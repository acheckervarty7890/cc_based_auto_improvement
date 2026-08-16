"""Where do v2, v3new and the eval rows sit in an MLP probe's post-readout space?

The question this answers
------------------------
``docs/heldout_v3_vs_v2_overlap.md`` ends with 31 eval rows that no probe family reaches,
and ``docs/probe_architecture_sweep.md`` shows no architecture reaches them either. Both
measure the *decision* (right/wrong, AUROC). This measures the **representation**: for the
three MLP heads, take the vector the sequence logit is a linear function of, and ask how
far apart the two training vintages and the eval splits sit in it.

Only the MLP heads have a post-readout space at all. ``linear_then_softmax`` and friends
read out straight to a scalar, and cosine distance between scalars is not a thing — which
is why the request scopes to the MLP probes.

What "after the readout" means, exactly
---------------------------------------
Each MLP head is ``Linear(5376 -> 64) -> GELU -> Linear(64 -> 1)``. The **post-readout
representation** here is the 64-d vector entering that last linear layer, i.e. the whole
head minus its final projection. Two properties make it the right object:

- The sequence logit is *exactly* ``W2 . h + b2`` for it, so cosine geometry in this space
  is the geometry the decision is made in.
- It is the same 64 dimensions for all three heads, so the three are comparable.

Per architecture (``h`` = 64-d post-readout, ``p`` = the pooled 5376-d input to the MLP,
reported as the pre-readout control):

``mean_then_mlp``      ``p`` = masked mean over tokens;  ``h`` = GELU(W1 p).
``attention_then_mlp`` ``p`` = the AttnLite attention-weighted context; ``h`` = GELU(W1 p).
``mlp_then_softmax``   the MLP runs **per token**, so both are pooled with the head's own
                       softmax weights ``w_t = softmax(z_t / T)``: ``h = sum_t w_t h_t``,
                       ``p = sum_t w_t x_t``. This is exact rather than an analogy: the
                       last layer is linear and ``sum_t w_t = 1``, so
                       ``sum_t w_t (W2 h_t + b2) = W2 (sum_t w_t h_t) + b2`` — the pooled
                       hidden is the vector the sequence logit reads. Asserted at runtime
                       against the classifier's own logits.

Metric
------
Cosine distance ``1 - cos`` between **group centroids**, reported raw and centered.
Centering is not cosmetic: ``why_close_but_wrong_centered.py`` established that a residual
stream's large shared component parks every raw pair at 0.86-0.96, and a GELU layer is
worse — its output is bounded below by -0.17 and mostly positive, so raw cosine there is
close to a measure of vector length.

**The centering vector is fixed, and that is load-bearing.** It is the grand mean over the
six primary groups with each row counted once, computed once and never recomputed. The
obvious alternative — centre each *pair* on its own pooled mean — is degenerate: the pooled
mean is then zero, so ``n_a * c_a = -n_b * c_b``, the two centroids are exactly
antiparallel, and every pair scores cosine distance 2.0000 regardless of the data. (This
script printed exactly that before the metric was fixed; the ``*_centered`` fields in the
progress sidecar are from that superseded code and should not be quoted.)

Three things make the between-group numbers readable:

- **a permutation null.** Pool two groups, re-split at the same two sizes, recompute. The
  null absorbs the effect that would otherwise be read as signal — a centroid's sampling
  noise goes as ``1/sqrt(n)``, so a 134-row eval split sits further from everything than a
  546-row vintage does for that reason alone. ``ratio = observed / null`` is the headline
  number. (Within-group spread, which this script reported first, is useless here: in 64
  dimensions a row's cosine distance to its own centroid is ~0.97 whatever the answer is.)
- **v2 <-> v3new** as the comparison point. Both are red-team data from the same generator
  and both are in training, so their distance is what "same kind of data" looks like.
- **per class**. The readout is trained to separate the classes, so a centroid gap between
  two groups with different class balance is mostly that balance. Distances are therefore
  also reported within the positive rows only and within the negative rows only.

Caveat that cannot be designed away here
----------------------------------------
v2 and v3new are both **in** the training set (the probe trains on iteration 3, which is
their union) while the eval rows are out of sample, so some of the v2/v3new-to-eval gap is
memorisation, not domain. The partial control is ``val_only``: the same numbers computed
from the red-team rows on the validation side, which get no gradient (tuberlens uses them
for early stopping only). If the gap survives there it is not purely fitting.

Usage:
    .venv_claude/bin/python scripts/mlp_readout_cosine.py
    .venv_claude/bin/python scripts/mlp_readout_cosine.py --seeds 42 --arms gptoss120b
    .venv_claude/bin/python scripts/mlp_readout_cosine.py --summarize-only
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
import attribution_vintage as V
from arch_sweep import ITERATION, build_train_val, load_eval_truth

OUT_DIR = A.REPO / "results_hu_harm_gemma27b_batch_ablation/arch_sweep"
PROGRESS = OUT_DIR / "readout_cosine_progress.jsonl"
SUMMARY = OUT_DIR / "readout_cosine.json"
REPS_DIR = OUT_DIR / "readout_reps"

ARCHITECTURES = ["mlp_then_softmax", "mean_then_mlp", "attention_then_mlp"]
SEEDS = [42, 43, 44]
ARMS = ["gptoss120b", "deepseekv4pro"]

#: Rows per forward chunk. 16 x 1024 x 5376 x 2 B is ~176 MB, which the GPU has spare
#: while the 10 GB of host-resident train/val activations stay put.
CHUNK = 16


# --- representation extraction -------------------------------------------------------


def _mlp_of(model) -> torch.nn.Sequential:
    """The ``Sequential`` MLP head, whichever attribute name its class uses."""
    for name in ("mlp", "linear", "classifier"):
        mod = getattr(model, name, None)
        if isinstance(mod, torch.nn.Sequential):
            return mod
    raise TypeError(f"no Sequential MLP head on {type(model).__name__}")


@torch.no_grad()
def _reps_chunk(model, arch: str, x: torch.Tensor, mask: torch.Tensor):
    """``(pre, post, logit)`` for one chunk. See the module docstring for definitions."""
    mlp = _mlp_of(model)
    trunk, last = mlp[:-1], mlp[-1]
    m = mask.bool()

    if arch == "mean_then_mlp":
        pre = x.masked_fill(~m.unsqueeze(-1), 0.0).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
    elif arch == "attention_then_mlp":
        scores = model.context_query(x).squeeze(-1) / model.scale
        w = torch.softmax(scores.masked_fill(~m, float("-inf")), dim=-1)
        pre = torch.einsum("bs,bse->be", w, x)
    elif arch == "mlp_then_softmax":
        h_tok = trunk(x)                                  # [b, s, hidden]
        z_tok = last(h_tok).squeeze(-1)                   # [b, s]
        temperature = model.kwargs["temperature"]
        w = torch.softmax(z_tok.masked_fill(~m, float("-inf")) / temperature, dim=1)
        # Pool the hidden layer, not the input, with the head's own weights; the pooled
        # input is carried alongside purely as the pre-readout control.
        post = torch.einsum("bs,bsh->bh", w, h_tok)
        pre = torch.einsum("bs,bse->be", w, x)
        return pre, post, last(post).squeeze(-1)
    else:
        raise ValueError(arch)

    post = trunk(pre)
    return pre, post, last(post).squeeze(-1)


@torch.no_grad()
def representations(probe, arch: str, acts: torch.Tensor, mask: torch.Tensor,
                    rows: list[int] | None = None):
    """``(pre [n, 5376], post [n, 64], logit [n])`` as float32 numpy, chunked.

    ``rows`` selects a subset **inside** the chunk loop rather than up front: fancy-
    indexing the whole red-team side of ``train`` would materialise a second copy of
    ~700 x 1024 x 5376 activations (7.7 GB) next to the 10 GB already resident, which is
    how this box gets OOM-killed.
    """
    model = probe._classifier.model
    model.eval()
    device, dtype = probe._classifier.device, probe._classifier.dtype
    order = list(range(acts.shape[0])) if rows is None else list(rows)
    pre, post, lg = [], [], []
    for i in range(0, len(order), CHUNK):
        sel = order[i : i + CHUNK]
        x = acts[sel].to(device=device, dtype=dtype)
        m = mask[sel].to(device=device)
        p, h, z = _reps_chunk(model, arch, x, m)
        pre.append(p.float().cpu().numpy())
        post.append(h.float().cpu().numpy())
        lg.append(z.float().cpu().numpy())
        del x, m, p, h, z
    return np.concatenate(pre), np.concatenate(post), np.concatenate(lg).ravel()


# --- cosine geometry -----------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - _unit(a) @ _unit(b))


def geometry(reps: dict[str, np.ndarray], centered: bool) -> dict:
    """Centroid cosine distances + within-group spread for one representation space.

    ``centered`` subtracts the grand mean over **all** rows first, so the shared component
    every group carries is removed once, globally, rather than per group (which would move
    each group's centroid to the origin and make the question vacuous).
    """
    names = [k for k in reps if len(reps[k])]
    if centered:
        grand = np.concatenate([reps[k] for k in names]).mean(0, keepdims=True)
        reps = {k: reps[k] - grand for k in names}

    cent = {k: reps[k].mean(0) for k in names}
    between = {
        f"{a}|{b}": _cos_dist(cent[a], cent[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    within = {
        k: float(np.mean(1.0 - _unit(reps[k]) @ _unit(cent[k]))) for k in names
    }
    return {"between": between, "within": within, "n": {k: int(len(reps[k])) for k in names}}


def _pair(g: dict, a: str, b: str) -> float:
    return g["between"].get(f"{a}|{b}", g["between"].get(f"{b}|{a}", float("nan")))


def all_geometries(post: dict[str, np.ndarray], pre: dict[str, np.ndarray],
                   labels: dict[str, np.ndarray]) -> dict:
    """Every space x centering x class-restriction combination, in one dict."""
    out = {}
    for space_name, space in (("post", post), ("pre", pre)):
        for centered in (True, False):
            tag = f"{space_name}_{'centered' if centered else 'raw'}"
            out[tag] = geometry(space, centered)
            for cls, want in (("pos", True), ("neg", False)):
                sub = {
                    k: v[labels[k] == want] for k, v in space.items()
                    if k in labels and (labels[k] == want).any()
                }
                out[f"{tag}_{cls}"] = geometry(sub, centered)
    return out


# --- self-test -----------------------------------------------------------------------


def self_test() -> int:
    """Assert the post-readout vector really is what the sequence logit reads.

    ``last(post)`` must reproduce the classifier's own logits for all three heads. For
    ``mlp_then_softmax`` that is the non-obvious one — it holds only because the last layer
    is linear and the softmax weights sum to 1 — and it is exactly the claim the whole
    measurement rests on, so it is checked on synthetic data in seconds before any real fit.
    """
    sys.path.insert(0, str(A.REPO / "src"))
    from tuberlens.interfaces.activations import Activation

    from agentic_redteam.probe_architectures import build_probe
    from test_probe_architectures import make_dataset

    train, val = make_dataset(128, 0, 4.0), make_dataset(64, 1, 4.0)
    failures = 0
    print("--- post-readout reconstruction of the sequence logit ---")
    for arch in ARCHITECTURES:
        with contextlib.redirect_stdout(io.StringIO()):
            probe = build_probe(
                arch, train, val, model_name="test", layer=0,
                pos_class_label="positive", neg_class_label="negative",
            )
            ref = probe._classifier.logits(Activation.from_dataset(val))
        ref = ref.float().cpu().numpy().ravel()
        _pre, post, mine = representations(
            probe, arch,
            val.other_fields["activations"], val.other_fields["attention_mask"],
        )
        gap = float(np.abs(ref - mine).max())
        # bf16 forward, so the tolerance is the dtype's, not float32's.
        ok = gap < 5e-2 * max(1.0, float(np.abs(ref).max()))
        failures += not ok
        print(f"  {arch:20s} post {post.shape}  max|logit gap| {gap:.2e}  "
              f"{'ok' if ok else 'FAIL'}")
    print("FAILED" if failures else "all ok")
    return failures


# --- progress ------------------------------------------------------------------------


def append_progress(row: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_progress() -> list[dict]:
    if not PROGRESS.exists():
        return []
    rows = []
    with PROGRESS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --- the run -------------------------------------------------------------------------


def run_arm(arm: str, seeds: list[int], architectures: list[str], resume: bool = True) -> None:
    done = {(r["arm"], r["architecture"], r["seed"]) for r in read_progress()} if resume else set()
    todo = [(s, a) for s in seeds for a in architectures if (arm, a, s) not in done]
    if not todo:
        print(f"=== {arm}: nothing to do", flush=True)
        return

    print(f"\n=== {arm} === {len(todo)} fit(s)", flush=True)
    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.probe_architectures import build_probe
    from tuberlens.interfaces.dataset import LabelledDataset

    keep, _ = V.vintages(arm, ITERATION)
    v2_idx = set(keep[2])

    asm = V.assemble_train_only(arm, ITERATION)
    probe_meta = asm.probe
    truth = load_eval_truth(probe_meta)
    n_rt = len(asm.redteam)
    rt_is_val = asm.rt_is_val.copy()
    rt_labels = asm.redteam.labels_torch().float().cpu().numpy().ravel() > 0.5
    n_base_train = len(asm.base_train)
    n_base_val = len(asm.base_val)

    # build_train_val consumes asm.redteam, so the row map is recorded first. train is
    # base_train ++ redteam[train side], val is base_val ++ redteam[val side], both in
    # ascending original order — so a red-team row's position is its rank on its side.
    train_rt = [i for i in range(n_rt) if not rt_is_val[i]]
    val_rt = [i for i in range(n_rt) if rt_is_val[i]]
    pos_in_train = {i: n_base_train + r for r, i in enumerate(train_rt)}
    pos_in_val = {i: n_base_val + r for r, i in enumerate(val_rt)}

    train, val = build_train_val(asm)
    print(f"  train {len(train)} rows, val {len(val)}; red-team {n_rt} "
          f"(v2 {len(v2_idx)}, v3new {n_rt - len(v2_idx)})", flush=True)

    for seed, arch in todo:
        t0 = time.time()
        seed_everything(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            probe = build_probe(
                arch, train, val,
                model_name=probe_meta.model_name,
                layer=probe_meta.layer,
                pos_class_label=probe_meta.pos_class_label,
                neg_class_label=probe_meta.neg_class_label,
                probe_description=probe_meta.description,
            )
        fit_s = time.time() - t0

        pre_g: dict[str, np.ndarray] = {}
        post_g: dict[str, np.ndarray] = {}
        lab_g: dict[str, np.ndarray] = {}

        # --- red-team side, split into the two vintages -------------------------------
        rt_pre = np.zeros((n_rt, train.other_fields["activations"].shape[-1]), dtype=np.float32)
        rt_post = None
        for ds, posmap in ((train, pos_in_train), (val, pos_in_val)):
            if not posmap:
                continue
            rt_rows = sorted(posmap)                      # red-team indices, ascending
            p, h, _z = representations(
                probe, arch,
                ds.other_fields["activations"], ds.other_fields["attention_mask"],
                rows=[posmap[i] for i in rt_rows],
            )
            if rt_post is None:
                rt_post = np.zeros((n_rt, h.shape[1]), dtype=np.float32)
            for j, i in enumerate(rt_rows):
                rt_pre[i] = p[j]
                rt_post[i] = h[j]
            del p, h
            gc.collect()

        idx_v2 = np.array(sorted(v2_idx), dtype=int)
        idx_new = np.array([i for i in range(n_rt) if i not in v2_idx], dtype=int)
        for name, idx in (("v2", idx_v2), ("v3new", idx_new)):
            pre_g[name] = rt_pre[idx]
            post_g[name] = rt_post[idx]
            lab_g[name] = rt_labels[idx]
            vsub = idx[rt_is_val[idx]]
            pre_g[f"{name}_val"] = rt_pre[vsub]
            post_g[f"{name}_val"] = rt_post[vsub]
            lab_g[f"{name}_val"] = rt_labels[vsub]

        # --- eval side, one blob at a time --------------------------------------------
        aurocs = {}
        for split in A.EVAL_SPLITS:
            ds = LabelledDataset.load_from(
                A.EVAL_DATASET_DIR / f"{split}.jsonl",
                pos_class_label=probe_meta.pos_class_label,
                neg_class_label=probe_meta.neg_class_label,
                combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
                convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
            )
            blob = torch.load(
                A.EVAL_ACTIVATIONS_DIR / f"{split}-acts_full.pt",
                weights_only=False, mmap=True,
            )
            p, h, z = representations(probe, arch, blob["activations"], blob["attention_mask"])
            key = split.replace("eval_", "")
            pre_g[key] = p
            post_g[key] = h
            lab_g[key] = truth[split]["y"] > 0.5
            aurocs[split] = A.auroc_both(truth[split]["y"], z)
            del ds, blob, p, h, z
            gc.collect()

        geo = all_geometries(post_g, pre_g, lab_g)

        REPS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            REPS_DIR / f"{arm}_{arch}_seed{seed}.npz",
            **{f"post_{k}": v for k, v in post_g.items()},
            **{f"label_{k}": v for k, v in lab_g.items()},
        )

        g = geo["post_centered"]
        print(
            f"  seed {seed} {arch:20s} fit {fit_s:6.1f}s  "
            f"v2|v3new {_pair(g, 'v2', 'v3new'):.4f}  "
            f"v2|ant_hh {_pair(g, 'v2', 'ant_hh'):.4f}  "
            f"within v2 {g['within']['v2']:.4f}",
            flush=True,
        )
        append_progress({
            "arm": arm, "architecture": arch, "seed": seed,
            "fit_seconds": fit_s,
            "best_epoch": (None if getattr(probe._classifier, "best_epoch", None) is None
                           else int(probe._classifier.best_epoch)),
            "n_rt": n_rt, "n_v2": int(len(idx_v2)), "n_v3new": int(len(idx_new)),
            "auroc": aurocs,
            "geometry": geo,
        })
        del probe, pre_g, post_g, rt_pre, rt_post
        gc.collect()
        torch.cuda.empty_cache()

    del asm, train, val
    gc.collect()
    torch.cuda.empty_cache()


# --- reporting -----------------------------------------------------------------------
#
# Everything below reads the saved per-row post-readout vectors (``readout_reps/*.npz``)
# rather than the ``geometry`` blob in the progress sidecar. The sidecar's ``*_centered``
# entries are superseded and should not be quoted: they centre on the grand mean of the
# groups being compared, and for a *pair* of groups that is degenerate — the pooled mean
# is zero, so ``n_a * c_a = -n_b * c_b`` and the two centroids are exactly antiparallel,
# giving cosine distance 2.0000 for every pair regardless of the data. The fix here is a
# **fixed** centring vector: the grand mean over all six primary groups, each row counted
# once, computed once and never recomputed — including inside the permutation loop.

EVAL_KEYS = ["ai_dilemmas", "ant_hh", "balanced_refusal", "daily_dilemmas"]
PRIMARY = ["v2", "v3new"] + EVAL_KEYS


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    a = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if not len(a):
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def load_fits() -> list[dict]:
    """One entry per saved fit: arm, architecture, seed, and the post-readout vectors."""
    out = []
    for f in sorted(REPS_DIR.glob("*.npz")):
        # "<arm>_<architecture>_seed<N>.npz"; arm never contains "_", architecture does.
        head, seed_s = f.stem.split("_seed")
        arm = head.split("_")[0]
        with np.load(f) as z:
            reps = {k[len("post_"):]: z[k] for k in z.files if k.startswith("post_")}
            labels = {k[len("label_"):]: z[k] for k in z.files if k.startswith("label_")}
        out.append({
            "arm": arm,
            "architecture": head[len(arm) + 1 :],
            "seed": int(seed_s),
            "reps": reps,
            "labels": labels,
        })
    return out


def fixed_center(reps: dict[str, np.ndarray]) -> np.ndarray:
    """Grand mean over the six primary groups, each row counted exactly once.

    The ``*_val`` groups are subsets of ``v2``/``v3new``, so including them would weight
    those rows twice. Defined off the group *set*, not off any pair, which is what keeps
    the pairwise distances below non-degenerate.
    """
    rows = [reps[g] for g in PRIMARY if g in reps and len(reps[g])]
    return np.concatenate(rows).mean(0, keepdims=True)


def distance_matrix(reps: dict[str, np.ndarray], center: np.ndarray | None,
                    groups: list[str]) -> dict[str, float]:
    names = [g for g in groups if g in reps and len(reps[g])]
    cent = {g: (reps[g] - center if center is not None else reps[g]).mean(0)
            for g in names}
    return {
        f"{a}|{b}": _cos_dist(cent[a], cent[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }


def permutation_null(a: np.ndarray, b: np.ndarray, center: np.ndarray | None,
                     n_perm: int, seed: int) -> dict:
    """Observed centroid distance vs. the null that ``a`` and ``b`` are one cloud.

    The within-group spread this script originally reported is the wrong yardstick: in 64
    dimensions a row's cosine distance to its own centroid is ~0.97 whatever the answer
    is, because the centroid averages many near-orthogonal vectors. It measures
    dimensionality, not separation.

    Permuting the labels measures the right thing. The pooled rows are re-split at the
    **same two sizes**, so the null already contains the effect that would otherwise be
    read as signal: a centroid's sampling noise goes as ``1/sqrt(n)``, so a 134-row eval
    split sits further from everything than a 546-row vintage does for that reason alone.

    ``center`` is applied **before** permuting and never recomputed — recomputing it per
    split is what makes the pairwise-centred version degenerate.
    """
    pooled = np.concatenate([a, b])
    if center is not None:
        pooled = pooled - center
    na = len(a)
    obs = _cos_dist(pooled[:na].mean(0), pooled[na:].mean(0))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        idx = rng.permutation(len(pooled))
        null[k] = _cos_dist(pooled[idx[:na]].mean(0), pooled[idx[na:]].mean(0))
    sd = float(null.std(ddof=1))
    return {
        "obs": obs,
        "null_mean": float(null.mean()),
        "null_sd": sd,
        "z": float((obs - null.mean()) / sd) if sd > 0 else float("nan"),
        "p": float((null >= obs).mean()),
        "n_a": int(len(a)), "n_b": int(len(b)),
    }


def _by_arm_arch(fits: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out = defaultdict(list)
    for f in fits:
        out[(f["arm"], f["architecture"])].append(f)
    return out


def _order(groups: set) -> list[str]:
    return [g for g in PRIMARY if g in groups]


def print_matrix(fits: list[dict], centered: bool, subset: str = "all") -> dict:
    """Full pairwise cosine-distance matrix, averaged over seeds.

    ``subset`` restricts to one class ("pos"/"neg") so a centroid gap between two groups
    with different class balance is not read as a domain gap — the readout is trained to
    separate the classes, so class composition moves these centroids more than anything
    else does.
    """
    tag = ("centered" if centered else "raw") + ("" if subset == "all" else f", {subset} rows")
    print(f"\n{'-' * 100}\nPOST-READOUT centroid cosine distance — {tag}\n{'-' * 100}")
    out = {}
    for (arm, arch), group in sorted(_by_arm_arch(fits).items()):
        per_pair = defaultdict(list)
        ns = {}
        for f in group:
            reps = f["reps"]
            if subset != "all":
                want = subset == "pos"
                reps = {g: v[f["labels"][g] == want] for g, v in reps.items()
                        if g in f["labels"] and (f["labels"][g] == want).any()}
            c = fixed_center(reps) if centered else None
            for k, v in distance_matrix(reps, c, PRIMARY).items():
                per_pair[k].append(v)
            for g in PRIMARY:
                if g in reps:
                    ns[g] = len(reps[g])
        names = _order(set(ns))
        print(f"\n  {arm} / {arch}   (n: " + ", ".join(f"{g}={ns[g]}" for g in names) + ")")
        print("    " + " " * 18 + "".join(f"{g[:16]:>18s}" for g in names))
        for a in names:
            line = f"    {a:18s}"
            for b in names:
                if a == b:
                    line += f"{'.':>18s}"
                    continue
                mu, sd = _mean_sd(per_pair.get(f"{a}|{b}", per_pair.get(f"{b}|{a}", [])))
                line += f"{mu:11.4f}+/-{sd:.3f}"
            print(line)
        out[f"{arm}|{arch}"] = {k: _mean_sd(v)[0] for k, v in per_pair.items()}
    return out


def print_null_table(fits: list[dict], centered: bool, n_perm: int) -> dict:
    print(f"\n{'-' * 100}\nPERMUTATION NULL "
          f"({'centered' if centered else 'raw'}, {n_perm} draws per pair per seed)"
          f"\n{'-' * 100}")
    out = {}
    for (arm, arch), group in sorted(_by_arm_arch(fits).items()):
        acc = defaultdict(list)
        for f in group:
            c = fixed_center(f["reps"]) if centered else None
            names = _order(set(f["reps"]))
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    acc[(a, b)].append(permutation_null(
                        f["reps"][a], f["reps"][b], c, n_perm, f["seed"]
                    ))
        print(f"\n  {arm} / {arch}")
        print(f"    {'pair':38s} {'n':>11s} {'observed':>10s} {'null':>9s} "
              f"{'ratio':>7s} {'z':>9s} {'p_max':>7s}")
        for (a, b), vals in sorted(acc.items(), key=lambda kv: -_mean_sd(
                [v["obs"] / v["null_mean"] for v in kv[1]])[0]):
            obs, _ = _mean_sd([v["obs"] for v in vals])
            nul, _ = _mean_sd([v["null_mean"] for v in vals])
            zz, _ = _mean_sd([v["z"] for v in vals])
            pp = max(v["p"] for v in vals)
            sizes = f"{vals[0]['n_a']}v{vals[0]['n_b']}"
            ratio = obs / nul if nul else float("nan")
            print(f"    {a + ' | ' + b:38s} {sizes:>11s} "
                  f"{obs:10.4f} {nul:9.4f} {ratio:7.1f} {zz:9.1f} {pp:7.3f}")
            out[f"{arm}|{arch}|{a}|{b}"] = {
                "obs": obs, "null_mean": nul, "ratio": obs / nul if nul else None,
                "z": zz, "p_max": pp, "n_a": vals[0]["n_a"], "n_b": vals[0]["n_b"],
                "n_seeds": len(vals),
            }
    print("\n  ratio = observed / null mean, i.e. how many times further apart the two "
          "groups sit than\n  two random halves of the same cloud at those sample sizes. "
          "p is the WORST over seeds.")
    return out


def print_vintage_comparison(fits: list[dict], centered: bool) -> None:
    print(f"\n{'-' * 100}\nWhich vintage sits closer to each eval split? "
          f"({'centered' if centered else 'raw'})\n{'-' * 100}")
    for (arm, arch), group in sorted(_by_arm_arch(fits).items()):
        cells = []
        for k in EVAL_KEYS:
            d2, dn = [], []
            for f in group:
                c = fixed_center(f["reps"]) if centered else None
                m = distance_matrix(f["reps"], c, PRIMARY)
                d2.append(_pair({"between": m}, "v2", k))
                dn.append(_pair({"between": m}, "v3new", k))
            gap = _mean_sd(d2)[0] - _mean_sd(dn)[0]
            cells.append(f"{k}={'v3new' if gap > 0 else 'v2':5s}({gap:+.4f})")
        print(f"  {arm:15s} {arch:20s} " + " ".join(cells))
    print("  (sign is d(v2,split) - d(v3new,split); positive means v3new sits closer)")


def print_val_control(fits: list[dict], centered: bool) -> None:
    print(f"\n{'-' * 100}\nIN-SAMPLE CONTROL: red-team rows on the validation side only "
          f"(no gradient)\n{'-' * 100}")
    groups = ["v2_val", "v3new_val"] + EVAL_KEYS
    for (arm, arch), group in sorted(_by_arm_arch(fits).items()):
        per_pair = defaultdict(list)
        for f in group:
            c = fixed_center(f["reps"]) if centered else None
            for k, v in distance_matrix(f["reps"], c, groups).items():
                per_pair[k].append(v)
        base = _mean_sd(per_pair.get("v2_val|v3new_val", []))[0]
        cells = [f"v2val|v3newval={base:.4f}"]
        for k in EVAL_KEYS:
            cells.append(f"{k}={_mean_sd(per_pair.get(f'v2_val|{k}', []))[0]:.4f}")
        print(f"  {arm:15s} {arch:20s} " + "  ".join(cells))
    print("  (distances from the v2 VALIDATION rows, which the fit never took a gradient "
          "step on)")


def print_pre_readout(rows: list[dict]) -> None:
    """The 5376-d pooled input to the MLP, raw — the readout's own input, for contrast.

    Only the raw metric is quoted: the per-row pre-readout vectors are 37 MB a fit and are
    not saved, so the fixed-centring and the permutation null cannot be recomputed for
    them from the sidecar. Raw-to-raw against the post table is still a like-for-like
    comparison, and it is the one that answers whether the readout creates the separation
    or inherits it.
    """
    print(f"\n{'-' * 100}\nPRE-READOUT (5376-d pooled MLP input), raw — control\n"
          f"{'-' * 100}")
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["architecture"])].append(r)
    for (arm, arch), group in sorted(by.items()):
        cells = []
        for a, b in [("v2", "v3new")] + [("v2", k) for k in EVAL_KEYS]:
            mu, _ = _mean_sd([_pair(r["geometry"]["pre_raw"], a, b) for r in group])
            cells.append(f"{a}|{b}={mu:.4f}")
        print(f"  {arm:15s} {arch:20s} " + "  ".join(cells))


def print_refit_check(rows: list[dict]) -> None:
    print("\nRefit check (mean eval AUROC, pipeline scale) — should match arch_sweep")
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["architecture"])].append(r)
    for (arm, arch), group in sorted(by.items()):
        m = [float(np.mean([r["auroc"][s]["pipeline"] for s in A.EVAL_SPLITS]))
             for r in group]
        mu, sd = _mean_sd(m)
        seeds = sorted(r["seed"] for r in group)
        print(f"  {arm:15s} {arch:20s} {mu:.4f} +/- {sd:.4f}  seeds {seeds}")


def print_report(rows: list[dict], n_perm: int = 500) -> dict:
    fits = load_fits()
    if not fits:
        print("no saved representations yet")
        return {}
    print(f"\n{'=' * 100}\nMLP post-readout cosine geometry — {len(fits)} fits, "
          f"seeds {sorted({f['seed'] for f in fits})}\n{'=' * 100}")
    print_refit_check(rows)

    out = {}
    out["matrix_centered"] = print_matrix(fits, centered=True)
    out["matrix_raw"] = print_matrix(fits, centered=False)
    if n_perm:
        out["null_centered"] = print_null_table(fits, centered=True, n_perm=n_perm)
    print_vintage_comparison(fits, centered=True)
    out["matrix_centered_pos"] = print_matrix(fits, centered=True, subset="pos")
    out["matrix_centered_neg"] = print_matrix(fits, centered=True, subset="neg")
    print_val_control(fits, centered=True)
    print_pre_readout(rows)
    return out


def summarize(n_perm: int = 500) -> None:
    rows = [r for r in read_progress() if "geometry" in r]
    tables = print_report(rows, n_perm=n_perm)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        json.dumps({"fits": rows, "tables": tables}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {SUMMARY}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--architectures", nargs="+", default=ARCHITECTURES)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--permutations", type=int, default=500,
                    help="permutation-null draws per pair (0 skips the null table)")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(1 if self_test() else 0)
    if args.summarize_only:
        summarize(args.permutations)
        return

    for arm in args.arms:
        run_arm(arm, args.seeds, args.architectures, resume=not args.no_resume)
    summarize(args.permutations)


if __name__ == "__main__":
    main()
