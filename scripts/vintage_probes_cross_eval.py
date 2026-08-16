#!/usr/bin/env python
"""Refit the hu_harm vintage probes, keep them, and score them on the OTHER two concepts.

What this does
--------------
``scripts/attribution_vintage.py`` produced
``results_hu_harm_gemma27b_batch_ablation/vintage/SUMMARY.md``: 80 real ``ProbeFactory``
refits (2 arms x 4 vintages x 10 seeds) on the hu_harm base training data plus one
*vintage* of iteration-3 red-team pairs, scored on the four ``eval_dataset_hu_ha`` splits.
It recorded **metrics only** — the fitted probes were discarded.

This script refits the same 80, **keeps them**, and additionally scores every one on the
high-stakes (``eval_datasets``, 4 splits) and instruction-following (``eval_instructions``,
7 splits) eval sets. Those are *different concepts*, so this is a transfer measurement:
does a probe trained to detect harm-to-human rank high-stakes conversations, or
instruction-following ones, above their opposites?

Vintage membership is imported from ``attribution_vintage`` rather than reimplemented, so
the training sets here are the same sets by construction, not by coincidence.

Why the eval activations already exist
--------------------------------------
All three concepts' probes are ``google/gemma-3-27b-it`` layer 32, and an activation blob
is a function of the conversation, the model and the layer — **not** of the probe. Both
foreign concepts' full-split blobs are already cached locally (fetched by
``scripts/eval_probe_versions.py``), and all three runs load their splits under the same
two message transforms. So no gemma-3-27b forward is run anywhere in this script.

Labels across concepts
----------------------
A foreign split's ``labels`` strings are its own concept's, not the probe's, so a split
cannot be loaded with ``probe.pos_class_label``. Each eval directory therefore declares its
own (pos, neg) pair in ``EVAL_DIRS``, and AUROC is computed as "does the hu_harm probe's
score rank *that directory's* positive class above its negative one". **An AUROC below 0.5
is a real result, not a bug** — it means the harm score is anti-correlated with that
concept — so nothing here flips or absolute-values it.

Two phases, because of memory
-----------------------------
``fit`` holds the assembly (base + all iteration-3 red-team activations, ~13 GB peak) and
touches no eval blob. ``score`` drops the assembly entirely, then walks **splits outer,
probes inner**: one blob is resident at a time and every probe is scored against it in a
single pass. The alternative — scoring inside the fit loop — would hold the 13 GB assembly
alongside the 11.3 GB ``anthropic`` blob, over this box's cgroup ceiling.

The single pass is what makes ``score`` cheap. ``_classifier.logits`` re-reads the whole
activation tensor per probe (~4 min for ``anthropic``, so ~5 h for 80 probes on that split
alone). ``multi_probe_logits`` evaluates all 80 heads in one traversal: the per-token
linear becomes one ``h @ W`` with W of shape (5376, 80), and each probe's softmax pooling
is applied along the token axis of the same result. Run ``--verify-forward`` to check it
against ``_classifier.logits`` — it must agree, or the numbers are not comparable to
SUMMARY.md.

Usage::

    .venv_claude/bin/python scripts/vintage_probes_cross_eval.py fit
    .venv_claude/bin/python scripts/vintage_probes_cross_eval.py score --verify-forward
    .venv_claude/bin/python scripts/vintage_probes_cross_eval.py report
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")  # tuberlens tqdm-wraps every epoch on stderr

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

import attribution_lib as A  # noqa: E402
import attribution_refit as R  # noqa: E402
import attribution_vintage as V  # noqa: E402

OUT_DIR = REPO / "results_hu_harm_vintage_cross_eval"
PROBE_DIR = OUT_DIR / "probes"
FIT_PROGRESS = OUT_DIR / "fit_progress.jsonl"
SCORE_ROWS = OUT_DIR / "score_rows.jsonl"

ITERATION = 3
VINTAGES = [0, 1, 2, 3]
SEEDS = list(range(42, 52))  # the sweep's 10 seeds

# Each eval directory, its blob cache, and the (pos, neg) label strings ITS OWN splits
# use. The probe's labels are irrelevant here — see the module docstring.
EVAL_DIRS = {
    "hu_ha": {
        "concept": "harmful_to_human (the probes' own concept)",
        "dataset_dir": REPO / "eval_dataset_hu_ha",
        "cache_dir": REPO / "results_hu_harm_gemma27b_batch_ablation/eval_activations",
        "pos": "harmful_to_human",
        "neg": "not_harmful_to_human",
        "splits": A.EVAL_SPLITS,
    },
    "hs": {
        "concept": "high-stakes",
        "dataset_dir": REPO / "eval_datasets",
        "cache_dir": REPO / "results_hs_gemma27b_batch_ablation/eval_activations",
        "pos": "high-stakes",
        "neg": "low-stakes",
        "splits": ["anthropic", "mt", "mts", "toolace"],
    },
    "instructions": {
        "concept": "instruction-following",
        "dataset_dir": REPO / "eval_instructions",
        "cache_dir": REPO / "results_instructions_gemma27b_shared/eval_activations",
        "pos": "assistant_follows_the_instruction",
        "neg": "assistant_does_not_follow_the_instruction",
        "splits": [
            "anthropic_harmless_refusal", "bbq_substitution", "hc_context_drift",
            "hc_contradiction", "mm_substitution", "oig_context_drift", "oig_omission",
        ],
    },
}


def probe_path(arm: str, vintage: int, seed: int) -> Path:
    return PROBE_DIR / f"{arm}_v{vintage}_s{seed}.pkl"


def _append(path: Path, row: dict) -> None:
    """One self-contained row, flushed and fsync'd — a hard kill costs one unit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # truncated by a hard kill; that unit simply re-runs
    return out


# ---------------------------------------------------------------- phase A: fit


def fit(args) -> None:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    done = {(r["arm"], r["vintage"], r["seed"]) for r in _rows(FIT_PROGRESS)
            if probe_path(r["arm"], r["vintage"], r["seed"]).exists()}

    for arm in args.arms:
        keep, report = V.vintages(arm, ITERATION)
        todo = [(s, k) for s in args.seeds for k in args.vintages
                if (arm, k, s) not in done]
        print(f"\n=== {arm}: {len(todo)} fit(s) to run, {len(done)} already on disk",
              flush=True)
        for k in args.vintages:
            print(f"  vintage {k}: {len(keep[k])} of {report['n_rows_final']} "
                  f"iter{ITERATION} rows", flush=True)
        if not todo:
            continue

        asm = V.assemble_train_only(arm, ITERATION)  # ~1 min, ~10 GB; reused by every fit
        n_rows = len(asm.redteam)

        # Seeds outer, vintages inner: after each seed the run holds a COMPLETE curve
        # for that seed, so stopping early still yields a usable comparison.
        for seed, k in todo:
            drop = set(range(n_rows)) - set(keep[k])
            t0 = time.time()
            probe, n_tr, n_val = R.refit(asm, drop_rows=drop, seed=seed)
            fit_s = time.time() - t0
            with probe_path(arm, k, seed).open("wb") as fh:
                pickle.dump(probe, fh)
            _append(FIT_PROGRESS, {
                "arm": arm, "vintage": k, "seed": seed,
                "n_redteam_rows": len(keep[k]), "n_train": n_tr, "n_val": n_val,
                # None when validation AUROC never improved — routine on the tiny
                # base-only vintage, not an error.
                "best_epoch": (None if probe._classifier.best_epoch is None
                               else int(probe._classifier.best_epoch)),
                "fit_seconds": fit_s,
            })
            print(f"  seed {seed} vintage {k}: fit {fit_s:6.1f}s  "
                  f"train={n_tr} val={n_val} best_epoch={probe._classifier.best_epoch}",
                  flush=True)
            del probe
            gc.collect()
            torch.cuda.empty_cache()

        del asm
        gc.collect()


# -------------------------------------------------------------- phase B: score


def multi_probe_logits(acts, mask, W, B, T, device: str, batch_size: int,
                       dtype: torch.dtype = torch.bfloat16):
    """Sequence logits of P ``LinearThenSoftmax`` heads in ONE pass over ``acts``.

    Mirrors ``LinearThenAgg.forward`` + ``LinearThenSoftmax.agg`` (tuberlens
    ``pytorch_modules.py``) with the head axis vectorised: per token the linear becomes
    ``h @ W + B`` for all P heads at once, pad positions are zeroed, and each head's
    softmax pooling runs along the token axis at its own temperature.

    **``dtype`` defaults to bfloat16 on purpose.** These probes are *stored* in bf16 and
    ``_classifier.logits`` runs the head in bf16, whose ULP at these logit magnitudes
    (|s| up to ~25) is ~0.1 — so an fp32 recomputation of the same weights is a
    genuinely different scale, not a more precise one, and moves ``eval_ant_hh``'s AUROC
    by ~0.006 (see the precision note in ``attribution_lib``'s docstring). SUMMARY.md was
    produced through the bf16 path, so reproducing it requires bf16 here. Pass fp32 only
    for a comparison that is fp32 on both sides.

    ``W`` (E, P), ``B`` (P,), ``T`` (P,) given in fp32 and cast here. Returns (n_rows, P)
    fp32 (the container only — the arithmetic happened in ``dtype``).
    """
    n = acts.shape[0]
    out = torch.empty((n, W.shape[1]), dtype=torch.float32)
    W, B, T = W.to(device, dtype), B.to(device, dtype), T.to(device, dtype)
    for i in range(0, n, batch_size):
        h = acts[i:i + batch_size].to(device, dtype)
        mk = mask[i:i + batch_size].to(device).bool()[:, :, None]     # (b, t, 1)
        z = h @ W + B                                                 # (b, t, P)
        z = z.masked_fill(~mk, 0.0)
        p = torch.softmax(z.masked_fill(~mk, float("-inf")) / T, dim=1)
        out[i:i + batch_size] = (z * p).sum(dim=1).float().cpu()
        del h, mk, z, p
    return out


def _load_heads(arm_filter=None):
    """Every saved probe as ``(key, w, b, temperature)``, in a stable order."""
    heads, keys = [], []
    for path in sorted(PROBE_DIR.glob("*.pkl")):
        arm, v, s = path.stem.rsplit("_", 2)
        if arm_filter and arm not in arm_filter:
            continue
        probe = A.load_probe(path)
        w, b, t = A.probe_params(probe)
        keys.append({"arm": arm, "vintage": int(v[1:]), "seed": int(s[1:])})
        heads.append((w, b, t))
        del probe
    if not heads:
        raise SystemExit(f"no probes found in {PROBE_DIR} — run the `fit` phase first")
    W = torch.stack([h[0] for h in heads], dim=1).float()   # (E, P)
    B = torch.tensor([h[1] for h in heads], dtype=torch.float32)
    T = torch.tensor([h[2] for h in heads], dtype=torch.float32)
    return keys, W, B, T


def _load_split(spec: dict, split: str):
    """The split's dataset (own labels) and its cached activation blob, mmap'd."""
    from tuberlens.interfaces.dataset import LabelledDataset

    ds = LabelledDataset.load_from(
        spec["dataset_dir"] / f"{split}.jsonl",
        pos_class_label=spec["pos"],
        neg_class_label=spec["neg"],
        combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
    )
    y = ds.labels_torch().float().cpu().numpy()
    blob_path = spec["cache_dir"] / f"{split}-acts_full.pt"
    if not blob_path.exists():
        raise SystemExit(f"missing activation blob {blob_path}")
    blob = torch.load(blob_path, weights_only=False, mmap=True)
    if len(blob["activations"]) != len(ds):
        raise SystemExit(
            f"{split}: blob has {len(blob['activations'])} rows but the split has {len(ds)}"
        )
    return ds, y, blob


def _verify_forward(ds, blob, keys, W, B, T, device, batch_size) -> str:
    """Check the fused multi-head pass against tuberlens' own per-probe path.

    Compares head 0's logits from ``multi_probe_logits`` with
    ``probe._classifier.logits``. If these disagree beyond float noise, every number
    this script produces is on a different scale from SUMMARY.md's.
    """
    import contextlib
    import io

    from tuberlens.interfaces.activations import Activation

    k = keys[0]
    probe = A.load_probe(probe_path(k["arm"], k["vintage"], k["seed"]))
    attached = ds.assign(
        activations=blob["activations"],
        attention_mask=blob["attention_mask"],
        input_ids=blob["input_ids"],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        ref = probe._classifier.logits(Activation.from_dataset(attached))
    ref = ref.float().cpu().numpy()
    del attached
    gc.collect()

    lines = []
    for name, dt in (("bf16 (default)", torch.bfloat16), ("fp32", torch.float32)):
        fast = multi_probe_logits(
            blob["activations"], blob["attention_mask"],
            W[:, :1], B[:1], T[:1], device, batch_size, dtype=dt,
        ).numpy()[:, 0]
        dev = float(np.max(np.abs(ref - fast)))
        scale = float(np.max(np.abs(ref))) or 1.0
        # AUROC is what actually has to match; a logit residual inside one bf16 ULP can
        # still reorder ties, so report the metric-level consequence too.
        y = ds.labels_torch().float().cpu().numpy()
        d_auroc = abs(A.auroc_pipeline(y, ref) - A.auroc_pipeline(y, fast))
        lines.append(f"{name}: max |Δlogit| {dev:.3e} ({dev / scale:.1e} rel), "
                     f"|Δauroc_pipeline| {d_auroc:.2e}")
    del probe
    gc.collect()
    return " | ".join(lines) + f"  [{len(ref)} rows, vs _classifier.logits]"


def score(args) -> None:
    keys, W, B, T = _load_heads(args.arms if args.arms != sorted(A.ARMS) else None)
    print(f"loaded {len(keys)} probe heads, embed_dim={W.shape[0]}", flush=True)

    done = {(r["eval_set"], r["split"], r["arm"], r["vintage"], r["seed"])
            for r in _rows(SCORE_ROWS)}
    verified = not args.verify_forward

    for name, spec in EVAL_DIRS.items():
        if args.eval_sets and name not in args.eval_sets:
            continue
        for split in spec["splits"]:
            if all((name, split, k["arm"], k["vintage"], k["seed"]) in done for k in keys):
                print(f"[{name}/{split}] already scored for every probe — skipping",
                      flush=True)
                continue
            t0 = time.time()
            ds, y, blob = _load_split(spec, split)

            if not verified:
                print(f"[{name}/{split}] forward check: "
                      f"{_verify_forward(ds, blob, keys, W, B, T, args.device, args.batch_size)}",
                      flush=True)
                verified = True

            s = multi_probe_logits(
                blob["activations"], blob["attention_mask"],
                W, B, T, args.device, args.batch_size,
            ).numpy()
            print(f"[{name}/{split}] {len(ds)} rows x {len(keys)} probes "
                  f"in {time.time() - t0:.0f}s  (pos rate {y.mean():.2f})", flush=True)

            for i, k in enumerate(keys):
                if (name, split, k["arm"], k["vintage"], k["seed"]) in done:
                    continue
                _append(SCORE_ROWS, {
                    "eval_set": name, "concept": spec["concept"], "split": split,
                    "n_rows": len(ds), **k, **A.auroc_both(y, s[:, i]),
                })
            del ds, blob, s
            gc.collect()


# -------------------------------------------------------------- phase C: report


def report(args) -> None:
    import pandas as pd

    rows = _rows(SCORE_ROWS)
    if not rows:
        raise SystemExit(f"no scored rows in {SCORE_ROWS} — run the `score` phase first")
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "cross_eval_rows.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 60)

    # Mean over an eval set's splits, per (arm, vintage, seed), then mean ± sd over seeds
    per_seed = (df.groupby(["eval_set", "arm", "vintage", "seed"])["pipeline"]
                  .mean().reset_index())
    agg = (per_seed.groupby(["eval_set", "arm", "vintage"])["pipeline"]
                   .agg(["mean", "std", "count"]).round(4).reset_index())
    print("\nMean AUROC over each eval set's splits (pipeline scale), mean ± sd over seeds")
    print(agg.to_string(index=False))
    agg.to_csv(OUT_DIR / "cross_eval_summary.csv", index=False)

    for es in df["eval_set"].unique():
        sub = df[df["eval_set"] == es]
        grid = (sub.groupby(["arm", "vintage", "split"])["pipeline"]
                   .mean().unstack("split").round(4))
        print(f"\nPer-split AUROC (mean over seeds) — {es}")
        print(grid.to_string())
        grid.to_csv(OUT_DIR / f"cross_eval_{es}_by_split.csv")

    # Reproduction check: hu_ha must match the sweep that wrote SUMMARY.md.
    old = REPO / "results_hu_harm_gemma27b_batch_ablation/vintage/vintage_progress.jsonl"
    if old.exists():
        prev = []
        for r in _rows(old):
            m = np.mean([r["auroc"][sp]["pipeline"] for sp in A.EVAL_SPLITS])
            prev.append({"arm": r["arm"], "vintage": r["vintage"], "prev_mean": m})
        pv = (pd.DataFrame(prev).groupby(["arm", "vintage"])["prev_mean"]
                .agg(["mean", "std"]).round(4).reset_index()
                .rename(columns={"mean": "orig_mean", "std": "orig_sd"}))
        mine = agg[agg["eval_set"] == "hu_ha"][["arm", "vintage", "mean", "std"]]
        cmp = mine.merge(pv, on=["arm", "vintage"]).rename(
            columns={"mean": "new_mean", "std": "new_sd"})
        cmp["Δmean"] = (cmp["new_mean"] - cmp["orig_mean"]).round(4)
        print("\nReproduction of SUMMARY.md (hu_ha, mean over 4 splits and 10 seeds)")
        print(cmp.to_string(index=False))
        cmp.to_csv(OUT_DIR / "reproduction_vs_summary.csv", index=False)
        print("\nRefits are independent ProbeFactory fits at the same seeds, so Δmean "
              "should sit well inside the sd column, not at zero.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in (("fit", fit), ("score", score), ("report", report)):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
        if name in ("fit", "score"):
            p.add_argument("--arms", nargs="+", choices=sorted(A.ARMS),
                           default=sorted(A.ARMS))
        if name == "fit":
            p.add_argument("--vintages", nargs="+", type=int, default=VINTAGES)
            p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
        if name == "score":
            p.add_argument("--eval-sets", nargs="*", choices=sorted(EVAL_DIRS),
                           default=None)
            p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--verify-forward", action="store_true",
                           help="check the fused pass against _classifier.logits once")

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
