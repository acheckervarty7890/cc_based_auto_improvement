"""Can a probe learn *where* in a conversation to look, and does that need more data?

Every pooling head measured so far weights a token by its **content**: ``pre_mean``
averages uniformly, ``linear_then_softmax`` weights each token by its own logit, and
``attention`` weights it by a learned query. This script adds the axis none of them
touch — weighting a token by its **position** — and reads it against the three red-team
vintages, so the two questions are answered together:

1. Does replacing mean pooling with a *learned* map over token positions buy anything?
2. Does the answer depend on how much red-team data there is? A head with 1024 extra
   parameters that helps at 878 rows and hurts at 368 has been bought with data, not
   with architecture, and the vintage axis is the only way to see that.

The architectures
-----------------
==============================  ==========================  =========================
architecture                    pooling over the token axis  readout
==============================  ==========================  =========================
``linear_then_softmax``         each token by its own logit  linear   (deployed head)
``pre_mean``                    uniform ``1/L``              linear
``token_linear_then_softmax``   learned ``w[s]`` per index   linear
``token_mlp_then_softmax``      learned MLP over indices     linear
==============================  ==========================  =========================

The two baselines are refit here rather than read off
``results_.../vintage/vintage_summary.csv``: those fits went through
``ProbeFactory.build``, which does **not** restore the epoch early stopping selects
(``docs/attribution_findings.md`` §1), while everything routed through ``build_probe``
does. Mixing the two would compare an architecture against a different trainer.

What "vintage k" means is ``attribution_vintage``'s definition, unchanged: every row of
the iteration-3 dump whose originating red-team success was already present at iteration
k, carrying iteration 3's content. Vintages are not strictly nested and each is exactly
50/50 by construction — see that module's docstring for why.

Three things to hold in mind when reading the output
----------------------------------------------------
- **Position is measured from the start.** The blobs are right-padded, so index ``s`` is
  "the s-th token", and the last real token of a conversation sits at a different index
  in every row (median 300, max 1024). A positional map can therefore learn *ignore the
  opening turns*, but it cannot point at the assistant's reply. If position carries
  anything, an end-aligned variant is the follow-up, not a tweak.
- **Length is not divided out.** ``pre_mean`` divides by the token count and these heads
  do not, so a longer conversation yields a larger pooled vector and the logit can track
  length directly. The four eval splits have very different lengths (blob widths 121,
  144, 288, 859), so this is a live confound and is measured explicitly — see the
  logit/length correlation table, and ``--normalize`` for the arm that removes it.
- **The softmax downstream is a no-op.** At ``n_slots=1`` the pooled sequence has length
  1, so ``LinearThenSoftmax`` reduces to ``w . pooled + b``. That is the design: the
  request was to replace the mean with a learned token map and then read out.

Usage
-----
    .venv_claude/bin/python scripts/token_pool_vintage.py                  # 72 fits
    .venv_claude/bin/python scripts/token_pool_vintage.py --summarize-only
    .venv_claude/bin/python scripts/token_pool_vintage.py --check-padding  # seconds
    .venv_claude/bin/python scripts/token_pool_vintage.py --normalize      # control arm

Resumes at ``(arm, vintage, architecture, seed, variant)`` from
``token_pool_progress.jsonl``; every row carries that probe's fp32 logits on all 866 eval
rows, so every table below is re-derivable with ``--summarize-only`` and no refitting.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# tuberlens' trainer wraps every epoch in tqdm on stderr; 72 fits would bury the log.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arch_sweep as S
import attribution_lib as A
import attribution_vintage as V

OUT_DIR = A.REPO / "results_hu_harm_gemma27b_batch_ablation/token_pool"
PROGRESS = OUT_DIR / "token_pool_progress.jsonl"

ITERATION = 3
SEEDS = [42, 43, 44]
VINTAGES = [1, 2, 3]

#: Baselines first, so a run killed early still holds the comparison point. Within the
#: baselines, ``pre_mean`` is the one the new heads generalise — it is this same pooling
#: with the position weights frozen at ``1/L`` — and ``linear_then_softmax`` is what is
#: actually deployed.
ARCHITECTURES = [
    "linear_then_softmax",
    "pre_mean",
    "token_linear_then_softmax",
    "token_mlp_then_softmax",
]

#: The heads under test, as opposed to the two baselines they are read against.
TOKEN_ARCHES = ["token_linear_then_softmax", "token_mlp_then_softmax"]

#: Which architectures the ``--normalize`` control arm applies to. Dividing the pooled
#: sum by the token count makes the new heads length-invariant the way ``pre_mean`` is,
#: so the contrast isolates how much of any difference was length rather than position.
NORMALIZE_ARCHES = TOKEN_ARCHES


# --- eval-side extras ----------------------------------------------------------------


def eval_lengths(eval_dir: Path) -> dict[str, np.ndarray]:
    """Real token count per eval row, per split.

    Only needed for the logit/length diagnostic, and read straight off the cached blobs'
    attention masks — ``mmap=True`` so the 4.5 GB of activations sitting next to them in
    the same files are never faulted in.
    """
    out = {}
    for split in A.EVAL_SPLITS:
        blob = torch.load(
            eval_dir / f"{split}-acts_full.pt", weights_only=False, mmap=True
        )
        out[split] = blob["attention_mask"].sum(dim=1).cpu().numpy().astype(np.int64)
        del blob
    return out


def check_padding(eval_dir: Path, n_redteam: int = 24) -> int:
    """Assert the invariant the token pool rests on, on the real blobs.

    ``TokenPool`` reconciles its fixed 1024-wide position map with narrower tensors by
    slicing the weight instead of padding the input, which is exact only if padding is on
    the **right** and pad positions hold **exact zeros**. Both hold today across every
    blob in this repo, but neither is enforced anywhere, and if either broke the effect
    would be a probe scoring differently on a split purely because that split's blob is a
    different width — plausible numbers, no error. Cheap enough to re-check before
    committing to a multi-hour sweep.
    """
    import random

    bad = 0

    def check(path: Path, label: str) -> None:
        nonlocal bad
        blob = torch.load(path, weights_only=False, mmap=True)
        acts, mask = blob["activations"], blob["attention_mask"].bool()
        leading = int((~mask[:, 0]).sum())
        worst = 0.0
        for i in range(len(acts)):
            pad = ~mask[i]
            if pad.any():
                worst = max(worst, float(acts[i][pad].abs().max()))
        ok = leading == 0 and worst == 0.0
        bad += 0 if ok else 1
        print(f"  {'ok ' if ok else 'BAD'} {label:52s} width={acts.shape[1]:>5d} "
              f"rows_with_leading_pad={leading:<5d} max|act| at pad={worst}")
        del blob

    print("--- padding invariant: right-aligned, exact zeros ---")
    for split in A.EVAL_SPLITS:
        check(eval_dir / f"{split}-acts_full.pt", split)
    for path in A.base_blob_paths():
        check(path, path.name[:52])

    rng = random.Random(0)
    for arm in sorted(A.ARMS):
        ds = A.load_redteam_dataset(arm, ITERATION)
        picks = rng.sample(range(len(ds)), min(n_redteam, len(ds)))
        for i in picks:
            check(A.redteam_blob_path(ds.inputs[i]), f"{arm} red-team row {i}")
        del ds
    return bad


# --- progress ------------------------------------------------------------------------


def _key(row: dict) -> tuple:
    return (
        row["arm"],
        int(row["vintage"]),
        row["architecture"],
        int(row["seed"]),
        row.get("variant", ""),
    )


def read_progress(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # truncated by a hard kill; that fit simply re-runs
    return rows


# --- the sweep -----------------------------------------------------------------------


def build_split(asm, keep_rows: list[int]):
    """``(train, val)`` for one vintage, built **once** and reused by every architecture.

    ``attribution_refit.refit`` rebuilds this per fit, which is right when each fit drops
    a different row set. Here four architectures share a vintage, so rebuilding would pay
    the ~10 GB concatenation four times over for an identical result. Peak RSS is
    unchanged — it is ``asm.redteam`` (9.7 GB) plus these copies (~9.9 GB) either way,
    the measured 19.9 GB of ``arch_sweep.build_train_val`` — but three of every four
    rebuilds disappear.

    ``asm.redteam`` is **not** consumed: ``_concatenate_consuming`` pops the pad fields
    out of the parts it is handed, and an indexed selection is already a fresh copy, so
    the next vintage still finds the full set intact.
    """
    from agentic_redteam.retrain import _concatenate_consuming

    keep = set(keep_rows)
    tr = [i for i in range(len(asm.redteam)) if not asm.rt_is_val[i] and i in keep]
    va = [i for i in range(len(asm.redteam)) if asm.rt_is_val[i] and i in keep]
    train = _concatenate_consuming([asm.base_train[:], asm.redteam[tr]])
    val = _concatenate_consuming([asm.base_val[:], asm.redteam[va]])
    return train, val


def run_arm(arm: str, seeds: list[int], vintages: list[int], architectures: list[str],
            eval_dir: Path, resume: bool = True,
            variants: list[tuple[str, dict]] | None = None) -> None:
    import contextlib
    import io

    variants = variants or [("", {})]
    print(f"\n=== {arm} ===", flush=True)

    keep, report = V.vintages(arm, ITERATION)
    for k in vintages:
        v = report["vintages"][k]
        print(f"  vintage {k}: {v['n_rows']:4d} of the iter{ITERATION} rows "
              f"({v['n_pairs_kept']} of {v['n_pairs_in_dump']} iter{k} pairs survive)",
              flush=True)

    done = {_key(r) for r in read_progress(PROGRESS)} if resume else set()
    todo = [
        (s, k, arch, vname, vhp)
        for s in seeds
        for k in vintages
        for arch in architectures
        for vname, vhp in variants
        if (arm, k, arch, s, vname) not in done
    ]
    if not todo:
        print("  every (vintage, architecture, seed) already recorded", flush=True)
        return
    print(f"  {len(todo)} fit(s) to run", flush=True)

    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.probe_architectures import build_probe

    asm = V.assemble_train_only(arm, ITERATION)
    truth = S.load_eval_truth(asm.probe)

    # Seed outer, vintage next: after seed s lands, the sweep holds a COMPLETE
    # architecture x vintage grid at s seeds. Vintage before architecture so the shared
    # train/val split is built once per (seed, vintage) rather than once per fit.
    split_key: tuple | None = None
    train = val = None
    for seed, k, arch, vname, vhp in todo:
        if split_key != (seed, k):
            train = val = None
            gc.collect()
            t0 = time.time()
            train, val = build_split(asm, keep[k])
            split_key = (seed, k)
            print(f"  [vintage {k}] train={len(train)} val={len(val)} "
                  f"(built in {time.time() - t0:.0f}s)", flush=True)

        t0 = time.time()
        try:
            seed_everything(seed)
            with contextlib.redirect_stdout(io.StringIO()):
                probe = build_probe(
                    arch, train, val,
                    model_name=asm.probe.model_name,
                    layer=asm.probe.layer,
                    pos_class_label=asm.probe.pos_class_label,
                    neg_class_label=asm.probe.neg_class_label,
                    probe_description=asm.probe.description,
                    hyperparams=dict(vhp) or None,
                )
        except Exception as exc:  # noqa: BLE001 — one fit failing must not kill the sweep
            print(f"  seed {seed} v{k} {arch:28s} FAILED: {exc!r}", flush=True)
            _append(PROGRESS, {"arm": arm, "vintage": k, "architecture": arch,
                               "seed": seed, "variant": vname, "error": repr(exc)})
            continue
        fit_s = time.time() - t0

        t1 = time.time()
        logits = S.eval_logits(probe, eval_dir)
        score_s = time.time() - t1

        accs, aurocs = {}, {}
        for split in A.EVAL_SPLITS:
            s = np.array(logits[split])
            y = truth[split]["y"]
            accs[split] = float(((s >= 0) == (y > 0.5)).mean())
            aurocs[split] = A.auroc_both(y, s)
        mean_auc = float(np.mean([aurocs[sp]["pipeline"] for sp in A.EVAL_SPLITS]))
        n_params = int(sum(p.numel() for p in probe._classifier.model.parameters()))

        tag = f"{arch}{('/' + vname) if vname else ''}"
        print(f"  seed {seed} v{k} {tag:30s} fit {fit_s:6.1f}s score {score_s:4.1f}s "
              f"params={n_params:>7d} best_epoch="
              f"{getattr(probe._classifier, 'best_epoch', None)}  MEAN={mean_auc:.4f}",
              flush=True)
        print("     " + "  ".join(
            f"{sp.replace('eval_', ''):14s} acc={accs[sp]:.3f} "
            f"auc={aurocs[sp]['pipeline']:.4f}" for sp in A.EVAL_SPLITS), flush=True)

        _append(PROGRESS, {
            "arm": arm,
            "vintage": k,
            "architecture": arch,
            "variant": vname,
            "hyperparams": dict(vhp),
            "seed": seed,
            "n_redteam_rows": len(keep[k]),
            "n_train": len(train),
            "n_val": len(val),
            "n_params": n_params,
            "best_epoch": (
                None if getattr(probe._classifier, "best_epoch", None) is None
                else int(probe._classifier.best_epoch)
            ),
            "fit_seconds": fit_s,
            "accuracy": accs,
            "auroc": aurocs,
            "logits": logits,
        })
        del probe
        gc.collect()
        torch.cuda.empty_cache()

    del train, val, asm
    gc.collect()
    torch.cuda.empty_cache()


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# --- analysis ------------------------------------------------------------------------


def _fits(rows: list[dict], variant: str = "") -> list[dict]:
    return [r for r in rows if "error" not in r and r.get("variant", "") == variant]


def _mean_auc(r: dict) -> float:
    return float(np.mean([r["auroc"][sp]["pipeline"] for sp in A.EVAL_SPLITS]))


def print_vintage_table(rows: list[dict], variant: str = "") -> None:
    """Mean eval AUROC per (arm, vintage, architecture), the headline.

    Read **down** a vintage block to compare architectures at fixed data, and **across**
    the three blocks to see whether an architecture's standing depends on how much
    red-team data it was given.
    """
    by: dict = defaultdict(list)
    for r in _fits(rows, variant):
        by[(r["arm"], r["vintage"], r["architecture"])].append(_mean_auc(r))
    if not by:
        return
    label = f" (variant: {variant})" if variant else ""
    print(f"\n\n########## eval AUROC by vintage x architecture{label} ##########")
    for arm in sorted({k[0] for k in by}):
        print(f"\n  {arm}")
        print(f"    {'architecture':30s} " + " ".join(
            f"{'vintage ' + str(k):>19s}" for k in VINTAGES))
        for arch in ARCHITECTURES:
            cells = []
            for k in VINTAGES:
                v = np.array(by.get((arm, k, arch), []))
                if not len(v):
                    cells.append(f"{'-':>19s}")
                    continue
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                cells.append(f"{v.mean():.4f}+/-{sd:.4f} ({len(v)})".rjust(19))
            if any(c.strip() != "-" for c in cells):
                print(f"    {arch:30s} " + " ".join(cells))


def print_delta_table(rows: list[dict], variant: str = "") -> None:
    """Each token-pooled head against the two baselines, **paired by seed**.

    Unpaired means hide a small architecture effect behind the seed spread, which on
    these fits is +/-0.01-0.03 AUROC — comparable to the differences being looked for.
    Every architecture at a given (arm, vintage, seed) trains on the identical rows with
    the identical initial RNG state, so the per-seed difference is a genuinely paired
    contrast and its sd is the sd of the *difference*, not of the level.
    """
    idx: dict = {}
    for r in _fits(rows, variant):
        idx[(r["arm"], r["vintage"], r["architecture"], r["seed"])] = _mean_auc(r)
    pairs = [(new, base) for new in TOKEN_ARCHES
             for base in ("pre_mean", "linear_then_softmax")]
    print("\n\n########## paired deltas vs. the two baselines (same arm/vintage/seed) ##########")
    print("  positive = the token-pooled head scores higher on mean eval AUROC")
    for arm in sorted({k[0] for k in idx}):
        print(f"\n  {arm}")
        print(f"    {'contrast':56s} " + " ".join(
            f"{'vintage ' + str(k):>18s}" for k in VINTAGES))
        for new, base in pairs:
            cells = []
            for k in VINTAGES:
                d = [idx[(arm, k, new, s)] - idx[(arm, k, base, s)]
                     for s in SEEDS
                     if (arm, k, new, s) in idx and (arm, k, base, s) in idx]
                if not d:
                    cells.append(f"{'-':>18s}")
                    continue
                d = np.array(d)
                sd = d.std(ddof=1) if len(d) > 1 else 0.0
                cells.append(f"{d.mean():+.4f}+/-{sd:.4f}".rjust(18))
            if any(c.strip() != "-" for c in cells):
                print(f"    {new + '  -  ' + base:56s} " + " ".join(cells))


def print_per_split(rows: list[dict], variant: str = "") -> None:
    """Per-split AUROC at the full vintage, where the length confound is visible."""
    by: dict = defaultdict(lambda: defaultdict(list))
    for r in _fits(rows, variant):
        if r["vintage"] != max(VINTAGES):
            continue
        for sp in A.EVAL_SPLITS:
            by[(r["arm"], r["architecture"])][sp].append(r["auroc"][sp]["pipeline"])
    if not by:
        return
    print(f"\n\n########## per-split AUROC at vintage {max(VINTAGES)} ##########")
    for arm in sorted({k[0] for k in by}):
        print(f"\n  {arm}")
        print(f"    {'architecture':30s} " + " ".join(
            f"{sp.replace('eval_', ''):>17s}" for sp in A.EVAL_SPLITS))
        for arch in ARCHITECTURES:
            e = by.get((arm, arch))
            if not e:
                continue
            print(f"    {arch:30s} " + " ".join(
                f"{np.mean(e[sp]):.4f}+/-{np.std(e[sp], ddof=1) if len(e[sp]) > 1 else 0:.3f}"
                for sp in A.EVAL_SPLITS))


def print_length_confound(rows: list[dict], lengths: dict, truth: dict,
                          variant: str = "") -> None:
    """How much of each head's logit is just "how long was the conversation".

    The token pool sums over positions **without dividing by length**, so a longer
    conversation contributes a larger pooled vector and the logit can ride on length
    alone. That is not automatically wrong — length may genuinely correlate with the
    label — so the diagnostic reports both: the logit/length rank correlation, and the
    length/label correlation the probe would be entitled to exploit. A head whose
    logit/length correlation greatly exceeds the label's is reading the wrong signal.

    Spearman rather than Pearson, computed within each split and then averaged, because
    the splits have very different length distributions (blob widths 121 to 859) and a
    pooled correlation would mostly measure that between-split spread.
    """
    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = np.sqrt((ra**2).sum() * (rb**2).sum())
        return float((ra * rb).sum() / denom) if denom else 0.0

    fits = _fits(rows, variant)
    if not fits:
        return
    print("\n\n########## does the logit just track conversation length? ##########")
    print("  Spearman(logit, n_tokens) within each split, averaged over splits and seeds.")
    print("  The last column is Spearman(n_tokens, label) — the correlation a probe is")
    print("  entitled to exploit; a head far above it is reading length, not the concept.")
    ref = float(np.mean([
        spearman(lengths[sp].astype(float), truth[sp]["y"]) for sp in A.EVAL_SPLITS
    ]))
    by: dict = defaultdict(list)
    for r in fits:
        if r["vintage"] != max(VINTAGES):
            continue
        by[(r["arm"], r["architecture"])].append(float(np.mean([
            spearman(np.array(r["logits"][sp]), lengths[sp].astype(float))
            for sp in A.EVAL_SPLITS
        ])))
    for arm in sorted({k[0] for k in by}):
        print(f"\n  {arm}   [length vs. label: {ref:+.3f}]")
        for arch in ARCHITECTURES:
            v = by.get((arm, arch))
            if not v:
                continue
            v = np.array(v)
            print(f"    {arch:30s} {v.mean():+.3f}+/-"
                  f"{v.std(ddof=1) if len(v) > 1 else 0.0:.3f}")


def print_core_recovery(rows: list[dict], truth: dict, variant: str = "") -> None:
    """Of the 31 rows every heldout probe family got wrong, how many does each head fix?

    Same construction and the same two caveats as ``arch_sweep``: 31 rows is noisy (a
    chance-level ranking scores 15.5 +/- 2.8 under the balanced rule) and the core was
    defined by linear heads trained on this data, so any head from that family is
    expected to fail those rows on selection alone. It is a screen, not the headline.
    """
    y, split_all, _keys, offsets = S._flat_truth(truth)
    truth_bool = y > 0.5
    cores, info = S.load_hard_core()
    if not cores.get("balanced"):
        return
    mask = np.zeros(len(y), dtype=bool)
    for split, idx in cores["balanced"]:
        mask[offsets[split] + idx] = True

    print(f"\n\n########## hard-core recovery — {int(mask.sum())} rows, balanced rule "
          f"##########")
    by: dict = defaultdict(list)
    for r in _fits(rows, variant):
        if r["vintage"] != max(VINTAGES):
            continue
        flat = np.concatenate([np.array(r["logits"][sp]) for sp in A.EVAL_SPLITS])
        pred = np.zeros(len(flat), dtype=bool)
        for sp in A.EVAL_SPLITS:
            m = split_all == sp
            pred[m] = S._balanced_pred(flat[m], int(truth_bool[m].sum()))
        by[(r["arm"], r["architecture"])].append(int(((pred == truth_bool) & mask).sum()))
    for arm in sorted({k[0] for k in by}):
        print(f"\n  {arm}")
        for arch in ARCHITECTURES:
            v = by.get((arm, arch))
            if not v:
                continue
            v = np.array(v, dtype=float)
            print(f"    {arch:30s} {v.mean():5.1f}+/-"
                  f"{v.std(ddof=1) if len(v) > 1 else 0.0:.1f} of {int(mask.sum())}")
    print(f"\n  (heldout four-way cores: {info['balanced']})")


def print_variant_control(rows: list[dict]) -> None:
    """The ``--normalize`` arm against its own un-normalised fits, paired by seed."""
    idx: dict = defaultdict(dict)
    for r in rows:
        if "error" in r:
            continue
        idx[r.get("variant", "")][
            (r["arm"], r["vintage"], r["architecture"], r["seed"])
        ] = _mean_auc(r)
    if "lengthnorm" not in idx:
        return
    print("\n\n########## control: dividing the pooled sum by the token count ##########")
    print("  Paired against the same architecture/arm/vintage/seed without the division.")
    print("  A large positive delta means the un-normalised head was losing to length.")
    for arm in sorted({k[0] for k in idx["lengthnorm"]}):
        print(f"\n  {arm}")
        print(f"    {'architecture':30s} " + " ".join(
            f"{'vintage ' + str(k):>18s}" for k in VINTAGES))
        for arch in NORMALIZE_ARCHES:
            cells = []
            for k in VINTAGES:
                d = [idx["lengthnorm"][key] - idx[""][key]
                     for key in [(arm, k, arch, s) for s in SEEDS]
                     if key in idx["lengthnorm"] and key in idx[""]]
                if not d:
                    cells.append(f"{'-':>18s}")
                    continue
                d = np.array(d)
                sd = d.std(ddof=1) if len(d) > 1 else 0.0
                cells.append(f"{d.mean():+.4f}+/-{sd:.4f}".rjust(18))
            if any(c.strip() != "-" for c in cells):
                print(f"    {arch:30s} " + " ".join(cells))


def summarize(eval_dir: Path) -> None:
    rows = read_progress(PROGRESS)
    if not rows:
        print("no progress rows yet")
        return
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"\n{len(errs)} failed fit(s):")
        for r in errs[:10]:
            print(f"  {r['arm']} v{r['vintage']} {r['architecture']} "
                  f"seed {r['seed']}: {r['error']}")

    probe = A.load_probe(A.ARMS[sorted(A.ARMS)[0]] / f"probe_iter{ITERATION}.pkl")
    truth = S.load_eval_truth(probe)
    del probe
    gc.collect()
    lengths = eval_lengths(eval_dir)

    print_vintage_table(rows)
    print_delta_table(rows)
    print_per_split(rows)
    print_length_confound(rows, lengths, truth)
    print_core_recovery(rows, truth)
    print_variant_control(rows)
    print_vintage_table(rows, variant="lengthnorm")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_fits": len(rows) - len(errs),
        "n_errors": len(errs),
        "results": [
            {k: v for k, v in r.items() if k != "logits"} for r in rows
        ],
    }
    path = OUT_DIR / "token_pool.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--vintages", type=int, nargs="+", default=VINTAGES)
    ap.add_argument("--architectures", nargs="+", default=ARCHITECTURES)
    ap.add_argument("--eval-dir", type=Path, default=A.EVAL_ACTIVATIONS_DIR)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--check-padding", action="store_true",
                    help="verify the right-padding / exact-zero invariant TokenPool "
                         "rests on, then exit")
    ap.add_argument("--normalize", action="store_true",
                    help=f"control arm: re-run {NORMALIZE_ARCHES} dividing the pooled "
                         "sum by the token count, so they are length-invariant the way "
                         "pre_mean is")
    args = ap.parse_args()

    if args.check_padding:
        bad = check_padding(args.eval_dir)
        print("\nall blobs satisfy the invariant" if not bad
              else f"\n{bad} blob(s) VIOLATE it — TokenPool's weight slicing is unsafe")
        raise SystemExit(1 if bad else 0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        architectures = NORMALIZE_ARCHES if args.normalize else args.architectures
        variants = [("lengthnorm", {"normalize": True})] if args.normalize else None
        for arm in args.arm:
            run_arm(arm, args.seeds, args.vintages, architectures, args.eval_dir,
                    resume=not args.no_resume, variants=variants)
    summarize(args.eval_dir)


if __name__ == "__main__":
    main()
