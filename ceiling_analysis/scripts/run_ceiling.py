#!/usr/bin/env python
"""Ceiling analysis: how well can this probe family do on the eval sets at all?

The red-team loop trains a probe on conversations that are not drawn from the eval
distribution, and the run comparison CSVs report eval AUROC. To read those numbers you need
to know what the *achievable* eval AUROC is for this probe family — a `linear_then_softmax`
head over gemma-3-27b layer 32 — when the training data IS eval-distribution. That is what
this estimates, by k-fold cross-validation **inside** the eval sets:

    fold k:  train on the rows outside fold k
             early-stop on the reserved dev validation slice   <- identical to the sweep
             score fold k

Every eval row gets exactly one out-of-fold score, and the per-split metrics come from
those. Validation is the same fixed dev slice every sweep fit uses, so a ceiling probe and a
sweep probe differ *only* in their training data — which is the comparison the whole
analysis rests on.

Two details keep it honest:

* **The eval splits are never pooled into one tensor.** Each split keeps its own blob and
  its own width; training subsets and scoring both work per split and stream through the
  memory maps. Pooling `anthropic_hh_balanced` with the rest would cost ~48 GB for nothing.
* **A training-size ladder runs alongside.** A ceiling estimated from a training set that is
  itself too small is not a ceiling. `--train-sizes` repeats the CV at several sizes; if the
  top two agree, the estimate is saturated rather than data-limited.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca_common as C  # noqa: E402


def assign_folds(labels: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per row, stratified by label (splits are kept separate throughout)."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=int)
    for label in np.unique(labels):
        idx = rng.permutation(np.where(labels == label)[0])
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


def train_parts(eval_srcs, folds, labels, held_out: int, max_train: int, rng,
                extra=None):
    """`(source, rows)` for every split outside fold `held_out`, capped at max_train total.

    `extra` appends further in-distribution training data (the dev training pool) that is not
    part of any fold, so it is added whole to every fold and never scored.
    """
    avail = {n: np.where(folds[n] != held_out)[0] for n in eval_srcs}
    total = sum(len(v) for v in avail.values())
    take = min(max_train, total)
    parts = []
    for name, idx in avail.items():
        quota = int(round(take * len(idx) / total))
        if quota < len(idx):
            keep = C.stratified_sample(labels[name][idx], quota, rng)
            idx = idx[keep]
        parts.append((eval_srcs[name], list(idx)))
    if extra is not None:
        parts.append(extra)
    return parts


def run_concept(concept: C.Concept, args) -> dict:
    # An unnormalized run writes the plain filenames it always did; a `--norm` run writes
    # its own, so the study never overwrites the baseline it is measured against — and so
    # `--norm none` can be diffed against that baseline as a determinism check.
    suffix = "" if args.norm is None else f"__norm-{args.norm}"
    if args.fit_seed is not None:
        suffix += f"__fit{args.fit_seed}"
    out_path = C.RESULTS / f"ceiling_{concept.name}{suffix}.json"
    log_path = C.RESULTS / f"ceiling_{concept.name}{suffix}.jsonl"

    eval_srcs = C.eval_sources(concept)
    labels = {n: C.source_labels(s) for n, s in eval_srcs.items()}
    for n, s in eval_srcs.items():
        print(f"[{concept.name}] eval {n}: {len(s)} rows, max real length "
              f"{int(s.lengths().max())}", flush=True)

    dev_src, val_idx, pool_idx = C.dev_partition(concept)
    val_d = C.ragged_from_parts([(dev_src, val_idx)])
    print(f"[{concept.name}] validation: {len(val_idx)} dev rows "
          f"(packed {val_d.nbytes/1e9:.2f} GB); dev training pool {len(pool_idx)}",
          flush=True)

    folds = {n: assign_folds(labels[n], args.folds, args.seed) for n in eval_srcs}
    n_total = sum(len(s) for s in eval_srcs.values())
    sizes = args.train_sizes or [n_total]
    # An optional top rung that adds the dev training pool to every fold. The eval-only
    # ladder is bounded by the eval set itself — with 5 folds the largest training set is
    # 4/5 of it — so if the ladder is still climbing at the top rung, the estimate is a
    # lower bound rather than a ceiling. The dev pool is drawn from the same sources as the
    # eval splits and is disjoint from them, so adding it buys more in-distribution
    # training data without contaminating what is scored.
    rungs = [(size, None) for size in sizes]
    if args.add_dev_pool:
        rungs.append((sizes[-1], (dev_src, pool_idx)))

    results = {"concept": concept.name, "norm": C.NORM, "fit_seed": C.FIT_SEED,
               "n_eval_rows": n_total,
               "n_folds": args.folds, "n_validation": len(val_idx), "by_train_size": {}}

    for size, extra in rungs:
        tag = str(size) if extra is None else f"{size}+dev{len(extra[1])}"
        oof = {n: np.full(len(s), np.nan) for n, s in eval_srcs.items()}
        used = []
        for k in range(args.folds):
            t0 = time.time()
            rng = np.random.default_rng(args.seed * 1000 + k)
            parts = train_parts(eval_srcs, folds, labels, k, size, rng, extra=extra)
            n_train = sum(len(idx) for _, idx in parts)
            used.append(n_train)
            train = C.ragged_from_parts(parts)
            print(f"[{concept.name}] size={tag} fold {k}: {n_train} train rows "
                  f"(packed {train.nbytes/1e9:.2f} GB)", flush=True)
            probe = C.fit(train, val_d, concept, seed=C.FIT_SEED)
            del train
            C.free_gpu()
            for name, src in eval_srcs.items():
                idx = np.where(folds[name] == k)[0]
                if len(idx):
                    oof[name][idx] = C.score_source(probe, src, idx)
            del probe
            C.free_gpu()
            print(f"[{concept.name}] size={tag} fold {k} done in {time.time()-t0:.0f}s",
                  flush=True)

        per_split = {}
        for name, src in eval_srcs.items():
            assert not np.isnan(oof[name]).any(), f"{name}: unscored rows"
            per_split[name] = C.metrics_from_scores(labels[name], oof[name])
        mean = {m: float(np.mean([v[m] for v in per_split.values()]))
                for m in ("auroc", "accuracy", "tpr_at_fpr")}
        entry = {"train_rows_per_fold": used, "per_split": per_split, "mean": mean}
        results["by_train_size"][tag] = entry
        C.append_jsonl(log_path, {"concept": concept.name, "norm": C.NORM,
                                  "train_size": tag, **entry})
        print(f"[{concept.name}] size={tag}: MEAN eval AUROC {mean['auroc']:.4f} | "
              + " ".join(f"{k}={v['auroc']:.4f}" for k, v in per_split.items()), flush=True)

    top = list(results["by_train_size"])[-1]
    results["ceiling_rung"] = top
    results["ceiling"] = results["by_train_size"][top]["mean"]
    results["ceiling_per_split"] = {
        k: v["auroc"] for k, v in results["by_train_size"][top]["per_split"].items()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[{concept.name}] wrote {out_path}", flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=list(C.CONCEPTS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-sizes", nargs="*", type=int, default=None,
                    help="training rows per fold; the LAST rung is reported as the ceiling")
    ap.add_argument("--add-dev-pool", action="store_true",
                    help="add a top rung that also trains on the dev training pool")
    ap.add_argument("--fit-seed", type=int, default=None,
                    help="seed for the head's init and batch order (default 42). --seed "
                         "still governs the folds and the training-size subsampling, so "
                         "varying THIS re-fits the same data and measures run-to-run "
                         "noise. Also suffixes the output filenames.")
    ap.add_argument("--norm", choices=None, default=None,
                    help="normalization step in front of the head (see ca_norm.KINDS). "
                         "Omit for the unnormalized architecture every experiment used; "
                         "passing it also suffixes the output filenames.")
    args = ap.parse_args()
    if args.fit_seed is not None:
        C.FIT_SEED = args.fit_seed
    if args.norm is not None:
        import ca_norm

        if args.norm not in ca_norm.KINDS:
            ap.error(f"--norm must be one of {ca_norm.KINDS}")
        C.NORM = args.norm
    for name in args.concepts:
        run_concept(C.CONCEPTS[name], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
