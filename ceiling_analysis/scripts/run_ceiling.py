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
import ca_data as D  # noqa: E402


def assign_folds(labels: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per row, stratified by label (splits are kept separate throughout)."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=int)
    for label in np.unique(labels):
        idx = rng.permutation(np.where(labels == label)[0])
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


def build_train(eval_srcs, folds, labels, held_out: int, max_train: int, rng):
    """Rows of every split outside fold `held_out`, subsampled to <= max_train in total."""
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
    return D.build_pool(parts)


def score_source(probe, src, idx, chunk: int = 64) -> np.ndarray:
    """Score selected rows of a source, streaming so no split is ever materialized whole."""
    out = np.empty(len(idx), dtype=float)
    for start in range(0, len(idx), chunk):
        sl = list(idx[start : start + chunk])
        ds = src.take(sl)
        [ds_d], _ = C.to_device([ds])
        out[start : start + len(sl)] = np.asarray(probe.predict_proba(ds_d))
        del ds, ds_d
    return out


def run_concept(concept: C.Concept, args) -> dict:
    out_path = C.RESULTS / f"ceiling_{concept.name}.json"
    log_path = C.RESULTS / f"ceiling_{concept.name}.jsonl"

    eval_srcs = C.eval_sources(concept)
    labels = {n: C.source_labels(s) for n, s in eval_srcs.items()}
    for n, s in eval_srcs.items():
        print(f"[{concept.name}] eval {n}: {len(s)} rows, max real length "
              f"{int(s.lengths().max())}", flush=True)

    dev_src, val_idx, pool_idx = C.dev_partition(concept)
    dev_val = dev_src.take(val_idx)
    [val_d], val_gpu = C.to_device([dev_val])
    print(f"[{concept.name}] validation: {len(dev_val)} dev rows "
          f"({C.nbytes(dev_val)/1e9:.2f} GB, gpu={val_gpu}); dev training pool "
          f"{len(pool_idx)}", flush=True)

    folds = {n: assign_folds(labels[n], args.folds, args.seed) for n in eval_srcs}
    n_total = sum(len(s) for s in eval_srcs.values())
    sizes = args.train_sizes or [n_total]

    results = {"concept": concept.name, "n_eval_rows": n_total, "n_folds": args.folds,
               "n_validation": len(dev_val), "by_train_size": {}}

    for size in sizes:
        oof = {n: np.full(len(s), np.nan) for n, s in eval_srcs.items()}
        used = []
        for k in range(args.folds):
            t0 = time.time()
            rng = np.random.default_rng(args.seed * 1000 + k)
            train = build_train(eval_srcs, folds, labels, k, size, rng)
            used.append(len(train))
            [train_d], gpu = C.to_device([train])
            print(f"[{concept.name}] size={size} fold {k}: {len(train)} train rows "
                  f"({C.nbytes(train)/1e9:.2f} GB, gpu={gpu})", flush=True)
            probe = C.fit_probe(train_d, val_d, concept, seed=C.FIT_SEED)
            del train_d, train
            C.free_gpu()
            for name, src in eval_srcs.items():
                idx = np.where(folds[name] == k)[0]
                if len(idx):
                    oof[name][idx] = score_source(probe, src, idx)
            del probe
            C.free_gpu()
            print(f"[{concept.name}] size={size} fold {k} done in {time.time()-t0:.0f}s",
                  flush=True)

        per_split = {}
        for name, src in eval_srcs.items():
            assert not np.isnan(oof[name]).any(), f"{name}: unscored rows"
            per_split[name] = C.metrics_from_scores(labels[name], oof[name])
        mean = {m: float(np.mean([v[m] for v in per_split.values()]))
                for m in ("auroc", "accuracy", "tpr_at_fpr")}
        entry = {"train_rows_per_fold": used, "per_split": per_split, "mean": mean}
        results["by_train_size"][str(size)] = entry
        C.append_jsonl(log_path, {"concept": concept.name, "train_size": size, **entry})
        print(f"[{concept.name}] size={size}: MEAN eval AUROC {mean['auroc']:.4f} | "
              + " ".join(f"{k}={v['auroc']:.4f}" for k, v in per_split.items()), flush=True)

    top = str(sizes[-1])
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
                    help="training rows per fold; the LAST one is reported as the ceiling")
    args = ap.parse_args()
    for name in args.concepts:
        run_concept(C.CONCEPTS[name], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
