#!/usr/bin/env python
"""Ceiling analysis: how well can this probe family do on the eval sets at all?

The red-team loop trains a probe on conversations that are not drawn from the eval
distribution, and the run comparison CSVs report eval AUROC. To read those numbers you need
to know what the *achievable* eval AUROC is for this probe family — a `linear_then_softmax`
head over gemma-3-27b layer 32 — when the training data IS eval-distribution. That is what
this estimates, by k-fold cross-validation **inside** the eval sets:

    fold k:  train on the other k-1 folds (eval rows)
             early-stop on the reserved dev validation slice   <- identical to the sweep
             score fold k

Every eval row therefore gets exactly one out-of-fold score, and the per-split metrics are
computed from those. Validation is the same fixed dev slice every sweep fit uses, so a
ceiling probe and a sweep probe differ *only* in their training data.

Two details that keep this honest:

* **The eval splits are never pooled into one tensor.** Each split keeps its own padded
  width (anthropic_hh_balanced alone is thousands of rows), and both the CV training subsets
  and the scoring work per split. Pooling would cost tens of GB for nothing.
* **A training-size ladder runs alongside.** A ceiling estimated from a training set that is
  itself too small is not a ceiling. `--train-sizes` fits the same CV at several training
  sizes; if the top two agree, the estimate is saturated rather than data-limited.
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


def assign_folds(ds, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per row, stratified by label (the splits are handled separately)."""
    rng = np.random.default_rng(seed)
    y = C.labels_array(ds)
    folds = np.empty(len(ds), dtype=int)
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        idx = rng.permutation(idx)
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


def build_train(eval_splits, folds, held_out: int, max_train: int, rng):
    """Rows of every split outside fold `held_out`, subsampled to <= max_train in total."""
    avail = {name: np.where(folds[name] != held_out)[0] for name in eval_splits}
    total = sum(len(v) for v in avail.values())
    take = min(max_train, total)
    parts = []
    for name, idx in avail.items():
        sub = eval_splits[name][[int(i) for i in idx]]
        quota = int(round(take * len(idx) / total))
        if quota < len(sub):
            keep = C.stratified_sample(sub, quota, rng)
            sub = sub[keep]
        parts.append(sub)
    return C.trim(C.pool(parts))


def run_concept(concept: C.Concept, args) -> dict:
    out_path = C.RESULTS / f"ceiling_{concept.name}.json"
    log_path = C.RESULTS / f"ceiling_{concept.name}.jsonl"

    print(f"[{concept.name}] loading eval splits ...", flush=True)
    eval_splits = {k: C.trim(v) for k, v in C.load_eval_splits(concept).items()}
    for k, v in eval_splits.items():
        print(f"   {k}: {len(v)} rows, {C.nbytes(v)/1e9:.2f} GB", flush=True)
    dev_val, dev_pool, _ = C.dev_partition(concept)
    print(f"   dev validation slice: {len(dev_val)} rows "
          f"({C.nbytes(dev_val)/1e9:.2f} GB); dev training pool {len(dev_pool)}", flush=True)

    folds = {name: assign_folds(ds, args.folds, args.seed) for name, ds in eval_splits.items()}
    n_total = sum(len(v) for v in eval_splits.values())
    sizes = args.train_sizes or [n_total]

    results = {"concept": concept.name, "n_eval_rows": n_total,
               "n_folds": args.folds, "by_train_size": {}}

    for size in sizes:
        oof = {name: np.full(len(ds), np.nan) for name, ds in eval_splits.items()}
        used_train = []
        for k in range(args.folds):
            t0 = time.time()
            rng = np.random.default_rng(args.seed * 1000 + k)
            train = build_train(eval_splits, folds, k, size, rng)
            used_train.append(len(train))
            val = dev_val
            [train_d, val_d], on_gpu = C.to_device([train, val])
            print(f"[{concept.name}] size={size} fold {k}: train {len(train)} rows "
                  f"({C.nbytes(train)/1e9:.2f} GB, gpu={on_gpu})", flush=True)
            probe = C.fit_probe(train_d, val_d, concept, seed=C.FIT_SEED)
            for name, ds in eval_splits.items():
                idx = np.where(folds[name] == k)[0]
                if len(idx) == 0:
                    continue
                sub = ds[[int(i) for i in idx]]
                [sub_d], _ = C.to_device([sub])
                oof[name][idx] = np.asarray(probe.predict_proba(sub_d))
                del sub_d, sub
            del train_d, val_d, train, probe
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{concept.name}] size={size} fold {k} done in "
                  f"{time.time()-t0:.0f}s", flush=True)

        per_split = {}
        for name, ds in eval_splits.items():
            assert not np.isnan(oof[name]).any(), f"{name}: unscored rows"
            per_split[name] = C.metrics_from_scores(C.labels_array(ds), oof[name])
        mean = {m: float(np.mean([v[m] for v in per_split.values()]))
                for m in ("auroc", "accuracy", "tpr_at_fpr")}
        entry = {"train_rows_per_fold": used_train, "per_split": per_split, "mean": mean}
        results["by_train_size"][str(size)] = entry
        C.append_jsonl(log_path, {"concept": concept.name, "train_size": size, **entry})
        print(f"[{concept.name}] size={size}: mean eval AUROC {mean['auroc']:.4f} "
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
                    help="training-set sizes per fold (the last one is the ceiling)")
    args = ap.parse_args()
    for name in args.concepts:
        run_concept(C.CONCEPTS[name], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
