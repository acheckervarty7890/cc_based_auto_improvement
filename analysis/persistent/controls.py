#!/usr/bin/env python
"""Phase 2 - in-distribution controls, per eval row.

The always-wrong set means two very different things depending on the answer to one
question: can a probe of this family get these rows right *at all* when its training data
is drawn from the eval distribution? So two controls score every eval row:

``dev_only``  one probe trained on the dev training pool alone. ``dev_samples/hu_ha`` is
              drawn from the same four sources as the eval splits and is verified disjoint
              from them, so this is in-distribution training with no leakage.
``ceiling``   5-fold CV inside the eval set plus the dev pool, every row scored by the
              fold that did not train on it. This is the ceiling study's own estimate of
              what the family can reach; it is re-run here only because that study kept
              aggregates and this one needs the per-row scores.

Both use ``ca_common``'s fit unchanged — one ``linear_then_softmax`` head at seed 42,
early-stopped on the ceiling study's reserved 25% dev slice — so the numbers land on its
curves. They are single probes, not the runs' 10-member ensembles, which is the right
comparison for "is this row learnable" and the wrong one for "is this probe good".

    analysis/persistent/controls.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pe_common as PE  # noqa: E402

sys.path.insert(0, str(PE.REPO / "ceiling_analysis" / "scripts"))

FOLDS = 5
FOLD_SEED = 42


def main() -> int:
    import ca_common as C
    import run_ceiling as RC

    concept = C.CONCEPTS[PE.CONCEPT]
    srcs = C.eval_sources(concept)
    names = sorted(srcs)
    labels = {n: C.source_labels(srcs[n]) for n in names}
    dev_src, val_idx, pool_idx = C.dev_partition(concept)
    val_d = C.ragged_from_parts([(dev_src, val_idx)])
    print(f"dev: {len(val_idx)} validation / {len(pool_idx)} training pool", flush=True)

    t0 = time.time()
    probe = C.fit(C.ragged_from_parts([(dev_src, pool_idx)]), val_d, concept,
                  seed=C.FIT_SEED)
    dev_only = {n: C.score_source(probe, srcs[n]) for n in names}
    del probe
    C.free_gpu()
    print(f"dev_only: fit + scored in {time.time() - t0:.0f}s", flush=True)

    folds = {n: RC.assign_folds(labels[n], FOLDS, FOLD_SEED) for n in names}
    n_total = sum(len(srcs[n]) for n in names)
    oof = {n: np.full(len(srcs[n]), np.nan) for n in names}
    for k in range(FOLDS):
        t0 = time.time()
        rng = np.random.default_rng(FOLD_SEED * 1000 + k)
        parts = RC.train_parts(srcs, folds, labels, k, n_total, rng,
                               extra=(dev_src, pool_idx))
        probe = C.fit(C.ragged_from_parts(parts), val_d, concept, seed=C.FIT_SEED)
        C.free_gpu()
        for n in names:
            idx = np.where(folds[n] == k)[0]
            if len(idx):
                oof[n][idx] = C.score_source(probe, srcs[n], idx)
        del probe
        C.free_gpu()
        print(f"ceiling fold {k}: {time.time() - t0:.0f}s", flush=True)

    per_split = {}
    for n in names:
        assert not np.isnan(oof[n]).any(), f"{n}: unscored rows"
        per_split[n] = C.metrics_from_scores(labels[n], oof[n])
        print(f"  ceiling {n:22s} auroc {per_split[n]['auroc']:.4f} "
              f"acc {per_split[n]['accuracy']:.4f}")

    PE.RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PE.RESULTS / "controls.npz",
        dev_only=np.concatenate([dev_only[n] for n in names]),
        ceiling=np.concatenate([oof[n] for n in names]),
        split=np.array([n for n in names for _ in range(len(srcs[n]))]))
    PE.write_json(PE.RESULTS / "controls.json",
                  {"n_folds": FOLDS, "fold_seed": FOLD_SEED, "fit_seed": C.FIT_SEED,
                   "n_dev_validation": len(val_idx), "n_dev_pool": len(pool_idx),
                   "ceiling_per_split": per_split,
                   "ceiling_mean_auroc": float(np.mean([v["auroc"]
                                                        for v in per_split.values()]))})
    print("wrote", PE.RESULTS / "controls.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
