#!/usr/bin/env python
"""Phase 1 - score all 45 probes on all four hu_ha eval splits.

Writes ``results/scores.npz``: a 45 x 866 probability matrix plus the row and probe
labels. Everything reads the cached full-split activation blobs, so gemma-3-27b is never
loaded and the whole phase is seconds.

The loop is inverted relative to the obvious one: a chunk of rows is materialized on the
GPU **once** and then scored by all 45 probes, rather than each probe walking the split.
The probes are 118 KB heads; the activations are 4.3 GB. Reading them 45 times would be
the entire cost of the phase.

The run's own comparison CSV is then reproduced from the matrix and the deviation
asserted. That check is the reason this phase exists as a file rather than a notebook: it
is what licenses every later claim about *which* rows are wrong.

    PROBE_FUSED_ENSEMBLE=0 analysis/persistent/score.py
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pe_common as PE  # noqa: E402

sys.path.insert(0, str(PE.REPO / "src"))
sys.path.insert(0, str(PE.REPO / "ceiling_analysis" / "scripts"))

CHUNK = int(os.environ.get("PE_CHUNK", 32))
# Above this the reproduction is not the same computation any more and the check should
# fail loudly rather than be widened. 4th-decimal AUROC drift is fp16 reduction order
# (chunk widths differ from tuberlens' own batching); the accuracy tolerance is one row of
# the smallest split, which is what a single cell sitting within 1e-3 of 0.5 costs.
TOL_AUROC = 5e-4
TOL_ACC = 1.0 / 134


def check_reproduction(P, y, split, arm, it) -> float:
    """Rebuild each run's published per-split AUROC/accuracy from the score matrix."""
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score

    worst_auroc = worst_acc = 0.0
    n = 0
    for run in PE.RUNS:
        pub = pd.read_csv(io.StringIO(PE.comparison_csv(run)))
        for k in np.where(arm == run.arm)[0]:
            for s in np.unique(split):
                row = pub[(pub["round"] == f"iter{it[k]}") & (pub["dataset"] == s)]
                if row.empty:
                    continue
                m = split == s
                worst_auroc = max(worst_auroc,
                                  abs(roc_auc_score(y[m], P[k, m]) - row["auroc"].iloc[0]))
                worst_acc = max(worst_acc,
                                abs(accuracy_score(y[m], P[k, m] > 0.5)
                                    - row["accuracy"].iloc[0]))
                n += 1
    total = len(arm) * len(np.unique(split))
    print(f"  reproduced {n} of {total} published (probe, split) cells "
          f"(the rest are iterations a run's CSV does not carry): "
          f"max |dAUROC| {worst_auroc:.2e}, max |dacc| {worst_acc:.2e}")
    assert worst_auroc < TOL_AUROC, f"AUROC drift {worst_auroc:.2e} > {TOL_AUROC}"
    assert worst_acc <= TOL_ACC + 1e-9, f"accuracy drift {worst_acc:.2e} > {TOL_ACC:.2e}"
    return worst_auroc


def main() -> int:
    if os.environ.get("PROBE_FUSED_ENSEMBLE") != "0":
        print("warning: PROBE_FUSED_ENSEMBLE is not 0; the runs scored sequentially and "
              "the reproduction check is calibrated to that", file=sys.stderr)
    import ca_common as C

    reg = PE.probe_registry()
    probes = [PE.load_probe(p) for *_, p in reg]
    print(f"loaded {len(probes)} probes "
          f"({sorted({len(getattr(p, 'members', [p])) for p in probes})}-member ensembles)",
          flush=True)

    srcs = C.eval_sources(C.CONCEPTS[PE.CONCEPT])
    names = sorted(srcs)
    P_parts, y_parts, split_col = [], [], []
    for s in names:
        src = srcs[s]
        n = len(src)
        y = C.labels_array(src.dataset)
        # length-sorted so a chunk is padded to its own longest row, not the split's
        order = np.argsort(src.lengths(), kind="stable")
        P = np.empty((len(probes), n))
        t0 = time.time()
        for start in range(0, n, CHUNK):
            pos = order[start:start + CHUNK]
            ds = src.take([int(i) for i in pos])
            [ds_d], _ = C.to_device([ds])
            for k, probe in enumerate(probes):
                P[k, pos] = np.asarray(probe.predict_proba(ds_d))
            del ds, ds_d
        C.free_gpu()
        print(f"  {s}: {n} rows x {len(probes)} probes in {time.time() - t0:.0f}s",
              flush=True)
        P_parts.append(P)
        y_parts.append(y)
        split_col += [s] * n

    P = np.concatenate(P_parts, axis=1)
    y = np.concatenate(y_parts).astype(int)
    split = np.array(split_col)
    arm = np.array([a for _, a, _, _ in reg])
    it = np.array([i for _, _, i, _ in reg])

    drift = check_reproduction(P, y, split, arm, it)
    PE.RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PE.RESULTS / "scores.npz", proba=P, labels=y, split=split,
        experiment=np.array([e for e, _, _, _ in reg]), arm=arm, iteration=it,
        reproduction_max_auroc_drift=np.array(drift))
    print(f"wrote {PE.RESULTS / 'scores.npz'} {P.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
