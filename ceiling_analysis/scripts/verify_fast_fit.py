#!/usr/bin/env python
"""Check the ragged fit path against the stock tuberlens fit, on real data.

`ca_fit` argues that trimming a batch to its own longest row cannot change the numbers,
because padded positions contribute exactly 0.0 to the aggregation. That argument is worth
exactly as much as a test of it. This fits the same data both ways — `ProbeFactory.build`
over dense, padded activations vs `train_head` over ragged ones — under the same seed, and
compares the resulting eval scores.

Run it on `hu_ha`, whose pools are small enough that the dense path is cheap.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca_common as C  # noqa: E402
import ca_data as D  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="hu_ha")
    ap.add_argument("--n-train", type=int, default=400)
    args = ap.parse_args()

    concept = C.CONCEPTS[args.concept]
    dev_src, val_idx, pool_idx = C.dev_partition(concept)
    rt_src = C.redteam_source(concept)
    rng = np.random.default_rng(7)
    sel = C.stratified_sample(C.source_labels(rt_src), args.n_train, rng)
    train = D.build_pool([(rt_src, sel)])
    val = dev_src.take(val_idx)
    [train_d, val_d], gpu = C.to_device([train, val])
    print(f"train {len(train)} rows, val {len(val)} rows, gpu={gpu}", flush=True)

    t0 = time.time()
    ref = C.fit_probe(train_d, val_d, concept, seed=C.FIT_SEED)
    t_ref = time.time() - t0
    t0 = time.time()
    fast = C.fit_probe_fast(train, val, concept, seed=C.FIT_SEED)
    t_fast = time.time() - t0

    eval_srcs = C.eval_sources(concept)
    diffs = []
    for name, src in eval_srcs.items():
        a = C.score_source(ref, src)
        b = C.score_source(fast, src)
        y = C.source_labels(src)
        m_a = C.metrics_from_scores(y, a)["auroc"]
        m_b = C.metrics_from_scores(y, b)["auroc"]
        diffs.append(abs(m_a - m_b))
        print(f"  {name:26s} reference AUROC {m_a:.6f}  ragged {m_b:.6f}  "
              f"max |dp| {np.abs(a - b).max():.2e}", flush=True)
    print(f"\nreference fit {t_ref:.1f}s   ragged fit {t_fast:.1f}s   "
          f"speedup {t_ref / max(t_fast, 1e-9):.1f}x", flush=True)
    print(f"max |AUROC difference| = {max(diffs):.2e}", flush=True)
    print(f"best_epoch: reference {ref._classifier.best_epoch}, "
          f"ragged {fast._classifier.best_epoch}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
