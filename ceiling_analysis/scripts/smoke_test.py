#!/usr/bin/env python
"""End-to-end shakedown of the analysis on tiny data, on whichever device is free.

Exercises the parts that are cheap to get wrong and expensive to discover late: the
source/partition plumbing, the nested stratified draws, the ragged packing, a (3-epoch)
fit, a fine-tune, scoring, the JSONL result log and its resume keys. It does not produce a
result worth reading — it produces a stack trace or nothing.
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca_common as C  # noqa: E402
import ca_fit as F  # noqa: E402
import run_sweep as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="hu_ha")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-train", type=int, default=48)
    args = ap.parse_args()

    F.DEVICE = args.device
    F.DTYPE = torch.float32 if args.device == "cpu" else torch.bfloat16
    # tuberlens' own scoring path builds its `ActivationDataset` on `global_settings.DEVICE`
    # regardless of where the probe lives, so a CPU run has to move that too.
    from tuberlens.config import global_settings

    global_settings.DEVICE = F.DEVICE
    global_settings.DTYPE = F.DTYPE
    concept = C.CONCEPTS[args.concept]

    hp = C.hyperparams()
    hp.update(epochs=3, patience=3)
    C.hyperparams = lambda: dict(hp)

    dev_src, val_idx, pool_idx = C.dev_partition(concept)
    rt_src = C.redteam_source(concept)
    print(f"dev {len(dev_src)} (val {len(val_idx)}, pool {len(pool_idx)}), "
          f"red-team+base {len(rt_src)}", flush=True)

    rng = np.random.default_rng(0)
    order = S.stratified_order(C.source_labels(dev_src)[pool_idx],
                               [dev_src.dataset.other_fields["dev_split"][i] for i in pool_idx],
                               rng)
    points = C.sweep_points(len(pool_idx))
    print("points", points, flush=True)
    # nesting: every prefix must be contained in the next
    for a, b in zip(points, points[1:]):
        assert set(order[:a]) <= set(order[:b]), "dev subsets are not nested"
    # balance: the halfway prefix should stay ~balanced
    y = C.source_labels(dev_src)[pool_idx]
    mid = points[len(points) // 2]
    frac = y[order[:mid]].mean()
    assert 0.4 <= frac <= 0.6, f"prefix is unbalanced: {frac}"
    print(f"nesting + balance ok (positive fraction at N={mid}: {frac:.3f})", flush=True)

    rt_sel = C.stratified_sample(C.source_labels(rt_src), args.n_train, rng)
    val = C.ragged_from_parts([(dev_src, val_idx[:24])])
    train = C.ragged_from_parts([(rt_src, rt_sel)])
    print(f"packed train {train.nbytes/1e6:.0f} MB, val {val.nbytes/1e6:.0f} MB", flush=True)

    probe = C.fit(train, val, concept)
    print("fit ok; val AUROC", round(C.ragged_val_auroc(probe, val), 4), flush=True)

    dev_train = C.ragged_from_parts([(dev_src, pool_idx[:24])])
    # the sweep fine-tunes a *copy* of the stage-1 probe once per point, so the copy has to
    # be independent — a shared module would let each point train on top of the last
    clone = copy.deepcopy(probe)
    before = C.ragged_val_auroc(probe, val)
    probe2, info = C.finetune(clone, dev_train, val)
    after_original = C.ragged_val_auroc(probe, val)
    assert before == after_original, "fine-tuning the copy mutated the original probe"
    print("finetune (on a deepcopy) ok;", info, flush=True)

    name, src = next(iter(C.eval_sources(concept).items()))
    idx = np.arange(min(32, len(src)))
    scores = C.score_source(probe2, src, idx)
    assert scores.shape == idx.shape and np.isfinite(scores).all()
    print(f"scored {name}[{len(idx)}] ok, mean p={scores.mean():.3f}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.jsonl"
        C.append_jsonl(path, {"concept": concept.name, "arm": "mixed", "dev_seed": 0,
                              "n_dev": 0, "mean": {"auroc": 0.5}})
        keys = C.done_keys(path, S.KEY_FIELDS)
        assert (concept.name, "mixed", 0, 0) in keys, keys
    print("result log + resume keys ok", flush=True)
    print("\nSMOKE TEST PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
