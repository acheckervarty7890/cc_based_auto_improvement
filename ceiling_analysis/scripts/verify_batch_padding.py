#!/usr/bin/env python
"""Check the cached red-team activations against a fresh extraction, row by row.

They must match **exactly**, and the reason is worth stating, because the fast way to
extract them does not.

`BATCH_SIZE=4` is ~8x faster than tuberlens' default of 1 — the CPU-offloaded half of
gemma-3-27b is streamed once per forward rather than once per conversation. But batched
extraction does not reproduce single-row extraction: measured with
`verify_extraction_noise.py`, a batch of 4 moves a conversation's activations by ~1e-2
relative L2, while a repeated single-row extraction moves them by exactly 0 and a local
single-row extraction reproduces the *published* eval/dev blobs by exactly 0 as well. It is
not the padding (gemma pads left, so batching shifts every real token's position) — a batch
of four copies of one conversation, which needs no padding at all, drifts by the same 1e-2.
It is bf16 matmul reduction order.

Which means the cheap knob was not free: it would have left every red-team activation ~1%
off the eval and dev activations the same probe is scored against, a difference applied to
exactly one side of the training data. So the cache is built at batch size 1, and this
asserts it stayed that way.
"""


from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="hu_ha")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from tuberlens.model import LLMModel

    from agentic_redteam.model_loading import load_extraction_model
    from agentic_redteam.retrain import (
        _apply_message_transforms,
        _redteam_activation_cache_path,
    )

    concept = C.CONCEPTS[args.concept]
    ds = C.load_jsonl_dataset(concept.redteam_jsonl, concept,
                              field_mapping={"label": "labels"})
    ds = _apply_message_transforms(ds, C.COMBINE, C.CONVERT)
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(len(ds), size=args.n, replace=False)

    model = load_extraction_model(C.MODEL_NAME, C.LAYER, verbose=True)
    worst = 0.0
    for i in rows:
        i = int(i)
        cached_path = _redteam_activation_cache_path(
            concept.redteam_cache_dir, ds.inputs[i], C.MODEL_NAME, C.LAYER,
            C.COMBINE, C.CONVERT
        )
        cached = LLMModel.load_activations(cached_path)
        keep = cached.attention_mask[0].bool()
        batched = cached.activations[0][keep].float()

        single = model.get_activations([ds.inputs[i]], layer=C.LAYER)
        keep1 = single.attention_mask[0].bool()
        solo = single.activations[0][keep1].float()

        if solo.shape != batched.shape:
            print(f"  row {i}: SHAPE MISMATCH {tuple(solo.shape)} vs {tuple(batched.shape)}",
                  flush=True)
            worst = float("inf")
            continue
        rel = float(
            (solo.double() - batched.double()).norm() / solo.double().norm()
        )
        worst = max(worst, rel)
        print(f"  row {i}: {tuple(solo.shape)}  relative L2 = {rel:.2e}", flush=True)

    print(f"\nworst relative deviation: {worst:.2e}", flush=True)
    if worst > 0:
        print("MISMATCH — the cache was not built at batch size 1 (or the model changed). "
              "Delete it and re-run extract_redteam_activations.py.", flush=True)
        return 1
    print("exact match: the cache reproduces a fresh single-row extraction", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
