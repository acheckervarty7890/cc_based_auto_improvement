#!/usr/bin/env python
"""Check that batched extraction gives the same activations as one-at-a-time extraction.

The red-team activations here were extracted with `BATCH_SIZE=4`, which is ~8x faster than
tuberlens' default of 1 (the CPU-offloaded half of gemma-3-27b is streamed once per forward,
not once per conversation). But gemma's tokenizer pads **left**, so at batch > 1 a
conversation's real tokens no longer start at position 0 — while every published eval and
dev blob, extracted one row at a time, has no intra-batch padding at all.

Left padding should be harmless: attention masks the pad positions out entirely, and RoPE is
relative, so shifting all of a sequence's real tokens by the same offset leaves every
query-key distance (and every sliding-window span) unchanged. "Should" is not "does",
though, and if it were wrong every red-team activation in this analysis would be subtly
mismatched against the eval and dev sets. So this re-extracts a sample of conversations at
batch size 1 and compares them, token for token, against what the batched run stored.
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
        scale = solo.abs().mean().item()
        rel = (solo - batched).abs().max().item() / max(scale, 1e-6)
        cos = torch.nn.functional.cosine_similarity(
            solo.flatten(), batched.flatten(), dim=0
        ).item()
        worst = max(worst, rel)
        print(f"  row {i}: {tuple(solo.shape)}  max|delta|/mean|a| = {rel:.4f}  "
              f"cosine = {cos:.6f}", flush=True)

    print(f"\nworst relative deviation: {worst:.4f}", flush=True)
    print("fp16 storage alone gives ~1e-3 relative error, so anything at that scale is "
          "storage precision, not a padding effect.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
