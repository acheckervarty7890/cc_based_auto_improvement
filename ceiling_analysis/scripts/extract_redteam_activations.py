#!/usr/bin/env python
"""Extract layer-32 gemma-3-27b activations for the red-team + base training conversations.

This is the only place in the analysis that loads the 27B model. Everything else runs on
blobs: the eval and dev activations come from Kaggle, and the conversations extracted here
land in the repo's own **content-keyed per-conversation** cache
(``retrain._redteam_activation_cache_path``), so the extraction is resumable — a crash
re-does only the rows that had not landed yet — and every later fit reads it back without a
model.

The extraction itself is `retrain._activate_redteam_cached`, unchanged: same chunking, same
per-row write-through, same width behaviour. The model is loaded through
`model_loading.load_extraction_model`, which truncates the stack to layers 0..32 — 33 of
gemma-3-27b's 62 layers, exactly (the stack is causal), and the difference between a load
that fits and one that offloads its executed tail to disk.

    ceiling_analysis/scripts/extract_redteam_activations.py --concepts hu_ha highstakes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca_common as C  # noqa: E402


def extract(concept: C.Concept, *, dry_run: bool) -> int:
    from agentic_redteam.retrain import (
        _activate_redteam_cached,
        _apply_message_transforms,
        _redteam_activation_cache_path,
    )

    parts = []
    for label, path, mapping in (
        ("redteam", concept.redteam_jsonl, {"label": "labels"}),
        ("base", concept.base_jsonl, None),
    ):
        ds = C.load_jsonl_dataset(path, concept, field_mapping=mapping)
        ds = _apply_message_transforms(ds, C.COMBINE, C.CONVERT)
        parts.append((label, ds))

    cache_dir = concept.redteam_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = 0
    for label, ds in parts:
        miss = sum(
            0
            if _redteam_activation_cache_path(
                cache_dir, m, C.MODEL_NAME, C.LAYER, C.COMBINE, C.CONVERT
            ).exists()
            else 1
            for m in ds.inputs
        )
        print(f"  {concept.name}/{label}: {len(ds)} rows, {miss} uncached", flush=True)
        todo += miss
    if dry_run or todo == 0:
        return todo

    from agentic_redteam.model_loading import load_extraction_model

    loaded = {"model": None}

    def get_model():
        if loaded["model"] is None:
            print("  loading extraction model ...", flush=True)
            t0 = time.time()
            loaded["model"] = load_extraction_model(C.MODEL_NAME, C.LAYER, verbose=True)
            print(f"  model loaded in {time.time() - t0:.0f}s", flush=True)
        return loaded["model"]

    for label, ds in parts:
        t0 = time.time()
        _activate_redteam_cached(
            ds, cache_dir, C.MODEL_NAME, C.LAYER, C.COMBINE, C.CONVERT, get_model, True
        )
        print(
            f"  {concept.name}/{label}: done in {time.time() - t0:.0f}s",
            flush=True,
        )
    loaded["model"] = None
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return todo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=list(C.CONCEPTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for name in args.concepts:
        print(f"=== {name} ===", flush=True)
        extract(C.CONCEPTS[name], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
