#!/usr/bin/env python
"""Time activation extraction with and without layer truncation.

Run this on the box that will do the real work, *before* committing to a multi-hour
retrain. It reports seconds/sample and where accelerate actually put the weights,
which is the number that decides whether a 770-sample retrain takes 40 minutes or 30
hours.

    .venv_claude/bin/python scripts/bench_extraction.py \
        --model google/gemma-3-27b-it --layer 32 --samples 5

    # sweep the batch size too (BATCH_SIZE is read by tuberlens' get_activations)
    BATCH_SIZE=4 .venv_claude/bin/python scripts/bench_extraction.py ...

Add --full to also time the untruncated model for a direct A/B. That doubles the run
and may disk-offload, so it is off by default.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path


def _placement(model) -> str:
    """Summarize accelerate's device_map: how many modules landed where."""
    device_map = getattr(model.model, "hf_device_map", None)
    if not device_map:
        try:
            devices = Counter(str(p.device) for p in model.model.parameters())
        except Exception:  # noqa: BLE001
            return "unknown"
        return ", ".join(f"{d}: {n} tensors" for d, n in sorted(devices.items()))
    counts = Counter(str(v) for v in device_map.values())
    return ", ".join(f"{d}: {n} modules" for d, n in sorted(counts.items()))


def _load_samples(path: Path | None, n: int):
    from tuberlens.interfaces.dataset import Message

    if path is None:
        # Roughly the length distribution of a real red-team conversation (~535 tokens).
        body = (
            "A patient in the ICU is deteriorating and the on-call physician is "
            "unreachable. Walk me through the escalation path in detail. "
        )
        return [
            [
                Message(role="user", content=body * (2 + i % 4)),
                Message(role="assistant", content=body * (2 + (i + 1) % 4)),
            ]
            for i in range(n)
        ]

    convos = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            raw = row.get("inputs") or row.get("sample") or row.get("messages")
            if isinstance(raw, str):
                raw = json.loads(raw)
            convos.append([Message(role=m["role"], content=m["content"]) for m in raw])
            if len(convos) >= n:
                break
    if not convos:
        raise SystemExit(f"No conversations parsed from {path}")
    return convos


def _time_one(model_name: str, layer: int, samples, *, truncate: bool) -> float:
    from agentic_redteam.model_loading import load_extraction_model

    os.environ["AGENTIC_REDTEAM_TRUNCATE_LAYERS"] = "1" if truncate else "0"

    label = "truncated" if truncate else "full"
    print(f"\n--- {label} ---", flush=True)
    t0 = time.monotonic()
    model = load_extraction_model(model_name, layer, verbose=True)
    load_s = time.monotonic() - t0
    print(f"  load:      {load_s:6.1f}s   layers={model.n_layers}  "
          f"batch_size={model.batch_size}")
    print(f"  placement: {_placement(model)}")

    # One warm-up sample: the first forward pays lazy CUDA init and, under offload,
    # the first cold read of every offloaded shard.
    model.get_activations(samples[:1], layer=layer, show_progress=False)

    t0 = time.monotonic()
    acts = model.get_activations(samples, layer=layer, show_progress=False)
    extract_s = time.monotonic() - t0
    per_sample = extract_s / len(samples)
    print(f"  extract:   {extract_s:6.1f}s for {len(samples)} samples "
          f"-> {per_sample:.2f} s/sample")
    print(f"  shape:     {tuple(acts.activations.shape)}")

    del model, acts
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--from-jsonl", type=Path, default=None,
                    help="Take conversations from a JSONL (red-team log or eval split) "
                         "instead of the synthetic ones, for realistic lengths.")
    ap.add_argument("--full", action="store_true",
                    help="Also time the untruncated model (slow; may disk-offload).")
    ap.add_argument("--retrain-size", type=int, default=770,
                    help="Sample count to extrapolate a retrain's wall-clock from.")
    args = ap.parse_args()

    import torch

    print(f"model={args.model}  layer={args.layer}  samples={args.samples}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={props.name}  vram={props.total_memory / 2**30:.1f} GiB")
    print(f"BATCH_SIZE={os.environ.get('BATCH_SIZE', '1 (default)')}")

    samples = _load_samples(args.from_jsonl, args.samples)

    truncated = _time_one(args.model, args.layer, samples, truncate=True)
    full = _time_one(args.model, args.layer, samples, truncate=False) if args.full else None

    print("\n=== extrapolated to one retrain "
          f"({args.retrain_size} samples) ===")
    print(f"  truncated: {truncated * args.retrain_size / 3600:6.2f} h")
    if full is not None:
        print(f"  full:      {full * args.retrain_size / 3600:6.2f} h"
              f"   (truncation is {full / truncated:.1f}x faster)")


if __name__ == "__main__":
    main()
