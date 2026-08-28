#!/usr/bin/env python
"""Download google/gemma-3-27b-it into the local HF cache.

Gated model: needs HF_TOKEN (from .env) on an account whose licence request was
approved. Resumable — snapshot_download skips whatever is already complete, so a
killed run just re-invokes.
"""
import os, sys, time
from huggingface_hub import snapshot_download

REPO = "google/gemma-3-27b-it"
tok = os.environ.get("HF_TOKEN")
if not tok:
    sys.exit("HF_TOKEN not set (source .env first)")

t0 = time.monotonic()
path = snapshot_download(
    REPO,
    token=tok,
    max_workers=8,
    # weights + tokenizer + config; skip the .gguf/consolidated duplicates if any
    ignore_patterns=["*.gguf", "original/*"],
)
print(f"\n{REPO} -> {path}")
print(f"Elapsed {(time.monotonic() - t0) / 60:.1f} min")
