#!/usr/bin/env python
"""Download google/gemma-3-27b-it into the local HF cache (resumable)."""
import os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
tok = None
tf = REPO / "hf_token.txt"
if tf.is_file():
    for line in tf.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip().upper() in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
                tok = v.strip().strip("'\"")
        elif line.startswith("hf_"):
            tok = line
if tok:
    os.environ["HF_TOKEN"] = tok

from huggingface_hub import snapshot_download

REPO_ID = "google/gemma-3-27b-it"
t0 = time.time()
for attempt in range(1, 21):
    try:
        p = snapshot_download(
            REPO_ID,
            token=tok,
            max_workers=8,
            ignore_patterns=["*.gguf", "*.msgpack", "*.h5", "*.onnx"],
        )
        print(f"DONE {REPO_ID} -> {p}  ({time.time()-t0:.0f}s)", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(min(60, 5 * attempt))
print("FAILED after retries", flush=True)
sys.exit(1)
