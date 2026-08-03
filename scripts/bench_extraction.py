#!/usr/bin/env python
"""Time activation extraction with and without layer truncation.

Run this on the box that will do the real work, *before* committing to a multi-hour
retrain. It reports seconds/sample and where accelerate actually put the weights,
which is the number that decides whether a 770-sample retrain takes 40 minutes or 30
hours.

    .venv_claude/bin/python scripts/bench_extraction.py \
        --model google/gemma-3-27b-it --layer 32 --samples 5

    # realistic lengths matter — synthetic samples run short and flatter the result
    .venv_claude/bin/python scripts/bench_extraction.py \
        --from-jsonl results_.../deepseekv4pro_probing_fp.jsonl --samples 5

    # sweep the batch size (BATCH_SIZE is read by tuberlens' get_activations)
    BATCH_SIZE=4 .venv_claude/bin/python scripts/bench_extraction.py ...

Add --full to also time the untruncated model for a direct A/B.

**Each arm runs in its own subprocess.** This is not tidiness: ``device_map="auto"``
infers its GPU budget from memory *free at load time*, so a model still resident from
a previous arm makes the next one place everything on CPU — which for a 27B model is
minutes per sample and effectively uninterruptible (the forward sits in a C-level
matmul, so Ctrl-C does not land). Dropping the reference and calling
``torch.cuda.empty_cache()`` in-process is not sufficient. ``--abort-after`` is a
second guard: the first sample is timed alone and the arm gives up if it blows past
the threshold, rather than grinding through all of them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

_ARMS = ("truncated", "full")


def _free_vram() -> str:
    import torch

    if not torch.cuda.is_available():
        return "no cuda"
    free, total = torch.cuda.mem_get_info()
    return f"{free / 2**30:.1f} / {total / 2**30:.1f} GiB free"


def _placement(model) -> tuple[str, bool]:
    """Summarize accelerate's device_map, and flag a fully-CPU placement.

    A fully-CPU map is the contamination signature described in the module docstring —
    worth calling out loudly, because the arm will otherwise look merely "slow".
    """
    device_map = getattr(model.model, "hf_device_map", None)
    if not device_map:
        try:
            counts = Counter(str(p.device) for p in model.model.parameters())
        except Exception:  # noqa: BLE001
            return "unknown", False
        label = ", ".join(f"{d}: {n} tensors" for d, n in sorted(counts.items()))
        return label, not any(d.startswith("cuda") for d in counts)
    counts = Counter(str(v) for v in device_map.values())
    label = ", ".join(f"{d}: {n} modules" for d, n in sorted(counts.items()))
    on_gpu = any(d.isdigit() or d.startswith("cuda") for d in counts)
    return label, not on_gpu


def _load_samples(path: Path | None, n: int):
    from tuberlens.interfaces.dataset import Message

    if path is None:
        # Roughly the shape of a red-team conversation, but SHORT — real ones average
        # ~535 tokens, so treat synthetic timings as a lower bound.
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
            raw = row.get("sample") or row.get("inputs") or row.get("messages")
            if isinstance(raw, str):
                raw = json.loads(raw)
            if not raw:
                continue
            convos.append([Message(role=m["role"], content=m["content"]) for m in raw])
            if len(convos) >= n:
                break
    if not convos:
        raise SystemExit(f"No conversations parsed from {path}")
    return convos


def _run_arm(args) -> dict:
    """Time one arm in this process. Returns a result dict (printed as JSON)."""
    import torch

    from agentic_redteam.model_loading import load_extraction_model

    os.environ["AGENTIC_REDTEAM_TRUNCATE_LAYERS"] = (
        "1" if args.arm == "truncated" else "0"
    )
    samples = _load_samples(args.from_jsonl, args.samples)

    print(f"  vram before load: {_free_vram()}", flush=True)
    t0 = time.monotonic()
    model = load_extraction_model(args.model, args.layer, verbose=True)
    load_s = time.monotonic() - t0
    placement, all_cpu = _placement(model)
    print(f"  load:      {load_s:6.1f}s   layers={model.n_layers}  "
          f"batch_size={model.batch_size}")
    print(f"  placement: {placement}")
    if all_cpu:
        print("  !! nothing was placed on the GPU. Either the GPU is occupied (a "
              "previous model still resident) or the model cannot be split — "
              "extraction will be minutes per sample.", flush=True)

    # First sample alone: it pays lazy CUDA init and the first cold read of every
    # offloaded shard, and it is the cheapest place to notice a hopeless arm.
    t0 = time.monotonic()
    first = model.get_activations(samples[:1], layer=args.layer, show_progress=False)
    first_s = time.monotonic() - t0
    print(f"  warm-up:   {first_s:6.1f}s for 1 sample "
          f"({first.activations.shape[1]} tokens)", flush=True)
    if first_s > args.abort_after:
        print(f"  !! aborting this arm: {first_s:.0f}s > --abort-after "
              f"{args.abort_after}s. Extrapolated retrain: "
              f"{first_s * args.retrain_size / 3600:.1f} h.", flush=True)
        return {"arm": args.arm, "aborted": True, "per_sample": first_s,
                "placement": placement}

    t0 = time.monotonic()
    acts = model.get_activations(samples, layer=args.layer, show_progress=False)
    extract_s = time.monotonic() - t0
    per_sample = extract_s / len(samples)
    print(f"  extract:   {extract_s:6.1f}s for {len(samples)} samples "
          f"-> {per_sample:.2f} s/sample")
    print(f"  shape:     {tuple(acts.activations.shape)}  "
          f"({acts.activations.shape[1]} tokens/sample after padding)")

    del model, acts, first
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"arm": args.arm, "aborted": False, "per_sample": per_sample,
            "placement": placement}


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
                    help="Also time the untruncated model, in its own subprocess.")
    ap.add_argument("--retrain-size", type=int, default=770,
                    help="Sample count to extrapolate a retrain's wall-clock from.")
    ap.add_argument("--abort-after", type=float, default=120.0,
                    help="Give up on an arm whose first sample exceeds this many "
                         "seconds (default 120).")
    ap.add_argument("--arm", choices=_ARMS, default=None,
                    help=argparse.SUPPRESS)  # internal: run a single arm in-process
    args = ap.parse_args()

    if args.arm is not None:
        result = _run_arm(args)
        print("__RESULT__" + json.dumps(result))
        return

    import torch

    print(f"model={args.model}  layer={args.layer}  samples={args.samples}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={props.name}  vram={props.total_memory / 2**30:.1f} GiB")
    print(f"BATCH_SIZE={os.environ.get('BATCH_SIZE', '1 (default)')}")
    if args.from_jsonl is None:
        print("samples=synthetic (short — real conversations average ~535 tokens; "
              "pass --from-jsonl for a representative number)")

    results: dict[str, dict] = {}
    for arm in _ARMS if args.full else ("truncated",):
        print(f"\n--- {arm} (subprocess) ---", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--arm", arm,
               "--model", args.model, "--layer", str(args.layer),
               "--samples", str(args.samples),
               "--retrain-size", str(args.retrain_size),
               "--abort-after", str(args.abort_after)]
        if args.from_jsonl is not None:
            cmd += ["--from-jsonl", str(args.from_jsonl)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if line.startswith("__RESULT__"):
                results[arm] = json.loads(line[len("__RESULT__"):])
            else:
                print(line)
        if proc.returncode != 0:
            print(f"  arm failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")

    print(f"\n=== extrapolated to one retrain ({args.retrain_size} samples) ===")
    for arm in _ARMS:
        r = results.get(arm)
        if r is None:
            continue
        note = "  (ABORTED — lower bound)" if r["aborted"] else ""
        print(f"  {arm:10s} {r['per_sample']:7.2f} s/sample  -> "
              f"{r['per_sample'] * args.retrain_size / 3600:6.2f} h{note}")
    if len(results) == 2 and not results["truncated"]["aborted"]:
        speedup = results["full"]["per_sample"] / results["truncated"]["per_sample"]
        print(f"  truncation is {speedup:.1f}x faster"
              + ("  (full arm aborted, so this is a lower bound)"
                 if results["full"]["aborted"] else ""))


if __name__ == "__main__":
    main()
