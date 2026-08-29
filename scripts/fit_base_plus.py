#!/usr/bin/env python
"""Fit `base ∪ <jsonl>` — the accepted 62 deliberately left out — and score dev + eval.

Every other experiment in this arm floors on `base ∪ 62 accepted`, so a generated set is
always measured as an increment on top of the loop's own output. This asks what a set is
worth *instead of* the accepted rows rather than *in addition to* them.

Reference points, same setup:

    base only (50 rows)              dev 0.7573   eval 0.7779
    62 accepted only (no base)       dev 0.6250   eval 0.6172
    base ∪ 62 (probe_iter13)         dev 0.8311   eval 0.8148
    base ∪ 62 ∪ slot1 (88 rows)      dev 0.8657   eval 0.8483

`--no-base` drops the base data too, fitting the samples alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PROBE_DIR = REPO / "probes/gen_gemma27b_instructions_nemotron"
BASE_DATA = REPO / "data/instructions_llama70b_50.jsonl"
DEV_DATA = REPO / "dev_samples/instructions"
EVAL_DIR = REPO / "eval_sets/instructions"
BASE_CACHE = REPO / "cache_gen_gemma27b_instructions/base_activations"
EVAL_CACHE = REPO / "cache_gen_gemma27b_instructions/eval_activations"
BASE_PROBE = PROBE_DIR / "probe_iter12.pkl"  # byte-identical to probe_iter13

SEED, COMBINE, CONVERT = 42, True, True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", type=Path, help="JSONL of {inputs, labels} rows")
    ap.add_argument("--no-base", action="store_true", help="fit the samples alone")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import retrain_probe, warm_sample_activation_cache

    rows = [json.loads(l) for l in args.samples.open() if l.strip()]
    rows = [{"inputs": r["inputs"], "labels": r["labels"]} for r in rows]
    npos = sum(1 for r in rows if r["labels"].endswith("follows_the_instruction"))
    label = ("no base ∪ " if args.no_base else "base ∪ ") + f"{len(rows)} ({args.samples.name})"
    print(f"{label}: {npos} pos / {len(rows) - npos} neg")

    out = args.out or PROBE_DIR / f"baseplus_{args.samples.stem}.pkl"
    warm_sample_activation_cache(rows, base_probe_path=BASE_PROBE,
                                 base_activation_cache_dir=BASE_CACHE,
                                 combine_consecutive_messages=COMBINE,
                                 convert_tool_to_assistant=CONVERT, verbose=False)
    res = retrain_probe(
        samples=rows, base_probe_path=BASE_PROBE,
        base_training_data_path=None if args.no_base else BASE_DATA,
        new_probe_path=out, dev_data_path=DEV_DATA, seed=SEED, base_data_fraction=1.0,
        base_activation_cache_dir=BASE_CACHE, combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT, verbose=False,
    )
    df = evaluate_probe(out, EVAL_DIR, EVAL_CACHE, max_samples=None, seed=SEED,
                        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT)
    ev = float(df.loc[df["dataset"] == "mean", "auroc"].iloc[0])
    print(f"\n{res.n_training_samples_total} training rows")
    print(f"  dev  mean {res.dev_auroc['mean']:.5f}")
    for k, v in sorted(res.dev_auroc.items()):
        if k != "mean":
            print(f"      {k:<28} {v:.5f}")
    print(f"  eval mean {ev:.5f}")
    for _, r in df[df["dataset"] != "mean"].iterrows():
        print(f"      {r['dataset']:<28} {r['auroc']:.5f}")


if __name__ == "__main__":
    main()
