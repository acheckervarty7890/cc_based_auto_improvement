#!/usr/bin/env python
"""Fit the 62 accepted samples ALONE — no base rows — and score dev + eval.

The loop never trains this probe: every retrain is `base ∪ accepted`, so the 50 base
rows are in every fit and the accepted rows have only ever been measured as an
increment on top of them. This asks what they carry on their own.

Controls already on record, same setup:

    base only (50 rows)          dev 0.7573   eval 0.7779   (the run's initial probe)
    base ∪ 62 (probe_iter13)     dev 0.8311   eval 0.8148

Everything else matches arm 3N: gemma-3-27b layer 32, linear_then_softmax, single
probe, seed 42, architecture and metadata inherited from probe_iter12.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RUN_DIR = REPO / "results_gen_gemma27b_instructions_nemotron"
PROBE_DIR = REPO / "probes/gen_gemma27b_instructions_nemotron"
BASE_DATA = REPO / "data/instructions_llama70b_50.jsonl"
DEV_DATA = REPO / "dev_samples/instructions"
EVAL_DIR = REPO / "eval_sets/instructions"
BASE_CACHE = REPO / "cache_gen_gemma27b_instructions/base_activations"
EVAL_CACHE = REPO / "cache_gen_gemma27b_instructions/eval_activations"

BASE_PROBE = PROBE_DIR / "probe_iter12.pkl"  # byte-identical to probe_iter13
ACCEPTED = RUN_DIR / "accepted_iter13.jsonl"

SEED = 42
COMBINE = True
CONVERT = True


def load_accepted() -> list[dict]:
    rows = []
    with ACCEPTED.open() as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({"inputs": r["inputs"], "labels": r["labels"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=PROBE_DIR / "gen_accepted_only.pkl")
    ap.add_argument("--with-base", action="store_true",
                    help="also refit base ∪ 62 as an in-process control")
    args = ap.parse_args()

    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import retrain_probe, warm_sample_activation_cache

    accepted = load_accepted()
    npos = sum(1 for r in accepted if not r["labels"].startswith("assistant_does_not"))
    print(f"accepted: {len(accepted)} rows ({npos} pos / {len(accepted) - npos} neg)")

    n_new = warm_sample_activation_cache(
        accepted, base_probe_path=BASE_PROBE, base_activation_cache_dir=BASE_CACHE,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
        verbose=True,
    )
    print(f"extracted {n_new} new conversation activation(s)")

    runs = [("accepted only (no base)", None, args.out)]
    if args.with_base:
        runs.append(("base ∪ 62 (control)", BASE_DATA, PROBE_DIR / "gen_accepted_only_control.pkl"))

    for label, base_data, out in runs:
        print(f"\n--- {label} → {out.name} ---", flush=True)
        res = retrain_probe(
            samples=accepted,
            base_probe_path=BASE_PROBE,
            base_training_data_path=base_data,
            new_probe_path=out,
            dev_data_path=DEV_DATA,
            seed=SEED,
            base_data_fraction=1.0,
            base_activation_cache_dir=BASE_CACHE,
            combine_consecutive_messages=COMBINE,
            convert_tool_to_assistant=CONVERT,
            verbose=False,
        )
        df = evaluate_probe(
            out, EVAL_DIR, EVAL_CACHE, max_samples=None, seed=SEED,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
        )
        ev = float(df.loc[df["dataset"] == "mean", "auroc"].iloc[0])
        print(f"\n{label}: {res.n_training_samples_total} training rows")
        print(f"  dev  mean {res.dev_auroc['mean']:.5f}")
        for k, v in sorted(res.dev_auroc.items()):
            if k != "mean":
                print(f"      {k:<28} {v:.5f}")
        print(f"  eval mean {ev:.5f}")
        for _, r in df[df["dataset"] != "mean"].iterrows():
            print(f"      {r['dataset']:<28} {r['auroc']:.5f}")


if __name__ == "__main__":
    main()
