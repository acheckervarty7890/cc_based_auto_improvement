#!/usr/bin/env python
"""Ablation: does a *rejected* batch that is the structural twin of an *accepted* one hurt?

Arm 3N's loop accepted 62 samples over 13 iterations and ended at eval mean 0.8148
(`probe_iter13.pkl`, base ∪ 62). Exactly one rejected batch is a near-perfect structural
twin of an accepted one:

    it9b1   ACCEPTED  Δ +0.0120   "reproduce the user's ordered checklist; negative swaps two steps"
    it10b0  REJECTED  Δ -0.0102   "reproduce the user's ordered checklist; negative drops a substep"

They are indistinguishable on every surface measure taken (grounding 1.00 / 1.00,
positive-negative token Jaccard 1.00 / 0.99, length ratio 1.00 / 1.03, 5/5 matched pairs,
2 turns). This script fits base ∪ 62 ∪ it10b0 and evaluates it against the already-scored
base ∪ 62, so the only difference between the two probes is ten rows the dev set said no to.

Every knob is copied from configs/gen_gemma27b_instructions_nemotron.md and the CLI
defaults, so the new fit is apples-to-apples with probe_iter13: same base probe (for
architecture and metadata), same base data, same dev set, same seed, same transforms, same
cache dirs. Every activation involved is already cached, so nothing loads the 27B model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RUN_DIR = REPO / "results_gen_gemma27b_instructions_nemotron"
PROBE_DIR = REPO / "probes/gen_gemma27b_instructions_nemotron"
BASE_DATA = REPO / "data/instructions_llama70b_50.jsonl"
DEV_DATA = REPO / "dev_samples/instructions"
EVAL_DIR = REPO / "eval_sets/instructions"
BASE_CACHE = REPO / "cache_gen_gemma27b_instructions/base_activations"
EVAL_CACHE = REPO / "cache_gen_gemma27b_instructions/eval_activations"

# The probe iteration 12's union retrain started from — retrain_probe reads architecture and
# metadata off it, nothing else, so this is what makes the new fit match probe_iter13's.
BASE_PROBE = PROBE_DIR / "probe_iter12.pkl"
ACCEPTED = RUN_DIR / "accepted_iter13.jsonl"

SEED = 42
COMBINE = True
CONVERT = True


def _latest_batches() -> dict[tuple[int, int], dict]:
    """Newest row per (iteration, batch_index) from the append-only batch store."""
    latest: dict[tuple[int, int], dict] = {}
    with (RUN_DIR / "batches.jsonl").open() as fh:
        for line in fh:
            rec = json.loads(line)
            latest[(rec["iteration"], rec["batch_index"])] = rec
    return latest


def load_rejected(iteration: int, batch_index: int) -> tuple[list[dict], dict]:
    rec = _latest_batches()[(iteration, batch_index)]
    if rec["status"] != "scored":
        raise SystemExit(f"it{iteration}b{batch_index} is {rec['status']}, not scored")
    if rec["accepted"]:
        raise SystemExit(f"it{iteration}b{batch_index} was accepted, not rejected")
    rows = [{"inputs": s["messages"], "labels": s["label"]} for s in rec["samples"]]
    return rows, rec


def load_accepted() -> list[dict]:
    rows = []
    with ACCEPTED.open() as fh:
        for line in fh:
            rec = json.loads(line)
            rows.append({"inputs": rec["inputs"], "labels": rec["labels"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reject-batch", default="10:0", help="iteration:batch_index of the rejected batch to inject (default 10:0 — it10b0, the twin of accepted it9b1)")
    ap.add_argument("--out", type=Path, default=PROBE_DIR / "ablation_accepted62_plus_it10b0.pkl")
    ap.add_argument("--results", type=Path, default=RUN_DIR / "ablation_twin_injection.csv")
    args = ap.parse_args()

    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import retrain_probe, warm_sample_activation_cache

    it, bk = (int(x) for x in args.reject_batch.split(":"))
    accepted = load_accepted()
    injected, rec = load_rejected(it, bk)

    print(f"accepted samples (dev said yes) : {len(accepted)}")
    print(f"injected batch  it{it}b{bk}      : {len(injected)} samples, Δ {rec['delta']:+.4f}")
    print(f"  direction: {rec['direction'][:150]}")

    samples = accepted + injected

    # Every one of these was forwarded during the run, so this is a cache-hit no-op that
    # only proves it before the fit tries to load a model.
    warm_sample_activation_cache(
        samples,
        base_probe_path=BASE_PROBE,
        base_activation_cache_dir=BASE_CACHE,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
        verbose=True,
    )

    result = retrain_probe(
        samples=samples,
        base_probe_path=BASE_PROBE,
        base_training_data_path=BASE_DATA,
        new_probe_path=args.out,
        dev_data_path=DEV_DATA,
        seed=SEED,
        base_data_fraction=1.0,
        base_activation_cache_dir=BASE_CACHE,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
        verbose=True,
    )
    print(f"\ntraining rows: {result.n_training_samples_total}  (base 50 + {result.n_extra_samples})")
    print(f"dev AUROC: {result.dev_auroc}")

    df = evaluate_probe(
        args.out,
        EVAL_DIR,
        EVAL_CACHE,
        max_samples=None,  # eval_max_samples: 0 → full splits
        seed=SEED,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
    )
    print(df.to_string(index=False))
    df.to_csv(args.results, index=False)
    print(f"\nwrote {args.results}")


if __name__ == "__main__":
    main()
