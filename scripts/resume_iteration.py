#!/usr/bin/env python
"""Resume a single iterative-retrain iteration (Option B).

Re-runs one iteration's red-team + retrain (+eval) without redoing the earlier
iterations, to finish a run that crashed mid-red-teaming. It mirrors the body of
``cli.iterative_retrain_main``'s loop for a single iteration index ``i``:

  1. Warm-start from the probe iteration ``i`` was attacking (--base-probe).
  2. Red-team it across every configured error type, appending to the SAME
     JSONL stores — already-collected attempts dedup, so no probe/judge call is
     re-spent on a byte-identical conversation. New rows are tagged
     ``iteration=i`` / global rounds ``i*rounds .. i*rounds+rounds-1``.
  3. Retrain on base data ∪ all successes → --new-probe, reusing the existing
     contrastive cache (so preprocessing only pays for newly-seen successes).
  4. Evaluate the new probe (skip with --no-eval).

Because the attacker is stochastic, step 2 re-spends roughly one iteration's
worth of attacker+judge API calls (dedup only saves on exact regenerations).

Defaults resume the 2026-06-01 run that crashed in iteration 2: i=2, base probe
probe_iter2.pkl → new probe_iter3.pkl in probes/run_0601.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_redteam.attacker import run_redteam_sync
from agentic_redteam.config import load_config
from agentic_redteam.evaluation import (
    DEFAULT_EVAL_MAX_SAMPLES,
    DEFAULT_SEED,
    evaluate_probe,
    seed_everything,
)
from agentic_redteam.retrain import retrain_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("configs/example_config_openrouter.md"),
        help="Markdown config (attacker/judge/preprocessing + jsonl paths)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=2,
        help="0-based iteration index to (re)run; sets round tags and base_round",
    )
    parser.add_argument(
        "--base-probe",
        type=Path,
        default=Path("probes/run_0601/probe_iter2.pkl"),
        help="Probe this iteration attacks + inherits architecture from",
    )
    parser.add_argument(
        "--new-probe",
        type=Path,
        default=Path("probes/run_0601/probe_iter3.pkl"),
        help="Output path for the retrained probe",
    )
    parser.add_argument(
        "--base-training-data",
        type=Path,
        default=Path(
            "/home/ankush/Documents/tuberlens/data/high-stakes/"
            "combined_deployment_22_04_25.jsonl"
        ),
        help="Base training data concatenated with red-team successes",
    )
    parser.add_argument(
        "--probe-out-dir",
        type=Path,
        default=Path("probes/run_0601"),
        help="Dir holding the contrastive cache + postprocessed snapshot",
    )
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Skip filter+contrastive preprocessing during retrain",
    )
    parser.add_argument(
        "--skip-redteam",
        action="store_true",
        help="Skip step 2 (red-team) and only retrain from existing successes",
    )
    # ---- eval knobs (match the iterative CLI defaults) ----
    parser.add_argument("--no-eval", action="store_true", help="Skip the eval step")
    parser.add_argument(
        "--eval-dataset-dir", type=Path, default=Path("eval_datasets")
    )
    parser.add_argument(
        "--activations-cache-dir",
        type=Path,
        default=Path("results/eval_activations"),
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=DEFAULT_EVAL_MAX_SAMPLES,
        help="Balanced subsample per split (0 = full split)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-field", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)

    # The attacker loads the probe under attack from config.probe.path.
    config.probe.path = args.base_probe

    error_types = config.probe.error_types
    jsonl_paths = [config.jsonl_path_for(et) for et in error_types]
    base_round = args.iteration * config.attacker.rounds
    preprocessing = None if args.no_preprocessing else config.preprocessing
    contrastive_cache_path = args.probe_out_dir / "contrastive_cache.jsonl"
    postprocessed_out_path = (
        args.probe_out_dir / f"redteam_postprocessed_{args.new_probe.stem}.jsonl"
    )

    print(f"Resume iteration {args.iteration} (Option B)")
    print(f"  config:          {args.config}")
    print(f"  base probe:      {args.base_probe}")
    print(f"  error types:     {error_types}  (rounds {base_round}..{base_round + config.attacker.rounds - 1})")
    print(f"  red-team JSONLs: {[str(p) for p in jsonl_paths]}")
    print(f"  preprocessing:   {'off' if preprocessing is None else preprocessing.model}")
    print(f"  → new probe:     {args.new_probe}\n")

    # ---- Step 2: red-team the base probe across all error types ----
    if not args.skip_redteam:
        for et in error_types:
            jsonl_path = config.jsonl_path_for(et)
            print(f"\n--- Error type: {et} → {jsonl_path} ---")
            summaries = run_redteam_sync(
                config,
                base_round_num=base_round,
                error_type=et,
                jsonl_path=jsonl_path,
                iteration=args.iteration,
            )
            et_new = sum(s.new_successes for s in summaries)
            print(f"  {et}: {et_new} new successes across {len(summaries)} model-rounds")
    else:
        print("Skipping red-team (--skip-redteam); retraining from existing successes.")

    # ---- Step 3: retrain on base data ∪ all successes ----
    result = retrain_probe(
        jsonl_path=jsonl_paths,
        base_probe_path=args.base_probe,
        base_training_data_path=args.base_training_data,
        new_probe_path=args.new_probe,
        layer=None,  # inherit from base probe
        probe_spec=None,  # inherit architecture from base probe
        preprocessing=preprocessing,
        contrastive_cache_path=contrastive_cache_path,
        postprocessed_out_path=postprocessed_out_path,
        min_judge_confidence=config.judge.confidence_threshold,
        test_size=args.test_size,
        split_field=args.split_field,
        seed=args.seed,
        base_activation_cache_dir=args.probe_out_dir / "base_activation_cache",
        combine_consecutive_messages=config.eval.combine_consecutive_messages,
        convert_tool_to_assistant=config.eval.convert_tool_to_assistant,
        verbose=True,
    )
    print(
        f"\nIteration {args.iteration}: trained on {result.n_training_samples_total} "
        f"samples ({result.n_redteam_samples} from red-team) → {result.new_probe_path}"
    )

    # ---- Step 4: eval ----
    if args.no_eval:
        print(f"\nFinal probe: {args.new_probe}")
        return 0

    max_samples = args.eval_max_samples if args.eval_max_samples > 0 else None
    print(f"\n===== EVALUATING {args.new_probe} =====")
    df = evaluate_probe(
        args.new_probe,
        args.eval_dataset_dir,
        args.activations_cache_dir,
        max_samples=max_samples,
        seed=args.seed,
    )
    print(df.to_string(index=False))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.results_dir / f"{args.new_probe.stem}_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved eval table to {out_csv}")
    print(f"\nFinal probe: {args.new_probe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
