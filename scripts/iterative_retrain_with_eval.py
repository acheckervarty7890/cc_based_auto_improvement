#!/usr/bin/env python
"""Run iterative red-team + retrain cycles and evaluate every probe (base + each
retrained version) against the local eval_datasets so the user can see how each
round changes per-split AUROC / accuracy.

The activation cache (`--activations-cache-dir`) is reused across rounds, so the
Llama-3.2-1B activation pass on each eval split runs at most once.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from agentic_redteam.attacker import run_redteam_sync
from agentic_redteam.config import load_config
from agentic_redteam.retrain import retrain_probe

EVAL_DATASETS = ["anthropic", "mts"]


def evaluate_probe(
    probe_path: Path,
    eval_dataset_dir: Path,
    activations_cache_dir: Path,
) -> pd.DataFrame:
    from tuberlens.evaluation import get_performances
    from tuberlens.interfaces.dataset import LabelledDataset

    with probe_path.open("rb") as f:
        probe = pickle.load(f)

    eval_datasets = {
        name: LabelledDataset.load_from(
            eval_dataset_dir / f"{name}.jsonl",
            pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label,
        )
        for name in EVAL_DATASETS
    }

    activations_cache_dir.mkdir(parents=True, exist_ok=True)
    return get_performances(
        probe,
        eval_datasets,
        activations_save_path=str(activations_cache_dir / "acts.pt"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Iterative red-team + retrain with per-round eval_datasets evaluation."
    )
    parser.add_argument("config", type=Path, help="Markdown config path")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--base-training-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=None)
    parser.add_argument(
        "--eval-dataset-dir",
        type=Path,
        default=Path(__file__).parent.parent / "eval_datasets",
    )
    parser.add_argument(
        "--probe-out-dir",
        type=Path,
        default=Path(__file__).parent.parent / "probes",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
    )
    parser.add_argument(
        "--activations-cache-dir",
        type=Path,
        default=None,
        help="Where to cache Llama activations for the eval splits "
        "(default: <results-dir>/eval_activations).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    args.probe_out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    activations_cache_dir = (
        args.activations_cache_dir or args.results_dir / "eval_activations"
    )

    eval_results: dict[str, pd.DataFrame] = {}

    base_probe_path = Path(config.probe.path)
    print(f"\n========= EVALUATING BASE PROBE: {base_probe_path} =========")
    eval_results["base"] = evaluate_probe(
        base_probe_path, args.eval_dataset_dir, activations_cache_dir
    )
    print(eval_results["base"].to_string(index=False))

    current_probe_path = base_probe_path

    for i in range(args.iterations):
        print(f"\n########## ITERATION {i} ##########")
        config.probe.path = current_probe_path
        summaries = run_redteam_sync(config, round_num=i)
        new_total = sum(s.new_successes for s in summaries)
        print(
            f"Iteration {i}: {new_total} new successes across {len(summaries)} models"
        )

        new_probe_path = args.probe_out_dir / f"probe_iter{i + 1}.pkl"
        result = retrain_probe(
            jsonl_path=config.output.jsonl_path,
            base_probe_path=current_probe_path,
            base_training_data_path=args.base_training_data,
            new_probe_path=new_probe_path,
            validation_dataset_path=args.validation_data,
            verbose=True,
        )
        print(
            f"Iteration {i}: trained on {result.n_training_samples_total} samples "
            f"({result.n_redteam_samples} from red-team) → {result.new_probe_path}"
        )
        current_probe_path = new_probe_path

        round_label = f"iter{i + 1}"
        print(f"\n========= EVALUATING {round_label}: {new_probe_path} =========")
        eval_results[round_label] = evaluate_probe(
            new_probe_path, args.eval_dataset_dir, activations_cache_dir
        )
        print(eval_results[round_label].to_string(index=False))

    print("\n========= COMPARISON ACROSS ROUNDS =========")
    comparison_rows = []
    for round_label, df in eval_results.items():
        df_copy = df.copy()
        df_copy.insert(0, "round", round_label)
        comparison_rows.append(df_copy)
    comparison = pd.concat(comparison_rows, ignore_index=True)
    print(comparison.to_string(index=False))

    comparison_csv = args.results_dir / "iter_run_comparison.csv"
    comparison.to_csv(comparison_csv, index=False)
    print(f"\nSaved comparison table to {comparison_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
