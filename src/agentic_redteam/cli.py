"""Console entry points for `redteam` and `iterative-retrain`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentic_redteam.attacker import run_redteam_sync
from agentic_redteam.config import load_config
from agentic_redteam.evaluation import (
    DEFAULT_EVAL_MAX_SAMPLES,
    DEFAULT_SEED,
    evaluate_probe,
    seed_everything,
)
from agentic_redteam.retrain import (
    DEFAULT_FRESH_PROBE_ARCH,
    retrain_probe,
    train_initial_probe,
)


def run_redteam_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one round of agentic red-teaming against a tuberlens probe."
    )
    parser.add_argument("config", type=Path, help="Path to the markdown config file")
    parser.add_argument(
        "--round", type=int, default=0, help="Base round number to record in the JSONL log"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"Run ID: {config.output.run_id}")
    print(f"Probe:  {config.probe.path}")
    print(f"Error types: {config.probe.error_types}")
    print(f"Rounds per error type: {config.attacker.rounds}")
    print(f"Concurrency: {config.attacker.concurrency}")
    if config.attacker.persistence_from_last_rounds is not None:
        print(f"Persistence from last rounds: {config.attacker.persistence_from_last_rounds}")
    print(
        "Models: "
        + ", ".join(f"{m.name} ({m.provider})" for m in config.attacker.models)
    )
    print()

    total_new = 0
    for et in config.probe.error_types:
        jsonl_path = config.jsonl_path_for(et)
        print(f"\n--- Error type: {et} → {jsonl_path} ---")
        base_round = args.round * config.attacker.rounds
        summaries = run_redteam_sync(
            config, base_round_num=base_round, error_type=et, jsonl_path=jsonl_path,
        )
        for s in summaries:
            print(
                f"  {s.model}: {s.new_successes} new successes "
                f"({s.total_messages} agent messages; stop={s.stop_reason})"
            )
            total_new += s.new_successes

    print(f"\n=== Total new successes: {total_new} ===")
    return 0


def iterative_retrain_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train an initial probe, then iterate: red-team → retrain "
        "(→ optional eval), n times."
    )
    parser.add_argument("config", type=Path, help="Path to the markdown config file")
    parser.add_argument(
        "--iterations", type=int, default=2, help="Number of red-team + retrain cycles (n)"
    )
    parser.add_argument(
        "--base-training-data",
        type=Path,
        required=True,
        help="JSONL/CSV used to train the initial probe and concatenated with red-team "
        "successes on every retrain",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the training data held out for validation (create_train_test_split)",
    )
    parser.add_argument(
        "--split-field",
        type=str,
        default=None,
        help="Optional field kept grouped together when splitting train/validation",
    )
    parser.add_argument(
        "--probe-out-dir",
        type=Path,
        default=Path("probes"),
        help="Where to write probes (probe_iter0.pkl = initial, probe_iterN.pkl per cycle)",
    )
    parser.add_argument(
        "--layer", type=int, default=None, help="Layer to probe; overrides probe.layer in config"
    )
    parser.add_argument(
        "--probe-arch",
        nargs="?",
        const=DEFAULT_FRESH_PROBE_ARCH,
        default=None,
        help="Probe architecture (ProbeType name); overrides probe.architecture in config. "
        f"Bare flag uses {DEFAULT_FRESH_PROBE_ARCH!r}. On retrains, omitting inherits the "
        "current probe's architecture.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the initial probe and each retrained probe on the eval datasets",
    )
    parser.add_argument(
        "--eval-dataset-dir",
        type=Path,
        default=Path("eval_datasets"),
        help="Directory of local eval split JSONLs (used with --eval)",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=DEFAULT_EVAL_MAX_SAMPLES,
        help="Balanced subsample size per eval split (used with --eval); 0 = full split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (train/val split + reproducible eval subsampling)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Where to write the eval comparison CSV (used with --eval)",
    )
    parser.add_argument(
        "--activations-cache-dir",
        type=Path,
        default=None,
        help="Cache dir for eval activations (default <results-dir>/eval_activations)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    args.probe_out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    error_types = config.probe.error_types
    layer = args.layer if args.layer is not None else config.probe.layer
    arch = args.probe_arch if args.probe_arch is not None else config.probe.architecture
    eval_max_samples = args.eval_max_samples if args.eval_max_samples > 0 else None
    contrastive_cache_path = args.probe_out_dir / "contrastive_cache.jsonl"
    if config.preprocessing is not None:
        print(
            "Preprocessing enabled: filter_dataset + generate_contrastive_dataset "
            f"({config.preprocessing.model} via {config.preprocessing.provider})"
        )

    # ---- Step 1: obtain the initial probe (warm-start if present, else train fresh) ----
    if config.probe.path is not None and Path(config.probe.path).exists():
        current_probe_path = Path(config.probe.path)
        print(f"Warm-starting from existing probe: {current_probe_path}")
    else:
        missing = [
            name
            for name, value in (
                ("probe.model", config.probe.model),
                ("probe.layer", layer),
                ("probe.pos_class_label", config.probe.pos_class_label),
                ("probe.neg_class_label", config.probe.neg_class_label),
            )
            if value is None
        ]
        if missing:
            parser.error(
                "Training the initial probe from scratch requires "
                + ", ".join(missing)
                + " — set them in the config probe: section, or point probe.path at an "
                "existing probe to warm-start."
            )
        current_probe_path = args.probe_out_dir / "probe_iter0.pkl"
        print(f"Training initial probe → {current_probe_path}")
        train_initial_probe(
            base_training_data_path=args.base_training_data,
            model_name=config.probe.model,
            layer=layer,
            new_probe_path=current_probe_path,
            pos_class_label=config.probe.pos_class_label,
            neg_class_label=config.probe.neg_class_label,
            probe_description=config.probe.description,
            probe_spec=arch,
            test_size=args.test_size,
            split_field=args.split_field,
            verbose=True,
        )

    activations_cache_dir = args.activations_cache_dir or (
        args.results_dir / "eval_activations"
    )
    eval_results: dict = {}

    def _maybe_eval(label: str, probe_path: Path) -> None:
        if not args.eval:
            return
        args.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== EVALUATING {label}: {probe_path} =====")
        df = evaluate_probe(
            probe_path,
            args.eval_dataset_dir,
            activations_cache_dir,
            max_samples=eval_max_samples,
            seed=args.seed,
        )
        eval_results[label] = df
        print(df.to_string(index=False))

    _maybe_eval("iter0", current_probe_path)

    print(f"\nInitial probe: {current_probe_path}")
    print(f"Error types: {error_types}")
    print(f"Rounds per error type: {config.attacker.rounds}")

    # ---- Steps 2-4, repeated n times ----
    for i in range(args.iterations):
        print(f"\n########## ITERATION {i} ##########")
        config.probe.path = current_probe_path
        base_round = i * config.attacker.rounds

        iteration_new_total = 0
        for et in error_types:
            jsonl_path = config.jsonl_path_for(et)
            print(f"\n--- Error type: {et} → {jsonl_path} ---")
            summaries = run_redteam_sync(
                config, base_round_num=base_round, error_type=et, jsonl_path=jsonl_path,
            )
            et_new = sum(s.new_successes for s in summaries)
            iteration_new_total += et_new
            print(f"  {et}: {et_new} new successes across {len(summaries)} model-rounds")

        print(f"Iteration {i}: {iteration_new_total} new successes total")

        jsonl_paths = [config.jsonl_path_for(et) for et in error_types]
        new_probe_path = args.probe_out_dir / f"probe_iter{i + 1}.pkl"
        result = retrain_probe(
            jsonl_path=jsonl_paths,
            base_probe_path=current_probe_path,
            base_training_data_path=args.base_training_data,
            new_probe_path=new_probe_path,
            layer=args.layer,
            probe_spec=args.probe_arch,
            preprocessing=config.preprocessing,
            contrastive_cache_path=contrastive_cache_path,
            test_size=args.test_size,
            split_field=args.split_field,
            verbose=True,
        )
        print(
            f"Iteration {i}: trained on {result.n_training_samples_total} samples "
            f"({result.n_redteam_samples} from red-team) → {result.new_probe_path}"
        )
        current_probe_path = new_probe_path
        _maybe_eval(f"iter{i + 1}", current_probe_path)

    if args.eval and eval_results:
        import pandas as pd

        rows = []
        for label, df in eval_results.items():
            d = df.copy()
            d.insert(0, "round", label)
            rows.append(d)
        comparison = pd.concat(rows, ignore_index=True)
        print("\n===== COMPARISON ACROSS ROUNDS =====")
        print(comparison.to_string(index=False))
        out_csv = args.results_dir / "iter_run_comparison.csv"
        comparison.to_csv(out_csv, index=False)
        print(f"\nSaved comparison table to {out_csv}")

    print(f"\nFinal probe: {current_probe_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("redteam", "iterative-retrain"):
        cmd = sys.argv.pop(1)
        if cmd == "redteam":
            sys.exit(run_redteam_main())
        else:
            sys.exit(iterative_retrain_main())
    else:
        print("Usage: python -m agentic_redteam.cli {redteam|iterative-retrain} ...")
        sys.exit(2)
