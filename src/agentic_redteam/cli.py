"""Console entry points for `redteam` and `iterative-retrain`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentic_redteam.attacker import run_redteam_sync
from agentic_redteam.config import load_config
from agentic_redteam.retrain import retrain_probe


def run_redteam_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one round of agentic red-teaming against a tuberlens probe."
    )
    parser.add_argument("config", type=Path, help="Path to the markdown config file")
    parser.add_argument(
        "--round", type=int, default=0, help="Round number to record in the JSONL log"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"Run ID: {config.output.run_id}")
    print(f"Probe:  {config.probe.path} (error_type={config.probe.error_type})")
    print(f"Output: {config.output.jsonl_path}")
    print(f"Models: {', '.join(config.attacker.models)}")
    print()

    summaries = run_redteam_sync(config, round_num=args.round)

    print("\n=== Summary ===")
    total_new = 0
    for s in summaries:
        print(
            f"  {s.model}: {s.new_successes} new successes "
            f"({s.total_messages} agent messages; stop={s.stop_reason})"
        )
        total_new += s.new_successes
    print(f"  Total new successes this round: {total_new}")
    return 0


def iterative_retrain_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run iterative red-team → retrain → evaluate cycles."
    )
    parser.add_argument("config", type=Path, help="Path to the markdown config file")
    parser.add_argument(
        "--iterations", type=int, default=2, help="Number of red-team + retrain cycles"
    )
    parser.add_argument(
        "--base-training-data",
        type=Path,
        default=None,
        help="Optional JSONL/CSV consumed by tuberlens.LabelledDataset.load_from",
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=None,
        help="Optional held-out dataset for training validation",
    )
    parser.add_argument(
        "--probe-out-dir",
        type=Path,
        default=Path("probes"),
        help="Where to write retrained probes (one .pkl per iteration)",
    )
    parser.add_argument(
        "--layer", type=int, default=None, help="Layer to probe; defaults to base probe's layer"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    args.probe_out_dir.mkdir(parents=True, exist_ok=True)

    current_probe_path = config.probe.path
    print(f"Initial probe: {current_probe_path}")

    for i in range(args.iterations):
        print(f"\n########## ITERATION {i} ##########")
        # Point the config at the current probe so each iteration attacks the freshest version.
        config.probe.path = current_probe_path
        summaries = run_redteam_sync(config, round_num=i)
        new_total = sum(s.new_successes for s in summaries)
        print(f"Iteration {i}: {new_total} new successes across {len(summaries)} models")

        new_probe_path = args.probe_out_dir / f"probe_iter{i + 1}.pkl"
        result = retrain_probe(
            jsonl_path=config.output.jsonl_path,
            base_probe_path=current_probe_path,
            base_training_data_path=args.base_training_data,
            new_probe_path=new_probe_path,
            layer=args.layer,
            validation_dataset_path=args.validation_data,
            verbose=True,
        )
        print(
            f"Iteration {i}: trained on {result.n_training_samples_total} samples "
            f"({result.n_redteam_samples} from red-team) → {result.new_probe_path}"
        )
        current_probe_path = new_probe_path

    print(f"\nFinal probe: {current_probe_path}")
    return 0


if __name__ == "__main__":
    # Allow `python -m agentic_redteam.cli redteam <args>` style invocation.
    if len(sys.argv) > 1 and sys.argv[1] in ("redteam", "iterative-retrain"):
        cmd = sys.argv.pop(1)
        if cmd == "redteam":
            sys.exit(run_redteam_main())
        else:
            sys.exit(iterative_retrain_main())
    else:
        print("Usage: python -m agentic_redteam.cli {redteam|iterative-retrain} ...")
        sys.exit(2)
