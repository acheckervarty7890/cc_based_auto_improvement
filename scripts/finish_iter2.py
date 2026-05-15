#!/usr/bin/env python
"""Finish iteration 2 after the iter1 OOM: retrain probe_iter1 on the existing
JSONL successes (no --validation-data, to avoid the CUDA OOM that killed the
prior run) and evaluate the resulting probe on the cached eval_datasets
activations.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from agentic_redteam.retrain import retrain_probe
from iterative_retrain_with_eval import evaluate_probe  # noqa: E402


def main() -> int:
    base_probe = REPO / "probes" / "probe_iter1.pkl"
    jsonl = REPO / "results" / "iter_run_2026-05-15.jsonl"
    base_train = REPO / "data" / "combined_deployment_22_04_25.jsonl"
    new_probe = REPO / "probes" / "probe_iter2.pkl"
    eval_dir = REPO / "eval_datasets"
    activations_cache = REPO / "results" / "eval_activations"

    print(f"Retraining {base_probe} -> {new_probe} (NO validation data this time)")
    result = retrain_probe(
        jsonl_path=jsonl,
        base_probe_path=base_probe,
        base_training_data_path=base_train,
        new_probe_path=new_probe,
        validation_dataset_path=None,
        verbose=True,
    )
    print(
        f"Trained on {result.n_training_samples_total} samples "
        f"({result.n_redteam_samples} from red-team) -> {result.new_probe_path}"
    )

    print(f"\n========= EVALUATING iter2: {new_probe} =========")
    df = evaluate_probe(new_probe, eval_dir, activations_cache)
    print(df.to_string(index=False))

    out_csv = REPO / "results" / "iter2_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved iter2 eval table to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
