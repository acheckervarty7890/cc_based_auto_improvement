"""Evaluate a high-stakes probe against the local dev eval datasets.

Datasets are loaded from ../eval_datasets/{name}.jsonl rather than downloaded.
The four dev splits (anthropic, mt, mts, toolace) are the ones we have locally;
the test-only mental_health / redteaming splits would need separate download.
"""

import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from tuberlens.evaluation import get_performances
from tuberlens.interfaces.dataset import LabelledDataset, subsample_balanced_subset
from tuberlens.model import LLMModel

# `mt` and `toolace` in eval_datasets/ are CSVs misnamed with a .jsonl
# extension, so tuberlens's JSONL loader can't read them — they are skipped
# here. Add them back (and convert to CSV or rename) if you need them.
DATASETS = ["anthropic", "mts"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a high-stakes probe on the local dev eval datasets."
    )
    parser.add_argument(
        "--probe_path",
        type=Path,
        default=Path("probe_llama1b.pkl"),
        help="Path to the trained probe (default: probe_llama1b.pkl in cwd)",
    )
    parser.add_argument(
        "--eval_data_dir",
        type=Path,
        default=Path("../eval_datasets"),
        help="Directory containing the local {anthropic,mt,mts,toolace}.jsonl files",
    )
    parser.add_argument(
        "--results_file",
        type=Path,
        default=Path("evaluation_results.csv"),
        help="Where to write the per-dataset metric CSV",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=None,
        help="Downsample to N samples per class for faster eval (None = full)",
    )
    parser.add_argument(
        "--activations_save_path",
        type=str,
        default=None,
        help="Optional path to save/load activations (skip recompute on rerun)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert os.environ.get("HF_TOKEN"), (
        "HF_TOKEN not set. Export your HuggingFace token before running, e.g.:\n"
        "  export HF_TOKEN=hf_xxx"
    )

    seed_everything(args.seed)

    with args.probe_path.open("rb") as f:
        probe = pickle.load(f)
    assert probe.model_name is not None
    assert probe.layer is not None
    print(f"Probe: {probe.description}")

    model = LLMModel.load(probe.model_name)

    eval_datasets = {
        name: LabelledDataset.load_from(
            args.eval_data_dir / f"{name}.jsonl",
            pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label,
        )
        for name in DATASETS
    }
    for name, ds in eval_datasets.items():
        print(f"  {name}: {len(ds)} samples")

    if args.samples_per_class is not None:
        eval_datasets = {
            name: subsample_balanced_subset(ds, n_per_class=args.samples_per_class)
            for name, ds in eval_datasets.items()
        }
        print(f"Subsampled to {args.samples_per_class} samples per class.")

    performances = get_performances(
        probe,
        eval_datasets,
        activations_save_path=args.activations_save_path,
    )

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    performances.to_csv(args.results_file, index=False)
    print(f"Wrote {args.results_file}")
