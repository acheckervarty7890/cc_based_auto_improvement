# Basic Configuration
import argparse
import os
import pickle
from pathlib import Path


from tuberlens.interfaces.dataset import LabelledDataset
from tuberlens.interfaces.probes import ProbeSpec, ProbeType
from tuberlens.training import train_probe
from tuberlens.utils import create_train_test_split
import random
import numpy as np
import torch

assert os.environ.get("HF_TOKEN"), (
    "HF_TOKEN not set. Export your HuggingFace token before running, e.g.:\n"
    "  export HF_TOKEN=hf_xxx\n"
    "Needed to download meta-llama/Llama-3.2-1B-Instruct (gated repo)."
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a high-stakes probe.")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name to use",
    )
    parser.add_argument("--layer", type=int, default=8, help="Layer number to probe")

    parser.add_argument(
        "--output_probe_path",
        type=str,
        default="probe_llama1b.pkl",
        help="Output path for the trained probe",
    )

    parser.add_argument(
        "--probe_training_data",
        type=str,
        default="combined_deployment_22_04_25.jsonl",
        help="Path to the probe training data file",
    )
    parser.add_argument(
        "--split", type=str, default=None, help="Field to use for train-test split"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--activations_save_path",
        type=str,
        default=None,
        help="Path to save activations",
    )
    parser.add_argument(
        "--using_kaggle",
        action="store_true",
        help="Whether the script will upload activations to Kaggle",
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    layer = args.layer

    pos_class_label = "high-stakes"
    neg_class_label = "low-stakes"
    probe_description = f"A linear probe on {args.model} detecting whether the conversation is high-stakes."

    stakes_dataset = Path(args.probe_training_data)
    # stakes_dataset = CONCEPT_DIR / "high_stakes_training_data_without_inputs.jsonl"
    # stakes_dataset = CONCEPT_DIR / "high_stakes_training_data.jsonl"
    dataset = LabelledDataset.load_from(
        stakes_dataset, pos_class_label=pos_class_label, neg_class_label=neg_class_label
    )
    train_dataset, validation_dataset = create_train_test_split(
        dataset, split_field=args.split
    )

    print(
        f"Read {len(train_dataset)} samples for training and {len(validation_dataset)} samples for validation."
    )

    probe = train_probe(
        train_dataset,
        validation_dataset,
        args.model,
        layer,
        # start_turn_index=0,  # Include system and user message
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
        probe_description=probe_description,
        probe_spec=ProbeSpec(
            name=ProbeType.linear_then_softmax,
            hyperparams={},
        ),
        activations_save_path=args.activations_save_path,
        using_kaggle=args.using_kaggle,
    )

    os.makedirs(Path(args.output_probe_path).parent, exist_ok=True)
    with open(Path(args.output_probe_path), "wb") as f:
        pickle.dump(probe, f)
