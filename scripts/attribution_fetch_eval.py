"""Pull the four ``eval_datasets/`` activation blobs from Kaggle into a local cache.

The red-team and base blobs have their own fetcher already
(``publish_kaggle_hs_redteam_activations.py restore``); this is the eval half, and it
just wires the arm's probe metadata into ``kaggle_activations.prefetch_eval_activations``
so every blob is validated against the probe's model/layer and the split's row count
before it lands. A split that cannot be fetched raises rather than silently falling
back to computing it — computing one of these means hours of gemma-3-27b forwards.

The slug/file templates default to the ones both experiment9 configs carry
(``{split}gemmaevalpt`` / ``{split}-gemmaeval.pt``). All four high-stakes split stems
are plain alphanumerics, so ``{split}`` is a legal Kaggle slug here and the ``{slug}``
form the human-harm splits need is not required.

Usage:
    .venv_claude/bin/python scripts/attribution_fetch_eval.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import attribution_lib as A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(A.ARMS), default="gptoss120b",
                    help="only used for probe metadata; both arms share the probe spec")
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--owner", default="anku7890")
    ap.add_argument("--dataset-slug", default="{split}gemmaevalpt")
    ap.add_argument("--file-name", default="{split}-gemmaeval.pt")
    ap.add_argument("--cache-dir", type=Path, default=A.EVAL_ACTIVATIONS_DIR)
    args = ap.parse_args()

    from tuberlens.interfaces.dataset import LabelledDataset

    from agentic_redteam.kaggle_activations import (
        KaggleActivationSource,
        prefetch_eval_activations,
    )

    probe = A.load_probe(A.ARMS[args.arm] / f"probe_iter{args.iteration}.pkl")
    eval_datasets = {
        split: LabelledDataset.load_from(
            A.EVAL_DATASET_DIR / f"{split}.jsonl",
            pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label,
            combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
        )
        for split in A.EVAL_SPLITS
    }

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    status = prefetch_eval_activations(
        args.cache_dir,
        eval_datasets,
        KaggleActivationSource(args.owner, args.dataset_slug, args.file_name),
        model_name=probe.model_name,
        layer=int(probe.layer),
        cache_stem="acts_full.pt",
    )
    for split, state in sorted(status.items()):
        print(f"  {split:24s} {state}")


if __name__ == "__main__":
    main()
