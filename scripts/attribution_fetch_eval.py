"""Pull the four ``eval_datasets/`` activation blobs from Kaggle into a local cache.

The red-team and base blobs have their own fetcher already
(``publish_kaggle_hs_redteam_activations.py restore``); this is the eval half, and it
just wires the arm's probe metadata into ``kaggle_activations.prefetch_eval_activations``
so every blob is validated against the probe's model/layer and the split's row count
before it lands. A split that cannot be fetched raises rather than silently falling
back to computing it — computing one of these means hours of gemma-3-27b forwards.

``--concept`` picks which eval set to fetch, defaulting to ``hs`` — the four
``eval_datasets/`` splits these arms were trained and scored on. The other two concepts
are here for ``vintage_cross_concept_eval.py``, which scores the same probes off their
own concept and so needs those blobs in the cache dirs its ``TARGETS`` table names.
Everything about the fetch except the split list and the slug template is identical
across the three, because every blob is the same gemma-3-27b-it L32 extraction under the
same two loader transforms.

The slug template differs per concept, and only for a Kaggle naming reason: dataset
slugs are lowercase alphanumerics and hyphens, so a stem containing an underscore has no
dataset named after it verbatim. All four high-stakes stems are plain words and use
``{split}``; every human-harm and instruction stem has an underscore and needs the
hyphenated ``{slug}`` form. The *file* inside the dataset is unrestricted and stays on
``{split}`` throughout.

Usage:
    .venv_claude/bin/python scripts/attribution_fetch_eval.py
    .venv_claude/bin/python scripts/attribution_fetch_eval.py --concept instructions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import attribution_lib as A
from vintage_cross_concept_eval import TARGETS

# Per-concept Kaggle slug template. The file name inside each dataset is the same
# everywhere, so it stays a flag rather than a per-concept field.
CONCEPT_SLUGS = {
    "hs": "{split}gemmaevalpt",
    "hu_harm": "{slug}-gemmaevalpt",
    "instructions": "{slug}-gemmaevalpt",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", choices=sorted(CONCEPT_SLUGS), default="hs",
                    help="which eval set to fetch (splits, cache dir and slug template)")
    ap.add_argument("--arm", choices=sorted(A.ARMS), default="gptoss120b",
                    help="only used for probe metadata; both arms share the probe spec")
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--owner", default="anku7890")
    ap.add_argument("--dataset-slug", default=None,
                    help="override the concept's slug template")
    ap.add_argument("--file-name", default="{split}-gemmaeval.pt")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="override the concept's cache dir")
    args = ap.parse_args()

    target = TARGETS[args.concept]
    cache_dir = args.cache_dir or target.activations_dir
    dataset_slug = args.dataset_slug or CONCEPT_SLUGS[args.concept]

    from tuberlens.interfaces.dataset import LabelledDataset

    from agentic_redteam.kaggle_activations import (
        KaggleActivationSource,
        prefetch_eval_activations,
    )

    probe = A.load_probe(A.ARMS[args.arm] / f"probe_iter{args.iteration}.pkl")
    # The probe supplies model_name/layer, which is all the validation needs from it.
    # The LABELS come from the target concept: a split loaded under another concept's
    # class strings keeps its labels as raw strings (dataset.py:311-317) — harmless for
    # the row count prefetch checks, but it would be a trap for anything reading them.
    eval_datasets = {
        split: LabelledDataset.load_from(
            target.eval_dir / f"{split}.jsonl",
            pos_class_label=target.pos_label,
            neg_class_label=target.neg_label,
            combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
        )
        for split in target.splits()
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"concept {args.concept}: {len(eval_datasets)} split(s) -> {cache_dir}")
    status = prefetch_eval_activations(
        cache_dir,
        eval_datasets,
        KaggleActivationSource(args.owner, dataset_slug, args.file_name),
        model_name=probe.model_name,
        layer=int(probe.layer),
        cache_stem="acts_full.pt",
    )
    for split, state in sorted(status.items()):
        print(f"  {split:24s} {state}")


if __name__ == "__main__":
    main()
