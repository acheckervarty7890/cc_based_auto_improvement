"""Pull the high-stakes and harmful_to_human eval-split activations from Kaggle.

``attribution_fetch_eval.py`` fetches this branch's own seven ``eval_instructions``
blobs. This is the same call for the *other two* concepts' eval sets, so an
instruction-following probe can be scored on them without a gemma-3-27b forward pass.

That is only sound because all three concepts' runs use the **same frozen extraction
model and layer** — ``google/gemma-3-27b-it`` L32 (``configs/gemma27_config.md``,
``configs/gemma27_config_hu_harm.md``, and this branch's two arm configs) — and the same
two loader transforms. An activation is a property of the conversation and the extractor,
not of the probe head, so the blobs are interchangeable across concepts. Every blob is
validated against model name, layer and the split's row count before it may be used;
``prefetch_eval_activations`` raises rather than falling back to computing one.

Two slug conventions, because the datasets were published at different times:

- **high-stakes** (``eval_datasets/``): ``anku7890/{split}gemmaevalpt`` — no hyphen. Its
  four split stems (anthropic/mt/mts/toolace) carry no underscore, so ``{split}`` is a
  legal Kaggle slug on its own.
- **harmful_to_human** (``eval_dataset_hu_ha/``): ``anku7890/{slug}-gemmaevalpt``. Every
  stem there has an underscore, which Kaggle rejects in a slug, hence ``{slug}``.

The file *inside* each dataset follows ``{split}-gemmaeval.pt`` in both cases.

Usage:
    KAGGLE_CONFIG_DIR=$PWD/kaggle .venv_claude/bin/python scripts/fetch_crossconcept_eval.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A  # noqa: E402

# Each concept: the eval-split dir, the Kaggle slug template, and the class labels the
# split JSONLs use (load_from matches them exactly).
CONCEPTS = {
    "hs": {
        "dir": A.REPO / "eval_datasets",
        "dataset_slug": "{split}gemmaevalpt",
        "pos": "high-stakes",
        "neg": "low-stakes",
        "cache": A.REPO / "results_instructions_gemma27b_shared/eval_activations_hs",
    },
    "hh": {
        "dir": A.REPO / "eval_dataset_hu_ha",
        "dataset_slug": "{slug}-gemmaevalpt",
        "pos": "harmful_to_human",
        "neg": "not_harmful_to_human",
        "cache": A.REPO / "results_instructions_gemma27b_shared/eval_activations_hh",
    },
}


def splits_of(concept: str) -> list[str]:
    """Every ``*.jsonl`` in the concept's dir — what ``evaluate_probe(splits=None)`` finds."""
    return sorted(p.stem for p in CONCEPTS[concept]["dir"].glob("*.jsonl"))


def load_splits(concept: str):
    from tuberlens.interfaces.dataset import LabelledDataset

    c = CONCEPTS[concept]
    return {
        split: LabelledDataset.load_from(
            c["dir"] / f"{split}.jsonl",
            pos_class_label=c["pos"],
            neg_class_label=c["neg"],
            combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
        )
        for split in splits_of(concept)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", nargs="+", choices=sorted(CONCEPTS),
                    default=sorted(CONCEPTS))
    ap.add_argument("--arm", choices=sorted(A.ARMS), default="gptoss120b",
                    help="only read for probe model/layer; both arms share them")
    ap.add_argument("--owner", default="anku7890")
    ap.add_argument("--file-name", default="{split}-gemmaeval.pt")
    args = ap.parse_args()

    from agentic_redteam.kaggle_activations import (
        KaggleActivationSource,
        prefetch_eval_activations,
    )

    probe = A.load_probe(A.ARMS[args.arm] / "probe_iter3.pkl")
    print(f"validating against {probe.model_name} L{probe.layer}", flush=True)

    for concept in args.concept:
        c = CONCEPTS[concept]
        datasets = load_splits(concept)
        print(f"\n=== {concept}: {len(datasets)} split(s) — "
              + ", ".join(f"{k}({len(v)})" for k, v in datasets.items()), flush=True)
        status = prefetch_eval_activations(
            c["cache"],
            datasets,
            KaggleActivationSource(args.owner, c["dataset_slug"], args.file_name),
            model_name=probe.model_name,
            layer=int(probe.layer),
            cache_stem="acts_full.pt",
        )
        for split, st in sorted(status.items()):
            print(f"  {split:26s} {st}", flush=True)


if __name__ == "__main__":
    main()
