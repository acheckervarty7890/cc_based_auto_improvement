#!/usr/bin/env python
"""Download every published eval + dev activation blob for the three gemma-3-27b concepts.

Goes through the repo's own ``kaggle_activations`` prefetch functions rather than the
Kaggle CLI, so each blob is validated against the probe's ``model_name``/``layer`` and the
split's row count before it may be used, and lands at the exact path the loop looks for:

* eval — one file per split, ``<activations_cache_dir>/<split>-acts_full.pt``, the name
  ``get_performances`` derives.
* dev  — the splits are fetched individually and concatenated into the ONE content-hashed
  blob ``_dev_activation_cache_path`` names, in ``sorted(glob("*.jsonl"))`` order.

Transforms are taken from the run configs (``combine_consecutive_messages`` and
``convert_tool_to_assistant`` both on), which is part of the dev cache key.

    KAGGLE_CONFIG_DIR=<dir holding kaggle.json> .venv_claude/bin/python \
        scripts/fetch_all_kaggle_activations.py [--concept highstakes] [--kind eval]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from agentic_redteam.kaggle_activations import (  # noqa: E402
    KaggleActivationSource,
    prefetch_dev_activations,
    prefetch_eval_activations,
)
from agentic_redteam.retrain import _dev_activation_cache_path  # noqa: E402

MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
OWNER = "anku7890"
COMBINE = True
CONVERT = True

CONCEPTS = {
    "highstakes": dict(
        eval_dir=REPO / "eval_sets/highstakes",
        dev_dir=REPO / "dev_samples/highstakes",
        pos="high-stakes",
        neg="low-stakes",
        cache=REPO / "cache_gen_gemma27b_highstakes",
    ),
    "hu_harm": dict(
        eval_dir=REPO / "eval_sets/hu_ha",
        dev_dir=REPO / "dev_samples/hu_ha",
        pos="harmful_to_human",
        neg="not_harmful_to_human",
        cache=REPO / "cache_gen_gemma27b_hu_harm",
    ),
    "instructions": dict(
        eval_dir=REPO / "eval_sets/instructions",
        dev_dir=REPO / "dev_samples/instructions",
        pos="assistant_follows_the_instruction",
        neg="assistant_does_not_follow_the_instruction",
        cache=REPO / "cache_gen_gemma27b_instructions",
    ),
}

EVAL_SOURCE = KaggleActivationSource(OWNER, "{slug}-gemmaevalpt", "{split}-gemmaeval.pt")
DEV_SOURCE = KaggleActivationSource(OWNER, "{slug}-gemmadevpt", "{split}-gemmadev.pt")


def _load_eval_datasets(cfg: dict) -> dict:
    from tuberlens.interfaces.dataset import LabelledDataset

    out = {}
    for path in sorted(Path(cfg["eval_dir"]).glob("*.jsonl")):
        out[path.stem] = LabelledDataset.load_from(
            path,
            pos_class_label=cfg["pos"],
            neg_class_label=cfg["neg"],
            combine_consecutive_messages=COMBINE,
            convert_tool_to_assistant=CONVERT,
        )
    return out


def do_eval(name: str, cfg: dict) -> None:
    cache_dir = Path(cfg["cache"]) / "eval_activations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    datasets = _load_eval_datasets(cfg)
    print(f"\n### {name} / eval — {len(datasets)} splits → {cache_dir}", flush=True)
    result = prefetch_eval_activations(
        cache_dir,
        datasets,
        EVAL_SOURCE,
        model_name=MODEL_NAME,
        layer=LAYER,
        cache_stem="acts_full.pt",
    )
    for split, status in sorted(result.items()):
        print(f"    {split:32s} {status}", flush=True)


def do_dev(name: str, cfg: dict) -> None:
    base_cache = Path(cfg["cache"]) / "base_activations"
    base_cache.mkdir(parents=True, exist_ok=True)
    dev_files = sorted(Path(cfg["dev_dir"]).glob("*.jsonl"))
    target = _dev_activation_cache_path(base_cache, dev_files, MODEL_NAME, LAYER, COMBINE, CONVERT)
    print(f"\n### {name} / dev — {len(dev_files)} splits → {target}", flush=True)
    status = prefetch_dev_activations(
        target, dev_files, DEV_SOURCE, model_name=MODEL_NAME, layer=LAYER
    )
    print(f"    dev blob: {status}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", choices=sorted(CONCEPTS), action="append")
    ap.add_argument("--kind", choices=("eval", "dev"), action="append")
    args = ap.parse_args()
    concepts = args.concept or list(CONCEPTS)
    kinds = args.kind or ["eval", "dev"]

    t0 = time.monotonic()
    failures: list[str] = []
    for name in concepts:
        cfg = CONCEPTS[name]
        for kind in kinds:
            try:
                (do_eval if kind == "eval" else do_dev)(name, cfg)
            except Exception as e:  # noqa: BLE001 — report every concept, don't stop at the first
                failures.append(f"{name}/{kind}: {type(e).__name__}: {e}")
                print(f"    FAILED {name}/{kind}: {type(e).__name__}: {e}", flush=True)

    print(f"\nElapsed {(time.monotonic() - t0) / 60:.1f} min")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("All blobs present and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
