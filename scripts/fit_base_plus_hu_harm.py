#!/usr/bin/env python
"""Fit `base ∪ <jsonl>` for the HUMAN HARM concept and score dev + eval.

The hu_harm counterpart of `scripts/fit_base_plus.py` (which is pinned to the
instructions concept and its nemotron arm). Every knob is copied from
`configs/gen_gemma27b_hu_harm.md` so a fit here is apples-to-apples with that arm's
`probe_iter*.pkl`:

    probe      google/gemma-3-27b-it, layer 32, linear_then_softmax, single (no ensemble)
    base       data/hu_harm_llama70b_50.jsonl
    dev        dev_samples/hu_ha        (290 rows, 4 splits)
    eval       eval_sets/hu_ha          (866 rows, 4 splits, FULL — no subsampling)
    transforms combine_consecutive_messages = convert_tool_to_assistant = True
    seed       42

`--base-only` skips the fit and scores `probes/gen_gemma27b_hu_harm/probe_iter0.pkl`
— the probe trained on the 50 base rows alone — as the reference point every generated
set is measured against.

Measured, this setup:

    base only (50 rows, probe_iter0)          dev 0.87589   eval 0.85232
    base ∪ hu_harm_gptoss_600 (650 rows)      dev 0.89724   eval 0.87323

The eval and dev activations come from Kaggle (`prefetch_*`, the `kaggle:` block of the
arm-2 config), so neither is ever extracted locally. Only the generated samples and the
50 base rows go through the 27B model, once each, into the shared per-sample cache.

Examples:
    ${REPO_ROOT}/.venv_claude/bin/python scripts/fit_base_plus_hu_harm.py --base-only
    ${REPO_ROOT}/.venv_claude/bin/python scripts/fit_base_plus_hu_harm.py \\
        data/hu_harm_gptoss_600.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PROBE_DIR = REPO / "probes/gen_gemma27b_hu_harm"
BASE_DATA = REPO / "data/hu_harm_llama70b_50.jsonl"
DEV_DATA = REPO / "dev_samples/hu_ha"
EVAL_DIR = REPO / "eval_sets/hu_ha"
BASE_CACHE = REPO / "cache_gen_gemma27b_hu_harm/base_activations"
EVAL_CACHE = REPO / "cache_gen_gemma27b_hu_harm/eval_activations"
BASE_PROBE = PROBE_DIR / "probe_iter0.pkl"  # trained on the 50 base rows alone

MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
POS_LABEL = "harmful_to_human"
NEG_LABEL = "not_harmful_to_human"

SEED, COMBINE, CONVERT = 42, True, True

# The kaggle: block of configs/gen_gemma27b_hu_harm.md, verbatim.
KAGGLE_OWNER = "anku7890"
EVAL_SLUG, EVAL_FILE = "{slug}-gemmaevalpt", "{split}-gemmaeval.pt"
DEV_SLUG, DEV_FILE = "{slug}-gemmadevpt", "{split}-gemmadev.pt"


def _eval_source():
    from agentic_redteam.kaggle_activations import KaggleActivationSource

    return KaggleActivationSource(KAGGLE_OWNER, EVAL_SLUG, EVAL_FILE)


def prefetch_dev() -> None:
    """Assemble the dev blob from Kaggle into the exact path the fit looks for."""
    from agentic_redteam.kaggle_activations import (
        KaggleActivationSource,
        prefetch_dev_activations,
    )
    from agentic_redteam.retrain import _dev_activation_cache_path

    dev_files = sorted(DEV_DATA.glob("*.jsonl"))
    if not dev_files:
        raise SystemExit(f"{DEV_DATA} holds no *.jsonl splits")
    BASE_CACHE.mkdir(parents=True, exist_ok=True)
    prefetch_dev_activations(
        _dev_activation_cache_path(BASE_CACHE, dev_files, MODEL_NAME, LAYER, COMBINE, CONVERT),
        dev_files,
        KaggleActivationSource(KAGGLE_OWNER, DEV_SLUG, DEV_FILE),
        model_name=MODEL_NAME,
        layer=LAYER,
        verbose=True,
    )


def report(name: str, n_rows: int | None, dev: dict[str, float] | None, df) -> None:
    ev = float(df.loc[df["dataset"] == "mean", "auroc"].iloc[0])
    print(f"\n=== {name}" + (f"  ({n_rows} training rows)" if n_rows else "") + " ===")
    if dev is not None:
        print(f"  dev  mean {dev['mean']:.5f}")
        for k, v in sorted(dev.items()):
            if k != "mean":
                print(f"      {k:<28} {v:.5f}")
    print(f"  eval mean {ev:.5f}")
    for _, r in df[df["dataset"] != "mean"].iterrows():
        print(f"      {r['dataset']:<28} {r['auroc']:.5f}")
    return ev


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("samples", type=Path, nargs="?", help="JSONL of {inputs, labels} rows")
    ap.add_argument("--base-only", action="store_true",
                    help="score probe_iter0 (base 50 rows) — the reference point")
    ap.add_argument("--no-base", action="store_true", help="fit the samples alone")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-prefetch", action="store_true",
                    help="assume the eval/dev activation caches are already populated")
    args = ap.parse_args()

    if not args.base_only and args.samples is None:
        ap.error("give a samples JSONL, or --base-only")

    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import (
        retrain_probe,
        score_probe_on_dev,
        warm_sample_activation_cache,
    )

    if not args.skip_prefetch:
        prefetch_dev()

    if args.base_only:
        dev = score_probe_on_dev(
            BASE_PROBE, DEV_DATA, BASE_CACHE,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=False,
        )
        df = evaluate_probe(
            BASE_PROBE, EVAL_DIR, EVAL_CACHE, max_samples=None, seed=SEED,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            kaggle_source=_eval_source(),
        )
        report(f"base only ({BASE_PROBE.name})", 50, dev, df)
        return

    # On disk `inputs` is a JSON-encoded string (the tuberlens LabelledDataset schema
    # every file under data/ and eval_sets/ uses). The in-memory `samples` path of
    # retrain_probe wants it already parsed — `_dicts_to_labelled_dataset` calls
    # `m.get("role")` on each message — so decode it here. Rows that already carry a
    # list (a `_dump_labelled_dataset` snapshot, e.g. accepted_iter*.jsonl) pass through.
    rows = [json.loads(line) for line in args.samples.open() if line.strip()]
    rows = [
        {
            "inputs": json.loads(r["inputs"]) if isinstance(r["inputs"], str) else r["inputs"],
            "labels": r["labels"],
        }
        for r in rows
    ]
    npos = sum(1 for r in rows if r["labels"] == POS_LABEL)
    nneg = sum(1 for r in rows if r["labels"] == NEG_LABEL)
    if npos + nneg != len(rows):
        raise SystemExit(f"{args.samples}: {len(rows) - npos - nneg} rows carry an unknown label")
    prefix = "no base ∪ " if args.no_base else "base ∪ "
    print(f"{prefix}{len(rows)} ({args.samples.name}): {npos} {POS_LABEL} / {nneg} {NEG_LABEL}")

    out = args.out or PROBE_DIR / f"baseplus_{args.samples.stem}.pkl"
    warm_sample_activation_cache(
        rows, base_probe_path=BASE_PROBE, base_activation_cache_dir=BASE_CACHE,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT, verbose=True,
    )
    res = retrain_probe(
        samples=rows, base_probe_path=BASE_PROBE,
        base_training_data_path=None if args.no_base else BASE_DATA,
        new_probe_path=out, dev_data_path=DEV_DATA, seed=SEED, base_data_fraction=1.0,
        base_activation_cache_dir=BASE_CACHE, combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT, verbose=True,
    )
    df = evaluate_probe(
        out, EVAL_DIR, EVAL_CACHE, max_samples=None, seed=SEED,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
        kaggle_source=_eval_source(),
    )
    report(f"{prefix}{args.samples.name}", res.n_training_samples_total, res.dev_auroc, df)


if __name__ == "__main__":
    main()
