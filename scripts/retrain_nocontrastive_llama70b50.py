#!/usr/bin/env python
"""Retrain the llama70b50 probes on red-team data WITHOUT contrastive pairs, then eval.

Ablation of the contrastive-augmentation step for three attacker runs:

    15July_hh_llama70b50            (attacker: Llama-3.3-70B)
    15July_hh_llama70b50_gpt51chat  (attacker: gpt-5.1-chat)
    15July_hh_llama70b50_gptoss120b (attacker: gpt-oss-120b)

The original pipeline retrained each probe on ``base ∪ preprocessed(successes)``
where preprocessing = ``filter_dataset`` (drop confounders) + ``generate_contrastive_dataset``
(add opposite-class pairs). This script reproduces the same three retrain iterations
per run but **keeps filter_dataset and drops the contrastive-pair generation** — i.e.
plain filtered red-team successes concatenated with the base data.

Everything the original runs produced is REUSED, nothing is overwritten:
  * The per-conversation red-team activation cache and the base-split activation
    cache are shared with the originals (content-keyed), so the filtered originals
    hit the cache and no Llama forward passes are recomputed (no model load on a
    full hit).
  * The eval activation cache (``acts_full.pt`` per split) is reused read-only.
  * New probes, postprocessed snapshots, iteration subsets and comparison CSVs are
    written under fresh ``*_nocontrastive`` paths.

"Keep filter, drop contrastive" is implemented by monkeypatching
``agentic_redteam.preprocessing.generate_contrastive_dataset`` to an identity that
returns the filtered records unchanged (no LLM calls). ``retrain_probe`` is then
driven with the preprocessing config so ``filter_dataset`` still runs.

Run:
    .venv_claude/bin/python scripts/retrain_nocontrastive_llama70b50.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# --- monkeypatch: filter stays, contrastive becomes a no-op -----------------
import agentic_redteam.preprocessing as _pp


def _no_contrastive(records, **_kwargs):
    """Identity stand-in for generate_contrastive_dataset: keep the filtered
    originals, add no opposite-class pairs, make no LLM calls."""
    return list(records)


_pp.generate_contrastive_dataset = _no_contrastive

from agentic_redteam.config import PreprocessingConfig  # noqa: E402
from agentic_redteam.retrain import retrain_probe  # noqa: E402
from agentic_redteam.evaluation import evaluate_probe  # noqa: E402

# --- fixed inputs shared by all three runs ----------------------------------
BASE_TRAINING_DATA = REPO / "data" / "hu_harm_llama70b_50.jsonl"
EVAL_DATASET_DIR = REPO / "eval_sets/hu_ha"
# Shared, already-populated caches from the original runs (see the configs).
BASE_ACT_CACHE = REPO / "results_hu_harm_llama70b50" / "llama1b_base_activations"
EVAL_ACT_CACHE = REPO / "results_hu_harm_prompt_llama1b" / "llama1b_eval_activations"

# Training / eval knobs, matching the original configs exactly so the caches hit.
FILTER_PERCENTILE = 0.8
TEST_SIZE = 0.2
SEED = 42
COMBINE_CONSECUTIVE = True
CONVERT_TOOL_TO_ASSISTANT = True
EVAL_MAX_SAMPLES = None  # eval_max_samples: 0 in config → full split → acts_full.pt

# Preprocessing config: only filter_percentile is used (contrastive is patched out),
# so provider/model are placeholders and never trigger an LLM call.
PREPROC = PreprocessingConfig(
    model="__unused_no_contrastive__",
    provider="openrouter",
    filter_percentile=FILTER_PERCENTILE,
)

# One entry per attacker run.
RUNS = [
    {
        "name": "15July_hh_llama70b50",
        "fp": REPO / "results_hu_harm_llama70b50" / "llama1b_probing_fp.jsonl",
        "fn": REPO / "results_hu_harm_llama70b50" / "llama1b_probing_fn.jsonl",
    },
    {
        "name": "15July_hh_llama70b50_gpt51chat",
        "fp": REPO / "results_hu_harm_llama70b50_gpt51chat" / "gpt51chat_probing_fp.jsonl",
        "fn": REPO / "results_hu_harm_llama70b50_gpt51chat" / "gpt51chat_probing_fn.jsonl",
    },
    {
        "name": "15July_hh_llama70b50_gptoss120b",
        "fp": REPO / "results_hu_harm_llama70b50_gptoss120b" / "gptoss120b_probing_fp.jsonl",
        "fn": REPO / "results_hu_harm_llama70b50_gptoss120b" / "gptoss120b_probing_fn.jsonl",
    },
]

# Reproduce the same 3 retrains the original loop did. Cutoff k → probe_iter{k+1},
# trained on successes with iteration <= k (iter1: iter-0 successes; iter2: 0+1;
# iter3: 0+1+2), matching how the JSONL grew across the original iterations.
CUTOFFS = [0, 1, 2]

NEW_RESULTS_DIR = REPO / "results_hu_harm_llama70b50_nocontrastive"


def write_iteration_subset(src: Path, dst: Path, cutoff: int) -> int:
    """Copy rows of `src` with 0 <= iteration <= cutoff into `dst`. Returns row count."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open() as fh, dst.open("w") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            it = rec.get("iteration", -1)
            if isinstance(it, int) and 0 <= it <= cutoff:
                out.write(json.dumps(rec) + "\n")
                n += 1
    return n


def eval_to_rows(probe_path: Path, label: str) -> pd.DataFrame:
    df = evaluate_probe(
        probe_path,
        EVAL_DATASET_DIR,
        EVAL_ACT_CACHE,
        max_samples=EVAL_MAX_SAMPLES,
        seed=SEED,
        combine_consecutive_messages=COMBINE_CONSECUTIVE,
        convert_tool_to_assistant=CONVERT_TOOL_TO_ASSISTANT,
    )
    df = df.copy()
    df.insert(0, "round", label)
    return df


def main() -> None:
    NEW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for run in RUNS:
        name = run["name"]
        orig_probe_dir = REPO / "probes" / name
        base_probe = orig_probe_dir / "probe_iter0.pkl"  # arch + metadata donor (unchanged)
        out_dir = REPO / "probes" / f"{name}_nocontrastive"
        subset_dir = out_dir / "subsets"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}\nRUN: {name}\n{'=' * 70}")
        if not base_probe.exists():
            raise FileNotFoundError(f"Base probe not found: {base_probe}")

        eval_frames: list[pd.DataFrame] = []

        # iter0 = the untouched base probe (no red-team data, no contrastive) —
        # evaluate the existing probe to fill the iter0 row of the CSV.
        print("Evaluating iter0 (original base probe, unchanged) ...")
        eval_frames.append(eval_to_rows(base_probe, "iter0"))

        for k in CUTOFFS:
            it_label = f"iter{k + 1}"
            print(f"\n--- {name}: {it_label} (successes with iteration <= {k}) ---")

            fp_sub = subset_dir / f"{it_label}_fp.jsonl"
            fn_sub = subset_dir / f"{it_label}_fn.jsonl"
            n_fp = write_iteration_subset(run["fp"], fp_sub, k)
            n_fn = write_iteration_subset(run["fn"], fn_sub, k)
            print(f"Subset rows: fp={n_fp}, fn={n_fn}")

            new_probe = out_dir / f"probe_{it_label}.pkl"
            postproc = out_dir / f"redteam_postprocessed_{it_label}.jsonl"

            result = retrain_probe(
                jsonl_path=[fp_sub, fn_sub],
                base_probe_path=base_probe,
                base_training_data_path=BASE_TRAINING_DATA,
                new_probe_path=new_probe,
                preprocessing=PREPROC,          # filter_dataset runs; contrastive patched out
                contrastive_cache_path=None,    # unused (no contrastive generation)
                postprocessed_out_path=postproc,
                test_size=TEST_SIZE,
                seed=SEED,
                base_activation_cache_dir=BASE_ACT_CACHE,   # REUSE existing activations
                combine_consecutive_messages=COMBINE_CONSECUTIVE,
                convert_tool_to_assistant=CONVERT_TOOL_TO_ASSISTANT,
                verbose=True,
            )
            print(f"Saved probe -> {new_probe}  (n_redteam={result.n_redteam_samples})")

            print(f"Evaluating {it_label} ...")
            eval_frames.append(eval_to_rows(new_probe, it_label))

        comparison = pd.concat(eval_frames, ignore_index=True)
        out_csv = NEW_RESULTS_DIR / f"{name}_nocontrastive_comparison.csv"
        comparison.to_csv(out_csv, index=False)
        print(f"\nSaved comparison table -> {out_csv}")
        print(comparison.to_string(index=False))

    print(f"\nAll done. Comparison CSVs are under {NEW_RESULTS_DIR}")


if __name__ == "__main__":
    main()
