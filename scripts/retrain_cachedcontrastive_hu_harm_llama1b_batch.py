#!/usr/bin/env python
"""Retrain experiment10_cloud's two BATCH arms WITH contrastive pairs, but using
ONLY pairs already present in the on-disk contrastive cache — no LLM calls.

Why this exists: the natural companion to
``retrain_nocontrastive_hu_harm_llama1b_batch.py --min-judge-confidence 10`` is
the *original* pipeline (filter + contrastive) with only the confidence gate
changed. That subset is not fully covered by the cache each arm's original run
wrote — ``filter_dataset`` refits on the smaller conf-10 set and keeps a few
records the original run had dropped, so their pairs were never generated. With
no ``OPENROUTER_API_KEY`` those would trip the circuit breaker.

So ``generate_contrastive_dataset`` is monkeypatched to a **cache-only** variant
that emulates the real function's failure path exactly:

  * cache hit  → keep the source record AND append its cached contrastive pair;
  * cache miss → **drop the source record too**, which is precisely what the real
    function does when every generation attempt fails (``preprocessing.py``:
    "both the source record and its (missing) contrastive pair are dropped,
    rather than keeping an unpaired source").

So the output is exactly what the real pipeline would have produced had those
uncached generations failed — and, importantly, the exact 50/50 class balance
that pairing gives is preserved.

APPROXIMATION, stated plainly: the dropped records are not a random sample —
they are exactly the records the *original* (conf>=7) run's filter_dataset had
discarded. Measured miss counts are small (dsv4-pro 4/2/22, gpt-oss 1/2/5 across
the three iteration cutoffs, out of 136/221/309 and 181/250/355 filtered
records), but iter3 of the dsv4-pro arm loses ~7% of its records. Treat that arm's
iter3 as approximate. Re-run with an API key for the exact figure.

Every other knob is identical to
``retrain_nocontrastive_hu_harm_llama1b_batch.py`` — same caches, same
seed/test_size/transforms, same iteration cutoffs, same probe donor.

Run:
    .venv_claude/bin/python scripts/retrain_cachedcontrastive_hu_harm_llama1b_batch.py \
        --min-judge-confidence 10 --tag _conf10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# --- monkeypatch: filter stays, contrastive becomes cache-only ---------------
import agentic_redteam.preprocessing as _pp

_STATS = {"hits": 0, "misses": 0}


def _cached_only_contrastive(
    records,
    pos_class_label: str,
    neg_class_label: str,
    text_key: str = "inputs",
    label_key: str = "labels",
    cache_path=None,
    concept_description: str = "",
    label_guidance=None,
    **_unused,
):
    """Cache-only stand-in for generate_contrastive_dataset. No LLM calls.

    Mirrors the real function's contract: returns ``kept_sources + pairs``, and
    drops the source of any record whose pair is not cached (the real function's
    behaviour when generation fails after all retries).
    """
    records = list(records)
    valid = [
        r for r in records if r.get(label_key) in (pos_class_label, neg_class_label)
    ]
    if not valid:
        print("No records with valid labels found; skipping contrastive generation.")
        return records

    cache = _pp._load_cache(Path(cache_path) if cache_path is not None else None)

    def _target(label: str) -> str:
        return neg_class_label if label == pos_class_label else pos_class_label

    pairs: list[dict] = []
    dropped_ids: set[int] = set()
    for record in valid:
        messages = _pp._extract_messages(record, text_key)
        target_label = _target(str(record.get(label_key, "")))
        key = _pp._cache_key(
            messages,
            target_label,
            _pp._guidance_fingerprint(concept_description, target_label, label_guidance),
        )
        cached = cache.get(key)
        if cached is not None:
            pairs.append(cached)
        else:
            dropped_ids.add(id(record))

    _STATS["hits"] += len(pairs)
    _STATS["misses"] += len(dropped_ids)

    if dropped_ids:
        records = [r for r in records if id(r) not in dropped_ids]
        print(
            f"  [contrastive] CACHE-ONLY MODE: dropped {len(dropped_ids)} source "
            f"records (and their pairs) with no cached contrastive partner"
        )
    print(
        f"Contrastive generation (cache-only): {len(pairs)} pairs reused from "
        f"{len(valid)} source records; {len(dropped_ids)} uncached"
    )
    return records + pairs


_pp.generate_contrastive_dataset = _cached_only_contrastive

from agentic_redteam.config import PreprocessingConfig  # noqa: E402
from agentic_redteam.retrain import retrain_probe  # noqa: E402
from agentic_redteam.evaluation import evaluate_probe  # noqa: E402

# --- fixed inputs, identical to the nocontrastive script ---------------------
BASE_TRAINING_DATA = REPO / "data" / "hu_harm_llama70b_50.jsonl"
EVAL_DATASET_DIR = REPO / "eval_sets/hu_ha"
BASE_ACT_CACHE = REPO / "results_hu_harm_llama70b50_batch_ablation" / "base_activations"
EVAL_ACT_CACHE = REPO / "results_hu_harm_llama70b50_batch_ablation" / "eval_activations"

FILTER_PERCENTILE = 0.8
MIN_JUDGE_CONFIDENCE = 7
TEST_SIZE = 0.2
SEED = 42
COMBINE_CONSECUTIVE = True
CONVERT_TOOL_TO_ASSISTANT = True
EVAL_MAX_SAMPLES = None
ASSISTANT_CENTRIC = True

# model/provider are placeholders: the patched generator never reaches an LLM.
PREPROC = PreprocessingConfig(
    model="__unused_cache_only__",
    provider="openrouter",
    filter_percentile=FILTER_PERCENTILE,
    assistant_centric=ASSISTANT_CENTRIC,
)

# Each arm's own contrastive cache, written by its original run.
RUNS = [
    {
        "name": "hu_harm_llama1b_deepseekv4pro_batch",
        "probe_dir": REPO / "probes" / "hu_harm_llama1b_deepseekv4pro_batch",
        "fp": REPO
        / "results_hu_harm_llama70b50_deepseekv4pro_batch"
        / "deepseekv4pro_probing_fp.jsonl",
        "fn": REPO
        / "results_hu_harm_llama70b50_deepseekv4pro_batch"
        / "deepseekv4pro_probing_fn.jsonl",
    },
    {
        "name": "hu_harm_llama1b_gptoss120b_batch",
        "probe_dir": REPO / "probes" / "hu_harm_llama1b_gptoss120b_batch",
        "fp": REPO
        / "results_hu_harm_llama70b50_gptoss120b_batch"
        / "gptoss120b_probing_fp.jsonl",
        "fn": REPO
        / "results_hu_harm_llama70b50_gptoss120b_batch"
        / "gptoss120b_probing_fn.jsonl",
    },
]

CUTOFFS = [0, 1, 2]
NEW_RESULTS_DIR = REPO / "results_hu_harm_llama70b50_batch_nocontrastive"


def write_iteration_subset(src: Path, dst: Path, cutoff: int) -> int:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-judge-confidence", type=int, default=MIN_JUDGE_CONFIDENCE)
    p.add_argument("--tag", default="", help="Suffix for probe dirs and CSVs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    min_conf = args.min_judge_confidence
    tag = args.tag
    print(f">>> CACHE-ONLY contrastive; min_judge_confidence={min_conf} tag={tag!r}")

    NEW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for run in RUNS:
        name = run["name"]
        base_probe = run["probe_dir"] / "probe_iter0.pkl"
        # The arm's ORIGINAL contrastive cache — read-only in this mode, since the
        # patched generator never writes.
        cache_path = run["probe_dir"] / "contrastive_cache.jsonl"
        out_dir = REPO / "probes" / f"{name}_cachedcontrastive{tag}"
        subset_dir = out_dir / "subsets"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}\nRUN: {name}\n{'=' * 70}")
        for required in (base_probe, cache_path, run["fp"], run["fn"]):
            if not Path(required).exists():
                raise FileNotFoundError(f"Missing input: {required}")

        eval_frames = [eval_to_rows(base_probe, "iter0")]
        current_probe = base_probe

        for k in CUTOFFS:
            it_label = f"iter{k + 1}"
            print(f"\n--- {name}: {it_label} (successes with iteration <= {k}) ---")

            fp_sub = subset_dir / f"{it_label}_fp.jsonl"
            fn_sub = subset_dir / f"{it_label}_fn.jsonl"
            write_iteration_subset(run["fp"], fp_sub, k)
            write_iteration_subset(run["fn"], fn_sub, k)

            new_probe = out_dir / f"probe_{it_label}.pkl"
            result = retrain_probe(
                jsonl_path=[fp_sub, fn_sub],
                base_probe_path=current_probe,
                base_training_data_path=BASE_TRAINING_DATA,
                new_probe_path=new_probe,
                preprocessing=PREPROC,
                contrastive_cache_path=cache_path,  # read-only: patched gen never writes
                postprocessed_out_path=out_dir / f"redteam_postprocessed_{it_label}.jsonl",
                min_judge_confidence=min_conf,
                test_size=TEST_SIZE,
                seed=SEED,
                base_activation_cache_dir=BASE_ACT_CACHE,
                combine_consecutive_messages=COMBINE_CONSECUTIVE,
                convert_tool_to_assistant=CONVERT_TOOL_TO_ASSISTANT,
                verbose=True,
            )
            print(
                f"Saved probe -> {new_probe}  (n_redteam={result.n_redteam_samples}, "
                f"n_total={result.n_training_samples_total})"
            )
            current_probe = new_probe
            eval_frames.append(eval_to_rows(new_probe, it_label))

        comparison = pd.concat(eval_frames, ignore_index=True)
        out_csv = NEW_RESULTS_DIR / f"{name}_cachedcontrastive{tag}_comparison.csv"
        comparison.to_csv(out_csv, index=False)
        print(f"\nSaved comparison table -> {out_csv}")
        print(comparison.to_string(index=False))

    print(
        f"\nAll done. Cache-only totals: {_STATS['hits']} pairs reused, "
        f"{_STATS['misses']} source records dropped for want of a cached pair."
    )


if __name__ == "__main__":
    main()
