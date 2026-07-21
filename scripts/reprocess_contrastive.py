#!/usr/bin/env python
"""Re-apply the contrastive-generation procedure to an already-completed run.

Motivation
----------
Older runs kept a red-team source sample even when its contrastive pair failed
to generate (leaving it unpaired). ``generate_contrastive_dataset`` now retries a
failed generation ``max_generation_retries`` times and, if it still fails, drops
*both* the source and its (missing) pair. This script re-runs that exact
procedure over a finished run's persisted successes and writes corrected
``redteam_postprocessed_iter{N}`` datasets — **without retraining any probe**.

Why it's cheap: the already-generated pairs are reused from the run's
``contrastive_cache.jsonl`` (keyed by source), so only the *previously-failed*
sources incur fresh LLM calls (now with retries).

Faithful reproduction of each iteration's snapshot
--------------------------------------------------
``redteam_postprocessed_iter{N}.jsonl`` was built from the red-team successes
accumulated **up to that iteration's retrain**: iter1 ← iteration-0 successes,
iter2 ← iterations 0–1, iter3 ← iterations 0–2. The cumulative JSONL now holds
every iteration, so we reconstruct each snapshot by keeping only rows with
``iteration <= N-1`` (and ``judge_confidence >= judge.confidence_threshold``,
matching the retrain gate). We then run the same
filter_dataset → generate_contrastive_dataset → message-transforms → dump
pipeline the retrain uses (see ``retrain._build_redteam_dataset`` /
``_apply_message_transforms`` / ``_dump_labelled_dataset``).

Requires ``OPENROUTER_API_KEY`` (the hh preprocessing model is
``openai/gpt-5.1-chat`` via OpenRouter) for the retried sources.

Defaults target the 13 July hh run; override via flags for another run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_redteam.config import load_config
from agentic_redteam.persistence import JsonlStore
from agentic_redteam.retrain import (
    _apply_message_transforms,
    _build_redteam_dataset,
    _dump_labelled_dataset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_cumulative_successes(jsonl_paths, max_iteration: int, min_confidence: int):
    """Successful records with ``0 <= iteration <= max_iteration`` and
    ``judge_confidence >= min_confidence``, pooled across the given JSONL files."""
    recs = []
    for p in jsonl_paths:
        p = Path(p)
        if not p.exists():
            print(f"  [warn] JSONL not found, skipping: {p}")
            continue
        recs.extend(JsonlStore(path=p).iter_successes())
    kept = [
        r
        for r in recs
        if 0 <= r.iteration <= max_iteration and r.judge_confidence >= min_confidence
    ]
    return kept


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "gemma27_hu_harm_prompt.md",
        help="Config whose preprocessing/probe/eval sections + JSONL paths to reuse.",
    )
    parser.add_argument(
        "--probe-out-dir",
        type=Path,
        default=REPO_ROOT / "probes" / "13July_hh_guard",
        help="Dir holding contrastive_cache.jsonl and where cleaned files are written.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Which postprocessed_iter{N} snapshots to regenerate (default 1 2 3).",
    )
    parser.add_argument(
        "--out-suffix",
        default="_cleaned",
        help="Suffix inserted before .jsonl for outputs (default '_cleaned'; "
        "use '' to OVERWRITE the originals — not recommended).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if config.preprocessing is None:
        parser.error(f"{args.config} has no preprocessing: section; nothing to reprocess.")

    pos = config.probe.pos_class_label
    neg = config.probe.neg_class_label
    min_conf = int(config.judge.confidence_threshold or 0)
    combine = bool(getattr(config.eval, "combine_consecutive_messages", False))
    convert = bool(getattr(config.eval, "convert_tool_to_assistant", False))
    contrastive_cache = args.probe_out_dir / "contrastive_cache.jsonl"

    # Resolve the per-error-type JSONL paths exactly as the run wrote them.
    error_types = config.probe.error_types
    jsonl_paths = [config.jsonl_path_for(et) for et in error_types]

    print(f"Config:            {args.config}")
    print(f"Classes:           pos={pos!r}  neg={neg!r}")
    print(f"Judge conf gate:   >= {min_conf}")
    print(f"Msg transforms:    combine={combine}  convert_tool_to_assistant={convert}")
    print(f"Preprocessing:     {config.preprocessing.provider}/{config.preprocessing.model}  "
          f"filter_pct={config.preprocessing.filter_percentile}  "
          f"max_retries={config.preprocessing.max_generation_retries}")
    print(f"Contrastive cache: {contrastive_cache} "
          f"({'exists' if contrastive_cache.exists() else 'MISSING'})")
    print(f"Red-team JSONLs:   {[str(p) for p in jsonl_paths]}")
    print(f"Iterations:        {args.iterations}\n")

    summary = []
    for n in args.iterations:
        max_iter = n - 1
        print(f"\n########## postprocessed_iter{n}  (cumulative iteration <= {max_iter}) ##########")
        successes = _load_cumulative_successes(jsonl_paths, max_iter, min_conf)
        print(f"Red-team successes loaded: {len(successes)} "
              f"(iteration<= {max_iter}, judge_confidence >= {min_conf})")
        if not successes:
            print("  nothing to process; skipping.")
            continue

        # Exact retrain preprocessing path: filter_dataset -> contrastive (retry+drop).
        redteam_dataset = _build_redteam_dataset(
            successes, pos, neg, config.preprocessing, contrastive_cache, verbose=True
        )
        # Match the dump order used in retrain: transforms are applied before dumping.
        redteam_dataset = _apply_message_transforms(redteam_dataset, combine, convert)

        out_path = args.probe_out_dir / f"redteam_postprocessed_iter{n}{args.out_suffix}.jsonl"
        n_written = _dump_labelled_dataset(redteam_dataset, out_path)
        print(f"Saved {n_written} cleaned red-team samples to {out_path}")

        orig = args.probe_out_dir / f"redteam_postprocessed_iter{n}.jsonl"
        orig_n = sum(1 for _ in orig.open()) if orig.exists() else None
        summary.append((n, orig_n, n_written, out_path))

    print("\n===== SUMMARY =====")
    print(f"{'iter':>4}  {'original':>8}  {'cleaned':>8}  {'delta':>6}  output")
    for n, orig_n, new_n, out_path in summary:
        delta = "n/a" if orig_n is None else f"{new_n - orig_n:+d}"
        print(f"{n:>4}  {str(orig_n):>8}  {new_n:>8}  {delta:>6}  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
