#!/usr/bin/env python
"""Score every iteration of an experiment's probes on its eval splits, off published activations.

What this is for
----------------
``cli.iterative_retrain_main --eval`` scores each probe as the run produces it, and writes a
per-run comparison CSV. This script is the *after the fact* version: given probes that already
exist (``probe_iter{0..N}.pkl``, on whatever branch they were committed to) it re-scores all of
them on the concept's eval splits and emits one tidy CSV covering every (arm, iteration, split).

Why it is not a loop over ``evaluate_probe``
--------------------------------------------
``evaluate_probe`` is probe-major: each call loads every split's activation blob, scores one
probe, and drops them. For the high-stakes splits that is ~20 GB of blobs re-read per probe, and
this box's cgroup ceiling sits close enough to that to OOM (see the memory note on running one
activation-resident job at a time). So the loops are inverted: **split-major, probe-minor**. One
split's blob is loaded once, every probe of that concept is scored against it, and it is freed
before the next split. Peak residency is one split, and the blob is read from disk once instead
of once per probe.

Activations come from Kaggle
----------------------------
Both concepts' full-split gemma-3-27b L32 activations are published per split, so nothing here
loads the 27B model — ``prefetch_eval_activations`` downloads and validates each blob against
the probe's ``model_name`` / ``layer`` and the split's row count, and ``get_performances`` sees
a dataset that already carries its activations. The instructions blobs exist despite both
instruction configs stating they do not (``anku7890/{slug}-gemmaevalpt``, the same template the
hu_ha splits use); Kaggle's listing endpoints 403 for this account, which is how they came to be
believed missing.

``--verify-tokens`` re-tokenizes the first row of each split through tuberlens' own
``tokenize_inputs`` and compares it to the ``input_ids`` stored in the blob. That is the only
check that the published activations were computed under the *same message transforms* this
script loads the splits with — row count and probe metadata both match either way. It needs the
tokenizer only (no weights), and the gemma-3-27b snapshot is already cached locally.

Results are checkpointed per (concept, split, arm, iteration) to a JSONL as they land, and
re-running skips what is already there, so an interrupted download or eval resumes.

Typical use::

    .venv_claude/bin/python scripts/eval_probe_versions.py --verify-tokens
    .venv_claude/bin/python scripts/eval_probe_versions.py --concept hs --iterations 1,2,3
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Every experiment covered here: where its splits, its published blobs, its cache dir and its
# arms' probes live. Probe directories sit on the experiment's own branch, so they are given as
# absolute paths to a checkout (``git worktree add --detach`` is enough — nothing is written).
EXPERIMENTS: dict[str, dict] = {
    "hs": {
        "concept": "high-stakes",
        "eval_dataset_dir": REPO_ROOT / "eval_datasets",
        "cache_dir": REPO_ROOT / "results_hs_gemma27b_batch_ablation" / "eval_activations",
        # hs split stems are single words, so the dataset slug can use {split} directly.
        "kaggle_dataset_slug": "{split}gemmaevalpt",
        "kaggle_file_name": "{split}-gemmaeval.pt",
        "arms": {
            "gptoss120b": "/home/ubuntu/wt_exp9/probes/hs_gemma27b_gptoss120b_batch",
            "deepseekv4pro": "/home/ubuntu/wt_exp9/probes/hs_gemma27b_deepseekv4pro_batch",
        },
    },
    "instructions": {
        "concept": "instruction-following",
        "eval_dataset_dir": REPO_ROOT / "eval_instructions",
        "cache_dir": REPO_ROOT / "results_instructions_gemma27b_shared" / "eval_activations",
        # Every instructions split stem contains an underscore, which Kaggle slugs forbid — so
        # the DATASET name must use {slug} (the stem hyphenated). The FILE inside it is
        # unrestricted and keeps the raw stem.
        "kaggle_dataset_slug": "{slug}-gemmaevalpt",
        "kaggle_file_name": "{split}-gemmaeval.pt",
        "arms": {
            "gptoss": "/home/ubuntu/wt_instr/probes/instructions_gemma27b_gptoss",
            "nemotron": "/home/ubuntu/wt_instr/probes/instructions_gemma27b_nemotron",
        },
    },
}

# Both runs' configs set these; the published blobs were computed under them, so they are not
# knobs here — changing one would invalidate every cached blob.
COMBINE_CONSECUTIVE_MESSAGES = True
CONVERT_TOOL_TO_ASSISTANT = True
SEED = 42
CACHE_STEM = "acts_full.pt"  # what evaluate_probe derives for full splits (max_samples=None)
FPR = 0.01


def _load_probe(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _done_keys(results_path: Path) -> set[tuple]:
    """Keys already scored, so a re-run resumes instead of repeating."""
    if not results_path.exists():
        return set()
    done = set()
    with results_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["experiment"], r["split"], r["arm"], r["iteration"]))
    return done


def _verify_tokens(split: str, dataset, model_name: str, n: int = 1) -> str:
    """Re-tokenize the first ``n`` rows and compare against the blob's stored input_ids.

    The row count and the probe's model/layer are checked by ``_validate_blob``, but neither
    would notice a blob computed with different message transforms. The input_ids would.
    Returns a human-readable verdict rather than raising: a mismatch is worth reporting loudly
    but the caller decides whether it invalidates the run.
    """
    import torch
    from transformers import AutoTokenizer
    from tuberlens.model import LLMModel, tokenize_inputs

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = dict(LLMModel.default_tokenize_kwargs)
    stored = dataset.other_fields["input_ids"]
    for i in range(min(n, len(dataset))):
        want = tokenize_inputs(tokenizer, [dataset.inputs[i]], **kwargs)["input_ids"][0]
        got = stored[i]
        # The blob is zero-padded to the split's max length; compare only the real tokens.
        got = got[: len(want)]
        if not torch.equal(got.to(want.dtype), want):
            first = int((got.to(want.dtype) != want).nonzero()[0][0])
            return (
                f"MISMATCH at row {i}, token {first}: blob {got[first]} vs recomputed "
                f"{want[first]} — the published activations were NOT computed under "
                f"combine={COMBINE_CONSECUTIVE_MESSAGES} convert={CONVERT_TOOL_TO_ASSISTANT}"
            )
    return f"ok ({min(n, len(dataset))} row(s) token-identical)"


def run_experiment(
    name: str,
    *,
    iterations: list[int],
    results_path: Path,
    splits: list[str] | None,
    verify_tokens: bool,
) -> None:
    import numpy as np
    from tuberlens.evaluation import calculate_metrics
    from tuberlens.interfaces.dataset import LabelledDataset

    from agentic_redteam.evaluation import _assign_cached_activations, seed_everything
    from agentic_redteam.kaggle_activations import (
        KaggleActivationSource,
        prefetch_eval_activations,
    )

    spec = EXPERIMENTS[name]
    eval_dir = Path(spec["eval_dataset_dir"])
    cache_dir = Path(spec["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the probes up front: a missing pickle should fail before any download.
    probes: dict[tuple[str, int], object] = {}
    for arm, probe_dir in spec["arms"].items():
        for it in iterations:
            path = Path(probe_dir) / f"probe_iter{it}.pkl"
            if not path.exists():
                raise FileNotFoundError(f"{name}/{arm}: no probe at {path}")
            probes[(arm, it)] = _load_probe(path)
    reference = next(iter(probes.values()))

    split_names = splits or sorted(p.stem for p in eval_dir.glob("*.jsonl"))
    source = KaggleActivationSource(
        owner="anku7890",
        dataset_slug=spec["kaggle_dataset_slug"],
        file_name=spec["kaggle_file_name"],
    )
    done = _done_keys(results_path)

    print(f"\n=== {name} ({spec['concept']}) — {len(split_names)} splits x {len(probes)} probes")
    for split in split_names:
        keys = [(name, split, arm, it) for (arm, it) in probes]
        if all(k in done for k in keys):
            print(f"[{name}/{split}] already scored for every probe — skipping")
            continue

        seed_everything(SEED)  # full splits are not subsampled, but keep the call site identical
        dataset = LabelledDataset.load_from(
            eval_dir / f"{split}.jsonl",
            pos_class_label=reference.pos_class_label,
            neg_class_label=reference.neg_class_label,
            combine_consecutive_messages=COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=CONVERT_TOOL_TO_ASSISTANT,
        )

        t0 = time.time()
        status = prefetch_eval_activations(
            cache_dir,
            {split: dataset},
            source,
            model_name=reference.model_name,
            layer=int(reference.layer),
            cache_stem=CACHE_STEM,
        )
        holder = {split: dataset}
        _assign_cached_activations(holder, cache_dir / CACHE_STEM)
        dataset = holder[split]
        if "activations" not in dataset.other_fields:
            raise RuntimeError(
                f"{name}/{split}: activations were not attached from the cache — refusing to "
                f"fall back to computing them (that would load gemma-3-27b)"
            )
        print(
            f"[{name}/{split}] {len(dataset)} rows, {status[split]}, "
            f"ready in {time.time() - t0:.0f}s"
        )

        if verify_tokens:
            print(f"[{name}/{split}] token check: {_verify_tokens(split, dataset, reference.model_name)}")

        y_true = np.array([label.to_int() for label in dataset.labels])
        with results_path.open("a") as fh:
            for (arm, it), probe in probes.items():
                if (name, split, arm, it) in done:
                    continue
                preds = probe.predict_proba(dataset)
                metrics = calculate_metrics(y_true, preds, fpr=FPR)
                row = {
                    "experiment": name,
                    "concept": spec["concept"],
                    "split": split,
                    "arm": arm,
                    "iteration": it,
                    "n_rows": len(dataset),
                    **metrics,
                }
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"    {arm:<14} iter{it}  auroc={metrics['auroc']:.4f}  "
                    f"acc={metrics['accuracy']:.4f}  tpr@{FPR}={metrics['tpr_at_fpr']:.4f}"
                )
        del dataset, holder


def write_csv(results_path: Path, csv_path: Path) -> None:
    import pandas as pd

    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows).sort_values(["experiment", "arm", "iteration", "split"])
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {len(df)} rows to {csv_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", action="append", choices=sorted(EXPERIMENTS), default=None,
                    help="experiment to score (repeatable; default: all)")
    ap.add_argument("--iterations", default="0,1,2,3",
                    help="comma-separated probe iterations (default 0,1,2,3)")
    ap.add_argument("--splits", default=None, help="comma-separated split stems (default: all)")
    ap.add_argument("--results-jsonl", default=str(REPO_ROOT / "results_probe_versions" / "eval_rows.jsonl"))
    ap.add_argument("--csv", default=str(REPO_ROOT / "results_probe_versions" / "probe_versions_eval.csv"))
    ap.add_argument("--verify-tokens", action="store_true",
                    help="re-tokenize one row per split and compare with the blob's input_ids")
    args = ap.parse_args(argv)

    iterations = [int(x) for x in args.iterations.split(",") if x.strip() != ""]
    splits = [s for s in args.splits.split(",")] if args.splits else None
    results_path = Path(args.results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    for name in args.concept or list(EXPERIMENTS):
        run_experiment(
            name,
            iterations=iterations,
            results_path=results_path,
            splits=splits,
            verify_tokens=args.verify_tokens,
        )

    write_csv(results_path, Path(args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
