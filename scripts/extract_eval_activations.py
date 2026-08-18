#!/usr/bin/env python
"""Compute eval-split activations through the TRUNCATED extraction loader.

``evaluate_probe`` does not extract eval activations the way the rest of this repo
extracts activations. It hands the splits to tuberlens' ``get_performances``, which
calls ``LLMModel.load(probe.model_name)`` directly (``tuberlens/evaluation.py:79``)
with no ``model_kwargs`` of its own. That path therefore gets **none** of what
``model_loading.load_extraction_model`` gives the red-team and retrain paths:

* no truncation to layers ``0..probe.layer`` — for ``google/gemma-3-27b-it`` at layer 32
  that is 29 of 62 layers, ~24 GB of bf16 weights, downloaded and dispatched without
  ever running a forward, and (since ``device_map="auto"`` fills the GPU in layer order)
  exactly what pushes the *executed* tail onto disk. **This is the one that still can't
  be fixed from the outside**, and it is the expensive one;
* no ``AGENTIC_REDTEAM_MAX_MEMORY`` pin — setting that env var does not affect
  ``get_performances`` at all, because that path never reads it. tuberlens' own
  ``MAX_MEMORY`` / ``MODEL_MAX_MEMORY`` settings *do* reach it (they are resolved inside
  ``LLMModel.load``), as does ``OFFLOAD_BUFFERS``, which now defaults on — so a budget
  and buffer offloading can be pinned there without going through this script. The
  layers still can't.

It never mattered before because every gemma run so far carried a ``kaggle:`` section,
so its eval was a pure cache hit and no model was loaded. It matters the first time a
gemma-sized probe has to compute its own eval activations — which is what the
instruction-following experiment does, over 1302 rows of ``eval_sets/instructions``.

So: compute them here instead, through ``load_extraction_model``, and write each split
to the exact filename ``get_performances`` derives
(``<cache-dir>/<split>-acts_full.pt``). The subsequent ``--eval`` then finds every split
cached, ``_assign_cached_activations`` attaches the blobs before ``get_performances``
runs, and no model is loaded at all. Afterwards publish them with
``scripts/publish_kaggle_eval_activations.py`` so no box ever pays this again.

THE TRANSFORMS ARE THE FOOTGUN. ``combine_consecutive_messages`` /
``convert_tool_to_assistant`` change how each conversation is tokenized but NOT how many
rows a split has — and the blob validation on both this script and the download side
checks model name, layer and row count only. Extract under transforms that differ from
the eval run's and you get blobs that pass every check while encoding different token
sequences. That is why ``--config`` is the intended way to run this: it takes the
transforms from the same ``eval:`` section the run will use. The explicit
``--[no-]combine-consecutive-messages`` / ``--[no-]convert-tool-to-assistant`` flags exist
for runs driven without a config, and must then be set to match by hand.

Only FULL splits are supported (``eval_max_samples: 0``), matching the ``acts_full.pt``
cache stem that ``kaggle:`` and the publisher address. A subsampled cache is keyed to a
``(max_samples, seed)`` pair and is not publishable.

Typical use — precompute the instruction-following eval activations, then publish::

    AGENTIC_REDTEAM_MAX_MEMORY="0=21GiB,cpu=45GiB" \\
    .venv_claude/bin/python scripts/extract_eval_activations.py \\
        --config configs/gptoss120b_instructions_gemma27b_batch_target60.md \\
        --probe probes/instructions_gemma27b_gptoss/probe_iter0.pkl \\
        --eval-dataset-dir eval_sets/instructions

    .venv_claude/bin/python scripts/publish_kaggle_eval_activations.py \\
        --source-dir results_instructions_gemma27b_shared/eval_activations \\
        --eval-dataset-dir eval_sets/instructions \\
        --dataset-slug "{slug}-instr-gemmaevalpt" --dry-run

Splits already cached and valid are skipped, so an interrupted run resumes at the split
it died on. ``--dry-run`` reports what would be computed and loads no model.
"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_redteam.kaggle_activations import (  # noqa: E402
    KaggleActivationError,
    _validate_blob,
)

# The only cache stem `kaggle:` and the publisher address. acts_n<N>_seed<S>.pt is a
# subsample keyed to that pair — it is a valid local cache but must never be published,
# and this script has no reason to produce one.
CACHE_STEM = "acts_full.pt"


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute full-split eval activations via the truncated extraction loader.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--config",
        type=Path,
        help="Run config. Supplies the output dir and — importantly — the eval transforms, "
        "so the blobs are tokenized the way the run that consumes them expects.",
    )
    ap.add_argument(
        "--probe",
        type=Path,
        help="Probe pickle giving the extraction model and layer. Any probe with the right "
        "model+layer works: eval activations depend on those, not on the probe head. "
        "Defaults to the config's probe.path.",
    )
    ap.add_argument(
        "--eval-dataset-dir",
        type=Path,
        required=True,
        help="Dir of eval split JSONLs (each *.jsonl is a split, keyed by filename stem).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        help="Where to write <split>-acts_full.pt. Defaults to the config's "
        "output.activations_cache_dir — i.e. the dir the run's --eval will read.",
    )
    ap.add_argument("--splits", nargs="+", help="Splits to compute (default: all in --eval-dataset-dir).")
    ap.add_argument(
        "--combine-consecutive-messages",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Eval transform. Default: from --config. MUST match the consuming run.",
    )
    ap.add_argument(
        "--convert-tool-to-assistant",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Eval transform. Default: from --config. MUST match the consuming run.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute splits that are already cached and valid.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be computed; load no model.",
    )
    return ap.parse_args(argv)


def _resolve(args: argparse.Namespace) -> dict:
    cfg = None
    if args.config is not None:
        from agentic_redteam.config import load_config

        cfg = load_config(args.config)

    probe_path = args.probe or (cfg.probe.path if cfg else None)
    out_dir = args.out_dir or (cfg.output.activations_cache_dir if cfg else None)
    combine = args.combine_consecutive_messages
    convert = args.convert_tool_to_assistant
    if combine is None and cfg is not None:
        combine = cfg.eval.combine_consecutive_messages
    if convert is None and cfg is not None:
        convert = cfg.eval.convert_tool_to_assistant

    missing = [
        name
        for name, value in (
            ("--probe (or config probe.path)", probe_path),
            ("--out-dir (or config output.activations_cache_dir)", out_dir),
        )
        if not value
    ]
    if missing:
        raise SystemExit("ERROR: missing required setting(s): " + ", ".join(missing))

    if combine is None or convert is None:
        raise SystemExit(
            "ERROR: the eval transforms are unset. They change how each conversation is "
            "tokenized but not how many rows a split has, so blobs computed under the "
            "wrong setting pass every validation check while being wrong. Pass --config, "
            "or set --[no-]combine-consecutive-messages and --[no-]convert-tool-to-assistant "
            "to match the run that will consume these blobs."
        )

    # A subsampled eval would need acts_n<N>_seed<S>.pt, which is not what this writes.
    if cfg is not None and cfg.eval.eval_max_samples not in (0, None):
        raise SystemExit(
            f"ERROR: {args.config} sets eval.eval_max_samples={cfg.eval.eval_max_samples}. "
            "This script computes FULL-split blobs (acts_full.pt); a subsampled run reads a "
            "different cache name and cannot use them."
        )

    return {
        "probe_path": Path(probe_path),
        "out_dir": Path(out_dir),
        "combine": bool(combine),
        "convert": bool(convert),
    }


def _load_splits(eval_dir: Path, splits, probe, *, combine: bool, convert: bool) -> dict:
    """Load each split exactly as ``evaluate_probe`` does.

    No ``subsample_balanced_subset`` and no ``seed_everything``: these are full splits, so
    there is nothing to subsample and no RNG whose state could change the rows.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    if splits is None:
        splits = sorted(p.stem for p in eval_dir.glob("*.jsonl"))
        if not splits:
            raise SystemExit(f"ERROR: no eval split JSONLs (*.jsonl) in {eval_dir}")

    out = {}
    for name in splits:
        path = eval_dir / f"{name}.jsonl"
        if not path.is_file():
            raise SystemExit(f"ERROR: split {name!r} has no JSONL at {path}")
        out[name] = LabelledDataset.load_from(
            path,
            pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label,
            combine_consecutive_messages=combine,
            convert_tool_to_assistant=convert,
        )
    return out


def _is_usable(path: Path, *, split: str, model_name: str, layer: int, n_rows: int) -> bool:
    """Is there already a valid blob at ``path``?

    A blob that exists but fails validation is reported and recomputed rather than
    trusted — the common cause is a run killed mid-save, which leaves a truncated .pt.
    """
    if not path.exists():
        return False
    try:
        _validate_blob(path, split=split, model_name=model_name, layer=layer, n_rows=n_rows)
        return True
    except KaggleActivationError as exc:
        print(f"  {split}: existing blob is unusable, recomputing — {exc}")
        return False


def main(argv=None) -> int:
    args = _parse_args(argv)
    settings = _resolve(args)

    probe_path = settings["probe_path"]
    if not probe_path.exists():
        raise SystemExit(f"ERROR: probe not found: {probe_path}")
    with probe_path.open("rb") as f:
        probe = pickle.load(f)
    model_name, layer = probe.model_name, int(probe.layer)

    out_dir = settings["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = _load_splits(
        args.eval_dataset_dir,
        args.splits,
        probe,
        combine=settings["combine"],
        convert=settings["convert"],
    )

    print(f"probe      : {probe_path}")
    print(f"           : {model_name} L{layer}")
    print(f"splits     : {args.eval_dataset_dir} ({len(datasets)}: {', '.join(datasets)})")
    print(f"out dir    : {out_dir}")
    print(f"transforms : combine={settings['combine']} convert_tool={settings['convert']}")
    print()

    todo, skipped = [], []
    for split, dataset in datasets.items():
        target = out_dir / f"{split}-{CACHE_STEM}"
        if not args.force and _is_usable(
            target, split=split, model_name=model_name, layer=layer, n_rows=len(dataset)
        ):
            skipped.append(split)
            continue
        todo.append((split, dataset, target))

    for split in skipped:
        print(f"  SKIP {split}: already cached")
    for split, dataset, target in todo:
        print(f"  TODO {split}: {len(dataset)} rows -> {target.name}")
    total_rows = sum(len(d) for _, d, _ in todo)
    print(f"\n{len(todo)} split(s) to compute, {total_rows} rows total.")

    if not todo:
        print("Nothing to do — every split is already cached.")
        return 0
    if args.dry_run:
        print("--dry-run: no model loaded, nothing computed.")
        return 0

    # Imported here so --dry-run and the fully-cached case never pull in torch/tuberlens.
    from agentic_redteam.model_loading import load_extraction_model

    model = load_extraction_model(model_name, layer, verbose=True)
    try:
        done_rows = 0
        run_start = time.time()
        for split, dataset, target in todo:
            # Write to a sibling temp path and rename on success, so an interrupted run
            # never leaves a half-written blob where the next --eval would find it. The
            # rename is atomic within the dir and costs nothing (same filesystem).
            tmp = target.with_name(f".{target.name}.partial")
            tmp.unlink(missing_ok=True)
            print(f"\n>>> {split}: {len(dataset)} rows")
            t0 = time.time()
            # Same call get_performances would have made (tuberlens/evaluation.py:80-90):
            # same layer, same default max_length=1024, one call per split — so the blob
            # is what the eval would have produced, just off a truncated model.
            model.get_activations(
                dataset.inputs,
                layer=layer,
                show_progress=True,
                save_path=str(tmp),
            )
            elapsed = time.time() - t0
            _validate_blob(
                tmp, split=split, model_name=model_name, layer=layer, n_rows=len(dataset)
            )
            tmp.replace(target)
            done_rows += len(dataset)
            gb = target.stat().st_size / 1e9
            rate = elapsed / max(len(dataset), 1)
            remaining = total_rows - done_rows
            eta = remaining * ((time.time() - run_start) / max(done_rows, 1))
            print(
                f"    {split}: {gb:.2f} GB in {elapsed / 60:.1f} min "
                f"({rate:.1f} s/sample); {remaining} rows left, ETA {eta / 60:.0f} min"
            )
    finally:
        # Mirrors ProbeJudge.release: torch's caching allocator holds freed memory as
        # reserved, and every tuberlens load re-infers its layer split from what is FREE
        # at load time — so leaving this resident would push the next phase's load onto
        # CPU/disk offload.
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — teardown must not mask a real failure above
            pass

    print(f"\nComputed {len(todo)} split(s) into {out_dir}.")
    print("Publish them so no box computes them again:")
    print(
        "  .venv_claude/bin/python scripts/publish_kaggle_eval_activations.py \\\n"
        f"      --source-dir {out_dir} \\\n"
        f"      --eval-dataset-dir {args.eval_dataset_dir} \\\n"
        '      --dataset-slug "{slug}-gemmaevalpt" --dry-run'
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KaggleActivationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
