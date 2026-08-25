#!/usr/bin/env python
"""Download precomputed eval AND dev activations for one concept from Kaggle.

The download half of ``agentic_redteam.kaggle_activations``, exposed standalone so a
box can be primed *before* a run rather than during it. ``iterative_retrain_main``
already prefetches both (see ``cli.py``), but only as part of starting a run; this
script does the same two calls with nothing else attached, which is what you want when
provisioning a machine or checking that every blob a config names actually exists.

Defaults are the ``instructions`` concept against the gemma-3-27b-it L32 probe — the
slug/file templates and cache dirs are copied from
``configs/gptoss120b_instructions_gemma27b_*.md``. Point ``--eval-dataset-dir`` /
``--dev-data`` elsewhere for another concept.

The transforms matter: the eval cache is keyed by PATH, and the dev blob's name is a
content hash over the dev files plus ``model | layer | combine | convert``. Passing
transforms that differ from the run's would either write a blob the fit never looks for
(dev) or leave one the eval silently reuses against a different representation (eval).
The defaults here match those configs' ``eval:`` section, which is how the published
blobs were computed.

Needs Kaggle credentials: KAGGLE_CONFIG_DIR naming the DIRECTORY holding kaggle.json,
or KAGGLE_API_TOKEN. A split that cannot be fetched or fails validation RAISES — the
point of the cache is to avoid hours of local 27B extraction, so falling back silently
would defeat it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_redteam.kaggle_activations import (  # noqa: E402
    KaggleActivationSource,
    prefetch_dev_activations,
    prefetch_eval_activations,
)
from agentic_redteam.retrain import _dev_activation_cache_path  # noqa: E402


def _jsonl_rows(path: Path) -> int:
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--eval-dataset-dir", type=Path,
                    default=REPO_ROOT / "eval_sets/instructions")
    ap.add_argument("--dev-data", type=Path,
                    default=REPO_ROOT / "dev_samples/instructions")
    ap.add_argument("--activations-cache-dir", type=Path,
                    default=REPO_ROOT / "results_instructions_gemma27b_shared/eval_activations")
    ap.add_argument("--base-activation-cache-dir", type=Path,
                    default=REPO_ROOT / "results_instructions_gemma27b_shared/base_activations",
                    help="Where the assembled dev blob lands (same dir the fit reads)")
    ap.add_argument("--owner", default="anku7890")
    ap.add_argument("--eval-dataset-slug", default="{slug}-gemmaevalpt")
    ap.add_argument("--eval-file-name", default="{split}-gemmaeval.pt")
    ap.add_argument("--dev-dataset-slug", default="{slug}-gemmadevpt")
    ap.add_argument("--dev-file-name", default="{split}-gemmadev.pt")
    ap.add_argument("--model-name", default="google/gemma-3-27b-it")
    ap.add_argument("--layer", type=int, default=32)
    # Must match the run's eval: section — see the module docstring.
    ap.add_argument("--combine-consecutive-messages", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--convert-tool-to-assistant", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--skip-eval", action="store_true", help="Fetch only the dev set")
    ap.add_argument("--skip-dev", action="store_true", help="Fetch only the eval splits")
    args = ap.parse_args(argv)

    if not args.skip_eval:
        eval_files = sorted(args.eval_dataset_dir.glob("*.jsonl"))
        if not eval_files:
            print(f"ERROR: no *.jsonl in {args.eval_dataset_dir}", file=sys.stderr)
            return 2
        # prefetch_eval_activations only ever takes len() of each value, so a list of
        # the right length stands in for the LabelledDataset and costs no tuberlens
        # import (and no transforms, which cannot change a row count anyway).
        eval_datasets = {f.stem: [None] * _jsonl_rows(f) for f in eval_files}
        print(f"=== EVAL: {len(eval_datasets)} split(s) from {args.eval_dataset_dir}")
        statuses = prefetch_eval_activations(
            args.activations_cache_dir,
            eval_datasets,
            KaggleActivationSource(args.owner, args.eval_dataset_slug, args.eval_file_name),
            model_name=args.model_name,
            layer=args.layer,
            verbose=True,
        )
        for split, status in sorted(statuses.items()):
            print(f"  {split}: {status} ({len(eval_datasets[split])} rows)")

    if not args.skip_dev:
        dev_path = args.dev_data
        dev_files = sorted(dev_path.glob("*.jsonl")) if dev_path.is_dir() else [dev_path]
        if not dev_files:
            print(f"ERROR: no *.jsonl in {dev_path}", file=sys.stderr)
            return 2
        target = _dev_activation_cache_path(
            args.base_activation_cache_dir,
            dev_files,
            args.model_name,
            args.layer,
            args.combine_consecutive_messages,
            args.convert_tool_to_assistant,
        )
        print(f"=== DEV: {len(dev_files)} split(s) from {dev_path}")
        print(f"    assembling into {target}")
        status = prefetch_dev_activations(
            target,
            dev_files,
            KaggleActivationSource(args.owner, args.dev_dataset_slug, args.dev_file_name),
            model_name=args.model_name,
            layer=args.layer,
            verbose=True,
        )
        print(f"  dev: {status} ({sum(_jsonl_rows(f) for f in dev_files)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
