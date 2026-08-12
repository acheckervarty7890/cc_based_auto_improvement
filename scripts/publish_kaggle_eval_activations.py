#!/usr/bin/env python
"""Publish locally-computed eval activation blobs to Kaggle, one dataset per split.

This is the *upload* half of ``agentic_redteam.kaggle_activations``. That module
downloads precomputed activations so a cloud run's eval is a pure cache hit and never
loads the 27B model; this script is how those datasets come to exist in the first
place, from blobs an earlier local run already computed.

Why it is needed for the human-harm concept specifically: the high-stakes splits
(anthropic/mt/mts/toolace) were published long ago as ``anku7890/{split}gemmaevalpt``,
but no equivalents exist for the ``eval_dataset_hu_ha`` splits — and they could not be
named that way regardless, because Kaggle dataset slugs are lowercase alphanumerics and
hyphens while every hu_ha split stem contains underscores. Hence the ``{slug}``
placeholder (see ``KaggleActivationSource``) and the hyphenated default template here.

Each blob is validated against the probe it claims to belong to (model name, layer) and
against the split's own row count BEFORE upload — the same check the download side runs,
applied early so a mismatched blob is never published rather than being caught hours
later on a remote box.

The staged copy is a hard link where the filesystem allows it, so publishing ~4.6 GB
does not first write ~4.6 GB. Note the file is *renamed* in staging: local caches are
``<split>-acts_full.pt`` while the published file name follows ``--file-name``
(``<split>-gemmaeval.pt``), matching what the configs' ``kaggle.eval_file_name`` asks for.

Typical use (from the repo root, after a local run has filled the cache):

    .venv_claude/bin/python scripts/publish_kaggle_eval_activations.py \
        --source-dir archive/results3/gemma27_activations --dry-run

Drop ``--dry-run`` to actually upload. Datasets are created PRIVATE unless ``--public``
is passed; a split whose dataset already exists is skipped unless ``--new-version`` is
given. Needs KAGGLE_CONFIG_DIR (the DIRECTORY holding kaggle.json) or KAGGLE_API_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_redteam.kaggle_activations import (  # noqa: E402
    KaggleActivationError,
    KaggleActivationSource,
    _authenticate,
    _validate_blob,
)


def _split_rows(split_path: Path) -> int:
    """Number of records in an eval split JSONL (blank lines ignored)."""
    with split_path.open() as fh:
        return sum(1 for line in fh if line.strip())


def _stage(blob: Path, staging: Path, file_name: str) -> Path:
    """Place ``blob`` into ``staging`` under ``file_name``, cheaply if possible.

    A hard link costs nothing and is what normally happens; it fails across
    filesystems (and on some network mounts), in which case a symlink is tried —
    Kaggle's uploader opens the path, so a link it can follow is enough — and a real
    copy is the last resort.
    """
    dest = staging / file_name
    try:
        os.link(blob, dest)
        return dest
    except OSError:
        pass
    try:
        dest.symlink_to(blob.resolve())
        return dest
    except OSError:
        shutil.copy2(blob, dest)
        return dest


def _title_for(split: str) -> str:
    """Kaggle requires a title of at least 6 characters; splits are all longer."""
    return f"{split} gemma-3-27b L32 eval activations"[:50]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", required=True, type=Path,
                    help="Directory holding <split>-<cache-stem> blobs (e.g. archive/results3/gemma27_activations)")
    ap.add_argument("--eval-dataset-dir", type=Path, default=REPO_ROOT / "eval_dataset_hu_ha",
                    help="Eval split JSONLs, used to discover splits and check row counts")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="Split stems to publish (default: every *.jsonl in --eval-dataset-dir)")
    ap.add_argument("--owner", default="anku7890", help="Kaggle username that will own the datasets")
    ap.add_argument("--dataset-slug", default="{slug}-gemmaevalpt",
                    help="Slug template; {slug} is the hyphenated split stem, {split} the raw one. "
                         "Must match kaggle.eval_dataset_slug in the configs.")
    ap.add_argument("--file-name", default="{split}-gemmaeval.pt",
                    help="Name of the file inside each dataset. Must match kaggle.eval_file_name.")
    ap.add_argument("--cache-stem", default="acts_full.pt",
                    help="Suffix of the LOCAL blob names (<split>-<cache-stem>)")
    ap.add_argument("--model-name", default="google/gemma-3-27b-it", help="Probe's extraction model")
    ap.add_argument("--layer", type=int, default=32, help="Probe's layer")
    ap.add_argument("--public", action="store_true", help="Publish public (default: private)")
    ap.add_argument("--new-version", action="store_true",
                    help="If a dataset already exists, push a new version instead of skipping it")
    ap.add_argument("--dry-run", action="store_true", help="Validate and print the plan; upload nothing")
    args = ap.parse_args(argv)

    source_dir = args.source_dir.resolve()
    eval_dir = args.eval_dataset_dir.resolve()
    if not source_dir.is_dir():
        print(f"ERROR: --source-dir {source_dir} is not a directory", file=sys.stderr)
        return 2
    if not eval_dir.is_dir():
        print(f"ERROR: --eval-dataset-dir {eval_dir} is not a directory", file=sys.stderr)
        return 2

    splits = args.splits or sorted(p.stem for p in eval_dir.glob("*.jsonl"))
    if not splits:
        print(f"ERROR: no *.jsonl splits found in {eval_dir}", file=sys.stderr)
        return 2

    source = KaggleActivationSource(args.owner, args.dataset_slug, args.file_name)

    # --- validate everything BEFORE uploading anything -------------------------------
    plan = []
    problems = []
    for split in splits:
        blob = source_dir / f"{split}-{args.cache_stem}"
        split_path = eval_dir / f"{split}.jsonl"
        if not blob.is_file():
            problems.append(f"{split}: no blob at {blob}")
            continue
        if not split_path.is_file():
            problems.append(f"{split}: no split JSONL at {split_path}")
            continue
        n_rows = _split_rows(split_path)
        try:
            _validate_blob(blob, split=split, model_name=args.model_name,
                           layer=args.layer, n_rows=n_rows)
        except KaggleActivationError as e:
            problems.append(str(e))
            continue
        plan.append((split, blob, n_rows, source.handle(split), source.file_for(split)))

    for p in problems:
        print(f"INVALID  {p}", file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} split(s) failed validation — nothing was uploaded.", file=sys.stderr)
        return 1

    total_gb = sum(b.stat().st_size for _, b, _, _, _ in plan) / 1e9
    print(f"{len(plan)} split(s), {total_gb:.2f} GB, "
          f"{'PUBLIC' if args.public else 'private'} datasets owned by {args.owner}:\n")
    for split, blob, n_rows, handle, fname in plan:
        print(f"  {split:<24} {n_rows:>5} rows  {blob.stat().st_size/1e9:>6.2f} GB")
        print(f"  {'':<24} {blob}")
        print(f"  {'':<24} -> kaggle.com/datasets/{handle}  ({fname})")
    print()
    if args.dry_run:
        print("--dry-run: nothing uploaded. Re-run without it to publish.")
        return 0

    api = _authenticate()
    failures = 0
    for split, blob, _n_rows, handle, fname in plan:
        slug = handle.split("/", 1)[1]
        with tempfile.TemporaryDirectory(dir=str(source_dir), prefix=f".publish_{split}_") as tmp:
            staging = Path(tmp)
            _stage(blob, staging, fname)
            (staging / "dataset-metadata.json").write_text(json.dumps({
                "title": _title_for(split),
                "id": f"{args.owner}/{slug}",
                "licenses": [{"name": "CC0-1.0"}],
            }, indent=2))
            print(f">>> uploading {split} -> {handle} ...", flush=True)
            try:
                if args.new_version:
                    api.dataset_create_version(
                        str(staging), version_notes=f"{split} activations "
                        f"({args.model_name} L{args.layer})", dir_mode="skip",
                        convert_to_csv=False,
                    )
                else:
                    api.dataset_create_new(
                        str(staging), public=args.public, dir_mode="skip",
                        convert_to_csv=False,
                    )
            except Exception as e:  # noqa: BLE001 — report and continue to the next split
                failures += 1
                print(f"    FAILED {split}: {e}", file=sys.stderr)
                continue
            print(f"    done: kaggle.com/datasets/{handle}")

    if failures:
        print(f"\n{failures} of {len(plan)} upload(s) failed.", file=sys.stderr)
        return 1
    print(f"\nAll {len(plan)} dataset(s) published. "
          f"The configs' kaggle.eval_dataset_slug must read {args.dataset_slug!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
