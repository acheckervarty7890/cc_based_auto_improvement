#!/usr/bin/env python
"""Bulk-download every concept's published eval AND dev activation blob from Kaggle.

Layout written (both usable as a path-keyed cache, i.e. the exact name
``get_performances`` derives, ``<split>-acts_full.pt``)::

    activations/eval/<concept>/<split>-acts_full.pt
    activations/dev/<concept>/<split>-acts_full.pt

The dev blobs are kept PER SPLIT here rather than assembled into the single
content-hashed blob ``retrain._dev_activation_cache_path`` looks for: that name depends
on the run's transform flags, which differ per config, and assembly is cheap once the
bytes are local (see ``kaggle_activations.prefetch_dev_activations``).

Order is hu_ha -> instructions -> highstakes, because high-stakes is by far the largest
(the anthropic_hh_balanced eval blob alone is ~33 GB) and everything else should land
first. Each blob is validated against the probe's model/layer and the split's own row
count before it is accepted, exactly as the download half of the library does; a split
that cannot be fetched is reported and the run continues to the next one.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("KAGGLE_CONFIG_DIR", str(REPO_ROOT / "kaggle"))

from agentic_redteam.kaggle_activations import (  # noqa: E402
    KaggleActivationError,
    _authenticate,
    _extract_downloaded,
    _jsonl_rows,
    _slugify,
    _validate_blob,
)

MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
CACHE_STEM = "acts_full.pt"
OWNER = "anku7890"
OUT_ROOT = REPO_ROOT / "activations"

# High-stakes last: it is ~48 GB of eval blobs against ~7 GB for the other two concepts.
CONCEPT_ORDER = ["hu_ha", "instructions", "highstakes"]

# Primary naming is {slug}-gemma{eval,dev}pt / {split}-gemma{eval,dev}.pt. A couple of
# splits predate that convention, so a fallback handle is tried before giving up — the
# row-count check is what decides whether the older blob is actually the same split.
FALLBACK_HANDLES = {
    ("eval", "mts_balanced"): ["mtsgemmaevalpt", "mtsevalpt"],
    ("dev", "anthropic_hh_balanced"): ["anthropicgemmaevalpt", "anthropicevalpt"],
    ("dev", "mt_balanced"): ["mtgemmaevalpt"],
    ("dev", "mts_balanced"): ["mtsgemmaevalpt"],
    ("dev", "toolace_balanced"): ["toolacegemmaevalpt"],
}


def _candidates(kind: str, split: str) -> list[str]:
    slug = _slugify(split)
    primary = f"{slug}-gemma{kind}pt"
    return [primary] + FALLBACK_HANDLES.get((kind, split), [])


def _resolve(api, kind: str, split: str) -> tuple[str, str] | None:
    """Return (handle, remote file name) for the first candidate dataset that exists."""
    for slug in _candidates(kind, split):
        handle = f"{OWNER}/{slug}"
        try:
            files = api.dataset_list_files(handle).files
        except Exception:
            continue
        names = [f.name for f in files if str(f.name).endswith(".pt")]
        if len(names) == 1:
            return handle, names[0]
        preferred = f"{split}-gemma{kind}.pt"
        if preferred in names:
            return handle, preferred
        if names:
            return handle, sorted(names)[0]
    return None


def _fetch(api, kind: str, split: str, target: Path, n_rows: int) -> str:
    if target.exists():
        try:
            _validate_blob(target, split=split, model_name=MODEL_NAME, layer=LAYER, n_rows=n_rows)
            return "cached"
        except KaggleActivationError as e:
            print(f"    existing blob rejected, refetching: {e}", flush=True)
            target.replace(target.with_suffix(target.suffix + ".rejected"))

    resolved = _resolve(api, kind, split)
    if resolved is None:
        raise KaggleActivationError(
            f"{split}: no {kind} dataset found (tried {_candidates(kind, split)})"
        )
    handle, remote_name = resolved
    staging = target.parent / f".staging-{split}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        print(f"    downloading {handle}:{remote_name}", flush=True)
        t0 = time.time()
        api.dataset_download_file(handle, remote_name, path=str(staging), quiet=True)
        blob = _extract_downloaded(staging, split)
        gb = blob.stat().st_size / 1e9
        print(f"    {gb:.2f} GB in {time.time()-t0:.0f}s; validating", flush=True)
        _validate_blob(blob, split=split, model_name=MODEL_NAME, layer=LAYER, n_rows=n_rows)
        blob.replace(target)
        return "downloaded"
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    api = _authenticate()
    print(f"[kaggle] authenticated as {api.get_config_value('username')}", flush=True)

    jobs = []
    for concept in CONCEPT_ORDER:
        for kind, root in (("eval", "eval_sets"), ("dev", "dev_samples")):
            d = REPO_ROOT / root / concept
            if not d.is_dir():
                continue
            for split_path in sorted(d.glob("*.jsonl")):
                jobs.append((kind, concept, split_path))

    failures = []
    for kind, concept, split_path in jobs:
        split = split_path.stem
        n_rows = _jsonl_rows(split_path)
        out_dir = OUT_ROOT / kind / concept
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{split}-{CACHE_STEM}"
        print(f"[{kind}/{concept}] {split} ({n_rows} rows)", flush=True)
        try:
            status = _fetch(api, kind, split, target, n_rows)
            print(f"    {status}: {target}", flush=True)
        except Exception as e:  # noqa: BLE001 — report and keep going
            failures.append((kind, concept, split, str(e)))
            print(f"    FAILED: {e}", flush=True)

    print("\n==== summary ====", flush=True)
    for kind in ("eval", "dev"):
        for concept in CONCEPT_ORDER:
            d = OUT_ROOT / kind / concept
            if d.is_dir():
                blobs = sorted(d.glob(f"*-{CACHE_STEM}"))
                gb = sum(b.stat().st_size for b in blobs) / 1e9
                print(f"{kind}/{concept}: {len(blobs)} blob(s), {gb:.2f} GB", flush=True)
    if failures:
        print(f"\n{len(failures)} split(s) failed:", flush=True)
        for kind, concept, split, err in failures:
            print(f"  {kind}/{concept}/{split}: {err}", flush=True)
        return 1
    print("\nall splits present", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
