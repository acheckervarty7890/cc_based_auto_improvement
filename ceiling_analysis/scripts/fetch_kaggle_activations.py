#!/usr/bin/env python
"""Download the published gemma-3-27b eval/dev activation blobs for both concepts.

The eval and dev activations for `highstakes` and `hu_ha` are published on Kaggle as
one dataset per split (owner `anku7890`), addressed exactly the way the run configs
address them: dataset `{slug}-gemma{eval,dev}pt`, file `{split}-gemma{eval,dev}.pt`,
where `{slug}` is the split stem hyphenated (Kaggle rejects underscores in a slug).

Blobs land under ACTS_ROOT/<concept>/{eval,dev}/ under their published file names.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ACTS_ROOT = REPO / "ceiling_acts"

CONCEPTS = {
    "highstakes": {
        "eval_dir": REPO / "eval_sets/highstakes",
        "dev_dir": REPO / "dev_samples/highstakes",
    },
    "hu_ha": {
        "eval_dir": REPO / "eval_sets/hu_ha",
        "dev_dir": REPO / "dev_samples/hu_ha",
    },
}


def slugify(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def targets(concept: str) -> list[tuple[str, str, str, Path]]:
    """(kind, dataset_ref, remote_file, local_path) for every split of one concept."""
    out = []
    for kind, suffix in (("eval", "gemmaeval"), ("dev", "gemmadev")):
        d = CONCEPTS[concept][f"{kind}_dir"]
        for jsonl in sorted(d.glob("*.jsonl")):
            stem = jsonl.stem
            ref = f"anku7890/{slugify(stem)}-{suffix}pt"
            remote = f"{stem}-{suffix}.pt"
            local = ACTS_ROOT / concept / kind / remote
            out.append((kind, ref, remote, local))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=list(CONCEPTS))
    ap.add_argument("--kinds", nargs="*", default=["dev", "eval"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    rc = 0
    for concept in args.concepts:
        for kind, ref, remote, local in targets(concept):
            if kind not in args.kinds:
                continue
            if local.is_file() and local.stat().st_size > 0:
                print(f"OK    {concept}/{kind}/{remote}  ({local.stat().st_size/1e9:.2f} GB, cached)", flush=True)
                continue
            print(f"GET   {ref} :: {remote} -> {local}", flush=True)
            if args.dry_run:
                continue
            staging = local.parent / "_staging"
            staging.mkdir(parents=True, exist_ok=True)
            try:
                api.dataset_download_file(ref, remote, path=str(staging), force=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL  {ref}: {exc}", file=sys.stderr, flush=True)
                rc = 1
                continue
            for z in [p for p in staging.glob("*") if p.suffix == ".zip"]:
                shutil.unpack_archive(str(z), str(staging))
                z.unlink()
            src = staging / remote
            if not src.is_file():
                cands = list(staging.glob("*.pt"))
                if len(cands) == 1:
                    src = cands[0]
                else:
                    print(f"FAIL  {ref}: no {remote} after unpack ({[p.name for p in staging.iterdir()]})",
                          file=sys.stderr, flush=True)
                    rc = 1
                    continue
            src.replace(local)
            shutil.rmtree(staging, ignore_errors=True)
            print(f"DONE  {local}  ({local.stat().st_size/1e9:.2f} GB)", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
