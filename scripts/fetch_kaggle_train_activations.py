#!/usr/bin/env python
"""Download precomputed TRAINING activations from Kaggle into a base activation cache.

This is the training-side counterpart of ``fetch_kaggle_eval_activations.py``. That one
fills the *eval* cache so ``get_performances`` never loads the 27B model; this one fills
the cache ``retrain_probe`` reads, so a retrain on a box with no GPU-hours to spare never
runs a forward pass either.

Two kinds of blob live in that one directory, and both are needed:

    <cache-dir>/base_acts_<model>_L<layer>_<hash>_{train,val}.pt
        The base training data's activations, one whole-split blob per side. Keyed on a
        hash of the base data file plus model|layer|seed|test_size|split_field|combine|
        convert (``retrain._base_activation_cache_paths``), so it is only a hit for a run
        using the same base file and the same split parameters.

    <cache-dir>/redteam_acts_<model>_L<layer>/<key>.pt
        The red-team set, cached PER CONVERSATION (``retrain._redteam_activation_cache_path``)
        rather than as one blob, because the set grows every iteration and a whole-set key
        would never hit twice. ``<key>`` is a hash of that conversation's own transformed
        messages plus model|layer|combine|convert.

Because the extraction LLM is frozen, a conversation's blob is valid no matter which
iteration computed it — which is why the published per-iteration datasets are each
*self-contained* (iter3 carries all 878 conversations, not a delta on iter2; the
iterations do not nest).

Published layout, one dataset per (concept, arm, iteration) plus one shared base:

    anku7890/hu-harm-gemma27b-base                     the two base_acts_*.pt
    anku7890/hu-harm-gemma27b-deepseekv4pro-iter{1,2,3}  redteam_acts_*/<key>.pt + manifest
    anku7890/hu-harm-gemma27b-gptoss120b-iter{1,2,3}     ditto for the other attacker arm

Each dataset carries a ``manifest.json`` naming the model, layer, transform flags and the
conversations it covers. This script checks the manifest against what you asked for
BEFORE writing anything into the cache, then validates every blob's header — the caches
otherwise load by path without checking their inputs, which is fine for blobs you
computed yourself and not for blobs fetched from a remote store.

Usage:

    python scripts/fetch_kaggle_train_activations.py --dry-run
    python scripts/fetch_kaggle_train_activations.py \
        --cache-dir results/base_activations --arm deepseekv4pro --iteration 3

Needs KAGGLE_CONFIG_DIR set to the DIRECTORY holding kaggle.json (the API joins the
filename on itself, and ``os.makedirs`` a wrong path, so pointing it at the file fails
silently), or KAGGLE_API_TOKEN.

Disk: the download arrives as a zip and is unzipped in a staging dir inside --cache-dir,
so transiently needs ~2x the dataset size free. Staging is removed on success.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

DEFAULT_OWNER = "anku7890"
DEFAULT_MODEL_NAME = "google/gemma-3-27b-it"
DEFAULT_LAYER = 32
BASE_SLUG = "hu-harm-gemma27b-base"
REDTEAM_SLUG = "hu-harm-gemma27b-{arm}-iter{iteration}"


class KaggleActivationError(RuntimeError):
    """A Kaggle activation dataset could not be fetched, or failed validation."""


# --------------------------------------------------------------------------- #
# kaggle plumbing
# --------------------------------------------------------------------------- #
def _authenticate():
    """Return an authenticated ``KaggleApi``.

    ``KaggleApi.authenticate()`` ends in ``exit(1)`` when no credential resolves.
    ``SystemExit`` is a ``BaseException``, so an ordinary ``except Exception`` would let
    it escape and kill the process — convert it into our own error instead.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise KaggleActivationError(
            "The 'kaggle' package is not installed. Install it with: pip install kaggle"
        ) from e
    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as e:
        raise KaggleActivationError(
            "Kaggle authentication failed. Set KAGGLE_CONFIG_DIR to the DIRECTORY holding "
            "kaggle.json (not the file itself), or export KAGGLE_API_TOKEN with a token "
            "from https://www.kaggle.com/settings/api."
        ) from e
    return api


def _download_dataset(api, handle: str, staging: Path) -> None:
    """Pull every file of ``handle`` into ``staging``, unzipped, subdirectories intact."""
    staging.mkdir(parents=True, exist_ok=True)
    try:
        api.dataset_download_files(handle, path=str(staging), unzip=True, quiet=False)
    except Exception as e:
        raise KaggleActivationError(f"{handle}: download failed: {e}") from e
    # Older SDKs ignore unzip=True for some payloads; finish the job by hand.
    for archive in sorted(staging.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
        archive.unlink()


def _read_manifest(staging: Path, handle: str) -> dict:
    path = staging / "manifest.json"
    if not path.is_file():
        raise KaggleActivationError(
            f"{handle}: no manifest.json in the download — refusing to trust it. "
            f"Got {sorted(p.name for p in staging.iterdir())[:8]}"
        )
    return json.loads(path.read_text())


def _check_manifest(man: dict, handle: str, *, model_name: str, layer: int,
                    combine: bool, convert: bool) -> None:
    """Raise unless the manifest describes the run we are about to feed."""
    problems = []
    if man.get("model_name") not in (None, model_name):
        problems.append(f"model_name={man['model_name']!r} (want {model_name!r})")
    if man.get("layer") is not None and int(man["layer"]) != int(layer):
        problems.append(f"layer={man['layer']} (want {layer})")
    for key, want in (("combine_consecutive_messages", combine),
                      ("convert_tool_to_assistant", convert)):
        got = man.get(key)
        if got is not None and bool(got) != bool(want):
            problems.append(f"{key}={got} (want {want})")
    if problems:
        raise KaggleActivationError(
            f"{handle}: manifest does not match this run — " + "; ".join(problems)
            + ". These activations would be silently wrong; refusing to install them."
        )


# --------------------------------------------------------------------------- #
# blob validation
# --------------------------------------------------------------------------- #
def _blob_header(path: Path) -> dict:
    """Read a blob's metadata without paging its tensors in (``mmap=True``)."""
    try:
        import torch
    except ImportError as e:
        raise KaggleActivationError(
            "The 'torch' package is not installed, so blobs cannot be validated."
        ) from e
    return torch.load(path, map_location="cpu", mmap=True)


def _validate_blob(path: Path, *, model_name: str, layer: int, n_rows: int | None) -> None:
    try:
        data = _blob_header(path)
    except KaggleActivationError:
        raise
    except Exception as e:
        raise KaggleActivationError(f"could not read {path}: {e}") from e

    missing = {"activations", "attention_mask", "input_ids"} - set(data)
    if missing:
        raise KaggleActivationError(
            f"{path.name} is missing tensor field(s) {sorted(missing)} — "
            "it does not look like a tuberlens activation blob."
        )
    problems = []
    if data.get("model_name") not in (None, model_name):
        problems.append(f"model_name={data['model_name']!r} (want {model_name!r})")
    if data.get("layer") is not None and int(data["layer"]) != int(layer):
        problems.append(f"layer={data['layer']} (want {layer})")
    got = int(data["activations"].shape[0])
    if n_rows is not None and got != n_rows:
        problems.append(f"{got} rows (want {n_rows})")
    if problems:
        raise KaggleActivationError(
            f"{path.name} does not match this run — " + "; ".join(problems)
        )


# --------------------------------------------------------------------------- #
# install
# --------------------------------------------------------------------------- #
def _install(staging: Path, cache_dir: Path, *, model_name: str, layer: int,
             sample_every: int, overwrite: bool) -> tuple[int, int]:
    """Move every ``.pt`` from ``staging`` into ``cache_dir``, keeping subdirectories.

    Returns ``(installed, skipped)``. Staging is on the same filesystem as the cache
    (it is created inside it), so this is a rename, not a second copy of several GB.
    Every ``sample_every``-th blob has its header validated; validating all 878 costs a
    torch.load each and buys little once the manifest has matched.
    """
    blobs = sorted(p for p in staging.rglob("*.pt") if p.is_file())
    if not blobs:
        raise KaggleActivationError(f"no .pt files in the download at {staging}")

    installed = skipped = 0
    for i, src in enumerate(blobs):
        rel = src.relative_to(staging)
        dest = cache_dir / rel
        if dest.exists() and not overwrite:
            skipped += 1
            continue
        if i % max(1, sample_every) == 0:
            _validate_blob(src, model_name=model_name, layer=layer, n_rows=None)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        installed += 1
    return installed, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="Base activation cache dir to fill (what retrain_probe is given "
                         "as --base-activation-cache-dir)")
    ap.add_argument("--owner", default=DEFAULT_OWNER)
    ap.add_argument("--arm", default="deepseekv4pro",
                    help="Attacker arm whose red-team activations to fetch "
                         "(deepseekv4pro | gptoss120b)")
    ap.add_argument("--iteration", type=int, default=3,
                    help="Which iteration's (self-contained) red-team set to fetch")
    ap.add_argument("--base-only", action="store_true", help="Fetch only the base blobs")
    ap.add_argument("--redteam-only", action="store_true", help="Fetch only the red-team blobs")
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--no-combine-consecutive-messages", dest="combine",
                    action="store_false", default=True)
    ap.add_argument("--no-convert-tool-to-assistant", dest="convert",
                    action="store_false", default=True)
    ap.add_argument("--sample-every", type=int, default=50,
                    help="Validate every Nth blob header (0/1 = all)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace blobs already in the cache (default: keep them)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and check credentials; download nothing")
    args = ap.parse_args(argv)

    if args.base_only and args.redteam_only:
        print("ERROR: --base-only and --redteam-only are mutually exclusive", file=sys.stderr)
        return 2

    cache_dir = args.cache_dir.resolve()
    wanted: list[str] = []
    if not args.redteam_only:
        wanted.append(f"{args.owner}/{BASE_SLUG}")
    if not args.base_only:
        wanted.append(f"{args.owner}/"
                      + REDTEAM_SLUG.format(arm=args.arm, iteration=args.iteration))

    print(f"cache dir : {cache_dir}")
    print(f"probe     : {args.model_name} layer {args.layer} "
          f"(combine={args.combine}, convert={args.convert})")
    for h in wanted:
        print(f"fetch     : kaggle.com/datasets/{h}")
    if args.dry_run:
        _authenticate()
        print("\n--dry-run: credentials OK, nothing downloaded.")
        return 0

    cache_dir.mkdir(parents=True, exist_ok=True)
    api = _authenticate()

    total_installed = total_skipped = 0
    for handle in wanted:
        print(f"\n>>> {handle}", flush=True)
        staging = Path(tempfile.mkdtemp(dir=str(cache_dir), prefix=".fetch_"))
        try:
            _download_dataset(api, handle, staging)
            man = _read_manifest(staging, handle)
            _check_manifest(man, handle, model_name=args.model_name, layer=args.layer,
                            combine=args.combine, convert=args.convert)
            n_expected = man.get("n_conversations")
            n_blobs = sum(1 for p in staging.rglob("*.pt") if p.is_file())
            if n_expected is not None and n_blobs != int(n_expected):
                raise KaggleActivationError(
                    f"{handle}: manifest claims {n_expected} conversations but the "
                    f"download holds {n_blobs} .pt files — incomplete or wrong dataset."
                )
            inst, skip = _install(staging, cache_dir, model_name=args.model_name,
                                  layer=args.layer, sample_every=args.sample_every,
                                  overwrite=args.overwrite)
            total_installed += inst
            total_skipped += skip
            print(f"    kind={man.get('kind')} n_blobs={n_blobs} "
                  f"installed={inst} already-present={skip}")
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    print(f"\nDone: {total_installed} blob(s) installed, {total_skipped} already present.")
    print(f"Pass --base-activation-cache-dir {cache_dir} to the retrain.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KaggleActivationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
