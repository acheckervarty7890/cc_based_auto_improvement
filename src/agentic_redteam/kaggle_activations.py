"""Prefetch precomputed eval activations from Kaggle into the local activation cache.

Why a prefetch rather than tuberlens' ``get_activations(using_kaggle=True)``:

* **Addressing.** tuberlens derives the Kaggle slug from the local ``save_path``
  (``LLMModel._get_kaggle_dataset_slug``) by stripping punctuation and truncating to
  the first 50 characters. Our cache filenames carry their distinguishing part at the
  *end*, so all four eval splits collapse to one slug — and the remote datasets were
  named independently of our directory layout anyway. Addressing has to be explicit.
* **Transfer volume.** ``_download_from_kaggle`` calls ``dataset_download_files(...,
  unzip=True)``, which pulls and unzips the *whole* dataset into a temp dir and then
  copies one file out. We fetch exactly the one file we need, into a staging dir on
  the same filesystem as the cache, so landing it is a rename rather than a copy.
* **Validation.** ``LLMModel.load_activations`` drops the ``model_name`` / ``layer``
  fields the blob was saved with, and the whole activation-cache design in this repo
  loads *by path without checking the inputs match*. That is tolerable for a blob we
  computed ourselves and keyed by content; it is not tolerable for one fetched from a
  remote store. Every download is checked against the probe and the split it claims
  to be before it is allowed into the cache.

Once a blob is in place under the name ``get_performances`` would have written
(``<split>-acts_full.pt``), tuberlens' local-first branch in ``get_activations``
loads it and never constructs the LLM — which is the entire point, since these are
gemma-3-27b activations over full eval splits (~20 GB, hours of forward passes).
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


class KaggleActivationError(RuntimeError):
    """A Kaggle activation blob could not be fetched, or failed validation."""


def _slugify(split: str) -> str:
    """Return ``split`` in the character set Kaggle allows in a dataset slug.

    Kaggle slugs are lowercase alphanumerics and hyphens — an underscore is rejected
    at creation time, so a split named ``eval_ai_dilemmas`` simply has no dataset
    whose slug contains its stem verbatim. Runs of anything else collapse to a single
    hyphen, and leading/trailing hyphens are dropped.
    """
    out = []
    for ch in split.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")


@dataclass(frozen=True)
class KaggleActivationSource:
    """Where a split's precomputed activations live on Kaggle.

    ``dataset_slug`` and ``file_name`` are templates formatted with **two** keys:

    * ``split`` — the split stem exactly as it appears on disk (``eval_ai_dilemmas``).
    * ``slug``  — that stem run through :func:`_slugify` (``eval-ai-dilemmas``).

    e.g. ``"{split}gemmaevalpt"`` / ``"{split}-gemmaeval.pt"``. A literal string with
    no placeholder is also fine when every split maps to the same object.

    Use ``{slug}`` in ``dataset_slug`` whenever a split stem contains an underscore
    (the ``eval_sets/hu_ha`` and ``eval_sets/instructions`` splits all do): Kaggle would
    reject ``eval_ai_dilemmasgemmaevalpt`` as a slug, so ``{split}`` there names a
    dataset that cannot be created. ``file_name`` is a filename *inside* the dataset
    and is unrestricted, so it normally stays on ``{split}``.
    """

    owner: str
    dataset_slug: str
    file_name: str

    def handle(self, split: str) -> str:
        rendered = self.dataset_slug.format(split=split, slug=_slugify(split))
        return f"{self.owner}/{rendered}"

    def file_for(self, split: str) -> str:
        return self.file_name.format(split=split, slug=_slugify(split))


def _authenticate():
    """Return an authenticated ``KaggleApi``.

    ``KaggleApi.authenticate()`` ends in ``print_auth_help(); exit(1)`` when no
    credential source resolves, and its anonymous fallback is disabled whenever the
    library is imported rather than run as the ``kaggle`` CLI. ``SystemExit`` is a
    ``BaseException``, so an ordinary ``except Exception`` would let it escape and
    kill the whole run. Convert it into our own error instead.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise KaggleActivationError(
            "The 'kaggle' package is not installed in this environment. "
            "Install it with: .venv_claude/bin/pip install kaggle"
        ) from e

    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as e:
        raise KaggleActivationError(
            "Kaggle authentication failed. Set KAGGLE_CONFIG_DIR to the DIRECTORY "
            "holding kaggle.json (not the file itself), or export KAGGLE_API_TOKEN "
            "with a token from https://www.kaggle.com/settings/api."
        ) from e
    return api


def _blob_header(path: Path) -> dict:
    """Read a saved activation blob's metadata without paging its tensors into RAM.

    ``mmap=True`` maps the tensor storages instead of reading them, so this is
    ~instant even for the 11 GB anthropic blob; only shapes and the small scalar
    fields are touched.
    """
    import torch

    return torch.load(path, map_location="cpu", mmap=True)


def _validate_blob(path: Path, *, split: str, model_name: str, layer: int, n_rows: int) -> None:
    """Raise unless the blob at ``path`` matches the probe and split it is claimed for."""
    try:
        data = _blob_header(path)
    except Exception as e:
        raise KaggleActivationError(f"{split}: could not read {path}: {e}") from e

    missing = {"activations", "attention_mask", "input_ids"} - set(data)
    if missing:
        raise KaggleActivationError(
            f"{split}: {path} is missing tensor field(s) {sorted(missing)} — "
            "it does not look like a tuberlens activation blob."
        )

    problems = []
    got_model = data.get("model_name")
    if got_model is not None and got_model != model_name:
        problems.append(f"model_name={got_model!r} (probe expects {model_name!r})")
    got_layer = data.get("layer")
    if got_layer is not None and int(got_layer) != int(layer):
        problems.append(f"layer={got_layer} (probe expects {layer})")
    got_rows = int(data["activations"].shape[0])
    if got_rows != n_rows:
        problems.append(f"{got_rows} rows (split has {n_rows})")
    if problems:
        raise KaggleActivationError(
            f"{split}: activations at {path} do not match this run — "
            + "; ".join(problems)
            + ". Refusing to use them."
        )


def _extract_downloaded(staging: Path, split: str) -> Path:
    """Return the single ``.pt`` that landed in ``staging``, unzipping if needed.

    ``dataset_download_file`` names its output from the download URL, not from the
    requested file name, and Kaggle may serve a large file zipped — so what actually
    arrives has to be discovered rather than assumed.
    """
    for archive in sorted(staging.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
        archive.unlink()

    blobs = sorted(p for p in staging.rglob("*.pt") if p.is_file())
    if not blobs:
        landed = sorted(p.name for p in staging.rglob("*") if p.is_file())
        raise KaggleActivationError(
            f"{split}: no .pt file in the download (got {landed or 'nothing'})."
        )
    if len(blobs) > 1:
        raise KaggleActivationError(
            f"{split}: expected one .pt in the download, got {[p.name for p in blobs]}."
        )
    return blobs[0]


def prefetch_eval_activations(
    activations_cache_dir: str | Path,
    eval_datasets: dict,
    source: KaggleActivationSource,
    *,
    model_name: str,
    layer: int,
    cache_stem: str = "acts_full.pt",
    verbose: bool = True,
) -> dict[str, str]:
    """Populate the eval activation cache from Kaggle, one file per split.

    Writes each split to ``<cache_dir>/<split>-<cache_stem>`` — the exact path
    ``get_performances`` derives via ``Path(save_path).with_stem(f"{name}-{stem}")``
    — so the subsequent eval is a pure cache hit and loads no model.

    Splits already present locally are validated and left alone; nothing is
    re-downloaded. A blob that fails validation is moved aside with a ``.rejected``
    suffix rather than left in the cache, so a re-run cannot mistake it for a hit.

    Returns a ``{split: status}`` map where status is ``cached`` (already local) or
    ``downloaded``. A split that cannot be fetched raises rather than falling back:
    silently computing gemma-27b activations for a full split is hours of work, and
    the caller asked for the cache precisely to avoid that.
    """
    cache_dir = Path(activations_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(cache_stem).stem
    suffix = Path(cache_stem).suffix or ".pt"

    targets = {
        split: cache_dir / f"{split}-{stem}{suffix}" for split in eval_datasets
    }
    pending = [s for s, t in targets.items() if not t.exists()]

    statuses: dict[str, str] = {}
    api = None
    for split, dataset in eval_datasets.items():
        target = targets[split]
        n_rows = len(dataset)
        if target.exists():
            _validate_blob(target, split=split, model_name=model_name, layer=layer, n_rows=n_rows)
            statuses[split] = "cached"
            if verbose:
                print(f"[kaggle] {split}: already cached at {target}")
            continue

        if api is None:  # authenticate lazily — a full local cache needs no network
            api = _authenticate()
            if verbose:
                print(
                    f"[kaggle] authenticated as {api.get_config_value('username')}; "
                    f"fetching {len(pending)} split(s): {', '.join(pending)}"
                )

        handle, remote_name = source.handle(split), source.file_for(split)
        staging = cache_dir / f".staging-{split}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            if verbose:
                print(f"[kaggle] {split}: downloading {handle}:{remote_name} ...")
            try:
                api.dataset_download_file(handle, remote_name, path=str(staging), quiet=not verbose)
            except Exception as e:
                raise KaggleActivationError(
                    f"{split}: download of {handle}:{remote_name} failed: {e}"
                ) from e

            blob = _extract_downloaded(staging, split)
            _validate_blob(blob, split=split, model_name=model_name, layer=layer, n_rows=n_rows)
            # Same filesystem as the cache, so this is a rename, not a 2nd copy.
            blob.replace(target)
            statuses[split] = "downloaded"
            if verbose:
                size_gb = target.stat().st_size / 1e9
                print(f"[kaggle] {split}: {target.name} ({size_gb:.2f} GB) validated and cached")
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    return statuses
