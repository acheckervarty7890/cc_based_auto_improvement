#!/usr/bin/env python
"""Download precomputed eval activations from Kaggle into a local activation cache.

STANDALONE: this file deliberately duplicates the parts of
``agentic_redteam.kaggle_activations`` it needs and imports nothing from this repo, so
it can be copied to a bare box on its own. Its only third-party requirements are
``kaggle`` (the download) and ``torch`` (reading a blob's header to validate it).

Each blob is written to ``<cache-dir>/<split>-acts_full.pt`` — the exact path
tuberlens' ``get_performances`` derives via ``Path(save_path).with_stem(f"{name}-{stem}")``
— so a later eval finds it in ``get_activations``' local-first branch and never loads
the 27B extraction model. That is the whole point: these are full-split gemma-3-27b
activations, tens of GB and hours of forward passes to recompute.

Three concepts are published, differing only in which eval splits they cover and how
their datasets are named on Kaggle:

    hs_ls          high-stakes / low-stakes   anku7890/{split}gemmaevalpt
    hu_harm        harmful_to_human           anku7890/{slug}-gemmaevalpt
    instructions   instruction-following      anku7890/{slug}-gemmaevalpt

The ``{slug}`` (hyphenated stem) form exists because Kaggle dataset slugs are lowercase
alphanumerics and hyphens, and every hu_harm / instructions split stem contains an
underscore. The high-stakes stems are single words, so they need no such rewrite. The
*file* inside each dataset is unrestricted and is ``{split}-gemmaeval.pt`` for all three.

Every blob — freshly downloaded *or* already sitting in the cache — is checked against
the probe's model name and layer and against the split's row count before it is allowed
to count as a hit. This matters because ``LLMModel.load_activations`` discards the
model/layer the blob was saved with and the caches otherwise load *by path without
checking their inputs*: fine for a content-keyed blob you computed yourself, not for one
fetched from a remote store. A split that cannot be fetched raises rather than falling
back to computing it.

Usage:

    python fetch_kaggle_eval_activations.py --concept hu_harm --dry-run
    python fetch_kaggle_eval_activations.py --concept hs_ls --cache-dir path/to/eval_activations

Needs ``KAGGLE_CONFIG_DIR`` set to the DIRECTORY holding kaggle.json (the API joins the
filename on itself, and ``os.makedirs`` a wrong path, so pointing it at the file fails
silently), or ``KAGGLE_API_TOKEN``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

# The published activations are all gemma-3-27b-it layer 32 — the probe the eval-side
# caches in this repo were built for. Override with --probe (read off a pickled probe,
# which needs tuberlens importable) or with --model-name / --layer.
DEFAULT_MODEL_NAME = "google/gemma-3-27b-it"
DEFAULT_LAYER = 32
DEFAULT_OWNER = "anku7890"
DEFAULT_FILE_NAME = "{split}-gemmaeval.pt"
DEFAULT_CACHE_STEM = "acts_full.pt"


@dataclass(frozen=True)
class Concept:
    """A published set of eval-split blobs.

    ``splits`` maps split stem -> row count. The counts are baked in rather than read
    from the eval JSONLs so this script needs no data from the repo; pass
    ``--eval-dataset-dir`` to count the local files instead, which is the stricter
    check if you have them (it catches a split JSONL that has since changed).
    """

    dataset_slug: str
    splits: dict[str, int]
    file_name: str = DEFAULT_FILE_NAME


CONCEPTS: dict[str, Concept] = {
    # high-stakes / low-stakes: stems are single words, already slug-safe under {split}
    "hs_ls": Concept(
        "{split}gemmaevalpt",
        {"anthropic": 1028, "mt": 278, "mts": 274, "toolace": 328},
    ),
    # harmful_to_human: every stem has an underscore, so the DATASET slug must use {slug}
    "hu_harm": Concept(
        "{slug}-gemmaevalpt",
        {
            "eval_ai_dilemmas": 136,
            "eval_ant_hh": 134,
            "eval_balanced_refusal": 400,
            "eval_daily_dilemmas": 196,
        },
    ),
    # instruction-following: same {slug} rewrite as hu_harm — every stem has an underscore.
    # Splits live in eval_sets/instructions/; blobs were computed by
    # scripts/extract_eval_activations.py under combine_consecutive_messages: true +
    # convert_tool_to_assistant: true (the eval: section of the
    # *_instructions_gemma27b_batch_target60.md configs). Row counts and layer/model are
    # validated on download, but the transforms are NOT — a run whose eval: differs from
    # that pair would silently score mismatched activations.
    "instructions": Concept(
        "{slug}-gemmaevalpt",
        {
            "anthropic_harmless_refusal": 200,
            "bbq_substitution": 200,
            "hc_context_drift": 194,
            "hc_contradiction": 200,
            "mm_substitution": 200,
            "oig_context_drift": 194,
            "oig_omission": 114,
        },
    ),
}


class KaggleActivationError(RuntimeError):
    """A Kaggle activation blob could not be fetched, or failed validation."""


def _slugify(split: str) -> str:
    """Return ``split`` in the character set Kaggle allows in a dataset slug.

    Kaggle slugs are lowercase alphanumerics and hyphens — an underscore is rejected at
    creation time, so a split named ``eval_ai_dilemmas`` has no dataset whose slug
    contains its stem verbatim. Runs of anything else collapse to a single hyphen, and
    leading/trailing hyphens are dropped.
    """
    out: list[str] = []
    for ch in split.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")


@dataclass(frozen=True)
class KaggleActivationSource:
    """Where a split's precomputed activations live on Kaggle.

    ``dataset_slug`` and ``file_name`` are templates formatted with two keys: ``split``
    (the stem exactly as it appears on disk) and ``slug`` (that stem through
    :func:`_slugify`). A literal string with no placeholder is fine when every split
    maps to the same object.
    """

    owner: str
    dataset_slug: str
    file_name: str

    def handle(self, split: str) -> str:
        return f"{self.owner}/{self.dataset_slug.format(split=split, slug=_slugify(split))}"

    def file_for(self, split: str) -> str:
        return self.file_name.format(split=split, slug=_slugify(split))


def _authenticate():
    """Return an authenticated ``KaggleApi``.

    ``KaggleApi.authenticate()`` ends in ``print_auth_help(); exit(1)`` when no credential
    source resolves, and its anonymous fallback is disabled whenever the library is
    imported rather than run as the ``kaggle`` CLI. ``SystemExit`` is a ``BaseException``,
    so an ordinary ``except Exception`` would let it escape and kill the process — convert
    it into our own error instead.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise KaggleActivationError(
            "The 'kaggle' package is not installed in this environment. "
            "Install it with: pip install kaggle"
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


def _blob_header(path: Path) -> dict:
    """Read a saved activation blob's metadata without paging its tensors into RAM.

    ``mmap=True`` maps the tensor storages instead of reading them, so this is ~instant
    even for the 11 GB anthropic blob; only shapes and the small scalar fields are touched.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise KaggleActivationError(
            "The 'torch' package is not installed in this environment, so downloaded "
            "blobs cannot be validated. Install it with: pip install torch"
        ) from e

    return torch.load(path, map_location="cpu", mmap=True)


def _validate_blob(path: Path, *, split: str, model_name: str, layer: int, n_rows: int) -> None:
    """Raise unless the blob at ``path`` matches the probe and split it is claimed for."""
    try:
        data = _blob_header(path)
    except KaggleActivationError:
        raise
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

    ``dataset_download_file`` names its output from the download URL rather than from the
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
    split_rows: dict[str, int],
    source: KaggleActivationSource,
    *,
    model_name: str,
    layer: int,
    cache_stem: str = DEFAULT_CACHE_STEM,
    verbose: bool = True,
) -> dict[str, str]:
    """Populate the eval activation cache from Kaggle, one file per split.

    Splits already present locally are validated and left alone; nothing is
    re-downloaded. Returns a ``{split: status}`` map where status is ``cached`` (already
    local) or ``downloaded``.
    """
    cache_dir = Path(activations_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(cache_stem).stem
    suffix = Path(cache_stem).suffix or ".pt"

    targets = {split: cache_dir / f"{split}-{stem}{suffix}" for split in split_rows}
    pending = [s for s, t in targets.items() if not t.exists()]

    statuses: dict[str, str] = {}
    api = None
    for split, n_rows in split_rows.items():
        target = targets[split]
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


def _count_jsonl_rows(split_path: Path) -> int:
    """Number of records in an eval split JSONL (blank lines ignored)."""
    with split_path.open() as fh:
        return sum(1 for line in fh if line.strip())


def _probe_metadata(probe_path: Path) -> tuple[str, int]:
    """``(model_name, layer)`` off a pickled probe. Needs tuberlens importable."""
    import pickle

    with probe_path.open("rb") as fh:
        probe = pickle.load(fh)
    return str(probe.model_name), int(probe.layer)


def _resolve_split_rows(concept: Concept, splits: list[str] | None, eval_dir: Path | None) -> dict[str, int]:
    """``{split: row count}``, from the local eval JSONLs when given one, else baked in."""
    wanted = splits or list(concept.splits)
    if eval_dir is None:
        unknown = [s for s in wanted if s not in concept.splits]
        if unknown:
            raise KaggleActivationError(
                f"no baked-in row count for split(s) {', '.join(unknown)} — pass "
                "--eval-dataset-dir so the count can be read from the split JSONL."
            )
        return {s: concept.splits[s] for s in wanted}

    missing = [s for s in wanted if not (eval_dir / f"{s}.jsonl").is_file()]
    if missing:
        raise KaggleActivationError(
            f"no split JSONL for {', '.join(missing)} in {eval_dir}"
        )
    return {s: _count_jsonl_rows(eval_dir / f"{s}.jsonl") for s in wanted}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--concept", required=True, choices=sorted(CONCEPTS),
                    help="Which probe concept's eval splits to fetch")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Activation cache to fill (default: ./eval_activations_<concept>). "
                         "Point this at the run's output.activations_cache_dir to prefill it.")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="Split stems to fetch (default: all of the concept's splits)")
    ap.add_argument("--eval-dataset-dir", type=Path, default=None,
                    help="Read row counts from <dir>/<split>.jsonl instead of the baked-in "
                         "counts — the stricter check when the split files are available")
    ap.add_argument("--probe", type=Path, default=None,
                    help="Pickled probe to read model_name/layer from, instead of the defaults")
    ap.add_argument("--model-name", default=None,
                    help=f"Extraction model the blobs must declare (default {DEFAULT_MODEL_NAME})")
    ap.add_argument("--layer", type=int, default=None,
                    help=f"Layer the blobs must declare (default {DEFAULT_LAYER})")
    ap.add_argument("--owner", default=DEFAULT_OWNER, help="Kaggle username owning the datasets")
    ap.add_argument("--dataset-slug", default=None,
                    help="Slug template override; {slug} = hyphenated stem, {split} = raw stem")
    ap.add_argument("--file-name", default=None, help="Template for the file inside each dataset")
    ap.add_argument("--cache-stem", default=DEFAULT_CACHE_STEM,
                    help="Local blob names are <split>-<cache-stem> (must match what eval derives)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan (handles, targets, what is already there); download nothing")
    args = ap.parse_args(argv)

    concept = CONCEPTS[args.concept]
    eval_dir = args.eval_dataset_dir.resolve() if args.eval_dataset_dir else None
    cache_dir = (args.cache_dir or Path(f"eval_activations_{args.concept}")).resolve()
    if eval_dir is not None and not eval_dir.is_dir():
        print(f"ERROR: --eval-dataset-dir {eval_dir} is not a directory", file=sys.stderr)
        return 2

    if args.probe is not None:
        model_name, layer = _probe_metadata(args.probe.resolve())
    else:
        model_name, layer = DEFAULT_MODEL_NAME, DEFAULT_LAYER
    # explicit flags win over both, so a probe can supply one field and a flag override the other
    if args.model_name:
        model_name = args.model_name
    if args.layer is not None:
        layer = args.layer

    source = KaggleActivationSource(
        args.owner,
        args.dataset_slug or concept.dataset_slug,
        args.file_name or concept.file_name,
    )
    stem, suffix = Path(args.cache_stem).stem, Path(args.cache_stem).suffix or ".pt"

    try:
        split_rows = _resolve_split_rows(concept, args.splits, eval_dir)
    except KaggleActivationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"concept   {args.concept}")
    print(f"rows from {eval_dir if eval_dir else 'baked-in split sizes'}")
    print(f"cache     {cache_dir}")
    print(f"probe     {model_name} layer {layer}\n")
    for split, n_rows in split_rows.items():
        target = cache_dir / f"{split}-{stem}{suffix}"
        print(f"  {split:<24} {n_rows:>5} rows  {'present' if target.exists() else 'MISSING'}")
        print(f"  {'':<24} kaggle.com/datasets/{source.handle(split)}  ({source.file_for(split)})")
        print(f"  {'':<24} -> {target}")
    print()

    if args.dry_run:
        print("--dry-run: nothing downloaded. Blobs already present were NOT validated; "
              "re-run without --dry-run to validate them and fetch the rest.")
        return 0

    try:
        statuses = prefetch_eval_activations(
            cache_dir,
            split_rows,
            source,
            model_name=model_name,
            layer=layer,
            cache_stem=args.cache_stem,
        )
    except KaggleActivationError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    print()
    for split, state in sorted(statuses.items()):
        print(f"  {split:<24} {state}")
    total_gb = sum((cache_dir / f"{s}-{stem}{suffix}").stat().st_size for s in statuses) / 1e9
    print(f"\n{len(statuses)} split(s) ready in {cache_dir} ({total_gb:.2f} GB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
