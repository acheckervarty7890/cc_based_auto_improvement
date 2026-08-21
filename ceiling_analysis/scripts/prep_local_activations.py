#!/usr/bin/env python
"""Build ``ceiling_acts/<acts_name>/`` out of the caches a run on THIS box already wrote.

The upstream ceiling analysis fetched its eval and dev blobs from Kaggle and extracted the
red-team and base conversations with a fresh gemma-3-27b load. Neither is necessary here:
experiment22 ran on this machine, so every activation the analysis needs is already on
disk, in the run's own caches. This script re-addresses them under the names
``ca_common`` expects. **No LLM is loaded and nothing is recomputed** — every byte written
is a copy or a slice of a byte the run already produced.

Four kinds of source, each handled differently because each is cached differently:

* **eval** — one blob per split, ``<split>-acts_full.pt``, exactly what
  ``get_performances`` derives. Symlinked to ``<split>-gemmaeval.pt``.
* **dev** — ONE blob for the whole dev dir (``_dev_activation_cache_path``), because the
  dev set is used whole and is never split. The analysis wants it per split, so it is
  sliced in ``sorted(*.jsonl)`` order — the order ``retrain._load_dev_dataset``
  concatenates in, which is what makes a row index mean the same row it would to a run.
* **red-team** — already per conversation and content-keyed exactly as ``ca_common``
  keys it, so the whole directory is linked file by file.
* **base** — cached as one whole-split blob, not per conversation, so its 50 rows are
  split out and written under the per-conversation key. Safe because
  ``stable_train_test_split`` appends indices in ascending order and this run's
  ``test_size`` is 0.0, so blob row *i* is base row *i*; the script asserts that against
  each row's tokenized ``input_ids`` rather than trusting it.

Every blob written here is right-padded (real tokens at positions ``0..L-1``), matching
the published blobs the upstream analysis assumed and the trims in ``ca_common`` require.
The sources are already right-padded on this box — extraction runs at ``BATCH_SIZE=1``,
and ``_concatenate_consuming`` pads on the right — which the script verifies before
slicing anything, since a left-padded source would silently corrupt every width trim
downstream.

Usage::

    .venv_claude/bin/python ceiling_analysis/scripts/prep_local_activations.py \
        --concept hu_ha_dd_gptoss120b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ca_common as C  # noqa: E402

REPO = C.REPO
sys.path.insert(0, str(REPO / "src"))

DEFAULT_RUN_CACHE = REPO / "results_hu_harm_gemma27b_batch_ablation" / "base_activations"
DEFAULT_EVAL_CACHE = REPO / "results_hu_harm_gemma27b_batch_ablation" / "eval_activations"


def _load(path: Path, *, mmap: bool = True) -> dict:
    return torch.load(path, map_location="cpu", mmap=mmap, weights_only=False)


def _assert_right_padded(blob: dict, what: str) -> None:
    """Refuse to slice a left-padded blob.

    ``tokenize_inputs`` pads with the tokenizer's own ``padding_side``, which is *left*
    for gemma-3-it, so any extraction call carrying more than one row comes back
    left-padded. Slicing ``[:w]`` off such a row cuts real tokens instead of padding, and
    nothing downstream would notice — the mask would still say the row is valid.
    """
    mask = blob["attention_mask"]
    if int((mask[:, 0] == 0).sum()) > 0:
        raise SystemExit(
            f"{what}: left-padded (some rows start with mask 0). Every trim in this "
            f"analysis assumes real tokens at 0..L-1; refusing to slice it."
        )


def _link(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return "ok"
        dst.unlink()
    dst.symlink_to(src.resolve())
    return "linked"


def prep_eval(concept: C.Concept, eval_cache: Path) -> None:
    print(f"[eval] {concept.eval_blob_dir}")
    for jsonl in sorted(concept.eval_dir.glob("*.jsonl")):
        src = eval_cache / f"{jsonl.stem}-acts_full.pt"
        if not src.exists():
            raise SystemExit(f"missing eval blob {src} — run the eval, or fetch it")
        ds = C.load_jsonl_dataset(jsonl, concept)
        blob = _load(src)
        if blob["model_name"] != C.MODEL_NAME or int(blob["layer"]) != C.LAYER:
            raise SystemExit(f"{src}: {blob['model_name']} L{blob['layer']}, expected "
                             f"{C.MODEL_NAME} L{C.LAYER}")
        if blob["activations"].shape[0] != len(ds):
            raise SystemExit(
                f"{src}: {blob['activations'].shape[0]} rows, {jsonl.name} has {len(ds)}"
            )
        _assert_right_padded(blob, str(src))
        state = _link(src, concept.eval_blob_dir / f"{jsonl.stem}-{concept.eval_blob_suffix}.pt")
        print(f"   {jsonl.stem:24s} {len(ds):5d} rows  w={blob['attention_mask'].shape[1]:5d}  {state}")


def prep_dev(concept: C.Concept, run_cache: Path) -> None:
    """Slice the one whole-dev-set blob into the per-split blobs the analysis wants."""
    from agentic_redteam.retrain import _dev_activation_cache_path

    files = sorted(concept.dev_dir.glob("*.jsonl"))
    src = _dev_activation_cache_path(run_cache, files, C.MODEL_NAME, C.LAYER, C.COMBINE, C.CONVERT)
    if not src.exists():
        raise SystemExit(
            f"missing dev blob {src.name} under {run_cache} — it is written by the first "
            f"retrain of a --dev-data run; has one finished?"
        )
    print(f"[dev]  {src.name} -> {concept.dev_blob_dir}")
    blob = _load(src)
    _assert_right_padded(blob, str(src))
    counts = [len(C.load_jsonl_dataset(f, concept)) for f in files]
    if sum(counts) != blob["activations"].shape[0]:
        raise SystemExit(
            f"{src}: {blob['activations'].shape[0]} rows, the dev splits sum to "
            f"{sum(counts)} — the blob is not this dev dir's"
        )
    concept.dev_blob_dir.mkdir(parents=True, exist_ok=True)
    off = 0
    for jsonl, n in zip(files, counts):
        dst = concept.dev_blob_dir / f"{jsonl.stem}-{concept.dev_blob_suffix}.pt"
        rows = slice(off, off + n)
        mask = blob["attention_mask"][rows].clone()
        # Trim to this split's own longest real row. Lossless: the source is right-padded
        # and the mask already zeroes everything past it.
        w = int(mask.sum(dim=1).max())
        if not dst.exists():
            torch.save(
                {
                    "activations": blob["activations"][rows, :w].clone(),
                    "attention_mask": mask[:, :w],
                    "input_ids": blob["input_ids"][rows, :w].clone(),
                    "model_name": C.MODEL_NAME,
                    "layer": C.LAYER,
                },
                dst,
            )
            state = "written"
        else:
            state = "ok"
        print(f"   {jsonl.stem:24s} {n:5d} rows  w={w:5d}  {state}")
        off += n


def prep_redteam_cache(concept: C.Concept, run_cache: Path) -> None:
    """Link the run's per-conversation red-team blobs; write the base rows beside them."""
    src_dir = run_cache / f"redteam_acts_{C.MODEL_NAME.replace('/', '_')}_L{C.LAYER}"
    if not src_dir.is_dir():
        raise SystemExit(f"no per-conversation red-team cache at {src_dir}")
    dst_dir = concept.redteam_cache_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for src in src_dir.glob("*.pt"):
        if _link(src, dst_dir / src.name) == "linked":
            linked += 1
    print(f"[rt]   {src_dir.name}: {len(list(src_dir.glob('*.pt')))} blobs "
          f"({linked} newly linked) -> {dst_dir}")


def prep_base(concept: C.Concept, run_cache: Path) -> None:
    """Split the whole-split base blob into per-conversation blobs under the analysis key.

    The base rows are the one thing the run does *not* cache per conversation, and
    ``ca_common.redteam_source`` needs them in that form. Blob row *i* is base row *i*
    (ascending indices, ``test_size=0.0``), and that is checked here per row against a
    fresh tokenization rather than assumed.
    """
    from agentic_redteam.retrain import _base_activation_cache_paths

    ds = C.load_jsonl_dataset(concept.base_jsonl, concept)
    train_blob, _val_blob = _base_activation_cache_paths(
        run_cache, concept.base_jsonl, C.MODEL_NAME, C.LAYER,
        C.FIT_SEED, 0.0, None, C.COMBINE, C.CONVERT, 1.0,
    )
    if not train_blob.exists():
        raise SystemExit(f"missing base blob {train_blob.name} under {run_cache}")
    print(f"[base] {train_blob.name} -> {concept.redteam_cache_dir}")
    blob = _load(train_blob)
    _assert_right_padded(blob, str(train_blob))
    if blob["activations"].shape[0] != len(ds):
        raise SystemExit(
            f"{train_blob}: {blob['activations'].shape[0]} rows, {concept.base_jsonl.name} "
            f"has {len(ds)} — this blob was not computed at test_size=0.0"
        )
    tds, paths = C._redteam_conv_sources(ds, concept.redteam_cache_dir)
    # Tokenize exactly as `tokenize_inputs` does — the chat template emits the special
    # tokens itself, so the tokenizer must be called with add_special_tokens=False or
    # every row reads one token wide. `token_budget.count_tokens` carries the same two
    # traps; this is the id-level version of it.
    from agentic_redteam.token_budget import _load_tokenizer, apply_message_transforms

    tok = _load_tokenizer(C.MODEL_NAME)
    written = 0
    for i, dst in enumerate(paths):
        n = int(blob["attention_mask"][i].sum())
        ids = blob["input_ids"][i, :n].tolist()
        if tok is not None:
            dicts = apply_message_transforms(
                tds.inputs[i],
                combine_consecutive_messages=C.COMBINE,
                convert_tool_to_assistant=C.CONVERT,
            )
            rendered = tok.apply_chat_template([dicts], tokenize=False)
            if isinstance(rendered, str):
                rendered = [rendered]
            want = tok(rendered, add_special_tokens=False)["input_ids"][0]
            if ids != list(want):
                raise SystemExit(
                    f"base row {i}: blob input_ids do not match a fresh tokenization of "
                    f"{concept.base_jsonl.name} row {i} — the blob's row order is not the "
                    f"file's, so slicing it would mislabel every base activation."
                )
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "activations": blob["activations"][i : i + 1, :n].clone(),
                "attention_mask": blob["attention_mask"][i : i + 1, :n].clone(),
                "input_ids": blob["input_ids"][i : i + 1, :n].clone(),
                "model_name": C.MODEL_NAME,
                "layer": C.LAYER,
            },
            dst,
        )
        written += 1
    print(f"   {len(paths)} base rows verified against their tokenization, {written} written")


def check_redteam_hits(concept: C.Concept) -> None:
    """Report how many of this arm's red-team conversations have a cached activation."""
    if not concept.redteam_jsonl.exists():
        print(f"[hits] {concept.redteam_jsonl.name}: not written yet — skipped")
        return
    ds = C.load_jsonl_dataset(concept.redteam_jsonl, concept, field_mapping={"label": "labels"})
    _, paths = C._redteam_conv_sources(ds, concept.redteam_cache_dir)
    hit = sum(1 for p in paths if p.exists())
    print(f"[hits] {concept.redteam_jsonl.name}: {hit}/{len(paths)} cached")
    if hit != len(paths):
        raise SystemExit(
            f"{len(paths) - hit} red-team conversations have no cached activation. They "
            f"would have to be extracted with a 27B load; this script never does that."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concept", required=True, choices=sorted(C.CONCEPTS))
    ap.add_argument("--run-cache", type=Path, default=DEFAULT_RUN_CACHE,
                    help="the run's base_activation_cache_dir (base, dev and red-team blobs)")
    ap.add_argument("--eval-cache", type=Path, default=DEFAULT_EVAL_CACHE,
                    help="the run's eval activations dir (<split>-acts_full.pt)")
    args = ap.parse_args()

    concept = C.CONCEPTS[args.concept]
    print(f"concept {concept.name}  (activations under {concept._acts_dir})")
    prep_eval(concept, args.eval_cache)
    prep_dev(concept, args.run_cache)
    prep_redteam_cache(concept, args.run_cache)
    prep_base(concept, args.run_cache)
    check_redteam_hits(concept)
    print("done — nothing was extracted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
