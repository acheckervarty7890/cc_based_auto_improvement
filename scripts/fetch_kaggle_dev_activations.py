#!/usr/bin/env python
"""Download the high-stakes DEV activations published on Kaggle, per split.

`evaluate_probe` already has a Kaggle path for the EVAL splits (the config's
`kaggle:` section → `prefetch_eval_activations`). The DEV set has none: it is one
content-keyed blob, not a per-split cache, so the first fit recomputes all 1908 rows
through gemma-3-27b — ~50 minutes of forwards that this repo has already published
the answer to. This fetches the per-split blobs; `scripts/stage_local_hs_activations.py
--dev-source` then assembles them into the single blob `_dev_activation_cache_path`
derives.

WHY A SEPARATE SCRIPT AND NOT `kaggle:` TEMPLATES. `KaggleActivationSource` formats
one template pair over the split stem, and the dev blobs are NOT reachable that way:
they were published when today's dev splits were the eval set, under the names those
splits had then. `dev_samples/highstakes/anthropic_hh_balanced.jsonl` lives in
`anku7890/anthropicgemmaevalpt` as `anthropic-gemmaeval.pt` — neither `{split}` nor
`{slug}` produces that. So the mapping is an explicit table (`DEV_SOURCES`), which is
also where a future dev set would be added.

WHAT MAKES THIS SAFE. Two checks, both refusing rather than warning:

1. `_validate_blob` — the same one the eval download side runs: model name, layer and
   ROW COUNT. The count is the discriminator that matters here, since the dev and eval
   splits share their stems today: dev is 1028/278/274/328, eval is 2984/604/86/734, so
   a blob pointed at the wrong one cannot pass.

2. `--verify-tokens` (default on) — row count alone cannot catch a blob whose rows are
   the right ones in the wrong ORDER, which would silently pair every activation with
   another row's label. So each split's stored `input_ids` are compared against the dev
   JSONL tokenized locally through the same path extraction uses (chat template,
   `add_special_tokens=False`, the config's two transforms), on a sample of rows
   (`--verify-rows`, 0 = all). CPU only — no GPU, no model weights.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from agentic_redteam.config import load_config  # noqa: E402
from agentic_redteam.kaggle_activations import (  # noqa: E402
    KaggleActivationError,
    _authenticate,
    _extract_downloaded,
    _validate_blob,
)

CACHE_STEM = "acts_full.pt"

# dev split stem -> (kaggle dataset slug, file name inside it). The dev splits were the
# eval set when these were published, hence the pre-rename names.
DEV_SOURCES = {
    "anthropic_hh_balanced": ("anthropicgemmaevalpt", "anthropic-gemmaeval.pt"),
    "mt_balanced": ("mtgemmaevalpt", "mt-gemmaeval.pt"),
    "mts_balanced": ("mtsgemmaevalpt", "mts-gemmaeval.pt"),
    "toolace_balanced": ("toolacegemmaevalpt", "toolace-gemmaeval.pt"),
}


def _rows(jsonl: Path) -> int:
    with jsonl.open() as fh:
        return sum(1 for line in fh if line.strip())


def _download(api, owner: str, slug: str, remote_name: str, target: Path, split: str) -> None:
    staging = target.parent / f".staging-dev-{split}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        handle = f"{owner}/{slug}"
        print(f"[kaggle] {split}: downloading {handle}:{remote_name} ...", flush=True)
        api.dataset_download_file(handle, remote_name, path=str(staging), quiet=False)
        blob = _extract_downloaded(staging, split)
        blob.replace(target)          # same filesystem: a rename, not a second copy
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _verify_tokens(blob: Path, jsonl: Path, cfg, split: str, n_check: int) -> int:
    """Compare stored input_ids against the dev rows tokenized locally. 0 = all rows.

    The local side goes through exactly what extraction does — ``apply_message_transforms``
    with the config's two flags, then the chat template batched one dialogue at a time with
    ``add_special_tokens=False`` — i.e. ``token_budget.count_tokens``'s path, which was
    written against ``tokenize_inputs`` itself rather than a reading of it.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    from agentic_redteam.token_budget import _load_tokenizer, apply_message_transforms

    dataset = LabelledDataset.load_from(
        jsonl,
        pos_class_label=cfg.probe.pos_class_label,
        neg_class_label=cfg.probe.neg_class_label,
        combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
        convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant,
    )
    inputs = list(dataset.inputs)
    data = torch.load(blob, map_location="cpu", mmap=True, weights_only=False)
    ids, mask = data["input_ids"], data["attention_mask"]

    tok = _load_tokenizer(cfg.probe.model)
    if tok is None:
        print(f"  SKIP  {split}: no tokenizer for {cfg.probe.model}", file=sys.stderr)
        return 0

    step = 1 if n_check <= 0 else max(1, len(inputs) // n_check)
    idxs = list(range(0, len(inputs), step))
    bad = 0
    for i in idxs:
        dicts = apply_message_transforms(
            inputs[i],
            combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
            convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant,
        )
        rendered = tok.apply_chat_template([dicts], tokenize=False)
        if isinstance(rendered, str):
            rendered = [rendered]
        want = tok(rendered, add_special_tokens=False)["input_ids"][0][: ids.shape[1]]
        # Padding side varies by tokenizer; compare the span the attention mask marks,
        # which is padding-side agnostic. Truncated rows (>1024 tokens) compare their
        # kept prefix, which is what extraction stored.
        got = ids[i][mask[i].bool()].tolist()
        if got != want[: len(got)] or len(got) != min(len(want), ids.shape[1]):
            bad += 1
            if bad <= 3:
                diff = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), "length")
                print(f"  MISMATCH {split} row {i}: stored {len(got)} tokens, local "
                      f"{len(want)}; first difference at {diff}", file=sys.stderr)
    if bad:
        print(f"  BAD   {split}: {bad}/{len(idxs)} sampled rows do not match the local "
              f"tokenization — wrong split, or right rows in the wrong order.",
              file=sys.stderr)
    else:
        print(f"  OK    {split}: {len(idxs)} sampled rows match the local tokenization")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path,
                    help="Run config; supplies the probe model/layer, the transforms, "
                         "validation.dev_data and the kaggle owner")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT / "acts_new/hs_dev",
                    help="Where to write <split>-acts_full.pt (feed this to "
                         "stage_local_hs_activations.py --dev-source)")
    ap.add_argument("--verify-tokens", action=argparse.BooleanOptionalAction, default=True,
                    help="Check stored input_ids against the local tokenization (default on)")
    ap.add_argument("--verify-rows", type=int, default=40,
                    help="Rows per split to token-check, spread evenly (0 = all)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.validation.dev_data is None:
        raise SystemExit(f"ERROR: {args.config} sets no validation.dev_data — nothing to fetch.")
    if cfg.kaggle is None:
        raise SystemExit(f"ERROR: {args.config} has no kaggle: section — no owner to fetch from.")

    dev_dir = Path(cfg.validation.dev_data)
    model_name, layer = cfg.probe.model, int(cfg.probe.layer)
    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"dev splits : {dev_dir}\nprobe      : {model_name} L{layer}\ndest       : {args.dest}\n")

    api = None
    problems = 0
    for jsonl in sorted(dev_dir.glob("*.jsonl")):
        split = jsonl.stem
        n = _rows(jsonl)
        target = args.dest / f"{split}-{CACHE_STEM}"

        if target.is_file():
            try:
                _validate_blob(target, split=split, model_name=model_name, layer=layer, n_rows=n)
                print(f"  OK    {split}: already fetched ({n} rows)")
            except KaggleActivationError as exc:
                print(f"  STALE {split}: {exc} — refetching", file=sys.stderr)
                target.unlink()
        if not target.is_file():
            if split not in DEV_SOURCES:
                print(f"  MISS  {split}: no published dataset known for this split "
                      f"(add it to DEV_SOURCES)", file=sys.stderr)
                problems += 1
                continue
            slug, remote_name = DEV_SOURCES[split]
            if api is None:
                api = _authenticate()
            try:
                _download(api, cfg.kaggle.owner, slug, remote_name, target, split)
                _validate_blob(target, split=split, model_name=model_name, layer=layer, n_rows=n)
                print(f"  FETCH {split}: {n} rows, "
                      f"{target.stat().st_size / 1e9:.2f} GB -> {target}")
            except (KaggleActivationError, Exception) as exc:  # noqa: BLE001 — report and continue
                print(f"  BAD   {split}: {exc}", file=sys.stderr)
                target.unlink(missing_ok=True)
                problems += 1
                continue

        if args.verify_tokens:
            problems += _verify_tokens(target, jsonl, cfg, split, args.verify_rows)

    if problems:
        print(f"\n{problems} problem(s) — see above. NOT safe to stage.", file=sys.stderr)
        return 1
    print("\nFetched and verified. Assemble the dev blob with:\n"
          f"  .venv_claude/bin/python scripts/stage_local_hs_activations.py \\\n"
          f"      --config {args.config} --dev-source {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
