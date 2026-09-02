#!/usr/bin/env python
"""Point a run at the high-stakes activations THIS BOX already has, so nothing is
extracted or downloaded again.

Two caches have to be filled, and they are filled differently:

* **Eval** (``output.activations_cache_dir``) is per split, keyed by path:
  ``<split>-acts_full.pt``. Blobs computed by ``scripts/extract_eval_activations.py``
  already carry those exact names, so this only has to place them — as a HARD LINK, so
  48 GB of eval blobs are not copied. Once they are there ``prefetch_eval_activations``
  validates them and skips the Kaggle download, and ``_assign_cached_activations``
  attaches them before ``get_performances`` can load the 27B model.

* **Dev** (``output.base_activation_cache_dir``) is ONE blob for the whole dev set,
  named by ``_dev_activation_cache_path`` — a content hash of the dev JSONLs plus
  ``model | layer | combine | convert``. There is no per-split form of it and no Kaggle
  fetch path for it, so the first fit would otherwise recompute all 1908 rows through
  gemma-3-27b. This assembles it instead, by concatenating the per-split dev blobs in
  ``sorted(dev_dir.glob("*.jsonl"))`` order — exactly the order ``_load_dev_dataset``
  concatenates the splits in, so row i of the blob is row i of the dev dataset.

The concatenation is only sound because every high-stakes blob is padded to the same
1024 tokens (``LabelledDataset.concatenate`` would zero-pad to the common max anyway).
That is asserted, not assumed.

Every blob is checked with the same ``_validate_blob`` the Kaggle download side runs
(model name, layer, row count) BEFORE it is placed, so a stale or truncated blob is
rejected here rather than silently training a probe on the wrong activations.

    .venv_claude/bin/python scripts/stage_local_hs_activations.py \
        --config configs/gptoss120b_hs_gemma27b_ens10dev.md --dry-run

Drop --dry-run to place them. Re-running is a no-op once everything is staged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from agentic_redteam.config import load_config  # noqa: E402
from agentic_redteam.kaggle_activations import (  # noqa: E402
    KaggleActivationError,
    _validate_blob,
)
from agentic_redteam.retrain import _dev_activation_cache_path  # noqa: E402

CACHE_STEM = "acts_full.pt"


def _rows(jsonl: Path) -> int:
    with jsonl.open() as fh:
        return sum(1 for line in fh if line.strip())


def _place(blob: Path, target: Path) -> str:
    """Hard-link ``blob`` to ``target``; fall back to a symlink across filesystems."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.hardlink_to(blob)
        return "hardlink"
    except OSError:
        target.symlink_to(blob.resolve())
        return "symlink"


def _stage_eval(source_dir: Path, eval_dir: Path, cache_dir: Path, *, model_name: str,
                layer: int, dry_run: bool) -> int:
    problems = 0
    for jsonl in sorted(eval_dir.glob("*.jsonl")):
        split = jsonl.stem
        blob = source_dir / f"{split}-{CACHE_STEM}"
        target = cache_dir / f"{split}-{CACHE_STEM}"
        n = _rows(jsonl)
        if target.exists():
            try:
                _validate_blob(target, split=split, model_name=model_name, layer=layer, n_rows=n)
                print(f"  OK    {split}: already staged ({n} rows)")
                continue
            except KaggleActivationError as exc:
                print(f"  STALE {split}: {exc} — replacing", file=sys.stderr)
                if not dry_run:
                    target.unlink()
        if not blob.is_file():
            print(f"  MISS  {split}: no blob at {blob}", file=sys.stderr)
            problems += 1
            continue
        try:
            _validate_blob(blob, split=split, model_name=model_name, layer=layer, n_rows=n)
        except KaggleActivationError as exc:
            print(f"  BAD   {exc}", file=sys.stderr)
            problems += 1
            continue
        size = blob.stat().st_size / 1e9
        if dry_run:
            print(f"  PLAN  {split}: {n} rows, {size:.2f} GB -> {target}")
        else:
            how = _place(blob, target)
            print(f"  STAGE {split}: {n} rows, {size:.2f} GB -> {target} ({how})")
    return problems


def _stage_dev(source_dir: Path, dev_dir: Path, cache_dir: Path, *, model_name: str,
               layer: int, combine: bool, convert: bool, dry_run: bool) -> int:
    dev_files = sorted(dev_dir.glob("*.jsonl"))
    if not dev_files:
        print(f"  MISS  no dev splits in {dev_dir}", file=sys.stderr)
        return 1
    target = _dev_activation_cache_path(cache_dir, dev_files, model_name, layer, combine, convert)
    total_rows = sum(_rows(f) for f in dev_files)
    if target.exists():
        acts = torch.load(target, map_location="cpu", mmap=True, weights_only=False)
        if acts["activations"].shape[0] == total_rows:
            print(f"  OK    dev blob already staged: {target.name} ({total_rows} rows)")
            return 0
        print(f"  STALE dev blob {target.name} has {acts['activations'].shape[0]} rows, "
              f"want {total_rows} — rebuilding", file=sys.stderr)

    parts = []
    problems = 0
    for f in dev_files:
        split = f.stem
        blob = source_dir / f"{split}-{CACHE_STEM}"
        n = _rows(f)
        if not blob.is_file():
            print(f"  MISS  {split}: no blob at {blob}", file=sys.stderr)
            problems += 1
            continue
        try:
            _validate_blob(blob, split=split, model_name=model_name, layer=layer, n_rows=n)
        except KaggleActivationError as exc:
            print(f"  BAD   {exc}", file=sys.stderr)
            problems += 1
            continue
        parts.append((split, blob, n))
    if problems:
        return problems

    print(f"  PLAN  dev blob: {' + '.join(f'{s}({n})' for s, _, n in parts)} "
          f"= {total_rows} rows -> {target}")
    if dry_run:
        return 0

    heads = [torch.load(b, map_location="cpu", mmap=True, weights_only=False) for _, b, _ in parts]
    seq_lens = {h["activations"].shape[1] for h in heads}
    hidden = {h["activations"].shape[2] for h in heads}
    dtypes = {h["activations"].dtype for h in heads}
    # Concatenating along dim 0 is only row-for-row correct if every part shares the
    # padded length; LabelledDataset.concatenate would zero-pad to the common max, which
    # is a different tensor than the one the fit would have computed for a longer row.
    assert len(seq_lens) == 1 and len(hidden) == 1 and len(dtypes) == 1, (
        f"parts disagree on shape/dtype: seq={seq_lens} hidden={hidden} dtype={dtypes}"
    )
    seq, hid, dtype = seq_lens.pop(), hidden.pop(), dtypes.pop()

    out = {
        "activations": torch.empty((total_rows, seq, hid), dtype=dtype),
        "attention_mask": torch.empty((total_rows, seq), dtype=heads[0]["attention_mask"].dtype),
        "input_ids": torch.empty((total_rows, seq), dtype=heads[0]["input_ids"].dtype),
        "layer": layer,
        "model_name": model_name,
    }
    at = 0
    for (split, _, n), head in zip(parts, heads):
        for key in ("activations", "attention_mask", "input_ids"):
            out[key][at:at + n] = head[key]
        at += n
        print(f"        copied {split}: rows {at - n}..{at}", flush=True)
    assert at == total_rows, f"copied {at} rows, expected {total_rows}"
    del heads

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.partial")
    torch.save(out, tmp)
    tmp.replace(target)
    print(f"  STAGE dev blob: {target} ({target.stat().st_size / 1e9:.2f} GB)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path,
                    help="Run config; supplies the two cache dirs, the probe model/layer, "
                         "the eval transforms and validation.dev_data")
    ap.add_argument("--eval-dataset-dir", type=Path, default=REPO_ROOT / "eval_sets/highstakes",
                    help="Eval splits the run will score against (must match --eval-dataset-dir)")
    ap.add_argument("--eval-source", type=Path, default=REPO_ROOT / "acts_new/hs_eval",
                    help="Dir holding locally computed <split>-acts_full.pt for the EVAL splits")
    ap.add_argument("--dev-source", type=Path, default=REPO_ROOT / "acts_new/hs_dev",
                    help="Dir holding per-split <split>-acts_full.pt for the DEV splits")
    ap.add_argument("--dry-run", action="store_true", help="Report the plan; place nothing")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.validation.dev_data is None:
        raise SystemExit(f"ERROR: {args.config} sets no validation.dev_data — nothing to stage.")
    model_name, layer = cfg.probe.model, int(cfg.probe.layer)
    combine = cfg.eval.combine_consecutive_messages
    convert = cfg.eval.convert_tool_to_assistant

    print(f"config     : {args.config}")
    print(f"probe      : {model_name} L{layer}")
    print(f"transforms : combine={combine} convert_tool={convert}")
    print(f"eval cache : {cfg.output.activations_cache_dir}")
    print(f"base cache : {cfg.output.base_activation_cache_dir}")
    print()
    print("EVAL splits")
    problems = _stage_eval(args.eval_source, args.eval_dataset_dir,
                           Path(cfg.output.activations_cache_dir),
                           model_name=model_name, layer=layer, dry_run=args.dry_run)
    print("\nDEV set")
    problems += _stage_dev(args.dev_source, Path(cfg.validation.dev_data),
                           Path(cfg.output.base_activation_cache_dir),
                           model_name=model_name, layer=layer, combine=combine,
                           convert=convert, dry_run=args.dry_run)
    if problems:
        print(f"\n{problems} problem(s) — see above.", file=sys.stderr)
        return 1
    print("\n--dry-run: nothing placed." if args.dry_run else "\nStaged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
