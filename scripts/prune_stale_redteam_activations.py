#!/usr/bin/env python
"""Delete per-conversation red-team activation blobs that nothing on this branch wants.

WHY THIS EXISTS
---------------
``scripts/shorten_long_contrastive_pairs.py`` (commit 98b5011) rewrote 105 contrastive
pairs that overran the 1024-token activation window. The red-team blob cache is
CONTENT-ADDRESSED — the key is a hash of the conversation's own transformed messages
(``retrain._redteam_activation_cache_path``) — so a rewritten conversation gets a *new*
key and its old blob is simply never asked for again. Stale blobs are therefore harmless
to correctness, and that is exactly what makes them easy to leave lying around: a box
that ran before the shortening keeps ~0.6 GB of activations for conversations that no
longer exist, indefinitely.

So this is a disk-space tool, not a correctness one, and it is written to be safe in the
one way that matters: it computes the set of keys the postprocessed dumps on this branch
actually reference, and deletes only blobs outside that set. Nothing is derived from
timestamps or from what "looks old".

Prune AFTER restoring, not before. Wiping the directory and re-fetching would move ~1500
blobs across the network to replace 105; restoring first (which writes only what is
missing) and pruning after transfers only the genuinely new ones.

THE KEEP SET IS EVERY ITERATION, NOT JUST THE LATEST. The iterations do not nest —
``filter_dataset`` refits each cycle, so 16-84 conversations present in one iteration are
absent from the next — so a blob unreferenced by iter3 may well be what iter1 trained on.
``--iterations`` narrows this deliberately; the default keeps everything any dump names.

Dev-sample activations are kept the same way, by reading ``dev_samples/`` — they live in
the same content-addressed cache and are referenced by no dump.

    python scripts/prune_stale_redteam_activations.py --dry-run
    python scripts/prune_stale_redteam_activations.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CACHE = (
    REPO_ROOT / "results_hu_harm_gemma27b_batch_ablation" / "base_activations"
)
ARM_DIRS = [
    REPO_ROOT / "probes" / "hu_harm_gemma27b_deepseekv4pro_batch",
    REPO_ROOT / "probes" / "hu_harm_gemma27b_gptoss120b_batch",
]
DEFAULT_DEV_DIR = REPO_ROOT / "dev_samples"
COMBINE = True
CONVERT = True


def _keys_for(conversations: list[list[dict]], cache_dir: Path, model: str, layer: int) -> set[str]:
    """Cache-blob stems for these conversations, transforms applied exactly as at train time."""
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

    from agentic_redteam.retrain import _redteam_activation_cache_path

    keys = set()
    for messages in conversations:
        dialogue = [
            TLMessage(role=str(m["role"]), content=str(m["content"])) for m in messages
        ]
        # convert first, then combine — the order load_from and _apply_message_transforms
        # both use. Reversing it changes the content hash and so changes every key.
        if CONVERT:
            dialogue = LabelledDataset._convert_tool_to_assistant(dialogue)
        if COMBINE:
            dialogue = LabelledDataset._combine_consecutive_messages(dialogue)
        keys.add(
            _redteam_activation_cache_path(
                cache_dir, dialogue, model, layer, COMBINE, CONVERT
            ).stem
        )
    return keys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--probe-dir", type=Path, nargs="*", default=ARM_DIRS)
    ap.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    ap.add_argument("--iterations", type=int, nargs="*", default=None,
                    help="Only keep blobs referenced by these iterations' dumps "
                         "(default: every dump present — the iterations do not nest)")
    ap.add_argument("--model-name", default="google/gemma-3-27b-it")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    blob_dir = args.cache_dir / f"redteam_acts_{args.model_name.replace('/', '_')}_L{args.layer}"
    if not blob_dir.is_dir():
        # Try the layout the cache actually uses rather than guessing at it.
        candidates = sorted(args.cache_dir.glob("redteam_acts_*"))
        if len(candidates) != 1:
            print(f"no single redteam_acts_* dir under {args.cache_dir} "
                  f"(found {[c.name for c in candidates]}) — nothing to prune")
            return 0
        blob_dir = candidates[0]
    print(f"blob dir : {blob_dir}")

    conversations: list[list[dict]] = []
    n_dumps = 0
    for probe_dir in args.probe_dir:
        for dump in sorted(probe_dir.glob("redteam_postprocessed_iter*.jsonl")):
            it = int(dump.stem.rsplit("iter", 1)[-1])
            if args.iterations is not None and it not in args.iterations:
                continue
            n_dumps += 1
            for line in dump.read_text().splitlines():
                if line.strip():
                    conversations.append(json.loads(line)["inputs"])
    n_redteam = len(conversations)

    n_dev = 0
    if args.dev_dir and args.dev_dir.is_dir():
        for path in sorted(args.dev_dir.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                inputs = row["inputs"]
                conversations.append(json.loads(inputs) if isinstance(inputs, str) else inputs)
                n_dev += 1

    keep = _keys_for(conversations, args.cache_dir, args.model_name, args.layer)
    print(f"keep set : {len(keep)} distinct conversation(s) from {n_dumps} dump(s) "
          f"({n_redteam} red-team rows + {n_dev} dev rows)")

    present = sorted(blob_dir.glob("*.pt"))
    stale = [p for p in present if p.stem not in keep]
    freed = sum(p.stat().st_size for p in stale)
    print(f"on disk  : {len(present)} blob(s); {len(stale)} stale "
          f"({freed / 1e9:.2f} GB)")

    if not stale:
        return 0
    if args.dry_run:
        for p in stale[:10]:
            print(f"  would delete {p.name}")
        if len(stale) > 10:
            print(f"  ... and {len(stale) - 10} more")
        return 0
    for p in stale:
        p.unlink()
    print(f"deleted {len(stale)} stale blob(s), freed {freed / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
