"""Extract gemma-3-27b activations for the v2 (minimal-edit) contrastive counterparts.

Everything else in this analysis has run off cached activations. These conversations are
new — the v2 prompt rewrote them — so they are the one step that needs real forward
passes through ``google/gemma-3-27b-it``.

Cost on this box, stated up front
---------------------------------
Even truncated to layers 0..32 the model is **30.9 GB** of bf16 weights against 8 GB of
VRAM and 15 GB of host RAM, so accelerate cannot place it and each forward streams
weights off disk. Measured here: **24.5 s/sample**, so ~1 hour for 155 conversations —
well short of the 48–264 s/sample the retrain notes warn about, but not free.

**Layer truncation has to be OFF on a box that disk-offloads.**
``AGENTIC_REDTEAM_TRUNCATE_LAYERS=0`` is required here, which is the opposite of the
usual advice. ``_truncated_config`` rebuilds the config with ``num_hidden_layers = 33``,
so the checkpoint's layers 33..61 have no module in the device map; transformers'
``get_disk_only_shard_files`` then looks them up and dies on ``KeyError: ''``. That
function only runs when the device map contains ``"disk"``, which never happened on the
cloud box where the model fit in GPU + CPU — so this is an interaction with offload, not
a regression in the truncation.

Running the full 62-layer stack instead costs load time and per-forward streaming but is
**numerically identical at layer 32**: the stack is causal, so layer 32's output depends
only on layers 0..32 whether or not the deeper ones are instantiated. That is the same
argument that lets the truncation exist at all, and it is why no activation cache key
mentions truncation — blobs computed either way are interchangeable, and these blobs sit
in the same store as the ones the cloud box wrote.

Which is why it delegates to ``retrain._activate_redteam_cached`` rather than calling
``get_activations`` itself. That helper computes misses in chunks of tuberlens'
``BATCH_SIZE`` and **writes each row's blob the moment its chunk returns**, so a crash
at row 150 of 157 costs one row rather than the run — the property that matters most
when a run is this long. It also skips anything already on disk, so re-running is free
and safe.

Blobs are written into the same store the rest of the analysis reads
(``attribution_lib.SHARED_CACHE``) under the production key
(``retrain._redteam_activation_cache_path``: the conversation's own transformed messages
plus model / layer / transform flags). So the scorer finds them with no wiring — and
because the key is content-addressed, a v2 counterpart cannot collide with the v1 one it
replaces.

Transforms are applied before extraction, in ``LabelledDataset.load_from`` order, because
that is what the key and every existing blob in the store assume.

Usage:
    .venv_claude/bin/python scripts/extract_contrastive_v2_activations.py --plan
    .venv_claude/bin/python scripts/extract_contrastive_v2_activations.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A

DEFAULT_IN = A.REPO / "results_hs_gemma27b_batch_ablation/contrastive_v2/cohort_contrastive_v2.jsonl"


def load_new_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path} — run scripts/regen_cohort_contrastive.py first")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_messages(raw: list[dict]):
    """Config transforms applied, matching how every blob in the store is keyed."""
    from tuberlens.interfaces.dataset import Message

    return A.apply_transforms(
        [Message(role=m["role"], content=m["content"]) for m in raw]
    )


def build_dataset(rows: list[dict], which: str):
    """A ``LabelledDataset`` of the conversations to extract.

    ``which`` selects ``new_messages`` (the v2 counterparts) or ``source_messages``.
    Sources are already in the store from the original run — the option exists only so a
    fresh checkout can rebuild everything.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    key = {"new": "new_messages", "source": "source_messages"}[which]
    inputs, ids, labels = [], [], []
    seen = set()
    for i, r in enumerate(rows):
        msgs = to_messages(r[key])
        canon = A.canon(msgs)
        if canon in seen:  # a v2 rewrite can coincide with another's; extract once
            continue
        seen.add(canon)
        inputs.append(msgs)
        ids.append(f"{r['arm']}-{which}-{r['source_key']}")
        # For extraction the label is inert — activations do not depend on it — but
        # LabelledDataset wants the field, so carry the true one rather than a dummy.
        labels.append(r["target_label"] if which == "new" else r["source_label"])
    return LabelledDataset(inputs=inputs, ids=ids, other_fields={"labels": labels})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--which", choices=("new", "source", "both"), default="new")
    ap.add_argument("--cache-dir", type=Path, default=A.SHARED_CACHE)
    ap.add_argument("--plan", action="store_true",
                    help="report how many conversations are already cached and exit")
    ap.add_argument("--limit", type=int, default=0,
                    help="extract at most N uncached conversations this run (0 = all). "
                         "Useful to time a few before committing to the whole set.")
    args = ap.parse_args()

    from agentic_redteam.model_loading import extraction_batch_size, load_extraction_model
    from agentic_redteam.retrain import _activate_redteam_cached

    rows = load_new_pairs(args.in_path)
    print(f"{len(rows)} pair(s) in {args.in_path.name}", flush=True)

    which = ["new", "source"] if args.which == "both" else [args.which]
    for w in which:
        ds = build_dataset(rows, w)
        paths = [A.redteam_blob_path(m) for m in ds.inputs]
        have = [p.exists() for p in paths]
        n_miss = have.count(False)
        print(f"\n=== {w} ===", flush=True)
        print(f"  {len(ds)} unique conversation(s); {have.count(True)} cached, "
              f"{n_miss} to extract", flush=True)
        if args.plan or n_miss == 0:
            continue

        if args.limit and n_miss > args.limit:
            # Keep the cached ones (free) and only the first N misses, so the helper's
            # own hit/miss partition does the right thing and progress is preserved.
            keep = []
            budget = args.limit
            for i, hit in enumerate(have):
                if hit:
                    keep.append(i)
                elif budget:
                    keep.append(i)
                    budget -= 1
            from tuberlens.interfaces.dataset import LabelledDataset
            ds = LabelledDataset(
                inputs=[ds.inputs[i] for i in keep],
                ids=[ds.ids[i] for i in keep],
                other_fields={"labels": [ds.other_fields["labels"][i] for i in keep]},
            )
            print(f"  --limit: extracting {args.limit} of them this run", flush=True)

        truncating = os.environ.get("AGENTIC_REDTEAM_TRUNCATE_LAYERS", "1") != "0"
        print(f"  batch size {extraction_batch_size()} (tuberlens BATCH_SIZE); "
              f"blobs -> {args.cache_dir}", flush=True)
        print(f"  ~{n_miss * 24.5 / 60:.0f} min at the 24.5 s/sample measured on this box "
              f"(disk-offloaded)", flush=True)
        if truncating:
            # Loud, because the failure is a bare KeyError deep in transformers that
            # gives no hint about the cause. See the module docstring.
            print("  WARNING: AGENTIC_REDTEAM_TRUNCATE_LAYERS is not 0. If this box "
                  "disk-offloads, the load will die with KeyError: '' in "
                  "get_disk_only_shard_files. Set it to 0.", flush=True)

        loaded: dict = {"model": None}

        def get_model():
            if loaded["model"] is None:
                print("  loading extraction model (layers 0..%d) ..." % A.PROBE_LAYER,
                      flush=True)
                loaded["model"] = load_extraction_model(
                    A.PROBE_MODEL, A.PROBE_LAYER, verbose=True
                )
            return loaded["model"]

        t0 = time.time()
        _activate_redteam_cached(
            ds,
            args.cache_dir,
            A.PROBE_MODEL,
            A.PROBE_LAYER,
            A.COMBINE_CONSECUTIVE_MESSAGES,
            A.CONVERT_TOOL_TO_ASSISTANT,
            get_model,
            True,
        )
        # Free the extraction model before the next pass or the process exit — a
        # gemma-sized model keeps multi-GB of CPU-offloaded shards resident for as long
        # as it is referenced (see the conventions on freeing GPU memory between phases).
        if loaded["model"] is not None:
            import gc

            import torch

            loaded["model"] = None
            gc.collect()
            torch.cuda.empty_cache()

        done = sum(1 for m in ds.inputs if A.redteam_blob_path(m).exists())
        print(f"  {done}/{len(ds)} cached after {time.time() - t0:.0f}s", flush=True)

    print("\nre-run any time: cached conversations are skipped, so this is resumable.")


if __name__ == "__main__":
    main()
