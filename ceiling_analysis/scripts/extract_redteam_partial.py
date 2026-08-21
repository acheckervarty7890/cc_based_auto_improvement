#!/usr/bin/env python
"""Start the red-team extraction before gemma-3-27b has finished downloading.

`load_extraction_model` truncates the stack to layers `0..probe.layer`, so a probe on
layer 32 of 62 only ever instantiates language layers 0..32, the embeddings, the final
norm, the vision tower and the projector. On the published `google/gemma-3-27b-it`
checkpoint every one of those tensors lives in shards **1-7 and 12**; shards 8-11 hold
language layers 37..61 and nothing else. So the extraction can begin as soon as 8 of the
12 shards are on disk — which on a fresh box is most of an hour earlier.

The one thing in the way is that `from_pretrained` opens **every** shard named in
`model.safetensors.index.json`, whether or not the truncated model needs it. This builds
`gemma_partial/` — symlinks to the shards present, plus an index with the entries for the
absent ones filtered out — and points the loader at that.

**This does not change a single weight that is read**, so the activations are identical to
ones extracted from the complete download: same truncation the normal path already
applies, applied to the files as well as to the config. Two things keep that honest:

* the cache is keyed on `ca_common.MODEL_NAME`, never on this directory, so the blobs are
  indistinguishable from ordinary ones and interoperate with them;
* `verify_batch_padding.py` re-extracts cached rows with a fresh **complete** load and
  compares. On the run this was written for it reported `0.00e+00` on every sampled row.

It refuses to build an index that is missing a tensor the truncated model needs, so a
checkpoint whose shards are laid out differently fails loudly instead of silently
returning a randomly-initialised layer.

    HF_TOKEN=... ceiling_analysis/scripts/extract_redteam_partial.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca_common as C  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PARTIAL = REPO / "gemma_partial"
HF_HUB = Path.home() / ".cache/huggingface/hub/models--google--gemma-3-27b-it/snapshots"


def snapshot_dir() -> Path:
    dirs = sorted(HF_HUB.glob("*/"))
    if not dirs:
        raise SystemExit(f"no gemma snapshot under {HF_HUB}")
    return dirs[0]


def build_partial(snap: Path, layer: int) -> Path:
    """Mirror `snap` into PARTIAL, keeping only shards that are actually present."""
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    present = {
        f.name for f in snap.iterdir()
        if f.name.endswith(".safetensors") and f.is_file() and f.stat().st_size > 0
    }

    needed, missing = set(), []
    for tensor, shard in weight_map.items():
        m = re.search(r"^language_model\.model\.layers\.(\d+)\.", tensor)
        if m and int(m.group(1)) > layer:
            continue  # not instantiated once num_hidden_layers = layer + 1
        needed.add(shard)
        if shard not in present:
            missing.append((tensor, shard))
    if missing:
        raise SystemExit(
            f"{len(missing)} tensor(s) the layer-{layer} model needs are in shards that "
            f"have not downloaded yet, e.g. {missing[0][0]} in {missing[0][1]}. "
            "Wait for those shards rather than loading a partly-random model."
        )

    PARTIAL.mkdir(exist_ok=True)
    kept = {k: v for k, v in weight_map.items() if v in present}
    (PARTIAL / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": index.get("metadata", {}), "weight_map": kept}, indent=2)
    )
    for f in snap.iterdir():
        if f.name == "model.safetensors.index.json":
            continue
        if f.name.endswith(".safetensors") and f.name not in present:
            continue
        dst = PARTIAL / f.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(os.path.realpath(f))
    print(f"[partial] {len(present)}/12 shards present; index keeps {len(kept)} tensors, "
          f"drops {len(weight_map) - len(kept)} (layers > {layer})", flush=True)
    return PARTIAL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=list(C.CONCEPTS))
    args = ap.parse_args()

    from agentic_redteam.model_loading import load_extraction_model
    from agentic_redteam.retrain import (
        _activate_redteam_cached,
        _apply_message_transforms,
        _redteam_activation_cache_path,
    )

    model_path = build_partial(snapshot_dir(), C.LAYER)
    loaded = {"model": None}

    def get_model():
        if loaded["model"] is None:
            t0 = time.time()
            loaded["model"] = load_extraction_model(str(model_path), C.LAYER, verbose=True)
            print(f"  model loaded in {time.time() - t0:.0f}s", flush=True)
        return loaded["model"]

    for name in args.concepts:
        concept = C.CONCEPTS[name]
        print(f"=== {name} ===", flush=True)
        cache_dir = concept.redteam_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        for label, path, mapping in (
            ("redteam", concept.redteam_jsonl, {"label": "labels"}),
            ("base", concept.base_jsonl, None),
        ):
            ds = C.load_jsonl_dataset(path, concept, field_mapping=mapping)
            ds = _apply_message_transforms(ds, C.COMBINE, C.CONVERT)
            miss = sum(
                0
                if _redteam_activation_cache_path(
                    cache_dir, m, C.MODEL_NAME, C.LAYER, C.COMBINE, C.CONVERT
                ).exists()
                else 1
                for m in ds.inputs
            )
            print(f"  {name}/{label}: {len(ds)} rows, {miss} uncached", flush=True)
            if miss == 0:
                continue
            t0 = time.time()
            _activate_redteam_cached(
                ds, cache_dir, C.MODEL_NAME, C.LAYER, C.COMBINE, C.CONVERT, get_model, True
            )
            print(f"  {name}/{label}: done in {time.time() - t0:.0f}s", flush=True)

    loaded["model"] = None
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
