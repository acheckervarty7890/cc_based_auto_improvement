#!/usr/bin/env python
"""Put the batched-extraction difference on a scale, by measuring what a *no-op* difference costs.

`verify_batch_padding` shows that batch-4 (left-padded) and batch-1 activations for the same
conversation differ a little. A bare number there means nothing without a reference, because
two things already guarantee a nonzero difference: bf16 arithmetic is not associative, so
any change in tensor shapes reorders reductions, and the published eval/dev blobs were
computed on *another machine*, with its own kernels and its own CPU/GPU offload split.

So this measures three quantities on the same rows, in float64, as relative L2 error:

  repeat      the same conversation extracted twice at batch 1 here — pure nondeterminism
  published   local batch-1 extraction vs the published dev blob — machine-to-machine drift,
              which every activation in this analysis already carries
  batched     local batch-4 vs local batch-1 — the effect actually under suspicion

If `batched` sat at or below `published`, batching would have changed nothing this analysis
was not already living with. It does not: measured on `hu_ha`, `repeat` and `published` are
both exactly 0 while `batched` is ~1e-2. Extraction is bit-exact and machine-independent at
batch size 1, so there is no drift to hide in — which is why the red-team cache here is
built one row at a time despite being ~8x slower.

A fourth measurement in the same session ruled out the obvious culprit: gemma's tokenizer
pads *left*, so batching shifts every real token's position, but a batch of four identical
copies of one conversation — needing no padding at all — drifts by the same ~1e-2. It is
bf16 matmul reduction order, not padding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca_common as C  # noqa: E402


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a64, b64 = a.double(), b.double()
    return float((a64 - b64).norm() / a64.norm())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="hu_ha")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from agentic_redteam.model_loading import load_extraction_model

    concept = C.CONCEPTS[args.concept]
    dev_src = C.dev_source(concept)
    rng = np.random.default_rng(args.seed)
    rows = sorted(int(i) for i in rng.choice(len(dev_src), size=args.n, replace=False))
    inputs = [dev_src.dataset.inputs[i] for i in rows]

    model = load_extraction_model(C.MODEL_NAME, C.LAYER, verbose=True)

    def extract(batch_inputs, batch_size):
        model.batch_size = batch_size
        act = model.get_activations(batch_inputs, layer=C.LAYER)
        out = []
        for r in range(len(batch_inputs)):
            keep = act.attention_mask[r].bool()
            out.append(act.activations[r][keep].float())
        return out

    solo = [extract([x], 1)[0] for x in inputs]
    solo2 = [extract([x], 1)[0] for x in inputs]
    batched = extract(inputs, len(inputs))

    print(f"\n{'row':>6} {'tokens':>7} {'repeat':>10} {'published':>10} {'batched':>10}",
          flush=True)
    stats = {"repeat": [], "published": [], "batched": []}
    for k, i in enumerate(rows):
        pub_full, pub_mask, _ = next(dev_src.slabs([i]))
        keep = pub_mask[0].bool()
        pub = pub_full[0][keep].float()
        n = solo[k].shape[0]
        if pub.shape[0] != n or batched[k].shape[0] != n:
            print(f"{i:>6} SHAPE MISMATCH published={pub.shape[0]} "
                  f"batched={batched[k].shape[0]} solo={n}", flush=True)
            continue
        vals = {
            "repeat": rel_l2(solo[k], solo2[k]),
            "published": rel_l2(solo[k], pub),
            "batched": rel_l2(solo[k], batched[k]),
        }
        for key, v in vals.items():
            stats[key].append(v)
        print(f"{i:>6} {n:>7} {vals['repeat']:>10.2e} {vals['published']:>10.2e} "
              f"{vals['batched']:>10.2e}", flush=True)

    print("\nmedian relative L2 error:", flush=True)
    for key, vs in stats.items():
        if vs:
            print(f"  {key:<10} {np.median(vs):.3e}", flush=True)
    if stats["batched"]:
        print(
            "\nreading: `repeat` and `published` at 0 mean extraction is bit-exact and "
            "machine-independent at batch size 1 — there is no drift to hide behind, so the "
            "`batched` figure is the whole effect of raising BATCH_SIZE. This is why the "
            "red-team cache is built one row at a time.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
