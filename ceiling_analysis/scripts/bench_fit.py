#!/usr/bin/env python
"""Measure what a probe-head fit actually costs at high-stakes shapes, host vs GPU resident.

The high-stakes training pool at the top of the sweep is ~2.3k rows padded to 1024 tokens —
about 25 GB of fp16 activations, just past a 24 GB card. Whether that matters depends
entirely on how expensive host residency is *on this box*, which is a measurement, not a
guess: `retrain._to_device_for_fit` reports 18.35 ms/sample host-resident vs 0.16
GPU-resident on the box it was tuned on, and the decision of how many draw seeds the
high-stakes sweep can afford follows directly from the number here.
"""

from __future__ import annotations

import argparse
import time

import torch


def bench(n_rows: int, width: int, dim: int, batch: int, device_resident: bool,
          n_batches: int | None = None) -> float:
    from tuberlens.interfaces.activations import ActivationDataset
    from tuberlens.probes.pytorch_modules import LinearThenSoftmax
    from torch.utils.data import DataLoader

    dev = "cuda"
    acts = torch.randn((n_rows, width, dim), dtype=torch.float16)
    mask = torch.ones((n_rows, width), dtype=torch.bool)
    ids = torch.zeros((n_rows, width), dtype=torch.int64)
    y = (torch.rand(n_rows) > 0.5).to(torch.bfloat16)
    if device_resident:
        acts, mask, ids, y = (t.to(dev) for t in (acts, mask, ids, y))
    ds = ActivationDataset(acts, mask, ids, y)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = LinearThenSoftmax(dim, temperature=5).to(dev).to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()

    torch.cuda.synchronize()
    t0 = time.time()
    seen = 0
    for i, (ba, bm, _bi, by) in enumerate(dl):
        out = model(ba, bm)
        loss = crit(out, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        seen += ba.shape[0]
        if n_batches and i + 1 >= n_batches:
            break
    torch.cuda.synchronize()
    dt = time.time() - t0
    del acts, mask, ids, y, ds, dl, model, opt
    torch.cuda.empty_cache()
    return dt / seen * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=5376)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batches", type=int, default=24)
    args = ap.parse_args()
    for resident in (True, False):
        ms = bench(args.rows, args.width, args.dim, args.batch, resident, args.batches)
        where = "gpu" if resident else "host"
        print(f"{where}-resident: {ms:.2f} ms/sample  "
              f"=> {ms * 2323 / 1000:.1f} s/epoch at 2323 rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
