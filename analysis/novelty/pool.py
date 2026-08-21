#!/usr/bin/env python
"""Phase 0 -- reduce every activation blob to one mask-weighted mean vector per row.

    .venv_claude/bin/python analysis/novelty/pool.py --experiment instructions
    .venv_claude/bin/python analysis/novelty/pool.py --experiment highstakes

Writes ``results/pooled_<experiment>.npz`` holding, for each source, a
``(rows, 5376)`` float32 matrix plus its labels:

    eval::<split>     the eval split, file order
    dev               the held-out dev set, sorted-glob concatenation order
    base_train        the 50-row base training split
    rt::<arm>         that arm's final postprocessed red-team snapshot, file order

plus ``rt::<arm>::first_iter`` (which iteration introduced each row) and
``rt::<arm>::hash`` (its activation-cache digest, the row's stable identity).

Mean-pooling is a *proxy* for what the probe does -- ``linear_then_softmax`` pools with
a softmax over per-token linear scores, not a flat mean. The proxy is deliberate: the
novelty scores have to be probe-independent or the whole analysis is circular (a point
would be "novel" partly because the probe already mishandles it). The probe's own view
is brought in separately, and explicitly, by ``novelty.py``'s direction decomposition.

High-stakes reads 46 GiB of eval blobs and a 19.6 GiB dev blob; both stream in row
chunks, so peak resident memory is a chunk, not a split.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments as X  # noqa: E402
import loaders as L  # noqa: E402


def pool_experiment(exp: X.Experiment, chunk: int, verbose: bool = True) -> dict:
    out: dict[str, np.ndarray] = {}
    t0 = time.monotonic()

    for split in exp.splits():
        blob = L.eval_blob(exp, split)
        vecs = L.pool_blob(blob, chunk=chunk)
        labels = L.jsonl_labels(exp, exp.eval_dir / f"{split}.jsonl")
        if len(labels) != len(vecs):
            raise ValueError(f"eval/{split}: {len(vecs)} activation rows vs {len(labels)} labels")
        out[f"eval::{split}"] = vecs
        out[f"eval::{split}::labels"] = labels
        if verbose:
            print(f"  eval::{split:32s} {vecs.shape[0]:5d} rows  ({time.monotonic() - t0:5.0f}s)", flush=True)

    from agentic_redteam.retrain import _load_dev_dataset

    ds, _files = _load_dev_dataset(exp.dev_dir, exp.pos, exp.neg, X.COMBINE, X.CONVERT, verbose=False)
    out["dev"] = L.pool_blob(L.dev_blob(exp), chunk=chunk)
    out["dev::labels"] = L.labels_of(ds)
    # Which dev split each row came from: _load_dev_dataset concatenates sorted(glob).
    split_of = []
    for f in sorted(exp.dev_dir.glob("*.jsonl")):
        split_of.extend([f.stem] * len(L.load_jsonl_dataset(f, exp)))
    out["dev::split"] = np.array(split_of)
    if len(out["dev"]) != len(out["dev::labels"]):
        raise ValueError("dev: activation/label row mismatch")
    if verbose:
        print(f"  dev{'':35s} {out['dev'].shape[0]:5d} rows  ({time.monotonic() - t0:5.0f}s)", flush=True)

    base = L.load_jsonl_dataset(exp.base_data, exp)
    out["base_train"] = L.pool_blob(L.base_blob(exp), chunk=chunk)
    out["base_train::labels"] = L.labels_of(base)
    if verbose:
        print(f"  base_train{'':28s} {out['base_train'].shape[0]:5d} rows", flush=True)

    for arm_key, arm in exp.arms.items():
        ds = L.redteam_dataset(exp, arm)
        paths = L.redteam_paths(exp, ds)
        out[f"rt::{arm_key}"] = L.pool_rows(paths)
        out[f"rt::{arm_key}::labels"] = L.labels_of(ds)
        prov = L.redteam_provenance(exp, arm)
        out[f"rt::{arm_key}::first_iter"] = np.array(prov["first_iter"])
        out[f"rt::{arm_key}::hash"] = np.array(prov["hash"])
        if verbose:
            iters = np.array(prov["first_iter"])
            per = ", ".join(f"it{i}:{int((iters == i).sum())}" for i in sorted(set(iters.tolist())))
            print(f"  rt::{arm_key:31s} {out[f'rt::{arm_key}'].shape[0]:5d} rows  ({per})", flush=True)

    if verbose:
        print(f"  pooled in {time.monotonic() - t0:.0f}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=sorted(X.EXPERIMENTS))
    ap.add_argument("--chunk", type=int, default=32, help="rows per mmap read (lower = less RAM)")
    ap.add_argument("--refresh", action="store_true", help="recompute even if the npz exists")
    args = ap.parse_args()

    exp = X.get(args.experiment)
    X.RESULTS.mkdir(parents=True, exist_ok=True)
    dest = X.RESULTS / f"pooled_{exp.key}.npz"
    if dest.exists() and not args.refresh:
        print(f"{dest} exists; pass --refresh to recompute")
        return 0

    print(f"=== pooling {exp.key} (chunk={args.chunk}) ===", flush=True)
    data = pool_experiment(exp, args.chunk)
    np.savez_compressed(dest, **data)
    print(f"Saved -> {dest} ({dest.stat().st_size / 2**20:.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
