#!/usr/bin/env python
"""Where do the red-team samples sit, in activation space, relative to eval and dev?

The question this answers: each iteration's red-team set is written by an attacker aiming
at the probe's blind spots, not sampled from the eval distribution. So do those samples
drift away from the eval/dev manifold, and does each iteration drift FURTHER than the last?
If they do, a probe retrained on them is being pulled toward a region the eval set never
visits — which is one mechanism behind an eval score that rises for two iterations and then
falls.

REPRESENTATION. Every set is reduced to one vector per conversation: the mask-weighted mean
of the layer-32 activations over the conversation's REAL tokens (padding excluded via
attention_mask). The probe is per-token with a masked aggregation, so a masked mean is the
summary closest to what the head actually consumes, and it makes sets of different padded
widths directly comparable — which matters because red-team blobs are stored at their own
width while eval/dev blobs are padded to 1024.

WHAT IS COMPARED. Per iteration, only the rows NEW in that iteration (the postprocessed
dumps are cumulative, so iteration k's contribution is dump_k minus dump_{k-1}). Against:
eval (4408 rows), dev (1908) and the base training set.

METRICS, and why each is here:
  centroid cosine        - crude but interpretable: is the set centred somewhere else?
  kNN distance           - per-sample: how far is a red-team row from its nearest eval rows?
                           A set can share a centroid and still live in a different region.
  outside-manifold rate  - share of red-team rows whose nearest-eval-neighbour distance
                           exceeds the 95th percentile of eval's own nearest-neighbour
                           distances. Calibrated against how spread eval already is.
  MMD^2 (RBF)            - a proper two-sample statistic; median-heuristic bandwidth.
  dispersion             - mean pairwise cosine distance WITHIN the set: are later
                           iterations tighter (mode collapse) or wider (more diverse)?
  drift                  - centroid cosine distance from iteration 1's centroid, i.e. is
                           each iteration moving further in a consistent direction?

Pooled vectors are cached in --out-dir/pooled.npz, so a re-run re-does only the metrics and
a crashed run resumes without re-reading ~100 GB of blobs.

    .venv_claude/bin/python scripts/analyze_redteam_activation_space.py \
        --probe-dir probes/hs_gemma27b_gptoss120b_ens3 \
        --cache-dir results_hs_gemma27b_devval/base_activations \
        --eval-cache-dir results_hs_gemma27b_devval/eval_activations \
        --out-dir analysis/redteam_space_ens3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _pool(blob_path: Path, chunk: int = 64, sample: int | None = None,
          seed: int = 0) -> np.ndarray:
    """Mask-weighted mean over real tokens -> (rows, hidden) float32, read in chunks.

    ``sample`` reads only that many randomly chosen rows. The blob is opened with
    mmap, so an unread row is never faulted in from disk — sampling 500 of 2984 rows
    costs ~1/6th of the I/O, which is what makes a whole-experiment pass take minutes
    instead of tens of minutes. The draw is seeded, so a re-run pools the same rows.
    """
    d = torch.load(blob_path, map_location="cpu", mmap=True, weights_only=False)
    acts, mask = d["activations"], d["attention_mask"]
    idx = np.arange(acts.shape[0])
    if sample is not None and sample < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(len(idx), sample, replace=False))
    out = np.empty((len(idx), acts.shape[2]), dtype=np.float32)
    for i in range(0, len(idx), chunk):
        rows = torch.from_numpy(idx[i : i + chunk])
        a = acts[rows].to(torch.float32)
        m = mask[rows].to(torch.float32).unsqueeze(-1)
        out[i : i + chunk] = ((a * m).sum(1) / m.sum(1).clamp(min=1)).numpy()
    return out


def _pool_redteam(rows, cache_dir: Path, model_name: str, layer: int,
                  combine: bool, convert: bool, sample: int | None = None,
                  seed: int = 0) -> tuple[np.ndarray, int]:
    """Pool one red-team dump by looking each conversation up in the per-sample cache."""
    from tuberlens.interfaces.dataset import Message

    from agentic_redteam.retrain import _redteam_activation_cache_path

    if sample is not None and sample < len(rows):
        pick = np.sort(np.random.default_rng(seed).choice(len(rows), sample, replace=False))
        rows = [rows[i] for i in pick]
    vecs, missing = [], 0
    for row in rows:
        msgs = [Message(role=m["role"], content=m["content"]) for m in row["messages"]]
        p = _redteam_activation_cache_path(cache_dir, msgs, model_name, layer, combine, convert)
        if not p.exists():
            missing += 1
            continue
        vecs.append(_pool(p)[0])
    return (np.stack(vecs) if vecs else np.empty((0, 0), np.float32)), missing


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine DISTANCE between two already-normalised (1, d) centroids, as a scalar.

    The dot of two (1, d) arrays is (1, 1), and numpy >= 1.25 refuses float() on any
    array with ndim > 0, so squeeze explicitly instead of leaning on a deprecation.
    """
    return float(np.asarray(1.0 - a @ b.T).reshape(-1)[0])


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)


def _knn_dist(a: np.ndarray, b: np.ndarray, k: int = 10, block: int = 512) -> np.ndarray:
    """Mean cosine distance from each row of a to its k nearest rows of b."""
    an, bn = _norm(a), _norm(b)
    out = np.empty(len(an), dtype=np.float32)
    for i in range(0, len(an), block):
        sims = an[i : i + block] @ bn.T
        k_eff = min(k, sims.shape[1])
        top = np.partition(sims, -k_eff, axis=1)[:, -k_eff:]
        out[i : i + block] = 1.0 - top.mean(1)
    return out


def _self_knn_dist(a: np.ndarray, k: int = 10) -> np.ndarray:
    an = _norm(a)
    sims = an @ an.T
    np.fill_diagonal(sims, -np.inf)
    k_eff = min(k, sims.shape[1] - 1)
    return 1.0 - np.partition(sims, -k_eff, axis=1)[:, -k_eff:].mean(1)


def _mmd2(x: np.ndarray, y: np.ndarray, cap: int = 1500, seed: int = 0) -> float:
    """Unbiased MMD^2 with an RBF kernel, median-heuristic bandwidth, subsampled."""
    rng = np.random.default_rng(seed)
    if len(x) > cap:
        x = x[rng.choice(len(x), cap, replace=False)]
    if len(y) > cap:
        y = y[rng.choice(len(y), cap, replace=False)]
    z = np.vstack([x, y])
    d2 = np.maximum(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1), 0) if len(z) < 900 else None
    if d2 is None:  # memory-safe path
        sq = (z**2).sum(1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0)
    med = np.median(d2[d2 > 0]) or 1.0
    kmat = np.exp(-d2 / med)
    n, m = len(x), len(y)
    kxx = (kmat[:n, :n].sum() - np.trace(kmat[:n, :n])) / (n * (n - 1))
    kyy = (kmat[n:, n:].sum() - np.trace(kmat[n:, n:])) / (m * (m - 1))
    kxy = kmat[:n, n:].mean()
    return float(kxx + kyy - 2 * kxy)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-dir", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path, help="base_activations dir (holds redteam_acts_*/ and the dev blob)")
    ap.add_argument("--eval-cache-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model-name", default="google/gemma-3-27b-it")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--combine-consecutive-messages", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--convert-tool-to-assistant", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--refresh", action="store_true", help="recompute pooled vectors even if cached")
    ap.add_argument("--sample-eval", type=int, default=None, help="rows to sample per EVAL split (default: all)")
    ap.add_argument("--sample-dev", type=int, default=None, help="rows to sample from the dev set (default: all)")
    ap.add_argument("--sample-rt", type=int, default=None, help="rows to sample per red-team iteration (default: all)")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pooled_path = args.out_dir / "pooled.npz"
    t0 = time.time()

    if pooled_path.exists() and not args.refresh:
        print(f"Loading cached pooled vectors from {pooled_path}")
        store = {k: v for k, v in np.load(pooled_path).items()}
    else:
        store: dict[str, np.ndarray] = {}
        for blob in sorted(args.eval_cache_dir.glob("*-acts_full.pt")):
            name = blob.name.replace("-acts_full.pt", "")
            store[f"eval::{name}"] = _pool(blob, sample=args.sample_eval)
            print(f"  pooled eval {name}: {store[f'eval::{name}'].shape} ({time.time()-t0:.0f}s)", flush=True)
        dev = sorted(args.cache_dir.glob("dev_acts_*.pt"))
        if dev:
            store["dev"] = _pool(dev[0], sample=args.sample_dev)
            print(f"  pooled dev: {store['dev'].shape} ({time.time()-t0:.0f}s)", flush=True)
        base = sorted(args.cache_dir.glob("base_acts_*_train.pt"))
        if base:
            store["base_train"] = _pool(base[0])
            print(f"  pooled base_train: {store['base_train'].shape}", flush=True)

        # NOTE: pass the PARENT dir — _redteam_activation_cache_path appends
        # redteam_acts_<model>_L<layer>/ itself. Handing it the subdir doubles the path.
        rt_cache = args.cache_dir
        seen: set[str] = set()
        for dump in sorted(args.probe_dir.glob("redteam_postprocessed_iter*.jsonl"),
                           key=lambda p: int("".join(c for c in p.stem if c.isdigit()))):
            it = int("".join(c for c in dump.stem if c.isdigit()))
            rows = []
            for line in dump.open():
                if not line.strip():
                    continue
                rec = json.loads(line)
                msgs = rec["inputs"]
                if isinstance(msgs, str):
                    try:
                        msgs = json.loads(msgs)
                    except json.JSONDecodeError:
                        import ast
                        msgs = ast.literal_eval(msgs)
                key = json.dumps(msgs, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"messages": msgs})
            vecs, missing = _pool_redteam(rows, rt_cache, args.model_name, args.layer,
                                          args.combine_consecutive_messages,
                                          args.convert_tool_to_assistant,
                                          sample=args.sample_rt)
            store[f"rt::iter{it}"] = vecs
            print(f"  pooled red-team iter{it}: {len(rows)} new rows, {vecs.shape} "
                  f"({missing} missing from cache) ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(pooled_path, **store)
        print(f"Saved pooled vectors -> {pooled_path} ({time.time()-t0:.0f}s)")

    ev = np.vstack([v for k, v in store.items() if k.startswith("eval::")])
    dv = store.get("dev")
    rt_keys = sorted([k for k in store if k.startswith("rt::")], key=lambda k: int(k.split("iter")[1]))

    ev_c, dv_c = _norm(ev.mean(0, keepdims=True)), (_norm(dv.mean(0, keepdims=True)) if dv is not None else None)
    ev_self = _self_knn_dist(ev[np.random.default_rng(0).choice(len(ev), min(2000, len(ev)), replace=False)])
    thresh = float(np.quantile(ev_self, 0.95))
    print(f"\neval self-kNN cosine distance: mean {ev_self.mean():.4f}, p95 {thresh:.4f} "
          f"(rows beyond p95 count as outside the eval manifold)\n")

    first_c = None
    print(f"{'set':<12}{'n':>6}{'cos->eval':>11}{'cos->dev':>10}{'kNN->eval':>11}{'outside%':>10}"
          f"{'MMD2 eval':>11}{'dispersion':>12}{'drift(it1)':>12}")
    for key in rt_keys:
        x = store[key]
        if x.size == 0:
            continue
        c = _norm(x.mean(0, keepdims=True))
        if first_c is None:
            first_c = c
        knn = _knn_dist(x, ev)
        row = (f"{key.replace('rt::',''):<12}{len(x):>6}"
               f"{_cos(c, ev_c):>11.4f}"
               f"{(_cos(c, dv_c) if dv_c is not None else float('nan')):>10.4f}"
               f"{knn.mean():>11.4f}{100*float((knn > thresh).mean()):>10.1f}"
               f"{_mmd2(x, ev):>11.4f}{_self_knn_dist(x).mean():>12.4f}"
               f"{_cos(c, first_c):>12.4f}")
        print(row, flush=True)

    if dv is not None:
        knn = _knn_dist(dv, ev)
        print(f"{'dev (ref)':<12}{len(dv):>6}{_cos(dv_c, ev_c):>11.4f}{0.0:>10.4f}"
              f"{knn.mean():>11.4f}{100*float((knn > thresh).mean()):>10.1f}"
              f"{_mmd2(dv, ev):>11.4f}{_self_knn_dist(dv[:1500]).mean():>12.4f}{float('nan'):>12.4f}")
    if "base_train" in store:
        b = store["base_train"]
        knn = _knn_dist(b, ev)
        bc = _norm(b.mean(0, keepdims=True))
        print(f"{'base (ref)':<12}{len(b):>6}{_cos(bc, ev_c):>11.4f}"
              f"{(_cos(bc, dv_c) if dv_c is not None else float('nan')):>10.4f}"
              f"{knn.mean():>11.4f}{100*float((knn > thresh).mean()):>10.1f}"
              f"{_mmd2(b, ev):>11.4f}{_self_knn_dist(b).mean():>12.4f}{float('nan'):>12.4f}")

    print(f"\nDone in {time.time()-t0:.0f}s. Pooled vectors: {pooled_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
