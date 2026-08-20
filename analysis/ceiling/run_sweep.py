"""PART 2 — how much does labelled dev data buy on top of red-teaming?

Sweep N = the number of held-out dev samples the probe is allowed to learn from,
over an equidistant grid from 0 to the size of the dev pool, and measure eval
performance two ways:

  joint     one from-scratch fit on  base ∪ red-team ∪ dev[:N]   (the "add the samples
            into the red-teaming training data and retrain" arm)
  finetune  fit on base ∪ red-team first, then continue training THAT probe on
            dev[:N] alone with `initialize_model=False` (the "red-team first, then
            finetune on dev" arm). Same member seeds, so joint and finetune at a
            given N differ only in how the dev rows were consumed.

Both need a validation set for early stopping, and it must be the SAME one at every
N or the curves are not comparable — a probe that early-stops against a set that
shrinks as N grows is being selected differently at each point. So the 436-row dev
set is cut once, deterministically and stratified by (split, label), into

  DEV_VAL   100 rows — validation only, never trained on, in any condition
  DEV_POOL  336 rows — the sweep's pool; `dev[:N]` means DEV_POOL[order][:N]

and the grid runs 0, 42, ..., 336. `order` is a stratified interleave, so every
prefix is class- and split-balanced: N=42 is 6 rows from each of the 7 splits, half
positive, not 42 rows of whatever happened to sort first.

N=0 is the red-team-only probe and is shared by both arms by construction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import folds as F
import harness as H

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

DEV_VAL_N = 100
GRID_STEP = 42  # 336 / 8 -> 0, 42, ..., 336


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gptoss", choices=list(H.ARMS))
    ap.add_argument("--ensemble", type=int, default=10)
    ap.add_argument("--ways", default="joint,finetune")
    ap.add_argument("--ft-lr", default="default,1e-4",
                    help="comma list; 'default' keeps the probe's own 5e-3")
    ap.add_argument("--iteration", type=int, default=5)
    ap.add_argument("--grid-step", type=int, default=GRID_STEP)
    ap.add_argument("--no-redteam", action="store_true",
                    help="drop the red-team rows entirely: train on base + dev[:N] only. "
                         "Gives the exchange rate between labelled dev rows and the whole "
                         "red-teaming loop, since the N=0 point is then just the iter0 probe.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    H.silence_tqdm()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else OUT / f"sweep_{args.arm}.jsonl"
    log_path = out_path.with_suffix(".log")
    logf = log_path.open("a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["way"], r["n_dev"], r.get("ft_lr")))
        log(f"resuming: {len(done)} cells already recorded in {out_path}")

    t0 = time.time()
    ev = H.load_eval_splits()
    dev = H.load_dev()
    base = H.load_base()
    if args.no_redteam:
        rt_train = base
    else:
        rt = H.load_redteam(args.arm, args.iteration)
        rt_train = H.concat(base, rt)      # stays on the HOST: every fit takes a copy
    n_rt = len(rt_train)
    for d in ev.values():                  # eval is scored ~50 times; park it once
        H.stage(d)
    log(f"loaded eval={sum(len(d) for d in ev.values())} dev={len(dev)} "
        f"redteam-train={n_rt} in {time.time()-t0:.0f}s")

    # ---- cut the dev set once: DEV_VAL (validation, never trained on) vs DEV_POOL
    dev_y = H.labels_of(dev)
    dev_split = F.dev_split_of_row()
    assert len(dev_split) == len(dev), (len(dev_split), len(dev))
    strata = [f"{s}|{y}" for s, y in zip(dev_split, dev_y)]
    order = F.stratified_order(strata, seed=H.SEED)
    val_idx = np.sort(order[:DEV_VAL_N])
    pool_order = [i for i in order if i not in set(val_idx.tolist())]
    log(f"dev cut: val={len(val_idx)} (pos {int(dev_y[val_idx].sum())}) "
        f"pool={len(pool_order)} (pos {int(dev_y[pool_order].sum())})")

    dev_val = H.take(dev, val_idx)
    H.stage(dev_val)

    grid = list(range(0, len(pool_order) + 1, args.grid_step))
    if grid[-1] != len(pool_order):
        grid.append(len(pool_order))
    log(f"grid: {grid}")

    ft_lrs = [None if s == "default" else float(s) for s in args.ft_lr.split(",")]
    ways = args.ways.split(",")
    seeds = list(H.ENSEMBLE_SEEDS[: args.ensemble])

    def emit(way, n, per, ft_lr=None, extra=None):
        rec = {"arm": args.arm, "way": way, "n_dev": n, "ft_lr": ft_lr,
               "no_redteam": bool(args.no_redteam),
               "n_redteam_train": n_rt, "ensemble": args.ensemble,
               "per_split": per, "macro": H.macro(per)} | (extra or {})
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        m = rec["macro"]
        log(f">>> {way:9s} lr={str(ft_lr):7s} N={n:4d}  AUROC {m['auroc']:.4f}  "
            f"acc {m['accuracy']:.4f}  TPR@1%FPR {m['tpr_at_fpr']:.4f} (<=1%: {m['tpr_at_fpr_le']:.4f})")

    def fit_members(train, tag):
        out = []
        for i, s in enumerate(seeds):
            t = time.time()
            with H.Quiet() as q:
                out.append(H.fit_member(train, dev_val, s))
            log(f"    [{tag}] member {i+1}/{len(seeds)} {q.epochs_run()} ep "
                f"{time.time()-t:.0f}s")
        return out

    # ---- N = 0: the red-team-only probe, shared by both arms
    train0 = H.take(rt_train, range(n_rt))
    H.stage(train0)
    log(f"fitting N=0 (red-team only), train={n_rt} | {H.gpu_free_gib():.1f} GiB free")
    base_members = fit_members(train0, "N=0")
    base_probe = H.ensemble_of(base_members, seeds)
    per0 = H.score_splits(base_probe, ev)
    for way in ways:
        for lr in (ft_lrs if way == "finetune" else [None]):
            if (way, 0, lr) not in done:
                emit(way, 0, per0, ft_lr=lr, extra={"note": "red-team only; shared baseline"})
    del train0
    H.free()

    # ---- joint: retrain from scratch on red-team ∪ dev[:N]
    if "joint" in ways:
        for n in grid[1:]:
            if ("joint", n, None) in done:
                continue
            sub = H.take(dev, [pool_order[i] for i in range(n)])
            train = H.concat(H.take(rt_train, range(n_rt)), sub)
            H.stage(train)
            log(f"[joint] N={n}: train={len(train)} | {H.gpu_free_gib():.1f} GiB free")
            ms = fit_members(train, f"joint/N{n}")
            per = H.score_splits(H.ensemble_of(ms, seeds), ev)
            emit("joint", n, per)
            del train, sub, ms
            H.free()

    # ---- finetune: continue the N=0 members on dev[:N] alone
    if "finetune" in ways:
        for lr in ft_lrs:
            for n in grid[1:]:
                if ("finetune", n, lr) in done:
                    continue
                sub = H.take(dev, [pool_order[i] for i in range(n)])
                H.stage(sub)
                log(f"[finetune lr={lr}] N={n} | {H.gpu_free_gib():.1f} GiB free")
                ms = []
                for i, s in enumerate(seeds):
                    t = time.time()
                    with H.Quiet() as q:
                        ms.append(H.finetune_member(base_members[i], sub, dev_val, s, lr=lr))
                    log(f"    [ft/N{n}] member {i+1}/{len(seeds)} {q.epochs_run()} ep "
                        f"{time.time()-t:.0f}s")
                per = H.score_splits(H.ensemble_of(ms, seeds), ev)
                emit("finetune", n, per, ft_lr=lr)
                del sub, ms
                H.free()

    log(f"done in {(time.time()-t0)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
