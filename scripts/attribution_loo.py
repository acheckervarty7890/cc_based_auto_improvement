"""Leave-one-pair-out over an arm's iter3 red-team set, paired and seed-averaged.

For every (source, contrastive) pair this measures the change in each eval split's
AUROC when that pair is removed from the training data — the literal question, done
by real refits rather than any surrogate.

Two design points carry the whole thing:

**Pairing.** The seed-to-seed spread of this probe is enormous (sd 0.023 on
balanced_refusal, range 0.078), because model selection separates neighbouring epochs
by one or two validation pairs out of 6624. A single pair out of 389 cannot be seen
against that. So every pass trains a baseline column and up to K-1 drop columns
*together*, from one shared initialisation and one shared shuffle stream, and the
statistic is the paired difference. Only the drop-set differs between columns.

**Averaging.** Even paired, a removal can flip the epoch at which early stopping
fires, which is discontinuous. Repeating over ``--seeds`` independent seeds and
reporting mean +/- standard error over those paired differences is what turns this
into something with an error bar rather than an anecdote.

Cost: at K=64 a model costs ~0.5 s, so 389 pairs x 20 seeds is about an hour per arm.

**Resumable at pass granularity.** Every finished pass appends a row to
``<arm>_iter{N}_loo_progress.jsonl`` and a restart skips the passes already in it, so
an interrupted run resumes rather than starting over. This is possible because each
pass is *self-contained*: it trains its own baseline column from the same shared
initialisation and shuffle as its drop columns, and the reported delta is against
that column — not against a baseline recorded once per seed. So a pass can be
computed at any time, in any order, and still be paired correctly.

Usage:
    .venv_claude/bin/python scripts/attribution_loo.py --arm gptoss120b --seeds 20
    # after an interruption, the identical command resumes; --no-resume starts over
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import attribution_fasttrain as F  # noqa: E402
import attribution_lib as A  # noqa: E402
import attribution_pack as P  # noqa: E402


def _append_progress(path: Path, row: dict) -> None:
    """Append one finished pass. Flushed immediately — the point is surviving a kill."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_progress(path: Path, shape_key: dict, deltas, baselines, splits,
                   resume: bool) -> set:
    """Refill ``deltas``/``baselines`` from the sidecar; return the done pass keys.

    Rows whose chunking differs from this run's are skipped rather than trusted: the
    pass index only identifies a set of pairs relative to a given ``--k``. A torn
    final line (killed mid-write) is skipped the same way, so an unclean stop costs
    one pass, not the file.
    """
    done: set = set()
    if not resume or not path.exists():
        return done
    n_skipped = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            if any(row.get(k) != v for k, v in shape_key.items()):
                n_skipped += 1
                continue
            si = row["seed"] - A.SEED
            if not 0 <= si < deltas.shape[2]:
                continue  # a seed outside this run's range
            baselines[si] = np.asarray(row["baseline"])
            rd = np.asarray(row["deltas"])
            for j, pid in enumerate(row["pair_ids"]):
                deltas[pid, :, si, :] = rd[j]
            done.add((row["seed"], row["pass"]))
    if n_skipped:
        print(f"  ({n_skipped} sidecar row(s) ignored: different chunking or truncated)")
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(A.ARMS), required=True)
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--k", type=int, default=64, help="columns per pass (1 is baseline)")
    ap.add_argument("--limit-pairs", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse finished passes from the progress sidecar (default on)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=A.REPO / "results_hu_harm_gemma27b_batch_ablation/attribution",
    )
    args = ap.parse_args()

    train, val, ev, meta = P.build(args.arm, args.iteration)
    pairs = meta["pairs"]
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    n_pairs = len(pairs)
    splits = A.EVAL_SPLITS + ["mean"]

    # deltas[pair, split, seed, scale]; NaN until filled
    deltas = np.full((n_pairs, len(splits), args.seeds, 2), np.nan, dtype=np.float64)
    baselines = np.full((args.seeds, len(splits), 2), np.nan, dtype=np.float64)

    per_pass = args.k - 1
    n_pass = int(np.ceil(n_pairs / per_pass))
    print(f"{args.arm}: {n_pairs} pairs, {args.seeds} seeds, K={args.k} "
          f"-> {n_pass} passes/seed, {n_pass * args.seeds} passes total", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / f"{args.arm}_iter{args.iteration}_loo_progress.jsonl"
    # The layout of a pass depends on how the pairs were chunked, so rows written
    # under a different --k (or a different pair count) describe different work and
    # must not be reused.
    shape_key = {"per_pass": per_pass, "n_pairs": n_pairs, "iteration": args.iteration}
    done = _load_progress(
        progress_path, shape_key, deltas, baselines, splits, args.resume
    )
    if done:
        print(f"resuming: {len(done)} of {n_pass * args.seeds} passes already done",
              flush=True)

    t_start = time.time()
    n_run = 0
    for si in range(args.seeds):
        seed = A.SEED + si
        for pi in range(n_pass):
            if (seed, pi) in done:
                continue
            chunk = pairs[pi * per_pass : (pi + 1) * per_pass]
            k = len(chunk) + 1
            keep = torch.ones((train.n, k), dtype=torch.bool)
            vkeep = torch.ones((val.n, k), dtype=torch.bool)
            for j, pair in enumerate(chunk, start=1):
                for r in pair.packed_train_rows:
                    keep[r, j] = False
                for r in pair.packed_val_rows:
                    vkeep[r, j] = False

            W, b, _, _ = F.train_many(
                train, val, keep, seed=seed, val_keep_mask=vkeep, shared_init=True
            )
            res = F.score_eval(ev, W, b)
            row_base = np.empty((len(splits), 2))
            row_deltas = np.empty((len(chunk), len(splits), 2))
            for scale_i, scale in enumerate(("pipeline", "rank")):
                for split_i, split in enumerate(splits):
                    col = res[split][scale]
                    # This pass's OWN baseline column — same init, same shuffle, no
                    # drop — so the pass is self-contained and resumable in any order.
                    base = float(col[0])
                    row_base[split_i, scale_i] = base
                    baselines[si, split_i, scale_i] = base
                    for j, pair in enumerate(chunk, start=1):
                        d = float(col[j]) - base
                        deltas[pair.pair_id, split_i, si, scale_i] = d
                        row_deltas[j - 1, split_i, scale_i] = d

            _append_progress(
                progress_path,
                {
                    **shape_key,
                    "seed": seed,
                    "pass": pi,
                    "pair_ids": [p.pair_id for p in chunk],
                    "baseline": row_base.tolist(),
                    "deltas": row_deltas.tolist(),
                },
            )
            done.add((seed, pi))
            n_run += 1
            el = time.time() - t_start
            left = n_pass * args.seeds - len(done)
            print(f"  seed {seed} pass {pi + 1}/{n_pass}  "
                  f"({el / n_run:.0f}s/pass, {el / 60:.0f}m elapsed, "
                  f"{left * el / n_run / 60:.0f}m left)", flush=True)

    out = args.out_dir / f"{args.arm}_iter{args.iteration}_loo.npz"
    np.savez_compressed(
        out,
        deltas=deltas,
        baselines=baselines,
        splits=np.array(splits),
        scales=np.array(["pipeline", "rank"]),
        pair_source_idx=np.array([p.source_idx for p in pairs]),
        pair_generated_idx=np.array([p.generated_idx for p in pairs]),
        n_train_rows=np.array([len(p.train_rows) for p in pairs]),
        n_val_rows=np.array([len(p.val_rows) for p in pairs]),
        source_label=np.array([p.source_label for p in pairs]),
    )
    print(f"\nwrote {out}")

    # A quick read of the result: how many pairs move any split beyond its own noise?
    mean = np.nanmean(deltas[..., 0], axis=2)
    se = np.nanstd(deltas[..., 0], axis=2, ddof=1) / np.sqrt(args.seeds)
    print(f"\n{'split':22s} {'|mean delta| median':>19s} {'max':>9s} "
          f"{'signif +':>9s} {'signif -':>9s}")
    for i, split in enumerate(splits):
        sig = np.abs(mean[:, i]) > 2 * se[:, i]
        print(f"{split:22s} {np.median(np.abs(mean[:, i])):19.5f} "
              f"{np.max(np.abs(mean[:, i])):9.5f} "
              f"{int((sig & (mean[:, i] > 0)).sum()):9d} "
              f"{int((sig & (mean[:, i] < 0)).sum()):9d}")
    print("\n'signif +' = removing the pair RAISES that split's AUROC (pair is harmful);"
          "\n'signif -' = removing it lowers AUROC (pair is doing useful work)."
          "\nThreshold is |mean| > 2 x standard error over seeds — see the caveat in the"
          "\nwrite-up about how many of these survive a multiplicity correction.")


if __name__ == "__main__":
    main()
