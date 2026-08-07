"""Stage 3: act on the leave-one-pair-out result and check that acting on it helps.

A ranked list of per-pair deltas is a hypothesis, not a finding. This drops the
flagged sets for real, retrains, and measures whether the eval AUROC actually moves
— which is the only statement worth making after a sweep whose per-pair effects sit
close to the noise.

Four things make the check honest rather than self-confirming:

**Two flagging rules, not one.** ``2 SE`` is the loose reading; Benjamini-Hochberg at
q=0.10 over all pairs x 4 splits is the multiplicity-corrected one. With 389 pairs and
four splits, roughly 5% of pairs clear a 2-SE bar by chance alone, so the loose set is
expected to be mostly false positives — reporting both is what shows that.

**A random control set.** A same-sized set of randomly chosen pairs is dropped too.
If dropping 40 flagged pairs helps exactly as much as dropping 40 arbitrary ones, the
flagging carried no information and the effect was just "less training data" (or
nothing). Without this column the other numbers cannot be interpreted.

**The same pairing as the sweep.** Every variant is a column in one pass, sharing the
initialisation and the shuffle stream with the baseline column, so each seed yields a
paired difference and the seed spread (sd up to 0.023) cancels instead of swamping.

**Both metric scales.** ``pipeline`` reproduces the bf16-saturated probabilities the
comparison CSVs are built from; ``rank`` is the tie-free AUROC of the logits.

Usage:
    .venv_claude/bin/python scripts/attribution_verify.py --arm gptoss120b --seeds 50
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import attribution_fasttrain as F  # noqa: E402
import attribution_lib as A  # noqa: E402
import attribution_pack as P  # noqa: E402


def _bh_reject(pvals: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of rejections at false-discovery rate ``q``."""
    n = pvals.size
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = pvals[order] <= thresh
    out = np.zeros(n, dtype=bool)
    if passed.any():
        out[order[: np.flatnonzero(passed)[-1] + 1]] = True
    return out


def flag_sets(loo_path: Path, q: float = 0.10, rng_seed: int = 0):
    """Turn the LOO cube into the drop-sets to verify, plus a size-matched control."""
    from scipy import stats

    d = np.load(loo_path, allow_pickle=True)
    deltas = d["deltas"][..., 0]              # pipeline scale
    splits = [str(s) for s in d["splits"]]
    keep = [i for i, s in enumerate(splits) if s != "mean"]
    n_pairs, _, n_seeds = deltas.shape

    mean = np.nanmean(deltas[:, keep, :], axis=2)
    se = np.nanstd(deltas[:, keep, :], axis=2, ddof=1) / np.sqrt(n_seeds)
    # Floor rather than mask: se == 0 means every seed returned the identical delta.
    # For a pair that changes nothing that delta is 0 and the floor keeps t at 0
    # (correctly inert); for one with a consistent non-zero effect it is the most
    # significant case there is, and masking it to NaN would have filed it as inert.
    t = mean / np.maximum(se, 1e-12)
    # One-sided: does REMOVING the pair raise this split's AUROC?
    p_up = stats.t.sf(t, df=n_seeds - 1)

    harmful_2se = np.nanmax(np.where(t > 2, 1.0, -1.0), axis=1) > 0
    useful_2se = np.nanmin(np.where(t < -2, -1.0, 1.0), axis=1) < 0
    inert = ~harmful_2se & ~useful_2se

    flat = np.nan_to_num(p_up, nan=1.0).ravel()
    harmful_bh = _bh_reject(flat, q).reshape(p_up.shape).any(axis=1)

    rng = np.random.default_rng(rng_seed)
    control = np.zeros(n_pairs, dtype=bool)
    n_ctrl = int(harmful_2se.sum())
    if n_ctrl:
        control[rng.choice(n_pairs, size=n_ctrl, replace=False)] = True

    return {
        "n_pairs": n_pairs,
        "n_seeds": int(n_seeds),
        "splits": [splits[i] for i in keep],
        "mean": mean,
        "se": se,
        "sets": {
            "drop_harmful_bh": harmful_bh,
            "drop_harmful_2se": harmful_2se,
            "drop_inert": inert,
            "drop_harmful2se_and_inert": harmful_2se | inert,
            "drop_random_control": control,
        },
        "useful_2se": useful_2se,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(A.ARMS), required=True)
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=A.REPO / "results_hu_harm_gemma27b_batch_ablation/attribution",
    )
    args = ap.parse_args()

    loo_path = args.out_dir / f"{args.arm}_iter{args.iteration}_loo.npz"
    if not loo_path.exists():
        raise SystemExit(f"no LOO result at {loo_path} — run attribution_loo.py first")

    info = flag_sets(loo_path, q=args.fdr)
    names = list(info["sets"])
    print(f"{args.arm}: {info['n_pairs']} pairs, LOO used {info['n_seeds']} seeds")
    for name in names:
        print(f"  {name:28s} {int(info['sets'][name].sum()):4d} pairs")
    print(f"  {'(pairs flagged as useful)':28s} {int(info['useful_2se'].sum()):4d} pairs")

    train, val, ev, meta = P.build(args.arm, args.iteration)
    pairs = meta["pairs"]
    k = len(names) + 1

    results = {n: [] for n in names}
    base_abs = []
    for si in range(args.seeds):
        seed = A.SEED + si
        keep = torch.ones((train.n, k), dtype=torch.bool)
        vkeep = torch.ones((val.n, k), dtype=torch.bool)
        for j, name in enumerate(names, start=1):
            for pid in np.flatnonzero(info["sets"][name]):
                for r in pairs[pid].packed_train_rows:
                    keep[r, j] = False
                for r in pairs[pid].packed_val_rows:
                    vkeep[r, j] = False

        W, b, _, _ = F.train_many(
            train, val, keep, seed=seed, val_keep_mask=vkeep, shared_init=True
        )
        res = F.score_eval(ev, W, b)
        base_abs.append({s: {sc: float(res[s][sc][0]) for sc in ("pipeline", "rank")}
                         for s in A.EVAL_SPLITS + ["mean"]})
        for j, name in enumerate(names, start=1):
            results[name].append(
                {s: {sc: float(res[s][sc][j] - res[s][sc][0])
                     for sc in ("pipeline", "rank")}
                 for s in A.EVAL_SPLITS + ["mean"]}
            )
        print(f"  seed {seed} done ({si + 1}/{args.seeds})", flush=True)

    print(f"\npaired change vs the full-data baseline, pipeline scale, "
          f"mean +/- SE over {args.seeds} seeds")
    header = f"{'variant':28s}" + "".join(f"{s.replace('eval_', ''):>22s}"
                                          for s in A.EVAL_SPLITS + ["mean"])
    print(header)
    for name in names:
        cells = ""
        for s in A.EVAL_SPLITS + ["mean"]:
            v = np.array([r[s]["pipeline"] for r in results[name]])
            m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            star = "*" if abs(m) > 2 * se else " "
            cells += f"{m:+.4f}+-{se:.4f}{star}".rjust(22)
        print(f"{name:28s}{cells}")
    for s in A.EVAL_SPLITS + ["mean"]:
        v = np.array([r[s]["pipeline"] for r in base_abs])
        print(f"  baseline {s:24s} {v.mean():.5f} +- {v.std(ddof=1) / np.sqrt(len(v)):.5f}")
    print("\n* = |mean| > 2 SE. Read drop_random_control FIRST: any variant that does not"
          "\nbeat it by more than its error bar has not demonstrated that the flagging"
          "\nidentified anything — only that the training set changed size.")

    out = args.out_dir / f"{args.arm}_iter{args.iteration}_verify.json"
    out.write_text(json.dumps({
        "arm": args.arm,
        "seeds": args.seeds,
        "fdr_q": args.fdr,
        "set_sizes": {n: int(info["sets"][n].sum()) for n in names},
        "n_useful_2se": int(info["useful_2se"].sum()),
        "pair_ids": {n: np.flatnonzero(info["sets"][n]).tolist() for n in names},
        "baseline_abs": base_abs,
        "deltas": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
