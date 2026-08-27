#!/usr/bin/env python
"""Weighted-mean fusion of layer probes: sweep the weights, for every subset containing L32.

`scripts/probe_agreement.py` showed the top-3 layer probes rank differently (pairwise
Spearman 0.36-0.53) yet no equal-weight combiner beat L32 alone. Equal weights are a strong
assumption when the components span 0.61-0.71 AUROC, so this sweeps the weights instead:
every subset of the extracted layers that CONTAINS L32, over a simplex grid.

**Fusion is done in LOGIT space by default.** The probes saturate -- 77% of L32's scores sit
below 0.01 -- so a weighted mean of raw probabilities is dominated by whichever component is
least compressed, and the weight then controls something other than influence. `--space
prob` runs the raw-probability version for comparison.

**Three numbers per subset, and only two of them are honest:**

* ``best_eval``   -- the best AUROC any weight vector achieves ON THE EVAL SPLIT. This is
  selection on the test set and is reported as a CEILING, never as an achievable score. Its
  purpose is to bound the whole family: if the ceiling does not clear the best single probe,
  no weighting scheme can, and there is nothing left to try.
* ``heldout``     -- weights chosen on the 32 `oig_omission` dev rows this protocol neither
  trains nor validates on, then applied to eval. Clean, in-distribution, but 32 rows is a
  noisy basis for choosing weights and the variance shows.
* ``val``         -- weights chosen on the 404-row validation set. Larger, but early stopping
  saw those rows, so it is mildly optimistic rather than fully independent.

Weights are non-negative and sum to 1 -- a weighted *mean*, as asked. Signed weights are the
unconstrained linear stack, which `probe_agreement` already bounded with a
leave-one-out logistic regression fitted on the eval labels.

    .venv_claude/bin/python scripts/weighted_combine_sweep.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# The logit floor. This MUST sit below the smallest positive probability any probe emits,
# not at a "small-looking" round number: the `linear_then_last` heads saturate to 1e-28 and
# below, so a 1e-6 floor tied 92 of 114 eval scores together and read that probe as 0.6631
# when its true AUROC is 0.7867. AUROC is rank-based, so any clipping that merges distinct
# scores silently rewrites the metric. float64's tiny (~2.2e-308) is below anything a
# sigmoid can produce while keeping log() finite; exact zeros clip to it and stay tied with
# each other, which is faithful -- they were already tied in the raw scores.
# `_assert_rank_preserving` checks this held, per arm, rather than trusting the constant.
# The bounds are ASYMMETRIC and must be: `1 - tiny` rounds to exactly 1.0 in float64
# (tiny is ~1e-292 below the spacing near 1), so a symmetric clip yields log(1/0) = inf.
# The upper bound is instead the largest double strictly below 1.
EPS_LO = np.finfo(float).tiny
EPS_HI = np.nextafter(1.0, 0.0)


def _simplex(k: int, step: float):
    """All non-negative weight vectors of length k summing to 1, on a grid of `step`."""
    n = int(round(1.0 / step))
    def rec(k_left, total):
        if k_left == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for rest in rec(k_left - 1, total - i):
                yield (i,) + rest
    for w in rec(k, n):
        yield tuple(x / n for x in w)


def _grid_step(k: int) -> float:
    return {1: 1.0, 2: 0.01, 3: 0.05, 4: 0.1, 5: 0.125, 6: 0.2}[k]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/probe_agreement_all6.json")
    ap.add_argument("--anchor", default="L32", help="Every subset must contain this arm.")
    ap.add_argument("--space", choices=["logit", "prob"], default="logit")
    ap.add_argument("--max-subset-size", type=int, default=0,
                    help="Cap on subset size including the anchor (0 = no cap). Required "
                         "once the arm count is large: enumerating every subset of 66 arms "
                         "is 2^65. Size 2 (the anchor paired with each other arm) is where "
                         "the signal is legible anyway.")
    ap.add_argument("--top-partners", type=int, default=8,
                    help="For subsets of size >= 3, draw partners only from the N arms that "
                         "did best paired with the anchor. Enumerating all triples over 65 "
                         "partners is 2080 subsets x a simplex grid each; shortlisting on "
                         "the pair result keeps it tractable and loses only combinations "
                         "whose members were individually useless with the anchor.")
    ap.add_argument("--top-rows", type=int, default=0,
                    help="Print only the N best rows (0 = all). The JSON always holds all.")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/weighted_sweep.json")
    args = ap.parse_args(argv)

    from sklearn.metrics import roc_auc_score

    d = json.loads(args.scores.read_text())
    names = d["arms"]
    if args.anchor not in names:
        raise SystemExit(f"{args.anchor} not among {names}")

    def prep(scores):
        p = np.asarray(scores, dtype=float)
        if args.space == "prob":
            return p
        p = np.clip(p, EPS_LO, EPS_HI)
        # log(p) - log1p(-p) rather than log(p/(1-p)): the ratio underflows to 0 for the
        # 1e-28-scale probabilities the saturating heads emit.
        return np.log(p) - np.log1p(-p)

    def _assert_rank_preserving(where, raw, transformed, y):
        """Every per-arm transform must leave that arm's own AUROC untouched.

        Fusion may legitimately change a score; the per-arm SPACE CHANGE may not. Two bugs
        in this analysis (6-dp rounding of stored scores, and a 1e-6 logit floor) both
        showed up exactly here -- as a per-arm AUROC that moved before any weighting
        happened -- and both were invisible in the fused numbers, which merely looked
        disappointing. So this is checked rather than assumed.
        """
        bad = []
        for k in raw:
            a, b = roc_auc_score(y, raw[k]), roc_auc_score(y, transformed[k])
            if abs(a - b) > 1e-9:
                bad.append((k, a, b))
        if bad:
            print(f"!! {where}: {len(bad)} arm(s) whose AUROC MOVED under the "
                  f"{args.space} transform -- the fusion below is not trustworthy:")
            for k, a, b in bad[:10]:
                print(f"     {k:<34} raw {a:.4f} -> transformed {b:.4f}")
            raise SystemExit("rank-destroying transform; fix EPS or use --space prob")

    RAW = {
        "eval": ({k: np.asarray(d["scores"][k], float) for k in names},
                 np.array(d["labels"])),
        "heldout": ({k: np.asarray(d["heldout"][k]["scores"], float) for k in names},
                    np.array(d["heldout"][names[0]]["labels"])),
        "val": ({k: np.asarray(d["val"][k]["scores"], float) for k in names},
                np.array(d["val"][names[0]]["labels"])),
    }
    SETS = {}
    for where, (raw, yy) in RAW.items():
        tr = {k: prep(v) for k, v in raw.items()}
        _assert_rank_preserving(where, raw, tr, yy)
        SETS[where] = (tr, yy)
    print(f"[sweep] per-arm {args.space} transform verified rank-preserving on all "
          f"{len(names)} arms x 3 sets")

    def auroc(where, subset, w):
        X, y = SETS[where]
        s = sum(wi * X[k] for wi, k in zip(w, subset))
        return float(roc_auc_score(y, s))

    solo = {k: auroc("eval", (k,), (1.0,)) for k in names}
    base = solo[args.anchor]
    print(f"space={args.space}  anchor={args.anchor}  "
          f"eval n={len(SETS['eval'][1])}  heldout n={len(SETS['heldout'][1])}  "
          f"val n={len(SETS['val'][1])}")
    print("single-arm eval AUROC: " + "  ".join(f"{k} {v:.4f}" for k, v in solo.items()))
    print(f"\nbaseline to beat = {args.anchor} alone = {base:.4f}\n")

    others = [k for k in names if k != args.anchor]
    cap = args.max_subset_size or (len(others) + 1)

    def _sweep(subset):
        step = _grid_step(min(len(subset), 6))
        grid = list(_simplex(len(subset), step))
        ev = np.array([auroc("eval", subset, w) for w in grid])
        ho = np.array([auroc("heldout", subset, w) for w in grid])
        va = np.array([auroc("val", subset, w) for w in grid])
        i_ev, i_ho, i_va = int(ev.argmax()), int(ho.argmax()), int(va.argmax())
        return {
            "subset": list(subset), "n_grid": len(grid),
            "best_eval_CEILING": round(float(ev[i_ev]), 4),
            "best_eval_weights": [round(x, 3) for x in grid[i_ev]],
            "heldout_selected_eval": round(float(ev[i_ho]), 4),
            "heldout_weights": [round(x, 3) for x in grid[i_ho]],
            "val_selected_eval": round(float(ev[i_va]), 4),
            "val_weights": [round(x, 3) for x in grid[i_va]],
        }

    rows = [_sweep((args.anchor,))]
    pair_rows = [_sweep((args.anchor, o)) for o in others]
    rows += pair_rows
    if cap >= 3:
        # Shortlist partners on the pair result before going to size 3+, see --top-partners.
        rank = sorted(pair_rows, key=lambda r: -r["best_eval_CEILING"])
        short = [r["subset"][1] for r in rank[: args.top_partners]]
        print(f"[sweep] size>=3 partners shortlisted to: {', '.join(short)}\n")
        for r in range(2, min(cap, len(short) + 1)):
            for combo in itertools.combinations(short, r):
                rows.append(_sweep((args.anchor,) + combo))

    rows.sort(key=lambda r: -r["best_eval_CEILING"])
    rows = rows[: args.top_rows] if args.top_rows else rows
    w = max(len("+".join(r["subset"])) for r in rows)
    print(f"{'subset':<{w}}{'ceiling':>9}{'  (weights)':<26}"
          f"{'heldout-sel':>13}{'val-sel':>9}")
    for r in rows:
        tag = "+".join(r["subset"])
        wt = "(" + ",".join(f"{x:g}" for x in r["best_eval_weights"]) + ")"
        print(f"{tag:<{w}}{r['best_eval_CEILING']:>9.4f}  {wt:<24}"
              f"{r['heldout_selected_eval']:>13.4f}{r['val_selected_eval']:>9.4f}")

    best_ceiling = max(r["best_eval_CEILING"] for r in rows)
    best_ho = max(r["heldout_selected_eval"] for r in rows)
    best_va = max(r["val_selected_eval"] for r in rows)
    print(f"\n{args.anchor} alone                          {base:.4f}")
    print(f"best CEILING over all subsets+weights  {best_ceiling:.4f}  ({best_ceiling-base:+.4f})"
          f"   <- selection on the test set")
    print(f"best with weights from held-out dev    {best_ho:.4f}  ({best_ho-base:+.4f})")
    print(f"best with weights from validation      {best_va:.4f}  ({best_va-base:+.4f})")

    # The pair curve is the readable part: how does AUROC move as weight shifts off L32?
    # Pair curves are the readable part: how AUROC moves as weight shifts off the anchor.
    # With many arms, show only the partners that actually did something.
    curve_for = [r["subset"][1] for r in
                 sorted(pair_rows, key=lambda r: -r["best_eval_CEILING"])[
                     : (args.top_partners if len(others) > 12 else len(others))]]
    print(f"\n--- pair sweeps: w*{args.anchor} + (1-w)*other, eval AUROC ---")
    ws = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    ow = max(len(o) for o in curve_for)
    print(f"{'other':<{ow}}" + "".join(f"{x:>7.2f}" for x in ws))
    for o in curve_for:
        print(f"{o:<{ow}}" + "".join(f"{auroc('eval', (args.anchor, o), (x, 1 - x)):>7.4f}"
                                     for x in ws))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"space": args.space, "anchor": args.anchor, "singles": solo,
         "baseline": base, "subsets": rows}, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
