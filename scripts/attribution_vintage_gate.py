"""Does the iteration-3 red-team data actually change eval AUROC, or is the dip noise?

``attribution_vintage.py`` fit each vintage once, at one seed, and reported that
gptoss120b's ``eval_balanced_refusal`` fell 0.9155 -> 0.8698 from vintage 2 to
vintage 3. That comparison cannot support any claim: the arm's own noise floor
(``<arm>_noisefloor.json``, 10 seeds on one fixed training set) puts
``eval_balanced_refusal`` at sd **0.0233**, range **0.0778** — larger than the
observed drop. Two independent fits of the *same* data differ by more than the
"effect".

So this script asks the question in the only form that can answer it: **paired**.
Every pass trains all vintages as columns of one ``train_many`` call, sharing one
initialisation and one shuffle stream (common random numbers), so the columns differ
only in their drop-set and the seed noise cancels in the difference. Repeating over
``--seeds`` independent seeds turns each contrast into a mean +/- standard error.

The control column is what makes the result interpretable
--------------------------------------------------------
Vintage 3 is vintage 2 plus 116 further pairs, so "vintage 3 is worse" has two very
different possible causes: *those particular* conversations are harmful to the split,
or *any* 116 extra pairs would be. ``--controls`` adds columns holding vintage 3 minus
a **random** 116 pairs — same size as vintage 2, different composition. If vintage 2
and the random controls score alike, the iteration-3 samples carry no special blame
and there is nothing to attribute; if vintage 2 beats the controls, the specific
composition matters and a per-pair sweep is warranted.

Validation rows are dropped too (``val_keep_mask``), not just training rows: ~31% of
pairs straddle the content-deterministic split, and the val set is what early stopping
reads, so a train-only drop would be a different intervention than the real one.

Usage:
    .venv_claude/bin/python scripts/attribution_vintage_gate.py --arm gptoss120b --seeds 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_fasttrain as F  # noqa: E402
import attribution_lib as A  # noqa: E402
import attribution_pack as P  # noqa: E402
from attribution_vintage import vintages  # noqa: E402


def pair_vintages(arm: str, iteration: int, pairs) -> dict[int, set[int]]:
    """``{vintage: set of pair ids}``, lifted from the row-level vintage sets.

    ``vintages()`` works in red-team *row* indices; the drop-set machinery works in
    pairs. A pair belongs to a vintage when either of its rows does — which is always
    both, since membership is decided per source success and both rows follow.
    """
    keep_rows, report = vintages(arm, iteration)
    pair_of_row: dict[int, int] = {}
    for p in pairs:
        for idx in (p.source_idx, p.generated_idx):
            if idx is not None:
                pair_of_row[idx] = p.pair_id
    out = {}
    for k, rows in keep_rows.items():
        out[k] = {pair_of_row[r] for r in rows if r in pair_of_row}
    return out, report


def masks_for(keep_pairs: set[int], pairs, n_train: int, n_val: int):
    """``(train_keep, val_keep)`` boolean vectors for one column.

    Base rows are always kept; a red-team row is kept iff its pair is in the set.
    """
    tr = np.ones(n_train, dtype=bool)
    va = np.ones(n_val, dtype=bool)
    for p in pairs:
        if p.pair_id in keep_pairs:
            continue
        for r in p.packed_train_rows:
            tr[r] = False
        for r in p.packed_val_rows:
            va[r] = False
    return tr, va


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gptoss120b", choices=sorted(A.ARMS))
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=A.SEED)
    ap.add_argument("--controls", type=int, default=3,
                    help="random-drop control columns per pass (0 disables)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    train, val, eval_packed, meta = P.build(args.arm, args.iteration)
    pairs = meta["pairs"]
    pv, report = pair_vintages(args.arm, args.iteration, pairs)
    all_pairs = {p.pair_id for p in pairs}
    new_pairs = pv[args.iteration] - pv[args.iteration - 1]
    print(
        f"pairs: v1={len(pv[1])} v2={len(pv[2])} v3={len(pv[3])} "
        f"(all={len(all_pairs)}); new in v3 = {len(new_pairs)}",
        flush=True,
    )

    columns: list[tuple[str, set[int]]] = [
        (f"v{k}", pv[k]) for k in sorted(pv) if k >= 1
    ]
    n_ctrl = args.controls
    results: dict[str, list[dict]] = {}

    for si in range(args.seeds):
        seed = args.seed0 + si
        # Control draws differ per seed on purpose: a single fixed random subset would
        # itself be one draw of a noisy quantity, and averaging over draws is what
        # makes "any 116 pairs" a population statement rather than an anecdote.
        rng = np.random.default_rng(10_000 + seed)
        cols = list(columns)
        for c in range(n_ctrl):
            # Vintage 2 is "all pairs minus these particular 116"; a control is "all
            # pairs minus *some* 116", so the two differ only in which pairs were cut.
            drop = set(
                rng.choice(sorted(all_pairs), size=len(new_pairs), replace=False).tolist()
            )
            cols.append((f"ctrl{c}", all_pairs - drop))

        tr_masks, va_masks = [], []
        for _, keep in cols:
            t, v = masks_for(keep, pairs, train.n, val.n)
            tr_masks.append(t)
            va_masks.append(v)
        keep_mask = torch.from_numpy(np.stack(tr_masks, axis=1))
        val_keep = torch.from_numpy(np.stack(va_masks, axis=1))

        t0 = time.time()
        W, b, best_auroc, best_epoch = F.train_many(
            train, val, keep_mask, seed, val_keep_mask=val_keep
        )
        scores = F.score_eval(eval_packed, W, b)
        dt = time.time() - t0

        line = [f"seed {seed} ({dt:.0f}s)"]
        for j, (name, keep) in enumerate(cols):
            row = {
                "seed": seed,
                "n_pairs": len(keep),
                "best_epoch": int(best_epoch[j]),
                "auroc": {
                    sp: {sc: float(scores[sp][sc][j]) for sc in ("pipeline", "rank")}
                    for sp in list(A.EVAL_SPLITS) + ["mean"]
                },
            }
            results.setdefault(name, []).append(row)
            line.append(
                f"{name}: refusal={row['auroc']['eval_balanced_refusal']['pipeline']:.4f} "
                f"mean={row['auroc']['mean']['pipeline']:.4f}"
            )
        print("  ".join(line), flush=True)

    # --- paired contrasts ----------------------------------------------------------
    def col(name, split, scale="pipeline"):
        return np.array([r["auroc"][split][scale] for r in results[name]])

    ctrl_names = [f"ctrl{c}" for c in range(n_ctrl)]
    contrasts = [("v3", "v2"), ("v2", "v1"), ("v3", "v1")]
    if n_ctrl:
        contrasts.append(("v2", "ctrl_mean"))
        contrasts.append(("v3", "ctrl_mean"))

    print("\n=== paired contrasts (pipeline scale), mean +/- SE over seeds ===")
    summary = {}
    for split in list(A.EVAL_SPLITS) + ["mean"]:
        print(f"\n{split}")
        for a, b_ in contrasts:
            va = col(a, split)
            vb = (
                np.mean([col(c, split) for c in ctrl_names], axis=0)
                if b_ == "ctrl_mean"
                else col(b_, split)
            )
            d = va - vb
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            sig = "" if not np.isfinite(se) or se == 0 else (
                "  ***" if abs(d.mean()) > 2 * se else ""
            )
            print(
                f"  {a:4s} - {b_:9s}  {d.mean():+.4f} +/- {se:.4f}"
                f"   [{a}={va.mean():.4f}, {b_}={vb.mean():.4f}]{sig}"
            )
            summary[f"{split}|{a}-{b_}"] = {
                "delta": float(d.mean()),
                "se": float(se),
                "n": int(len(d)),
            }

    print("\n=== per-column absolute AUROC (mean over seeds, pipeline) ===")
    for name in [c for c, _ in [(n, None) for n in results]]:
        r = col(name, "eval_balanced_refusal")
        m = col(name, "mean")
        print(
            f"  {name:7s} refusal={r.mean():.4f} (sd {r.std(ddof=1):.4f})   "
            f"mean={m.mean():.4f} (sd {m.std(ddof=1):.4f})"
        )

    out = args.out or (
        A.REPO / f"results_hu_harm_gemma27b_batch_ablation/vintage/{args.arm}_gate.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"report": report, "n_new_pairs": len(new_pairs),
             "per_seed": results, "contrasts": summary},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
