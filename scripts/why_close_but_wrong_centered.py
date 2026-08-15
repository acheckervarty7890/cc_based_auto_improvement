"""Does the uncentered cosine change the conclusions of ``why_close_but_wrong.py``?

That script measures cosine on **raw mean-pooled** activations, with no centering. A
residual stream has a large shared component, so every pair sits at 0.86-0.96 and the
absolute value is mostly that common direction rather than anything about the two
conversations. Its load-bearing comparisons are all *within* the same metric (same-label
vs opposite-label; a source vs its own counterpart vs the new-row hop), which is valid on
any monotone scale -- but the nearest-neighbour results (which v2 row is closest, and how
a k-NN built on them classifies) genuinely depend on the geometry, and centering changes
the geometry.

So: rerun sections A-D under three representations and see whether anything moves.

    raw        mean-pooled, L2-normalised          (what the published numbers use)
    centered   minus the grand mean of all rows, then L2-normalised
    whitened   per-dimension z-score, then L2-normalised

Usage:
    .venv_claude/bin/python scripts/why_close_but_wrong_centered.py
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
import attribution_vintage as V
from why_close_but_wrong import load_rows, unit


def reps(X: np.ndarray) -> dict[str, np.ndarray]:
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    return {"raw": X, "centered": X - mu, "whitened": (X - mu) / sd}


def main() -> None:
    from sklearn.neighbors import KNeighborsClassifier

    out = {}
    for arm in ("gptoss120b", "deepseekv4pro"):
        print(f"\n########## {arm} ##########", flush=True)
        X0, y, logit2, src, is_gen, ds = load_rows(arm)
        keep, _ = V.vintages(arm, 3)
        idx3 = np.array(keep[3], dtype=int)
        v2 = np.array(keep[2], dtype=int)
        new = np.setdiff1d(idx3, v2)
        succ = new[~is_gen[new]]

        by_src = {}
        for i in idx3:
            by_src.setdefault(src[i], {})[bool(is_gen[i])] = i
        pair_idx = [(d[False], d[True]) for d in by_src.values() if True in d and False in d]

        arm_out = {}
        rng = np.random.default_rng(0)
        ri, rj = rng.choice(v2, 4000), rng.choice(v2, 4000)
        keep_m = ri != rj
        ri, rj = ri[keep_m], rj[keep_m]

        print(f"{'rep':<10} {'same-lab':>9} {'opp-lab':>9} {'Δ':>7} "
              f"{'own-pair':>9} {'new→v2 NN':>10} {'NN same-lab':>12} {'kNN k=1':>8} {'k=5':>7} {'k=15':>7}")
        for name, X in reps(X0).items():
            U = unit(X)
            cs = (U[ri] * U[rj]).sum(1)
            same = cs[y[ri] == y[rj]].mean()
            opp = cs[y[ri] != y[rj]].mean()
            own = np.mean([float(U[a] @ U[b]) for a, b in pair_idx])
            nn = unit(X[succ]) @ unit(X[v2]).T
            nn_lab = float((y[v2[nn.argmax(1)]] == y[succ]).mean())
            ks = {}
            for k in (1, 5, 15):
                knn = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(X[v2], y[v2])
                ks[k] = float((knn.predict(X[succ]) == y[succ]).mean())
            arm_out[name] = {
                "same_label": float(same), "opp_label": float(opp),
                "delta": float(same - opp), "own_pair": own,
                "nn_cos_successes": float(nn.max(1).mean()),
                "nn_label_agreement": nn_lab, "knn": ks,
            }
            print(f"{name:<10} {same:>9.4f} {opp:>9.4f} {same-opp:>7.4f} "
                  f"{own:>9.4f} {nn.max(1).mean():>10.4f} {nn_lab:>11.1%} "
                  f"{ks[1]:>7.1%} {ks[5]:>6.1%} {ks[15]:>6.1%}")

        out[arm] = arm_out
        del X0
        gc.collect()

    p = Path("results_hu_harm_gemma27b_batch_ablation/vintage/why_close_but_wrong_centered.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")
    print("\nchance for the kNN columns is 50% (the success set is class-balanced)")


if __name__ == "__main__":
    main()
