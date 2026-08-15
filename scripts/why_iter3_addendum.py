"""Follow-ups to ``why_iter3_null.py`` that its first pass left open.

Three things, all on the same deterministic instrument (mean-pooled layer-32
activations + L2 logistic), so the numbers are directly comparable to that script's:

A. **Provenance separability, cross-validated.** The first pass reported an *in-sample*
   AUROC of 1.0000 for "was this row written by the attacker or by the contrastive
   generator". At 5376 features and ~800 rows that is what perfect overfitting looks
   like and says nothing. 5-fold CV says whether the axis is really there.

B. **Does alignment with the eval splits improve across vintages at all?** The first
   pass measured ``cos(w_v3, w_oracle[split])`` only. The trend across v1 -> v2 -> v3 is
   the decision-relevant quantity: if it is flat, no number of further iterations of
   this loop converges on what the eval splits need, and the fix has to change the
   loop rather than run it longer.

C. **Are iteration 3's pairs weak, or merely redundant?** Train on the 116 new pairs
   *alone* (plus base) and compare with an equally sized random draw from the earlier
   vintage. Same size, different provenance -- the same control logic the gate used.

Usage:
    .venv_claude/bin/python scripts/why_iter3_addendum.py
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
from why_iter3_null import (
    C_GRID,
    base_features,
    eval_features,
    fit_logistic,
    pick_c,
    redteam_features,
    score,
    cos,
)


def cv_auroc(x, y, c=0.03, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(x, y):
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(C=c, max_iter=4000).fit(sc.transform(x[tr]), y[tr])
        s[te] = clf.decision_function(sc.transform(x[te]))
    return float(roc_auc_score(y, s))


def main() -> None:
    from sklearn.metrics import roc_auc_score

    eval_x, eval_y = {}, {}
    for split in A.EVAL_SPLITS:
        eval_x[split], eval_y[split] = eval_features(split)
    oracle_w = {}
    for split in A.EVAL_SPLITS:
        sc, clf = fit_logistic(eval_x[split], eval_y[split], 0.03)
        oracle_w[split] = clf.coef_.ravel() / sc.scale_

    (btr_x, btr_y), (bva_x, bva_y) = base_features()
    out = {}

    for arm in ("gptoss120b", "deepseekv4pro"):
        print(f"\n########## {arm} ##########", flush=True)
        rt_x, rt_y, rt_src, rt_gen, rt_ds = redteam_features(arm)
        keep, _ = V.vintages(arm, 3)
        rt_is_val = np.array([A.is_val(m) for m in rt_ds.inputs], dtype=bool)
        idx3 = np.array(keep[3], dtype=int)
        arm_out = {}

        print("\n=== A. provenance axis, cross-validated ===", flush=True)
        gen = rt_gen[idx3].astype(float)
        a_cv = cv_auroc(rt_x[idx3], gen)
        arm_out["provenance_cv_auroc"] = a_cv
        print(f"  source-vs-generated, 5-fold CV AUROC = {a_cv:.4f}"
              f"   (in-sample was 1.0000 and meaningless at 5376 features)")

        print("\n=== B. alignment with each split's own direction, by vintage ===", flush=True)
        def fit_vintage(idx):
            if len(idx) == 0:
                x, y, xv, yv = btr_x, btr_y, bva_x, bva_y
            else:
                tr_i, va_i = idx[~rt_is_val[idx]], idx[rt_is_val[idx]]
                x = np.concatenate([btr_x, rt_x[tr_i]])
                y = np.concatenate([btr_y, rt_y[tr_i]])
                xv = np.concatenate([bva_x, rt_x[va_i]])
                yv = np.concatenate([bva_y, rt_y[va_i]])
            c = pick_c(x, y, xv, yv)
            sc, clf = fit_logistic(x, y, c)
            return sc, clf, clf.coef_.ravel() / sc.scale_

        align = {}
        for v in (1, 2, 3):
            _, _, w = fit_vintage(np.array(keep[v], dtype=int))
            align[v] = {s: cos(w, oracle_w[s]) for s in A.EVAL_SPLITS}
            align[v]["mean"] = float(np.mean([align[v][s] for s in A.EVAL_SPLITS]))
        arm_out["oracle_alignment"] = align
        hdr = f"{'vintage':>8} " + " ".join(f"{s.replace('eval_',''):>16}" for s in A.EVAL_SPLITS) + f" {'mean':>8}"
        print(hdr)
        for v in (1, 2, 3):
            print(f"{v:>8} " + " ".join(f"{align[v][s]:>16.4f}" for s in A.EVAL_SPLITS) + f" {align[v]['mean']:>8.4f}")

        print("\n=== C. iteration-3 pairs alone vs a size-matched earlier draw ===", flush=True)
        new_mask = np.zeros(len(rt_x), dtype=bool)
        new_mask[idx3] = True
        new_mask[np.array(keep[2], dtype=int)] = False
        new_idx = np.flatnonzero(new_mask)
        v2_idx = np.array(keep[2], dtype=int)

        rows = {}

        def eval_of(idx, tag):
            sc, clf, _ = fit_vintage(np.asarray(idx, dtype=int))
            r = {s: float(roc_auc_score(eval_y[s], score(sc, clf, eval_x[s]))) for s in A.EVAL_SPLITS}
            r["mean"] = float(np.mean([r[s] for s in A.EVAL_SPLITS]))
            rows[tag] = r
            print(f"  {tag:28s} n={len(idx):4d} " + " ".join(f"{r[s]:.4f}" for s in A.EVAL_SPLITS) + f"   mean {r['mean']:.4f}")

        eval_of(new_idx, "v3-only (116 new pairs)")
        rng = np.random.default_rng(0)
        ctrl = []
        for k in range(3):
            pick = rng.choice(v2_idx, size=len(new_idx), replace=False)
            eval_of(pick, f"random {len(new_idx)} rows from v2 #{k}")
            ctrl.append(rows[f"random {len(new_idx)} rows from v2 #{k}"]["mean"])
        arm_out["subset_fits"] = rows
        arm_out["ctrl_mean"] = float(np.mean(ctrl))
        print(f"  -> v3-only mean {rows['v3-only (116 new pairs)']['mean']:.4f} "
              f"vs size-matched v2 draws {np.mean(ctrl):.4f} ± {np.std(ctrl):.4f}")

        out[arm] = arm_out
        del rt_x
        gc.collect()

    p = Path("results_hu_harm_gemma27b_batch_ablation/vintage/why_iter3_addendum.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
