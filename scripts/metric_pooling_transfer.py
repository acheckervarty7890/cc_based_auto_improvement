"""Does the pooling that wins the metric sweep also buy eval AUROC?

``metric_geometry.py`` answers a *geometric* question — in which representation does
"close" mean "same label". If ``pool:last`` wins that, the natural follow-up is whether
the win is confined to neighbourhood structure or reaches the thing the doc actually
cares about: transfer to the four hu_ha eval splits.

So this re-runs sections 1, 2 and 4 of ``why_iter3_null.py`` verbatim — same L2 logistic
regression, same C grid, same content-deterministic split, same vintages — with the *only*
change being which pooling the 5376-dim feature vector comes from:

1. VINTAGE CURVE   v0 (base only) .. v3, mean eval AUROC per vintage.
2. CEILING         best linear probe fitted on each eval split itself (5-fold CV) — the
                   upper bound on what a linear readout of this pooling can do there.
3. ALIGNMENT       cos(w_v3, w_oracle[split]), the note's "load-bearing number".

Reading it: a pooling that raises the *ceiling* is exposing information mean pooling
destroys, and the doc's §2 transfer gap is measured against that ceiling — so if the
ceiling moves, the gap the loop is trying to close moves with it. A pooling that raises
the *achieved* v3 number moves the deployed metric.

Both the deployed probe head and every measurement in the doc pool differently (softmax
over per-token logits, and mean, respectively), so a gain here is a claim about the
representation, not about any probe that has been trained.

Usage:
    .venv_claude/bin/python scripts/metric_geometry_features.py --only-eval  # all poolings
    .venv_claude/bin/python scripts/metric_pooling_transfer.py
    .venv_claude/bin/python scripts/metric_pooling_transfer.py --poolings mean last
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
import attribution_vintage as V
from metric_geometry_features import _pool_all
from why_iter3_null import C_GRID, cos, fit_logistic, pick_c, score

REPS_DIR = Path("results_hu_harm_gemma27b_batch_ablation/vintage/metric_reps")
OUT = Path("results_hu_harm_gemma27b_batch_ablation/vintage/metric_pooling_transfer.json")


def base_features(arm: str) -> tuple[dict, np.ndarray, dict, np.ndarray]:
    """All poolings of the base training data's train and val sides.

    Only 50 conversations, so this is re-pooled here rather than cached — and it must use
    the *arm's own* probe for the ``probe``/``topz16`` poolings, which is why it is not in
    the shared eval cache.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    probe2 = A.load_probe(A.ARMS[arm] / "probe_iter2.pkl")
    w, b, T = A.probe_params(probe2)
    ds = LabelledDataset.load_from(
        A.BASE_TRAINING_DATA,
        pos_class_label=probe2.pos_class_label,
        neg_class_label=probe2.neg_class_label,
        combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
    )
    tr, va = A.split_sides(ds)
    out = []
    for part, path in ((tr, A.base_blob_paths()[0]), (va, A.base_blob_paths()[1])):
        blob = torch.load(path, weights_only=False, mmap=True)
        pooled = _pool_all(blob["activations"], blob["attention_mask"], w, b, T)
        out.append(
            ({k: v for k, v in pooled.items() if not k.startswith("_")},
             part.labels_torch().float().cpu().numpy())
        )
        del blob
        gc.collect()
    del probe2
    return out[0][0], out[0][1], out[1][0], out[1][1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", default=["gptoss120b", "deepseekv4pro"])
    ap.add_argument("--poolings", nargs="+", default=["mean", "last", "probe"])
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    ze = np.load(REPS_DIR / "eval.npz", allow_pickle=True)
    report: dict = {"ceiling": {}, "arms": {}}

    print("=== 2. CEILING: best linear probe fitted ON the split (5-fold CV) ===", flush=True)
    print(f"{'pooling':<10} " + " ".join(f"{s.replace('eval_',''):>17}" for s in A.EVAL_SPLITS) + f" {'mean':>8}")
    for pool in args.poolings:
        row = {}
        for split in A.EVAL_SPLITS:
            x, y = ze[f"X_{pool}_{split}"], ze[f"y_{split}"]
            best = -1.0
            for c in C_GRID:
                s = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
                    sc = StandardScaler().fit(x[tr])
                    clf = LogisticRegression(C=c, max_iter=4000).fit(sc.transform(x[tr]), y[tr])
                    s[te] = clf.decision_function(sc.transform(x[te]))
                best = max(best, roc_auc_score(y, s))
            row[split] = float(best)
        row["mean"] = float(np.mean([row[s] for s in A.EVAL_SPLITS]))
        report["ceiling"][pool] = row
        print(f"{pool:<10} " + " ".join(f"{row[s]:>17.4f}" for s in A.EVAL_SPLITS) + f" {row['mean']:>8.4f}", flush=True)

    for arm in args.arm:
        print(f"\n########## {arm} ##########", flush=True)
        z = np.load(REPS_DIR / f"{arm}.npz", allow_pickle=True)
        keep, _ = V.vintages(arm, 3)
        rt_is_val = z["is_val"]
        rt_y = z["y"]
        btr_x, btr_y, bva_x, bva_y = base_features(arm)
        arm_out: dict = {}

        for pool in args.poolings:
            t0 = time.time()
            rt_x = z[f"X_{pool}"]
            curve, weights = {}, {}
            for v in (0, 1, 2, 3):
                idx = np.array(keep[v], dtype=int)
                if len(idx):
                    tr_i, va_i = idx[~rt_is_val[idx]], idx[rt_is_val[idx]]
                    x = np.concatenate([btr_x[pool], rt_x[tr_i]])
                    y = np.concatenate([btr_y, rt_y[tr_i]])
                    xv = np.concatenate([bva_x[pool], rt_x[va_i]])
                    yv = np.concatenate([bva_y, rt_y[va_i]])
                else:
                    x, y, xv, yv = btr_x[pool], btr_y, bva_x[pool], bva_y
                c = pick_c(x, y, xv, yv)
                sc, clf = fit_logistic(x, y, c)
                weights[v] = clf.coef_.ravel() / sc.scale_
                row = {
                    s: float(roc_auc_score(ze[f"y_{s}"], score(sc, clf, ze[f"X_{pool}_{s}"])))
                    for s in A.EVAL_SPLITS
                }
                row["mean"] = float(np.mean([row[s] for s in A.EVAL_SPLITS]))
                row["C"] = c
                curve[v] = row

            align = {}
            for split in A.EVAL_SPLITS:
                sc_s, clf_s = fit_logistic(ze[f"X_{pool}_{split}"], ze[f"y_{split}"], 0.03)
                align[split] = cos(weights[3], clf_s.coef_.ravel() / sc_s.scale_)
            align["mean"] = float(np.mean([align[s] for s in A.EVAL_SPLITS]))

            arm_out[pool] = {
                "curve": curve,
                "alignment_w3_oracle": align,
                "cos_w2_w3": cos(weights[2], weights[3]),
                "gap_to_ceiling": {
                    s: report["ceiling"][pool][s] - curve[3][s] for s in A.EVAL_SPLITS
                },
            }
            print(
                f"  {pool:<8} v0 {curve[0]['mean']:.4f}  v1 {curve[1]['mean']:.4f}  "
                f"v2 {curve[2]['mean']:.4f}  v3 {curve[3]['mean']:.4f}   "
                f"(v3-v2 {curve[3]['mean']-curve[2]['mean']:+.4f})  "
                f"ceiling {report['ceiling'][pool]['mean']:.4f}  "
                f"align {align['mean']:.4f}   [{time.time()-t0:.0f}s]",
                flush=True,
            )
            del rt_x
            gc.collect()

        report["arms"][arm] = arm_out
        del z, btr_x, bva_x
        gc.collect()

    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
