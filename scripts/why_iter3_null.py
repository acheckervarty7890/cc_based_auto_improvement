"""Why does the *last* red-team iteration add no measurable eval AUROC?

The vintage sweep (``attribution_vintage.py``, 10 seeds x 4 vintages x 2 arms) put
``v3 - v2`` at -0.001 (gptoss) and -0.002 (deepseek) on the mean eval AUROC, against a
seed sd of ~0.011-0.027. That is a null, but the sweep cannot say *why*: it measures
the pipeline end to end, and the pipeline's own seed noise is as large as the effect
being looked for.

This script separates the candidate causes, using only activations already on disk.
The instrument is deliberately **not** the pipeline trainer: mean-pooled activations
plus an L2 logistic regression is convex and deterministic, so every number below is
exactly reproducible and a v3-v2 difference of 0.002 is readable. It is a different
probe head from the deployed one, so its absolute AUROCs are not comparable to the
committed CSVs -- only the contrasts within this script are.

Six questions, one section each:

1. VINTAGE CURVE, NOISE-FREE -- is ``v3 - v2`` still zero when the trainer's seed
   noise is removed entirely? Distinguishes "the data is redundant" from "the
   instrument was too blunt".
2. CEILING -- how well can *any* linear probe on these activations do on each eval
   split, fitted on that split itself (5-fold CV)? A split already at its ceiling
   cannot be improved by more training data of any kind.
3. TRANSFER GAP -- ceiling minus achieved, per split, per vintage. Where the
   remaining headroom actually is.
4. DIRECTION SATURATION -- cosine between the weight vectors fitted on v1/v2/v3.
   If the direction stops moving, later data is redundant by construction.
5. COVERAGE -- do the pairs iteration 3 added occupy a region the earlier vintages
   did not? Nearest-neighbour cosine of each new pair against the v2 set, and of the
   red-team set against each eval split (does red-team data live where eval lives?).
6. THE SOURCE/GENERATED CONFOUND -- ``preprocessing`` mints one LLM-written
   opposite-class counterpart per success, and the successes are 71% false-negative,
   so "written by the attacker" correlates with "harmful". Measures how separable
   source from generated is, and how much of the fitted direction that axis explains.

Usage:
    .venv_claude/bin/python scripts/why_iter3_null.py --arm gptoss120b deepseekv4pro
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
import attribution_vintage as V

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


# --- representations ---------------------------------------------------------------


def _pool(acts: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Mean over the unpadded tokens, in fp32.

    The deployed head pools with a softmax over its own per-token logits, which makes
    the representation a function of the weights being fitted -- fine for the probe,
    useless as a fixed feature space. Mean pooling is the weight-independent stand-in;
    section 1 checks it reproduces the pipeline's vintage ordering before anything else
    is read off it.
    """
    m = mask.bool()
    h = acts.float()
    h = h * m[..., None]
    return (h.sum(-2) / m.sum(-1, keepdim=True).clamp(min=1)).numpy()


def eval_features(split: str) -> tuple[np.ndarray, np.ndarray]:
    from tuberlens.interfaces.dataset import LabelledDataset

    ds = LabelledDataset.load_from(
        A.EVAL_DATASET_DIR / f"{split}.jsonl",
        pos_class_label="harmful_to_human",
        neg_class_label="not_harmful_to_human",
        combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
    )
    y = ds.labels_torch().float().cpu().numpy()
    blob = torch.load(
        A.EVAL_ACTIVATIONS_DIR / f"{split}-acts_full.pt", weights_only=False, mmap=True
    )
    x = _pool(blob["activations"], blob["attention_mask"])
    del blob
    gc.collect()
    return x, y


def redteam_features(arm: str, iteration: int = 3):
    """``(X, y, keys, is_generated)`` for the iteration-3 dump, in dump order."""
    ds = A.load_redteam_dataset(arm, iteration)
    gen2src = V._generated_to_source(arm)
    labels = ds.other_fields["labels"]
    xs, keys, is_gen = [], [], []
    for i, msgs in enumerate(ds.inputs):
        blob = torch.load(A.redteam_blob_path(msgs), weights_only=False)
        xs.append(_pool(blob["activations"], blob["attention_mask"])[0])
        key = A.canon(msgs)
        keys.append(gen2src.get(key, key))
        is_gen.append(key in gen2src)
        del blob
    y = np.array([1.0 if l == "positive" else 0.0 for l in labels], dtype=np.float32)
    return np.stack(xs), y, keys, np.array(is_gen), ds


def base_features(pos="harmful_to_human", neg="not_harmful_to_human"):
    from tuberlens.interfaces.dataset import LabelledDataset

    ds = LabelledDataset.load_from(
        A.BASE_TRAINING_DATA,
        pos_class_label=pos,
        neg_class_label=neg,
        combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
    )
    tr, va = A.split_sides(ds)
    out = []
    for part, path in ((tr, A.base_blob_paths()[0]), (va, A.base_blob_paths()[1])):
        blob = torch.load(path, weights_only=False, mmap=True)
        out.append((_pool(blob["activations"], blob["attention_mask"]),
                    part.labels_torch().float().cpu().numpy()))
        del blob
    return out[0], out[1]


# --- fitting -----------------------------------------------------------------------


def fit_logistic(x, y, c=0.03):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(x)
    clf = LogisticRegression(C=c, max_iter=4000).fit(sc.transform(x), y)
    return sc, clf


def score(sc, clf, x):
    return clf.decision_function(sc.transform(x))


def pick_c(x, y, xval, yval):
    """Choose C on the held-out validation side, mirroring how the pipeline early-stops."""
    from sklearn.metrics import roc_auc_score

    best, best_c = -1.0, C_GRID[0]
    for c in C_GRID:
        sc, clf = fit_logistic(x, y, c)
        try:
            a = roc_auc_score(yval, score(sc, clf, xval))
        except ValueError:
            continue
        if a > best:
            best, best_c = a, c
    return best_c


# --- sections ----------------------------------------------------------------------


def section_ceiling(eval_x, eval_y):
    """5-fold CV AUROC of a linear probe fitted on the split itself."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    out = {}
    for split in A.EVAL_SPLITS:
        x, y = eval_x[split], eval_y[split]
        best = -1.0
        for c in C_GRID:
            s = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
                sc = StandardScaler().fit(x[tr])
                clf = LogisticRegression(C=c, max_iter=4000).fit(sc.transform(x[tr]), y[tr])
                s[te] = clf.decision_function(sc.transform(x[te]))
            best = max(best, roc_auc_score(y, s))
        out[split] = best
    out["mean"] = float(np.mean([out[s] for s in A.EVAL_SPLITS]))
    return out


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", default=["gptoss120b", "deepseekv4pro"])
    ap.add_argument("--out", default="results_hu_harm_gemma27b_batch_ablation/vintage/why_iter3.json")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    print("loading eval activations ...", flush=True)
    eval_x, eval_y = {}, {}
    for split in A.EVAL_SPLITS:
        eval_x[split], eval_y[split] = eval_features(split)
        print(f"  {split}: {eval_x[split].shape}", flush=True)

    print("\n=== 2. CEILING: best linear probe fitted ON the split (5-fold CV) ===", flush=True)
    ceiling = section_ceiling(eval_x, eval_y)
    for k, v in ceiling.items():
        print(f"  {k:24s} {v:.4f}")

    (btr_x, btr_y), (bva_x, bva_y) = base_features()
    report = {"ceiling": ceiling, "arms": {}}

    for arm in args.arm:
        print(f"\n########## {arm} ##########", flush=True)
        print("loading red-team activations ...", flush=True)
        rt_x, rt_y, rt_src, rt_gen, rt_ds = redteam_features(arm)
        keep, vreport = V.vintages(arm, 3)
        rt_is_val = np.array([A.is_val(m) for m in rt_ds.inputs], dtype=bool)

        arm_out = {"vintage_report": vreport, "curve": {}, "cos": {}, "coverage": {}}
        weights = {}

        print("\n=== 1. VINTAGE CURVE, deterministic instrument ===", flush=True)
        header = f"{'vintage':>8} {'rows':>5} " + " ".join(f"{s.replace('eval_',''):>16}" for s in A.EVAL_SPLITS) + f" {'mean':>8}"
        print(header)
        for v in (0, 1, 2, 3):
            idx = np.array(keep[v], dtype=int)
            if len(idx):
                tr_i = idx[~rt_is_val[idx]]
                va_i = idx[rt_is_val[idx]]
                x = np.concatenate([btr_x, rt_x[tr_i]])
                y = np.concatenate([btr_y, rt_y[tr_i]])
                xv = np.concatenate([bva_x, rt_x[va_i]])
                yv = np.concatenate([bva_y, rt_y[va_i]])
            else:
                x, y, xv, yv = btr_x, btr_y, bva_x, bva_y
            c = pick_c(x, y, xv, yv)
            sc, clf = fit_logistic(x, y, c)
            w = clf.coef_.ravel() / sc.scale_
            weights[v] = w
            row = {}
            for split in A.EVAL_SPLITS:
                row[split] = float(roc_auc_score(eval_y[split], score(sc, clf, eval_x[split])))
            row["mean"] = float(np.mean([row[s] for s in A.EVAL_SPLITS]))
            row["C"] = c
            row["n_train"] = int(len(y))
            arm_out["curve"][v] = row
            print(f"{v:>8} {len(idx):>5} " + " ".join(f"{row[s]:>16.4f}" for s in A.EVAL_SPLITS) + f" {row['mean']:>8.4f}   (C={c})")

        print("\n=== 3. TRANSFER GAP (ceiling - v3) ===", flush=True)
        for split in A.EVAL_SPLITS:
            g = ceiling[split] - arm_out["curve"][3][split]
            print(f"  {split:24s} ceiling {ceiling[split]:.4f}  v3 {arm_out['curve'][3][split]:.4f}  gap {g:+.4f}")

        print("\n=== 4. DIRECTION SATURATION ===", flush=True)
        for a, b in ((1, 2), (2, 3), (1, 3)):
            arm_out["cos"][f"w{a}_w{b}"] = cos(weights[a], weights[b])
            print(f"  cos(w_v{a}, w_v{b}) = {arm_out['cos'][f'w{a}_w{b}']:.4f}")
        # how much does the boundary move, relative to its own length?
        for a, b in ((1, 2), (2, 3)):
            d = weights[b] / np.linalg.norm(weights[b]) - weights[a] / np.linalg.norm(weights[a])
            arm_out["cos"][f"move_{a}_{b}"] = float(np.linalg.norm(d))
            print(f"  |w_v{b}/|w| - w_v{a}/|w|| = {arm_out['cos'][f'move_{a}_{b}']:.4f}")
        # alignment with each split's own best direction
        for split in A.EVAL_SPLITS:
            sc, clf = fit_logistic(eval_x[split], eval_y[split], 0.03)
            wsplit = clf.coef_.ravel() / sc.scale_
            arm_out["cos"][f"w3_{split}"] = cos(weights[3], wsplit)
            print(f"  cos(w_v3, w_oracle[{split}]) = {arm_out['cos'][f'w3_{split}']:.4f}")

        print("\n=== 5. COVERAGE / NOVELTY ===", flush=True)
        new_mask = np.zeros(len(rt_x), dtype=bool)
        new_mask[np.array(keep[3], dtype=int)] = True
        new_mask[np.array(keep[2], dtype=int)] = False
        v2_idx = np.array(keep[2], dtype=int)

        def unit(m):
            return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)

        U_new, U_v2 = unit(rt_x[new_mask]), unit(rt_x[v2_idx])
        nn_new = (U_new @ U_v2.T).max(1)
        # baseline: how similar is a v2 row to the *rest* of v2?
        S = U_v2 @ U_v2.T
        np.fill_diagonal(S, -1)
        nn_v2 = S.max(1)
        arm_out["coverage"]["nn_new_to_v2_mean"] = float(nn_new.mean())
        arm_out["coverage"]["nn_v2_to_v2_mean"] = float(nn_v2.mean())
        print(f"  new-in-v3 rows: {int(new_mask.sum())}")
        print(f"  nearest-neighbour cosine, new-in-v3 -> v2 : mean {nn_new.mean():.4f}  p10 {np.percentile(nn_new,10):.4f}  p90 {np.percentile(nn_new,90):.4f}")
        print(f"  nearest-neighbour cosine, v2 -> v2 (self) : mean {nn_v2.mean():.4f}  p10 {np.percentile(nn_v2,10):.4f}  p90 {np.percentile(nn_v2,90):.4f}")

        for split in A.EVAL_SPLITS:
            Ue = unit(eval_x[split])
            nn_e_to_rt = (Ue @ unit(rt_x[np.array(keep[3], dtype=int)]).T).max(1)
            nn_e_to_v2 = (Ue @ U_v2.T).max(1)
            nn_e_to_e = Ue @ Ue.T
            np.fill_diagonal(nn_e_to_e, -1)
            arm_out["coverage"][split] = {
                "eval_to_v3": float(nn_e_to_rt.mean()),
                "eval_to_v2": float(nn_e_to_v2.mean()),
                "eval_to_self": float(nn_e_to_e.max(1).mean()),
            }
            print(f"  {split:24s} eval->v3 {nn_e_to_rt.mean():.4f}   eval->v2 {nn_e_to_v2.mean():.4f}   eval->own-split {nn_e_to_e.max(1).mean():.4f}")

        print("\n=== 6. SOURCE vs GENERATED CONFOUND ===", flush=True)
        idx3 = np.array(keep[3], dtype=int)
        gen = rt_gen[idx3].astype(float)
        lab = rt_y[idx3]
        agree = float(max((gen == lab).mean(), (gen != lab).mean()))
        sc_g, clf_g = fit_logistic(rt_x[idx3], gen, 0.03)
        w_gen = clf_g.coef_.ravel() / sc_g.scale_
        s_in = score(sc_g, clf_g, rt_x[idx3])
        arm_out["confound"] = {
            "label_provenance_agreement": agree,
            "source_vs_generated_auroc_insample": float(roc_auc_score(gen, s_in)),
            "cos_w3_wprovenance": cos(weights[3], w_gen),
        }
        print(f"  P(label predictable from provenance alone) = {agree:.1%}")
        print(f"  source-vs-generated separability (in-sample AUROC) = {arm_out['confound']['source_vs_generated_auroc_insample']:.4f}")
        print(f"  cos(w_v3, w_provenance) = {arm_out['confound']['cos_w3_wprovenance']:+.4f}")
        # what the provenance direction alone scores on eval
        for split in A.EVAL_SPLITS:
            a = roc_auc_score(eval_y[split], score(sc_g, clf_g, eval_x[split]))
            arm_out["confound"][f"provenance_auroc_{split}"] = float(a)
            print(f"    provenance direction alone on {split:24s} {a:.4f}")

        report["arms"][arm] = arm_out
        del rt_x
        gc.collect()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
