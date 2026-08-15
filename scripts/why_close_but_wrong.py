"""If iteration 3's samples sit that close to iteration 2's, why was the probe trained
on iteration 2 confidently wrong on them?

``why_iter3_null.py`` reported nearest-neighbour cosine 0.9432 from the new-in-v3 rows to
the v2 set, against v2's own self-NN of 0.9728 -- and the same rows were red-team
successes against ``probe_iter2``, at median score 0.777 (false positives) and 0.037
(false negatives). Those two facts are only in tension if cosine proximity in the pooled
activation space implies the probe treats two conversations alike. This script tests
whether it does.

Five measurements, all on activations already on disk:

A. **What the cosine scale even is.** Everything in an LLM residual stream shares a large
   mean component, so 0.94 is not "almost the same point" unless the spread says so.
   Calibrates against within-class, between-class and random-pair cosines.
B. **The contrastive pair itself.** ``preprocessing`` mints each generated counterpart by
   editing its source toward the opposite label. If a source and its own opposite-label
   counterpart are as close as 0.94, then a 0.94 neighbour carries no label information
   at all, and the training set is class-mixed at exactly that scale by construction.
C. **What the nearest v2 neighbour's label is** for each new-in-v3 success.
D. **k-NN in this space, fitted on v2, predicting the new rows.** If a nearest-neighbour
   classifier is also wrong on them, the probe is not doing anything strange -- proximity
   is simply not predictive here.
E. **Distance in the probe's own metric.** ``probe_iter2``'s logit for each new-in-v3
   success against the logit of its nearest v2 neighbour. The probe reads one direction
   through a softmax pooling; two conversations can be neighbours in the full space and
   far apart along that one direction.

Usage:
    .venv_claude/bin/python scripts/why_close_but_wrong.py
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
import attribution_vintage as V
from why_iter3_null import _pool, eval_features


def load_rows(arm: str, iteration: int = 3):
    """Pooled vector + ``probe_iter2`` logit for every row of the iteration-3 dump."""
    ds = A.load_redteam_dataset(arm, iteration)
    gen2src = V._generated_to_source(arm)
    probe2 = A.load_probe(A.ARMS[arm] / "probe_iter2.pkl")
    w, b, T = A.probe_params(probe2)
    w = w.cuda() if torch.cuda.is_available() else w

    pooled, logits, src_keys, is_gen = [], [], [], []
    for msgs in ds.inputs:
        blob = torch.load(A.redteam_blob_path(msgs), weights_only=False)
        acts, mask = blob["activations"], blob["attention_mask"]
        pooled.append(_pool(acts, mask)[0])
        h = acts.float().to(w.device)
        mk = mask.bool().to(w.device)
        h = h * mk[..., None]
        z = (h @ w + b).masked_fill(~mk, 0.0)
        p = torch.softmax(z.masked_fill(~mk, float("-inf")) / T, dim=-1)
        logits.append(float((z * p).sum(-1).item()))
        key = A.canon(msgs)
        src_keys.append(gen2src.get(key, key))
        is_gen.append(key in gen2src)
        del blob, h, mk, z, p
    y = np.array(
        [1.0 if l == "positive" else 0.0 for l in ds.other_fields["labels"]], dtype=np.float32
    )
    return (np.stack(pooled), y, np.array(logits, dtype=np.float32),
            np.array(src_keys), np.array(is_gen), ds)


def unit(m):
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def q(a):
    a = np.asarray(a)
    return f"mean {a.mean():.4f}  p10 {np.percentile(a,10):.4f}  p50 {np.median(a):.4f}  p90 {np.percentile(a,90):.4f}"


def main() -> None:
    out = {}
    for arm in ("gptoss120b", "deepseekv4pro"):
        print(f"\n########## {arm} ##########", flush=True)
        X, y, logit2, src, is_gen, ds = load_rows(arm)
        keep, _ = V.vintages(arm, 3)
        idx3 = np.array(keep[3], dtype=int)
        v2 = np.array(keep[2], dtype=int)
        new = np.setdiff1d(idx3, v2)
        U = unit(X)
        arm_out = {}

        print("\n=== A. what the cosine scale is in this space ===", flush=True)
        rng = np.random.default_rng(0)
        rand_i = rng.choice(v2, 4000)
        rand_j = rng.choice(v2, 4000)
        keep_m = rand_i != rand_j
        rand_cos = (U[rand_i[keep_m]] * U[rand_j[keep_m]]).sum(1)
        same = (U[rand_i[keep_m]] * U[rand_j[keep_m]]).sum(1)[y[rand_i[keep_m]] == y[rand_j[keep_m]]]
        diff = (U[rand_i[keep_m]] * U[rand_j[keep_m]]).sum(1)[y[rand_i[keep_m]] != y[rand_j[keep_m]]]
        print(f"  random v2 pair                       {q(rand_cos)}")
        print(f"  random v2 pair, SAME label           {q(same)}")
        print(f"  random v2 pair, OPPOSITE label       {q(diff)}")
        arm_out["scale"] = {"random": float(rand_cos.mean()),
                            "same_label": float(same.mean()),
                            "opp_label": float(diff.mean())}

        print("\n=== B. a source vs its OWN opposite-label counterpart ===", flush=True)
        by_src = {}
        for i in idx3:
            by_src.setdefault(src[i], {})[bool(is_gen[i])] = i
        pair_cos = []
        for k, d in by_src.items():
            if True in d and False in d:
                pair_cos.append(float(U[d[False]] @ U[d[True]]))
        pair_cos = np.array(pair_cos)
        print(f"  source <-> its generated counterpart {q(pair_cos)}   (n={len(pair_cos)})")
        print(f"  ... these two rows carry OPPOSITE labels by construction.")
        arm_out["pair_cosine"] = {"mean": float(pair_cos.mean()),
                                  "p90": float(np.percentile(pair_cos, 90))}
        nn_new = (unit(X[new]) @ unit(X[v2]).T)
        print(f"  new-in-v3 -> nearest v2 row          {q(nn_new.max(1))}")
        frac = float((pair_cos > np.median(nn_new.max(1))).mean())
        print(f"  -> {frac:.0%} of opposite-label pairs are CLOSER than the median new-row/v2 nearest neighbour")
        arm_out["opp_pairs_closer_than_median_nn"] = frac

        print("\n=== C. label of the nearest v2 neighbour ===", flush=True)
        nn_idx = v2[nn_new.argmax(1)]
        agree = float((y[nn_idx] == y[new]).mean())
        print(f"  nearest v2 neighbour has the SAME label: {agree:.1%} of new-in-v3 rows")
        succ = new[~is_gen[new]]
        nn_s = (unit(X[succ]) @ unit(X[v2]).T)
        nn_si = v2[nn_s.argmax(1)]
        print(f"  ... restricted to the red-team SUCCESSES only ({len(succ)} rows): {float((y[nn_si]==y[succ]).mean()):.1%}")
        arm_out["nn_label_agreement"] = {"all_new": agree,
                                         "successes_only": float((y[nn_si] == y[succ]).mean())}

        print("\n=== D. k-NN fitted on v2, predicting the new rows ===", flush=True)
        from sklearn.metrics import roc_auc_score
        from sklearn.neighbors import KNeighborsClassifier

        knn_out = {}
        for k in (1, 5, 15):
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(X[v2], y[v2])
            acc = float((knn.predict(X[new]) == y[new]).mean())
            acc_s = float((knn.predict(X[succ]) == y[succ]).mean())
            knn_out[k] = {"all_new": acc, "successes_only": acc_s}
            print(f"  k={k:2d}: accuracy on all new-in-v3 rows {acc:.1%}   on the successes only {acc_s:.1%}")
        arm_out["knn"] = knn_out
        print("  (chance is 50% -- the new rows are exactly class-balanced)")

        print("\n=== E. distance in the probe's OWN metric ===", flush=True)
        d_logit = np.abs(logit2[succ] - logit2[nn_si])
        c_succ = nn_s.max(1)
        print(f"  probe_iter2 logit of the success            {q(logit2[succ])}")
        print(f"  probe_iter2 logit of its nearest v2 row     {q(logit2[nn_si])}")
        print(f"  |logit difference| across that 0.94 hop     {q(d_logit)}")
        opp = float((np.sign(logit2[succ]) != np.sign(logit2[nn_si])).mean())
        print(f"  -> {opp:.0%} of successes sit on the OPPOSITE side of the boundary from their nearest v2 neighbour")
        arm_out["probe_metric"] = {
            "mean_abs_logit_gap": float(d_logit.mean()),
            "frac_opposite_side": opp,
            "mean_nn_cosine": float(c_succ.mean()),
        }
        # how far a unit of cosine distance goes, in logits
        gap = 1.0 - c_succ
        print(f"  logit change per unit of cosine distance    {float(np.median(d_logit/np.maximum(gap,1e-6))):.1f}")
        arm_out["probe_metric"]["median_logits_per_cosine"] = float(
            np.median(d_logit / np.maximum(gap, 1e-6))
        )

        out[arm] = arm_out
        del X, U
        gc.collect()

    p = Path("results_hu_harm_gemma27b_batch_ablation/vintage/why_close_but_wrong.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
