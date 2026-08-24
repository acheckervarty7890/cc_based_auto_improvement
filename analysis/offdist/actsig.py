#!/usr/bin/env python
"""Phase 3: does the surface-flagged group have a signature in activation space?

Every row is reduced to one vector — the **mask-weighted mean** of its layer-32 token
activations — and then four questions are asked of it:

* **Is it off the eval manifold?** Distance to the eval centroid, and mean distance to the
  10 nearest eval rows. Calibrated against the eval set's own self-kNN distribution, so
  "outside" means "further from eval than eval's own 95th percentile is", not an arbitrary
  cut.
* **Does the surface score know about it?** Spearman correlation between `p_redteam` (the
  TF-IDF discriminator, text only) and activation-space novelty. If they agree, the cheap
  text measure is a proxy for the expensive one.
* **Does it lie along the probe's decision axis?** The head is `LinearThenSoftmax`: a
  per-token logit `w·h + b` pooled by a softmax over positions, so `w` is the decision
  direction. A row's displacement from the eval centroid is split into its component along
  `w` and the rest. Novelty that is orthogonal to `w` cannot move a score directly — it can
  only act by rotating `w` at the next refit.
* **Is the flagged group linearly separable from the rest?** Out-of-fold AUROC of a logistic
  regression on the pooled vectors. A high value means the surface flag names something the
  representation also encodes, rather than a purely lexical accident.

`w` is averaged over the run's own 10 ensemble members after normalisation; their pairwise
cosine agreement is reported, since averaging directions that disagree would be meaningless.

Reads the same blobs the ceiling analysis reads. No LLM is loaded.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import od_common as O  # noqa: E402

sys.path.insert(0, str(O.REPO / "ceiling_analysis" / "scripts"))
import ca_common as C  # noqa: E402

N_BASE = 50


def pooled(source, idx=None) -> np.ndarray:
    """Mask-weighted mean over real tokens, streamed slab by slab."""
    idx = list(range(len(source))) if idx is None else list(idx)
    out = np.zeros((len(idx), int(source.dim)), dtype=np.float32)
    pos = 0
    for acts, mask, _ids in source.slabs(idx):
        m = mask.to(torch.float32).unsqueeze(-1)
        s = (acts.to(torch.float32) * m).sum(1) / m.sum(1).clamp(min=1)
        out[pos : pos + len(s)] = s.numpy()
        pos += len(s)
    return out


def probe_direction(arm: O.Arm) -> tuple[np.ndarray, float]:
    """Unit decision direction, averaged over the run's ensemble members."""
    sys.path.insert(0, str(O.REPO / "src"))
    from agentic_redteam.ensemble import iter_probe_members

    with (arm.probe_dir / "probe_iter5.pkl").open("rb") as f:
        probe = pickle.load(f)
    ws = []
    for m in iter_probe_members(probe):
        w = m._classifier.model.linear.weight.detach().to(torch.float32).cpu().numpy().ravel()
        ws.append(w / (np.linalg.norm(w) + 1e-12))
    W = np.stack(ws)
    cos = W @ W.T
    iu = np.triu_indices(len(W), 1)
    agreement = float(np.mean(cos[iu])) if len(W) > 1 else 1.0
    w = W.mean(0)
    return w / (np.linalg.norm(w) + 1e-12), agreement


def knn_mean(A: np.ndarray, B: np.ndarray, k: int, exclude_self: bool = False) -> np.ndarray:
    """Mean euclidean distance from each row of A to its k nearest rows of B."""
    out = np.empty(len(A), dtype=np.float64)
    Bn = (B ** 2).sum(1)
    step = 256
    kk = k + (1 if exclude_self else 0)
    for s in range(0, len(A), step):
        a = A[s : s + step]
        d2 = np.maximum((a ** 2).sum(1)[:, None] + Bn[None, :] - 2 * a @ B.T, 0.0)
        part = np.partition(d2, kk - 1, axis=1)[:, :kk]
        part.sort(axis=1)
        if exclude_self:
            part = part[:, 1:]
        out[s : s + len(a)] = np.sqrt(part).mean(1)
    return out


def separability(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y)) < 2:
        return float("nan")
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, oof))


def run_arm(arm: O.Arm, args) -> None:
    from scipy.stats import spearmanr

    concept = C.CONCEPTS[arm.concept]
    flags = [json.loads(l) for l in
             (O.RESULTS / f"flags_{arm.key}.jsonl").read_text().splitlines() if l.strip()]
    p_surface = np.array([f["p_redteam"] for f in flags])
    topic = np.array([f["topic"] for f in flags])

    rt_src = C.redteam_source(concept)
    H_rt = pooled(rt_src, range(N_BASE, len(rt_src)))
    eval_srcs = C.eval_sources(concept)
    ev_parts = {name: pooled(src) for name, src in eval_srcs.items()}
    ev_split_names = list(ev_parts)
    # Which split each eval row came from, kept alongside the pooled vectors so the
    # write-up can plot the eval cloud split by split rather than as one undifferentiated
    # reference blob.
    ev_split = np.concatenate([np.full(len(h), i, dtype=np.int16)
                               for i, h in enumerate(ev_parts.values())])
    H_ev = np.concatenate(list(ev_parts.values()))
    print(f"[{arm.key}] pooled {len(H_rt)} red-team, {len(H_ev)} eval "
          f"({H_rt.shape[1]}-d)", flush=True)

    centroid = H_ev.mean(0)
    d_rt = H_rt - centroid
    d_ev = H_ev - centroid

    knn_rt = knn_mean(H_rt, H_ev, args.k)
    knn_ev = knn_mean(H_ev, H_ev, args.k, exclude_self=True)
    thresh = float(np.quantile(knn_ev, 0.95))
    outside = knn_rt > thresh

    w, agreement = probe_direction(arm)
    proj_rt = d_rt @ w
    proj_ev = d_ev @ w
    norm_rt = np.linalg.norm(d_rt, axis=1)
    norm_ev = np.linalg.norm(d_ev, axis=1)
    orth_frac_ev = (np.sqrt(np.maximum(norm_ev ** 2 - proj_ev ** 2, 0.0))
                    / np.maximum(norm_ev, 1e-12))
    along = np.abs(proj_rt)
    orth = np.sqrt(np.maximum(norm_rt ** 2 - proj_rt ** 2, 0.0))
    orth_frac = orth / np.maximum(norm_rt, 1e-12)

    rho_knn = spearmanr(p_surface, knn_rt).statistic
    rho_dist = spearmanr(p_surface, norm_rt).statistic

    q = np.quantile(p_surface, 1 - args.flag_frac)
    flagged = (p_surface >= q).astype(int)
    sep_flag = separability(H_rt, flagged, seed=args.seed)
    y_rt_ev = np.r_[np.ones(len(H_rt), int), np.zeros(len(H_ev), int)]
    sep_corpus = separability(np.concatenate([H_rt, H_ev]), y_rt_ev, seed=args.seed)

    per_eval_split = {}
    for i, name in enumerate(ev_split_names):
        m = ev_split == i
        per_eval_split[name] = {
            "n": int(m.sum()),
            "abs_proj_on_w": float(np.abs(proj_ev[m]).mean()),
            "orth": float(np.sqrt(np.maximum(
                np.linalg.norm(d_ev[m], axis=1) ** 2 - proj_ev[m] ** 2, 0.0)).mean()),
            "self_knn": float(knn_ev[m].mean()),
        }

    per_topic = {}
    for c in sorted(set(topic.tolist())):
        m = topic == c
        per_topic[int(c)] = {
            "n": int(m.sum()),
            "knn_to_eval": float(knn_rt[m].mean()),
            "outside_frac": float(outside[m].mean()),
            "abs_proj_on_w": float(along[m].mean()),
            "orth_frac": float(orth_frac[m].mean()),
            "mean_p_redteam": float(p_surface[m].mean()),
        }

    summary = {
        "arm": arm.key,
        "n_redteam": int(len(H_rt)), "n_eval": int(len(H_ev)), "dim": int(H_rt.shape[1]),
        "k": args.k,
        "eval_self_knn_p95": thresh,
        "eval_self_knn_median": float(np.median(knn_ev)),
        "redteam_knn_median": float(np.median(knn_rt)),
        "outside_frac": float(outside.mean()),
        "ensemble_direction_agreement_cos": agreement,
        "mean_abs_proj_on_w_redteam": float(along.mean()),
        "mean_abs_proj_on_w_eval": float(np.abs(proj_ev).mean()),
        "mean_orthogonal_fraction": float(orth_frac.mean()),
        # The same fraction for the eval rows against their own centroid. It is the baseline
        # the red-team figure has to be read against: near-total orthogonality to a single
        # direction is what 5376 dimensions hand any row, so the finding is which way the
        # displacement points, not that it is orthogonal.
        "mean_orthogonal_fraction_eval": float(orth_frac_ev.mean()),
        "spearman_p_redteam_vs_knn": float(rho_knn),
        "spearman_p_redteam_vs_centroid_dist": float(rho_dist),
        "separability_flagged_vs_rest_auroc": sep_flag,
        "separability_redteam_vs_eval_auroc": sep_corpus,
        "flag_frac": args.flag_frac,
        "per_topic": per_topic,
        "per_eval_split": per_eval_split,
    }
    O.write_json(O.RESULTS / f"actsig_{arm.key}.json", summary)
    np.savez_compressed(
        O.RESULTS / f"actsig_{arm.key}.npz",
        knn_to_eval=knn_rt, centroid_dist=norm_rt, proj_on_w=proj_rt,
        orth_frac=orth_frac, p_redteam=p_surface, topic=topic,
        # The eval side in the same coordinates, so the reference cloud can be drawn
        # instead of standing in as a single centroid marker.
        proj_on_w_eval=proj_ev, centroid_dist_eval=np.linalg.norm(d_ev, axis=1),
        knn_self_eval=knn_ev, eval_split=ev_split,
        eval_split_names=np.array(ev_split_names, dtype="U64"),
    )
    print(f"[{arm.key}] outside eval's own p95 kNN: {outside.mean():.1%}  "
          f"orthogonal fraction {orth_frac.mean():.1%}  "
          f"rho(surface, knn) {rho_knn:+.3f}  "
          f"separable flagged/rest {sep_flag:.4f}  redteam/eval {sep_corpus:.4f}", flush=True)
    C.free_gpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(O.ARMS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--flag-frac", type=float, default=0.30,
                    help="top fraction by p_redteam treated as the flagged group")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for key in args.arms:
        run_arm(O.ARMS[key], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
