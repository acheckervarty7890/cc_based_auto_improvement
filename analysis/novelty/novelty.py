#!/usr/bin/env python
"""Phase 1 -- score how novel each red-team row is relative to the eval manifold.

    .venv_claude/bin/python analysis/novelty/novelty.py --experiment instructions --arm gptoss

Reads ``results/pooled_<experiment>.npz`` and writes ``results/novelty_<exp>_<arm>.npz``
plus a printed summary. Six scores per red-team row, each answering a different
question, because "far from eval" on its own does not distinguish the failure modes
that matter:

``knn_eval``   mean cosine distance to the k nearest EVAL rows. The headline "too far
               from eval space" score. kNN rather than distance-to-centroid because the
               eval set is 4-7 heterogeneous splits: what matters is distance to the
               nearest split's manifold, not to a global mean that sits between them.
``knn_dev``    same against dev. A row far from eval but near dev is a *validation*
               anomaly, not an eval one -- and dev is what early stopping selects on.
``knn_rt``     mean cosine distance to the k nearest OTHER red-team rows.
``relative``   ``knn_eval - knn_rt``. Separates a coherent novel attack cluster (far
               from eval, dense among its own kind -> large) from an isolated one-off
               (far from everything -> near zero). These deserve opposite treatment and
               a single distance cannot tell them apart.
``mahal``      Mahalanobis distance to the eval distribution in PCA space, with
               Ledoit-Wolf shrinkage. Raw-space Mahalanobis is meaningless here (5376
               dims, ~1300 eval rows), so the covariance is estimated after projection.
``outside``    1 if ``knn_eval`` exceeds the p95 of eval's own self-kNN distance -- i.e.
               the row sits further from eval than 95% of eval sits from itself. This is
               the calibrated version of "too far": the threshold comes from the eval
               set's own dispersion instead of being picked by hand.

And the decomposition the plain distance analysis cannot do. The probe is a single
direction ``w`` (``LinearThenSoftmax`` = ``Linear(5376 -> 1)`` + a softmax pool), so a
row's displacement from its local eval neighbourhood splits cleanly:

``along``      component of the displacement along ``w`` -- moves the row along the
               decision axis, so it drags the boundary directly.
``orth``       component orthogonal to ``w`` -- invisible to the current probe's score,
               but it is exactly what *rotates* ``w`` during a refit. A row can be
               enormously novel and score-neutral, and those are the rows that quietly
               change what the probe learns.

Direction only, no fitting, no eval labels are consulted -- so nothing here is selected
on the thing Phase 3 measures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments as X  # noqa: E402

K_DEFAULT = 10
PCA_COMPONENTS = 200


# ------------------------------------------------------------------ helpers


def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.clip(n, 1e-12, None)


def knn_cosine(a: np.ndarray, b: np.ndarray, k: int, exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Mean cosine distance from each row of `a` to its k nearest rows of `b`.

    Returns (mean distance, indices of the k neighbours). Cosine because the pooled
    vectors' norms vary with conversation length, which is a property of the corpus
    rather than of the content.
    """
    A, B = _unit(a.astype(np.float32)), _unit(b.astype(np.float32))
    sim = A @ B.T
    if exclude_self:
        np.fill_diagonal(sim, -np.inf)
    kk = min(k, sim.shape[1] - (1 if exclude_self else 0))
    idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    top = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-top, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    top = np.take_along_axis(top, order, axis=1)
    return (1.0 - top).mean(axis=1), idx


def probe_direction(probe_path: Path) -> tuple[np.ndarray, dict]:
    """Unit weight vector of the probe's linear head (members averaged for ensembles).

    Members are unit-normalised before averaging so a member with a larger weight norm
    does not dominate the mean direction; the pairwise cosine spread is reported so a
    disagreeing ensemble is visible rather than silently averaged away.
    """
    import pickle

    with probe_path.open("rb") as f:
        probe = pickle.load(f)
    members = getattr(probe, "members", None) or [probe]
    ws = []
    for m in members:
        w = m._classifier.model.linear.weight.detach().to(torch_float32()).cpu().numpy().reshape(-1)
        ws.append(w)
    W = _unit(np.stack(ws).astype(np.float32))
    mean_dir = _unit(W.mean(axis=0)[None, :])[0]
    info = {"n_members": len(ws)}
    if len(ws) > 1:
        sims = W @ W.T
        iu = np.triu_indices(len(ws), 1)
        info["member_cos_mean"] = float(sims[iu].mean())
        info["member_cos_min"] = float(sims[iu].min())
    return mean_dir, info


def torch_float32():
    import torch

    return torch.float32


# ------------------------------------------------------------------ scoring


def score_arm(exp: X.Experiment, arm: X.Arm, k: int = K_DEFAULT, n_components: int = PCA_COMPONENTS) -> dict:
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA

    pooled = np.load(X.RESULTS / f"pooled_{exp.key}.npz", allow_pickle=True)
    splits = exp.splits()
    eval_X = np.concatenate([pooled[f"eval::{s}"] for s in splits])
    eval_split = np.concatenate([[s] * len(pooled[f"eval::{s}"]) for s in splits])
    dev_X = pooled["dev"]
    rt_X = pooled[f"rt::{arm.name}"]
    rt_labels = pooled[f"rt::{arm.name}::labels"]
    rt_iter = pooled[f"rt::{arm.name}::first_iter"]
    rt_hash = pooled[f"rt::{arm.name}::hash"]

    # --- distance scores (raw pooled space, cosine) -----------------------------
    knn_eval, nn_idx = knn_cosine(rt_X, eval_X, k)
    knn_dev, _ = knn_cosine(rt_X, dev_X, k)
    knn_rt, _ = knn_cosine(rt_X, rt_X, k, exclude_self=True)
    eval_self, _ = knn_cosine(eval_X, eval_X, k, exclude_self=True)
    dev_to_eval, _ = knn_cosine(dev_X, eval_X, k)
    p95 = float(np.quantile(eval_self, 0.95))
    relative = knn_eval - knn_rt
    outside = (knn_eval > p95).astype(np.int8)

    # --- Mahalanobis in PCA space ----------------------------------------------
    # Fit the projection on eval ALONE: it defines the manifold the red-team rows are
    # being measured against, and including them would rotate the basis toward the very
    # directions we are trying to detect.
    pca = PCA(n_components=min(n_components, eval_X.shape[0] - 1, eval_X.shape[1]), random_state=0)
    E = pca.fit_transform(eval_X)
    R = pca.transform(rt_X)
    lw = LedoitWolf().fit(E)
    mahal = np.sqrt(np.clip(lw.mahalanobis(R), 0, None))
    mahal_eval_self = np.sqrt(np.clip(lw.mahalanobis(E), 0, None))

    # --- probe-direction decomposition -----------------------------------------
    w, winfo = probe_direction(arm.probe_pkl())
    local_mu = eval_X[nn_idx].mean(axis=1)          # local eval neighbourhood centroid
    disp = rt_X - local_mu
    along = disp @ w
    orth = np.linalg.norm(disp - np.outer(along, w), axis=1)
    disp_norm = np.linalg.norm(disp, axis=1)
    # Scale-free version: what fraction of the displacement is on the decision axis.
    along_frac = np.abs(along) / np.clip(disp_norm, 1e-12, None)

    return {
        "hash": rt_hash,
        "labels": rt_labels,
        "first_iter": rt_iter,
        "knn_eval": knn_eval,
        "knn_dev": knn_dev,
        "knn_rt": knn_rt,
        "relative": relative,
        "mahal": mahal,
        "outside": outside,
        "along": along,
        "orth": orth,
        "disp_norm": disp_norm,
        "along_frac": along_frac,
        "nn_split": eval_split[nn_idx[:, 0]],
        "_eval_self_knn": eval_self,
        "_eval_self_p95": np.array([p95]),
        "_dev_to_eval_knn": dev_to_eval,
        "_mahal_eval_self": mahal_eval_self,
        "_probe_members": np.array([winfo["n_members"]]),
        "_probe_dir": w,
    }


def summarise(exp: X.Experiment, arm: X.Arm, s: dict) -> str:
    p95 = float(s["_eval_self_p95"][0])
    lines = [
        f"=== {exp.key} / {arm.name} : {len(s['knn_eval'])} red-team rows ===",
        f"eval self-kNN cosine distance : mean {s['_eval_self_knn'].mean():.4f}  p95 {p95:.4f}",
        f"dev  ->eval kNN               : mean {s['_dev_to_eval_knn'].mean():.4f}  "
        f"outside% {100 * (s['_dev_to_eval_knn'] > p95).mean():5.1f}",
        f"rt   ->eval kNN               : mean {s['knn_eval'].mean():.4f}  "
        f"outside% {100 * s['outside'].mean():5.1f}",
        "",
        f"{'iter':>6} {'n':>5} {'knn_eval':>9} {'knn_dev':>8} {'relative':>9} {'mahal':>8} "
        f"{'outside%':>9} {'|along|':>8} {'orth':>8} {'along_frac':>11}",
    ]
    for it in sorted(set(s["first_iter"].tolist())):
        m = s["first_iter"] == it
        lines.append(
            f"{it:>6} {m.sum():>5} {s['knn_eval'][m].mean():>9.4f} {s['knn_dev'][m].mean():>8.4f} "
            f"{s['relative'][m].mean():>9.4f} {s['mahal'][m].mean():>8.2f} "
            f"{100 * s['outside'][m].mean():>9.1f} {np.abs(s['along'][m]).mean():>8.3f} "
            f"{s['orth'][m].mean():>8.3f} {s['along_frac'][m].mean():>11.3f}"
        )
    lines.append(
        f"{'ALL':>6} {len(s['knn_eval']):>5} {s['knn_eval'].mean():>9.4f} {s['knn_dev'].mean():>8.4f} "
        f"{s['relative'].mean():>9.4f} {s['mahal'].mean():>8.2f} "
        f"{100 * s['outside'].mean():>9.1f} {np.abs(s['along']).mean():>8.3f} "
        f"{s['orth'].mean():>8.3f} {s['along_frac'].mean():>11.3f}"
    )
    # Correlation between novelty and the decision axis: if these are near zero, the
    # far rows are far in directions the current probe cannot see at all.
    c = np.corrcoef(s["knn_eval"], np.abs(s["along"]))[0, 1]
    c2 = np.corrcoef(s["knn_eval"], s["orth"])[0, 1]
    lines += [
        "",
        f"corr(knn_eval, |along w|) = {c:+.3f}   corr(knn_eval, orth) = {c2:+.3f}",
        f"probe direction from {int(s['_probe_members'][0])} member(s)",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=sorted(X.EXPERIMENTS))
    ap.add_argument("--arm", default=None, help="default: every arm of the experiment")
    ap.add_argument("-k", type=int, default=K_DEFAULT)
    ap.add_argument("--pca", type=int, default=PCA_COMPONENTS)
    args = ap.parse_args()

    exp = X.get(args.experiment)
    arms = [exp.arms[args.arm]] if args.arm else list(exp.arms.values())
    for arm in arms:
        s = score_arm(exp, arm, args.k, args.pca)
        print(summarise(exp, arm, s))
        print()
        dest = X.RESULTS / f"novelty_{exp.key}_{arm.name}.npz"
        np.savez_compressed(dest, **s)
        print(f"Saved -> {dest}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
