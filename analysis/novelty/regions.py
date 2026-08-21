#!/usr/bin/env python
"""Phase 2 -- turn per-row novelty scores into named REGIONS.

    .venv_claude/bin/python analysis/novelty/regions.py --experiment instructions --arm gptoss

A scalar threshold ("drop everything past p95") is the wrong instrument for the question
being asked: the far tail of an auto-improvement loop holds both the genuinely
off-distribution junk and the most valuable novel attacks, and a threshold mixes them.
Clustering first lets Phase 3 ablate a *region* at a time and attribute the effect.

Two partitions are produced, because they answer different questions and because on
this data the first one mostly declines to answer:

``--method hdbscan`` (default) finds density clusters over a PCA of the red-team rows
themselves (not of eval -- here we want the structure *within* the attack set), on
cosine-normalised vectors. It labels genuine one-offs as noise (-1) instead of forcing
them into a cluster, which is the honest outcome when there is no cluster; that noise
set is itself worth ablating, since "isolated junk" and "coherent novel family" are
exactly the two cases ``relative`` distinguishes. **On all four arms HDBSCAN puts the
large majority of rows in noise** -- the attack sets are diffuse rather than
family-structured -- so it alone does not give Phase 3 enough to ablate.

``--method kmeans`` therefore provides a complementary forced partition: every row lands
in a region, so the whole set is covered by region-level conditions. It makes no claim
that the regions are natural kinds; it is a covering, and it is read as one.

Writes ``results/regions_<exp>_<arm>.json``: per-region size, novelty stats, label
balance, iteration composition, and the row indices -- plus ``examples`` holding a few
verbatim conversations per region so the regions can be *read* and named rather than
guessed at from centroids.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments as X  # noqa: E402

MIN_CLUSTER_FRAC = 0.04     # a region must hold >= 4% of the arm's rows to be a region
N_EXAMPLES = 3
PCA_COMPONENTS = 50


def _text_of(messages: list[dict], limit: int = 900) -> str:
    out = []
    for m in messages:
        body = str(m.get("content", "")).replace("\n", " ").strip()
        out.append(f"[{m.get('role','?')}] {body}")
    s = "\n".join(out)
    return s[:limit] + (" ..." if len(s) > limit else "")


def cluster_arm(
    exp: X.Experiment,
    arm: X.Arm,
    min_cluster_frac: float = MIN_CLUSTER_FRAC,
    method: str = "hdbscan",
    k: int = 6,
) -> dict:
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    pooled = np.load(X.RESULTS / f"pooled_{exp.key}.npz", allow_pickle=True)
    nov = np.load(X.RESULTS / f"novelty_{exp.key}_{arm.name}.npz", allow_pickle=True)
    Xrt = pooled[f"rt::{arm.name}"].astype(np.float32)
    n = len(Xrt)

    unit = Xrt / np.clip(np.linalg.norm(Xrt, axis=1, keepdims=True), 1e-12, None)
    n_comp = min(PCA_COMPONENTS, n - 1, unit.shape[1])
    Z = PCA(n_components=n_comp, random_state=0).fit_transform(unit)

    min_size = max(5, int(round(min_cluster_frac * n)))
    if method == "kmeans":
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Z)
    else:
        labels = HDBSCAN(min_cluster_size=min_size, metric="euclidean", copy=True).fit_predict(Z)

    rows = [json.loads(l) for l in arm.redteam_jsonl().read_text(encoding="utf-8").splitlines() if l.strip()]

    regions = []
    for c in sorted(set(labels.tolist())):
        m = labels == c
        idx = np.flatnonzero(m)
        # Representatives: the rows closest to the region's centroid, so the examples
        # are typical of the region rather than its outermost edge.
        centre = Z[m].mean(axis=0)
        order = idx[np.argsort(np.linalg.norm(Z[m] - centre, axis=1))]
        regions.append(
            {
                "id": int(c),
                "name": ("noise/isolated" if c == -1 else f"region_{c}"),
                "n": int(m.sum()),
                "knn_eval": float(nov["knn_eval"][m].mean()),
                "relative": float(nov["relative"][m].mean()),
                "mahal": float(nov["mahal"][m].mean()),
                "outside_pct": float(100 * nov["outside"][m].mean()),
                "along_abs": float(np.abs(nov["along"][m]).mean()),
                "orth": float(nov["orth"][m].mean()),
                "along_frac": float(nov["along_frac"][m].mean()),
                "pos_frac": float((nov["labels"][m] == 1).mean()),
                "iter_hist": {str(int(i)): int(((nov["first_iter"] == i) & m).sum())
                              for i in sorted(set(nov["first_iter"].tolist()))},
                "nn_split_mode": str(max(set(nov["nn_split"][m].tolist()),
                                         key=list(nov["nn_split"][m]).count)),
                "indices": [int(i) for i in idx],
                "examples": [
                    {"idx": int(i), "label": rows[i]["label"], "text": _text_of(rows[i]["inputs"])}
                    for i in order[:N_EXAMPLES]
                ],
            }
        )
    regions.sort(key=lambda r: -r["knn_eval"])
    return {
        "experiment": exp.key,
        "arm": arm.name,
        "method": method,
        "n_rows": n,
        "min_cluster_size": min_size,
        "n_regions": len([r for r in regions if r["id"] != -1]),
        "k": k,
        "regions": regions,
    }


def summarise(res: dict) -> str:
    lines = [
        f"=== {res['experiment']} / {res['arm']} : {res['n_rows']} rows -> "
        f"{res['n_regions']} regions ({res['method']}, min_cluster_size={res['min_cluster_size']}) ===",
        f"{'region':>16} {'n':>5} {'knn_eval':>9} {'relative':>9} {'outside%':>9} "
        f"{'orth':>7} {'along_frac':>11} {'pos%':>6} {'nearest eval split':>26}",
    ]
    for r in res["regions"]:
        lines.append(
            f"{r['name']:>16} {r['n']:>5} {r['knn_eval']:>9.4f} {r['relative']:>9.4f} "
            f"{r['outside_pct']:>9.1f} {r['orth']:>7.2f} {r['along_frac']:>11.3f} "
            f"{100 * r['pos_frac']:>6.1f} {r['nn_split_mode']:>26}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=sorted(X.EXPERIMENTS))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--min-cluster-frac", type=float, default=MIN_CLUSTER_FRAC)
    ap.add_argument("--method", default="hdbscan", choices=("hdbscan", "kmeans"))
    ap.add_argument("-k", type=int, default=6, help="k-means region count")
    args = ap.parse_args()

    exp = X.get(args.experiment)
    arms = [exp.arms[args.arm]] if args.arm else list(exp.arms.values())
    for arm in arms:
        res = cluster_arm(exp, arm, args.min_cluster_frac, args.method, args.k)
        print(summarise(res))
        suffix = "" if args.method == "hdbscan" else f"_{args.method}"
        dest = X.RESULTS / f"regions_{exp.key}_{arm.name}{suffix}.json"
        dest.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"Saved -> {dest}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
