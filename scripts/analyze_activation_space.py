#!/usr/bin/env python
"""Where do red-team samples sit in activation space, relative to dev and eval?

Two questions, asked of the SAME layer-32 gemma-3-27b activations the probe is
trained and scored on:

1. Are the red-team samples far from the dev/eval distribution? If iterative
   retraining is drifting the probe onto a region of input space the evaluation
   never visits, that shows up here as red-team samples sitting off on their own.
2. Do successive iterations form clusters that move further apart? A rotation
   that keeps re-skinning one template produces a tight, drifting blob; one that
   genuinely explores produces overlapping, spread-out ones.

Everything is read from caches that already exist on disk — no extraction model
is loaded and nothing is recomputed:

  * red-team: `<base-cache>/redteam_acts_<model>_L<layer>/<key>.pt`, one blob per
    conversation, keyed by content. Keys are recomputed here with retrain's own
    `_redteam_activation_cache_path`, so a low hit rate means the key inputs have
    drifted and is reported rather than silently ignored.
  * dev: the single `dev_acts_*.pt` blob.
  * eval: `<eval-cache>/<split>-acts_full.pt`, one per split.

Per-sample representation is a **mask-weighted mean over tokens** -> one 5376-d
vector. That is deliberately NOT the probe's own pooling: `LinearThenSoftmax`
weights tokens by the current head, so it changes every iteration and would
confound "did the data move" with "did the probe move". Mean pooling is fixed,
so all iterations are measured on the same ruler.

Usage:
    python scripts/analyze_activation_space.py --out-dir analysis/activation_space
    python scripts/analyze_activation_space.py --stage vectors   # build cache only
    python scripts/analyze_activation_space.py --stage metrics   # reuse cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# This is meant to run ALONGSIDE a live training run. Never touch the GPU (the
# run has ~11 GiB of activations staged there and 1.5 GiB free), and keep the
# thread count low so the fits keep their cores. Both must be set before torch
# is imported anywhere.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# The two arms of experiment17, and the shared content-keyed activation cache
# both of them wrote into.
ARMS = {
    "gptoss120b": {
        "probe_dir": "probes/hu_harm_gemma27b_gptoss120b_batch_ens10_devval",
        "results": "results_hu_harm_gemma27b_gptoss120b_batch_ens10_devval",
        "jsonl_stem": "gptoss120b_probing",
    },
    "deepseekv4pro": {
        "probe_dir": "probes/hu_harm_gemma27b_deepseekv4pro_batch_ens10_devval",
        "results": "results_hu_harm_gemma27b_deepseekv4pro_batch_ens10_devval",
        "jsonl_stem": "deepseekv4pro_probing",
    },
}
BASE_CACHE = REPO / "results_hu_harm_gemma27b_batch_ablation/base_activations"
EVAL_CACHE = REPO / "results_hu_harm_gemma27b_batch_ablation/eval_activations"
DEV_DIR = REPO / "dev_samples/hu_ha"
MODEL = "google/gemma-3-27b-it"
LAYER = 32
# Must match the run's eval: block, or the recomputed cache keys miss.
COMBINE, CONVERT = True, True


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Phase 1 — turn cached activations into one vector per sample
# --------------------------------------------------------------------------- #
def mean_pool(acts, mask):
    """Mask-weighted mean over tokens -> (batch, hidden), float32."""
    import torch

    a = acts.to(torch.float32)
    m = mask.to(torch.float32).unsqueeze(-1)
    return (a * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def pool_blob(path, want_rows=None):
    import torch

    d = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    v = mean_pool(d["activations"], d["attention_mask"])
    if want_rows is not None and v.shape[0] != want_rows:
        raise ValueError(f"{path}: {v.shape[0]} rows, expected {want_rows}")
    return v.numpy()


def as_messages(sample):
    """`sample` is `{"messages": [{role, content}, ...]}`; the cache-key helper
    wants objects with `.role` / `.content`, as `_records_to_labelled_dataset`
    hands it. Iterating the raw dict yields its KEYS, which fails with a confusing
    `'str' object has no attribute 'role'` — hence this."""
    from types import SimpleNamespace

    raw = sample["messages"] if isinstance(sample, dict) else sample
    return [SimpleNamespace(role=m["role"], content=m["content"]) for m in raw]


def redteam_samples():
    """Every red-team conversation, tagged (arm, iteration, error_type, success).

    Read from the attempt JSONLs rather than the postprocessed dumps: the dumps are
    cumulative AND contain LLM-generated contrastive pairs, which are not things the
    attacker found and would blur the question. `success` is kept so successes (what
    actually becomes training data) can be separated from the full attempt stream.
    """
    rows = []
    for arm, cfg in ARMS.items():
        for et in ("fp", "fn"):
            p = REPO / cfg["results"] / f"{cfg['jsonl_stem']}_{et}.jsonl"
            if not p.exists():
                continue
            for line in p.open():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "sample" not in r:
                    continue
                # Only SUCCESSES ever reach the extraction step, so only they have
                # a cached blob. Failures are scored by the probe and discarded.
                if not r.get("success"):
                    continue
                rows.append(
                    {
                        "arm": arm,
                        "iteration": int(r.get("iteration", -1)),
                        "error_type": r.get("error_type", ""),
                        "success": bool(r.get("success", False)),
                        "messages": r["sample"],
                    }
                )
    return rows


def build_vectors(out_dir: Path, skip_eval: bool = False):
    import numpy as np
    from agentic_redteam.retrain import _redteam_activation_cache_path

    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list] = {}
    meta: list[dict] = []
    vecs: list = []

    # ---- red-team ---------------------------------------------------------- #
    rows = redteam_samples()
    log(f"red-team attempts in JSONLs: {len(rows)}")
    hits = misses = 0
    seen: set[str] = set()
    for r in rows:
        try:
            key_path = _redteam_activation_cache_path(
                BASE_CACHE, as_messages(r["messages"]), MODEL, LAYER, COMBINE, CONVERT
            )
        except Exception:
            misses += 1
            continue
        if key_path is None or not key_path.exists():
            misses += 1
            continue
        k = str(key_path)
        if k in seen:          # same conversation reached by both arms / rounds
            continue
        seen.add(k)
        try:
            v = pool_blob(key_path)
        except Exception:
            misses += 1
            continue
        vecs.append(v[0])
        meta.append(
            {
                "group": f"rt_{r['arm']}_iter{r['iteration']}",
                "arm": r["arm"],
                "iteration": r["iteration"],
                "error_type": r["error_type"],
                "success": r["success"],
                "kind": "redteam",
            }
        )
        hits += 1
        if hits % 200 == 0:
            log(f"  pooled {hits} red-team blobs")
    log(f"red-team: {hits} pooled, {misses} without a cached blob "
        f"({hits / max(1, hits + misses):.0%} hit rate)")

    # ---- dev --------------------------------------------------------------- #
    dev_blobs = sorted(BASE_CACHE.glob("dev_acts_*.pt"))
    for b in dev_blobs:
        v = pool_blob(b)
        log(f"dev blob {b.name}: {v.shape[0]} rows")
        for i in range(v.shape[0]):
            vecs.append(v[i])
            meta.append({"group": "dev", "kind": "dev", "iteration": -1,
                         "arm": "", "error_type": "", "success": False})

    # ---- eval -------------------------------------------------------------- #
    if skip_eval:
        log("skipping eval splits (--no-eval): dev alone is the reference")
    for b in [] if skip_eval else sorted(EVAL_CACHE.glob("*-acts_full.pt")):
        split = b.name.replace("-acts_full.pt", "")
        v = pool_blob(b)
        log(f"eval split {split}: {v.shape[0]} rows")
        for i in range(v.shape[0]):
            vecs.append(v[i])
            meta.append({"group": f"eval_{split}", "kind": "eval", "iteration": -1,
                         "arm": "", "error_type": "", "success": False})

    X = np.stack(vecs).astype("float32")
    np.save(out_dir / "vectors.npy", X)
    (out_dir / "meta.json").write_text(json.dumps(meta))
    log(f"saved {X.shape} vectors -> {out_dir/'vectors.npy'}")
    return X, meta


# --------------------------------------------------------------------------- #
# Phase 2 — the actual questions
# --------------------------------------------------------------------------- #
def analyze(X, meta, out_dir: Path):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, silhouette_score
    from sklearn.model_selection import cross_val_score

    out: list[str] = []
    def w(s=""):
        out.append(s)
        print(s, flush=True)

    groups = np.array([m["group"] for m in meta])
    kinds = np.array([m["kind"] for m in meta])
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-9)

    order = sorted(set(groups), key=lambda g: (not g.startswith("rt_"), g))
    w("# Activation-space analysis: red-team vs dev/eval\n")
    w(f"{X.shape[0]} samples, {X.shape[1]}-d mask-mean-pooled layer-{LAYER} activations.\n")
    w("## Group sizes\n")
    w("| group | n |")
    w("| --- | --- |")
    for g in order:
        w(f"| {g} | {(groups == g).sum()} |")

    # --- 1. centroid geometry ---------------------------------------------- #
    cent = {g: Xn[groups == g].mean(0) for g in order}
    for g in cent:
        cent[g] /= max(np.linalg.norm(cent[g]), 1e-9)
    has_eval = bool((kinds == "eval").sum())
    w("\n## 1. Distance from dev / eval\n")
    w("Cosine distance between group centroids, and the group's own spread "
      "(mean cosine distance of its members to its own centroid). A group whose "
      "distance-to-dev is small relative to its own spread is *inside* the dev "
      "distribution, not beside it.\n")
    w("| group | cos-dist to dev centroid | cos-dist to eval centroid | own spread |")
    w("| --- | --- | --- | --- |")
    dev_c = Xn[kinds == "dev"].mean(0); dev_c /= max(np.linalg.norm(dev_c), 1e-9)
    if has_eval:
        ev_c = Xn[kinds == "eval"].mean(0); ev_c /= max(np.linalg.norm(ev_c), 1e-9)
    for g in order:
        Xg = Xn[groups == g]
        spread = float((1 - Xg @ cent[g]).mean())
        ev = f"{1 - float(cent[g] @ ev_c):.4f}" if has_eval else "n/a"
        w(f"| {g} | {1 - float(cent[g] @ dev_c):.4f} | {ev} | {spread:.4f} |")

    # --- 2. nearest-neighbour novelty --------------------------------------- #
    w("\n## 2. Nearest-neighbour novelty\n")
    w("For each sample, cosine distance to its nearest DEV neighbour. The dev row "
      "is dev-to-dev (excluding self) and is the yardstick: red-team groups far "
      "above it occupy regions dev never covers.\n")
    dev_X = Xn[kinds == "dev"]
    w("| group | mean NN-dist to dev | p90 |")
    w("| --- | --- | --- |")
    for g in ["dev"] + [g for g in order if g != "dev"]:
        Xg = Xn[groups == g]
        sims = Xg @ dev_X.T
        if g == "dev":
            np.fill_diagonal(sims, -np.inf)
        nn = 1 - sims.max(1)
        w(f"| {g} | {nn.mean():.4f} | {np.percentile(nn, 90):.4f} |")

    # --- 3. linear separability -------------------------------------------- #
    w("\n## 3. Linear separability (5-fold CV AUC)\n")
    w("How cleanly a linear model tells each red-team group from dev. 0.5 = "
      "indistinguishable, 1.0 = trivially separable.\n")
    w("| group vs dev | AUC |")
    w("| --- | --- |")
    for g in [g for g in order if g.startswith("rt_")]:
        Xg = Xn[groups == g]
        if len(Xg) < 20:
            continue
        Xy = np.vstack([Xg, dev_X])
        y = np.r_[np.ones(len(Xg)), np.zeros(len(dev_X))]
        auc = cross_val_score(
            LogisticRegression(max_iter=2000), Xy, y, cv=5, scoring="roc_auc"
        ).mean()
        w(f"| {g} | {auc:.3f} |")

    # --- 4. do iterations drift apart? -------------------------------------- #
    w("\n## 4. Do the iterations separate from each other?\n")
    for arm in ARMS:
        gs = [g for g in order if g.startswith(f"rt_{arm}_")]
        if len(gs) < 2:
            continue
        w(f"\n### {arm}\n")
        w("Pairwise cosine distance between iteration centroids:\n")
        w("| | " + " | ".join(g.split("_")[-1] for g in gs) + " |")
        w("| --- |" + " --- |" * len(gs))
        for a in gs:
            cells = [f"{1 - float(cent[a] @ cent[b]):.4f}" for b in gs]
            w(f"| {a.split('_')[-1]} | " + " | ".join(cells) + " |")
        mask = np.isin(groups, gs)
        if mask.sum() > 50 and len(set(groups[mask])) > 1:
            sil = silhouette_score(Xn[mask], groups[mask], metric="cosine")
            w(f"\nSilhouette of iteration labels: **{sil:.3f}** "
              "(~0 = iterations overlap; >0.25 = they form distinct clusters)")

    # --- 5. picture ---------------------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        p = PCA(n_components=2).fit(Xn[kinds == "dev"] if not has_eval else Xn[kinds != "redteam"])
        P = p.transform(Xn)
        fig, ax = plt.subplots(figsize=(9, 7))
        for g in order:
            m = groups == g
            rt = g.startswith("rt_")
            ax.scatter(P[m, 0], P[m, 1], s=6 if rt else 4, alpha=0.55 if rt else 0.3,
                       label=f"{g} (n={m.sum()})")
        ax.set_title("Layer-32 activations (mask-mean-pooled), PCA fit on dev+eval")
        ax.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]:.1%})")
        ax.legend(fontsize=6, markerscale=2, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "pca.png", dpi=150)
        w(f"\n![PCA](pca.png)\n")
    except Exception as exc:
        w(f"\n(figure skipped: {exc})")

    (out_dir / "REPORT.md").write_text("\n".join(out) + "\n")
    log(f"wrote {out_dir/'REPORT.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "analysis/activation_space")
    ap.add_argument("--stage", choices=["all", "vectors", "metrics"], default="all")
    ap.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the eval splits (4.3 GB of blobs) and use dev alone as the "
        "reference distribution. This is the cheap pass: dev is one 849 MB file "
        "and the red-team blobs are ~1-3 MB each, so the whole thing is a few GB "
        "of sequential reads instead of 5+.",
    )
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    a = ap.parse_args()

    import torch

    torch.set_num_threads(max(1, a.threads))
    a.out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    if a.stage in ("all", "vectors"):
        X, meta = build_vectors(a.out_dir, skip_eval=a.no_eval)
    if a.stage == "vectors":
        return 0
    if a.stage == "metrics":
        X = np.load(a.out_dir / "vectors.npy")
        meta = json.loads((a.out_dir / "meta.json").read_text())
    analyze(X, meta, a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
