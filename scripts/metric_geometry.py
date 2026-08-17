"""Is there a geometry in which "close to the training set" means "same label"?

``why_close_but_wrong{,_centered}.py`` established the negative result that the rest of
``docs/why_last_iteration_adds_nothing.md`` rests on: in mean-pooled layer-32 activations,
a new-in-v3 red-team success sits as close to the v2 training set (cosine 0.498 centered)
as a *deliberately opposite-label* counterpart sits to its own source (0.494). Proximity
carries no label information there, which is why loop-fix 2 says a novelty guard must not
be built on raw activation cosine, and loop-fix 4's acquisition rule has nothing to
measure "far from the existing training set" with.

Those two fixes both *need* a metric. This script looks for one, by running every
candidate geometry through the same acceptance test, so they are comparable to each other
and to the published numbers.

The candidates, one per suggestion:

**Pooling** (the representation, not the metric on it). Mean pooling averages a decisive
token over up to 1024 positions; the deployed head instead pools with a softmax over its
own per-token logits. ``pool:{mean,last,last32,probe,topz16}``.

**The probe's own readout.** ``probe:logit`` is the sequence logit s of ``probe_iter2``
(1-D — the metric the *definition of success* is written in); ``probe:proj`` is w.x on the
mean pooling; ``probe:wscaled`` rescales each dimension by |w_j|; ``probe:jac`` is the
weight-gradient ds/dw, i.e. two rows are close when they push the probe the same way.

**Supervised metric learning.** ``sup:lda`` (shrinkage Fisher direction), ``sup:winwhite``
(Mahalanobis under the pooled within-class covariance) and ``sup:nca`` (Neighbourhood
Components Analysis, which optimises k-NN accuracy directly), each fit on the v2 rows'
labels, with ``unsup:pca64`` as the same-dimensionality unsupervised control so a gain can
be attributed to the supervision rather than to the projection.

**Text.** ``text:tfidf`` (lexical) and ``text:minilm`` (all-MiniLM-L6-v2 sentence
embeddings, chunked and averaged) — probe-independent, and the only candidates that could
run at submit time without an activation.

**Nonlinear.** ``nl:expstretch`` is exp(8·cos) on the baseline — a *monotone* transform,
included precisely to demonstrate that stretching a distance inflates the headline gap
while leaving every decision-relevant number bit-identical. ``nl:tsne2`` and ``nl:umap``
are the manifold embeddings.

Every candidate is scored on the same five things (see :func:`acceptance`):

    pair_auroc_new  does similarity predict "same label" on rows nothing was fit on?
    hop_auroc       can it tell an own-counterpart hop from a new-row->v2 hop? This is
                    the novelty guard's question: at 0.5 no threshold separates a clone
                    from a genuinely new error, which is what kills raw cosine.
    nn_agree/knn    §5a's nearest-neighbour and k-NN tests, verbatim
    durable_auroc   does "far from the v2 training set" predict the ~40% of successes
                    that EVERY reseeded vintage-2 probe misses (v2probe_on_new_v3.json)?
                    This is loop-fix 4's acquisition question.

Fits use the v2 rows only; every number is read off rows held out from that fit (the
new-in-v3 rows), so a supervised metric cannot win by memorising. ``--cv`` additionally
reports the in-sample pair statistic under 5-fold CV.

Usage:
    .venv_claude/bin/python scripts/metric_geometry_features.py   # once, builds the cache
    .venv_claude/bin/python scripts/metric_geometry.py
    .venv_claude/bin/python scripts/metric_geometry.py --metrics pool:mean sup:nca
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPS_DIR = Path("results_hu_harm_gemma27b_batch_ablation/vintage/metric_reps")
OUT_JSON = Path("results_hu_harm_gemma27b_batch_ablation/vintage/metric_geometry.json")
V2PROBE = Path("results_hu_harm_gemma27b_batch_ablation/vintage/v2probe_on_new_v3.json")
ARMS = ("gptoss120b", "deepseekv4pro")
EVAL_SPLITS = [
    "eval_ai_dilemmas",
    "eval_ant_hh",
    "eval_balanced_refusal",
    "eval_daily_dilemmas",
]
EXP_K = 8.0  # the stretch factor of nl:expstretch
N_PAIRS = 4000
N_BOOT = 400


# --- data -------------------------------------------------------------------------


@dataclass
class ArmData:
    arm: str
    bases: dict[str, np.ndarray]  # pooling name -> [n, d] (or [n] of str for "text")
    y: np.ndarray
    is_gen: np.ndarray
    src_keys: np.ndarray
    v2: np.ndarray  # rows of the iteration-2 vintage
    v3: np.ndarray  # rows of the iteration-3 vintage (everything)
    new: np.ndarray  # v3 \ v2
    succ: np.ndarray  # new rows that are red-team successes (not generated)
    pair_src: np.ndarray  # source rows of the (source, generated) couples in v3
    pair_gen: np.ndarray
    durable: np.ndarray  # over `succ`: missed by all 10 reseeded v2 probes
    w: np.ndarray
    n_tok: np.ndarray
    seq: np.ndarray
    eval_bases: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


def load_arm(arm: str, with_eval: bool = True) -> ArmData:
    z = np.load(REPS_DIR / f"{arm}.npz", allow_pickle=True)
    y = z["y"]
    is_gen = z["is_gen"]
    src = z["src_keys"]
    v2, v3 = z["idx_v2"], z["idx_v3"]
    new = np.setdiff1d(v3, v2)
    succ = new[~is_gen[new]]

    by_src: dict[str, dict[bool, int]] = {}
    for i in v3:
        by_src.setdefault(src[i], {})[bool(is_gen[i])] = int(i)
    couples = [(d[False], d[True]) for d in by_src.values() if True in d and False in d]
    pair_src = np.array([a for a, _ in couples], dtype=int)
    pair_gen = np.array([b for _, b in couples], dtype=int)

    always = {
        int(r["row"])
        for r in json.loads(V2PROBE.read_text())[arm]["always_wrong_source_rows"]
    }
    durable = np.array([int(i) in always for i in succ], dtype=bool)

    import attribution_lib as A

    probe2 = A.load_probe(A.ARMS[arm] / "probe_iter2.pkl")
    w = A.probe_params(probe2)[0].detach().cpu().numpy().astype(np.float32)
    del probe2
    gc.collect()

    bases = {p: z[f"X_{p}"] for p in ("mean", "last", "last32", "probe", "topz16")}
    bases["jac"] = z["jac"]
    bases["seq"] = z["seq_logit"].astype(np.float32)[:, None]
    bases["text"] = z["texts"]

    eval_bases: dict[str, dict[str, np.ndarray]] = {}
    if with_eval and (REPS_DIR / "eval.npz").exists():
        ze = np.load(REPS_DIR / "eval.npz", allow_pickle=True)
        for split in EVAL_SPLITS:
            eval_bases[split] = {
                "mean": ze[f"X_mean_{split}"],
                "probe": ze[f"X_probe_{split}"],
                "text": ze[f"texts_{split}"],
                "_y": ze[f"y_{split}"],
            }

    return ArmData(
        arm=arm, bases=bases, y=y, is_gen=is_gen, src_keys=src, v2=v2, v3=v3, new=new,
        succ=succ, pair_src=pair_src, pair_gen=pair_gen, durable=durable, w=w,
        n_tok=z["n_tok"], seq=z["seq_logit"].astype(np.float32), eval_bases=eval_bases,
    )


# --- similarity -------------------------------------------------------------------


def _unit(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def sim_pairs(Z: np.ndarray, kind: str, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Similarity of the paired rows ``(i[k], j[k])`` — higher means more alike."""
    if kind.startswith("cosine"):
        U = _unit(Z)
        s = (U[i] * U[j]).sum(1)
        return np.exp(EXP_K * s) if kind == "cosine_exp" else s
    return -np.linalg.norm(Z[i] - Z[j], axis=1)


def sim_mat(Z: np.ndarray, kind: str, rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
    if kind.startswith("cosine"):
        U = _unit(Z)
        S = U[rows_a] @ U[rows_b].T
        return np.exp(EXP_K * S) if kind == "cosine_exp" else S
    from scipy.spatial.distance import cdist

    return -cdist(Z[rows_a], Z[rows_b]).astype(np.float32)


# --- metric registry ----------------------------------------------------------------
# A metric is (base, fit) where fit(X_fit, y_fit) -> (transform, kind). `transform` may
# be applied to any matrix in the same base, which is what lets a metric also be measured
# on the eval splits. Fits only ever see the v2 rows.


def _identity(kind: str):
    def fit(X, y):
        return (lambda M: M), kind

    return fit


def _fit_centered(X, y):
    mu = X.mean(0, keepdims=True)
    return (lambda M: M - mu), "cosine"


def _fit_whitened(X, y):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    return (lambda M: (M - mu) / sd), "cosine"


def _fit_pca(k: int, whiten: bool, kind: str = "euclidean"):
    def fit(X, y):
        from sklearn.decomposition import PCA

        p = PCA(n_components=min(k, X.shape[0] - 1, X.shape[1]), whiten=whiten,
                random_state=0).fit(X)
        return (lambda M: p.transform(M).astype(np.float32)), kind

    return fit


def _fit_wproj(w: np.ndarray):
    def fit(X, y):
        return (lambda M: (M @ w)[:, None].astype(np.float32)), "euclidean"

    return fit


def _fit_wscaled(w: np.ndarray):
    a = np.abs(w)[None, :]

    def fit(X, y):
        return (lambda M: M * a), "cosine"

    return fit


def _fit_lda(k: int = 128):
    """Shrinkage Fisher direction, fitted inside a PCA subspace.

    Full-dimensional LDA would estimate a 5376x5376 covariance from ~430 rows — hopelessly
    rank-deficient, and an O(d^3) solve that dominates the whole sweep. The PCA prefix is
    the standard remedy and matches ``sup:winwhite``, so the two supervised metrics differ
    only in what they do with the covariance.
    """

    def fit(X, y):
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        p = PCA(n_components=min(k, X.shape[0] - 1, X.shape[1]), random_state=0).fit(X)
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
            p.transform(X), y
        )
        return (
            lambda M: m.decision_function(p.transform(M)).astype(np.float32)[:, None]
        ), "euclidean"

    return fit


def _fit_winwhite(k: int = 128, shrink: float = 0.1):
    """Mahalanobis under the pooled *within-class* covariance.

    The classical supervised metric: divide out the directions along which same-label
    rows already vary, so what is left is the between-class structure. Shrunk toward a
    scaled identity because 128 dims are estimated from ~430 rows.
    """

    def fit(X, y):
        from sklearn.decomposition import PCA

        p = PCA(n_components=min(k, X.shape[0] - 1, X.shape[1]), random_state=0).fit(X)
        Z = p.transform(X)
        C = np.zeros((Z.shape[1], Z.shape[1]))
        for c in np.unique(y):
            M = Z[y == c]
            M = M - M.mean(0, keepdims=True)
            C += M.T @ M
        C /= max(len(Z) - len(np.unique(y)), 1)
        C = (1 - shrink) * C + shrink * (np.trace(C) / C.shape[0]) * np.eye(C.shape[0])
        ev, EV = np.linalg.eigh(C)
        W = (EV * np.maximum(ev, 1e-8) ** -0.5) @ EV.T
        return (lambda M: (p.transform(M) @ W).astype(np.float32)), "euclidean"

    return fit


def _fit_nca(pca_k: int = 64, out_k: int = 32):
    def fit(X, y):
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NeighborhoodComponentsAnalysis

        p = PCA(n_components=min(pca_k, X.shape[0] - 1, X.shape[1]), random_state=0).fit(X)
        Z = p.transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n = NeighborhoodComponentsAnalysis(
                n_components=min(out_k, Z.shape[1]), random_state=0, max_iter=200
            ).fit(Z, y)
        return (lambda M: n.transform(p.transform(M)).astype(np.float32)), "euclidean"

    return fit


def _fit_tfidf(X, y):
    from sklearn.feature_extraction.text import TfidfVectorizer

    v = TfidfVectorizer(
        min_df=2, max_features=20000, ngram_range=(1, 2), sublinear_tf=True
    ).fit(list(X))

    def transform(M):
        return np.asarray(v.transform(list(M)).todense(), dtype=np.float32)

    return transform, "cosine"


def build_registry(d: ArmData) -> dict[str, dict]:
    """``name -> {base, fit, supervised, transductive}``."""
    reg: dict[str, dict] = {}

    def add(name, base, fit, supervised=False, transductive=False):
        reg[name] = {"base": base, "fit": fit, "supervised": supervised,
                     "transductive": transductive}

    for p in ("mean", "last", "last32", "probe", "topz16"):
        add(f"pool:{p}", p, _identity("cosine"))
    add("lin:centered", "mean", _fit_centered)
    add("lin:whitened", "mean", _fit_whitened)
    add("lin:pcawhite", "mean", _fit_pca(128, whiten=True))
    add("unsup:pca64", "mean", _fit_pca(64, whiten=False))

    add("probe:logit", "seq", _identity("euclidean"))
    add("probe:proj", "mean", _fit_wproj(d.w))
    add("probe:wscaled", "mean", _fit_wscaled(d.w))
    add("probe:jac", "jac", _identity("cosine"))

    add("sup:lda", "mean", _fit_lda(), supervised=True)
    add("sup:winwhite", "mean", _fit_winwhite(), supervised=True)
    add("sup:nca", "mean", _fit_nca(), supervised=True)

    add("text:tfidf", "text", _fit_tfidf)
    if "minilm" in d.bases:
        add("text:minilm", "minilm", _identity("cosine"))

    add("nl:expstretch", "mean", _identity("cosine_exp"))
    add("nl:tsne2", "mean", _fit_tsne(), transductive=True)
    try:
        import umap  # noqa: F401

        add("nl:umap", "mean", _fit_umap())
    except Exception:
        pass
    return reg


def _fit_tsne():
    """t-SNE has no out-of-sample transform, so it must be fit on the rows it embeds.

    The transform therefore ignores its argument and returns the joint embedding, and the
    metric is marked transductive: a guard built on it would have to re-embed the whole
    corpus per candidate, which is why the flag is reported rather than hidden.
    """

    def fit(X, y):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        def transform(M):
            # sklearn's own guidance: reduce to ~50 dims first, or the neighbour graph is
            # built in 5376 dimensions where all distances are nearly equal
            P = PCA(n_components=min(50, M.shape[0] - 1, M.shape[1]), random_state=0)
            return (
                TSNE(
                    n_components=2, random_state=0, init="pca",
                    perplexity=min(30, max(5, len(M) // 4)),
                )
                .fit_transform(P.fit_transform(M))
                .astype(np.float32)
            )

        return transform, "euclidean"

    return fit


def _fit_umap(k: int = 16):
    def fit(X, y):
        import umap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = umap.UMAP(n_components=k, random_state=0, n_neighbors=15).fit(X)
        return (lambda M: m.transform(M).astype(np.float32)), "euclidean"

    return fit


# --- the acceptance test -------------------------------------------------------------


def _auroc(labels, scores) -> float:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _boot_auroc(labels, scores, n=N_BOOT, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    labels, scores = np.asarray(labels), np.asarray(scores)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue
        vals.append(_auroc(labels[idx], scores[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pair_stats(d: ArmData, Z, kind, rows, n_pairs=N_PAIRS, seed=0, attr=None) -> dict:
    """Does similarity predict "these two rows share an attribute" (the label by default)?

    ``auroc`` is the monotone-invariant form of the note's same-minus-opposite gap: it is
    unchanged by any stretch of the distance, which the raw ``delta`` is not. Passing
    ``attr=d.is_gen`` asks the same question of provenance instead.
    """
    attr = d.y if attr is None else attr
    rng = np.random.default_rng(seed)
    i, j = rng.choice(rows, n_pairs), rng.choice(rows, n_pairs)
    m = i != j
    i, j = i[m], j[m]
    s = sim_pairs(Z, kind, i, j)
    same = (attr[i] == attr[j]).astype(int)
    return {
        "same_label": float(s[same == 1].mean()),
        "opp_label": float(s[same == 0].mean()),
        "delta": float(s[same == 1].mean() - s[same == 0].mean()),
        "auroc": _auroc(same, s),
        "n_pairs": int(len(i)),
    }


def knn_acc(d: ArmData, Z, kind, train, test, ks=(1, 5, 15)) -> dict:
    """k-NN label transfer, reported three ways because raw accuracy is misleading here.

    §5a reads its k-NN numbers against a 50% chance line, on the stated grounds that "the
    new rows are exactly class-balanced". The 232 new *rows* are (each success is paired
    with an opposite-label counterpart), but the test set there is the 116 **successes**,
    which are 71% positive for gptoss and 83% for deepseek — because 71%/83% of them are
    false negatives, whose true label is the positive class. So the constant classifier
    scores 71%/83% and an accuracy of 52% is far *below* chance rather than at it.

    ``bal_k`` (mean per-class recall) and ``auroc_k`` (over the fraction of positive
    neighbours) are both immune to that, so they are what the report reads.
    """
    S = sim_mat(Z, kind, test, train)
    order = np.argsort(-S, axis=1)
    ytr, yte = d.y[train], d.y[test]
    out: dict = {
        "positive_rate_test": float(yte.mean()),
        "positive_rate_train": float(ytr.mean()),
        "majority_baseline": float(max(yte.mean(), 1 - yte.mean())),
    }
    for k in ks:
        lab = ytr[order[:, :k]]
        frac = lab.mean(1)
        pred = (frac > 0.5).astype(float)
        tie = frac == 0.5
        pred[tie] = lab[tie, 0]
        out[str(k)] = float((pred == yte).mean())
        out[f"bal_{k}"] = float(
            np.mean([float((pred[yte == c] == c).mean()) for c in (0.0, 1.0) if (yte == c).any()])
        )
        out[f"auroc_{k}"] = _auroc(yte.astype(int), frac)
    return out


def _nn_agreement(d: ArmData, Z, kind, ref: np.ndarray, test: np.ndarray) -> dict:
    """Nearest-neighbour label agreement, with the same imbalance correction."""
    S = sim_mat(Z, kind, test, ref)
    nn_idx = ref[S.argmax(1)]
    agree = d.y[nn_idx] == d.y[test]
    yte = d.y[test]
    return {
        "agreement": float(agree.mean()),
        "balanced": float(
            np.mean([float(agree[yte == c].mean()) for c in (0.0, 1.0) if (yte == c).any()])
        ),
        "majority_baseline": float(max(yte.mean(), 1 - yte.mean())),
    }


def acceptance(d: ArmData, Z: np.ndarray, kind: str) -> dict:
    """Every §5a measurement, plus the two the loop fixes actually need."""
    out: dict = {}
    out["pair_v2"] = pair_stats(d, Z, kind, d.v2)
    out["pair_new"] = pair_stats(d, Z, kind, d.new, n_pairs=min(N_PAIRS, 4 * len(d.new) ** 2))

    own = sim_pairs(Z, kind, d.pair_src, d.pair_gen)
    S = sim_mat(Z, kind, d.succ, d.v2)
    nn = S.max(1)
    nn_idx = d.v2[S.argmax(1)]

    lab = np.r_[np.ones(len(own)), np.zeros(len(nn))]
    sc = np.r_[own, nn]
    lo, hi = _boot_auroc(lab, sc)
    out["hop"] = {
        "own_pair_sim": float(own.mean()),
        "new_to_v2_nn_sim": float(nn.mean()),
        "auroc": _auroc(lab, sc),
        "ci95": [lo, hi],
        "n_own": int(len(own)),
        "n_new": int(len(nn)),
    }

    # The unpaired form above inherits a confound from §5a: `own` ranges over every v3
    # couple while `nn` is a maximum over 546 v2 rows, and a nearest-neighbour similarity
    # grows with the size of the set searched. The paired form fixes both — same anchor
    # rows, and each anchor's own counterpart against that same anchor's nearest v2 row.
    src2gen = dict(zip(d.pair_src.tolist(), d.pair_gen.tolist()))
    pos = {int(s): k for k, s in enumerate(d.succ)}
    anchors = [int(s) for s in d.succ if int(s) in src2gen]
    if anchors:
        own_a = sim_pairs(
            Z, kind, np.array(anchors), np.array([src2gen[s] for s in anchors])
        )
        nn_a = nn[[pos[s] for s in anchors]]
        lo2, hi2 = _boot_auroc(
            np.r_[np.ones(len(own_a)), np.zeros(len(nn_a))], np.r_[own_a, nn_a]
        )
        out["hop_paired"] = {
            "n_anchors": len(anchors),
            "own_pair_sim": float(own_a.mean()),
            "nn_sim": float(nn_a.mean()),
            "frac_own_closer": float((own_a > nn_a).mean()),
            "auroc": _auroc(
                np.r_[np.ones(len(own_a)), np.zeros(len(nn_a))], np.r_[own_a, nn_a]
            ),
            "ci95": [lo2, hi2],
        }
    # Does the metric see the SCENARIO, independently of the label? A novelty guard needs
    # both: a re-skin of a stored success (same scenario, same label) must come out close,
    # and an opposite-label rewrite must not. The couples give the second half directly;
    # this gives the first half's prerequisite, by asking whether a source's own
    # counterpart — same scenario, opposite label — is at least closer than an unrelated
    # row. It is also what exposes the 1-D metrics: along a single direction two rows can
    # be adjacent while sharing nothing, so `probe:logit` and `sup:lda` clear the hop bar
    # for a reason that has nothing to do with content.
    rng = np.random.default_rng(1)
    ri, rj = rng.choice(d.v3, N_PAIRS), rng.choice(d.v3, N_PAIRS)
    cross = (ri != rj) & (d.src_keys[ri] != d.src_keys[rj])
    rand_sim = sim_pairs(Z, kind, ri[cross], rj[cross])
    lab_s = np.r_[np.ones(len(own)), np.zeros(len(rand_sim))]
    out["scenario"] = {
        "auroc": _auroc(lab_s, np.r_[own, rand_sim]),
        "own_pair_sim": float(own.mean()),
        "random_cross_scenario_sim": float(rand_sim.mean()),
        "n_random": int(len(rand_sim)),
    }

    out["nn"] = _nn_agreement(d, Z, kind, d.v2, d.succ)
    out["nn_label_agreement"] = out["nn"]["agreement"]  # §5a's column, verbatim
    out["knn"] = knn_acc(d, Z, kind, d.v2, d.succ)

    # acquisition: does distance from the training set flag the durable holes?
    dist = -nn
    lo, hi = _boot_auroc(d.durable.astype(int), dist)
    out["durable"] = {
        "auroc": _auroc(d.durable.astype(int), dist),
        "ci95": [lo, hi],
        "n_durable": int(d.durable.sum()),
        "n_succ": int(len(d.succ)),
    }

    # --- provenance controls -----------------------------------------------------
    # The doc's "design smell": every training pair is one attacker-written row plus one
    # gpt-5.1-written counterpart, that axis is 99.9% linearly decodable, and because 71%
    # of successes are false negatives it predicts the LABEL 69-70% of the time. So a
    # metric can score well above chance on the label tests by tracking authorship
    # instead. Two checks: how strongly the metric sees provenance at all, and every
    # neighbour test redone against the source-written half of v2 only — where the
    # successes and the reference rows share an author and the shortcut is unavailable.
    out["pair_provenance"] = pair_stats(d, Z, kind, d.v2, attr=d.is_gen)
    v2_src = d.v2[~d.is_gen[d.v2]]
    if len(v2_src) >= 15:
        Ss = sim_mat(Z, kind, d.succ, v2_src)
        nn_s = Ss.max(1)
        nn_s_idx = v2_src[Ss.argmax(1)]
        lo3, hi3 = _boot_auroc(d.durable.astype(int), -nn_s)
        out["source_only"] = {
            "n_v2_source": int(len(v2_src)),
            "nn": _nn_agreement(d, Z, kind, v2_src, d.succ),
            "nn_label_agreement": float((d.y[nn_s_idx] == d.y[d.succ]).mean()),
            "knn": knn_acc(d, Z, kind, v2_src, d.succ),
            "durable_auroc": _auroc(d.durable.astype(int), -nn_s),
            "durable_ci95": [lo3, hi3],
        }
    return out


def pair_stats_cv(d: ArmData, spec: dict, folds: int = 5) -> dict:
    """In-sample pair statistic done honestly: fit on 4 folds of v2, measure on the 5th."""
    from sklearn.model_selection import StratifiedKFold

    base = d.bases[spec["base"]]
    rows = d.v2
    aurocs, deltas = [], []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(
        np.zeros(len(rows)), d.y[rows]
    ):
        transform, kind = spec["fit"](base[rows[tr]], d.y[rows[tr]])
        Z = transform(base)
        st = pair_stats(d, Z, kind, rows[te], n_pairs=2000)
        aurocs.append(st["auroc"])
        deltas.append(st["delta"])
        del Z
    return {"auroc": float(np.mean(aurocs)), "auroc_sd": float(np.std(aurocs, ddof=1)),
            "delta": float(np.mean(deltas))}


def eval_proximity(d: ArmData, spec: dict, transform, kind: str) -> dict | None:
    """§5's question re-asked in this metric: do the new pairs move the training set
    toward the eval distribution?

    Size-matched, because v2 has ~4x the rows of new-in-v3 and a nearest-neighbour
    similarity grows with the size of the set searched. 20 draws of |new| rows from v2.
    """
    if spec["base"] not in ("mean", "probe", "text") or spec["transductive"]:
        return None
    Zrt = transform(d.bases[spec["base"]])
    rng = np.random.default_rng(0)
    out = {}
    for split in EVAL_SPLITS:
        Ze = transform(d.eval_bases[split][spec["base"]])
        # stack so sim_mat can index one array
        Z = np.concatenate([Zrt, Ze]) if Zrt.ndim == 2 else np.concatenate([Zrt, Ze])
        ev = np.arange(len(Zrt), len(Z))
        to_new = sim_mat(Z, kind, ev, d.succ).max(1)
        draws = []
        for _ in range(20):
            sub = rng.choice(d.v2, size=len(d.succ), replace=False)
            draws.append(sim_mat(Z, kind, ev, sub).max(1).mean())
        out[split] = {
            "eval_to_new_v3": float(to_new.mean()),
            "eval_to_v2_sizematched": float(np.mean(draws)),
            "sd_over_draws": float(np.std(draws, ddof=1)),
        }
        del Ze, Z
    return out


# --- text embeddings -----------------------------------------------------------------


def add_minilm(d: ArmData, model="sentence-transformers/all-MiniLM-L6-v2") -> None:
    """Mean-pooled MiniLM embeddings, cached to disk.

    The conversations run to thousands of characters against MiniLM's 256-token training
    window, so each is embedded in 256-token windows and the windows averaged — truncating
    to the first window would discard exactly the assistant turn that carries the label.
    """
    cache = REPS_DIR / f"{d.arm}_minilm.npy"
    cache_eval = REPS_DIR / "eval_minilm.npz"
    if cache.exists() and cache_eval.exists():
        d.bases["minilm"] = np.load(cache)
        ze = np.load(cache_eval)
        for split in EVAL_SPLITS:
            if split in d.eval_bases:
                d.eval_bases[split]["minilm"] = ze[split]
        return

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    mod = AutoModel.from_pretrained(model).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = mod.to(dev)

    @torch.no_grad()
    def embed(texts) -> np.ndarray:
        out = np.empty((len(texts), mod.config.hidden_size), dtype=np.float32)
        for i, t in enumerate(texts):
            enc = tok(
                str(t),
                truncation=True,
                max_length=256,
                return_overflowing_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(dev)
            am = enc["attention_mask"].to(dev)
            h = mod(input_ids=ids, attention_mask=am).last_hidden_state
            m = am[..., None].float()
            # mean-pool each 256-token window, then average the windows
            v = ((h * m).sum(1) / m.sum(1).clamp(min=1)).mean(0).cpu().numpy()
            out[i] = v / (np.linalg.norm(v) + 1e-12)
        return out

    d.bases["minilm"] = embed(d.bases["text"])
    np.save(cache, d.bases["minilm"])
    if not cache_eval.exists() and d.eval_bases:
        payload = {s: embed(d.eval_bases[s]["text"]) for s in EVAL_SPLITS}
        np.savez(cache_eval, **payload)
    if cache_eval.exists():
        ze = np.load(cache_eval)
        for split in EVAL_SPLITS:
            if split in d.eval_bases:
                d.eval_bases[split]["minilm"] = ze[split]
    del mod
    gc.collect()


# --- driver ---------------------------------------------------------------------------


def run_arm(d: ArmData, only: list[str] | None, do_cv: bool, do_eval: bool) -> dict:
    reg = build_registry(d)
    names = [n for n in reg if not only or n in only]
    results = {}
    print(
        f"\n{'metric':<16} {'pairAUR':>7} {'scenAUR':>7} {'provAUR':>7} {'hopPair':>7} "
        f"{'nnBal':>6} {'kNN15aur':>8} {'srcKNN5bal':>10} "
        f"{'durAUR':>7} {'srcDur':>7}",
        flush=True,
    )
    for name in names:
        spec = reg[name]
        base = d.bases[spec["base"]]
        transform, kind = spec["fit"](base[d.v2], d.y[d.v2])
        Z = transform(base)
        res = acceptance(d, Z, kind)
        res["kind"] = kind
        res["supervised"] = spec["supervised"]
        res["transductive"] = spec["transductive"]
        res["dim"] = int(Z.shape[1]) if Z.ndim == 2 else 1
        if do_cv and spec["supervised"]:
            res["pair_cv"] = pair_stats_cv(d, spec)
        if do_eval:
            try:
                ev = eval_proximity(d, spec, transform, kind)
                if ev:
                    res["eval_proximity"] = ev
            except Exception as e:  # a metric that cannot embed eval rows is not a failure
                res["eval_proximity_error"] = f"{type(e).__name__}: {e}"
        results[name] = res
        so = res.get("source_only", {})
        print(
            f"{name:<16} {res['pair_new']['auroc']:>7.3f} "
            f"{res['scenario']['auroc']:>7.3f} "
            f"{res['pair_provenance']['auroc']:>7.3f} "
            f"{res.get('hop_paired', res['hop'])['auroc']:>7.3f} "
            f"{res['nn']['balanced']:>6.3f} "
            f"{res['knn']['auroc_15']:>8.3f} "
            f"{so.get('knn', {}).get('bal_5', float('nan')):>10.3f} "
            f"{res['durable']['auroc']:>7.3f} "
            f"{so.get('durable_auroc', float('nan')):>7.3f}",
            flush=True,
        )
        del Z
        gc.collect()
    return results


def reference_predictors(d: ArmData) -> dict:
    """Acquisition baselines that need no metric at all, for calibration."""
    out = {}
    lab = d.durable.astype(int)
    for name, sc in (
        ("abs_seq_logit", np.abs(d.seq[d.succ])),
        ("neg_abs_seq_logit", -np.abs(d.seq[d.succ])),
        ("n_tokens", d.n_tok[d.succ].astype(float)),
        ("label_is_positive", d.y[d.succ]),
    ):
        lo, hi = _boot_auroc(lab, sc)
        out[name] = {"auroc": _auroc(lab, sc), "ci95": [lo, hi]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", default=list(ARMS))
    ap.add_argument("--metrics", nargs="+", default=None)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-minilm", action="store_true")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    report = {}
    for arm in args.arm:
        print(f"\n########## {arm} ##########", flush=True)
        d = load_arm(arm)
        print(
            f"  rows {len(d.y)}  v2 {len(d.v2)}  new-in-v3 {len(d.new)}  "
            f"successes {len(d.succ)}  durable {int(d.durable.sum())}  "
            f"couples {len(d.pair_src)}",
            flush=True,
        )
        if not args.no_minilm:
            try:
                add_minilm(d)
                print("  minilm embeddings ready", flush=True)
            except Exception as e:
                print(f"  minilm unavailable: {type(e).__name__}: {e}", flush=True)
        report[arm] = {
            "n": {"rows": len(d.y), "v2": len(d.v2), "new": len(d.new),
                  "successes": len(d.succ), "durable": int(d.durable.sum()),
                  "couples": len(d.pair_src)},
            "metrics": run_arm(d, args.metrics, not args.no_cv, not args.no_eval),
            "reference_predictors": reference_predictors(d),
        }
        del d
        gc.collect()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
