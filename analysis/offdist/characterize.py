#!/usr/bin/env python
"""Phase 1: how do the red-team rows differ from the eval rows, at the surface?

Three descriptions, all computed from text alone — no activations, no LLM:

1. **Structure.** Turn counts, who talks and how much, formatting, and whether the
   assistant refuses. Reported as red-team vs eval means.

2. **Lexicon and topic.** A TF-IDF + logistic-regression discriminator is trained to tell a
   red-team conversation from an eval conversation, **out of fold**, so every row gets an
   honest `p_redteam`: 0.5 means "this row could have come from either corpus", 1.0 means
   "unmistakably red-team". That score is this study's off-distribution axis, and the
   discriminator's own AUROC says how separable the two corpora are at all. k-means over
   the same TF-IDF space gives topics, each summarised by the terms that distinguish it.

3. **Labelling convention.** The eval splits encode a mapping from assistant behaviour to
   label. The one part of it that can be read off mechanically is refusal: measured here,
   an eval conversation whose assistant refuses is labelled `not_harmful_to_human` 94.8% of
   the time. A red-team row that refuses *and* is labelled `harmful_to_human` is therefore
   labelled the opposite way round from how eval labels that behaviour, and is flagged.

   The refusal detector extends `llm_judge._REFUSAL_MARKERS` (the repo's own list) with a
   regex for the softer forms an assistant uses and a judge does not, and it keeps
   `_strip_quoted_spans` verbatim — this concept's data is *about* refusal, so a quoted
   refusal phrase is common and a plain substring scan reads the citation as the utterance.
   Its precision is not assumed: it is reported against the eval labels, which is what makes
   "the convention" a measurement rather than a premise.

Also recorded, because the question was asked directly: for every contrastive pair, whether
the two halves differ in the ASSISTANT's turns (the axis the eval pairs contrast on) or in
the USER's turns (the opposite axis).

Writes `results/flags_<arm>.jsonl` (one row per red-team conversation) and
`results/surface_<arm>.json`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import od_common as O  # noqa: E402

# Softer refusal forms an assistant uses and a judge does not. Kept as one regex rather than
# folded into the marker tuple so the borrowed list stays recognisably the borrowed list.
_BROAD_REFUSAL = re.compile(
    r"\b(i\s(can'?t|cannot|won'?t|will not)\b"
    r"|i'?m\s(sorry|afraid|not able|unable|not going to|not comfortable)"
    r"|i am\s(sorry|afraid|not able|unable|not going to|not comfortable)"
    r"|i\s(must|have to|need to|will)\s(decline|refuse)"
    r"|i'?d rather not|against my (guidelines|programming|principles)"
    r"|not something i can|can'?t (help|assist|provide|share|do) (with )?that)",
    re.I,
)


def refuses(messages) -> bool:
    from agentic_redteam.llm_judge import _strip_quoted_spans

    text = _strip_quoted_spans(O.joined(messages, "assistant"))
    return bool(_BROAD_REFUSAL.search(text)) or O.assistant_refuses(messages)


def eval_convention(evrows) -> dict:
    """P(harmful | assistant refuses) on the eval splits — the convention, measured."""
    r = [e for e in evrows if refuses(e["messages"])]
    nr = [e for e in evrows if not refuses(e["messages"])]
    ph = lambda xs: float(np.mean([x["label"] == O.POS for x in xs])) if xs else float("nan")
    per_split = {}
    for split in sorted({e["split"] for e in evrows}):
        rows = [e for e in evrows if e["split"] == split]
        rr = [e for e in rows if refuses(e["messages"])]
        per_split[split] = {"n": len(rows), "n_refuse": len(rr), "p_harm_given_refuse": ph(rr)}
    return {
        "n": len(evrows), "n_refuse": len(r), "refusal_rate": len(r) / len(evrows),
        "p_harm_given_refuse": ph(r), "p_harm_given_no_refuse": ph(nr),
        "per_split": per_split,
    }


def discriminator(rt_texts, ev_texts, seed=0, n_folds=5):
    """Out-of-fold P(red-team) per row, plus the discriminator's AUROC and top terms.

    Out of fold matters: an in-sample probability from a high-dimensional TF-IDF model is
    near-separable by memorisation, which would make every row look equally off-manifold.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X_text = list(rt_texts) + list(ev_texts)
    y = np.r_[np.ones(len(rt_texts), int), np.zeros(len(ev_texts), int)]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60_000,
                          sublinear_tf=True, strip_accents="unicode")
    X = vec.fit_transform(X_text)
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    full = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    terms = np.array(vec.get_feature_names_out())
    coef = full.coef_[0]
    top_rt = [(terms[i], float(coef[i])) for i in np.argsort(coef)[::-1][:30]]
    top_ev = [(terms[i], float(coef[i])) for i in np.argsort(coef)[:30]]
    return {
        "p_redteam": oof[: len(rt_texts)],
        "p_redteam_eval": oof[len(rt_texts):],
        "auroc": float(roc_auc_score(y, oof)),
        "top_terms_redteam": top_rt,
        "top_terms_eval": top_ev,
        "X_rt": X[: len(rt_texts)],
        "vectorizer": vec,
    }


def topics(X_rt, vec, k=8, seed=0):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X_rt)
    terms = np.array(vec.get_feature_names_out())
    out = []
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        centroid = np.asarray(X_rt[members].mean(axis=0)).ravel()
        top = [terms[i] for i in np.argsort(centroid)[::-1][:10]]
        out.append({"cluster": c, "n": int(len(members)), "top_terms": top})
    return km.labels_, out


def pair_axis(rows, pairs):
    """Per pair: does the contrast live in the assistant's turns or the user's?"""
    out = {}
    for i, j in pairs:
        a, b = rows[i]["messages"], rows[j]["messages"]
        du = 1 - difflib.SequenceMatcher(
            None, O.joined(a, "user"), O.joined(b, "user"), autojunk=False).ratio()
        da = 1 - difflib.SequenceMatcher(
            None, O.joined(a, "assistant"), O.joined(b, "assistant"), autojunk=False).ratio()
        axis = "user" if du > da else "assistant"
        for x in (i, j):
            out[x] = {"pair_with": j if x == i else i, "pair_axis": axis,
                      "user_delta": round(du, 4), "assistant_delta": round(da, 4)}
    return out


def run(arm: O.Arm, evrows, conv, args) -> None:
    rows = O.load_redteam(arm)
    pairs = O.recover_pairs(arm, rows)
    axis = pair_axis(rows, pairs)

    rt_texts = [O.joined(r["messages"]) for r in rows]
    ev_texts = [O.joined(e["messages"]) for e in evrows]
    d = discriminator(rt_texts, ev_texts, seed=args.seed)
    labels_km, topic_info = topics(d["X_rt"], d["vectorizer"], k=args.k, seed=args.seed)

    feats = [O.structural_features(r["messages"]) for r in rows]
    ev_feats = [O.structural_features(e["messages"]) for e in evrows]
    for f, r in zip(feats, rows):
        f["refuses"] = int(refuses(r["messages"]))
    for f, e in zip(ev_feats, evrows):
        f["refuses"] = int(refuses(e["messages"]))

    out_rows = []
    for n, r in enumerate(rows):
        a = axis.get(n, {})
        out_rows.append({
            "i": r["i"], "pool_index": r["pool_index"], "id": r["id"], "label": r["label"],
            "p_redteam": float(d["p_redteam"][n]),
            "topic": int(labels_km[n]),
            "refuses": feats[n]["refuses"],
            # The convention flag: eval labels a refusing conversation not_harmful 94.8% of
            # the time, so a refusing row labelled harmful runs the mapping backwards.
            "convention_inverted": int(feats[n]["refuses"] and r["label"] == O.POS),
            "pair_with": a.get("pair_with"),
            "pair_axis": a.get("pair_axis"),
            "structural": feats[n],
        })
    path = O.RESULTS / f"flags_{arm.key}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    keys = [k for k in feats[0] if k != "refuses"] + ["refuses"]
    struct = {k: {"redteam": float(np.mean([f[k] for f in feats])),
                  "eval": float(np.mean([f[k] for f in ev_feats]))} for k in keys}
    n_inv = sum(r["convention_inverted"] for r in out_rows)
    summary = {
        "arm": arm.key, "attacker": arm.attacker,
        "n_redteam": len(rows), "n_eval": len(evrows), "n_pairs": len(pairs),
        "discriminator_auroc": d["auroc"],
        "p_redteam_quantiles": {q: float(np.quantile(d["p_redteam"], q))
                                for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "p_redteam_eval_median": float(np.median(d["p_redteam_eval"])),
        "top_terms_redteam": d["top_terms_redteam"],
        "top_terms_eval": d["top_terms_eval"],
        "topics": topic_info,
        "topic_mean_p_redteam": {int(c): float(np.mean(d["p_redteam"][labels_km == c]))
                                 for c in range(args.k)},
        "structural_means": struct,
        "refusal_rate_redteam": float(np.mean([f["refuses"] for f in feats])),
        "eval_convention": conv,
        "n_convention_inverted": int(n_inv),
        "pair_axis_counts": {
            "assistant": sum(1 for v in axis.values() if v["pair_axis"] == "assistant") // 2,
            "user": sum(1 for v in axis.values() if v["pair_axis"] == "user") // 2,
        },
    }
    O.write_json(O.RESULTS / f"surface_{arm.key}.json", summary)
    print(f"[{arm.key}] discriminator AUROC {d['auroc']:.4f}  "
          f"median p_redteam {np.median(d['p_redteam']):.3f}  "
          f"refusal {summary['refusal_rate_redteam']:.1%} (eval {conv['refusal_rate']:.1%})  "
          f"convention-inverted {n_inv}  "
          f"pairs assistant/user {summary['pair_axis_counts']['assistant']}/"
          f"{summary['pair_axis_counts']['user']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(O.ARMS))
    ap.add_argument("--k", type=int, default=8, help="k-means topics over the red-team set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    evrows = O.load_eval()
    conv = eval_convention(evrows)
    print(f"[eval] refusal rate {conv['refusal_rate']:.1%}; "
          f"P(harmful | refuses) = {conv['p_harm_given_refuse']:.3f}, "
          f"P(harmful | does not) = {conv['p_harm_given_no_refuse']:.3f}", flush=True)
    O.write_json(O.RESULTS / "eval_convention.json", conv)
    for key in args.arms:
        run(O.ARMS[key], evrows, conv, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
