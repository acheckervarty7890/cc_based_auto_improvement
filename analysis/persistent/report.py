#!/usr/bin/env python
"""Phase 3 - the analysis. Writes ``results/summary.json`` and ``results/SUMMARY.md``.

Every number the page and the write-up quote is computed here and nowhere else, so the two
cannot drift apart or from ``scores.npz``.

    analysis/persistent/report.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pe_common as PE  # noqa: E402

sys.path.insert(0, str(PE.REPO / "src"))
# The refusal detector is offdist's, deliberately not re-implemented: this concept's data
# is *about* refusal, so a plain substring scan reads a quoted refusal as an uttered one —
# the failure CLAUDE.md records. Importing it also keeps the two studies' "refuses" column
# meaning the same thing.
sys.path.insert(0, str(PE.REPO / "analysis" / "offdist"))
import od_common as OD  # noqa: E402


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 1.0


def build() -> dict:
    d = PE.load_scores()
    P, y, split = d["proba"], d["labels"].astype(int), d["split"]
    exp, arm, it = d["experiment"], d["arm"], d["iteration"]
    nprobe, nrow = P.shape
    rows = PE.eval_rows()
    assert len(rows) == nrow
    assert all((r["label"] == PE.POS) == bool(v) for r, v in zip(rows, y)), \
        "eval_rows() order does not match the score matrix columns"

    W = PE.errors_at_half(P, y)
    k = W.sum(0)
    core = W.all(0)
    core_idx = np.where(core)[0]
    splits = sorted(set(split.tolist()))
    ctl = np.load(PE.RESULTS / "controls.npz", allow_pickle=True)
    assert (ctl["split"] == split).all()
    ctl_json = PE.read_json(PE.RESULTS / "controls.json")

    S: dict = {}

    S["setup"] = {
        "n_probes": nprobe, "n_rows": nrow,
        "runs": [{"experiment": r.experiment, "arm": r.arm, "label": r.label,
                  "n_probes": r.n_iters + 1} for r in PE.RUNS],
        "split_sizes": {s: int((split == s).sum()) for s in splits},
        "per_probe_error_rate": {"mean": float(W.mean()), "min": float(W.mean(1).min()),
                                 "max": float(W.mean(1).max())},
        "reproduction_max_auroc_drift": float(d["reproduction_max_auroc_drift"]),
    }

    # ---------------------------------------------------------------- concentration
    S["concentration"] = {
        "never_wrong": int((k == 0).sum()),
        "wrong_1_to_11": int(((k >= 1) & (k <= 11)).sum()),
        "wrong_12_to_33": int(((k >= 12) & (k <= 33)).sum()),
        "wrong_34_to_44": int(((k >= 34) & (k <= 44)).sum()),
        "always_wrong": int(core.sum()),
        "total_errors": int(W.sum()),
        "distinct_rows_ever_wrong": int((k > 0).sum()),
        "expected_always_if_independent": float(nrow * W.mean() ** nprobe),
        "histogram": np.bincount(k, minlength=nprobe + 1).tolist(),
    }

    # ------------------------------------------------------------------- the core set
    marg = np.abs(P[:, core_idx] - 0.5)
    S["core"] = {
        "n": int(core.sum()),
        "by_split": {s: {"n": int((split[core_idx] == s).sum()),
                         "of": int((split == s).sum())} for s in splits},
        "n_positive": int(y[core_idx].sum()),
        "n_negative": int((1 - y[core_idx]).sum()),
        "median_abs_margin": float(np.median(marg)),
        "median_abs_margin_never_wrong": float(np.median(np.abs(P[:, k == 0] - 0.5))),
        "n_confidently_wrong_every_probe": int((marg.min(0) > 0.4).sum()),
        "mean_within_split_percentile": float(np.mean(
            [[(P[j, split == rows[i]["split"]] < P[j, i]).mean() for i in core_idx]
             for j in range(nprobe)])),
    }

    # --------------------------------------------------- arms, experiments, iteration
    # iter0 is bit-identical in all five runs, so it is excluded from every persistence
    # intersection: leaving it in makes "already wrong at iter0" true by construction.
    i0_rows = np.where(it == 0)[0]
    i0 = W[i0_rows[0]]
    S["ancestor"] = {
        "identical_across_runs": bool(np.abs(P[i0_rows] - P[i0_rows[0]]).max() == 0),
        "max_abs_score_delta": float(np.abs(P[i0_rows] - P[i0_rows[0]]).max()),
        "n_errors": int(i0.sum()),
    }

    retrained = it > 0
    per_arm = {a: W[(arm == a) & retrained].all(0) for a in PE.ARMS}
    core_r = W[retrained].all(0)
    S["arms"] = {}
    for a in PE.ARMS:
        others = np.any([per_arm[b] for b in PE.ARMS if b != a], axis=0)
        pa = per_arm[a]
        S["arms"][a] = {
            "label": PE.ARM_LABEL[a],
            "persistent": int(pa.sum()),
            "private": int((pa & ~others).sum()),
            "inherited_from_iter0": int((pa & i0).sum()),
            "new_since_iter0": int((pa & ~i0).sum()),
            "positive_share": float(y[pa].mean()),
        }
    union = np.any(list(per_arm.values()), axis=0)
    S["arms_summary"] = {
        "intersection": int(core_r.sum()), "union": int(union.sum()),
        "intersection_inherited_from_iter0": int((core_r & i0).sum()),
        "jaccard": {a: {b: jaccard(per_arm[a], per_arm[b]) for b in PE.ARMS}
                    for a in PE.ARMS},
    }
    e22 = W[(exp == "exp22") & retrained].all(0)
    e23 = W[(exp == "exp23") & retrained].all(0)
    S["experiments"] = {
        "exp22_persistent": int(e22.sum()), "exp23_persistent": int(e23.sum()),
        "shared": int((e22 & e23).sum()), "exp22_only": int((e22 & ~e23).sum()),
        "exp23_only": int((e23 & ~e22).sum()), "jaccard": jaccard(e22, e23),
    }

    keep = np.where(retrained)[0]
    buckets = {"same_arm": [], "same_experiment": [], "different_experiment": []}
    for ii in range(len(keep)):
        for jj in range(ii + 1, len(keep)):
            a, b = keep[ii], keep[jj]
            key = ("same_arm" if arm[a] == arm[b]
                   else "same_experiment" if exp[a] == exp[b]
                   else "different_experiment")
            buckets[key].append(jaccard(W[a], W[b]))
    S["travel"] = {kk: {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v))}
                   for kk, v in buckets.items()}

    S["trajectory"] = {
        a: [{"iteration": int(i), "errors": int(W[j].sum()),
             "share_inherited": float((W[j] & i0).sum() / max(W[j].sum(), 1))}
            for j, i in sorted(zip(np.where(arm == a)[0], it[arm == a]), key=lambda t: t[1])]
        for a in PE.ARMS}
    still = sum(per_arm[a][i0] for a in PE.ARMS)
    S["ancestor"]["fate_of_its_errors"] = {
        "fixed_by_all_arms": int((still == 0).sum()),
        "fixed_by_none": int((still == len(PE.ARMS)).sum()),
        "histogram_n_arms_still_failing": {str(n): int((still == n).sum())
                                           for n in range(len(PE.ARMS) + 1)},
    }

    # ------------------------------------------------------------------ calibration
    rate = (P > 0.5).mean(1)
    med_thresh_acc = {}
    for a in PE.ARMS:
        sel = arm == a
        Wm = (P[sel] > np.median(P[sel], axis=1, keepdims=True)) != y[None, :].astype(bool)
        med_thresh_acc[a] = {"positive_call_rate": float(rate[sel].mean()),
                             "accuracy_at_half": float(1 - W[sel].mean()),
                             "accuracy_at_median": float(1 - Wm.mean())}
    S["calibration"] = {
        "positive_call_rate": {"mean": float(rate.mean()), "min": float(rate.min()),
                               "max": float(rate.max()), "true_base_rate": 0.5},
        "median_score": float(np.median(P, axis=1).mean()),
        "false_negative_share_of_errors": float((W & (y[None, :] == 1)).sum() / W.sum()),
        "by_arm": med_thresh_acc,
    }

    # ------------------------------------------------------- threshold-free core set
    Wr = PE.errors_at_split_median(P, y, split)
    core_rank = Wr.all(0)
    rank_idx = np.where(core_rank)[0]
    ctl_wrong = {n: (ctl[n] > 0.5) != y.astype(bool) for n in ("dev_only", "ceiling")}
    S["ranking_core"] = {
        "n": int(core_rank.sum()),
        "overlap_with_half_core": int((core_rank & core).sum()),
        "jaccard_with_half_core": jaccard(core_rank, core),
        "n_wrong_for_at_least_44": int((k >= 44).sum()),
        "by_split": {s: int((split[rank_idx] == s).sum()) for s in splits},
        "rows": [{
            "index": int(i), "split": rows[i]["split"],
            "label": PE.POS if y[i] else PE.NEG,
            "mean_percentile": float(np.mean(
                [(P[j, split == rows[i]['split']] < P[j, i]).mean() for j in range(nprobe)])),
            "ceiling_wrong": bool(ctl_wrong["ceiling"][i]),
            "user": next((m["content"] for m in rows[i]["messages"]
                          if m.get("role") == "user"), "")[:160],
        } for i in rank_idx],
    }

    # ---------------------------------------------------------- learnability controls
    both = ctl_wrong["ceiling"][core_idx] & ctl_wrong["dev_only"][core_idx]
    S["controls"] = {
        "ceiling_mean_auroc": ctl_json["ceiling_mean_auroc"],
        "ceiling_per_split_auroc": {n: v["auroc"]
                                    for n, v in ctl_json["ceiling_per_split"].items()},
        "n_dev_pool": ctl_json["n_dev_pool"], "n_dev_validation": ctl_json["n_dev_validation"],
        **{n: {"overall_error": float(w.mean()),
               "errors_on_core": int(w[core_idx].sum()),
               "errors_on_never_wrong": int(w[k == 0].sum())}
           for n, w in ctl_wrong.items()},
        "core_wrong_for_both_controls": int(both.sum()),
        "core_recovered_by_in_distribution": int(core.sum() - both.sum()),
    }

    # ------------------------------------------------------------------- eval hygiene
    h = [hashlib.sha256(json.dumps(r["messages"], sort_keys=True).encode()).hexdigest()
         for r in rows]
    cnt = Counter(h)
    by_h = defaultdict(list)
    for i, x in enumerate(h):
        by_h[x].append(i)
    pair_counts: dict[str, int] = defaultdict(int)
    for x, idxs in by_h.items():
        if len(idxs) > 1:
            assert len({y[i] for i in idxs}) == 1, f"same text, different labels: {idxs}"
            pair_counts[" + ".join(sorted({rows[i]["split"] for i in idxs}))] += 1
    S["duplicates"] = {
        "n_conversations_repeated": sum(1 for v in cnt.values() if v > 1),
        "n_rows_involved": sum(1 for x in h if cnt[x] > 1),
        "by_split_pair": dict(pair_counts),
        "labels_always_agree": True,
        "core_rows_that_are_duplicates": int(sum(1 for i in core_idx if cnt[h[i]] > 1)),
        "core_distinct_conversations": len({h[i] for i in core_idx}),
    }

    # --------------------------------------------------------------------- what they are
    feats = [OD.structural_features(r["messages"]) for r in rows]
    ref = np.array([f["refuses"] for f in feats])
    S["character"] = {
        "features": {key: {"core": float(np.mean([feats[i][key] for i in core_idx])),
                           "all": float(np.mean([f[key] for f in feats]))}
                     for key in ("refuses", "n_messages", "chars_total",
                                 "chars_assistant", "assistant_share")},
        "refusal_crosstab": {
            ("refuses" if r_ else "complies"): {
                "n": int((ref == r_).sum()),
                "p_positive": float(y[ref == r_].mean()),
                "mean_k": float(k[ref == r_].mean())}
            for r_ in (0, 1)},
    }

    # Per-row error count and split, in column order — the page's heat strip is one cell
    # per eval row, so it needs the raw vector rather than the histogram.
    S["rows_k"] = k.tolist()
    S["rows_split"] = split.tolist()
    S["rows_label"] = y.tolist()

    S["core_rows"] = [{
        "index": int(i), "split": rows[i]["split"],
        "label": PE.POS if y[i] else PE.NEG,
        "mean_p": float(P[:, i].mean()), "min_p": float(P[:, i].min()),
        "max_p": float(P[:, i].max()),
        "ceiling_p": float(ctl["ceiling"][i]), "dev_only_p": float(ctl["dev_only"][i]),
        "ceiling_wrong": bool(ctl_wrong["ceiling"][i]),
        "dev_only_wrong": bool(ctl_wrong["dev_only"][i]),
        "duplicate": bool(cnt[h[i]] > 1),
        "in_ranking_core": bool(core_rank[i]),
        "user": next((m["content"] for m in rows[i]["messages"]
                      if m.get("role") == "user"), ""),
        "assistant": next((m["content"] for m in rows[i]["messages"]
                           if m.get("role") == "assistant"), ""),
        "explanation": rows[i]["explanation"],
    } for i in core_idx]
    return S


def markdown(S: dict) -> str:
    c, a, e, t = S["concentration"], S["arms"], S["experiments"], S["travel"]
    L = []
    w = L.append
    w("# Which eval rows is every probe wrong about?\n")
    w(f"{S['setup']['n_probes']} probes — experiment22's two arms (6 each) and "
      f"experiment23's three (11 each) — scored on all {S['setup']['n_rows']} "
      f"`eval_sets/hu_ha` rows off the cached activations. The matrix reproduces every "
      f"run's published comparison CSV to {S['setup']['reproduction_max_auroc_drift']:.1e} "
      f"AUROC.\n")
    w("## The errors are shared, not idiosyncratic\n")
    w("| rows wrong for… | count | % of eval |")
    w("| --- | --- | --- |")
    for lab, key in (("0 probes", "never_wrong"), ("1–11", "wrong_1_to_11"),
                     ("12–33", "wrong_12_to_33"), ("34–44", "wrong_34_to_44"),
                     ("**all 45**", "always_wrong")):
        w(f"| {lab} | {c[key]} | {100*c[key]/S['setup']['n_rows']:.1f}% |")
    w(f"\nPer-probe error rate is {S['setup']['per_probe_error_rate']['mean']:.3f}; if the "
      f"45 probes erred independently at that rate the last row would hold "
      f"{c['expected_always_if_independent']:.1e} rows. The {c['total_errors']} errors land "
      f"on {c['distinct_rows_ever_wrong']} distinct rows.\n")
    w(f"The {S['core']['n']} always-wrong rows are {S['core']['n_positive']} positives and "
      f"{S['core']['n_negative']} negative — almost entirely `{PE.POS}` conversations "
      f"scored as not-harmful — concentrated in "
      + ", ".join(f"`{s}` ({v['n']}/{v['of']})" for s, v in S["core"]["by_split"].items()
                  if v["n"]) + ".\n")
    w("## Arm and experiment barely move it\n")
    w("| relation between two probes | pairs | mean Jaccard of error sets |")
    w("| --- | --- | --- |")
    for key, lab in (("same_arm", "same arm, different iteration"),
                     ("same_experiment", "same experiment, different arm"),
                     ("different_experiment", "different experiment")):
        w(f"| {lab} | {t[key]['n']} | {t[key]['mean']:.3f} |")
    w(f"\nSame-arm ≈ same-experiment ≈ cross-experiment. Each arm fails "
      f"{min(v['persistent'] for v in a.values())}–"
      f"{max(v['persistent'] for v in a.values())} rows at every retrained iteration; the "
      f"five-way intersection is {S['arms_summary']['intersection']}, "
      f"{100*S['arms_summary']['intersection']/S['arms_summary']['union']:.0f}% of the "
      f"{S['arms_summary']['union']}-row union.\n")
    w("| arm | persistent | private to it | inherited from iter0 |")
    w("| --- | --- | --- | --- |")
    for k_, v in a.items():
        w(f"| {v['label']} | {v['persistent']} | {v['private']} | {v['inherited_from_iter0']} |")
    w(f"\nAll five `probe_iter0` are bit-identical (max |Δp| = "
      f"{S['ancestor']['max_abs_score_delta']:.1e}); that shared 50-sample probe makes "
      f"{S['ancestor']['n_errors']} errors, and "
      f"{100*S['arms_summary']['intersection_inherited_from_iter0']/S['arms_summary']['intersection']:.0f}% "
      f"of the universal core was already wrong there. Of its {S['ancestor']['n_errors']} "
      f"errors, {S['ancestor']['fate_of_its_errors']['fixed_by_all_arms']} are fixed by "
      f"every arm and {S['ancestor']['fate_of_its_errors']['fixed_by_none']} by none. "
      f"Retraining shrinks the error set without relocating it.\n")
    w("## Most of it is the 0.5 threshold\n")
    cal = S["calibration"]
    w(f"These probes call {cal['positive_call_rate']['mean']:.0%} of eval rows positive "
      f"against a 50% base rate, their median score is {cal['median_score']:.3f}, and "
      f"{cal['false_negative_share_of_errors']:.1%} of all errors are false negatives. At "
      f"each probe's own median threshold every arm gains accuracy:\n")
    w("| arm | positive-call rate | accuracy at 0.5 | accuracy at its median |")
    w("| --- | --- | --- | --- |")
    for k_, v in cal["by_arm"].items():
        w(f"| {a[k_]['label']} | {v['positive_call_rate']:.3f} | "
          f"{v['accuracy_at_half']:.3f} | {v['accuracy_at_median']:.3f} |")
    r = S["ranking_core"]
    w(f"\nThreshold-free — wrong side of its split's median for all 45 probes — the core is "
      f"**{r['n']} rows, not {S['core']['n']}** (overlap {r['overlap_with_half_core']}). All "
      f"{r['n']} are in `eval_ant_hh`. The other rows are ranked reasonably (mean "
      f"within-split percentile {S['core']['mean_within_split_percentile']:.2f}) and fail "
      f"only because the threshold sits above them.\n")
    for row in r["rows"]:
        w(f"- `{row['index']}` **{'POS' if row['label'] == PE.POS else 'NEG'}** "
          f"percentile {row['mean_percentile']:.2f}, ceiling "
          f"{'wrong' if row['ceiling_wrong'] else 'right'} — {row['user'][:80]}")
    ctl = S["controls"]
    w(f"\n## Are they learnable?\n")
    w(f"An in-distribution probe (5-fold CV inside eval plus the dev pool, mean AUROC "
      f"{ctl['ceiling_mean_auroc']:.4f}) gets {S['core']['n']-ctl['ceiling']['errors_on_core']} "
      f"of the {S['core']['n']} right; a dev-only probe gets "
      f"{S['core']['n']-ctl['dev_only']['errors_on_core']}. Only "
      f"{ctl['core_wrong_for_both_controls']} rows defeat all 45 run probes **and** both "
      f"controls. The core is mostly a red-team training-distribution deficit, not "
      f"intrinsic ambiguity.\n")
    dup = S["duplicates"]
    w("## One data-hygiene finding\n")
    w(f"{dup['n_conversations_repeated']} conversations appear more than once across "
      f"splits — {dup['n_rows_involved']} rows, "
      f"{100*dup['n_rows_involved']/S['setup']['n_rows']:.1f}% of the eval set ("
      + ", ".join(f"{k_}: {v}" for k_, v in dup["by_split_pair"].items())
      + f"). Labels always agree. {dup['core_rows_that_are_duplicates']} of the "
      f"{S['core']['n']} core rows are duplicates, so they are "
      f"{dup['core_distinct_conversations']} distinct conversations — and the per-split "
      f"eval means are not independent of each other.\n")
    return "\n".join(L) + "\n"


def main() -> int:
    S = build()
    PE.write_json(PE.RESULTS / "summary.json", S)
    (PE.RESULTS / "SUMMARY.md").write_text(markdown(S), encoding="utf-8")
    print(f"wrote {PE.RESULTS/'summary.json'} and {PE.RESULTS/'SUMMARY.md'}")
    print(f"  core (p>0.5, all {S['setup']['n_probes']} probes): {S['core']['n']} rows")
    print(f"  core (threshold-free):                {S['ranking_core']['n']} rows")
    print(f"  resist both in-distribution controls: "
          f"{S['controls']['core_wrong_for_both_controls']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
