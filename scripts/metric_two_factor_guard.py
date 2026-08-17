"""If no single metric is both scenario-aware and label-aware, use two.

``metric_geometry.py`` finds the candidate geometries splitting into two disjoint
families and nothing in between:

    scenario-aware, label-blind   pool:mean, text:tfidf, text:minilm, lin:whitened, ...
                                  scenAUR 0.90-1.00, pairAUR 0.51-0.59
    label-aware, scenario-blind   sup:lda, pool:last, probe:logit, probe:proj
                                  pairAUR 0.59-0.76, scenAUR 0.29-0.47

A novelty guard needs both at once — it must reject a re-skin (same scenario, *same*
label) while accepting an opposite-label rewrite (same scenario, *different* label) — so
neither family can implement it alone, which is the mechanism behind the doc's loop-fix 2
warning. But the two properties being carried by different metrics is not an obstacle if
the guard is allowed to consult both:

    reject(candidate) iff  sim_scenario(candidate, stored) > tau  AND  same side of the
                           probe's boundary

This script measures the one thing that decides whether that conjunction works. The gate
is only sound if the pairs it must protect really do straddle the boundary, so:

1. **Straddle rate** — what fraction of (source, generated counterpart) couples sit on
   opposite sides of ``probe_iter2``'s boundary, and on opposite sides of the *judge's*
   label. If this is high, the gate accepts them by construction.
2. **False-rejection rate** — sweeping tau over the scenario metric, what fraction of the
   couples (must accept) and of the genuinely-new successes (must accept) each guard
   rejects, one-factor versus gated.

**What this cannot measure**, stated plainly rather than papered over: the *true*-rejection
rate. A guard is judged on catching clones, and this corpus contains none to catch — the
difflib guard reports 0% at tau=0.8 and no semantic-clone labels exist. So the result here
bounds the damage a guard does, not the good it does; establishing the latter needs a set
of deliberately re-skinned successes, which has to be generated.

Usage:
    .venv_claude/bin/python scripts/metric_two_factor_guard.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_geometry import ARMS, load_arm, sim_mat, sim_pairs

OUT = Path("results_hu_harm_gemma27b_batch_ablation/vintage/metric_two_factor_guard.json")
SCENARIO_METRICS = [("pool:mean", "mean"), ("text:tfidf", "text"), ("pool:topz16", "topz16")]
TAUS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def _tfidf(d, fit_rows):
    from sklearn.feature_extraction.text import TfidfVectorizer

    v = TfidfVectorizer(
        min_df=2, max_features=20000, ngram_range=(1, 2), sublinear_tf=True
    ).fit(list(d.bases["text"][fit_rows]))
    return np.asarray(v.transform(list(d.bases["text"])).todense(), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", default=list(ARMS))
    args = ap.parse_args()

    report: dict = {}
    for arm in args.arm:
        d = load_arm(arm, with_eval=False)
        src2gen = dict(zip(d.pair_src.tolist(), d.pair_gen.tolist()))
        anchors = np.array([int(s) for s in d.succ if int(s) in src2gen])
        gens = np.array([src2gen[int(s)] for s in anchors])

        # 1. do the pairs the gate must protect actually straddle the boundary?
        side = (d.seq >= 0).astype(int)  # probe_iter2's predicted class
        straddle_probe = float((side[d.pair_src] != side[d.pair_gen]).mean())
        straddle_label = float((d.y[d.pair_src] != d.y[d.pair_gen]).mean())
        # The number that decides whether the gate can work. Restricted to the couples
        # built from a NEW SUCCESS — the only ones a guard meets in anger — the probe is
        # wrong on the source by definition of success, so the source's predicted side
        # equals its counterpart's true side and the pair does NOT straddle. The gate is
        # therefore blindest exactly where it is needed.
        straddle_anchor = float((side[anchors] != side[gens]).mean())
        print(f"\n########## {arm} ##########")
        print(f"  couples: {len(d.pair_src)}   anchors (new successes with a counterpart): {len(anchors)}")
        print(f"  straddle probe_iter2's boundary, ALL couples    : {straddle_probe:.1%}")
        print(f"  straddle probe_iter2's boundary, ANCHOR couples : {straddle_anchor:.1%}")
        print(f"  carry opposite judge/constructed labels         : {straddle_label:.1%}")

        arm_out: dict = {
            "n_couples": int(len(d.pair_src)),
            "n_anchors": int(len(anchors)),
            "straddle_probe_boundary": straddle_probe,
            "straddle_probe_boundary_anchors": straddle_anchor,
            "straddle_label": straddle_label,
            "metrics": {},
        }

        for name, base in SCENARIO_METRICS:
            Z = _tfidf(d, d.v2) if base == "text" else d.bases[base]
            kind = "cosine"
            # the guard's own view: each candidate against the stored SUCCESSES
            stored = d.v2[~d.is_gen[d.v2]]
            S_new = sim_mat(Z, kind, d.succ, stored)
            nn_new = S_new.max(1)
            nn_new_idx = stored[S_new.argmax(1)]
            own = sim_pairs(Z, kind, anchors, gens)

            rows = []
            for tau in TAUS:
                # one-factor: reject on similarity alone
                rej_new = float((nn_new > tau).mean())
                # a counterpart is judged against ITS OWN source, the hardest case: the
                # guard sees a conversation that shares a scenario with a stored success
                rej_own = float((own > tau).mean())
                # gated: additionally require the same side of the boundary
                gate_new = float(
                    ((nn_new > tau) & (side[d.succ] == side[nn_new_idx])).mean()
                )
                gate_own = float(
                    ((own > tau) & (side[anchors] == side[gens])).mean()
                )
                rows.append(
                    {"tau": tau, "reject_new_1f": rej_new, "reject_own_1f": rej_own,
                     "reject_new_gated": gate_new, "reject_own_gated": gate_own}
                )
            arm_out["metrics"][name] = {
                "mean_own_sim": float(own.mean()),
                "mean_nn_sim": float(nn_new.mean()),
                "sweep": rows,
            }
            print(f"\n  --- {name} (scenario metric; stored set = {len(stored)} v2 successes) ---")
            print(f"  {'tau':>6} {'rej new 1f':>11} {'rej own 1f':>11} "
                  f"{'rej new gated':>14} {'rej own gated':>14}")
            for r in rows:
                print(f"  {r['tau']:>6.2f} {r['reject_new_1f']:>10.1%} "
                      f"{r['reject_own_1f']:>10.1%} {r['reject_new_gated']:>13.1%} "
                      f"{r['reject_own_gated']:>13.1%}")
        report[arm] = arm_out
        del d

    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}")
    print(
        "\nBoth 'rej own' columns are FALSE rejections — an opposite-label rewrite is the\n"
        "one thing the guard must never drop. 'rej new' is also a false rejection here,\n"
        "since every one of these successes is a genuinely new error. No column measures\n"
        "correct rejections: this corpus has no clones (see the module docstring)."
    )


if __name__ == "__main__":
    main()
