#!/usr/bin/env python
"""Probe scores for the successes and their partners, before and after training on them.

`pair_probe_scores.py` scored each pair against the probe of the day — the one it was
submitted against. This is the flat before/after: every source and every generated partner
scored by

* **`probe_iter0`** — the from-scratch probe, trained on the 50 base rows and nothing else.
  BEFORE: neither half was in its training set, so both are fully out-of-sample here.
* **`probe_iter5`** — the final probe, after five retrain cycles on these very rows. AFTER:
  both halves are now in-sample, so this measures how the boundary moved to fit them.

Reported per arm, split into source vs partner and by true class, as: mean/median score,
share the probe assigns to the positive class, and share it classifies correctly (prediction
== the label the retrain used — the judge's for a source, the generation target for a
partner). No LLM is loaded; every row is already in the ceiling analysis's pool.

    analysis/offdist/before_after_scores.py [--arms ...] [--before 0] [--after 5]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import od_common as O  # noqa: E402
from ablate import N_BASE  # noqa: E402
import pair_probe_scores as PPS  # noqa: E402

sys.path.insert(0, str(O.REPO / "ceiling_analysis" / "scripts"))
import ca_common as C  # noqa: E402

THRESHOLD = 0.5  # the run's own `probe.threshold`


def score_all(arm: O.Arm, iteration: int, src, idx) -> np.ndarray:
    with (arm.probe_dir / f"probe_iter{iteration}.pkl").open("rb") as f:
        probe = pickle.load(f)
    s = C.score_source(probe, src, [N_BASE + i for i in idx], chunk=64)
    del probe
    C.free_gpu()
    return np.asarray(s)


def stats(scores: np.ndarray, truth: np.ndarray) -> dict:
    pred_pos = scores >= THRESHOLD
    return {
        "n": int(len(scores)),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "std": float(scores.std()),
        "predicted_positive": float(pred_pos.mean()),
        "correct": float((pred_pos == truth).mean()),
    }


def run_arm(arm: O.Arm, args) -> dict:
    concept = C.CONCEPTS[arm.concept]
    rows = O.load_redteam(arm)
    label = {r["i"]: r["label"] for r in rows}
    flags = [json.loads(l) for l in
             (O.RESULTS / f"flags_{arm.key}.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    role = {f["i"]: f.get("pair_role") for f in flags}

    groups = {g: sorted(i for i in role if role[i] == g) for g in ("source", "generated")}
    src = C.redteam_source(concept)

    all_idx = sorted(i for g in groups.values() for i in g)
    before = dict(zip(all_idx, score_all(arm, args.before, src, all_idx)))
    after = dict(zip(all_idx, score_all(arm, args.after, src, all_idx)))

    out = {"arm": arm.key, "before_iter": args.before, "after_iter": args.after,
           "threshold": THRESHOLD, "groups": {}}

    # Why the BEFORE probe scores some successes correctly at all: a success fooled the
    # probe of the DAY (probe_iter{k}, the iteration it was found in), which is the before
    # probe only for that iteration's own batch. Breaking the source side down by true
    # class AND the iteration that found it shows exactly this — iteration-0 rows score ~0%
    # (the before probe IS the one they beat), while benign rows found later score ~100%
    # (the before probe predates the false-positive drift that made them successes).
    found = PPS.pair_iterations(arm, rows, flags)
    for cls in (O.POS, O.NEG):
        per_iter = {}
        for k in sorted(set(found.values())):
            g = [i for i in groups["source"] if label[i] == cls and found.get(i) == k]
            if not g:
                continue
            v = np.array([before[i] for i in g])
            per_iter[str(k)] = {"n": len(g),
                                "correct": float(np.mean((v >= THRESHOLD) == (cls == O.POS))),
                                "mean": float(v.mean())}
        out.setdefault("source_before_by_found_iter", {})[cls] = per_iter
    for g, idx in groups.items():
        truth = np.array([label[i] == O.POS for i in idx])
        entry = {}
        for when, sc in (("before", before), ("after", after)):
            v = np.array([sc[i] for i in idx])
            entry[when] = stats(v, truth)
            # by true class
            entry[when]["by_class"] = {
                O.POS: stats(v[truth], truth[truth]) if truth.any() else None,
                O.NEG: stats(v[~truth], truth[~truth]) if (~truth).any() else None,
            }
        out["groups"][g] = entry

    O.write_json(O.RESULTS / f"before_after_scores_{arm.key}.json", out)
    return out


def report(out: dict) -> None:
    a, b_it, a_it = out["arm"], out["before_iter"], out["after_iter"]
    print(f"\n=== {a} — probe scores before (iter{b_it}) and after (iter{a_it}) training on "
          f"these rows (threshold {out['threshold']}) ===")
    print(f"{'half':<12}{'n':>5}{'':>4}{'mean score':>22}{'% predicted positive':>24}"
          f"{'% classified correctly':>26}")
    print(f"{'':<12}{'':>5}{'':>4}{'before -> after':>22}{'before -> after':>24}"
          f"{'before -> after':>26}")
    for g, lab in (("source", "successes"), ("generated", "partners")):
        e = out["groups"][g]
        bf, af = e["before"], e["after"]
        print(f"{lab:<12}{bf['n']:>5}{'':>4}"
              f"{bf['mean']:>10.3f} ->{af['mean']:>8.3f}"
              f"{bf['predicted_positive']:>13.1%} ->{af['predicted_positive']:>8.1%}"
              f"{bf['correct']:>14.1%} ->{af['correct']:>9.1%}")
    print("  by true class (share classified correctly, before -> after):")
    for g, lab in (("source", "successes"), ("generated", "partners")):
        e = out["groups"][g]
        for cls in (O.POS, O.NEG):
            cb, ca = e["before"]["by_class"][cls], e["after"]["by_class"][cls]
            if cb and ca:
                print(f"    {lab:<10} {cls:<22} n={cb['n']:>4}  "
                      f"{cb['correct']:>6.1%} -> {ca['correct']:>6.1%}")
    brk = out.get("source_before_by_found_iter")
    if brk:
        print("  why iter{}'s correct rate on the successes is not 0 — success is defined vs "
              "the probe of the DAY, not iter{}:".format(out["before_iter"], out["before_iter"]))
        print("    (successes, split by true class x the iteration that found them, "
              "iter{} correct)".format(out["before_iter"]))
        for cls, per in brk.items():
            cells = "  ".join(f"it{k}:{d['correct']:.0%}(n={d['n']})" for k, d in per.items())
            print(f"    {cls:<22} {cells}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(O.ARMS))
    ap.add_argument("--before", type=int, default=0)
    ap.add_argument("--after", type=int, default=5)
    args = ap.parse_args()
    for key in args.arms:
        report(run_arm(O.ARMS[key], args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
