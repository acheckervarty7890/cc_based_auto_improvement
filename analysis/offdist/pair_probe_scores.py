#!/usr/bin/env python
"""Would the probe have got the *generated* half wrong too?

Every red-team success is, by construction, a conversation the live probe misclassified —
that is what made it a success. Its contrastive partner is not: it was written afterwards by
`generate_contrastive_dataset` and went straight into the next retrain **without ever being
scored**. So one half of every pair has a known verdict against the probe of the day and the
other half has none at all.

This fills that gap. Each partner is scored by `probe_iter{k}`, where `k` is the iteration
its *source* was found in — the last probe that had not yet seen either half of the pair, and
the one the source itself defeated. The question is whether the generation step is quietly
manufacturing a second misclassification per pair, or whether the partners are rows the probe
already handled correctly.

Two properties make the number readable:

* **The sources are the control, and their answer is known in advance.** They must come out
  at 0% correct — they are successes, i.e. rows where probe and judge disagreed. Anything
  else means the row-to-probe wiring is wrong, so it is measured rather than assumed.
* **Nothing here is in-sample.** `probe_iter{k}` is trained on successes from iterations
  `< k`, and this pair was found *at* `k`. `--also-final` additionally scores everything with
  `probe_iter5`, which has seen every row and is reported separately for exactly that reason.

Labels are the ones the retrain used: the judge's for a source, the generation target for a
partner. The threshold is the run's own (`probe.threshold`). No LLM is loaded — every
conversation is already in the pool the ceiling analysis built.

    analysis/offdist/pair_probe_scores.py [--arms ...] [--also-final]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import od_common as O  # noqa: E402
from ablate import N_BASE  # noqa: E402

sys.path.insert(0, str(O.REPO / "ceiling_analysis" / "scripts"))
import ca_common as C  # noqa: E402

RESULTS_DIR = {
    "gptoss120b": ("results_hu_harm_gemma27b_gptoss120b_datadesc", "gptoss120b_datadesc"),
    "deepseekv4pro": ("results_hu_harm_gemma27b_deepseekv4pro_datadesc",
                      "deepseekv4pro_datadesc"),
}


def successes_by_text(arm: O.Arm) -> dict[str, dict]:
    """Every attempt the run recorded as a success, keyed by its conversation text.

    Both error-type logs, since a rotation attacks each in turn and the postprocessed set
    mixes them. The row carries `iteration` (which probe was under attack) and `probe_score`
    (what that probe gave it), so the source side needs no re-scoring at all.
    """
    d, prefix = RESULTS_DIR[arm.key]
    out: dict[str, dict] = {}
    for et in ("fp", "fn"):
        path = O.REPO / d / f"{prefix}_probing_{et}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("success"):
                out[O.canonical(r["sample"]["messages"])] = r
    return out


def pair_iterations(arm: O.Arm, rows, flags) -> dict[int, int]:
    """Red-team row index -> the iteration its pair's SOURCE was found in.

    A partner inherits its source's iteration: the two enter training together, so the last
    probe that had seen neither is the one the source was submitted against.
    """
    succ = successes_by_text(arm)
    by_i = {r["i"]: r for r in rows}
    role = {f["i"]: f.get("pair_role") for f in flags}
    partner = {f["i"]: f.get("pair_with") for f in flags}
    out: dict[int, int] = {}
    for f in flags:
        if role.get(f["i"]) != "source":
            continue
        rec = succ.get(O.canonical(by_i[f["i"]]["messages"]))
        if rec is None:
            continue
        out[f["i"]] = int(rec["iteration"])
        j = partner.get(f["i"])
        if j is not None:
            out[j] = int(rec["iteration"])
    return out


def load_probe(arm: O.Arm, iteration: int):
    import pickle

    path = arm.probe_dir / f"probe_iter{iteration}.pkl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def run_arm(arm: O.Arm, args) -> dict:
    concept = C.CONCEPTS[arm.concept]
    rows = O.load_redteam(arm)
    flags = [json.loads(l) for l in
             (O.RESULTS / f"flags_{arm.key}.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    role = {f["i"]: f.get("pair_role") for f in flags}
    label = {r["i"]: r["label"] for r in rows}
    iters = pair_iterations(arm, rows, flags)

    missing = [i for i in role if i not in iters]
    if missing:
        print(f"  {len(missing)} rows could not be traced to a success record; skipped")

    src = C.redteam_source(concept)
    threshold = 0.5  # the run's own `probe.threshold`; stated rather than inferred

    # One probe load per iteration, scoring all of that iteration's rows at once.
    by_iter: dict[int, list[int]] = defaultdict(list)
    for i, k in iters.items():
        by_iter[k].append(i)

    scores: dict[int, float] = {}
    for k in sorted(by_iter):
        idx = sorted(by_iter[k])
        probe = load_probe(arm, k)
        s = C.score_source(probe, src, [N_BASE + i for i in idx], chunk=32)
        scores.update(dict(zip(idx, (float(v) for v in s))))
        print(f"  probe_iter{k}: scored {len(idx)} rows", flush=True)
        del probe
        C.free_gpu()

    final: dict[int, float] = {}
    if args.also_final:
        probe = load_probe(arm, args.final_iteration)
        idx = sorted(iters)
        s = C.score_source(probe, src, [N_BASE + i for i in idx], chunk=32)
        final = dict(zip(idx, (float(v) for v in s)))
        del probe
        C.free_gpu()

    def summarise(which: dict[int, float], group: str) -> dict:
        ids = [i for i in which if role.get(i) == group]
        sc = np.array([which[i] for i in ids])
        pos = sc >= threshold
        truth = np.array([label[i] == O.POS for i in ids])
        return {
            "n": len(ids),
            "mean_score": float(sc.mean()) if len(sc) else float("nan"),
            "predicted_positive": float(pos.mean()) if len(sc) else float("nan"),
            "correct": float((pos == truth).mean()) if len(sc) else float("nan"),
        }

    out = {
        "arm": arm.key,
        "threshold": threshold,
        "n_traced": len(iters),
        "per_iteration_counts": {str(k): len(v) for k, v in sorted(by_iter.items())},
        "probe_of_the_day": {g: summarise(scores, g) for g in ("source", "generated")},
    }
    if final:
        out["final_probe"] = {g: summarise(final, g) for g in ("source", "generated")}

    # Per-iteration breakdown for the partners: the interesting series, since a later probe
    # has been retrained on more of this attacker's material.
    per_it = {}
    for k, idx in sorted(by_iter.items()):
        gen = [i for i in idx if role.get(i) == "generated"]
        if not gen:
            continue
        sc = np.array([scores[i] for i in gen])
        truth = np.array([label[i] == O.POS for i in gen])
        per_it[str(k)] = {"n": len(gen),
                          "correct": float(((sc >= threshold) == truth).mean()),
                          "mean_score": float(sc.mean())}
    out["generated_by_iteration"] = per_it

    O.write_json(O.RESULTS / f"pair_probe_scores_{arm.key}.json", out)
    return out


def report(out: dict) -> None:
    a = out["arm"]
    print(f"\n=== {a} — scored by the probe under attack when the pair was found "
          f"(threshold {out['threshold']}) ===")
    print(f"{'half':<26}{'n':>6}{'mean score':>13}{'pred. positive':>16}{'probe correct':>15}")
    for g, lab in (("source", "the attacker's success"), ("generated", "its generated partner")):
        d = out["probe_of_the_day"][g]
        print(f"{lab:<26}{d['n']:>6}{d['mean_score']:>13.3f}"
              f"{d['predicted_positive']:>15.1%}{d['correct']:>15.1%}")
    if out.get("final_probe"):
        print("  (same rows under the final probe, which has trained on all of them: "
              + ", ".join(f"{g} {out['final_probe'][g]['correct']:.1%}"
                          for g in ("source", "generated")) + ")")
    if out.get("generated_by_iteration"):
        print("  partners by the iteration their pair was found in:")
        for k, d in out["generated_by_iteration"].items():
            print(f"    iter {k}: n={d['n']:>4}  probe correct {d['correct']:>6.1%}  "
                  f"mean score {d['mean_score']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(O.ARMS))
    ap.add_argument("--also-final", action="store_true",
                    help="also score every row with the final probe (in-sample; reported "
                         "separately for that reason)")
    ap.add_argument("--final-iteration", type=int, default=5)
    args = ap.parse_args()
    for key in args.arms:
        report(run_arm(O.ARMS[key], args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
