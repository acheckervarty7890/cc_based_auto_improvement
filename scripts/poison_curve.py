#!/usr/bin/env python
"""How much rejected data does `base ∪ 62 ∪ 107 generated` absorb before it falls back?

Starts from the strongest probe this arm produced and adds the loop's REJECTED batches
ten rows at a time, refitting and re-evaluating after each one, until the eval mean drops
to or below `base ∪ 62` (probe_iter13, 0.8148).

Batches are added in the run's own chronological order — (iteration, batch_index) — which
is arbitrary with respect to how harmful they are, so this measures the typical case, not
the worst one. Only batches carrying the full ten samples are used; the short,
connection-truncated ones are skipped so every step adds the same dose.

Writes one CSV row per step as it goes, so a long run is readable while it is still going.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RUN_DIR = REPO / "results_gen_gemma27b_instructions_nemotron"
PROBE_DIR = REPO / "probes/gen_gemma27b_instructions_nemotron"
BASE_DATA = REPO / "data/instructions_llama70b_50.jsonl"
DEV_DATA = REPO / "dev_samples/instructions"
EVAL_DIR = REPO / "eval_sets/instructions"
BASE_CACHE = REPO / "cache_gen_gemma27b_instructions/base_activations"
EVAL_CACHE = REPO / "cache_gen_gemma27b_instructions/eval_activations"

BASE_PROBE = PROBE_DIR / "probe_iter12.pkl"  # byte-identical to probe_iter13
ACCEPTED = RUN_DIR / "accepted_iter13.jsonl"
GENERATED = REPO / "data/instructions_like_accepted62.jsonl"

TARGET = 0.8148  # base ∪ 62 = probe_iter13
SEED = 42
COMBINE = True
CONVERT = True

SPLITS = [
    "anthropic_harmless_refusal", "bbq_substitution", "hc_context_drift",
    "hc_contradiction", "mm_substitution", "oig_context_drift", "oig_omission",
]


def latest_batches() -> dict[tuple[int, int], dict]:
    latest: dict[tuple[int, int], dict] = {}
    with (RUN_DIR / "batches.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            latest[(r["iteration"], r["batch_index"])] = r
    return latest


def jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({"inputs": r["inputs"], "labels": r["labels"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-steps", type=int, default=30, help="stop after this many rejected batches even if still above target")
    ap.add_argument("--order", choices=["chronological", "worst-first"], default="chronological")
    ap.add_argument("--out", type=Path, default=RUN_DIR / "poison_curve.csv")
    args = ap.parse_args()

    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import retrain_probe

    latest = latest_batches()
    rejected = [
        r for r in latest.values()
        if r["status"] == "scored" and not r["accepted"] and len(r["samples"]) == 10
    ]
    if args.order == "chronological":
        rejected.sort(key=lambda r: (r["iteration"], r["batch_index"]))
    else:
        rejected.sort(key=lambda r: r["delta"])
    print(f"{len(rejected)} rejected batches of exactly 10 samples, order={args.order}")

    floor = jsonl(ACCEPTED) + jsonl(GENERATED)
    print(f"starting set: base 50 + {len(floor)} (62 accepted + 107 generated)")
    print(f"target (base ∪ 62): eval mean {TARGET:.4f}\n")

    fh = args.out.open("w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["step", "batch", "delta_at_scoring", "n_rows", "dev_mean", "eval_mean",
                     "vs_target", *SPLITS])

    def run(step: int, tag: str, samples: list[dict], scoring_delta) -> float:
        out = PROBE_DIR / f"poison_step{step:02d}.pkl"
        res = retrain_probe(
            samples=samples,
            base_probe_path=BASE_PROBE,
            base_training_data_path=BASE_DATA,
            new_probe_path=out,
            dev_data_path=DEV_DATA,
            seed=SEED,
            base_data_fraction=1.0,
            base_activation_cache_dir=BASE_CACHE,
            combine_consecutive_messages=COMBINE,
            convert_tool_to_assistant=CONVERT,
            verbose=False,
        )
        df = evaluate_probe(
            out, EVAL_DIR, EVAL_CACHE, max_samples=None, seed=SEED,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
        )
        per = {r["dataset"]: float(r["auroc"]) for _, r in df.iterrows()}
        dev, ev = res.dev_auroc["mean"], per["mean"]
        writer.writerow([step, tag, scoring_delta, len(samples), f"{dev:.5f}", f"{ev:.5f}",
                         f"{ev - TARGET:+.5f}", *[f"{per[s]:.5f}" for s in SPLITS]])
        fh.flush()
        print(f"STEP {step:>2}  +{tag:<7}  rows {len(samples):>3}  dev {dev:.5f}  "
              f"eval {ev:.5f}  vs target {ev - TARGET:+.5f}")
        out.unlink(missing_ok=True)  # the curve is the artifact, not 30 pickles
        return ev

    ev = run(0, "none", floor, "")
    if ev <= TARGET:
        print("\nAlready at or below target before adding anything.")
        fh.close()
        return

    cur = list(floor)
    for step, rec in enumerate(rejected[: args.max_steps], start=1):
        tag = f"it{rec['iteration']}b{rec['batch_index']}"
        cur = cur + [{"inputs": s["messages"], "labels": s["label"]} for s in rec["samples"]]
        ev = run(step, tag, cur, f"{rec['delta']:+.4f}")
        if ev <= TARGET:
            print(f"\nBack to base ∪ 62 ({TARGET:.4f}) after {step} rejected batches "
                  f"({step * 10} rows).")
            break
    else:
        print(f"\nStill above target after {min(len(rejected), args.max_steps)} batches.")
    fh.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
