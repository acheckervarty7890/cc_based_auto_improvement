#!/usr/bin/env python
"""Fit `base ∪ 62 accepted ∪ it10b0 ∪ <generated>` and score it.

Companion to `scripts/ablation_twin_injection.py`. The reference points are:

    base ∪ 62                 = probe_iter13   dev 0.8311   eval 0.8148
    base ∪ 62 ∪ it10b0        = ablation       dev 0.8220   eval 0.8108

This script adds generated rows on top of the second and asks whether they recover the
loss. Two modes:

    --score-families   one candidate fit per generated family, dev AUROC only. No eval is
                       computed, so family selection can be made without looking at the
                       test set.
    (default)          fit the union of `--families` (all of them unless restricted) and
                       run the full eval once.

Every knob matches the arm 3N config and the CLI defaults, so each fit is
apples-to-apples with probe_iter13.
"""

from __future__ import annotations

import argparse
import collections
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
INJECTED_KEY = (10, 0)

SEED = 42
COMBINE = True
CONVERT = True

# Measured, for the printed comparison.
REF = {
    "base+62": {"dev": 0.8311, "eval": 0.8148},
    "base+62+it10b0": {"dev": 0.8220, "eval": 0.8108},
}


def load_accepted() -> list[dict]:
    rows = []
    with ACCEPTED.open() as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({"inputs": r["inputs"], "labels": r["labels"]})
    return rows


def load_injected() -> list[dict]:
    latest = {}
    with (RUN_DIR / "batches.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            latest[(r["iteration"], r["batch_index"])] = r
    rec = latest[INJECTED_KEY]
    return [{"inputs": s["messages"], "labels": s["label"]} for s in rec["samples"]]


def load_generated(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({"inputs": r["inputs"], "labels": r["labels"], "family": r.get("family", "?")})
    return rows


def fit(samples: list[dict], out: Path, label: str):
    from agentic_redteam.retrain import retrain_probe

    print(f"\n--- {label}: base ∪ {len(samples)} samples → {out.name} ---")
    return retrain_probe(
        samples=[{"inputs": s["inputs"], "labels": s["labels"]} for s in samples],
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated", type=Path, default=REPO / "data/instructions_like_accepted62.jsonl")
    ap.add_argument("--families", default=None, help="comma-separated family ids to include (default: all)")
    ap.add_argument("--score-families", action="store_true", help="dev-only score of each family separately")
    ap.add_argument("--tag", default="all", help="name for the output probe / csv")
    args = ap.parse_args()

    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import warm_sample_activation_cache

    accepted, injected = load_accepted(), load_injected()
    generated = load_generated(args.generated)
    by_family = collections.OrderedDict()
    for r in generated:
        by_family.setdefault(r["family"], []).append(r)

    print(f"accepted 62        : {len(accepted)}")
    print(f"injected it10b0    : {len(injected)}")
    print(f"generated          : {len(generated)} across {len(by_family)} families")
    for f, rows in by_family.items():
        npos = sum(1 for r in rows if not r["labels"].startswith("assistant_does_not"))
        print(f"    {f:>8}: {len(rows):>3}  ({npos} pos / {len(rows) - npos} neg)")

    floor = accepted + injected

    # One model load covers every uncached conversation in play.
    warm_sample_activation_cache(
        floor + generated,
        base_probe_path=BASE_PROBE,
        base_activation_cache_dir=BASE_CACHE,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
        verbose=True,
    )

    print("\nreference points")
    for k, v in REF.items():
        print(f"    {k:<16} dev {v['dev']:.4f}   eval {v['eval']:.4f}")

    if args.score_families:
        base = fit(floor, PROBE_DIR / "cand_floor.pkl", "floor (62 + it10b0)")
        floor_dev = base.dev_auroc["mean"]
        print(f"floor dev mean: {floor_dev:.5f}")
        print(f"\n{'family':>8} {'n':>3} {'dev mean':>9} {'Δ vs floor':>11}")
        for f, rows in by_family.items():
            res = fit(floor + rows, PROBE_DIR / f"cand_{f}.pkl", f"floor + {f}")
            d = res.dev_auroc["mean"]
            print(f"{f:>8} {len(rows):>3} {d:>9.5f} {d - floor_dev:>+11.5f}")
        return

    keep = set(args.families.split(",")) if args.families else set(by_family)
    chosen = [r for r in generated if r["family"] in keep]
    print(f"\nincluding {len(chosen)} generated rows from families: {sorted(keep)}")

    out = PROBE_DIR / f"gen_{args.tag}.pkl"
    res = fit(floor + chosen, out, f"62 + it10b0 + {len(chosen)} generated")
    dev = res.dev_auroc["mean"]
    print(f"training rows: {res.n_training_samples_total}")
    print(f"dev  mean: {dev:.5f}   (base+62 = {REF['base+62']['dev']:.4f}, "
          f"+it10b0 = {REF['base+62+it10b0']['dev']:.4f})")

    df = evaluate_probe(
        out, EVAL_DIR, EVAL_CACHE,
        max_samples=None, seed=SEED,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
    )
    print(df.to_string(index=False))
    csv = RUN_DIR / f"gen_{args.tag}_eval.csv"
    df.to_csv(csv, index=False)
    mean = float(df.loc[df["dataset"] == "mean", "auroc"].iloc[0])
    print(f"\neval mean: {mean:.5f}")
    print(f"  vs base+62        ({REF['base+62']['eval']:.4f}): {mean - REF['base+62']['eval']:+.5f}")
    print(f"  vs base+62+it10b0 ({REF['base+62+it10b0']['eval']:.4f}): {mean - REF['base+62+it10b0']['eval']:+.5f}")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
