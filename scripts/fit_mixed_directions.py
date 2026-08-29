#!/usr/bin/env python
"""Fit `base ∪ 62 accepted ∪ <generated>` and score it against `base ∪ 62`.

Companion to `fit_with_generated.py`, which floors on `base ∪ 62 ∪ it10b0`. Here the
floor is the loop's own final training set — `probe_iter13` — so every candidate answers
one question: what do these extra rows add to the probe the loop actually produced?

    --score-families   one candidate fit per generated family, dev AND eval.
    (default)          fit the union of --families and eval it once.

Every knob matches the arm 3N config and the CLI defaults, so each fit is
apples-to-apples with probe_iter13.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RUN_DIR = REPO / "results_gen_gemma27b_instructions_nemotron"
PROBE_DIR = REPO / "probes/gen_gemma27b_instructions_nemotron"
BASE_DATA = REPO / "data/instructions_llama70b_50.jsonl"
DEV_DATA = REPO / "dev_samples/instructions"
EVAL_DIR = REPO / "eval_sets/instructions"
BASE_CACHE = REPO / "cache_gen_gemma27b_instructions/base_activations"
EVAL_CACHE = REPO / "cache_gen_gemma27b_instructions/eval_activations"

BASE_PROBE = PROBE_DIR / "probe_iter12.pkl"  # byte-identical to probe_iter13
ACCEPTED = RUN_DIR / "accepted_iter13.jsonl"

SEED = 42
COMBINE = True
CONVERT = True


def load_accepted() -> list[dict]:
    with ACCEPTED.open() as fh:
        return [{"inputs": json.loads(l)["inputs"], "labels": json.loads(l)["labels"]} for l in fh]


def load_generated(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({"inputs": r["inputs"], "labels": r["labels"], "family": r.get("family", "?")})
    return rows


def fit(samples: list[dict], out: Path, label: str):
    from agentic_redteam.retrain import retrain_probe

    print(f"\n--- {label}: base ∪ {len(samples)} samples → {out.name} ---", flush=True)
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


def eval_mean(probe_path: Path):
    from agentic_redteam.evaluation import evaluate_probe

    df = evaluate_probe(
        probe_path, EVAL_DIR, EVAL_CACHE,
        max_samples=None, seed=SEED,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
    )
    return float(df.loc[df["dataset"] == "mean", "auroc"].iloc[0]), df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated", type=Path, default=REPO / "data/instructions_mixed_directions.jsonl")
    ap.add_argument("--families", default=None, help="comma-separated family ids (default: all)")
    ap.add_argument("--score-families", action="store_true")
    ap.add_argument("--tag", default="mixed")
    ap.add_argument("--warm-only", action="store_true", help="extract activations and stop")
    args = ap.parse_args()

    from agentic_redteam.retrain import warm_sample_activation_cache

    accepted = load_accepted()
    generated = load_generated(args.generated)
    by_family = collections.OrderedDict()
    for r in generated:
        by_family.setdefault(r["family"], []).append(r)

    print(f"accepted 62 : {len(accepted)}")
    print(f"generated   : {len(generated)} across {len(by_family)} families")
    for f, rows in by_family.items():
        npos = sum(1 for r in rows if not r["labels"].startswith("assistant_does_not"))
        print(f"    {f:>8}: {len(rows):>3}  ({npos} pos / {len(rows) - npos} neg)")

    # One model load covers every uncached conversation in play.
    n_new = warm_sample_activation_cache(
        accepted + generated,
        base_probe_path=BASE_PROBE,
        base_activation_cache_dir=BASE_CACHE,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
        verbose=True,
    )
    print(f"\nextracted {n_new} new conversation activation(s)")
    if args.warm_only:
        return

    floor_probe = PROBE_DIR / f"cand_{args.tag}_floor.pkl"
    floor = fit(accepted, floor_probe, "floor (base + 62 accepted)")
    floor_dev = floor.dev_auroc["mean"]
    floor_eval, _ = eval_mean(floor_probe)
    print(f"\nfloor: dev {floor_dev:.5f}   eval {floor_eval:.5f}   "
          f"(published probe_iter13: dev 0.8311, eval 0.8148)")

    rows_out = [{"set": "floor (base+62)", "n_added": 0, "dev": floor_dev, "eval": floor_eval,
                 "d_dev": 0.0, "d_eval": 0.0, "sources": ""}]

    src = {d["family"]: "+".join(d["sources"])
           for d in json.loads((RUN_DIR / "mixed_directions.json").read_text())}

    if args.score_families:
        for f, rows in by_family.items():
            out = PROBE_DIR / f"cand_{args.tag}_{f}.pkl"
            res = fit(accepted + rows, out, f"floor + {f}")
            dev = res.dev_auroc["mean"]
            ev, _ = eval_mean(out)
            rows_out.append({"set": f, "n_added": len(rows), "dev": dev, "eval": ev,
                             "d_dev": dev - floor_dev, "d_eval": ev - floor_eval,
                             "sources": src.get(f, "")})
            print(f"    {f}: dev {dev:.5f} ({dev - floor_dev:+.5f})   "
                  f"eval {ev:.5f} ({ev - floor_eval:+.5f})", flush=True)

    keep = set(args.families.split(",")) if args.families else set(by_family)
    chosen = [r for r in generated if r["family"] in keep]
    union_out = PROBE_DIR / f"gen_{args.tag}_union.pkl"
    res = fit(accepted + chosen, union_out, f"floor + all {len(chosen)} generated")
    dev = res.dev_auroc["mean"]
    ev, df = eval_mean(union_out)
    rows_out.append({"set": f"union ({len(keep)} families)", "n_added": len(chosen), "dev": dev,
                     "eval": ev, "d_dev": dev - floor_dev, "d_eval": ev - floor_eval, "sources": ""})
    print(df.to_string(index=False))

    import csv
    csv_path = RUN_DIR / f"{args.tag}_directions_results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    print(f"\n{'set':>22} {'n':>4} {'dev':>8} {'Δdev':>9} {'eval':>8} {'Δeval':>9}  sources")
    for r in rows_out:
        print(f"{r['set']:>22} {r['n_added']:>4} {r['dev']:>8.5f} {r['d_dev']:>+9.5f} "
              f"{r['eval']:>8.5f} {r['d_eval']:>+9.5f}  {r['sources']}")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
