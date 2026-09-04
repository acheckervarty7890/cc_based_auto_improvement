#!/usr/bin/env python3
"""Compare N replicates of ARM 7's iteration 4 against the original run.

Iteration 4 of the nemotron / high-stakes / memo-only arm retrained probe_iter4
(eval mean 0.76858) into probe_iter5 (0.81088) — a +0.0423 jump. Each replicate
re-ran that one cycle from the same state, with the same cross-iteration memo,
and differs only in LLM sampling. This prints:

  1. a sanity row — every replicate re-evaluates probe_iter4 before it starts, and
     that number MUST reproduce the original's 0.768578; if it does not, the
     reconstruction is wrong and nothing below means anything;
  2. the four finals and their spread, against the original as a fifth point;
  3. per-split finals, since the arm's mean moves mostly through mts/mt;
  4. what each replicate FOUND — attempts and successes by error type;
  5. how much the replicates' successes overlap each other and the original,
     by canonical conversation text (the same key JsonlStore dedups on).

Usage: compare_iter4_replicates.py [--reps 1 2 3 4]
"""
from __future__ import annotations
import argparse, csv, json, statistics
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIG_RES = ROOT / "results_hs_gemma27b_nemotron_nemobase_itermemo150"
ORIG_STEM = "nemobase_itermemo150_probing"
ORIG_CSV = "nemobase_itermemo150_comparison.csv"
SPLITS = ["anthropic_hh_balanced", "mt_balanced", "mts_balanced", "toolace_balanced"]
ITER = 4


def read_csv(path: Path) -> dict[str, dict[str, float]]:
    """{round_label: {dataset: auroc}} — 'mean' included, it is a row in the CSV."""
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["round"], {})[row["dataset"]] = float(row["auroc"])
    return out


def canon(sample) -> str:
    """Canonical conversation text — mirrors what JsonlStore dedups on."""
    msgs = sample if isinstance(sample, list) else sample.get("messages", [])
    return "\n".join(f"{m.get('role')}:{m.get('content')}" for m in msgs)


def load_attempts(res_dir: Path, stem: str, iteration: int) -> dict[str, list[dict]]:
    """{'fp': [...], 'fn': [...]} — rows of one iteration only."""
    out = {}
    for et in ("fp", "fn"):
        path = res_dir / f"{stem}_{et}.jsonl"
        rows = []
        if path.exists():
            with path.open() as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("iteration") == iteration:
                        rows.append(r)
        out[et] = rows
    return out


def fmt(v, w=8):
    return f"{v:>{w}.5f}" if isinstance(v, float) else f"{str(v):>{w}}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()

    orig_csv = read_csv(ORIG_RES / ORIG_CSV)
    start = orig_csv.get(f"iter{ITER}", {}).get("mean")
    orig_final = orig_csv.get(f"iter{ITER+1}", {}).get("mean")

    runs = []           # (label, csv dict, results dir, stem)
    for n in args.reps:
        res = ROOT / f"results_hs_nm_iter4_rep{n}"
        runs.append((f"rep{n}", read_csv(res / f"rep{n}_comparison.csv"), res, f"rep{n}_probing"))

    print(f"ARM 7 (nemotron · high-stakes · memo-only) — iteration {ITER} replicated "
          f"{len(runs)}x\ninput probe probe_iter{ITER}.pkl · output probe_iter{ITER+1}.pkl\n")

    # 1. sanity ---------------------------------------------------------------
    print("1. RECONSTRUCTION CHECK — each replicate re-evaluates the input probe first")
    print(f"   original  iter{ITER} mean = {start:.6f}" if start else "   original: CSV missing")
    for label, c, *_ in runs:
        got = c.get(f"iter{ITER}", {}).get("mean")
        if got is None:
            print(f"   {label:<6} iter{ITER} mean =    (pending)")
        else:
            ok = "OK" if start is not None and abs(got - start) < 1e-9 else "*** MISMATCH ***"
            print(f"   {label:<6} iter{ITER} mean = {got:.6f}   {ok}")
    print()

    # 2. finals ---------------------------------------------------------------
    print(f"2. FINALS — probe_iter{ITER+1} eval mean (start {start:.5f})" if start else "2. FINALS")
    finals = []
    if orig_final is not None:
        print(f"   {'original':<10}{fmt(orig_final)}   delta {orig_final-start:+.5f}   (ran across the abort/resume)")
        finals.append(("original", orig_final))
    for label, c, *_ in runs:
        v = c.get(f"iter{ITER+1}", {}).get("mean")
        if v is None:
            print(f"   {label:<10}{'pending':>8}")
        else:
            print(f"   {label:<10}{fmt(v)}   delta {v-start:+.5f}")
            finals.append((label, v))
    vals = [v for _, v in finals]
    reps_only = [v for l, v in finals if l != "original"]
    if len(vals) >= 2:
        print(f"\n   all {len(vals)}      mean {statistics.mean(vals):.5f}  sd {statistics.stdev(vals):.5f}  "
              f"range {min(vals):.5f}-{max(vals):.5f}  spread {max(vals)-min(vals):.5f}")
    if len(reps_only) >= 2:
        print(f"   replicates  mean {statistics.mean(reps_only):.5f}  sd {statistics.stdev(reps_only):.5f}  "
              f"range {min(reps_only):.5f}-{max(reps_only):.5f}  spread {max(reps_only)-min(reps_only):.5f}")
        if orig_final is not None:
            z = (orig_final - statistics.mean(reps_only)) / statistics.stdev(reps_only)
            print(f"   original sits {z:+.2f} sd from the replicate mean")
    if start is not None and reps_only:
        n_up = sum(v > start for v in reps_only)
        print(f"   replicates that improved on the input probe: {n_up}/{len(reps_only)}")
    print()

    # 3. per split ------------------------------------------------------------
    print(f"3. PER SPLIT — probe_iter{ITER+1}")
    print(f"   {'':<10}" + "".join(f"{s[:14]:>16}" for s in SPLITS) + f"{'mean':>10}")
    rows = [(f"iter{ITER} start", orig_csv.get(f"iter{ITER}", {})),
            ("original", orig_csv.get(f"iter{ITER+1}", {}))]
    rows += [(l, c.get(f"iter{ITER+1}", {})) for l, c, *_ in runs]
    for label, d in rows:
        if not d:
            continue
        print(f"   {label:<10}" + "".join(f"{d.get(s, float('nan')):>16.5f}" for s in SPLITS)
              + f"{d.get('mean', float('nan')):>10.5f}")
    print()

    # 4. yield ----------------------------------------------------------------
    print(f"4. WHAT EACH RUN FOUND in iteration {ITER}")
    print(f"   {'':<10}{'fp att':>8}{'fp succ':>9}{'fn att':>8}{'fn succ':>9}{'total att':>11}{'total succ':>12}{'rate':>8}")
    attempts = {"original": load_attempts(ORIG_RES, ORIG_STEM, ITER)}
    for label, _, res, stem in runs:
        attempts[label] = load_attempts(res, stem, ITER)
    for label, per_et in attempts.items():
        fa, fs = len(per_et["fp"]), sum(bool(r.get("success")) for r in per_et["fp"])
        na, ns = len(per_et["fn"]), sum(bool(r.get("success")) for r in per_et["fn"])
        ta, ts = fa + na, fs + ns
        rate = f"{100*ts/ta:.1f}%" if ta else "-"
        print(f"   {label:<10}{fa:>8}{fs:>9}{na:>8}{ns:>9}{ta:>11}{ts:>12}{rate:>8}")
    print()

    # 5. overlap --------------------------------------------------------------
    print(f"5. OVERLAP of iteration-{ITER} SUCCESSES (canonical conversation text)")
    succ = {lab: {canon(r["sample"]) for et in ("fp", "fn") for r in per_et[et] if r.get("success")}
            for lab, per_et in attempts.items()}
    labels = [l for l in succ if succ[l]]
    for a, b in combinations(labels, 2):
        inter = len(succ[a] & succ[b])
        union = len(succ[a] | succ[b])
        print(f"   {a:<10} n {b:<10} shared {inter:>4} / union {union:>4}   jaccard {inter/union:.3f}"
              if union else f"   {a} n {b}: both empty")
    if len(labels) > 1:
        everywhere = set.intersection(*(succ[l] for l in labels))
        anywhere = set.union(*(succ[l] for l in labels))
        print(f"\n   found by ALL {len(labels)} runs: {len(everywhere)}   "
              f"found by at least one: {len(anywhere)}")


if __name__ == "__main__":
    main()
