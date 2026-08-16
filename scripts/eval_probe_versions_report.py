#!/usr/bin/env python
"""Summarize scripts/eval_probe_versions.py output, and check it against the original runs.

Two things this prints:

**The comparison itself** — mean AUROC / accuracy / TPR@1%FPR per (concept, arm, iteration),
plus the per-split AUROC grid, which is where the interesting variation lives: a mean that
barely moves can hide one split gaining 0.2 while another loses it.

**A reproduction check.** Every one of these probes was already scored once, in-run, by
``cli.iterative_retrain_main --eval``; those CSVs are committed on each experiment's branch.
Re-scoring them here off the published Kaggle activations should reproduce those numbers, and
the size of the residual is the interesting quantity: accuracy and TPR are expected to match
exactly, while AUROC can differ by ~1e-3 because it depends on the *ordering* of scores and
near-ties get reordered by float non-determinism between one GPU and another. A residual much
larger than that would mean the blobs, the splits or the transforms are not what the original
run used — i.e. the numbers below are not comparable to the published ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The in-run comparison CSVs, on each experiment's own branch (via the worktrees).
ORIGINAL_CSVS = {
    ("hs", "gptoss120b"): "/home/ubuntu/wt_exp9/results_hs_gemma27b_gptoss120b_batch/gptoss120b_comparison.csv",
    ("hs", "deepseekv4pro"): "/home/ubuntu/wt_exp9/results_hs_gemma27b_deepseekv4pro_batch/deepseekv4pro_comparison.csv",
    ("instructions", "gptoss"): "/home/ubuntu/wt_instr/results_instructions_gemma27b_gptoss/gptoss120b_comparison.csv",
    ("instructions", "nemotron"): "/home/ubuntu/wt_instr/results_instructions_gemma27b_nemotron/nemotron_comparison.csv",
}

METRICS = ["auroc", "accuracy", "tpr_at_fpr"]


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(REPO_ROOT / "results_probe_versions" / "eval_rows.jsonl"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results_probe_versions"))
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 200, "display.max_columns", 50)

    for experiment, edf in df.groupby("experiment"):
        print(f"\n{'=' * 78}\n{experiment}  ({edf['concept'].iloc[0]})\n{'=' * 78}")

        means = (
            edf.groupby(["arm", "iteration"])[METRICS].mean().round(4).reset_index()
        )
        print("\nMean over splits (unweighted, matching the in-run CSVs' 'mean' row):")
        print(means.to_string(index=False))

        grid = edf.pivot_table(index=["arm", "iteration"], columns="split", values="auroc").round(4)
        print("\nPer-split AUROC:")
        print(grid.to_string())

        grid.to_csv(out_dir / f"{experiment}_auroc_by_split.csv")
        means.to_csv(out_dir / f"{experiment}_means.csv", index=False)

    # --- reproduction check against the original in-run CSVs -------------------------
    print(f"\n{'=' * 78}\nReproduction vs. the original in-run comparison CSVs\n{'=' * 78}")
    print(f"{'experiment/arm':<28} {'n':>4}  " + "  ".join(f"{'max |Δ ' + m + '|':>16}" for m in METRICS))
    for (experiment, arm), path in ORIGINAL_CSVS.items():
        p = Path(path)
        if not p.exists():
            print(f"{experiment + '/' + arm:<28} original CSV not found at {p}")
            continue
        orig = pd.read_csv(p)
        orig = orig[orig["dataset"] != "mean"].copy()
        orig["iteration"] = orig["round"].str.removeprefix("iter").astype(int)
        orig = orig.rename(columns={"dataset": "split"})

        mine = df[(df["experiment"] == experiment) & (df["arm"] == arm)]
        merged = mine.merge(orig, on=["split", "iteration"], suffixes=("_new", "_orig"))
        if merged.empty:
            print(f"{experiment + '/' + arm:<28} no overlapping rows")
            continue
        deltas = [f"{(merged[m + '_new'] - merged[m + '_orig']).abs().max():>16.2e}" for m in METRICS]
        print(f"{experiment + '/' + arm:<28} {len(merged):>4}  " + "  ".join(deltas))

    print(
        "\nExpected: |Δaccuracy| and |Δtpr| ~0 (identical thresholded predictions), |Δauroc| "
        "up to ~1e-3 (score ties reordered by float non-determinism across GPUs)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
