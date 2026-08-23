#!/usr/bin/env python
"""Turn the ceiling CV and the dev-sample sweep into tables, curves and a written answer.

Reads `results/ceiling_<concept>.json` and `results/sweep_<concept>.jsonl` and writes, per
concept, a CSV of the curve, a PNG of it against the ceiling, and a markdown section that
answers the question the analysis was set up to ask: **how many dev samples does red-team
training data need before eval performance reaches the ceiling** — separately for the two
ways of adding them.

"Reaches the ceiling" needs a definition that is not just eyeballing, so two are reported:

* `n_at_gap_frac` — the smallest N whose mean eval AUROC closes a given fraction (90% / 95%)
  of the gap between the N=0 red-team-only probe and the ceiling, *and* which no later point
  falls back below. The trailing condition matters: a single noisy point crossing early is
  not the answer to "how many samples are required".
* `n_within_tol` — the smallest such N landing within an absolute tolerance of the ceiling.

Both are computed on the across-seed mean, with the across-seed spread reported alongside so
a reader can see whether a crossing is inside the noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca_common as C  # noqa: E402

# batch_size x gradient_accumulation_steps for `linear_then_softmax`'s defaults. A training
# set smaller than this produces fewer batches per epoch than the accumulation period, so
# `(batch_idx + 1) % gradient_accumulation_steps == 0` never fires and `optimizer.step()` is
# never called — the head is returned at its random initialisation. tuberlens' loop does this
# too; it is not an artifact of the ragged transcription.
MIN_TRAIN_FOR_A_STEP = 16 * 4

ARM_LABEL = {
    "mixed": "mixed into the red-team training data",
    "finetune": "red-team first, then fine-tuned on the dev samples",
    "dev_only": "dev samples alone (control, no red-team data)",
}
ARM_COLOR = {"mixed": "#2f6fb5", "finetune": "#c8641c", "dev_only": "#6d6d6d"}


REFERENCE = {
    "highstakes": [
        ("experiment19 gpt-oss-120b, ens3, dev-validated",
         "exp19_hs_gptoss120b_ens3_comparison.csv"),
        ("experiment18 gpt-oss-120b, single probe, dev-validated",
         "exp18_hs_gptoss120b_devval_comparison.csv"),
    ],
    "hu_ha": [
        ("experiment17 gpt-oss-120b, ens10, dev-validated",
         "exp17_hu_harm_gptoss120b_ens10_devval_comparison.csv"),
    ],
    # experiment22's two dataset-description arms. Each arm's context is its OWN run, not
    # the other's: the probe description reaches the judge, so the arms' training labels are
    # not row-for-row comparable, and only their eval numbers read against each other.
    # Both CSVs are the run's comparison table with its backfilled early rounds merged in
    # (cli.py rewrites the CSV in process, so a resumed run's own file starts at the
    # iteration it resumed from).
    "hu_ha_dd_gptoss120b": [
        ("experiment22 arm 1 gpt-oss-120b, ens10, dev-validated",
         "exp22_hu_harm_gptoss120b_datadesc_comparison.csv"),
    ],
    "hu_ha_dd_deepseekv4pro": [
        ("experiment22 arm 2 deepseek-v4-pro, ens10, dev-validated",
         "exp22_hu_harm_deepseekv4pro_datadesc_comparison.csv"),
    ],
}


def reference_rows(concept: str) -> list[tuple[str, str, float]]:
    """(run, round, mean eval AUROC) from the experiment runs' own comparison CSVs.

    These probes were trained on base + red-team data with the whole dev set as validation —
    i.e. they are the runs this analysis is trying to put a scale under. Their numbers are
    not identical in setup to the N=0 point here (they are ensembles, and they early-stop
    against all 1908/290 dev rows rather than the reserved 25%), so they are context, not a
    control.
    """
    out = []
    for label, name in REFERENCE.get(concept, []):
        path = C.REPO / "ceiling_analysis/data/reference" / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        means = df[df.dataset == "mean"]
        if means.empty:
            means = df.groupby("round", as_index=False)["auroc"].mean()
        for _, row in means.iterrows():
            out.append((label, str(row["round"]), float(row["auroc"])))
    return out


def load_sweep(concept: str) -> pd.DataFrame:
    path = C.RESULTS / f"sweep_{concept}.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        flat = {k: v for k, v in r.items() if k not in ("mean", "per_split")}
        flat.update({f"mean_{k}": v for k, v in r["mean"].items()})
        for split, m in r["per_split"].items():
            flat[f"auroc_{split}"] = m["auroc"]
        rows.append(flat)
    df = pd.DataFrame(rows)
    # a rerun appends; keep the last row written for each key
    return df.drop_duplicates(subset=["arm", "dev_seed", "n_dev"], keep="last")


def curve(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    sub = df[df.arm == arm]
    if sub.empty:
        return sub
    g = sub.groupby("n_dev")["mean_auroc"]
    out = pd.DataFrame({"n_dev": g.mean().index, "auroc_mean": g.mean().to_numpy(),
                        "auroc_std": g.std(ddof=0).fillna(0.0).to_numpy(),
                        "n_seeds": g.count().to_numpy()})
    return out.sort_values("n_dev").reset_index(drop=True)


def first_sustained(xs: np.ndarray, ys: np.ndarray, target: float) -> int | None:
    """Smallest x whose y is >= target and which no later point falls back below."""
    ok = ys >= target
    for i in range(len(xs)):
        if ok[i] and ok[i:].all():
            return int(xs[i])
    return None


def summarize(concept: str, ceiling: float, df: pd.DataFrame, tol: float,
              gap_fracs=(0.9, 0.95)) -> dict:
    base = df[(df.arm == "mixed") & (df.n_dev == 0)]["mean_auroc"]
    baseline = float(base.mean()) if len(base) else float("nan")
    out = {"concept": concept, "ceiling": ceiling, "redteam_only": baseline,
           "gap": ceiling - baseline, "arms": {}}
    for arm in ("mixed", "finetune", "dev_only"):
        c = curve(df, arm)
        if c.empty:
            continue
        xs, ys = c.n_dev.to_numpy(), c.auroc_mean.to_numpy()
        entry = {
            "best_auroc": float(ys.max()),
            "best_n_dev": int(xs[int(np.argmax(ys))]),
            "final_auroc": float(ys[-1]),
            "n_within_tol": first_sustained(xs, ys, ceiling - tol),
        }
        for f in gap_fracs:
            entry[f"n_at_gap_{int(f*100)}"] = first_sustained(
                xs, ys, baseline + f * (ceiling - baseline)
            )
        out["arms"][arm] = entry
    return out


def plot(concept: str, ceiling: float, df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm in ("mixed", "finetune", "dev_only"):
        c = curve(df, arm)
        if c.empty:
            continue
        ax.plot(c.n_dev, c.auroc_mean, marker="o", ms=4, color=ARM_COLOR[arm],
                label=ARM_LABEL[arm])
        ax.fill_between(c.n_dev, c.auroc_mean - c.auroc_std, c.auroc_mean + c.auroc_std,
                        color=ARM_COLOR[arm], alpha=0.15, linewidth=0)
    ax.axhline(ceiling, color="#111111", ls="--", lw=1.2,
               label=f"ceiling (eval-trained CV) = {ceiling:.4f}")
    ax.set_xlabel("dev samples added to the training data (N)")
    ax.set_ylabel("mean eval AUROC over splits")
    ax.set_title(f"{concept}: closing the gap to the ceiling")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=list(C.CONCEPTS))
    ap.add_argument("--tol", type=float, default=0.01,
                    help="absolute AUROC tolerance for 'reached the ceiling'")
    args = ap.parse_args()

    sections, summaries = [], []
    for concept in args.concepts:
        cpath = C.RESULTS / f"ceiling_{concept}.json"
        spath = C.RESULTS / f"sweep_{concept}.jsonl"
        if not (cpath.exists() and spath.exists()):
            print(f"skipping {concept}: missing results", flush=True)
            continue
        ceil = json.loads(cpath.read_text())
        # The ceiling is the BEST rung, not the last one. The dev-pool rung adds training
        # data but is not guaranteed to score higher — on high-stakes it lands 0.001 below
        # the eval-only rung, which is the saturation signal, not a worse ceiling.
        rung_scores = {k: float(v["mean"]["auroc"]) for k, v in ceil["by_train_size"].items()}
        best_rung = max(rung_scores, key=rung_scores.get)
        ceiling = rung_scores[best_rung]
        df = load_sweep(concept)
        df.to_csv(C.RESULTS / f"sweep_{concept}.csv", index=False)
        for arm in ("mixed", "finetune", "dev_only"):
            c = curve(df, arm)
            if not c.empty:
                c.to_csv(C.RESULTS / f"curve_{concept}_{arm}.csv", index=False)
        plot(concept, ceiling, df, C.RESULTS / f"curve_{concept}.png")
        s = summarize(concept, ceiling, df, args.tol)
        summaries.append(s)

        lines = [f"### {concept}", ""]
        lines.append(f"* ceiling (eval-trained {ceil['n_folds']}-fold CV, best rung "
                     f"`{best_rung}`): **{ceiling:.4f}** mean eval AUROC")
        lines.append(f"* red-team only (N=0): **{s['redteam_only']:.4f}** "
                     f"— gap {s['gap']:+.4f}")
        rungs = list(ceil["by_train_size"].items())
        for size, entry in rungs:
            lines.append(f"* ceiling CV at {size} training rows/fold: "
                         f"{entry['mean']['auroc']:.4f}")
        if len(rungs) >= 2:
            ordered = sorted(rung_scores.items(), key=lambda kv: kv[1])
            climb = ordered[-1][1] - ordered[-2][1]
            if climb > 0.005:
                lines.append(
                    f"* **the ladder is still climbing** (+{climb:.4f} on the top step), so "
                    f"{ceiling:.4f} is a *lower bound* on the ceiling, not a plateau — the "
                    f"eval set simply has no more in-distribution rows to train on"
                )
            else:
                lines.append(
                    f"* the top two rungs agree to {abs(climb):.4f}, so the estimate is "
                    f"saturated rather than training-size limited"
                )
        lines.append("")
        lines.append("| arm | best AUROC | at N | N to close 90% of gap | 95% | "
                     f"N within {args.tol} of ceiling |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for arm, e in s["arms"].items():
            lines.append(
                f"| {ARM_LABEL[arm]} | {e['best_auroc']:.4f} | {e['best_n_dev']} | "
                f"{e['n_at_gap_90']} | {e['n_at_gap_95']} | {e['n_within_tol']} |"
            )
        dead = df[df.n_train < MIN_TRAIN_FOR_A_STEP][["arm", "n_dev", "n_train"]]
        if not dead.empty:
            pts = sorted({(r.arm, int(r.n_dev), int(r.n_train)) for r in dead.itertuples()})
            lines.append("")
            lines.append(
                f"> **Points that never take an optimizer step.** With the default "
                f"`batch_size` 16 and `gradient_accumulation_steps` 4, a training set below "
                f"{MIN_TRAIN_FOR_A_STEP} samples yields fewer batches per epoch than the "
                f"accumulation period, so `optimizer.step()` is never called and the head is "
                f"returned at its random initialisation. This is tuberlens' own loop, not an "
                f"artifact of this analysis, and it applies to: "
                + ", ".join(f"`{a}` at N={n} ({t} train rows)" for a, n, t in pts)
                + ". Read those points as 'no training happened', not as 'the data did not "
                  "help'. The `mixed` arm is unaffected — it always carries the red-team set."
            )
        refs = reference_rows(concept)
        if refs:
            lines.append("")
            lines.append("For context, the probes those experiment runs actually produced "
                         "(their own comparison CSVs, mean eval AUROC per retrain round):")
            lines.append("")
            lines.append("| run | round | mean eval AUROC |")
            lines.append("| --- | --- | --- |")
            for label, rnd, auroc in refs:
                lines.append(f"| {label} | {rnd} | {auroc:.4f} |")
        lines.append("")
        lines.append(f"![{concept}](curve_{concept}.png)")
        lines.append("")
        sections.append("\n".join(lines))

    (C.RESULTS / "summary.json").write_text(json.dumps(summaries, indent=2))
    (C.RESULTS / "SUMMARY.md").write_text(
        "# Ceiling analysis — results\n\n" + "\n".join(sections), encoding="utf-8"
    )
    print("wrote", C.RESULTS / "SUMMARY.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
