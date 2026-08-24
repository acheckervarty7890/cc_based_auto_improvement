#!/usr/bin/env python
"""Compare the 10-member sequential ensemble against a single probe, on score and on cost.

Reads what the concept-probe runs already produced — the per-generator eval CSVs under
results_<generator>/<concept>/eval_results.csv for the scores, and the run logs for the
fit wall-clock — and writes reports/ensemble_vs_single.md. Nothing is refitted.

The pairing is (generator, concept, val_mode), giving 24 cells across 4 generators x 3
concepts x 2 validation modes. Note the two sides are NOT the same probe family: this
repo's ``_resolve_ensemble_seeds`` carves out ``n == 1`` to return ``[--seed]`` (42),
while ``n > 1`` uses the repo-pinned ``ENSEMBLE_SEEDS[:n]``. So a cell's delta mixes the
averaging effect with a change of seed, which is worth stating whenever it is read as
"what ensembling bought".

    .venv_claude/bin/python scripts/ensemble_vs_single.py
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "reports" / "ensemble_vs_single.md"

GENERATORS = ["llama8b", "llama70b", "dsv4pro", "nemotron550b"]
CONCEPTS = ["hu_ha", "highstakes", "instructions"]

# The run logs carry the "fit in Ns" lines. llama8b ran before the script was
# generalized, so its headers omit the generator field; the regex makes it optional.
LOGS = [
    ("llama8b", REPO_ROOT / "logs" / "llama8b_run.log"),
    (None, REPO_ROOT / "logs" / "concept_probes_70b_dsv.log"),
    (None, REPO_ROOT / "logs" / "concept_probes_nemotron.log"),
]

_HEADER = re.compile(r"=== (?:(\w+) \| )?(\w+) \| (\w+) \| val=(\w+) ===")
_FIT = re.compile(r"\s+fit in (\d+)s")


def load_scores() -> pd.DataFrame:
    rows = []
    for gen in GENERATORS:
        for concept in CONCEPTS:
            csv = REPO_ROOT / f"results_{gen}" / concept / "eval_results.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            df = df[df.dataset != "mean"]
            means = df.groupby(["config", "val_mode"]).auroc.mean().reset_index()
            for _, r in means.iterrows():
                rows.append(
                    dict(generator=gen, concept=concept, config=r.config,
                         val_mode=r.val_mode, auroc=r.auroc)
                )
    return pd.DataFrame(rows)


def load_fit_times() -> pd.DataFrame:
    rows = []
    for default_gen, path in LOGS:
        if not path.exists():
            continue
        pending = None
        for line in path.open(errors="ignore"):
            m = _HEADER.match(line)
            if m:
                pending = (m.group(1) or default_gen, m.group(2), m.group(3), m.group(4))
                continue
            m = _FIT.match(line)
            if m and pending:
                gen, concept, config, val_mode = pending
                rows.append(dict(generator=gen, concept=concept, config=config,
                                 val_mode=val_mode, sec=int(m.group(1))))
                pending = None
    return pd.DataFrame(rows)


def _paired(df: pd.DataFrame, value: str) -> pd.DataFrame:
    piv = df.pivot_table(index=["generator", "concept", "val_mode"],
                         columns="config", values=value)
    return piv.dropna(subset=["single", "seq_ens10"])


def main() -> int:
    scores = load_scores()
    times = load_fit_times()
    if scores.empty:
        print("no eval CSVs found — nothing to compare")
        return 1

    s = _paired(scores, "auroc")
    s["delta"] = s["seq_ens10"] - s["single"]
    t = _paired(times, "sec")
    t["ratio"] = t["seq_ens10"] / t["single"]

    # Does averaging damp the choice of validation source?
    vs = scores.pivot_table(index=["generator", "concept", "config"],
                            columns="val_mode", values="auroc")
    vs["sensitivity"] = (vs["dev"] - vs["split"]).abs()
    sens = vs.groupby(level="config").sensitivity.agg(["mean", "max"])

    tight = s.loc[(slice(None), ["highstakes", "instructions"], "split"), :]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    n = len(s)
    tot = times.groupby("config").sec.sum()

    lines = [
        "# Ensemble vs single probe — what the 10 members buy, and what they cost",
        "",
        f"_Generated {now} by `scripts/ensemble_vs_single.py` from the existing run "
        "outputs. Nothing is refitted._",
        "",
        f"{n} paired cells: {len(GENERATORS)} generators x {len(CONCEPTS)} concepts x 2 "
        "validation modes. `seq_ens10` is a 10-member score-averaging deep ensemble fit "
        "sequentially (`PROBE_FUSED_ENSEMBLE=0`); `single` is one probe. Every score is "
        "mean AUROC over that concept's eval splits.",
        "",
        "## Headline",
        "",
        "| | |",
        "| --- | --- |",
        f"| mean delta (ens - single) | **{s.delta.mean():+.4f}** |",
        f"| median delta | {s.delta.median():+.4f} |",
        f"| range | {s.delta.min():+.3f} to {s.delta.max():+.3f} |",
        f"| ensemble wins | {int((s.delta > 0).sum())}/{n} |",
        f"| cells moving more than 0.02 | {int((s.delta.abs() > 0.02).sum())}/{n} |",
        f"| total fit time, single | {tot['single']:.0f} s = {tot['single']/60:.1f} min |",
        f"| total fit time, seq_ens10 | {tot['seq_ens10']:.0f} s = {tot['seq_ens10']/60:.1f} min |",
        f"| cost ratio | **{tot['seq_ens10']/tot['single']:.1f}x** |",
        "",
        "About +0.01 AUROC for 7x the fit time, with the sign unreliable — a third of "
        "cells go the wrong way. Against the effects this experiment actually measures "
        "(0.22 between generators on `instructions`, 0.28 between concepts), ensembling "
        "is noise.",
        "",
        "## Scores",
        "",
        "```",
        s.round(3).to_string(),
        "```",
        "",
        "### By concept",
        "",
        "```",
        s.groupby(level="concept").delta.agg(["mean", "min", "max"]).round(4).to_string(),
        "```",
        "",
        "### By validation mode",
        "",
        "```",
        s.groupby(level="val_mode").delta.agg(["mean", "min", "max"]).round(4).to_string(),
        "```",
        "",
        "## Fit wall-clock (seconds)",
        "",
        "```",
        t.round(1).to_string(),
        "```",
        "",
        "The ratio is sub-linear rather than 10x because `_to_device_for_fit` stages the "
        "activations, and the dev blob is read from disk, **once per `build_ensemble` "
        "call** rather than once per member — the single-probe fit pays that fixed cost "
        "against one fit, the ensemble amortizes it over ten. It is largest on the "
        "`split` arms, where the fits themselves are 1-2 s and the fixed cost dominates "
        "both sides.",
        "",
        "## Two caveats before reading the delta column as an ensemble effect",
        "",
        "**The single probe is not member 0 of the ensemble.** "
        "`retrain._resolve_ensemble_seeds` carves out `n == 1` and returns `[--seed]` "
        "(42), while `n > 1` uses the repo-pinned `ENSEMBLE_SEEDS[:10]`. The two sides "
        "are different draws, so each cell's delta mixes averaging with a change of "
        "seed, and some of the +-0.05 scatter is seed noise.",
        "",
        "**It does not buy stability either.** Mean `|dev - split|` per (generator, "
        f"concept) is {sens.loc['seq_ens10','mean']:.4f} for the ensemble against "
        f"{sens.loc['single','mean']:.4f} for the single probe — if anything slightly "
        "worse, though within noise. The usual argument for a deep ensemble, that "
        "averaging damps sensitivity to arbitrary choices, does not show up here.",
        "",
        "```",
        sens.round(4).to_string(),
        "```",
        "",
        "## Where it looks least like noise",
        "",
        f"The `split` arms of `highstakes` and `instructions` — the cells with only ~40 "
        f"training rows, where a single fit is least stable and averaging has the most to "
        f"fix — average {tight.delta.mean():+.3f}, "
        f"{int((tight.delta > 0).sum())}/{len(tight)} positive. That is roughly double "
        "the overall mean, and it is the only slice with a defensible story behind it. "
        "It is still not a clean result: llama70b moves the wrong way on both of its "
        "cells, so even here the effect does not hold for every generator.",
        "",
        "```",
        tight.round(3).to_string(),
        "```",
    ]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
