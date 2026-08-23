#!/usr/bin/env python
"""Regenerate reports/<generator>_concept_probes/ from whatever the runs have produced.

Safe to call repeatedly while a run is still going: a concept without an eval CSV is
reported as pending rather than blocking the report. The published outputs are small (a
markdown report plus one CSV per concept); the activations, fit caches and probe pickles
stay untracked under activations/ and results_<generator>/.

A per-generator findings narrative is read from
reports/<generator>_concept_probes/FINDINGS.md when that file exists, so it survives a
regeneration. A cross-generator comparison is written to reports/concept_probes_summary.md.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONCEPTS = ["hu_ha", "instructions", "highstakes"]
N_PROBES = 4

GENERATORS = {
    "llama8b": ("meta-llama Llama-3.1-8B", "data/<concept>_llama8b.jsonl"),
    "llama70b": ("meta-llama Llama-3.3-70B", "data/<concept>_llama70b_50.jsonl"),
    "dsv4pro": ("deepseek v4 pro", "data/<concept>_dsv4pro.jsonl"),
}


def results_dir(gen: str) -> Path:
    return REPO_ROOT / f"results_{gen}"


def out_dir(gen: str) -> Path:
    return REPO_ROOT / "reports" / f"{gen}_concept_probes"


def _table(csv_path: Path) -> str:
    import pandas as pd

    df = pd.read_csv(csv_path)
    piv = df[df.dataset != "mean"].pivot_table(
        index=["config", "val_mode"], columns="dataset", values="auroc"
    )
    piv["MEAN"] = piv.mean(axis=1)
    return piv.round(3).to_string()


def _arm_means(gen: str):
    """{concept: Series indexed by (config, val_mode)} of mean AUROC over splits."""
    import pandas as pd

    rows = {}
    for c in CONCEPTS:
        csv = results_dir(gen) / c / "eval_results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df = df[df.dataset != "mean"]
        rows[c] = df.groupby(["config", "val_mode"])["auroc"].mean()
    return rows


def write_generator_report(gen: str) -> bool:
    o = out_dir(gen)
    o.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    model, pattern = GENERATORS[gen]
    lines = [
        f"# Concept probes trained on the {gen} synthetic cuts",
        "",
        f"_Generated {now}._",
        "",
        "## What is being measured",
        "",
        f"Each probe is trained on its own concept's `{pattern}` — ~50 balanced two-turn",
        f"conversations written by **{model}** to exhibit that concept's positive and",
        "negative class — and scored on that concept's real eval splits at full size. So this",
        "is a transfer question: how much of a concept does a 50-row synthetic cut teach a",
        "probe? Model `google/gemma-3-27b-it`, layer 32, arch `linear_then_softmax`, seed 42,",
        "activations precomputed.",
        "",
        "| axis | values |",
        "| --- | --- |",
        "| probe config | `single` · `seq_ens10` (10 members, `PROBE_FUSED_ENSEMBLE=0`, fit sequentially under the repo-pinned `ENSEMBLE_SEEDS`) |",
        "| validation | `dev` (the concept's `dev_samples/` set; all ~50 rows train) · `split` (0.2 of the training data via `stable_train_test_split`, leaving ~40 to train) |",
        "",
        "4 probes per concept, 12 in total.",
        "",
        "## Mean AUROC across eval splits",
        "",
    ]
    means = _arm_means(gen)
    if means:
        arms = sorted({k for m in means.values() for k in m.index})
        lines.append("| arm | " + " | ".join(means) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in means) + " |")
        for arm in arms:
            cells = [f"{means[c][arm]:.3f}" if arm in means[c].index else "—" for c in means]
            lines.append(f"| `{arm[0]}` / val=`{arm[1]}` | " + " | ".join(cells) + " |")
        lines.append("")
    else:
        lines += ["_No eval results yet._", ""]

    for c in CONCEPTS:
        csv = results_dir(gen) / c / "eval_results.csv"
        probes = results_dir(gen) / c / "probes"
        n = len(list(probes.glob("*.pkl"))) if probes.is_dir() else 0
        if csv.exists():
            shutil.copy2(csv, o / f"{c}_eval_results.csv")
            lines += [f"## {c} — AUROC per eval split", "", "```", _table(csv), "```", ""]
        else:
            lines += [f"## {c}", "",
                      f"_In progress — {n}/{N_PROBES} probes fitted, no eval yet._", ""]

    findings = o / "FINDINGS.md"
    if findings.exists():
        lines += [findings.read_text().rstrip(), ""]

    lines += [
        "## Reproducing",
        "",
        "```bash",
        f".venv_claude/bin/python scripts/concept_probes.py --generator {gen} --phase all",
        "```",
        "",
        "`prepare` extracts every requested generator's conversations in ONE model load and",
        "slices each (concept, validation mode) train/val activation cache out of that master,",
        "addressing rows by conversation content. The dev blobs and the activation cache dir",
        "are shared across generators and with `results_generalization/` — base blobs are keyed",
        "on the training file's own hash and dev blobs on the dev files' bytes, so neither can",
        "collide. After `prepare`, no fit loads a model.",
    ]
    (o / "REPORT.md").write_text("\n".join(lines) + "\n")
    return bool(means)


def write_summary() -> None:
    """Cross-generator comparison: mean AUROC per (generator, concept, arm)."""
    import pandas as pd

    out = REPO_ROOT / "reports" / "concept_probes_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    frames = []
    for gen in GENERATORS:
        for c in CONCEPTS:
            csv = results_dir(gen) / c / "eval_results.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            df = df[df.dataset != "mean"].copy()
            df["generator"] = gen
            df["concept"] = c
            frames.append(df)
    lines = [
        "# Concept probes — three generators compared",
        "",
        f"_Generated {now}._",
        "",
        "The same experiment run on three ~50-row synthetic cuts per concept, one per",
        "generating model. Every cell is mean AUROC over that concept's eval splits.",
        "",
    ]
    if not frames:
        lines += ["_No results yet._", ""]
    else:
        all_df = pd.concat(frames, ignore_index=True)
        lines += ["## Mean AUROC by generator and concept (averaged over all four arms)", "",
                  "```",
                  all_df.pivot_table(index="generator", columns="concept",
                                     values="auroc").round(3).to_string(),
                  "```", "",
                  "## Mean AUROC by generator, concept and arm", "",
                  "```",
                  all_df.pivot_table(index=["generator", "config", "val_mode"],
                                     columns="concept", values="auroc").round(3).to_string(),
                  "```", ""]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main() -> int:
    gens = sys.argv[1:] or list(GENERATORS)
    for gen in gens:
        if not results_dir(gen).is_dir():
            continue
        done = write_generator_report(gen)
        print(f"wrote {out_dir(gen)/'REPORT.md'}{'' if done else ' (no eval rows yet)'}")
    write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
