#!/usr/bin/env python
"""Regenerate reports/llama8b_concept_probes/ from whatever the run has produced.

Safe to call repeatedly while the run is still going: concepts without an eval CSV are
reported as pending rather than blocking the report. The published outputs are small (a
markdown report plus one CSV per concept); the activations, fit caches and probe pickles
stay untracked under activations/ and results_llama8b/.

The findings section is read from reports/llama8b_concept_probes/FINDINGS.md if that
file exists, so the narrative can be written once and survive a regeneration.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results_llama8b"
OUT = REPO_ROOT / "reports" / "llama8b_concept_probes"
CONCEPTS = ["hu_ha", "instructions", "highstakes"]
N_PROBES = 4


def _table(csv_path: Path) -> str:
    import pandas as pd

    df = pd.read_csv(csv_path)
    piv = df[df.dataset != "mean"].pivot_table(
        index=["config", "val_mode"], columns="dataset", values="auroc"
    )
    piv["MEAN"] = piv.mean(axis=1)
    return piv.round(3).to_string()


def _summary_table() -> list[str]:
    """Mean AUROC per concept x arm, across whatever concepts have finished."""
    import pandas as pd

    rows = {}
    for c in CONCEPTS:
        csv = RESULTS / c / "eval_results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df = df[df.dataset != "mean"]
        m = df.groupby(["config", "val_mode"])["auroc"].mean()
        rows[c] = m
    if not rows:
        return ["_No eval results yet._", ""]
    arms = sorted({k for m in rows.values() for k in m.index})
    out = ["| arm | " + " | ".join(rows) + " |",
           "| --- | " + " | ".join("---" for _ in rows) + " |"]
    for arm in arms:
        cells = [f"{rows[c][arm]:.3f}" if arm in rows[c].index else "—" for c in rows]
        out.append(f"| `{arm[0]}` / val=`{arm[1]}` | " + " | ".join(cells) + " |")
    return out + [""]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Concept probes trained on the llama-3.1-8b synthetic cuts",
        "",
        f"_Generated {now}._",
        "",
        "## What is being measured",
        "",
        "Each probe is trained on its own concept's `data/<concept>_llama8b.jsonl` — ~50",
        "balanced two-turn conversations written by llama-3.1-8b to exhibit that concept's",
        "positive and negative class — and scored on that concept's real eval splits at full",
        "size. So this is a transfer question: how much of a concept does a 50-row synthetic",
        "cut teach a probe? Model `google/gemma-3-27b-it`, layer 32, arch",
        "`linear_then_softmax`, seed 42, activations precomputed.",
        "",
        "| axis | values |",
        "| --- | --- |",
        "| probe config | `single` · `seq_ens10` (10 members, `PROBE_FUSED_ENSEMBLE=0`, fit sequentially under the repo-pinned `ENSEMBLE_SEEDS`) |",
        "| validation | `dev` (the concept's `dev_samples/` set; all ~50 rows train) · `split` (0.2 of the training data via `stable_train_test_split`, leaving ~40 to train) |",
        "",
        "| concept | training file | rows | labels |",
        "| --- | --- | --- | --- |",
        "| `hu_ha` | `data/hu_harm_llama8b.jsonl` | 49 (24/25) | `harmful_to_human` / `not_harmful_to_human` |",
        "| `highstakes` | `data/highstakes_llama8b.jsonl` | 50 (25/25) | `high-stakes` / `low-stakes` |",
        "| `instructions` | `data/instructions_llama8b.jsonl` | 50 (25/25) | `assistant_follows_the_instruction` / `assistant_does_not_follow_the_instruction` |",
        "",
        "4 probes per concept, 12 in total.",
        "",
        "## Mean AUROC across eval splits",
        "",
    ]
    lines += _summary_table()

    for c in CONCEPTS:
        csv = RESULTS / c / "eval_results.csv"
        probes = RESULTS / c / "probes"
        n = len(list(probes.glob("*.pkl"))) if probes.is_dir() else 0
        if csv.exists():
            shutil.copy2(csv, OUT / f"{c}_eval_results.csv")
            lines += [f"## {c} — AUROC per eval split", "", "```", _table(csv), "```", ""]
        else:
            lines += [f"## {c}", "", f"_In progress — {n}/{N_PROBES} probes fitted, no eval yet._", ""]

    findings = OUT / "FINDINGS.md"
    if findings.exists():
        lines += [findings.read_text().rstrip(), ""]

    lines += [
        "## Reproducing",
        "",
        "```bash",
        ".venv_claude/bin/python scripts/llama8b_concept_probes.py --phase all",
        "```",
        "",
        "`prepare` extracts all 149 conversations once into a master blob and slices each",
        "(concept, validation mode) train/val activation cache out of it, so the six",
        "combinations do not each trigger their own gemma-3-27b load. The dev blobs and the",
        "activation cache dir are shared with `results_generalization/` — the base blobs are",
        "keyed on the training file's own hash and the dev blobs on the dev files' bytes, so",
        "neither can collide. After `prepare`, no fit loads a model.",
    ]

    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
