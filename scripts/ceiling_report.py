#!/usr/bin/env python
"""Regenerate reports/cross_concept_ceiling/ from results_ceiling/.

Safe to call while the run is in flight — arms not yet finished simply do not appear.
The narrative, if any, is read from reports/cross_concept_ceiling/FINDINGS.md.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV = REPO_ROOT / "results_ceiling" / "cross_concept_ceiling.csv"
OUT = REPO_ROOT / "reports" / "cross_concept_ceiling"


def main() -> int:
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Cross-concept ceiling",
        "",
        f"_Generated {now}._",
        "",
        "## What is being measured",
        "",
        "**Ceiling** is the `ceiling_analysis` branch's definition: the best eval-set",
        "performance this probe family (`linear_then_softmax` head on",
        "`google/gemma-3-27b-it` layer 32) can reach *when trained on eval-distribution",
        "data*, estimated by 5-fold cross-validation **inside the eval sets themselves** —",
        "fit on the rows outside fold k, early-stop against a fixed reserved dev slice,",
        "score fold k. Every eval row gets exactly one out-of-fold score, so a ceiling probe",
        "differs from an ordinary probe only in its training data.",
        "",
        "This run asks that of the three concepts **pooled**:",
        "",
        "| arm | training pool |",
        "| --- | --- |",
        "| `within/<concept>` | CV inside one concept's own eval splits — the per-concept ceiling, and the baseline the cross numbers must be read against |",
        "| `cross/native` | all three concepts at once, each row positive iff it is its own concept's positive class |",
        "| `cross/aligned` | the same pool with `instructions` **inverted**, so all three positives are the *problematic* response |",
        "",
        "`hu_ha` and `highstakes` call the problematic response positive (harmful,",
        "high-stakes); `instructions` calls the desirable one positive",
        "(`assistant_follows_the_instruction`). Pooling as-is therefore asks one direction to",
        "put \"harmful\" and \"correctly followed the instruction\" on the same side, which is",
        "not obviously the intended question — hence both orientations. The flip is not a",
        "symmetry of the pooled task (it re-labels a third of the rows), so `aligned` is not",
        "`1 - native`.",
        "",
        "Each eval split is balanced-subsampled to 100 rows (this repo's own",
        "`--eval-max-samples` default), giving 1500 pooled rows; the full 6576-row pool",
        "padded to the longest split's 1024 tokens would be 72 GB of fp16 activations and",
        "fits neither the box nor the card. Validation is a fixed 300-row balanced slice of",
        "`dev_samples/`, 100 per concept, identical for every fold and every arm. Single",
        "probes, never ensembles, seed 42. No model is loaded — every activation comes from",
        "the precomputed blobs.",
        "",
    ]

    if not CSV.exists():
        lines += ["_No results yet._", ""]
    else:
        shutil.copy2(CSV, OUT / "cross_concept_ceiling.csv")
        df = pd.read_csv(CSV)
        for scope, title in (("pooled", "Pooled AUROC (all rows of the arm's own pool)"),
                             ("concept", "AUROC by concept"),
                             ("split", "AUROC by eval split")):
            sub = df[df.scope == scope]
            if sub.empty:
                continue
            lines += [f"## {title}", "", "```",
                      sub.pivot_table(index="name", columns="arm",
                                      values="auroc").round(3).to_string(),
                      "```", ""]
        acc = df[df.scope == "concept"]
        if not acc.empty:
            lines += ["## Accuracy and TPR@1%FPR by concept", "", "```",
                      acc.pivot_table(index="name", columns="arm",
                                      values=["accuracy", "tpr_at_fpr"]).round(3).to_string(),
                      "```", ""]

    findings = OUT / "FINDINGS.md"
    if findings.exists():
        lines += [findings.read_text().rstrip(), ""]

    lines += [
        "## Reproducing",
        "",
        "```bash",
        ".venv_claude/bin/python scripts/cross_concept_ceiling.py",
        "```",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
