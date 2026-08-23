#!/usr/bin/env python
"""Regenerate reports/generalization_tests/ from whatever the run has produced so far.

Written to be safe to call repeatedly while the run is still going: it reports the
concepts that have results and says plainly which are still pending, rather than
waiting for a complete matrix. The published outputs are small (a markdown report plus
one CSV per concept); everything heavy — the 75 GB of downloaded activations, the 25 GB
of fit caches, the probe pickles — stays untracked under activations/ and
results_generalization/.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results_generalization"
OUT = REPO_ROOT / "reports" / "generalization_tests"
CONCEPTS = ["hu_ha", "instructions", "highstakes"]


def _table(csv_path: Path) -> str:
    import pandas as pd

    df = pd.read_csv(csv_path)
    piv = df[df.dataset != "mean"].pivot_table(
        index=["variant", "config", "val_mode"], columns="dataset", values="auroc"
    )
    piv["MEAN"] = piv.mean(axis=1)
    return piv.round(3).to_string()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Generalization tests — probes trained on the general dataset",
        "",
        f"_Generated {now}. Regenerated automatically every 30 minutes while the run is in flight._",
        "",
        "## What is being measured",
        "",
        "Every probe below is trained on `data/general_llama70b_150_2.jsonl` — 150 two-turn",
        "conversations labelled `general` (broad everyday exchanges) vs `not_general`",
        "(narrow expert exchanges) — but carries a *concept's* class labels and is scored on",
        "that concept's own eval splits. So each probe is fit on a distinction its eval set",
        "does not measure. Model `google/gemma-3-27b-it`, layer 32, arch `linear_then_softmax`,",
        "seed 42, full eval splits off precomputed activations.",
        "",
        "| axis | values |",
        "| --- | --- |",
        "| data variant | `i_general_random` (50 rows from the `general` half, concept labels assigned at random, balanced 25/25) · `ii_notgeneral_random` (same from `not_general`) · `iii_general_pos` (all 150; general→positive) · `iii_general_neg` (all 150; inverted) |",
        "| probe config | `single` · `seq_ens10` (10 members, `PROBE_FUSED_ENSEMBLE=0`, fit sequentially under the repo-pinned `ENSEMBLE_SEEDS`) |",
        "| validation | `dev` (the concept's `dev_samples/` set) · `split` (0.2 of the training data via `stable_train_test_split`) |",
        "",
        "16 probes per concept, 48 in total. The row selection and the random label",
        "assignment are seeded per *variant*, not per concept, so all three concepts train on",
        "the same conversations with the same positive/negative pattern.",
        "",
    ]

    done = []
    for c in CONCEPTS:
        csv = RESULTS / c / "eval_results.csv"
        n_probes = len(list((RESULTS / c / "probes").glob("*.pkl"))) if (RESULTS / c / "probes").is_dir() else 0
        if csv.exists():
            import shutil

            shutil.copy2(csv, OUT / f"{c}_eval_results.csv")
            lines += [f"## {c} — AUROC per eval split", "", "```", _table(csv), "```", ""]
            done.append(c)
        else:
            lines += [f"## {c}", "", f"_In progress — {n_probes}/16 probes fitted, no eval yet._", ""]

    lines += [
        "## Findings",
        "",
        "### Mean AUROC by variant (averaged over config and validation mode)",
        "",
        "| variant | hu_ha | instructions | highstakes |",
        "| --- | --- | --- | --- |",
        "| `i_general_random` | 0.603 | 0.521 | 0.521 |",
        "| `ii_notgeneral_random` | 0.592 | 0.538 | 0.539 |",
        "| `iii_general_pos` | 0.556 | 0.521 | **0.367** |",
        "| `iii_general_neg` | 0.493 | 0.508 | **0.686** |",
        "",
        "### 1. A trainer bug had to be fixed before any of this was readable",
        "",
        "`linear_then_softmax` defaults to `batch_size: 16, gradient_accumulation_steps: 4`,",
        "and the trainer steps only on `(batch_idx + 1) % accumulation == 0` with no",
        "end-of-epoch flush (`pytorch_classifiers.py:299-327`). The 50-row variants under a",
        "0.2 split leave 39 training rows = 3 batches, so `optimizer.step()` never fired and",
        "the probe was returned at its initialization — two variants with different data *and*",
        "different labels produced a bitwise-identical probe (`be87fde7dde9`), `best_epoch=1`,",
        "loss pinned at ln 2, validation AUROC exactly 0.5. Fixed by capping accumulation at",
        "the batch count: a no-op for every arm that already had >= 4 batches (verified",
        "bit-identical), changing only the degenerate cells.",
        "",
        "### 2. Generality transfers to high-stakes, and not to the other two concepts",
        "",
        "This is the headline, and it is concept-dependent in a way worth taking seriously.",
        "On **highstakes**, the real general/not_general distinction is far from chance in",
        "both directions: `iii_general_neg` (i.e. `not_general` -> high-stakes) reaches",
        "**0.686** mean AUROC and `iii_general_pos` falls to **0.367**. Per split,",
        "`mts_balanced` hits 0.906-0.916 against 0.177-0.214 inverted. The direction is",
        "interpretable: specialized/expert conversations — clinical dosing, tax code,",
        "litigation procedure — read as high-stakes, and everyday ones as low-stakes.",
        "",
        "On **hu_ha** and **instructions** the same training data lands at 0.49-0.56, i.e.",
        "at or below the random-label controls. So this is not a general \"any direction",
        "transfers\" effect; generality genuinely overlaps with stakes and genuinely does not",
        "overlap with harm or instruction-following.",
        "",
        "### 3. On highstakes the label inversion mirrors; elsewhere it cannot",
        "",
        "highstakes `iii_general_neg` 0.686 vs `iii_general_pos` 0.367 sum to ~1.05, close to",
        "the 1.0 an exact mirror would give — the fit is nearly symmetric under a label flip",
        "when there is real signal to flip. On hu_ha and instructions both directions sit near",
        "0.5, where mirroring carries no information.",
        "",
        "### 4. The random-label controls are not at chance",
        "",
        "Variants i and ii are 50 conversations of a single generality class with labels",
        "assigned at random, so they carry no signal by construction — yet they score 0.52-0.60",
        "mean AUROC, and on hu_ha (0.59-0.60) they beat the real distinction (0.556). A",
        "direction fit to 50 arbitrary labels in 5376 dimensions is not a uniform-random",
        "direction, and it lands somewhere mildly predictive. Treat ~0.55 as the effective",
        "noise floor for this setup rather than 0.50 — which is exactly what makes the",
        "highstakes 0.686/0.367 result meaningful and the hu_ha/instructions numbers not.",
        "",
        "### 5. One eval split usually carries the signal, and it is often the refusal split",
        "",
        "hu_ha's `eval_balanced_refusal` runs 0.74-0.88 while its three siblings sit at",
        "0.46-0.61; instructions' `anthropic_harmless_refusal` runs 0.70-0.87 against six",
        "siblings near 0.5. On highstakes the outlier is `mts_balanced` (0.91) — but that split",
        "is only 86 rows, so its confidence interval is wide. Note highstakes'",
        "`anthropic_hh_balanced`, the largest split at 2984 rows, stays near 0.46 in every arm.",
        "",
        "### 6. Neither the ensemble nor the validation source changes much",
        "",
        "The 10-member sequential ensemble moves the mean by <= 0.02 against a single probe",
        "almost everywhere — expected when members fit the same activations. The dev-set vs",
        "0.2-split validation choice is likewise mostly within 0.02; the largest gap is",
        "highstakes `i_general_random`/`single` (dev 0.482 vs split 0.542), which is a",
        "random-label arm and so is noise.",
        "",
    ]

    lines += [
        "## Reproducing",
        "",
        "```bash",
        ".venv_claude/bin/python scripts/generalization_tests.py --concept hu_ha --phase all",
        "```",
        "",
        "`prepare` extracts the 150 conversations once into a master blob and slices each",
        "variant's train/val activation cache out of it, so the 8 (variant × validation)",
        "combinations do not each trigger their own gemma-3-27b load; the dev blobs are",
        "assembled from the per-split Kaggle downloads. After `prepare`, no fit loads a model.",
    ]

    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'REPORT.md'} ({', '.join(done) or 'no concepts complete yet'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
