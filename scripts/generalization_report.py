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
        "## Findings so far",
        "",
        "**A trainer bug had to be fixed before any of this was readable.** `linear_then_softmax`",
        "defaults to `batch_size: 16, gradient_accumulation_steps: 4`, and the trainer steps only",
        "on `(batch_idx + 1) % accumulation == 0` with no end-of-epoch flush",
        "(`pytorch_classifiers.py:299-327`). The 50-row variants under a 0.2 split leave 39",
        "training rows = 3 batches, so `optimizer.step()` never fired and the probe was returned",
        "at its initialization — two variants with different data *and* different labels produced",
        "a bitwise-identical probe (`be87fde7dde9`), `best_epoch=1`, loss pinned at ln 2,",
        "validation AUROC exactly 0.5. Fixed by capping accumulation at the batch count, which is",
        "a no-op for every arm that already had ≥4 batches (verified bit-identical) and changes",
        "only the degenerate cells.",
        "",
        "**The random-label controls transfer as well as, or better than, the real signal.** On",
        "hu_ha, variants i and ii average 0.58–0.62 mean AUROC while `iii_general_pos` — the",
        "actual general/not_general distinction — averages 0.54–0.56. A probe fit to 50",
        "conversations with meaningless labels transfers better than one fit to the real one.",
        "",
        "**One split carries almost all of the signal, and it is the refusal split in both",
        "concepts.** hu_ha's `eval_balanced_refusal` runs 0.74–0.88 while its three siblings sit",
        "at 0.46–0.61; instructions' `anthropic_harmless_refusal` runs 0.70–0.87 while its six",
        "siblings sit near 0.5. General-vs-specialized language plausibly correlates with",
        "refusal-shaped language; nothing else here is far from chance.",
        "",
        "**Label inversion is not a clean mirror.** `iii_general_neg` sits at ~0.49 throughout",
        "while `iii_general_pos` is ~0.55. If inversion were exact these would reflect around",
        "0.5, so the fits are not symmetric under a label flip.",
        "",
        "**The 10-member sequential ensemble barely moves anything** (≤0.02 mean vs the single",
        "probe), which is the expected result when members over the same activations agree.",
        "",
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
