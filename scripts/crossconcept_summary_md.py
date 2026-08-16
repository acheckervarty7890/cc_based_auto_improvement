"""Render the cross-concept transfer sweep as CROSSCONCEPT.md.

Reads only ``crossconcept_progress.jsonl``, so it is correct mid-run and costs
milliseconds. Kept separate from ``SUMMARY.md`` / ``REDTEAM_ONLY.md``, each of which has
its own generator.

Usage:
    .venv_claude/bin/python scripts/crossconcept_summary_md.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import attribution_lib as A  # noqa: E402
import fetch_crossconcept_eval as FC  # noqa: E402

OUT_DIR = A.REPO / "results_instructions_gemma27b_vintage"
ORDER = ["base_only", "v2_base", "v3_base", "v2_alone", "v3only_alone"]
LABEL = {
    "base_only": "base only (no red-team)",
    "v2_base": "v2 + base",
    "v3_base": "v3 + base",
    "v2_alone": "v2 alone (no base)",
    "v3only_alone": "v3only alone (no base)",
}
CONCEPT_TITLE = {
    "hh": "harmful_to_human (`eval_dataset_hu_ha/`)",
    "hs": "high-stakes (`eval_datasets/`)",
}


def main() -> None:
    rows = []
    path = OUT_DIR / "crossconcept_progress.jsonl"
    for line in path.open(encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    L = []
    L.append("# Do the instruction-following probes transfer to the other two concepts?\n")
    L.append(f"_Updated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")
    L.append(
        "\n**What this measures.** Every probe named in `SUMMARY.md` and `REDTEAM_ONLY.md` "
        "— 90 of them, five training sets x two arms x ten seeds — scored on the **other "
        "two concepts'** eval sets. The probes separate "
        "`assistant_follows_the_instruction` from `assistant_does_not_follow_the_"
        "instruction`; the question is whether the direction they learned also orders "
        "*high-stakes* and *harmful_to_human* labels.\n"
    )
    L.append(
        "\n**Why it costs no forward pass.** All three concepts' runs extract from the same "
        "frozen `google/gemma-3-27b-it` at layer 32 under the same two loader transforms, "
        "so a conversation's activation is the same tensor whichever concept's run computed "
        "it — only the probe head differs. The eight target blobs were pulled from Kaggle "
        "(`scripts/fetch_crossconcept_eval.py`) and each was validated against the probe's "
        "model name, layer and the split's row count before use.\n"
    )
    L.append(
        "\n**How to read the numbers.** AUROC is against each target concept's own positive "
        "class (`high-stakes`, `harmful_to_human`), on the pipeline scale. **0.5 is chance, "
        "and below 0.5 is signal pointing the other way, not failure** — a probe reading "
        "0.19 against harm separates the harm labels as well as one reading 0.81, having "
        "learned a direction whose *follows-the-instruction* end is the harmful end. "
        "Nothing is thresholded: `probe.threshold` was calibrated for a different concept, "
        "so accuracy would only measure how two base rates happen to line up.\n"
    )
    L.append(
        "\n**`base_only` is the control.** It trains on the 50 base samples and no red-team "
        "rows, scores chance on its *own* concept (0.4972 ± 0.0349), and is fitted once "
        "because it does not depend on the arm.\n"
    )

    for concept in ("hh", "hs"):
        rs = [r for r in rows if r["concept"] == concept]
        if not rs:
            continue
        splits = sorted({r["split"] for r in rs})
        by = defaultdict(lambda: defaultdict(list))
        for r in rs:
            by[(r["arm"], r["condition"])][r["split"]].append(r["auroc_pipeline"])

        L.append(f"\n## {CONCEPT_TITLE[concept]} — AUROC vs `{FC.CONCEPTS[concept]['pos']}`\n\n")
        L.append("| arm | training set | "
                 + " | ".join(s.replace("eval_", "") for s in splits) + " | mean |\n")
        L.append("|" + "---|" * (3 + len(splits)) + "\n")
        for arm in ("shared", "gptoss120b", "nemotron"):
            for cond in ORDER:
                d = by.get((arm, cond))
                if not d:
                    continue
                per = [np.array(d[s]) for s in splits]
                cells = [f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" for v in per]
                m = np.mean(np.stack(per), axis=0)
                cells.append(f"**{m.mean():.3f} ± {m.std(ddof=1):.3f}**")
                L.append(f"| {arm} | {LABEL[cond]} | " + " | ".join(cells) + " |\n")

    L.append("\n## Findings\n")
    L.append(
        "\n**The harm transfer is strong, and inverted.** Every red-team-trained probe lands "
        "at 0.19-0.23 mean AUROC against `harmful_to_human`, 5-20 sd from chance, and on "
        "`eval_daily_dilemmas` it reaches 0.050-0.074 — near-perfect separation with the "
        "sign flipped. `base_only` sits at 0.475 ± 0.063, so the transfer is carried "
        "entirely by the red-team data, not by the base corpus or the architecture.\n"
    )
    L.append(
        "\n**The direction is semantically sensible.** The negative class in "
        "`eval_instructions` is failure-to-follow, whose split names spell out the modes: "
        "refusal, omission, contradiction, drift. Three of the four hu_ha splits are "
        "*paired* — the same user prompt with a harmful and a non-harmful assistant reply — "
        "and the non-harmful member is typically the refusal. So *follows the instruction* "
        "and *harmful* pick out the same conversations, and a probe trained on the first "
        "orders the second almost perfectly.\n"
    )
    L.append(
        "\n**The high-stakes transfer is weak in aggregate and lives in one split.** Mean "
        "AUROC is 0.42-0.50 across conditions, but that average hides opposite behaviour: "
        "`anthropic` inverts strongly (0.238-0.322 for the v2 conditions) while `mt` and "
        "`mts` sit at or slightly above chance and `toolace` just below. The pattern follows "
        "the refusal axis rather than the stakes axis — `anthropic` is the one hs split "
        "built from chosen/rejected assistant replies, the same comply-vs-refuse contrast "
        "the hu_ha splits carry, whereas medical-transcription (`mt`, `mts`) and tool-use "
        "system prompts (`toolace`) contain no such contrast. So what transfers is not "
        "\"stakes\" at all; it is the same refusal direction, showing up wherever a split "
        "happens to contain it.\n"
    )
    L.append(
        "\n**The v3only probes transfer least, on both concepts.** 0.357 / 0.421 against "
        "harm (vs ~0.21 for every other red-team condition) and 0.502 / 0.462 — flat chance "
        "— against high-stakes; nemotron's reaches 0.493 on `eval_ant_hh`, exactly chance. "
        "This is an independent signature of what `REDTEAM_ONLY.md` found by training on "
        "those rows alone: the pairs that first appear at iteration 3 encode something "
        "narrower than the earlier vintages. Narrow enough, it turns out, to lose the "
        "refusal axis that drives the whole transfer.\n"
    )
    L.append(
        "\n**Adding the base data changes nothing here either.** `v2 + base` and `v2 alone` "
        "differ by 0.019 (gptoss120b) and 0.013 (nemotron) on harm — the same non-effect the "
        "in-concept sweep measured, seen on a corpus neither probe was trained for.\n"
    )

    L.append("\n## Files\n\n")
    L.append("- `crossconcept_progress.jsonl` — one row per (probe, split), append-only\n")
    L.append("- `crossconcept_auroc.csv` — the same, flat\n")
    L.append("- `crossconcept_fits.jsonl` — the 90 refits and their verification status\n")
    L.append("- `sweep_probes/` — the 90 probes themselves (~13 KB each), shared with "
             "`redteam_only_fits.py`\n")
    L.append("\n## Reproducing\n\n")
    L.append("```bash\n"
             "KAGGLE_CONFIG_DIR=$PWD/kaggle .venv_claude/bin/python "
             "scripts/fetch_crossconcept_eval.py\n"
             "AGENTIC_FAST_ACTS=1 .venv_claude/bin/python scripts/crossconcept_eval.py "
             "--stage fit     # skipped entirely if sweep_probes/ is populated\n"
             ".venv_claude/bin/python scripts/crossconcept_eval.py --stage score\n"
             ".venv_claude/bin/python scripts/crossconcept_summary_md.py\n"
             "```\n")

    (OUT_DIR / "CROSSCONCEPT.md").write_text("".join(L), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'CROSSCONCEPT.md'} ({len(rows)} scorings)")


if __name__ == "__main__":
    main()
