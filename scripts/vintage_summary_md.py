"""Render the vintage sweep's current state as a human-readable SUMMARY.md.

Run from the checkpointer every poll, so whatever is on the remote when the box dies
explains itself without anyone having to re-read a JSONL. Reads only the append-only
progress sidecar, so it is correct mid-run and costs milliseconds.

Ported from ``experiment11_cloud``'s hu_harm version onto this branch's
instruction-following experiment.

Usage:
    .venv_claude/bin/python scripts/vintage_summary_md.py
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

OUT_DIR = A.REPO / "results_instructions_gemma27b_vintage"
VINTAGE_DESC = {
    0: "base training data only, no red-team rows",
    1: "iter-3 pairs whose source success existed at iteration 1",
    2: "…existed at iteration 2",
    3: "all iteration-3 pairs",
}


def main() -> None:
    prog = OUT_DIR / "vintage_progress.jsonl"
    rows = []
    if prog.exists():
        for line in prog.open(encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # row truncated by a hard kill

    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["vintage"])].append(r)

    L = []
    L.append("# Red-team vintage sweep — instruction-following concept\n")
    L.append(f"_Updated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")
    L.append(
        "\n**What this measures.** Each row is a real `ProbeFactory` refit on the base "
        "training data (`data/instructions_llama70b_50.jsonl`) plus one *vintage* of "
        "iteration-3 red-team pairs, scored on the seven `eval_instructions` splits from "
        "cached activations. Only set membership varies between vintages — content, "
        "activations and every hyperparameter are iteration 3's — so the vintages are "
        "directly comparable to each other in a way the original `probe_iter1/2/3` are "
        "not (those came from separate retrains with their own filter draws and "
        "contrastive generations).\n"
    )
    L.append(
        "\n**Setup.** Probe `google/gemma-3-27b-it` L32, labels "
        "`assistant_follows_the_instruction` / `assistant_does_not_follow_the_instruction`. "
        "Arms are the two attackers of `run_gemma27b_instructions_attackers.sh`: "
        "`gptoss120b` (`openai/gpt-oss-120b`) and `nemotron` "
        "(`nvidia/nemotron-3-ultra-550b-a55b`). All activations — base split, per-conversation "
        "red-team, and the seven eval splits — were pulled from Kaggle "
        "(`anku7890/instructions-gemma27b-*`, `anku7890/*-gemmaevalpt`); no gemma-3-27b "
        "forward pass runs here.\n"
    )
    L.append("\n**Vintages**\n")
    for k, desc in VINTAGE_DESC.items():
        L.append(f"- `v{k}` — {desc}\n")
    L.append(
        "\n**The over-1024-token filter.** `get_activations` pads *or truncates* every "
        "conversation to 1024 tokens, so a longer one is trained on from its opening alone. "
        "These runs predate `token_budget.py`, so nothing length-guarded the contrastive "
        "generator. Rows whose conversation exceeds the cap — **and** rows whose cached "
        "activation is stored truncated, which `get_activations` also produces for a "
        "short conversation that merely shared an extraction batch with an over-long one — "
        "are dropped together with their pair partner, keeping every vintage exactly 50/50. "
        "Per-arm counts are in `*_vintage.json` under `report.long_filter`.\n"
    )
    L.append(
        "\n**Read the sd, not just the mean.** These are independent `ProbeFactory` fits "
        "with independent initialisations, and seed alone moves some splits by more than "
        "the between-vintage gaps. A single-seed comparison of two vintages means nothing; "
        "quantifying that is what this sweep exists for.\n"
    )

    L.append(f"\n## Progress: {len(rows)} fits recorded\n\n")
    for arm in sorted({r["arm"] for r in rows}):
        per_v = defaultdict(list)
        for r in rows:
            if r["arm"] == arm:
                per_v[r["vintage"]].append(r["seed"])
        L.append(f"- **{arm}**: " + ", ".join(
            f"v{v}×{len(ss)}" for v, ss in sorted(per_v.items())
        ) + "\n")

    for arm in sorted({r["arm"] for r in rows}):
        L.append(f"\n## {arm} — mean ± sd over seeds (pipeline scale)\n\n")
        splits = list(A.EVAL_SPLITS) + ["mean"]
        L.append("| vintage | rows | seeds | "
                 + " | ".join(s.replace("eval_", "") for s in splits) + " |\n")
        L.append("|" + "---|" * (3 + len(splits)) + "\n")
        for (a, v), rs in sorted(by.items()):
            if a != arm:
                continue
            cells = []
            for sp in splits:
                x = np.array([r["auroc"][sp]["pipeline"] for r in rs])
                sd = x.std(ddof=1) if len(x) > 1 else float("nan")
                cells.append(
                    f"{x.mean():.4f}" + ("" if len(x) < 2 else f" ± {sd:.4f}")
                )
            L.append(f"| v{v} | {rs[0]['n_redteam_rows']} | {len(rs)} | "
                     + " | ".join(cells) + " |\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "SUMMARY.md").write_text("".join(L), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'SUMMARY.md'} ({len(rows)} fits)")


if __name__ == "__main__":
    main()
