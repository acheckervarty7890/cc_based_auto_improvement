"""Render the high-stakes vintage sweep's current state as a human-readable SUMMARY.md.

Can be run every poll while the sweep is in flight, so whatever is on the remote when
the box dies explains itself without anyone having to re-read a JSONL. Reads only the
append-only progress sidecar (plus the per-arm membership reports, if they exist), so
it is correct mid-run and costs milliseconds.

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

OUT_DIR = A.REPO / "results_hs_gemma27b_batch_ablation/vintage"
VINTAGE_DESC = {
    0: "base training data only, no red-team rows",
    1: "iter-3 pairs whose source success existed at iteration 1",
    2: "…existed at iteration 2",
    3: "all iteration-3 pairs",
}


def _drop_line() -> str:
    """One sentence per arm on what the over-length filter removed."""
    out = []
    for arm in sorted(A.ARMS):
        path = OUT_DIR / f"{arm}_vintage_report.json"
        if not path.exists():
            continue
        rep = json.loads(path.read_text(encoding="utf-8")).get("overlong_drop", {})
        if not rep:
            continue
        out.append(
            f"- **{arm}**: {rep.get('n_at_cap', 0)} row(s) at the cap "
            f"({rep.get('n_at_cap_generated', 0)} generated, "
            f"{rep.get('n_at_cap_source', 0)} attacker-written) → "
            f"{rep.get('n_pairs_dropped', 0)} pair(s), "
            f"{rep.get('n_dropped', 0)} row(s) removed from every vintage\n"
        )
    return "".join(out)


def _readout(by: dict) -> str:
    """Per-arm read-out, computed from the sidecar rather than written by hand.

    Two things are worth stating for every arm and are easy to get wrong by eye: which
    vintage actually scored best, and whether the curve is monotone. A later vintage
    scoring *below* an earlier one means the red-team data added in between made the
    probe worse on the eval splits — the sweep's most actionable outcome — but only if
    the gap clears the seed noise, so it is tested against the pooled sd rather than
    reported on the point estimates.
    """
    out = []
    for arm in sorted({a for a, _ in by}):
        stats = {}
        for (a, v), rs in by.items():
            if a != arm:
                continue
            x = np.array([r["auroc"]["mean"]["pipeline"] for r in rs])
            stats[v] = (x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0, len(x))
        if not stats:
            continue
        best = max(stats, key=lambda v: stats[v][0])
        vs = sorted(stats)
        curve = " → ".join(f"v{v} {stats[v][0]:.4f}" for v in vs)
        line = f"- **{arm}**: {curve}; best is **v{best}**."

        regressions = []
        for i, v in enumerate(vs):
            for w in vs[i + 1:]:
                d = stats[v][0] - stats[w][0]
                pooled = float(np.hypot(stats[v][1], stats[w][1])) or 1e-12
                if d > 0 and d / pooled >= 2.0:
                    regressions.append(f"v{w} is {d:.4f} below v{v} ({d / pooled:.1f}σ)")
        if regressions:
            line += " **Non-monotone**: " + "; ".join(regressions) + "."
        out.append(line + "\n")
    return "".join(out)


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
    L.append("# Red-team vintage sweep — HIGH-STAKES (experiment9)\n")
    L.append(f"_Updated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")
    L.append(
        "\n**What this measures.** Each row is a real `ProbeFactory` refit on the base "
        "training data (`data/hs_ls_200.jsonl`) plus one *vintage* of iteration-3 "
        "red-team pairs, scored on the four `eval_datasets/` splits from cached "
        "activations. Only set membership varies between vintages — content, "
        "activations and every hyperparameter are iteration 3's — so the vintages are "
        "directly comparable to each other in a way the original `probe_iter1/2/3` are "
        "not (those came from separate retrains with their own filter draws and "
        "contrastive generations).\n"
    )
    L.append("\n**Vintages**\n")
    for k, desc in VINTAGE_DESC.items():
        L.append(f"- `v{k}` — {desc}\n")

    drop = _drop_line()
    if drop:
        L.append(
            "\n**Over-length pairs are dropped.** `get_activations` truncates at 1024 "
            "tokens, so a conversation at that width lost its tail — and for an "
            "LLM-written contrastive counterpart the tail is disproportionately the "
            "part carrying the opposite-class label. Every over-cap row in both arms is "
            "a *generated* row; not one attacker-written source overran. The affected "
            "pair is removed **whole**, because dropping the generated half alone would "
            "orphan its success and break the exact 50/50 balance that makes the "
            "vintages comparable.\n\n"
        )
        L.append(drop)

    L.append(
        "\n**Read the sd, not just the mean.** These are unpaired refits with "
        "independent initialisations, so the seed-to-seed sd is the quantity that makes "
        "or breaks a single-seed reading — where it is comparable to a between-vintage "
        "gap, that gap is not evidence of anything.\n"
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
        L.append("| vintage | rows | seeds | " + " | ".join(splits) + " |\n")
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

    readout = _readout(by)
    if readout:
        L.append("\n## Read-out\n\n")
        L.append(readout)
        L.append(
            "\nThe σ figures are against the pooled seed sd of the two vintages "
            "compared, so a flagged regression is one the seed noise cannot explain. "
            "Only gaps of ≥2σ are flagged.\n"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "SUMMARY.md").write_text("".join(L), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'SUMMARY.md'} ({len(rows)} fits)")


if __name__ == "__main__":
    main()
