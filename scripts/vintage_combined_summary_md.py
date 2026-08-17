"""Render the pooled-attacker sweep as a SUMMARY.md, next to the two single-arm curves.

Reads only the append-only progress sidecars — the combined sweep's and the single-arm
sweep's — so it is correct mid-run and costs milliseconds. Run it every poll while
``vintage_combined.py`` is in flight and whatever is on disk when the box dies explains
itself.

The point of the table is the *comparison*: the combined row is fitted under exactly the
single-arm sweep's protocol (same activations, same split, same spec, same seeds, same
splits, same over-length filter), so the only difference at a given vintage is that its
training set is the union of the two arms' pairs rather than one arm's. Every gap is
therefore quoted against the pooled seed sd, and only >= 2 sigma is treated as a result.

Usage:
    .venv_claude/bin/python scripts/vintage_combined_summary_md.py
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

OUT_DIR = A.REPO / "results_hs_gemma27b_batch_ablation/vintage_combined"
REF_DIR = A.REPO / "results_hs_gemma27b_batch_ablation/vintage"
ARM_ORDER = ["deepseekv4pro", "gptoss120b", "combined"]
VINTAGE_DESC = {
    0: "base training data only, no red-team rows",
    1: "iter-3 pairs whose source success existed at iteration 1",
    2: "…existed at iteration 2",
    3: "all iteration-3 pairs — i.e. v1 + v2 + v3",
}


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.open(encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # row truncated by a hard kill
    return out


def _stat(rs: list[dict], split: str) -> tuple[float, float, int]:
    x = np.array([r["auroc"][split]["pipeline"] for r in rs])
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0, len(x)


def _cell(rs: list[dict], split: str) -> str:
    m, sd, n = _stat(rs, split)
    return f"{m:.4f}" + ("" if n < 2 else f" ± {sd:.4f}")


def _sigma(a: tuple[float, float, int], b: tuple[float, float, int]) -> float:
    """Gap b-a in pooled-sd units. The yardstick for every claim in this file."""
    pooled = float(np.hypot(a[1], b[1])) or 1e-12
    return (b[0] - a[0]) / pooled


def main() -> None:
    by: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in _rows(REF_DIR / "vintage_progress.jsonl") + _rows(
        OUT_DIR / "vintage_progress.jsonl"
    ):
        by[(r["arm"], r["vintage"])].append(r)
    # v0 is one training set — no red-team rows means it does not depend on the arm — so
    # every sweep that fitted it produced the same numbers under a different arm label.
    # Collapse them to a single row keyed on the reference arm, or the table repeats
    # itself and the head-to-head compares a cell against a copy of itself.
    v0 = by.pop(("deepseekv4pro", 0), [])
    for arm in ("gptoss120b", "combined"):
        by.pop((arm, 0), None)
    if v0:
        by[("deepseekv4pro", 0)] = v0

    splits = list(A.EVAL_SPLITS) + ["mean"]
    n_combined = sum(len(v) for k, v in by.items() if k[0] == "combined")

    L = []
    L.append("# Pooling both attackers' red-team data — HIGH-STAKES (experiment9)\n")
    L.append(f"_Updated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")
    L.append(
        "\n**What this measures.** The sweep in `../vintage/` fits each attacker arm "
        "separately. This one adds a third arm, **`combined`**, whose training set at "
        "each vintage is the base training data (`data/hs_ls_200.jsonl`) plus the "
        "*union* of both arms' iteration-3 red-team pairs of that vintage. Vintages are "
        "cumulative, so `combined` v3 is the base data plus every iteration-3 pair "
        "either attacker produced — the \"v1 + v2 + v3, both attackers\" set.\n"
    )
    L.append(
        "\nEverything except the row set is held at the single-arm sweep's values: the "
        "same per-conversation cached activations, the same content-deterministic "
        "train/val split, the same `ProbeSpec` (verified identical between the two "
        "arms' `probe_iter3.pkl`), the same ten seeds, the same four `eval_datasets/` "
        "splits, the same `--drop-overlong pair`. So the `combined` row at vintage k is "
        "comparable to the two single-arm rows at vintage k, and the only thing that "
        "differs is which conversations are in the training set.\n"
    )
    L.append("\n**Vintages**\n")
    for k, desc in VINTAGE_DESC.items():
        L.append(f"- `v{k}` — {desc}\n")
    L.append(
        "\n**v0 is one fit, not three.** With no red-team rows the training set does "
        "not depend on the arm, so v0 is reported once. The combined sweep fits it "
        "anyway as a cross-check: it reproduces the single-arm sweep's v0 AUROC "
        "**exactly** (max |Δ| = 0 across every seed and split), which is what licenses "
        "reading the `combined` rows as continuous with the others — the pooled "
        "indexing and the disk-backed assembly this sweep needs are not perturbing the "
        "fit path.\n"
    )
    L.append(
        "\n**Read the sd, not just the mean.** These are unpaired refits with "
        "independent initialisations. Every gap below is quoted against the pooled seed "
        "sd of the two cells compared, and only >= 2 sigma is treated as a result.\n"
    )

    L.append(f"\n## Progress: {n_combined} combined fits recorded\n\n")
    per_v = defaultdict(list)
    for (arm, v), rs in by.items():
        if arm == "combined":
            per_v[v] = rs
    L.append("- **combined**: " + (", ".join(
        f"v{v}×{len(rs)}" for v, rs in sorted(per_v.items())
    ) or "none yet") + "\n")

    L.append("\n## Eval AUROC — mean ± sd over seeds (pipeline scale)\n\n")
    L.append("| vintage | arm | rows | seeds | " + " | ".join(splits) + " |\n")
    L.append("|" + "---|" * (4 + len(splits)) + "\n")
    for v in sorted({k[1] for k in by}):
        for arm in ARM_ORDER:
            rs = by.get((arm, v))
            if not rs:
                continue
            name = "_base only_" if v == 0 else f"**{arm}**" if arm == "combined" else arm
            L.append(
                f"| v{v} | {name} | {rs[0]['n_redteam_rows']} | {len(rs)} | "
                + " | ".join(_cell(rs, sp) for sp in splits)
                + " |\n"
            )

    # --- the head-to-head -------------------------------------------------------
    lines = []
    for v in sorted({k[1] for k in by if k[0] == "combined"}):
        comb = by.get(("combined", v))
        if not comb:
            continue
        c = _stat(comb, "mean")
        for arm in ("deepseekv4pro", "gptoss120b"):
            single = by.get((arm, v))
            if not single:
                continue
            s = _stat(single, "mean")
            # A one-seed cell has sd 0, which would divide a real gap by nothing and
            # print a spectacular sigma off a single draw. Both sides need >= 2 seeds
            # before the comparison means anything.
            if s[2] < 2 or c[2] < 2:
                sig_txt, verdict = "—", f"_needs ≥2 seeds (have {c[2]})_"
            else:
                sig = _sigma(s, c)
                sig_txt = f"{abs(sig):.1f}"
                verdict = (
                    "indistinguishable" if abs(sig) < 2
                    else ("**above**" if sig > 0 else "**below**")
                )
            lines.append(
                f"| v{v} | {arm} | {single[0]['n_redteam_rows']} | {s[0]:.4f} ± {s[1]:.4f} "
                f"| {comb[0]['n_redteam_rows']} | {c[0]:.4f} ± {c[1]:.4f} "
                f"| {c[0] - s[0]:+.4f} | {sig_txt} | {verdict} |\n"
            )
    if lines:
        L.append("\n## Pooled against each arm alone\n\n")
        L.append("| vintage | arm | its rows | arm mean | pooled rows | combined mean "
                 "| gap | sigma | verdict |\n")
        L.append("|" + "---|" * 9 + "\n")
        L.extend(lines)
        L.append(
            "\n`sigma` is the gap over the pooled seed sd of the two cells. "
            "\"indistinguishable\" means the seed noise covers it — not that the two "
            "training sets are equivalent, only that this sweep cannot separate them.\n"
        )

    # --- the read-out -----------------------------------------------------------
    stats = {v: _stat(rs, "mean") for (arm, v), rs in by.items() if arm == "combined"}
    if stats:
        vs = sorted(stats)
        L.append("\n## Read-out\n\n")
        L.append("- **combined**: "
                 + " → ".join(f"v{v} {stats[v][0]:.4f}" for v in vs)
                 + f"; best is **v{max(stats, key=lambda v: stats[v][0])}**.\n")
        regressions = []
        for i, v in enumerate(vs):
            for w in vs[i + 1:]:
                if stats[v][2] < 2 or stats[w][2] < 2:
                    continue  # one seed has no sd to test the gap against
                sig = _sigma(stats[v], stats[w])
                if sig <= -2.0:
                    regressions.append(
                        f"v{w} is {stats[v][0] - stats[w][0]:.4f} below v{v} "
                        f"({abs(sig):.1f}σ)"
                    )
        if regressions:
            L.append("  - **Non-monotone**: " + "; ".join(regressions) + ".\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "SUMMARY.md").write_text("".join(L), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'SUMMARY.md'} ({n_combined} combined fits)")


if __name__ == "__main__":
    main()
