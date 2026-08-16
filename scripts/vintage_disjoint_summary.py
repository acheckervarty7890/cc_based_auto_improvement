#!/usr/bin/env python
"""Join the cumulative and disjoint vintage sweeps into one comparison.

The cumulative sweep answers "how good is the probe if the pipeline had stopped after
iteration k". It cannot answer "how good is iteration k's *own* contribution", because a
later vintage there is both **newer** data and **more** data — v3 contains v2 contains
v1. A flattening curve is therefore ambiguous: either the later iterations' finds are
individually weaker, or they are simply redundant with what is already in the set.

`attribution_vintage.py --membership disjoint` breaks that tie by fitting each increment
**alone** on top of the base data: v1, then v2-minus-v1, then v3-minus-v2. This script
reads both sweeps' progress sidecars and reports the two contrasts that matter:

1. **Increment vs. the cumulative set that contains it** — v2-only against v2, v3-only
   against v3. If the increment alone matches the whole set, the earlier vintages were
   adding nothing that the later data does not already carry.
2. **Increment vs. increment** — is iteration 3's new data individually better or worse
   than iteration 1's, per fit and per row?

`v1` is the same set under both memberships (there is no earlier vintage to subtract), so
its fits are read from the cumulative sweep rather than refitted. That is an identity of
construction, not an approximation: `vintages()` builds v1 from the same row indices in
the same order either way, and the fit is deterministic in (data, seed) — which the
cumulative sweep itself demonstrates, since its v0 lands on identical numbers for two
arms that were fitted separately.

All comparisons are **unpaired** (independent initialisations), so a gap is quoted in
units of the pooled seed sd of the two cells compared, and only >= 2 sigma is called a
result.

Usage:
    .venv_claude/bin/python scripts/vintage_disjoint_summary.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A

CUMULATIVE_DIR = A.REPO / "results_hs_gemma27b_batch_ablation/vintage"
DISJOINT_DIR = A.REPO / "results_hs_gemma27b_batch_ablation/vintage_disjoint"
SCALE = "pipeline"


def load(progress: Path) -> dict:
    """``{(arm, vintage): [row, ...]}`` from a sweep's progress sidecar."""
    by: dict[tuple[str, int], list[dict]] = defaultdict(list)
    if not progress.exists():
        raise SystemExit(f"missing {progress} — run that sweep first")
    for line in progress.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # row truncated by a hard kill
        by[(r["arm"], int(r["vintage"]))].append(r)
    return by


def cell(rows: list[dict], split: str) -> tuple[float, float, int]:
    """``(mean, sd, n_seeds)`` of one split's AUROC over a cell's fits."""
    vals = [r["auroc"][split][SCALE] for r in rows]
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0, len(vals))


def sigma(a: tuple[float, float, int], b: tuple[float, float, int]) -> tuple[float, float]:
    """``(gap, gap / pooled sd)`` for two unpaired cells; inf sigma when both sds vanish."""
    gap = a[0] - b[0]
    pooled = (a[1] ** 2 + b[1] ** 2) ** 0.5
    return gap, (abs(gap) / pooled if pooled else float("inf"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cumulative-dir", type=Path, default=CUMULATIVE_DIR)
    ap.add_argument("--disjoint-dir", type=Path, default=DISJOINT_DIR)
    args = ap.parse_args()

    cum = load(args.cumulative_dir / "vintage_progress.jsonl")
    dis = load(args.disjoint_dir / "vintage_progress.jsonl")
    arms = sorted({a for a, _ in dis})
    splits = list(A.EVAL_SPLITS) + ["mean"]

    # v1 is the same set under both memberships — read it from the cumulative sweep.
    slices: dict[tuple[str, int], list[dict]] = {}
    for arm in arms:
        slices[(arm, 1)] = cum[(arm, 1)]
        for k in (2, 3):
            if (arm, k) in dis:
                slices[(arm, k)] = dis[(arm, k)]

    rows_out = []
    for (arm, k), rs in sorted(slices.items()):
        for split in splits:
            m, sd, n = cell(rs, split)
            rows_out.append({
                "arm": arm, "slice": f"v{k}" if k == 1 else f"v{k}-only",
                "n_redteam_rows": rs[0]["n_redteam_rows"], "n_seeds": n,
                "dataset": split, "scale": SCALE,
                "mean": m, "sd": sd,
            })
    out_csv = args.disjoint_dir / "disjoint_vs_cumulative.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {out_csv.relative_to(A.REPO)} ({len(rows_out)} rows)")

    L: list[str] = []
    L.append("# Red-team vintage sweep — DISJOINT slices (experiment9, high-stakes)\n")
    L.append(f"_Updated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")
    L.append(
        "\n**What this measures.** The companion sweep in `../vintage/` is *cumulative*: "
        "vintage k holds every iteration-3 pair whose source success existed by iteration "
        "k, so v3 contains v2 contains v1. That conflates two things — a later vintage is "
        "both newer data and **more** data — so its flattening curve cannot say whether "
        "the later iterations' finds are individually weaker or merely redundant.\n\n"
        "Here each vintage is fitted **alone** on top of the base training data: `v1`, "
        "then `v2-only` (v2 minus v1), then `v3-only` (v3 minus v2). Everything else is "
        "held at the cumulative sweep's values — same content, same cached activations, "
        "same hyperparameters, same ten seeds, same four `eval_datasets/` splits.\n"
    )
    L.append(
        "\n`v1` is identical under both memberships (nothing earlier to subtract), so its "
        "fits are the cumulative sweep's, not refits. The subtraction runs downward — v3 "
        "loses *cumulative* v2, then v2 loses *cumulative* v1 — so each slice is measured "
        "against the set the previous vintage actually held.\n"
    )
    L.append(
        "\n**Unpaired, so read the sigma.** Every gap below is quoted against the pooled "
        "seed sd of the two cells compared; only >= 2 sigma is treated as a result.\n"
    )

    for arm in arms:
        L.append(f"\n## {arm} — mean +/- sd over seeds ({SCALE} scale)\n\n")
        L.append("| slice | rows | seeds | " + " | ".join(A.EVAL_SPLITS) + " | mean |\n")
        L.append("|---" * (len(A.EVAL_SPLITS) + 4) + "|\n")
        for k in (1, 2, 3):
            rs = slices.get((arm, k))
            if not rs:
                continue
            name = "v1" if k == 1 else f"v{k}-only"
            cells = " | ".join(
                f"{cell(rs, s)[0]:.4f} ± {cell(rs, s)[1]:.4f}" for s in splits
            )
            L.append(f"| {name} | {rs[0]['n_redteam_rows']} | {cell(rs, 'mean')[2]} | {cells} |\n")

    L.append("\n## The increment against the set that contains it\n\n")
    L.append("| arm | increment | rows | cumulative | rows | increment mean | cumulative mean | gap | sigma |\n")
    L.append("|---|---|---|---|---|---|---|---|---|\n")
    for arm in arms:
        for k in (2, 3):
            if (arm, k) not in slices or (arm, k) not in cum:
                continue
            inc, whole = cell(slices[(arm, k)], "mean"), cell(cum[(arm, k)], "mean")
            gap, s = sigma(inc, whole)
            L.append(
                f"| {arm} | v{k}-only | {slices[(arm, k)][0]['n_redteam_rows']} | v{k} | "
                f"{cum[(arm, k)][0]['n_redteam_rows']} | {inc[0]:.4f} ± {inc[1]:.4f} | "
                f"{whole[0]:.4f} ± {whole[1]:.4f} | {gap:+.4f} | {s:.1f} |\n"
            )

    L.append("\n## The increments against each other\n\n")
    L.append("| arm | pair | rows | gap | sigma |\n|---|---|---|---|---|\n")
    for arm in arms:
        for a_k, b_k in ((2, 1), (3, 1), (3, 2)):
            if (arm, a_k) not in slices or (arm, b_k) not in slices:
                continue
            a, b = cell(slices[(arm, a_k)], "mean"), cell(slices[(arm, b_k)], "mean")
            gap, s = sigma(a, b)
            na = slices[(arm, a_k)][0]["n_redteam_rows"]
            nb = slices[(arm, b_k)][0]["n_redteam_rows"]
            name = lambda k: "v1" if k == 1 else f"v{k}-only"  # noqa: E731
            L.append(f"| {arm} | {name(a_k)} − {name(b_k)} | {na} vs {nb} | {gap:+.4f} | {s:.1f} |\n")

    L.append("\n## Read-out\n\n")
    for arm in arms:
        parts = []
        for k in (1, 2, 3):
            if (arm, k) in slices:
                m = cell(slices[(arm, k)], "mean")[0]
                n = slices[(arm, k)][0]["n_redteam_rows"]
                parts.append(f"{'v1' if k == 1 else f'v{k}-only'} {m:.4f} ({n} rows)")
        L.append(f"- **{arm}**: " + " → ".join(parts) + ".\n")
        for k in (2, 3):
            if (arm, k) not in slices or (arm, k) not in cum:
                continue
            inc, whole = cell(slices[(arm, k)], "mean"), cell(cum[(arm, k)], "mean")
            gap, s = sigma(inc, whole)
            verdict = (
                "indistinguishable from the whole cumulative set"
                if s < 2 else
                ("above" if gap > 0 else "below") + " the cumulative set"
            )
            L.append(
                f"  - v{k}-only ({slices[(arm, k)][0]['n_redteam_rows']} rows) is {verdict} "
                f"({cum[(arm, k)][0]['n_redteam_rows']} rows) — {gap:+.4f}, {s:.1f} sigma.\n"
            )

    path = args.disjoint_dir / "SUMMARY.md"
    path.write_text("".join(L), encoding="utf-8")
    print(f"wrote {path.relative_to(A.REPO)}")


if __name__ == "__main__":
    main()
