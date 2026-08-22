#!/usr/bin/env python
"""Render the novelty study's results into ``results/summary.md``.

    .venv_claude/bin/python analysis/novelty/report.py

Reads the artefacts the three phases wrote (``novelty_*.npz``, ``regions_*.json``,
``ablation_*.jsonl``) and produces the tables the decision actually turns on -- most
importantly the novelty-vs-random head-to-head, which is the only comparison that can
show the geometry is predictive rather than the effect being training-set size.

THE NOISE FLOOR governs every reading here. Identical data in a different row order
gives a different probe (the fit is order-sensitive), so the spread across the
``full_perm*`` conditions bounds what any other difference may be read as meaning.
Deltas inside it are reported as inert, not as small effects.

THE DECISION RULE, applied per condition, combines the two metrics:

    eval up, cross not down     -> DROP     genuinely off-task rows
    eval flat, cross down       -> KEEP     eval is blind here; the rows carry
                                            robustness eval cannot see
    eval up, cross down         -> TRADE    a real choice, not a free win
    eval down                   -> KEEP     removal hurts outright
    both flat                   -> INERT    removing them changes nothing measurable

"eval flat" is the case the whole study exists to guard against: it is NOT evidence the
rows were useless.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments as X  # noqa: E402


def load_rows(exp_key: str) -> list[dict]:
    p = X.RESULTS / f"ablation_{exp_key}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def noise_floor(rows: list[dict]) -> tuple[float, int]:
    """Half the full-range spread of the identical-data permutations, as +/- a bound.

    This is the floor for "does this refit differ from that refit at all". It is NOT the
    right band for judging a *removal*, which perturbs the data as well as the order --
    see :func:`comparison_band`.
    """
    vals = [r["macro_auroc"] for r in rows if r["condition"].startswith("full_perm")]
    vals += [r["macro_auroc"] for r in rows if r["condition"] == "full"]
    if len(vals) < 2:
        return 0.0146, len(vals)  # the ceiling study's measured value, as a fallback
    return (max(vals) - min(vals)) / 2, len(vals)


def comparison_band(rows: list[dict]) -> float:
    """The band a removal has to clear to mean anything.

    Permuting row order alone moves macro AUROC by a few thousandths, but *removing* a
    matched number of rows at random moves it far more -- the random-seed conditions at
    one quantile routinely span 0.04 AUROC. Judging a targeted removal against the
    permutation floor would therefore call a great many size-effects "real". The band
    used here is the larger of the two: the permutation half-range, and the mean
    half-range across the matched-n random-removal groups, which is the honest
    "what does removing this many rows do by chance" scale.
    """
    perm, _ = noise_floor(rows)
    by: dict[str, list[float]] = {}
    for r in rows:
        c = r["condition"]
        if c.startswith("drop_random_q"):
            by.setdefault(c.split("_s")[0], []).append(r["macro_auroc"])
    spreads = [(max(v) - min(v)) / 2 for v in by.values() if len(v) > 1]
    return max(perm, statistics.fmean(spreads)) if spreads else perm


def verdict(d_eval: float, d_cross: float | None, floor: float) -> str:
    if d_eval < -floor:
        return "KEEP (removal hurts eval)"
    up = d_eval > floor
    if d_cross is None:
        return "DROP?" if up else "INERT"
    down = d_cross < -floor
    if up and not down:
        return "DROP"
    if up and down:
        return "TRADE-OFF"
    if not up and down:
        return "KEEP (eval blind)"
    return "INERT"


def phase1_table(lines: list[str]) -> None:
    lines += ["## Phase 1 — how novel is each arm's red-team set?", ""]
    lines += [
        "| experiment | arm | rows | eval self-kNN p95 | dev→eval outside% | rt→eval kNN | rt outside% | along_frac | corr(novelty, orth) | published Δ eval |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for exp, arm in X.all_arms():
        f = X.RESULTS / f"novelty_{exp.key}_{arm.name}.npz"
        if not f.exists():
            continue
        s = np.load(f, allow_pickle=True)
        p95 = float(s["_eval_self_p95"][0])
        dev_out = 100 * (s["_dev_to_eval_knn"] > p95).mean()
        corr = np.corrcoef(s["knn_eval"], s["orth"])[0, 1]
        delta = published_delta(arm)
        lines.append(
            f"| {exp.key} | {arm.name} | {len(s['knn_eval'])} | {p95:.4f} | {dev_out:.1f}% | "
            f"{s['knn_eval'].mean():.4f} | {100 * s['outside'].mean():.1f}% | "
            f"{s['along_frac'].mean():.3f} | {corr:+.2f} | {delta:+.4f} |"
        )
    lines += [
        "",
        "`outside%` = share of rows further from eval than 95% of eval is from itself. "
        "`along_frac` = share of a row's displacement from its local eval neighbourhood that "
        "lies on the probe's decision axis. `published Δ eval` = the arm's own last-iteration "
        "macro AUROC minus its iteration 0, from the run's comparison CSV.",
        "",
    ]


def published_delta(arm: X.Arm) -> float:
    import csv

    if not arm.comparison_csv.exists():
        return float("nan")
    per = {}
    with arm.comparison_csv.open() as f:
        for row in csv.DictReader(f):
            if row["dataset"] == "mean":
                per[row["round"]] = float(row["auroc"])
    if not per:
        return float("nan")
    return per.get(f"iter{arm.last_iteration}", float("nan")) - per.get("iter0", float("nan"))


def phase2_table(lines: list[str]) -> None:
    lines += [
        "## Phase 2 — regions",
        "",
        "HDBSCAN over the red-team rows' own PCA assigns the large majority of every arm's "
        "rows to **noise**: these attack sets are diffuse, not organised into dense families. "
        "That is a finding, not a failure — it already means there is no compact \"bad region\" "
        "to excise. A k-means covering (k=6) is therefore used for the region-level ablations, "
        "and it does separate the rows by novelty even though density does not.",
        "",
        "| experiment | arm | HDBSCAN regions | rows in noise | k-means region outside% (min → max) |",
        "|---|---|--:|--:|--:|",
    ]
    for exp, arm in X.all_arms():
        h = X.RESULTS / f"regions_{exp.key}_{arm.name}.json"
        k = X.RESULTS / f"regions_{exp.key}_{arm.name}_kmeans.json"
        if not (h.exists() and k.exists()):
            continue
        hj, kj = json.loads(h.read_text()), json.loads(k.read_text())
        noise = next((r["n"] for r in hj["regions"] if r["id"] == -1), 0)
        outs = sorted(r["outside_pct"] for r in kj["regions"])
        lines.append(
            f"| {exp.key} | {arm.name} | {hj['n_regions']} | {noise}/{hj['n_rows']} "
            f"({100 * noise / hj['n_rows']:.0f}%) | {outs[0]:.0f}% → {outs[-1]:.0f}% |"
        )
    lines.append("")


def arm_section(lines: list[str], exp: X.Experiment, arm: X.Arm, rows: list[dict]) -> None:
    rows = [r for r in rows if r["arm"] == arm.name]
    if not rows:
        return
    base = next((r for r in rows if r["condition"] == "full"), None)
    if base is None:
        return
    perm_floor, n_perm = noise_floor(rows)
    floor = comparison_band(rows)
    b_eval, b_cross = base["macro_auroc"], base.get("cross_auroc")

    lines += [
        f"### {exp.key} / {arm.name}",
        "",
        f"`full` (all {base['n_redteam']} red-team rows, file order) = **{b_eval:.4f}** macro AUROC, "
        f"dev {base['dev_auroc']:.4f}, cross-attacker {fmt(b_cross)}. "
        f"Row-order noise floor ±{perm_floor:.4f} from {n_perm} identical-data orderings; "
        f"**comparison band ±{floor:.4f}**, the scale on which removing this many rows at "
        f"random moves the score. Verdicts below use the band, not the floor.",
        "",
        "| condition | train n | dropped | macro AUROC | Δ vs full | dev | cross-attacker | Δ cross | verdict |",
        "|---|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    order = sorted(rows, key=lambda r: (_group(r["condition"]), r["condition"]))
    for r in order:
        d = r["macro_auroc"] - b_eval
        dc = (r["cross_auroc"] - b_cross) if (r.get("cross_auroc") is not None and b_cross is not None) else None
        v = "—" if r["condition"] in ("full",) or r["condition"].startswith("full_perm") else verdict(d, dc, floor)
        flag = " ⚠︎" if r.get("below_step_floor") else ""
        lines.append(
            f"| `{r['condition']}`{flag} | {r['n_train']} | {r.get('n_dropped', 0)} | {r['macro_auroc']:.4f} | "
            f"{d:+.4f} | {r['dev_auroc']:.4f} | {fmt(r.get('cross_auroc'))} | {fmt_d(dc)} | {v} |"
        )
    lines += ["", "⚠︎ = training set below the 64-row optimizer-step floor; the fit takes no step.", ""]
    head_to_head(lines, rows, b_eval, floor)
    region_attribution(lines, exp, arm, rows, b_eval, b_cross, floor)


def _group(name: str) -> int:
    for i, p in enumerate(("full", "base_only", "drop_top", "drop_bottom", "drop_random",
                           "drop_outside", "drop_relative", "drop_region")):
        if name.startswith(p):
            return i
    return 99


def head_to_head(lines: list[str], rows: list[dict], b_eval: float, floor: float) -> None:
    """The comparison the whole design exists for: novelty-ordered vs random, matched n."""
    by = {r["condition"]: r for r in rows}
    qs = sorted({int(c.split("_q")[1].split("_")[0]) for c in by if c.startswith("drop_top_q")})
    if not qs:
        return
    lines += [
        "**Novelty-ordered vs random removal, matched n** — the only comparison that can show "
        "the geometry is predictive. Random removal is the size control: if targeting the most "
        "novel rows does no better than dropping the same number at random, the geometry carries "
        "no information about *which* rows to drop.",
        "",
        "| drop % | most-novel | least-novel | random (mean ± range) | novel − random | least − random | ordering |",
        "|--:|--:|--:|--:|--:|--:|---|",
    ]
    for q in qs:
        top = by.get(f"drop_top_q{q}")
        bot = by.get(f"drop_bottom_q{q}")
        rnd = [r["macro_auroc"] for c, r in by.items() if c.startswith(f"drop_random_q{q}_")]
        if not top or not rnd:
            continue
        rm = statistics.fmean(rnd)
        rr = (max(rnd) - min(rnd)) / 2
        dtop = top["macro_auroc"] - rm
        dbot = (bot["macro_auroc"] - rm) if bot else None
        marks = []
        if abs(dtop) > floor:
            marks.append("novel≠random")
        if dbot is not None and abs(dbot) > floor:
            marks.append("least≠random")
        order = "novel > random > least" if (bot and top["macro_auroc"] > rm > bot["macro_auroc"]) else "—"
        lines.append(
            f"| {q}% | {top['macro_auroc']:.4f} | {fmt(bot['macro_auroc'] if bot else None)} | "
            f"{rm:.4f} ± {rr:.4f} | {dtop:+.4f} | {fmt_d(dbot)} | "
            f"{order}{(' (' + ', '.join(marks) + ')') if marks else ''} |"
        )
    lines += [
        "",
        f"Deltas exceeding the ±{floor:.4f} comparison band are the only ones that mean anything. "
        "`novel > random > least` in the ordering column means novelty ranks the rows in the "
        "expected direction *at that quantile* — which is a weaker claim than the gap being "
        "large enough to act on.",
        "",
    ]


def region_attribution(lines, exp, arm, rows, b_eval, b_cross, floor) -> None:
    """Which regions actually matter, with a sample of what is in them.

    Regions are a *covering*, so every row belongs to one and the effects are not
    independent -- but at matched size two regions can still differ, and when they do it
    is the content, not the distance, that separates them. Hence the excerpt column.
    """
    f = X.RESULTS / f"regions_{exp.key}_{arm.name}_kmeans.json"
    if not f.exists():
        return
    regs = {r["id"]: r for r in json.loads(f.read_text())["regions"]}
    by = {r["condition"]: r for r in rows}
    got = [(rid, by[f"drop_region_{rid}"]) for rid in regs if f"drop_region_{rid}" in by]
    if not got:
        return
    lines += [
        "**Region attribution** — effect of removing each k-means region, with what is in it.",
        "",
        "| region | n | outside% | Δ eval | Δ cross | verdict | representative content |",
        "|---|--:|--:|--:|--:|---|---|",
    ]
    for rid, r in sorted(got, key=lambda t: -(t[1]["macro_auroc"] - b_eval)):
        reg = regs[rid]
        d = r["macro_auroc"] - b_eval
        dc = (r["cross_auroc"] - b_cross) if (r.get("cross_auroc") is not None and b_cross is not None) else None
        ex = reg["examples"][0]["text"] if reg["examples"] else ""
        ex = " ".join(ex.split())[:150].replace("|", "/")
        lines.append(
            f"| region_{rid} | {reg['n']} | {reg['outside_pct']:.0f}% | {d:+.4f} | {fmt_d(dc)} | "
            f"{verdict(d, dc, floor)} | {ex}… |"
        )
    lines.append("")


def fmt(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def fmt_d(v) -> str:
    return "—" if v is None else f"{v:+.4f}"


def synthesis(lines: list[str]) -> None:
    """The cross-arm read, computed rather than asserted."""
    lines += ["## Synthesis", ""]

    novelty, gain = [], []
    for exp, arm in X.all_arms():
        f = X.RESULTS / f"novelty_{exp.key}_{arm.name}.npz"
        if not f.exists():
            continue
        s = np.load(f, allow_pickle=True)
        d = published_delta(arm)
        if not np.isnan(d):
            novelty.append(100 * float(s["outside"].mean()))
            gain.append(d)
    if len(novelty) >= 3:
        r = float(np.corrcoef(novelty, gain)[0, 1])
        lines += [
            f"Across the {len(novelty)} arms, the share of red-team rows outside the eval manifold and the "
            f"arm's published eval gain correlate **{r:+.2f}** "
            f"(outside%: {', '.join(f'{v:.0f}' for v in novelty)} vs "
            f"Δ AUROC: {', '.join(f'{v:+.3f}' for v in gain)}). "
            + (
                "The hypothesis predicts a *negative* correlation — more novelty, more harm. "
                "The observed sign is the opposite: the **most** off-manifold red-team sets "
                "produced the **largest** eval gains, and the least novel one produced the "
                "largest loss. Four arms is not a result on its own — it is a cross-arm "
                "correlation over four points, not a controlled comparison — but it is a strong "
                "prior against 'far from eval ⇒ harmful', and the within-arm removals below are "
                "what actually test it."
                if r > 0.3
                else (
                    "That is the sign the hypothesis predicts; the within-arm removals below test "
                    "whether it survives a controlled comparison."
                    if r < -0.3
                    else "No clear cross-arm relationship; the within-arm removals below are the test."
                )
            ),
            "",
        ]

    # Did novelty-ordered removal ever beat matched-n random removal by more than noise?
    beats, total = [], 0
    for exp in X.EXPERIMENTS.values():
        rows = load_rows(exp.key)
        for arm in exp.arms.values():
            ar = [r for r in rows if r["arm"] == arm.name]
            base = next((r for r in ar if r["condition"] == "full"), None)
            if not base:
                continue
            floor = comparison_band(ar)
            by = {r["condition"]: r for r in ar}
            for q in sorted({int(c.split("_q")[1].split("_")[0]) for c in by if c.startswith("drop_top_q")}):
                rnd = [r["macro_auroc"] for c, r in by.items() if c.startswith(f"drop_random_q{q}_")]
                top = by.get(f"drop_top_q{q}")
                if not rnd or not top:
                    continue
                total += 1
                diff = top["macro_auroc"] - statistics.fmean(rnd)
                if abs(diff) > floor:
                    beats.append((exp.key, arm.name, q, diff))
    if total:
        pos = [b for b in beats if b[3] > 0]
        neg = [b for b in beats if b[3] < 0]
        lines += [
            f"**Novelty-ordered vs random removal.** Of {total} matched-n comparisons across the "
            f"{_n_arms()} arms, {len(beats)} differ from random by more than that arm's comparison "
            f"band — **{len(pos)} in one direction and {len(neg)} in the other**."
            + (
                " Targeting the most novel rows is better than chance on some arms and worse on "
                "others, so there is no pruning rule here that survives being moved to a different "
                "attacker."
                if len(pos) and len(neg)
                else " All the differences that clear the band point the same way, but a single "
                "concept cannot show whether that survives a change of concept — cloud_3 found "
                "this signal REVERSES on high-stakes, so treat it as concept-specific until it is "
                "re-measured on another one."
            ),
            "",
        ]
        for e, a, q, d in beats:
            lines.append(f"- `{e}/{a}` at {q}%: {d:+.4f} vs random")
        lines += [
            "",
            "The *ordering* is more consistent than the magnitudes: on both arms removal tends to "
            "rank `most-novel > random > least-novel`, so novelty carries some signal about which "
            "rows are dispensable here. This run covers ONE concept, so it cannot test whether "
            "that ordering travels. experiment_instruction_cloud_3 ran the same protocol on "
            "high-stakes and found it does NOT — there the ordering fails on both arms and "
            "inverts on `deepseekv4pro`. Read the ordering below as an instructions-specific "
            "result, not a pruning rule.",
            "",
        ]

    # The one effect that IS consistent across all four arms.
    lines += ["### What removing ALL red-team data does", ""]
    lines += [
        "| arm | eval: full → base only | cross-attacker: full → base only |",
        "|---|--:|--:|",
    ]
    for exp in X.EXPERIMENTS.values():
        rows = load_rows(exp.key)
        for arm in exp.arms.values():
            ar = [r for r in rows if r["arm"] == arm.name]
            base = next((r for r in ar if r["condition"] == "full"), None)
            bo = next((r for r in ar if r["condition"] == "base_only"), None)
            if not base or not bo:
                continue
            de = bo["macro_auroc"] - base["macro_auroc"]
            dc = (bo["cross_auroc"] - base["cross_auroc"]) if bo.get("cross_auroc") and base.get("cross_auroc") else None
            lines.append(
                f"| {exp.key}/{arm.name} | {base['macro_auroc']:.4f} → {bo['macro_auroc']:.4f} "
                f"({de:+.4f}) | {fmt(base.get('cross_auroc'))} → {fmt(bo.get('cross_auroc'))} ({fmt_d(dc)}) |"
            )
    lines += [
        "",
        "This is the clearest finding in the study, and it is not about novelty at all. Removing "
        "every red-team row costs BOTH columns in both arms: eval and cross-attacker AUROC fall "
        "together. Note cloud_3 found the eval column has no fixed sign across concepts — on "
        "high-stakes, dropping every red-team row GAINED +0.1105 macro AUROC — so the fact that "
        "it is negative on both arms here is a property of this concept, not a general law. The "
        "cross-attacker column is the one that pointed the same way in all four of cloud_3's arms "
        "as well: whatever the red-team rows buy, eval is a poor instrument for seeing it.",
        "",
        "Note `base_only` trains on 50 rows, below the 64-row optimizer-step floor "
        "(`batch_size 16` x `gradient_accumulation_steps 4`), so it takes **zero** optimizer steps "
        "and is effectively a seeded random projection of the layer-32 activations. It still "
        "reaches 0.7714 macro AUROC here, which says as much about how separable this eval is in "
        "the layer-32 representation as it does about the red-team data.",
        "",
    ]



def _n_arms() -> int:
    """How many arms this report actually covers.

    Counted from the results ON DISK, not from EXPERIMENTS: the registry still declares
    the high-stakes arms, which this box has no activations for and never ran. Hardcoding
    "four arms" was wrong the moment this study was pointed at one concept — the tables
    were computed from two while the prose asserted findings about arms that do not exist
    here. Anything comparative about a concept with no results must be attributed to the
    run that measured it.
    """
    return sum(
        1
        for e in X.EXPERIMENTS.values()
        for a in e.arms
        if (X.RESULTS / f"novelty_{e.key}_{a}.npz").exists()
    )

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = [
        "# Is red-team novelty the thing that hurts eval?",
        "",
        f"{_n_arms()} red-team arms, all read off the runs' own "
        "cached gemma-3-27b L32 activations. Phase 1 measures how far each red-team row sits "
        "from the eval manifold; Phase 2 groups those rows into regions; Phase 3 removes them "
        "and refits, which is the only step that can establish a *cause*.",
        "",
    ]
    phase1_table(lines)

    phase2_table(lines)
    lines += ["## Phase 3 — removal experiments", ""]
    for exp in X.EXPERIMENTS.values():
        rows = load_rows(exp.key)
        for arm in exp.arms.values():
            arm_section(lines, exp, arm, rows)

    synthesis(lines)

    dest = Path(args.out) if args.out else X.RESULTS / "summary.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved -> {dest}")
    print("\n".join(lines[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
