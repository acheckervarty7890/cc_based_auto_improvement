#!/usr/bin/env python
"""Turn the three phases into `results/SUMMARY.md`, one section per question asked.

Every removal condition is reported against **matched random removal of the same number of
rows**, three seeds. The random spread is the noise band: a condition only counts as an
effect if it lands outside `mean ± band_sd` of its own control, and conditions with no
control of that size are marked rather than compared to a different size.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import od_common as O  # noqa: E402


def load_ablation(arm: str) -> list[dict]:
    p = O.RESULTS / f"ablation_{arm}.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    seen, out = set(), []
    for r in reversed(rows):                       # a rerun appends; keep the newest
        k = (r["condition"], r["seed"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return list(reversed(out))


def controls(rows) -> dict[int, tuple[float, float, int]]:
    """removal size -> (mean, sd, n_seeds) over the matched random seeds.

    Keyed on the size the control was ASKED for (which is in its condition name), not the
    size it achieved. `matched_random` removes whole pairs until it has removed at least the
    target, so a seed can overshoot by one pair and land on 275 where another landed on 276.
    Keying on the achieved count scatters one 3-seed control across three 1-seed buckets,
    every one of which then reports a standard deviation of zero — a noise band of zero,
    which would mark essentially every condition as an effect.
    """
    byn: dict[int, list[float]] = {}
    for r in rows:
        if r["condition"].startswith("random_"):
            byn.setdefault(int(r["condition"].split("_")[1]), []).append(r["mean"]["auroc"])
    return {n: (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0, len(v))
            for n, v in byn.items()}


def _cond(rows, name):
    return next((r for r in rows if r["condition"] == name), None)


def headline(loaded, band: float) -> str:
    """The four answers, in one table each, computed from what is on disk."""
    L = ["## Answers", ""]
    L += ["| | " + " | ".join(k for k, *_ in loaded) + " |",
          "| --- | " + " | ".join("---" for _ in loaded) + " |"]
    def row(label, fn):
        L.append(f"| {label} | " + " | ".join(fn(*x) for x in loaded) + " |")
    row("red-team vs eval, text discriminator AUROC",
        lambda k, s, a, r: f"{s['discriminator_auroc']:.4f}")
    row("red-team vs eval, activation-space AUROC",
        lambda k, s, a, r: f"{a['separability_redteam_vs_eval_auroc']:.4f}")
    row("rows past eval's own p95 self-kNN radius",
        lambda k, s, a, r: f"{a['outside_frac']:.1%}")
    row("refusal rate (eval: {:.1%})".format(loaded[0][1]["eval_convention"]["refusal_rate"]),
        lambda k, s, a, r: f"{s['refusal_rate_redteam']:.1%}")
    row("rows labelled opposite to the eval convention",
        lambda k, s, a, r: str(s["n_convention_inverted"]))
    row("pairs contrasting on the assistant turn",
        lambda k, s, a, r: f"{s['pair_axis_counts']['assistant']}/{s['n_pairs']}")
    row("displacement orthogonal to the probe direction",
        lambda k, s, a, r: f"{a['mean_orthogonal_fraction']:.1%}")
    row("full (all red-team data)",
        lambda k, s, a, r: f"{_cond(r,'full')['mean']['auroc']:.4f}" if _cond(r,'full') else "-")
    row("base_only (no red-team data)",
        lambda k, s, a, r: f"{_cond(r,'base_only')['mean']['auroc']:.4f}" if _cond(r,'base_only') else "-")
    row("best single removal, vs its matched random",
        lambda k, s, a, r: _best(r, band))
    L += ["",
          "1. **Not \"some portion\" — effectively all of it.** Both corpora separate at "
          "AUROC >= 0.998 on text alone and >= 0.9999 in activation space. There is no "
          "eval-like subset of the red-team data to keep; there is only a gradient.",
          "2. **Nothing is labelled backwards, and no pair contrasts on the wrong turn.** "
          "What is off is the *mix*: refusal, which is a large part of what the eval "
          "negative class is made of, is 5-13x rarer in red-team data.",
          "3. **Removing the most off-distribution rows does not reliably help.** It clears "
          "the matched-random band in some conditions and loses in others, and the "
          "most-eval-like removal does about as well - so surface distance from eval is not "
          "the axis that predicts harm. The one consistent effect points the other way: "
          "dropping the 30% with the LONGEST assistant replies - the rows least like eval "
          "structurally - is significantly WORSE than random in both arms.",
          "4. **The novelty is real but almost entirely orthogonal to the decision axis** "
          "(~99.9%), so it cannot move an eval score directly; it can only act by rotating "
          "`w` at the next refit. The surface flag is a weak proxy for representation-level "
          "novelty (Spearman ~0.3) yet still names a group that is linearly separable in "
          "activation space at ~0.90.", ""]
    return "\n".join(L)


def _best(rows, band: float) -> str:
    if not rows:
        return "-"
    ctrl = controls(rows)
    best, bd = None, -9.0
    for r in rows:
        n = r["condition"]
        if n in ("full", "base_only") or n.startswith("random_"):
            continue
        c = ctrl.get(r["n_removed"])
        if not c:
            continue
        d = r["mean"]["auroc"] - c[0]
        if d > bd:
            best, bd = r, d
    return f"`{best['condition']}` {bd:+.4f}" if best else "-"


def section(arm_key: str, surf: dict, acts: dict, rows: list[dict], band: float) -> str:
    arm = O.ARMS[arm_key]
    L = [f"## {arm_key} — attacker `{arm.attacker}`", ""]
    conv = surf["eval_convention"]

    L += ["### 1. How different are the red-team rows from the eval rows?", ""]
    L += [f"A TF-IDF discriminator separates the two corpora **out of fold at AUROC "
          f"{surf['discriminator_auroc']:.4f}**, and in activation space a linear model does "
          f"it at **{acts['separability_redteam_vs_eval_auroc']:.4f}**. So the honest answer "
          f"to \"can you identify *some portion* that is different\" is that essentially "
          f"**all of it is**: the median red-team row scores "
          f"{surf['p_redteam_quantiles']['0.5']:.3f} on the discriminator against "
          f"{surf['p_redteam_eval_median']:.3f} for the median eval row, and the two "
          f"distributions barely touch. What varies across rows is the *degree*, and that is "
          f"what the removal conditions below are ordered by.", ""]
    L += ["| structural feature | red-team | eval |", "| --- | --- | --- |"]
    for k in ("chars_total", "chars_assistant", "chars_user", "assistant_share",
              "n_newlines", "has_bullets", "has_numbered", "n_questions", "refuses"):
        v = surf["structural_means"][k]
        L.append(f"| {k} | {v['redteam']:.3f} | {v['eval']:.3f} |")
    L += ["", f"Terms the discriminator weights toward red-team: "
              f"{', '.join('`%s`' % t for t, _ in surf['top_terms_redteam'][:12])}.", "",
          f"Toward eval: {', '.join('`%s`' % t for t, _ in surf['top_terms_eval'][:12])}.", ""]

    L += ["### 2. Are any samples or pairs set up the opposite way to the eval pairs?", ""]
    L += [f"Measured on the eval splits, the labelling convention is unambiguous: an eval "
          f"conversation whose assistant refuses is labelled `{O.POS}` only "
          f"**{conv['p_harm_given_refuse']:.1%}** of the time, against "
          f"{conv['p_harm_given_no_refuse']:.1%} for one that does not "
          f"({conv['n_refuse']}/{conv['n']} eval rows refuse). A red-team row that refuses "
          f"*and* is labelled `{O.POS}` therefore runs that mapping backwards.", ""]
    L += [f"**{surf['n_convention_inverted']} such rows** in this arm. The pairs themselves "
          f"are not inverted either: {surf['pair_axis_counts']['assistant']} of "
          f"{surf['n_pairs']} contrast on the **assistant's** turn — the axis the eval pairs "
          f"contrast on — and {surf['pair_axis_counts']['user']} on the user's.", ""]
    L += [f"The real mismatch is one of *composition*, not direction: "
          f"**{surf['refusal_rate_redteam']:.1%}** of red-team rows contain a refusal against "
          f"**{conv['refusal_rate']:.1%}** of eval rows. Refusal is a large part of what the "
          f"eval negative class is made of, and the red-team negative class is almost never "
          f"made of it.", ""]

    if rows:
        base = next((r for r in rows if r["condition"] == "full"), None)
        ctrl = controls(rows)
        L += ["### 3. Does removing them improve eval AUROC?", ""]
        L += [f"Baseline (`full`, all red-team data): **{base['mean']['auroc']:.4f}** — the "
              f"ceiling study's N=0 point, reproduced.", ""]
        L += ["| condition | rows removed | eval AUROC | Δ vs full | matched random | "
              "Δ vs random | outside band? |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            name = r["condition"]
            if name in ("full",) or name.startswith("random_"):
                continue
            n = r["n_removed"]
            a = r["mean"]["auroc"]
            c = ctrl.get(n)
            if c is None:
                L.append(f"| `{name}` | {n} | {a:.4f} | {a-base['mean']['auroc']:+.4f} | "
                         f"— | — | no control |")
                continue
            m, sd, n_seeds = c
            lim = max(sd, band)
            verdict = "**yes**" if abs(a - m) > lim else "no"
            L.append(f"| `{name}` | {n} | {a:.4f} | {a-base['mean']['auroc']:+.4f} | "
                     f"{m:.4f} ± {sd:.4f} (n={n_seeds}) | {a-m:+.4f} | {verdict} |")
        L += ["", f"The band is `max(control sd, {band})`; a condition inside it removed "
                  f"*some* data, not *these* data.", ""]

    L += ["### 4. Do the flagged rows have an activation-space signature?", ""]
    L += [f"* **{acts['outside_frac']:.1%}** of red-team rows sit further from the eval set "
          f"than eval's own 95th-percentile self-kNN radius "
          f"(k={acts['k']}, threshold {acts['eval_self_knn_p95']:.2f}; median red-team "
          f"{acts['redteam_knn_median']:.2f} against median eval "
          f"{acts['eval_self_knn_median']:.2f}).",
          f"* That displacement is **{acts['mean_orthogonal_fraction']:.1%} orthogonal** to "
          f"the probe's decision direction `w` (10 ensemble members, pairwise cosine "
          f"{acts['ensemble_direction_agreement_cos']:.3f}). Novelty orthogonal to `w` cannot "
          f"move a score directly — it can only act by rotating `w` at the next refit.",
          f"* The surface score is a **weak** proxy for it: Spearman ρ between `p_redteam` and "
          f"kNN-to-eval is {acts['spearman_p_redteam_vs_knn']:+.3f}. Text-level oddness and "
          f"representation-level oddness are not the same ordering.",
          f"* But the flagged group is **real in the representation**: the top "
          f"{acts['flag_frac']:.0%} by `p_redteam` separate from the rest at out-of-fold "
          f"AUROC **{acts['separability_flagged_vs_rest_auroc']:.4f}**.", ""]
    L += ["| topic | n | top terms | kNN to eval | outside | mean p_redteam |",
          "| --- | --- | --- | --- | --- | --- |"]
    for t in surf["topics"]:
        c = str(t["cluster"])
        pt = acts["per_topic"][c]
        L.append(f"| {c} | {t['n']} | {', '.join(t['top_terms'][:6])} | "
                 f"{pt['knn_to_eval']:.2f} | {pt['outside_frac']:.0%} | "
                 f"{pt['mean_p_redteam']:.3f} |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(O.ARMS))
    ap.add_argument("--band", type=float, default=0.005,
                    help="floor on the noise band, in AUROC")
    args = ap.parse_args()
    parts = ["# Off-distribution red-team samples — results", "",
             "Which red-team rows are unlike the eval set, whether any are labelled the "
             "opposite way round from how eval labels that behaviour, whether removing them "
             "helps, and whether they show up in activation space.", "",
             "Fits are the ceiling analysis's: one `linear_then_softmax` head, seed 42, "
             "early-stopped on that study's reserved 25% dev slice. Removals are by "
             "contrastive **pair**, so the class balance never moves with the flag.", ""]
    loaded = []
    for key in args.arms:
        loaded.append((key, O.read_json(O.RESULTS / f"surface_{key}.json"),
                       O.read_json(O.RESULTS / f"actsig_{key}.json"), load_ablation(key)))
    parts.append(headline(loaded, args.band))
    for key, surf, acts, rows in loaded:
        parts.append(section(key, surf, acts, rows, args.band))
    (O.RESULTS / "SUMMARY.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", O.RESULTS / "SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
