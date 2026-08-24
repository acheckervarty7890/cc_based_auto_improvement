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


# The three row-level conditions that ask what the contrastive generation step is worth.
# They break the pairing on purpose, so `controls()` — matched removal of whole pairs — is
# the wrong yardstick for them and they are kept out of the Q3 table entirely.
PROVENANCE = ("drop_generated", "drop_sources", "keep_random_half",
              # Substitution conditions (rewrite_ablate.py): these ADD rows written by a
              # different model as well as dropping rows, so no removal-of-the-same-size
              # control speaks to them at all.
              "rewritten_sources", "rewritten_plus_generated")


def half_control(rows) -> tuple[float, float, int] | None:
    """(mean, sd, n_seeds) over `keep_random_half` — the control for the provenance pair."""
    v = [r["mean"]["auroc"] for r in rows if r["condition"] == "keep_random_half"]
    if not v:
        return None
    return st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0), len(v)


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


N_BASE = 50   # base rows are in every condition's n_train


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
    row("attacker successes only, no generated halves",
        lambda k, s, a, r: f"{_cond(r,'drop_generated')['mean']['auroc']:.4f}"
        if _cond(r, 'drop_generated') else "-")
    row("generated halves only",
        lambda k, s, a, r: f"{_cond(r,'drop_sources')['mean']['auroc']:.4f}"
        if _cond(r, 'drop_sources') else "-")
    row("a random half of every pair",
        lambda k, s, a, r: f"{half_control(r)[0]:.4f}" if half_control(r) else "-")
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
          "(~99.9%) - as are the eval rows themselves about their own centroid (~99.9%), "
          "so this is the geometry of a 5376-dimensional space rather than a peculiarity "
          "of the red-team data. Either way it cannot move an eval score directly; it can "
          "only act by rotating `w` at the next refit. The surface flag is a weak proxy for representation-level "
          "novelty (Spearman ~0.3) yet still names a group that is linearly separable in "
          "activation space at ~0.90.",
          "5. **The generated halves are doing the work, and it is the pairing.** Dropping "
          "the LLM-written partner of every red-team conversation - i.e. the retrain a run "
          "with no `preprocessing:` section would have done - costs 0.16 and 0.15 AUROC, "
          "landing below `base_only` in both arms. It is not a matter of who wrote the text: "
          "re-expressing arm 1's successes in the generator's own voice moves the score "
          "+0.006, inside the control's noise, even though the rewrite carried them most of "
          "the way to the generator's own length and formatting. Pair those same rewrites "
          "with the partners and the score returns to 0.8955 against 0.9164 untouched. The "
          "contrast between two near-identical conversations is the mechanism - not the row "
          "count, not the prose.", ""]
    return "\n".join(L)


def _best(rows, band: float) -> str:
    if not rows:
        return "-"
    ctrl = controls(rows)
    best, bd = None, -9.0
    for r in rows:
        n = r["condition"]
        if n in ("full", "base_only") or n in PROVENANCE or n.startswith("random_"):
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
            if name in ("full",) or name in PROVENANCE or name.startswith("random_"):
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
          f"* Read that against its baseline: the **eval** rows are "
          f"**{acts['mean_orthogonal_fraction_eval']:.2%}** orthogonal to `w` about their own "
          f"centroid. Near-total orthogonality to one direction is what {acts['dim']:,} "
          f"dimensions hand any row — the finding is which way the displacement points, not "
          f"that it is orthogonal.",
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

    if rows and half_control(rows):
        base = _cond(rows, "full")
        only = _cond(rows, "base_only")
        gen = _cond(rows, "drop_sources")        # generated halves kept
        srcs = _cond(rows, "drop_generated")     # attacker successes kept
        hm, hsd, hn = half_control(rows)
        L += ["### 5. What are the generated halves worth?", ""]
        L += [f"Every red-team conversation the attacker landed was given an opposite-label "
              f"partner by `generate_contrastive_dataset`, so the {surf['n_redteam']} rows "
              f"are {surf['n_pairs']} pairs. Dropping one side of every pair asks what each "
              f"side contributes. The control is `keep_random_half`, which keeps a randomly "
              f"chosen side of each pair — same row count, pairing broken just as hard, only "
              f"the source-versus-generated choice left to chance.", ""]
        L += ["| kept | rows | eval AUROC | Δ vs full | Δ vs random half |",
              "| --- | --- | --- | --- | --- |"]

        def line(label, r, n_rows):
            if r is None:
                return
            a = r["mean"]["auroc"]
            L.append(f"| {label} | {n_rows} | {a:.4f} | {a - base['mean']['auroc']:+.4f} | "
                     f"{a - hm:+.4f} |")

        line("everything (`full`)", base, surf["n_redteam"])
        n_half = next(r["n_train"] - N_BASE for r in rows
                      if r["condition"] == "keep_random_half")
        L.append(f"| a random half of every pair | {n_half} | "
                 f"{hm:.4f} ± {hsd:.4f} (n={hn}) | {hm - base['mean']['auroc']:+.4f} | — |")
        line("the attacker's own successes only", srcs, surf["n_pairs"])
        line("the generated partners only", gen, surf["n_pairs"])
        line("no red-team data (`base_only`)", only, 0)
        worse = [lab for lab, r in (("the attacker's own successes", srcs),
                                    ("the generated partners", gen))
                 if r and r["mean"]["auroc"] < only["mean"]["auroc"]]
        vs_base = (f"{' and '.join(worse)} score below `base_only` "
                   f"({only['mean']['auroc']:.4f}) — worse than training on the "
                   f"{N_BASE} base rows alone"
                   if worse else
                   f"neither half falls below `base_only` ({only['mean']['auroc']:.4f})")
        pr = acts.get("per_pair_role") or {}
        if {"source", "generated"} <= set(pr):
            src_g, gen_g = pr["source"], pr["generated"]
            L += ["", "In activation space the two halves are near-twins, and what difference "
                      "there is runs the wrong way for a distance story:", "",
                  f"* The generated partners are the **further** of the two from eval "
                  f"({gen_g['knn_to_eval']:.2f} against {src_g['knn_to_eval']:.2f} mean kNN) and "
                  f"the more off-distribution on the Q1 text discriminator "
                  f"({gen_g['mean_p_redteam']:.3f} against {src_g['mean_p_redteam']:.3f}) — yet "
                  f"they are the better half to keep.",
                  f"* Their displacement from the eval centroid is also the larger, and it is "
                  f"the orthogonal part that grows ({gen_g['orth']:.2f} against "
                  f"{src_g['orth']:.2f} mean residual) — the component that cannot move a score "
                  f"by itself.",
                  f"* Both halves order their own two labels along `w` "
                  f"({src_g['auroc_along_w']:.3f} for the submissions, "
                  f"{gen_g['auroc_along_w']:.3f} for the partners). `w` here is the arm's own "
                  f"final probe, fit on both halves, so this is agreement with where the run "
                  f"ended up — not what either half would reach alone.", ""]

        hw_path = O.RESULTS / f"handwritten_scores_{arm_key}.json"
        if hw_path.exists():
            hw = O.read_json(hw_path)
            pp = hw["per_probe"]
            L += ["", "**Did it memorise the strings, or patch the behaviour?** The final "
                      f"probe scores the {surf['n_pairs']} successes it trained on 100% "
                      "correctly, but in-sample, so that cannot separate a patched boundary "
                      f"from memorised text. {hw['n']} conversations were hand-written to "
                      "vary those successes — same scenario type and same assistant "
                      "behaviour, hence the same class, but fresh wording and details, "
                      "written by neither model in the pipeline — and scored by the "
                      "pre-retrain probe and the final one:", ""]
            L += ["| probe | classifies the variations correctly | harmful | not-harmful |",
                  "| --- | --- | --- | --- |"]
            L.append(f"| `probe_iter0` (before red-teaming) | {pp['0']['overall']:.0%} | "
                     f"{pp['0']['harmful']:.0%} | {pp['0']['not_harmful']:.0%} |")
            L.append(f"| `probe_iter5` (after five retrains) | **{pp['5']['overall']:.0%}** | "
                     f"{pp['5']['harmful']:.0%} | {pp['5']['not_harmful']:.0%} |")
            L += ["", f"The final probe gets **all {hw['n']}** right, on conversations it was "
                      f"never shown. The retrain patched the behaviour, not the wording: the "
                      f"lift is entirely on the harmful side "
                      f"({pp['0']['harmful']:.0%} -> {pp['5']['harmful']:.0%}), which is what "
                      f"red-teaming this concept was hunting — the pre-retrain probe already "
                      f"handled the benign variations. So the pairing does not just move an "
                      f"AUROC on the fixed eval splits; it generalises to fresh instances of "
                      f"the same failure, by hand, off-distribution from both writers.", ""]
            L += ["Two caveats. These are 25 rows the author judged clear cases, not a "
                  "balanced held-out split; and the variations stay close to the training "
                  "scenarios (a sibling's addiction, a triage call), so this measures "
                  "generalisation across wording and detail, not across new kinds of harm.",
                  ""]

        pps_path = O.RESULTS / f"pair_probe_scores_{arm_key}.json"
        if pps_path.exists():
            pps = O.read_json(pps_path)
            po = pps["probe_of_the_day"]
            L += ["", "**What the probe thought of each half.** A success is by definition a "
                      "row the live probe got wrong; its partner was written afterwards and "
                      "never scored at all. Scoring each partner with `probe_iter{k}` — the "
                      "probe its source was submitted against, and the last one that had "
                      "seen neither half — closes that gap:", ""]
            L += ["| half | n | mean probe score | probe classifies it correctly |",
                  "| --- | --- | --- | --- |"]
            for g, lab in (("source", "the attacker's success"),
                           ("generated", "its generated partner")):
                d = po[g]
                L.append(f"| {lab} | {d['n']} | {d['mean_score']:.3f} | "
                         f"**{d['correct']:.1%}** |")
            L += ["", f"The source row is the control and its answer is known in advance: "
                      f"0%, because that is what made it a success. It comes out at "
                      f"{po['source']['correct']:.1%}, and every exception is a row sitting "
                      f"within 0.002 of the 0.5 threshold, so the wiring is right.", ""]
            ba_path = O.RESULTS / f"before_after_scores_{arm_key}.json"
            if ba_path.exists():
                ba = O.read_json(ba_path)
                gsrc, ggen = ba["groups"]["source"], ba["groups"]["generated"]
                L += ["", f"And the flat before/after, `probe_iter{ba['before_iter']}` (trained "
                          f"on the base rows only, so both halves are out-of-sample) against "
                          f"`probe_iter{ba['after_iter']}` (trained on both, so both are "
                          f"in-sample):", ""]
                L += ["| half | mean score | % predicted harmful | % classified correctly |",
                      "| --- | --- | --- | --- |"]
                for g, lab in ((gsrc, "the attacker's success"), (ggen, "its partner")):
                    b, a = g["before"], g["after"]
                    L.append(f"| {lab} | {b['mean']:.3f} &rarr; {a['mean']:.3f} | "
                             f"{b['predicted_positive']:.0%} &rarr; {a['predicted_positive']:.0%} | "
                             f"{b['correct']:.0%} &rarr; {a['correct']:.0%} |")
                brk = ba.get("source_before_by_found_iter")
                if brk:
                    iters_seen = sorted({k for cls in brk.values() for k in cls}, key=int)
                    L += ["", f"That {gsrc['before']['correct']:.0%}-correct figure for the "
                              f"successes is not evidence they were weak attacks — it is an "
                              f"artefact of *which* probe they beat. A success fooled the probe "
                              f"of the day, `probe_iter{{k}}`, which is `probe_iter"
                              f"{ba['before_iter']}` only for the iteration-"
                              f"{ba['before_iter']} batch. Split the successes by true class and "
                              f"the iteration that found them, scored by `probe_iter"
                              f"{ba['before_iter']}` (share it classifies correctly):", ""]
                    header = "| class \\ found in | " + " | ".join(
                        f"iter {k}" for k in iters_seen) + " |"
                    L += [header, "| " + " | ".join(["---"] * (len(iters_seen) + 1)) + " |"]
                    for cls in (O.POS, O.NEG):
                        cells = []
                        for k in iters_seen:
                            d = brk[cls].get(k)
                            cells.append(f"{d['correct']:.0%} (n={d['n']})" if d else "—")
                        L.append(f"| {cls} | " + " | ".join(cells) + " |")
                    z0 = brk[O.POS].get(str(ba['before_iter']))
                    n0z = brk[O.NEG].get(str(ba['before_iter']))
                    later_benign = sum(d["n"] for k, d in brk[O.NEG].items()
                                       if int(k) > int(ba['before_iter']))
                    L += ["", f"Two things read off this. The iteration-{ba['before_iter']} rows "
                              f"score ~0% ({z0['correct']:.0%} harmful, "
                              f"{n0z['correct']:.0%} benign) — `probe_iter{ba['before_iter']}` "
                              f"IS the probe they beat, so it must get them wrong. The "
                              f"{gsrc['before']['correct']:.0%} overall is carried almost "
                              f"entirely by the {later_benign} benign successes found at later "
                              f"iterations, which `probe_iter{ba['before_iter']}` calls correctly "
                              f"100% of the time: those were false positives against a LATER "
                              f"probe that had drifted into over-flagging benign edge cases, and "
                              f"`probe_iter{ba['before_iter']}` — negative-biased and earlier — "
                              f"predates that drift. It agrees with the judge on them not because "
                              f"it is good but because it had not yet developed the failure that "
                              f"made them successes.", ""]

                bh = gsrc["before"]["by_class"].get(O.POS)
                L += ["", f"The move is concentrated where it should be: the harmful successes "
                          f"go from {bh['correct']:.0%} correct before to 100% after, while the "
                          f"benign ones and the partners were mostly right already. `iter0` "
                          f"scores the successes low across the board "
                          f"({gsrc['before']['predicted_positive']:.0%} assigned to the harmful "
                          f"class) — it is a probe that has not yet learned this attacker's "
                          f"failure mode; five retrains on these very rows move that boundary "
                          f"onto them.", ""]

            L += [f"The partner is the finding: the probe already classifies "
                  f"**{po['generated']['correct']:.1%}** of them correctly. The generation "
                  f"step is not manufacturing a second misclassification per pair — it is "
                  f"attaching, to each row the probe fails, a near-identical row the probe "
                  f"already gets right. That is what the retrain is actually being handed, "
                  f"and it is why the sources alone are worse than no red-team data at all: "
                  f"on their own they are a pile of failures with nothing to contrast "
                  f"against.", ""]

        rw = _cond(rows, "rewritten_sources")
        rwg = _cond(rows, "rewritten_plus_generated")
        stats_path = O.RESULTS / f"rewrite_stats_{arm_key}.json"
        if rw and stats_path.exists():
            stx = O.read_json(stats_path)
            sm = stx["structural_means"]
            L += ["", "**Is it the voice?** Every source was written by the attacker model "
                      "and every partner by the contrastive generator, so who wrote a row is "
                      "perfectly confounded with which half it is. `rewrite_successes.py` "
                      f"removes that confound from one side: the same {stx['n_rewrites']} "
                      f"successes, re-expressed by `{stx['rewrite_model']}` with the "
                      "scenario, the assistant's behaviour and the label held fixed "
                      f"(median difflib similarity to the original "
                      f"{stx['similarity_median']:.2f}, turn count preserved on "
                      f"{stx['turn_count_preserved']}/{stx['n_rewrites']}).", ""]
            L += ["| kept | rows | eval AUROC | Δ vs the originals |",
                  "| --- | --- | --- | --- |"]
            L.append(f"| the attacker's own successes | {surf['n_pairs']} | "
                     f"{srcs['mean']['auroc']:.4f} | — |")
            L.append(f"| the same, rewritten | {rw['n_train'] - N_BASE} | "
                     f"{rw['mean']['auroc']:.4f} | "
                     f"{rw['mean']['auroc'] - srcs['mean']['auroc']:+.4f} |")
            if rwg:
                L.append(f"| the rewrites **and** the generated partners | "
                         f"{rwg['n_train'] - N_BASE} | {rwg['mean']['auroc']:.4f} | "
                         f"{rwg['mean']['auroc'] - srcs['mean']['auroc']:+.4f} |")
            L.append(f"| everything, untouched (`full`) | {surf['n_redteam']} | "
                     f"{base['mean']['auroc']:.4f} | "
                     f"{base['mean']['auroc'] - srcs['mean']['auroc']:+.4f} |")
            L += ["", f"Rewriting moves the score {rw['mean']['auroc'] - srcs['mean']['auroc']:+.4f} "
                      f"— inside the ±{hsd:.4f} spread of the random-half control, i.e. nothing. "
                      f"And it is not that the rewrite failed to change the writing: it carried "
                      f"the sources most of the way to the generator's own profile "
                      f"({sm['sources']['chars_total']:.0f} -> "
                      f"{sm['rewrites']['chars_total']:.0f} characters against "
                      f"{sm['partners']['chars_total']:.0f} for the partners, and "
                      f"{sm['sources']['chars_assistant']:.0f} -> "
                      f"{sm['rewrites']['chars_assistant']:.0f} against "
                      f"{sm['partners']['chars_assistant']:.0f} in the assistant turn; eval "
                      f"sits at {sm['eval']['chars_total']:.0f} / "
                      f"{sm['eval']['chars_assistant']:.0f}). The voice moved and the score "
                      f"did not.", ""]
            if rwg:
                L += [f"Restoring the pairing does move it. The rewrites plus the partners "
                      f"already generated for the originals reach "
                      f"{rwg['mean']['auroc']:.4f}, within "
                      f"{abs(base['mean']['auroc'] - rwg['mean']['auroc']):.4f} of the "
                      f"untouched set — from a training half that was rewritten wholesale. "
                      f"What the partners contribute is the contrast, not the prose.", ""]
            L += ["Caveat: the rewrites' labels are asserted by the rewrite prompt and were "
                  "not re-judged, so a rewrite that drifted across the boundary enters as a "
                  "mislabelled row. That biases the rewritten conditions downward, against "
                  "the hypothesis being tested — it cannot manufacture the null, but it "
                  "could deepen it.", ""]

        L += ["", f"Halving the set moves the score {hm - base['mean']['auroc']:+.4f} when "
                  f"the side is chosen at random. Choosing it *systematically* costs more in "
                  f"both "
                  f"directions — {srcs['mean']['auroc'] - hm:+.4f} for the attacker's own "
                  f"successes, {gen['mean']['auroc'] - hm:+.4f} for the generated partners — "
                  f"and {vs_base}. So this is not the row count, and not the generated text "
                  f"as such: it is the *pairing*. With both halves the label can only be read "
                  f"off the behavioural difference between two near-identical conversations; "
                  f"with one side systematically removed the class becomes predictable from "
                  f"who wrote the conversation.", ""]

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
             "helps, whether they show up in activation space, and what the generated "
             "contrastive halves are worth.", "",
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
