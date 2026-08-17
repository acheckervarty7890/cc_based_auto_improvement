"""Render ``metric_geometry.json`` as the markdown tables the write-up quotes.

Kept separate from the sweep so the doc's tables can be regenerated without refitting
anything, and so the verdict rule is written down once, in code, rather than applied by
eye in prose.

A metric has to clear three independent bars, and they are independent on purpose — the
two loop fixes need different things from it:

``label``        does similarity track the label at all, on rows nothing was fit on?
                 ``pair_new.auroc >= 0.60`` in **both** arms. Below that the metric is
                 measuring something other than the concept and nothing downstream is
                 trustworthy.
``scenario``     does it see *what the conversation is about*, not just which side of the
                 boundary it falls? ``scenario.auroc >= 0.75`` — a source's own
                 counterpart must be closer than an unrelated row. This is what a guard
                 needs to recognise a re-skin at all, and it is what disqualifies the 1-D
                 metrics: along a single direction two rows can be adjacent while sharing
                 no content, so ``probe:logit`` and ``sup:lda`` clear the hop bar for a
                 reason that would not survive contact with an actual clone.
``guard``        requires ``label`` AND ``scenario`` AND ``hop_paired.auroc <= 0.25`` —
                 an opposite-label counterpart must land FARTHER from its source than the
                 source's nearest v2 row. At 0.5 the two are indistinguishable, which is
                 the finding that rules out raw cosine; above 0.5 the metric would reject
                 label flips as clones, which is worse than useless. All three are needed
                 because the guard must be close on same-scenario-same-label and far on
                 same-scenario-opposite-label, and no single one of these bars implies it.
``acquisition``  does distance-to-training-set flag the successes every reseeded v2 probe
                 misses? ``durable.auroc >= 0.65`` with a bootstrap CI clear of 0.5.

``deployable`` is separate again: a transductive embedding (t-SNE) has no out-of-sample
transform, so using it as a submit-time guard means re-embedding the whole corpus per
candidate. It can pass every bar above and still not be usable.

Usage:
    .venv_claude/bin/python scripts/metric_geometry_report.py
    .venv_claude/bin/python scripts/metric_geometry_report.py --out docs/metric_tables.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IN_JSON = Path("results_hu_harm_gemma27b_batch_ablation/vintage/metric_geometry.json")
ARMS = ("gptoss120b", "deepseekv4pro")

LABEL_BAR = 0.60
SCENARIO_BAR = 0.75
GUARD_BAR = 0.25
ACQ_BAR = 0.65


def verdict(per_arm: list[dict]) -> str:
    """Every bar must be cleared in *both* arms — a metric that works in one is noise."""
    label = all(m["pair_new"]["auroc"] >= LABEL_BAR for m in per_arm)
    scenario = all(m["scenario"]["auroc"] >= SCENARIO_BAR for m in per_arm)
    hop = all(m.get("hop_paired", m["hop"])["auroc"] <= GUARD_BAR for m in per_arm)
    acq = all(
        m["durable"]["auroc"] >= ACQ_BAR and m["durable"]["ci95"][0] > 0.5
        for m in per_arm
    )
    tags = []
    if label:
        tags.append("label")
    if scenario:
        tags.append("scenario")
    if label and scenario and hop:
        tags.append("**guard**")
    if acq:
        tags.append("**acquisition**")
    if any(m.get("transductive") for m in per_arm):
        tags.append("(transductive — not deployable)")
    return " + ".join(tags) if tags else "—"


def fmt(v, spec=".3f") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(IN_JSON))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rep = json.loads(Path(args.json).read_text())
    arms = [a for a in ARMS if a in rep]
    names = list(rep[arms[0]]["metrics"])
    lines: list[str] = []

    lines.append("## Sizes\n")
    lines.append("| arm | rows | v2 | new-in-v3 | successes | durable | couples |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in arms:
        n = rep[a]["n"]
        lines.append(
            f"| {a} | {n['rows']} | {n['v2']} | {n['new']} | {n['successes']} "
            f"| {n['durable']} | {n['couples']} |"
        )

    lines.append("\n## Acceptance test\n")
    lines.append(
        "`pairAUR` = AUROC of similarity predicting *same label*, on the new-in-v3 rows "
        "(held out from every fit). `scenAUR` = AUROC separating a source's own "
        "counterpart from an unrelated row — whether the metric sees content at all. "
        "`provAUR` = the same test for *authorship*, the doc's design smell. `hopPair` = "
        "AUROC separating a source's own opposite-label counterpart from that same "
        "source's nearest v2 row; **low is good** (the counterpart must look farther). "
        "`kNN15aur` is AUROC over the fraction of positive neighbours, because the "
        "success sets are 71% / 83% positive rather than class-balanced. `durAUR` = "
        "AUROC of distance-to-v2 predicting a durable hole.\n"
    )
    for a in arms:
        lines.append(f"\n### {a}\n")
        lines.append(
            "| metric | dim | pairAUR | scenAUR | provAUR | hopPair | nnBal | kNN15aur "
            "| durAUR (CI95) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for nm in names:
            m = rep[a]["metrics"][nm]
            d = m["durable"]
            lines.append(
                f"| `{nm}` | {m['dim']} | {fmt(m['pair_new']['auroc'])} "
                f"| {fmt(m['scenario']['auroc'])} "
                f"| {fmt(m['pair_provenance']['auroc'])} "
                f"| {fmt(m.get('hop_paired', m['hop'])['auroc'])} "
                f"| {fmt(m['nn']['balanced'])} "
                f"| {fmt(m['knn']['auroc_15'])} "
                f"| {fmt(d['auroc'])} ({fmt(d['ci95'][0], '.2f')}–{fmt(d['ci95'][1], '.2f')}) |"
            )

    lines.append("\n## Verdicts (both arms must clear each bar)\n")
    lines.append(
        f"`label` pairAUR ≥ {LABEL_BAR}; `guard` hopPair ≤ {GUARD_BAR}; "
        f"`acquisition` durAUR ≥ {ACQ_BAR} with CI clear of 0.5.\n"
    )
    lines.append("| metric | verdict |")
    lines.append("|---|---|")
    for nm in names:
        lines.append(f"| `{nm}` | {verdict([rep[a]['metrics'][nm] for a in arms])} |")

    lines.append("\n## §5a's own columns, for continuity with the published note\n")
    lines.append(
        "Raw similarity means. These are *scale-dependent* — `nl:expstretch` moves them "
        "while changing nothing else — so they are here to line up against the note, not "
        "to be compared across metrics.\n"
    )
    for a in arms:
        lines.append(f"\n### {a}\n")
        lines.append(
            "| metric | same-label | opp-label | Δ | own counterpart | new→v2 NN "
            "| frac own closer | kNN 1/5/15 (raw acc) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for nm in names:
            m = rep[a]["metrics"][nm]
            p, h = m["pair_v2"], m["hop"]
            hp = m.get("hop_paired", {})
            k = m["knn"]
            lines.append(
                f"| `{nm}` | {fmt(p['same_label'], '.4f')} | {fmt(p['opp_label'], '.4f')} "
                f"| {fmt(p['delta'], '.4f')} | {fmt(h['own_pair_sim'], '.4f')} "
                f"| {fmt(h['new_to_v2_nn_sim'], '.4f')} "
                f"| {fmt(hp.get('frac_own_closer'), '.2f')} "
                f"| {k['1']:.1%} / {k['5']:.1%} / {k['15']:.1%} |"
            )

    lines.append("\n## Class balance of the k-NN test set\n")
    lines.append("| arm | positive rate (successes) | majority-class baseline |")
    lines.append("|---|---|---|")
    for a in arms:
        k = rep[a]["metrics"][names[0]]["knn"]
        lines.append(
            f"| {a} | {k['positive_rate_test']:.1%} | {k['majority_baseline']:.1%} |"
        )

    lines.append("\n## Acquisition baselines that need no metric\n")
    lines.append("| arm | predictor | AUROC (CI95) |")
    lines.append("|---|---|---|")
    for a in arms:
        for nm, v in rep[a]["reference_predictors"].items():
            lines.append(
                f"| {a} | `{nm}` | {fmt(v['auroc'])} "
                f"({fmt(v['ci95'][0], '.2f')}–{fmt(v['ci95'][1], '.2f')}) |"
            )

    # eval proximity: §5's question re-asked
    lines.append("\n## Does the new vintage move the training set toward eval?\n")
    lines.append(
        "Size-matched nearest-neighbour similarity from each eval row to the 116/86 "
        "new-in-v3 successes, versus to an equal-sized random draw from v2 (20 draws). "
        "§5 asked this in raw cosine; a metric that passed the label bar asking it again "
        "is the point.\n"
    )
    lines.append("| arm | metric | split | eval→new-v3 | eval→v2 (size-matched) | Δ |")
    lines.append("|---|---|---|---|---|---|")
    for a in arms:
        for nm in names:
            ev = rep[a]["metrics"][nm].get("eval_proximity")
            if not ev:
                continue
            for split, v in ev.items():
                dlt = v["eval_to_new_v3"] - v["eval_to_v2_sizematched"]
                lines.append(
                    f"| {a} | `{nm}` | {split.replace('eval_','')} "
                    f"| {fmt(v['eval_to_new_v3'], '.4f')} "
                    f"| {fmt(v['eval_to_v2_sizematched'], '.4f')} | {dlt:+.4f} |"
                )

    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
