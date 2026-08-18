#!/usr/bin/env python
"""Render ``cohort_novelty.json`` as a standalone report page.

Every number on the page is read out of the JSON — nothing is transcribed by hand,
which is the only way a report and its data cannot silently disagree.

Usage:
    .venv_claude/bin/python scripts/build_cohort_novelty_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Where each experiment's analysis JSON lives, and where its page is written. The
# high-stakes run predates the multi-experiment layout and keeps its original path.
SOURCES = {
    # The high-stakes JSON moved to the shared results dir with the other two, but its
    # PAGE keeps its original path: that path is the published artifact's identity, and
    # writing the same report somewhere else would strand the existing link.
    "hs": (
        REPO / "results_cohort_novelty/hs_cohort_novelty.json",
        REPO / "results_hs_gemma27b_batch_ablation/vintage/cohort_novelty_report.html",
    ),
    "hu_harm": (
        REPO / "results_cohort_novelty/hu_harm_cohort_novelty.json",
        REPO / "results_cohort_novelty/hu_harm_cohort_novelty_report.html",
    ),
    "instructions": (
        REPO / "results_cohort_novelty/instructions_cohort_novelty.json",
        REPO / "results_cohort_novelty/instructions_cohort_novelty_report.html",
    ),
}

# Provenance line per experiment: which branch and run the data came from.
PROVENANCE = {
    "hs": ("experiment9_cloud", "high-stakes &middot; gemma-3-27b L32"),
    "hu_harm": ("experiment11_cloud", "harmful-to-human &middot; gemma-3-27b L32"),
    "instructions": (
        "experiment_instruction_cloud_1",
        "instruction-following &middot; gemma-3-27b L32",
    ),
}

CSS = """
:root {
  color-scheme: light dark;
  --ground: #eef2f4;
  --surface: #ffffff;
  --surface-sunk: #f6f9fa;
  --ink: #14202a;
  --ink-soft: #3b4c58;
  --muted: #5f7280;
  --rule: #d8e2e7;
  --rule-strong: #b9c8d0;
  --accent: #0e6f78;
  --accent-soft: #d9ecee;
  --arm2: #9a5518;
  --caution: #7d5f10;
  --caution-soft: #f3ecd8;
  --ramp: 14, 111, 120;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0f1519;
    --surface: #172128;
    --surface-sunk: #131c22;
    --ink: #e6edf1;
    --ink-soft: #c2d0d8;
    --muted: #93a6b2;
    --rule: #26333b;
    --rule-strong: #384953;
    --accent: #47aeb8;
    --accent-soft: #1c343a;
    --arm2: #d08b4c;
    --caution: #d7b465;
    --caution-soft: #2a2517;
    --ramp: 71, 174, 184;
  }
}
:root[data-theme="dark"] {
  --ground: #0f1519;
  --surface: #172128;
  --surface-sunk: #131c22;
  --ink: #e6edf1;
  --ink-soft: #c2d0d8;
  --muted: #93a6b2;
  --rule: #26333b;
  --rule-strong: #384953;
  --accent: #47aeb8;
  --accent-soft: #1c343a;
  --arm2: #d08b4c;
  --caution: #d7b465;
  --caution-soft: #2a2517;
  --ramp: 71, 174, 184;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  font-size: 16px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}

.frame {
  max-width: 1080px;
  margin: 0 auto;
  padding: clamp(28px, 5vw, 72px) clamp(18px, 4vw, 40px) 96px;
  display: flex;
  flex-direction: column;
  gap: 44px;
}

.prose { max-width: 68ch; display: flex; flex-direction: column; gap: 16px; }
.prose p { margin: 0; color: var(--ink-soft); }
.prose p strong { color: var(--ink); font-weight: 640; }

h1, h2, h3 {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-weight: 600;
  text-wrap: balance;
  margin: 0;
  color: var(--ink);
}
h1 { font-size: clamp(30px, 4.4vw, 44px); line-height: 1.14; letter-spacing: -0.012em; }
h2 { font-size: clamp(21px, 2.6vw, 26px); line-height: 1.25; }
h3 { font-size: 17px; line-height: 1.35; }

.eyebrow {
  font-size: 11.5px;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 620;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

header.head { display: flex; flex-direction: column; gap: 14px; }
header.head .sub { max-width: 62ch; color: var(--muted); font-size: 15px; }

/* --- verdict ------------------------------------------------------------- */
.verdict {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--accent);
  border-radius: 3px;
  padding: clamp(20px, 3vw, 30px);
  display: flex;
  flex-direction: column;
  gap: 22px;
}
.verdict .lede {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: clamp(19px, 2.3vw, 23px);
  line-height: 1.42;
  margin: 0;
  text-wrap: pretty;
}
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1px; background: var(--rule); border: 1px solid var(--rule); }
.tile { background: var(--surface); padding: 16px 18px; display: flex; flex-direction: column; gap: 5px; }
.tile .num {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
  font-size: 27px;
  line-height: 1.05;
  color: var(--accent);
  font-weight: 600;
}
.tile .lab { font-size: 12.5px; color: var(--muted); line-height: 1.4; }

/* --- sections ------------------------------------------------------------ */
section { display: flex; flex-direction: column; gap: 18px; }
.sec-head { display: flex; align-items: baseline; gap: 12px; border-bottom: 1px solid var(--rule-strong); padding-bottom: 10px; }
.sec-num {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  color: var(--accent);
  font-weight: 640;
  padding-top: 3px;
}

/* --- tables -------------------------------------------------------------- */
.tw { overflow-x: auto; border: 1px solid var(--rule); background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
caption {
  caption-side: top;
  text-align: left;
  padding: 13px 16px 11px;
  font-size: 12.5px;
  color: var(--muted);
  border-bottom: 1px solid var(--rule);
  background: var(--surface-sunk);
}
th, td { padding: 8px 16px; text-align: right; border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; }
thead th {
  font-size: 11.5px;
  letter-spacing: 0.055em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 620;
  white-space: nowrap;
}
tbody tr:last-child td { border-bottom: none; }
td.num, th.num {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
}
td.hi { color: var(--accent); font-weight: 620; }
td.lo { color: var(--muted); }
tr.rule-top td { border-top: 1px solid var(--rule-strong); }

.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  padding: 2px 8px;
  border-radius: 2px;
  border: 1px solid var(--rule-strong);
  color: var(--ink-soft);
  white-space: nowrap;
}
.chip::before { content: ""; width: 7px; height: 7px; border-radius: 1px; background: var(--accent); }
.chip.arm2::before { background: var(--arm2); }

/* --- heatmap ------------------------------------------------------------- */
.maps { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 20px; }
.map { border: 1px solid var(--rule); background: var(--surface); overflow-x: auto; }
.map .map-head { padding: 13px 16px 11px; border-bottom: 1px solid var(--rule); background: var(--surface-sunk); display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.map table { font-size: 13px; }
.map td.cell {
  text-align: center;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
  width: 58px;
  cursor: default;
  border-bottom: 2px solid var(--surface);
  border-right: 2px solid var(--surface);
}
.map td.terms { font-size: 12.5px; color: var(--ink-soft); line-height: 1.35; min-width: 190px; }
.map td.terms code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11.5px; color: var(--muted); }
.scale { display: flex; align-items: center; gap: 8px; font-size: 11.5px; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.scale .bar { width: 88px; height: 9px; border: 1px solid var(--rule-strong); }

#tip {
  position: fixed;
  z-index: 40;
  pointer-events: none;
  opacity: 0;
  transition: opacity .1s ease;
  background: var(--surface);
  color: var(--ink);
  border: 1px solid var(--rule-strong);
  box-shadow: 0 6px 22px rgba(0, 0, 0, .16);
  padding: 8px 11px;
  font-size: 12.5px;
  line-height: 1.45;
  max-width: 260px;
}
#tip .t { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11.5px; color: var(--muted); display: block; }
@media (prefers-reduced-motion: reduce) { #tip { transition: none; } }

/* --- notes --------------------------------------------------------------- */
.note {
  border: 1px solid var(--rule);
  border-left: 3px solid var(--caution);
  background: var(--caution-soft);
  padding: 15px 18px;
  font-size: 14.5px;
  color: var(--ink-soft);
  display: flex;
  flex-direction: column;
  gap: 9px;
}
.note .eyebrow { color: var(--caution); }
.note p { margin: 0; }
ul.plain { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 10px; color: var(--ink-soft); font-size: 15px; }
ul.plain li::marker { color: var(--accent); }
footer {
  border-top: 1px solid var(--rule);
  padding-top: 18px;
  font-size: 12.5px;
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  line-height: 1.7;
}
a { color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
"""

JS = """
(function () {
  var tip = document.getElementById('tip');
  function show(e, html) {
    tip.innerHTML = html;
    tip.style.opacity = '1';
    var r = tip.getBoundingClientRect();
    var x = Math.min(e.clientX + 14, window.innerWidth - r.width - 10);
    var y = Math.max(e.clientY - r.height - 12, 8);
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }
  document.querySelectorAll('[data-tip]').forEach(function (el) {
    el.addEventListener('mousemove', function (e) { show(e, el.getAttribute('data-tip')); });
    el.addEventListener('mouseleave', function () { tip.style.opacity = '0'; });
  });
})();
"""


def f3(x: float) -> str:
    return f"{x:.3f}"


def cell_style(share: float, vmax: float) -> str:
    """Sequential single-hue fill, light -> dark, over the surface token."""
    a = 0.0 if vmax <= 0 else min(share / vmax, 1.0)
    # Gamma < 1 keeps the low end visible without flattening the top.
    a = a ** 0.72
    ink = "var(--surface)" if a > 0.62 else "var(--ink-soft)"
    return f"background: rgba(var(--ramp), {a:.3f}); color: {ink};"


def table(caption: str, head: list[str], rows: list[list[str]], cls: str = "") -> str:
    ths = "".join(f'<th scope="col">{h}</th>' for h in head)
    trs = ""
    for r in rows:
        tds = "".join(
            c if c.startswith("<td") else f'<td class="num">{c}</td>' for c in r[1:]
        )
        trs += f'<tr><th scope="row">{r[0]}</th>{tds}</tr>'
    return (
        f'<div class="tw {cls}"><table><caption>{caption}</caption>'
        f'<thead><tr><th scope="col"></th>{ths}</tr></thead><tbody>{trs}</tbody></table></div>'
    )


def chip(label: str, second: bool) -> str:
    return f'<span class="chip{" arm2" if second else ""}">{label}</span>'


def heatmap(label: str, r: dict, second: bool) -> str:
    t = r["topics"]
    sizes = {c: r["cohort_sizes"][f"C{c}"] for c in (1, 2, 3)}
    hists = {c: t["histograms"][f"C{c}"] for c in (1, 2, 3)}
    k = t["k"]
    order = sorted(range(k), key=lambda i: -(hists[3][i] / max(sizes[3], 1)))
    vmax = max(hists[c][i] / max(sizes[c], 1) for c in (1, 2, 3) for i in range(k))

    rows = ""
    for i in order:
        cells = ""
        for c in (1, 2, 3):
            n = hists[c][i]
            share = n / max(sizes[c], 1)
            tip = (
                f"<span class='t'>{label} &middot; C{c}</span>"
                f"<b>{n}</b> of {sizes[c]} pairs &mdash; {share*100:.0f}% of the cohort"
                f"<span class='t'>{t['cluster_terms'][i]}</span>"
            )
            cells += (
                f'<td class="cell" style="{cell_style(share, vmax)}" '
                f'data-tip="{tip}">{n}</td>'
            )
        rows += (
            f'<tr><td class="terms"><code>{t["cluster_terms"][i]}</code></td>{cells}</tr>'
        )

    return f"""
    <div class="map">
      <div class="map-head">
        {chip(label, second)}
        <span class="scale">0<span class="bar" style="background: linear-gradient(90deg, rgba(var(--ramp),0.02), rgba(var(--ramp),1))"></span>{vmax*100:.0f}% of cohort</span>
      </div>
      <table>
        <thead><tr><th scope="col">cluster, by top terms</th>
        <th scope="col" class="num">C1</th><th scope="col" class="num">C2</th><th scope="col" class="num">C3</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def verdict_sentences(d: dict, clones: int, worst_cross: float,
                      seps: list[float], null_hi: float, guard_rate: float) -> str:
    """The headline, stated from the measurements rather than assumed.

    Three experiments run through this builder and they need not agree, so each clause
    is chosen by what the numbers actually show. Nothing here asserts a conclusion the
    data does not carry.
    """
    n_arms = len(d["arms"])
    scope = "in both arms" if n_arms == 2 else f"across {n_arms} arms"

    if clones == 0:
        dup = ("There is not one near-duplicate anywhere &mdash; the closest any later "
               f"conversation comes to an earlier one is {worst_cross:.2f}, against the "
               "0.80 this repo calls a clone.")
    else:
        dup = (f"{clones} later conversations land at or above the 0.80 clone threshold "
               f"against an earlier one; the worst is {worst_cross:.2f}.")

    if guard_rate >= 0.10:
        dup += (f" That has to be read against the guard, which rejected "
                f"{guard_rate*100:.0f}% of everything submitted here: this is what got "
                "through a filter, not what the attackers wrote.")
    elif guard_rate > 0:
        dup += (f" The guard rejected only {guard_rate*100:.1f}% of submissions, so this "
                "is close to organic.")

    lo = min(seps)
    if lo >= 0.85:
        sep = ("Wording alone separates the cohorts near-perfectly "
               f"({lo:.2f}&ndash;{max(seps):.2f} AUROC against a null of {null_hi:.2f}), "
               "so each iteration's additions sit in a measurably different region.")
    elif lo >= 0.70:
        sep = (f"Wording separates the cohorts clearly ({lo:.2f}&ndash;{max(seps):.2f} "
               f"AUROC against a null of {null_hi:.2f}), though less sharply for some "
               "pairs of iterations than others.")
    else:
        sep = (f"Separability is uneven: {lo:.2f}&ndash;{max(seps):.2f} AUROC against a "
               f"null of {null_hi:.2f}, so at least one pair of cohorts is close to "
               "exchangeable in its wording.")

    if clones == 0 and lo >= 0.85:
        verdict = f"The increments are genuinely different, by every measure applied, {scope}."
    elif clones == 0:
        verdict = f"The increments are not duplicates {scope}, but they differ unevenly."
    else:
        verdict = f"The increments repeat earlier material in part {scope}."
    return f"{verdict} {dup} {sep}"


def build(d: dict) -> str:
    arms = d["arms"]
    keys = list(arms)
    labels = d.get("labels") or {k: k for k in keys}
    A0, A1 = arms[keys[0]], arms[keys[1]]
    L0, L1 = labels[keys[0]], labels[keys[1]]
    branch, subtitle = PROVENANCE[d["experiment"]]

    # --- headline numbers, all read from the JSON ---
    worst_cross = max(
        arms[a]["difflib"][f"C{c}_vs_earlier"]["max"] for a in arms for c in (2, 3)
    )
    n_pairs = sum(arms[a]["n_pairs"] for a in arms)
    clones = sum(
        round(arms[a]["difflib"][f"C{c}_vs_earlier"]["frac_ge_0.8"]
              * arms[a]["difflib"][f"C{c}_vs_earlier"]["n"])
        for a in arms for c in (2, 3)
    )
    sep_keys = ("C1_vs_C2", "C2_vs_C3", "C1_vs_C3")
    seps = [arms[a]["separability"][k]["auroc"] for a in arms for k in sep_keys]
    null_hi = max(arms[a]["separability"][k]["null_p95"] for a in arms for k in sep_keys)

    sizes_rows = [
        [chip(labels[a], i > 0),
         str(arms[a]["cohort_sizes"]["C1"]),
         str(arms[a]["cohort_sizes"]["C2"]),
         str(arms[a]["cohort_sizes"]["C3"]),
         str(arms[a]["n_pairs"])]
        for i, a in enumerate(keys)
    ]

    def dup_table(block: str, caption: str) -> str:
        rows = []
        for a in keys:
            if "C2_vs_earlier" not in (arms[a].get(block) or {}):
                return ""
            for c in (2, 3):
                s = arms[a][block][f"C{c}_vs_earlier"]
                rows.append([
                    f"{labels[a]} &nbsp; C{c} &rarr; earlier",
                    f3(s["median"]), f3(s["p90"]), f3(s["max"]),
                    f'<td class="num lo">{round(s["frac_ge_0.8"] * s["n"])} of {s["n"]}</td>',
                ])
        return table(caption, ["median", "p90", "max", "&ge; 0.8"], rows)

    nn_rows = []
    for a in keys:
        row = [labels[a]]
        for c in (2, 3):
            w = arms[a]["cosine"][f"word_C{c}_within"]["median"]
            e = arms[a]["cosine"][f"word_C{c}_vs_earlier"]["median"]
            row += [f3(w), f'<td class="num hi">{f3(e)}</td>']
        nn_rows.append(row)

    sep_rows = []
    for key, lab in zip(sep_keys, ("C1 vs C2", "C2 vs C3", "C1 vs C3")):
        sep_rows.append([
            lab,
            f'<td class="num hi">{f3(A0["separability"][key]["auroc"])}</td>',
            f'<td class="num hi">{f3(A1["separability"][key]["auroc"])}</td>',
            f'<td class="num lo">{f3(A0["separability"][key]["null_mean"])} / '
            f'{f3(max(A0["separability"][key]["null_p95"], A1["separability"][key]["null_p95"]))}</td>',
        ])

    et_rows = []
    for a in keys:
        for key, v in sorted(arms[a]["separability_within_error_type"].items()):
            if key.endswith("sizes"):
                continue
            pretty = key.replace("false_positive_", "").replace("false_negative_", "")
            direction = ("false positive" if key.startswith("false_positive")
                         else "false negative")
            et_rows.append([
                f"{labels[a]} &nbsp; <span class='chip'>{direction}</span> &nbsp; "
                f"{pretty.replace('_', ' ')}",
                f'<td class="num hi">{f3(v["auroc"])}</td>',
                f'<td class="num lo">{f3(v["null_mean"])}</td>',
                f'<td class="num lo">{v["n_a"]} / {v["n_b"]}</td>',
            ])
    et_note = ("" if et_rows else
               "<p>No pair of cohorts held enough successes of a single error type "
               "(15 minimum each side) for this control to run on this experiment.</p>")

    js_rows = []
    for key, lab in zip(sep_keys, ("C1 vs C2", "C2 vs C3", "C1 vs C3")):
        js_rows.append([
            lab,
            f'<td class="num hi">{f3(A0["topics"][key]["js"])}</td>',
            f'<td class="num lo">{f3(A0["topics_all_tokens"][key]["js"])}</td>',
            f'<td class="num hi">{f3(A1["topics"][key]["js"])}</td>',
            f'<td class="num lo">{f3(A1["topics_all_tokens"][key]["js"])}</td>',
            f'<td class="num lo">{f3(max(A0["topics"][key]["null_p95"], A1["topics"][key]["null_p95"]))}</td>',
        ])

    prov_rows = []
    for a in keys:
        for c in (1, 2, 3):
            p = arms[a]["provenance"][f"C{c}"]
            prov_rows.append([
                f"{labels[a]} &nbsp; C{c}",
                str(p["n_pairs"]),
                f'<td class="num">{p["frac_false_positive"]*100:.0f}%</td>',
                f3(p["mean_probe_score"]),
                f'<td class="num lo">{p["mean_words"]:.0f}</td>',
            ])

    guard_rows = []
    guard_total_rej = guard_total_sub = 0
    for a in keys:
        g = arms[a].get("guard")
        if not g:
            continue
        guard_total_rej += g["n_rejected"]
        guard_total_sub += g["n_submissions"]
        iters = g["iterations_with_rejections"]
        guard_rows.append([
            chip(labels[a], keys.index(a) > 0),
            f'<td class="num hi">{g["n_rejected"]}</td>',
            f'<td class="num lo">{g["n_submissions"]}</td>',
            f'<td class="num">{g["reject_rate"]*100:.1f}%</td>',
            f'<td class="num lo">{", ".join(str(i) for i in iters) if iters else "&mdash;"}</td>',
        ])
    guard_rate = guard_total_rej / guard_total_sub if guard_total_sub else 0.0
    guard_table_html = table(
        "Submit-time guard activity",
        ["rejected", "of submissions", "rate", "iterations it fired in"], guard_rows,
    ) if guard_rows else ""

    if guard_rate >= 0.10:
        guard_note = (
            f"<p><strong>This run is guard-dominated.</strong> {guard_total_rej} of "
            f"{guard_total_sub} submissions ({guard_rate*100:.0f}%) were turned away as "
            "near-duplicates of an existing success, and it kept firing in every "
            "iteration. The attackers were repeatedly producing clones; the low "
            "similarity numbers above describe the filtered survivors, and say nothing "
            "reassuring about what the attackers would have submitted unfiltered.</p>"
        )
    elif guard_total_rej == 0:
        guard_note = (
            "<p><strong>The guard never fired.</strong> Nothing was filtered, so the "
            "absence of near-duplicates above is entirely a property of what the "
            "attackers produced.</p>"
        )
    else:
        guard_note = (
            f"<p>The guard fired {guard_total_rej} times in {guard_total_sub} "
            f"submissions ({guard_rate*100:.1f}%), so it cannot account for the "
            "similarity levels above.</p>"
        )

    # --- cross-cohort duplicates that DID get through --------------------------------
    dup_pairs = [(a, p) for a in keys
                 for p in (arms[a].get("cross_cohort_duplicates") or {}).get("pairs", [])]
    has_dup_block = any("cross_cohort_duplicates" in arms[a] for a in keys)
    if not has_dup_block or not dup_pairs:
        dups_section = ""
        if has_dup_block:
            dups_section = f"""
  <section>
    <div class="sec-head"><span class="sec-num">01c</span><h2>Duplicates that got through</h2></div>
    <div class="prose"><p>None. No cross-cohort pair of openers reaches 0.80 in either
    arm, so the guard's per-error-type blind spot (below) was never exercised here.</p></div>
  </section>"""
    else:
        n_cross = sum((arms[a].get("cross_cohort_duplicates") or {}).get("n_cross_error_type", 0)
                      for a in keys)
        n_same = sum((arms[a].get("cross_cohort_duplicates") or {}).get("n_same_error_type", 0)
                     for a in keys)
        n_twin = sum((arms[a].get("cross_cohort_duplicates") or {}).get("n_contradictory_twins", 0)
                     for a in keys)
        rows = []
        for a, pr in sorted(dup_pairs, key=lambda x: -x[1]["opener_similarity"]):
            asst = pr["assistant_similarity"]
            twin = asst is not None and asst >= 0.8
            rows.append([
                f'{labels[a]} &nbsp; C{pr["cohort_new"]} &rarr; C{pr["cohort_earlier"]}',
                f'<td class="num hi">{pr["opener_similarity"]:.3f}</td>',
                f'<td class="num{" hi" if twin else " lo"}">'
                f'{"&mdash;" if asst is None else f"{asst:.3f}"}</td>',
                f'<td class="num lo">{pr["error_type_new"].replace("_", " ")} vs '
                f'{pr["error_type_earlier"].replace("_", " ")}</td>',
                f'<td class="num">{"contradictory twin" if twin else "matched contrast"}</td>',
            ])
        dups_section = f"""
  <section>
    <div class="sec-head"><span class="sec-num">01c</span><h2>Duplicates that got through, and why</h2></div>
    <div class="prose">
      <p><strong>{n_cross} of the {n_cross + n_same} cross-cohort opener duplicates are
      <em>cross-error-type</em></strong> &mdash; a false-positive success cloning a
      false-negative one, or the reverse. (These are <em>pairs</em>; the {clones}
      conversations counted above are the distinct later ones involved, since one
      conversation can clone more than one predecessor.) That is a structural blind spot, not bad luck:
      each hunt writes its own JSONL and gets its own store, so the guard can only ever
      compare a candidate against successes of the <em>same</em> error type.</p>
      <p>What such a pair <em>is</em> depends on the reply. The two successes carry
      opposite judge labels by construction, so if the assistant turns diverge the pair is
      a matched minimal contrast &mdash; same prompt, one compliant answer and one not &mdash;
      which for an assistant-centric concept is desirable training data. If the replies are
      near-identical too, it is instead two nearly identical rows carrying opposite labels.
      <strong>{n_twin} of {len(dup_pairs)} fall on the wrong side of that line.</strong></p>
    </div>
    {table("Every cross-cohort opener pair at or above the guard's own 0.80",
           ["opener", "assistant", "error types", "what it is"], rows)}
  </section>"""

    self_pair = A0["anchor_self_pair_difflib"]["median"]

    def top_clusters(arm_data: dict, cohort: int, n: int):
        t = arm_data["topics"]
        h = t["histograms"][f"C{cohort}"]
        order = sorted(range(t["k"]), key=lambda i: -h[i])[:n]
        return [(i, h[i], t["cluster_terms"][i]) for i in order]

    def terms3(s: str) -> str:
        return ", ".join(s.split(", ")[:3])

    def rotation_para(a: str) -> str:
        """One sentence per arm on where C3 went and what it left."""
        r = arms[a]
        t = r["topics"]
        c3 = top_clusters(r, 3, 3)
        c1 = top_clusters(r, 1, 3)
        c1_now = [t["histograms"]["C3"][i] for i, _, _ in c1]
        unseen = [
            (i, t["histograms"]["C3"][i], t["cluster_terms"][i])
            for i in range(t["k"])
            if t["histograms"]["C3"][i] > 0
            and t["histograms"]["C1"][i] == 0 and t["histograms"]["C2"][i] == 0
        ]
        share = sum(n for _, n, _ in c3) / max(r["cohort_sizes"]["C3"], 1)
        bits = (f"<strong>{labels[a]}</strong>: C3's three biggest clusters hold "
                f"{sum(n for _, n, _ in c3)} of its {r['cohort_sizes']['C3']} pairs "
                f"({share*100:.0f}%) &mdash; {terms3(c3[0][2])} ({c3[0][1]}), "
                f"{terms3(c3[1][2])} ({c3[1][1]}), {terms3(c3[2][2])} ({c3[2][1]}). "
                f"Its three biggest C1 clusters &mdash; {terms3(c1[0][2])} ({c1[0][1]}), "
                f"{terms3(c1[1][2])} ({c1[1][1]}), {terms3(c1[2][2])} ({c1[2][1]}) "
                f"&mdash; fall to {c1_now[0]}, {c1_now[1]} and {c1_now[2]}.")
        if unseen:
            biggest = max(unseen, key=lambda u: u[1])
            bits += (f" The {biggest[1]}-pair <em>{terms3(biggest[2])}</em> cluster had "
                     f"<strong>zero</strong> members in C1 and C2 combined.")
        if t["C3_abandoned_clusters"]:
            bits += (f" It abandons {t['C3_abandoned_clusters']} clusters the earlier "
                     "cohorts populated.")
        return f"<p>{bits}</p>"

    asst_table = dup_table(
        "difflib_assistant_side",
        "Same metric on the ASSISTANT side &mdash; where the label lives, and where the "
        "submit-time guard never looks",
    )
    asst_block = A0.get("difflib_assistant_side") or {}
    n_no_asst = sum((arms[a].get("difflib_assistant_side") or {}).get(
        "n_without_assistant_turn", 0) for a in keys)
    asst_excl = ""
    if n_no_asst:
        asst_excl = (
            f"<p><strong>{n_no_asst} successes across the two arms have no assistant turn "
            "at all</strong> and are excluded from this table. They have to be: "
            "<code>difflib</code> scores two empty strings as a perfect 1.0, so leaving "
            "them in manufactures duplicates out of nothing. Their existence is itself a "
            "finding when the label is defined by a reply that is not there.</p>"
        )

    asst_note = ""
    if asst_table and d.get("assistant_centric"):
        asst_note = (
            "<p>This concept is <strong>assistant-centric</strong>: the label is set by "
            "the reply, not the prompt. The live guard only ever compares first user "
            "turns, so a repeated <em>answer</em> under a fresh prompt is exactly what it "
            "cannot see &mdash; which makes this the load-bearing duplication check here, "
            "not a supplement to the one above.</p>"
        )
    elif asst_table:
        asst_note = (
            "<p>The guard only ever compares first user turns. This concept is a property "
            "of the whole scenario rather than of the reply, so the opener is the right "
            "place to look &mdash; but the reply side is checked too, since nothing "
            "guarantees the two agree.</p>"
        )

    base_anchor = d.get("anchors", {}).get("base_data_within")
    base_para = ""
    if base_anchor:
        base_para = (
            f"<li><strong>The base-data anchor is not length-comparable.</strong> Its "
            f"openers average {base_anchor['mean_opener_chars']:.0f} characters against "
            f"the red team's, and difflib's ratio is length-sensitive. Treat it as a "
            f"direction, not a number.</li>"
        )

    return f"""<title>Red-Team Vintage Novelty &mdash; {d["concept"]}</title>
<style>{CSS}</style>
<div id="tip" role="status" aria-live="polite"></div>
<div class="frame">

  <header class="head">
    <span class="eyebrow">{subtitle} &middot; {branch}</span>
    <h1>Are v1, v2 and v3 made of genuinely different samples?</h1>
    <p class="sub">The vintage sweep reads an AUROC curve off three nested red-team sets.
    That reading only means something if each iteration <em>added</em> material rather than
    re-finding what the last one had. This is that check for the
    <strong>{d["concept"]}</strong> concept &mdash; {n_pairs} pairs across both arms, text
    only, no activations.</p>
  </header>

  <div class="verdict">
    <p class="lede">{verdict_sentences(d, clones, worst_cross, seps, null_hi, guard_rate)}</p>
    <div class="tiles">
      <div class="tile"><span class="num">{clones}</span><span class="lab">cross-cohort near-duplicates, at the &tau;&nbsp;=&nbsp;0.8 the repo calls a clone</span></div>
      <div class="tile"><span class="num">{worst_cross:.2f}</span><span class="lab">highest cross-cohort similarity seen at all &mdash; the clone threshold is 0.80</span></div>
      <div class="tile"><span class="num">{min(seps):.2f}&ndash;{max(seps):.2f}</span><span class="lab">AUROC telling cohorts apart from wording alone, where a permutation null tops out at {null_hi:.2f}</span></div>
    </div>
  </div>

  <section>
    <div class="sec-head"><h2>The cohorts</h2></div>
    <div class="prose">
      <p><strong>C1 = v1</strong>, <strong>C2 = v2 &minus; v1</strong>,
      <strong>C3 = v3 &minus; v2</strong>, assigned by <em>first</em> appearance of the
      source success in <code>redteam_postprocessed_iter{{k}}.jsonl</code>. The unit is the
      pair, so a success and its contrastive counterpart always move together.</p>
    </div>
    {table("Pairs per cohort, iteration-3 universe", ["C1", "C2", "C3", "total"], sizes_rows)}
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">01</span><h2>Near-duplication</h2></div>
    <div class="prose">
      <p>Similarity is <code>difflib.SequenceMatcher(autojunk=False)</code> on the first 600
      characters of the first user turn: the repo's own clone metric, and the one
      <code>near_dup_guard</code> enforced live at &tau;&nbsp;=&nbsp;0.8.</p>
    </div>
    {dup_table("difflib", "Nearest neighbour among all earlier cohorts")}
    <div class="prose">{asst_note}{asst_excl}</div>
    {asst_table}
    <div class="prose">
      <p>For scale, on the same metric: a success measured against the contrastive
      counterpart written <em>from it</em> scores {f3(self_pair)} (median).</p>
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">01b</span><h2>How much of that is the guard?</h2></div>
    <div class="prose">
      <p>These runs had <code>near_dup_guard</code> on at &tau;&nbsp;=&nbsp;0.8, rejecting a
      candidate <em>before</em> the probe and judge if its opener cloned an existing
      success. Rejections never reach the JSONL, so the runlog is the only record of how
      hard that filter was working &mdash; and it decides whether the row above reports an
      <em>organic</em> absence of clones or an <em>enforced</em> one.</p>
    </div>
    {guard_table_html}
    <div class="prose">{guard_note}</div>
  </section>

  {dups_section}

  <section>
    <div class="sec-head"><span class="sec-num">02</span><h2>Is each increment more like itself than like anything earlier?</h2></div>
    {table("TF-IDF cosine nearest neighbour, word 1&ndash;2grams, whole conversation",
           ["C2 within", "C2 &rarr; earlier", "C3 within", "C3 &rarr; earlier"], nn_rows)}
    <div class="prose">
      <p>A cohort that re-worked old ground would show "&rarr; earlier" at or above
      "within".</p>
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">03</span><h2>Can a bag-of-words model tell the cohorts apart?</h2></div>
    <div class="prose">
      <p>5-fold CV AUROC, against a null from permuting the labels on the same features
      &mdash; the only honest baseline at these sample sizes over ~20k features.</p>
    </div>
    {table("Cohort separability from wording alone", [L0, L1, "null mean / p95"], sep_rows)}
    <div class="prose">
      <p><strong>The confound, controlled.</strong> The cohorts do not hold the attack
      direction fixed, and false-positive and false-negative successes are different
      content classes by definition. Repeating the test <em>inside</em> one error type:</p>
      {et_note}
    </div>
    {table("Separability within a single error type", ["AUROC", "null", "n"], et_rows) if et_rows else ""}
    {table("Where each cohort came from", ["pairs", "false pos.", "mean probe score", "mean words"], prov_rows)}
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">04</span><h2>What changes: the scenario mix</h2></div>
    <div class="prose">
      <p>15-way KMeans over the pooled sources, on <strong>content words only</strong> &mdash;
      English stop words, contraction fragments and the role prefixes are stripped, so a
      cluster is named by its subject rather than by how chatty its conversations are. Cells
      are pair counts, shaded by share of that cohort. Rows are ordered by how much of C3
      they hold.</p>
    </div>
    <div class="maps">
      {heatmap(L0, A0, False)}
      {heatmap(L1, A1, True)}
    </div>
    {table("Jensen-Shannon divergence between cluster histograms",
           [L0, "&nbsp;&nbsp;(all tokens)", L1, "&nbsp;&nbsp;(all tokens)", "null p95"],
           js_rows)}
    <div class="prose">
      {rotation_para(keys[0])}
      {rotation_para(keys[1])}
    </div>
  </section>

  <section>
    <div class="sec-head"><h2>Caveats</h2></div>
    <ul class="plain">
      <li><strong>Cohort is confounded with the probe.</strong> Each iteration attacks a
      <em>retrained</em> probe, so "different samples" is partly "different target". The
      samples differ; this does not isolate attacker exploration from probe drift.</li>
      <li><strong>Lexical and topical, not semantic.</strong> Two conversations could be the
      same strategy with different props and still land far apart here.</li>
      <li><strong>The topic stop list includes a judgement call.</strong> Beyond function
      words it drops a dozen generic discourse verbs, so the all-token JS numbers are
      reported alongside. Sections 01&ndash;03 use the <em>unfiltered</em> text.</li>
      {base_para}
    </ul>
  </section>

  <footer>
    scripts/vintage_cohort_novelty.py --experiment {d["experiment"]}<br>
    every figure on this page is read from that run's JSON at build time by
    scripts/build_cohort_novelty_report.py
  </footer>
</div>
<script>{JS}</script>
"""


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="hs", choices=sorted(SOURCES))
    args = ap.parse_args()

    src, out = SOURCES[args.experiment]
    d = json.loads(src.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(d), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
