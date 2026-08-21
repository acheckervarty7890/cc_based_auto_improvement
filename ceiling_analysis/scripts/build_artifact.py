#!/usr/bin/env python
"""Generate the ceiling-analysis write-up as a self-contained HTML page.

Every number on the page is read out of ``ceiling_analysis/results/`` — none is typed by
hand — so the page cannot drift from the JSONL it describes. Re-run this after any re-run
of the sweep and republish the same file path to refresh the artifact in place.

Published at https://claude.ai/code/artifact/35391e55-8c9c-40dd-a01a-f353a215de5a
(companion to "The Instruction Probe Ceiling", the same analysis on the third concept —
the two share a design system deliberately).

    ceiling_analysis/scripts/build_artifact.py [out.html]
"""
from __future__ import annotations

import collections
import json
import statistics as st
from pathlib import Path

REPO = Path("/workspace/cc_based_auto_improvement")
RES = REPO / "ceiling_analysis/results"
import sys

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "ceilings.html"

ARMS = ("mixed", "finetune", "dev_only")
ARM_LABEL = {"mixed": "mixed into the red-team set",
             "finetune": "red-team first, then fine-tune",
             "dev_only": "dev samples alone (control)"}
ARM_CLASS = {"mixed": "s-mix", "finetune": "s-ft", "dev_only": "s-dev"}
MIN_STEP = 16 * 4  # batch_size x grad_accum: below this, optimizer.step() never fires


def load(concept):
    d = json.loads((RES / f"ceiling_{concept}.json").read_text())
    rows = [json.loads(l) for l in (RES / f"sweep_{concept}.jsonl").read_text().splitlines() if l.strip()]
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["arm"], r["n_dev"])].append(r["mean"]["auroc"])
    curve = {}
    for (arm, n), v in agg.items():
        curve.setdefault(arm, {})[n] = (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0, len(v))
    curve = {a: dict(sorted(p.items())) for a, p in curve.items()}
    rungs = list(d["by_train_size"].items())
    scores = {k: v["mean"]["auroc"] for k, v in rungs}
    best_rung = max(scores, key=scores.get)
    ordered = sorted(scores.values())
    n0 = curve["mixed"][0][0]
    zero_row = next(r for r in rows if r["n_dev"] == 0 and r["arm"] == "mixed")
    return {
        "name": concept, "raw": d, "rows": rows, "curve": curve,
        "rungs": rungs, "best_rung": best_rung, "ceiling": scores[best_rung],
        "top_step": ordered[-1] - ordered[-2],
        "n0": n0, "gap": scores[best_rung] - n0,
        "n0_per_split": {k: v["auroc"] for k, v in zero_row["per_split"].items()},
        "ceil_per_split": {k: v["auroc"] for k, v in d["by_train_size"][best_rung]["per_split"].items()},
        "points": sorted({n for a in curve for n in curve[a]}),
    }


def best_of(c, arm):
    pts = c["curve"][arm]
    n, (v, s, _) = max(pts.items(), key=lambda kv: kv[1][0])
    return n, v, s


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------- charts
def line_chart(c, ylo, yhi, xlabel, dead_zone=None):
    """One concept's sweep: three arms, mean +/- across-seed spread, ceiling rule."""
    W, H = 760, 344
    L, R, T, B = 66, 744, 20, 288
    xs = c["points"]
    xmax = max(xs)

    def X(n):
        return L + (R - L) * (n / xmax)

    def Y(v):
        return B - (B - T) * ((v - ylo) / (yhi - ylo))

    o = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
         f'aria-label="mean eval AUROC against number of in-distribution dev samples">']

    # dead zone: training sets too small for a single optimizer step
    if dead_zone:
        o.append(f'<rect class="deadzone" x="{L}" y="{T}" width="{X(dead_zone)-L:.1f}" height="{B-T}"/>')
        o.append(f'<text class="zonelab" x="{X(dead_zone)-6:.1f}" y="{T+14}" text-anchor="end">'
                 f'no optimizer step</text>')

    # y grid
    step = 0.02 if (yhi - ylo) > 0.06 else 0.01
    v = ylo
    while v <= yhi + 1e-9:
        o.append(f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{R}" y2="{Y(v):.1f}"/>')
        o.append(f'<text class="tick ytick" x="{L-9}" y="{Y(v)+4:.1f}">{v:.2f}</text>')
        v = round(v + step, 4)

    # x ticks
    for n in xs:
        o.append(f'<text class="tick" x="{X(n):.1f}" y="{B+20}">{n}</text>')

    # ceiling
    o.append(f'<line class="ceiling" x1="{L}" y1="{Y(c["ceiling"]):.1f}" x2="{R}" y2="{Y(c["ceiling"]):.1f}"/>')
    o.append(f'<text class="ceiling-label" x="{R}" y="{Y(c["ceiling"])-7:.1f}" text-anchor="end">'
             f'ceiling {c["ceiling"]:.4f}</text>')

    for arm in ARMS:
        pts = [(n, m, s) for n, (m, s, _) in c["curve"][arm].items() if ylo <= m <= yhi]
        if not pts:
            continue
        cls = ARM_CLASS[arm]
        d = " ".join(("M" if i == 0 else "L") + f"{X(n):.1f},{Y(m):.1f}"
                     for i, (n, m, _) in enumerate(pts))
        o.append(f'<path class="series {cls}" d="{d}"/>')
        for n, m, s in pts:
            if s > 0:
                o.append(f'<line class="whisk {cls}" x1="{X(n):.1f}" y1="{Y(min(m+s,yhi)):.1f}" '
                         f'x2="{X(n):.1f}" y2="{Y(max(m-s,ylo)):.1f}"/>')
            o.append(f'<circle class="dot {cls}" cx="{X(n):.1f}" cy="{Y(m):.1f}" r="3.2"/>')

    o.append(f'<text class="axis" x="{(L+R)/2:.0f}" y="{H-6}" text-anchor="middle">{esc(xlabel)}</text>')
    o.append("</svg>")
    return "\n".join(o)


def gap_bars(c, pretty):
    """Per split: where the red-team probe lands, and how far the ceiling is."""
    splits = list(c["ceil_per_split"])
    rowh, gap = 40, 8
    W = 760
    L, R = 232, 636
    H = len(splits) * (rowh + gap) + 6
    lo = min(min(c["n0_per_split"].values()), min(c["ceil_per_split"].values())) - 0.03
    lo = max(0.60, round(lo, 2))

    def X(v):
        return L + (R - L) * ((v - lo) / (1.0 - lo))

    o = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
         f'aria-label="red-team-only AUROC versus ceiling, per eval split">']
    for i, s in enumerate(splits):
        y = i * (rowh + gap) + 4
        a, ceil = c["n0_per_split"][s], c["ceil_per_split"][s]
        o.append(f'<text class="rowlab" x="{L-14}" y="{y+rowh/2+4:.0f}" text-anchor="end">'
                 f'{esc(pretty.get(s, s))}</text>')
        o.append(f'<rect class="track" x="{L}" y="{y}" width="{R-L}" height="{rowh}" rx="2"/>')
        o.append(f'<rect class="gapbar" x="{X(a):.1f}" y="{y}" width="{max(X(ceil)-X(a),0):.1f}" '
                 f'height="{rowh}" rx="2"/>')
        o.append(f'<rect class="achieved" x="{L}" y="{y}" width="{X(a)-L:.1f}" height="{rowh}" rx="2"/>')
        o.append(f'<line class="ceilmark" x1="{X(ceil):.1f}" y1="{y-3}" x2="{X(ceil):.1f}" y2="{y+rowh+3}"/>')
        o.append(f'<text class="barval" x="{R+16}" y="{y+rowh/2+4:.0f}">{a:.3f} &#8594; {ceil:.3f}</text>')
    o.append("</svg>")
    return "\n".join(o)


# ---------------------------------------------------------------------------- tables
def ladder_table(c):
    rows = []
    for tag, e in c["rungs"]:
        hi = ' class="hi"' if tag == c["best_rung"] else ""
        note = "best rung &#8212; reported as the ceiling" if tag == c["best_rung"] else ""
        rows.append(f'<tr{hi}><td>{esc(tag)}</td><td class="n">{e["mean"]["auroc"]:.4f}</td>'
                    f'<td class="n">{e["mean"]["accuracy"]:.4f}</td>'
                    f'<td class="n">{e["mean"]["tpr_at_fpr"]:.4f}</td><td class="note">{note}</td></tr>')
    return ("<table><thead><tr><th>training rows per fold</th><th class=\"n\">mean AUROC</th>"
            "<th class=\"n\">accuracy</th><th class=\"n\">TPR@1%FPR</th><th></th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>")


def curve_table(c):
    head = "".join(f'<th class="n">{n}</th>' for n in c["points"])
    body = []
    for arm in ARMS:
        cells = []
        for n in c["points"]:
            p = c["curve"][arm].get(n)
            if p is None:
                cells.append('<td class="n dash">&#8212;</td>')
                continue
            m, s, _ = p
            dead = ' class="n dead"' if (arm == "dev_only" and n and n < MIN_STEP) or \
                                        (arm == "finetune" and n and n < MIN_STEP) else ' class="n"'
            cells.append(f'<td{dead}>{m:.4f}</td>')
        body.append(f'<tr><td>{esc(ARM_LABEL[arm])}</td>{"".join(cells)}</tr>')
    return ('<table><thead><tr><th>N dev samples</th>' + head + '</tr></thead><tbody>'
            + "".join(body) + "</tbody></table>")


def close_table(c):
    rows = []
    for arm in ARMS:
        n, v, s = best_of(c, arm)
        frac = (v - c["n0"]) / c["gap"]
        rows.append(f'<tr><td>{esc(ARM_LABEL[arm])}</td><td class="n">{v:.4f}</td>'
                    f'<td class="n">{n}</td><td class="n">{frac*100:.0f}%</td>'
                    f'<td class="n">{c["ceiling"]-v:+.4f}</td></tr>')
    return ('<table><thead><tr><th>arm</th><th class="n">best mean AUROC</th><th class="n">at N</th>'
            '<th class="n">gap closed</th><th class="n">short of ceiling</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def runs_table(pairs):
    rows = "".join(f'<tr><td>{esc(a)}</td><td class="n">{b:.4f}</td></tr>' for a, b in pairs)
    return ('<table><thead><tr><th>published probe</th><th class="n">mean eval AUROC</th>'
            "</tr></thead><tbody>" + rows + "</tbody></table>")


# ---------------------------------------------------------------------------- page
hs = load("highstakes")
hh = load("hu_ha")

HS_PRETTY = {"anthropic_hh_balanced": "anthropic hh", "mt_balanced": "mt",
             "mts_balanced": "mts", "toolace_balanced": "toolace"}
HH_PRETTY = {"eval_ai_dilemmas": "ai dilemmas", "eval_ant_hh": "anthropic hh",
             "eval_balanced_refusal": "balanced refusal", "eval_daily_dilemmas": "daily dilemmas"}

hs_runs = [("experiment19 &#183; gpt-oss-120b, 3-probe ensemble &#183; iter3", 0.9276),
           ("experiment19 &#183; iter5 (final)", 0.9150),
           ("experiment18 &#183; gpt-oss-120b, single probe &#183; iter2", 0.9250),
           ("experiment18 &#183; iter3 (final)", 0.8988)]
hh_runs = [("experiment17 &#183; gpt-oss-120b, 10-probe ensemble &#183; iter4", 0.8826),
           ("experiment17 &#183; iter5 (final)", 0.8751)]

n_fits = len(hs["rows"]) + len(hh["rows"]) + sum(
    len(c["rungs"]) * c["raw"]["n_folds"] for c in (hs, hh))

hs_mix_n, hs_mix_v, _ = best_of(hs, "mixed")
hs_ft_n, hs_ft_v, _ = best_of(hs, "finetune")
hs_dev_n, hs_dev_v, _ = best_of(hs, "dev_only")
hh_mix_n, hh_mix_v, _ = best_of(hh, "mixed")
hh_dev_n, hh_dev_v, _ = best_of(hh, "dev_only")

HTML = f"""<title>Stakes and Harm Ceilings</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@400;600&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>
:root{{
  --paper:#F2F3F7; --surface:#FFFFFF; --surface-2:#EAECF3; --rule:#D3D7E4;
  --ink:#16192B; --muted:#5B6076; --faint:#878CA3;
  --accent:#2F4B99; --accent-soft:#8FA3D8;
  --limit:#A8761C; --limit-soft:#E3C98B;
  --good:#1F7A5C; --warn:#9C3B34;
  --mix:#2F4B99; --ft:#1F7A5C; --dev:#B4562A;
  --dead:#E4E6EE;
}}
@media (prefers-color-scheme: dark){{
  :root:not([data-theme="light"]){{
    --paper:#10121C; --surface:#171A26; --surface-2:#1E2231; --rule:#2C3145;
    --ink:#E8EAF2; --muted:#9AA0B8; --faint:#767C93;
    --accent:#8AA6FF; --accent-soft:#3C4A75;
    --limit:#E0A93E; --limit-soft:#6B5526;
    --good:#4FCFA6; --warn:#E4736E;
    --mix:#8AA6FF; --ft:#4FCFA6; --dev:#E8956A;
    --dead:#1A1E2B;
  }}
}}
:root[data-theme="dark"]{{
  --paper:#10121C; --surface:#171A26; --surface-2:#1E2231; --rule:#2C3145;
  --ink:#E8EAF2; --muted:#9AA0B8; --faint:#767C93;
  --accent:#8AA6FF; --accent-soft:#3C4A75;
  --limit:#E0A93E; --limit-soft:#6B5526;
  --good:#4FCFA6; --warn:#E4736E;
  --mix:#8AA6FF; --ft:#4FCFA6; --dev:#E8956A;
  --dead:#1A1E2B;
}}

*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Serif",Georgia,"Times New Roman",serif;
  font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 28px 96px}}
.measure{{max-width:66ch}}

h1,h2,h3,.eyebrow,.tick,.rowlab,.barval,.legend,th,.num,.kpi-v,.pill,.axis,
.ceiling-label,.zonelab,.note,.kpi-k{{
  font-family:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif}}
h1{{font-size:clamp(2.4rem,5.2vw,3.6rem);line-height:1.04;font-weight:600;
   letter-spacing:-.015em;margin:0 0 .5rem;text-wrap:balance}}
h2{{font-size:1.85rem;font-weight:600;letter-spacing:-.01em;margin:0;text-wrap:balance}}
h3{{font-size:1.12rem;font-weight:600;margin:0 0 .35rem;text-wrap:balance}}
p{{margin:0}}
a{{color:var(--accent)}}
code,.mono,.num,.kpi-v,td.n,th.n{{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  font-variant-numeric:tabular-nums}}
code{{font-size:.86em;background:var(--surface-2);padding:.1em .35em;border-radius:3px}}
.eyebrow{{font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;
  color:var(--faint);font-weight:600}}

header.hero{{padding:76px 0 40px;border-bottom:1px solid var(--rule)}}
.hero .sub{{font-size:1.2rem;color:var(--muted);max-width:62ch;margin-top:.7rem}}
.meta{{display:flex;flex-wrap:wrap;gap:8px 26px;margin-top:26px;
  font-family:"IBM Plex Mono",monospace;font-size:.78rem;color:var(--faint)}}

section{{padding:52px 0;border-bottom:1px solid var(--rule);
  display:flex;flex-direction:column;gap:22px}}
section:last-of-type{{border-bottom:none}}
.sec-head{{display:flex;flex-direction:column;gap:6px}}

.pair{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:26px}}
.panel{{display:flex;flex-direction:column;gap:14px}}
.panel-head{{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;
  padding-bottom:8px;border-bottom:2px solid var(--ink)}}
.panel-head h3{{margin:0}}
.panel-head .tag{{font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;
  color:var(--faint);font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:600}}

.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.kpi{{background:var(--surface);border:1px solid var(--rule);border-radius:5px;
  padding:15px 17px;display:flex;flex-direction:column;gap:3px}}
.kpi-v{{font-size:1.85rem;font-weight:600;letter-spacing:-.02em;line-height:1}}
.kpi-k{{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);
  font-weight:600}}
.kpi-n{{font-size:.82rem;color:var(--muted);line-height:1.45;
  font-family:"IBM Plex Serif",serif}}
.kpi.is-limit .kpi-v{{color:var(--limit)}}
.kpi.is-good .kpi-v{{color:var(--good)}}
.kpi.is-warn .kpi-v{{color:var(--warn)}}

figure{{margin:0;background:var(--surface);border:1px solid var(--rule);
  border-radius:5px;padding:20px 20px 14px;display:flex;flex-direction:column;gap:12px}}
figcaption{{font-size:.85rem;color:var(--muted);line-height:1.5}}
.scroll{{overflow-x:auto}}
.chart{{display:block;width:100%;min-width:520px;height:auto}}
.grid{{stroke:var(--rule);stroke-width:1}}
.tick{{fill:var(--faint);font-size:11px;text-anchor:middle}}
.ytick{{text-anchor:end}}
.axis{{fill:var(--faint);font-size:11px;letter-spacing:.06em;text-transform:uppercase}}
.deadzone{{fill:var(--dead)}}
.zonelab{{fill:var(--faint);font-size:10.5px;letter-spacing:.07em;text-transform:uppercase}}
.ceiling{{stroke:var(--limit);stroke-width:1.5;stroke-dasharray:7 5}}
.ceiling-label{{fill:var(--limit);font-size:11px;letter-spacing:.06em}}
.series{{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}}
.whisk{{stroke-width:1.4;opacity:.75}}
.dot{{stroke:var(--surface);stroke-width:1.4}}
.s-mix{{stroke:var(--mix)}} circle.s-mix{{fill:var(--mix)}}
.s-ft{{stroke:var(--ft)}} circle.s-ft{{fill:var(--ft)}}
.s-dev{{stroke:var(--dev);stroke-dasharray:6 4}} circle.s-dev{{fill:var(--dev)}}
.legend{{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:8px 20px;
  font-size:.78rem;color:var(--muted)}}
.legend li{{display:flex;align-items:center;gap:7px}}
.swatch{{width:15px;height:3px;border-radius:2px;background:var(--muted);display:inline-block}}
.legend .s-mix .swatch{{background:var(--mix)}}
.legend .s-ft .swatch{{background:var(--ft)}}
.legend .s-dev .swatch{{background:var(--dev)}}

.track{{fill:var(--surface-2)}}
.achieved{{fill:var(--accent)}}
.gapbar{{fill:var(--limit-soft)}}
.ceilmark{{stroke:var(--limit);stroke-width:2}}
.rowlab{{fill:var(--ink);font-size:12.5px}}
.barval{{fill:var(--muted);font-size:11.5px;font-family:"IBM Plex Mono",monospace}}

table{{border-collapse:collapse;width:100%;font-size:.85rem;min-width:520px}}
th,td{{padding:8px 11px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}}
th:first-child,td:first-child{{text-align:left;white-space:normal}}
thead th{{font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;color:var(--faint);
  font-weight:600;border-bottom:1px solid var(--ink)}}
td.n{{font-size:.84rem}}
td.dead{{color:var(--faint);text-decoration:line-through;text-decoration-thickness:1px}}
td.dash{{color:var(--faint)}}
td.note{{text-align:left;white-space:normal;font-size:.72rem;letter-spacing:.06em;
  text-transform:uppercase;color:var(--limit)}}
tr.hi td{{background:var(--surface-2);font-weight:600}}

.findings{{display:flex;flex-direction:column;gap:0}}
.finding{{display:grid;grid-template-columns:118px 1fr;gap:26px;
  padding:22px 0;border-top:1px solid var(--rule)}}
.finding:first-child{{border-top:none}}
.pill{{font-size:.66rem;letter-spacing:.11em;text-transform:uppercase;font-weight:600;
  padding:4px 9px;border-radius:3px;align-self:start;text-align:center;
  border:1px solid currentColor}}
.p-result{{color:var(--accent)}}
.p-limit{{color:var(--limit)}}
.p-method{{color:var(--muted)}}
.finding p{{color:var(--muted);font-size:.93rem;margin-top:.3rem}}
.finding strong{{color:var(--ink);font-weight:600}}

.callout{{background:var(--surface);border:1px solid var(--rule);
  border-left:3px solid var(--limit);border-radius:4px;padding:16px 20px;
  font-size:.92rem;color:var(--muted)}}
.callout strong{{color:var(--ink)}}
.callout.is-accent{{border-left-color:var(--accent)}}

footer{{padding:40px 0 0;color:var(--faint);font-size:.8rem;
  font-family:"IBM Plex Mono",monospace;line-height:1.7}}

@media (max-width:640px){{
  body{{font-size:16px}}
  .finding{{grid-template-columns:1fr;gap:8px}}
  header.hero{{padding:48px 0 30px}}
}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:3px}}
@media (prefers-reduced-motion:no-preference){{
  .series{{stroke-dasharray:2400;stroke-dashoffset:2400;
    animation:draw 1.1s cubic-bezier(.4,0,.2,1) forwards}}
  .s-dev{{animation:none;stroke-dasharray:6 4}}
  @keyframes draw{{to{{stroke-dashoffset:0}}}}
}}
</style>

<div class="wrap">
<header class="hero">
  <div class="eyebrow">gemma-3-27b &#183; layer 32 &#183; high-stakes and harmful-to-human probes</div>
  <h1>What the red-team loop left on the table</h1>
  <p class="sub">Two probes trained on attacker-generated conversations, measured against
  the best the same probe family reaches when it trains on the evaluation distribution
  instead &#8212; and against a control that removes the red-team data entirely.</p>
  <div class="meta">
    <span>{hs["raw"]["n_eval_rows"]} + {hh["raw"]["n_eval_rows"]} eval rows &#183; 8 splits</span>
    <span>1&#8239;870 conversations extracted once</span>
    <span>{n_fits} probe fits, single probes throughout</span>
    <span>3 draw seeds per point</span>
  </div>
</header>

<section>
  <div class="sec-head">
    <div class="eyebrow">Part 1 &#8212; the ceilings</div>
    <h2>One of these is a ceiling. The other is a lower bound.</h2>
  </div>
  <p class="measure">A ceiling here is five-fold cross-validation <em>inside</em> the eval
  set: fit on the rows outside a fold, early-stop against a reserved slice of dev that no
  fit ever trains on, score the held-out fold. A training-size ladder runs alongside,
  because an estimate made from too little data is not a ceiling &#8212; it is a
  measurement of how little data there was. The two concepts answer that test
  differently, and it changes how their numbers should be read.</p>

  <div class="pair">
    <div class="panel">
      <div class="panel-head"><h3>high-stakes</h3><span class="tag">saturated</span></div>
      <div class="kpis">
        <div class="kpi is-limit"><span class="kpi-k">ceiling</span>
          <span class="kpi-v">{hs["ceiling"]:.4f}</span>
          <span class="kpi-n">rung <code>{hs["best_rung"]}</code></span></div>
        <div class="kpi"><span class="kpi-k">red-team only</span>
          <span class="kpi-v">{hs["n0"]:.4f}</span>
          <span class="kpi-n">842 post-processed successes + 50 base rows</span></div>
        <div class="kpi"><span class="kpi-k">gap</span>
          <span class="kpi-v">{hs["gap"]:.4f}</span>
          <span class="kpi-n">mean eval AUROC</span></div>
      </div>
      <div class="scroll">{ladder_table(hs)}</div>
      <p class="measure">Adding the whole 1&#8239;431-row dev pool on top of the full
      eval-internal training set makes it <em>worse</em>, by {abs(hs["rungs"][-1][1]["mean"]["auroc"] - hs["ceiling"]):.4f}.
      The ladder has flattened: more in-distribution data buys nothing, so
      {hs["ceiling"]:.4f} is where this probe family actually tops out.</p>
    </div>

    <div class="panel">
      <div class="panel-head"><h3>harmful-to-human</h3><span class="tag">still climbing</span></div>
      <div class="kpis">
        <div class="kpi is-limit"><span class="kpi-k">ceiling &#8805;</span>
          <span class="kpi-v">{hh["ceiling"]:.4f}</span>
          <span class="kpi-n">rung <code>{hh["best_rung"]}</code></span></div>
        <div class="kpi"><span class="kpi-k">red-team only</span>
          <span class="kpi-v">{hh["n0"]:.4f}</span>
          <span class="kpi-n">928 post-processed successes + 50 base rows</span></div>
        <div class="kpi"><span class="kpi-k">gap &#8805;</span>
          <span class="kpi-v">{hh["gap"]:.4f}</span>
          <span class="kpi-n">mean eval AUROC</span></div>
      </div>
      <div class="scroll">{ladder_table(hh)}</div>
      <p class="measure">Every rung improves on the one below it, and the top step is still
      worth <strong>+{hh["top_step"]:.4f}</strong>. Nothing here has flattened &#8212; the
      concept simply runs out of in-distribution rows at {hh["raw"]["n_eval_rows"]} eval and 218 dev.
      Read {hh["ceiling"]:.4f} as a floor under the true ceiling, and every gap below it as
      an underestimate.</p>
    </div>
  </div>

  <div class="callout"><strong>Why the asymmetry matters.</strong> High-stakes has five
  times the eval data. Where the ladder saturates, the distance from a probe to the ceiling
  is a real generalization gap. Where it does not, that distance is partly just the eval
  set being small &#8212; so the harmful-to-human numbers below are the more conservative
  of the two, not the more alarming.</div>
</section>

<section>
  <div class="sec-head">
    <div class="eyebrow">Part 2 &#8212; the sweep</div>
    <h2>What labelled in-distribution data buys</h2>
  </div>
  <p class="measure">Starting from red-team-only training data, add N rows drawn from the
  held-out dev set &#8212; same sources as the eval splits, verified disjoint from them
  &#8212; at ten equidistant points, three draw seeds each. The subsets are nested and
  stratified by label and split, so each line is a learning curve rather than ten unrelated
  draws. Two ways of spending the data, plus a control that spends no red-team data at all.
  Whiskers are the spread across the three seeds.</p>

  <figure>
    <div class="scroll">{line_chart(hs, 0.90, 0.99, "dev samples added to training")}</div>
    <ul class="legend">
      <li class="s-mix"><span class="swatch"></span>mixed into the red-team set</li>
      <li class="s-ft"><span class="swatch"></span>red-team first, then fine-tune</li>
      <li class="s-dev"><span class="swatch"></span>dev samples alone (control)</li>
    </ul>
    <figcaption><strong>high-stakes.</strong> The first {hs["points"][1]} samples are worth
    {hs["curve"]["mixed"][hs["points"][1]][0]-hs["n0"]:+.4f} &#8212; more than half the gap.
    Everything after that is a crawl: the remaining 1&#8239;272 samples buy about
    {hs_mix_v-hs["curve"]["mixed"][hs["points"][1]][0]:+.3f} more. Fine-tuning gets within
    0.01 of the ceiling at N={hs_ft_n}, where mixing needs the whole pool. The whiskers
    close at the last point because all three draw seeds select the same full pool
    there &#8212; it is one fit, not three that agreed.</figcaption>
  </figure>

  <div class="scroll">{curve_table(hs)}</div>

  <figure>
    <div class="scroll">{line_chart(hh, 0.89, 0.99, "dev samples added to training", dead_zone=MIN_STEP)}</div>
    <ul class="legend">
      <li class="s-mix"><span class="swatch"></span>mixed into the red-team set</li>
      <li class="s-ft"><span class="swatch"></span>red-team first, then fine-tune</li>
      <li class="s-dev"><span class="swatch"></span>dev samples alone (control)</li>
    </ul>
    <figcaption><strong>harmful-to-human.</strong> The whole dev pool is 218 rows, and it
    never gets close: the best point of any arm is {max(hh_mix_v, hh_dev_v):.4f} against a
    ceiling of at least {hh["ceiling"]:.4f}. The shaded band is the region where a training
    set is too small to take a single optimizer step &#8212; the control scores 0.5299 there,
    off the bottom of this scale, and fine-tuning returns its input probe unchanged.
    </figcaption>
  </figure>

  <div class="scroll">{curve_table(hh)}</div>

  <div class="pair">
    <div class="panel">
      <div class="panel-head"><h3>high-stakes</h3><span class="tag">gap {hs["gap"]:.4f}</span></div>
      <div class="scroll">{close_table(hs)}</div>
    </div>
    <div class="panel">
      <div class="panel-head"><h3>harmful-to-human</h3><span class="tag">gap &#8805; {hh["gap"]:.4f}</span></div>
      <div class="scroll">{close_table(hh)}</div>
    </div>
  </div>
</section>

<section>
  <div class="sec-head">
    <div class="eyebrow">the control</div>
    <h2>The red-team data stops mattering once real labels arrive</h2>
  </div>
  <p class="measure">The dashed line on both charts is the same sweep with the red-team
  training set removed entirely &#8212; N dev rows and nothing else. It exists because a
  rising curve on its own cannot tell you whether the dev samples carry the signal or
  whether the combination does. On both concepts it answers that question the same way.</p>

  <div class="kpis">
    <div class="kpi"><span class="kpi-k">high-stakes, full pool</span>
      <span class="kpi-v">{hs["curve"]["dev_only"][hs_dev_n][0]:.4f}</span>
      <span class="kpi-n">control, vs {hs["curve"]["mixed"][hs_dev_n][0]:.4f} mixed and
      {hs["curve"]["finetune"][hs_dev_n][0]:.4f} fine-tuned at the same N</span></div>
    <div class="kpi is-warn"><span class="kpi-k">harmful, full pool</span>
      <span class="kpi-v">{hh["curve"]["dev_only"][hh_dev_n][0]:.4f}</span>
      <span class="kpi-n">control <em>beats</em> both arms carrying red-team data
      ({hh["curve"]["mixed"][hh_dev_n][0]:.4f} and {hh["curve"]["finetune"][hh_dev_n][0]:.4f})</span></div>
    <div class="kpi"><span class="kpi-k">where it does help</span>
      <span class="kpi-v">N &#8804; {hs["points"][2]}</span>
      <span class="kpi-n">below a few hundred rows the red-team set is a real head start on
      both concepts</span></div>
  </div>

  <p class="measure">Read together with the curves: the red-team data is worth a lot at
  N&#8239;=&#8239;0 and decays from there. By the top of the high-stakes range the three arms
  sit within {max(abs(hs["curve"][a][hs_dev_n][0]-hs["curve"]["dev_only"][hs_dev_n][0]) for a in ARMS):.4f}
  of each other &#8212; a spread comparable to the across-seed noise on individual points.
  On harmful-to-human the control finishes highest outright. This is the same result the
  instruction-following analysis reached independently, on a third concept and a different
  attacker rotation.</p>
</section>

<section>
  <div class="sec-head">
    <div class="eyebrow">where the gap lives</div>
    <h2>One split per concept carries almost all of it</h2>
  </div>
  <p class="measure">The mean over splits hides the shape. Solid bar: what the red-team-only
  probe reaches on that split. Gold segment and tick: the distance to the cross-validated
  ceiling. Both concepts have a single split doing most of the damage, and in both cases it
  is the one with the least regular structure &#8212; tool-call transcripts for high-stakes,
  the unpaired Anthropic HH split for harm.</p>

  <div class="pair">
    <figure>
      <div class="scroll">{gap_bars(hs, HS_PRETTY)}</div>
      <figcaption><strong>high-stakes.</strong> <span class="mono">mts</span> is nearly
      solved at N=0. <span class="mono">toolace</span> is
      {hs["ceil_per_split"]["toolace_balanced"]-hs["n0_per_split"]["toolace_balanced"]:.3f} short
      and <span class="mono">mt</span> {hs["ceil_per_split"]["mt_balanced"]-hs["n0_per_split"]["mt_balanced"]:.3f}
      short &#8212; together they are the whole gap.</figcaption>
    </figure>
    <figure>
      <div class="scroll">{gap_bars(hh, HH_PRETTY)}</div>
      <figcaption><strong>harmful-to-human.</strong>
      <span class="mono">anthropic hh</span> is
      {hh["ceil_per_split"]["eval_ant_hh"]-hh["n0_per_split"]["eval_ant_hh"]:.3f} short of its
      ceiling &#8212; it is also the only one of the four splits whose rows are not paired
      prompt-for-prompt across the two classes.</figcaption>
    </figure>
  </div>
</section>

<section>
  <div class="sec-head">
    <div class="eyebrow">for scale</div>
    <h2>What the loop actually shipped</h2>
  </div>
  <p class="measure">The probes those red-team runs produced, scored on the same eval splits
  by their own comparison CSVs. These are ensembles where noted; every number elsewhere on
  this page is a single probe, so the comparison is indicative rather than like-for-like
  &#8212; an ensemble is worth roughly the difference you see between these and the
  N&#8239;=&#8239;0 refits.</p>
  <div class="pair">
    <div class="panel">
      <div class="panel-head"><h3>high-stakes</h3><span class="tag">ceiling {hs["ceiling"]:.4f}</span></div>
      <div class="scroll">{runs_table(hs_runs)}</div>
    </div>
    <div class="panel">
      <div class="panel-head"><h3>harmful-to-human</h3><span class="tag">ceiling &#8805; {hh["ceiling"]:.4f}</span></div>
      <div class="scroll">{runs_table(hh_runs)}</div>
    </div>
  </div>
  <div class="callout is-accent"><strong>The exchange rate.</strong> Five iterations of
  red-teaming &#8212; an attacker rotation, a judge on every candidate, contrastive
  preprocessing and a retrain per cycle &#8212; moved high-stakes from
  {hs_runs[3][1]:.4f} to {hs_runs[0][1]:.4f} at its best round. On these curves,
  {hs["points"][1]} labelled in-distribution rows are worth
  {hs["curve"]["mixed"][hs["points"][1]][0]:.4f}, fitted in about a minute on activations
  that were already cached.</div>
</section>

<section>
  <div class="sec-head">
    <div class="eyebrow">findings</div>
    <h2>What to take away</h2>
  </div>
  <div class="findings">
    <div class="finding"><span class="pill p-result">result</span><div>
      <h3>Red-team training leaves 8 points of AUROC unclaimed</h3>
      <p>{hs["gap"]:.4f} on high-stakes against a saturated ceiling, and at least
      {hh["gap"]:.4f} on harmful-to-human. Neither is a capacity limit: the same head, the
      same layer and the same 27B model reach the ceiling when the training rows come from
      the eval distribution. Every point of it is a generalization gap.</p></div></div>
    <div class="finding"><span class="pill p-result">result</span><div>
      <h3>A few hundred in-distribution rows do most of the closing</h3>
      <p>On high-stakes, {hs["points"][1]} dev rows close
      {(hs["curve"]["mixed"][hs["points"][1]][0]-hs["n0"])/hs["gap"]*100:.0f}% of the gap and
      the remaining 1&#8239;272 close another
      {(hs_mix_v-hs["curve"]["mixed"][hs["points"][1]][0])/hs["gap"]*100:.0f}%. The curve is
      steep then flat, so the useful question is not <em>how much</em> labelled data to
      collect but <em>whether any is obtainable at all</em>.</p></div></div>
    <div class="finding"><span class="pill p-result">result</span><div>
      <h3>Fine-tuning afterwards beats folding the data in</h3>
      <p>On high-stakes, fine-tuning reaches within 0.01 of the ceiling at N={hs_ft_n} where
      mixing needs all {hs_dev_n}. The mechanism is plain: a few hundred dev rows are a
      minority inside 892 red-team rows, but they are the entire signal in a fine-tune. The
      per-point differences are small &#8212; read the direction, not any single
      number.</p></div></div>
    <div class="finding"><span class="pill p-limit">limit</span><div>
      <h3>The control matches or beats both arms at the top of the range</h3>
      <p>Dev rows alone reach {hs["curve"]["dev_only"][hs_dev_n][0]:.4f} on high-stakes and
      {hh["curve"]["dev_only"][hh_dev_n][0]:.4f} on harmful-to-human &#8212; the latter above
      every condition that included red-team data. Once in-distribution labels exist the
      red-team set stops contributing. Its value lies in the case where they do
      not.</p></div></div>
    <div class="finding"><span class="pill p-limit">limit</span><div>
      <h3>Training sets under 64 rows never take an optimizer step</h3>
      <p><code>batch_size 16</code> with <code>gradient_accumulation_steps 4</code> means
      <code>optimizer.step()</code> fires only when <code>(batch_idx+1) % 4 == 0</code>. At
      24 or 48 rows that condition never holds, the fit burns its epoch budget changing
      nothing, and the head comes back at its random initialisation &#8212; 0.5299 for the
      harmful-to-human control. This is tuberlens' own loop, not an artifact of the
      analysis, and it affects any small-data fit in the repo.</p></div></div>
    <div class="finding"><span class="pill p-method">method</span><div>
      <h3>The validation set stops training but never restores a checkpoint</h3>
      <p><code>PytorchAdamClassifier.train</code> saves its best state with
      <code>state_dict().copy()</code>, which is <em>shallow</em> &#8212; the entries are the
      live parameter tensors, which go on training. The closing
      <code>load_state_dict</code> therefore copies every parameter onto itself and the
      probe returned is the <em>last</em> epoch's, not the best one. Patience still stops
      training the right number of epochs after the best, so validation is far from inert.
      Every probe in the experiment runs was produced this way, so this analysis reproduces
      it rather than fixing it &#8212; a corrected version would not be comparable to the
      runs it exists to explain.</p></div></div>
    <div class="finding"><span class="pill p-method">method</span><div>
      <h3>Extraction is bit-exact at batch size 1, and only at batch size 1</h3>
      <p>The same conversation extracted twice agrees exactly, and agrees exactly with the
      published activation blobs the probes are scored against. Raising the batch size to 4
      is eight times faster and moves every activation by 1.0e-2 relative &#8212; bf16
      reduction order, not padding, since four identical copies drift by the same amount.
      Batching would have perturbed one side of the training data and not the other, so the
      1&#8239;870 red-team conversations were extracted one at a time.</p></div></div>
  </div>
</section>

<footer>
  every fit runs on precomputed layer-32 activations &#183; the 27B model is loaded once, to extract<br>
  validation is a fixed 25% stratified slice of dev, never trained on, identical at every point<br>
  both ceilings reproduce a prior run of the same code to four decimals on every split<br>
  ceiling_analysis &#183; branch ceiling_analysis
</footer>
</div>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT} ({len(HTML)/1024:.1f} KB)")
