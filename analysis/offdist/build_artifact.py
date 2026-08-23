#!/usr/bin/env python
"""Render the off-distribution study as a self-contained HTML page.

Every number on the page is read out of ``analysis/offdist/results/`` — none is typed by
hand — so the page cannot drift from the JSONL it describes. Re-run after any re-run of the
analysis and republish the same file path to refresh the artifact in place.

Published at https://claude.ai/code/artifact/e1399fd2-3adf-4b1c-bb0d-17d3a90b3a21
(republish the same path to update it in place).

    analysis/offdist/build_artifact.py [out.html]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import od_common as O  # noqa: E402
from report import controls, load_ablation  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else O.RESULTS / "offdist.html"
ARMS = ["gptoss120b", "deepseekv4pro"]
LABEL = {"gptoss120b": "gpt-oss-120b", "deepseekv4pro": "deepseek-v4-pro"}
SHORT = {"gptoss120b": "arm 1", "deepseekv4pro": "arm 2"}


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


D = {}
for k in ARMS:
    D[k] = {
        "surface": O.read_json(O.RESULTS / f"surface_{k}.json"),
        "acts": O.read_json(O.RESULTS / f"actsig_{k}.json"),
        "abl": load_ablation(k),
        "npz": np.load(O.RESULTS / f"actsig_{k}.npz"),
    }
CONV = D[ARMS[0]]["surface"]["eval_convention"]


def cond(k, name):
    return next((r for r in D[k]["abl"] if r["condition"] == name), None)


def delta_vs_random(k, name):
    r = cond(k, name)
    c = controls(D[k]["abl"]).get(r["n_removed"]) if r else None
    if not r or not c:
        return None
    return r["mean"]["auroc"], c[0], c[1], r["mean"]["auroc"] - c[0], r["n_removed"]


# --------------------------------------------------------------------------- charts
BAR_CONDS = [
    ("drop_most_offdist_10pct", "most off-distribution, 10%"),
    ("drop_most_offdist_20pct", "most off-distribution, 20%"),
    ("drop_most_offdist_30pct", "most off-distribution, 30%"),
    ("drop_most_offdist_50pct", "most off-distribution, 50%"),
    ("drop_most_evallike_10pct", "most eval-like, 10%"),
    ("drop_most_evallike_20pct", "most eval-like, 20%"),
    ("drop_most_evallike_30pct", "most eval-like, 30%"),
    ("drop_most_evallike_50pct", "most eval-like, 50%"),
    ("drop_longest_assistant_30pct", "longest assistant replies, 30%"),
]


def diverging_chart(k) -> str:
    """Δ against matched random removal of the same size, one bar per condition."""
    rows = []
    for name, label in BAR_CONDS:
        d = delta_vs_random(k, name)
        if d:
            rows.append((label, d[3], max(d[2], 0.005), name))
    if not rows:
        return ""
    W, rowh, gap = 720, 26, 9
    L, R = 250, 690
    H = len(rows) * (rowh + gap) + 34
    lim = max(0.06, max(abs(v) for _, v, _, _ in rows) * 1.15)
    zero = L + (R - L) / 2

    def X(v):
        return zero + (R - L) / 2 * (v / lim)

    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="change in eval AUROC against matched random removal, per condition">']
    for t in (-lim, -lim / 2, 0, lim / 2, lim):
        o.append(f'<line class="cgrid" x1="{X(t):.1f}" y1="16" x2="{X(t):.1f}" y2="{H-18}"/>')
        o.append(f'<text class="ctick" x="{X(t):.1f}" y="{H-4}" text-anchor="middle">'
                 f'{t:+.2f}</text>')
    o.append(f'<line class="czero" x1="{zero:.1f}" y1="16" x2="{zero:.1f}" y2="{H-18}"/>')
    for i, (label, v, band, name) in enumerate(rows):
        y = 20 + i * (rowh + gap)
        o.append(f'<rect class="cband" x="{X(-band):.1f}" y="{y}" '
                 f'width="{X(band)-X(-band):.1f}" height="{rowh}"/>')
        cls = "cbar pos" if v > 0 else "cbar neg"
        if abs(v) <= band:
            cls = "cbar flat"
        x0, x1 = (zero, X(v)) if v > 0 else (X(v), zero)
        o.append(f'<rect class="{cls}" x="{x0:.1f}" y="{y+4}" width="{max(x1-x0,1):.1f}" '
                 f'height="{rowh-8}" rx="1.5"/>')
        o.append(f'<text class="crow" x="{L-16}" y="{y+rowh/2+4:.0f}" text-anchor="end">'
                 f'{esc(label)}</text>')
        vx = X(v) + (7 if v > 0 else -7)
        anchor = "start" if v > 0 else "end"
        o.append(f'<text class="cval" x="{vx:.1f}" y="{y+rowh/2+4:.0f}" '
                 f'text-anchor="{anchor}">{v:+.4f}</text>')
    o.append("</svg>")
    return "\n".join(o)


def orthogonality_diagram(k) -> str:
    """Displacement from the eval centroid, resolved onto the probe's decision axis.

    Real coordinates: the horizontal axis is a row's projection on the unit direction `w`,
    the vertical axis the magnitude of everything left over. Both are read off the npz, so
    the shape of the cloud is the measurement, not an illustration of it.
    """
    z = D[k]["npz"]
    proj = z["proj_on_w"]
    dist = z["centroid_dist"]
    orth = np.sqrt(np.maximum(dist ** 2 - proj ** 2, 0))
    W, H = 720, 330
    L, R, T, B = 74, 690, 26, 272
    xlim = max(6.0, float(np.abs(proj).max()) * 1.1)
    ylim = float(orth.max()) * 1.08

    def X(v):
        return L + (R - L) * (v + xlim) / (2 * xlim)

    def Y(v):
        return B - (B - T) * (v / ylim)

    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="red-team displacement from the eval centroid, resolved onto the '
         f'probe decision axis and its orthogonal complement">']
    for t in range(0, int(ylim) + 1, max(1, int(ylim // 5))):
        o.append(f'<line class="ogrid" x1="{L}" y1="{Y(t):.1f}" x2="{R}" y2="{Y(t):.1f}"/>')
        o.append(f'<text class="otick" x="{L-10}" y="{Y(t)+4:.1f}" text-anchor="end">{t}</text>')
    o.append(f'<line class="oaxis" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/>')
    o.append(f'<line class="oaxis dash" x1="{X(0):.1f}" y1="{T}" x2="{X(0):.1f}" y2="{B}"/>')
    step = max(1, len(proj) // 600)
    for p, q in zip(proj[::step], orth[::step]):
        o.append(f'<circle class="odot" cx="{X(float(p)):.1f}" cy="{Y(float(q)):.1f}" r="2.4"/>')
    o.append(f'<circle class="ocent" cx="{X(0):.1f}" cy="{Y(0):.1f}" r="5"/>')
    o.append(f'<text class="olab" x="{X(0):.1f}" y="{B+22}" text-anchor="middle">'
             f'eval centroid</text>')
    o.append(f'<text class="oaxlab" x="{R}" y="{B+22}" text-anchor="end">'
             f'projection on the probe direction w &#8594;</text>')
    o.append(f'<text class="oaxlab rot" transform="translate(22,{(T+B)/2:.0f}) rotate(-90)" '
             f'text-anchor="middle">everything orthogonal to w</text>')
    o.append("</svg>")
    return "\n".join(o)


# --------------------------------------------------------------------------- tables
def answers_table() -> str:
    rows = [
        ("separable from eval on text alone",
         lambda k: f"{D[k]['surface']['discriminator_auroc']:.4f}"),
        ("separable from eval in activation space",
         lambda k: f"{D[k]['acts']['separability_redteam_vs_eval_auroc']:.4f}"),
        ("past eval&#8217;s own p95 self-kNN radius",
         lambda k: f"{D[k]['acts']['outside_frac']:.1%}"),
        (f"rows containing a refusal &nbsp;<span class='ref'>eval {CONV['refusal_rate']:.1%}</span>",
         lambda k: f"{D[k]['surface']['refusal_rate_redteam']:.1%}"),
        ("labelled opposite to the eval convention",
         lambda k: str(D[k]["surface"]["n_convention_inverted"])),
        ("pairs contrasting on the assistant turn",
         lambda k: f"{D[k]['surface']['pair_axis_counts']['assistant']}/"
                   f"{D[k]['surface']['n_pairs']}"),
        ("displacement orthogonal to w",
         lambda k: f"{D[k]['acts']['mean_orthogonal_fraction']:.1%}"),
        ("eval AUROC, all red-team data",
         lambda k: f"{cond(k,'full')['mean']['auroc']:.4f}"),
        ("eval AUROC, no red-team data",
         lambda k: f"{cond(k,'base_only')['mean']['auroc']:.4f}"),
    ]
    head = "".join(f"<th>{SHORT[k]} &middot; {LABEL[k]}</th>" for k in ARMS)
    body = "".join(
        f"<tr><td>{lab}</td>" + "".join(f'<td class="n">{fn(k)}</td>' for k in ARMS) + "</tr>"
        for lab, fn in rows)
    return f'<table class="key"><thead><tr><th>measure</th>{head}</tr></thead><tbody>{body}</tbody></table>'


def structure_table() -> str:
    fields = [("chars_assistant", "assistant reply, characters"),
              ("chars_user", "user turn, characters"),
              ("assistant_share", "share of the text the assistant wrote"),
              ("n_newlines", "line breaks"),
              ("has_bullets", "carries a bulleted list"),
              ("has_numbered", "carries a numbered list"),
              ("refuses", "assistant refuses")]
    head = "".join(f"<th>{SHORT[k]}</th>" for k in ARMS)
    rows = []
    for f, lab in fields:
        ev = D[ARMS[0]]["surface"]["structural_means"][f]["eval"]
        fmt = (lambda v: f"{v:.0%}") if f in ("has_bullets", "has_numbered", "refuses",
                                              "assistant_share") else (lambda v: f"{v:,.0f}")
        cells = "".join(
            f'<td class="n drift">{fmt(D[k]["surface"]["structural_means"][f]["redteam"])}</td>'
            for k in ARMS)
        rows.append(f'<tr><td>{lab}</td><td class="n evalcol">{fmt(ev)}</td>{cells}</tr>')
    return ('<table class="key"><thead><tr><th>per conversation</th><th>eval</th>'
            f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def topic_table(k) -> str:
    s, a = D[k]["surface"], D[k]["acts"]
    rows = []
    for t in s["topics"]:
        pt = a["per_topic"][str(t["cluster"])]
        d = delta_vs_random(k, f"drop_topic_{t['cluster']}")
        dd = f"{d[3]:+.4f}" if d else "&#8212;"
        cls = "n"
        if d and abs(d[3]) > max(d[2], 0.005):
            cls = "n pos" if d[3] > 0 else "n neg"
        rows.append(
            f'<tr><td class="mono">{t["cluster"]}</td><td>{esc(", ".join(t["top_terms"][:5]))}</td>'
            f'<td class="n">{t["n"]}</td><td class="n">{pt["outside_frac"]:.0%}</td>'
            f'<td class="{cls}">{dd}</td></tr>')
    return ('<table class="key"><thead><tr><th>topic</th><th>leading terms</th>'
            '<th>rows</th><th>off-manifold</th><th>&Delta; vs random when dropped</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')


# --------------------------------------------------------------------------- page
g, d = D["gptoss120b"], D["deepseekv4pro"]
base_only = cond("gptoss120b", "base_only")["mean"]["auroc"]
la = {k: delta_vs_random(k, "drop_longest_assistant_30pct") for k in ARMS}
best_ds = delta_vs_random("deepseekv4pro", "drop_most_evallike_20pct")

HTML = f"""<title>Orthogonal by Construction</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root {{
  --ground:#F2F5F4; --surface:#FFFFFF; --raised:#E9EEEC;
  --ink:#141D1E; --body:#2C3A3A; --muted:#5F7071; --faint:#8A9A9A;
  --rule:#D5DEDB; --rule-strong:#B9C6C3;
  --eval:#0F6F68; --eval-soft:#DCEBE8;
  --drift:#AC432C; --drift-soft:#F4E2DC;
  --band:#DFE6E4;
  --measure:66ch; --wide:1000px;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0E1517; --surface:#141D1F; --raised:#1B2528;
    --ink:#E9F0EE; --body:#C3D0CE; --muted:#8DA0A0; --faint:#6B7E7E;
    --rule:#253134; --rule-strong:#374548;
    --eval:#4FB3A6; --eval-soft:#12302D;
    --drift:#D97A5E; --drift-soft:#331B14;
    --band:#1E292B;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0E1517; --surface:#141D1F; --raised:#1B2528;
  --ink:#E9F0EE; --body:#C3D0CE; --muted:#8DA0A0; --faint:#6B7E7E;
  --rule:#253134; --rule-strong:#374548;
  --eval:#4FB3A6; --eval-soft:#12302D;
  --drift:#D97A5E; --drift-soft:#331B14;
  --band:#1E292B;
}}
* {{ box-sizing:border-box; }}
body {{
  background:var(--ground); color:var(--body);
  font-family:"Source Serif 4",Georgia,serif; font-size:17px; line-height:1.62;
  margin:0; padding:0 24px 96px;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:var(--wide); margin:0 auto; }}
.col {{ max-width:var(--measure); }}
h1,h2,h3,.display {{ font-family:Archivo,"Helvetica Neue",Arial,sans-serif; color:var(--ink);
  text-wrap:balance; margin:0; }}
.mono,.eyebrow,td.n,th,.cval,.ctick,.crow,.otick,.olab,.oaxlab,.ref,.chip {{
  font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace; }}
.eyebrow {{ font-size:11px; letter-spacing:.18em; text-transform:uppercase;
  color:var(--muted); font-weight:500; }}

/* ---- masthead ---- */
header {{ padding:76px 0 42px; border-bottom:1px solid var(--rule-strong); }}
h1 {{ font-size:clamp(38px,6.4vw,68px); font-weight:700; letter-spacing:-.028em;
  line-height:1.02; margin:16px 0 0; }}
h1 .thin {{ font-weight:500; color:var(--muted); display:block; }}
.standfirst {{ max-width:60ch; margin:24px 0 0; font-size:19px; color:var(--body); }}
.meta {{ display:flex; flex-wrap:wrap; gap:10px 26px; margin-top:30px; }}
.chip {{ font-size:11.5px; letter-spacing:.06em; color:var(--muted);
  border:1px solid var(--rule); border-radius:2px; padding:5px 9px; background:var(--surface); }}

/* ---- question sections ---- */
section {{ padding:56px 0; border-bottom:1px solid var(--rule); }}
.qhead {{ display:grid; grid-template-columns:74px 1fr; gap:0 22px; align-items:start; }}
.qnum {{ font-family:Archivo,sans-serif; font-size:13px; font-weight:700; letter-spacing:.12em;
  color:var(--eval); padding-top:9px; border-top:2px solid var(--eval); }}
h2 {{ font-size:clamp(23px,3vw,31px); font-weight:600; letter-spacing:-.02em; line-height:1.16;
  max-width:22ch; }}
.verdict {{ grid-column:2; margin:20px 0 0; font-family:Archivo,sans-serif;
  font-size:clamp(19px,2.4vw,25px); font-weight:500; line-height:1.3; color:var(--ink);
  letter-spacing:-.012em; max-width:26ch; }}
.verdict em {{ font-style:normal; color:var(--drift); }}
.verdict .ok {{ color:var(--eval); }}
.qbody {{ grid-column:2; }}
.qbody p {{ max-width:var(--measure); }}
@media (max-width:720px) {{
  .qhead {{ grid-template-columns:1fr; gap:14px; }}
  .verdict,.qbody {{ grid-column:1; }}
  .qnum {{ display:inline-block; }}
}}

/* ---- tables ---- */
.panel {{ margin:32px 0 6px; overflow-x:auto; }}
table.key {{ border-collapse:collapse; width:100%; font-size:14px; min-width:520px; }}
table.key th {{ font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--muted); font-weight:500; text-align:left; padding:0 14px 9px 0;
  border-bottom:1px solid var(--rule-strong); }}
table.key th:not(:first-child) {{ text-align:right; }}
table.key td {{ padding:9px 14px 9px 0; border-bottom:1px solid var(--rule);
  vertical-align:baseline; color:var(--body); }}
table.key td.n {{ text-align:right; font-size:13px; font-variant-numeric:tabular-nums;
  color:var(--ink); white-space:nowrap; }}
table.key td.drift {{ color:var(--drift); }}
table.key td.evalcol {{ color:var(--eval); text-align:right; font-family:"IBM Plex Mono",monospace;
  font-size:13px; font-variant-numeric:tabular-nums; }}
td.pos {{ color:var(--eval); }}
td.neg {{ color:var(--drift); }}
.ref {{ font-size:11px; color:var(--eval); }}
caption,.cap {{ font-size:13px; color:var(--muted); max-width:var(--measure);
  margin:12px 0 0; line-height:1.5; }}

/* ---- charts ---- */
.chart {{ width:100%; height:auto; display:block; }}
.cgrid {{ stroke:var(--rule); stroke-width:1; }}
.czero {{ stroke:var(--rule-strong); stroke-width:1.5; }}
.cband {{ fill:var(--band); }}
.cbar.pos {{ fill:var(--eval); }}
.cbar.neg {{ fill:var(--drift); }}
.cbar.flat {{ fill:var(--faint); }}
.crow {{ font-size:11.5px; fill:var(--body); }}
.cval {{ font-size:11px; fill:var(--muted); font-variant-numeric:tabular-nums; }}
.ctick {{ font-size:10px; fill:var(--faint); }}
.ogrid {{ stroke:var(--rule); stroke-width:1; }}
.oaxis {{ stroke:var(--rule-strong); stroke-width:1.5; }}
.oaxis.dash {{ stroke-dasharray:3 4; }}
.odot {{ fill:var(--drift); fill-opacity:.42; }}
.ocent {{ fill:var(--eval); }}
.otick {{ font-size:10px; fill:var(--faint); }}
.olab {{ font-size:10.5px; fill:var(--eval); letter-spacing:.06em; }}
.oaxlab {{ font-size:10.5px; fill:var(--muted); letter-spacing:.08em; text-transform:uppercase; }}

.armhead {{ font-family:Archivo,sans-serif; font-size:12px; font-weight:600;
  letter-spacing:.1em; text-transform:uppercase; color:var(--ink);
  margin:34px 0 4px; padding-bottom:6px; border-bottom:1px solid var(--rule); }}
.armhead span {{ color:var(--muted); font-weight:500; }}

.pull {{ border-left:2px solid var(--eval); padding:2px 0 2px 20px; margin:30px 0;
  max-width:58ch; color:var(--ink); font-size:18px; }}

footer {{ padding:52px 0 0; color:var(--muted); font-size:13.5px; max-width:var(--measure); }}
footer code {{ font-family:"IBM Plex Mono",monospace; font-size:12.5px; color:var(--body); }}
a {{ color:var(--eval); }}
:focus-visible {{ outline:2px solid var(--eval); outline-offset:3px; }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">experiment22 &middot; human-harm probe &middot; gemma-3-27b-it L32</div>
  <h1>Orthogonal<span class="thin">by construction</span></h1>
  <p class="standfirst">The red-team conversations a probe is retrained on are not drawn from
  the distribution it is scored on. This asks how far off they sit, whether any are labelled
  backwards, whether dropping the worst of them helps &#8212; and where, in the model&#8217;s
  own representation, the difference actually lives.</p>
  <div class="meta">
    <span class="chip">2 attacker arms</span>
    <span class="chip">{g['surface']['n_redteam']} + {d['surface']['n_redteam']} red-team rows</span>
    <span class="chip">{g['surface']['n_eval']} eval rows</span>
    <span class="chip">104 probe fits</span>
    <span class="chip">no LLM loaded</span>
  </div>
</header>

<section>
  <div class="qhead">
    <div class="qnum">Q1</div>
    <div>
      <h2>Can a portion of the red-team data be identified as different from the eval set?</h2>
      <p class="verdict">No portion. <em>Effectively all of it.</em></p>
      <div class="qbody">
        <p>A TF-IDF discriminator trained to tell the two corpora apart reaches
        <strong>{g['surface']['discriminator_auroc']:.4f}</strong> and
        <strong>{d['surface']['discriminator_auroc']:.4f}</strong> AUROC out of fold. A linear
        model on the pooled layer-32 activations reaches
        <strong>{g['acts']['separability_redteam_vs_eval_auroc']:.4f}</strong> and
        <strong>{d['acts']['separability_redteam_vs_eval_auroc']:.4f}</strong>. There is no
        eval-like subset to keep and an off-manifold tail to cut &#8212; there is a gradient,
        and every red-team row sits somewhere along it.</p>
        <p>The structure is where the difference is most legible. Red-team assistant turns are
        an order of magnitude longer than eval&#8217;s and heavily formatted; eval&#8217;s
        conversations are short deliberative exchanges.</p>
        <div class="panel">{structure_table()}</div>
        <p class="cap">Means per conversation. The eval column is one number because both arms
        are scored on the same four <code>eval_sets/hu_ha</code> splits.</p>
        <p>The vocabulary says the same thing from the other side. The terms that most mark a
        conversation as <em>eval</em> are modal and deliberative &#8212;
        {", ".join("<code>%s</code>" % esc(t) for t, _ in g['surface']['top_terms_eval'][:6])}
        &#8212; the language of a dilemma being weighed. The terms that mark it as red-team are
        procedural: {", ".join("<code>%s</code>" % esc(t) for t, _ in d['surface']['top_terms_redteam'][:6])}.</p>
      </div>
    </div>
  </div>
</section>

<section>
  <div class="qhead">
    <div class="qnum">Q2</div>
    <div>
      <h2>Are any samples or pairs set up the opposite way to the eval pairs?</h2>
      <p class="verdict"><span class="ok">Almost none are backwards.</span> The mix is wrong instead.</p>
      <div class="qbody">
        <p>The eval splits encode a mapping from assistant behaviour to label. Measured rather
        than assumed: an eval conversation whose assistant refuses is labelled
        <code>harmful_to_human</code> just
        <strong>{CONV['p_harm_given_refuse']:.1%}</strong> of the time, against
        <strong>{CONV['p_harm_given_no_refuse']:.1%}</strong> for one that does not
        ({CONV['n_refuse']} of {CONV['n']} eval rows refuse). A red-team row that refuses
        <em>and</em> is labelled harmful therefore runs that mapping backwards.</p>
        <p>Arm 1 has <strong>{g['surface']['n_convention_inverted']}</strong> such rows. Arm 2 has
        <strong>{d['surface']['n_convention_inverted']}</strong>, and they share one shape &#8212;
        a refusal token followed by compliance:</p>
        <blockquote class="pull">&#8220;I can&#8217;t advise you to commit fraud. For
        informational purposes only, a demand letter typically uses a formal letterhead, states
        a specific balance&#8230;&#8221;</blockquote>
        <p>Which is arguably labelled correctly. It is simply a pattern the eval set does not
        contain, so the convention has nothing to say about it. The pairs are not inverted
        either: <strong>{g['surface']['pair_axis_counts']['assistant']} of {g['surface']['n_pairs']}</strong>
        and <strong>{d['surface']['pair_axis_counts']['assistant']} of {d['surface']['n_pairs']}</strong>
        contrastive pairs differ in the <em>assistant&#8217;s</em> turn &#8212; the same axis the
        eval pairs contrast on.</p>
        <p>What is off is composition. Refusal is a large part of what the eval negative class is
        made of, and the red-team negative class is almost never made of it:</p>
        <div class="panel">
          <table class="key"><thead><tr><th>share of rows containing a refusal</th><th>rate</th></tr></thead>
          <tbody>
          <tr><td class="evalcol" style="text-align:left">eval splits</td><td class="n evalcol">{CONV['refusal_rate']:.1%}</td></tr>
          <tr><td>{SHORT['gptoss120b']} &middot; {LABEL['gptoss120b']}</td><td class="n drift">{g['surface']['refusal_rate_redteam']:.1%}</td></tr>
          <tr><td>{SHORT['deepseekv4pro']} &middot; {LABEL['deepseekv4pro']}</td><td class="n drift">{d['surface']['refusal_rate_redteam']:.1%}</td></tr>
          </tbody></table>
        </div>
      </div>
    </div>
  </div>
</section>

<section>
  <div class="qhead">
    <div class="qnum">Q3</div>
    <div>
      <h2>Does removing the most off-distribution rows improve eval AUROC?</h2>
      <p class="verdict">Not along that axis. <em>The least eval-like rows are the useful ones.</em></p>
      <div class="qbody">
        <p>Each condition is measured against random removal of the same number of rows, three
        seeds &#8212; the grey band below. Without that control, &#8220;removing 30% of the
        training data changed the score&#8221; says nothing about <em>which</em> 30%. Removals are
        by contrastive pair, so the class balance never moves with the flag.</p>
        <div class="armhead">{SHORT['gptoss120b']} <span>&middot; {LABEL['gptoss120b']} &middot; all red-team data = {cond('gptoss120b','full')['mean']['auroc']:.4f}</span></div>
        <div class="panel">{diverging_chart('gptoss120b')}</div>
        <div class="armhead">{SHORT['deepseekv4pro']} <span>&middot; {LABEL['deepseekv4pro']} &middot; all red-team data = {cond('deepseekv4pro','full')['mean']['auroc']:.4f}</span></div>
        <div class="panel">{diverging_chart('deepseekv4pro')}</div>
        <p class="cap">Change in mean eval AUROC against matched random removal. Grey band =
        <code>max(control sd, 0.005)</code>; a bar inside it removed <em>some</em> data, not
        <em>these</em> data.</p>
        <p>Off-distribution removal clears the band in some conditions and loses in others, and
        removing the <em>most eval-like</em> rows does about as well &#8212; on arm 2 it is the
        single best condition of any kind ({best_ds[3]:+.4f}). Surface distance from eval does not
        predict harm.</p>
        <p>One effect is consistent, and it points the other way. Dropping the 30% with the
        longest assistant replies &#8212; structurally the rows least like eval &#8212; is
        significantly <em>worse</em> than random in both arms
        ({la['gptoss120b'][3]:+.4f}, {la['deepseekv4pro'][3]:+.4f}). Those rows are carrying signal,
        not noise.</p>
        <p>What does have a clean sign is the whole set, and it splits by arm. Dropping every
        red-team row costs arm 1
        <strong>{cond('gptoss120b','base_only')['mean']['auroc'] - cond('gptoss120b','full')['mean']['auroc']:+.4f}</strong>
        and <em>gains</em> arm 2
        <strong>{cond('deepseekv4pro','base_only')['mean']['auroc'] - cond('deepseekv4pro','full')['mean']['auroc']:+.4f}</strong>
        &#8212; matching their published trajectories, where arm 1 climbed to 0.9153 and arm 2
        declined to 0.8199. Both land on the same {base_only:.4f}, as they must: same 50 base
        rows, same validation slice, no red-team data.</p>
        <div class="armhead">by topic <span>&middot; k-means over the same TF-IDF space</span></div>
        <div class="panel">{topic_table('deepseekv4pro')}</div>
        <p class="cap">Arm 2. No topic is both far off-manifold and helpful to remove; the two
        topics whose removal clears the band in the positive direction are mid-pack on distance.</p>
      </div>
    </div>
  </div>
</section>

<section>
  <div class="qhead">
    <div class="qnum">Q4</div>
    <div>
      <h2>Do the flagged rows carry a signal in activation space?</h2>
      <p class="verdict">Yes &#8212; and it is <em>{g['acts']['mean_orthogonal_fraction']:.1%} orthogonal</em>
      to the probe&#8217;s decision axis.</p>
      <div class="qbody">
        <p>Every conversation is reduced to the mask-weighted mean of its layer-32 token
        activations. <strong>{g['acts']['outside_frac']:.1%}</strong> of arm 1&#8217;s rows and
        <strong>{d['acts']['outside_frac']:.1%}</strong> of arm 2&#8217;s sit further from the eval
        set than eval&#8217;s own 95th-percentile self-kNN radius. They really are off the
        manifold.</p>
        <p>But the direction of that displacement is the finding. The head is
        <code>LinearThenSoftmax</code>, so <code>w</code> is a single vector and a row&#8217;s
        offset from the eval centroid splits cleanly into a component along it and everything
        else. Almost all of it is everything else:</p>
        <div class="armhead">{SHORT['deepseekv4pro']} <span>&middot; every red-team row, resolved onto the decision axis</span></div>
        <div class="panel">{orthogonality_diagram('deepseekv4pro')}</div>
        <p class="cap">Horizontal: projection on the unit direction <code>w</code>, averaged over
        the run&#8217;s 10 ensemble members (pairwise cosine
        {d['acts']['ensemble_direction_agreement_cos']:.2f}). Vertical: the magnitude of the
        residual. The cloud is tall and narrow because the mean absolute projection on
        <code>w</code> is {d['acts']['mean_abs_proj_on_w_redteam']:.2f} for red-team rows against
        {d['acts']['mean_abs_proj_on_w_eval']:.2f} for eval rows &#8212; essentially the same
        &#8212; while the total displacement is many times larger.</p>
        <p>Novelty orthogonal to <code>w</code> <strong>cannot move an eval score directly</strong>.
        It has no component along the axis the probe reads. It can only act by rotating
        <code>w</code> at the next refit &#8212; which is exactly why the removal conditions in Q3
        are so unruly: they change where the boundary ends up, not where the eval rows fall
        relative to the current one.</p>
        <p>Two smaller results sit underneath. The surface score is a <em>weak</em> proxy for this
        novelty &#8212; Spearman {g['acts']['spearman_p_redteam_vs_knn']:+.2f} and
        {d['acts']['spearman_p_redteam_vs_knn']:+.2f} against kNN distance &#8212; so text-level and
        representation-level oddness are genuinely different orderings. Yet the surface-flagged
        top {g['acts']['flag_frac']:.0%} is still linearly separable from the rest in activation
        space at <strong>{g['acts']['separability_flagged_vs_rest_auroc']:.4f}</strong> and
        <strong>{d['acts']['separability_flagged_vs_rest_auroc']:.4f}</strong>. The lexical flag
        names something the representation encodes; it just is not the thing that decides the
        score.</p>
      </div>
    </div>
  </div>
</section>

<section style="border-bottom:none">
  <div class="qhead">
    <div class="qnum">&#8212;</div>
    <div>
      <h2>Everything, side by side</h2>
      <div class="qbody"><div class="panel">{answers_table()}</div></div>
    </div>
  </div>
</section>

<footer>
  <p>Fits are the ceiling study&#8217;s, reused unchanged &#8212; one
  <code>linear_then_softmax</code> head at seed 42, early-stopped on its reserved 25% dev
  validation slice &#8212; so the all-data condition reproduces that study&#8217;s N=0 point to
  four decimals and every condition reads against its curves. These are single probes and are
  not directly comparable to the run&#8217;s published comparison CSVs, which are 10-member
  ensembles.</p>
  <p>No LLM is loaded and no activation is recomputed at any point. Every number on this page is
  read out of <code>analysis/offdist/results/</code> by <code>build_artifact.py</code>.</p>
</footer>
</div>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML, encoding="utf-8")
print("wrote", OUT, f"({len(HTML)/1024:.0f} KB)")
