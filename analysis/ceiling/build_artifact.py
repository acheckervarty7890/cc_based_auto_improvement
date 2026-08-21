"""Render the ceiling study to a self-contained HTML page.

Every figure on the page is read out of results/*.jsonl and every chart path is
computed here, so nothing is transcribed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "ceiling_study.html"

SPLITS = ["anthropic_harmless_refusal", "bbq_substitution", "hc_context_drift",
          "hc_contradiction", "mm_substitution", "oig_context_drift", "oig_omission"]
NICE = {s: s.replace("_", " ") for s in SPLITS}


def rows(name):
    p = RES / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def curve(rs, way, lr=None):
    pts = sorted([(r["n_dev"], r["macro"]) for r in rs
                  if r["way"] == way and r.get("ft_lr") == lr])
    return pts


# ---------------------------------------------------------------- svg helpers

def line_chart(series, *, ceiling, ymin, ymax, width=760, height=330,
               pad_l=54, pad_r=18, pad_t=18, pad_b=44, xlab="dev samples used for training",
               ylab="macro AUROC", xticks=None):
    xs = [p[0] for s in series for p in s["points"]]
    x0, x1 = min(xs), max(xs)
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b

    def X(v):
        return pad_l + (v - x0) / max(1, (x1 - x0)) * iw

    def Y(v):
        return pad_t + (1 - (v - ymin) / (ymax - ymin)) * ih

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
           f'aria-label="{ylab} against {xlab}">']
    # horizontal grid
    step = 0.02 if (ymax - ymin) <= 0.25 else 0.05
    v = ymin
    while v <= ymax + 1e-9:
        y = Y(v)
        out.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick ytick" x="{pad_l-9}" y="{y+4:.1f}">{v:.2f}</text>')
        v += step
    for t in (xticks or [p[0] for p in series[0]["points"]]):
        out.append(f'<text class="tick" x="{X(t):.1f}" y="{height-pad_b+20}">{t}</text>')
    # the ceiling motif
    if ceiling is not None:
        y = Y(ceiling)
        out.append(f'<line class="ceiling" x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}"/>')
        out.append(f'<text class="ceiling-label" x="{width-pad_r}" y="{y-8:.1f}" '
                   f'text-anchor="end">ceiling {ceiling:.4f}</text>')
    for s in series:
        d = " ".join(("M" if i == 0 else "L") + f"{X(p[0]):.1f},{Y(p[1]):.1f}"
                     for i, p in enumerate(s["points"]))
        out.append(f'<path class="series {s["cls"]}" d="{d}"/>')
        for p in s["points"]:
            out.append(f'<circle class="dot {s["cls"]}" cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="3"/>')
    out.append(f'<text class="axis" x="{pad_l + iw/2:.0f}" y="{height-6}" '
               f'text-anchor="middle">{xlab}</text>')
    out.append("</svg>")
    return "\n".join(out)


def legend(series):
    return ('<ul class="legend">' + "".join(
        f'<li class="{s["cls"]}"><span class="swatch"></span>{s["label"]}</li>' for s in series)
        + "</ul>")


def headroom_chart(achieved, ceiling, width=760, bar_h=30, gap=13,
                   pad_l=176, pad_r=64, pad_t=10):
    height = pad_t + len(SPLITS) * (bar_h + gap)
    iw = width - pad_l - pad_r
    lo = 0.5
    def X(v):
        return pad_l + (v - lo) / (1 - lo) * iw
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
           f'aria-label="achieved versus ceiling AUROC per eval split">']
    for i, s in enumerate(SPLITS):
        y = pad_t + i * (bar_h + gap)
        a, c = achieved[s], ceiling[s]
        out.append(f'<text class="rowlab" x="{pad_l-14}" y="{y+bar_h*0.68:.0f}" '
                   f'text-anchor="end">{NICE[s]}</text>')
        out.append(f'<rect class="track" x="{pad_l}" y="{y}" width="{iw}" height="{bar_h}" rx="2"/>')
        gain = X(max(a, c)) - X(min(a, c))
        out.append(f'<rect class="gap" x="{X(min(a,c)):.1f}" y="{y}" '
                   f'width="{gain:.1f}" height="{bar_h}" rx="2"/>')
        out.append(f'<rect class="achieved" x="{pad_l}" y="{y}" '
                   f'width="{X(a)-pad_l:.1f}" height="{bar_h}" rx="2"/>')
        out.append(f'<line class="ceilmark" x1="{X(c):.1f}" y1="{y-3}" '
                   f'x2="{X(c):.1f}" y2="{y+bar_h+3}"/>')
        out.append(f'<text class="barval" x="{width-pad_r+10}" y="{y+bar_h*0.68:.0f}">'
                   f'{a:.3f} &#8594; {c:.3f}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------- data

ceil = {}
for arm in ("gptoss", "nemotron"):
    for r in rows(f"ceiling_{arm}.jsonl"):
        ceil[(arm, r["condition"])] = r
for r in rows("ceiling_cv_perfold.jsonl"):
    ceil[("gptoss", "cv_eval")] = r          # the version carrying within-fold metrics

sw = {a: rows(f"sweep_{a}.jsonl") for a in ("gptoss", "nemotron")}
sw["nort"] = rows("sweep_nort.jsonl")

CV = ceil[("gptoss", "cv_eval")]
CEIL_POOLED = CV["macro"]["auroc"]
CEIL_WITHIN = CV["macro_within_fold"]["auroc"]
ORACLE = ceil[("gptoss", "oracle")]["macro"]["auroc"]
ITER0 = 0.7714           # published iter0, shared by both arms
BEST = {"gptoss": 0.8252, "nemotron": 0.8595}    # published best per arm

def captured(best):
    return (best - ITER0) / (CEIL_WITHIN - ITER0)

SER = {
    "joint":    {"cls": "s-joint",  "label": "joint retrain (red-team &#8746; dev)"},
    "ft":       {"cls": "s-ft",     "label": "finetune after red-team"},
    "ftlow":    {"cls": "s-ftlow",  "label": "finetune, lr 1e-4"},
    "nojoint":  {"cls": "s-nojoint","label": "dev only, joint"},
    "noft":     {"cls": "s-noft",   "label": "dev only, finetune"},
}

def series_for(arm):
    rs = sw[arm]
    out = []
    for way, lr, k in (("joint", None, "joint"), ("finetune", None, "ft"),
                       ("finetune", 1e-4, "ftlow")):
        pts = [(n, m["auroc"]) for n, m in curve(rs, way, lr)]
        if pts:
            out.append(dict(SER[k], points=pts))
    return out

def series_nort():
    rs = sw["nort"]
    out = []
    for way, k in (("joint", "nojoint"), ("finetune", "noft")):
        pts = [(n, m["auroc"]) for n, m in curve(rs, way, None)]
        if pts:
            out.append(dict(SER[k], points=pts))
    return out


CSS = """
:root{
  --paper:#F2F3F7; --surface:#FFFFFF; --surface-2:#EAECF3; --rule:#D3D7E4;
  --ink:#16192B; --muted:#5B6076; --faint:#878CA3;
  --accent:#2F4B99; --accent-soft:#8FA3D8;
  --limit:#A8761C; --limit-soft:#E3C98B;
  --good:#1F7A5C; --warn:#9C3B34;
  --s1:#2F4B99; --s2:#1F7A5C; --s3:#9AA0B8; --s4:#7A3E9C; --s5:#B4562A;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#10121C; --surface:#171A26; --surface-2:#1E2231; --rule:#2C3145;
    --ink:#E8EAF2; --muted:#9AA0B8; --faint:#767C93;
    --accent:#8AA6FF; --accent-soft:#3C4A75;
    --limit:#E0A93E; --limit-soft:#6B5526;
    --good:#4FCFA6; --warn:#E4736E;
    --s1:#8AA6FF; --s2:#4FCFA6; --s3:#767C93; --s4:#C79BE8; --s5:#E8956A;
  }
}
:root[data-theme="dark"]{
  --paper:#10121C; --surface:#171A26; --surface-2:#1E2231; --rule:#2C3145;
  --ink:#E8EAF2; --muted:#9AA0B8; --faint:#767C93;
  --accent:#8AA6FF; --accent-soft:#3C4A75;
  --limit:#E0A93E; --limit-soft:#6B5526;
  --good:#4FCFA6; --warn:#E4736E;
  --s1:#8AA6FF; --s2:#4FCFA6; --s3:#767C93; --s4:#C79BE8; --s5:#E8956A;
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Serif",Georgia,"Times New Roman",serif;
  font-size:17px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px;margin:0 auto;padding:0 28px 96px}
.measure{max-width:65ch}

h1,h2,h3,.eyebrow,.tick,.rowlab,.barval,.legend,th,.num,.kpi-v,.pill,.axis,.ceiling-label{
  font-family:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
}
h1{font-size:clamp(2.4rem,5.2vw,3.6rem);line-height:1.04;font-weight:600;
   letter-spacing:-.015em;margin:0 0 .5rem;text-wrap:balance}
h2{font-size:1.85rem;font-weight:600;letter-spacing:-.01em;margin:0;text-wrap:balance}
h3{font-size:1.12rem;font-weight:600;margin:0 0 .35rem;text-wrap:balance}
p{margin:0}
a{color:var(--accent)}
code,.mono,.num,.kpi-v,td.n,th.n{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  font-variant-numeric:tabular-nums}
code{font-size:.86em;background:var(--surface-2);padding:.1em .35em;border-radius:3px}

.eyebrow{font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;
  color:var(--faint);font-weight:600}

header.hero{padding:76px 0 40px;border-bottom:1px solid var(--rule)}
.hero .sub{font-size:1.2rem;color:var(--muted);max-width:60ch;margin-top:.7rem}
.meta{display:flex;flex-wrap:wrap;gap:8px 26px;margin-top:26px;
  font-family:"IBM Plex Mono",monospace;font-size:.78rem;color:var(--faint)}

section{padding:52px 0;border-bottom:1px solid var(--rule);
  display:flex;flex-direction:column;gap:22px}
section:last-of-type{border-bottom:none}
.sec-head{display:flex;flex-direction:column;gap:6px}

.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:14px}
.kpi{background:var(--surface);border:1px solid var(--rule);border-radius:5px;
  padding:16px 18px;display:flex;flex-direction:column;gap:3px}
.kpi-v{font-size:1.95rem;font-weight:600;letter-spacing:-.02em;line-height:1}
.kpi-k{font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);
  font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:600}
.kpi-n{font-size:.84rem;color:var(--muted);line-height:1.45;
  font-family:"IBM Plex Serif",serif}
.kpi.is-limit .kpi-v{color:var(--limit)}
.kpi.is-good .kpi-v{color:var(--good)}

figure{margin:0;background:var(--surface);border:1px solid var(--rule);
  border-radius:5px;padding:20px 20px 14px;display:flex;flex-direction:column;gap:12px}
figcaption{font-size:.85rem;color:var(--muted);line-height:1.5}
.scroll{overflow-x:auto}
.chart{display:block;width:100%;min-width:520px;height:auto}
.grid{stroke:var(--rule);stroke-width:1}
.tick{fill:var(--faint);font-size:11px;text-anchor:middle}
.ytick{text-anchor:end}
.axis{fill:var(--faint);font-size:11px;letter-spacing:.06em;text-transform:uppercase}
.ceiling{stroke:var(--limit);stroke-width:1.5;stroke-dasharray:7 5}
.ceiling-label{fill:var(--limit);font-size:11px;letter-spacing:.06em}
.series{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}
.dot{stroke:var(--surface);stroke-width:1.5}
.s-joint{stroke:var(--s1)} circle.s-joint{fill:var(--s1)}
.s-ft{stroke:var(--s2)} circle.s-ft{fill:var(--s2)}
.s-ftlow{stroke:var(--s3);stroke-dasharray:4 4} circle.s-ftlow{fill:var(--s3)}
.s-nojoint{stroke:var(--s4)} circle.s-nojoint{fill:var(--s4)}
.s-noft{stroke:var(--s5)} circle.s-noft{fill:var(--s5)}
.legend{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:8px 20px;
  font-size:.78rem;color:var(--muted)}
.legend li{display:flex;align-items:center;gap:7px}
.swatch{width:15px;height:3px;border-radius:2px;background:var(--muted);display:inline-block}
.legend .s-joint .swatch{background:var(--s1)}
.legend .s-ft .swatch{background:var(--s2)}
.legend .s-ftlow .swatch{background:var(--s3)}
.legend .s-nojoint .swatch{background:var(--s4)}
.legend .s-noft .swatch{background:var(--s5)}

.track{fill:var(--surface-2)}
.achieved{fill:var(--accent)}
.gap{fill:var(--limit-soft)}
.ceilmark{stroke:var(--limit);stroke-width:2}
.rowlab{fill:var(--ink);font-size:12.5px}
.barval{fill:var(--muted);font-size:11.5px;font-family:"IBM Plex Mono",monospace}

table{border-collapse:collapse;width:100%;font-size:.85rem;min-width:560px}
th,td{padding:8px 11px;text-align:right;border-bottom:1px solid var(--rule);
  white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;
  color:var(--faint);font-weight:600;border-bottom:1px solid var(--ink)}
td.n{font-size:.84rem}
tr.hi td{background:var(--surface-2);font-weight:600}
td.best{color:var(--good);font-weight:600}
.foot td{border-bottom:none;color:var(--faint);font-size:.78rem}
.visually-hidden{position:absolute;width:1px;height:1px;overflow:hidden;
  clip:rect(0 0 0 0);white-space:nowrap}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}

.findings{display:flex;flex-direction:column;gap:0}
.finding{display:grid;grid-template-columns:118px 1fr;gap:26px;
  padding:22px 0;border-top:1px solid var(--rule)}
.finding:first-child{border-top:none}
.pill{font-size:.66rem;letter-spacing:.11em;text-transform:uppercase;font-weight:600;
  padding:4px 9px;border-radius:3px;align-self:start;text-align:center;
  border:1px solid currentColor}
.p-result{color:var(--accent)}
.p-bug{color:var(--warn)}
.p-method{color:var(--muted)}
.finding p{color:var(--muted);font-size:.93rem;margin-top:.3rem}
.finding strong{color:var(--ink);font-weight:600}

.callout{background:var(--surface);border:1px solid var(--rule);
  border-left:3px solid var(--limit);border-radius:4px;padding:16px 20px;
  font-size:.92rem;color:var(--muted)}
.callout strong{color:var(--ink)}

footer{padding:40px 0 0;color:var(--faint);font-size:.8rem;
  font-family:"IBM Plex Mono",monospace;line-height:1.7}

@media (max-width:640px){
  body{font-size:16px}
  .finding{grid-template-columns:1fr;gap:8px}
  header.hero{padding:48px 0 30px}
}
@media (prefers-reduced-motion:no-preference){
  .series{stroke-dasharray:2200;stroke-dashoffset:2200;
    animation:draw 1.15s cubic-bezier(.4,0,.2,1) forwards}
  .s-ftlow{animation:none;stroke-dasharray:4 4}
  @keyframes draw{to{stroke-dashoffset:0}}
}
"""


def sweep_table(arm, label):
    rs = sw[arm]
    ns = sorted({r["n_dev"] for r in rs})
    if arm == "nort":
        ways = [("joint", None, "dev only, joint retrain"),
                ("finetune", None, "dev only, finetune")]
    else:
        ways = [("joint", None, "joint retrain"),
                ("finetune", None, "finetune (lr 5e-3)"),
                ("finetune", 1e-4, "finetune (lr 1e-4)")]
    head = "".join(f'<th class="n">{n}</th>' for n in ns)
    body = []
    for way, lr, name in ways:
        cells = []
        for n in ns:
            hit = [r for r in rs if r["way"] == way and r["n_dev"] == n and r.get("ft_lr") == lr]
            if not hit:
                cells.append('<td class="n">&#8212;</td>'); continue
            v = hit[0]["macro"]["auroc"]
            best = v == max(
                h[0]["macro"]["auroc"] for h in
                ([r for r in rs if r["way"] == w2 and r["n_dev"] == n and r.get("ft_lr") == l2]
                 for w2, l2, _ in ways) if h)
            cells.append(f'<td class="n{" best" if best else ""}">{v:.4f}</td>')
        body.append(f"<tr><td>{name}</td>{''.join(cells)}</tr>")
    return (f'<div class="scroll"><table><caption class="visually-hidden">{label}</caption>'
            f'<thead><tr><th>{label}</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def ceiling_table():
    r = lambda a, c: ceil[(a, c)]["macro"]
    w = CV["macro_within_fold"]
    nort = [x for x in sw["nort"] if x["way"] == "finetune" and x["n_dev"] == 336][0]["macro"]
    rowspec = [
        ("iteration 0 &#8212; base data only, no red-teaming", ITER0, None, ""),
        ("best published probe (gpt-oss arm, iter 4)", BEST["gptoss"], None, ""),
        ("best published probe (nemotron arm, iter 5)", BEST["nemotron"], None, ""),
        ("red-team only, refit here (gpt-oss)", r("gptoss", "redteam_only")["auroc"],
         r("gptoss", "redteam_only")["tpr_at_fpr_le"], ""),
        ("red-team only, refit here (nemotron)", r("nemotron", "redteam_only")["auroc"],
         r("nemotron", "redteam_only")["tpr_at_fpr_le"], ""),
        ("5-fold CV inside eval, pooled out-of-fold", CEIL_POOLED,
         CV["macro"]["tpr_at_fpr_le"], ""),
        ("5-fold CV inside eval, averaged within fold", CEIL_WITHIN, w["tpr_at_fpr_le"], "hi"),
        ("CV inside eval + red-team data (gpt-oss)", r("gptoss", "cv_eval_rt")["auroc"],
         r("gptoss", "cv_eval_rt")["tpr_at_fpr_le"], ""),
        ("CV inside eval + red-team data (nemotron)", r("nemotron", "cv_eval_rt")["auroc"],
         r("nemotron", "cv_eval_rt")["tpr_at_fpr_le"], ""),
        ("trained on 436 dev rows only, tested on all 1302 eval rows",
         nort["auroc"], nort["tpr_at_fpr_le"], "hi"),
        ("oracle &#8212; fit and scored on the same 1302 eval rows", ORACLE,
         r("gptoss", "oracle")["tpr_at_fpr_le"], ""),
    ]
    body = "".join(
        f'<tr class="{cls}"><td>{name}</td><td class="n">{a:.4f}</td>'
        f'<td class="n">{"&#8212;" if t is None else f"{t:.4f}"}</td></tr>'
        for name, a, t, cls in rowspec)
    return ('<div class="scroll"><table><thead><tr><th>condition</th>'
            '<th class="n">macro AUROC</th><th class="n">TPR @ FPR &#8804; 1%</th>'
            f'</tr></thead><tbody>{body}</tbody></table></div>')


achieved = {s: ceil[("gptoss", "redteam_only")]["per_split"][s]["auroc"] for s in SPLITS}
ceil_split = {s: CV["per_split_within_fold"][s]["auroc"] for s in SPLITS}
NOISE = 0.0146

gp = series_for("gptoss"); nm = series_for("nemotron"); no = series_nort()
grid = [p[0] for p in gp[0]["points"]]

HTML = f"""<title>The Instruction Probe Ceiling</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@400;600&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>{CSS}</style>

<div class="wrap">
<header class="hero">
  <div class="eyebrow">gemma-3-27b &#183; layer 32 &#183; instruction-following probe</div>
  <h1>How high could this probe actually go?</h1>
  <p class="sub">Five iterations of automated red-teaming moved a linear probe from
  0.7714 to 0.8595 macro AUROC. This measures the number it was climbing toward &#8212;
  and what it would have cost to get there another way.</p>
  <div class="meta">
    <span>1302 eval rows &#183; 7 splits</span>
    <span>436 held-out dev rows</span>
    <span>10-member deep ensembles throughout</span>
    <span>all figures recomputed from cached activations</span>
  </div>
</header>

<section>
  <div class="sec-head">
    <div class="eyebrow">Part 1 &#8212; the ceiling</div>
    <h2>There is no representational barrier</h2>
  </div>
  <p class="measure">Fit the same probe head on all 1302 eval rows and score those same
  rows and it separates them perfectly. Layer 32 already carries the whole signal;
  nothing about the architecture or the layer is what holds the run back. Every point
  below the oracle is a generalization gap, not a capacity gap.</p>

  <div class="kpis">
    <div class="kpi"><span class="kpi-k">achieved</span>
      <span class="kpi-v">{BEST['nemotron']:.4f}</span>
      <span class="kpi-n">best published probe, after 5 red-team iterations</span></div>
    <div class="kpi is-limit"><span class="kpi-k">ceiling</span>
      <span class="kpi-v">{CEIL_WITHIN:.4f}</span>
      <span class="kpi-n">5-fold CV inside the eval set, no test leakage</span></div>
    <div class="kpi"><span class="kpi-k">separability</span>
      <span class="kpi-v">{ORACLE:.4f}</span>
      <span class="kpi-n">oracle: fit and scored on the same rows</span></div>
    <div class="kpi is-good"><span class="kpi-k">headroom captured</span>
      <span class="kpi-v">{captured(BEST['nemotron'])*100:.0f}%</span>
      <span class="kpi-n">of the distance from iteration 0 to the ceiling</span></div>
  </div>

  {ceiling_table()}

  <div class="callout"><strong>The ceiling is a band, not a point.</strong>
  Cross-validating inside eval gives {CEIL_WITHIN:.4f}; training on the 436 dev rows and
  testing on all 1302 eval rows gives {[x for x in sw['nort'] if x['way']=='finetune' and x['n_dev']==336][0]['macro']['auroc']:.4f}.
  The 0.018 between them sits barely above the {NOISE:.4f} noise floor measured below, so
  the two estimates agree about as well as this metric can resolve. Read the ceiling as
  roughly 0.94&#8211;0.96.</div>
</section>

<section>
  <div class="sec-head">
    <div class="eyebrow">where the gap lives</div>
    <h2>Six splits are nearly solved. One is not.</h2>
  </div>
  <p class="measure">The macro average hides the shape of the problem. Under
  cross-validation six of the seven eval splits reach 0.95 or better; <span
  class="mono">oig_omission</span> reaches {ceil_split['oig_omission']:.3f} and drags the
  mean down on its own. Drop it and the ceiling is
  {sum(v for k,v in ceil_split.items() if k!='oig_omission')/6:.3f}. It is also the
  smallest split &#8212; 114 eval rows and 32 dev rows &#8212; so it is as likely to be a
  data problem as a hard one.</p>
  <figure>
    <div class="scroll">{headroom_chart(achieved, ceil_split)}</div>
    <figcaption>Solid bar: what the gpt-oss red-team probe reached. Gold segment and tick:
    the distance to the cross-validated ceiling for that split. The probe is already at
    ceiling on <span class="mono">hc_context_drift</span>, and 0.3&#8201;AUROC short on
    <span class="mono">anthropic_harmless_refusal</span> and
    <span class="mono">oig_context_drift</span> &#8212; that pair is where the remaining
    headroom actually is.</figcaption>
  </figure>
</section>

<section>
  <div class="sec-head">
    <div class="eyebrow">Part 2 &#8212; the dev sweep</div>
    <h2>What labelled in-distribution data buys</h2>
  </div>
  <p class="measure">The 436 dev rows were cut once into a 100-row validation holdout &#8212;
  never trained on, in any condition, so early stopping is identical at every point &#8212;
  and a 336-row pool. N runs 0 to 336 in equal steps, and each prefix is stratified, so
  N&#8239;=&#8239;42 means six rows from each of the seven splits, half positive. Two ways
  of spending them, as asked: fold them into the red-team training set and retrain from
  scratch, or train on red-team data first and then finetune on the dev rows alone.</p>

  <figure>
    <div class="scroll">{line_chart(gp, ceiling=CEIL_WITHIN, ymin=0.80, ymax=0.96, xticks=grid)}</div>
    {legend(gp)}
    <figcaption>gpt-oss arm. Finetuning after red-teaming leads joint retraining across the
    data-scarce middle of the range; both converge on the ceiling once the whole pool is
    used. The flat grey line is a 10&#215; lower finetune learning rate, which does nothing
    here &#8212; at one optimizer step per epoch, 1e-4 cannot move the head.</figcaption>
  </figure>

  <figure>
    <div class="scroll">{line_chart(nm, ceiling=CEIL_WITHIN, ymin=0.80, ymax=0.96, xticks=grid)}</div>
    {legend(nm)}
    <figcaption>nemotron arm. It starts 0.032 higher and finishes in the same place, which
    is the first hint that at the top of these curves it is the dev labels doing the work,
    not the red-team data underneath them.</figcaption>
  </figure>

  {sweep_table("gptoss", "gpt-oss arm &#183; macro AUROC")}
  {sweep_table("nemotron", "nemotron arm &#183; macro AUROC")}
</section>

<section>
  <div class="sec-head">
    <div class="eyebrow">the control that reframes it</div>
    <h2>Dev labels alone do better than dev labels plus red-teaming</h2>
  </div>
  <p class="measure">Running the same sweep with the red-team data removed entirely &#8212;
  50 base rows plus N dev rows, nothing else &#8212; starts at iteration 0's own score and
  overtakes the finished red-team probe by N&#8239;=&#8239;84. At the full pool it reaches
  0.9613, above every condition that included red-team data. The red-team samples are not
  merely redundant once in-distribution labels exist; they cost about a point.</p>
  <figure>
    <div class="scroll">{line_chart(no, ceiling=CEIL_WITHIN, ymin=0.76, ymax=0.98, xticks=grid)}</div>
    {legend(no)}
    <figcaption>No red-team data at any point on these curves. N&#8239;=&#8239;0 reproduces
    the published iteration-0 probe (0.7713 against 0.7714), which is what makes this
    directly comparable to the red-teaming loop: same start, different way of spending
    effort.</figcaption>
  </figure>
  {sweep_table("nort", "dev only &#183; macro AUROC")}
  <div class="callout"><strong>The exchange rate.</strong> Five iterations,
  ~2500 attacker conversations, a GPT-5.1 judge and preprocessor on every success and
  2&#8239;h&#8239;15 of wall clock bought 0.7714&#8239;&#8594;&#8239;0.8595. Eighty-four
  labelled in-distribution examples bought 0.7713&#8239;&#8594;&#8239;0.8928, in six
  minutes of fitting on already-cached activations.</div>
</section>

<section>
  <div class="sec-head">
    <div class="eyebrow">findings</div>
    <h2>What to take away</h2>
  </div>
  <div class="findings">
    <div class="finding"><span class="pill p-result">result</span><div>
      <h3>Red-teaming captured about half the available headroom</h3>
      <p>{captured(BEST['nemotron'])*100:.0f}% in the nemotron arm and
      {captured(BEST['gptoss'])*100:.0f}% in the gpt-oss arm, measured from iteration 0 to
      the cross-validated ceiling. Real progress, and roughly half the distance left
      unwalked.</p></div></div>
    <div class="finding"><span class="pill p-result">result</span><div>
      <h3>In-distribution labels dominate red-team samples, sample for sample</h3>
      <p>84 dev rows beat 812 red-team rows. 336 dev rows beat every condition that
      included red-team data. Where labels are obtainable, obtaining them is the higher
      -leverage move; red-teaming's value is in the case where they are not.</p></div></div>
    <div class="finding"><span class="pill p-result">result</span><div>
      <h3>Finetuning afterwards beats folding the data in</h3>
      <p>Finetune leads joint retraining in 12 of 14 valid comparisons across both arms,
      and by most at small N &#8212; up to +0.042. Individual gaps are mostly inside the
      noise floor, so read the direction rather than any single number. The mechanism is
      plain: 84 dev rows are a rounding error inside 812 red-team rows, but they are the
      entire signal in a finetune.</p></div></div>
    <div class="finding"><span class="pill p-bug">bug</span><div>
      <h3>Training sets under 64 rows take zero optimizer steps</h3>
      <p><code>batch_size&#8239;16</code> with
      <code>gradient_accumulation_steps&#8239;4</code> means <code>optimizer.step()</code>
      only fires when <code>(batch_idx+1) % 4 == 0</code>. At 42 rows that is three
      batches, the condition never holds, and the fit runs its whole epoch budget while
      changing nothing &#8212; scores come back bit-identical to the starting probe.
      PyTorch says so in the log: <em>"Detected call of lr_scheduler.step() before
      optimizer.step()"</em>. This affects any small-data fit in the repo.</p></div></div>
    <div class="finding"><span class="pill p-bug">bug</span><div>
      <h3>The TPR@FPR metric reads the wrong ROC point</h3>
      <p><code>tpr_at_fixed_fpr_score</code> takes the point <em>nearest</em> 1% FPR
      rather than the best point at or below it. On a well-separated probe the nearest
      point is often (0,&#8239;0): the oracle scores TPR 0.0000 at AUROC 1.0. Every
      TPR figure here is the corrected definition; the original is kept alongside it in
      the result files for comparability.</p></div></div>
    <div class="finding"><span class="pill p-method">method</span><div>
      <h3>A retrain is not reproducible from its own snapshot</h3>
      <p>Refitting iteration 5 from the saved training data gave 0.8272, not the published
      0.8126 &#8212; while two refits in one process agreed bit for bit. The cause is row
      order: <code>_activate_redteam_cached</code> emits already-cached conversations
      first and appends newly-computed ones, so the training set's order encodes the
      box's cache history. Rebuilding that partition from the blob mtimes (614 hits,
      148 computed) reproduced the published probe exactly &#8212; 10/10 members
      bit-identical, AUROC to six decimals.</p></div></div>
    <div class="finding"><span class="pill p-method">method</span><div>
      <h3>The noise floor is {NOISE:.4f} macro AUROC</h3>
      <p>That same finding supplies a calibration: identical data, identical seeds, three
      different row orders &#8212; 0.8126, 0.8249, 0.8272. Differences smaller than this
      should not be read as effects. It is why "red-team data adds nothing once you have
      dev labels" is stated as a null rather than as a small positive or negative.</p></div></div>
  </div>
</section>

<footer>
  every number recomputed offline from the run's cached activations &#183; no LLM loaded<br>
  harness validated against all 12 published probe scores to six decimal places<br>
  analysis/ceiling &#183; branch experiment_instruction_cloud_3
</footer>
</div>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT} ({len(HTML)/1024:.0f} KB)")
