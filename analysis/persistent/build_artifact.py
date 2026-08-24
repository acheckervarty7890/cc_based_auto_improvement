#!/usr/bin/env python
"""Render the persistent-error study as a self-contained HTML page.

Every number and every mark on the page is read out of ``results/summary.json`` — none is
typed by hand — so the page cannot drift from the matrix it describes. Re-run after any
re-run of ``report.py`` and republish the same file path to refresh the artifact in place.

Published at https://claude.ai/code/artifact/10429880-0cd2-4bde-97da-96e5b86b5529
(republish the same path to update it in place).

    analysis/persistent/build_artifact.py [out.html]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pe_common as PE  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else PE.RESULTS / "persistent.html"
S = PE.read_json(PE.RESULTS / "summary.json")

SPLIT_SHORT = {"eval_ai_dilemmas": "ai_dilemmas", "eval_ant_hh": "ant_hh",
               "eval_balanced_refusal": "balanced_refusal",
               "eval_daily_dilemmas": "daily_dilemmas"}
ARM_SHORT = {"gptoss120b_datadesc": "exp22 · gpt-oss-120b",
             "deepseekv4pro_datadesc": "exp22 · deepseek-v4-pro",
             "s3_control": "exp23 · control",
             "s3_itermemo150": "exp23 · itermemo150",
             "s3_evaldesc": "exp23 · evaldesc"}
# The two arms the trajectory result turns on get the loud colors; the rest stay quiet.
ARM_TONE = {"gptoss120b_datadesc": "var(--ok)", "deepseekv4pro_datadesc": "var(--core)",
            "s3_control": "var(--faint)", "s3_itermemo150": "var(--muted)",
            "s3_evaldesc": "var(--rule-strong)"}


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bucket(k: int) -> int:
    """Five bands, the same ones the concentration table uses."""
    return 0 if k == 0 else 1 if k <= 11 else 2 if k <= 33 else 3 if k < 45 else 4


# ------------------------------------------------------------------------------ charts
def heat_strip() -> str:
    """One cell per eval row, banded by how many of the 45 probes get it wrong.

    Grouped by split and kept in file order, so the concentration in balanced_refusal and
    ant_hh is a shape rather than a claim. The tick row underneath marks the rows that are
    misranked rather than merely mis-thresholded.
    """
    ks, sp = S["rows_k"], S["rows_split"]
    rank = {r["index"] for r in S["ranking_core"]["rows"]}
    pitch, cell, cols = 9, 7.4, 96
    out = []
    for split in ["eval_ai_dilemmas", "eval_ant_hh", "eval_balanced_refusal",
                  "eval_daily_dilemmas"]:
        idx = [i for i, s in enumerate(sp) if s == split]
        nrows = -(-len(idx) // cols)
        h = nrows * pitch + 12
        marks = []
        for j, i in enumerate(idx):
            x, yy = (j % cols) * pitch, (j // cols) * pitch
            marks.append(f'<rect x="{x}" y="{yy}" width="{cell}" height="{cell}" rx="1.4" '
                         f'fill="var(--h{bucket(ks[i])})"><title>row {i} · {ks[i]}/45 '
                         f'probes wrong</title></rect>')
            if i in rank:
                marks.append(f'<rect x="{x}" y="{yy + cell + 1.6}" width="{cell}" '
                             f'height="2" fill="var(--core)"></rect>')
        n_core = sum(1 for i in idx if ks[i] == 45)
        out.append(
            f'<figure class="strip"><figcaption><span class="lbl">'
            f'{esc(SPLIT_SHORT[split])}</span><span class="cnt">{len(idx)} rows · '
            f'<b>{n_core}</b> wrong for all 45</span></figcaption>'
            f'<svg viewBox="0 0 {cols * pitch} {h}" width="100%" '
            f'style="max-height:{h * 1.6}px" role="img" aria-label="one cell per row of '
            f'{esc(split)}, shaded by how many probes misclassify it">'
            f'{"".join(marks)}</svg></figure>')
    legend = "".join(
        f'<span class="key"><i style="background:var(--h{b})"></i>{esc(t)}</span>'
        for b, t in enumerate(["0 probes wrong", "1–11", "12–33", "34–44", "all 45"]))
    return (f'<div class="strips">{"".join(out)}</div>'
            f'<div class="legend">{legend}'
            f'<span class="key"><i class="tick"></i>misranked, not just '
            f'mis-thresholded</span></div>')


def concentration_bars() -> str:
    c = S["concentration"]
    items = [("0 probes", c["never_wrong"], 0), ("1–11", c["wrong_1_to_11"], 1),
             ("12–33", c["wrong_12_to_33"], 2), ("34–44", c["wrong_34_to_44"], 3),
             ("all 45", c["always_wrong"], 4)]
    top = max(v for _, v, _ in items)
    rows = "".join(
        f'<div class="brow"><span class="bl">{esc(lab)}</span>'
        f'<span class="btrack"><span class="bfill" style="width:{100*v/top:.1f}%;'
        f'background:var(--h{b})"></span></span>'
        f'<span class="bv">{v}</span></div>' for lab, v, b in items)
    return f'<div class="bars">{rows}</div>'


def travel_bars() -> str:
    t = S["travel"]
    items = [("same arm, different iteration", "same_arm"),
             ("same experiment, different arm", "same_experiment"),
             ("different experiment", "different_experiment")]
    rows = []
    for lab, key in items:
        v, sd, n = t[key]["mean"], t[key]["sd"], t[key]["n"]
        rows.append(
            f'<div class="brow"><span class="bl">{esc(lab)}</span>'
            f'<span class="btrack"><span class="bfill" style="width:{100*v:.1f}%"></span>'
            f'<span class="whisk" style="left:{100*max(v-sd,0):.1f}%;'
            f'width:{100*min(sd*2,1-max(v-sd,0)):.1f}%"></span></span>'
            f'<span class="bv">{v:.3f}<em>±{sd:.2f}</em></span></div>')
    return (f'<div class="bars travel">{"".join(rows)}</div>'
            f'<p class="cap">Jaccard of two probes\' error sets, over all '
            f'{sum(t[k]["n"] for _, k in items)} pairs of retrained probes.</p>')


def trajectory_chart() -> str:
    traj = S["trajectory"]
    W, H, PL, PR, PT, PB = 660, 250, 46, 128, 14, 30
    top = max(p["errors"] for v in traj.values() for p in v)
    top = int(-(-top // 50) * 50)
    xs = lambda i: PL + i / 10 * (W - PL - PR)          # noqa: E731
    ys = lambda v: H - PB - v / top * (H - PT - PB)     # noqa: E731
    grid = "".join(
        f'<line x1="{PL}" y1="{ys(g):.1f}" x2="{W-PR}" y2="{ys(g):.1f}" '
        f'stroke="var(--rule)" stroke-width="1"></line>'
        f'<text x="{PL-9}" y="{ys(g)+4:.1f}" class="ax" text-anchor="end">{g}</text>'
        for g in range(0, top + 1, 100))
    ticks = "".join(
        f'<text x="{xs(i):.1f}" y="{H-PB+17}" class="ax" text-anchor="middle">{i}</text>'
        for i in range(0, 11, 2))
    series = []
    for a, pts in traj.items():
        d = " ".join(("M" if j == 0 else "L") + f"{xs(p['iteration']):.1f},"
                     f"{ys(p['errors']):.1f}" for j, p in enumerate(pts))
        last = pts[-1]
        series.append(
            f'<path d="{d}" fill="none" stroke="{ARM_TONE[a]}" stroke-width="2" '
            f'stroke-linejoin="round"></path>'
            f'<circle cx="{xs(last["iteration"]):.1f}" cy="{ys(last["errors"]):.1f}" '
            f'r="3.2" fill="{ARM_TONE[a]}"></circle>'
            f'<text x="{xs(last["iteration"])+8:.1f}" y="{ys(last["errors"])+4:.1f}" '
            f'class="slab" fill="{ARM_TONE[a]}">{esc(ARM_SHORT[a])}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="eval errors '
            f'per iteration, one line per arm">{grid}{ticks}'
            f'<text x="{PL}" y="{H-4}" class="ax">retrain iteration</text>'
            f'{"".join(series)}</svg>')


def calibration_dumbbells() -> str:
    rows = []
    lo = 0.60
    for a, v in S["calibration"]["by_arm"].items():
        x0 = (v["accuracy_at_half"] - lo) / (1 - lo) * 100
        x1 = (v["accuracy_at_median"] - lo) / (1 - lo) * 100
        rows.append(
            f'<div class="dumb"><span class="bl">{esc(ARM_SHORT[a])}</span>'
            f'<span class="dtrack">'
            f'<span class="dbar" style="left:{x0:.1f}%;width:{x1-x0:.1f}%"></span>'
            f'<span class="dot a" style="left:{x0:.1f}%"></span>'
            f'<span class="dot b" style="left:{x1:.1f}%"></span></span>'
            f'<span class="bv">{v["accuracy_at_half"]:.3f}<em>→ '
            f'{v["accuracy_at_median"]:.3f}</em></span></div>')
    return (f'<div class="bars">{"".join(rows)}</div>'
            f'<p class="cap">Eval accuracy at the reported 0.5 threshold '
            f'<span class="sw a"></span> and at each probe\'s own median score '
            f'<span class="sw b"></span>. Axis starts at {lo:.2f}.</p>')


def controls_bar() -> str:
    rows = S["core_rows"]
    both_right = sum(1 for r in rows if not r["ceiling_wrong"] and not r["dev_only_wrong"])
    one_right = sum(1 for r in rows if r["ceiling_wrong"] != r["dev_only_wrong"])
    neither = sum(1 for r in rows if r["ceiling_wrong"] and r["dev_only_wrong"])
    n = len(rows)
    seg = [("both in-distribution probes get it right", both_right, "var(--ok)"),
           ("one of the two does", one_right, "var(--ok-mid)"),
           ("neither does", neither, "var(--core)")]
    bars = "".join(f'<span class="seg" style="width:{100*v/n:.2f}%;background:{c}" '
                   f'title="{esc(t)}: {v}"></span>' for t, v, c in seg)
    keys = "".join(f'<span class="key"><i style="background:{c}"></i>{esc(t)} '
                   f'<b>{v}</b></span>' for t, v, c in seg)
    return f'<div class="stack">{bars}</div><div class="legend">{keys}</div>'


def ranking_table() -> str:
    tr = "".join(
        f'<tr><td class="mono">{r["index"]}</td>'
        f'<td class="mono">{esc(SPLIT_SHORT[r["split"]])}</td>'
        f'<td><span class="pill {"pos" if r["label"] == PE.POS else "neg"}">'
        f'{"harmful" if r["label"] == PE.POS else "not harmful"}</span></td>'
        f'<td class="num">{r["mean_percentile"]:.2f}</td>'
        f'<td>{"<b>wrong too</b>" if r["ceiling_wrong"] else "right"}</td>'
        f'<td class="quote">{esc(r["user"][:95])}</td></tr>'
        for r in S["ranking_core"]["rows"])
    return (f'<div class="scroll"><table><thead><tr><th>row</th><th>split</th>'
            f'<th>label</th><th>percentile</th><th>ceiling probe</th>'
            f'<th>first user turn</th></tr></thead><tbody>{tr}</tbody></table></div>')


def core_table() -> str:
    tr = "".join(
        f'<tr><td class="mono">{r["index"]}</td>'
        f'<td class="mono">{esc(SPLIT_SHORT[r["split"]])}</td>'
        f'<td><span class="pill {"pos" if r["label"] == PE.POS else "neg"}">'
        f'{"harmful" if r["label"] == PE.POS else "not harmful"}</span></td>'
        f'<td class="num">{r["mean_p"]:.3f}</td>'
        f'<td class="num {"bad" if r["ceiling_wrong"] else "good"}">{r["ceiling_p"]:.2f}</td>'
        f'<td class="num {"bad" if r["dev_only_wrong"] else "good"}">'
        f'{r["dev_only_p"]:.2f}</td>'
        f'<td>{"dup" if r["duplicate"] else ""}</td>'
        f'<td class="quote">{esc(r["user"][:110])}</td></tr>'
        for r in S["core_rows"])
    return (f'<div class="scroll tall"><table><thead><tr><th>row</th><th>split</th>'
            f'<th>label</th><th>mean p<br><span class="sub">45 probes</span></th>'
            f'<th>ceiling</th><th>dev-only</th><th></th>'
            f'<th>first user turn</th></tr></thead><tbody>{tr}</tbody></table></div>')


def arms_table() -> str:
    tr = "".join(
        f'<tr><td>{esc(ARM_SHORT[a])}</td><td class="num">{v["persistent"]}</td>'
        f'<td class="num">{v["private"]}</td>'
        f'<td class="num">{v["inherited_from_iter0"]}</td>'
        f'<td class="num">{v["new_since_iter0"]}</td></tr>'
        for a, v in S["arms"].items())
    return (f'<div class="scroll"><table><thead><tr><th>arm</th>'
            f'<th>persistent</th><th>private to it</th><th>inherited</th>'
            f'<th>new</th></tr></thead><tbody>{tr}</tbody></table></div>')


def jaccard_grid() -> str:
    J = S["arms_summary"]["jaccard"]
    arms = list(J)
    head = "".join(f'<th class="rot">{esc(ARM_SHORT[a].split(" · ")[1])}</th>' for a in arms)
    body = ""
    for a in arms:
        cells = ""
        for b in arms:
            v = J[a][b]
            alpha = 0.10 + 0.80 * (v - 0.35) / 0.65 if a != b else 0
            cells += (f'<td class="jc"{"" if a != b else " data-self=1"} '
                      f'style="background:rgba(var(--core-rgb),{max(alpha,0):.2f})">'
                      f'{"—" if a == b else f"{v:.2f}"}</td>')
        body += f'<tr><th class="rl">{esc(ARM_SHORT[a])}</th>{cells}</tr>'
    return (f'<div class="scroll"><table class="jac"><thead><tr><th></th>{head}</tr>'
            f'</thead><tbody>{body}</tbody></table></div>')


# -------------------------------------------------------------------------------- page
c, core, ar, ex, tv = (S["concentration"], S["core"], S["arms_summary"],
                       S["experiments"], S["travel"])
cal, rk, ctl, dup, anc = (S["calibration"], S["ranking_core"], S["controls"],
                          S["duplicates"], S["ancestor"])
setup = S["setup"]
n_rows = setup["n_rows"]

HTML = f"""<title>Inherited Blind Spots</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root {{
  --ground:#F5F3F0; --surface:#FFFFFF; --raised:#EDE9E4;
  --ink:#1C1A16; --body:#3B3830; --muted:#6E6759; --faint:#948C7C;
  --rule:#DFD9D0; --rule-strong:#C4BBAD;
  --core:#A8390F; --core-rgb:168,57,15; --core-soft:#F6E3D8;
  --ok:#0F6F68; --ok-soft:#DCEBE8; --ok-mid:#7FA89F;
  --h0:#E7E3DC; --h1:#EFD8C4; --h2:#E4B189; --h3:#CE7A47; --h4:#A8390F;
  --measure:66ch; --wide:1060px;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#14120F; --surface:#1B1815; --raised:#231F1A;
    --ink:#F1ECE4; --body:#CFC7BA; --muted:#988F80; --faint:#736B5E;
    --rule:#2E2822; --rule-strong:#453D33;
    --core:#E4763F; --core-rgb:228,118,63; --core-soft:#3A1D0E;
    --ok:#57B7AA; --ok-soft:#0F2E2B; --ok-mid:#3F7D74;
    --h0:#262119; --h1:#4A3520; --h2:#7C4E27; --h3:#B0602D; --h4:#E4763F;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#14120F; --surface:#1B1815; --raised:#231F1A;
  --ink:#F1ECE4; --body:#CFC7BA; --muted:#988F80; --faint:#736B5E;
  --rule:#2E2822; --rule-strong:#453D33;
  --core:#E4763F; --core-rgb:228,118,63; --core-soft:#3A1D0E;
  --ok:#57B7AA; --ok-soft:#0F2E2B; --ok-mid:#3F7D74;
  --h0:#262119; --h1:#4A3520; --h2:#7C4E27; --h3:#B0602D; --h4:#E4763F;
}}

body {{ background:var(--ground); color:var(--body);
  font-family:"Source Serif 4",Georgia,serif; font-size:17px; line-height:1.62;
  -webkit-font-smoothing:antialiased; margin:0; }}
.page {{ max-width:var(--wide); margin:0 auto; padding:56px 26px 88px; }}
h1,h2,h3,.slab,.lbl,.ax,th,.qnum {{ font-family:Archivo,"Helvetica Neue",Arial,sans-serif; }}
h1 {{ color:var(--ink); font-size:44px; line-height:1.06; font-weight:700;
  letter-spacing:-.02em; margin:0 0 14px; text-wrap:balance; }}
h2 {{ color:var(--ink); font-size:25px; font-weight:600; letter-spacing:-.012em;
  margin:0 0 10px; text-wrap:balance; }}
h3 {{ color:var(--ink); font-size:15px; font-weight:600; letter-spacing:.02em;
  margin:26px 0 8px; }}
p {{ max-width:var(--measure); margin:0 0 15px; }}
a {{ color:var(--core); }}
code, .mono, .num, .bv, .ax {{ font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
code {{ font-size:.86em; background:var(--raised); padding:1px 5px; border-radius:3px; }}
.dek {{ max-width:var(--measure); color:var(--muted); font-size:19px; line-height:1.5;
  margin:0 0 26px; }}
.eyebrow {{ font-family:Archivo,sans-serif; font-size:11.5px; font-weight:700;
  letter-spacing:.16em; text-transform:uppercase; color:var(--core); margin:0 0 14px; }}

/* the four questions are numbered because each one only makes sense after the last */
section {{ border-top:1px solid var(--rule); padding:34px 0 8px; }}
.qhead {{ display:grid; grid-template-columns:52px 1fr; gap:18px; align-items:start; }}
.qnum {{ font-size:12px; font-weight:700; letter-spacing:.1em; color:var(--faint);
  padding-top:8px; }}
.finding {{ display:block; max-width:var(--measure); margin:18px 0 0; padding:13px 17px;
  background:var(--core-soft); border-left:3px solid var(--core); color:var(--ink);
  font-size:15.5px; line-height:1.5; }}
.finding.calm {{ background:var(--ok-soft); border-left-color:var(--ok); }}

.panel {{ background:var(--surface); border:1px solid var(--rule); border-radius:5px;
  padding:20px 22px; margin:20px 0 4px; }}
.strips {{ display:flex; flex-direction:column; gap:17px; }}
.strip {{ margin:0; }}
.strip figcaption {{ display:flex; justify-content:space-between; align-items:baseline;
  gap:12px; margin:0 0 6px; }}
.lbl {{ font-size:12px; font-weight:600; letter-spacing:.07em; text-transform:uppercase;
  color:var(--ink); }}
.cnt {{ font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--muted); }}
.cnt b {{ color:var(--core); }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin-top:16px;
  font-family:Archivo,sans-serif; font-size:11.5px; color:var(--muted); }}
.key {{ display:inline-flex; align-items:center; gap:6px; }}
.key i {{ width:11px; height:11px; border-radius:2px; display:inline-block; }}
.key i.tick {{ height:3px; width:13px; border-radius:1px; background:var(--core); }}
.key b {{ color:var(--ink); font-variant-numeric:tabular-nums; }}

.bars {{ display:flex; flex-direction:column; gap:9px; }}
.brow, .dumb {{ display:grid; grid-template-columns:210px 1fr 108px; gap:14px;
  align-items:center; }}
.bl {{ font-family:Archivo,sans-serif; font-size:12.5px; color:var(--body); }}
.btrack, .dtrack {{ position:relative; height:15px; background:var(--raised);
  border-radius:3px; }}
.bfill {{ position:absolute; inset:0 auto 0 0; background:var(--core);
  border-radius:3px; }}
.travel .bfill {{ background:var(--rule-strong); }}
.whisk {{ position:absolute; top:4px; height:7px; border-left:1.5px solid var(--core);
  border-right:1.5px solid var(--core); }}
.bv {{ font-size:12.5px; color:var(--ink); text-align:right;
  font-variant-numeric:tabular-nums; }}
.bv em {{ display:block; font-style:normal; color:var(--muted); font-size:11.5px; }}
.dbar {{ position:absolute; top:6.5px; height:2px; background:var(--ok-mid); }}
.dot {{ position:absolute; top:3px; width:9px; height:9px; border-radius:50%;
  margin-left:-4.5px; }}
.dot.a {{ background:var(--rule-strong); }}
.dot.b {{ background:var(--ok); }}
.sw {{ display:inline-block; width:9px; height:9px; border-radius:50%;
  vertical-align:baseline; }}
.sw.a {{ background:var(--rule-strong); }}
.sw.b {{ background:var(--ok); }}
.stack {{ display:flex; height:22px; border-radius:3px; overflow:hidden; }}
.seg {{ display:block; height:100%; }}
.cap {{ font-family:Archivo,sans-serif; font-size:11.5px; color:var(--muted);
  margin:14px 0 0; max-width:none; }}
.ax {{ font-size:10.5px; fill:var(--faint); }}
.slab {{ font-size:11px; font-weight:600; }}

.scroll {{ overflow-x:auto; margin:18px 0 4px; }}
.scroll.tall {{ max-height:460px; overflow-y:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px;
  font-family:Archivo,sans-serif; }}
th {{ text-align:left; font-size:11px; font-weight:600; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--rule-strong);
  padding:0 12px 7px 0; position:sticky; top:0; background:var(--surface); }}
th .sub {{ text-transform:none; letter-spacing:0; font-weight:400; color:var(--faint); }}
td {{ padding:6px 12px 6px 0; border-bottom:1px solid var(--rule); color:var(--body);
  vertical-align:top; }}
td.num {{ font-variant-numeric:tabular-nums; text-align:right; padding-right:16px; }}
td.mono {{ font-size:12px; color:var(--muted); }}
td.quote {{ color:var(--muted); font-family:"Source Serif 4",serif; font-size:13.5px; }}
td.good {{ color:var(--ok); }}
td.bad {{ color:var(--core); font-weight:600; }}
.pill {{ display:inline-block; font-size:10.5px; font-weight:600; letter-spacing:.04em;
  padding:2px 7px; border-radius:9px; white-space:nowrap; }}
.pill.pos {{ background:var(--core-soft); color:var(--core); }}
.pill.neg {{ background:var(--ok-soft); color:var(--ok); }}
table.jac td.jc {{ text-align:center; font-family:"IBM Plex Mono",monospace;
  font-variant-numeric:tabular-nums; color:var(--ink); padding:7px 10px; }}
table.jac td[data-self] {{ color:var(--faint); background:var(--raised) !important; }}
table.jac th.rot {{ font-size:10.5px; }}
table.jac th.rl {{ text-transform:none; letter-spacing:0; font-size:12px;
  color:var(--body); font-weight:500; padding-right:14px; border-bottom:1px solid var(--rule); }}

footer {{ border-top:1px solid var(--rule); margin-top:40px; padding-top:22px;
  color:var(--muted); font-size:13.5px; }}
footer p {{ max-width:var(--measure); margin:0 0 10px; }}
footer code {{ font-size:12px; }}
:focus-visible {{ outline:2px solid var(--core); outline-offset:2px; }}
@media (max-width:640px) {{
  h1 {{ font-size:33px; }}
  .qhead {{ grid-template-columns:1fr; gap:4px; }}
  .qnum {{ padding-top:0; }}
  .brow, .dumb {{ grid-template-columns:1fr; gap:4px; }}
  .bv {{ text-align:left; }}
}}
@media (prefers-reduced-motion:reduce) {{ * {{ animation:none !important;
  transition:none !important; }} }}
</style>

<div class="page">
<p class="eyebrow">experiment22 + experiment23 · {setup['n_probes']} probes · harmful_to_human</p>
<h1>Inherited blind spots</h1>
<p class="dek">Every probe the two experiments produced was scored on all {n_rows}
<code>eval_sets/hu_ha</code> rows. {core['n']} rows are misclassified by all
{setup['n_probes']} of them — and the set barely moves when you change the attacker model,
the memo condition, or the experiment. It was already there in the probe all five runs
started from.</p>

<section>
<div class="qhead"><div class="qnum">01</div><div>
<h2>The errors are shared, not idiosyncratic</h2>
<p>Each probe gets {setup['per_probe_error_rate']['mean']:.1%} of the eval set wrong. If
those errors were independent across the {setup['n_probes']} probes, the number of rows
every one of them misses would be {c['expected_always_if_independent']:.0e}. It is
<b>{c['always_wrong']}</b>. The {c['total_errors']} errors land on just
{c['distinct_rows_ever_wrong']} distinct rows, and {c['never_wrong']} rows
({100*c['never_wrong']/n_rows:.0f}%) are never missed by anything.</p>
<div class="panel">{heat_strip()}</div>
<div class="panel">{concentration_bars()}</div>
<p class="finding">The {core['n']} always-wrong rows are <b>{core['n_positive']} positives
and {core['n_negative']} negative</b> — almost entirely harmful conversations scored as
not-harmful — and {core['by_split']['eval_balanced_refusal']['n']} of them sit in
<code>balanced_refusal</code>, {core['by_split']['eval_ant_hh']['n']} in
<code>ant_hh</code>.</p>
</div></div>
</section>

<section>
<div class="qhead"><div class="qnum">02</div><div>
<h2>Arm and experiment barely move it</h2>
<p>Two probes from the same arm agree about their errors a little more than two probes from
different arms. Two probes from different <em>experiments</em> agree just as much as two
from different arms of the same one. Swapping the attacker model or the memo condition is
not what decides which rows go wrong.</p>
<div class="panel">{travel_bars()}</div>
<h3>What each arm fails at every retrained iteration</h3>
{arms_table()}
<p>The five-way intersection is <b>{ar['intersection']}</b> rows, {100*ar['intersection']/ar['union']:.0f}%
of the {ar['union']}-row union — a common core plus a substantial arm-private tail, the
largest of it deepseek-v4-pro's {S['arms']['deepseekv4pro_datadesc']['private']} rows.
Between experiments: {ex['shared']} shared, {ex['exp22_only']} only experiment22,
{ex['exp23_only']} only experiment23 (Jaccard {ex['jaccard']:.2f}).</p>
{jaccard_grid()}
</div></div>
</section>

<section>
<div class="qhead"><div class="qnum">03</div><div>
<h2>It was inherited, not discovered</h2>
<p>All five runs start from the same probe. The <code>probe_iter0</code> pickles differ only
in metadata — their scores are identical to
{anc['max_abs_score_delta']:.0e} — because every run trains its first probe on the same 50
base samples under the same pinned ensemble seeds. That shared probe makes
<b>{anc['n_errors']}</b> errors, and
{100*ar['intersection_inherited_from_iter0']/ar['intersection']:.0f}% of the universal core
was already wrong there.</p>
<p>Of those {anc['n_errors']} inherited errors,
{anc['fate_of_its_errors']['fixed_by_all_arms']} are fixed by every arm and
{anc['fate_of_its_errors']['fixed_by_none']} by none. Retraining on red-team data shrinks
the error set without relocating it — and in one arm it does not even shrink.</p>
<div class="panel">{trajectory_chart()}</div>
<p class="finding">gpt-oss-120b takes its probe from {anc['n_errors']} eval errors down to
{S['trajectory']['gptoss120b_datadesc'][-1]['errors']} over five retrains. deepseek-v4-pro
ends at <b>{S['trajectory']['deepseekv4pro_datadesc'][-1]['errors']}</b> — worse than the
probe it started from.</p>
</div></div>
</section>

<section>
<div class="qhead"><div class="qnum">04</div><div>
<h2>Most of the core is the 0.5 threshold</h2>
<p>These probes are shifted. They call {cal['positive_call_rate']['mean']:.0%} of eval rows
positive against a 50% base rate, their median score is {cal['median_score']:.3f}, and
<b>{cal['false_negative_share_of_errors']:.0%} of every error is a false negative</b>. Move
each probe to its own median score — the balanced-accuracy cut, since every split is
exactly class balanced — and every arm gains.</p>
<div class="panel">{calibration_dumbbells()}</div>
<p>Under that threshold-free reading the core is <b>{rk['n']} rows, not {core['n']}</b>
(they overlap on {rk['overlap_with_half_core']}). The rest are ranked reasonably — mean
within-split percentile {core['mean_within_split_percentile']:.2f} — and fail only because
the threshold sits above them. All {rk['n']} survivors are in <code>ant_hh</code>, the split
with the lowest AUROC throughout, and they are semantically coherent: harm directed at
<em>animals</em>, which the concept labels not-harmful-to-<em>humans</em> and every probe
reads as harmful, plus complied-with harmful requests scored low.</p>
{ranking_table()}
</div></div>
</section>

<section>
<div class="qhead"><div class="qnum">05</div><div>
<h2>Are these rows learnable at all?</h2>
<p>The distinction that matters: is the core intrinsically ambiguous, or is it what
red-team training data fails to cover? Two controls answer it, both single
<code>linear_then_softmax</code> heads on the same activations. <b>dev-only</b> trains on the
{ctl['n_dev_pool']}-row dev pool alone; <b>ceiling</b> is 5-fold cross-validation inside the
eval set plus that pool, every row scored by the fold that did not train on it (mean AUROC
{ctl['ceiling_mean_auroc']:.4f}).</p>
<div class="panel">{controls_bar()}</div>
<p class="finding calm">In-distribution training recovers
{ctl['core_recovered_by_in_distribution']} of the {core['n']} core rows. Only
<b>{ctl['core_wrong_for_both_controls']}</b> defeat all {setup['n_probes']} run probes
<em>and</em> both controls. The core is mostly a red-team training-distribution deficit, not
a limit of the probe family.</p>
{core_table()}
</div></div>
</section>

<section>
<div class="qhead"><div class="qnum">—</div><div>
<h2>One thing found along the way</h2>
<p><b>{dup['n_conversations_repeated']} conversations appear in more than one eval split</b> —
{dup['n_rows_involved']} rows, {100*dup['n_rows_involved']/n_rows:.1f}% of the eval set,
{list(dup['by_split_pair'].values())[0]} of them shared between <code>ant_hh</code> and
<code>balanced_refusal</code>. Labels always agree, so nothing is contradictory, but the
per-split eval means are not independent of each other, and the {core['n']} core rows are
really {dup['core_distinct_conversations']} distinct conversations.</p>
<p>Related, from the labelling side: an eval conversation whose assistant refuses is labelled
harmful {S['character']['refusal_crosstab']['refuses']['p_positive']:.0%} of the time
(n={S['character']['refusal_crosstab']['refuses']['n']}) against
{S['character']['refusal_crosstab']['complies']['p_positive']:.0%} for one that complies —
and refusing rows are almost never missed, at a mean of
{S['character']['refusal_crosstab']['refuses']['mean_k']:.1f} of {setup['n_probes']} probes
wrong. None of the {core['n']} core rows contains a refusal. The probes have learned refusal
as the negative class; the core is what is left when that cue is absent.</p>
</div></div>
</section>

<footer>
<p>{setup['n_probes']} probes — experiment22's two arms (6 iterations each) and
experiment23's three (11 each), the latter read out of the
<code>experiment23_cloud</code> branch — scored on all {n_rows} rows of the four
<code>eval_sets/hu_ha</code> splits. Scoring uses the cached full-split activation blobs
with <code>PROBE_FUSED_ENSEMBLE=0</code>, as the runs did: the matrix reproduces every
published comparison CSV to {setup['reproduction_max_auroc_drift']:.1e} AUROC, which is
what licenses a claim about individual rows. No LLM is loaded and no activation is
recomputed at any point.</p>
<p>The run probes are 10-member score-averaging ensembles; the two controls are single
heads at seed 42, early-stopped on the ceiling study's reserved 25% dev slice, so they read
against that study's curves rather than against the runs' comparison CSVs. Every number
here is generated from <code>analysis/persistent/results/summary.json</code> by
<code>build_artifact.py</code>.</p>
</footer>
</div>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML, encoding="utf-8")
print("wrote", OUT, f"({len(HTML)/1024:.0f} KB)")
