#!/usr/bin/env python
"""Build a self-contained HTML viewer for a probe-to-probe eval regression.

Written for the set-A arm's iteration-2 -> iteration-3 drop on the ``mt`` / ``mts``
splits. ``probe_iter3`` trains on exactly ``probe_iter2``'s data plus 14 hand-written
contrastive pairs, so the viewer puts those two things side by side:

  * **The added pairs** — each low-stakes original next to its high-stakes
    contrastive. For the 11 pairs whose two members share a byte-identical message
    prefix, that prefix is rendered *once*, spanning both columns, because sharing
    is the whole point: the probe cannot use anything in it to tell the pair apart.
    Each pair carries its per-token diagnostics (the peak per-token score on the
    shared prefix vs. the differing tail, and the share of softmax pooling weight on
    the shared prefix, under both probes).
  * **The regressions** — every eval sample the first probe got right and the second
    got wrong, with both scores, the label, the split's provenance columns, and the
    conversation the probe actually saw.

Inputs are the two files ``scripts/analyze_iter3_regression.py`` writes, plus the
eval splits themselves (joined by ``split`` + ``idx``) and the arm's comparison CSV.

Usage:
    .venv_claude/bin/python scripts/build_iter3_regression_viewer.py \
        --probe-dir probes/hs_llama1b_deepseekv4pro_guidance_setA \
        --out viewers/setA_iter3_regression_viewer.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def esc(x) -> str:
    return html.escape(str(x), quote=True)


# ------------------------------------------------------------------- loading


def load_eval_conversations(eval_dir: Path, splits, combine: bool, convert: bool):
    """{split: [messages, ...]} exactly as the probe saw them (same loader path)."""
    from tuberlens.interfaces.dataset import LabelledDataset

    out = {}
    for name in splits:
        ds = LabelledDataset.load_from(
            eval_dir / f"{name}.jsonl",
            pos_class_label="high-stakes",
            neg_class_label="low-stakes",
            combine_consecutive_messages=combine,
            convert_tool_to_assistant=convert,
        )
        out[name] = [
            [{"role": m.role, "content": m.content} for m in dialogue]
            for dialogue in ds.inputs
        ]
    return out


def load_pairs(probe_dir: Path, redteam_a: str, redteam_b: str):
    ids_a = {json.loads(l)["id"] for l in (probe_dir / redteam_a).open()}
    pairs: dict[str, dict] = {}
    for line in (probe_dir / redteam_b).open():
        row = json.loads(line)
        if row["id"] in ids_a:
            continue
        kind, key = row["id"].split("-", 1)
        pairs.setdefault(key, {})[kind] = row
    return pairs


def load_csv(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ rendering

MSG_ROLE_CLASS = {"system": "sys", "user": "usr", "assistant": "asst", "tool": "tool"}


def render_turns(messages, cls: str = "") -> str:
    out = []
    for m in messages:
        role = m.get("role", "?")
        out.append(
            f'<div class="msg {MSG_ROLE_CLASS.get(role, "usr")} {cls}">'
            f'<span class="role">{esc(role)}</span>'
            f'<div class="body">{esc(m.get("content", ""))}</div></div>'
        )
    return "".join(out)


def shared_message_count(orig, contra) -> int:
    n = 0
    for a, b in zip(orig, contra):
        if a["role"] == b["role"] and a["content"] == b["content"]:
            n += 1
        else:
            break
    return n


def render_pair(n: int, key: str, pair: dict, stats: dict | None,
                label_a: str, label_b: str) -> str:
    orig, contra = pair["orig"], pair["contra"]
    om, cm = orig["inputs"], contra["inputs"]
    shared = shared_message_count(om, cm)

    head = [
        f'<span class="idx">pair {n + 1}</span>',
        f'<code class="key">{esc(key)}</code>',
    ]
    if shared:
        head.append(
            f'<span class="chip warn">first {shared} of {len(om)} messages '
            f"byte-identical</span>"
        )
    else:
        head.append('<span class="chip">no shared message</span>')
    if stats:
        frac = stats["shared_body_tokens"] / stats["orig_body_tokens"]
        head.append(
            f'<span class="chip">{stats["shared_body_tokens"]} of '
            f'{stats["orig_body_tokens"]} body tokens shared ({frac:.0%})</span>'
        )

    body = []
    if shared:
        body.append(
            '<div class="shared"><div class="sharedhead">Shared prefix — identical in '
            "both members, so nothing here can separate the classes. Llama is causal, "
            "so the layer-8 activations at these positions are the same in both."
            "</div>" + render_turns(om[:shared]) + "</div>"
        )
    body.append(
        '<div class="cols">'
        f'<div class="col"><div class="colhead neg">original — '
        f"{esc(label_a)}</div>{render_turns(om[shared:])}</div>"
        f'<div class="col"><div class="colhead pos">contrastive — '
        f"{esc(label_b)}</div>{render_turns(cm[shared:])}</div>"
        "</div>"
    )

    if stats:
        rows = []
        for which, name in (("a", "probe_iter2"), ("b", "probe_iter3")):
            for tag in ("orig", "contra"):
                rows.append(
                    f"<tr><td>{esc(name)}</td><td>{esc(tag)}</td>"
                    f'<td class="num">{stats[f"{which}_{tag}_shared_max"]:+.2f}</td>'
                    f'<td class="num">{stats[f"{which}_{tag}_tail_max"]:+.2f}</td>'
                    f'<td class="num">{stats[f"{which}_{tag}_shared_weight"]:.2f}</td>'
                    f'<td class="num">{stats[f"{which}_{tag}_pooled"]:+.2f}</td></tr>'
                )
        body.append(
            '<details class="tokstats"><summary>Per-token diagnostics</summary>'
            '<div class="scroll"><table><thead><tr><th>probe</th><th>member</th>'
            "<th>peak score<br>on shared prefix</th><th>peak score<br>on unique tail</th>"
            "<th>pooling weight<br>on shared prefix</th><th>pooled<br>score</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></details>"
        )

    hay = " ".join(m["content"] for m in om + cm)[:4000]
    return (
        f'<article class="pair" data-shared="{1 if shared else 0}" '
        f'data-hay="{esc(hay.lower())}">'
        f'<header class="pairhead">{"".join(head)}</header>'
        + "".join(body)
        + "</article>"
    )


META_ORDER = (
    "medical_specialty", "section_header", "sample_name", "category",
    "scale_labels", "scale_label_confidence", "ids", "source",
)
EXPLAIN_FIELDS = ("scale_label_explanation", "harm_explanation")


def render_sample(row: dict, messages, pos_label: str) -> str:
    meta = row.get("meta", {})
    is_pos = row["y"] == 1
    chips = [
        f'<span class="chip {"pos" if is_pos else "neg"}">{esc(row["label"])}</span>',
        f'<span class="chip">{esc(row["split"])} #{row["idx"]}</span>',
    ]
    for k in META_ORDER:
        v = meta.get(k)
        if v not in (None, ""):
            chips.append(f'<span class="chip">{esc(k)}: {esc(v)}</span>')
    nchars = sum(len(m["content"]) for m in messages)
    chips.append(f'<span class="chip">{len(messages)} msgs, {nchars} chars</span>')

    arrow = (
        f'<span class="score ok">{row["score_a"]:.3f}</span>'
        f'<span class="arr">&rarr;</span>'
        f'<span class="score bad">{row["score_b"]:.3f}</span>'
        f'<span class="delta down">{row["delta"]:+.3f}</span>'
    )

    explain = ""
    for k in EXPLAIN_FIELDS:
        v = meta.get(k)
        if v not in (None, "") and "Filled in based on" not in str(v):
            explain = f'<div class="explain"><b>{esc(k)}</b>: {esc(v)}</div>'
            break

    hay = " ".join(m["content"] for m in messages)[:4000].lower()
    return (
        f'<article class="samp" data-split="{esc(row["split"])}" '
        f'data-cls="{"pos" if is_pos else "neg"}" '
        f'data-delta="{row["delta"]:.5f}" data-hay="{esc(hay)}">'
        f'<header class="samphead"><div class="chips">{"".join(chips)}</div>'
        f'<div class="scores">{arrow}</div></header>'
        f"{explain}"
        f'<details><summary>Conversation the probe scored</summary>'
        f'{render_turns(messages)}</details>'
        "</article>"
    )


def render_metrics_table(rows) -> str:
    if not rows:
        return ""
        # comparison.csv missing — the mechanism tab still works
    by_round: dict[str, dict[str, str]] = {}
    for r in rows:
        by_round.setdefault(r["round"], {})[r["dataset"]] = r["auroc"]
    datasets = ["anthropic", "mt", "mts", "toolace", "mean"]
    head = "".join(f"<th>{esc(d)}</th>" for d in datasets)
    body = []
    for rnd, cols in by_round.items():
        cells = []
        for d in datasets:
            v = cols.get(d)
            cells.append(f'<td class="num">{float(v):.3f}</td>' if v else "<td></td>")
        body.append(f"<tr><td>{esc(rnd)}</td>{''.join(cells)}</tr>")
    return (
        '<div class="scroll"><table class="metrics"><thead><tr><th>probe</th>'
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


#: Categorical slots, assigned in fixed order and never cycled. Rendered through
#: CSS custom properties (``--s1``…) so each has a light and a dark step.
N_SERIES_SLOTS = 8

# chart geometry, in viewBox units
_CW, _CH = 900, 360
_ML, _MR, _MT, _MB = 54, 132, 14, 38
_PW, _PH = _CW - _ML - _MR, _CH - _MT - _MB
_LABEL_GAP = 15.0  # min vertical distance between two endpoint labels


def _spread_labels(ys, gap: float, lo: float, hi: float):
    """De-collide label positions, keeping order. Returns the adjusted y per input."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = [ys[i] for i in order]
    for k in range(1, len(placed)):  # push down
        placed[k] = max(placed[k], placed[k - 1] + gap)
    overflow = placed[-1] - hi if placed else 0
    if overflow > 0:
        placed = [p - overflow for p in placed]
    for k in range(len(placed) - 2, -1, -1):  # pull back up, never past `lo`
        placed[k] = min(placed[k], placed[k + 1] - gap)
    if placed and placed[0] < lo:
        shift = lo - placed[0]
        for k in range(len(placed)):
            placed[k] = min(placed[k] + shift, hi)
    out = [0.0] * len(ys)
    for slot, i in enumerate(order):
        out[i] = placed[slot]
    return out


def render_auroc_chart(metrics_rows) -> str:
    """Line chart of AUROC per probe iteration, one line per eval split.

    Self-contained inline SVG: the page ships no chart library. The table below it
    is the accessible fallback (and the relief for the light-mode series whose
    contrast against the panel sits under 3:1).
    """
    if not metrics_rows:
        return ""
    rounds, datasets = [], []
    for r in metrics_rows:
        if r["round"] not in rounds:
            rounds.append(r["round"])
        d = r["dataset"]
        if d != "mean" and d not in datasets:
            datasets.append(d)
    vals = {}
    for r in metrics_rows:
        try:
            vals[(r["round"], r["dataset"])] = float(r["auroc"])
        except (TypeError, ValueError):
            pass
    if len(rounds) < 2 or not vals:
        return ""  # nothing to trend; the table already says it

    series = []
    for i, d in enumerate(datasets[:N_SERIES_SLOTS]):
        series.append({"name": d, "color": f"var(--s{i + 1})", "mean": False})
    if any(k[1] == "mean" for k in vals):
        series.append({"name": "mean", "color": "var(--fg)", "mean": True})

    lo = min(0.5, (int(min(vals.values()) * 10) / 10))
    hi = 1.0
    xs = [_ML + i * (_PW / (len(rounds) - 1)) for i in range(len(rounds))]
    ypx = lambda v: _MT + (1 - (v - lo) / (hi - lo)) * _PH  # noqa: E731

    for s in series:
        s["vals"] = [vals.get((rnd, s["name"])) for rnd in rounds]
        s["y"] = [None if v is None else ypx(v) for v in s["vals"]]

    # ---- axes
    ticks = [lo + i * 0.1 for i in range(int(round((hi - lo) / 0.1)) + 1)]
    grid = "".join(
        f'<line class="grid" x1="{_ML}" x2="{_ML + _PW}" '
        f'y1="{ypx(t):.1f}" y2="{ypx(t):.1f}"/>'
        f'<text class="ytick" x="{_ML - 10}" y="{ypx(t) + 4:.1f}">{t:.1f}</text>'
        for t in ticks
    )
    grid += (
        f'<line class="axis" x1="{_ML}" x2="{_ML + _PW}" '
        f'y1="{ypx(lo):.1f}" y2="{ypx(lo):.1f}"/>'
    )
    if abs(lo - 0.5) < 1e-9:
        grid += (
            f'<text class="chance" x="{_ML + 4}" y="{ypx(0.5) - 7:.1f}">chance</text>'
        )
    xlab = "".join(
        f'<text class="xtick" x="{xs[i]:.1f}" y="{_MT + _PH + 24}">{esc(r)}</text>'
        for i, r in enumerate(rounds)
    )

    # ---- lines + markers
    marks = []
    for s in series:
        pts = [(xs[i], y) for i, y in enumerate(s["y"]) if y is not None]
        if len(pts) > 1:
            d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
            marks.append(
                f'<path class="ln{" mean" if s["mean"] else ""}" d="{d}" '
                f'style="stroke:{s["color"]}"/>'
            )
        for x, y in pts:
            marks.append(
                f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4" '
                f'style="fill:{s["color"]}"/>'
            )

    # ---- endpoint labels (identity + value without a hover), leader-lined when nudged
    ends = [(i, s) for i, s in enumerate(series) if s["y"][-1] is not None]
    placed = _spread_labels(
        [s["y"][-1] for _, s in ends], _LABEL_GAP, _MT + 6, _MT + _PH - 2
    )
    tags = []
    for (_, s), ly in zip(ends, placed):
        y0 = s["y"][-1]
        x0 = xs[-1]
        if abs(ly - y0) > 2:
            tags.append(
                f'<path class="leader" d="M{x0 + 6:.1f} {y0:.1f} '
                f'L{x0 + 14:.1f} {ly:.1f} L{x0 + 20:.1f} {ly:.1f}" '
                f'style="stroke:{s["color"]}"/>'
            )
        else:
            tags.append(
                f'<line class="leader" x1="{x0 + 6:.1f}" x2="{x0 + 20:.1f}" '
                f'y1="{ly:.1f}" y2="{ly:.1f}" style="stroke:{s["color"]}"/>'
            )
        tags.append(
            f'<text class="endlab" x="{x0 + 25:.1f}" y="{ly + 4:.1f}">'
            f'<tspan class="v">{s["vals"][-1]:.3f}</tspan>'
            f'<tspan class="n" dx="5">{esc(s["name"])}</tspan></text>'
        )

    # ---- hover layer
    hover = (
        f'<line class="cross" x1="0" x2="0" y1="{_MT}" y2="{_MT + _PH}" hidden/>'
        f'<rect class="hit" x="{_ML - 20}" y="{_MT}" width="{_PW + 40}" '
        f'height="{_PH}" fill="transparent"/>'
    )

    legend = "".join(
        f'<span class="lgd"><i class="{"key mean" if s["mean"] else "key"}" '
        f'style="background:{s["color"]}"></i>{esc(s["name"])}</span>'
        for s in series
    )
    data = json.dumps(
        {
            "rounds": rounds,
            "x": [round(x, 1) for x in xs],
            "w": _CW,
            "series": [
                {"name": s["name"], "color": s["color"], "vals": s["vals"]}
                for s in series
            ],
        }
    )
    return f"""
<figure class="chart" id="aurocfig">
  <div class="legendrow">{legend}</div>
  <div class="plotwrap">
    <svg id="aurocsvg" viewBox="0 0 {_CW} {_CH}" tabindex="0" role="img"
         aria-label="AUROC per probe iteration, one line per eval split">
      {grid}{xlab}{"".join(marks)}{"".join(tags)}{hover}
    </svg>
    <div class="tip" id="auroctip" hidden></div>
  </div>
  <figcaption>AUROC on each eval split after every retrain. Hover (or focus the plot
  and use &larr;/&rarr;) for all splits at one iteration; the table below carries the
  same numbers.</figcaption>
</figure>
<script type="application/json" id="aurocdata">{data}</script>"""


def render_mechanism(per_split_summary, metrics_rows) -> str:
    split_rows = "".join(
        f"<tr><td>{esc(s['split'])}</td>"
        f'<td class="num">{s["n"]}</td>'
        f'<td class="num">{s["auroc_a"]:.3f}</td><td class="num">{s["auroc_b"]:.3f}</td>'
        f'<td class="num down">{s["auroc_b"] - s["auroc_a"]:+.3f}</td>'
        f'<td class="num">{s["regressions"]}</td><td class="num">{s["improvements"]}</td>'
        f'<td class="num">{s["shift_pos"]:+.3f}</td><td class="num">{s["shift_neg"]:+.3f}</td>'
        "</tr>"
        for s in per_split_summary
    )
    return f"""
<section class="tabpanel" id="panel-mech">
  <div class="note">
    <b>What changed.</b> <code>probe_iter3</code> trains on exactly
    <code>probe_iter2</code>'s data plus 14 contrastive pairs. In 11 of those 14 the
    two members share a byte-identical opening — a screaming emergency (CODE RED,
    MAYDAY, NERC Alert Level 3) — and differ only in the final turn, where the
    low-stakes member retracts it ("the reactor is a microwave in the break room")
    and the high-stakes member continues it for real. Because Llama is causal, the
    activations over that shared opening are <i>identical</i> in both members, so the
    only way to fit the pair is to stop scoring the opening.
  </div>
  <h2>Per-split effect</h2>
  <div class="scroll"><table><thead><tr><th>split</th><th>n</th>
    <th>AUROC iter2</th><th>AUROC iter3</th><th>&Delta;</th>
    <th>regressions</th><th>improvements</th>
    <th>score shift<br>positives</th><th>score shift<br>negatives</th>
  </tr></thead><tbody>{split_rows}</tbody></table></div>
  <h2>Whole-arm AUROC</h2>
  {render_auroc_chart(metrics_rows)}
  {render_metrics_table(metrics_rows)}
</section>"""


# ----------------------------------------------------------------- page shell

STYLE = """
:root {
  color-scheme: light dark;
  --bg:#fff; --panel:#f6f7f9; --panel2:#eef0f3; --line:#d9dde3;
  --fg:#1a1d21; --muted:#666e79; --accent:#3f6fd8;
  --pos-bg:#fdeeee; --pos-fg:#9d2727; --neg-bg:#eaf5ee; --neg-fg:#1e6b3c;
  --warn:#8a5b00; --warn-bg:#fdf3dc;
  --up:#1e6b3c; --down:#9d2727;
  /* categorical chart slots, fixed order, light steps */
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
    --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
    --pos-bg:#2c1c1e; --pos-fg:#ff9a9a; --neg-bg:#17281d; --neg-fg:#86e0a5;
    --warn:#f0c060; --warn-bg:#33290f;
    --up:#86e0a5; --down:#ff9a9a;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  }
}
:root[data-theme="light"] {
  --bg:#fff; --panel:#f6f7f9; --panel2:#eef0f3; --line:#d9dde3;
  --fg:#1a1d21; --muted:#666e79; --accent:#3f6fd8;
  --pos-bg:#fdeeee; --pos-fg:#9d2727; --neg-bg:#eaf5ee; --neg-fg:#1e6b3c;
  --warn:#8a5b00; --warn-bg:#fdf3dc; --up:#1e6b3c; --down:#9d2727;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
}
:root[data-theme="dark"] {
  --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
  --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
  --pos-bg:#2c1c1e; --pos-fg:#ff9a9a; --neg-bg:#17281d; --neg-fg:#86e0a5;
  --warn:#f0c060; --warn-bg:#33290f; --up:#86e0a5; --down:#ff9a9a;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1280px; margin:0 auto; padding:22px 18px 80px; }
h1 { margin:0 0 4px; font-size:21px; letter-spacing:-.01em; }
h2 { margin:26px 0 8px; font-size:16px; }
.sub { color:var(--muted); font-size:13px; margin:4px 0; }
.themebtn { position:fixed; top:14px; right:14px; z-index:9; appearance:none; cursor:pointer;
  background:var(--panel); color:var(--muted); border:1px solid var(--line);
  border-radius:999px; padding:6px 12px; font-size:12px; font-weight:600; }
.note { margin:16px 0 0; padding:11px 14px; border-left:3px solid var(--accent);
  background:var(--panel); border-radius:0 6px 6px 0; font-size:13.5px; color:var(--muted); }
.note b { color:var(--fg); }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.92em; }
.tabs { display:flex; flex-wrap:wrap; gap:6px; margin-top:20px; border-bottom:1px solid var(--line); }
.tabs button { appearance:none; cursor:pointer; background:transparent; color:var(--muted);
  border:0; border-bottom:2px solid transparent; padding:9px 13px; font-size:13.5px; font-weight:600; }
.tabs button[aria-selected="true"] { color:var(--fg); border-bottom-color:var(--accent); }
.tabpanel { display:none; }
.tabpanel.active { display:block; }
.controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:16px 0 4px; }
.controls input, .controls select { background:var(--panel); color:var(--fg);
  border:1px solid var(--line); border-radius:6px; padding:6px 9px; font-size:13px; }
.controls input[type="search"] { min-width:220px; flex:1 1 220px; }
.count { color:var(--muted); font-size:12.5px; }
.scroll { overflow-x:auto; margin:8px 0; }
table { border-collapse:collapse; font-size:13px; min-width:520px; }
th, td { border-bottom:1px solid var(--line); padding:6px 11px; text-align:left; vertical-align:bottom; }
th { color:var(--muted); font-weight:600; font-size:12px; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
td.down { color:var(--down); }
table.metrics tbody tr:last-child td { font-weight:600; }
.pair, .samp { border:1px solid var(--line); border-radius:9px; background:var(--panel);
  margin:14px 0; padding:12px 14px; }
.pairhead, .samphead { display:flex; flex-wrap:wrap; gap:8px; align-items:center;
  justify-content:space-between; }
.chips { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.idx { font-weight:700; font-size:13px; }
.key { color:var(--muted); font-size:12px; }
.chip { display:inline-block; background:var(--panel2); color:var(--muted);
  border:1px solid var(--line); border-radius:999px; padding:2px 9px; font-size:11.5px; }
.chip.pos { background:var(--pos-bg); color:var(--pos-fg); border-color:transparent; font-weight:600; }
.chip.neg { background:var(--neg-bg); color:var(--neg-fg); border-color:transparent; font-weight:600; }
.chip.warn { background:var(--warn-bg); color:var(--warn); border-color:transparent; font-weight:600; }
.shared { margin:12px 0; border:1px dashed var(--line); border-radius:8px;
  background:var(--panel2); padding:10px 12px; }
.sharedhead { font-size:12px; color:var(--muted); margin-bottom:8px; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:10px; }
@media (max-width:860px) { .cols { grid-template-columns:1fr; } }
.col { min-width:0; }
.colhead { font-size:12px; font-weight:700; padding:5px 9px; border-radius:6px; margin-bottom:8px; }
.colhead.pos { background:var(--pos-bg); color:var(--pos-fg); }
.colhead.neg { background:var(--neg-bg); color:var(--neg-fg); }
.msg { margin:7px 0; }
.msg .role { display:inline-block; font-size:10.5px; text-transform:uppercase;
  letter-spacing:.06em; color:var(--muted); font-weight:700; }
.msg .body { white-space:pre-wrap; overflow-wrap:anywhere; font-size:13.5px;
  background:var(--bg); border:1px solid var(--line); border-left-width:3px;
  border-radius:0 6px 6px 0; padding:8px 10px; margin-top:3px; }
.msg.usr .body { border-left-color:var(--accent); }
.msg.asst .body { border-left-color:var(--muted); }
.msg.sys .body { border-left-color:var(--warn); }
.scores { display:flex; align-items:center; gap:7px; font-variant-numeric:tabular-nums; }
.score { font-size:14px; font-weight:700; padding:2px 7px; border-radius:5px; }
.score.ok { background:var(--neg-bg); color:var(--neg-fg); }
.score.bad { background:var(--pos-bg); color:var(--pos-fg); }
.arr { color:var(--muted); }
.delta { font-size:12px; color:var(--down); font-weight:600; }
.explain { margin:9px 0 0; font-size:13px; color:var(--muted); }
.explain b { color:var(--fg); }
details { margin-top:9px; }
summary { cursor:pointer; color:var(--accent); font-size:12.5px; font-weight:600; }
details.tokstats table { min-width:0; }
.empty { color:var(--muted); font-size:13px; padding:18px 0; }

/* ---- AUROC-per-iteration chart ---- */
.chart { margin:12px 0 4px; padding:12px 14px 10px; border:1px solid var(--line);
  border-radius:9px; background:var(--panel); }
.legendrow { display:flex; flex-wrap:wrap; gap:14px; margin-bottom:6px;
  font-size:12.5px; color:var(--muted); }
.lgd { display:inline-flex; align-items:center; gap:6px; }
.key { display:inline-block; width:15px; height:2px; border-radius:1px; }
.key.mean { height:0; border-top:2px dashed currentColor; background:none !important;
  color:var(--fg); }
.plotwrap { position:relative; }
#aurocsvg { display:block; width:100%; height:auto; overflow:visible; outline:none; }
#aurocsvg:focus-visible { outline:2px solid var(--accent); outline-offset:3px;
  border-radius:6px; }
#aurocsvg .grid, #aurocsvg .axis { stroke:var(--line); stroke-width:1; }
#aurocsvg .axis { stroke:var(--muted); }
#aurocsvg text { font:12px -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  fill:var(--muted); }
#aurocsvg .ytick { text-anchor:end; font-variant-numeric:tabular-nums; }
#aurocsvg .xtick { text-anchor:middle; fill:var(--fg); font-weight:600; }
#aurocsvg .chance { font-size:10.5px; }
#aurocsvg .ln { fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }
#aurocsvg .ln.mean { stroke-width:2; stroke-dasharray:7 5; }
#aurocsvg .dot { stroke:var(--panel); stroke-width:2; }
#aurocsvg .leader { fill:none; stroke-width:1.25; opacity:.8; }
#aurocsvg .endlab { text-anchor:start; }
#aurocsvg .endlab .v { fill:var(--fg); font-weight:700; font-variant-numeric:tabular-nums; }
#aurocsvg .endlab .n { font-size:11px; }
/* SVG elements ignore the `hidden` attribute's UA rule — hide it explicitly */
#aurocsvg .cross { stroke:var(--muted); stroke-width:1; opacity:.55; }
#aurocsvg .cross[hidden] { display:none; }
#aurocsvg .hit { cursor:crosshair; }
#aurocsvg .dot.dim { opacity:.28; }
.tip { position:absolute; top:6px; z-index:5; pointer-events:none;
  background:var(--bg); border:1px solid var(--line); border-radius:8px;
  padding:8px 10px; font-size:12.5px; min-width:132px;
  box-shadow:0 6px 20px rgba(0,0,0,.16); }
.tip h4 { margin:0 0 6px; font-size:11px; text-transform:uppercase;
  letter-spacing:.06em; color:var(--muted); }
.tip .row { display:flex; align-items:center; gap:7px; margin:3px 0; }
.tip .rk { display:inline-block; width:12px; height:2px; border-radius:1px; flex:none; }
.tip .rk.mean { height:0; border-top:2px dashed var(--fg); background:none !important; }
.tip .rv { font-weight:700; font-variant-numeric:tabular-nums; color:var(--fg);
  min-width:42px; }
.tip .rn { color:var(--muted); }
figcaption { margin-top:8px; font-size:12px; color:var(--muted); }
@media (max-width:720px) { #aurocsvg text { font-size:14px; } }
"""

SCRIPT = """
(function () {
  var root = document.documentElement;
  var btn = document.getElementById("themebtn");
  function setTheme(t) {
    root.setAttribute("data-theme", t);
    try { localStorage.setItem("theme", t); } catch (e) {}
    btn.textContent = t === "dark" ? "light mode" : "dark mode";
  }
  var cur = null;
  try { cur = localStorage.getItem("theme"); } catch (e) {}
  if (!cur) cur = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  setTheme(cur);
  btn.addEventListener("click", function () {
    setTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });

  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tabs button"));
  tabs.forEach(function (t) {
    t.addEventListener("click", function () {
      tabs.forEach(function (o) {
        o.setAttribute("aria-selected", String(o === t));
        var p = document.getElementById("panel-" + o.dataset.tab);
        if (p) p.classList.toggle("active", o === t);
      });
    });
  });

  document.querySelectorAll("[data-filters]").forEach(function (panel) {
    var q = panel.querySelector('input[type="search"]');
    var cls = panel.querySelector("select.cls");
    var sort = panel.querySelector("select.sort");
    var count = panel.querySelector(".count");
    var host = panel.querySelector(".list");
    var cards = Array.prototype.slice.call(host.querySelectorAll(".samp, .pair"));
    function apply() {
      var needle = (q && q.value || "").trim().toLowerCase();
      var want = cls ? cls.value : "all";
      var shown = 0;
      cards.forEach(function (c) {
        var ok = true;
        if (want !== "all" && c.dataset.cls !== want) ok = false;
        if (ok && needle && (c.dataset.hay || "").indexOf(needle) === -1) ok = false;
        c.style.display = ok ? "" : "none";
        if (ok) shown++;
      });
      if (sort && sort.value !== "none") {
        var dir = sort.value === "worst" ? 1 : -1;
        cards.slice().sort(function (a, b) {
          return dir * (parseFloat(a.dataset.delta) - parseFloat(b.dataset.delta));
        }).forEach(function (c) { host.appendChild(c); });
      }
      if (count) count.textContent = shown + " of " + cards.length + " shown";
    }
    [q, cls, sort].forEach(function (el) {
      if (el) el.addEventListener(el.tagName === "SELECT" ? "change" : "input", apply);
    });
    apply();
  });

  // ---- AUROC chart: crosshair + one tooltip listing every series
  var raw = document.getElementById("aurocdata");
  var svg = document.getElementById("aurocsvg");
  var tip = document.getElementById("auroctip");
  if (raw && svg && tip) {
    var D = JSON.parse(raw.textContent);
    var cross = svg.querySelector(".cross");
    var hit = svg.querySelector(".hit");
    var dots = Array.prototype.slice.call(svg.querySelectorAll(".dot"));
    var cur = -1;

    function nearest(vx) {
      var best = 0;
      for (var i = 1; i < D.x.length; i++) {
        if (Math.abs(D.x[i] - vx) < Math.abs(D.x[best] - vx)) best = i;
      }
      return best;
    }
    function show(i) {
      if (i === cur) return;
      cur = i;
      cross.setAttribute("x1", D.x[i]);
      cross.setAttribute("x2", D.x[i]);
      cross.removeAttribute("hidden");
      dots.forEach(function (d) {
        var on = Math.abs(parseFloat(d.getAttribute("cx")) - D.x[i]) < 0.6;
        d.classList.toggle("dim", !on);
      });
      tip.textContent = "";
      var h = document.createElement("h4");
      h.textContent = D.rounds[i];                       // untrusted label → textContent
      tip.appendChild(h);
      D.series.forEach(function (s) {
        var v = s.vals[i];
        if (v === null || v === undefined) return;
        var row = document.createElement("div");
        row.className = "row";
        var k = document.createElement("i");
        k.className = "rk" + (s.name === "mean" ? " mean" : "");
        k.style.background = s.color;
        var val = document.createElement("span");
        val.className = "rv";
        val.textContent = v.toFixed(3);
        var nm = document.createElement("span");
        nm.className = "rn";
        nm.textContent = s.name;
        row.appendChild(k); row.appendChild(val); row.appendChild(nm);
        tip.appendChild(row);
      });
      var pct = (D.x[i] / D.w) * 100;
      var right = pct > 55;
      tip.style.left = right ? "auto" : "calc(" + pct + "% + 14px)";
      tip.style.right = right ? "calc(" + (100 - pct) + "% + 14px)" : "auto";
      tip.hidden = false;
    }
    function hide() {
      cur = -1;
      tip.hidden = true;
      cross.setAttribute("hidden", "");
      dots.forEach(function (d) { d.classList.remove("dim"); });
    }
    function vx(ev) {
      var r = svg.getBoundingClientRect();
      return ((ev.clientX - r.left) / r.width) * D.w;
    }
    hit.addEventListener("pointermove", function (e) { show(nearest(vx(e))); });
    hit.addEventListener("pointerleave", hide);
    svg.addEventListener("focus", function () { show(cur < 0 ? 0 : cur); });
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", function (e) {
      var d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!d) return;
      e.preventDefault();
      show(Math.max(0, Math.min(D.x.length - 1, (cur < 0 ? 0 : cur) + d)));
    });
  }
})();
"""


def filter_bar(with_class: bool, with_sort: bool) -> str:
    bits = ['<input type="search" placeholder="search conversation text…">']
    if with_class:
        bits.append(
            '<select class="cls"><option value="all">both classes</option>'
            '<option value="pos">high-stakes only</option>'
            '<option value="neg">low-stakes only</option></select>'
        )
    if with_sort:
        bits.append(
            '<select class="sort"><option value="worst">biggest score drop first</option>'
            '<option value="least">smallest drop first</option>'
            '<option value="none">split order</option></select>'
        )
    bits.append('<span class="count"></span>')
    return f'<div class="controls">{"".join(bits)}</div>'


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-dir", type=Path, required=True)
    p.add_argument("--per-sample", default="probe_iter2_vs_probe_iter3_mt_mts.jsonl")
    p.add_argument("--token-stats", default="probe_iter3_pair_token_stats.json")
    p.add_argument("--redteam-a", default="redteam_postprocessed_iter2.jsonl")
    p.add_argument("--redteam-b", default="redteam_postprocessed_iter3.jsonl")
    p.add_argument("--comparison-csv", default="comparison.csv")
    p.add_argument("--eval-dataset-dir", type=Path, default=REPO_ROOT / "eval_datasets")
    p.add_argument(
        "--combine-consecutive-messages", action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--convert-tool-to-assistant", action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--title",
        default="set A — what the 14 iteration-3 pairs did to mt / mts",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows = [json.loads(l) for l in (args.probe_dir / args.per_sample).open()]
    stats = json.load((args.probe_dir / args.token_stats).open())
    stats_by_key = {s["key"]: s for s in stats}
    pairs = load_pairs(args.probe_dir, args.redteam_a, args.redteam_b)
    metrics = load_csv(args.probe_dir / args.comparison_csv)

    splits = sorted({r["split"] for r in rows})
    convs = load_eval_conversations(
        args.eval_dataset_dir, splits,
        args.combine_consecutive_messages, args.convert_tool_to_assistant,
    )

    from sklearn.metrics import roc_auc_score

    summary = []
    for name in splits:
        sub = [r for r in rows if r["split"] == name]
        y = [r["y"] for r in sub]
        summary.append(
            {
                "split": name,
                "n": len(sub),
                "auroc_a": roc_auc_score(y, [r["score_a"] for r in sub]),
                "auroc_b": roc_auc_score(y, [r["score_b"] for r in sub]),
                "regressions": sum(1 for r in sub if r["status"] == "regression"),
                "improvements": sum(1 for r in sub if r["status"] == "improvement"),
                "shift_pos": (
                    sum(r["delta"] for r in sub if r["y"] == 1)
                    / max(1, sum(1 for r in sub if r["y"] == 1))
                ),
                "shift_neg": (
                    sum(r["delta"] for r in sub if r["y"] == 0)
                    / max(1, sum(1 for r in sub if r["y"] == 0))
                ),
            }
        )

    pos_label = next((r["label"] for r in rows if r["y"] == 1), "high-stakes")
    neg_label = next((r["label"] for r in rows if r["y"] == 0), "low-stakes")

    tabs = [("mech", "Mechanism")]
    panels = [render_mechanism(summary, metrics)]

    pair_cards = "".join(
        render_pair(i, k, v, stats_by_key.get(k), neg_label, pos_label)
        for i, (k, v) in enumerate(pairs.items())
    )
    tabs.append(("pairs", f"The {len(pairs)} added pairs"))
    panels.append(
        f"""
<section class="tabpanel" id="panel-pairs" data-filters>
  <div class="note">The only difference between <code>probe_iter2</code>'s training
  data and <code>probe_iter3</code>'s. Each original is a low-stakes red-team success;
  each contrastive is the hand-written opposite-class counterpart. Where the two share
  a byte-identical opening it is shown once, spanning both columns.</div>
  {filter_bar(with_class=False, with_sort=False)}
  <div class="list">{pair_cards or '<p class="empty">No added pairs.</p>'}</div>
</section>"""
    )

    for name in splits:
        regs = [r for r in rows if r["split"] == name and r["status"] == "regression"]
        regs.sort(key=lambda r: r["delta"])
        cards = "".join(render_sample(r, convs[name][r["idx"]], pos_label) for r in regs)
        tabs.append((f"reg-{name}", f"{name} regressions ({len(regs)})"))
        s = next(x for x in summary if x["split"] == name)
        panels.append(
            f"""
<section class="tabpanel" id="panel-reg-{esc(name)}" data-filters>
  <div class="note">Samples <code>probe_iter2</code> classified correctly at the 0.5
  threshold and <code>probe_iter3</code> got wrong. Split AUROC
  {s['auroc_a']:.3f} &rarr; {s['auroc_b']:.3f}; mean score shift
  {s['shift_pos']:+.3f} on the positive class and {s['shift_neg']:+.3f} on the
  negative class.</div>
  {filter_bar(with_class=True, with_sort=True)}
  <div class="list">{cards or '<p class="empty">No regressions.</p>'}</div>
</section>"""
        )

    tabbar = "".join(
        f'<button data-tab="{esc(t)}" aria-selected="{"true" if i == 0 else "false"}">'
        f"{esc(lbl)}</button>"
        for i, (t, lbl) in enumerate(tabs)
    )
    panels[0] = panels[0].replace('class="tabpanel"', 'class="tabpanel active"', 1)

    total_reg = sum(s["regressions"] for s in summary)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(args.title)}</title>
<style>{STYLE}</style></head>
<body>
<button class="themebtn" id="themebtn">dark mode</button>
<div class="wrap">
  <h1>{esc(args.title)}</h1>
  <p class="sub">{len(pairs)} added contrastive pairs &middot; {total_reg} regressed
  eval samples across {len(splits)} splits &middot; probe
  <code>{esc(args.probe_dir.name)}</code></p>
  <div class="tabs">{tabbar}</div>
  {"".join(panels)}
</div>
<script>{SCRIPT}</script>
</body></html>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({len(page)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
