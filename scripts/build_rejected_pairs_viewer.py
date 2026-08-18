#!/usr/bin/env python
"""Build a self-contained HTML viewer of the contrastive pairs rejected in review.

Input is a ``review.jsonl`` produced by the pair-review pass over a run's
``redteam_postprocessed_iter*.jsonl`` + ``contrastive_cache.jsonl``: one row per
original <-> contrastive pair, carrying the verdict (KEEP / DROP), the reason a pair
was dropped, which iterations it appeared in, and both conversations with their
labels.

Only the DROPped pairs are embedded (that is what the viewer is for); the KEEP count
is still reported in the header so the rejection rate is visible. Each pair is
bucketed into a rejection category derived from its reason text, and the categories
become filter chips.

With ``--comparison-csv`` pairs of (original, pruned) run CSVs, a second tab shows
what removing these pairs did to the retrained probes' eval metrics.

Usage:
    .venv_claude/bin/python scripts/build_rejected_pairs_viewer.py \
        --review probes/hs_llama1b_deepseekv4pro_guidance_pruned/review.jsonl \
        --orig-csv results_hs_llama1b_deepseekv4pro_guidance/deepseekv4pro_comparison.csv \
        --pruned-csv probes/hs_llama1b_deepseekv4pro_guidance_pruned/comparison.csv \
        --title "deepseek-v4-pro (guidance) - rejected contrastive pairs" \
        --out viewers/deepseekv4pro_guidance_rejected_viewer.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


# ---------------------------------------------------------------- categories

# Ordered: first matching predicate wins, so put the specific ones first.
CATEGORIES: list[tuple[str, str, str, tuple[str, ...]]] = [
    (
        "meta",
        "Attacker meta-chatter",
        "The original is not a conversation at all — it is the attacker announcing "
        "it hit its success target. The generator then invented an unrelated scenario "
        "around the phrase, so the two members are not a class flip.",
        ("meta-chatter",),
    ),
    (
        "mirror",
        "Contrastive mirrors the original",
        "The contrastive keeps the original's structure — it also ends by revealing "
        "the emergency was a simulation or a false alarm — so both members read as "
        "the same class.",
        ("mirror", "keeps the ORIGINAL structure", "keeps the original structure",
         "same kind of conversation"),
    ),
    (
        "pet",
        "Pet-fish care as high-stakes",
        "The original is routine aquarium care labelled high-stakes, and the "
        "contrastive switches topic rather than flipping the class.",
        ("pet-fish", "pet-care"),
    ),
    (
        "label",
        "Label does not hold",
        "One member's own label is not clearly right on the conversation's merits — "
        "too neutral to be high-stakes, or the reverse.",
        ("not clearly high-stakes", "not clearly low-stakes", "not clear",
         "not high-stakes", "deflates", "the stated situation is real"),
    ),
]
FALLBACK_CAT = ("other", "Other", "Reviewed as not a valid contrastive pair.")


def categorise(reason: str) -> str:
    for key, _label, _desc, needles in CATEGORIES:
        if any(n in reason for n in needles):
            return key
    return FALLBACK_CAT[0]


def category_meta() -> dict[str, tuple[str, str]]:
    meta = {k: (lbl, desc) for k, lbl, desc, _ in CATEGORIES}
    meta[FALLBACK_CAT[0]] = (FALLBACK_CAT[1], FALLBACK_CAT[2])
    return meta


# ---------------------------------------------------------------- data loading

def load_review(path: Path) -> tuple[list[dict], int, dict[str, str]]:
    """Return ``(rejected_rows, n_kept, canonical->human label map)``.

    The postprocessed dump labels samples with tuberlens' canonical
    ``positive``/``negative`` while the contrastive cache carries the probe's own
    class strings, so the two sides of a pair speak different vocabularies. The
    contrastive is by construction the *opposite* class of its original, which is
    enough to recover the mapping from the data itself — no probe load needed.
    """
    rejected: list[dict] = []
    kept = 0
    votes: dict[str, dict[str, int]] = {"positive": {}, "negative": {}}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        oc = (row.get("original") or {}).get("label")
        cl = (row.get("contrastive") or {}).get("label")
        if oc in ("positive", "negative") and cl:
            other = "negative" if oc == "positive" else "positive"
            votes[other][cl] = votes[other].get(cl, 0) + 1
        if row.get("verdict") != "DROP":
            kept += 1
            continue
        row["category"] = categorise(row.get("reason", ""))
        rejected.append(row)
    rejected.sort(key=lambda r: r["pair_id"])

    label_map = {
        canonical: max(counts, key=counts.get)
        for canonical, counts in votes.items()
        if counts
    }
    return rejected, kept, label_map


def load_comparison(path: Path | None) -> dict[tuple[str, str], dict[str, float]]:
    if path is None or not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["round"], row["dataset"])
            out[key] = {
                k: float(row[k])
                for k in ("auroc", "accuracy", "tpr_at_fpr")
                if row.get(k) not in (None, "")
            }
    return out


# ---------------------------------------------------------------- rendering

def esc(s: object) -> str:
    return html.escape(str(s), quote=False)


def render_turns(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = esc(m.get("role", ""))
        text = esc(m.get("content", ""))
        parts.append(
            f'<div class="turn"><div class="role">{role}</div>'
            f'<div class="text">{text}</div></div>'
        )
    return "".join(parts)


LABEL_CLASS = {"positive": "pos", "negative": "neg"}


def display_label(label: str, label_map: dict[str, str]) -> str:
    """Human-readable class name, with the canonical enum kept as a hint."""
    lab = (label or "?").strip()
    human = label_map.get(lab)
    return f"{human} ({lab})" if human else lab


def side_class(label: str) -> str:
    lab = (label or "").lower()
    if lab in LABEL_CLASS:
        return LABEL_CLASS[lab]
    if "high" in lab:
        return "pos"
    if "low" in lab:
        return "neg"
    return ""


def render_pair(row: dict, cat_labels: dict[str, tuple[str, str]],
                label_map: dict[str, str]) -> str:
    orig = row.get("original") or {}
    contr = row.get("contrastive") or {}
    cat = row["category"]
    cat_label = cat_labels[cat][0]
    iters = row.get("in_iters") or []
    iter_txt = ", ".join(f"iter{i}" for i in iters) if iters else "—"

    gen = (row.get("generation_explanation") or "").strip()
    gen_block = (
        '<details class="why"><summary>Generator’s own justification</summary>'
        f'<div class="body">{esc(gen)}</div></details>'
        if gen
        else ""
    )

    haystack = json.dumps(
        [orig.get("inputs", []), contr.get("inputs", []), row.get("reason", "")],
        ensure_ascii=False,
    ).lower()

    return f"""<article class="pair" data-cat="{esc(cat)}" data-hay="{esc(haystack)}">
  <div class="head">
    <span class="idx">pair {row['pair_id']}</span>
    <span class="badge cat cat-{esc(cat)}">{esc(cat_label)}</span>
    <span class="badge">{esc(iter_txt)}</span>
  </div>
  <div class="reason"><span class="lbl">Why rejected</span>{esc(row.get('reason', ''))}</div>
  <div class="cols">
    <div class="side {side_class(orig.get('label', ''))}">
      <div class="lbl">Original — {esc(display_label(orig.get('label', '?'), label_map))}</div>
      {render_turns(orig.get('inputs', []))}
    </div>
    <div class="side {side_class(contr.get('label', ''))}">
      <div class="lbl">Contrastive — {esc(display_label(contr.get('label', '?'), label_map))}</div>
      {render_turns(contr.get('inputs', []))}
    </div>
  </div>
  {gen_block}
</article>"""


def render_impact(orig: dict, pruned: dict) -> str:
    if not orig or not pruned:
        return ""
    rounds = sorted(
        {r for r, _ in orig} & {r for r, _ in pruned},
        key=lambda r: int("".join(c for c in r if c.isdigit()) or 0),
    )
    splits: list[str] = []
    for r in rounds:
        for rr, ds in orig:
            if rr == r and ds not in splits:
                splits.append(ds)
    splits = [s for s in splits if s != "mean"] + (["mean"] if any(
        (r, "mean") in orig for r in rounds
    ) else [])

    def fmt(v: float | None) -> str:
        return "—" if v is None else f"{v:.4f}"

    def delta(a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return '<td class="d">—</td>'
        d = b - a
        cls = "up" if d > 0.0005 else ("down" if d < -0.0005 else "flat")
        return f'<td class="d {cls}">{d:+.4f}</td>'

    blocks = []
    for metric, nice in (("auroc", "AUROC"), ("accuracy", "Accuracy"),
                         ("tpr_at_fpr", "TPR @ 1% FPR")):
        head = "".join(
            f"<th>{esc(r)} orig</th><th>{esc(r)} pruned</th><th>Δ</th>" for r in rounds
        )
        body = []
        for s in splits:
            cells = []
            for r in rounds:
                a = orig.get((r, s), {}).get(metric)
                b = pruned.get((r, s), {}).get(metric)
                cells.append(f"<td>{fmt(a)}</td><td>{fmt(b)}</td>{delta(a, b)}")
            tr_cls = ' class="mean"' if s == "mean" else ""
            body.append(f"<tr{tr_cls}><td>{esc(s)}</td>{''.join(cells)}</tr>")
        blocks.append(
            f'<div class="card"><h2>{esc(nice)}</h2><div class="tblwrap">'
            f'<table class="ev"><thead><tr><th>split</th>{head}</tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table></div></div>"
        )
    return "".join(blocks)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light dark;
  --bg:#fff; --panel:#f6f7f9; --panel2:#eef0f3; --line:#d9dde3;
  --fg:#1a1d21; --muted:#666e79; --accent:#3f6fd8;
  --pos-bg:#fdeeee; --pos-fg:#9d2727; --neg-bg:#eaf5ee; --neg-fg:#1e6b3c;
  --warn:#8a5b00; --warn-bg:#fdf3dc;
  --up:#1e6b3c; --down:#9d2727;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
    --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
    --pos-bg:#2c1c1e; --pos-fg:#ff9a9a; --neg-bg:#17281d; --neg-fg:#86e0a5;
    --warn:#f0c060; --warn-bg:#33290f;
    --up:#86e0a5; --down:#ff9a9a;
  }
}
:root[data-theme="light"] {
  --bg:#fff; --panel:#f6f7f9; --panel2:#eef0f3; --line:#d9dde3;
  --fg:#1a1d21; --muted:#666e79; --accent:#3f6fd8;
  --pos-bg:#fdeeee; --pos-fg:#9d2727; --neg-bg:#eaf5ee; --neg-fg:#1e6b3c;
  --warn:#8a5b00; --warn-bg:#fdf3dc;
  --up:#1e6b3c; --down:#9d2727;
}
:root[data-theme="dark"] {
  --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
  --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
  --pos-bg:#2c1c1e; --pos-fg:#ff9a9a; --neg-bg:#17281d; --neg-fg:#86e0a5;
  --warn:#f0c060; --warn-bg:#33290f;
  --up:#86e0a5; --down:#ff9a9a;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1280px; margin:0 auto; padding:22px 18px 80px; }
h1 { margin:0 0 4px; font-size:21px; letter-spacing:-.01em; }
.sub { color:var(--muted); font-size:13px; }
.path { color:var(--muted); font-size:12px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; margin-top:2px; overflow-wrap:anywhere; }
.themebtn { position:fixed; top:14px; right:14px; z-index:9; appearance:none; cursor:pointer;
  background:var(--panel); color:var(--muted); border:1px solid var(--line);
  border-radius:999px; padding:6px 12px; font-size:12px; font-weight:600; }
.note { margin:16px 0 0; padding:10px 13px; border-left:3px solid var(--accent);
  background:var(--panel); border-radius:0 6px 6px 0; font-size:13px; color:var(--muted); }
.note code { font-family:ui-monospace,Menlo,monospace; font-size:12px; }
.tabs { display:flex; flex-wrap:wrap; gap:6px; margin-top:20px; border-bottom:1px solid var(--line); }
.tab { appearance:none; border:1px solid transparent; border-bottom:none; background:transparent;
  color:var(--muted); cursor:pointer; padding:9px 15px; font-size:14px; font-weight:600;
  border-radius:7px 7px 0 0; margin-bottom:-1px; }
.tab:hover { color:var(--fg); background:var(--panel); }
.tab[aria-selected="true"] { color:var(--fg); background:var(--bg);
  border-color:var(--line); border-bottom:1px solid var(--bg); }
.tab .cnt { font-weight:400; color:var(--muted); font-size:12px; margin-left:5px; }
.panel[hidden] { display:none; }
.stats { display:flex; flex-wrap:wrap; gap:10px; margin:18px 0 6px; }
.stat { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:9px 13px; min-width:112px; }
.stat .k { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.stat .v { font-size:19px; font-weight:650; margin-top:2px; }
.toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:14px 0 8px; }
.toolbar input[type=search] { flex:1 1 260px; min-width:200px; padding:8px 11px; font-size:14px;
  background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:7px; }
.chip { appearance:none; cursor:pointer; padding:7px 12px; font-size:13px; font-weight:550;
  background:var(--panel); color:var(--muted); border:1px solid var(--line); border-radius:999px; }
.chip[aria-pressed="true"] { background:var(--accent); border-color:var(--accent); color:#fff; }
.count { color:var(--muted); font-size:13px; margin-left:auto; }
.catdesc { font-size:12.5px; color:var(--muted); margin:0 0 16px; min-height:1.2em; }
.pair { border:1px solid var(--line); border-radius:10px; margin-bottom:14px; background:var(--panel); overflow:hidden; }
.pair[hidden] { display:none; }
.pair .head { display:flex; flex-wrap:wrap; gap:8px; align-items:center;
  padding:8px 13px; border-bottom:1px solid var(--line); background:var(--panel2); }
.idx { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; color:var(--muted); }
.badge { font-size:11px; font-weight:700; letter-spacing:.03em; text-transform:uppercase;
  padding:3px 8px; border-radius:999px; border:1px solid var(--line); color:var(--muted); }
.badge.cat { color:var(--warn); background:var(--warn-bg); border-color:var(--warn); }
.reason { padding:10px 15px; border-bottom:1px solid var(--line); background:var(--bg); font-size:13.5px; }
.lbl { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); margin-bottom:6px; display:block; }
.turn { margin-bottom:9px; }
.turn:last-child { margin-bottom:0; }
.role { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.07em;
  color:var(--muted); margin-bottom:3px; }
.text { white-space:pre-wrap; overflow-wrap:anywhere; font-size:14px; }
.cols { display:grid; grid-template-columns:1fr 1fr; }
.side { padding:12px 15px; background:var(--bg); }
.side + .side { border-left:1px solid var(--line); }
@media (max-width:800px) {
  .cols { grid-template-columns:1fr; }
  .side + .side { border-left:none; border-top:1px solid var(--line); }
}
.side.pos { background:var(--pos-bg); }
.side.neg { background:var(--neg-bg); }
.side.pos .lbl { color:var(--pos-fg); }
.side.neg .lbl { color:var(--neg-fg); }
details.why { border-top:1px solid var(--line); background:var(--panel); }
details.why > summary { cursor:pointer; padding:8px 15px; font-size:12px; font-weight:600;
  color:var(--muted); list-style:none; }
details.why > summary::-webkit-details-marker { display:none; }
details.why > summary::before { content:"\25b8 "; }
details.why[open] > summary::before { content:"\25be "; }
details.why .body { padding:0 15px 12px; font-size:13px; color:var(--muted);
  white-space:pre-wrap; overflow-wrap:anywhere; }
.empty { padding:40px 0; text-align:center; color:var(--muted); }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:16px; margin-top:18px; }
.card h2 { font-size:15px; margin:0 0 10px; }
.tblwrap { overflow-x:auto; }
table.ev { border-collapse:collapse; width:100%; font-size:12.5px; font-variant-numeric:tabular-nums; }
table.ev th, table.ev td { padding:5px 9px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
table.ev th:first-child, table.ev td:first-child { text-align:left; }
table.ev thead th { color:var(--muted); font-weight:600; }
table.ev tr.mean td { font-weight:700; border-top:1px solid var(--line); }
table.ev td.d.up { color:var(--up); font-weight:600; }
table.ev td.d.down { color:var(--down); font-weight:600; }
table.ev td.d.flat { color:var(--muted); }
</style>
</head>
<body>
<button class="themebtn" id="themebtn" type="button">theme</button>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="path">__SRC__</div>

  <div class="note">__NOTE__</div>

  <div class="tabs" role="tablist">__TABS__</div>

  <section class="panel" id="panel-pairs" role="tabpanel">
    <div class="stats">__STATS__</div>
    <div class="toolbar">
      <input type="search" id="q" placeholder="Search conversations and reasons…" autocomplete="off">
      __CHIPS__
      <span class="count" id="count"></span>
    </div>
    <p class="catdesc" id="catdesc"></p>
    <div id="list">__PAIRS__</div>
    <div class="empty" id="empty" hidden>No pairs match.</div>
  </section>

  <section class="panel" id="panel-impact" role="tabpanel" hidden>
    <div class="note">Eval of the probes retrained on base data ∪ red-team set,
      before and after removing the rejected pairs. Everything else — base split,
      seed, transforms, activation caches, architecture — is held fixed.
      A control run through the same code path with the <em>unpruned</em> set
      reproduced the original to within ~0.001 mean AUROC, so treat that as the
      noise floor.</div>
    __IMPACT__
  </section>
</div>
<script>
const CATDESC = __CATDESC__;

// theme toggle
(function () {
  const root = document.documentElement;
  const btn = document.getElementById("themebtn");
  let cur = localStorage.getItem("theme");
  if (!cur) cur = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  root.setAttribute("data-theme", cur);
  btn.addEventListener("click", () => {
    cur = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", cur);
    localStorage.setItem("theme", cur);
  });
})();

// tabs
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", on ? "true" : "false");
      document.getElementById(t.dataset.panel).hidden = !on;
    });
  });
});

// filtering
const pairs = Array.from(document.querySelectorAll(".pair"));
const q = document.getElementById("q");
const countEl = document.getElementById("count");
const emptyEl = document.getElementById("empty");
const catdescEl = document.getElementById("catdesc");
let activeCat = "all";

function apply() {
  const needle = q.value.trim().toLowerCase();
  let shown = 0;
  for (const el of pairs) {
    const okCat = activeCat === "all" || el.dataset.cat === activeCat;
    const okQ = !needle || el.dataset.hay.includes(needle);
    const on = okCat && okQ;
    el.hidden = !on;
    if (on) shown++;
  }
  countEl.textContent = shown + " of " + pairs.length + " shown";
  emptyEl.hidden = shown !== 0;
  catdescEl.textContent = CATDESC[activeCat] || "";
}

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    activeCat = chip.dataset.cat;
    document.querySelectorAll(".chip").forEach((c) =>
      c.setAttribute("aria-pressed", c === chip ? "true" : "false")
    );
    apply();
  });
});
q.addEventListener("input", apply);
apply();
</script>
</body>
</html>
"""


def render(
    rejected: list[dict],
    n_kept: int,
    title: str,
    subtitle: str,
    source: str,
    note: str,
    impact_html: str,
    label_map: dict[str, str],
) -> str:
    cat_labels = category_meta()
    counts: dict[str, int] = {}
    for r in rejected:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    total = len(rejected) + n_kept
    stats = [
        ("Pairs reviewed", f"{total}"),
        ("Rejected", f"{len(rejected)}"),
        ("Kept", f"{n_kept}"),
        ("Rows removed", f"{len(rejected) * 2}"),
        ("Rejection rate", f"{len(rejected) / total * 100:.1f}%" if total else "—"),
    ]
    stats_html = "".join(
        f'<div class="stat"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
        for k, v in stats
    )

    chips = [
        f'<button class="chip" type="button" data-cat="all" aria-pressed="true">'
        f"All <span>({len(rejected)})</span></button>"
    ]
    ordered = [k for k, *_ in CATEGORIES] + [FALLBACK_CAT[0]]
    for key in ordered:
        if not counts.get(key):
            continue
        chips.append(
            f'<button class="chip" type="button" data-cat="{esc(key)}" aria-pressed="false">'
            f"{esc(cat_labels[key][0])} <span>({counts[key]})</span></button>"
        )

    catdesc = {"all": "Every pair removed from the retraining set."}
    for key in ordered:
        if counts.get(key):
            catdesc[key] = cat_labels[key][1]

    tabs = [
        '<button class="tab" role="tab" data-panel="panel-pairs" aria-selected="true">'
        f'Rejected pairs <span class="cnt">{len(rejected)}</span></button>'
    ]
    if impact_html:
        tabs.append(
            '<button class="tab" role="tab" data-panel="panel-impact" '
            'aria-selected="false">Impact on eval</button>'
        )

    pairs_html = "\n".join(render_pair(r, cat_labels, label_map) for r in rejected)

    return (
        _TEMPLATE.replace("__TITLE__", esc(title))
        .replace("__SUBTITLE__", subtitle)
        .replace("__SRC__", esc(source))
        .replace("__NOTE__", note)
        .replace("__TABS__", "".join(tabs))
        .replace("__STATS__", stats_html)
        .replace("__CHIPS__", "".join(chips))
        .replace("__PAIRS__", pairs_html)
        .replace("__IMPACT__", impact_html)
        .replace("__CATDESC__", json.dumps(catdesc, ensure_ascii=False))
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review", type=Path, required=True,
                    help="review.jsonl with one {pair_id, verdict, reason, ...} row per pair")
    ap.add_argument("--orig-csv", type=Path, default=None,
                    help="comparison CSV of the original (unpruned) run")
    ap.add_argument("--pruned-csv", type=Path, default=None,
                    help="comparison CSV of the pruned rerun")
    ap.add_argument("--title", default="Rejected contrastive pairs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rejected, n_kept, label_map = load_review(args.review)
    if not rejected:
        print("No rejected pairs in the review file — nothing to render.")
        return 1

    impact_html = render_impact(
        load_comparison(args.orig_csv), load_comparison(args.pruned_csv)
    )

    total = len(rejected) + n_kept
    subtitle = (
        f"{len(rejected)} of {total} original↔contrastive pairs removed from the "
        f"retraining set ({len(rejected) * 2} rows)"
    )
    note = (
        "A pair was rejected unless the <b>original</b> clearly belongs to its label "
        "<b>and</b> the <b>contrastive</b> clearly belongs to the opposite one. Both "
        "members of a rejected pair are removed, so the set stays class-balanced. "
        "Pairs where the concept call is merely arguable were kept."
    )
    src = f"{args.review}"
    if args.orig_csv:
        src += f"  •  {args.orig_csv}"
    if args.pruned_csv:
        src += f"  •  {args.pruned_csv}"

    html_out = render(rejected, n_kept, args.title, subtitle, src, note,
                      impact_html, label_map)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_out, encoding="utf-8")
    print(
        f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB) — "
        f"{len(rejected)} rejected pairs, {n_kept} kept"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
