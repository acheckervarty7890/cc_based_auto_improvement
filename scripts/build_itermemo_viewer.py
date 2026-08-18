#!/usr/bin/env python
"""Build a self-contained HTML viewer for the cross-iteration-memo ablation.

Two things, in one file:

1. **The cross-iteration memos**, run 1 beside run 2, one row per iteration. Only the
   *memo* arms have any — the nomemo arms ran with ``cross_iteration_memos: false``, so
   their absence is the ablated variable, not missing data, and the viewer says so rather
   than silently showing three columns instead of four.

2. **Every red-team success, beside the contrastive pair it was turned into** before it
   entered the retrain. This is the part that is otherwise hard to see: the JSONL holds the
   attack, ``contrastive_cache.jsonl`` holds the LLM-written opposite-class twin, and
   nothing joins them. The retrain trains on *both* halves — the success as a
   ``low-stakes`` sample and its twin as ``high-stakes`` — so reading them side by side is
   the only way to check the pair is a minimal edit rather than a different scenario.

Pairing goes through ``contrastive_cache.jsonl``'s ``original_messages`` field, the
authoritative link (see build_contrastive_pairs_viewer.py for why positional alignment on
the postprocessed dump is wrong once the cache is warm). Each arm has its OWN cache, so the
64 seeded pre-history successes — identical conversations in all four arms — carry four
independently generated twins; comparing them across tabs shows how much the contrastive
step itself varies.

Successes from the seeded pre-history (``iteration: -1``, imported from experiment6_cloud)
are included and badged as such: they are not this run's finds, but they ARE in every
retrain's training set, so a view of "what trained these probes" that omitted them would
mislead.

Usage:
    .venv_claude/bin/python scripts/build_itermemo_viewer.py
    .venv_claude/bin/python scripts/build_itermemo_viewer.py --out viewers/other.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ARMS = [
    ("R1 nomemo", "1", "nomemo", "results_itermemo_nomemo", "probes/itermemo_nomemo"),
    ("R1 memo", "1", "memo", "results_itermemo_memo", "probes/itermemo_memo"),
    ("R2 nomemo", "2", "nomemo", "results_itermemo_nomemo_run2", "probes/itermemo_nomemo_run2"),
    ("R2 memo", "2", "memo", "results_itermemo_memo_run2", "probes/itermemo_memo_run2"),
]


def canon(messages: list[dict]) -> str:
    return json.dumps(
        [{"role": m.get("role", ""), "content": m.get("content", "")} for m in messages],
        sort_keys=True,
    )


def load_cache(path: Path) -> dict[str, dict]:
    """Map canonical SOURCE conversation -> the contrastive record generated from it."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)["record"]
        src = rec.get("original_messages")
        if src:
            out[canon(src)] = rec
    return out


def build_data() -> dict:
    pool: list[list[dict]] = []
    index: dict[str, int] = {}

    def intern(messages: list[dict]) -> int:
        k = canon(messages)
        if k not in index:
            index[k] = len(pool)
            pool.append(
                [{"r": m.get("role", ""), "c": m.get("content", "")} for m in messages]
            )
        return index[k]

    arms_out = []
    memos_by_run: dict[str, list[dict]] = {}

    for label, run, kind, res, probe_dir in ARMS:
        res_p, probe_p = REPO / res, REPO / probe_dir
        cache = load_cache(probe_p / "contrastive_cache.jsonl")

        items, n_paired = [], 0
        jsonl = res_p / "gptoss120b_probing.jsonl"
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("success"):
                continue
            msgs = r["sample"]["messages"]
            rec = cache.get(canon(msgs))
            if rec:
                n_paired += 1
            items.append(
                {
                    "it": int(r.get("iteration", -1)),
                    "rd": int(r.get("round", 0)),
                    "sc": round(float(r.get("probe_score", 0.0)), 3),
                    "jl": r.get("judge_label", ""),
                    "jc": int(r.get("judge_confidence", 0)),
                    "jr": r.get("judge_reason", ""),
                    "am": r.get("attacker_model", ""),
                    "src": intern(msgs),
                    "gen": intern(rec["inputs"]) if rec else None,
                    "gl": (rec or {}).get("labels", ""),
                    "sl": (rec or {}).get("original_label", ""),
                    "ex": (rec or {}).get("generation_explanation", ""),
                    "gm": (rec or {}).get("generation_model", ""),
                }
            )

        arms_out.append(
            {
                "label": label,
                "run": run,
                "kind": kind,
                "jsonl": str(jsonl.relative_to(REPO)),
                "cache": str((probe_p / "contrastive_cache.jsonl").relative_to(REPO)),
                "items": items,
                "n_paired": n_paired,
            }
        )

        memo_path = res_p / "gptoss120b_probing.iteration_memos.jsonl"
        if memo_path.exists():
            rows = [
                json.loads(l)
                for l in memo_path.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            memos_by_run[run] = [
                {
                    "it": int(m.get("iteration", -1)),
                    "text": m.get("text", ""),
                    "na": int(m.get("n_attempts", 0)),
                    "ns": int(m.get("n_successes", 0)),
                    "words": len(m.get("text", "").split()),
                    "path": str(memo_path.relative_to(REPO)),
                }
                for m in rows
            ]

    return {"pool": pool, "arms": arms_out, "memos": memos_by_run}


CSS = """
:root {
  color-scheme: light dark;
  --bg:#fff; --panel:#f6f7f9; --panel2:#eef0f3; --line:#d9dde3;
  --fg:#1a1d21; --muted:#666e79; --accent:#3f6fd8;
  --pos-bg:#fdeeee; --pos-fg:#9d2727; --neg-bg:#eaf5ee; --neg-fg:#1e6b3c;
  --seed:#8a5b00; --seed-bg:#fdf3dc;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
    --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
    --pos-bg:#2c1c1e; --pos-fg:#ff9a9a; --neg-bg:#17281d; --neg-fg:#86e0a5;
    --seed:#f0c060; --seed-bg:#33290f;
  }
}
:root[data-theme="dark"] {
  --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
  --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
  --pos-bg:#2c1c1e; --pos-fg:#ff9a9a; --neg-bg:#17281d; --neg-fg:#86e0a5;
  --seed:#f0c060; --seed-bg:#33290f;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1440px; margin:0 auto; padding:22px 18px 80px; }
h1 { font-size:21px; margin:0 0 4px; }
h2 { font-size:16px; margin:26px 0 10px; }
.sub { color:var(--muted); font-size:13px; margin-bottom:16px; }
.path { color:var(--muted); font-size:12px; font-family:ui-monospace,Menlo,monospace; }
.note { background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:6px; padding:10px 13px; font-size:13.5px; color:var(--muted); margin:12px 0 18px; }
.note code { font-family:ui-monospace,Menlo,monospace; font-size:12px; }
.tabs { display:flex; gap:2px; border-bottom:1px solid var(--line); margin:18px 0 0; flex-wrap:wrap; }
.tab { appearance:none; border:1px solid transparent; border-bottom:none; background:none;
  color:var(--muted); font:inherit; font-size:14px; padding:8px 14px; cursor:pointer;
  border-radius:6px 6px 0 0; margin-bottom:-1px; }
.tab[aria-selected="true"] { color:var(--fg); background:var(--bg);
  border-color:var(--line); border-bottom:1px solid var(--bg); font-weight:600; }
.controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:16px 0 12px; }
select, input[type=search] { font:inherit; font-size:13.5px; padding:6px 9px; border-radius:6px;
  border:1px solid var(--line); background:var(--bg); color:var(--fg); }
input[type=search] { min-width:260px; }
.count { color:var(--muted); font-size:13px; }
.card { border:1px solid var(--line); border-radius:9px; margin:0 0 16px; overflow:hidden;
  background:var(--panel); }
.head { display:flex; gap:8px; align-items:center; flex-wrap:wrap;
  padding:9px 13px; border-bottom:1px solid var(--line); background:var(--panel2); }
.badge { font-size:11.5px; padding:2px 8px; border-radius:20px; border:1px solid var(--line);
  background:var(--bg); color:var(--muted); white-space:nowrap; }
.badge.seed { background:var(--seed-bg); color:var(--seed); border-color:transparent; }
.badge.pos { background:var(--pos-bg); color:var(--pos-fg); border-color:transparent; }
.badge.neg { background:var(--neg-bg); color:var(--neg-fg); border-color:transparent; }
.grid { display:grid; grid-template-columns:1fr 1fr; }
@media (max-width:900px) { .grid { grid-template-columns:1fr; } }
.side { padding:11px 14px; background:var(--bg); min-width:0; }
.side + .side { border-left:1px solid var(--line); }
@media (max-width:900px) { .side + .side { border-left:none; border-top:1px solid var(--line); } }
.sidehead { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted);
  margin-bottom:8px; display:flex; gap:7px; align-items:center; }
.msg { margin:0 0 8px; }
.role { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.content { white-space:pre-wrap; overflow-wrap:anywhere; font-size:13.5px; margin-top:2px; }
.foot { padding:9px 14px; border-top:1px solid var(--line); font-size:13px; color:var(--muted); }
.foot b { color:var(--fg); font-weight:600; }
.memogrid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
@media (max-width:900px) { .memogrid { grid-template-columns:1fr; } }
.memo { border:1px solid var(--line); border-radius:9px; background:var(--panel); overflow:hidden; }
.memo .body { padding:12px 14px; white-space:pre-wrap; font-size:13.5px; background:var(--bg); }
.missing { padding:12px 14px; font-size:13.5px; color:var(--muted); background:var(--bg); }
.hidden { display:none; }
.more { color:var(--accent); cursor:pointer; font-size:12.5px; user-select:none; }
"""


def render(data: dict, out: Path, title: str) -> None:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>{html.escape(title)}</h1>
<div class="sub">Cross-iteration-memo ablation &mdash; gpt-oss-120b attacker, batch mode,
<code>view_limit: 0</code>, high-stakes / llama-1b probe, <code>error_type: false_positive</code>.</div>
<div class="tabs" id="tabs"></div>
<div id="view"></div>
</div>
<script>
const D = {payload};
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c =>
  ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;"}}[c]));

function convHTML(idx) {{
  if (idx === null || idx === undefined) return '<div class="missing">no contrastive pair &mdash; generation failed for this conversation and failures are never cached, so every retrain retried and re-failed it. It trained as a lone negative, with no positive twin.</div>';
  return D.pool[idx].map(m =>
    '<div class="msg"><div class="role">' + esc(m.r) + '</div><div class="content">' +
    esc(m.c) + '</div></div>').join("");
}}

function itLabel(it) {{
  return it < 0 ? '<span class="badge seed">seeded pre-history</span>'
                : '<span class="badge">iteration ' + it + '</span>';
}}

function renderMemos() {{
  const runs = ["1", "2"];
  let h = '<div class="note">The <b>nomemo</b> arms of both runs produced no memos at all &mdash; ' +
    'they ran with <code>cross_iteration_memos: false</code>. That absence is the ablated variable. ' +
    'Each memo is written by the judge after its iteration\\u2019s rotation and <i>before</i> the retrain, ' +
    'then injected into the NEXT iteration\\u2019s attacker system prompts &mdash; so the memo for the ' +
    'final iteration is written but never consumed.</div>';
  const its = [...new Set(runs.flatMap(r => (D.memos[r] || []).map(m => m.it)))].sort((a,b)=>a-b);
  for (const it of its) {{
    h += '<h2>Iteration ' + it + (it === Math.max(...its) ? ' &mdash; written, never consumed' : '') + '</h2>';
    h += '<div class="memogrid">';
    for (const r of runs) {{
      const m = (D.memos[r] || []).find(x => x.it === it);
      h += '<div class="memo"><div class="head"><span class="badge">run ' + r + ' &middot; memo arm</span>' +
        (m ? '<span class="badge">' + m.ns + '/' + m.na + ' succeeded</span><span class="badge">' +
             m.words + ' words</span>' : '') + '</div>';
      h += m ? '<div class="body">' + esc(m.text) + '</div>'
             : '<div class="missing">no memo for this iteration</div>';
      h += '</div>';
    }}
    h += '</div>';
  }}
  return h;
}}

function renderArm(ai) {{
  const arm = D.arms[ai];
  const its = [...new Set(arm.items.map(i => i.it))].sort((a,b)=>a-b);
  let h = '<div class="note">Every successful red-team conversation from <code>' + esc(arm.jsonl) +
    '</code>, each beside the opposite-class twin written for it during preprocessing (' +
    '<code>' + esc(arm.cache) + '</code>). <b>Both halves enter the retrain</b> &mdash; the attack as ' +
    'a <i>low-stakes</i> sample, the twin as <i>high-stakes</i>. ' + arm.n_paired + ' of ' +
    arm.items.length + ' successes have a twin.' +
    (arm.kind === "nomemo" ? ' This arm ran <b>without</b> cross-iteration memos.'
                           : ' This arm ran <b>with</b> cross-iteration memos.') + '</div>';
  h += '<div class="controls"><label>Iteration <select id="fit"><option value="">all</option>' +
    its.map(i => '<option value="' + i + '">' + (i < 0 ? 'seeded pre-history' : i) + '</option>').join("") +
    '</select></label>' +
    '<input type="search" id="fq" placeholder="search conversation text\\u2026">' +
    '<span class="count" id="cnt"></span></div><div id="list"></div>';
  return h;
}}

function paint(ai) {{
  const arm = D.arms[ai];
  const fit = document.getElementById("fit").value;
  const q = document.getElementById("fq").value.trim().toLowerCase();
  const rows = arm.items.filter(i => {{
    if (fit !== "" && String(i.it) !== fit) return false;
    if (!q) return true;
    const t = D.pool[i.src].map(m => m.c).join(" ").toLowerCase() +
      (i.gen != null ? D.pool[i.gen].map(m => m.c).join(" ").toLowerCase() : "");
    return t.includes(q);
  }});
  document.getElementById("cnt").textContent =
    rows.length + " of " + arm.items.length + " successes";
  document.getElementById("list").innerHTML = rows.map((i, n) =>
    '<div class="card"><div class="head">' + itLabel(i.it) +
      '<span class="badge">round ' + i.rd + '</span>' +
      '<span class="badge pos">probe ' + i.sc.toFixed(3) + ' &rarr; high-stakes</span>' +
      '<span class="badge neg">judge ' + esc(i.jl) + ' (conf ' + i.jc + ')</span>' +
      '<span class="badge">' + esc(i.am) + '</span></div>' +
    '<div class="grid">' +
      '<div class="side"><div class="sidehead">red-team success' +
        (i.sl ? ' &mdash; trains as <b>' + esc(i.sl) + '</b>' : '') + '</div>' + convHTML(i.src) + '</div>' +
      '<div class="side"><div class="sidehead">contrastive twin' +
        (i.gl ? ' &mdash; trains as <b>' + esc(i.gl) + '</b>' : '') + '</div>' + convHTML(i.gen) + '</div>' +
    '</div>' +
    '<div class="foot"><b>Judge:</b> ' + esc(i.jr) +
      (i.ex ? '<br><b>Contrastive edit:</b> ' + esc(i.ex) : '') + '</div></div>'
  ).join("") || '<div class="missing">nothing matches this filter.</div>';
}}

function show(tab) {{
  document.querySelectorAll("#tabs .tab").forEach((b, n) =>
    b.setAttribute("aria-selected", String(n === tab)));
  const v = document.getElementById("view");
  if (tab === 0) {{ v.innerHTML = renderMemos(); return; }}
  const ai = tab - 1;
  v.innerHTML = renderArm(ai);
  document.getElementById("fit").onchange = () => paint(ai);
  document.getElementById("fq").oninput = () => paint(ai);
  paint(ai);
}}

const names = ["Iteration memos"].concat(D.arms.map(a =>
  a.label + " (" + a.items.length + ")"));
document.getElementById("tabs").innerHTML = names.map(n =>
  '<button class="tab" role="tab">' + esc(n) + '</button>').join("");
document.querySelectorAll("#tabs .tab").forEach((b, n) => b.onclick = () => show(n));
show(0);
</script></body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=REPO / "viewers/itermemo_ablation_viewer.html")
    p.add_argument("--title", default="Cross-iteration memos — memos, successes and contrastive pairs")
    args = p.parse_args(argv)

    data = build_data()
    render(data, args.out, args.title)

    print(f"wrote {args.out.relative_to(REPO)}  ({args.out.stat().st_size / 1e6:.2f} MB)")
    print(f"  {len(data['pool'])} distinct conversations pooled")
    for a in data["arms"]:
        seeded = sum(1 for i in a["items"] if i["it"] < 0)
        print(f"  {a['label']:<12} {len(a['items']):>4} successes "
              f"({seeded} seeded + {len(a['items']) - seeded} own), "
              f"{a['n_paired']} with a contrastive twin")
    for run, memos in sorted(data["memos"].items()):
        print(f"  run {run} memo arm: {len(memos)} iteration memos "
              f"({', '.join(str(m['words']) + 'w' for m in memos)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
