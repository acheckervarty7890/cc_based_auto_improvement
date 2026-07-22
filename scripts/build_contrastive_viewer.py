#!/usr/bin/env python
"""Build a self-contained HTML viewer for the contrastive-rerun output.

Reads ``contrastive_rerun.jsonl`` (one row per request) and its companion
``contrastive_rerun.samples.jsonl`` (the 15 selected records + exact prompts),
embeds both as JSON directly into a single ``.html`` file (so it opens by
double-click with no server / no CORS issues), and writes it next to the data.

Also derives, per malformed row, a human-readable *reason* it failed the
``_is_well_formed_conversation`` check (single assistant turn / empty list / etc.)
so the viewer can show at a glance that "malformed" == model refusal.

Usage:
    .venv_claude/bin/python scripts/build_contrastive_viewer.py \
        [--dir archive/results_hu_harm_llama70b50_gpt51chat]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO_ROOT / "archive" / "results_hu_harm_llama70b50_gpt51chat"


def malformed_reason(generated_messages) -> str:
    """Mirror _is_well_formed_conversation to explain a rejection in words."""
    msgs = generated_messages or []
    if not msgs:
        return "empty generated_messages ([]) — refusal returned no conversation"
    body = msgs[1:] if msgs[0].get("role") == "system" else msgs
    if not body:
        return "only a system message"
    if len(body) == 1 and body[0].get("role") == "assistant":
        return "single assistant turn (no user turn) — refusal packaged as one assistant message"
    for i, m in enumerate(body):
        expected = "user" if i % 2 == 0 else "assistant"
        if m.get("role") != expected:
            return f"role at position {i} is '{m.get('role')}', expected '{expected}' (non-alternating)"
    return "well-formed"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Contrastive rerun — gpt51chat (not_harmful_to_human &rarr; harmful_to_human)</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --border:#2a2f3a;
    --text:#e6e9ef; --muted:#9aa4b2; --accent:#6ea8fe;
    --pair:#2ea043; --malformed:#d29922; --unparseable:#db6d28;
    --no_choices:#f85149; --exception:#f85149;
  }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f6f7f9; --panel:#fff; --panel2:#f0f2f5; --border:#dfe3ea;
            --text:#1a1d24; --muted:#5a6472; --accent:#1f6feb; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { padding:20px 24px; border-bottom:1px solid var(--border); background:var(--panel); position:sticky; top:0; z-index:5; }
  h1 { margin:0 0 4px; font-size:18px; }
  .sub { color:var(--muted); font-size:13px; }
  .wrap { max-width:1100px; margin:0 auto; padding:20px 24px 80px; }
  .stats { display:flex; flex-wrap:wrap; gap:10px; margin:14px 0 0; }
  .stat { background:var(--panel2); border:1px solid var(--border); border-radius:8px; padding:8px 14px; }
  .stat b { font-size:18px; } .stat span { color:var(--muted); font-size:12px; display:block; }
  .controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:18px 0 6px; }
  .btn { cursor:pointer; user-select:none; border:1px solid var(--border); background:var(--panel);
    color:var(--text); border-radius:20px; padding:5px 13px; font-size:13px; }
  .btn.active { border-color:var(--accent); color:var(--accent); font-weight:600; }
  .tabbar { display:flex; gap:6px; margin-top:6px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px;
    margin:12px 0; overflow:hidden; }
  .card-head { display:flex; align-items:center; gap:10px; padding:11px 15px; cursor:pointer; flex-wrap:wrap; }
  .card-head:hover { background:var(--panel2); }
  .badge { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.4px;
    padding:3px 9px; border-radius:6px; color:#fff; }
  .badge.pair{background:var(--pair)} .badge.malformed{background:var(--malformed)}
  .badge.unparseable{background:var(--unparseable)} .badge.no_choices{background:var(--no_choices)}
  .badge.exception{background:var(--exception)}
  .idlabel { font-weight:600; } .muted { color:var(--muted); }
  .snippet { color:var(--muted); flex:1; min-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .card-body { display:none; padding:0 15px 15px; border-top:1px solid var(--border); }
  .card.open .card-body { display:block; }
  .field { margin-top:12px; }
  .field .k { font-size:12px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted); margin-bottom:4px; }
  pre { background:var(--panel2); border:1px solid var(--border); border-radius:8px; padding:11px 13px;
    margin:0; white-space:pre-wrap; word-break:break-word; font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace; max-height:420px; overflow:auto; }
  .reason { background:rgba(210,153,34,.12); border:1px solid var(--malformed); border-radius:8px;
    padding:9px 12px; color:var(--text); font-size:13px; }
  .msg { border-left:3px solid var(--border); padding:6px 12px; margin:6px 0; background:var(--panel2); border-radius:0 6px 6px 0; }
  .msg .role { font-size:11px; text-transform:uppercase; letter-spacing:.5px; color:var(--accent); font-weight:700; }
  .hidden { display:none !important; }
  .empty { color:var(--muted); font-style:italic; }
  code.inline { background:var(--panel2); padding:1px 6px; border-radius:5px; font-family:ui-monospace,monospace; }
</style>
</head>
<body>
<header>
  <h1>Contrastive rerun &mdash; gpt51chat</h1>
  <div class="sub">Direction: <code class="inline">not_harmful_to_human &rarr; harmful_to_human</code>
    &nbsp;·&nbsp; each of 15 seeded-random red-team successes re-requested <b id="nper"></b>&times;.
    &nbsp;·&nbsp; <span id="mtime"></span></div>
  <div class="stats" id="stats"></div>
</header>
<div class="wrap">
  <div class="tabbar">
    <div class="btn active" data-tab="requests" onclick="showTab('requests')">Requests</div>
    <div class="btn" data-tab="samples" onclick="showTab('samples')">Selected samples &amp; prompts</div>
  </div>

  <div id="tab-requests">
    <div class="controls" id="filters"></div>
    <div id="rows"></div>
  </div>

  <div id="tab-samples" class="hidden">
    <div id="samples"></div>
  </div>
</div>

<script>
const ROWS = __ROWS__;
const SAMPLES = __SAMPLES__;

const STATUS_ORDER = ["malformed","unparseable","pair","no_choices","exception"];
const STATUS_DESC = {
  pair: "well-formed pair returned (may still be a safety/refusal reply — check text)",
  malformed: "parsed, but not a valid user-first alternating conversation — refusal shape",
  unparseable: "no valid JSON object with generated_messages — prose refusal",
  no_choices: "200 + error body, no choices (policy block / provider error)",
  exception: "the SDK raised an error"
};

function esc(s){ return (s==null?"":String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

// ---- stats ----
const counts = {};
ROWS.forEach(r => counts[r.status]=(counts[r.status]||0)+1);
const nper = Math.max(...ROWS.map(r=>r.run_index))+1;
document.getElementById("nper").textContent = nper;
document.getElementById("mtime").textContent = ROWS.length + " total requests · 15 samples";
const statsEl = document.getElementById("stats");
statsEl.innerHTML = `<div class="stat"><b>${ROWS.length}</b><span>requests</span></div>` +
  STATUS_ORDER.filter(s=>counts[s]).map(s =>
    `<div class="stat"><b>${counts[s]}</b><span>${s}</span></div>`).join("");

// ---- filters ----
let active = "all";
const filtersEl = document.getElementById("filters");
const mk = (key,label) => {
  const b=document.createElement("div"); b.className="btn"+(key===active?" active":"");
  b.textContent=label; b.onclick=()=>{active=key; render(); [...filtersEl.children].forEach(c=>c.classList.toggle("active", c===b));};
  return b;
};
filtersEl.appendChild(mk("all","all ("+ROWS.length+")"));
STATUS_ORDER.filter(s=>counts[s]).forEach(s=>filtersEl.appendChild(mk(s, s+" ("+counts[s]+")")));

// ---- malformed reason (mirrors _is_well_formed_conversation) ----
function malformedReason(msgs){
  msgs = msgs||[];
  if(!msgs.length) return "empty generated_messages ([]) — refusal returned no conversation";
  const body = msgs[0].role==="system" ? msgs.slice(1) : msgs;
  if(!body.length) return "only a system message";
  if(body.length===1 && body[0].role==="assistant")
    return "single assistant turn (no user turn) — refusal packaged as one assistant message";
  for(let i=0;i<body.length;i++){ const exp=i%2===0?"user":"assistant";
    if(body[i].role!==exp) return `role at position ${i} is '${body[i].role}', expected '${exp}' (non-alternating)`; }
  return "well-formed";
}

function msgsHtml(msgs){
  if(!msgs || !msgs.length) return '<div class="empty">(no messages)</div>';
  return msgs.map(m=>`<div class="msg"><div class="role">${esc(m.role)}</div><div>${esc(m.content)}</div></div>`).join("");
}

function render(){
  const el=document.getElementById("rows"); el.innerHTML="";
  const list = ROWS.filter(r=>active==="all"||r.status===active);
  list.forEach(r=>{
    const card=document.createElement("div"); card.className="card";
    const reason = r.status==="malformed" ? malformedReason(r.generated_messages) : "";
    card.innerHTML = `
      <div class="card-head" onclick="this.parentElement.classList.toggle('open')">
        <span class="badge ${r.status}">${r.status}</span>
        <span class="idlabel">sample ${r.sample_index} · run ${r.run_index}</span>
        <span class="muted">${r.latency_s}s</span>
        <span class="snippet">${esc(r.first_user_text)}</span>
      </div>
      <div class="card-body">
        <div class="field"><div class="k">Status</div><div>${esc(r.status)} — ${esc(STATUS_DESC[r.status]||"")}</div></div>
        ${reason? `<div class="field"><div class="k">Why malformed</div><div class="reason">${esc(reason)}</div></div>`:""}
        ${r.explanation!=null? `<div class="field"><div class="k">Model explanation</div><pre>${esc(r.explanation)}</pre></div>`:""}
        ${r.error!=null? `<div class="field"><div class="k">Error body</div><pre>${esc(typeof r.error==="string"?r.error:JSON.stringify(r.error,null,2))}</pre></div>`:""}
        ${r.generated_messages!=null? `<div class="field"><div class="k">Generated messages</div>${msgsHtml(r.generated_messages)}</div>`:""}
        <div class="field"><div class="k">Raw text</div><pre>${esc(r.raw_text!=null?r.raw_text:"(none)")}</pre></div>
      </div>`;
    el.appendChild(card);
  });
  if(!list.length) el.innerHTML='<p class="empty">No requests with this status.</p>';
}

function renderSamples(){
  const el=document.getElementById("samples"); el.innerHTML="";
  SAMPLES.forEach(s=>{
    const card=document.createElement("div"); card.className="card";
    card.innerHTML = `
      <div class="card-head" onclick="this.parentElement.classList.toggle('open')">
        <span class="idlabel">sample ${s.sample_index}</span>
        <span class="muted">${esc(s.current_label)} &rarr; ${esc(s.target_label)}</span>
        <span class="snippet">run_id ${esc(s.run_id)} · round ${esc(s.round)} · iter ${esc(s.iteration)}</span>
      </div>
      <div class="card-body">
        <div class="field"><div class="k">Original conversation (source)</div>${msgsHtml(s.messages)}</div>
        <div class="field"><div class="k">System prompt (exact, from repo)</div><pre>${esc(s.system_prompt)}</pre></div>
        <div class="field"><div class="k">User prompt (exact, from repo)</div><pre>${esc(s.user_prompt)}</pre></div>
      </div>`;
    el.appendChild(card);
  });
}

function showTab(t){
  document.getElementById("tab-requests").classList.toggle("hidden", t!=="requests");
  document.getElementById("tab-samples").classList.toggle("hidden", t!=="samples");
  document.querySelectorAll(".tabbar .btn").forEach(b=>b.classList.toggle("active", b.dataset.tab===t));
}

render(); renderSamples();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--rows", default="contrastive_rerun.jsonl")
    ap.add_argument("--samples", default="contrastive_rerun.samples.jsonl")
    ap.add_argument("--out", default="contrastive_rerun_viewer.html")
    args = ap.parse_args()

    rows = load_jsonl(args.dir / args.rows)
    samples = load_jsonl(args.dir / args.samples)
    if not rows:
        raise SystemExit(f"No rows found at {args.dir / args.rows}")

    html = (
        HTML_TEMPLATE
        .replace("__ROWS__", json.dumps(rows, ensure_ascii=False))
        .replace("__SAMPLES__", json.dumps(samples, ensure_ascii=False))
    )
    out_path = args.dir / args.out
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote viewer ({len(rows)} requests, {len(samples)} samples) -> {out_path}")


if __name__ == "__main__":
    main()
