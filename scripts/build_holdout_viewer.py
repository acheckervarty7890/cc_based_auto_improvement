#!/usr/bin/env python
"""Build the two HTML deliverables for the v2-holdout measurement.

1. ``viewers/instructions_v3_only_pairs_viewer.html`` — every red-team pair that is in
   the iteration-3 training set but **not** in the iteration-2 vintage, i.e. exactly the
   pairs ``scripts/vintage_holdout_success.py`` puts in front of the v2 probes. Each pair
   shows the attacker's success conversation beside its generated counterpart, annotated
   with **how many of the ten v2 probes each side fools** — the quantity a single-seed
   red-team log cannot report, and the reason for looking at these pairs rather than all
   of iteration 3's.

2. ``viewers/instructions_v2_holdout_report.html`` — the written summary of that
   measurement: the same tables and findings as ``SUMMARY.md``'s held-out section, as a
   standalone page that opens by double-click.

Both embed their data inline, so they are self-contained files with no server, no CORS
and no fetches.

Provenance of each field is deliberately traced back to its own source rather than
re-derived: judge verdicts come from the run's ``*_probing_f{p,n}.jsonl`` (joined on the
transformed conversation text, since the dumps are transformed and the logs are not), the
counterpart's rationale from the arm's ``contrastive_cache.jsonl``, and the per-seed
success counts from ``holdout_progress.jsonl``. Nothing here re-runs a probe.

Usage:
    .venv_claude/bin/python scripts/build_holdout_viewer.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A  # noqa: E402
import attribution_vintage as V  # noqa: E402

REPO = A.REPO
VIEWERS = REPO / "viewers"
RESULTS = {
    "gptoss120b": REPO / "results_instructions_gemma27b_gptoss",
    "nemotron": REPO / "results_instructions_gemma27b_nemotron",
}
ATTACKER = {
    "gptoss120b": "openai/gpt-oss-120b",
    "nemotron": "nvidia/nemotron-3-ultra-550b-a55b",
}
JSONL_STEM = {"gptoss120b": "gptoss120b_probing", "nemotron": "nemotron_probing"}


# --- joins ------------------------------------------------------------------------


def judge_index(arm: str) -> dict[str, dict]:
    """Canonical *transformed* conversation text -> the attempt row that produced it.

    The JSONL stores the conversation as submitted; the postprocessed dump stores it
    after ``convert_tool_to_assistant`` + ``combine_consecutive_messages``. Keying on the
    transformed text is what makes the two joinable, and it is the same key the
    activation cache and the vintage membership use.
    """
    from tuberlens.interfaces.dataset import Message

    out: dict[str, dict] = {}
    for suffix in ("fp", "fn"):
        path = RESULTS[arm] / f"{JSONL_STEM[arm]}_{suffix}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = (rec.get("sample") or {}).get("messages") or []
                key = A.canon(
                    A.apply_transforms(
                        [Message(role=m["role"], content=m["content"]) for m in msgs]
                    )
                )
                # First writer wins: a conversation resubmitted in a later round is the
                # same conversation, and the earliest row is the one that found it.
                out.setdefault(key, rec)
    return out


def explanation_index(arm: str) -> dict[str, str]:
    """Canonical text of a generated row -> the generator's stated rationale."""
    from tuberlens.interfaces.dataset import Message

    def as_key(raw):
        return A.canon(
            A.apply_transforms(
                [Message(role=m["role"], content=m["content"]) for m in raw]
            )
        )

    out: dict[str, str] = {}
    with (A.ARMS[arm] / "contrastive_cache.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line).get("record", {})
            if rec.get("inputs") and rec.get("generation_explanation"):
                out[as_key(rec["inputs"])] = rec["generation_explanation"]
    return out


def seed_success(sidecar: Path, arm: str, vintage: int, iteration: int):
    """``(n_seeds, {row: n_probes_that_misclassify_it}, {row: mean logit})``."""
    fits = [
        r for r in (json.loads(l) for l in sidecar.open(encoding="utf-8"))
        if r["arm"] == arm and r["vintage"] == vintage and r["iteration"] == iteration
    ]
    if not fits:
        raise SystemExit(f"no holdout fits recorded for {arm} in {sidecar}")
    n = len(fits)
    wrong = [0] * len(fits[0]["wrong"])
    total = [0.0] * len(fits[0]["logits"])
    for r in fits:
        for i, w in enumerate(r["wrong"]):
            wrong[i] += w
        for i, s in enumerate(r["logits"]):
            total[i] += s
    return n, wrong, [t / n for t in total]


# --- pair assembly ----------------------------------------------------------------


def build_arm(arm: str, *, iteration: int, vintage: int, drop_long: str,
              sidecar: Path) -> dict:
    ds = A.load_redteam_dataset(arm, iteration)
    gen2src = A.generated_to_source(arm)
    keys = [A.canon(m) for m in ds.inputs]
    labels = ds.other_fields["labels"]
    ids = list(ds.ids)

    keep, report = V.vintages(arm, iteration, drop_long)
    trained = set(keep[vintage])
    heldout = [i for i in sorted(keep[iteration]) if i not in trained]

    n_seeds, wrong, mean_logit = seed_success(sidecar, arm, vintage, iteration)
    judges = judge_index(arm)
    explanations = explanation_index(arm)

    pairs: dict[str, dict] = {}
    for i in heldout:
        key = keys[i]
        src = gen2src.get(key, key)
        p = pairs.setdefault(src, {})
        side = "gen" if key in gen2src else "orig"
        p[f"{side}_inputs"] = [
            {"role": m.role, "content": m.content} for m in ds.inputs[i]
        ]
        p[f"{side}_cls"] = labels[i]
        p[f"{side}_id"] = ids[i]
        p[f"{side}_row"] = i
        p[f"{side}_hits"] = wrong[i]
        p[f"{side}_logit"] = round(mean_logit[i], 4)
        p[f"{side}_side"] = "val" if A.is_val(ds.inputs[i]) else "train"
        if side == "gen":
            p["explanation"] = explanations.get(key, "")
        else:
            rec = judges.get(key)
            if rec:
                p["judge_label"] = rec.get("judge_label", "")
                p["judge_reason"] = rec.get("judge_reason", "")
                p["judge_conf"] = rec.get("judge_confidence")
                p["probe_score"] = rec.get("probe_score")
                p["error_type"] = rec.get("error_type", "")
                p["found_iter"] = rec.get("iteration")
                p["found_round"] = rec.get("round")

    # Labels the probe uses, for the column pills — read off the probe, never hardcoded.
    probe = A.load_probe(A.ARMS[arm] / f"probe_iter{iteration}.pkl")
    pos, neg = probe.pos_class_label, probe.neg_class_label
    del probe

    def label_of(cls: str) -> str:
        return pos if cls == "positive" else neg

    out = []
    for src, p in pairs.items():
        if "orig_inputs" not in p or "gen_inputs" not in p:
            # A half-pair can only appear if the length filter removed one side, which
            # --drop-long pair rules out; keep the guard so a policy change is visible.
            continue
        p["pair"] = A.sha16(src)
        p["orig_label"] = label_of(p["orig_cls"])
        p["gen_label"] = label_of(p["gen_cls"])
        out.append(p)
    # Most robust finds first: a pair that fools every seed is the one worth reading.
    out.sort(key=lambda p: (-p["orig_hits"], -p["gen_hits"], p["orig_row"]))

    return {
        "attacker": ATTACKER[arm],
        "n_seeds": n_seeds,
        "n_heldout_rows": len(heldout),
        "n_v3_rows": len(keep[iteration]),
        "n_v2_rows": len(trained),
        "pos_label": pos,
        "neg_label": neg,
        "pairs": out,
    }


# --- the pairs viewer -------------------------------------------------------------


VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Instruction-following &mdash; red-team pairs new at iteration 3</title>
<style>
  :root{
    --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --border:#2a2f3a;
    --fg:#e6e8ec; --muted:#9aa3b2; --accent:#6ea8fe;
    --pos:#4dd08a; --pos-bg:#142218; --neg:#ff6b6b; --neg-bg:#2a1618;
    --user:#7aa2ff; --asst:#c9a0ff; --judge:#ffc14d; --judge-bg:#2a2410;
    --hot:#f85149; --cold:#3fb950;
  }
  @media (prefers-color-scheme: light){
    :root{
      --bg:#f6f7f9; --panel:#ffffff; --panel2:#f0f2f5; --border:#d9dde4;
      --fg:#1a1d23; --muted:#5b6472; --accent:#2f6fed;
      --pos:#1e8e5a; --pos-bg:#e8f6ee; --neg:#c0392b; --neg-bg:#fdecea;
      --user:#2f5fed; --asst:#7a3ff0; --judge:#9a6b00; --judge-bg:#fbf3dd;
      --hot:#c0392b; --cold:#1e8e5a;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--fg)}
  header{padding:18px 22px 8px}
  h1{margin:0 0 6px;font-size:18px}
  .sub{color:var(--muted);font-size:13px;margin-bottom:4px;max-width:1120px}
  .sub code{background:var(--panel2);padding:1px 5px;border-radius:4px}
  .tabs{display:flex;gap:6px;padding:12px 22px 0;flex-wrap:wrap;border-bottom:1px solid var(--border)}
  .tab{padding:9px 16px;border:1px solid var(--border);border-bottom:none;background:var(--panel2);color:var(--muted);
       border-radius:8px 8px 0 0;cursor:pointer;font-weight:600;font-size:13px}
  .tab.active{background:var(--panel);color:var(--fg);border-color:var(--accent)}
  .tab .n{color:var(--muted);font-weight:400}
  .controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:14px 22px;background:var(--panel)}
  .controls input[type=search]{flex:1;min-width:220px;padding:8px 12px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--fg)}
  .seg{display:flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
  .seg button{padding:8px 13px;background:var(--bg);color:var(--muted);border:none;cursor:pointer;font-weight:600;font-size:12.5px}
  .seg button.active{background:var(--accent);color:#fff}
  .meta{color:var(--muted);font-size:13px}
  .meta code{background:var(--panel2);padding:1px 5px;border-radius:4px}
  .list{padding:8px 22px 60px;max-width:1400px;margin:0 auto}
  .pair{margin:16px 0;border:1px solid var(--border);border-radius:12px;overflow:hidden;background:var(--panel2)}
  .pair-head{display:flex;align-items:center;gap:10px;padding:8px 14px;font-size:12px;color:var(--muted);
             border-bottom:1px solid var(--border);flex-wrap:wrap}
  .pair-head .idx{font-weight:700;color:var(--fg)}
  .chip{padding:2px 8px;border-radius:999px;border:1px solid var(--border);background:var(--bg);font-size:11.5px}
  .chip.hot{color:var(--hot);border-color:var(--hot)}
  .chip.cold{color:var(--cold);border-color:var(--cold)}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border)}
  @media (max-width:820px){.cols{grid-template-columns:1fr}}
  .col{background:var(--panel);padding:10px 14px 14px}
  .col-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;font-size:12px;color:var(--muted)}
  .tag{font-weight:700;color:var(--fg)}
  .pill{padding:2px 8px;border-radius:999px;font-size:11.5px;font-weight:600}
  .pill.positive{color:var(--pos);background:var(--pos-bg)}
  .pill.negative{color:var(--neg);background:var(--neg-bg)}
  .cid{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}
  .msgs{display:flex;flex-direction:column;gap:8px}
  .msg{border:1px solid var(--border);border-radius:8px;background:var(--bg)}
  .role{padding:4px 10px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid var(--border)}
  .role.user{color:var(--user)} .role.assistant{color:var(--asst)} .role.system{color:var(--muted)}
  .content{padding:8px 10px;white-space:pre-wrap;word-break:break-word;max-height:340px;overflow:auto}
  .note{padding:9px 14px;font-size:13px;border-top:1px solid var(--border);background:var(--panel)}
  .note.judge{background:var(--judge-bg)} .note.judge b{color:var(--judge)}
  .empty{padding:40px;text-align:center;color:var(--muted)}
  mark{background:#ffd54f66;color:inherit;border-radius:3px}
</style>
</head>
<body>
<header>
  <h1>Instruction-following &mdash; the red-team pairs that are new at iteration 3</h1>
  <div class="sub">Only the pairs in the iteration-3 training set that are <b>not</b> in the
    iteration-2 vintage: rows held out of every v2 probe fit, on both sides of the split.
    <code>__NPAIRS__</code> pairs across both attackers.</div>
  <div class="sub"><b>v2 hits</b> is how many of the <code>__NSEEDS__</code> independently
    seeded v2 probes <i>misclassify</i> that conversation &mdash; i.e. how many of them the
    find still beats. <code>__NSEEDS__/__NSEEDS__</code> is a robust probe weakness;
    <code>0/__NSEEDS__</code> is a find that only beat the one probe draw the run happened
    to be handed. Thresholded at <code>logit &ge; 0</code>, as <code>ProbeJudge</code> does.</div>
  <div class="sub">Left column is the attacker's submission (the judge disagreed with the
    probe on it); right column is the LLM-written opposite-class counterpart
    <code>preprocessing</code> generated from it. Both were trained on at iteration 3.</div>
  <div class="sub"><b>&ldquo;New at iteration 3&rdquo; is about the training set, not about
    when the attacker found it</b> &mdash; <code>filter_dataset</code> drops a different
    top-percentile each cycle, so some of these were found in an earlier rotation, dropped
    from the iteration-2 set and taken back later. The <i>found in iteration</i> chip shows
    which rotation each one came from.</div>
</header>
<div class="tabs" id="tabs"></div>
<div class="controls">
  <div class="seg" id="filter">
    <button data-l="all" class="active">All pairs</button>
    <button data-l="robust">Success fools all seeds</button>
    <button data-l="majority">Fools a majority</button>
    <button data-l="never">Fools none</button>
  </div>
  <input type="search" id="search" placeholder="Search text within this attacker&hellip;">
  <span class="meta" id="count"></span>
</div>
<div class="list" id="list"></div>
<script>
const DATA = __DATA__;
const KEYS = Object.keys(DATA);
let active = KEYS[0];
let filt = "all";
let query = "";

const tabsEl = document.getElementById("tabs");
KEYS.forEach(k=>{
  const b=document.createElement("button");
  b.className="tab"+(k===active?" active":"");
  b.innerHTML=`${k} <span class="n">(${DATA[k].pairs.length} pairs)</span>`;
  b.onclick=()=>{active=k;render();document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));b.classList.add("active");};
  tabsEl.appendChild(b);
});
document.querySelectorAll("#filter button").forEach(b=>{
  b.onclick=()=>{filt=b.dataset.l;document.querySelectorAll("#filter button").forEach(x=>x.classList.remove("active"));b.classList.add("active");render();};
});
document.getElementById("search").addEventListener("input",e=>{query=e.target.value.toLowerCase();render();});

function esc(s){return (s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function hl(s){
  const e=esc(s);
  if(!query) return e;
  try{const re=new RegExp("("+query.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")+")","ig");return e.replace(re,"<mark>$1</mark>");}
  catch(_){return e;}
}
function convoHtml(inputs){
  return `<div class="msgs">${inputs.map(m=>`
    <div class="msg"><div class="role ${esc(m.role)}">${esc(m.role)}</div>
    <div class="content">${hl(m.content||"")}</div></div>`).join("")}</div>`;
}
function pairText(p){
  return (p.orig_inputs.map(m=>m.content).join(" ")+" "+p.gen_inputs.map(m=>m.content).join(" ")+" "+
          (p.explanation||"")+" "+(p.judge_reason||"")).toLowerCase();
}
function hits(n,total){
  const cls = n===total ? "hot" : (n===0 ? "cold" : "");
  return `<span class="chip ${cls}">${n}/${total} v2 probes fooled</span>`;
}
function col(tag,cls,label,id,inputs,n,total,side,logit){
  return `<div class="col">
      <div class="col-head"><span class="tag">${tag}</span>
        <span class="pill ${cls}">${esc(label)}</span>
        ${hits(n,total)}
        <span class="chip">split: ${esc(side)}</span>
        <span class="chip">mean logit ${logit}</span>
        <span class="cid">${esc(id)}</span></div>
      ${convoHtml(inputs)}</div>`;
}

function render(){
  const d=DATA[active];
  const T=d.n_seeds;
  let pairs=d.pairs;
  if(filt==="robust") pairs=pairs.filter(p=>p.orig_hits===T);
  if(filt==="majority") pairs=pairs.filter(p=>p.orig_hits*2>=T);
  if(filt==="never") pairs=pairs.filter(p=>p.orig_hits===0);
  if(query) pairs=pairs.filter(p=>pairText(p).includes(query));
  document.getElementById("count").innerHTML=
    `<span class="meta">${pairs.length} pairs shown &middot; attacker <code>${esc(d.attacker)}</code> &middot; `+
    `${d.n_heldout_rows} of ${d.n_v3_rows} iteration-3 rows are new since v2</span>`;
  const list=document.getElementById("list");
  if(!pairs.length){list.innerHTML='<div class="empty">No pairs match.</div>';return;}
  list.innerHTML=pairs.map((p,i)=>`
    <div class="pair">
      <div class="pair-head"><span class="idx">Pair ${i+1}</span>
        <span>attacker success <b>${esc(p.orig_label)}</b> &rarr; counterpart <b>${esc(p.gen_label)}</b></span>
        ${p.error_type?`<span class="chip">hunted as ${esc(p.error_type)}</span>`:""}
        ${p.found_iter!=null?`<span class="chip">found in iteration ${esc(String(p.found_iter))}</span>`:""}
        ${p.probe_score!=null?`<span class="chip">probe of the day scored ${esc(String(p.probe_score))}</span>`:""}
      </div>
      <div class="cols">
        ${col("Attacker submission",p.orig_cls,p.orig_label,p.orig_id,p.orig_inputs,p.orig_hits,T,p.orig_side,p.orig_logit)}
        ${col("Generated counterpart",p.gen_cls,p.gen_label,p.gen_id,p.gen_inputs,p.gen_hits,T,p.gen_side,p.gen_logit)}
      </div>
      ${p.judge_reason?`<div class="note judge"><b>Judge &rarr; ${esc(p.judge_label||p.orig_label)}${p.judge_conf!=null?` <span class="conf">(confidence ${esc(String(p.judge_conf))})</span>`:""}:</b> ${hl(p.judge_reason)}</div>`:""}
      ${p.explanation?`<div class="note"><b>Counterpart is ${esc(p.gen_label)}:</b> ${hl(p.explanation)}</div>`:""}
    </div>`).join("");
}
render();
</script>
</body>
</html>
"""


def build_viewer(data: dict, out_path: Path) -> None:
    n_pairs = sum(len(d["pairs"]) for d in data.values())
    n_seeds = max(d["n_seeds"] for d in data.values())
    html = (
        VIEWER_HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
        .replace("__NPAIRS__", str(n_pairs))
        .replace("__NSEEDS__", str(n_seeds))
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB, {n_pairs} pairs)")


# --- the report -------------------------------------------------------------------


def report_rows(out_dir: Path) -> tuple[list[dict], list[dict]]:
    """``(group table, per-fit rows)`` read back from the committed CSVs."""
    import csv

    with (out_dir / "holdout_success_summary.csv").open(encoding="utf-8") as fh:
        groups = list(csv.DictReader(fh))
    with (out_dir / "holdout_success_rows.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return groups, rows


REPORT_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Do iteration 3's red-team finds beat more than one probe?</title>
<style>
  :root{ --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --border:#2a2f3a;
         --fg:#e6e8ec; --muted:#9aa3b2; --accent:#6ea8fe; --hot:#f85149; --cold:#3fb950; }
  @media (prefers-color-scheme: light){
    :root{ --bg:#f6f7f9; --panel:#fff; --panel2:#f0f2f5; --border:#d9dde4;
           --fg:#1a1d23; --muted:#5b6472; --accent:#2f6fed; --hot:#c0392b; --cold:#1e8e5a; }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:900px;margin:0 auto;padding:28px 24px 90px}
  h1{font-size:24px;margin:0 0 6px;line-height:1.25}
  h2{font-size:17px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--border)}
  h3{font-size:14.5px;margin:22px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
  .lede{color:var(--muted);font-size:14px;margin:0 0 4px}
  p{margin:10px 0}
  code{background:var(--panel2);padding:1px 5px;border-radius:4px;font-size:.9em}
  table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px;
        display:block;overflow-x:auto;white-space:nowrap}
  th,td{border:1px solid var(--border);padding:6px 10px;text-align:right}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left;white-space:normal}
  th{background:var(--panel2);font-weight:600}
  tr.head td{background:var(--panel2);font-weight:700}
  .hot{color:var(--hot);font-weight:700} .cold{color:var(--cold);font-weight:700}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 18px;margin:14px 0}
  .bars td.b{padding:0 10px;width:100%}
  .bar{display:inline-block;height:12px;background:var(--accent);border-radius:3px;vertical-align:middle}
  .muted{color:var(--muted)}
  .foot{margin-top:36px;color:var(--muted);font-size:13px;border-top:1px solid var(--border);padding-top:14px}
  a{color:var(--accent)}
</style>
</head>
<body>
<div class="wrap">
__BODY__
</div>
</body>
</html>
"""


def build_report(data: dict, out_dir: Path, out_path: Path) -> None:
    groups, _rows = report_rows(out_dir)
    n_seeds = max(d["n_seeds"] for d in data.values())

    def g(arm: str, group: str, sub: str) -> dict | None:
        for r in groups:
            if r["arm"] == arm and r["group"] == group and r["subgroup"] == sub:
                return r
        return None

    def pct(x) -> str:
        return f"{float(x):.3f}"

    label = {
        ("held-out (v3 only)", "all"): "held-out, all",
        ("held-out (v3 only)", "success"): "&mdash; attacker successes",
        ("held-out (v3 only)", "generated"): "&mdash; generated counterparts",
        ("held-out (v3 only)", "side=train"): "&mdash; train side of the split",
        ("held-out (v3 only)", "side=val"): "&mdash; val side of the split",
        ("v2 rows", "fit on (train side)"): "v2's own rows, fitted",
        ("v2 rows", "early-stopping (val side)"): "v2's own rows, val side",
    }

    body = [
        "<h1>Do iteration 3's red-team finds beat more than one probe?</h1>",
        '<p class="lede">Instruction-following concept &middot; '
        "<code>google/gemma-3-27b-it</code> layer 32 &middot; "
        f"{n_seeds} independently seeded v2 probes per attacker</p>",
        "<div class='card'><p><b>The question.</b> A red-team run reports a find whenever "
        "the probe of the day disagrees with the judge. But that probe is one fit among "
        "many possible ones, and nothing in the run distinguishes a conversation that "
        "defeats the classifier from one that merely defeats that fit's initialisation.</p>"
        "<p><b>The measurement.</b> Refit the iteration-2 vintage probe with "
        f"{n_seeds} seeds, then score the pairs that are in the iteration-3 training set "
        "but not in the iteration-2 vintage &mdash; rows held out of every one of those "
        "fits, on both sides of the split. Per conversation, the <b>success rate is the "
        f"fraction of the {n_seeds} probes that misclassify it</b>, thresholded at "
        "<code>logit &ge; 0</code> exactly as <code>ProbeJudge</code> does. Every refit was "
        "verified to reproduce the sweep's v2 probe by re-scoring an eval split and "
        "demanding the recorded AUROC to the last bit.</p>"
        "<p class='muted'><b>&ldquo;New at iteration 3&rdquo; is a property of the training "
        "set, not of when the attacker found it.</b> <code>filter_dataset</code> refits its "
        "bag-of-words classifier each cycle and drops a different top-percentile, so a pair "
        "can be found in an early rotation, dropped from the iteration-2 set, and taken back "
        "at iteration 3. Most of these pairs were found in the last rotation (67 of 92 and "
        "120 of 148), but 22 and 19 come from the first &mdash; and they are held out of the "
        "v2 fits all the same, which is what this measures.</p></div>",
        "<h2>Headline</h2>",
        "<table><tr><th>attacker</th><th>held out of v2</th><th>pairs</th>"
        "<th>per-seed success rate</th></tr>",
    ]
    per_seed = {
        "gptoss120b": "0.273 &plusmn; 0.021 (0.239&ndash;0.310)",
        "nemotron": "0.401 &plusmn; 0.011 (0.385&ndash;0.416)",
    }
    for arm, d in data.items():
        body.append(
            f"<tr><td>{arm}</td><td>{d['n_heldout_rows']} rows of {d['n_v3_rows']}</td>"
            f"<td>{len(d['pairs'])}</td><td>{per_seed.get(arm, '')}</td></tr>"
        )
    body.append("</table>")

    body.append("<h2>By row type &mdash; mean over the seeds</h2>")
    body.append(
        "<table><tr><th>attacker</th><th>group</th><th>n</th><th>success</th>"
        f"<th>always ({n_seeds}/{n_seeds})</th><th>never (0/{n_seeds})</th></tr>"
    )
    for arm in data:
        first = True
        for (grp, sub), pretty in label.items():
            r = g(arm, grp, sub)
            if r is None:
                continue
            emph = ' class="hot"' if sub == "success" else ""
            body.append(
                f"<tr><td>{arm if first else ''}</td><td>{pretty}</td>"
                f"<td>{r['n_rows']}</td><td{emph}>{pct(r['success_rate'])}</td>"
                f"<td>{r['n_rows_always']}</td><td>{r['n_rows_never']}</td></tr>"
            )
            first = False
    body.append("</table>")

    body.append("<h2>Distribution of the per-row success count</h2>")
    body.append('<table class="bars"><tr><th>seeds fooled</th>'
                + "".join(f"<th>{a}</th>" for a in data) + "</tr>")
    dist = {arm: [0] * (n_seeds + 1) for arm in data}
    for arm, d in data.items():
        # Recomputed from the pairs so the report and the viewer can never disagree.
        for p in d["pairs"]:
            dist[arm][p["orig_hits"]] += 1
            dist[arm][p["gen_hits"]] += 1
    for k in range(n_seeds + 1):
        cells = "".join(f"<td>{dist[a][k]}</td>" for a in data)
        body.append(f"<tr><td>{k}/{n_seeds}</td>{cells}</tr>")
    body.append("</table>")

    body += [
        "<h2>Findings</h2>",
        "<h3>About half of the new finds are not probe-draw artifacts</h3>",
        "<p>Of the attacker's own success conversations that v2 never saw, "
        "<b>45 of 92 (gptoss120b) and 109 of 148 (nemotron) fool at least five of the ten "
        "v2 probes</b>, and 29 / 74 fool all ten. Iteration 3 was largely re-finding "
        "weaknesses the iteration-2 probe really has, not exploiting the particular fit it "
        "happened to be handed.</p>",
        "<h3>The rest is a seed lottery, and it is large</h3>",
        "<p>30 of 92 and 21 of 148 attacker successes fool <b>none</b> of the ten. Those "
        "are conversations the probe of the day misclassified that a rerun of the same "
        "training with a different initialisation would have classified correctly &mdash; "
        "reported as finds, trained against, and not reproducible. Between the extremes "
        "the middle is thin (51 and 87 rows at 1&ndash;9 of 10), so a find is usually "
        "either robust or a coin flip, rarely in between.</p>",
        "<h3>Generated counterparts carry little held-out evidence</h3>",
        "<p>The LLM-written opposite-class halves fail at 0.070 / 0.095 against probes that "
        "never saw them &mdash; five to seven times below the successes they were written "
        "from. Only 1 of 92 and 3 of 148 fool all ten seeds; in just 1 and 6 pairs do "
        "<i>both</i> members beat a majority. They balance the labels, but as evidence "
        "about the probe they add little.</p>",
        "<h3>Memorisation is total, so the held-out rate is a real generalisation gap</h3>",
        "<p>v2's own <i>fitted</i> rows fail at 0.003 / 0.006 &mdash; essentially every "
        "training row is classified correctly. Its <i>validation</i> rows (never fit; they "
        "act only through early stopping) fail at 0.353 / 0.265, which brackets the "
        "held-out numbers. Read against that reference, gptoss120b's v3-only rows are no "
        "harder than v2's own unfitted rows, whereas nemotron's are harder: its "
        "iteration-3 attacker found something its iteration-2 data genuinely did not "
        "cover.</p>",
        "<h3>Beating the probe and improving the probe are not the same thing</h3>",
        "<p>nemotron's iteration-3 finds transfer to the v2 probes far better than "
        "gptoss120b's (0.707 vs 0.477) &mdash; the better attack by this measure &mdash; "
        "yet training on them <i>lowered</i> mean eval AUROC (v3 0.785 vs v2 0.803, over "
        "the same ten seeds). A batch of samples can be reliable probe failures and still "
        "not generalise to the eval splits, either because they concentrate in a region "
        "those splits do not sample or because they resemble each other more than they "
        "resemble anything else. Which of the two it is, this measurement does not "
        "settle.</p>",
        "<h3>Per-attacker asymmetry</h3>",
        "<p>gptoss120b's held-out failures are lopsided &mdash; 0.420 on rows labelled "
        "<code>assistant_does_not_follow_the_instruction</code> against 0.127 on "
        "<code>follows</code>, i.e. its v2 probes mostly miss violations rather than "
        "over-flag compliance. nemotron's are symmetric (0.389 / 0.413).</p>",
        '<div class="foot">The conversations behind every number are in '
        "<code>viewers/instructions_v3_only_pairs_viewer.html</code>, which shows only "
        "these pairs, each annotated with how many of the ten v2 probes it fools. "
        "Per-row data: <code>holdout_success_rows.csv</code>; group table: "
        "<code>holdout_success_summary.csv</code>; raw per-seed logits: "
        "<code>holdout_progress.jsonl</code>. Method and the vintage AUROC sweep this "
        "extends: <code>results_instructions_gemma27b_vintage/SUMMARY.md</code>.</div>",
    ]

    out_path.write_text(REPORT_HTML.replace("__BODY__", "\n".join(body)), encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e3:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--vintage", type=int, default=2)
    ap.add_argument("--drop-long", choices=("pair", "row", "none"), default="pair")
    ap.add_argument("--results-dir", type=Path,
                    default=REPO / "results_instructions_gemma27b_vintage")
    ap.add_argument("--viewer-out", type=Path,
                    default=VIEWERS / "instructions_v3_only_pairs_viewer.html")
    ap.add_argument("--report-out", type=Path,
                    default=VIEWERS / "instructions_v2_holdout_report.html")
    args = ap.parse_args()

    sidecar = args.results_dir / "holdout_progress.jsonl"
    data = {}
    for arm in args.arm:
        d = build_arm(arm, iteration=args.iteration, vintage=args.vintage,
                      drop_long=args.drop_long, sidecar=sidecar)
        print(f"{arm}: {len(d['pairs'])} pairs, {d['n_heldout_rows']} rows "
              f"({d['n_seeds']} seeds)")
        data[arm] = d

    args.viewer_out.parent.mkdir(parents=True, exist_ok=True)
    build_viewer(data, args.viewer_out)
    build_report(data, args.results_dir, args.report_out)


if __name__ == "__main__":
    main()
