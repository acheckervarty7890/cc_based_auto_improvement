"""Build a self-contained HTML viewer: one tab per topic, conversations from every run.

Tab set = every topic holding at least ``--min-share`` of the successes in at
least one (run, iteration) cell, with fp/fn pooled as the denominator. Each tab
lists every success in that topic across all runs, tagged with the run it came
from, its iteration and error type, the probe/judge disagreement, and whether it
actually reached a retrain. The generated contrastive pair sits behind a toggle.

Usage mirrors analyze_redteam_successes.py:

    python scripts/build_topic_viewer.py \
        --arm NAME:RESULTS_DIR:PROBE_DIR [--arm ...] \
        --topic-names scripts/redteam_topic_names.json \
        --out viewers/redteam_topics_viewer.html
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import analyze_redteam_successes as core

RUN_COLORS = ["#3f6fd8", "#c2410c", "#0f766e", "#7c3aed", "#a16207", "#be123c"]


def select_topics(arms, labels_by_arm, min_share: float) -> dict[int, list[str]]:
    """Topics clearing ``min_share`` in some (run, iteration) cell → why they qualified."""
    qualifying: dict[int, list[str]] = defaultdict(list)
    for arm in arms:
        by_iter: dict[int, list[int]] = defaultdict(list)
        for s, topic in zip(arm.successes, labels_by_arm[arm.name]):
            by_iter[s.iteration].append(topic)
        for it, topics in sorted(by_iter.items()):
            n = len(topics)
            for topic, cnt in Counter(topics).items():
                if n and cnt / n >= min_share:
                    qualifying[topic].append(f"{arm.name} · iter {it} — {100 * cnt / n:.0f}%")
    return qualifying


def build_payload(
    arms,
    topic_names: dict[int, str],
    k: int,
    seed: int,
    min_share: float,
    include_never_trained: bool,
) -> dict:
    all_success = [s for a in arms for s in a.successes]
    labels, top_terms, _, _ = core.fit_topic_model(
        [core.topic_text(s.messages) for s in all_success], k, seed
    )
    labels = [int(x) for x in labels]

    labels_by_arm: dict[str, list[int]] = {}
    off = 0
    for arm in arms:
        labels_by_arm[arm.name] = labels[off : off + len(arm.successes)]
        off += len(arm.successes)

    qualifying = select_topics(arms, labels_by_arm, min_share)

    run_ids = [a.name for a in arms]
    totals = Counter(labels)

    convs = []
    n_dropped = 0
    dropped_by_topic: Counter = Counter()
    for arm_idx, arm in enumerate(arms):
        for s, topic in zip(arm.successes, labels_by_arm[arm.name]):
            if topic not in qualifying:
                continue
            status = 2 if s.in_final_training else (1 if s.in_any_training else 0)
            if status == 0 and not include_never_trained:
                n_dropped += 1
                dropped_by_topic[topic] += 1
                continue
            convs.append(
                {
                    "t": topic,
                    "r": arm_idx,
                    "it": s.iteration,
                    "et": "fp" if s.error_type == "false_positive" else "fn",
                    "ps": round(s.probe_score, 4),
                    "pp": 1 if s.probe_predicts_positive else 0,
                    "jl": s.judge_label,
                    "jr": s.judge_reason,
                    "st": status,
                    "m": [[m["role"], m["content"]] for m in s.messages],
                    "c": [[m.get("role", ""), m.get("content", "")] for m in s.contrastive]
                    if s.contrastive
                    else None,
                    "ce": s.contrastive_explanation,
                    "cm": s.contrastive_model,
                }
            )

    topics = []
    wiped = []  # qualified on all successes, but nothing survived the never-trained cut
    for topic in sorted(qualifying, key=lambda c: -totals[c]):
        by_run = Counter(c["r"] for c in convs if c["t"] == topic)
        n = sum(by_run.values())
        if not n:
            wiped.append(
                {"name": topic_names.get(topic, f"cluster {topic}"), "dropped": dropped_by_topic[topic]}
            )
            continue
        topics.append(
            {
                "id": topic,
                "name": topic_names.get(topic, f"cluster {topic}"),
                "terms": top_terms[topic][:8],
                "n": n,
                "byRun": [by_run.get(i, 0) for i in range(len(arms))],
                "why": qualifying[topic],
                "dropped": dropped_by_topic[topic],
            }
        )

    pos = next((s.pos_class_label for s in all_success if s.pos_class_label), "positive")
    neg = next((s.neg_class_label for s in all_success if s.neg_class_label), "negative")

    return {
        "runs": [
            {"id": rid, "color": RUN_COLORS[i % len(RUN_COLORS)], "n": len(arms[i].successes)}
            for i, rid in enumerate(run_ids)
        ],
        "topics": topics,
        "convs": convs,
        "meta": {
            "k": k,
            "seed": seed,
            "minShare": min_share,
            "pos": pos,
            "neg": neg,
            "nSuccessTotal": len(all_success),
            "nShown": len(convs),
            "nTopicsTotal": k,
            "nDropped": n_dropped,
            "wiped": wiped,
            "includeNeverTrained": include_never_trained,
        },
    }


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Red-team successes by topic</title>
<style>
:root {
  color-scheme: light dark;
  --bg:#fff; --panel:#f6f7f9; --panel2:#eef0f3; --line:#d9dde3;
  --fg:#1a1d21; --muted:#666e79; --accent:#3f6fd8;
  --user-bg:#eef2fb; --asst-bg:#f4f5f7;
  --pos-bg:#fdeeee; --pos-fg:#9d2727; --neg-bg:#eaf5ee; --neg-fg:#1e6b3c;
  --warn-bg:#fdf3dc; --warn-fg:#8a5b00;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
    --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
    --user-bg:#1b2433; --asst-bg:#1f242a;
    --pos-bg:#2c1c1e; --pos-fg:#ff9a9a; --neg-bg:#17281d; --neg-fg:#86e0a5;
    --warn-bg:#33290f; --warn-fg:#f0c060;
  }
}
:root[data-theme="light"] {
  --bg:#fff; --panel:#f6f7f9; --panel2:#eef0f3; --line:#d9dde3;
  --fg:#1a1d21; --muted:#666e79; --accent:#3f6fd8;
  --user-bg:#eef2fb; --asst-bg:#f4f5f7;
  --pos-bg:#fdeeee; --pos-fg:#9d2727; --neg-bg:#eaf5ee; --neg-fg:#1e6b3c;
  --warn-bg:#fdf3dc; --warn-fg:#8a5b00;
}
:root[data-theme="dark"] {
  --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
  --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
  --user-bg:#1b2433; --asst-bg:#1f242a;
  --pos-bg:#2c1c1e; --pos-fg:#ff9a9a; --neg-bg:#17281d; --neg-fg:#86e0a5;
  --warn-bg:#33290f; --warn-fg:#f0c060;
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
header { padding:20px 22px 12px; border-bottom:1px solid var(--line); }
h1 { margin:0 0 6px; font-size:20px; letter-spacing:-0.01em; }
.sub { color:var(--muted); font-size:13px; max-width:80ch; }
.wrap { padding:0 22px 60px; }
.bar {
  display:flex; gap:6px; overflow-x:auto; padding:12px 0;
  position:sticky; top:0; background:var(--bg); z-index:5;
  border-bottom:1px solid var(--line); margin-bottom:14px;
}
.tab {
  flex:0 0 auto; padding:7px 12px; border:1px solid var(--line); border-radius:999px;
  background:var(--panel); color:var(--fg); cursor:pointer; font-size:13px; white-space:nowrap;
}
.tab:hover { border-color:var(--accent); }
.tab.on { background:var(--accent); border-color:var(--accent); color:#fff; }
.tab .c { opacity:.7; margin-left:6px; font-variant-numeric:tabular-nums; }
.controls {
  display:flex; flex-wrap:wrap; gap:10px; align-items:center;
  padding:12px; background:var(--panel); border:1px solid var(--line);
  border-radius:10px; margin-bottom:14px;
}
.controls label { font-size:13px; color:var(--muted); display:flex; align-items:center; gap:5px; }
select, input[type=search] {
  font:inherit; font-size:13px; padding:5px 8px; border-radius:7px;
  border:1px solid var(--line); background:var(--bg); color:var(--fg);
}
input[type=search] { min-width:230px; }
.chip {
  display:inline-flex; align-items:center; gap:5px; padding:4px 10px; border-radius:999px;
  border:1px solid var(--line); font-size:12px; cursor:pointer; user-select:none; background:var(--bg);
}
.chip.off { opacity:.38; }
.chip .dot { width:9px; height:9px; border-radius:50%; }
.topichead { margin-bottom:14px; }
.topichead h2 { margin:0 0 4px; font-size:17px; }
.terms { color:var(--muted); font-size:12.5px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.why { margin-top:8px; display:flex; flex-wrap:wrap; gap:6px; }
.why span {
  font-size:11.5px; padding:3px 8px; border-radius:6px;
  background:var(--warn-bg); color:var(--warn-fg);
}
.card { border:1px solid var(--line); border-radius:10px; margin-bottom:12px; overflow:hidden; background:var(--panel); }
.chead {
  display:flex; flex-wrap:wrap; gap:7px; align-items:center;
  padding:9px 12px; background:var(--panel2); border-bottom:1px solid var(--line);
}
.run { font-weight:600; font-size:12.5px; padding:3px 9px; border-radius:6px; color:#fff; }
.tag {
  font-size:11.5px; padding:3px 8px; border-radius:6px;
  background:var(--bg); border:1px solid var(--line); color:var(--muted);
  font-variant-numeric:tabular-nums;
}
.tag.pos { background:var(--pos-bg); color:var(--pos-fg); border-color:transparent; }
.tag.neg { background:var(--neg-bg); color:var(--neg-fg); border-color:transparent; }
.tag.warn { background:var(--warn-bg); color:var(--warn-fg); border-color:transparent; }
.msgs { padding:10px 12px; }
.msg { padding:9px 11px; border-radius:8px; margin-bottom:7px; white-space:pre-wrap; word-break:break-word; font-size:14px; }
.msg.user { background:var(--user-bg); }
.msg.assistant { background:var(--asst-bg); }
.msg .role { display:block; font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-bottom:4px; }
.pairbtn {
  margin:0 12px 12px; padding:6px 11px; font:inherit; font-size:12.5px; cursor:pointer;
  background:var(--bg); color:var(--accent); border:1px solid var(--line); border-radius:7px;
}
.pairbtn:hover { border-color:var(--accent); }
.pair { margin:0 12px 12px; padding:10px 12px; border:1px dashed var(--line); border-radius:8px; }
.pair h4 { margin:0 0 8px; font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.reason { margin:0 12px 10px; padding:9px 11px; border-radius:8px; background:var(--bg); border:1px solid var(--line); font-size:13.5px; }
.reason .who { display:block; font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-bottom:4px; }
.pair .reason { margin:0 0 10px; }
.note {
  margin:0 0 14px; padding:11px 13px; border-radius:9px;
  background:var(--warn-bg); color:var(--warn-fg); font-size:13px; border:1px solid transparent;
}
.note b { font-weight:650; }
.none { color:var(--muted); font-size:13px; padding:12px 0; }
.more {
  display:block; width:100%; padding:10px; margin-top:6px; font:inherit; font-size:13px;
  cursor:pointer; background:var(--panel); color:var(--accent);
  border:1px solid var(--line); border-radius:9px;
}
.count { color:var(--muted); font-size:13px; margin-bottom:10px; }
#theme { position:absolute; top:18px; right:22px; }
</style>
</head>
<body>
<header>
  <button class="chip" id="theme">theme</button>
  <h1>Red-team successes by topic</h1>
  <div class="sub" id="sub"></div>
</header>
<div class="wrap">
  <div class="bar" id="tabs"></div>
  <div id="note"></div>
  <div class="controls" id="controls"></div>
  <div id="topichead" class="topichead"></div>
  <div class="count" id="count"></div>
  <div id="list"></div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const PAGE = 25;
const STATUS = [
  {label:'filtered out — never trained', cls:'warn'},
  {label:'trained earlier, cut from final', cls:''},
  {label:'in final retrain', cls:''},
];
let active = D.topics[0].id;
let shown = PAGE;
const runOn = D.runs.map(() => true);
let fIter = 'all', fErr = 'all', fStatus = 'all', q = '';

const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

document.getElementById('sub').textContent =
  `${D.meta.nShown.toLocaleString()} successes that reached training, across the ` +
  `${D.topics.length} topics that reached ${Math.round(D.meta.minShare*100)}% of at least one run's iteration ` +
  `(false-positive and false-negative pooled). Topics come from one TF-IDF + KMeans model (k=${D.meta.k}, ` +
  `seed ${D.meta.seed}) fit across every run, so a topic means the same thing everywhere. ` +
  `Positive class: ${D.meta.pos}; negative: ${D.meta.neg}.`;

if (!D.meta.includeNeverTrained && D.meta.nDropped) {
  const wiped = D.meta.wiped || [];
  document.getElementById('note').innerHTML =
    `<div class="note"><b>${D.meta.nDropped.toLocaleString()} successes hidden</b> — they were dropped by ` +
    `<code>filter_dataset</code> before contrastive generation and so never entered any retrain. ` +
    (wiped.length
      ? `That empties ${wiped.length} otherwise-qualifying topic${wiped.length===1?'':'s'} completely, ` +
        `so ${wiped.length===1?'it has':'they have'} no tab: ` +
        wiped.map(w => `<b>${esc(w.name)}</b> (${w.dropped})`).join(', ') + '.'
      : '') +
    `</div>`;
}

function topic(id) { return D.topics.find(t => t.id === id); }

function renderTabs() {
  document.getElementById('tabs').innerHTML = D.topics.map(t =>
    `<button class="tab ${t.id===active?'on':''}" data-id="${t.id}">${esc(t.name)}<span class="c">${t.n}</span></button>`
  ).join('');
}

function renderControls() {
  const iters = [...new Set(D.convs.map(c => c.it))].sort((a,b) => a-b);
  document.getElementById('controls').innerHTML =
    D.runs.map((r,i) =>
      `<span class="chip run-chip ${runOn[i]?'':'off'}" data-i="${i}">
         <span class="dot" style="background:${r.color}"></span>${esc(r.id)}</span>`).join('') +
    `<label>iteration <select id="fIter"><option value="all">all</option>` +
      iters.map(i => `<option value="${i}" ${fIter==String(i)?'selected':''}>${i}</option>`).join('') +
    `</select></label>` +
    `<label>error type <select id="fErr">
       <option value="all">all</option><option value="fp">false positive</option>
       <option value="fn">false negative</option></select></label>` +
    `<label>training <select id="fStatus">
       <option value="all">all</option><option value="2">in final retrain</option>
       <option value="1">trained earlier, cut from final</option>` +
       (D.meta.includeNeverTrained ? `<option value="0">filtered out — never trained</option>` : '') +
    `</select></label>` +
    `<input type="search" id="q" placeholder="search conversation text…" value="${esc(q)}">`;

  document.querySelectorAll('.run-chip').forEach(el => el.onclick = () => {
    const i = +el.dataset.i; runOn[i] = !runOn[i]; shown = PAGE; render();
  });
  const bind = (id, fn) => { const el = document.getElementById(id); el.onchange = e => { fn(e.target.value); shown = PAGE; render(); }; };
  bind('fIter', v => fIter = v); bind('fErr', v => fErr = v); bind('fStatus', v => fStatus = v);
  const qi = document.getElementById('q');
  qi.oninput = e => { q = e.target.value.toLowerCase(); shown = PAGE; render(); };
  document.getElementById('fErr').value = fErr;
  document.getElementById('fStatus').value = fStatus;
}

function matches(c) {
  if (c.t !== active) return false;
  if (!runOn[c.r]) return false;
  if (fIter !== 'all' && String(c.it) !== fIter) return false;
  if (fErr !== 'all' && c.et !== fErr) return false;
  if (fStatus !== 'all' && String(c.st) !== fStatus) return false;
  if (q && !c.m.some(m => m[1].toLowerCase().includes(q))) return false;
  return true;
}

function msgHtml(msgs) {
  return msgs.map(m =>
    `<div class="msg ${m[0]==='user'?'user':'assistant'}"><span class="role">${esc(m[0])}</span>${esc(m[1])}</div>`
  ).join('');
}

function card(c, idx) {
  const run = D.runs[c.r];
  const st = STATUS[c.st];
  const probeCls = c.pp ? 'pos' : 'neg';
  const probeLbl = c.pp ? D.meta.pos : D.meta.neg;
  const judgeCls = c.jl === D.meta.pos ? 'pos' : 'neg';
  return `<div class="card">
    <div class="chead">
      <span class="run" style="background:${run.color}">${esc(run.id)}</span>
      <span class="tag">iteration ${c.it}</span>
      <span class="tag">${c.et === 'fp' ? 'false positive' : 'false negative'}</span>
      <span class="tag ${probeCls}">probe ${c.ps.toFixed(3)} → ${esc(probeLbl)}</span>
      <span class="tag ${judgeCls}">judge → ${esc(c.jl || 'unparsed')}</span>
      <span class="tag ${st.cls}">${st.label}</span>
    </div>
    <div class="msgs">${msgHtml(c.m)}</div>
    ${c.jr ? `<div class="reason"><span class="who">judge reasoning — why it chose ${esc(c.jl || 'its label')}</span>${esc(c.jr)}</div>` : ''}
    ${c.c ? `<button class="pairbtn" data-i="${idx}">show generated contrastive pair</button>
             <div class="pair" id="pair-${idx}" hidden>
               <h4>generated contrastive pair (opposite class)</h4>
               ${c.ce ? `<div class="reason"><span class="who">contrastive generator reasoning${c.cm ? ' — ' + esc(c.cm) : ''}</span>${esc(c.ce)}</div>` : ''}
               ${msgHtml(c.c)}</div>` : ''}
  </div>`;
}

function render() {
  renderTabs();
  const t = topic(active);
  document.getElementById('topichead').innerHTML =
    `<h2>${esc(t.name)}</h2>
     <div class="terms">${t.terms.map(esc).join(' · ')}</div>
     <div class="why">${t.why.map(w => `<span>${esc(w)}</span>`).join('')}</div>`;

  const hits = D.convs.filter(matches);
  const perRun = D.runs.map((r,i) => `${r.id}: ${hits.filter(h => h.r===i).length}`).join('  ·  ');
  document.getElementById('count').textContent =
    `${hits.length} conversation${hits.length===1?'':'s'}  —  ${perRun}`;

  const list = document.getElementById('list');
  if (!hits.length) { list.innerHTML = '<div class="none">Nothing matches these filters.</div>'; return; }
  const page = hits.slice(0, shown);
  list.innerHTML = page.map((c,i) => card(c, i)).join('') +
    (hits.length > shown ? `<button class="more" id="more">show ${Math.min(PAGE, hits.length-shown)} more (${hits.length-shown} remaining)</button>` : '');

  list.querySelectorAll('.pairbtn').forEach(btn => btn.onclick = () => {
    const el = document.getElementById('pair-' + btn.dataset.i);
    el.hidden = !el.hidden;
    btn.textContent = el.hidden ? 'show generated contrastive pair' : 'hide generated contrastive pair';
  });
  const more = document.getElementById('more');
  if (more) more.onclick = () => { shown += PAGE; render(); };
}

document.getElementById('tabs').onclick = e => {
  const b = e.target.closest('.tab');
  if (!b) return;
  active = +b.dataset.id; shown = PAGE;
  render(); window.scrollTo({top:0});
};
document.getElementById('theme').onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : cur === 'light' ? '' : 'dark';
  if (next) document.documentElement.setAttribute('data-theme', next);
  else document.documentElement.removeAttribute('data-theme');
};

renderControls();
render();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--arm", action="append", default=[], metavar="NAME:RESULTS_DIR:PROBE_DIR")
    ap.add_argument("--clusters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-share", type=float, default=0.10)
    ap.add_argument(
        "--include-never-trained",
        action="store_true",
        help="also show successes filter_dataset dropped before any retrain (hidden by default)",
    )
    ap.add_argument("--topic-names", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    specs = [tuple(a.split(":", 2)) for a in args.arm] or core.DEFAULT_ARMS
    arms = []
    for name, res, probe in specs:
        res_p = Path(res) if Path(res).is_absolute() else args.root / res
        probe_p = Path(probe) if Path(probe).is_absolute() else args.root / probe
        if not res_p.exists():
            print(f"skip {name}: {res_p} missing")
            continue
        arms.append(core.load_arm(name, res_p, probe_p))
        print(f"{name}: {len(arms[-1].successes)} successes")

    topic_names: dict[int, str] = {}
    if args.topic_names and args.topic_names.exists():
        topic_names = {int(k): v for k, v in json.loads(args.topic_names.read_text()).items()}

    payload = build_payload(
        arms, topic_names, args.clusters, args.seed, args.min_share, args.include_never_trained
    )
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    args.out.write_text(HTML.replace("__DATA__", blob), encoding="utf-8")
    mb = args.out.stat().st_size / 1e6
    print(
        f"wrote {args.out} ({mb:.1f} MB) — {len(payload['topics'])} topics, "
        f"{len(payload['convs'])} conversations"
    )


if __name__ == "__main__":
    main()
