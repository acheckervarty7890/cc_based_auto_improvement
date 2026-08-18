#!/usr/bin/env python
"""Build a self-contained HTML viewer for the hu_harm / llama-1b AUROC results.

Two tabs, deliberately separated because they are NOT cell-comparable (see the
banner the page renders):

  Cross-experiment — the 9 arms of experiment{1,3,4,5,10}_cloud. AUROC per
      iteration per eval split, plus red-team attempts/successes per iteration
      and the postprocessed row count that actually trained each probe.
  exp10 ablation — the 5 preprocessing/confidence variants of experiment10_cloud's
      two batch arms, all scored on one identical local eval-activation blob set,
      with the full sample pipeline (raw successes → confidence gate →
      postprocessed rows → total training samples).

Data comes from scripts/collect_hu_harm_auroc_data.py; run that first (or let
this script invoke it via --collect).

    .venv_claude/bin/python scripts/build_hu_harm_auroc_viewer.py --collect
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Reference categorical palette, slots 1-5 in documented order. palette.md
# publishes this order as passing every hard gate on the *adjacent* pairlist in
# both modes (worst adjacent CVD dE 9.1 light / 8.4 dark; normal-vision 19.6 /
# 19.3), which is the pairlist line charts use. Three light slots sit under 3:1
# on the light surface, so the relief rule applies — hence direct labels on every
# line AND a full table view below each chart.
PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]

TEMPLATE = """<meta charset="utf-8">
<title>harmful_to_human / llama-1b — AUROC and training samples</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f7f7f5;
  --surface-1: #fcfcfb;
  --surface-2: #f0efec;
  --border:    #dcdbd6;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #7a7975;
  --grid: #e6e5e1;
  --good: #0f7a4a;
  --bad:  #b3261e;
  --series-1: __S1__; --series-2: __S2__; --series-3: __S3__;
  --series-4: __S4__; --series-5: __S5__;
  --warn-bg: #fdf6e3; --warn-br: #e0c86a;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --surface-2: #242422;
    --border: #3a3a37;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --grid: #2e2e2b;
    --good: #4ec98a; --bad: #f08a84;
    --series-1: __D1__; --series-2: __D2__; --series-3: __D3__;
    --series-4: __D4__; --series-5: __D5__;
    --warn-bg: #2b2718; --warn-br: #6b5c22;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --surface-2: #242422;
  --border: #3a3a37;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
  --grid: #2e2e2b;
  --good: #4ec98a; --bad: #f08a84;
  --series-1: __D1__; --series-2: __D2__; --series-3: __D3__;
  --series-4: __D4__; --series-5: __D5__;
  --warn-bg: #2b2718; --warn-br: #6b5c22;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px 20px 64px;
  background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 21px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 16px; margin: 32px 0 10px; letter-spacing: -0.01em; }
h3 { font-size: 13px; margin: 0 0 8px; color: var(--text-secondary); font-weight: 600; }
.sub { color: var(--text-secondary); margin: 0 0 18px; }
.banner {
  background: var(--warn-bg); border: 1px solid var(--warn-br);
  border-radius: 8px; padding: 10px 13px; margin: 0 0 20px;
  color: var(--text-primary); font-size: 13px;
}
.banner b { font-weight: 650; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 18px; }
.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border); margin: 0 0 20px; }
.tab {
  appearance: none; background: none; border: 0; border-bottom: 2px solid transparent;
  padding: 9px 14px; font: inherit; font-weight: 550; color: var(--text-secondary);
  cursor: pointer;
}
.tab[aria-selected="true"] { color: var(--text-primary); border-bottom-color: var(--series-1); }
select {
  font: inherit; padding: 5px 9px; border-radius: 6px;
  border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary);
}
label.ctl { color: var(--text-secondary); font-size: 13px; }
.facets { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); }
.facet {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 12px 6px;
}
.legend { display: flex; flex-wrap: wrap; gap: 10px 16px; margin: 2px 0 10px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-secondary); }
.swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; }
.tablewrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface-1); }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 13px; }
th, td { padding: 6px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--grid); }
th { background: var(--surface-2); color: var(--text-secondary); font-weight: 600; position: sticky; top: 0; }
th:first-child, td:first-child { text-align: left; }
th.grp { text-align: center; border-left: 1px solid var(--border); }
td.grpstart, th.grpstart { border-left: 1px solid var(--border); }
tbody tr:hover td { background: var(--surface-2); }
tr.arm-head td { background: var(--surface-2); font-weight: 650; color: var(--text-primary); }
.best { font-weight: 700; }
.pos { color: var(--good); } .neg { color: var(--bad); }
.muted { color: var(--text-muted); }
.note { color: var(--text-secondary); font-size: 12.5px; margin: 8px 0 0; }
.tip {
  position: fixed; pointer-events: none; opacity: 0; transition: opacity .08s;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.16);
  z-index: 40; min-width: 150px;
}
.tip .r { display: flex; justify-content: space-between; gap: 14px; }
.tip .r b { font-weight: 650; font-variant-numeric: tabular-nums; }
.tip .hd { color: var(--text-secondary); margin-bottom: 5px; font-weight: 600; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
.hidden { display: none; }
</style>

<div class="wrap">
  <h1>harmful_to_human / llama-1b — AUROC and training samples</h1>
  <p class="sub">Every probe here is <code>meta-llama/Llama-3.2-1B-Instruct</code> layer 8, trained from
  <code>data/hu_harm_llama70b_50.jsonl</code> (50 samples) and scored on the four
  <code>eval_dataset_hu_ha</code> splits at full size.</p>

  <div class="banner">
    <b>The two tabs are not cell-comparable.</b> Cross-experiment AUROC comes from each branch's
    committed CSV, scored on whatever eval activations that run's machine computed. The exp10
    ablation was re-scored locally on one identical blob set — on which the unchanged
    <code>probe_iter0</code> reads 0.6175, not the committed 0.6209. Compare <em>within</em> a tab.
  </div>

  <div class="tabs" role="tablist">
    <button class="tab" role="tab" data-tab="cross" aria-selected="true">Cross-experiment · 9 arms</button>
    <button class="tab" role="tab" data-tab="abl" aria-selected="false">exp10 preprocessing ablation · 5 variants</button>
  </div>

  <div class="controls">
    <label class="ctl" for="split">Eval split</label>
    <select id="split"></select>
    <label class="ctl" for="metric">Metric</label>
    <select id="metric">
      <option value="auroc">AUROC</option>
      <option value="accuracy">Accuracy</option>
      <option value="tpr_at_fpr">TPR @ 1% FPR</option>
    </select>
  </div>

  <section id="panel-cross">
    <h2>AUROC by iteration</h2>
    <p class="note">One facet per experiment; <code>iter0</code> is the initial probe, <code>iterN</code>
    the probe retrained after red-team rotation <code>N&minus;1</code>. Colors are per-facet only.</p>
    <div class="facets" id="cross-facets"></div>

    <h2>Scores — every arm, every iteration</h2>
    <div class="tablewrap"><table id="cross-scores"></table></div>

    <h2>Samples per iteration</h2>
    <p class="note">Attempts and successes are red-team rotation <code>N</code> (which feeds the probe
    labelled <code>iter N+1</code>). "Trained on" is the postprocessed red-team row count
    (<code>filter_dataset</code> + contrastive pairs) that actually reached that retrain, plus 50 base samples.</p>
    <div class="tablewrap"><table id="cross-samples"></table></div>

    <h2>Parameters</h2>
    <div class="tablewrap"><table id="cross-params"></table></div>
  </section>

  <section id="panel-abl" class="hidden">
    <h2>AUROC by iteration — 5 preprocessing variants</h2>
    <p class="note">All ten curves scored on one identical local eval-activation blob set, so
    <code>iter0</code> is the same number everywhere by construction.</p>
    <div class="facets" id="abl-facets"></div>

    <h2>Scores — every variant, every iteration</h2>
    <div class="tablewrap"><table id="abl-scores"></table></div>

    <h2>Sample pipeline per iteration</h2>
    <p class="note">Successes accumulate across iterations (<code>iterN</code> trains on successes from
    rotations <code>0..N-1</code>). <b>raw</b> = red-team successes; <b>conf</b> = surviving the judge-confidence
    gate; <b>trained</b> = postprocessed rows after <code>filter_dataset</code> and, where enabled, contrastive
    pairing; <b>total</b> = trained + 50 base samples.</p>
    <div class="tablewrap"><table id="abl-samples"></table></div>
  </section>
</div>

<div class="tip" id="tip"></div>

<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const SPLITS = DATA.splits;
const SERIES_VARS = ['--series-1','--series-2','--series-3','--series-4','--series-5'];
const ITERS = ['iter0','iter1','iter2','iter3'];
const tip = document.getElementById('tip');
const fmt = v => (v === null || v === undefined) ? '—' : v.toFixed(4);
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

let split = 'mean', metric = 'auroc', tab = 'cross';

/* ---------- chart ---------- */
function lineChart(series, opts) {
  const W = 100, H = 62, ml = 13, mr = 15, mt = 5, mb = 9;
  const vals = series.flatMap(s => s.points.filter(p => p.y != null).map(p => p.y));
  if (!vals.length) return '<p class="note">no data</p>';
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = Math.max((hi - lo) * 0.18, 0.02);
  lo = Math.max(0, lo - pad); hi = Math.min(1, hi + pad);
  const x = i => ml + (W - ml - mr) * (i / (ITERS.length - 1));
  const y = v => mt + (H - mt - mb) * (1 - (v - lo) / (hi - lo));
  const ticks = [lo, (lo + hi) / 2, hi];

  let g = '';
  for (const t of ticks) {
    g += `<line x1="${ml}" x2="${W - mr}" y1="${y(t).toFixed(2)}" y2="${y(t).toFixed(2)}"
           stroke="var(--grid)" stroke-width="0.4"/>
          <text x="${ml - 2}" y="${(y(t) + 1.1).toFixed(2)}" font-size="3.1" text-anchor="end"
           fill="var(--text-muted)">${t.toFixed(2)}</text>`;
  }
  ITERS.forEach((it, i) => {
    g += `<text x="${x(i).toFixed(2)}" y="${H - 3}" font-size="3.2" text-anchor="middle"
           fill="var(--text-muted)">${i}</text>`;
  });

  let paths = '';
  series.forEach(s => {
    const pts = s.points.map((p, i) => p.y == null ? null : [x(i), y(p.y)]).filter(Boolean);
    if (!pts.length) return;
    paths += `<polyline fill="none" stroke="var(${s.colorVar})" stroke-width="1.05"
               stroke-linejoin="round" stroke-linecap="round"
               points="${pts.map(p => p[0].toFixed(2) + ',' + p[1].toFixed(2)).join(' ')}"/>`;
    pts.forEach(p => {
      paths += `<circle cx="${p[0].toFixed(2)}" cy="${p[1].toFixed(2)}" r="1.25"
                 fill="var(${s.colorVar})" stroke="var(--surface-1)" stroke-width="0.5"/>`;
    });
    const last = pts[pts.length - 1];
    if (opts.directLabel) {
      paths += `<text x="${(last[0] + 1.8).toFixed(2)}" y="${(last[1] + 1.1).toFixed(2)}"
                 font-size="3" fill="var(--text-secondary)">${esc(s.short)}</text>`;
    }
  });

  let hot = '';
  ITERS.forEach((it, i) => {
    const bw = (W - ml - mr) / (ITERS.length - 1);
    hot += `<rect x="${(x(i) - bw / 2).toFixed(2)}" y="0" width="${bw.toFixed(2)}" height="${H}"
             fill="transparent" data-i="${i}"/>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.title)}">
            ${g}${paths}<g class="hot">${hot}</g></svg>`;
}

function facet(title, series, directLabel) {
  const legend = series.length > 1
    ? `<div class="legend">${series.map(s =>
        `<span><i class="swatch" style="background:var(${s.colorVar})"></i>${esc(s.label)}</span>`).join('')}</div>`
    : '';
  const el = document.createElement('div');
  el.className = 'facet';
  el.innerHTML = `<h3>${esc(title)}</h3>${legend}${lineChart(series, {title, directLabel})}`;
  el.querySelectorAll('.hot rect').forEach(r => {
    r.addEventListener('mousemove', e => {
      const i = +r.dataset.i;
      tip.innerHTML = `<div class="hd">${esc(title)} · iter ${i}</div>` + series.map(s =>
        `<div class="r"><span style="color:var(${s.colorVar})">&#9632;</span>
         <span style="flex:1">${esc(s.label)}</span><b>${fmt(s.points[i].y)}</b></div>`).join('');
      tip.style.opacity = 1;
      tip.style.left = Math.min(e.clientX + 14, innerWidth - 210) + 'px';
      tip.style.top = Math.min(e.clientY + 14, innerHeight - 120) + 'px';
    });
    r.addEventListener('mouseleave', () => { tip.style.opacity = 0; });
  });
  return el;
}

const val = (auroc, it) => {
  const c = auroc[it]; if (!c || !c[split]) return null;
  const v = c[split][metric]; return v === undefined ? null : v;
};

/* ---------- cross-experiment ---------- */
function renderCross() {
  const host = document.getElementById('cross-facets');
  host.innerHTML = '';
  const byExp = {};
  DATA.cross_experiment.forEach(a => (byExp[a.exp] = byExp[a.exp] || []).push(a));
  Object.entries(byExp).forEach(([exp, arms]) => {
    const series = arms.map((a, i) => ({
      label: a.label, short: a.label.split(' ')[0],
      colorVar: SERIES_VARS[i % SERIES_VARS.length],
      points: ITERS.map(it => ({y: val(a.auroc, it)})),
    }));
    // One series gets no legend box, so the title has to name it.
    const title = series.length === 1 ? `${exp} — ${arms[0].label}` : exp;
    host.appendChild(facet(title, series, false));
  });

  // scores table
  let h = `<thead><tr><th>arm</th><th>exp</th>` +
    ITERS.map(it => `<th>${it}</th>`).join('') +
    `<th class="grpstart">best</th><th>&Delta; vs iter0</th></tr></thead><tbody>`;
  DATA.cross_experiment.forEach(a => {
    const vs = ITERS.map(it => val(a.auroc, it));
    const fin = vs.filter(v => v != null);
    const best = fin.length ? Math.max(...fin) : null;
    h += `<tr><td>${esc(a.label)}</td><td class="muted">${a.exp}</td>` +
      vs.map(v => `<td class="${v != null && v === best ? 'best' : ''}">${fmt(v)}</td>`).join('') +
      `<td class="grpstart best">${fmt(best)}</td>` +
      `<td class="${best != null && vs[0] != null ? (best - vs[0] >= 0 ? 'pos' : 'neg') : ''}">` +
      `${best != null && vs[0] != null ? (best - vs[0] >= 0 ? '+' : '') + (best - vs[0]).toFixed(4) : '—'}</td></tr>`;
  });
  document.getElementById('cross-scores').innerHTML = h + '</tbody>';

  // samples table
  let s = `<thead><tr><th>arm</th><th>rotation</th><th>attempts</th><th>successes</th><th>rate</th>` +
    `<th class="grpstart">fp succ/att</th><th>fn succ/att</th>` +
    `<th class="grpstart">&rarr; probe</th><th>trained on</th><th>pos</th><th>neg</th><th>total w/ base</th></tr></thead><tbody>`;
  DATA.cross_experiment.forEach(a => {
    const keys = Object.keys(a.rounds).map(Number).sort((x, y) => x - y);
    s += `<tr class="arm-head"><td colspan="12">${esc(a.label)} <span class="muted">· ${a.exp} · ${esc(a.branch)}</span></td></tr>`;
    let ta = 0, ts = 0;
    keys.forEach(k => {
      const r = a.rounds[String(k)];
      const att = r.fp_att + r.fn_att, suc = r.fp_succ + r.fn_succ;
      ta += att; ts += suc;
      const tr = a.train['iter' + (k + 1)];
      s += `<tr><td class="muted">&nbsp;</td><td>iter ${k}</td><td>${att}</td><td>${suc}</td>` +
        `<td>${(100 * suc / att).toFixed(1)}%</td>` +
        `<td class="grpstart">${r.fp_succ}/${r.fp_att}</td><td>${r.fn_succ}/${r.fn_att}</td>` +
        `<td class="grpstart muted">iter${k + 1}</td>` +
        `<td>${tr ? tr.total : '—'}</td><td>${tr ? tr.positive : '—'}</td><td>${tr ? tr.negative : '—'}</td>` +
        `<td>${tr ? tr.total + DATA.base_training_samples : '—'}</td></tr>`;
    });
    // any extra retrain beyond the recorded rotations (exp4 MEMO's 4th)
    Object.keys(a.train).forEach(it => {
      const n = +it.replace('iter', '');
      if (n - 1 > Math.max(...keys)) {
        const tr = a.train[it];
        s += `<tr><td class="muted">&nbsp;</td><td class="muted">iter ${n - 1}*</td><td colspan="5" class="muted">` +
          `rotation not in JSONL</td><td class="grpstart muted">${it}</td><td>${tr.total}</td>` +
          `<td>${tr.positive}</td><td>${tr.negative}</td><td>${tr.total + DATA.base_training_samples}</td></tr>`;
      }
    });
    s += `<tr><td class="muted">&nbsp;</td><td><b>total</b></td><td><b>${ta}</b></td><td><b>${ts}</b></td>` +
      `<td><b>${(100 * ts / ta).toFixed(1)}%</b></td><td class="grpstart" colspan="6"></td></tr>`;
  });
  document.getElementById('cross-samples').innerHTML = s + '</tbody>';

  let p = `<thead><tr><th>arm</th><th>exp</th><th>attacker</th><th>feedback</th><th>view_limit</th>` +
    `<th>cross-iter memo</th><th>judge</th><th>contrastive LLM</th></tr></thead><tbody>`;
  DATA.cross_experiment.forEach(a => {
    const q = a.params;
    p += `<tr><td>${esc(a.label)}</td><td class="muted">${a.exp}</td><td>${esc(q.attacker)}</td>` +
      `<td>${esc(q.feedback)}</td><td>${q.view_limit}</td><td>${esc(q.memo)}</td>` +
      `<td>${esc(q.judge)}</td><td>${esc(q.contrastive)}</td></tr>`;
  });
  document.getElementById('cross-params').innerHTML = p + '</tbody>';
}

/* ---------- ablation ---------- */
function renderAbl() {
  const host = document.getElementById('abl-facets');
  host.innerHTML = '';
  DATA.ablation.forEach(arm => {
    const series = arm.variants.map((v, i) => ({
      label: v.label, short: '',
      colorVar: SERIES_VARS[i % SERIES_VARS.length],
      points: ITERS.map(it => ({y: val(v.auroc, it)})),
    }));
    host.appendChild(facet(arm.label, series, false));
  });

  let h = `<thead><tr><th>arm</th><th>variant</th>` + ITERS.map(it => `<th>${it}</th>`).join('') +
    `<th class="grpstart">best</th></tr></thead><tbody>`;
  DATA.ablation.forEach(arm => {
    const bests = arm.variants.map(v => {
      const f = ITERS.map(it => val(v.auroc, it)).filter(x => x != null);
      return f.length ? Math.max(...f) : null;
    });
    const armBest = Math.max(...bests.filter(x => x != null));
    arm.variants.forEach((v, i) => {
      const vs = ITERS.map(it => val(v.auroc, it));
      h += `<tr><td class="muted">${i === 0 ? esc(arm.label) : ''}</td><td>${esc(v.label)}</td>` +
        vs.map(x => `<td class="${x != null && x === bests[i] ? 'best' : ''}">${fmt(x)}</td>`).join('') +
        `<td class="grpstart ${bests[i] === armBest ? 'best pos' : ''}">${fmt(bests[i])}</td></tr>`;
    });
  });
  document.getElementById('abl-scores').innerHTML = h + '</tbody>';

  let s = `<thead><tr><th>arm</th><th>variant</th><th>conf gate</th><th>contrastive</th><th>iter</th>` +
    `<th class="grpstart">raw succ</th><th>after conf</th><th>trained</th><th>pos</th><th>neg</th>` +
    `<th>pos %</th><th class="grpstart">total w/ base</th></tr></thead><tbody>`;
  DATA.ablation.forEach(arm => {
    s += `<tr class="arm-head"><td colspan="12">${esc(arm.label)}</td></tr>`;
    arm.variants.forEach(v => {
      ['iter1', 'iter2', 'iter3'].forEach((it, j) => {
        const t = v.train[it] || {};
        const pct = t.total ? (100 * t.positive / t.total) : null;
        s += `<tr><td class="muted">&nbsp;</td><td>${j === 0 ? esc(v.label) : ''}</td>` +
          `<td>${j === 0 ? '&ge; ' + v.min_judge_confidence : ''}</td>` +
          `<td>${j === 0 ? (v.contrastive ? 'yes' : 'no') : ''}</td><td>${it}</td>` +
          `<td class="grpstart">${t.successes_raw ?? '—'}</td><td>${t.successes_after_conf ?? '—'}</td>` +
          `<td>${t.total ?? '—'}</td><td>${t.positive ?? '—'}</td><td>${t.negative ?? '—'}</td>` +
          `<td>${pct == null ? '—' : pct.toFixed(1) + '%'}</td>` +
          `<td class="grpstart">${t.total_training ?? '—'}</td></tr>`;
      });
    });
  });
  document.getElementById('abl-samples').innerHTML = s + '</tbody>';
}

function renderAll() { renderCross(); renderAbl(); }

/* ---------- wiring ---------- */
const sel = document.getElementById('split');
SPLITS.forEach(sp => {
  const o = document.createElement('option');
  o.value = sp; o.textContent = sp === 'mean' ? 'mean (all four)' : sp;
  sel.appendChild(o);
});
sel.value = 'mean';
sel.addEventListener('change', () => { split = sel.value; renderAll(); });
document.getElementById('metric').addEventListener('change', e => { metric = e.target.value; renderAll(); });
function selectTab(name) {
  tab = (name === 'abl') ? 'abl' : 'cross';
  document.querySelectorAll('.tab').forEach(x =>
    x.setAttribute('aria-selected', String(x.dataset.tab === tab)));
  document.getElementById('panel-cross').classList.toggle('hidden', tab !== 'cross');
  document.getElementById('panel-abl').classList.toggle('hidden', tab !== 'abl');
}
document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => {
  selectTab(b.dataset.tab);
  history.replaceState(null, '', '#' + tab);
}));
addEventListener('hashchange', () => selectTab(location.hash.slice(1)));
selectTab(location.hash.slice(1));
renderAll();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=REPO / "scripts" / "hu_harm_auroc_data.json")
    ap.add_argument("--out", type=Path, default=REPO / "viewers" / "hu_harm_llama1b_auroc_viewer.html")
    ap.add_argument("--collect", action="store_true",
                    help="run collect_hu_harm_auroc_data.py first")
    args = ap.parse_args()

    if args.collect or not args.data.exists():
        subprocess.run(
            [str(REPO / ".venv_claude" / "bin" / "python"),
             str(REPO / "scripts" / "collect_hu_harm_auroc_data.py"), "-o", str(args.data)],
            check=True, cwd=REPO,
        )

    data = json.loads(args.data.read_text())
    # Token replacement, not %-formatting: the CSS/JS is full of literal % signs.
    html = TEMPLATE
    for i, c in enumerate(PALETTE_LIGHT):
        html = html.replace(f"__S{i + 1}__", c)
    for i, c in enumerate(PALETTE_DARK):
        html = html.replace(f"__D{i + 1}__", c)
    html = html.replace(
        "__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
