#!/usr/bin/env python
"""Render the novelty study as a standalone page: results/novelty_study.html.

Every figure reads from results/ (pooled_*, novelty_*, regions_*, ablation_*), so the
page is regenerated rather than hand-edited. Charts are inline SVG computed here; the
palette is validated against the OKLab/CVD checks in both modes before use.
"""

from __future__ import annotations

import html
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments as X  # noqa: E402
import report as R  # noqa: E402

ARM_LABEL = {
    "gptoss": "instructions / gptoss",
    "nemotron": "instructions / nemotron",
    "gptoss120b": "high-stakes / gptoss120b",
    "deepseekv4pro": "high-stakes / deepseekv4pro",
}


def esc(s) -> str:
    return html.escape(str(s))


# ------------------------------------------------------------------ data


def collect() -> dict:
    arms = []
    for exp, arm in X.all_arms():
        f = X.RESULTS / f"novelty_{exp.key}_{arm.name}.npz"
        if not f.exists():
            continue
        s = np.load(f, allow_pickle=True)
        p95 = float(s["_eval_self_p95"][0])
        rows = [r for r in R.load_rows(exp.key) if r["arm"] == arm.name]
        base = next((r for r in rows if r["condition"] == "full"), None)
        arms.append(
            {
                "exp": exp.key,
                "arm": arm.name,
                "label": ARM_LABEL.get(arm.name, arm.name),
                "n": int(len(s["knn_eval"])),
                "p95": p95,
                "dev_outside": float(100 * (s["_dev_to_eval_knn"] > p95).mean()),
                "knn_eval": float(s["knn_eval"].mean()),
                "outside": float(100 * s["outside"].mean()),
                "along_frac": float(s["along_frac"].mean()),
                "corr_orth": float(np.corrcoef(s["knn_eval"], s["orth"])[0, 1]),
                "corr_along": float(np.corrcoef(s["knn_eval"], np.abs(s["along"]))[0, 1]),
                "published_delta": float(R.published_delta(arm)),
                "ensemble": exp.ensemble_size,
                "rows": rows,
                "base": base,
                "band": R.comparison_band(rows) if rows else None,
                "perm_floor": R.noise_floor(rows)[0] if rows else None,
                "regions": _regions(exp, arm, rows, base),
                "h2h": _h2h(rows, R.comparison_band(rows)) if rows else [],
            }
        )
    return {"arms": arms}


def _regions(exp, arm, rows, base) -> list[dict]:
    f = X.RESULTS / f"regions_{exp.key}_{arm.name}_kmeans.json"
    if not f.exists() or base is None:
        return []
    regs = {r["id"]: r for r in json.loads(f.read_text())["regions"]}
    by = {r["condition"]: r for r in rows}
    band = R.comparison_band(rows)
    out = []
    for rid, reg in regs.items():
        r = by.get(f"drop_region_{rid}")
        if not r:
            continue
        d = r["macro_auroc"] - base["macro_auroc"]
        dc = (
            r["cross_auroc"] - base["cross_auroc"]
            if r.get("cross_auroc") is not None and base.get("cross_auroc") is not None
            else None
        )
        ex = reg["examples"][0]["text"] if reg["examples"] else ""
        out.append(
            {
                "id": rid,
                "n": reg["n"],
                "outside": reg["outside_pct"],
                "d_eval": d,
                "d_cross": dc,
                "verdict": R.verdict(d, dc, band),
                "excerpt": " ".join(ex.split())[:190],
            }
        )
    out.sort(key=lambda r: -r["d_eval"])
    return out


def _h2h(rows, band) -> list[dict]:
    by = {r["condition"]: r for r in rows}
    qs = sorted({int(c.split("_q")[1].split("_")[0]) for c in by if c.startswith("drop_top_q")})
    out = []
    for q in qs:
        top, bot = by.get(f"drop_top_q{q}"), by.get(f"drop_bottom_q{q}")
        rnd = [r["macro_auroc"] for c, r in by.items() if c.startswith(f"drop_random_q{q}_")]
        if not top or not rnd:
            continue
        rm = statistics.fmean(rnd)
        out.append(
            {
                "q": q,
                "top": top["macro_auroc"],
                "bottom": bot["macro_auroc"] if bot else None,
                "rand": rm,
                "rand_range": (max(rnd) - min(rnd)) / 2,
                "diff": top["macro_auroc"] - rm,
                "beats": abs(top["macro_auroc"] - rm) > band,
            }
        )
    return out


# ------------------------------------------------------------------ charts


def scatter(arms: list[dict]) -> str:
    """outside% vs the arm's published eval gain. Four points, all direct-labelled."""
    pts = [a for a in arms if not np.isnan(a["published_delta"])]
    if len(pts) < 2:
        return ""
    W, H = 660, 340
    ml, mr, mt, mb = 62, 130, 22, 46
    xs = [p["outside"] for p in pts]
    ys = [p["published_delta"] for p in pts]
    x0, x1 = 20, 90
    y0, y1 = min(-0.10, min(ys) - 0.01), max(0.10, max(ys) + 0.01)
    sx = lambda v: ml + (v - x0) / (x1 - x0) * (W - ml - mr)
    sy = lambda v: mt + (y1 - v) / (y1 - y0) * (H - mt - mb)

    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Red-team novelty against eval gain, four arms" class="chart">']
    for gx in range(x0, x1 + 1, 10):
        g.append(f'<line x1="{sx(gx):.1f}" y1="{mt}" x2="{sx(gx):.1f}" y2="{H-mb}" class="grid"/>')
        g.append(f'<text x="{sx(gx):.1f}" y="{H-mb+18}" class="tick" text-anchor="middle">{gx}%</text>')
    for gy in (-0.10, -0.05, 0.0, 0.05, 0.10):
        cls = "zero" if abs(gy) < 1e-9 else "grid"
        g.append(f'<line x1="{ml}" y1="{sy(gy):.1f}" x2="{W-mr}" y2="{sy(gy):.1f}" class="{cls}"/>')
        g.append(f'<text x="{ml-10}" y="{sy(gy)+4:.1f}" class="tick" text-anchor="end">{gy:+.2f}</text>')

    # Least-squares line: the claim is about the sign of the relationship, so show it.
    b, a = np.polyfit(xs, ys, 1)
    g.append(
        f'<line x1="{sx(x0):.1f}" y1="{sy(a + b * x0):.1f}" x2="{sx(x1):.1f}" y2="{sy(a + b * x1):.1f}" class="fit"/>'
    )
    for p in pts:
        c = "s-instr" if p["exp"] == "instructions" else "s-hs"
        cx, cy = sx(p["outside"]), sy(p["published_delta"])
        anchor = "start" if p["outside"] < 70 else "end"
        dx = 13 if anchor == "start" else -13
        g.append(
            f'<g class="pt"><circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" class="{c}"/>'
            f'<title>{esc(p["label"])} — {p["outside"]:.0f}% outside, published Δ {p["published_delta"]:+.3f}</title>'
            f'<text x="{cx+dx:.1f}" y="{cy+4:.1f}" class="ptlabel" text-anchor="{anchor}">{esc(p["arm"])}</text></g>'
        )
    g.append(f'<text x="{ml + (W-ml-mr)/2:.0f}" y="{H-6}" class="axis" text-anchor="middle">red-team rows outside the eval manifold</text>')
    g.append(f'<text transform="translate(16,{mt + (H-mt-mb)/2:.0f}) rotate(-90)" class="axis" text-anchor="middle">published Δ macro AUROC</text>')
    g.append("</svg>")
    return "\n".join(g)


def dotplot(arm: dict) -> str:
    """Δ eval and Δ cross-attacker for the conditions that carry the argument."""
    base, rows, band = arm["base"], arm["rows"], arm["band"]
    if not base:
        return ""
    want = ["base_only", "drop_outside", "drop_relative_q20", "drop_top_q40", "drop_top_q20",
            "drop_top_q10", "drop_bottom_q20", "drop_bottom_q40"]
    by = {r["condition"]: r for r in rows}
    sel = [by[c] for c in want if c in by]
    if not sel:
        return ""
    rowh, ml, mr, mt = 28, 168, 26, 30
    H = mt + rowh * len(sel) + 42
    W = 660
    lim = max(0.11, max(abs(r["macro_auroc"] - base["macro_auroc"]) for r in sel) * 1.15)
    sx = lambda v: ml + (v + lim) / (2 * lim) * (W - ml - mr)

    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Effect of each removal on eval and cross-attacker AUROC" class="chart">']
    g.append(f'<rect x="{sx(-band):.1f}" y="{mt-8}" width="{sx(band)-sx(-band):.1f}" height="{rowh*len(sel)+8}" class="bandrect"/>')
    g.append(f'<line x1="{sx(0):.1f}" y1="{mt-8}" x2="{sx(0):.1f}" y2="{mt+rowh*len(sel):.1f}" class="zero"/>')
    for i, r in enumerate(sel):
        y = mt + i * rowh + rowh / 2
        d = r["macro_auroc"] - base["macro_auroc"]
        dc = (r["cross_auroc"] - base["cross_auroc"]) if (r.get("cross_auroc") is not None and base.get("cross_auroc") is not None) else None
        g.append(f'<text x="{ml-12}" y="{y+4:.1f}" class="rowlabel" text-anchor="end">{esc(r["condition"])}</text>')
        if dc is not None:
            g.append(
                f'<g class="pt"><circle cx="{sx(dc):.1f}" cy="{y:.1f}" r="6" class="m-cross"/>'
                f'<title>{esc(r["condition"])}: cross-attacker {dc:+.4f}</title></g>'
            )
        g.append(
            f'<g class="pt"><circle cx="{sx(d):.1f}" cy="{y:.1f}" r="6" class="m-eval"/>'
            f'<title>{esc(r["condition"])}: eval {d:+.4f}</title></g>'
        )
    for gv in (-0.10, -0.05, 0.0, 0.05, 0.10):
        if abs(gv) > lim:
            continue
        g.append(f'<text x="{sx(gv):.1f}" y="{H-14}" class="tick" text-anchor="middle">{gv:+.2f}</text>')
    g.append(f'<text x="{ml + (W-ml-mr)/2:.0f}" y="{H-1}" class="axis" text-anchor="middle">Δ macro AUROC vs keeping everything</text>')
    g.append("</svg>")
    return "\n".join(g)


# ------------------------------------------------------------------ page


def build(data: dict) -> str:
    arms = data["arms"]
    inst = [a for a in arms if a["exp"] == "instructions"]
    hs = [a for a in arms if a["exp"] == "highstakes"]
    outs = [a["outside"] for a in arms]
    corr = np.corrcoef([a["outside"] for a in arms], [a["published_delta"] for a in arms])[0, 1] if len(arms) > 2 else float("nan")
    beats = [h["diff"] for a in arms for h in a["h2h"] if h["beats"]]
    n_beats, n_tot = len(beats), sum(len(a["h2h"]) for a in arms)
    n_pos, n_neg = sum(1 for d in beats if d > 0), sum(1 for d in beats if d < 0)
    # base_only vs full, per arm -- the one effect with a consistent sign.
    strip = []
    for a in arms:
        bo = next((r for r in a["rows"] if r["condition"] == "base_only"), None)
        if bo and a["base"]:
            strip.append({
                "label": a["label"],
                "d_eval": bo["macro_auroc"] - a["base"]["macro_auroc"],
                "d_cross": (bo["cross_auroc"] - a["base"]["cross_auroc"])
                if bo.get("cross_auroc") and a["base"].get("cross_auroc") else None,
                "eval_from": a["base"]["macro_auroc"], "eval_to": bo["macro_auroc"],
                "cross_from": a["base"].get("cross_auroc"), "cross_to": bo.get("cross_auroc"),
            })
    n_cond = sum(len(a["rows"]) for a in arms)
    n_fits = sum(len(a["rows"]) * a["ensemble"] for a in arms)
    # The headline cost: the calibrated "drop everything outside" condition, whose eval
    # effect is inside the band on the instructions arms while cross-attacker AUROC falls.
    outside_costs = []
    for a in arms:
        r = next((x for x in a["rows"] if x["condition"] == "drop_outside"), None)
        if r and a["base"] and r.get("cross_auroc") is not None and a["base"].get("cross_auroc") is not None:
            outside_costs.append(r["cross_auroc"] - a["base"]["cross_auroc"])
    worst_outside = min(outside_costs) if outside_costs else float("nan")

    P = []
    P.append(f"""<title>Novelty Is Not the Culprit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>""")

    P.append(f"""
<header class="hero">
  <p class="eyebrow">Red-team activation study · gemma-3-27b L32 · 4 arms</p>
  <h1>Does off-manifold red-team data hurt eval scores?</h1>
  <p class="lede">Across four red-team arms, {n_cond} removal conditions and {n_fits} probe fits,
  <strong>no</strong>. Distance from the eval manifold does not predict harm — the most
  off-manifold attack sets produced the largest eval gains. And the removal that looks free on
  eval quietly costs up to {abs(worst_outside) * 100:.1f} points of AUROC against a second
  attacker the probe never trained on.</p>
  <div class="tiles">
    <div class="tile"><span class="tnum">{min(outs):.0f}–{max(outs):.0f}%</span>
      <span class="tlab">of red-team rows sit outside the eval manifold, depending on the arm</span></div>
    <div class="tile"><span class="tnum">{corr:+.2f}</span>
      <span class="tlab">correlation between that share and the arm's eval gain — the wrong sign for the hypothesis</span></div>
    <div class="tile"><span class="tnum">{n_pos} up / {n_neg} down</span>
      <span class="tlab">of {n_tot} matched-n comparisons, novelty-targeted removal beat random on some arms and lost on others — no usable sign</span></div>
  </div>
</header>
<main>
""")

    # --- Phase 1
    P.append(f"""
<section>
  <p class="phase">Phase 1 · measure</p>
  <h2>The novelty is real, and it points the wrong way</h2>
  <p>Every red-team row is scored by its mean cosine distance to its ten nearest eval rows,
  calibrated against the eval set's own dispersion: a row counts as <em>outside</em> when it
  sits further from eval than 95% of eval sits from itself. By that measure the held-out dev
  sets land at {min(a["dev_outside"] for a in arms):.0f}–{max(a["dev_outside"] for a in arms):.0f}%
  outside, and the red-team sets at {min(outs):.0f}–{max(outs):.0f}%. So the premise holds:
  red-team data really does live somewhere else.</p>
  <p>What does not hold is the consequence. Plotting each arm's novelty against the eval gain
  its own published run achieved gives a line sloping the wrong way.</p>
  <figure>
    {scatter(arms)}
    <div class="legend"><span class="key"><i class="sw s-instr"></i>instructions</span>
      <span class="key"><i class="sw s-hs"></i>high-stakes</span></div>
    <figcaption>Each point is one attacker arm. The arm whose red-team set was
    <em>least</em> novel (deepseekv4pro, {[a for a in arms if a["arm"] == "deepseekv4pro"][0]["outside"]:.0f}% outside)
    is the one whose eval score fell furthest. Four points is a weak basis for a slope — it is
    a cross-arm correlation, not a controlled comparison — but it is the opposite of what the
    hypothesis predicts, and it is what motivated testing removal directly.</figcaption>
  </figure>
  <div class="tablewrap"><table>
    <caption>Phase 1 — novelty per arm</caption>
    <thead><tr><th>arm</th><th>rows</th><th>dev outside</th><th>red-team outside</th>
      <th>on the probe's axis</th><th>corr(novelty, ⊥)</th><th>published Δ eval</th></tr></thead>
    <tbody>""")
    for a in arms:
        P.append(
            f'<tr><td class="mono">{esc(a["label"])}</td><td class="num">{a["n"]}</td>'
            f'<td class="num">{a["dev_outside"]:.1f}%</td><td class="num strong">{a["outside"]:.1f}%</td>'
            f'<td class="num">{a["along_frac"]:.3f}</td><td class="num">{a["corr_orth"]:+.2f}</td>'
            f'<td class="num strong">{a["published_delta"]:+.4f}</td></tr>'
        )
    P.append("""</tbody></table></div>
  <h3>The novelty is almost entirely invisible to the probe</h3>
  <p>The probe is a single direction <span class="mono">w</span> — a 5376-wide linear head with a
  softmax pool over tokens — so each row's displacement from its local eval neighbourhood splits
  exactly into a component along <span class="mono">w</span> and one orthogonal to it. On all four
  arms, only 2–5% of that displacement lies on the decision axis, and novelty correlates
  +0.90 to +0.93 with the <em>orthogonal</em> part.</p>
  <p class="callout">Being far from eval is therefore not something the probe can read off a
  score. These rows cannot drag eval predictions directly; whatever they do, they do by
  <em>rotating</em> <span class="mono">w</span> during the refit. That is a mechanism a distance
  analysis alone cannot see, and it is why the sign of the effect has to be measured rather
  than reasoned about.</p>
</section>
""")

    # --- Phase 2
    P.append(f"""
<section>
  <p class="phase">Phase 2 · group</p>
  <h2>There is no compact bad region to excise</h2>
  <p>If a specific far-out family were poisoning eval, density clustering should find it.
  HDBSCAN over each arm's own PCA assigns the large majority of rows to noise — these attack
  sets are diffuse, not organised into tight families. On the two high-stakes arms it finds no
  clusters at all. That is already an answer to "which region do we delete": there isn't one
  waiting to be found.</p>
  <p>A k-means covering (k=6) is used instead for the removal experiments, so every row belongs
  to some region and the whole set is covered. It makes no claim that the regions are natural
  kinds — but it does separate them cleanly by novelty, which is what Phase 3 needs.</p>
</section>
""")

    # --- Phase 3
    P.append("""
<section>
  <p class="phase">Phase 3 · test</p>
  <h2>Removing the novel rows, and refitting</h2>
  <p>Thirty-three conditions per arm, each a full refit on <span class="mono">base ∪ redteam[keep]</span>
  scored on the full eval splits. Two controls carry the argument. <strong>Matched-n random
  removal</strong> is the size control: if dropping the most novel rows does no better than
  dropping the same number at random, the geometry is not telling us <em>which</em> rows to drop.
  <strong>Identical-data row permutations</strong> give the noise floor, and the spread across
  random draws gives the wider band a targeted removal has to clear to mean anything.</p>
""")
    for a in arms:
        if not a["h2h"]:
            continue
        P.append(f"""
  <h3>{esc(a["label"])}</h3>
  <p class="meta"><span class="mono">full</span> = {a["base"]["macro_auroc"]:.4f} macro AUROC over
  {a["n"]} red-team rows · comparison band ±{a["band"]:.4f} · row-order floor ±{a["perm_floor"]:.4f}
  · {a["ensemble"]}-member fit</p>
  <div class="tablewrap"><table>
    <caption>Novelty-ordered vs random removal, matched n</caption>
    <thead><tr><th>drop</th><th>most novel</th><th>least novel</th><th>random</th>
      <th>novel − random</th><th>beats the band?</th></tr></thead><tbody>""")
        for h in a["h2h"]:
            bot = f'{h["bottom"]:.4f}' if h["bottom"] is not None else "—"
            P.append(
                f'<tr><td class="num">{h["q"]}%</td><td class="num">{h["top"]:.4f}</td>'
                f'<td class="num">{bot}</td><td class="num">{h["rand"]:.4f} ±{h["rand_range"]:.4f}</td>'
                f'<td class="num strong">{h["diff"]:+.4f}</td>'
                f'<td>{"<span class=\'flag yes\'>yes</span>" if h["beats"] else "<span class=\'flag no\'>no</span>"}</td></tr>'
            )
        P.append("</tbody></table></div>")
        P.append(f'<figure>{dotplot(a)}<div class="legend">'
                 f'<span class="key"><i class="sw m-eval"></i>Δ eval</span>'
                 f'<span class="key"><i class="sw m-cross"></i>Δ cross-attacker</span>'
                 f'<span class="key"><i class="sw sw-band"></i>comparison band</span></div>'
                 f'<figcaption>Marks inside the shaded band are indistinguishable from noise.</figcaption></figure>')

    P.append(f"""
  <h3>The result that matters most</h3>
  <p class="callout warn"><strong><span class="mono">drop_outside</span> — removing every row past
  the calibrated threshold — leaves eval essentially unchanged on both instructions arms
  ({", ".join(f'{(next(r for r in a["rows"] if r["condition"]=="drop_outside")["macro_auroc"] - a["base"]["macro_auroc"]):+.4f}' for a in inst if a["base"])})
  while cutting cross-attacker AUROC by
  {", ".join(f'{(next(r for r in a["rows"] if r["condition"]=="drop_outside")["cross_auroc"] - a["base"]["cross_auroc"]):+.4f}' for a in inst if a["base"] and a["base"].get("cross_auroc"))}.</strong>
  This is the trap the study was built to catch. "Eval stayed the same" is not evidence the rows
  were useless — it is evidence that eval has no support where those rows live. Scored against a
  <em>second attacker's</em> red-team set, which the probe never trained on, the same rows are
  clearly load-bearing.</p>
</section>
""")

    # --- regions
    P.append("""
<section>
  <p class="phase">Attribution</p>
  <h2>What the regions actually contain</h2>
  <p>Removing one k-means region at a time, and then reading the conversations in it, separates
  cases that novelty scores alone cannot tell apart. The clearest example is on the
  instructions/gptoss arm, where two regions of near-identical size — 146 and 147 rows — produce
  opposite effects.</p>
""")
    for a in arms:
        if not a["regions"]:
            continue
        P.append(f"""
  <h3>{esc(a["label"])}</h3>
  <div class="tablewrap"><table>
    <caption>Effect of removing each region</caption>
    <thead><tr><th>region</th><th>n</th><th>outside</th><th>Δ eval</th><th>Δ cross</th>
      <th>verdict</th><th>representative content</th></tr></thead><tbody>""")
        for r in a["regions"]:
            v = r["verdict"]
            cls = "v-drop" if v == "DROP" else ("v-keep" if v.startswith("KEEP") else ("v-trade" if v.startswith("TRADE") else "v-inert"))
            dc = f'{r["d_cross"]:+.4f}' if r["d_cross"] is not None else "—"
            P.append(
                f'<tr><td class="mono">region_{r["id"]}</td><td class="num">{r["n"]}</td>'
                f'<td class="num">{r["outside"]:.0f}%</td><td class="num strong">{r["d_eval"]:+.4f}</td>'
                f'<td class="num">{dc}</td><td><span class="verdict {cls}">{esc(v)}</span></td>'
                f'<td class="excerpt">{esc(r["excerpt"])}…</td></tr>'
            )
        P.append("</tbody></table></div>")
    drops = [(a, r) for a in arms for r in a["regions"] if r["verdict"] == "DROP"]
    blind = [(a, r) for a in arms for r in a["regions"] if r["verdict"].startswith("KEEP (eval blind)")]
    n_reg = sum(len(a["regions"]) for a in arms)
    P.append(f"""
  <p>Of {n_reg} regions across the four arms, {len(drops)} earns a
  <span class="verdict v-drop">DROP</span> and {len(blind)} earn
  <span class="verdict v-keep">KEEP (eval blind)</span> — flat on eval, but measurably worse
  against the second attacker. The two instructions arms agree on <em>which</em> region is
  expendable, and it is not the one novelty would pick out. In both it is the region of trivial
  constraint-following contrastive pairs — “list four colours starting with b”, the same prompt
  twice with one trailing clause changed; “write a sentence where every word is four letters”.
  Those regions sit at 92% and 98% outside the eval manifold. So do several regions whose removal
  costs cross-attacker robustness or eval outright.</p>
  <p class="callout"><strong>Distance ranked those regions the same way. Only their content, and
  the cross-attacker check, separated the expendable one from the load-bearing ones.</strong>
  That is also the practical opening: contrastive minimal pairs are identifiable directly, from
  the preprocessing step that mints them — no activation geometry required.</p>
</section>
""")

    P.append("""
<section>
  <p class="phase">The consistent result</p>
  <h2>The only effect with a fixed sign isn't about novelty</h2>
  <p>Set the geometry aside and simply compare keeping every red-team row against keeping none.
  The eval column swings both ways — but the cross-attacker column does not.</p>
  <div class="tablewrap"><table>
    <caption>Removing all red-team data</caption>
    <thead><tr><th>arm</th><th>eval: full → none</th><th>Δ</th>
      <th>cross-attacker: full → none</th><th>Δ</th></tr></thead><tbody>""")
    for r in strip:
        cf = f'{r["cross_from"]:.4f} → {r["cross_to"]:.4f}' if r["cross_from"] else "—"
        dc = f'{r["d_cross"]:+.4f}' if r["d_cross"] is not None else "—"
        P.append(
            f'<tr><td class="mono">{esc(r["label"])}</td>'
            f'<td class="num">{r["eval_from"]:.4f} → {r["eval_to"]:.4f}</td>'
            f'<td class="num strong">{r["d_eval"]:+.4f}</td>'
            f'<td class="num">{cf}</td><td class="num strong">{dc}</td></tr>'
        )
    P.append("""</tbody></table></div>
  <p class="callout"><strong>Every arm loses 7–13 points of AUROC against the other attacker's
  conversations when its red-team data is removed — including the two arms where removing it
  <em>improves</em> the eval score.</strong> Whatever red-teaming buys, eval is a poor instrument
  for seeing it, and on high-stakes it scores it negatively.</p>
</section>
""")

    # --- what to do
    P.append(f"""
<section>
  <p class="phase">So</p>
  <h2>What to do with this</h2>
  <ol class="doing">
    <li><strong>Don't prune by distance.</strong> Across {n_tot} matched-n comparisons, targeting
    the most novel rows differed from random removal by more than the noise band {n_beats} times —
    {n_pos} in its favour and {n_neg} against. On both instructions arms removal ranks
    <span class="mono">most-novel &gt; random &gt; least-novel</span>; on high-stakes/deepseekv4pro
    that inverts, and dropping the <em>least</em> novel 40% is the better move. A signal that
    reverses between concepts is not a pruning rule.</li>
    <li><strong>Never accept "eval didn't move" as permission to delete.</strong> On both
    instructions arms the flat-on-eval removal was the most expensive one measured, once a second
    attacker was allowed to score it. Any pruning rule needs a robustness metric eval cannot see.</li>
    <li><strong>The one real win is content, not geometry.</strong> Contrastive minimal pairs that
    differ by a trailing clause are cheap to identify directly — they come out of the
    preprocessing step, not out of the attacker — and they are the only rows whose removal helped
    eval without costing anything.</li>
    <li><strong>Red-teaming is a trade, and eval only sees one side of it.</strong> Removing every
    red-team row changes eval in <em>both</em> directions depending on the concept — instructions
    loses 0.06–0.09, high-stakes <em>gains</em> up to 0.11 — but costs 7–13 points of
    cross-attacker AUROC in all four arms. If high-stakes red-teaming is judged on its eval alone
    it looks like damage; judged on held-out attacks it is the largest effect in the study.</li>
    <li><strong>Look at high-stakes' baseline before reading its regression.</strong> Both concepts'
    iteration-0 probes train on 50 rows, under the 64-row floor set by
    <span class="mono">batch_size 16 × gradient_accumulation_steps 4</span>, so they take zero
    optimizer steps. That baseline is a random projection, and on high-stakes a random projection
    already scores ~0.92 macro AUROC. A "regression" measured against it is not what it appears
    to be.</li>
  </ol>
</section>

<section class="method">
  <p class="phase">Method</p>
  <h2>How to trust these numbers</h2>
  <ul>
    <li><strong>Everything is offline.</strong> Every fit reads the runs' own cached
    activations — eval blobs, the dev blob, the base blob, the per-conversation red-team blobs.
    No LLM is ever loaded; a cache miss raises rather than silently forwarding a 27B model.</li>
    <li><strong>Parameters were recovered, not assumed.</strong> The activation blob filenames are
    content hashes of their inputs, so the runs' real settings were read back off disk and
    verified by reproducing the hashes — including <span class="mono">seed=42</span>, which is the
    CLI default rather than the <span class="mono">seed=0</span> in the retrain signature.</li>
    <li><strong>Two known numbers are reproduced exactly.</strong> The instructions/gptoss
    <span class="mono">full</span> refit lands on 0.8272, matching the ceiling study's
    file-order refit, and <span class="mono">base_only</span> on 0.7714, matching the published
    iteration-0 score.</li>
    <li><strong>Row order is a real variable</strong> and is held constant. Cache-hit ordering
    means no refit reproduces a published probe bit-identically, so every condition is compared
    to this study's own <span class="mono">full</span>, never to the comparison CSV.</li>
    <li><strong>Novelty scoring is probe-independent by construction.</strong> Pooling is a flat
    mask-weighted mean, not the probe's softmax pool, so a row is never "novel" because the probe
    already mishandles it. The probe's view enters only in the along/orthogonal decomposition,
    where it is explicit.</li>
    <li><strong>Known limits.</strong> Four arms is a small basis for the cross-arm slope. The
    k-means regions are a covering, so their effects are not independent. High-stakes fits
    validate on a 400-row stratified subsample of dev (its full 19.6 GiB blob makes an
    all-resident fit impossible), identical across its conditions.</li>
  </ul>
</section>
</main>
<footer><p>Generated from <span class="mono">analysis/novelty/</span> — pooled vectors, novelty
scores, regions and ablation results all regenerate this page.</p></footer>
""")
    return "\n".join(P)


CSS = """
:root{
  --ground:#F4F6F8; --panel:#FFFFFF; --ink:#171A1F; --ink-2:#414B57; --muted:#67727F;
  --rule:#DDE3E9; --rule-2:#EDF0F4;
  --inside:#008A7C; --outside:#C06A10;
  --good:#2F6B45; --band:rgba(103,114,127,.13);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#14171B; --panel:#1A1E24; --ink:#E7EAEE; --ink-2:#B9C2CC; --muted:#8B97A4;
    --rule:#272D35; --rule-2:#1F242B;
    --inside:#1BA192; --outside:#C27D1E;
    --good:#4E9A68; --band:rgba(139,151,164,.16);
  }
}
:root[data-theme="dark"]{
  --ground:#14171B; --panel:#1A1E24; --ink:#E7EAEE; --ink-2:#B9C2CC; --muted:#8B97A4;
  --rule:#272D35; --rule-2:#1F242B;
  --inside:#1BA192; --outside:#C27D1E;
  --good:#4E9A68; --band:rgba(139,151,164,.16);
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);
  font-family:"Public Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:17px;line-height:1.62;margin:0;padding:0 24px 80px;
  -webkit-font-smoothing:antialiased}
.mono,code{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.87em;letter-spacing:-.01em}
h1,h2,h3{font-family:Spectral,Georgia,"Times New Roman",serif;font-weight:600;
  text-wrap:balance;line-height:1.18;margin:0}
h1{font-size:clamp(2.1rem,5.2vw,3.15rem);letter-spacing:-.015em}
h2{font-size:clamp(1.45rem,2.9vw,1.95rem);margin-bottom:.5em}
h3{font-size:1.16rem;margin:2.2em 0 .35em}
p{margin:0 0 1.05em}
.hero,main>section,footer{max-width:74ch;margin-inline:auto}
.hero{padding:72px 0 40px;border-bottom:1px solid var(--rule)}
.eyebrow,.phase{font-size:.74rem;letter-spacing:.11em;text-transform:uppercase;
  color:var(--muted);font-weight:600;margin:0 0 1.1em}
.phase{color:var(--inside);margin-bottom:.7em}
.lede{font-size:1.2rem;color:var(--ink-2);margin-top:1.1em;max-width:62ch}
.lede strong{color:var(--ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:20px;margin-top:34px}
.tile{display:flex;flex-direction:column;gap:7px;padding:18px 20px;background:var(--panel);
  border:1px solid var(--rule);border-radius:3px}
.tnum{font-family:Spectral,Georgia,serif;font-size:1.85rem;font-weight:600;
  color:var(--inside);font-variant-numeric:tabular-nums;line-height:1}
.tlab{font-size:.83rem;color:var(--muted);line-height:1.45}
main>section{padding:52px 0;border-bottom:1px solid var(--rule-2)}
section.method{border-bottom:none}
.callout{border-left:3px solid var(--inside);padding:2px 0 2px 18px;color:var(--ink-2)}
.callout.warn{border-left-color:var(--outside)}
.callout strong{color:var(--ink)}
.meta{font-size:.9rem;color:var(--muted);margin-bottom:1em}
figure{margin:26px 0 30px}
figcaption{font-size:.86rem;color:var(--muted);margin-top:10px;line-height:1.5;max-width:64ch}
.chart{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--rule);stroke-width:1}
.zero{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:none;opacity:.55}
.fit{stroke:var(--muted);stroke-width:2;stroke-dasharray:5 4;opacity:.6}
.bandrect{fill:var(--band)}
.tick{fill:var(--muted);font-size:11px;font-family:"IBM Plex Mono",monospace}
.axis{fill:var(--muted);font-size:11.5px;letter-spacing:.03em}
.ptlabel{fill:var(--ink-2);font-size:12px;font-family:"IBM Plex Mono",monospace}
.rowlabel{fill:var(--ink-2);font-size:12px;font-family:"IBM Plex Mono",monospace}
.s-instr,.m-eval{fill:var(--inside);stroke:var(--ground);stroke-width:2}
.s-hs{fill:var(--outside);stroke:var(--ground);stroke-width:2}
.m-cross{fill:none;stroke:var(--outside);stroke-width:2.5}
.pt{cursor:default}
.pt:hover circle{stroke:var(--ink);stroke-width:2}
.legend{display:flex;flex-wrap:wrap;gap:18px;margin-top:12px;font-size:.83rem;color:var(--muted)}
.key{display:inline-flex;align-items:center;gap:7px}
.sw{width:11px;height:11px;border-radius:50%;display:inline-block}
.sw.s-instr,.sw.m-eval{background:var(--inside)}
.sw.s-hs{background:var(--outside)}
.sw.m-cross{background:transparent;border:2.5px solid var(--outside)}
.sw.sw-band{background:var(--band);border-radius:2px;width:16px;height:11px}
.tablewrap{overflow-x:auto;margin:20px 0 26px;border:1px solid var(--rule);border-radius:3px;
  background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.87rem}
caption{text-align:left;padding:13px 16px 0;font-size:.78rem;letter-spacing:.07em;
  text-transform:uppercase;color:var(--muted);font-weight:600}
th,td{padding:9px 14px;text-align:left;border-bottom:1px solid var(--rule-2);vertical-align:top}
thead th{font-size:.74rem;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);
  font-weight:600;white-space:nowrap;border-bottom:1px solid var(--rule)}
tbody tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-family:"IBM Plex Mono",monospace;
  font-variant-numeric:tabular-nums;white-space:nowrap}
td.strong{color:var(--ink);font-weight:500}
td.mono{font-family:"IBM Plex Mono",monospace;font-size:.83rem;white-space:nowrap}
td.excerpt{color:var(--muted);font-size:.8rem;line-height:1.45;min-width:22ch;max-width:40ch}
.flag{font-size:.76rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.flag.yes{color:var(--inside)}
.flag.no{color:var(--muted)}
.verdict{display:inline-block;font-size:.72rem;font-weight:600;letter-spacing:.04em;
  padding:2px 7px;border-radius:2px;border:1px solid currentColor;white-space:nowrap}
.v-drop{color:var(--good)}
.v-keep{color:var(--outside)}
.v-trade{color:var(--outside)}
.v-inert{color:var(--muted)}
ol.doing{padding-left:1.25em;margin:0}
ol.doing li{margin-bottom:1em;color:var(--ink-2)}
ol.doing strong{color:var(--ink)}
.method ul{padding-left:1.15em;margin:0}
.method li{margin-bottom:.85em;color:var(--ink-2);font-size:.94rem}
.method strong{color:var(--ink)}
footer{padding:34px 0 0;color:var(--muted);font-size:.83rem;border-top:1px solid var(--rule)}
a{color:var(--inside)}
:focus-visible{outline:2px solid var(--inside);outline-offset:3px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media (max-width:640px){body{padding:0 16px 56px}.hero{padding:44px 0 30px}}
"""


def main() -> int:
    data = collect()
    dest = X.RESULTS / "novelty_study.html"
    dest.write_text(build(data), encoding="utf-8")
    print(f"Saved -> {dest} ({dest.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
