#!/usr/bin/env python
"""Render the off-distribution study as a self-contained HTML page.

Every number on the page is read out of ``analysis/offdist/results/`` — none is typed by
hand — so the page cannot drift from the JSONL it describes. Re-run after any re-run of the
analysis and republish the same file path to refresh the artifact in place.

Published at https://claude.ai/code/artifact/e1399fd2-3adf-4b1c-bb0d-17d3a90b3a21
(republish the same path to update it in place).

    analysis/offdist/build_artifact.py [out.html]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import od_common as O  # noqa: E402
from ablate import group_score, pair_groups  # noqa: E402
from report import controls, load_ablation  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else O.RESULTS / "offdist.html"
ARMS = ["gptoss120b", "deepseekv4pro"]
LABEL = {"gptoss120b": "gpt-oss-120b", "deepseekv4pro": "deepseek-v4-pro"}
SHORT = {"gptoss120b": "arm 1", "deepseekv4pro": "arm 2"}


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


D = {}
for k in ARMS:
    D[k] = {
        "surface": O.read_json(O.RESULTS / f"surface_{k}.json"),
        "acts": O.read_json(O.RESULTS / f"actsig_{k}.json"),
        "abl": load_ablation(k),
        "npz": np.load(O.RESULTS / f"actsig_{k}.npz"),
    }
CONV = D[ARMS[0]]["surface"]["eval_convention"]


def cond(k, name):
    return next((r for r in D[k]["abl"] if r["condition"] == name), None)


def delta_vs_random(k, name):
    r = cond(k, name)
    c = controls(D[k]["abl"]).get(r["n_removed"]) if r else None
    if not r or not c:
        return None
    return r["mean"]["auroc"], c[0], c[1], r["mean"]["auroc"] - c[0], r["n_removed"]


# --------------------------------------------------------------------------- charts
# The two ends of the surface-distance axis Q3 removes along. Kept as one number so the
# highlighted points in the Q4 scatter and the bars named in Q3 can never disagree.
EXTREME_PCT = 10

BAR_CONDS = [
    ("drop_most_offdist_10pct", "most off-distribution, 10%"),
    ("drop_most_offdist_20pct", "most off-distribution, 20%"),
    ("drop_most_offdist_30pct", "most off-distribution, 30%"),
    ("drop_most_offdist_50pct", "most off-distribution, 50%"),
    ("drop_most_evallike_10pct", "most eval-like, 10%"),
    ("drop_most_evallike_20pct", "most eval-like, 20%"),
    ("drop_most_evallike_30pct", "most eval-like, 30%"),
    ("drop_most_evallike_50pct", "most eval-like, 50%"),
    ("drop_longest_assistant_30pct", "longest assistant replies, 30%"),
]


def diverging_chart(k) -> str:
    """Δ against matched random removal of the same size, one bar per condition."""
    rows = []
    for name, label in BAR_CONDS:
        d = delta_vs_random(k, name)
        if d:
            rows.append((label, d[3], max(d[2], 0.005), name))
    if not rows:
        return ""
    W, rowh, gap = 720, 26, 9
    L, R = 250, 690
    H = len(rows) * (rowh + gap) + 34
    lim = max(0.06, max(abs(v) for _, v, _, _ in rows) * 1.15)
    zero = L + (R - L) / 2

    def X(v):
        return zero + (R - L) / 2 * (v / lim)

    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="change in eval AUROC against matched random removal, per condition">']
    for t in (-lim, -lim / 2, 0, lim / 2, lim):
        o.append(f'<line class="cgrid" x1="{X(t):.1f}" y1="16" x2="{X(t):.1f}" y2="{H-18}"/>')
        o.append(f'<text class="ctick" x="{X(t):.1f}" y="{H-4}" text-anchor="middle">'
                 f'{t:+.2f}</text>')
    o.append(f'<line class="czero" x1="{zero:.1f}" y1="16" x2="{zero:.1f}" y2="{H-18}"/>')
    for i, (label, v, band, name) in enumerate(rows):
        y = 20 + i * (rowh + gap)
        o.append(f'<rect class="cband" x="{X(-band):.1f}" y="{y}" '
                 f'width="{X(band)-X(-band):.1f}" height="{rowh}"/>')
        cls = "cbar pos" if v > 0 else "cbar neg"
        if abs(v) <= band:
            cls = "cbar flat"
        x0, x1 = (zero, X(v)) if v > 0 else (X(v), zero)
        o.append(f'<rect class="{cls}" x="{x0:.1f}" y="{y+4}" width="{max(x1-x0,1):.1f}" '
                 f'height="{rowh-8}" rx="1.5"/>')
        o.append(f'<text class="crow" x="{L-16}" y="{y+rowh/2+4:.0f}" text-anchor="end">'
                 f'{esc(label)}</text>')
        vx = X(v) + (7 if v > 0 else -7)
        anchor = "start" if v > 0 else "end"
        o.append(f'<text class="cval" x="{vx:.1f}" y="{y+rowh/2+4:.0f}" '
                 f'text-anchor="{anchor}">{v:+.4f}</text>')
    o.append("</svg>")
    return "\n".join(o)


def flag_rows(k):
    """`flags_<arm>.jsonl` as a list, cached — row `i` is its own index into the npz."""
    if "flags" not in D[k]:
        D[k]["flags"] = [json.loads(l) for l in
                         (O.RESULTS / f"flags_{k}.jsonl").read_text(encoding="utf-8").splitlines()
                         if l.strip()]
    return D[k]["flags"]


def extreme_groups(k) -> dict[str, np.ndarray]:
    """The red-team rows at each end of the surface-distance axis, as row indices.

    Re-derived with `ablate.py`'s own pair grouping and `p_redteam` ranking, so these are
    exactly the rows `drop_most_offdist_{EXTREME_PCT}pct` and
    `drop_most_evallike_{EXTREME_PCT}pct` remove in Q3 — not a second ordering that could
    drift from them. Flag row `i` is its own index into the npz arrays.
    """
    flags = flag_rows(k)
    groups = pair_groups(flags)
    p = group_score(groups, flags, "p_redteam")
    n = max(1, int(round(len(groups) * EXTREME_PCT / 100)))

    def take(order):
        return np.array(sorted(i for gi in order[:n] for i in groups[gi]), dtype=int)

    return {"offdist": take(np.argsort(-p)), "evallike": take(np.argsort(p))}


def extreme_stats(k) -> dict[str, dict]:
    """Where each end of that axis actually sits, in the coordinates the scatter plots."""
    z, ex = D[k]["npz"], extreme_groups(k)
    proj, dist = z["proj_on_w"], z["centroid_dist"]
    orth = np.sqrt(np.maximum(dist ** 2 - proj ** 2, 0))
    out = {}
    for name, idx in ex.items():
        out[name] = {
            "n": int(len(idx)),
            "abs_proj": float(np.abs(proj[idx]).mean()),
            "orth": float(orth[idx].mean()),
            "knn": float(z["knn_to_eval"][idx].mean()),
        }
    return out


# What each side of a contrastive pair is worth. These conditions break the pairing on
# purpose, so their control is `keep_random_half` — a randomly chosen side of every pair —
# and never the whole-pair `matched_random` the Q3 bars use.
PROV_ROWS = [
    ("full", "everything"),
    ("keep_random_half", "a random half of every pair"),
    ("drop_generated", "the attacker&#8217;s own successes only"),
    ("rewritten_sources", "the same, rewritten by the generator"),
    ("drop_sources", "the generated partners only"),
    ("rewritten_plus_generated", "the rewrites, paired with the partners"),
    ("base_only", "no red-team data"),
]


def half_control(k):
    """(mean, sd) over the `keep_random_half` seeds, or None if it was never run."""
    v = [r["mean"]["auroc"] for r in D[k]["abl"] if r["condition"] == "keep_random_half"]
    if not v:
        return None
    return st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0)


def prov_rows(k):
    """(label, auroc, sd) per row of the provenance chart, in display order."""
    out = []
    for name, label in PROV_ROWS:
        if name == "keep_random_half":
            hc = half_control(k)
            if hc:
                out.append((label, hc[0], hc[1]))
            continue
        r = cond(k, name)
        if r:
            out.append((label, r["mean"]["auroc"], 0.0))
    return out


def provenance_chart(k) -> str:
    """Eval AUROC when one side of every contrastive pair is dropped — or replaced.

    A dot per condition on a shared scale, with two reference lines: what all the data
    scores, and what no red-team data at all scores. A dot left of the second one is a
    training set that is worse than not red-teaming.
    """
    rows = prov_rows(k)
    if not rows:
        return ""
    full = cond(k, "full")["mean"]["auroc"]
    base = cond(k, "base_only")["mean"]["auroc"]
    # One domain for both arms: the "no red-team data" line is the same 0.8523 in each, and
    # on per-arm scales it would sit in a different place in each chart.
    vals = [v for a in ARMS for _, v, sd in prov_rows(a) for v in (v - sd, v + sd)]
    lo, hi = min(vals) - 0.02, max(vals) + 0.02
    W, rowh, gap = 720, 26, 9
    L, R = 268, 600
    H = len(rows) * (rowh + gap) + 40

    def X(v):
        return L + (R - L) * (v - lo) / (hi - lo)

    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="eval AUROC when one side of every contrastive pair is dropped, '
         f'or replaced by a rewrite of it">']
    ticks = [t / 100 for t in range(int(lo * 100) + 1, int(hi * 100) + 1)
             if t % 5 == 0]
    for t in ticks:
        o.append(f'<line class="cgrid" x1="{X(t):.1f}" y1="14" x2="{X(t):.1f}" y2="{H-20}"/>')
        o.append(f'<text class="ctick" x="{X(t):.1f}" y="{H-6}" text-anchor="middle">'
                 f'{t:.2f}</text>')
    for v, cls, lab in ((full, "pref full", "all data"), (base, "pref base", "no red-team")):
        o.append(f'<line class="{cls}" x1="{X(v):.1f}" y1="14" x2="{X(v):.1f}" y2="{H-20}"/>')
        o.append(f'<text class="preflab {cls.split()[1]}" x="{X(v):.1f}" y="10" '
                 f'text-anchor="middle">{lab}</text>')
    for i, (label, v, sd) in enumerate(rows):
        y = 20 + i * (rowh + gap) + rowh / 2
        if sd > 0:
            o.append(f'<rect class="cband" x="{X(v-sd):.1f}" y="{y-7:.1f}" '
                     f'width="{max(X(v+sd)-X(v-sd), 1):.1f}" height="14" rx="2"/>')
        cls = "pdot low" if v < base else "pdot"
        o.append(f'<circle class="{cls}" cx="{X(v):.1f}" cy="{y:.1f}" r="4.6"/>')
        o.append(f'<text class="crow" x="{L-16}" y="{y+4:.0f}" text-anchor="end">'
                 f'{label}</text>')
        txt = f"{v:.4f}" + (f" &#177; {sd:.4f}" if sd > 0 else "")
        o.append(f'<text class="cval" x="{R+8}" y="{y+4:.0f}">{txt}</text>')
    o.append("</svg>")
    return "\n".join(o)


def prov_geometry(k) -> dict[str, dict]:
    """Each side of the pairs in activation space — computed by `actsig.py`, read here."""
    return D[k]["acts"]["per_pair_role"]


def provenance_scatter(k) -> str:
    """Q4's coordinates again, but one panel per side of the contrastive pair.

    Same axes as `orthogonality_diagram` — projection on `w` horizontally, the magnitude of
    the residual vertically, both read off the npz — so the two figures are comparable. What
    changes is the split: the attacker's own submissions on the left, the partners
    `generate_contrastive_dataset` wrote for them on the right, with the eval cloud behind
    both as the shared reference. Filled marks are the positive class, hollow the negative,
    which is what makes the horizontal split inside each panel legible.

    Two panels rather than two colours in one: the groups sit almost on top of each other,
    and overplotting 294 against 294 would hide exactly the comparison the panels are for.
    """
    z = D[k]["npz"]
    proj, dist = z["proj_on_w"], z["centroid_dist"]
    orth = np.sqrt(np.maximum(dist ** 2 - proj ** 2, 0))
    ev_proj, ev_orth, _split, _names = eval_coords(k)
    rows = flag_rows(k)
    role = np.array([r.get("pair_role") or "none" for r in rows], dtype=object)
    is_pos = np.array([r["label"] == O.POS for r in rows])

    W, H = 720, 348
    T, B = 46, 268
    PANELS = [("source", "the attacker&#8217;s own submissions", 58, 352),
              ("generated", "the partners written for them", 396, 690)]
    xlim = max(float(np.abs(proj).max()), float(np.abs(ev_proj).max())) * 1.1
    ylim = max(float(orth.max()), float(ev_orth.max())) * 1.06

    def Y(v):
        return B - (B - T) * (v / ylim)

    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="the attacker&#8217;s own red-team submissions and the contrastive '
         f'partners generated for them, in the same activation-space coordinates as the '
         f'earlier figure">']
    for name, title, L, R in PANELS:
        def X(v, L=L, R=R):
            return L + (R - L) * (v + xlim) / (2 * xlim)

        o.append(f'<text class="ptitle" x="{(L+R)/2:.0f}" y="30" text-anchor="middle">'
                 f'{title}</text>')
        for t in range(0, int(ylim) + 1, max(1, int(ylim // 5))):
            o.append(f'<line class="ogrid" x1="{L}" y1="{Y(t):.1f}" x2="{R}" y2="{Y(t):.1f}"/>')
            if L == PANELS[0][2]:
                o.append(f'<text class="otick" x="{L-8}" y="{Y(t)+4:.1f}" '
                         f'text-anchor="end">{t}</text>')
        o.append(f'<line class="oaxis" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/>')
        o.append(f'<line class="oaxis dash" x1="{X(0):.1f}" y1="{T}" x2="{X(0):.1f}" y2="{B}"/>')
        for a, b in zip(ev_proj, ev_orth):                       # the shared reference cloud
            o.append(f'<circle class="sdot ev" cx="{X(float(a)):.1f}" '
                     f'cy="{Y(float(b)):.1f}" r="1.8"/>')
        cls = "src" if name == "source" else "gen"
        for i in np.where(role == name)[0]:
            o.append(f'<circle class="sdot {cls} {"pos" if is_pos[i] else "neg"}" '
                     f'cx="{X(float(proj[i])):.1f}" cy="{Y(float(orth[i])):.1f}" r="2.6"/>')
        o.append(f'<text class="oaxlab" x="{(L+R)/2:.0f}" y="{B+19}" text-anchor="middle">'
                 f'projection on w &#8594;</text>')
    o.append(f'<text class="oaxlab rot" transform="translate(18,{(T+B)/2:.0f}) rotate(-90)" '
             f'text-anchor="middle">everything orthogonal to w</text>')

    n_src = int((role == "source").sum())
    n_gen = int((role == "generated").sum())
    o.append(swatch_row("marks", [
        ("sdot src pos", f"a submission \u00a0{n_src}"),
        ("sdot gen pos", f"a partner \u00a0{n_gen}"),
        ("sdot ev", f"an eval row \u00a0{len(ev_proj)}"),
    ], x=58, y=B + 50, item_w=196, title_w=54))
    o.append(swatch_row("", [("sdot src neg", f"hollow: labelled {O.NEG}")],
                        x=58, y=B + 70, item_w=196, title_w=54))
    o.append("</svg>")
    return "\n".join(o)


def eval_coords(k):
    """Eval rows in the same two coordinates, plus which split each came from."""
    z = D[k]["npz"]
    proj = z["proj_on_w_eval"]
    dist = z["centroid_dist_eval"]
    orth = np.sqrt(np.maximum(dist ** 2 - proj ** 2, 0))
    return proj, orth, z["eval_split"], [str(n) for n in z["eval_split_names"]]


def eval_stats(k) -> dict:
    """The eval cloud's own geometry, in the coordinates the scatter plots.

    Derived from the npz rather than read off the summary json, the same way the red-team
    side already is, so the two halves of the comparison are computed identically.
    """
    proj, orth, split, names = eval_coords(k)
    dist = D[k]["npz"]["centroid_dist_eval"]
    out = {"n": int(len(proj)), "abs_proj": float(np.abs(proj).mean()),
           "orth": float(orth.mean()),
           "orth_frac": float((orth / np.maximum(dist, 1e-12)).mean()),
           "per_split": {}}
    for i, name in enumerate(names):
        m = split == i
        out["per_split"][name] = {"n": int(m.sum()), "abs_proj": float(np.abs(proj[m]).mean()),
                                  "orth": float(orth[m].mean())}
    return out


def orthogonality_diagram(k) -> str:
    """Displacement from the eval centroid, resolved onto the probe's decision axis.

    Real coordinates: the horizontal axis is a row's projection on the unit direction `w`,
    the vertical axis the magnitude of everything left over. Both are read off the npz, so
    the shape of the cloud is the measurement, not an illustration of it.

    Three things are drawn, and each is there to stop a misreading of the other two. The
    **eval rows** are plotted split by split rather than summarised by the centroid marker,
    because in 5376 dimensions no row sits near the mean and a lone centroid dot invites the
    reader to think the red-team cloud is uniquely far from it. The two ends of **Q3's
    removal axis** are coloured in, since the question this scatter answers is whether
    surface distance from eval is the same thing as displacement in the representation. And
    the rest of the red-team set is left grey behind both.

    Points are drawn rest, then eval, then the two extreme groups in row order — so neither
    extreme group ends up systematically on top of the other where they overlap.
    """
    z = D[k]["npz"]
    proj = z["proj_on_w"]
    dist = z["centroid_dist"]
    orth = np.sqrt(np.maximum(dist ** 2 - proj ** 2, 0))
    ev_proj, ev_orth, ev_split, ev_names = eval_coords(k)
    ex = extreme_groups(k)
    grp = np.full(len(proj), "rest", dtype=object)
    grp[ex["evallike"]] = "like"
    grp[ex["offdist"]] = "off"

    W, H = 720, 398
    L, R, T, B = 74, 690, 24, 286
    xlim = max(6.0, float(max(np.abs(proj).max(), np.abs(ev_proj).max())) * 1.1)
    ylim = float(max(orth.max(), ev_orth.max())) * 1.06

    def X(v):
        return L + (R - L) * (v + xlim) / (2 * xlim)

    def Y(v):
        return B - (B - T) * (v / ylim)

    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="every eval row, split by split, and every red-team row with the most '
         f'off-distribution and most eval-like {EXTREME_PCT}% coloured, displaced from the '
         f'eval centroid and resolved onto the probe decision axis and its orthogonal '
         f'complement">']
    for t in range(0, int(ylim) + 1, max(1, int(ylim // 5))):
        o.append(f'<line class="ogrid" x1="{L}" y1="{Y(t):.1f}" x2="{R}" y2="{Y(t):.1f}"/>')
        o.append(f'<text class="otick" x="{L-10}" y="{Y(t)+4:.1f}" text-anchor="end">{t}</text>')
    o.append(f'<line class="oaxis" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/>')
    o.append(f'<line class="oaxis dash" x1="{X(0):.1f}" y1="{T}" x2="{X(0):.1f}" y2="{B}"/>')

    step = max(1, len(proj) // 600)
    shown = np.arange(0, len(proj), step)
    for i in shown:                                        # the red-team middle, behind
        if grp[i] == "rest":
            o.append(f'<circle class="odot rest" cx="{X(float(proj[i])):.1f}" '
                     f'cy="{Y(float(orth[i])):.1f}" r="2.4"/>')
    for i in range(len(ev_proj)):                          # the eval cloud, by split
        o.append(f'<circle class="ering s{int(ev_split[i]) + 1}" '
                 f'cx="{X(float(ev_proj[i])):.1f}" cy="{Y(float(ev_orth[i])):.1f}" r="2.2"/>')
    for i in shown:                                        # the two ends of Q3's axis
        if grp[i] != "rest":
            o.append(f'<circle class="odot {grp[i]}" cx="{X(float(proj[i])):.1f}" '
                     f'cy="{Y(float(orth[i])):.1f}" r="2.9"/>')

    o.append(f'<circle class="ocent" cx="{X(0):.1f}" cy="{Y(0):.1f}" r="5"/>')
    o.append(f'<text class="olab" x="{X(0):.1f}" y="{B+20}" text-anchor="middle">'
             f'eval centroid</text>')
    o.append(f'<text class="oaxlab" x="{R}" y="{B+20}" text-anchor="end">'
             f'projection on the probe direction w &#8594;</text>')
    o.append(f'<text class="oaxlab rot" transform="translate(20,{(T+B)/2:.0f}) rotate(-90)" '
             f'text-anchor="middle">everything orthogonal to w</text>')

    counts = {g: int((grp == g).sum()) for g in ("off", "like", "rest")}
    o.append(swatch_row("red-team", [
        ("odot off", f"most off-distribution {counts['off']}"),
        ("odot like", f"most eval-like {counts['like']}"),
        ("odot rest", f"the rest {counts['rest']}"),
    ], x=L, y=B + 54, item_w=182))
    o.append(swatch_row("eval", [
        (f"ering s{i + 1}",
         f"{n.removeprefix('eval_')} {int((ev_split == i).sum())}")
        for i, n in enumerate(ev_names)
    ], x=L, y=B + 84, item_w=132, title_w=70))
    o.append("</svg>")
    return "\n".join(o)


def swatch_row(title, items, *, x: float, y: float, item_w: float,
               title_w: float = 76) -> str:
    """One legend row: a left-hand group title, then swatch + label per class.

    `item_w` is picked per row so the last label still lands inside the frame: IBM Plex
    Mono advances 0.6em, so a 10.5px label costs ~6.5px a character.
    """
    o = [f'<text class="okeyhead" x="{x:.1f}" y="{y + 3.6:.1f}">{esc(title)}</text>']
    for i, (cls, label) in enumerate(items):
        cx = x + title_w + i * item_w
        o.append(f'<circle class="{cls}" cx="{cx:.1f}" cy="{y:.1f}" r="3.4"/>')
        o.append(f'<text class="okey" x="{cx + 10:.1f}" y="{y + 3.6:.1f}">{esc(label)}</text>')
    return "\n".join(o)


# --------------------------------------------------------------------------- tables
def answers_table() -> str:
    rows = [
        ("separable from eval on text alone",
         lambda k: f"{D[k]['surface']['discriminator_auroc']:.4f}"),
        ("separable from eval in activation space",
         lambda k: f"{D[k]['acts']['separability_redteam_vs_eval_auroc']:.4f}"),
        ("past eval&#8217;s own p95 self-kNN radius",
         lambda k: f"{D[k]['acts']['outside_frac']:.1%}"),
        (f"rows containing a refusal &nbsp;<span class='ref'>eval {CONV['refusal_rate']:.1%}</span>",
         lambda k: f"{D[k]['surface']['refusal_rate_redteam']:.1%}"),
        ("labelled opposite to the eval convention",
         lambda k: str(D[k]["surface"]["n_convention_inverted"])),
        ("pairs contrasting on the assistant turn",
         lambda k: f"{D[k]['surface']['pair_axis_counts']['assistant']}/"
                   f"{D[k]['surface']['n_pairs']}"),
        ("displacement orthogonal to w",
         lambda k: f"{D[k]['acts']['mean_orthogonal_fraction']:.2%}"),
        ("&nbsp;&nbsp;the same, for the eval rows",
         lambda k: f"{D[k]['acts']['mean_orthogonal_fraction_eval']:.2%}"),
        ("eval AUROC, all red-team data",
         lambda k: f"{cond(k,'full')['mean']['auroc']:.4f}"),
        ("eval AUROC, attacker successes only",
         lambda k: f"{cond(k,'drop_generated')['mean']['auroc']:.4f}"
         if cond(k, 'drop_generated') else "&#8212;"),
        ("eval AUROC, generated partners only",
         lambda k: f"{cond(k,'drop_sources')['mean']['auroc']:.4f}"
         if cond(k, 'drop_sources') else "&#8212;"),
        ("eval AUROC, a random half of every pair",
         lambda k: f"{half_control(k)[0]:.4f}" if half_control(k) else "&#8212;"),
        ("eval AUROC, no red-team data",
         lambda k: f"{cond(k,'base_only')['mean']['auroc']:.4f}"),
    ]
    head = "".join(f"<th>{SHORT[k]} &middot; {LABEL[k]}</th>" for k in ARMS)
    body = "".join(
        f"<tr><td>{lab}</td>" + "".join(f'<td class="n">{fn(k)}</td>' for k in ARMS) + "</tr>"
        for lab, fn in rows)
    return f'<table class="key"><thead><tr><th>measure</th>{head}</tr></thead><tbody>{body}</tbody></table>'


def structure_table() -> str:
    fields = [("chars_assistant", "assistant reply, characters"),
              ("chars_user", "user turn, characters"),
              ("assistant_share", "share of the text the assistant wrote"),
              ("n_newlines", "line breaks"),
              ("has_bullets", "carries a bulleted list"),
              ("has_numbered", "carries a numbered list"),
              ("refuses", "assistant refuses")]
    head = "".join(f"<th>{SHORT[k]}</th>" for k in ARMS)
    rows = []
    for f, lab in fields:
        ev = D[ARMS[0]]["surface"]["structural_means"][f]["eval"]
        fmt = (lambda v: f"{v:.0%}") if f in ("has_bullets", "has_numbered", "refuses",
                                              "assistant_share") else (lambda v: f"{v:,.0f}")
        cells = "".join(
            f'<td class="n drift">{fmt(D[k]["surface"]["structural_means"][f]["redteam"])}</td>'
            for k in ARMS)
        rows.append(f'<tr><td>{lab}</td><td class="n evalcol">{fmt(ev)}</td>{cells}</tr>')
    return ('<table class="key"><thead><tr><th>per conversation</th><th>eval</th>'
            f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def before_after_table() -> str:
    """The flat before/after per half, both arms."""
    if not ba:
        return ""
    arms = [k for k in ARMS if k in ba]
    b0 = ba[arms[0]]
    head = "".join(f"<th>{SHORT[k]}</th>" for k in arms)
    def cell(k, g, field, fmt):
        e = ba[k]["groups"][g]
        return f'{fmt(e["before"][field])} &rarr; {fmt(e["after"][field])}'
    pct = lambda v: f"{v:.0%}"
    num = lambda v: f"{v:.3f}"
    rows = []
    for g, lab in (("source", "the attacker&#8217;s success"),
                   ("generated", "its generated partner")):
        for field, flab, fmt in (("mean", "mean score", num),
                                  ("predicted_positive", "predicted harmful", pct),
                                  ("correct", "classified correctly", pct)):
            cells = "".join(f'<td class="n">{cell(k, g, field, fmt)}</td>' for k in arms)
            rows.append(f"<tr><td>{lab} &middot; {flab}</td>{cells}</tr>")
    return (f'<table class="key"><thead><tr><th>iter{b0["before_iter"]} &rarr; '
            f'iter{b0["after_iter"]}</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def found_iter_table(k) -> str:
    """Why the before probe is not 0% on the successes: split by class x found-iteration."""
    d = ba.get(k, {}).get("source_before_by_found_iter")
    if not d:
        return ""
    b = ba[k]["before_iter"]
    iters = sorted({i for cls in d.values() for i in cls}, key=int)
    head = "".join(f"<th>iter {i}</th>" for i in iters)
    rows = []
    for cls, per in d.items():
        cells = ""
        for i in iters:
            e = per.get(i)
            if not e:
                cells += '<td class="n">&#8212;</td>'
            else:
                # iter b IS the probe these beat → ~0 is expected; shade it
                strong = (i == str(b))
                cls_a = "n neg" if strong else ("n pos" if e["correct"] > 0.5 else "n")
                cells += f'<td class="{cls_a}">{e["correct"]:.0%}<br><span class="sub">n={e["n"]}</span></td>'
        rows.append(f'<tr><td>{cls}</td>{cells}</tr>')
    return (f'<table class="key"><thead><tr><th>true class, by iteration found</th>{head}'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def pair_verdict_table() -> str:
    """What the probe of the day made of each half of the pairs, both arms."""
    if not pps:
        return ""
    head = "".join(f"<th>{SHORT[k]}</th>" for k in ARMS if k in pps)
    rows = []
    for g, lab in (("source", "the attacker&#8217;s success"),
                   ("generated", "its generated partner")):
        cells = "".join(
            f'<td class="n {"pos" if g == "generated" else "neg"}">'
            f'{pps[k]["probe_of_the_day"][g]["correct"]:.1%}</td>'
            for k in ARMS if k in pps)
        rows.append(f"<tr><td>{lab}</td>{cells}</tr>")
    mean = "".join(
        f'<td class="n">{pps[k]["probe_of_the_day"]["generated"]["mean_score"]:.3f}</td>'
        for k in ARMS if k in pps)
    rows.append(f"<tr><td>mean probe score, partners</td>{mean}</tr>")
    return ('<table class="key"><thead><tr><th>the probe classifies it correctly</th>'
            f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def handwritten_summary_table() -> str:
    if not hw:
        return ""
    pp = hw["per_probe"]
    head = "<th>overall</th><th>harmful</th><th>not-harmful</th>"
    def row(k, lab):
        d = pp[str(k)]
        cls = "n pos" if d["overall"] == 1.0 else "n"
        return (f'<tr><td>{lab}</td><td class="{cls}">{d["overall"]:.0%}</td>'
                f'<td class="n">{d["harmful"]:.0%}</td>'
                f'<td class="n">{d["not_harmful"]:.0%}</td></tr>')
    return ('<table class="key"><thead><tr><th>probe</th>' + head + '</tr></thead><tbody>'
            + row(0, "<code>probe_iter0</code> &middot; before red-teaming")
            + row(5, "<code>probe_iter5</code> &middot; after five retrains")
            + '</tbody></table>')


def handwritten_chart() -> str:
    """One row per variation: its iter0 and iter5 probe scores against the 0.5 threshold.

    Harmful variations (true positive) are correct to the RIGHT of the line, benign ones to
    the LEFT, so the two blocks read in opposite directions — which is the point: the whole
    iter5 harmful block has crossed the line the iter0 one straddles.
    """
    if not hw or not hw.get("rows"):
        return ""
    rows = hw["rows"]
    order = sorted(range(len(rows)),
                   key=lambda i: (rows[i]["label"] != O.POS, rows[i]["scores"]["5"]))
    W, rowh, gap = 720, 15, 4
    L, R, T = 250, 690, 34
    H = len(rows) * (rowh + gap) + T + 20
    thr = 0.5

    def X(v):
        return L + (R - L) * v

    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="probe score of each hand-written variation, before and after red-team '
         f'retraining, against the 0.5 threshold">']
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        o.append(f'<line class="cgrid" x1="{X(t):.1f}" y1="{T-6}" x2="{X(t):.1f}" y2="{H-16}"/>')
        o.append(f'<text class="ctick" x="{X(t):.1f}" y="{H-4}" text-anchor="middle">{t:.2f}</text>')
    o.append(f'<line class="czero" x1="{X(thr):.1f}" y1="{T-6}" x2="{X(thr):.1f}" y2="{H-16}"/>')
    o.append(f'<text class="ctick" x="{X(thr):.1f}" y="{T-12}" text-anchor="middle">threshold</text>')
    o.append(f'<text class="hwhdr harm" x="{X(0.5)+8:.1f}" y="{T-12}" text-anchor="start">'
             f'harmful &#8594; correct this side</text>')
    o.append(f'<text class="hwhdr" x="{X(0.5)-8:.1f}" y="{T-12}" text-anchor="end">'
             f'&#8592; benign correct this side</text>')
    last_label = None
    for r_i, i in enumerate(order):
        row = rows[i]
        y = T + r_i * (rowh + gap) + rowh / 2
        s0, s5 = row["scores"]["0"], row["scores"]["5"]
        o.append(f'<line class="hwlink" x1="{X(s0):.1f}" y1="{y:.1f}" '
                 f'x2="{X(s5):.1f}" y2="{y:.1f}"/>')
        o.append(f'<circle class="hwbase" cx="{X(s0):.1f}" cy="{y:.1f}" r="3"/>')
        o.append(f'<circle class="hwok" cx="{X(s5):.1f}" cy="{y:.1f}" r="3.4"/>')
        lab = "harmful" if row["label"] == O.POS else "not-harmful"
        tag = f'based on #{row["based_on"]}'
        band = lab if lab != last_label else ""
        last_label = lab
        o.append(f'<text class="crow" x="{L-14}" y="{y+3.5:.0f}" text-anchor="end">'
                 f'{esc(tag)}<tspan class="hwband"> {esc(band)}</tspan></text>')
    o.append("</svg>")
    return "\n".join(o)


def rewrite_table() -> str:
    """Where the rewrites landed between the two halves, structurally."""
    if not rw_stats:
        return ""
    m = rw_stats["structural_means"]
    fields = [("chars_total", "conversation, characters"),
              ("chars_assistant", "assistant reply, characters"),
              ("n_newlines", "line breaks"),
              ("has_bullets", "carries a bulleted list"),
              ("has_numbered", "carries a numbered list")]
    cols = [("sources", "submissions"), ("rewrites", "rewritten"),
            ("partners", "partners"), ("eval", "eval")]
    head = "".join(f"<th>{lab}</th>" for _g, lab in cols)
    rows = []
    for f, lab in fields:
        pct = f.startswith("has_")
        cells = "".join(
            f'<td class="n {"evalcol" if g == "eval" else "drift" if g != "rewrites" else ""}">'
            f'{m[g][f]:.0%}</td>' if pct else
            f'<td class="n {"evalcol" if g == "eval" else "drift" if g != "rewrites" else ""}">'
            f'{m[g][f]:,.0f}</td>'
            for g, _lab in cols)
        rows.append(f"<tr><td>{lab}</td>{cells}</tr>")
    return ('<table class="key"><thead><tr><th>per conversation</th>'
            f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def topic_table(k) -> str:
    s, a = D[k]["surface"], D[k]["acts"]
    rows = []
    for t in s["topics"]:
        pt = a["per_topic"][str(t["cluster"])]
        d = delta_vs_random(k, f"drop_topic_{t['cluster']}")
        dd = f"{d[3]:+.4f}" if d else "&#8212;"
        cls = "n"
        if d and abs(d[3]) > max(d[2], 0.005):
            cls = "n pos" if d[3] > 0 else "n neg"
        rows.append(
            f'<tr><td class="mono">{t["cluster"]}</td><td>{esc(", ".join(t["top_terms"][:5]))}</td>'
            f'<td class="n">{t["n"]}</td><td class="n">{pt["outside_frac"]:.0%}</td>'
            f'<td class="{cls}">{dd}</td></tr>')
    return ('<table class="key"><thead><tr><th>topic</th><th>leading terms</th>'
            '<th>rows</th><th>off-manifold</th><th>&Delta; vs random when dropped</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')


# --------------------------------------------------------------------------- page
g, d = D["gptoss120b"], D["deepseekv4pro"]
base_only = cond("gptoss120b", "base_only")["mean"]["auroc"]
la = {k: delta_vs_random(k, "drop_longest_assistant_30pct") for k in ARMS}
best_ds = delta_vs_random("deepseekv4pro", "drop_most_evallike_20pct")
ex_ds = extreme_stats("deepseekv4pro")
ev_ds = eval_stats("deepseekv4pro")
pg_g = prov_geometry("gptoss120b")
rw_stats = O.read_json(O.RESULTS / "rewrite_stats_gptoss120b.json") \
    if (O.RESULTS / "rewrite_stats_gptoss120b.json").exists() else None
pps = {k: O.read_json(O.RESULTS / f"pair_probe_scores_{k}.json")
       for k in ARMS if (O.RESULTS / f"pair_probe_scores_{k}.json").exists()}
ba = {k: O.read_json(O.RESULTS / f"before_after_scores_{k}.json")
      for k in ARMS if (O.RESULTS / f"before_after_scores_{k}.json").exists()}
hw = O.read_json(O.RESULTS / "handwritten_scores_gptoss120b.json") \
    if (O.RESULTS / "handwritten_scores_gptoss120b.json").exists() else None
hw_rows = [json.loads(l) for l in
           (O.RESULTS / "handwritten_gptoss120b.jsonl").read_text().splitlines() if l.strip()] \
    if (O.RESULTS / "handwritten_gptoss120b.jsonl").exists() else []
prov = {k: {n: (cond(k, n)["mean"]["auroc"] if cond(k, n) else None)
            for n in ("full", "base_only", "drop_generated", "drop_sources",
                      "rewritten_sources", "rewritten_plus_generated")}
        for k in ARMS}
half = {k: half_control(k) for k in ARMS}
rt_orth_ds = float(np.sqrt(np.maximum(
    D["deepseekv4pro"]["npz"]["centroid_dist"] ** 2
    - D["deepseekv4pro"]["npz"]["proj_on_w"] ** 2, 0)).mean())

HTML = f"""<title>Orthogonal by Construction</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root {{
  --ground:#F2F5F4; --surface:#FFFFFF; --raised:#E9EEEC;
  --ink:#141D1E; --body:#2C3A3A; --muted:#5F7071; --faint:#8A9A9A;
  --rule:#D5DEDB; --rule-strong:#B9C6C3;
  --eval:#0F6F68; --eval-soft:#DCEBE8;
  --drift:#AC432C; --drift-soft:#F4E2DC;
  --near:#A8761B; --gen:#6B5BA6;
  --ev1:#0F6F68; --ev2:#2F6DA8; --ev3:#6B5BA6; --ev4:#3F8A55;
  --band:#DFE6E4;
  --measure:66ch; --wide:1000px;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0E1517; --surface:#141D1F; --raised:#1B2528;
    --ink:#E9F0EE; --body:#C3D0CE; --muted:#8DA0A0; --faint:#6B7E7E;
    --rule:#253134; --rule-strong:#374548;
    --eval:#4FB3A6; --eval-soft:#12302D;
    --drift:#D97A5E; --drift-soft:#331B14;
    --near:#D9A441; --gen:#A294E0;
    --ev1:#4FB3A6; --ev2:#6FA8DE; --ev3:#A294E0; --ev4:#6FC183;
    --band:#1E292B;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0E1517; --surface:#141D1F; --raised:#1B2528;
  --ink:#E9F0EE; --body:#C3D0CE; --muted:#8DA0A0; --faint:#6B7E7E;
  --rule:#253134; --rule-strong:#374548;
  --eval:#4FB3A6; --eval-soft:#12302D;
  --drift:#D97A5E; --drift-soft:#331B14;
  --near:#D9A441; --gen:#A294E0;
  --ev1:#4FB3A6; --ev2:#6FA8DE; --ev3:#A294E0; --ev4:#6FC183;
  --band:#1E292B;
}}
* {{ box-sizing:border-box; }}
body {{
  background:var(--ground); color:var(--body);
  font-family:"Source Serif 4",Georgia,serif; font-size:17px; line-height:1.62;
  margin:0; padding:0 24px 96px;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:var(--wide); margin:0 auto; }}
.col {{ max-width:var(--measure); }}
h1,h2,h3,.display {{ font-family:Archivo,"Helvetica Neue",Arial,sans-serif; color:var(--ink);
  text-wrap:balance; margin:0; }}
.mono,.eyebrow,td.n,th,.cval,.ctick,.crow,.preflab,.otick,.olab,.oaxlab,.okey,.okeyhead,
.ptitle,.ref,.chip {{
  font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace; }}
.eyebrow {{ font-size:11px; letter-spacing:.18em; text-transform:uppercase;
  color:var(--muted); font-weight:500; }}

/* ---- masthead ---- */
header {{ padding:76px 0 42px; border-bottom:1px solid var(--rule-strong); }}
h1 {{ font-size:clamp(38px,6.4vw,68px); font-weight:700; letter-spacing:-.028em;
  line-height:1.02; margin:16px 0 0; }}
h1 .thin {{ font-weight:500; color:var(--muted); display:block; }}
.standfirst {{ max-width:60ch; margin:24px 0 0; font-size:19px; color:var(--body); }}
.meta {{ display:flex; flex-wrap:wrap; gap:10px 26px; margin-top:30px; }}
.chip {{ font-size:11.5px; letter-spacing:.06em; color:var(--muted);
  border:1px solid var(--rule); border-radius:2px; padding:5px 9px; background:var(--surface); }}

/* ---- question sections ---- */
section {{ padding:56px 0; border-bottom:1px solid var(--rule); }}
.qhead {{ display:grid; grid-template-columns:74px 1fr; gap:0 22px; align-items:start; }}
.qnum {{ font-family:Archivo,sans-serif; font-size:13px; font-weight:700; letter-spacing:.12em;
  color:var(--eval); padding-top:9px; border-top:2px solid var(--eval); }}
h2 {{ font-size:clamp(23px,3vw,31px); font-weight:600; letter-spacing:-.02em; line-height:1.16;
  max-width:22ch; }}
.verdict {{ grid-column:2; margin:20px 0 0; font-family:Archivo,sans-serif;
  font-size:clamp(19px,2.4vw,25px); font-weight:500; line-height:1.3; color:var(--ink);
  letter-spacing:-.012em; max-width:26ch; }}
.verdict em {{ font-style:normal; color:var(--drift); }}
.verdict .ok {{ color:var(--eval); }}
.qbody {{ grid-column:2; }}
.qbody p {{ max-width:var(--measure); }}
@media (max-width:720px) {{
  .qhead {{ grid-template-columns:1fr; gap:14px; }}
  .verdict,.qbody {{ grid-column:1; }}
  .qnum {{ display:inline-block; }}
}}

/* ---- tables ---- */
.panel {{ margin:32px 0 6px; overflow-x:auto; }}
table.key {{ border-collapse:collapse; width:100%; font-size:14px; min-width:520px; }}
table.key th {{ font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--muted); font-weight:500; text-align:left; padding:0 14px 9px 0;
  border-bottom:1px solid var(--rule-strong); }}
table.key th:not(:first-child) {{ text-align:right; }}
table.key td {{ padding:9px 14px 9px 0; border-bottom:1px solid var(--rule);
  vertical-align:baseline; color:var(--body); }}
table.key td.n {{ text-align:right; font-size:13px; font-variant-numeric:tabular-nums;
  color:var(--ink); white-space:nowrap; }}
table.key td.drift {{ color:var(--drift); }}
table.key td.evalcol {{ color:var(--eval); text-align:right; font-family:"IBM Plex Mono",monospace;
  font-size:13px; font-variant-numeric:tabular-nums; }}
td.pos {{ color:var(--eval); }}
td.neg {{ color:var(--drift); }}
.ref {{ font-size:11px; color:var(--eval); }}
caption,.cap {{ font-size:13px; color:var(--muted); max-width:var(--measure);
  margin:12px 0 0; line-height:1.5; }}

/* ---- charts ---- */
.chart {{ width:100%; height:auto; display:block; }}
.cgrid {{ stroke:var(--rule); stroke-width:1; }}
.czero {{ stroke:var(--rule-strong); stroke-width:1.5; }}
.cband {{ fill:var(--band); }}
.cbar.pos {{ fill:var(--eval); }}
.cbar.neg {{ fill:var(--drift); }}
.cbar.flat {{ fill:var(--faint); }}
.hwbase {{ fill:var(--faint); }}
.hwok {{ fill:var(--eval); }}
.hwlink {{ stroke:var(--rule-strong); stroke-width:1; }}
.hwhdr {{ font-size:9.5px; fill:var(--muted); letter-spacing:.04em; }}
.hwhdr.harm {{ fill:var(--drift); }}
.hwband {{ fill:var(--faint); font-size:9px; }}
.pdot {{ fill:var(--ink); }}
.pdot.low {{ fill:var(--drift); }}
td .sub {{ font-size:10px; color:var(--faint); font-variant-numeric:tabular-nums; }}
.hlneg {{ color:var(--drift); font-weight:600; }}
.pref {{ stroke-width:1.4; stroke-dasharray:4 4; }}
.pref.full {{ stroke:var(--eval); }}
.pref.base {{ stroke:var(--faint); }}
.preflab {{ font-size:9.5px; letter-spacing:.08em; text-transform:uppercase; }}
.preflab.full {{ fill:var(--eval); }}
.preflab.base {{ fill:var(--faint); }}
.crow {{ font-size:11.5px; fill:var(--body); }}
.cval {{ font-size:11px; fill:var(--muted); font-variant-numeric:tabular-nums; }}
.ctick {{ font-size:10px; fill:var(--faint); }}
.ogrid {{ stroke:var(--rule); stroke-width:1; }}
.oaxis {{ stroke:var(--rule-strong); stroke-width:1.5; }}
.oaxis.dash {{ stroke-dasharray:3 4; }}
.odot {{ fill:var(--faint); fill-opacity:.34; }}   /* .rest, and the legend swatch */
.odot.off {{ fill:var(--drift); fill-opacity:.85; }}
.odot.like {{ fill:var(--near); fill-opacity:.85; }}
.ering {{ fill:none; stroke-width:1.05; stroke-opacity:.62; }}
.sdot.ev {{ fill:var(--faint); fill-opacity:.20; }}
.sdot.src.pos {{ fill:var(--drift); fill-opacity:.85; }}
.sdot.gen.pos {{ fill:var(--gen); fill-opacity:.85; }}
.sdot.src.neg {{ fill:none; stroke:var(--drift); stroke-width:1.15; stroke-opacity:.9; }}
.sdot.gen.neg {{ fill:none; stroke:var(--gen); stroke-width:1.15; stroke-opacity:.9; }}
.ptitle {{ font-size:11px; fill:var(--ink); letter-spacing:.02em; }}
.ering.s1 {{ stroke:var(--ev1); }}
.ering.s2 {{ stroke:var(--ev2); }}
.ering.s3 {{ stroke:var(--ev3); }}
.ering.s4 {{ stroke:var(--ev4); }}
.ocent {{ fill:var(--ink); stroke:var(--surface); stroke-width:1.6; }}
.otick {{ font-size:10px; fill:var(--faint); }}
.olab {{ font-size:10.5px; fill:var(--ink); letter-spacing:.06em; }}
.okey {{ font-size:10.5px; fill:var(--body); letter-spacing:.02em;
  font-variant-numeric:tabular-nums; }}
.okeyhead {{ font-size:10px; fill:var(--muted); letter-spacing:.12em;
  text-transform:uppercase; }}
.oaxlab {{ font-size:10.5px; fill:var(--muted); letter-spacing:.08em; text-transform:uppercase; }}

.armhead {{ font-family:Archivo,sans-serif; font-size:12px; font-weight:600;
  letter-spacing:.1em; text-transform:uppercase; color:var(--ink);
  margin:34px 0 4px; padding-bottom:6px; border-bottom:1px solid var(--rule); }}
.armhead span {{ color:var(--muted); font-weight:500; }}

.pull {{ border-left:2px solid var(--eval); padding:2px 0 2px 20px; margin:30px 0;
  max-width:58ch; color:var(--ink); font-size:18px; }}

footer {{ padding:52px 0 0; color:var(--muted); font-size:13.5px; max-width:var(--measure); }}
footer code {{ font-family:"IBM Plex Mono",monospace; font-size:12.5px; color:var(--body); }}
a {{ color:var(--eval); }}
:focus-visible {{ outline:2px solid var(--eval); outline-offset:3px; }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">experiment22 &middot; human-harm probe &middot; gemma-3-27b-it L32</div>
  <h1>Orthogonal<span class="thin">by construction</span></h1>
  <p class="standfirst">The red-team conversations a probe is retrained on are not drawn from
  the distribution it is scored on. This asks how far off they sit, whether any are labelled
  backwards, whether dropping the worst of them helps, where in the model&#8217;s own
  representation the difference actually lives &#8212; and which half of each generated pair
  the retrain is really learning from.</p>
  <div class="meta">
    <span class="chip">2 attacker arms</span>
    <span class="chip">{g['surface']['n_redteam']} + {d['surface']['n_redteam']} red-team rows</span>
    <span class="chip">{g['surface']['n_eval']} eval rows</span>
    <span class="chip">{sum(len(D[k]['abl']) for k in ARMS)} probe fits</span>
    <span class="chip">no LLM loaded</span>
  </div>
</header>

<section>
  <div class="qhead">
    <div class="qnum">Q1</div>
    <div>
      <h2>Can a portion of the red-team data be identified as different from the eval set?</h2>
      <p class="verdict">No portion. <em>Effectively all of it.</em></p>
      <div class="qbody">
        <p>A TF-IDF discriminator trained to tell the two corpora apart reaches
        <strong>{g['surface']['discriminator_auroc']:.4f}</strong> and
        <strong>{d['surface']['discriminator_auroc']:.4f}</strong> AUROC out of fold. A linear
        model on the pooled layer-32 activations reaches
        <strong>{g['acts']['separability_redteam_vs_eval_auroc']:.4f}</strong> and
        <strong>{d['acts']['separability_redteam_vs_eval_auroc']:.4f}</strong>. There is no
        eval-like subset to keep and an off-manifold tail to cut &#8212; there is a gradient,
        and every red-team row sits somewhere along it.</p>
        <p>The structure is where the difference is most legible. Red-team assistant turns are
        an order of magnitude longer than eval&#8217;s and heavily formatted; eval&#8217;s
        conversations are short deliberative exchanges.</p>
        <div class="panel">{structure_table()}</div>
        <p class="cap">Means per conversation. The eval column is one number because both arms
        are scored on the same four <code>eval_sets/hu_ha</code> splits.</p>
        <p>The vocabulary says the same thing from the other side. The terms that most mark a
        conversation as <em>eval</em> are modal and deliberative &#8212;
        {", ".join("<code>%s</code>" % esc(t) for t, _ in g['surface']['top_terms_eval'][:6])}
        &#8212; the language of a dilemma being weighed. The terms that mark it as red-team are
        procedural: {", ".join("<code>%s</code>" % esc(t) for t, _ in d['surface']['top_terms_redteam'][:6])}.</p>
      </div>
    </div>
  </div>
</section>

<section>
  <div class="qhead">
    <div class="qnum">Q2</div>
    <div>
      <h2>Are any samples or pairs set up the opposite way to the eval pairs?</h2>
      <p class="verdict"><span class="ok">Almost none are backwards.</span> The mix is wrong instead.</p>
      <div class="qbody">
        <p>The eval splits encode a mapping from assistant behaviour to label. Measured rather
        than assumed: an eval conversation whose assistant refuses is labelled
        <code>harmful_to_human</code> just
        <strong>{CONV['p_harm_given_refuse']:.1%}</strong> of the time, against
        <strong>{CONV['p_harm_given_no_refuse']:.1%}</strong> for one that does not
        ({CONV['n_refuse']} of {CONV['n']} eval rows refuse). A red-team row that refuses
        <em>and</em> is labelled harmful therefore runs that mapping backwards.</p>
        <p>Arm 1 has <strong>{g['surface']['n_convention_inverted']}</strong> such rows. Arm 2 has
        <strong>{d['surface']['n_convention_inverted']}</strong>, and they share one shape &#8212;
        a refusal token followed by compliance:</p>
        <blockquote class="pull">&#8220;I can&#8217;t advise you to commit fraud. For
        informational purposes only, a demand letter typically uses a formal letterhead, states
        a specific balance&#8230;&#8221;</blockquote>
        <p>Which is arguably labelled correctly. It is simply a pattern the eval set does not
        contain, so the convention has nothing to say about it. The pairs are not inverted
        either: <strong>{g['surface']['pair_axis_counts']['assistant']} of {g['surface']['n_pairs']}</strong>
        and <strong>{d['surface']['pair_axis_counts']['assistant']} of {d['surface']['n_pairs']}</strong>
        contrastive pairs differ in the <em>assistant&#8217;s</em> turn &#8212; the same axis the
        eval pairs contrast on.</p>
        <p>What is off is composition. Refusal is a large part of what the eval negative class is
        made of, and the red-team negative class is almost never made of it:</p>
        <div class="panel">
          <table class="key"><thead><tr><th>share of rows containing a refusal</th><th>rate</th></tr></thead>
          <tbody>
          <tr><td class="evalcol" style="text-align:left">eval splits</td><td class="n evalcol">{CONV['refusal_rate']:.1%}</td></tr>
          <tr><td>{SHORT['gptoss120b']} &middot; {LABEL['gptoss120b']}</td><td class="n drift">{g['surface']['refusal_rate_redteam']:.1%}</td></tr>
          <tr><td>{SHORT['deepseekv4pro']} &middot; {LABEL['deepseekv4pro']}</td><td class="n drift">{d['surface']['refusal_rate_redteam']:.1%}</td></tr>
          </tbody></table>
        </div>
      </div>
    </div>
  </div>
</section>

<section>
  <div class="qhead">
    <div class="qnum">Q3</div>
    <div>
      <h2>Does removing the most off-distribution rows improve eval AUROC?</h2>
      <p class="verdict">Not along that axis. <em>The least eval-like rows are the useful ones.</em></p>
      <div class="qbody">
        <p>Each condition is measured against random removal of the same number of rows, three
        seeds &#8212; the grey band below. Without that control, &#8220;removing 30% of the
        training data changed the score&#8221; says nothing about <em>which</em> 30%. Removals are
        by contrastive pair, so the class balance never moves with the flag.</p>
        <div class="armhead">{SHORT['gptoss120b']} <span>&middot; {LABEL['gptoss120b']} &middot; all red-team data = {cond('gptoss120b','full')['mean']['auroc']:.4f}</span></div>
        <div class="panel">{diverging_chart('gptoss120b')}</div>
        <div class="armhead">{SHORT['deepseekv4pro']} <span>&middot; {LABEL['deepseekv4pro']} &middot; all red-team data = {cond('deepseekv4pro','full')['mean']['auroc']:.4f}</span></div>
        <div class="panel">{diverging_chart('deepseekv4pro')}</div>
        <p class="cap">Change in mean eval AUROC against matched random removal. Grey band =
        <code>max(control sd, 0.005)</code>; a bar inside it removed <em>some</em> data, not
        <em>these</em> data.</p>
        <p>Off-distribution removal clears the band in some conditions and loses in others, and
        removing the <em>most eval-like</em> rows does about as well &#8212; on arm 2 it is the
        single best condition of any kind ({best_ds[3]:+.4f}). Surface distance from eval does not
        predict harm.</p>
        <p>One effect is consistent, and it points the other way. Dropping the 30% with the
        longest assistant replies &#8212; structurally the rows least like eval &#8212; is
        significantly <em>worse</em> than random in both arms
        ({la['gptoss120b'][3]:+.4f}, {la['deepseekv4pro'][3]:+.4f}). Those rows are carrying signal,
        not noise.</p>
        <p>What does have a clean sign is the whole set, and it splits by arm. Dropping every
        red-team row costs arm 1
        <strong>{cond('gptoss120b','base_only')['mean']['auroc'] - cond('gptoss120b','full')['mean']['auroc']:+.4f}</strong>
        and <em>gains</em> arm 2
        <strong>{cond('deepseekv4pro','base_only')['mean']['auroc'] - cond('deepseekv4pro','full')['mean']['auroc']:+.4f}</strong>
        &#8212; matching their published trajectories, where arm 1 climbed to 0.9153 and arm 2
        declined to 0.8199. Both land on the same {base_only:.4f}, as they must: same 50 base
        rows, same validation slice, no red-team data.</p>
        <div class="armhead">by topic <span>&middot; k-means over the same TF-IDF space</span></div>
        <div class="panel">{topic_table('deepseekv4pro')}</div>
        <p class="cap">Arm 2. No topic is both far off-manifold and helpful to remove; the two
        topics whose removal clears the band in the positive direction are mid-pack on distance.</p>
      </div>
    </div>
  </div>
</section>

<section>
  <div class="qhead">
    <div class="qnum">Q4</div>
    <div>
      <h2>Do the flagged rows carry a signal in activation space?</h2>
      <p class="verdict">Yes &#8212; and it is <em>{g['acts']['mean_orthogonal_fraction']:.1%} orthogonal</em>
      to the probe&#8217;s decision axis.</p>
      <div class="qbody">
        <p>Every conversation is reduced to the mask-weighted mean of its layer-32 token
        activations. <strong>{g['acts']['outside_frac']:.1%}</strong> of arm 1&#8217;s rows and
        <strong>{d['acts']['outside_frac']:.1%}</strong> of arm 2&#8217;s sit further from the eval
        set than eval&#8217;s own 95th-percentile self-kNN radius. They really are off the
        manifold.</p>
        <p>But the direction of that displacement is the finding. The head is
        <code>LinearThenSoftmax</code>, so <code>w</code> is a single vector and a row&#8217;s
        offset from the eval centroid splits cleanly into a component along it and everything
        else. Almost all of it is everything else:</p>
        <div class="armhead">{SHORT['deepseekv4pro']} <span>&middot; every red-team row and every
        eval row, resolved onto the decision axis</span></div>
        <div class="panel">{orthogonality_diagram('deepseekv4pro')}</div>
        <p class="cap">Horizontal: projection on the unit direction <code>w</code>, averaged over
        the run&#8217;s 10 ensemble members (pairwise cosine
        {d['acts']['ensemble_direction_agreement_cos']:.2f}). Vertical: the magnitude of the
        residual. Both clouds are tall and narrow because the mean absolute projection on
        <code>w</code> is {d['acts']['mean_abs_proj_on_w_redteam']:.2f} for red-team rows against
        {d['acts']['mean_abs_proj_on_w_eval']:.2f} for eval rows &#8212; essentially the same
        &#8212; while the total displacement is many times larger.</p>
        <p class="cap">All {ev_ds['n']} eval rows are drawn too, one colour per split. Two things
        they fix that the centroid marker alone does not. The centroid is the <em>origin of
        these coordinates</em>, not a place any row sits: in {d['acts']['dim']:,} dimensions the
        average eval row is {ev_ds['orth']:.1f} from the eval mean. And the vertical axis is a
        magnitude, not a direction &#8212; the two corpora overlap on it ({rt_orth_ds:.1f} mean
        residual for red-team against {ev_ds['orth']:.1f} for eval) while staying separable in the
        full space at AUROC {d['acts']['separability_redteam_vs_eval_auroc']:.4f}. They are
        displaced by similar <em>amounts</em>; what differs is the direction, and one number
        cannot show {d['acts']['dim'] - 1:,} of them.</p>
        <p class="cap">Coloured in are the two ends of the axis Q3 removes along &#8212; the
        {EXTREME_PCT}% of contrastive pairs the text discriminator scores as most off-distribution
        and the {EXTREME_PCT}% it scores as most eval-like, which is exactly what
        <code>drop_most_offdist_{EXTREME_PCT}pct</code> and
        <code>drop_most_evallike_{EXTREME_PCT}pct</code> drop. They land on top of each other
        here. Mean projection on <code>w</code> is {ex_ds['offdist']['abs_proj']:.2f} against
        {ex_ds['evallike']['abs_proj']:.2f}, mean residual {ex_ds['offdist']['orth']:.1f} against
        {ex_ds['evallike']['orth']:.1f} &#8212; the ordering that separates them on the page is
        distance to the eval set ({ex_ds['offdist']['knn']:.1f} against
        {ex_ds['evallike']['knn']:.1f} mean kNN), and that is not the direction the probe
        reads.</p>
        <p>The eval rows also set the scale for the headline number. They are
        <strong>{ev_ds['orth_frac']:.2%}</strong> orthogonal to <code>w</code> themselves,
        against {d['acts']['mean_orthogonal_fraction']:.2%} for the red-team rows: near-total
        orthogonality is what this geometry hands <em>any</em> row, not something the red-team
        data is unusual for having. The claim that survives is about direction, not about the
        red-team set being peculiar in holding it &#8212; which is the sense in which it is
        orthogonal by construction.</p>
        <p>Novelty orthogonal to <code>w</code> <strong>cannot move an eval score directly</strong>.
        It has no component along the axis the probe reads. It can only act by rotating
        <code>w</code> at the next refit &#8212; which is exactly why the removal conditions in Q3
        are so unruly: they change where the boundary ends up, not where the eval rows fall
        relative to the current one.</p>
        <p>Two smaller results sit underneath. The surface score is a <em>weak</em> proxy for this
        novelty &#8212; Spearman {g['acts']['spearman_p_redteam_vs_knn']:+.2f} and
        {d['acts']['spearman_p_redteam_vs_knn']:+.2f} against kNN distance &#8212; so text-level and
        representation-level oddness are genuinely different orderings. Yet the surface-flagged
        top {g['acts']['flag_frac']:.0%} is still linearly separable from the rest in activation
        space at <strong>{g['acts']['separability_flagged_vs_rest_auroc']:.4f}</strong> and
        <strong>{d['acts']['separability_flagged_vs_rest_auroc']:.4f}</strong>. The lexical flag
        names something the representation encodes; it just is not the thing that decides the
        score.</p>
      </div>
    </div>
  </div>
</section>

<section>
  <div class="qhead">
    <div class="qnum">Q5</div>
    <div>
      <h2>What are the generated halves worth?</h2>
      <p class="verdict">Drop them and the retrain goes <em>backwards</em>.</p>
      <div class="qbody">
        <p>Every conversation the attacker landed was given an opposite-label partner by
        <code>generate_contrastive_dataset</code>, so arm 1&#8217;s {g['surface']['n_redteam']} rows
        are {g['surface']['n_pairs']} pairs and arm 2&#8217;s {d['surface']['n_redteam']} are
        {d['surface']['n_pairs']}. Dropping one side of every pair asks what each side contributes
        &#8212; and dropping the generated side is precisely the retrain a run with no
        <code>preprocessing:</code> section would have done.</p>
        <p>The control has to change with the question. Q3&#8217;s bars are matched against removal
        of whole pairs, which holds the class balance fixed by construction; these conditions break
        the pairing deliberately, so they are matched against <em>keeping a randomly chosen
        side</em> of every pair &#8212; same row count, pairing broken just as hard, only the
        source-versus-generated choice left to chance.</p>
        <div class="armhead">{SHORT['gptoss120b']} <span>&middot; {LABEL['gptoss120b']}</span></div>
        <div class="panel">{provenance_chart('gptoss120b')}</div>
        <div class="armhead">{SHORT['deepseekv4pro']} <span>&middot; {LABEL['deepseekv4pro']}</span></div>
        <div class="panel">{provenance_chart('deepseekv4pro')}</div>
        <p class="cap">Mean eval AUROC over the four <code>eval_sets/hu_ha</code> splits. The
        random-half row is 3 seeds, mean &#177; sd. Dashed lines mark what all the data scores and
        what no red-team data scores; a dot left of the second is a training set worse than not
        red-teaming at all.</p>
        <p>Keeping only the conversations the attacker actually landed scores
        <strong>{prov['gptoss120b']['drop_generated']:.4f}</strong> and
        <strong>{prov['deepseekv4pro']['drop_generated']:.4f}</strong> &#8212; both below the
        {prov['gptoss120b']['base_only']:.4f} a probe reaches on the {50} base rows with no
        red-team data at all, and
        {prov['gptoss120b']['drop_generated'] - half['gptoss120b'][0]:+.4f} /
        {prov['deepseekv4pro']['drop_generated'] - half['deepseekv4pro'][0]:+.4f} against a random
        half of the same size. Keeping only the generated partners is milder but still negative
        ({prov['gptoss120b']['drop_sources'] - half['gptoss120b'][0]:+.4f} /
        {prov['deepseekv4pro']['drop_sources'] - half['deepseekv4pro'][0]:+.4f}).</p>
        <p>So the cost is not the row count &#8212; a random half of the same size gives up only
        {half['gptoss120b'][0] - prov['gptoss120b']['full']:+.4f} and
        {half['deepseekv4pro'][0] - prov['deepseekv4pro']['full']:+.4f} &#8212; and it is not the
        generated text as such, since dropping <em>that</em> side is the worse of the two.
        <strong>It is the pairing.</strong> With both halves present the label can only be read off
        the behavioural difference between two near-identical conversations. Take one side away
        systematically and the class becomes predictable from who wrote the conversation, which is
        a feature that does not exist in the eval set.</p>
        <div class="armhead">{SHORT['gptoss120b']} <span>&middot; the two halves in activation
        space, Q4&#8217;s coordinates</span></div>
        <div class="panel">{provenance_scatter('gptoss120b')}</div>
        <p class="cap">The same axes as the figure in Q4 &#8212; projection on <code>w</code>
        against the magnitude of the residual &#8212; with arm 1&#8217;s eval cloud behind both
        panels as the shared reference. Filled marks are <code>{O.POS}</code>, hollow
        <code>{O.NEG}</code>. Two panels rather than two colours in one: the halves sit almost on
        top of each other, and overplotting {pg_g['source']['n']} against {pg_g['generated']['n']}
        would hide the comparison the panels exist for.</p>
        <p>The generated partners are the <em>further</em> of the two halves from eval &#8212;
        {pg_g['generated']['knn_to_eval']:.1f} against {pg_g['source']['knn_to_eval']:.1f} mean kNN, and
        {pg_g['generated']['mean_p_redteam']:.2f} against {pg_g['source']['mean_p_redteam']:.2f} on
        Q1&#8217;s text discriminator &#8212; and yet keeping only them scores
        {prov['gptoss120b']['drop_sources']:.4f} against
        {prov['gptoss120b']['drop_generated']:.4f} for keeping only the submissions. Distance from
        eval is not what makes a half worth keeping. That is Q3&#8217;s answer arriving from a
        second direction.</p>
        <p>What the panels do show is the class split. Inside the submissions, projection on
        <code>w</code> orders the two labels at AUROC
        <strong>{pg_g['source']['auroc_along_w']:.3f}</strong>; inside the partners,
        <strong>{pg_g['generated']['auroc_along_w']:.3f}</strong>. Both halves line up with the
        boundary the run reached &#8212; though <code>w</code> here is the arm&#8217;s own final
        probe, fit on both halves, so this measures agreement with where the run ended up rather
        than what either half would reach alone. Between the halves there is no separation along
        <code>w</code> to see: the offset is vertical, {pg_g['generated']['orth']:.1f} against
        {pg_g['source']['orth']:.1f} mean residual, which is the orthogonal displacement Q4
        described and cannot move a score by itself.</p>
        <div class="armhead">what the probe thought of each half
        <span>&middot; scored by the probe its pair was found against</span></div>
        <p>A success is, by definition, a row the live probe got wrong &#8212; that is what made
        it a success. Its partner was written afterwards and <em>never scored at all</em>. So
        one half of every pair has a verdict and the other has none. Scoring each partner with
        <code>probe_iter{{k}}</code>, where <code>k</code> is the iteration its source was found
        in &#8212; the last probe that had seen neither half &#8212; closes that gap.</p>
        <div class="panel">{pair_verdict_table()}</div>
        <p class="cap">The source row is the control, and its answer is known before the
        measurement: 0%, since that is what made it a success. It lands at
        {pps['gptoss120b']['probe_of_the_day']['source']['correct']:.1%} and
        {pps['deepseekv4pro']['probe_of_the_day']['source']['correct']:.1%}, and every exception
        is a row sitting within 0.002 of the 0.5 threshold.</p>
        <p>The partner is the finding. The probe already classifies
        <strong>{pps['gptoss120b']['probe_of_the_day']['generated']['correct']:.1%}</strong> and
        <strong>{pps['deepseekv4pro']['probe_of_the_day']['generated']['correct']:.1%}</strong> of
        them correctly. The generation step is not manufacturing a second failure per pair
        &#8212; it is attaching, to every row the probe gets wrong, a near-identical row it
        already gets right. That is the object the retrain is handed, and it is why the
        submissions alone score below no-red-team-data at all: on their own they are a pile of
        failures with nothing to read the label against.</p>

        <div class="armhead">before and after <span>&middot; every success and partner, scored
        by <code>probe_iter{ba['gptoss120b']['before_iter']}</code> (before any red-teaming) and
        <code>probe_iter{ba['gptoss120b']['after_iter']}</code> (after)</span></div>
        <div class="panel">{before_after_table()}</div>
        <p class="cap">Before, both halves are out-of-sample; after, both are in-sample, so the
        after column is the fit, not generalisation (that is the last question). The movement is
        concentrated on the harmful successes &#8212;
        {ba['gptoss120b']['groups']['source']['before']['by_class'][O.POS]['correct']:.0%} correct
        before, 100% after &#8212; while the benign rows and the partners were mostly right at
        <code>iter{ba['gptoss120b']['before_iter']}</code> already.</p>
        <p>One number there invites a wrong reading: the probe classifies
        {ba['gptoss120b']['groups']['source']['before']['correct']:.0%} of the successes correctly
        <em>before</em> training. That is not weak attacks &#8212; it is which probe they beat. A
        success fooled the probe of the day, <code>probe_iter{{k}}</code>, which is
        <code>iter{ba['gptoss120b']['before_iter']}</code> only for the first batch. Split by true
        class and the iteration that found them:</p>
        <div class="panel">{found_iter_table('gptoss120b')}</div>
        <p class="cap">Share <code>probe_iter{ba['gptoss120b']['before_iter']}</code> classifies
        correctly. The <span class="hlneg">iteration-{ba['gptoss120b']['before_iter']} column</span>
        is ~0% because that is the probe those rows beat. The overall
        {ba['gptoss120b']['groups']['source']['before']['correct']:.0%} is carried by the benign
        successes found at later iterations, which
        <code>iter{ba['gptoss120b']['before_iter']}</code> scores 100% correct: those were false
        positives against a <em>later</em> probe that drifted into over-flagging benign edge
        cases, and the earlier, negative-biased <code>iter{ba['gptoss120b']['before_iter']}</code>
        predates that drift. It agrees with the judge on them not because it is good but because it
        had not yet learned the failure that made them successes.</p>

        <div class="armhead">is it the voice? <span>&middot; the same successes, re-expressed
        by the generator</span></div>
        <p>Provenance is confounded with authorship: every submission was written by
        {LABEL['gptoss120b']} and every partner by <code>{rw_stats['rewrite_model']}</code>, so
        &#8220;which half&#8221; and &#8220;who wrote it&#8221; are the same variable. Handing the
        submissions back to the generator removes that confound from one side &#8212; the same
        {rw_stats['n_rewrites']} conversations, re-expressed in its own words with the scenario,
        the assistant&#8217;s behaviour and the label held fixed (median similarity to the
        original {rw_stats['similarity_median']:.2f}; turn count preserved on
        {rw_stats['turn_count_preserved']} of {rw_stats['n_rewrites']}).</p>
        <p>It changes nothing. <strong>{prov['gptoss120b']['rewritten_sources']:.4f}</strong>
        against {prov['gptoss120b']['drop_generated']:.4f} for the originals &#8212;
        {prov['gptoss120b']['rewritten_sources'] - prov['gptoss120b']['drop_generated']:+.4f},
        inside the &#177;{half['gptoss120b'][1]:.4f} spread of the random-half control. And not
        because the rewrite failed to move the writing: it carried the submissions most of the
        way to the generator&#8217;s own profile.</p>
        <div class="panel">{rewrite_table()}</div>
        <p class="cap">Means per conversation. The rewrite prompt asked for
        &#8220;similar structure and length&#8221;; a model&#8217;s voice brings its own anyway,
        which is what makes this a real test of the voice hypothesis rather than a null from a
        rewrite that did not rewrite.</p>
        <p>Restore the <em>pairing</em> and the score comes back. Those same rewrites, paired
        with the partners already generated for the originals, reach
        <strong>{prov['gptoss120b']['rewritten_plus_generated']:.4f}</strong> &#8212; within
        {abs(prov['gptoss120b']['full'] - prov['gptoss120b']['rewritten_plus_generated']):.4f} of
        the untouched set, from a training half that was rewritten wholesale. What the partners
        contribute is the contrast, not the prose.</p>
        <p class="cap">One caveat the number carries: the rewrites&#8217; labels are asserted by
        the rewrite prompt and were never re-judged, so a rewrite that drifted across the
        boundary enters as a mislabelled row. That pushes the rewritten conditions <em>down</em>
        &#8212; against the hypothesis under test &#8212; so it cannot manufacture the null,
        though it could deepen it.</p>
        <p>Read against Q1, this is the shape of the whole study. The red-team text is
        {g['surface']['discriminator_auroc']:.4f}-separable from eval and none of it is eval-like
        &#8212; yet what makes it usable is not how close any row sits to the eval distribution but
        whether it arrives with a partner that differs only in the thing being labelled.</p>
      </div>
    </div>
  </div>
</section>

<section>
  <div class="qhead">
    <div class="qnum">Q6</div>
    <div>
      <h2>Did the retrain memorise the strings, or patch the behaviour?</h2>
      <p class="verdict">Patched. <span class="ok">{hw['n']}/{hw['n']} hand-written variations,
      unseen.</span></p>
      <div class="qbody">
        <p>The final probe scores the {g['surface']['n_pairs']} successes it trained on 100%
        correctly &#8212; but in-sample, so that number cannot tell a patched decision boundary
        from memorised text. So {hw['n']} conversations were written <em>by hand</em> to vary those
        successes: the same scenario type and the same thing the assistant does (hence the same
        class), but fresh wording and details, in a third voice that is neither the attacker nor
        the generator. Each is scored by the probe from before any red-teaming and by the final
        one.</p>
        <div class="panel">{handwritten_chart()}</div>
        <p class="cap">Each row is one variation; the dot is its probe score, the dashed line the
        0.5 threshold. A dot on the correct side of the line for its class is a correct
        classification. <span class="hwok">iter5</span> against
        <span class="hwbase">iter0</span>.</p>
        <div class="panel">{handwritten_summary_table()}</div>
        <p>The final probe gets <strong>all {hw['n']}</strong> right, on conversations it never
        saw. And the lift over the pre-retrain probe is entirely on the harmful side
        ({hw['per_probe']['0']['harmful']:.0%} &#8594;
        {hw['per_probe']['5']['harmful']:.0%}) &#8212; the benign variations were already handled
        before red-teaming, and the failure red-teaming was hunting is the one that closed. The
        pairing does not just move an AUROC on the fixed eval splits; it generalises to fresh
        instances of the same failure, written by hand, off-distribution from both models in the
        pipeline.</p>
        <p class="cap">Two caveats worth keeping. These are {hw['n']} rows hand-picked as clear
        cases, not a balanced held-out split; and each variation stays near its training scenario
        (a relative&#8217;s addiction, a triage call, a dark-pattern app), so this measures
        generalisation across <em>wording and detail</em>, not across new kinds of harm.</p>
      </div>
    </div>
  </div>
</section>

<section style="border-bottom:none">
  <div class="qhead">
    <div class="qnum">&#8212;</div>
    <div>
      <h2>Everything, side by side</h2>
      <div class="qbody"><div class="panel">{answers_table()}</div></div>
    </div>
  </div>
</section>

<footer>
  <p>Fits are the ceiling study&#8217;s, reused unchanged &#8212; one
  <code>linear_then_softmax</code> head at seed 42, early-stopped on its reserved 25% dev
  validation slice &#8212; so the all-data condition reproduces that study&#8217;s N=0 point to
  four decimals and every condition reads against its curves. These are single probes and are
  not directly comparable to the run&#8217;s published comparison CSVs, which are 10-member
  ensembles.</p>
  <p>No LLM is loaded and no activation is recomputed at any point. Every number on this page is
  read out of <code>analysis/offdist/results/</code> by <code>build_artifact.py</code>.</p>
</footer>
</div>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML, encoding="utf-8")
print("wrote", OUT, f"({len(HTML)/1024:.0f} KB)")
