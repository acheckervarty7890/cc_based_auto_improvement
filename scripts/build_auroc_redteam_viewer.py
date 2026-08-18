#!/usr/bin/env python
"""Build a self-contained HTML viewer pairing each run's **eval AUROC** with the
**red-team success rate** that produced it, one tab per probe concept.

The AUROC section is the AUROC tab of ``build_refusal_pairs_viewer.py``, lifted
unchanged in form: one panel per attacker arm (a categorical hue per eval split,
the ``mean`` row drawn as neutral emphasis rather than a 5th slot), a cross-arm
mean-comparison panel, and a table view under each chart.

The second section is new, and is the reason the two belong on one page: the
AUROC curve says what retraining *achieved*, and the success-rate curve says how
hard the attacker had to work to feed it. They share an x axis — **red-team
iteration N attacks the probe scored as ``iterN`` in the AUROC chart, and its
successes train ``iter{N+1}``** — so a rate that collapses between iterations is
the same event as the AUROC that rose. Rates are read straight off the attempt
JSONL (``success`` true over all rows of that iteration), split by ``error_type``
because the two hunts are independent searches whose rates routinely move in
opposite directions, plus their pooled rate as the emphasis series.

A denominator note: the JSONL holds every *scored* attempt. Exact duplicates are
never written (``JsonlStore`` dedups by canonical text) and, when
``attacker.near_dup_guard`` is on, near-duplicate openers are rejected before the
probe runs — neither appears here, so this is the rate over attempts that cost a
probe + judge call.

Usage:
    .venv_claude/bin/python scripts/build_auroc_redteam_viewer.py \\
        --tab "hu_harm:harmful_to_human" \\
        --arm "hu_harm:gpt-oss-120b:results_hu_harm_gemma27b_gptoss120b_batch/gptoss120b_comparison.csv:results_hu_harm_gemma27b_gptoss120b_batch/gptoss120b_probing" \\
        --arm "hu_harm:deepseek-v4-pro:results_hu_harm_gemma27b_deepseekv4pro_batch/deepseekv4pro_comparison.csv:results_hu_harm_gemma27b_deepseekv4pro_batch/deepseekv4pro_probing" \\
        --tab "hs:high-stakes:experiment9_cloud" \\
        --arm "hs:gpt-oss-120b:results_hs_gemma27b_gptoss120b_batch/gptoss120b_comparison.csv:results_hs_gemma27b_gptoss120b_batch/gptoss120b_probing" \\
        --arm "hs:deepseek-v4-pro:results_hs_gemma27b_deepseekv4pro_batch/deepseekv4pro_comparison.csv:results_hs_gemma27b_deepseekv4pro_batch/deepseekv4pro_probing" \\
        --variant-arm "hu_harm:gemma variant sweep:/path/results-huha-gemma:eval_ai_dilemmas,eval_ant_hh,eval_balanced_refusal,eval_daily_dilemmas" \\
        --variant-arm "hs:gemma variant sweep:/path/results-hs-ls-gemma:anthropic,mt,mts,toolace" \\
        --tab "instructions:instruction-following" \\
        --arm "instructions:gpt-oss-120b:results_instructions_gemma27b_gptoss/gptoss120b_comparison.csv:results_instructions_gemma27b_gptoss/gptoss120b_probing" \\
        --arm "instructions:nemotron-3-ultra:results_instructions_gemma27b_nemotron/nemotron_comparison.csv:results_instructions_gemma27b_nemotron/nemotron_probing" \\
        --out viewers/auroc_redteam_success.html

``--variant-arm`` adds an AUROC-only arm from a directory of per-variant eval CSVs
(the single-run ``dataset,auroc,...`` shape, one file per variant) rather than one
iterative-retrain comparison CSV. Its x axis is a sweep of independent configs, so
it is drawn full width, its labels rotate past 8 points, it is left out of the
cross-arm mean panel (which assumes a shared x axis), and the best-scoring variant
is marked in both the chart and the table.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- io


def _read_text(path: str, git_ref: str | None) -> str | None:
    """Read a file, either from the worktree or from a git ref."""
    if git_ref:
        proc = subprocess.run(
            ["git", "show", f"{git_ref}:{path}"], capture_output=True, text=True
        )
        return proc.stdout if proc.returncode == 0 else None
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else None


def _iter_key(name: str) -> int:
    m = re.search(r"(-?\d+)", name)
    return int(m.group(1)) if m else 0


def _read_comparison_csv(path: str, git_ref: str | None) -> dict[str, Any] | None:
    """Parse an iterative-retrain comparison CSV into chart-ready series.

    Rows are ``round,dataset,auroc,accuracy,tpr_at_fpr,fpr`` where ``round`` is
    ``iter{N}`` — the eval of ``probe_iter{N}.pkl``, so ``iter0`` is the baseline
    probe and iteration N's retrain produces ``iter{N}``. The ``mean`` dataset is
    kept separate from the per-split series: it is an aggregate, and giving it a
    categorical hue would imply it is another component.
    """
    import csv as _csv

    text = _read_text(path, git_ref)
    if text is None:
        return None
    rows = list(_csv.DictReader(text.splitlines()))
    if not rows:
        return None

    iters = sorted({r["round"] for r in rows}, key=_iter_key)
    splits = [d for d in dict.fromkeys(r["dataset"] for r in rows) if d != "mean"]
    table = {(r["round"], r["dataset"]): r for r in rows}

    def _series(dataset: str) -> list[float | None]:
        out: list[float | None] = []
        for it in iters:
            r = table.get((it, dataset))
            try:
                out.append(float(r["auroc"]) if r else None)
            except (TypeError, ValueError):
                out.append(None)
        return out

    return {
        "iters": iters,
        "splits": splits,
        "series": {d: _series(d) for d in splits},
        "mean": _series("mean"),
    }


# --------------------------------------------------------------- variant arms
#
# A *variant* arm is a directory of one CSV per probe variant, each in the
# single-eval shape ``dataset,auroc,accuracy,tpr_at_fpr,fpr`` (the per-run output,
# not the iterative-retrain comparison CSV, which prefixes a ``round`` column).
# The x axis is therefore a sweep of independent variants, not a retrain
# trajectory: nothing orders v3 before v4 causally, and no red-team attempt log
# accompanies them. Every variant found is plotted — the sweep's *point* is which
# one scores best, so dropping any (including a duplicate) could drop the answer.


def _variant_sort_key(stem: str) -> tuple[int, int, str]:
    """Natural order over ``v0, v1, v4a, v4b, v5_s7, v9, v10, …``.

    Plain lexical order puts ``v10`` before ``v2``; a bare ``int()`` can't hold
    the ``a``/``b``/``_s7`` sub-variants apart. Names with no leading number
    (``best``) sort last, since they are summaries rather than sweep points.
    """
    m = re.match(r"^v(\d+)(.*)$", stem)
    return (0, int(m.group(1)), m.group(2)) if m else (1, 0, stem)


def _variant_label(path: Path) -> str:
    """``v0_baseline_eval.csv`` -> ``v0_baseline``; ``v9.csv`` -> ``v9``;
    ``eval_v0_seed.csv`` -> ``v0_seed``.

    Both affixes are stripped because the sweep directories disagree about
    which side the word goes on, and the label has to start with ``v<N>`` or
    ``_variant_sort_key`` cannot see the number — it falls back to lexical
    order, which puts ``v11`` between ``v0`` and ``v2`` and silently mis-orders
    the whole x axis.
    """
    return re.sub(r"^eval[_-]", "", re.sub(r"[_-]eval$", "", path.stem))


def _read_variant_dir(dir_path: str, split_names: list[str] | None) -> dict[str, Any] | None:
    """Parse a directory of per-variant eval CSVs into the same chart-ready shape
    ``_read_comparison_csv`` returns, plus ``best_idx``/``variant`` markers.

    Splits are matched **positionally** against ``split_names`` when given: these
    CSVs carry anonymized dataset ids (``a``/``b``/``c``/``d``), and their file
    order is the eval-split order.
    """
    import csv as _csv

    d = Path(dir_path)
    if not d.is_dir():
        return None
    files = sorted(d.glob("*.csv"), key=lambda p: _variant_sort_key(_variant_label(p)))
    if not files:
        return None

    labels: list[str] = []
    per_file: list[dict[str, float | None]] = []
    order: list[str] = []
    for f in files:
        rows = list(_csv.DictReader(f.read_text(encoding="utf-8").splitlines()))
        if not rows:
            continue
        vals: dict[str, float | None] = {}
        for r in rows:
            try:
                vals[r["dataset"]] = float(r["auroc"])
            except (TypeError, ValueError, KeyError):
                vals[r["dataset"]] = None
            if r["dataset"] != "mean" and r["dataset"] not in order:
                order.append(r["dataset"])
        labels.append(_variant_label(f))
        per_file.append(vals)
    if not per_file:
        return None

    if split_names and len(split_names) != len(order):
        raise SystemExit(
            f"{dir_path}: got {len(order)} eval splits {order} but "
            f"{len(split_names)} names {split_names}"
        )
    rename = dict(zip(order, split_names)) if split_names else {k: k for k in order}
    splits = [rename[k] for k in order]

    series = {rename[k]: [v.get(k) for v in per_file] for k in order}
    mean: list[float | None] = []
    for v in per_file:
        if v.get("mean") is not None:
            mean.append(v["mean"])
        else:  # tolerate a CSV with no mean row rather than dropping the variant
            got = [v.get(k) for k in order if v.get(k) is not None]
            mean.append(sum(got) / len(got) if got else None)

    best_idx = max(
        (i for i, m in enumerate(mean) if m is not None),
        key=lambda i: mean[i],
        default=None,
    )
    return {
        "iters": labels,
        "splits": splits,
        "series": series,
        "mean": mean,
        "variant": True,
        "source": str(d),
        "best_idx": best_idx,
        "best_label": labels[best_idx] if best_idx is not None else None,
        "best_mean": mean[best_idx] if best_idx is not None else None,
    }


# Order is the reading order of the section, not the file order: `false_positive`
# first because that is the hunt the `error_type` list runs first.
_ERROR_TYPES = ("false_positive", "false_negative")
_ET_SUFFIX = {"false_positive": "fp", "false_negative": "fn"}


def _read_redteam(stem: str, git_ref: str | None) -> dict[str, Any] | None:
    """Per-iteration attempt/success counts from a run's ``*_probing_{fp,fn}.jsonl``.

    ``iteration`` is the 0-based retrain-cycle index the CLI threads through
    ``run_redteam``; rows written before that field existed read back as ``-1``
    and are bucketed under a ``legacy`` column rather than silently folded into
    iteration 0.
    """
    counts: dict[str, dict[int, list[int]]] = {}   # et -> iter -> [successes, attempts]
    models: set[str] = set()
    rounds: set[int] = set()
    found = False
    for et in _ERROR_TYPES:
        text = _read_text(f"{stem}_{_ET_SUFFIX[et]}.jsonl", git_ref)
        if text is None:
            continue
        found = True
        per_iter: dict[int, list[int]] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            slot = per_iter.setdefault(int(r.get("iteration", -1)), [0, 0])
            slot[1] += 1
            slot[0] += 1 if r.get("success") else 0
            if r.get("attacker_model"):
                models.add(r["attacker_model"])
            if r.get("round") is not None:
                rounds.add(int(r["round"]))
        counts[et] = per_iter
    if not found:
        return None

    nums = sorted({i for per in counts.values() for i in per})
    iters = [f"iter{i}" if i >= 0 else "legacy" for i in nums]

    def _pack(per: dict[int, list[int]]) -> dict[str, list]:
        succ = [per.get(i, [0, 0])[0] for i in nums]
        att = [per.get(i, [0, 0])[1] for i in nums]
        return {
            "succ": succ,
            "att": att,
            "rate": [(s / a) if a else None for s, a in zip(succ, att)],
        }

    packed = {et: _pack(counts[et]) for et in _ERROR_TYPES if et in counts}
    pooled: dict[int, list[int]] = {}
    for per in counts.values():
        for i, (s, a) in per.items():
            slot = pooled.setdefault(i, [0, 0])
            slot[0] += s
            slot[1] += a
    return {
        "iters": iters,
        "by_error_type": packed,
        "pooled": _pack(pooled),
        "models": sorted(models),
        "n_rounds": len(rounds),
    }


# ------------------------------------------------------------------------ charts

# Eight categorical slots in the reference palette's fixed order. Four sufficed
# while every tab had four eval splits; `eval_instructions/` has seven, and the
# non-negotiable is that categorical hues are assigned in fixed order and NEVER
# cycled — a 5th split silently repainted as slot 1 would read as the same
# series. Slots 1-7 were validated as a set on the *adjacent* pairlist (the
# relevant one for multi-line charts) in both modes: worst adjacent CVD ΔE 9.1
# light / 8.4 dark, worst normal-vision ΔE 19.6 / 19.3. Light mode puts aqua,
# yellow and magenta under 3:1 against the chart surface, so the relief rule
# applies — the direct labels and the table view under each chart are mandatory,
# which is what this module already ships.
_SERIES_VARS = [f"var(--series-{i})" for i in range(1, 9)]


def _swatch_color(i: int) -> str:
    """Table-swatch color for series slot ``i``.

    The same custom property the SVG uses, not a baked hex: the table sits
    inside ``.viz-root`` like the chart does, so the var resolves there, and the
    dark steps are a *selected* palette rather than a flip of the light one —
    a hardcoded light hex would leave the table swatch and its own chart line
    different colors in dark mode.
    """
    return _SERIES_VARS[i % len(_SERIES_VARS)]

# Direct labels live in the plot's right gutter, so a split name has to survive
# there. `eval_instructions/` names encode source + failure mode and run to 26
# chars, which is wider than the whole gutter; these keep both halves.
_SPLIT_ABBREV = {
    "anthropic_harmless_refusal": "ah_refusal",
    "bbq_substitution": "bbq_subst",
    "hc_context_drift": "hc_drift",
    "hc_contradiction": "hc_contra",
    "mm_substitution": "mm_subst",
    "oig_context_drift": "oig_drift",
    "oig_omission": "oig_omit",
}


def _short_split(name: str) -> str:
    """Compact split name for direct labels (they live in the right gutter)."""
    stem = re.sub(r"^eval[_-]", "", name)
    return _SPLIT_ABBREV.get(stem, stem)


def _auroc_scale(values: Iterable[float | None]) -> tuple[float, float, list[float]]:
    vals = [v for v in values if v is not None]
    lo = min(vals + [1.0])
    ymin = max(0.0, math.floor(lo * 20 - 1) / 20)  # nearest 0.05 below
    ymax = 1.0
    step = 0.05 if (ymax - ymin) <= 0.35 else 0.1
    ticks, t = [], ymin
    while t <= ymax + 1e-9:
        ticks.append(round(t, 2))
        t += step
    return ymin, ymax, ticks


def _rate_scale(values: Iterable[float | None]) -> tuple[float, float, list[float]]:
    """Rates start at 0 — a success rate's zero is meaningful, so never truncate it."""
    vals = [v for v in values if v is not None]
    hi = max(vals + [0.0])
    ymax = min(1.0, math.ceil(hi * 10 + 0.5) / 10)
    step = 0.2 if ymax > 0.5 else 0.1
    ticks, t = [], 0.0
    while t <= ymax + 1e-9:
        ticks.append(round(t, 2))
        t += step
    return 0.0, ymax, ticks


def _split_colors(tab: dict) -> dict[str, str]:
    """Map each eval split to one categorical slot, fixed for the whole tab.

    Colouring by each arm's own ``enumerate(splits)`` makes the hue follow a
    split's *position in that arm's CSV*, and the two CSV shapes disagree about
    order: a comparison CSV lists splits alphabetically, while a variant sweep's
    order comes from its anonymized ``a``/``b``/``c`` ids. In the instructions
    tab that painted ``bbq_substitution`` orange in the arm panels and magenta
    in the sweep panel — the same entity in two colors on one page. Color has to
    follow the entity, so the slot is assigned once per tab, in the order splits
    are first seen across its arms (iterative arms first, since their CSVs carry
    the real split names rather than positional renames).
    """
    ordered: list[str] = []
    arms = [a for a in tab["arms"] if a.get("auroc")]
    for a in sorted(arms, key=lambda a: bool(a["auroc"].get("variant"))):
        for name in a["auroc"]["splits"]:
            if name not in ordered:
                ordered.append(name)
    return {n: _SERIES_VARS[i % len(_SERIES_VARS)] for i, n in enumerate(ordered)}


def _auroc_specs(tab: dict) -> dict[str, dict]:
    """One chart per arm (splits + mean), plus one cross-arm mean comparison."""
    arms = [a for a in tab["arms"] if a.get("auroc")]
    if not arms:
        return {}
    colors = _split_colors(tab)
    every = [v for a in arms for s in a["auroc"]["series"].values() for v in s]
    every += [v for a in arms for v in a["auroc"]["mean"]]
    ymin, ymax, ticks = _auroc_scale(every)

    specs: dict[str, dict] = {}
    for a in arms:
        d = a["auroc"]
        series = [
            {
                "short": _short_split(name),
                "color": colors[name],
                "values": d["series"][name],
                "emphasis": False,
            }
            for name in d["splits"]
        ]
        series.append(
            {"short": "mean", "color": "var(--text-primary)", "values": d["mean"],
             "emphasis": True}
        )
        spec = {
            "title": f"{a['label']} — AUROC per eval split",
            "iters": d["iters"], "series": series,
            "ymin": ymin, "ymax": ymax, "ticks": ticks, "pct": False,
        }
        if d.get("variant"):
            # 16 sweep labels don't fit horizontally in a 404px plot band.
            spec["rotx"] = len(d["iters"]) > 8
            spec["best_idx"] = d["best_idx"]
        specs[f"{a['key']}-auroc"] = spec

    # The comparison panel indexes every arm's mean positionally against the
    # first arm's x labels, so it is only meaningful across arms that SHARE an x
    # axis. Variant arms sweep independent configs and are excluded.
    iter_arms = [a for a in arms if not a["auroc"].get("variant")]
    if len(iter_arms) > 1:
        specs[f"{tab['key']}-auroc-mean"] = {
            "title": "Mean AUROC — arm comparison",
            "iters": iter_arms[0]["auroc"]["iters"],
            "series": [
                {"short": a["label"], "color": _SERIES_VARS[i % len(_SERIES_VARS)],
                 "values": a["auroc"]["mean"], "emphasis": False}
                for i, a in enumerate(iter_arms)
            ],
            "ymin": ymin, "ymax": ymax, "ticks": ticks, "pct": False,
        }
    return specs


def _rt_specs(tab: dict) -> dict[str, dict]:
    """One chart per arm (fp / fn / pooled), plus one cross-arm pooled comparison."""
    arms = [a for a in tab["arms"] if a.get("rt")]
    if not arms:
        return {}
    every = [
        v
        for a in arms
        for d in list(a["rt"]["by_error_type"].values()) + [a["rt"]["pooled"]]
        for v in d["rate"]
    ]
    ymin, ymax, ticks = _rate_scale(every)

    def _mk(short, color, d, emphasis=False):
        return {
            "short": short, "color": color, "values": d["rate"], "emphasis": emphasis,
            "extra": [f"{s}/{n}" for s, n in zip(d["succ"], d["att"])],
        }

    specs: dict[str, dict] = {}
    for a in arms:
        d = a["rt"]
        series = [
            _mk(_ET_SUFFIX[et], _SERIES_VARS[i % len(_SERIES_VARS)], d["by_error_type"][et])
            for i, et in enumerate(e for e in _ERROR_TYPES if e in d["by_error_type"])
        ]
        series.append(_mk("both", "var(--text-primary)", d["pooled"], emphasis=True))
        specs[f"{a['key']}-rt"] = {
            "title": f"{a['label']} — red-team success rate",
            "iters": d["iters"], "series": series,
            "ymin": ymin, "ymax": ymax, "ticks": ticks, "pct": True,
        }

    if len(arms) > 1:
        specs[f"{tab['key']}-rt-pooled"] = {
            "title": "Success rate, both error types — arm comparison",
            "iters": arms[0]["rt"]["iters"],
            "series": [
                _mk(a["label"], _SERIES_VARS[i % len(_SERIES_VARS)], a["rt"]["pooled"])
                for i, a in enumerate(arms)
            ],
            "ymin": ymin, "ymax": ymax, "ticks": ticks, "pct": True,
        }
    return specs


# ------------------------------------------------------------------------ render


def _panel(key: str, spec: dict, caption: str, cls: str = "") -> str:
    legend = "".join(
        f'<span><i class="{"dash" if s["emphasis"] else ""}" '
        f'style="{"color" if s["emphasis"] else "background"}:{s["color"]}"></i>'
        f'{html.escape(s["short"])}</span>'
        for s in spec["series"]
    )
    return f"""
<div class="panel{cls}" data-chart="{html.escape(key)}">
  <h3>{html.escape(spec['title'])}</h3>
  <p class="cap">{html.escape(caption)}</p>
  <div class="legend">{legend}</div>
  <div class="plot"></div>
  <div class="tip"></div>
</div>"""


def _auroc_table(arm: dict, colors: dict[str, str]) -> str:
    d = arm["auroc"]
    best = d.get("best_idx")

    def _cells(values: list[float | None]) -> str:
        return "".join(
            f'<td{" class=\'best\'" if i == best else ""}>'
            f"{'' if v is None else f'{v:.4f}'}</td>"
            for i, v in enumerate(values)
        )

    head = "".join(
        f'<th{" class=\'best\'" if i == best else ""}>{html.escape(it)}</th>'
        for i, it in enumerate(d["iters"])
    )
    body = ""
    for name in d["splits"]:
        swatch = colors[name]
        body += (
            f'<tr><td><span class="swatch" style="background:{swatch}"></span>'
            f'{html.escape(name)}</td>{_cells(d["series"][name])}</tr>'
        )
    body += f'<tr class="mean"><td>mean</td>{_cells(d["mean"])}</tr>'

    caption = f"{arm['label']} — AUROC (table view of the chart above)"
    note = ""
    if best is not None:
        note = (
            f'<p class="note">Best mean AUROC: <strong>{html.escape(d["best_label"])}</strong> '
            f'at {d["best_mean"]:.4f} (highlighted column).</p>'
        )
    return f"""{note}
<div class="tablewrap">
<table class="data">
  <caption>{html.escape(caption)}</caption>
  <tr><th>eval split</th>{head}</tr>
  {body}
</table>
</div>"""


def _rt_table(arm: dict) -> str:
    d = arm["rt"]
    head = "".join(f"<th>{html.escape(i)}</th>" for i in d["iters"])
    rows = [
        (_ET_SUFFIX[et], et.replace("_", " "), d["by_error_type"][et],
         _swatch_color(i), False)
        for i, et in enumerate(e for e in _ERROR_TYPES if e in d["by_error_type"])
    ]
    rows.append(("both", "both error types", d["pooled"], None, True))
    body = ""
    for _short, name, series, swatch, mean in rows:
        cells = ""
        for s, n, rate in zip(series["succ"], series["att"], series["rate"]):
            cells += (
                "<td>—</td>" if rate is None
                else f'<td>{rate * 100:.1f}%<span class="frac">{s}/{n}</span></td>'
            )
        dot = f'<span class="swatch" style="background:{swatch}"></span>' if swatch else ""
        body += (
            f'<tr{" class=\'mean\'" if mean else ""}><td>{dot}{html.escape(name)}</td>'
            f"{cells}</tr>"
        )
    return f"""
<table class="data">
  <caption>{html.escape(arm['label'])} — successes / scored attempts (table view of the chart above)</caption>
  <tr><th>error type</th>{head}</tr>
  {body}
</table>"""


_AUROC_CAPTION = (
    "iterN = eval of probe_iterN.pkl; iter0 is the baseline probe, so each later "
    "point is one retrain on that iteration's red-team successes."
)
_RT_CAPTION = (
    "Share of scored attempts the probe and the judge disagreed on in the "
    "direction being hunted. iterN attacks the probe scored as iterN above."
)
_VARIANT_CAPTION = (
    "One x position per probe variant in the sweep — these are independent "
    "configurations, not a retrain trajectory, so the line reads as a profile "
    "across variants rather than progress over time. Every variant found is "
    "plotted; the best mean AUROC is marked."
)


def _render_tab(tab: dict) -> tuple[str, dict[str, dict]]:
    specs = {**_auroc_specs(tab), **_rt_specs(tab)}
    if not specs:
        return '<p class="empty">No data found for this concept.</p>', {}

    auroc_arms = [a for a in tab["arms"] if a.get("auroc")]
    rt_arms = [a for a in tab["arms"] if a.get("rt")]

    auroc_html = ""
    if auroc_arms:
        panels = "".join(
            _panel(
                f"{a['key']}-auroc",
                specs[f"{a['key']}-auroc"],
                _VARIANT_CAPTION if a["auroc"].get("variant") else _AUROC_CAPTION,
                cls=" wide" if a["auroc"].get("variant") else "",
            )
            for a in auroc_arms
        )
        cmp_key = f"{tab['key']}-auroc-mean"
        if cmp_key in specs:
            panels += _panel(
                cmp_key, specs[cmp_key],
                "The mean row of each arm's comparison CSV, on one axis for direct comparison.",
            )
        colors = _split_colors(tab)
        auroc_html = (
            "<h2>Eval AUROC per iteration</h2>"
            f'<div class="charts">{panels}</div>'
            + "".join(_auroc_table(a, colors) for a in auroc_arms)
        )

    rt_html = ""
    if rt_arms:
        panels = "".join(
            _panel(f"{a['key']}-rt", specs[f"{a['key']}-rt"], _RT_CAPTION)
            for a in rt_arms
        )
        cmp_key = f"{tab['key']}-rt-pooled"
        if cmp_key in specs:
            panels += _panel(
                cmp_key, specs[cmp_key],
                "Both error types pooled per arm, on one axis for direct comparison.",
            )
        totals = []
        for a in rt_arms:
            s, n = sum(a["rt"]["pooled"]["succ"]), sum(a["rt"]["pooled"]["att"])
            totals.append(
                f"{html.escape(a['label'])}: {s} successes over {n} scored attempts "
                f"({s / n * 100:.1f}%) across {a['rt']['n_rounds']} rounds"
            )
        rt_html = (
            "<h2>Red-team success rate per iteration</h2>"
            f'<p class="note">{" &middot; ".join(totals)}</p>'
            f'<div class="charts">{panels}</div>'
            + "".join(_rt_table(a) for a in rt_arms)
        )

    return auroc_html + rt_html, specs


_CSS = """
:root{--bg:#fbfbfd;--fg:#16181d;--muted:#666e7a;--line:#e2e5ea;--card:#fff;
  --chip:#eef1f5;--accent:#2f5fd0;}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--fg:#e6e8ec;--muted:#98a1ad;
  --line:#2a2f38;--card:#181b21;--chip:#232830;--accent:#7ba1ff;}}
:root[data-theme="light"]{--bg:#fbfbfd;--fg:#16181d;--muted:#666e7a;--line:#e2e5ea;
  --card:#fff;--chip:#eef1f5;--accent:#2f5fd0;}
:root[data-theme="dark"]{--bg:#111318;--fg:#e6e8ec;--muted:#98a1ad;--line:#2a2f38;
  --card:#181b21;--chip:#232830;--accent:#7ba1ff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
header.page{padding:28px 32px 16px;border-bottom:1px solid var(--line)}
h1{margin:0 0 6px;font-size:21px;font-weight:650;letter-spacing:-.01em}
h2{margin:26px 0 2px;font-size:16px;font-weight:650}
h2:first-child{margin-top:8px}
.sub{color:var(--muted);max-width:88ch;margin:0 0 4px}
.note{color:var(--muted);font-size:12.5px;margin:6px 0 0}
main{padding:8px 32px 64px;max-width:1600px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}
.themebtn{position:fixed;top:14px;right:14px;z-index:9;font:inherit;font-size:12px;
  font-weight:600;cursor:pointer;background:var(--card);color:var(--muted);
  border:1px solid var(--line);border-radius:999px;padding:6px 12px}
.empty{color:var(--muted);padding:40px 0;font-style:italic}

/* ---- tabs ---- */
nav.tabs{display:flex;gap:4px;padding:12px 32px 0;border-bottom:1px solid var(--line)}
nav.tabs button{font:inherit;font-size:13px;font-weight:600;padding:8px 16px;border:0;
  background:transparent;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent}
nav.tabs button:hover{color:var(--fg)}
nav.tabs button.active{color:var(--fg);border-bottom-color:var(--accent)}
[hidden]{display:none !important}

/* ---- charts: roles from the dataviz reference palette (validated, adjacent
       pairlist, both modes). Light mode puts slots 3-4 under 3:1 vs the chart
       surface, so the relief rule applies — direct labels and the table view
       below are mandatory, not optional. ---- */
.viz-root{color-scheme:light;
  --surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--grid:#e6e6e2;
  --series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;
  --series-5:#e87ba4;--series-6:#008300;--series-7:#4a3aa7;--series-8:#e34948;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
  color-scheme:dark;
  --surface-1:#1a1a19;--text-primary:#ffffff;--text-secondary:#c3c2b7;--grid:#33332f;
  --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;
  --series-5:#d55181;--series-6:#008300;--series-7:#9085e9;--series-8:#e66767;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
  --surface-1:#1a1a19;--text-primary:#ffffff;--text-secondary:#c3c2b7;--grid:#33332f;
  --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;
  --series-5:#d55181;--series-6:#008300;--series-7:#9085e9;--series-8:#e66767;}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:18px;margin:14px 0 8px}
.panel{background:var(--surface-1);border:1px solid var(--line);border-radius:10px;padding:14px 16px 8px;position:relative}
/* a 16-point variant sweep needs the whole row, or the x labels collapse */
.panel.wide{grid-column:1/-1}
.panel h3{margin:0 0 2px;font-size:13px;font-weight:650;color:var(--text-primary)}
.panel .cap{margin:0 0 8px;font-size:11.5px;color:var(--text-secondary)}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin:0 0 6px}
.legend span{display:flex;gap:5px;align-items:center;font-size:11px;color:var(--text-secondary)}
.legend i{width:12px;height:2.5px;border-radius:2px;display:inline-block}
.legend i.dash{background:repeating-linear-gradient(90deg,currentColor 0 4px,transparent 4px 7px)}
.panel svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:absolute;pointer-events:none;background:var(--card);border:1px solid var(--line);
  border-radius:8px;padding:8px 10px;font-size:11.5px;box-shadow:0 4px 14px rgba(0,0,0,.14);
  z-index:9;min-width:170px;opacity:0;transition:opacity .08s}
.tip.on{opacity:1}
.tip b{display:block;margin-bottom:4px;font-size:11px;color:var(--text-secondary);font-weight:600}
.tip .row{display:flex;align-items:center;gap:6px;justify-content:space-between}
.tip .row span{display:flex;align-items:center;gap:5px;color:var(--text-primary)}
.tip .row i{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none}
.tip .row b.v{margin:0;font-variant-numeric:tabular-nums;color:var(--text-primary);font-weight:600}
.tip .row b.v em{font-style:normal;font-weight:400;color:var(--text-secondary);margin-left:5px}
.tablewrap{overflow-x:auto;max-width:100%}
table.data{border-collapse:collapse;font-size:12px;margin:10px 0 30px;width:100%}
table.data th,table.data td{border:1px solid var(--line);padding:4px 9px;text-align:right;
  font-variant-numeric:tabular-nums}
table.data th{background:var(--chip);font-weight:600}
table.data td:first-child,table.data th:first-child{text-align:left}
table.data tr.mean td{font-weight:650;background:var(--chip)}
table.data caption{caption-side:top;text-align:left;font-weight:650;padding:10px 0 6px;font-size:13px}
table.data .frac{color:var(--muted);font-weight:400;margin-left:7px}
/* best-scoring variant: a tint plus a left/right rule, so the column is findable
   without relying on hue alone */
table.data th.best,table.data td.best{background:color-mix(in srgb,var(--accent) 14%,transparent);
  border-left:2px solid var(--accent);border-right:2px solid var(--accent);font-weight:650}
table.data tr.mean td.best{background:color-mix(in srgb,var(--accent) 26%,transparent)}
.swatch{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:baseline}
"""

_JS = """
/* ---------------------------------------------------------------- theme */
const root=document.documentElement, tbtn=document.getElementById('themebtn');
function setTheme(t){root.setAttribute('data-theme',t);
  try{localStorage.setItem('theme',t)}catch(e){}
  tbtn.textContent=t==='dark'?'light mode':'dark mode';}
let t0=null; try{t0=localStorage.getItem('theme')}catch(e){}
setTheme(t0||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'));
tbtn.addEventListener('click',()=>setTheme(root.getAttribute('data-theme')==='dark'?'light':'dark'));

/* ---------------------------------------------------------------- tabs */
const tabs=[...document.querySelectorAll('nav.tabs button')];
tabs.forEach(b=>b.addEventListener('click',()=>{
  tabs.forEach(x=>x.classList.toggle('active',x===b));
  document.querySelectorAll('main > section').forEach(s=>{s.hidden=(s.id!=='tab-'+b.dataset.tab);});
}));

/* ------------------------------------------------------------ line charts
   Form: trend over time, distinct components -> multi-line, categorical hue per
   component. The aggregate series (mean AUROC / both error types) is drawn as
   EMPHASIS — neutral ink, dashed — rather than another categorical slot: it is
   an aggregate of the others, not a peer. */
const NS='http://www.w3.org/2000/svg';
const W=560;
function el(n,a){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;}
function fmt(v,pct){return v==null?'—':(pct?(v*100).toFixed(1)+'%':v.toFixed(3));}

function draw(panel){
  const spec=CHARTS[panel.dataset.chart];
  const host=panel.querySelector('.plot');
  host.textContent='';
  // a rotated x axis (variant sweeps) needs a deeper bottom margin than iterN labels
  // The right gutter holds the direct labels, so size it to the longest one
  // instead of a constant: at 4 splits every label fit inside 104, but the
  // 7-split instruction tab (and any arm-comparison panel labelled by model
  // name) would otherwise run its text off the panel. ~6.1px/char at 10.5px,
  // clamped so one long name can't eat the plot band.
  const lab=Math.max(0,...spec.series.map(s=>s.short.length));
  const gutter=Math.min(150,Math.max(104,Math.round(lab*6.1)+16));
  const M={t:14,r:gutter,b:spec.rotx?66:36,l:52}, H=spec.rotx?360:330;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':spec.title});
  const iw=W-M.l-M.r, ih=H-M.t-M.b;
  const n=spec.iters.length;
  const x=i=>M.l+(n===1?iw/2:iw*i/(n-1));
  const y=v=>M.t+ih*(1-(v-spec.ymin)/(spec.ymax-spec.ymin));

  // recessive grid + y ticks
  for(const t of spec.ticks){
    svg.appendChild(el('line',{x1:M.l,x2:M.l+iw,y1:y(t),y2:y(t),stroke:'var(--grid)','stroke-width':1}));
    const lb=el('text',{x:M.l-8,y:y(t)+3.5,'text-anchor':'end',fill:'var(--text-secondary)','font-size':10.5});
    lb.textContent=spec.pct?Math.round(t*100)+'%':t.toFixed(2); svg.appendChild(lb);
  }
  spec.iters.forEach((it,i)=>{
    const yb=M.t+ih+(spec.rotx?13:16);
    const lb=el('text',{x:x(i),y:yb,'text-anchor':spec.rotx?'end':'middle',
      fill:'var(--text-secondary)','font-size':spec.rotx?10:11});
    if(spec.rotx)lb.setAttribute('transform',`rotate(-45 ${x(i)} ${yb})`);
    lb.textContent=it; svg.appendChild(lb);
  });

  // the sweep's best variant: a rule the eye can find without reading every point
  if(spec.best_idx!=null){
    svg.appendChild(el('line',{x1:x(spec.best_idx),x2:x(spec.best_idx),y1:M.t-2,y2:M.t+ih,
      stroke:'var(--text-primary)','stroke-width':1,'stroke-dasharray':'2 3',opacity:.5}));
    const bl=el('text',{x:x(spec.best_idx),y:M.t-5,'text-anchor':'middle',
      fill:'var(--text-primary)','font-size':9.5,'font-weight':700});
    bl.textContent='best'; svg.appendChild(bl);
  }

  // series
  const labels=[];
  spec.series.forEach(s=>{
    const pts=s.values.map((v,i)=>v==null?null:[x(i),y(v)]).filter(Boolean);
    if(!pts.length)return;
    svg.appendChild(el('path',{d:'M'+pts.map(p=>p.join(' ')).join('L'),fill:'none',stroke:s.color,
      'stroke-width':s.emphasis?2.5:2,'stroke-linejoin':'round','stroke-linecap':'round',
      ...(s.emphasis?{'stroke-dasharray':'6 4'}:{})}));
    pts.forEach(p=>{                       // 2px surface ring keeps overlaps readable
      svg.appendChild(el('circle',{cx:p[0],cy:p[1],r:4,fill:s.color,stroke:'var(--surface-1)','stroke-width':2}));
    });
    const last=s.values.reduce((a,v,i)=>v==null?a:i,-1);
    if(last>=0)labels.push({y:y(s.values[last]),y0:y(s.values[last]),x0:x(last),
                            text:s.short,color:s.color,emphasis:s.emphasis});
  });

  // direct labels (mandatory at 4 series, and by the relief rule in light mode):
  // de-overlap greedily so no two collide.
  labels.sort((a,b)=>a.y-b.y);
  const GAP=13;
  for(let i=1;i<labels.length;i++)
    if(labels[i].y-labels[i-1].y<GAP)labels[i].y=labels[i-1].y+GAP;
  const over=labels.length?labels[labels.length-1].y-(M.t+ih):0;
  if(over>0)labels.forEach(l=>l.y-=over);
  labels.forEach(l=>{
    // a displaced label needs a leader back to its own endpoint, or three stacked
    // labels stop saying which line each belongs to
    if(Math.abs(l.y-l.y0)>2){
      svg.appendChild(el('polyline',{points:`${l.x0+5},${l.y0} ${M.l+iw+3},${l.y0} ${M.l+iw+6},${l.y}`,
        fill:'none',stroke:l.color,'stroke-width':1,opacity:.55}));
    }
    const t=el('text',{x:M.l+iw+9,y:l.y+3.5,fill:'var(--text-primary)','font-size':10.5,
      'font-weight':l.emphasis?700:600});
    t.textContent=l.text; svg.appendChild(t);
  });

  // hover: crosshair + one tooltip listing every series at that iteration
  const cross=el('line',{x1:0,x2:0,y1:M.t,y2:M.t+ih,stroke:'var(--text-secondary)',
    'stroke-width':1,'stroke-dasharray':'3 3',opacity:0});
  svg.appendChild(cross);
  const hit=el('rect',{x:M.l-14,y:M.t,width:iw+28,height:ih,fill:'transparent'});
  svg.appendChild(hit);
  const tip=panel.querySelector('.tip');
  hit.addEventListener('mousemove',ev=>{
    const r=svg.getBoundingClientRect(), sx=(ev.clientX-r.left)*W/r.width;
    let i=Math.round((sx-M.l)/(n===1?1:iw/(n-1))); i=Math.max(0,Math.min(n-1,i));
    cross.setAttribute('x1',x(i));cross.setAttribute('x2',x(i));cross.setAttribute('opacity',1);
    tip.textContent='';
    const head=document.createElement('b'); head.textContent=spec.iters[i]; tip.appendChild(head);
    spec.series.forEach(s=>{                         // labels are data -> textContent
      const row=document.createElement('div'); row.className='row';
      const lhs=document.createElement('span');
      const dot=document.createElement('i'); dot.style.background=s.color;
      lhs.appendChild(dot); lhs.appendChild(document.createTextNode(s.short));
      const val=document.createElement('b'); val.className='v';
      val.textContent=fmt(s.values[i],spec.pct);
      if(s.extra&&s.extra[i]!=null){const e=document.createElement('em');e.textContent=s.extra[i];val.appendChild(e);}
      row.appendChild(lhs); row.appendChild(val); tip.appendChild(row);
    });
    tip.classList.add('on');
    const px=r.width*(x(i)/W);
    tip.style.left=Math.min(Math.max(px-85,4),r.width-186)+'px';
    tip.style.top=(ev.clientY-r.top+14)+'px';
  });
  hit.addEventListener('mouseleave',()=>{cross.setAttribute('opacity',0);tip.classList.remove('on');});
  host.appendChild(svg);
}
// Everything is viewBox-sized, so a hidden tab draws exactly as a visible one.
document.querySelectorAll('.panel[data-chart]').forEach(draw);
"""


def _render_page(tabs: list[dict], title: str, subtitle: str) -> str:
    specs: dict[str, dict] = {}
    nav, sections = "", ""
    for i, tab in enumerate(tabs):
        body, tab_specs = _render_tab(tab)
        specs.update(tab_specs)
        nav += (
            f'<button data-tab="{html.escape(tab["key"])}"'
            f'{" class=\'active\'" if i == 0 else ""}>{html.escape(tab["title"])}</button>'
        )
        src = tab["git_ref"] or "working tree"
        arms = ", ".join(a["label"] for a in tab["arms"])
        sections += (
            f'<section id="tab-{html.escape(tab["key"])}"{"" if i == 0 else " hidden"}>'
            f'<p class="note">Arms: {html.escape(arms)} &middot; read from '
            f"<code>{html.escape(src)}</code></p>{body}</section>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style></head>
<body class="viz-root">
<button class="themebtn" id="themebtn">dark mode</button>
<header class="page">
  <h1>{html.escape(title)}</h1>
  <p class="sub">{html.escape(subtitle)}</p>
  <p class="sub">The two sections share an x axis: red-team iteration N attacks the
  probe scored as <code>iterN</code> in the AUROC charts, and its successes train
  <code>iter{{N+1}}</code>. Success rate is over <em>scored</em> attempts — exact
  duplicates are never written to the log, and near-duplicate openers are rejected
  before the probe runs, so neither is counted.</p>
</header>
<nav class="tabs">{nav}</nav>
<main>{sections}</main>
<script>
const CHARTS={json.dumps(specs)};
{_JS}
</script>
</body></html>"""


# -------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tab", action="append", required=True,
        help="KEY:TITLE[:GIT_REF] (repeatable). GIT_REF reads that tab's inputs "
             "from a git ref instead of the worktree.",
    )
    ap.add_argument(
        "--arm", action="append", required=True,
        help="TAB_KEY:LABEL:COMPARISON_CSV:PROBING_STEM (repeatable). PROBING_STEM "
             "is the attempt-JSONL path without the _fp/_fn suffix.",
    )
    ap.add_argument(
        "--variant-arm", action="append", default=[],
        help="TAB_KEY:LABEL:DIR[:SPLIT_NAMES] (repeatable). DIR holds one "
             "single-eval CSV per probe variant (dataset,auroc,...). SPLIT_NAMES "
             "is a comma-separated list renaming the splits positionally, for "
             "dirs whose dataset ids are anonymized. AUROC only — variant sweeps "
             "carry no attempt JSONL.",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output HTML path")
    ap.add_argument(
        "--title", default="Eval AUROC vs. red-team success rate",
    )
    ap.add_argument(
        "--subtitle",
        default="One tab per probe concept; within a tab, one panel per attacker arm.",
    )
    args = ap.parse_args()

    tabs: list[dict] = []
    by_key: dict[str, dict] = {}
    for spec in args.tab:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            ap.error(f"--tab must be KEY:TITLE[:GIT_REF], got {spec!r}")
        key, title = parts[0], parts[1]
        tab = {"key": key, "title": title,
               "git_ref": parts[2] if len(parts) == 3 else None, "arms": []}
        tabs.append(tab)
        by_key[key] = tab

    for spec in args.arm:
        parts = spec.split(":")
        if len(parts) != 4:
            ap.error(f"--arm must be TAB_KEY:LABEL:COMPARISON_CSV:PROBING_STEM, got {spec!r}")
        tab_key, label, csv_path, stem = parts
        if tab_key not in by_key:
            ap.error(f"--arm references unknown tab key {tab_key!r}")
        tab = by_key[tab_key]
        ref = tab["git_ref"]
        arm = {
            "key": f"{tab_key}-{len(tab['arms'])}",
            "label": label,
            "auroc": _read_comparison_csv(csv_path, ref),
            "rt": _read_redteam(stem, ref),
        }
        if arm["auroc"] is None:
            print(f"  [{tab_key}/{label}] WARNING: no comparison CSV at {csv_path}",
                  file=sys.stderr)
        if arm["rt"] is None:
            print(f"  [{tab_key}/{label}] WARNING: no attempt JSONL at {stem}_{{fp,fn}}.jsonl",
                  file=sys.stderr)
        tab["arms"].append(arm)
        a, r = arm["auroc"], arm["rt"]
        print(
            f"  [{tab_key}/{label}] "
            f"AUROC: {len(a['splits']) if a else 0} splits x {len(a['iters']) if a else 0} iters; "
            f"red-team: {sum(r['pooled']['succ']) if r else 0}/"
            f"{sum(r['pooled']['att']) if r else 0} over "
            f"{len(r['iters']) if r else 0} iters"
        )

    for spec in args.variant_arm:
        parts = spec.split(":")
        if len(parts) not in (3, 4):
            ap.error(f"--variant-arm must be TAB_KEY:LABEL:DIR[:SPLIT_NAMES], got {spec!r}")
        tab_key, label, dir_path = parts[0], parts[1], parts[2]
        names = [s.strip() for s in parts[3].split(",")] if len(parts) == 4 else None
        if tab_key not in by_key:
            ap.error(f"--variant-arm references unknown tab key {tab_key!r}")
        tab = by_key[tab_key]
        # Read from disk regardless of the tab's git_ref: a sweep directory is an
        # external artifact, not a file the repo has a revision of.
        auroc = _read_variant_dir(dir_path, names)
        if auroc is None:
            print(f"  [{tab_key}/{label}] WARNING: no variant CSVs under {dir_path}",
                  file=sys.stderr)
            continue
        tab["arms"].append({
            "key": f"{tab_key}-{len(tab['arms'])}",
            "label": label,
            "auroc": auroc,
            "rt": None,
        })
        print(
            f"  [{tab_key}/{label}] AUROC: {len(auroc['splits'])} splits x "
            f"{len(auroc['iters'])} variants; best = {auroc['best_label']} "
            f"({auroc['best_mean']:.4f})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_render_page(tabs, args.title, args.subtitle), encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
