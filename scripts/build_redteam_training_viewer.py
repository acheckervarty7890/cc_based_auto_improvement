#!/usr/bin/env python
"""Build a self-contained HTML viewer of the red-team data that trained each probe.

One file covering several runs: a run selector, then a tab per iteration. Each tab
shows only the samples **new to that iteration** — the per-iteration dumps
(``<probe-dir>/redteam_postprocessed_iter{N}.jsonl``) are near-cumulative, so the
tab is a set-difference against every earlier iteration keyed on the conversation's
own content.

Iteration N's dump is what trained ``probe_iter{N}.pkl``, and it is built from the
red-team successes of iterations ``0..N-1`` — so tab "iter N" is sourced from
red-team iteration N-1. The tab header says so.

Every row is one of two kinds, tagged in the UI and shown **side by side** — a
success in the left column, the contrastive rewrite generated from it in the right:

- **red-team success** — a conversation the attacker produced where probe and judge
  disagreed in the ``error_type`` direction. Joined by conversation text against the
  run's attempt logs (``<results-dir>/*.jsonl``, excluding the ``.summaries`` /
  ``.runlog`` / ``.iteration_memos`` sidecars) to recover the **judge's reason**,
  plus probe score, judge label/confidence, attacker model, error type and round.
- **contrastive pair** — an opposite-class conversation ``generate_contrastive_dataset``
  wrote from one of those successes. Joined against ``<probe-dir>/contrastive_cache.jsonl``
  to recover the generator's **``generation_explanation``**, its model, and the source
  conversation it was derived from (which is what the pairing is keyed on).

The kind and label filters select which *sides* of a pair are shown, so filtering to
e.g. contrastive-only collapses each card to that one column; the search matches at the
card level, so a hit in one half still shows the other next to it. A half with no
partner in the same iteration (and every row of a ``_nocontrastive`` run) renders as a
single full-width column, with the rewrite's source inlined in a collapsed block.

Note ``filter_dataset`` re-runs over the whole accumulated set before each retrain, so
the number of new originals in a tab is generally *lower* than the number of successes
logged for that red-team iteration; the stats row reports both. A sample the filter
dropped one iteration and kept the next legitimately shows up as new in the later tab.

``--tiers RUN_LABEL:PATH`` attaches a tier JSONL to one run, adding **a tab per tier**
after the iteration tabs. A tier file classifies the run's *pairs* (rather than slicing
by iteration), one row per pair::

    {tier, tier_description, pair_id, in_iters,
     original: {inputs, label}, contrastive: {inputs, label}}

Each tier tab renders the same cards, except a card is always a whole pair — the two
halves are linked directly instead of being reconstructed by ``pair_up`` — because a
tier is a statement about the pair, and pruning a tier drops both halves together. The
two sides arrive in different label vocabularies (the dump's canonical
``positive``/``negative`` vs the probe's own class strings), so both are folded back to
the canonical pair. ``--tiers-note`` adds a sentence shown above every tier tab.

``--variant "LABEL:CSV"`` (repeatable) adds a **Pruning variants** tab: retrains of one
run's data with a slice removed, each supplied as a comparison CSV. They share the run's
eval splits and iteration axis, so they plot on the same chart engine — one line per arm,
with chips to pick the metric and which split (or their mean) to plot. Prefix a label with
``*`` to mark the arm the others are measured against: it is drawn thicker and the table's
last column becomes each arm's delta against it at the final iteration, rather than the
default "how far this line moved from its own first iteration". Capped at 6 arms, the
number of categorical slots in the palette — cycling hues would make two arms
indistinguishable.

A final **Eval scores** tab charts probe quality per iteration from each run's comparison
CSV (auto-discovered as ``<results-dir>/*comparison*.csv``, or given as a 4th ``--run``
field): one chart comparing the runs' means, one breaking the selected run into its
**components** — a line per eval split plus the mean — with AUROC / accuracy / TPR@1%FPR
chips, hover crosshair, and a table. Its rounds are ``iter0..iterK`` where **iter0 is the
baseline probe**, trained before any red-teaming, so it has no sample tab; ``iterN`` is the
probe trained on the samples in the "Iteration N" tab (cumulatively).

Usage:
    .venv_claude/bin/python scripts/build_redteam_training_viewer.py \
        --run "guidance:probes/hs_llama1b_deepseekv4pro_guidance:results_hs_llama1b_deepseekv4pro_guidance" \
        --run "no guidance:probes/hs_llama1b_deepseekv4pro_noguidance:results_hs_llama1b_deepseekv4pro_noguidance" \
        --tiers "guidance:probes/hs_llama1b_deepseekv4pro_guidance_pruned/toolace_similarity_tiers.jsonl" \
        --title "deepseek-v4-pro -> llama-1b high-stakes probe" \
        --out viewers/deepseekv4pro_redteam_training_viewer.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path

SIDECAR_RE = re.compile(r"\.(summaries|runlog|iteration_memos)\.jsonl$")
ITER_RE = re.compile(r"redteam_postprocessed_iter(\d+)\.jsonl$")
ROUND_RE = re.compile(r"(\d+)")

EVAL_METRICS = ("auroc", "accuracy", "tpr_at_fpr")

KIND_SUCCESS, KIND_CONTRASTIVE, KIND_UNKNOWN = 0, 1, 2


# ---------------------------------------------------------------- data loading

def canon(messages) -> str:
    """Canonical text of a conversation — the identity of a sample."""
    if isinstance(messages, dict):
        messages = messages.get("messages", messages.get("inputs", []))
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages
    return json.dumps(
        [[m.get("role", ""), m.get("content", "")] for m in messages],
        ensure_ascii=False,
    )


def as_messages(raw) -> list[dict]:
    if isinstance(raw, dict):
        raw = raw.get("messages", raw.get("inputs", []))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [{"role": "user", "content": raw}]
    return [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in raw
    ]


def read_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_successes(results_dir: Path) -> dict[str, dict]:
    """canon(conversation) -> the attempt record, for successful attempts only."""
    out: dict[str, dict] = {}
    if not results_dir.is_dir():
        return out
    for path in sorted(results_dir.glob("*.jsonl")):
        if SIDECAR_RE.search(path.name):
            continue
        for rec in read_jsonl(path):
            if rec.get("success"):
                out[canon(rec.get("sample"))] = rec
    return out


def load_contrastive(probe_dir: Path) -> dict[str, dict]:
    """canon(generated conversation) -> its cache record (carries the explanation)."""
    out: dict[str, dict] = {}
    path = probe_dir / "contrastive_cache.jsonl"
    if not path.is_file():
        return out
    for row in read_jsonl(path):
        rec = row.get("record", row)
        out[canon(rec.get("inputs"))] = rec
    return out


def iteration_files(probe_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for path in probe_dir.glob("redteam_postprocessed_iter*.jsonl"):
        m = ITER_RE.search(path.name)
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def find_comparison_csv(results_dir: Path) -> Path | None:
    hits = sorted(results_dir.glob("*comparison*.csv")) if results_dir.is_dir() else []
    if len(hits) > 1:
        print(f"  ! {results_dir}: {len(hits)} comparison CSVs, using {hits[0].name}")
    return hits[0] if hits else None


def load_eval(csv_path: Path | None) -> dict | None:
    """Per-iteration eval scores from a comparison CSV.

    ``round`` is ``iter{N}`` and ``dataset`` is an eval split or ``mean``. Note the
    rounds run ``iter0..iterK``: ``iter0`` is the **baseline** probe, before any
    red-team retraining, so it has no corresponding sample tab.
    """
    if csv_path is None or not csv_path.is_file():
        return None
    rounds: dict[int, dict[str, dict[str, float]]] = {}
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            m = ROUND_RE.search(row.get("round", ""))
            if not m:
                continue
            scores = {}
            for metric in EVAL_METRICS:
                try:
                    scores[metric] = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
            if scores:
                rounds.setdefault(int(m.group(1)), {})[row["dataset"]] = scores
    if not rounds:
        return None
    ordered = [{"n": n, "scores": rounds[n]} for n in sorted(rounds)]
    splits = [s for s in dict.fromkeys(k for r in ordered for k in r["scores"])
              if s != "mean"]
    return {"rounds": ordered, "splits": splits, "csv": str(csv_path)}


def logged_success_counts(successes: dict[str, dict]) -> dict[tuple[int, str], int]:
    counts: dict[tuple[int, str], int] = {}
    for rec in successes.values():
        key = (int(rec.get("iteration", -1)), str(rec.get("error_type", "")))
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------- run assembly

class ConvPool:
    """Distinct conversations stored once; items reference them by index."""

    def __init__(self) -> None:
        self.index: dict[str, int] = {}
        self.convs: list[list[dict]] = []

    def add(self, raw) -> int:
        key = canon(raw)
        idx = self.index.get(key)
        if idx is None:
            idx = len(self.convs)
            self.index[key] = idx
            self.convs.append(as_messages(raw))
        return idx


def pair_up(items: list[dict]) -> list[dict]:
    """Order each contrastive rewrite right after its source and link the two.

    The dump is ordered originals-then-generated, so left alone a page of cards is
    all one kind and a pair is never visible together. Each half gets ``pi``, the
    position of its partner in the returned list (``-1`` when it has none — the
    ``_nocontrastive`` runs, or a half whose partner landed in another iteration).
    Only the first rewrite of a source is paired; any extra renders on its own.
    """
    here = {x["c"] for x in items}
    children: dict[int, list[dict]] = {}
    for x in items:
        parent = x.get("oc", -1) if x["k"] == KIND_CONTRASTIVE else -1
        if parent >= 0 and parent in here:
            children.setdefault(parent, []).append(x)
    hoisted = {id(kid) for kids in children.values() for kid in kids}

    out: list[dict] = []
    for x in items:
        if id(x) in hoisted:
            continue  # emitted right after its source instead
        kids = children.get(x["c"], ())
        at = len(out)
        out.append(x)
        out.extend(kids)
        if kids:
            x["pi"] = at + 1
            kids[0]["pi"] = at

    for x in out:
        x.setdefault("pi", -1)
    return out


def enrich(item: dict, key: str, successes: dict, contrastive: dict, pool: ConvPool) -> dict:
    """Attach provenance to a bare {c, l} item: judge verdict, or generator reason."""
    hit = successes.get(key)
    cpair = contrastive.get(key)

    if hit is not None:
        item["k"] = KIND_SUCCESS
        item["sc"] = round(float(hit.get("probe_score", 0.0)), 4)
        item["pp"] = bool(hit.get("probe_predicts_positive"))
        item["jl"] = hit.get("judge_label", "")
        item["jc"] = hit.get("judge_confidence", "")
        item["jr"] = hit.get("judge_reason", "")
        item["am"] = hit.get("attacker_model", "")
        item["et"] = hit.get("error_type", "")
        item["ai"] = hit.get("iteration", -1)
        item["rd"] = hit.get("round", -1)
    elif cpair is not None:
        item["k"] = KIND_CONTRASTIVE
        item["ge"] = cpair.get("generation_explanation", "")
        item["gm"] = cpair.get("generation_model", "")
        item["ol"] = cpair.get("original_label", "")
        src = cpair.get("original_messages")
        item["oc"] = pool.add(src) if src else -1
        # the source success's own judge reason, when we have it
        shit = successes.get(canon(src)) if src else None
        item["sr"] = shit.get("judge_reason", "") if shit else ""
        item["se"] = shit.get("error_type", "") if shit else ""
    else:
        item["k"] = KIND_UNKNOWN
    return item


def build_tiers(path: Path, successes: dict, contrastive: dict, pool: ConvPool,
                pos_label: str, neg_label: str) -> list[dict]:
    """Group a toolace-similarity tier file into one tab per tier.

    Rows are whole pairs, so unlike the iteration tabs the two halves are always
    present together and are linked directly rather than reconstructed by ``pair_up``.

    The two sides arrive in different label vocabularies — an original carries the
    dump's canonical ``positive``/``negative``, a rewrite carries the probe's own
    ``high-stakes``/``low-stakes`` — so both are folded back to the canonical pair,
    which is what the label chips and pill colours downstream expect.
    """
    if not path or not path.is_file():
        return []

    canon_label = {str(pos_label).lower(): "positive", "positive": "positive",
                   str(neg_label).lower(): "negative", "negative": "negative"}

    groups: dict[str, dict] = {}
    for row in read_jsonl(path):
        tier = str(row.get("tier", "?"))
        g = groups.setdefault(tier, {
            "id": tier,
            "desc": row.get("tier_description", ""),
            "items": [],
            "iters": {},
        })
        for side in ("original", "contrastive"):
            side_row = row.get(side) or {}
            msgs = side_row.get("inputs")
            raw = str(side_row.get("label", ""))
            item = enrich(
                {"c": pool.add(msgs), "l": canon_label.get(raw.lower(), raw),
                 "tp": row.get("pair_id", -1), "tf": sorted(row.get("in_iters") or [])},
                canon(msgs), successes, contrastive, pool,
            )
            g["items"].append(item)
        # the two halves of a pair are adjacent by construction
        n = len(g["items"])
        g["items"][n - 2]["pi"] = n - 1
        g["items"][n - 1]["pi"] = n - 2
        # the dumps are near-cumulative, so attribute a pair to the iteration it
        # first entered the training set — the same rule the iteration tabs use
        seen_in = row.get("in_iters") or []
        if seen_in:
            first = str(min(seen_in))
            g["iters"][first] = g["iters"].get(first, 0) + 1

    for g in groups.values():
        g["items"].sort(key=lambda x: (x["tp"], x["k"]))
        pos = {id(x): i for i, x in enumerate(g["items"])}
        pairs = {}
        for x in g["items"]:
            pairs.setdefault(x["tp"], []).append(x)
        for sides in pairs.values():
            if len(sides) == 2:
                sides[0]["pi"], sides[1]["pi"] = pos[id(sides[1])], pos[id(sides[0])]
        g["pairs"] = len(pairs)

    return [groups[k] for k in sorted(groups)]


def build_run(label: str, probe_dir: Path, results_dir: Path, pool: ConvPool,
              csv_path: Path | None = None, tiers_path: Path | None = None) -> dict:
    successes = load_successes(results_dir)
    contrastive = load_contrastive(probe_dir)
    logged = logged_success_counts(successes)

    iters: list[dict] = []
    seen: set[str] = set()
    orphans = 0

    for n, path in iteration_files(probe_dir):
        rows = list(read_jsonl(path))
        items = []
        for row in rows:
            key = canon(row.get("inputs"))
            if key in seen:
                continue
            seen.add(key)

            item = enrich(
                {"c": pool.add(row.get("inputs")), "l": row.get("label", "")},
                key, successes, contrastive, pool,
            )
            if item["k"] == KIND_UNKNOWN:
                orphans += 1

            items.append(item)

        items = pair_up(items)

        # the red-team iteration this dump was built from
        src_iter = n - 1
        iters.append({
            "n": n,
            "src_iter": src_iter,
            "total": len(rows),
            "items": items,
            "logged": {
                "fp": logged.get((src_iter, "false_positive"), 0),
                "fn": logged.get((src_iter, "false_negative"), 0),
            },
        })

    # the human-readable class names, read off any attempt record
    any_rec = next(iter(successes.values()), {})
    pos_label = any_rec.get("pos_class_label", "positive")
    neg_label = any_rec.get("neg_class_label", "negative")

    return {
        "label": label,
        "probe_dir": str(probe_dir),
        "results_dir": str(results_dir),
        "pos_label": pos_label,
        "neg_label": neg_label,
        "orphans": orphans,
        "iters": iters,
        "tiers": build_tiers(tiers_path, successes, contrastive, pool, pos_label, neg_label),
        "eval": load_eval(csv_path if csv_path else find_comparison_csv(results_dir)),
    }


# ------------------------------------------------------------------- rendering

CSS = """
:root {
  color-scheme: light dark;
  --bg:#fff; --panel:#f6f7f9; --panel2:#eef0f3; --line:#d9dde3;
  --fg:#1a1d21; --muted:#666e79; --accent:#3f6fd8;
  --pos-bg:#fdeeee; --pos-fg:#9d2727; --neg-bg:#eaf5ee; --neg-fg:#1e6b3c;
  --suc-bg:#e9f0fd; --suc-fg:#28489c; --con-bg:#f5ecfb; --con-fg:#6b3a97;
  --warn:#8a5b00; --warn-bg:#fdf3dc;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
  --mean:#1a1d21; --chance:#b9b8b2;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
    --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
    --pos-bg:#2c1c1e; --pos-fg:#ff9a9a; --neg-bg:#17281d; --neg-fg:#86e0a5;
    --suc-bg:#1b2540; --suc-fg:#9db8f5; --con-bg:#2a1f38; --con-fg:#cfa8ee;
    --warn:#f0c060; --warn-bg:#33290f;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
    --mean:#e5e9ee; --chance:#4a4a47;
  }
}
:root[data-theme="light"] {
  --bg:#fff; --panel:#f6f7f9; --panel2:#eef0f3; --line:#d9dde3;
  --fg:#1a1d21; --muted:#666e79; --accent:#3f6fd8;
  --pos-bg:#fdeeee; --pos-fg:#9d2727; --neg-bg:#eaf5ee; --neg-fg:#1e6b3c;
  --suc-bg:#e9f0fd; --suc-fg:#28489c; --con-bg:#f5ecfb; --con-fg:#6b3a97;
  --warn:#8a5b00; --warn-bg:#fdf3dc;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
  --mean:#1a1d21; --chance:#b9b8b2;
}
:root[data-theme="dark"] {
  --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
  --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
  --pos-bg:#2c1c1e; --pos-fg:#ff9a9a; --neg-bg:#17281d; --neg-fg:#86e0a5;
  --suc-bg:#1b2540; --suc-fg:#9db8f5; --con-bg:#2a1f38; --con-fg:#cfa8ee;
  --warn:#f0c060; --warn-bg:#33290f;
  /* the chart steps belong here too: without them a light-OS viewer who toggles to
     dark keeps the light series colours and a near-black --mean on a near-black bg */
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  --mean:#e5e9ee; --chance:#4a4a47;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1360px; margin:0 auto; padding:22px 18px 90px; }
h1 { margin:0 0 4px; font-size:21px; letter-spacing:-.01em; }
.sub { color:var(--muted); font-size:13px; }
.path { color:var(--muted); font-size:12px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; margin-top:3px; }
.note { margin:16px 0 0; padding:10px 13px; border-left:3px solid var(--accent);
  background:var(--panel); border-radius:0 6px 6px 0; font-size:13px; color:var(--muted); }
.note code, code { font-family:ui-monospace,Menlo,monospace; font-size:12px; }
.runbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:18px 0 0; }
.runbar label { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
select, .toolbar input[type=search] { padding:8px 11px; font-size:14px; background:var(--panel);
  color:var(--fg); border:1px solid var(--line); border-radius:7px; font-family:inherit; }
.tabs { display:flex; flex-wrap:wrap; gap:6px; margin-top:16px; border-bottom:1px solid var(--line); }
.tab { appearance:none; border:1px solid transparent; border-bottom:none; background:transparent;
  color:var(--muted); cursor:pointer; padding:9px 15px; font-size:14px; font-weight:600;
  border-radius:7px 7px 0 0; margin-bottom:-1px; font-family:inherit; }
.tab:hover { color:var(--fg); background:var(--panel); }
.tab[aria-selected="true"] { color:var(--fg); background:var(--bg);
  border-color:var(--line); border-bottom:1px solid var(--bg); }
.tab .cnt { font-weight:400; color:var(--muted); font-size:12px; margin-left:5px; }
.tabnote { margin:14px 0 0; font-size:13px; color:var(--muted); }
.stats { display:flex; flex-wrap:wrap; gap:10px; margin:14px 0 6px; }
.stat { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:9px 13px; min-width:112px; }
.stat .k { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.stat .v { font-size:19px; font-weight:650; margin-top:2px; }
.toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:14px 0 18px; }
.toolbar input[type=search] { flex:1 1 240px; min-width:180px; }
.chips { display:flex; gap:6px; flex-wrap:wrap; }
.chip { appearance:none; font-family:inherit; font-size:13px; cursor:pointer; padding:7px 12px;
  border-radius:999px; border:1px solid var(--line); background:var(--panel); color:var(--muted); }
.chip[aria-pressed="true"] { background:var(--accent); border-color:var(--accent); color:#fff; }
.card { border:1px solid var(--line); border-radius:10px; margin:0 0 14px; background:var(--panel);
  overflow:hidden; }
.pair { display:grid; grid-template-columns:repeat(var(--cols),minmax(0,1fr)); }
.col { display:flex; flex-direction:column; min-width:0; }
.col + .col { border-left:1px solid var(--line); }
@media (max-width:900px) {
  .pair { grid-template-columns:1fr; }
  .col + .col { border-left:none; border-top:2px solid var(--line); }
}
.col > header { display:flex; flex-wrap:wrap; gap:7px; align-items:center; padding:10px 13px;
  border-bottom:1px solid var(--line); }
.badge { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
  padding:3px 8px; border-radius:5px; }
.b-suc { background:var(--suc-bg); color:var(--suc-fg); }
.b-con { background:var(--con-bg); color:var(--con-fg); }
.b-unk { background:var(--warn-bg); color:var(--warn); }
.pill { font-size:12px; padding:3px 8px; border-radius:5px; background:var(--panel2); color:var(--muted); }
.pill.pos { background:var(--pos-bg); color:var(--pos-fg); font-weight:600; }
.pill.neg { background:var(--neg-bg); color:var(--neg-fg); font-weight:600; }
.idx { margin-left:auto; font-size:12px; color:var(--muted); font-family:ui-monospace,Menlo,monospace; }
.conv { background:var(--bg); padding:4px 13px 10px; flex:1 0 auto; }
.msg { margin:9px 0 0; }
.msg .role { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); margin-bottom:3px; }
.msg .body { white-space:pre-wrap; overflow-wrap:anywhere; font-size:14px; }
.msg.system .body { color:var(--muted); font-style:italic; }
.why { padding:10px 13px; border-top:1px solid var(--line); font-size:13.5px; }
.why .k { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); margin-bottom:3px; }
.why p { margin:0; overflow-wrap:anywhere; }
details.src { border-top:1px solid var(--line); }
details.src > summary { padding:9px 13px; cursor:pointer; font-size:13px; color:var(--muted);
  user-select:none; }
details.src > summary:hover { color:var(--fg); }
details.src[open] > summary { border-bottom:1px solid var(--line); }
.meta { padding:9px 13px; border-top:1px solid var(--line); display:flex; flex-wrap:wrap; gap:6px; }
.empty { padding:36px 0; text-align:center; color:var(--muted); }

/* ---- eval tab: charts, legend, table (ported from build_contrastive_pairs_viewer) */
.evcard { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:16px 16px 12px; margin-top:18px; position:relative; }
.evcard h2 { font-size:15px; margin:0 0 3px; }
.evcard .src { color:var(--muted); font-size:11.5px; margin:0 0 10px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; }
.evcard .cap { color:var(--muted); font-size:12.5px; margin:0 0 12px; }
.chartbox { position:relative; }
.chartbox svg { display:block; width:100%; height:auto; overflow:visible; }
.legend { display:flex; flex-wrap:wrap; gap:9px 15px; margin:11px 2px 2px;
  font-size:12.5px; color:var(--muted); }
.legend .lg { display:inline-flex; align-items:center; gap:6px; }
.legend .sw { width:15px; height:3px; border-radius:2px; background:var(--c); }
.legend .sw.mean { height:0; width:17px; background:none; border-top:2.5px dashed var(--mean); }
.legend .sw.subject { height:4px; }
.s1 { --c:var(--s1); } .s2 { --c:var(--s2); } .s3 { --c:var(--s3); } .s4 { --c:var(--s4); }
.s5 { --c:var(--s5); } .s6 { --c:var(--s6); } .s7 { --c:var(--s7); } .s8 { --c:var(--s8); }
.smean { --c:var(--mean); }
.line { fill:none; stroke:var(--c); stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
.line.mean { stroke-width:2.5; stroke-dasharray:6 4; }
.line.subject { stroke-width:3; }
circle.pt { fill:var(--c); stroke:var(--panel); stroke-width:2; }
.axline { stroke:var(--line); stroke-width:1; }
.gridline { stroke:var(--line); stroke-width:1; opacity:.65; }
.chanceline { stroke:var(--chance); stroke-width:1; stroke-dasharray:3 3; }
.tick, .ylab { fill:var(--muted); font-size:11px; }
.endlab { font-size:10.5px; font-weight:650; fill:var(--c); paint-order:stroke;
  stroke:var(--panel); stroke-width:3px; stroke-linejoin:round; }
.crosshair { stroke:var(--line); stroke-width:1; stroke-dasharray:3 3; opacity:0; }
.crosshair.hi { opacity:1; }
.tip { position:absolute; pointer-events:none; z-index:5; min-width:172px; opacity:0;
  background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:8px 10px;
  font-size:12px; font-variant-numeric:tabular-nums; transition:opacity .08s;
  box-shadow:0 4px 16px rgba(0,0,0,.2); }
.tip .th { font-weight:700; color:var(--fg); margin-bottom:5px; }
.tip .row { display:flex; justify-content:space-between; gap:14px; color:var(--muted); }
.tip .row b { color:var(--fg); font-weight:600; }
.tip .dot { width:8px; height:8px; border-radius:2px; display:inline-block;
  margin-right:6px; vertical-align:middle; }
.tblwrap { overflow-x:auto; margin-top:6px; }
table.ev { border-collapse:collapse; width:100%; font-size:12.5px;
  font-variant-numeric:tabular-nums; }
table.ev th, table.ev td { padding:5px 9px; text-align:right; border-bottom:1px solid var(--line);
  white-space:nowrap; }
table.ev th:first-child, table.ev td:first-child { text-align:left; }
table.ev thead th { color:var(--muted); font-weight:600; }
table.ev tr.mean td { font-weight:700; border-top:1px solid var(--line); }
table.ev tr.subjectrow td { font-weight:700; }
table.ev td .key { width:9px; height:9px; border-radius:2px; display:inline-block;
  margin-right:7px; vertical-align:middle; background:var(--c); }
table.ev .up { color:var(--neg-fg); } table.ev .down { color:var(--pos-fg); }
.more { display:block; width:100%; padding:11px; margin-top:4px; font-family:inherit; font-size:14px;
  cursor:pointer; background:var(--panel); border:1px solid var(--line); border-radius:8px; color:var(--fg); }
"""

JS = """
const D = JSON.parse(document.getElementById('data').textContent);
const CONVS = D.convs, RUNS = D.runs, TIERNOTE = D.tier_note || '';
const VARIANTS = D.variants || [];
const KIND = ['red-team success', 'contrastive pair', 'unmatched'];
const PAGE = 25;  // cards, i.e. up to 2x that many samples

let runIdx = 0, iterIdx = 0, kindFilter = 'all', labelFilter = 'all', query = '', shown = PAGE;
let splitKey = 'mean';   // which eval split the pruning-variants chart plots

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s == null ? '' : s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function isPos(lbl) { return lbl === 'positive'; }
function labelText(run, lbl) {
  const human = isPos(lbl) ? run.pos_label : run.neg_label;
  return human && human !== lbl ? `${lbl} \\u00b7 ${human}` : lbl;
}

// iterIdx is an iteration index, 'tN' for the Nth tier tab, or 'eval'
function isTier() {
  return typeof iterIdx === 'string' && iterIdx[0] === 't' && iterIdx !== 'var';
}
function current() {
  const run = RUNS[runIdx];
  return isTier() ? run.tiers[+iterIdx.slice(1)] : run.iters[iterIdx];
}

// kind + label filters select which *sides* of a pair are shown; the search matches
// at the card level so a hit on one half still shows you the other to compare against.
function sidePasses(x) {
  if (kindFilter === 'suc' && x.k !== 0) return false;
  if (kindFilter === 'con' && x.k !== 1) return false;
  if (labelFilter !== 'all' && x.l !== labelFilter) return false;
  return true;
}

function hay(x) {
  return [
    CONVS[x.c].map((m) => m.content).join(' '),
    x.jr || '', x.ge || '', x.sr || '', x.am || '', x.et || '', x.gm || '',
  ].join(' ').toLowerCase();
}

/** Group items into cards ([success, rewrite] where paired), then apply the filters. */
function visibleCards() {
  const items = current().items;
  const q = query.trim().toLowerCase();
  const done = new Set();
  const cards = [];

  items.forEach((x, i) => {
    if (done.has(i)) return;
    done.add(i);
    let sides = [x];
    if (x.pi >= 0 && !done.has(x.pi)) {
      done.add(x.pi);
      sides.push(items[x.pi]);
    }
    sides.sort((a, b) => a.k - b.k);  // success left, rewrite right
    if (q && !sides.some((s) => hay(s).includes(q))) return;
    const keep = sides.filter(sidePasses);
    if (keep.length) cards.push({ sides: keep, paired: sides.length > 1 });
  });
  return cards;
}

function renderConv(ci) {
  return CONVS[ci].map((m) => `<div class="msg ${esc(m.role)}">`
    + `<div class="role">${esc(m.role)}</div>`
    + `<div class="body">${esc(m.content)}</div></div>`).join('');
}

function renderSide(x, tag, sideBySide) {
  const run = RUNS[runIdx];
  const cls = ['b-suc', 'b-con', 'b-unk'][x.k];
  let head = `<span class="badge ${cls}">${esc(KIND[x.k])}</span>`
    + `<span class="pill ${isPos(x.l) ? 'pos' : 'neg'}">${esc(labelText(run, x.l))}</span>`;
  let meta = '', why = '', src = '';

  if (x.k === 0) {
    head += `<span class="pill">${esc(x.et)}</span>`;
    why = `<div class="why"><div class="k">Judge reason &mdash; why it is `
      + `<strong>${esc(x.jl)}</strong> (confidence ${esc(x.jc)})</div>`
      + `<p>${esc(x.jr)}</p></div>`;
    meta = `<div class="meta">`
      + `<span class="pill">probe score ${esc(x.sc)}</span>`
      + `<span class="pill">probe said ${x.pp ? 'positive' : 'negative'}</span>`
      + `<span class="pill">${esc(x.am)}</span>`
      + `<span class="pill">red-team iter ${esc(x.ai)} &middot; round ${esc(x.rd)}</span>`
      + `</div>`;
  } else if (x.k === 1) {
    why = `<div class="why"><div class="k">Generator reason &mdash; why the rewrite is `
      + `<strong>${esc(labelText(run, x.l))}</strong></div><p>${esc(x.ge)}</p></div>`;
    // the source is the other column when both are shown; fall back to inlining it
    if (!sideBySide && x.oc >= 0) {
      src = `<details class="src"><summary>Source conversation it was rewritten from`
        + ` &mdash; labelled <strong>${esc(x.ol)}</strong>${x.se ? ' (' + esc(x.se) + ' success)' : ''}`
        + `</summary><div class="conv">${renderConv(x.oc)}</div>`
        + (x.sr ? `<div class="why"><div class="k">Judge reason for the source</div>`
                  + `<p>${esc(x.sr)}</p></div>` : '')
        + `</details>`;
    }
    meta = `<div class="meta"><span class="pill">generated by ${esc(x.gm)}</span></div>`;
  } else {
    why = `<div class="why"><div class="k">No provenance</div><p>This row matched neither an `
      + `attempt log nor the contrastive cache.</p></div>`;
  }

  return `<div class="col"><header>${head}${tag}</header>`
    + `<div class="conv">${renderConv(x.c)}</div>${why}${src}${meta}</div>`;
}

function renderCard(card, i) {
  const both = card.sides.length > 1;
  const cols = card.sides.map((x, j) => renderSide(
    x, j === 0 ? `<span class="idx">#${i + 1}</span>` : '', both,
  )).join('');
  return `<article class="card"><div class="pair" style="--cols:${card.sides.length}">`
    + `${cols}</div></article>`;
}

function renderRuns() {
  $('#run').innerHTML = RUNS.map((r, i) => `<option value="${i}">${esc(r.label)}</option>`).join('');
  $('#run').value = runIdx;
}

function renderTabs() {
  const run = RUNS[runIdx];
  let tabs = run.iters.map((it, i) => {
    const n = it.items.length;
    return `<button class="tab" role="tab" data-i="${i}" aria-selected="${i === iterIdx}">`
      + `Iteration ${it.n}<span class="cnt">${n}</span></button>`;
  }).join('');
  tabs += (run.tiers || []).map((g, i) =>
    `<button class="tab" role="tab" data-i="t${i}" aria-selected="${iterIdx === 't' + i}">`
    + `Tier ${esc(g.id)}<span class="cnt">${g.pairs}</span></button>`).join('');
  if (RUNS.some((r) => r.eval)) {
    tabs += `<button class="tab" role="tab" data-i="eval" aria-selected="${iterIdx === 'eval'}">`
      + `Eval scores<span class="cnt">${run.eval ? run.eval.rounds.length : 0}</span></button>`;
  }
  if (VARIANTS.length) {
    tabs += `<button class="tab" role="tab" data-i="var" aria-selected="${iterIdx === 'var'}">`
      + `Pruning variants<span class="cnt">${VARIANTS.length}</span></button>`;
  }
  $('#tabs').innerHTML = tabs;
  $('#paths').textContent = run.probe_dir + '  \\u00b7  ' + run.results_dir;
}

function render() {
  const evalView = iterIdx === 'eval', varView = iterIdx === 'var';
  // the standing note explains the per-iteration set-difference; no other tab is that
  $('#note').hidden = evalView || varView || isTier();
  $('#samples').hidden = evalView || varView;
  $('#evalview').hidden = !evalView;
  $('#varview').hidden = !varView;
  if (evalView) { renderEval(); return; }
  if (varView) { renderVariants(); return; }
  const tierView = isTier();

  const run = RUNS[runIdx], it = current();
  const nSuc = it.items.filter((x) => x.k === 0).length;
  const nCon = it.items.filter((x) => x.k === 1).length;
  const nUnk = it.items.filter((x) => x.k === 2).length;

  if (tierView) {
    const iters = Object.keys(it.iters).sort();
    $('#tabnote').innerHTML = (TIERNOTE ? TIERNOTE + ' ' : '')
      + `<strong>Tier ${esc(it.id)}</strong> &mdash; ${esc(it.desc)}. `
      + `Every card is a whole pair: the red-team success on the left and the contrastive `
      + `rewrite generated from it on the right, both of which are dropped together when `
      + `this tier is pruned.`;
    $('#stats').innerHTML = [
      ['pairs', it.pairs],
      ['conversations', it.items.length],
      ['red-team successes', nSuc],
      ['contrastive rewrites', nCon],
      ['first seen in iteration', iters.map((k) =>
        `${k}<span style="font-size:12px;color:var(--muted)">&nbsp;(${it.iters[k]})</span>`
      ).join(' &nbsp;') || '&mdash;'],
    ].concat(nUnk ? [['unmatched', nUnk]] : []).map(
      ([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`
    ).join('');
  } else {
    const logged = it.logged.fp + it.logged.fn;
    $('#tabnote').innerHTML = `<code>redteam_postprocessed_iter${it.n}.jsonl</code> &rarr; trained `
      + `<code>probe_iter${it.n}.pkl</code>, built from red-team iteration ${it.src_iter}. `
      + `Showing the ${it.items.length} rows <strong>new</strong> to this iteration `
      + `(the file holds ${it.total} in total; the rest already appeared in an earlier tab).`;

    $('#stats').innerHTML = [
      ['new rows', it.items.length],
      ['red-team successes', nSuc],
      ['contrastive pairs', nCon],
      ['successes logged (iter ' + it.src_iter + ')', logged + ' &nbsp;<span style="font-size:12px;color:var(--muted)">'
        + it.logged.fp + ' fp / ' + it.logged.fn + ' fn</span>'],
    ].concat(nUnk ? [['unmatched', nUnk]] : []).map(
      ([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`
    ).join('');
  }

  $('#lbl').innerHTML = ['all'].concat(run.labels).map((l) =>
    `<button class="chip" data-lbl="${esc(l)}" aria-pressed="${labelFilter === l}">`
    + `${l === 'all' ? 'all labels' : esc(labelText(run, l))}</button>`).join('');

  const vis = visibleCards();
  const nSamples = vis.reduce((a, c) => a + c.sides.length, 0);
  const slice = vis.slice(0, shown);
  $('#list').innerHTML = slice.length
    ? slice.map(renderCard).join('')
      + (vis.length > shown
        ? `<button class="more" id="more">Show ${Math.min(PAGE, vis.length - shown)} more `
          + `(${vis.length - shown} remaining)</button>` : '')
    : `<div class="empty">Nothing matches these filters.</div>`;
  $('#count').textContent = nSamples === it.items.length
    ? `${nSamples} samples shown` : `${nSamples} of ${it.items.length} samples shown`;

  const more = $('#more');
  if (more) more.onclick = () => { shown += PAGE; render(); };
}

// ----------------------------------------------------------------- eval charts
// Chart engine ported from build_contrastive_pairs_viewer.py so both viewers read
// the same way. Series colours are the repo's validated 4-hue categorical set.
const METRICS = [
  { key: 'auroc', label: 'AUROC', chance: 0.5 },
  { key: 'accuracy', label: 'Accuracy', chance: 0.5 },
  { key: 'tpr_at_fpr', label: 'TPR @ 1% FPR', chance: null },
];
let metric = METRICS[0];
const G = { W: 760, H: 330, mt: 16, mr: 148, mb: 34, ml: 48 };  // mr = direct-label gutter
const pw = G.W - G.ml - G.mr, ph = G.H - G.mt - G.mb;
const CHARTS = {};

function ptsFor(ev, name) {
  const out = [];
  ev.rounds.forEach((r) => {
    const s = r.scores[name];
    if (s && s[metric.key] != null) out.push([r.n, s[metric.key]]);
  });
  return out;
}

/** Components of the selected run: one line per eval split, plus the mean. */
function splitSeries(ev) {
  const out = ev.splits.map((sp, i) => (
    { name: sp, cls: 's' + ((i % 4) + 1), pts: ptsFor(ev, sp) }));
  out.push({ name: 'mean', cls: 'smean', mean: true, pts: ptsFor(ev, 'mean') });
  return out.filter((s) => s.pts.length);
}

/** One line per run (its mean), so the arms can be compared directly. */
function runSeries() {
  return RUNS.map((r, i) => (r.eval ? {
    name: r.label, cls: 's' + ((i % 4) + 1), subject: i === runIdx,
    pts: ptsFor(r.eval, 'mean'),
  } : null)).filter((s) => s && s.pts.length);
}

// A domain snapped to round ticks with a little headroom, so the y axis reads in
// familiar increments instead of tracking the data's exact extremes.
function domain(series) {
  const vals = [];
  series.forEach((s) => s.pts.forEach((p) => vals.push(p[1])));
  const lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
  const span = hi - lo;
  const step = span < 0.25 ? 0.05 : (span > 0.6 ? 0.2 : 0.1);
  const ymin = Math.max(0, Math.floor((lo - step / 2) / step) * step);
  const ymax = Math.min(1, Math.ceil((hi + step / 2) / step) * step);
  const ticks = [];
  for (let v = ymin; v <= ymax + 1e-9; v += step) ticks.push(Math.round(v * 1e4) / 1e4);
  return { ymin, ymax, ticks };
}

function xPos(it, its) {
  return G.ml + (its.length === 1 ? pw / 2
    : (it - its[0]) / (its[its.length - 1] - its[0]) * pw);
}
function yPos(v, d) { return G.mt + (1 - (v - d.ymin) / (d.ymax - d.ymin)) * ph; }

function drawChart(id, series, opts) {
  opts = opts || {};
  const seen = {};
  series.forEach((s) => s.pts.forEach((p) => { seen[p[0]] = 1; }));
  const its = Object.keys(seen).map(Number).sort((a, b) => a - b);
  const d = domain(series);
  CHARTS[id] = { series, its, dom: d };

  let s = '';
  d.ticks.forEach((v) => {
    const y = yPos(v, d);
    const isChance = metric.chance != null && Math.abs(v - metric.chance) < 1e-9;
    s += `<line class="${isChance ? 'chanceline' : 'gridline'}" x1="${G.ml}" y1="${y}"`
      + ` x2="${G.ml + pw}" y2="${y}"/>`
      + `<text class="ylab" x="${G.ml - 7}" y="${y + 3.5}" text-anchor="end">`
      + `${v.toFixed(2)}</text>`;
  });
  // Sits just above its own line, inside the plot — the right gutter belongs to the
  // direct labels, and a "chance" caption out there would collide with them.
  if (metric.chance != null && metric.chance >= d.ymin && metric.chance <= d.ymax) {
    s += `<text class="ylab" x="${G.ml + pw - 4}" y="${yPos(metric.chance, d) - 6}"`
      + ` text-anchor="end">chance</text>`;
  }
  s += `<line class="axline" x1="${G.ml}" y1="${G.mt + ph}" x2="${G.ml + pw}"`
    + ` y2="${G.mt + ph}"/>`;
  its.forEach((it) => {
    s += `<text class="tick" x="${xPos(it, its)}" y="${G.mt + ph + 16}"`
      + ` text-anchor="middle">iter ${it}</text>`;
  });
  if (opts.xtitle) {
    s += `<text class="tick" x="${G.ml + pw / 2}" y="${G.mt + ph + 31}"`
      + ` text-anchor="middle">${esc(opts.xtitle)}</text>`;
  }

  series.forEach((x) => {
    const pts = x.pts.map((p) => `${xPos(p[0], its)},${yPos(p[1], d)}`).join(' ');
    s += `<polyline class="line ${x.cls}${x.mean ? ' mean' : ''}`
      + `${x.subject ? ' subject' : ''}" points="${pts}"/>`;
    x.pts.forEach((p) => {
      s += `<circle class="pt ${x.cls}" cx="${xPos(p[0], its)}" cy="${yPos(p[1], d)}" r="4"/>`;
    });
  });

  // Direct labels at the line ends, pushed apart so near-equal finishes stay legible.
  // Past a handful of series they stop being legible however far you push them, so
  // they are dropped and the legend + table + hover carry identity instead.
  const columns = {};
  if (series.length <= (opts.maxLabels == null ? 5 : opts.maxLabels)) series.forEach((x) => {
    const last = x.pts[x.pts.length - 1];
    const lx = xPos(last[0], its) + 8;
    const k = Math.round(lx / 6);
    (columns[k] = columns[k] || []).push(
      { name: x.name, cls: x.cls, x: lx, y: yPos(last[1], d) });
  });
  Object.keys(columns).forEach((k) => {
    const group = columns[k].sort((a, b) => a.y - b.y);
    for (let i = 1; i < group.length; i++) {
      if (group[i].y - group[i - 1].y < 13) group[i].y = group[i - 1].y + 13;
    }
    // Pushing apart only moves labels down, so a cluster on the floor would spill onto
    // the x-axis ticks. Slide the stack back inside.
    const over = group[group.length - 1].y - (G.mt + ph);
    if (over > 0) group.forEach((l) => { l.y -= over; });
    const under = G.mt - group[0].y;
    if (under > 0) group.forEach((l) => { l.y += under; });
    group.forEach((l) => {
      s += `<text class="endlab ${l.cls}" x="${l.x}" y="${l.y + 3.5}">${esc(l.name)}</text>`;
    });
  });

  s += `<line class="crosshair" id="ch-${id}" y1="${G.mt}" y2="${G.mt + ph}"/>`;
  its.forEach((it) => {
    s += `<rect class="hz-${id}" x="${xPos(it, its) - pw / (its.length * 2) - 2}"`
      + ` y="${G.mt}" width="${pw / its.length + 4}" height="${ph}"`
      + ` fill="transparent" data-it="${it}"/>`;
  });
  return `<div class="chartbox"><svg id="svg-${id}" viewBox="0 0 ${G.W} ${G.H}" role="img"`
    + ` aria-label="${esc(opts.alt || (metric.label + ' chart'))}">${s}</svg>`
    + `<div class="tip" id="tip-${id}"></div></div>`;
}

function legend(series) {
  return '<div class="legend">' + series.map((x) =>
    `<span class="lg"><span class="sw ${x.mean ? 'mean ' : ''}`
    + `${x.subject ? 'subject ' : ''}${x.cls}"></span>${esc(x.name)}</span>`).join('') + '</div>';
}

function evalTable(id, series, opts) {
  opts = opts || {};
  const its = CHARTS[id].its;
  // Default: how far each series moved across its own run. With a reference series
  // (the variants chart) the useful comparison is instead sideways at the last
  // iteration — every arm against the arm it is meant to improve on.
  const ref = opts.deltaRef
    ? series.filter((s) => s.name === opts.deltaRef)[0] : null;
  const refLast = ref ? ref.pts[ref.pts.length - 1][1] : null;
  const head = '<tr><th>series</th>' + its.map((i) => `<th>iter ${i}</th>`).join('')
    + `<th>&Delta; vs ${ref ? esc(opts.deltaRef) : 'iter ' + its[0]}</th></tr>`;
  const body = series.map((x) => {
    const m = {};
    x.pts.forEach((p) => { m[p[0]] = p[1]; });
    const first = m[its[0]], last = x.pts[x.pts.length - 1][1];
    const delta = ref ? (x === ref ? null : last - refLast)
      : (first == null ? null : last - first);
    return `<tr class="${x.mean ? 'mean ' : ''}${x.subject ? 'subjectrow ' : ''}${x.cls}">`
      + `<td><span class="key"></span>${esc(x.name)}</td>`
      + its.map((i) => `<td>${m[i] == null ? '&mdash;' : m[i].toFixed(3)}</td>`).join('')
      + `<td class="${delta == null ? '' : delta >= 0 ? 'up' : 'down'}">`
      + `${delta == null ? '&mdash;' : (delta >= 0 ? '+' : '\\u2212') + Math.abs(delta).toFixed(3)}`
      + '</td></tr>';
  }).join('');
  return `<div class="tblwrap"><table class="ev"><thead>${head}</thead>`
    + `<tbody>${body}</tbody></table></div>`;
}

function wireHover(id) {
  const svg = document.getElementById('svg-' + id);
  if (!svg) return;
  const chart = CHARTS[id], box = svg.closest('.chartbox');
  const tip = document.getElementById('tip-' + id), ch = document.getElementById('ch-' + id);
  function show(it) {
    const x = xPos(it, chart.its);
    ch.setAttribute('x1', x); ch.setAttribute('x2', x); ch.classList.add('hi');
    tip.innerHTML = `<div class="th">iter ${it}</div>` + chart.series.map((sr) => {
      const hit = sr.pts.filter((p) => p[0] === it)[0];
      if (!hit) return '';
      return '<div class="row"><span><span class="dot" style="background:var(--'
        + (sr.mean ? 'mean' : sr.cls) + ')"></span>'
        + `${esc(sr.name)}</span><b>${hit[1].toFixed(3)}</b></div>`;
    }).join('');
    tip.style.opacity = 1;
    const rect = svg.getBoundingClientRect(), scale = rect.width / G.W;
    let px = x * scale + 14;
    if (px > rect.width - tip.offsetWidth - 6) px = x * scale - tip.offsetWidth - 14;
    tip.style.left = Math.max(4, px) + 'px';
    tip.style.top = (G.mt * scale + 6) + 'px';
  }
  box.querySelectorAll('.hz-' + id).forEach((rc) => {
    rc.addEventListener('mousemove', () => show(+rc.dataset.it));
    rc.addEventListener('mouseenter', () => show(+rc.dataset.it));
  });
  box.addEventListener('mouseleave', () => {
    ch.classList.remove('hi'); tip.style.opacity = 0;
  });
}

function renderEval() {
  const run = RUNS[runIdx];
  $('#metrics').innerHTML = METRICS.map((m) =>
    `<button class="chip" data-m="${m.key}" aria-pressed="${m.key === metric.key}">`
    + `${esc(m.label)}</button>`).join('');

  const across = runSeries();
  const comp = run.eval ? splitSeries(run.eval) : [];
  let out = '';

  if (across.length) {
    out += `<section class="evcard"><h2>${esc(metric.label)} per iteration &mdash; `
      + `across runs</h2><p class="cap">Each run's mean over its eval splits. `
      + `The selected run (<strong>${esc(run.label)}</strong>) is drawn thicker.</p>`
      + drawChart('runs', across, {
        alt: metric.label + ' per iteration, one line per run',
      }) + legend(across) + evalTable('runs', across) + '</section>';
  }

  if (comp.length) {
    out += `<section class="evcard"><h2>${esc(metric.label)} per iteration &mdash; `
      + `components of <strong>${esc(run.label)}</strong></h2>`
      + `<p class="cap">One line per eval split, plus the dashed mean they average to.</p>`
      + `<p class="src">${esc(run.eval.csv)}</p>`
      + drawChart('comp', comp, {
        alt: metric.label + ' per iteration, one line per eval split plus the mean',
      }) + legend(comp) + evalTable('comp', comp) + '</section>';
  } else {
    out += `<div class="empty">No comparison CSV found for ${esc(run.label)}.</div>`;
  }

  $('#eval').innerHTML = out;
  if (across.length) wireHover('runs');
  if (comp.length) wireHover('comp');

  $('#metrics').querySelectorAll('.chip').forEach((b) => {
    b.onclick = () => {
      metric = METRICS.filter((m) => m.key === b.dataset.m)[0];
      renderEval();
    };
  });
}

// ------------------------------------------------------------- pruning variants
// Retrains of one run's data with a slice of it removed. They share the run's eval
// splits and iteration axis, so they read on the same chart engine — one line per
// arm, the unpruned run drawn thicker as the reference everything is measured against.

/** One line per variant for the selected split (or its mean). */
function variantSeries() {
  return VARIANTS.map((v, i) => ({
    name: v.label,
    cls: 's' + (i + 1),           // fixed slot order, never cycled
    subject: !!v.reference,       // the arm the others are compared against
    pts: v.eval ? ptsFor(v.eval, splitKey) : [],
  })).filter((s) => s.pts.length);
}

function variantSplits() {
  const seen = {};
  VARIANTS.forEach((v) => (v.eval ? v.eval.splits : []).forEach((s) => { seen[s] = 1; }));
  return ['mean'].concat(Object.keys(seen).sort());
}

function renderVariants() {
  $('#vmetrics').innerHTML = METRICS.map((m) =>
    `<button class="chip" data-m="${m.key}" aria-pressed="${m.key === metric.key}">`
    + `${esc(m.label)}</button>`).join('');
  $('#vsplits').innerHTML = variantSplits().map((s) =>
    `<button class="chip" data-sp="${esc(s)}" aria-pressed="${s === splitKey}">`
    + `${s === 'mean' ? 'mean of splits' : esc(s)}</button>`).join('');

  const series = variantSeries();
  const ref = (VARIANTS.filter((v) => v.reference)[0] || {}).label;
  $('#vout').innerHTML = series.length
    ? `<section class="evcard"><h2>${esc(metric.label)} per iteration on `
      + `<strong>${esc(splitKey === 'mean' ? 'the mean of all splits' : splitKey)}</strong>`
      + ` &mdash; one line per pruning variant</h2>`
      + `<p class="cap">Every arm retrains the same three iterations off the same base `
      + `data; only which red-team pairs were removed differs. `
      + (ref ? `<strong>${esc(ref)}</strong> is the reference and is drawn thicker; `
             + `the table's last column measures each arm against it at the final `
             + `iteration. ` : '')
      + `An arm that starts at iter 1 has no iter 0 of its own &mdash; iter 0 is the `
      + `baseline probe, trained before any red-teaming, and is shared by all of them.</p>`
      + drawChart('var', series, {
        alt: metric.label + ' per iteration, one line per pruning variant',
        maxLabels: 0,   // the arms converge too tightly for end labels to stay legible
      }) + legend(series) + evalTable('var', series, { deltaRef: ref })
      + '</section>'
    : `<div class="empty">No variant CSVs were passed to the builder.</div>`;

  if (series.length) wireHover('var');
  $('#vmetrics').querySelectorAll('.chip').forEach((b) => {
    b.onclick = () => {
      metric = METRICS.filter((m) => m.key === b.dataset.m)[0];
      renderVariants();
    };
  });
  $('#vsplits').querySelectorAll('.chip').forEach((b) => {
    b.onclick = () => { splitKey = b.dataset.sp; renderVariants(); };
  });
}

function reset() { shown = PAGE; }

$('#run').onchange = (e) => {
  runIdx = +e.target.value; labelFilter = 'all';
  // stay on the eval tab when switching runs — that's where you compare arms
  // a tier tab is specific to one run's tier file, so fall back when it has none
  if (iterIdx === 'eval' || iterIdx === 'var') { /* not run-specific — keep */ }
  else if (!isTier() || !(RUNS[runIdx].tiers || [])[+iterIdx.slice(1)]) iterIdx = 0;
  reset(); renderTabs(); render();
};
$('#tabs').onclick = (e) => {
  const b = e.target.closest('.tab'); if (!b) return;
  const v = b.dataset.i;
  iterIdx = (v === 'eval' || v === 'var' || v[0] === 't') ? v : +v;
  reset(); renderTabs(); render();
};
$('#kind').onclick = (e) => {
  const b = e.target.closest('.chip'); if (!b) return;
  kindFilter = b.dataset.kind; reset();
  [...$('#kind').children].forEach((c) => c.setAttribute('aria-pressed', c === b));
  render();
};
$('#lbl').onclick = (e) => {
  const b = e.target.closest('.chip'); if (!b) return;
  labelFilter = b.dataset.lbl; reset(); render();
};
$('#q').oninput = (e) => { query = e.target.value; reset(); render(); };

renderRuns(); renderTabs(); render();
"""


def build_html(payload: dict, title: str, subtitle: str) -> str:
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(title)}</h1>
  <div class="sub">{html.escape(subtitle)}</div>
  <div class="path" id="paths"></div>

  <div class="note" id="note">
    Each tab holds only the samples <strong>new</strong> to that iteration &mdash; the
    <code>redteam_postprocessed_iter{{N}}.jsonl</code> dumps are near-cumulative, so a row is
    listed under the first iteration whose dump contains it. Iteration N's dump is what trained
    <code>probe_iter{{N}}.pkl</code> and is built from the red-team successes of iterations
    0&ndash;(N&minus;1). <code>filter_dataset</code> re-runs over the whole accumulated set before
    every retrain, so the new-success count is generally lower than the successes logged for that
    red-team iteration. Each card pairs a red-team success (left) with the contrastive rewrite
    generated from it (right); the kind and label filters pick which side is shown, while the
    search matches either half so you keep the comparison.
  </div>

  <div class="runbar">
    <label for="run">Run</label>
    <select id="run"></select>
  </div>

  <div class="tabs" id="tabs" role="tablist"></div>

  <div id="samples">
    <div class="tabnote" id="tabnote"></div>
    <div class="stats" id="stats"></div>

    <div class="toolbar">
      <input type="search" id="q" placeholder="Search conversations, judge reasons, generator reasons&hellip;">
      <div class="chips" id="kind">
        <button class="chip" data-kind="all" aria-pressed="true">all</button>
        <button class="chip" data-kind="suc" aria-pressed="false">successes</button>
        <button class="chip" data-kind="con" aria-pressed="false">contrastive</button>
      </div>
      <div class="chips" id="lbl"></div>
      <span class="pill" id="count"></span>
    </div>

    <div id="list"></div>
  </div>

  <div id="evalview" hidden>
    <div class="tabnote">
      Probe quality on the held-out eval splits, per retrain cycle. <strong>iter0 is the
      baseline probe</strong> &mdash; trained on the base data before any red-teaming, so it
      has no sample tab; iter<em>N</em> is the probe retrained on the data in the
      &ldquo;Iteration <em>N</em>&rdquo; tab (cumulatively). Hover a chart for the values at
      that iteration.
    </div>
    <div class="toolbar"><div class="chips" id="metrics"></div></div>
    <div id="eval"></div>
  </div>

  <div id="varview" hidden>
    <div class="tabnote">
      The same three retrain cycles run again with a slice of the red-team data removed,
      to see which slice the probe was actually losing ground to. Every arm shares the base
      training data, the split seed and the eval splits, so the only variable is which pairs
      were dropped &mdash; identical inputs give an identical probe, so equal lines mean the
      pruning had nothing to remove at that iteration rather than that it made no difference.
    </div>
    <div class="toolbar">
      <div class="chips" id="vmetrics"></div>
      <div class="chips" id="vsplits"></div>
    </div>
    <div id="vout"></div>
  </div>
</div>
<script type="application/json" id="data">{blob}</script>
<script>{JS}</script>
</body>
</html>
"""


# ------------------------------------------------------------------------ main

def parse_run(spec: str, root: Path) -> tuple[str, Path, Path, Path | None]:
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            f"--run must be LABEL:PROBE_DIR:RESULTS_DIR[:COMPARISON_CSV], got {spec!r}"
        )
    label, probe_dir, results_dir = parts[:3]
    csv_path = (root / parts[3]) if len(parts) == 4 and parts[3] else None
    return label, (root / probe_dir), (root / results_dir), csv_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    metavar="LABEL:PROBE_DIR:RESULTS_DIR",
                    help="a run to include; repeat for several (they share one viewer)")
    ap.add_argument("--title", default="Red-team data used for retraining")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--root", default=".", help="root the run paths resolve against")
    ap.add_argument("--tiers", action="append", default=[], metavar="RUN_LABEL:PATH",
                    help="a tier JSONL to attach to one run, adding a tab per tier; "
                         "rows are {tier, tier_description, pair_id, in_iters, "
                         "original:{inputs,label}, contrastive:{inputs,label}}")
    ap.add_argument("--tiers-note", default="",
                    help="sentence shown above every tier tab, saying what the tiering is for")
    ap.add_argument("--variant", action="append", default=[], metavar="LABEL:CSV",
                    help="a retrain-with-data-removed arm, as a comparison CSV; repeat for "
                         "several. They share one 'Pruning variants' tab. Prefix the label "
                         "with '*' to mark the arm the others are measured against.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    tiers_for: dict[str, Path] = {}
    for spec in args.tiers:
        run_label, _, rel = spec.partition(":")
        if not rel:
            raise SystemExit(f"--tiers must be RUN_LABEL:PATH, got {spec!r}")
        tiers_for[run_label.strip()] = root / rel.strip()

    pool = ConvPool()
    runs = []
    for spec in args.run:
        label, probe_dir, results_dir, csv_path = parse_run(spec, root)
        if not probe_dir.is_dir():
            raise SystemExit(f"probe dir not found: {probe_dir}")
        tiers_path = tiers_for.pop(label, None)
        if tiers_path and not tiers_path.is_file():
            raise SystemExit(f"tier file not found: {tiers_path}")
        run = build_run(label, probe_dir, results_dir, pool, csv_path, tiers_path)
        if not run["iters"]:
            raise SystemExit(f"no redteam_postprocessed_iter*.jsonl under {probe_dir}")
        present = {x["l"] for group in (run["iters"] + run["tiers"])
                   for x in group["items"]}
        run["labels"] = [l for l in ("positive", "negative") if l in present] \
            + sorted(present - {"positive", "negative"})
        runs.append(run)

        for it in run["iters"]:
            k = [0, 0, 0]
            for item in it["items"]:
                k[item["k"]] += 1
            print(f"{label:<16} iter{it['n']}: {len(it['items']):>4} new "
                  f"({k[0]} successes, {k[1]} contrastive"
                  + (f", {k[2]} UNMATCHED" if k[2] else "") + ")")
        for g in run["tiers"]:
            k = [0, 0, 0]
            for item in g["items"]:
                k[item["k"]] += 1
            print(f"{label:<16} tier{g['id']}: {g['pairs']:>4} pairs "
                  f"({k[0]} successes, {k[1]} contrastive"
                  + (f", {k[2]} UNMATCHED" if k[2] else "") + ")")
        ev = run["eval"]
        print(f"{label:<16} eval : " + (
            f"{len(ev['rounds'])} rounds x {len(ev['splits'])} splits "
            f"({', '.join(ev['splits'])}) from {Path(ev['csv']).name}"
            if ev else "no comparison CSV found — eval tab hidden"))

    if tiers_for:
        raise SystemExit(
            "--tiers named runs that are not in --run: " + ", ".join(sorted(tiers_for))
        )

    variants = []
    for spec in args.variant:
        label, _, rel = spec.partition(":")
        if not rel:
            raise SystemExit(f"--variant must be LABEL:CSV, got {spec!r}")
        label = label.strip()
        reference = label.startswith("*")
        label = label.lstrip("*").strip()
        csv_p = root / rel.strip()
        if not csv_p.is_file():
            raise SystemExit(f"variant CSV not found: {csv_p}")
        ev = load_eval(csv_p)
        if not ev:
            raise SystemExit(f"variant CSV had no usable rows: {csv_p}")
        variants.append({"label": label, "reference": reference, "eval": ev})
        print(f"{'variant':<16} {label:<22} {len(ev['rounds'])} rounds x "
              f"{len(ev['splits'])} splits" + ("  [reference]" if reference else ""))
    if len(variants) > 8:
        raise SystemExit(
            f"{len(variants)} variants, but the categorical palette has 8 slots — "
            "cycling hues would make two arms indistinguishable"
        )

    subtitle = args.subtitle or (
        f"{len(runs)} run(s) · {len(pool.convs)} distinct conversations · "
        "new samples per iteration, not cumulative"
    )
    payload = {"convs": pool.convs, "runs": runs, "tier_note": args.tiers_note,
               "variants": variants}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(payload, args.title, subtitle))
    size = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
