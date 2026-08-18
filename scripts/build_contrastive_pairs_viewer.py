#!/usr/bin/env python
"""Build a self-contained HTML viewer of the contrastive pairs used per retrain.

One viewer per run, with a tab per iteration. Iteration N's input is
``<probe-out-dir>/redteam_postprocessed_iter{N}.jsonl`` — the red-team samples
(after ``filter_dataset`` + ``generate_contrastive_dataset``) that trained
``probe_iter{N}.pkl``.

A final **eval** tab plots the probes' eval metrics per iteration — the mean plus
each split — read from the run's comparison CSV. Its ``iterN`` rows are the eval of
``probe_iter{N}.pkl``, so they line up with the contrastive tabs by probe index
(``iter0`` is the baseline probe, which no contrastive tab corresponds to). The CLI
only writes that CSV when the whole loop finishes, so ``--eval-log`` can recover the
eval tables of any extra rounds an interrupted run printed but never wrote out.

Pairing note: that dump carries only ``{id, inputs, label}``, and
``generate_contrastive_dataset`` returns ``records + generated`` where
``generated`` is *cache hits first, then newly generated* — so the naive
``row[i] <-> row[i + n/2]`` alignment only holds while the contrastive cache is
empty (iteration 1). From iteration 2 on it is wrong for most rows. The
authoritative link is ``contrastive_cache.jsonl``, whose records carry
``original_messages``; this script pairs through that and reports anything it
could not pair.

Distinct pairs are stored once and each iteration keeps only indices into them,
since the per-iteration files are near-cumulative.

Usage:
    .venv_claude/bin/python scripts/build_contrastive_pairs_viewer.py \
        --probe-dir probes/llama70b50_gpt51_memo \
        --comparison-csv results_hu_harm_llama70b50_gpt51_memo/gpt51_comparison.csv \
        --eval-log logs/run_hh_llama70b50_gpt51_memo_resume2.log \
        --baseline-dir ~/Documents/cresults --run-label "gpt-5.1 (memo)" \
        --title "gpt-5.1 — cross-iteration memos ON" \
        --out viewers/gpt51_memo_contrastive_viewer.html
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


# ---------------------------------------------------------------- data loading

def canon(messages: list[dict]) -> str:
    """Canonical text of a conversation — the identity of a sample."""
    return json.dumps(
        [{"role": m.get("role", ""), "content": m.get("content", "")} for m in messages],
        sort_keys=True,
    )


def load_contrastive_cache(path: Path) -> tuple[dict, dict]:
    """Return ``(by_source, by_generated)``, both keyed on canonical conversation."""
    by_source: dict[str, dict] = {}
    by_generated: dict[str, dict] = {}
    if not path.exists():
        return by_source, by_generated
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)["record"]
            by_source[canon(record["original_messages"])] = record
            by_generated[canon(record["inputs"])] = record
    return by_source, by_generated


def build_pairs(postprocessed: Path, by_source: dict, by_generated: dict):
    """Pair each source sample in one iteration file with its generated counterpart."""
    rows = [json.loads(l) for l in postprocessed.open(encoding="utf-8") if l.strip()]

    generated_rows: dict[str, dict] = {}
    sources: list[dict] = []
    for row in rows:
        key = canon(row["inputs"])
        if key in by_generated:
            generated_rows[key] = row
        else:
            sources.append(row)

    pairs: list[dict] = []
    unpaired: list[dict] = []
    matched: set[str] = set()
    for source in sources:
        record = by_source.get(canon(source["inputs"]))
        if record is None:
            unpaired.append({"reason": "no contrastive pair generated", "row": source})
            continue
        gen_key = canon(record["inputs"])
        generated = generated_rows.get(gen_key)
        if generated is None:
            unpaired.append({"reason": "pair absent from this iteration", "row": source})
            continue
        matched.add(gen_key)
        pairs.append(
            {
                "source": source,
                "generated": generated,
                "source_label": record.get("original_label", ""),
                "generated_label": record.get("labels", ""),
                "explanation": record.get("generation_explanation", ""),
                "generation_model": record.get("generation_model", ""),
            }
        )
    for key, row in generated_rows.items():
        if key not in matched:
            unpaired.append({"reason": "generated sample with no source in file", "row": row})
    return rows, pairs, unpaired


def collect_run(probe_dir: Path) -> dict:
    """Load every iteration of one run into a compact, de-duplicated payload."""
    by_source, by_generated = load_contrastive_cache(probe_dir / "contrastive_cache.jsonl")

    files = sorted(
        probe_dir.glob("redteam_postprocessed_iter*.jsonl"),
        key=lambda p: int(re.search(r"iter(\d+)", p.name).group(1)),
    )
    if not files:
        raise SystemExit(f"No redteam_postprocessed_iter*.jsonl found in {probe_dir}")

    pairs: list[dict] = []
    index_of: dict[str, int] = {}
    iterations: list[dict] = []
    previous: set[int] = set()   # the immediately preceding iteration, for the drop count
    seen_ever: set[int] = set()  # every pair shown so far, so tabs stay disjoint

    for path in files:
        iteration = int(re.search(r"iter(\d+)", path.name).group(1))
        rows, found, unpaired = build_pairs(path, by_source, by_generated)

        indices: list[int] = []
        for pair in found:
            key = canon(pair["source"]["inputs"])
            idx = index_of.get(key)
            if idx is None:
                idx = len(pairs)
                index_of[key] = idx
                pairs.append(
                    {
                        "src_msgs": pair["source"]["inputs"],
                        "gen_msgs": pair["generated"]["inputs"],
                        "src_label": pair["source_label"],
                        "gen_label": pair["generated_label"],
                        "src_class": pair["source"].get("label", ""),
                        "gen_class": pair["generated"].get("label", ""),
                        "why": pair["explanation"],
                        "model": pair["generation_model"],
                    }
                )
            indices.append(idx)

        current = set(indices)
        # Each tab shows ONLY what this iteration added. The per-iteration files are
        # cumulative (successes accumulate), so without this every tab would repeat
        # everything the earlier ones already showed. "Added" means first-ever
        # appearance, not merely absent from the previous iteration — `filter_dataset`
        # can drop a pair and let it back in later, and re-showing it would put the
        # same pair under two tabs.
        added = [i for i in indices if i not in seen_ever]
        returned = len([i for i in current if i in seen_ever and i not in previous])
        iterations.append(
            {
                "n": iteration,
                "file": path.name,
                "rows": len(rows),
                "idx": added,
                "cum": len(current),
                "dropped": len(previous - current) if previous else 0,
                "returned": returned,
                "unpaired": len(unpaired),
                "first": not previous,
            }
        )
        previous = current
        seen_ever |= current

    return {"pairs": pairs, "iterations": iterations}


# ------------------------------------------------------------------ eval scores

METRICS = ("auroc", "accuracy", "tpr_at_fpr")

_EVAL_HEADER = re.compile(r"=====\s*EVALUATING\s+iter(\d+):")
# One row of pandas' `to_string()` eval table: "  eval_ant_hh 0.641011  0.619403 ..."
_EVAL_ROW = re.compile(
    r"^\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)


def _round_num(label: str) -> int:
    """``"iter3"`` -> ``3``; anything unparseable sorts last."""
    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if m else 10**6


def load_comparison_csv(path: Path) -> dict[int, dict[str, dict[str, float]]]:
    """``{iteration: {split_or_"mean": {metric: value}}}`` from a comparison CSV."""
    rounds: dict[int, dict[str, dict[str, float]]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            scores = {}
            for metric in METRICS:
                try:
                    scores[metric] = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
            rounds.setdefault(_round_num(row["round"]), {})[row["dataset"]] = scores
    return rounds


def load_eval_logs(paths: list[Path]) -> dict[int, dict[str, dict[str, float]]]:
    """Recover eval tables printed to a run log.

    The CLI writes the comparison CSV only after the whole loop finishes, so an
    interrupted run leaves its later rounds' numbers *only* in the log. Each is an
    ``===== EVALUATING iterN =====`` marker followed (after tqdm/HF noise) by the
    pandas table, ending at its ``mean`` row.
    """
    rounds: dict[int, dict[str, dict[str, float]]] = {}
    for path in paths:
        if not path.exists():
            raise SystemExit(f"--eval-log not found: {path}")
        pending: int | None = None
        table: dict[str, dict[str, float]] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            header = _EVAL_HEADER.search(line)
            if header:
                pending, table = int(header.group(1)), {}
                continue
            if pending is None:
                continue
            row = _EVAL_ROW.match(line)
            if not row:
                continue
            name = row.group(1)
            if name == "dataset":  # the header row itself
                continue
            table[name] = dict(zip(METRICS, (float(row.group(i)) for i in (2, 3, 4))))
            if name == "mean":
                rounds[pending] = table
                pending, table = None, {}
    return rounds


# The "simple prompted" probe settings this repo's red-teaming runs are benchmarked
# against, as laid out in viewers/cp_baseline_comparison_redteaming.html. Each is a
# folder of one-CSV-per-step (same schema as a comparison CSV row group, minus the
# `round` column — the step is the filename), so the order has to be given explicitly.
# The skipped files are deliberate: results-all-hh/iter3b.csv duplicates final.csv,
# results-constraint1-hh/results_final.csv duplicates results_iter6.csv, and
# results_all.csv / noseed5.csv are separate one-offs, not steps of these sequences.
BASELINE_SPEC = [
    {
        "name": "everything-allowed", "dir": "results-all-hh", "cls": "s2",
        "steps": [("baseline", "baseline.csv"), ("iter1", "iter1.csv"),
                  ("iter2", "iter2.csv"), ("iter3", "iter3.csv"),
                  ("iter4", "iter4.csv"), ("final", "final.csv")],
    },
    {
        "name": "constraint 1", "dir": "results-constraint1-hh", "cls": "s3",
        "steps": [("baseline", "results_baseline.csv")]
                 + [(f"iter{i}", f"results_iter{i}.csv") for i in range(1, 8)],
    },
    {
        "name": "constraint 2", "dir": "results-constraint2-hh", "cls": "s4",
        "steps": [("baseline", "baseline.csv")]
                 + [(f"v{i}", f"v{i}.csv") for i in range(1, 7)]
                 + [("noseed", "noseed.csv")],
    },
]


def load_step_csv(path: Path) -> dict[str, dict[str, float]]:
    """``{split_or_"mean": {metric: value}}`` from one single-step eval CSV."""
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            scores = {}
            for metric in METRICS:
                try:
                    scores[metric] = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
            out[row["dataset"]] = scores
    return out


def _recover_split_names(steps: list[dict], reference: dict[float, str], where: str) -> None:
    """Rename anonymized splits (``a``/``b``/``c``/``d``) in place.

    The constraint folders hide their split names, but every one of these series starts
    from the *same* baseline probe, so their baseline row values are identical to the
    named folder's. Matching on those values recovers the mapping — rather than assuming
    ``a`` is alphabetically first — and fails loudly if the values don't line up, which
    is the only thing that would make a silent mis-labelling possible.
    """
    baseline = steps[0]["scores"]
    mapping: dict[str, str] = {}
    for key, scores in baseline.items():
        if key == "mean":
            continue
        name = reference.get(round(scores["auroc"], 12))
        if name is None:
            raise SystemExit(
                f"{where}: split {key!r} has baseline AUROC {scores['auroc']} which matches "
                "no split of the named baseline, so its identity cannot be recovered."
            )
        mapping[key] = name
    if len(set(mapping.values())) != len(mapping):
        raise SystemExit(f"{where}: two splits matched the same named split — {mapping}")
    for step in steps:
        step["scores"] = {mapping.get(k, k): v for k, v in step["scores"].items()}


def load_baselines(root: Path, splits: list[str]) -> list[dict]:
    """Read every `BASELINE_SPEC` series under ``root``, skipping ones not present."""
    loaded = []
    for spec in BASELINE_SPEC:
        folder = root / spec["dir"]
        if not folder.is_dir():
            print(f"  baseline: {spec['dir']} not under {root} — skipped")
            continue
        steps = []
        for label, filename in spec["steps"]:
            path = folder / filename
            if not path.exists():
                raise SystemExit(f"baseline step missing: {path}")
            steps.append({"label": label, "scores": load_step_csv(path)})
        loaded.append({"name": spec["name"], "cls": spec["cls"],
                       "dir": spec["dir"], "steps": steps})

    # One folder keeps the real split names; the others are keyed off its baseline values.
    named = next(
        (b for b in loaded
         if {k for k in b["steps"][0]["scores"] if k != "mean"} == set(splits)),
        None,
    )
    for b in loaded:
        found = {k for k in b["steps"][0]["scores"] if k != "mean"}
        if found == set(splits):
            continue
        if named is None:
            raise SystemExit(
                f"{b['dir']} names its splits {sorted(found)}, not this run's "
                f"{sorted(splits)}, and no folder under {root} uses the real names, so "
                "the mapping cannot be recovered."
            )
        reference = {
            round(v["auroc"], 12): k
            for k, v in named["steps"][0]["scores"].items() if k != "mean"
        }
        _recover_split_names(b["steps"], reference, b["dir"])
        print(f"  baseline: {b['dir']} split names recovered from {named['dir']}'s baseline")

    for b in loaded:
        found = {k for s in b["steps"] for k in s["scores"]} - {"mean"}
        if found != set(splits):
            raise SystemExit(
                f"{b['dir']} scores different eval splits than this run "
                f"({sorted(found)} vs {sorted(splits)}) — they are not comparable. "
                "--baseline-dir currently only covers the harmful-to-human splits."
            )
    return loaded


def collect_eval(csv_path: Path | None, log_paths: list[Path]) -> dict | None:
    """Merge CSV rounds with log-recovered ones, tagging where each came from.

    The CSV is authoritative — it carries full precision, where the log table is
    rounded to 6dp — so logs only ever *fill in* rounds the CSV lacks.
    """
    from_csv = load_comparison_csv(csv_path) if csv_path else {}
    from_log = load_eval_logs(log_paths)
    if not from_csv and not from_log:
        return None

    rounds = []
    for n in sorted(set(from_csv) | set(from_log)):
        scores = from_csv.get(n) or from_log[n]
        rounds.append(
            {"n": n, "scores": scores, "source": "csv" if n in from_csv else "log"}
        )

    splits = [s for s in dict.fromkeys(k for r in rounds for k in r["scores"]) if s != "mean"]
    return {
        "rounds": rounds,
        "splits": splits,
        "csv": str(csv_path) if csv_path else "",
        "logs": [str(p) for p in log_paths],
        "baselines": [],
        "baseline_dir": "",
    }


# ------------------------------------------------------------------- rendering

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
  --new:#8a5b00; --new-bg:#fdf3dc;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --mean:#1a1d21; --chance:#b9b8b2;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
    --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
    --pos-bg:#2c1c1e; --pos-fg:#ff9a9a; --neg-bg:#17281d; --neg-fg:#86e0a5;
    --new:#f0c060; --new-bg:#33290f;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --mean:#e5e9ee; --chance:#4a4a47;
  }
}
:root[data-theme="light"] {
  --bg:#fff; --panel:#f6f7f9; --panel2:#eef0f3; --line:#d9dde3;
  --fg:#1a1d21; --muted:#666e79; --accent:#3f6fd8;
  --pos-bg:#fdeeee; --pos-fg:#9d2727; --neg-bg:#eaf5ee; --neg-fg:#1e6b3c;
  --new:#8a5b00; --new-bg:#fdf3dc;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --mean:#1a1d21; --chance:#b9b8b2;
}
:root[data-theme="dark"] {
  --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
  --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
  --pos-bg:#2c1c1e; --pos-fg:#ff9a9a; --neg-bg:#17281d; --neg-fg:#86e0a5;
  --new:#f0c060; --new-bg:#33290f;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --mean:#e5e9ee; --chance:#4a4a47;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1280px; margin:0 auto; padding:22px 18px 80px; }
h1 { margin:0 0 4px; font-size:21px; letter-spacing:-.01em; }
.sub { color:var(--muted); font-size:13px; }
.path { color:var(--muted); font-size:12px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; margin-top:2px; }
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
.stats { display:flex; flex-wrap:wrap; gap:10px; margin:18px 0 6px; }
.stat { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:9px 13px; min-width:108px; }
.stat .k { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.stat .v { font-size:19px; font-weight:650; margin-top:2px; }
.toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:14px 0 18px; }
.toolbar input[type=search] { flex:1 1 260px; min-width:200px; padding:8px 11px; font-size:14px;
  background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:7px; }
.chip { appearance:none; cursor:pointer; padding:7px 12px; font-size:13px; font-weight:550;
  background:var(--panel); color:var(--muted); border:1px solid var(--line); border-radius:999px; }
.chip[aria-pressed="true"] { background:var(--accent); border-color:var(--accent); color:#fff; }
.count { color:var(--muted); font-size:13px; margin-left:auto; }
.pair { border:1px solid var(--line); border-radius:10px; margin-bottom:14px; background:var(--panel); overflow:hidden; }
.pair .head { display:flex; flex-wrap:wrap; gap:8px; align-items:center;
  padding:8px 13px; border-bottom:1px solid var(--line); background:var(--panel2); }
.idx { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; color:var(--muted); }
.badge { font-size:11px; font-weight:700; letter-spacing:.03em; text-transform:uppercase;
  padding:3px 8px; border-radius:999px; border:1px solid var(--line); color:var(--muted); }
.badge.new { color:var(--new); background:var(--new-bg); border-color:var(--new); }
.shared { padding:12px 15px; border-bottom:1px solid var(--line); background:var(--bg); }
.lbl { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); margin-bottom:6px; }
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
/* --- eval tab ------------------------------------------------------------- */
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:16px 16px 10px; margin-top:18px; position:relative; }
.card h2 { font-size:15px; margin:0 0 3px; }
.card .src { color:var(--muted); font-size:11.5px; margin:0 0 10px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; }
.chartbox { position:relative; }
.chartbox svg { display:block; width:100%; height:auto; overflow:visible; }
.legend { display:flex; flex-wrap:wrap; gap:9px 15px; margin:11px 2px 2px;
  font-size:12.5px; color:var(--muted); }
.legend .lg { display:inline-flex; align-items:center; gap:6px; }
.legend .sw { width:15px; height:3px; border-radius:2px; background:var(--c); }
.legend .sw.mean { height:0; width:17px; background:none; border-top:2.5px dashed var(--mean); }
.s1 { --c:var(--s1); } .s2 { --c:var(--s2); } .s3 { --c:var(--s3); } .s4 { --c:var(--s4); }
.smean { --c:var(--mean); }
.line { fill:none; stroke:var(--c); stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
.line.mean { stroke-width:2.5; stroke-dasharray:6 4; }
.line.subject { stroke-width:3; }
.legend .sw.subject { height:4px; }
.subjectrow td { font-weight:700; }
details.more-eval { margin-top:12px; border-top:1px solid var(--line); padding-top:8px; }
details.more-eval > summary { cursor:pointer; font-size:12.5px; font-weight:600; color:var(--muted); }
details.more-eval .tcap { font-size:12px; font-weight:700; color:var(--muted); margin:12px 0 0; }
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
table.ev td .key { width:9px; height:9px; border-radius:2px; display:inline-block;
  margin-right:7px; vertical-align:middle; background:var(--c); }
table.ev .up { color:var(--neg-fg); } table.ev .down { color:var(--pos-fg); }
.recovered { display:inline-block; margin-left:6px; font-size:10px; font-weight:700;
  letter-spacing:.03em; text-transform:uppercase; color:var(--new); }
.more { display:block; width:100%; margin-top:8px; padding:11px; cursor:pointer;
  background:var(--panel); color:var(--fg); border:1px solid var(--line);
  border-radius:8px; font-size:14px; font-weight:600; }
.themer { position:fixed; right:14px; bottom:14px; z-index:5; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="path">__PROBE_DIR__</div>

  <div class="note">
    Each card is one <strong>contrastive pair</strong>: a red-team success (the
    <em>source</em>) beside the opposite-class conversation an LLM wrote from it. Both are
    trained on. Pairs are resolved through <code>contrastive_cache.jsonl</code>'s
    <code>original_messages</code> link rather than by row position, which stops aligning
    after iteration&nbsp;1. The per-iteration files are cumulative, so each tab here shows
    <strong>only what that iteration added</strong> &mdash; not everything that trained
    <code>probe_iter&lt;N&gt;.pkl</code>. The running total is in
    <em>cumulative pairs</em> below.
  </div>

  <div class="tabs" id="tabs" role="tablist"></div>

  <div id="pairs-view">
    <div class="stats" id="stats"></div>
    <div class="toolbar">
      <input type="search" id="q" placeholder="Search the prompt or either reply&hellip;" autocomplete="off">
      <button class="chip" id="f-pos" aria-pressed="false">Source = __POS_LABEL__</button>
      <button class="chip" id="f-neg" aria-pressed="false">Source = __NEG_LABEL__</button>
      <span class="count" id="count"></span>
    </div>
    <div id="list"></div>
    <button class="more" id="more" hidden>Show more</button>
  </div>

  <div id="eval-view" hidden>
    <div class="note">
      Each <code>iter&nbsp;N</code> here is the eval of <code>probe_iter&lt;N&gt;.pkl</code>, so it
      lines up with the contrastive tabs by probe index: <code>iter&nbsp;N</code> scores the probe
      that the pairs under <strong>iteration&nbsp;N</strong> trained. <code>iter&nbsp;0</code> is the
      baseline &mdash; trained on the base data alone, before any red-team sample existed &mdash; so
      it has no contrastive tab. <strong>mean</strong> is the unweighted mean over the splits, each
      of which is class-balanced.
    </div>
    <div class="toolbar" id="metric-chips"></div>
    <div class="card" id="eval-card"></div>
    <div class="card" id="baseline-card" hidden></div>
  </div>
</div>
<button class="chip themer" id="themer">Toggle theme</button>

<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("payload").textContent);
  var PAIRS = DATA.pairs, ITERS = DATA.iterations, EVAL = DATA.eval;
  var PAGE = 25;

  var active = ITERS.length - 1, shown = PAGE;
  var view = "pairs";
  var filters = { q: "", src: null };

  var elTabs = document.getElementById("tabs"),
      elStats = document.getElementById("stats"),
      elList = document.getElementById("list"),
      elMore = document.getElementById("more"),
      elCount = document.getElementById("count"),
      elPairsView = document.getElementById("pairs-view"),
      elEvalView = document.getElementById("eval-view"),
      elEvalCard = document.getElementById("eval-card"),
      elMetrics = document.getElementById("metric-chips");

  PAIRS.forEach(function (p) {
    p._hay = p.src_msgs.concat(p.gen_msgs)
      .map(function (m) { return m.content || ""; }).join(" \n ").toLowerCase();
  });

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function renderTabs() {
    elTabs.innerHTML = "";
    ITERS.forEach(function (it, i) {
      var b = document.createElement("button");
      b.className = "tab";
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", view === "pairs" && i === active ? "true" : "false");
      b.innerHTML = "iteration " + it.n + '<span class="cnt">'
        + (it.first ? it.idx.length + " pairs" : "+" + it.idx.length + " new") + "</span>";
      b.onclick = function () { view = "pairs"; active = i; shown = PAGE; render(); };
      elTabs.appendChild(b);
    });
    if (!EVAL) return;
    var e = document.createElement("button");
    e.className = "tab";
    e.setAttribute("role", "tab");
    e.setAttribute("aria-selected", view === "eval" ? "true" : "false");
    e.innerHTML = 'eval scores<span class="cnt">' + EVAL.rounds.length + " probes</span>";
    e.onclick = function () { view = "eval"; render(); };
    elTabs.appendChild(e);
  }

  function renderStats() {
    var it = ITERS[active], pos = 0;
    it.idx.forEach(function (i) { if (PAIRS[i].src_label === DATA.pos_label) pos++; });
    var cells = [
      [it.first ? "pairs added" : "new pairs added", it.idx.length],
      ["cumulative pairs", it.cum],
      ["samples trained", it.rows],
      ["dropped since prev", it.first ? "—" : it.dropped],
      ["source = " + DATA.pos_label, pos],
      ["source = " + DATA.neg_label, it.idx.length - pos]
    ];
    if (it.returned) cells.push(["re-added after a drop", it.returned]);
    if (it.unpaired) cells.push(["unpaired rows", it.unpaired]);
    elStats.innerHTML = cells.map(function (c) {
      return '<div class="stat"><div class="k">' + esc(c[0]) + '</div><div class="v">' + esc(c[1]) + "</div></div>";
    }).join("")
      + '<div class="stat"><div class="k">source file</div><div class="v" style="font-size:12px;'
      + 'font-family:ui-monospace,Menlo,monospace;padding-top:5px">' + esc(it.file) + "</div></div>";
  }

  function visible() {
    var out = [];
    ITERS[active].idx.forEach(function (i) {
      var p = PAIRS[i];
      if (filters.src && p.src_label !== filters.src) return;
      if (filters.q && p._hay.indexOf(filters.q) === -1) return;
      out.push(i);
    });
    return out;
  }

  function turns(msgs) {
    return msgs.map(function (m) {
      return '<div class="turn"><div class="role">' + esc(m.role)
        + '</div><div class="text">' + esc(m.content) + "</div></div>";
    }).join("");
  }

  function card(index, ord) {
    var p = PAIRS[index];
    var n = Math.min(p.src_msgs.length, p.gen_msgs.length), same = 0;
    while (same < n
      && p.src_msgs[same].role === p.gen_msgs[same].role
      && p.src_msgs[same].content === p.gen_msgs[same].content) same++;

    var head = '<div class="head"><span class="idx">#' + ord + " · pair " + index + "</span>"
      + '<span class="badge">source: ' + esc(p.src_label) + "</span>"
      + '<span class="badge">pair: ' + esc(p.gen_label) + "</span></div>";

    var shared = same > 0
      ? '<div class="shared"><div class="lbl">shared turns</div>'
        + turns(p.src_msgs.slice(0, same)) + "</div>"
      : "";

    var srcTail = p.src_msgs.slice(same), genTail = p.gen_msgs.slice(same);
    var cols = '<div class="cols">'
      + '<div class="side ' + (p.src_class === "positive" ? "pos" : "neg") + '">'
      + '<div class="lbl">source — red-team success — ' + esc(p.src_label) + "</div>"
      + (srcTail.length ? turns(srcTail) : '<div class="text" style="opacity:.6">(identical to the pair)</div>')
      + "</div>"
      + '<div class="side ' + (p.gen_class === "positive" ? "pos" : "neg") + '">'
      + '<div class="lbl">contrastive pair — generated — ' + esc(p.gen_label) + "</div>"
      + (genTail.length ? turns(genTail) : '<div class="text" style="opacity:.6">(identical to the source)</div>')
      + "</div></div>";

    var why = p.why
      ? '<details class="why"><summary>Why this pair was written'
        + (p.model ? " (" + esc(p.model) + ")" : "") + '</summary><div class="body">'
        + esc(p.why) + "</div></details>"
      : "";

    return '<div class="pair">' + head + shared + cols + why + "</div>";
  }

  // ------------------------------------------------------------ eval scores
  var METRICS = [
    { key: "auroc", label: "AUROC", chance: 0.5 },
    { key: "accuracy", label: "Accuracy", chance: 0.5 },
    { key: "tpr_at_fpr", label: "TPR @ 1% FPR", chance: null }
  ];
  var metric = METRICS[0];
  // mr is the direct-label gutter — wide enough for the longest split name.
  var G = { W: 760, H: 330, mt: 16, mr: 148, mb: 34, ml: 48 };
  var pw = G.W - G.ml - G.mr, ph = G.H - G.mt - G.mb;
  var CHARTS = {};    // id -> {series, its, dom, tipTitle}; drawChart writes, hover reads

  function evalSeries() {
    function pts(name) {
      var out = [];
      EVAL.rounds.forEach(function (r) {
        var s = r.scores[name];
        if (s && s[metric.key] != null) out.push([r.n, s[metric.key]]);
      });
      return out;
    }
    var out = EVAL.splits.map(function (sp, i) {
      return { name: sp, cls: "s" + ((i % 4) + 1), pts: pts(sp) };
    });
    out.push({ name: "mean", cls: "smean", mean: true, pts: pts("mean") });
    return out.filter(function (s) { return s.pts.length; });
  }

  // A domain snapped to round ticks with a little headroom, so the y axis reads in
  // familiar increments instead of tracking the data's exact extremes.
  function domain(series) {
    var vals = [];
    series.forEach(function (s) { s.pts.forEach(function (p) { vals.push(p[1]); }); });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), span = hi - lo;
    var step = span < 0.25 ? 0.05 : (span > 0.6 ? 0.2 : 0.1);
    var ymin = Math.max(0, Math.floor((lo - step / 2) / step) * step);
    var ymax = Math.min(1, Math.ceil((hi + step / 2) / step) * step);
    var ticks = [];
    for (var v = ymin; v <= ymax + 1e-9; v += step) ticks.push(Math.round(v * 1e4) / 1e4);
    return { ymin: ymin, ymax: ymax, ticks: ticks };
  }

  function xPos(it, its) {
    return G.ml + (its.length === 1 ? pw / 2
      : (it - its[0]) / (its[its.length - 1] - its[0]) * pw);
  }
  function yPos(v, d) { return G.mt + (1 - (v - d.ymin) / (d.ymax - d.ymin)) * ph; }

  function drawChart(id, series, opts) {
    opts = opts || {};
    var seen = {};
    series.forEach(function (s) { s.pts.forEach(function (p) { seen[p[0]] = 1; }); });
    var its = Object.keys(seen).map(Number).sort(function (a, b) { return a - b; });
    var d = domain(series);
    CHARTS[id] = { series: series, its: its, dom: d, tipTitle: opts.tipTitle };

    var s = "";
    d.ticks.forEach(function (v) {
      var y = yPos(v, d);
      var isChance = metric.chance != null && Math.abs(v - metric.chance) < 1e-9;
      s += '<line class="' + (isChance ? "chanceline" : "gridline") + '" x1="' + G.ml
        + '" y1="' + y + '" x2="' + (G.ml + pw) + '" y2="' + y + '"/>'
        + '<text class="ylab" x="' + (G.ml - 7) + '" y="' + (y + 3.5)
        + '" text-anchor="end">' + v.toFixed(2) + "</text>";
    });
    // Sits just above its own line, inside the plot — the right gutter belongs to the
    // direct labels, and a "chance" caption out there would collide with them.
    if (metric.chance != null && metric.chance >= d.ymin && metric.chance <= d.ymax) {
      s += '<text class="ylab" x="' + (G.ml + pw - 4) + '" y="'
        + (yPos(metric.chance, d) - 6) + '" text-anchor="end">chance</text>';
    }
    s += '<line class="axline" x1="' + G.ml + '" y1="' + (G.mt + ph)
      + '" x2="' + (G.ml + pw) + '" y2="' + (G.mt + ph) + '"/>';
    its.forEach(function (it) {
      s += '<text class="tick" x="' + xPos(it, its) + '" y="' + (G.mt + ph + 16)
        + '" text-anchor="middle">' + esc(opts.xtick ? opts.xtick(it) : "iter " + it) + "</text>";
    });
    if (opts.xtitle) {
      s += '<text class="tick" x="' + (G.ml + pw / 2) + '" y="' + (G.mt + ph + 31)
        + '" text-anchor="middle">' + esc(opts.xtitle) + "</text>";
    }

    series.forEach(function (x) {
      var pts = x.pts.map(function (p) {
        return xPos(p[0], its) + "," + yPos(p[1], d);
      }).join(" ");
      s += '<polyline class="line ' + x.cls + (x.mean ? " mean" : "")
        + (x.subject ? " subject" : "") + '" points="' + pts + '"/>';
      x.pts.forEach(function (p) {
        s += '<circle class="pt ' + x.cls + '" cx="' + xPos(p[0], its)
          + '" cy="' + yPos(p[1], d) + '" r="4"/>';
      });
    });

    // Direct labels at the line ends, pushed apart so near-equal finishes stay legible.
    // Only labels sharing (nearly) the same x can actually collide — series that end at
    // different steps sit in different columns — so de-collide per column, not globally.
    var columns = {};
    series.forEach(function (x) {
      var last = x.pts[x.pts.length - 1];
      var lx = xPos(last[0], its) + 8;
      (columns[Math.round(lx / 6)] = columns[Math.round(lx / 6)] || []).push(
        { name: x.name, cls: x.cls, x: lx, y: yPos(last[1], d) });
    });
    Object.keys(columns).forEach(function (k) {
      var group = columns[k].sort(function (a, b) { return a.y - b.y; });
      for (var i = 1; i < group.length; i++) {
        if (group[i].y - group[i - 1].y < 13) group[i].y = group[i - 1].y + 13;
      }
      // Pushing apart only moves labels down, so a cluster on the floor (e.g. several
      // splits at TPR 0) would spill onto the x-axis ticks. Slide the stack back inside.
      var over = group[group.length - 1].y - (G.mt + ph);
      if (over > 0) group.forEach(function (l) { l.y -= over; });
      var under = G.mt - group[0].y;
      if (under > 0) group.forEach(function (l) { l.y += under; });
      group.forEach(function (l) {
        s += '<text class="endlab ' + l.cls + '" x="' + l.x + '" y="' + (l.y + 3.5) + '">'
          + esc(l.name) + "</text>";
      });
    });

    s += '<line class="crosshair" id="ch-' + id + '" y1="' + G.mt
      + '" y2="' + (G.mt + ph) + '"/>';
    its.forEach(function (it) {
      s += '<rect class="hz-' + id + '" x="' + (xPos(it, its) - pw / (its.length * 2) - 2)
        + '" y="' + G.mt + '" width="' + (pw / its.length + 4) + '" height="' + ph
        + '" fill="transparent" data-it="' + it + '"/>';
    });
    return '<div class="chartbox"><svg id="svg-' + id + '" viewBox="0 0 ' + G.W + " " + G.H
      + '" role="img" aria-label="' + esc(opts.alt || (metric.label + " chart")) + '">' + s
      + '</svg><div class="tip" id="tip-' + id + '"></div></div>';
  }

  function legend(series) {
    return '<div class="legend">' + series.map(function (x) {
      return '<span class="lg"><span class="sw ' + (x.mean ? "mean " : "")
        + (x.subject ? "subject " : "") + x.cls + '"></span>' + esc(x.name) + "</span>";
    }).join("") + "</div>";
  }

  function evalTable(series) {
    var its = CHARTS["ev"].its;
    var head = "<tr><th>split</th>" + its.map(function (i) {
      var r = EVAL.rounds.filter(function (x) { return x.n === i; })[0];
      return "<th>iter " + i + (r && r.source === "log"
        ? '<span class="recovered" title="recovered from the run log">log</span>' : "") + "</th>";
    }).join("") + "<th>&Delta; vs iter " + its[0] + "</th></tr>";

    var body = series.map(function (x) {
      var m = {};
      x.pts.forEach(function (p) { m[p[0]] = p[1]; });
      var first = m[its[0]], last = x.pts[x.pts.length - 1][1];
      var delta = first == null ? null : last - first;
      return '<tr class="' + (x.mean ? "mean " : "") + x.cls + '"><td><span class="key"></span>'
        + esc(x.name) + "</td>"
        + its.map(function (i) {
            return "<td>" + (m[i] == null ? "&mdash;" : m[i].toFixed(3)) + "</td>";
          }).join("")
        + '<td class="' + (delta == null ? "" : delta >= 0 ? "up" : "down") + '">'
        + (delta == null ? "&mdash;" : (delta >= 0 ? "+" : "−") + Math.abs(delta).toFixed(3))
        + "</td></tr>";
    }).join("");
    return '<div class="tblwrap"><table class="ev"><thead>' + head
      + "</thead><tbody>" + body + "</tbody></table></div>";
  }

  function renderMetricChips() {
    elMetrics.innerHTML = METRICS.map(function (m) {
      return '<button class="chip" data-m="' + m.key + '" aria-pressed="'
        + (m.key === metric.key) + '">' + esc(m.label) + "</button>";
    }).join("");
    elMetrics.querySelectorAll(".chip").forEach(function (b) {
      b.onclick = function () {
        metric = METRICS.filter(function (m) { return m.key === b.dataset.m; })[0];
        render();
      };
    });
  }

  function wireHover(id) {
    var svg = document.getElementById("svg-" + id);
    if (!svg) return;
    var chart = CHARTS[id],
        box = svg.closest(".chartbox"),
        tip = document.getElementById("tip-" + id),
        ch = document.getElementById("ch-" + id);
    function show(it) {
      var x = xPos(it, chart.its);
      ch.setAttribute("x1", x); ch.setAttribute("x2", x); ch.classList.add("hi");
      tip.innerHTML = '<div class="th">' + (chart.tipTitle ? chart.tipTitle(it) : "iter " + it)
        + "</div>"
        + chart.series.map(function (sr) {
            var hit = sr.pts.filter(function (p) { return p[0] === it; })[0];
            if (!hit) return "";
            var note = sr.labels && sr.labels[it] ? " · " + esc(sr.labels[it]) : "";
            return '<div class="row"><span><span class="dot" style="background:var(--'
              + (sr.mean || sr.subject ? "mean" : sr.cls) + ')"></span>' + esc(sr.name) + note
              + "</span><b>" + hit[1].toFixed(3) + "</b></div>";
          }).join("");
      tip.style.opacity = 1;
      var rect = svg.getBoundingClientRect(), scale = rect.width / G.W, px = x * scale + 14;
      if (px > rect.width - tip.offsetWidth - 6) px = x * scale - tip.offsetWidth - 14;
      tip.style.left = Math.max(4, px) + "px";
      tip.style.top = (G.mt * scale + 6) + "px";
    }
    box.querySelectorAll(".hz-" + id).forEach(function (rc) {
      rc.addEventListener("mousemove", function () { show(+rc.dataset.it); });
      rc.addEventListener("mouseenter", function () { show(+rc.dataset.it); });
    });
    box.addEventListener("mouseleave", function () {
      ch.classList.remove("hi"); tip.style.opacity = 0;
    });
  }

  // ------------------------------------- comparison against the prompted baselines
  // This run is one series among the simple-prompted settings, on a shared x axis of
  // "retrain step" (0 = each series' own baseline probe). `split` picks which score to
  // plot: "mean", or one eval split by name.
  function baselineSeries(split) {
    var subject = { name: DATA.run_label, cls: "smean", subject: true, labels: {}, pts: [] };
    EVAL.rounds.forEach(function (r, i) {
      subject.labels[i] = "iter" + r.n;
      if (r.scores[split] && r.scores[split][metric.key] != null) {
        subject.pts.push([i, r.scores[split][metric.key]]);
      }
    });
    var out = [subject];
    EVAL.baselines.forEach(function (b) {
      var labels = {}, pts = [];
      b.steps.forEach(function (st, i) {
        labels[i] = st.label;
        if (st.scores[split] && st.scores[split][metric.key] != null) {
          pts.push([i, st.scores[split][metric.key]]);
        }
      });
      out.push({ name: b.name, cls: b.cls, labels: labels, pts: pts });
    });
    return out.filter(function (s) { return s.pts.length; });
  }

  function baselineTable(series) {
    var last = 0;
    series.forEach(function (s) { last = Math.max(last, s.pts[s.pts.length - 1][0]); });
    var steps = [];
    for (var i = 0; i <= last; i++) steps.push(i);

    var head = "<tr><th>series</th>" + steps.map(function (i) {
      return "<th>step " + i + "</th>";
    }).join("") + "<th>best</th></tr>";

    var body = series.map(function (x) {
      var m = {};
      x.pts.forEach(function (p) { m[p[0]] = p[1]; });
      var best = Math.max.apply(null, x.pts.map(function (p) { return p[1]; }));
      var bestAt = x.pts.filter(function (p) { return p[1] === best; })[0][0];
      return '<tr class="' + (x.subject ? "subjectrow " : "") + x.cls
        + '"><td><span class="key"></span>' + esc(x.name) + "</td>"
        + steps.map(function (i) {
            if (m[i] == null) return "<td>&mdash;</td>";
            return '<td title="' + esc(x.labels[i] || "") + '">'
              + (i === bestAt ? "<b>" + m[i].toFixed(3) + "</b>" : m[i].toFixed(3)) + "</td>";
          }).join("")
        + "<td><b>" + best.toFixed(3) + "</b> <span style=\"opacity:.65\">"
        + esc(x.labels[bestAt] || ("step " + bestAt)) + "</span></td></tr>";
    }).join("");
    return '<div class="tblwrap"><table class="ev"><thead>' + head
      + "</thead><tbody>" + body + "</tbody></table></div>";
  }

  function renderBaselines() {
    if (!EVAL.baselines || !EVAL.baselines.length) {
      document.getElementById("baseline-card").hidden = true;
      return;
    }
    var card = document.getElementById("baseline-card");
    card.hidden = false;
    var series = baselineSeries("mean");
    var svg = drawChart("bl", series, {
      xtick: function (i) { return String(i); },
      xtitle: "retrain step (0 = baseline)",
      tipTitle: function (i) { return "step " + i; },
      alt: "mean " + metric.label + " per retrain step, this run against the "
        + "simple-prompted baselines"
    });
    var perSplit = EVAL.splits.map(function (sp) {
      return '<div class="tcap">' + esc(sp) + "</div>" + baselineTable(baselineSeries(sp));
    }).join("");

    card.innerHTML = "<h2>Mean " + esc(metric.label)
      + " vs. the simple-prompted baselines</h2>"
      + '<p class="src">' + esc(EVAL.baseline_dir
          + "/results-{all,constraint1,constraint2}-hh/*.csv") + "</p>"
      + '<div class="note" style="margin:0 0 12px;background:var(--panel2)">Each series walks its own retrain '
      + 'sequence, so <strong>step</strong> is a position in that sequence, not a shared '
      + 'clock &mdash; step 0 is every series\' own baseline probe, and the baselines\' '
      + 'steps are their config versions (hover a point for its name). The prompted '
      + 'baselines start from a baseline probe of their own (mean AUROC 0.616) rather than '
      + 'this run\'s <code>iter0</code> (0.614), so the starting points are close but not '
      + 'the same probe.</div>'
      + svg + legend(series) + baselineTable(series)
      + '<details class="more-eval"><summary>Per eval split</summary>' + perSplit
      + "</details>";
    wireHover("bl");
  }

  function renderEval() {
    renderMetricChips();
    var series = evalSeries();
    var svg = drawChart("ev", series, {
      alt: metric.label + " per iteration, one line per eval split plus the mean"
    });
    var src = EVAL.csv ? EVAL.csv : "";
    if (EVAL.rounds.some(function (r) { return r.source === "log"; })) {
      src += (src ? "  +  " : "") + EVAL.logs.join("  +  ")
        + "  (rounds the CSV never got: the comparison CSV is only written when the whole"
        + " loop finishes, so an interrupted run leaves them in the log alone)";
    }
    elEvalCard.innerHTML = "<h2>" + esc(metric.label) + " per iteration</h2>"
      + '<p class="src">' + esc(src) + "</p>" + svg + legend(series) + evalTable(series);
    wireHover("ev");
    renderBaselines();
  }

  function render() {
    renderTabs();
    if (view === "eval") {
      elPairsView.hidden = true;
      elEvalView.hidden = false;
      renderEval();
      return;
    }
    elEvalView.hidden = true;
    elPairsView.hidden = false;
    renderStats();
    var vis = visible(), total = ITERS[active].idx.length;
    elCount.textContent = vis.length + " of " + total
      + (ITERS[active].first ? " pairs" : " pairs added this iteration");
    if (!vis.length) {
      elList.innerHTML = '<div class="empty">'
        + (total ? "No pairs match these filters." : "This iteration added no new pairs.")
        + "</div>";
      elMore.hidden = true;
      return;
    }
    elList.innerHTML = vis.slice(0, shown).map(function (i, k) { return card(i, k + 1); }).join("");
    elMore.hidden = vis.length <= shown;
    elMore.textContent = "Show more (" + (vis.length - shown) + " remaining)";
  }

  document.getElementById("q").addEventListener("input", function (e) {
    filters.q = e.target.value.trim().toLowerCase(); shown = PAGE; render();
  });
  function chip(id, other, value) {
    var el = document.getElementById(id);
    el.addEventListener("click", function () {
      var on = el.getAttribute("aria-pressed") !== "true";
      el.setAttribute("aria-pressed", on ? "true" : "false");
      if (on && other) document.getElementById(other).setAttribute("aria-pressed", "false");
      filters.src = on ? value : null;
      shown = PAGE; render();
    });
  }
  chip("f-pos", "f-neg", DATA.pos_label);
  chip("f-neg", "f-pos", DATA.neg_label);
  elMore.addEventListener("click", function () { shown += PAGE; render(); });
  document.getElementById("themer").addEventListener("click", function () {
    var r = document.documentElement, cur = r.getAttribute("data-theme");
    if (!cur) cur = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    r.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
  });

  render();
})();
</script>
</body>
</html>
"""


def render(
    run: dict, *, title: str, subtitle: str, probe_dir: Path, eval_data: dict | None,
    run_label: str,
) -> str:
    pairs = run["pairs"]
    labels = {p["src_label"] for p in pairs} | {p["gen_label"] for p in pairs}
    # The positive class is whichever label sits on a sample dumped as "positive".
    pos_label = next(
        (p["src_label"] for p in pairs if p["src_class"] == "positive"),
        next((p["gen_label"] for p in pairs if p["gen_class"] == "positive"), "positive"),
    )
    neg_label = next((l for l in sorted(labels) if l != pos_label), "negative")

    payload = json.dumps(
        {
            "pairs": pairs,
            "iterations": run["iterations"],
            "pos_label": pos_label,
            "neg_label": neg_label,
            "eval": eval_data,
            "run_label": run_label,
        },
        ensure_ascii=False,
    ).replace("</", r"<\/")

    html = _TEMPLATE
    for token, value in (
        ("__TITLE__", title),
        ("__SUBTITLE__", subtitle),
        ("__PROBE_DIR__", str(probe_dir)),
        ("__POS_LABEL__", pos_label),
        ("__NEG_LABEL__", neg_label),
    ):
        html = html.replace(token, value)
    return html.replace("__PAYLOAD__", payload)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--probe-dir", type=Path, required=True,
                    help="Probe output dir with redteam_postprocessed_iter*.jsonl "
                         "and contrastive_cache.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="Output HTML path")
    ap.add_argument("--title", default="", help="Page heading")
    ap.add_argument("--subtitle", default="", help="One-line subheading")
    ap.add_argument("--comparison-csv", type=Path,
                    help="Run's eval comparison CSV; adds the eval-scores tab")
    ap.add_argument("--eval-log", type=Path, action="append", default=[],
                    metavar="LOG",
                    help="Run log to recover eval tables from, for rounds the comparison "
                         "CSV never got (an interrupted run never writes it). Repeatable.")
    ap.add_argument("--baseline-dir", type=Path,
                    help="cresults root (holding results-{all,constraint1,constraint2}-hh); "
                         "adds the simple-prompted comparison panel to the eval tab")
    ap.add_argument("--run-label", default="",
                    help="Short name for this run in the baseline comparison "
                         "(default: the probe dir name)")
    args = ap.parse_args()

    run = collect_run(args.probe_dir)
    eval_data = collect_eval(args.comparison_csv, args.eval_log)
    if args.baseline_dir:
        if not eval_data:
            raise SystemExit("--baseline-dir needs --comparison-csv / --eval-log too: "
                             "there is nothing to compare the baselines against.")
        eval_data["baselines"] = load_baselines(args.baseline_dir, eval_data["splits"])
        eval_data["baseline_dir"] = str(args.baseline_dir)
    title = args.title or f"Contrastive pairs — {args.probe_dir.name}"
    html = render(run, title=title, subtitle=args.subtitle, probe_dir=args.probe_dir,
                  eval_data=eval_data, run_label=args.run_label or args.probe_dir.name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")

    print(f"{args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(run['pairs'])} distinct pairs across {len(run['iterations'])} iterations "
          f"(each tab shows only that iteration's additions)")
    for it in run["iterations"]:
        print(f"    iter{it['n']}: {len(it['idx']):5d} shown  "
              f"(cumulative {it['cum']:5d}, {it['rows']:5d} samples)  "
              f"dropped {it['dropped']:3d}  re-added {it['returned']:3d}  "
              f"unpaired {it['unpaired']}")
    if eval_data:
        print(f"  eval tab: {len(eval_data['rounds'])} rounds over "
              f"{len(eval_data['splits'])} splits")
        for rnd in eval_data["rounds"]:
            mean = rnd["scores"].get("mean", {}).get("auroc")
            print(f"    iter{rnd['n']}: mean AUROC "
                  f"{'—' if mean is None else f'{mean:.4f}'}  [{rnd['source']}]")
        for b in eval_data["baselines"]:
            best = max(b["steps"], key=lambda s: s["scores"]["mean"]["auroc"])
            print(f"    baseline {b['name']:<20} {len(b['steps'])} steps  "
                  f"best mean AUROC {best['scores']['mean']['auroc']:.4f} "
                  f"at {best['label']}  ({b['dir']})")
    else:
        print("  eval tab: skipped (no --comparison-csv / --eval-log)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
