#!/usr/bin/env python
"""Score experiment9's HIGH-STAKES vintage probes on the human-harm and instruction eval sets.

``attribution_vintage.py`` fits v0..v3 for each arm and scores every fit on the four
``eval_datasets/`` splits — the concept those probes were trained for. This script asks
the complementary question: **what does the high-stakes red-team data do to the probe
off its own concept?** The vintage curve says iteration-2 and -3 red-team pairs keep
raising high-stakes AUROC; whether that is the probe learning *high-stakes* or learning
*something more generic the attacker happened to write* is not decidable from that
curve alone. Two eval sets the probe has never seen — human-harm and
instruction-following — are one way to tell those apart.

**Nothing is refitted here.** The 80 committed ``vintage/fits/*.pt`` checkpoints *are*
the probes behind ``vintage/SUMMARY.md``; they are reloaded through
``attribution_vintage._load_fitted`` — the same path the sweep's own resume uses, and
the one ``--self-check`` verifies against a live fit — and scored against two more eval
sets. So every number here is directly comparable, fit for fit and seed for seed, to the
high-stakes number in ``vintage_auroc.csv``.

WHAT THE AUROC MEANS OFF-CONCEPT
--------------------------------
The probe emits one score, "how high-stakes is this conversation". Scored on a
human-harm split, its AUROC is measured against *that split's* label — how well the
high-stakes score ranks harmful above not-harmful. So:

- **0.5** — the high-stakes score carries no information about the other concept.
- **> 0.5** — the two concepts are entangled in this probe: it ranks the other concept's
  positive class high without ever having been trained to.
- **< 0.5** — genuinely anti-correlated, and reported unflipped. ``1 - auroc`` is the
  strength of the reversed relation; flipping it in the table would hide the sign, which
  is the interesting part.

An off-concept AUROC is **not** a quality score for the probe — it was never asked to do
this job. It is only comparable *across vintages*, which is the whole point: v0 fixes
whatever entanglement the base training data already carries, and v1/v2/v3 show which
way the red-team data moves it.

WHY THE ACTIVATIONS ARE INTERCHANGEABLE ACROSS CONCEPTS
-------------------------------------------------------
Every published blob — high-stakes, human-harm and instruction alike — is
``google/gemma-3-27b-it`` layer 32 under the same two loader transforms
(``combine_consecutive_messages`` + ``convert_tool_to_assistant``, both true in all
three concepts' configs). An activation is a property of the conversation and the frozen
extraction model, not of the probe head that will read it, so the human-harm blobs
computed for the experiment11 arms and the instruction blobs computed for
experiment_instruction_cloud_1 are exactly what a high-stakes probe needs too. Only the
labels differ, and those come from the split JSONL.

LABELS COME FROM THE SPLIT, NOT FROM THE PROBE
----------------------------------------------
``LabelledDataset.from_pandas`` maps a label string to 1/0 when it matches
``pos_class_label``/``neg_class_label`` and otherwise **leaves it as the raw string**
(``dataset.py:311-317``) — no error. Loading a human-harm split under the high-stakes
probe's labels would therefore hand back a column of strings and blow up much later, in
``labels_torch()``. Each target concept below names its own pair, and the probe's own
labels are used for nothing but the self-check.

Usage:

    .venv_claude/bin/python scripts/vintage_cross_concept_eval.py --self-check
    .venv_claude/bin/python scripts/vintage_cross_concept_eval.py --concepts hu_harm instructions
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
from attribution_vintage import _blob, _load_fitted, fit_path

REPO = A.REPO
VINTAGE_DIR = REPO / "results_hs_gemma27b_batch_ablation/vintage"
OUT_DIR = VINTAGE_DIR / "cross_concept"


@dataclass(frozen=True)
class Target:
    """An eval set to score the high-stakes probes against.

    ``pos_label``/``neg_label`` are that concept's own class strings — see the note above
    on why they cannot be the probe's. ``activations_dir`` holds one
    ``<split>-acts_full.pt`` per split, the same path-keyed layout
    ``get_performances`` derives.
    """

    eval_dir: Path
    activations_dir: Path
    pos_label: str
    neg_label: str

    def splits(self) -> list[str]:
        return sorted(p.stem for p in self.eval_dir.glob("*.jsonl"))


TARGETS: dict[str, Target] = {
    # The concept the probes WERE trained for. Present so --self-check can reproduce
    # vintage_auroc.csv through this script's own scoring path rather than trusting it.
    "hs": Target(
        REPO / "eval_datasets",
        A.EVAL_ACTIVATIONS_DIR,
        "high-stakes",
        "low-stakes",
    ),
    "hu_harm": Target(
        REPO / "eval_dataset_hu_ha",
        REPO / "results_hu_harm_gemma27b_batch_ablation/eval_activations",
        "harmful_to_human",
        "not_harmful_to_human",
    ),
    "instructions": Target(
        REPO / "eval_instructions",
        REPO / "results_instructions_gemma27b_shared/eval_activations",
        "assistant_follows_the_instruction",
        "assistant_does_not_follow_the_instruction",
    ),
}


def redteam_rows_by_fit() -> dict[tuple[str, int], int]:
    """``{(arm, vintage): n_redteam_rows}`` off the sweep's progress sidecar.

    Only used to label the tables with the same row counts ``SUMMARY.md`` carries, so a
    reader can line the two up. Absent sidecar simply means the column reads 0.
    """
    path = VINTAGE_DIR / "vintage_progress.jsonl"
    out: dict[tuple[str, int], int] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[(row["arm"], int(row["vintage"]))] = int(row["n_redteam_rows"])
    return out


def load_fits(arms: list[str], vintages: list[int], seeds: list[int]) -> dict:
    """``{(arm, vintage, seed): probe}`` for every checkpoint that exists.

    A checkpoint is 13 KB, so all 80 are held at once and every split's blob is opened
    exactly once for the whole set — the same trade ``score_many`` makes for ten seeds.
    Missing checkpoints are reported and skipped rather than refitted: refitting here
    would silently mix probes from two different runs into one table.
    """
    probes, missing = {}, []
    for arm in arms:
        ref = A.load_probe(A.ARMS[arm] / "probe_iter3.pkl")
        for k in vintages:
            for seed in seeds:
                path = fit_path(VINTAGE_DIR, arm, k, seed)
                if not path.exists():
                    missing.append(path.name)
                    continue
                probe, _ = _load_fitted(ref, path)
                probes[(arm, k, seed)] = probe
    if missing:
        print(f"  WARNING: {len(missing)} checkpoint(s) missing: {missing[:5]}"
              + (" ..." if len(missing) > 5 else ""), flush=True)
    return probes


def score_target(name: str, target: Target, probes: dict) -> list[dict]:
    """AUROC of every probe on every split of one eval set, one blob open at a time.

    The blob is opened with ``mmap=True`` so it lives in page cache rather than
    anonymous memory (the human-harm ``eval_balanced_refusal`` blob alone is 3.7 GB on a
    15 GB box), and every probe is scored while it is warm before the next split is
    opened.
    """
    import contextlib
    import io

    from tuberlens.interfaces.activations import Activation
    from tuberlens.interfaces.dataset import LabelledDataset

    rows: list[dict] = []
    for split in target.splits():
        blob_path = target.activations_dir / f"{split}-acts_full.pt"
        if not blob_path.exists():
            raise SystemExit(
                f"{name}/{split}: no cached activations at {blob_path}. Fetch them first "
                f"(scripts/fetch_kaggle_eval_activations.py --concept {name})."
            )
        ds = LabelledDataset.load_from(
            target.eval_dir / f"{split}.jsonl",
            pos_class_label=target.pos_label,
            neg_class_label=target.neg_label,
            combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
        )
        y = ds.labels_torch().float().cpu().numpy()
        blob = _blob(blob_path)
        n_blob = int(blob["activations"].shape[0])
        if n_blob != len(ds):
            raise SystemExit(
                f"{name}/{split}: {blob_path.name} has {n_blob} rows but the split has "
                f"{len(ds)} — refusing to score mismatched activations."
            )
        ds = ds.assign(
            activations=blob["activations"],
            attention_mask=blob["attention_mask"],
            input_ids=blob["input_ids"],
        )
        acts = Activation.from_dataset(ds)
        for (arm, k, seed), probe in probes.items():
            # stderr too, not just stdout: the classifier's per-batch tqdm writes there,
            # and 80 probes x 7 splits of bars would bury the log. TQDM_DISABLE (which
            # attribution_vintage sets on import) is not honoured by this tqdm build.
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                s = probe._classifier.logits(acts)
            au = A.auroc_both(y, s.float().cpu().numpy())
            rows.append({
                "concept": name, "arm": arm, "vintage": k, "seed": seed,
                "dataset": split, "n_rows": len(ds),
                "auroc_pipeline": au["pipeline"], "auroc_rank": au["rank"],
            })
            del s
        print(f"  {name}/{split}: {len(ds)} rows x {len(probes)} probe(s) scored", flush=True)
        del ds, blob, acts
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def add_means(rows: list[dict], target: Target) -> list[dict]:
    """Append a per-(arm, vintage, seed) ``mean`` row across the concept's splits.

    Unweighted across splits, matching ``score_many``'s ``mean`` — the splits differ in
    size (114 to 400 rows) and a size-weighted mean would let the largest split speak
    for the concept.
    """
    splits = target.splits()
    by_fit: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by_fit.setdefault((r["concept"], r["arm"], r["vintage"], r["seed"]), {})[r["dataset"]] = r
    out = list(rows)
    for (concept, arm, k, seed), per_split in by_fit.items():
        if set(per_split) != set(splits):
            continue
        out.append({
            "concept": concept, "arm": arm, "vintage": k, "seed": seed,
            "dataset": "mean", "n_rows": sum(per_split[s]["n_rows"] for s in splits),
            "auroc_pipeline": float(np.mean([per_split[s]["auroc_pipeline"] for s in splits])),
            "auroc_rank": float(np.mean([per_split[s]["auroc_rank"] for s in splits])),
        })
    return out


def summarize(rows: list[dict], rt_rows: dict[tuple[str, int], int]) -> list[dict]:
    """Mean +/- sd over seeds per (concept, arm, vintage, split), both AUROC scales.

    Unpaired, exactly as ``attribution_vintage.summarize`` reports it: these are
    independent fits with independent initialisations, so the seed sd is the yardstick a
    between-vintage gap has to clear.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["concept"], r["arm"], r["vintage"], r["dataset"]), []).append(r)

    out = []
    for (concept, arm, k, split), rs in sorted(groups.items()):
        for scale in ("pipeline", "rank"):
            vals = [r[f"auroc_{scale}"] for r in rs]
            out.append({
                "concept": concept, "arm": arm, "vintage": k,
                "n_redteam_rows": rt_rows.get((arm, k), 0),
                "n_seeds": len(vals), "dataset": split, "scale": scale,
                "mean": float(np.mean(vals)),
                "sd": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                "min": float(min(vals)), "max": float(max(vals)),
            })
    return out


def _short(path: Path) -> str:
    """Repo-relative when it is under the repo, absolute otherwise (e.g. a --out-dir in /tmp)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {_short(path)} ({len(rows)} rows)", flush=True)


def on_concept_means() -> dict[tuple[str, int], float]:
    """``{(arm, vintage): high-stakes mean AUROC}`` from the sweep's own summary CSV.

    The read-out needs it as the reference line: an off-concept number is only
    interpretable next to what the same fit scored on the concept it was trained for.
    """
    path = VINTAGE_DIR / "vintage_summary.csv"
    out: dict[tuple[str, int], float] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == "mean":
                out[(row["arm"], int(row["vintage"]))] = float(row["mean"])
    return out


def readout(summary: list[dict], concepts: list[str], scale: str = "pipeline") -> list[str]:
    """The interpretation lines, computed rather than written by hand.

    Two questions per (concept, arm): does the off-concept mean move as red-team data is
    added, and is any single split carrying a signal? A move is called out only when it
    clears 2x the pooled seed sd of the two vintages compared — the same bar
    ``vintage/SUMMARY.md`` uses, and for the same reason: these are independent fits, so
    a gap inside the seed noise is not evidence.
    """
    on = on_concept_means()
    lines = ["## Read-out", ""]
    for concept in concepts:
        for arm in sorted({r["arm"] for r in summary if r["concept"] == concept}):
            def cell(k: int, split: str) -> dict | None:
                hit = [r for r in summary if r["concept"] == concept and r["arm"] == arm
                       and r["vintage"] == k and r["dataset"] == split and r["scale"] == scale]
                return hit[0] if hit else None

            vints = sorted({r["vintage"] for r in summary
                            if r["concept"] == concept and r["arm"] == arm})
            means = {k: cell(k, "mean") for k in vints}
            curve = " → ".join(f"v{k} {means[k]['mean']:.4f}" for k in vints if means[k])
            lines.append(f"- **{concept} / {arm}**: {curve}.")

            if 0 in means and means[0]:
                for k in [v for v in vints if v != 0]:
                    if not means[k]:
                        continue
                    gap = means[k]["mean"] - means[0]["mean"]
                    pooled = (means[0]["sd"] ** 2 + means[k]["sd"] ** 2) ** 0.5 or 1e-12
                    if abs(gap) >= 2 * pooled:
                        lines.append(
                            f"  - v{k} moves {gap:+.4f} against v0 ({abs(gap) / pooled:.1f}σ) — "
                            "the red-team data changed what this probe does off-concept."
                        )

            # The single most informative split, at whichever vintage it peaks.
            best = max(
                (r for r in summary if r["concept"] == concept and r["arm"] == arm
                 and r["scale"] == scale and r["dataset"] != "mean"),
                key=lambda r: abs(r["mean"] - 0.5),
            )
            lines.append(
                f"  - strongest single split: `{best['dataset']}` at v{best['vintage']}, "
                f"{best['mean']:.4f} ± {best['sd']:.4f} "
                f"({'above' if best['mean'] > 0.5 else 'below'} chance by "
                f"{abs(best['mean'] - 0.5):.4f})."
            )
            if on:
                on_curve = " → ".join(
                    f"v{k} {on[(arm, k)]:.4f}" for k in vints if (arm, k) in on
                )
                if on_curve:
                    lines.append(f"  - same fits ON high-stakes, for reference: {on_curve}.")
    lines.append("")
    return lines


def write_markdown(path: Path, summary: list[dict], concepts: list[str], scale: str = "pipeline") -> None:
    """One mean +/- sd table per (concept, arm), in ``vintage/SUMMARY.md``'s shape."""
    lines = [
        "# High-stakes vintage probes scored OFF their own concept",
        "",
        "Every row is one of the 80 committed `vintage/fits/*.pt` checkpoints — the probes "
        "behind `vintage/SUMMARY.md`, refitted nowhere — scored on an eval set it was never "
        "trained for. AUROC is against **that split's own positive class**, so 0.5 means the "
        "high-stakes score says nothing about the other concept and a value below 0.5 is a "
        "real anti-correlation, reported unflipped.",
        "",
        f"AUROC scale: `{scale}` (bf16 sigmoid then sklearn, as the pipeline reports it). "
        "The rank-faithful figures are in the CSVs alongside.",
        "",
    ] + readout(summary, concepts, scale)
    for concept in concepts:
        target = TARGETS[concept]
        splits = target.splits()
        lines += [f"## {concept} — {target.pos_label} vs {target.neg_label}", ""]
        arms = sorted({r["arm"] for r in summary if r["concept"] == concept})
        for arm in arms:
            lines += [
                f"### {arm}",
                "",
                "| vintage | rt rows | seeds | " + " | ".join(splits) + " | mean |",
                "|---" * (len(splits) + 4) + "|",
            ]
            vints = sorted({r["vintage"] for r in summary if r["concept"] == concept and r["arm"] == arm})
            for k in vints:
                cells = []
                for split in splits + ["mean"]:
                    hit = [r for r in summary
                           if r["concept"] == concept and r["arm"] == arm and r["vintage"] == k
                           and r["dataset"] == split and r["scale"] == scale]
                    cells.append(f"{hit[0]['mean']:.4f} ± {hit[0]['sd']:.4f}" if hit else "—")
                head = [r for r in summary
                        if r["concept"] == concept and r["arm"] == arm and r["vintage"] == k
                        and r["scale"] == scale]
                lines.append(
                    f"| v{k} | {head[0]['n_redteam_rows']} | {head[0]['n_seeds']} | "
                    + " | ".join(cells) + " |"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {_short(path)}", flush=True)


def self_check(probes: dict) -> None:
    """Re-score the high-stakes splits here and diff against ``vintage_auroc.csv``.

    The sweep and this script build the eval dataset, attach the blob and call the
    classifier through separately-written code paths, so agreement to the last bit is
    what licenses reading the off-concept tables as continuous with the on-concept one.
    Any disagreement means the two are not measuring the same probes and the comparison
    is void — so this exits non-zero rather than warning.
    """
    ref_path = VINTAGE_DIR / "vintage_auroc.csv"
    if not ref_path.exists():
        raise SystemExit(f"--self-check needs {ref_path}, which is missing")
    ref = {}
    with ref_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ref[(row["arm"], int(row["vintage"]), int(row["seed"]), row["dataset"])] = (
                float(row["auroc_pipeline"]), float(row["auroc_rank"])
            )

    rows = score_target("hs", TARGETS["hs"], probes)
    worst = 0.0
    compared = 0
    for r in rows:
        key = (r["arm"], r["vintage"], r["seed"], r["dataset"])
        if key not in ref:
            continue
        compared += 1
        worst = max(worst, abs(r["auroc_pipeline"] - ref[key][0]),
                    abs(r["auroc_rank"] - ref[key][1]))
    print(f"\n  self-check: {compared} fit x split cell(s) compared against "
          f"{ref_path.name}; max |diff| = {worst:.2e}", flush=True)
    if compared == 0:
        raise SystemExit("  self-check compared nothing — the keys do not line up")
    if worst > 1e-9:
        raise SystemExit("  self-check FAILED — this script does not reproduce the sweep")
    print("  self-check OK: these are the same probes the sweep scored\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concepts", nargs="+", default=["hu_harm", "instructions"],
                    choices=sorted(TARGETS), help="Eval sets to score against")
    ap.add_argument("--arms", nargs="+", default=sorted(A.ARMS), choices=sorted(A.ARMS))
    ap.add_argument("--vintages", nargs="+", type=int, default=[0, 1, 2, 3],
                    help="v0 is the no-red-team baseline every other vintage is read against")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--self-check", action="store_true",
                    help="Reproduce vintage_auroc.csv through this script's scoring path first")
    ap.add_argument("--from-csv", action="store_true",
                    help="Rebuild the summary and SUMMARY.md from the per-fit CSVs already in "
                         "--out-dir, scoring nothing. For editing the write-up without paying "
                         "for a rescore that would produce identical numbers.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.from_csv:
        rows = []
        for concept in args.concepts:
            path = args.out_dir / f"{concept}_auroc.csv"
            if not path.exists():
                raise SystemExit(f"--from-csv: {path} does not exist — score it first")
            with path.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    rows.append({**r, "vintage": int(r["vintage"]), "seed": int(r["seed"]),
                                 "n_rows": int(r["n_rows"]),
                                 "auroc_pipeline": float(r["auroc_pipeline"]),
                                 "auroc_rank": float(r["auroc_rank"])})
            print(f"  read {_short(path)}", flush=True)
        summary = summarize(rows, redteam_rows_by_fit())
        write_csv(args.out_dir / "cross_concept_summary.csv", summary)
        write_markdown(args.out_dir / "SUMMARY.md", summary, args.concepts)
        return

    print(f"loading {len(args.arms)} arm(s) x {len(args.vintages)} vintage(s) x "
          f"{len(args.seeds)} seed(s) of committed fits", flush=True)
    probes = load_fits(args.arms, args.vintages, args.seeds)
    print(f"  {len(probes)} probe(s) loaded", flush=True)
    if not probes:
        raise SystemExit("no checkpoints found — run attribution_vintage.py first")

    if args.self_check:
        self_check(probes)

    rt_rows = redteam_rows_by_fit()
    all_rows: list[dict] = []
    for concept in args.concepts:
        print(f"\n=== {concept} ===", flush=True)
        rows = add_means(score_target(concept, TARGETS[concept], probes), TARGETS[concept])
        write_csv(args.out_dir / f"{concept}_auroc.csv", rows)
        all_rows += rows

    summary = summarize(all_rows, rt_rows)
    write_csv(args.out_dir / "cross_concept_summary.csv", summary)
    write_markdown(args.out_dir / "SUMMARY.md", summary, args.concepts)


if __name__ == "__main__":
    main()
