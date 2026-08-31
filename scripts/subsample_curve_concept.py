#!/usr/bin/env python
"""Training-set size curve: refit `base ∪ <n rows>` for several n, several draws each.

A generated 600-row set beats the 50-row base probe by a wide margin. This asks how much
of that is the *size* of the addition and how much is the particular 600 rows: for each n
in `--sizes`, draw `--draws` independent random subsets, fit `base ∪ subset`, and score
dev + eval exactly as `fit_base_plus_concept.py` does. The spread across draws at one n is
the run-to-run noise; the movement between n's is the curve.

**Draws are class-balanced** — n/2 per class, not a uniform sample of the 600. The source
sets are exactly 300/300 and every dev and eval split is exactly balanced, so an
unbalanced draw would add a second, uncontrolled variable (the class ratio) on top of the
size being measured. `--unbalanced` takes the uniform sample instead.

Draws are seeded on `(file stem, n, draw index)`, so a given row of the output is
reproducible on its own and adding sizes or draws never moves the existing ones.

**No activations are extracted.** Run `fit_base_plus_concept.py` on a set once before
pointing this at it and every row is already in the per-sample cache, so each iteration
here is a probe-head fit on cached activations plus a cached eval; otherwise the first
draws each pay the extraction.

Output is one CSV row per fit, appended as it lands (so a kill loses at most one fit),
with `dev_mean` / `eval_mean` plus every per-split AUROC. Per-split columns are
`dev_<stem>` / `eval_<stem>`, with a leading `dev_`/`eval_` already on the stem not
doubled — the hu_ha splits carry the family in the filename, the highstakes splits use
the *same* stems for dev and eval and would otherwise collide.

Re-running skips `(samples, n, draw)` triples already in the CSV; `--no-resume`
recomputes them.

Example:
    ${REPO_ROOT}/.venv_claude/bin/python scripts/subsample_curve_concept.py \\
        --concept hu_harm data/hu_harm_gptoss_600.jsonl data/hu_harm_deepseekv4pro_600.jsonl \\
        --sizes 100 200 300 400 500 --draws 8 --out scripts/hu_harm_size_curve.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_base_plus_concept import (  # noqa: E402
    COMBINE,
    CONCEPTS,
    CONVERT,
    SEED,
    Concept,
    eval_source,
    load_rows,
)

# Candidate probes are throwaway — the CSV is the artefact. cache_*/ is gitignored.
SCRATCH_SUBDIR = "subsample_probes"

BASE_FIELDS = [
    "samples", "base", "n", "draw", "n_pos", "n_neg", "n_training_rows",
    "dev_mean", "eval_mean",
]


def split_column(family: str, stem: str) -> str:
    """`dev_`/`eval_` prefixed column name, without doubling a prefix already on the stem.

    hu_ha names its splits `dev_ai_dilemmas` / `eval_ai_dilemmas` — the family is already
    in the filename. highstakes uses the SAME stems (`anthropic_hh_balanced`, ...) for
    both dev and eval, so there the prefix is what keeps the two columns apart.
    """
    return stem if stem.startswith(f"{family}_") else f"{family}_{stem}"


def fields_for(concept: Concept) -> list[str]:
    dev = [split_column("dev", p.stem) for p in sorted(concept.dev_data.glob("*.jsonl"))]
    ev = [split_column("eval", p.stem) for p in sorted(concept.eval_dir.glob("*.jsonl"))]
    return BASE_FIELDS + dev + ev + ["seconds"]


def draw_subset(rows: list[dict], n: int, stem: str, draw: int, balanced: bool,
                concept: Concept) -> list[dict]:
    """One reproducible subset of ``n`` rows, seeded on (stem, n, draw)."""
    rng = random.Random(f"{stem}:{n}:{draw}")
    if not balanced:
        return rng.sample(rows, n)
    pos = [r for r in rows if r["labels"] == concept.pos_label]
    neg = [r for r in rows if r["labels"] == concept.neg_label]
    half, rest = n // 2, n - n // 2
    if len(pos) < half or len(neg) < rest:
        raise SystemExit(
            f"cannot draw {half}+{rest} from {len(pos)} {concept.pos_label} / "
            f"{len(neg)} {concept.neg_label}"
        )
    # Sample each class under its own stream so the positive half of a draw does not
    # shift when the negative half's size changes.
    out = rng.sample(pos, half) + rng.sample(neg, rest)
    rng.shuffle(out)
    return out


def done_keys(csv_path: Path, default_base: str) -> set[tuple[str, str, int, int]]:
    """Finished (base, samples, n, draw) keys.

    The base file is part of the key because the same generated set is fit on two
    different bases: the concept's llama70b-written 50 rows (every arm up to
    736c5691) and, for a single-source arm, that generator's own 50. Rows written
    before the column existed carry the concept default.
    """
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return {
            (r.get("base") or default_base, r["samples"], int(r["n"]), int(r["draw"]))
            for r in csv.DictReader(fh)
            if r.get("samples")
        }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--concept", required=True, choices=sorted(CONCEPTS))
    ap.add_argument("samples", type=Path, nargs="+", help="one or more {inputs, labels} JSONLs")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: scripts/<concept>_size_curve.csv")
    ap.add_argument("--base-data", type=Path, default=None,
                    help="override the concept's base training JSONL (see fit_base_plus_concept.py)")
    ap.add_argument("--unbalanced", action="store_true",
                    help="uniform sample of the set instead of n/2 per class")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute rows already present in the output CSV")
    args = ap.parse_args()

    concept = CONCEPTS[args.concept]
    base_data = args.base_data or concept.base_data
    out_csv = args.out or REPO / f"scripts/{concept.name}_size_curve.csv"

    from agentic_redteam.cli import _free_gpu
    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import retrain_probe

    scratch = concept.cache_dir / SCRATCH_SUBDIR
    scratch.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    seen = set() if args.no_resume else done_keys(out_csv, concept.base_data.name)

    fields = fields_for(concept)
    fresh = not out_csv.exists() or out_csv.stat().st_size == 0
    if not fresh:
        # Append under the header the file already has. The CSVs written before the
        # `base` column existed do not carry it, and writing the wider row into them
        # would shift every per-split value one column left rather than failing.
        with out_csv.open(newline="", encoding="utf-8") as _fh:
            existing = next(csv.reader(_fh), None)
        if existing and existing != fields:
            missing = [f for f in existing if f not in fields]
            if missing:
                raise SystemExit(f"{out_csv} has columns this run cannot fill: {missing}")
            if base_data != concept.base_data:
                raise SystemExit(
                    f"{out_csv} predates the `base` column, so its rows are all on "
                    f"{concept.base_data.name}; write this --base-data run to its own "
                    f"--out file rather than appending."
                )
            fields = existing
    fh = out_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if fresh:
        writer.writeheader()
        fh.flush()

    jobs = [
        (path, n, d)
        for path in args.samples
        for n in args.sizes
        for d in range(args.draws)
        if (base_data.name, path.name, n, d) not in seen
    ]
    print(f"{len(jobs)} fits to run ({len(seen)} already in {out_csv})", flush=True)

    cache: dict[Path, list[dict]] = {}
    for i, (path, n, d) in enumerate(jobs, 1):
        rows = cache.setdefault(path, load_rows(path, concept))
        subset = draw_subset(rows, n, path.stem, d, not args.unbalanced, concept)
        npos = sum(1 for r in subset if r["labels"] == concept.pos_label)
        t0 = time.time()
        out_pkl = scratch / f"{path.stem}_{base_data.stem}_n{n}_d{d}.pkl"
        res = retrain_probe(
            samples=subset, base_probe_path=concept.base_probe,
            base_training_data_path=base_data, new_probe_path=out_pkl,
            dev_data_path=concept.dev_data, seed=SEED, base_data_fraction=1.0,
            base_activation_cache_dir=concept.base_cache,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=False,
        )
        df = evaluate_probe(
            out_pkl, concept.eval_dir, concept.eval_cache, max_samples=None, seed=SEED,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            kaggle_source=eval_source(),
        )
        ev = {r["dataset"]: float(r["auroc"]) for _, r in df.iterrows()}
        row = {
            "samples": path.name, "base": base_data.name, "n": n, "draw": d,
            "n_pos": npos, "n_neg": len(subset) - npos,
            "n_training_rows": res.n_training_samples_total,
            "dev_mean": round(res.dev_auroc["mean"], 5),
            "eval_mean": round(ev["mean"], 5),
            "seconds": round(time.time() - t0, 1),
        }
        for split, v in res.dev_auroc.items():
            if split != "mean":
                row[split_column("dev", split)] = round(v, 5)
        for split, v in ev.items():
            if split != "mean":
                row[split_column("eval", split)] = round(v, 5)
        writer.writerow(row)
        fh.flush()
        print(
            f"[{i}/{len(jobs)}] {path.stem} n={n} draw={d}: "
            f"dev {row['dev_mean']:.5f}  eval {row['eval_mean']:.5f}  ({row['seconds']:.0f}s)",
            flush=True,
        )
        out_pkl.unlink(missing_ok=True)
        _free_gpu()

    fh.close()
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
