#!/usr/bin/env python
"""Size curve for MIXTURES of two generators' sets.

`subsample_curve_concept.py` answers "how much is one generator's set worth at n rows".
This answers "is a 50/50 blend of two generators worth more than either alone at the same
n" — the question the single-generator curves cannot, since every point there is one
source and the blends were never fit.

For each unordered pair of the concept's generators, each n in `--sizes`, and each draw,
it takes **n rows from EACH generator, n/2 per generator-and-class** — so `--sizes 100`
means 100 gptoss + 100 deepseek = 200 training rows, and n=600 uses both sets whole
(1200 rows). Sizes must be divisible by 2 (validated).

**No base training data is used at all.** The fit is the mix alone, so what is measured
is what the generated rows are worth by themselves, with no 50 real rows underneath. The
earlier finding that a 50-row base can outweigh 600 generated ones is exactly why this
run removes it. That makes these numbers NOT comparable with `<concept>_size_curve.csv`,
every point of which sits on a base.

At n = the full set size the draw is the identity — sampling 600 of 600 — so every draw
would be the same rows and the same fit. Draws beyond the first are skipped there rather
than burning identical fits.

Draws are seeded on `(pair, n, draw)` with each generator-and-class cell drawn from its
own stream, so a mix's gptoss half does not move when its partner changes, and adding
sizes or draws never moves existing rows.

**No activations are extracted** if `fit_base_plus_concept.py` has already been run on
each set (they are, for every set in this experiment), so each fit is cached-activation
probe-head work plus a cached eval.

Example:
    ${REPO_ROOT}/.venv_claude/bin/python scripts/mix_curve_concept.py \\
        --concept hu_harm --sizes 100 200 300 400 500 600 --draws 8 \\
        data/hu_harm_gptoss_600.jsonl data/hu_harm_deepseekv4pro_600.jsonl \\
        data/hu_harm_llama70b_600.jsonl
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
from subsample_curve_concept import fields_for, split_column  # noqa: E402

SCRATCH_SUBDIR = "mix_probes"

BASE_FIELDS = [
    "pair", "gen_a", "gen_b", "n", "draw",
    "n_pos", "n_neg", "n_training_rows", "dev_mean", "eval_mean",
]


def gen_tag(path: Path, concept: Concept) -> str:
    """`data/hu_harm_gptoss_600.jsonl` -> `gptoss`."""
    stem = path.stem
    for prefix in (f"{concept.name}_", "hu_harm_", "highstakes_", "instructions_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem.rsplit("_", 1)[0] if stem.endswith(("_600", "_50")) else stem


def is_whole_set(rows: list[dict], n: int, concept: Concept) -> bool:
    """True when drawing ``n`` from this set takes all of it, so a draw is the identity."""
    per_cell = n // 2
    return all(
        sum(1 for r in rows if r["labels"] == lab) <= per_cell
        for lab in (concept.pos_label, concept.neg_label)
    )


def draw_mix(rows_a: list[dict], rows_b: list[dict], n: int, tag_a: str, tag_b: str,
             draw: int, concept: Concept) -> list[dict]:
    """``n`` rows from EACH set, n/2 per generator-and-class (so 2n rows total).

    Each cell is drawn from its own RNG stream keyed on (tag, class, n, draw), so a
    generator's contribution is identical whichever partner it is paired with — which
    is what lets two mixes sharing a generator be compared without that half moving.
    """
    per_gen = n
    per_cell = per_gen // 2
    out: list[dict] = []
    for tag, rows in ((tag_a, rows_a), (tag_b, rows_b)):
        for label in (concept.pos_label, concept.neg_label):
            pool = [r for r in rows if r["labels"] == label]
            if len(pool) < per_cell:
                raise SystemExit(
                    f"{tag}: need {per_cell} {label} rows for n={n}, have {len(pool)}"
                )
            out += random.Random(f"{tag}:{label}:{n}:{draw}").sample(pool, per_cell)
    random.Random(f"{tag_a}+{tag_b}:{n}:{draw}").shuffle(out)
    return out


def done_keys(csv_path: Path) -> set[tuple[str, int, int]]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return {
            (r["pair"], int(r["n"]), int(r["draw"]))
            for r in csv.DictReader(fh) if r.get("pair")
        }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--concept", required=True, choices=sorted(CONCEPTS))
    ap.add_argument("samples", type=Path, nargs="+",
                    help="the concept's generator sets; every unordered pair is fit")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 300, 400, 500, 600])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: scripts/<concept>_mix_curve.csv")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    concept = CONCEPTS[args.concept]
    out_csv = args.out or REPO / f"scripts/{concept.name}_mix_curve.csv"
    bad = [n for n in args.sizes if n % 2]
    if bad:
        ap.error(f"sizes must be divisible by 2 (n/2 per generator-and-class cell): {bad}")
    if len(args.samples) < 2:
        ap.error("give at least two sample files to pair")

    from agentic_redteam.cli import _free_gpu
    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import retrain_probe

    scratch = concept.cache_dir / SCRATCH_SUBDIR
    scratch.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    seen = set() if args.no_resume else done_keys(out_csv)

    fields = BASE_FIELDS + [f for f in fields_for(concept) if f not in
                            ("samples", "base", "n", "draw", "n_pos", "n_neg",
                             "n_training_rows", "dev_mean", "eval_mean")]
    fresh = not out_csv.exists() or out_csv.stat().st_size == 0
    fh = out_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if fresh:
        writer.writeheader()
        fh.flush()

    loaded = {p: load_rows(p, concept) for p in args.samples}
    tags = {p: gen_tag(p, concept) for p in args.samples}

    jobs = []
    for pa, pb in itertools.combinations(args.samples, 2):
        pair = "+".join(sorted((tags[pa], tags[pb])))
        for n in args.sizes:
            whole = is_whole_set(loaded[pa], n, concept) and is_whole_set(loaded[pb], n, concept)
            for d in range(1 if whole else args.draws):
                if (pair, n, d) not in seen:
                    jobs.append((pa, pb, pair, n, d))
    print(f"{len(jobs)} mix fits to run ({len(seen)} already in {out_csv})", flush=True)

    for i, (pa, pb, pair, n, d) in enumerate(jobs, 1):
        subset = draw_mix(loaded[pa], loaded[pb], n, tags[pa], tags[pb], d, concept)
        npos = sum(1 for r in subset if r["labels"] == concept.pos_label)
        t0 = time.time()
        out_pkl = scratch / f"{pair}_n{n}_d{d}.pkl"
        res = retrain_probe(
            samples=subset, base_probe_path=concept.base_probe,
            base_training_data_path=None,   # mix alone — see the module docstring
            new_probe_path=out_pkl,
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
            "pair": pair, "gen_a": tags[pa], "gen_b": tags[pb],
            "n": n, "draw": d,
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
        print(f"[{i}/{len(jobs)}] {pair} n={n} draw={d}: dev {row['dev_mean']:.5f}  "
              f"eval {row['eval_mean']:.5f}  ({row['seconds']:.0f}s)", flush=True)
        out_pkl.unlink(missing_ok=True)
        _free_gpu()

    fh.close()
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
