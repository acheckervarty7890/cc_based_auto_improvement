#!/usr/bin/env python
"""Training-set size curve: refit `base ∪ <n rows>` for several n, several draws each.

Both 600-row human-harm sets (`data/hu_harm_gptoss_600.jsonl`,
`data/hu_harm_deepseekv4pro_600.jsonl`) beat the 50-row base probe by a wide margin.
This asks how much of that is the *size* of the addition and how much is the particular
600 rows: for each n in `--sizes`, draw `--draws` independent random subsets, fit
`base ∪ subset`, and score dev + eval exactly as `fit_base_plus_hu_harm.py` does. The
spread across draws at one n is the run-to-run noise; the movement between n's is the
curve.

**Draws are class-balanced** — n/2 per class, not a uniform sample of the 600. The
source sets are exactly 300/300 and every dev and eval split is exactly balanced, so an
unbalanced draw would add a second, uncontrolled variable (the class ratio) on top of
the size being measured. `--unbalanced` takes the uniform sample instead.

Draws are seeded on `(file stem, n, draw index)`, so a given row of the output is
reproducible on its own and adding sizes or draws never moves the existing ones.

**No activations are extracted.** Every row of both 600-row sets is already in the
per-sample cache from the two full fits, so each iteration here is a probe-head fit on
cached activations plus a cached eval. Run `fit_base_plus_hu_harm.py` on a set once
before pointing this at it, or the first draws will each pay the extraction.

Output is one CSV row per fit, appended as it lands (so a kill loses at most one fit),
with `dev_mean` / `eval_mean` plus every per-split AUROC. Re-running skips
`(samples, n, draw)` triples already in the CSV; `--no-resume` recomputes them.

Example:
    ${REPO_ROOT}/.venv_claude/bin/python scripts/subsample_curve_hu_harm.py \\
        data/hu_harm_gptoss_600.jsonl data/hu_harm_deepseekv4pro_600.jsonl \\
        --sizes 100 200 300 400 500 --draws 8 --out scripts/hu_harm_size_curve.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_base_plus_hu_harm import (  # noqa: E402
    BASE_CACHE,
    BASE_DATA,
    BASE_PROBE,
    COMBINE,
    CONVERT,
    DEV_DATA,
    EVAL_CACHE,
    EVAL_DIR,
    NEG_LABEL,
    POS_LABEL,
    SEED,
    _eval_source,
)

# Candidate probes are throwaway — the CSV is the artefact. cache_*/ is gitignored.
PROBE_SCRATCH = REPO / "cache_gen_gemma27b_hu_harm/subsample_probes"

FIELDS = [
    "samples", "n", "draw", "n_pos", "n_neg", "n_training_rows",
    "dev_mean", "eval_mean",
    "dev_ai_dilemmas", "dev_ant_hh", "dev_balanced_refusal", "dev_daily_dilemmas",
    "eval_ai_dilemmas", "eval_ant_hh", "eval_balanced_refusal", "eval_daily_dilemmas",
    "seconds",
]


def load_rows(path: Path) -> list[dict]:
    """Read a `{inputs, labels}` JSONL into the shape ``retrain_probe`` wants.

    `inputs` is a JSON-encoded string on disk; the in-memory samples path needs it
    parsed (``_dicts_to_labelled_dataset`` calls ``m.get("role")`` on each message).
    """
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({
            "inputs": json.loads(r["inputs"]) if isinstance(r["inputs"], str) else r["inputs"],
            "labels": r["labels"],
        })
    return rows


def draw_subset(rows: list[dict], n: int, stem: str, draw: int, balanced: bool) -> list[dict]:
    """One reproducible subset of ``n`` rows, seeded on (stem, n, draw)."""
    rng = random.Random(f"{stem}:{n}:{draw}")
    if not balanced:
        return rng.sample(rows, n)
    pos = [r for r in rows if r["labels"] == POS_LABEL]
    neg = [r for r in rows if r["labels"] == NEG_LABEL]
    half = n // 2
    if len(pos) < half or len(neg) < half:
        raise SystemExit(
            f"cannot draw {half}+{half} from {len(pos)} {POS_LABEL} / {len(neg)} {NEG_LABEL}"
        )
    # Sample each class under its own stream so the positive half of a draw does not
    # shift when the negative half's size changes.
    out = rng.sample(pos, half) + rng.sample(neg, n - half if n % 2 == 0 else half + 1)
    rng.shuffle(out)
    return out[:n]


def done_keys(csv_path: Path) -> set[tuple[str, int, int]]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return {
            (r["samples"], int(r["n"]), int(r["draw"]))
            for r in csv.DictReader(fh)
            if r.get("samples")
        }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("samples", type=Path, nargs="+", help="one or more {inputs, labels} JSONLs")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--out", type=Path, default=REPO / "scripts/hu_harm_size_curve.csv")
    ap.add_argument("--unbalanced", action="store_true",
                    help="uniform sample of the set instead of n/2 per class")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute rows already present in the output CSV")
    args = ap.parse_args()

    from agentic_redteam.cli import _free_gpu
    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import retrain_probe

    PROBE_SCRATCH.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = set() if args.no_resume else done_keys(args.out)

    fresh = not args.out.exists() or args.out.stat().st_size == 0
    fh = args.out.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if fresh:
        writer.writeheader()
        fh.flush()

    jobs = [
        (path, n, d)
        for path in args.samples
        for n in args.sizes
        for d in range(args.draws)
        if (path.name, n, d) not in seen
    ]
    print(f"{len(jobs)} fits to run ({len(seen)} already in {args.out})", flush=True)

    cache: dict[Path, list[dict]] = {}
    for i, (path, n, d) in enumerate(jobs, 1):
        rows = cache.setdefault(path, load_rows(path))
        subset = draw_subset(rows, n, path.stem, d, not args.unbalanced)
        npos = sum(1 for r in subset if r["labels"] == POS_LABEL)
        t0 = time.time()
        out_pkl = PROBE_SCRATCH / f"{path.stem}_n{n}_d{d}.pkl"
        res = retrain_probe(
            samples=subset, base_probe_path=BASE_PROBE, base_training_data_path=BASE_DATA,
            new_probe_path=out_pkl, dev_data_path=DEV_DATA, seed=SEED,
            base_data_fraction=1.0, base_activation_cache_dir=BASE_CACHE,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=False,
        )
        df = evaluate_probe(
            out_pkl, EVAL_DIR, EVAL_CACHE, max_samples=None, seed=SEED,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            kaggle_source=_eval_source(),
        )
        ev = {r["dataset"]: float(r["auroc"]) for _, r in df.iterrows()}
        row = {
            "samples": path.name, "n": n, "draw": d,
            "n_pos": npos, "n_neg": len(subset) - npos,
            "n_training_rows": res.n_training_samples_total,
            "dev_mean": round(res.dev_auroc["mean"], 5),
            "eval_mean": round(ev["mean"], 5),
            "seconds": round(time.time() - t0, 1),
        }
        for split, v in res.dev_auroc.items():
            if split != "mean":
                row[split] = round(v, 5)
        for split, v in ev.items():
            if split != "mean":
                row[split] = round(v, 5)
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
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
