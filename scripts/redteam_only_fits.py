"""Fit probes on red-team data **alone** — no base training data — and score the seven
``eval_instructions`` splits.

The vintage sweep (``scripts/attribution_vintage.py``) always trains on
``data/instructions_llama70b_50.jsonl`` *plus* a vintage of red-team pairs, which is what
the pipeline does. This asks the complementary question: **what does the red-team data
carry on its own?** Two conditions, ten seeds each, per arm:

- ``v2``     — every iteration-3 pair whose source success already existed at iteration 2,
               and nothing else. Directly comparable to the sweep's ``v2`` row, which is
               the same rows plus the 50 base samples.
- ``v3only`` — only the pairs that first appear at iteration 3 (``keep[3] - keep[2]``),
               i.e. the rows held out of every ``v2`` fit. Same set
               ``vintage_holdout_success.py`` puts *in front of* the v2 probes; here it is
               what the probe is trained *from*. 184 rows (gptoss120b) / 296 (nemotron).

Everything else is the sweep's: membership is derived per source success so pairs stay
atomic and the set stays 50/50 (``attribution_vintage.vintages``), the over-1024-token
filter is on by default (``--drop-long pair``), the train/val side of each row is the run's
own content-deterministic split, activations come off disk, and the fit goes through the
real ``ProbeFactory.build``.

The one thing to keep in mind reading the result: with the base data gone the *validation*
set is red-team rows only too, so early stopping is judged on the same distribution the
probe is fitted on. A high eval AUROC here means the red-team rows alone separate the
concept; it does not mean this is a better way to train.

Usage:
    AGENTIC_FAST_ACTS=1 .venv_claude/bin/python scripts/redteam_only_fits.py \
        --arm gptoss120b nemotron --seeds 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# tuberlens' trainer wraps every epoch in tqdm on stderr; a sweep would bury the log.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A  # noqa: E402
import attribution_refit as R  # noqa: E402
import attribution_vintage as V  # noqa: E402

CONDITIONS = ("v2", "v3only", "v3")

# Every fitted probe is pickled here, named ``{arm}__{condition}__seed{n}.pkl``.
#
# The vintage sweep this is modelled on kept each fit's AUROC and dropped the probe, and
# the first version of this script copied that. It is the wrong trade by three orders of
# magnitude: a probe is ~13 KB (a 5376-wide linear head plus metadata) while re-deriving
# one costs 20-130 s of GPU, so the whole 40-fit sweep is ~0.5 MB against ~1.5 h. Any
# later question that needs the probe *object* rather than its recorded AUROC — scoring a
# dataset that did not exist when the sweep ran, inspecting the weight vector, comparing
# two conditions per row — otherwise has to refit the entire sweep first, which is exactly
# what ``crossconcept_eval.py`` had to do.
#
# Shared with ``crossconcept_eval.py``, which fits the with-base conditions into the same
# directory under the same naming, so either script's probes satisfy the other.
PROBE_DIR = A.REPO / "results_instructions_gemma27b_vintage/sweep_probes"

# The filename tag is NOT this script's ``condition`` verbatim. One store holds probes from
# both sweeps, and there "v2" is ambiguous — the vintage sweep's v2 trains on the same
# red-team rows *plus* the base data, and is a different probe. The tag says which.
PROBE_TAG = {"v2": "v2_alone", "v3only": "v3only_alone", "v3": "v3_alone"}


def probe_path(probe_dir: Path, arm: str, condition: str, seed: int) -> Path:
    return probe_dir / f"{arm}__{PROBE_TAG.get(condition, condition)}__seed{seed}.pkl"


def save_probe(probe, probe_dir: Path, arm: str, condition: str, seed: int) -> Path:
    """Pickle one fitted probe. Refits are deterministic, so this is a cache, not state."""
    import pickle

    path = probe_path(probe_dir, arm, condition, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(probe, fh)
    return path


def condition_rows(keep: dict[int, list[int]], condition: str, iteration: int) -> list[int]:
    """Red-team row indices (into the iteration-``iteration`` dump) for one condition."""
    if condition == "v2":
        return sorted(keep[2])
    if condition == "v3":
        return sorted(keep[iteration])
    if condition == "v3only":
        return sorted(set(keep[iteration]) - set(keep[2]))
    raise ValueError(f"unknown condition {condition!r}")


def refit_redteam_only(asm: R.Assembled, rows: list[int], seed: int, quiet: bool = True):
    """``attribution_refit.refit`` with the base halves left out entirely.

    Not ``refit(drop_rows=...)`` with an empty base: that function concatenates
    ``asm.base_train`` in unconditionally, and the point here is to remove it. The rest is
    the same call — same probe spec, same ``seed_everything``, same ``ProbeFactory.build``
    — so a difference against the sweep is a difference in the training set and nothing
    else.
    """
    import contextlib
    import io

    from tuberlens.probes.probe_factory import ProbeFactory

    from agentic_redteam.evaluation import seed_everything

    keep = set(rows)
    train_idx = [i for i in rows if not asm.rt_is_val[i]]
    val_idx = [i for i in rows if asm.rt_is_val[i]]
    assert keep == set(train_idx) | set(val_idx)
    # Advanced indexing copies, so these are fresh tensors and asm.redteam is untouched —
    # which matters because it is reused across every fit of the run.
    train = asm.redteam[train_idx]
    val = asm.redteam[val_idx]

    # Transfer mechanics only — see scripts/fast_activations.py.
    import fast_activations as F

    if F.enabled():
        F.patch_getitems()
        F.maybe_resident(train, f"train {len(train)}")
        F.maybe_resident(val, f"val {len(val)}")

    seed_everything(seed)
    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        probe = ProbeFactory.build(
            probe_spec=asm.probe_spec,
            train_dataset=train,
            model_name=asm.probe.model_name,
            layer=asm.probe.layer,
            validation_dataset=val,
            use_store=False,
            pos_class_label=asm.probe.pos_class_label,
            neg_class_label=asm.probe.neg_class_label,
            probe_description=asm.probe.description,
        )
    return probe, len(train), len(val)


def _class_balance(asm: R.Assembled, rows: list[int]) -> dict:
    """Per-side label counts — a guard, since the val side is small on ``v3only``.

    The dump's ``label`` is the canonical class name (``"positive"`` / ``"negative"``, what
    ``_dump_labelled_dataset`` writes), not the probe's human-readable label, so count the
    values as they are rather than comparing against ``pos_class_label``.
    """
    labels = asm.redteam.other_fields["labels"]
    tr = [i for i in rows if not asm.rt_is_val[i]]
    va = [i for i in rows if asm.rt_is_val[i]]
    return {
        "n_train": len(tr),
        "n_val": len(va),
        "train": dict(Counter(labels[i] for i in tr)),
        "val": dict(Counter(labels[i] for i in va)),
    }


def _done_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    out = set()
    for line in path.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a row truncated by a hard kill; the fit simply re-runs
        out.add((r["arm"], r["condition"], int(r["seed"])))
    return out


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_arm(arm: str, conditions: list[str], seeds: list[int], *, iteration: int,
            drop_long: str, eval_dir: Path, progress: Path, probe_dir: Path,
            resume: bool = True) -> None:
    keep, report = V.vintages(arm, iteration, drop_long)
    rows_for = {c: condition_rows(keep, c, iteration) for c in conditions}

    print(f"\n=== {arm} ===", flush=True)
    for c in conditions:
        print(f"  {c:7s}: {len(rows_for[c]):4d} red-team rows "
              f"({len(rows_for[c]) // 2} pairs), no base data", flush=True)

    # A recorded AUROC row without its probe on disk is a fit that has to be redone if the
    # probe is ever wanted, so resume treats a missing pickle as unfinished. The scored row
    # is deduped by JsonlStore-style key on append, and the refit is deterministic, so
    # re-running one costs time and changes nothing.
    done = {k for k in _done_keys(progress)
            if probe_path(probe_dir, *k[:2], k[2]).exists()} if resume else set()
    todo = [(s, c) for s in seeds for c in conditions if (arm, c, s) not in done]
    if not todo:
        print("  every (condition, seed) already recorded — nothing to do", flush=True)
        return
    print(f"  {len(todo)} fit(s) to run, "
          f"{len([k for k in done if k[0] == arm])} already recorded", flush=True)

    asm = V.assemble_train_only(arm, iteration)
    for c in conditions:
        print(f"  {c:7s} balance: {_class_balance(asm, rows_for[c])}", flush=True)

    # Seeds outer, conditions inner: after seed s finishes, both conditions are measured at
    # s seeds, so a box that dies mid-run leaves a complete (if shorter) comparison.
    for seed, c in todo:
        t0 = time.time()
        probe, n_tr, n_val = refit_redteam_only(asm, rows_for[c], seed)
        fit_s = time.time() - t0
        t1 = time.time()
        res = V.score_streaming(asm, probe, eval_dir)
        print(
            f"  seed {seed} {c:7s}: fit {fit_s:5.1f}s  score {time.time() - t1:4.1f}s  "
            f"train={n_tr} val={n_val}  best_epoch={probe._classifier.best_epoch}",
            flush=True,
        )
        print(
            "     " + "  ".join(
                f"{sp:24s}={res[sp]['pipeline']:.4f}" for sp in A.EVAL_SPLITS
            ) + f"  MEAN={res['mean']['pipeline']:.4f}",
            flush=True,
        )
        saved = save_probe(probe, probe_dir, arm, c, seed)
        _append(progress, {
            "arm": arm,
            "condition": c,
            "seed": seed,
            "probe_path": str(saved.relative_to(A.REPO)),
            "iteration": iteration,
            "drop_long": drop_long,
            "base_data": False,
            "n_redteam_rows": len(rows_for[c]),
            "n_train": n_tr,
            "n_val": n_val,
            # None when validation AUROC never beat its initial value — possible on the
            # small v3only set, and not an error.
            "best_epoch": (None if probe._classifier.best_epoch is None
                           else int(probe._classifier.best_epoch)),
            "fit_seconds": fit_s,
            "auroc": res,
        })
        del probe
        gc.collect()
        torch.cuda.empty_cache()

    del asm
    gc.collect()


def summarize(rows: list[dict], out_dir: Path) -> None:
    """Mean +/- sd over seeds, plus a per-fit CSV.

    Unpaired, like the vintage sweep: these are independent ``ProbeFactory`` fits with
    independent initialisations, so the sd column is the thing that decides whether a
    between-condition gap is readable at all.
    """
    # Dedup by (arm, condition, seed), newest wins: a fit whose probe pickle went missing
    # is re-run, which appends a second (identical) row for that seed, and counting both
    # would report n=11 for a ten-seed sweep and shrink the sd.
    latest = {(r["arm"], r["condition"], r["seed"]): r for r in rows}
    rows = list(latest.values())

    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["condition"], r["n_redteam_rows"])].append(r)

    order = {c: i for i, c in enumerate(CONDITIONS)}
    print("\n\n=== eval AUROC, red-team rows only (no base data) — "
          "mean +/- sd over seeds (pipeline) ===")
    header = (f"{'arm':12s} {'cond':>7s} {'rows':>5s} {'n':>3s} "
              + " ".join(f"{sp:>19s}" for sp in A.EVAL_SPLITS) + f"{'MEAN':>19s}")
    print(header)
    print("-" * len(header))
    for key in sorted(by, key=lambda k: (k[0], order.get(k[1], 99))):
        arm, cond, n_rows = key
        rs = by[key]
        cells = []
        for sp in list(A.EVAL_SPLITS) + ["mean"]:
            v = np.array([r["auroc"][sp]["pipeline"] for r in rs])
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            cells.append(f"{v.mean():.4f}+/-{sd:.4f}")
        print(f"{arm:12s} {cond:>7s} {n_rows:>5d} {len(rs):>3d} "
              + " ".join(f"{c:>19s}" for c in cells))

    out_dir.mkdir(parents=True, exist_ok=True)
    per_fit = out_dir / "redteam_only_auroc.csv"
    with per_fit.open("w", encoding="utf-8") as fh:
        fh.write("arm,condition,seed,n_redteam_rows,n_train,n_val,dataset,"
                 "auroc_pipeline,auroc_rank\n")
        for r in sorted(rows, key=lambda x: (x["arm"], order.get(x["condition"], 99),
                                             x["seed"])):
            for sp in list(A.EVAL_SPLITS) + ["mean"]:
                fh.write(f"{r['arm']},{r['condition']},{r['seed']},{r['n_redteam_rows']},"
                         f"{r['n_train']},{r['n_val']},{sp},"
                         f"{r['auroc'][sp]['pipeline']},{r['auroc'][sp]['rank']}\n")

    stats = out_dir / "redteam_only_summary.csv"
    with stats.open("w", encoding="utf-8") as fh:
        fh.write("arm,condition,n_redteam_rows,n_seeds,dataset,mean,sd,min,max\n")
        for key in sorted(by, key=lambda k: (k[0], order.get(k[1], 99))):
            arm, cond, n_rows = key
            rs = by[key]
            for sp in list(A.EVAL_SPLITS) + ["mean"]:
                v = np.array([r["auroc"][sp]["pipeline"] for r in rs])
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                fh.write(f"{arm},{cond},{n_rows},{len(rs)},{sp},"
                         f"{v.mean()},{sd},{v.min()},{v.max()}\n")
    print(f"\nwrote {per_fit}\nwrote {stats}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                    default=["v2", "v3only"])
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--seed", type=int, default=A.SEED, help="first seed")
    ap.add_argument("--seeds", type=int, default=10, help="how many consecutive seeds")
    ap.add_argument("--drop-long", choices=("pair", "row", "none"), default="pair")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--membership-only", action="store_true",
                    help="print the row counts and exit, fitting nothing")
    ap.add_argument("--summarize-only", action="store_true",
                    help="re-aggregate the progress sidecar and exit. Use after running "
                         "the arms as separate processes.")
    ap.add_argument("--eval-dir", type=Path, default=A.EVAL_ACTIVATIONS_DIR)
    ap.add_argument("--out-dir", type=Path,
                    default=A.REPO / "results_instructions_gemma27b_vintage")
    ap.add_argument("--probe-dir", type=Path, default=PROBE_DIR,
                    help="where fitted probes are pickled (~13 KB each)")
    args = ap.parse_args()

    seeds = [args.seed + i for i in range(args.seeds)]
    progress = args.out_dir / "redteam_only_progress.jsonl"

    if args.membership_only:
        for arm in args.arm:
            keep, _ = V.vintages(arm, args.iteration, args.drop_long)
            counts = {c: len(condition_rows(keep, c, args.iteration)) for c in CONDITIONS}
            print(arm, json.dumps(counts))
        return

    if not args.summarize_only:
        for arm in args.arm:
            run_arm(arm, args.conditions, seeds, iteration=args.iteration,
                    drop_long=args.drop_long, eval_dir=args.eval_dir, progress=progress,
                    probe_dir=args.probe_dir, resume=args.resume)

    # Report from the SIDECAR: on a resumed run the earlier fits are on disk and nowhere
    # else, and a summary over this process's fits alone would understate the sweep.
    rows = [r for r in (json.loads(l) for l in progress.open(encoding="utf-8"))
            if r["arm"] in args.arm and r["condition"] in args.conditions]
    summarize(rows, args.out_dir)


if __name__ == "__main__":
    main()
