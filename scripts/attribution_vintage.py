"""Retrain each arm's probe on the iteration-1, -2 and -3 red-team *vintages*, and score
the eval AUROC of each — all from activations already on disk.

Ported from ``experiment11_cloud``'s ``harmful_to_human`` sweep (whose results are in
``results_hu_harm_gemma27b_batch_ablation/vintage/SUMMARY.md``) onto this branch's
**instruction-following** experiment: arms ``gptoss120b`` / ``nemotron``, probe
``google/gemma-3-27b-it`` L32, the seven ``eval_instructions`` splits.

The question this answers: **how much of the final probe's eval AUROC was already bought
by the red-team data that existed at iteration 1?** The committed ``*_comparison.csv``
cannot answer it, because ``probe_iter1`` was trained by a different retrain than
``probe_iter3`` — different filter percentile draw, different contrastive pairs. This
script holds everything else fixed and varies only *which* red-team conversations are in
the training set, so the points differ in their data and nothing else.

Why the unit of selection is the pair, not the row
--------------------------------------------------
``preprocessing`` turns each red-team success into two training rows — the success and an
LLM-written opposite-class counterpart — and ``retrain`` never sees one without the other.
Selecting rows individually would unbalance the labels and orphan half-pairs, so
membership is decided per **source success** and both of its rows follow.

Why membership is re-derived rather than read off the iteration-N dumps
----------------------------------------------------------------------
``redteam_postprocessed_iter{N}.jsonl`` is the verbatim training set of iteration N, but
its rows are **not** a subset of iteration 3's: ``filter_dataset`` refits its bag-of-words
classifier on the growing success set each cycle so a different top-percentile is dropped.
So a vintage here is *"every iteration-3 pair whose originating success was already
present in iteration N"*, carrying **iteration 3's** content — which is the only content
whose activations exist.

Two consequences worth stating plainly, since they bound what the numbers mean:

- The vintages are **not strictly nested** — a pair can be in the iteration-1 vintage but
  not the iteration-2 one, when the iteration-2 filter dropped it and iteration 3 took it
  back. This is faithful to what the runs actually trained on rather than an idealised
  add-only curve. The per-arm report prints the count.
- A vintage's rows are iteration 3's, so a pair whose generated half was rewritten between
  iterations enters here in its *final* form. The alternative has no activation blob.

The over-1024-token filter (``--drop-long``)
--------------------------------------------
``get_activations`` pads *or truncates* every conversation to 1024 tokens
(``tuberlens/model.py``), so a longer one is scored — and trained on — from its opening
alone, tail silently dropped. These runs predate ``token_budget.py``, so neither the
attacker nor the contrastive generator was length-guarded, and any such row was trained on
amputated. ``--drop-long pair`` (the default) removes every over-long row **and its pair
partner**, keeping the pair atomic and the vintage 50/50 the way the rest of this script
does; ``row`` removes only the over-long row itself; ``none`` disables the filter. The
exclusion is applied to the vintage-membership derivation too, so an over-long row cannot
admit a source into a vintage it would not otherwise be in.

Usage:
    .venv_claude/bin/python scripts/attribution_vintage.py --arm gptoss120b nemotron --seeds 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# tuberlens' trainer wraps every epoch in tqdm on stderr; a sweep would bury the log.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
import attribution_refit as R


# --- the over-1024-token exclusion ------------------------------------------------


def long_row_exclusions(arm: str, iterations, drop_long: str,
                        final_iteration: int | None = None) -> tuple[dict, dict]:
    """``({iteration: excluded row indices}, report)`` under the chosen policy.

    Computed per iteration dump, because membership is derived from the earlier dumps and
    the training rows come from the final one — an over-long row must be excluded from
    both roles or the two halves would disagree about what the training set is.

    Two detectors feed the same exclusion set. The token count is the one the request
    names, and it applies to every iteration. The blob check (see
    ``attribution_lib.truncated_blob_rows``) catches the rows that were *stored* amputated,
    which is a superset in practice because ``get_activations`` truncates a batch rather
    than a row — but it needs the activation on disk, so it only runs on the final
    iteration, the one whose blobs this sweep actually trains against.
    """
    if drop_long == "none":
        return {k: set() for k in iterations}, {"policy": "none"}

    gen2src = A.generated_to_source(arm)
    probe_model = "google/gemma-3-27b-it"
    excluded, report = {}, {"policy": drop_long, "per_iteration": {}}
    for k in iterations:
        ds = A.load_redteam_dataset(arm, k)
        over, rep = A.over_long_rows(ds, probe_model)
        blob_rep: dict = {}
        if k == final_iteration:
            trunc, blob_rep = A.truncated_blob_rows(ds, probe_model)
            over = over | trunc
        keys = [A.canon(m) for m in ds.inputs]
        # A row's pair is identified by its source-success key: a generated row maps
        # through the contrastive cache, a source row is its own key.
        src_key = [gen2src.get(key, key) for key in keys]
        if drop_long == "pair":
            doomed = {src_key[i] for i in over}
            drop = {i for i, s in enumerate(src_key) if s in doomed}
        else:
            drop = set(over)
        excluded[k] = drop
        report["per_iteration"][k] = {
            **rep,
            **blob_rep,
            "n_over_generated": sum(1 for i in over if keys[i] in gen2src),
            "n_excluded": len(drop),
        }
    return excluded, report


# --- vintage membership -----------------------------------------------------------


def _source_keys(arm: str, iteration: int, gen2src: dict[str, str],
                 exclude: set[int]) -> list[str | None]:
    """The source-success key of every row of ``redteam_postprocessed_iter{N}.jsonl``.

    A row is either a success (its own key) or a generated counterpart (its source's).
    Excluded rows come back as ``None`` so positions still line up with the dump, and so
    they can neither be selected into a vintage nor admit their source into one.
    """
    from tuberlens.interfaces.dataset import Message

    path = A.ARMS[arm] / f"redteam_postprocessed_iter{iteration}.jsonl"
    keys: list[str | None] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i in exclude:
                keys.append(None)
                continue
            row = json.loads(line)
            key = A.canon(
                [Message(role=m["role"], content=m["content"]) for m in row["inputs"]]
            )
            keys.append(gen2src.get(key, key))
    return keys


def vintages(arm: str, iteration: int = 3,
             drop_long: str = "pair") -> tuple[dict[int, list[int]], dict]:
    """``{vintage: iteration-3 row indices}`` plus a provenance report.

    ``vintage k`` is every row of the iteration-``iteration`` dump whose source success was
    already present in the iteration-``k`` dump. Vintage ``iteration`` is therefore the
    whole (post-filter) set, and vintage 0 (base training data only) is the empty list.
    """
    gen2src = A.generated_to_source(arm)
    iters = list(range(1, iteration + 1))
    excluded, long_report = long_row_exclusions(arm, iters, drop_long, final_iteration=iteration)

    final = _source_keys(arm, iteration, gen2src, excluded[iteration])
    report = {
        "arm": arm,
        "iteration": iteration,
        "n_rows_in_dump": len(final),
        "n_rows_final": sum(1 for s in final if s is not None),
        "long_filter": long_report,
        "vintages": {},
    }

    keep: dict[int, list[int]] = {0: []}
    for k in iters:
        seen = {s for s in _source_keys(arm, k, gen2src, excluded[k]) if s is not None}
        keep[k] = [i for i, src in enumerate(final) if src is not None and src in seen]
        report["vintages"][k] = {
            "n_rows": len(keep[k]),
            "n_pairs_in_dump": len(seen),
            "n_pairs_kept": len({final[i] for i in keep[k]}),
            "n_pairs_dropped": len(seen - {s for s in final if s is not None}),
        }
    if iteration >= 2:
        report["n_not_nested_1_in_2"] = len(
            {final[i] for i in keep[1]} - {final[i] for i in keep[2]}
        )
    return keep, report


# --- eval scoring, one split at a time --------------------------------------------


def score_streaming(asm: R.Assembled, probe, eval_dir: Path) -> dict:
    """Per-split AUROC, loading one eval blob at a time.

    ``attribution_refit.score`` scores against eval sets held in ``Assembled``, which is
    right when a sweep reuses them hundreds of times. Here the seven blobs are several GB —
    held alongside the padded red-team activations and the concatenated training copy, that
    is what pushes a modest box into swap. ``mmap=True`` keeps each blob in page cache
    rather than anonymous memory, and it is dropped before the next split is opened.
    """
    import contextlib
    import io

    from tuberlens.interfaces.activations import Activation
    from tuberlens.interfaces.dataset import LabelledDataset

    out: dict[str, dict[str, float]] = {}
    for split in A.EVAL_SPLITS:
        ds = LabelledDataset.load_from(
            A.EVAL_DATASET_DIR / f"{split}.jsonl",
            pos_class_label=asm.probe.pos_class_label,
            neg_class_label=asm.probe.neg_class_label,
            combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
        )
        y = ds.labels_torch().float().cpu().numpy()
        blob = torch.load(eval_dir / f"{split}-acts_full.pt", weights_only=False, mmap=True)
        ds = ds.assign(
            activations=blob["activations"],
            attention_mask=blob["attention_mask"],
            input_ids=blob["input_ids"],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            s = probe._classifier.logits(Activation.from_dataset(ds))
        out[split] = A.auroc_both(y, s.float().cpu().numpy())
        del ds, blob, s
        gc.collect()

    for scale in ("pipeline", "rank"):
        out.setdefault("mean", {})[scale] = float(
            np.mean([out[sp][scale] for sp in A.EVAL_SPLITS])
        )
    return out


# --- assembly without the eval sets -----------------------------------------------


def assemble_train_only(arm: str, iteration: int) -> R.Assembled:
    """``attribution_refit.assemble`` minus the eval sets (see ``score_streaming``)."""
    from agentic_redteam.retrain import _infer_probe_spec

    probe = A.load_probe(A.ARMS[arm] / f"probe_iter{iteration}.pkl")
    base = A.load_base_dataset(probe)
    base_train, base_val = A.split_sides(base)
    btr_blob, bval_blob = A.base_blob_paths()
    base_train = R._attach(base_train, btr_blob)
    base_val = R._attach(base_val, bval_blob)

    redteam = A.load_redteam_dataset(arm, iteration)
    t0 = time.time()
    redteam = R._attach_redteam(redteam)
    rt_is_val = np.array([A.is_val(m) for m in redteam.inputs], dtype=bool)
    gb = redteam.other_fields["activations"].nbytes / 1e9
    print(
        f"  base: {len(base_train)} train / {len(base_val)} val\n"
        f"  red-team activations: {len(redteam)} rows, {gb:.1f} GB "
        f"({time.time() - t0:.0f}s); {int(rt_is_val.sum())} on the val side",
        flush=True,
    )

    return R.Assembled(
        probe=probe,
        probe_spec=_infer_probe_spec(probe),
        base_train=base_train,
        base_val=base_val,
        redteam=redteam,
        rt_is_val=rt_is_val,
        eval_sets={},
    )


# --- main -------------------------------------------------------------------------


def _append_progress(path: Path, row: dict) -> None:
    """Append one finished (vintage, seed) fit, fsync'd.

    A full sweep is hours, so the box dying must cost one fit rather than the run: one
    self-contained row, flushed and fsync'd, re-read on restart to skip what is done.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _done_keys(path: Path) -> set[tuple[str, int, int]]:
    """``(arm, vintage, seed)`` triples already recorded in the progress sidecar."""
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a row truncated by a hard kill; the fit simply re-runs
            out.add((r["arm"], int(r["vintage"]), int(r["seed"])))
    return out


def run_arm(arm: str, iteration: int, seeds: list[int], eval_dir: Path, out_dir: Path,
            progress_path: Path, resume: bool = True,
            only_vintages: list[int] | None = None,
            drop_long: str = "pair") -> list[dict]:
    keep, report = vintages(arm, iteration, drop_long)
    print(f"\n=== {arm} ===", flush=True)
    lf = report["long_filter"]
    if lf["policy"] != "none":
        for k in sorted(lf["per_iteration"]):
            v = lf["per_iteration"][k]
            print(
                f"  iter{k} lengths: mean {v['mean_tokens']:.0f}, p95 {v['p95_tokens']:.0f}, "
                f"max {v['max_observed']} tok — {v['n_over']} row(s) over "
                f"{v['max_tokens']}, {v.get('n_truncated_blobs', '-')} stored truncated; "
                f"{v['n_excluded']} excluded under --drop-long {lf['policy']}",
                flush=True,
            )
    for k in sorted(report["vintages"]):
        v = report["vintages"][k]
        print(
            f"  vintage {k}: {v['n_rows']:4d} rows of iter{iteration} "
            f"({v['n_pairs_kept']} of {v['n_pairs_in_dump']} iter{k} pairs survive; "
            f"{v['n_pairs_dropped']} dropped before iter{iteration})",
            flush=True,
        )
    if "n_not_nested_1_in_2" in report:
        print(f"  pairs in vintage 1 but not vintage 2: {report['n_not_nested_1_in_2']}",
              flush=True)

    done = _done_keys(progress_path) if resume else set()
    wanted = sorted(keep) if only_vintages is None else [
        k for k in sorted(keep) if k in only_vintages
    ]
    todo = [(s, k) for s in seeds for k in wanted if (arm, k, s) not in done]
    if not todo:
        print("  every (vintage, seed) already recorded — nothing to do", flush=True)
        return []
    print(f"  {len(todo)} fit(s) to run, {len(done)} already recorded", flush=True)

    # Assembly costs ~1 min and several GB, so it happens once and every fit reuses it.
    asm = assemble_train_only(arm, iteration)
    n_rows = len(asm.redteam)
    # Rows the length filter removed are in no vintage — including the full one, which is
    # the point: vintage `iteration` must be "everything we would train on", not the dump.
    excluded_rows = set(range(n_rows)) - set(keep[iteration])

    rows = []
    # Seeds outer, vintages inner: after seed s completes, the sweep holds a COMPLETE curve
    # for s seeds. The other order would leave the last vintage unmeasured at every seed
    # until the very end, which is the wrong thing to own when the box may die mid-run.
    for seed, k in todo:
        drop = set(range(n_rows)) - set(keep[k])
        t0 = time.time()
        probe, n_tr, n_val = R.refit(asm, drop_rows=drop, seed=seed)
        fit_s = time.time() - t0
        t1 = time.time()
        res = score_streaming(asm, probe, eval_dir)
        print(
            f"  seed {seed} vintage {k}: fit {fit_s:5.1f}s  score {time.time() - t1:4.1f}s  "
            f"train={n_tr} val={n_val}  best_epoch={probe._classifier.best_epoch}",
            flush=True,
        )
        print(
            "     "
            + "  ".join(
                f"{sp.replace('eval_', ''):14s}={res[sp]['pipeline']:.4f}"
                for sp in A.EVAL_SPLITS
            )
            + f"  MEAN={res['mean']['pipeline']:.4f}",
            flush=True,
        )
        row = {
            "arm": arm,
            "vintage": k,
            "seed": seed,
            "n_redteam_rows": len(keep[k]),
            "n_train": n_tr,
            "n_val": n_val,
            "drop_long": drop_long,
            "n_rows_excluded_long": len(excluded_rows & set(range(n_rows))) if drop_long != "none" else 0,
            # None when validation AUROC never improved on its initial value — which happens
            # on the tiny base-only vintage, where a seed can leave the probe at chance for
            # all its epochs. Not an error, and it must not abort a sweep whose other fits
            # are fine.
            "best_epoch": (
                None if probe._classifier.best_epoch is None
                else int(probe._classifier.best_epoch)
            ),
            "fit_seconds": fit_s,
            "auroc": res,
        }
        rows.append(row)
        _append_progress(progress_path, row)
        del probe
        gc.collect()
        torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{arm}_vintage.json").write_text(
        json.dumps({"report": report, "results": rows}, indent=2), encoding="utf-8"
    )
    del asm
    gc.collect()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--seed", type=int, default=A.SEED, help="first seed")
    ap.add_argument("--seeds", type=int, default=1, help="how many consecutive seeds")
    ap.add_argument("--vintages", nargs="*", type=int, default=None,
                    help="which vintages to fit (default: all, including 0)")
    ap.add_argument("--drop-long", choices=("pair", "row", "none"), default="pair",
                    help="how to handle conversations over the 1024-token activation cap "
                         "(default: drop the row AND its pair partner)")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="skip (vintage, seed) pairs already in the progress sidecar")
    ap.add_argument("--membership-only", action="store_true",
                    help="print the vintage/length report and exit, fitting nothing")
    ap.add_argument("--summarize-only", action="store_true",
                    help="re-aggregate the progress sidecar and exit, fitting nothing. Use "
                         "after running the arms as separate processes: each of those only "
                         "knows its own rows, so the last one to finish would otherwise "
                         "leave CSVs covering one arm.")
    ap.add_argument("--eval-dir", type=Path, default=A.EVAL_ACTIVATIONS_DIR)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=A.REPO / "results_instructions_gemma27b_vintage",
    )
    args = ap.parse_args()

    seeds = [args.seed + i for i in range(args.seeds)]
    progress = args.out_dir / "vintage_progress.jsonl"

    if args.membership_only:
        for arm in args.arm:
            _, report = vintages(arm, args.iteration, args.drop_long)
            print(json.dumps(report, indent=2))
        return

    if not args.summarize_only:
        for arm in args.arm:
            run_arm(arm, args.iteration, seeds, args.eval_dir, args.out_dir, progress,
                    resume=args.resume, only_vintages=args.vintages,
                    drop_long=args.drop_long)

    # Report from the SIDECAR, not from this process's return values: on a resumed run the
    # rows computed before the restart are on disk and nowhere else, and a summary covering
    # only this process's fits would silently understate the sweep.
    all_rows = [
        r for r in (json.loads(l) for l in progress.open(encoding="utf-8"))
        if r["arm"] in args.arm
        and (args.vintages is None or r["vintage"] in args.vintages)
    ]

    summarize(all_rows, args.out_dir)


def summarize(all_rows: list[dict], out_dir: Path) -> None:
    """Aggregate the sidecar into a mean +/- sd table and a per-fit CSV.

    Reported unpaired (mean and sd over seeds per vintage), because these are separate
    ``ProbeFactory`` fits with independent initialisations — there is no shared shuffle
    stream to difference against. That is exactly why the sd column matters more than the
    mean here: it is the quantity that makes or breaks a single-seed reading.
    """
    from collections import defaultdict

    by = defaultdict(list)
    for r in all_rows:
        by[(r["arm"], r["vintage"], r["n_redteam_rows"])].append(r)

    print("\n\n=== eval AUROC by red-team vintage — mean +/- sd over seeds (pipeline) ===")
    header = (f"{'arm':14s} {'vint':>4s} {'rows':>5s} {'n':>3s} "
              + " ".join(f"{sp.replace('eval_', ''):>19s}" for sp in A.EVAL_SPLITS)
              + f"{'MEAN':>19s}")
    print(header)
    print("-" * len(header))
    for (arm, vint, rows_n), rs in sorted(by.items()):
        cells = []
        for sp in list(A.EVAL_SPLITS) + ["mean"]:
            v = np.array([r["auroc"][sp]["pipeline"] for r in rs])
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            cells.append(f"{v.mean():.4f}+/-{sd:.4f}")
        print(f"{arm:14s} {vint:>4d} {rows_n:>5d} {len(rs):>3d} "
              + " ".join(f"{c:>19s}" for c in cells))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "vintage_auroc.csv"
    with csv.open("w", encoding="utf-8") as fh:
        fh.write("arm,vintage,seed,n_redteam_rows,dataset,auroc_pipeline,auroc_rank\n")
        for r in sorted(all_rows, key=lambda x: (x["arm"], x["vintage"], x["seed"])):
            for sp in list(A.EVAL_SPLITS) + ["mean"]:
                fh.write(
                    f"{r['arm']},{r['vintage']},{r['seed']},{r['n_redteam_rows']},{sp},"
                    f"{r['auroc'][sp]['pipeline']},{r['auroc'][sp]['rank']}\n"
                )

    stats_path = out_dir / "vintage_summary.csv"
    with stats_path.open("w", encoding="utf-8") as fh:
        fh.write("arm,vintage,n_redteam_rows,n_seeds,dataset,mean,sd,min,max\n")
        for (arm, vint, rows_n), rs in sorted(by.items()):
            for sp in list(A.EVAL_SPLITS) + ["mean"]:
                v = np.array([r["auroc"][sp]["pipeline"] for r in rs])
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                fh.write(f"{arm},{vint},{rows_n},{len(rs)},{sp},"
                         f"{v.mean()},{sd},{v.min()},{v.max()}\n")
    print(f"\nwrote {csv}\nwrote {stats_path}")


if __name__ == "__main__":
    main()
