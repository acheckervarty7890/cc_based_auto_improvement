#!/usr/bin/env python
"""Does adding a handful of in-distribution DEV samples to the red-team training set
move eval performance, and how fast?

THE QUESTION
------------
The red-team pipeline trains on ``data/hu_harm_llama70b_50.jsonl`` plus attacker-written
conversations, and is scored on ``eval_dataset_hu_ha/`` — a distribution it has never
seen. ``scripts/eval_kfold_cv.py`` measures the other extreme: a probe trained on the
eval distribution itself. This script fills in the curve between them, by adding N rows
per split from ``dev_samples/`` — held-out companions to the four eval splits, drawn from
the same generators — to an otherwise unchanged iteration-3 retrain.

    N = 0    the pipeline as it stands (reference)
    N = 2    8 rows added  (2 per split, 1 positive + 1 negative)
    N = 8    32 rows
    N = 16   64 rows
    N = 30   120 rows — every dev sample there is

N counts rows PER SPLIT, so N=30 exhausts each 30-row dev file. The subsets are NESTED
(the N=2 rows are in the N=8 set, and so on) and class-balanced within each split, so
the five points are a dose-response curve rather than five unrelated draws.

DEV SAMPLES ARE NOT EVAL SAMPLES. Verified before this was written: zero conversations
and zero first-user prompts are shared between ``dev_samples/train_*.jsonl`` and
``eval_dataset_hu_ha/eval_*.jsonl``, and none of the 120 appear in the base training
data. Three of the four eval splits are internally paired (the same prompt with a
harmful and a non-harmful reply), so a prompt-level check is the one that matters and it
is clean.

WHAT IS HELD FIXED
------------------
Everything except N. Both attacker arms are run because the dev-sample effect has to be
separable from the arm:

    deepseekv4pro   probes/hu_harm_gemma27b_deepseekv4pro_batch  878 iter3 rows
    gptoss120b      probes/hu_harm_gemma27b_gptoss120b_batch     778 iter3 rows

The red-team set is read from ``redteam_postprocessed_iter3.jsonl`` — the dump
``retrain_probe`` writes *after* filtering and contrastive generation and *before* the
split, i.e. a verbatim record of what trained probe_iter3. So ``preprocessing`` is None
here: re-running filter+contrastive would refit the bag-of-words dropper on a different
set and change which rows survive, which is a second variable. This also means the
contrastive pairs used are the SHORTENED ones (commit 98b5011) — the dump on this branch
is the post-shortening one, and its rows' activation blobs are the newly published ones.

Architecture, base data, base/eval activation caches, transforms, test_size and the
train/val seed all come from the pipeline's own defaults and are identical across jobs.
Weight init is seed 42 for every job (``--seeds`` takes more if you want a noise floor;
with one seed a delta smaller than seed noise is not readable, and seed noise on this
setup has measured ~0.008 AUROC).

THE TRAIN/VAL SPLIT APPLIES TO DEV ROWS TOO
-------------------------------------------
``retrain_probe`` runs every non-base sample through ``stable_train_test_split``, so
roughly ``test_size`` of the dev rows land in validation rather than training. This is
not worked around — the point is to add them the way the pipeline adds data — but it does
mean "N=2" is 8 rows offered and typically 6-7 rows fitted. The split is
content-deterministic, so which dev rows go to validation is fixed and NESTED across the
N levels, and the plan printed by ``--dry-run`` reports the exact fit/val breakdown per
level. Read that before reading the curve.

STAGES
------
``extract``  push the 120 dev conversations through gemma-3-27b once and write each one
             into the per-conversation red-team activation cache. This is the ONLY stage
             that loads a model. Blobs are content-addressed, so both arms and all five
             N levels share them, and a re-run recomputes nothing.
``run``      one retrain + full-split eval per (arm, N, seed) job; one CSV each, skipped
             if the CSV exists, so the stage is resumable.
``analyze``  join the CSVs into ``dev_sample_summary.csv`` and print the curve.

The red-team activations themselves must already be in the cache — restore them from
Kaggle with ``scripts/publish_kaggle_redteam_activations.py restore`` (the run script
does this). Anything still missing is computed by ``run``, which would silently turn a
5-minute job into an hours-long one, so ``run`` refuses to start unless every red-team
blob it needs is present (``--allow-missing-acts`` overrides).

Typical use::

    python scripts/dev_sample_retrain.py --work-dir results/devsamples_kfold/dev --dry-run
    python scripts/dev_sample_retrain.py --work-dir results/devsamples_kfold/dev
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CACHE = REPO_ROOT / "results_hu_harm_gemma27b_batch_ablation"
DEFAULT_BASE_DATA = REPO_ROOT / "data" / "hu_harm_llama70b_50.jsonl"
DEFAULT_EVAL_DIR = REPO_ROOT / "eval_dataset_hu_ha"
DEFAULT_DEV_DIR = REPO_ROOT / "dev_samples"

ARMS = {
    "deepseekv4pro": "probes/hu_harm_gemma27b_deepseekv4pro_batch",
    "gptoss120b": "probes/hu_harm_gemma27b_gptoss120b_batch",
}
DEV_LEVELS = [0, 2, 8, 16, 30]
POS_LABEL = "harmful_to_human"
NEG_LABEL = "not_harmful_to_human"
# Both hu_harm configs' `eval:` section; the caches are keyed on these, so they are a
# property of the published blobs rather than a choice this script gets to make.
COMBINE = True
CONVERT = True
TEST_SIZE = 0.2
SPLIT_SEED = 42


# --------------------------------------------------------------------------- #
# selecting dev samples
# --------------------------------------------------------------------------- #
def _dev_pool(dev_dir: Path, seed: int) -> dict[str, dict[str, list[dict]]]:
    """``{split: {"positive": [...], "negative": [...]}}`` in a fixed, shuffled order.

    The order is shuffled (from a local ``default_rng``, not the global RNG) rather than
    left as file order: the dev files are grouped, so taking the first k in file order
    would sample one region of each generator's output. Shuffling once and taking
    prefixes keeps the N levels nested while making each level a spread of the file.
    """
    import numpy as np

    pool: dict[str, dict[str, list[dict]]] = {}
    for path in sorted(dev_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_class: dict[str, list[dict]] = {"positive": [], "negative": []}
        for row in rows:
            label = row.get("labels")
            if label == POS_LABEL:
                by_class["positive"].append(row)
            elif label == NEG_LABEL:
                by_class["negative"].append(row)
        # Namespaced per split so one split's order does not depend on another's size.
        # sha256, not hash(): Python salts str hashing per process, so hash() here would
        # reshuffle the dev pool on every invocation and the N levels would stop nesting
        # across a resumed run.
        digest = hashlib.sha256(f"{seed}:{path.stem}".encode()).hexdigest()[:8]
        rng = np.random.default_rng(int(digest, 16))
        for side in by_class:
            idx = np.arange(len(by_class[side]))
            rng.shuffle(idx)
            by_class[side] = [by_class[side][i] for i in idx]
        pool[path.stem] = by_class
    return pool


def _dev_rows(pool: dict, n_per_split: int) -> list[dict]:
    """The N-per-split, class-balanced, nested selection.

    An odd N would be split as ceil/floor; every level here is even, so this is N/2 of
    each class. A split short of N/2 in one class contributes what it has — reported
    rather than silently rebalanced, because quietly taking extra rows from the other
    class would break the balance the levels are supposed to hold fixed.
    """
    out: list[dict] = []
    for split in sorted(pool):
        want_pos = (n_per_split + 1) // 2
        want_neg = n_per_split // 2
        for side, want in (("positive", want_pos), ("negative", want_neg)):
            chosen = pool[split][side][:want]
            for row in chosen:
                out.append(row | {"_dev_split": split, "_dev_side": side})
    return out


# --------------------------------------------------------------------------- #
# building the training JSONL
# --------------------------------------------------------------------------- #
def _record(messages: list[dict], human_label: str, origin: str) -> dict:
    """Wrap a conversation in the AttemptRecord shape ``retrain_probe`` reads.

    ``retrain_probe`` takes red-team JSONL and keeps rows with ``success: true``,
    labelling each by ``judge_label``. Both the postprocessed dump and the dev samples
    are already labelled data, so every field except the conversation and its label is
    inert filler; ``attacker_model`` carries the origin so the file is self-describing.
    ``error_type`` is set consistently with the label only because ``_success_human_label``
    falls back to it when ``judge_label`` matches neither class — which cannot happen
    here, but a row whose two label fields disagreed would be a trap for anyone reading
    the file later.
    """
    return {
        "sample": {"messages": messages},
        "probe_score": 0.5,
        "probe_predicts_positive": human_label == NEG_LABEL,
        "judge_label": human_label,
        "judge_reason": "",
        "judge_confidence": 10,
        "success": True,
        "attacker_model": origin,
        "run_id": "dev_sample_retrain",
        "round": 0,
        "iteration": 0,
        "error_type": "false_negative" if human_label == POS_LABEL else "false_positive",
        "pos_class_label": POS_LABEL,
        "neg_class_label": NEG_LABEL,
    }


def _as_attempt(record: dict):
    """The dict form of a record as an ``AttemptRecord``.

    ``AttemptRecord`` parses from a JSONL *row*, not from a dict, so this round-trips
    through JSON rather than reimplementing the field coercions — which is the point:
    the same parser that reads a real run's log reads these, so a shape error here
    surfaces immediately instead of producing a subtly different object.
    """
    from agentic_redteam.persistence import AttemptRecord

    return AttemptRecord.from_jsonl_row(json.dumps(record, ensure_ascii=False))


def _redteam_records(dump_path: Path) -> list[dict]:
    """``redteam_postprocessed_iter{N}.jsonl`` rows as AttemptRecords.

    The dump stores the canonical enum value ("positive"/"negative") in ``label``; the
    record shape wants the human-readable class string, so it is mapped back.
    """
    out = []
    for line in dump_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = POS_LABEL if row["label"] == "positive" else NEG_LABEL
        out.append(_record(row["inputs"], label, "redteam_iter3"))
    return out


def _dev_records(rows: list[dict]) -> list[dict]:
    """Dev-sample rows as AttemptRecords. ``inputs`` is a JSON-encoded string on disk."""
    out = []
    for row in rows:
        messages = json.loads(row["inputs"]) if isinstance(row["inputs"], str) else row["inputs"]
        out.append(_record(messages, row["labels"], f"dev:{row['_dev_split']}"))
    return out


# --------------------------------------------------------------------------- #
# what the deterministic split will do to the dev rows
# --------------------------------------------------------------------------- #
def _fit_val_counts(records: list[dict]) -> tuple[int, int]:
    """How many of these conversations ``stable_train_test_split`` sends each way.

    Reproduced here rather than measured after the fact so ``--dry-run`` can report it.
    The transforms are applied first because ``retrain_probe`` splits the *transformed*
    dataset, and the hash is over message content.
    """
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

    from agentic_redteam.retrain import _split_unit_interval

    n_val = 0
    for rec in records:
        dialogue = [
            TLMessage(role=m["role"], content=m["content"])
            for m in rec["sample"]["messages"]
        ]
        if CONVERT:
            dialogue = LabelledDataset._convert_tool_to_assistant(dialogue)
        if COMBINE:
            dialogue = LabelledDataset._combine_consecutive_messages(dialogue)
        if _split_unit_interval(dialogue, None, SPLIT_SEED) < TEST_SIZE:
            n_val += 1
    return len(records) - n_val, n_val


# --------------------------------------------------------------------------- #
# stage: extract dev activations
# --------------------------------------------------------------------------- #
def stage_extract(args, pool) -> None:
    """Warm the per-conversation cache with all 120 dev conversations.

    Done once, up front, rather than letting the first ``run`` job compute them: the
    blobs are content-addressed, so every arm and every N level shares them, and doing
    it here keeps the model load in a stage that is obviously the expensive one. A
    conversation already cached is skipped by ``_activate_redteam_cached`` itself.
    """
    from agentic_redteam.model_loading import load_extraction_model
    from agentic_redteam.retrain import (
        _activate_redteam_cached,
        _apply_message_transforms,
        _redteam_activation_cache_path,
        _records_to_labelled_dataset,
    )
    rows = _dev_rows(pool, max(DEV_LEVELS))
    records = [_as_attempt(r) for r in _dev_records(rows)]
    dataset = _apply_message_transforms(
        _records_to_labelled_dataset(records), COMBINE, CONVERT
    )
    cache_dir = args.cache_dir / "base_activations"
    missing = [
        msgs
        for msgs in dataset.inputs
        if not _redteam_activation_cache_path(
            cache_dir, msgs, args.model_name, args.layer, COMBINE, CONVERT
        ).exists()
    ]
    print(f"extract: {len(dataset)} dev conversations, {len(missing)} not yet cached")
    if not missing:
        print("extract: nothing to do")
        return
    if args.dry_run:
        return

    loaded: dict = {"m": None}

    def _get_model():
        if loaded["m"] is None:
            loaded["m"] = load_extraction_model(args.model_name, args.layer, verbose=True)
        return loaded["m"]

    t0 = time.time()
    _activate_redteam_cached(
        dataset, cache_dir, args.model_name, args.layer, COMBINE, CONVERT, _get_model, True
    )
    loaded["m"] = None
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"extract: done in {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------- #
# stage: run
# --------------------------------------------------------------------------- #
def _jobs(args) -> list[tuple[str, int, int]]:
    return [
        (arm, n, seed)
        for arm in args.arms
        for n in args.dev_levels
        for seed in args.seeds
    ]


def _missing_redteam_blobs(records: list[dict], cache_dir: Path, args) -> int:
    from agentic_redteam.retrain import (
        _apply_message_transforms,
        _redteam_activation_cache_path,
        _records_to_labelled_dataset,
    )

    dataset = _apply_message_transforms(
        _records_to_labelled_dataset([_as_attempt(r) for r in records]),
        COMBINE,
        CONVERT,
    )
    return sum(
        1
        for msgs in dataset.inputs
        if not _redteam_activation_cache_path(
            cache_dir, msgs, args.model_name, args.layer, COMBINE, CONVERT
        ).exists()
    )


def stage_run(args, pool) -> None:
    import pandas as pd

    import agentic_redteam.evaluation as _ev
    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import retrain_probe

    real_seed_everything = _ev.seed_everything
    res_dir = args.work_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir / "base_activations"

    jobs = _jobs(args)
    todo = [j for j in jobs if not (res_dir / f"{j[0]}__n{j[1]}__s{j[2]}.csv").exists()]
    print(f"\nrun: {len(jobs)} job(s); {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to go", flush=True)

    if not args.allow_missing_acts:
        for arm in args.arms:
            dump = REPO_ROOT / ARMS[arm] / f"redteam_postprocessed_iter{args.iteration}.jsonl"
            records = _redteam_records(dump)
            n_missing = _missing_redteam_blobs(records, cache_dir, args)
            if n_missing and args.dry_run:
                # A dry run's job is to report the plan, including "the cache is not
                # ready" — raising here would make --dry-run unusable on a box that has
                # not restored yet, which is exactly when you want to run it.
                print(f"run: {arm}: {n_missing} of {len(records)} red-team activations "
                      f"NOT in {cache_dir} — restore before the real run")
                continue
            if n_missing:
                raise SystemExit(
                    f"{arm}: {n_missing} of the iteration-{args.iteration} red-team "
                    f"activations are not in {cache_dir}. Every job would forward them "
                    f"through {args.model_name}, turning minutes into hours. Restore "
                    f"them first:\n"
                    f"  python scripts/publish_kaggle_redteam_activations.py restore "
                    f"--arm {arm} --iterations {args.iteration} --cache-dir {cache_dir}\n"
                    f"or pass --allow-missing-acts to compute them here."
                )
            print(f"run: {arm} red-team activations all present")

    if args.dry_run:
        for arm, n, seed in todo:
            print(f"  {arm:16s} n={n:<3d} seed={seed}")
        return

    t_start = time.time()
    failed = []
    for done, (arm, n, seed) in enumerate(todo, 1):
        csv = res_dir / f"{arm}__n{n}__s{seed}.csv"
        probe_dir = REPO_ROOT / ARMS[arm]
        dump = probe_dir / f"redteam_postprocessed_iter{args.iteration}.jsonl"
        rt = _redteam_records(dump)
        dev = _dev_records(_dev_rows(pool, n))
        jsonl = res_dir / f".train_{arm}__n{n}__s{seed}.jsonl"
        jsonl.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rt + dev)
        )
        probe_out = res_dir / f".probe_{arm}__n{n}__s{seed}.pkl"
        dev_fit, dev_val = _fit_val_counts(dev) if dev else (0, 0)

        t0 = time.time()
        try:
            # Only the weight init varies with `seed`; retrain_probe's own seed stays at
            # SPLIT_SEED so the train/val split and every activation cache key are
            # unchanged across seeds. (retrain_probe re-applies seed_everything right
            # before the fit, which is the call being patched.)
            _ev.seed_everything = lambda _s, _S=seed: real_seed_everything(_S)
            try:
                retrain_probe(
                    jsonl_path=jsonl,
                    base_probe_path=probe_dir / "probe_iter0.pkl",
                    base_training_data_path=args.base_data,
                    new_probe_path=probe_out,
                    probe_spec=None,           # inherit the pipeline's architecture
                    preprocessing=None,        # the dump is ALREADY postprocessed
                    min_judge_confidence=0,
                    test_size=TEST_SIZE,
                    split_field=None,
                    seed=SPLIT_SEED,
                    base_activation_cache_dir=cache_dir,
                    combine_consecutive_messages=COMBINE,
                    convert_tool_to_assistant=CONVERT,
                    verbose=False,
                )
            finally:
                _ev.seed_everything = real_seed_everything
            df = evaluate_probe(
                probe_path=probe_out,
                eval_dataset_dir=args.eval_dir,
                activations_cache_dir=args.cache_dir / "eval_activations",
                max_samples=None,
                seed=SPLIT_SEED,
                combine_consecutive_messages=COMBINE,
                convert_tool_to_assistant=CONVERT,
            )
        except Exception as e:  # noqa: BLE001 — one job's failure must not sink the rest
            failed.append((arm, n, seed, repr(e)))
            print(f"  [{done}/{len(todo)}] {arm:16s} n={n:<3d} FAILED: "
                  f"{type(e).__name__}: {e}", flush=True)
            (res_dir / f"{arm}__n{n}__s{seed}.FAILED").write_text(
                f"{type(e).__name__}: {e}\n"
            )
            continue
        finally:
            probe_out.unlink(missing_ok=True)
            jsonl.unlink(missing_ok=True)

        df = df.assign(
            arm=arm, dev_per_split=n, init_seed=seed, iteration=args.iteration,
            n_redteam=len(rt), n_dev_offered=len(dev),
            n_dev_fit=dev_fit, n_dev_val=dev_val,
        )
        df.to_csv(csv, index=False)
        r = df[df.dataset == "mean"].iloc[0]
        elapsed = time.time() - t_start
        eta = elapsed / done * (len(todo) - done)
        print(f"  [{done}/{len(todo)}] {arm:16s} n={n:<3d} seed={seed} "
              f"dev={len(dev):3d} ({dev_fit} fit / {dev_val} val)  "
              f"AUROC={r.auroc:.4f} acc={r.accuracy:.4f} "
              f"({time.time() - t0:.0f}s, ETA {eta / 60:.0f}m)", flush=True)

    if failed:
        print(f"\nrun: {len(failed)} job(s) failed:")
        for arm, n, seed, e in failed:
            print(f"  {arm} n={n} s={seed}: {e}")


# --------------------------------------------------------------------------- #
# stage: analyze
# --------------------------------------------------------------------------- #
def stage_analyze(args) -> None:
    import pandas as pd

    res_dir = args.work_dir / "results"
    csvs = sorted(res_dir.glob("*.csv"))
    if not csvs:
        print("analyze: no result CSVs yet")
        return
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df.to_csv(args.work_dir / "dev_sample_all.csv", index=False)

    summary = (
        df.groupby(["arm", "dev_per_split", "dataset"])
        .agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
             accuracy=("accuracy", "mean"), tpr_at_fpr=("tpr_at_fpr", "mean"),
             n_dev_fit=("n_dev_fit", "max"), n_dev_val=("n_dev_val", "max"),
             n_seeds=("init_seed", "nunique"))
        .reset_index()
        .round(4)
    )
    summary.to_csv(args.work_dir / "dev_sample_summary.csv", index=False)

    print("\n" + "=" * 90)
    print("Eval AUROC vs. dev samples added per split (mean row = across the 4 splits)")
    print("=" * 90)
    mean = summary[summary.dataset == "mean"]
    for arm in sorted(mean.arm.unique()):
        sub = mean[mean.arm == arm].sort_values("dev_per_split")
        base = sub[sub.dev_per_split == 0]
        ref = float(base.auroc.iloc[0]) if len(base) else float("nan")
        print(f"\n{arm}")
        print(f"  {'N/split':>8} {'rows':>6} {'fit':>5} {'val':>5} "
              f"{'AUROC':>8} {'delta':>8} {'acc':>8}")
        for _, r in sub.iterrows():
            delta = r.auroc - ref
            print(f"  {int(r.dev_per_split):>8} {int(r.dev_per_split) * 4:>6} "
                  f"{int(r.n_dev_fit):>5} {int(r.n_dev_val):>5} "
                  f"{r.auroc:>8.4f} {delta:>+8.4f} {r.accuracy:>8.4f}")

    print("\nper-split detail:")
    piv = summary.pivot_table(
        index=["arm", "dataset"], columns="dev_per_split", values="auroc"
    ).round(4)
    print(piv.to_string())
    print(f"\nwrote {args.work_dir / 'dev_sample_summary.csv'}")
    (args.work_dir / "dev_sample_done.json").write_text(
        json.dumps({"n_jobs": len(csvs), "levels": args.dev_levels,
                    "arms": args.arms, "seeds": args.seeds}, indent=2)
    )


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                    help="Holds base_activations/ and eval_activations/")
    ap.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA)
    ap.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    ap.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--dev-levels", nargs="+", type=int, default=DEV_LEVELS,
                    help="Dev samples PER SPLIT (default: 0 2 8 16 30)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="Weight-init seeds. The train/val split seed is fixed at 42 so "
                         "the activation caches stay hits")
    ap.add_argument("--iteration", type=int, default=3,
                    help="Which redteam_postprocessed_iter{N}.jsonl to train from")
    ap.add_argument("--model-name", default="google/gemma-3-27b-it")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--dev-order-seed", type=int, default=42,
                    help="Seeds the per-split shuffle the nested N levels are prefixes of")
    ap.add_argument("--stages", nargs="+", default=["extract", "run", "analyze"],
                    choices=["extract", "run", "analyze"])
    ap.add_argument("--allow-missing-acts", action="store_true",
                    help="Let `run` compute red-team activations that are not cached "
                         "(loads the 27B model — normally a mistake)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    pool = _dev_pool(args.dev_dir, args.dev_order_seed)
    if not pool:
        raise SystemExit(f"no dev sample JSONLs in {args.dev_dir}")

    print(f"work dir  : {args.work_dir}")
    print(f"cache     : {args.cache_dir}")
    print(f"arms      : {', '.join(args.arms)}  (iteration {args.iteration})")
    print(f"dev levels: {args.dev_levels} per split; seeds {args.seeds}")
    print("dev pool  :")
    for split, sides in sorted(pool.items()):
        print(f"  {split:24s} {len(sides['positive'])} positive / "
              f"{len(sides['negative'])} negative")
    print("\nselection per level (train/val landing is deterministic):")
    for n in args.dev_levels:
        rows = _dev_rows(pool, n)
        if not rows:
            print(f"  N={n:<3d} -> 0 rows")
            continue
        fit, val = _fit_val_counts(_dev_records(rows))
        print(f"  N={n:<3d} -> {len(rows):3d} rows  ({fit} fitted, {val} to validation)")
    print()

    if "extract" in args.stages:
        stage_extract(args, pool)
    if "run" in args.stages:
        stage_run(args, pool)
    if "analyze" in args.stages and not args.dry_run:
        stage_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
