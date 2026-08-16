#!/usr/bin/env python
"""Does it matter WHEN the dev samples enter — mixed into the red-team fit, or as a
second finetuning stage on top of it?

THE QUESTION
------------
``scripts/dev_sample_retrain.py`` answered "what do N in-distribution dev samples buy?"
by **mixing** them into the red-team training set and fitting one probe on the union.
This script asks the same question with the same data and the same N levels, but
**sequentially**:

    stage 1   fit a fresh probe on base + red-team           (exactly the mixed N=0 job)
    stage 2   continue training THAT probe on the dev rows alone

Everything else is held identical to the mixed run — same arms, same iteration-3
postprocessed dumps, same activation caches, same ``stable_train_test_split`` (so the
same dev rows are fitted and the same ones land in validation), same weight-init seed.
The only thing that changes is whether the dev gradient signal is averaged in with the
red-team signal or applied after it. That makes the two summaries directly comparable
row for row, which ``analyze`` does.

WHY THIS IS NOT A ONE-LINE VARIATION
------------------------------------
Sequential training introduces two knobs the mixed design does not have, and neither has
an obviously-right setting, so both are run as a factorial rather than guessed:

``--val-modes``  What early stopping in stage 2 watches.
    ``mixed``  base+red-team validation ∪ the dev validation rows — the SAME validation
               set the mixed run used at that N. Model selection therefore sees the
               forgetting, and the comparison isolates the fit itself.
    ``dev``    the dev validation rows only. Selection is purely in-distribution, which
               is what someone finetuning on a target domain would actually do. Note
               N=2 leaves 2 validation rows, so its AUROC is 0 or 1 — read that level as
               "no model selection", not as a measurement.

``--lr-factors``  How hard stage 2 pulls. ``PytorchAdamClassifier.train`` builds a fresh
    AdamW and a fresh cosine schedule on every call, so a plain continuation restarts at
    the pipeline's lr=5e-3 over a ~90-row set — closer to "retrain the head on dev,
    warm-started" than to a finetune. ``0.1`` (lr and ``final_lr`` both scaled, so the
    schedule keeps its shape) is the gentler reading. Both are cheap; run both.

There is no stage-2 replay of red-team data — the request was dev-only finetuning, and
replay would be a third design with its own mixing ratio.

N=0 IS THE STAGE-1 PROBE. With no dev rows there is no second stage, so the N=0 row is
stage 1 itself. It should reproduce the mixed run's N=0 AUROC (0.9112 deepseekv4pro /
0.8978 gptoss120b) — ``analyze`` prints that check, and a mismatch means the two scripts
have drifted apart and nothing below it is comparable.

COST AND MEMORY
---------------
No model is ever loaded: every activation this needs is already in the caches the mixed
run used (red-team + the 120 dev conversations extracted by ``dev_sample_retrain.py``'s
``extract`` stage). ``fit`` refuses to start if any blob is missing rather than silently
forwarding conversations through gemma-3-27b.

Stage 1 is fitted **once per arm** and deep-copied into every finetune job, which is both
the honest design (all N levels branch off the identical probe) and the cheap one — the
~8 GB red-team training tensor is assembled once and freed before the finetunes start.
Peak host RAM is therefore the stage-1 assembly, the same peak the mixed run already
survived on this box; the finetunes run in a few GB.

``fit`` and ``eval`` are separate stages because they have disjoint memory profiles: the
training activations are gone by the time the 4.3 GB of eval blobs are read.

Typical use::

    python scripts/dev_sample_finetune.py --work-dir results/devsamples_kfold/finetune --dry-run
    python scripts/dev_sample_finetune.py --work-dir results/devsamples_kfold/finetune
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

# BEFORE any tuberlens import. `PytorchAdamClassifier.wandb_project` defaults to
# `global_settings.WANDB_PROJECT`, and that default is bound when the class body runs —
# so mutating the setting later has no effect, and an ambient WANDB_PROJECT (even the
# empty string, which is not None and so still counts as "configured") makes every one
# of the 34 fits try to `wandb.init` and die on a missing API key. This is a batch
# experiment; nothing here wants a W&B run per fit.
os.environ.pop("WANDB_PROJECT", None)
os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "disabled")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
# Import the mixed-run script rather than copying its selection logic: the two
# experiments are only comparable if "N=8" means the identical 32 conversations, and
# that is decided by _dev_pool/_dev_rows. Importing is side-effect-free (module level is
# constants and a sys.path insert).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import dev_sample_retrain as dsr  # noqa: E402

ARMS = dsr.ARMS
POS_LABEL, NEG_LABEL = dsr.POS_LABEL, dsr.NEG_LABEL
COMBINE, CONVERT = dsr.COMBINE, dsr.CONVERT
TEST_SIZE, SPLIT_SEED = dsr.TEST_SIZE, dsr.SPLIT_SEED

# N=0 is stage 1 itself and is emitted once per arm, not once per (val_mode, lr_factor).
DEV_LEVELS = [2, 8, 16, 30]
VAL_MODES = ["mixed", "dev"]
LR_FACTORS = [1.0, 0.1]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _msg_key(messages) -> str:
    """Identity of a (already-transformed) conversation.

    Same basis as ``_redteam_activation_cache_path`` so a row can be matched to its
    activation blob and to its counterpart in another dataset without relying on
    positional alignment.
    """
    return json.dumps(
        [{"role": m.role, "content": m.content} for m in messages],
        sort_keys=True,
        ensure_ascii=False,
    )


def _labelled(records: list[dict]):
    """Record dicts -> the transformed ``LabelledDataset`` ``retrain_probe`` would build."""
    from agentic_redteam.retrain import (
        _apply_message_transforms,
        _records_to_labelled_dataset,
    )

    return _apply_message_transforms(
        _records_to_labelled_dataset([dsr._as_attempt(r) for r in records]),
        COMBINE,
        CONVERT,
    )


def _no_model_factory(what: str):
    """A ``get_model`` that refuses instead of loading gemma-3-27b.

    Every activation this script needs was computed by the mixed run. A missing blob is
    a broken cache, and letting it fall through would turn a 30-second fit into hours of
    forwards on an 8 GB GPU — the failure mode is a slow success, which is the hardest
    kind to notice.
    """

    def _get_model():
        raise SystemExit(
            f"{what}: an activation blob is missing from the cache. Restore it with\n"
            f"  python scripts/publish_kaggle_redteam_activations.py restore ...\n"
            f"(or re-run scripts/dev_sample_retrain.py --stages extract for dev rows). "
            f"Pass --allow-missing-acts to compute it here instead — that loads the 27B "
            f"model."
        )

    return _get_model


def _free() -> None:
    from agentic_redteam.retrain import _release_free_heap

    gc.collect()
    _release_free_heap()


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("nan")


# --------------------------------------------------------------------------- #
# assembling one arm's activations
# --------------------------------------------------------------------------- #
def _activated_parts(dataset, cache_dir: Path, args, what: str) -> list:
    """Per-conversation activated parts, in dataset order.

    ``_redteam_activation_parts`` appends cache hits first and freshly-computed chunks
    after, so its output is only in dataset order when everything hit. That is the
    normal case here (and ``--allow-missing-acts`` is the escape hatch), but the whole
    index bookkeeping below depends on it, so it is checked rather than assumed.
    """
    from agentic_redteam.retrain import _redteam_activation_parts

    get_model = (
        _no_model_factory(what)
        if not args.allow_missing_acts
        else _make_lazy_model(args)
    )
    parts = _redteam_activation_parts(
        dataset,
        cache_dir,
        args.model_name,
        args.layer,
        COMBINE,
        CONVERT,
        get_model,
        False,
    )
    if parts is None:
        return []
    flat_keys = [_msg_key(msgs) for p in parts for msgs in p.inputs]
    if flat_keys != [_msg_key(msgs) for msgs in dataset.inputs]:
        raise SystemExit(
            f"{what}: activated parts are not in dataset order (some blobs were "
            f"computed rather than loaded). Restore the cache and re-run."
        )
    return parts


def _make_lazy_model(args):
    holder: dict = {"m": None}

    def _get():
        from agentic_redteam.model_loading import load_extraction_model

        if holder["m"] is None:
            holder["m"] = load_extraction_model(args.model_name, args.layer, verbose=True)
        return holder["m"]

    return _get


def assemble(arm: str, pool: dict, args) -> dict:
    """Everything one arm's fits need, off the caches, with no model load.

    Returns the stage-1 train/val sets, the merged dev train/val sets, and the position
    of every dev row inside them so an N level can be selected by ``__getitem__``
    without re-merging anything.
    """
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.model import LLMModel

    from agentic_redteam.retrain import (
        _base_activation_cache_paths,
        _concatenate_consuming,
        _cpu_unpickle,
        _infer_probe_spec,
        stable_train_test_split,
    )

    probe_dir = REPO_ROOT / ARMS[arm]
    with (probe_dir / "probe_iter0.pkl").open("rb") as f:
        base_probe = _cpu_unpickle(f)
    meta = {
        "probe_spec": _infer_probe_spec(base_probe),
        "model_name": base_probe.model_name,
        "layer": int(base_probe.layer),
        "pos": getattr(base_probe, "pos_class_label", POS_LABEL),
        "neg": getattr(base_probe, "neg_class_label", NEG_LABEL),
        "description": getattr(base_probe, "description", None),
        "hyper_params": dict(base_probe.hyper_params),
    }
    del base_probe

    cache_dir = args.cache_dir / "base_activations"

    # --- base split: one blob per side, exactly as retrain_probe keys it -------------
    base_ds = LabelledDataset.load_from(
        args.base_data,
        pos_class_label=meta["pos"],
        neg_class_label=meta["neg"],
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
    )
    base_train, base_val = stable_train_test_split(
        base_ds, test_size=TEST_SIZE, split_field=None, seed=SPLIT_SEED
    )
    train_blob, val_blob = _base_activation_cache_paths(
        cache_dir,
        args.base_data,
        meta["model_name"],
        meta["layer"],
        SPLIT_SEED,
        TEST_SIZE,
        None,
        COMBINE,
        CONVERT,
        1.0,
    )
    for p in (train_blob, val_blob):
        if not p.exists():
            raise SystemExit(f"base activation blob missing: {p}")

    def _assign_blob(ds, path):
        acts = LLMModel.load_activations(path)
        return ds.assign(
            activations=acts.activations,
            attention_mask=acts.attention_mask,
            input_ids=acts.input_ids,
        )

    base_train_a = _assign_blob(base_train, train_blob)
    base_val_a = _assign_blob(base_val, val_blob)

    # --- red-team and dev sets ------------------------------------------------------
    dump = probe_dir / f"redteam_postprocessed_iter{args.iteration}.jsonl"
    rt_ds = _labelled(dsr._redteam_records(dump))
    dev_ds = _labelled(dsr._dev_records(dsr._dev_rows(pool, max(args.dev_levels))))

    rt_train, rt_val = stable_train_test_split(
        rt_ds, test_size=TEST_SIZE, split_field=None, seed=SPLIT_SEED
    )
    dev_train, dev_val = stable_train_test_split(
        dev_ds, test_size=TEST_SIZE, split_field=None, seed=SPLIT_SEED
    )

    dev_train_keys = [_msg_key(m) for m in dev_train.inputs]
    dev_val_keys = [_msg_key(m) for m in dev_val.inputs]

    t0 = time.time()
    rt_train_parts = _activated_parts(rt_train, cache_dir, args, f"{arm} red-team train")
    rt_val_parts = _activated_parts(rt_val, cache_dir, args, f"{arm} red-team val")
    dev_train_parts = _activated_parts(dev_train, cache_dir, args, f"{arm} dev train")
    dev_val_parts = _activated_parts(dev_val, cache_dir, args, f"{arm} dev val")
    print(
        f"  activations loaded: base {len(base_train)}+{len(base_val)}, red-team "
        f"{len(rt_train)}+{len(rt_val)}, dev {len(dev_train)}+{len(dev_val)} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    n_nondev_val = len(base_val) + len(rt_val)

    # Merge the validation side FIRST, and merge it once with the dev rows already in
    # it. Every val set any job needs is then a row selection out of this one tensor:
    # re-concatenating per level would cost a full extra copy of it each time, and
    # holding a second copy alongside the ~8 GB training tensor is what OOMs this box.
    # (Padding width differs from a per-level merge, which changes nothing: the
    # classifier masks the pad region.)
    val_full = _concatenate_consuming(
        [base_val_a, *rt_val_parts, *dev_val_parts]
    )
    del base_val_a, rt_val_parts, dev_val_parts
    _free()

    dev_train_all = _concatenate_consuming(dev_train_parts)
    del dev_train_parts
    _free()

    stage1_val = val_full[list(range(n_nondev_val))]
    stage1_train = _concatenate_consuming([base_train_a, *rt_train_parts])
    del base_train_a, rt_train_parts
    _free()

    print(
        f"  stage-1 train {len(stage1_train)} rows / val {len(stage1_val)} rows; "
        f"dev pool {len(dev_train_all)} fit + {len(dev_val_keys)} val "
        f"(RSS {_rss_gb():.1f} GB)",
        flush=True,
    )

    return {
        "meta": meta,
        "stage1_train": stage1_train,
        "stage1_val": stage1_val,
        "val_full": val_full,
        "n_nondev_val": n_nondev_val,
        "dev_train_all": dev_train_all,
        "dev_train_keys": dev_train_keys,
        "dev_val_keys": dev_val_keys,
        "n_redteam": len(rt_ds),
    }


# --------------------------------------------------------------------------- #
# stage: fit
# --------------------------------------------------------------------------- #
def _scaled_args(hyper_params: dict, factor: float, epochs: int | None, n_rows: int) -> dict:
    """The stage-2 training args: lr and final_lr scaled, schedule shape preserved.

    ``gradient_accumulation_steps`` is additionally clamped to the number of batches the
    stage-2 set actually produces, and this is load-bearing rather than a tidy-up.
    ``PytorchAdamClassifier.train`` calls ``optimizer.step()`` only on
    ``(batch_idx + 1) % gradient_accumulation_steps == 0`` and zeroes the gradients at
    the top of every epoch — so with fewer batches per epoch than accumulation steps the
    condition never fires and **the weights never move**. The pipeline's setting is 4
    with a ~750-row training set (47 batches/epoch, 11 steps), which is fine; the dev-only
    stage 2 is 6-94 rows, i.e. 1-6 batches. Unclamped, N=2 and N=8 would run 200 epochs,
    early-stop, and return the stage-1 probe unchanged — a silent no-op that looks exactly
    like "finetuning does nothing", which is the conclusion this run is meant to test.
    Clamping keeps the pipeline's effective batch size wherever the data supports it and
    guarantees at least one optimizer step per epoch where it does not.
    """
    out = copy.deepcopy(hyper_params)
    out["optimizer_args"] = dict(out["optimizer_args"])
    out["optimizer_args"]["lr"] = out["optimizer_args"]["lr"] * factor
    out["final_lr"] = out["final_lr"] * factor
    if epochs is not None:
        out["epochs"] = epochs
    n_batches = max(1, -(-n_rows // out["batch_size"]))
    out["gradient_accumulation_steps"] = max(
        1, min(out.get("gradient_accumulation_steps", 1), n_batches)
    )
    return out


def _job_name(arm: str, n: int, val_mode: str, lr_factor: float, seed: int) -> str:
    if n == 0:
        return f"{arm}__n0__stage1__s{seed}"
    return f"{arm}__n{n}__val-{val_mode}__lr{lr_factor:g}__s{seed}"


def _jobs(args) -> list[dict]:
    jobs = []
    for arm in args.arms:
        for seed in args.seeds:
            jobs.append(
                {"arm": arm, "n": 0, "val_mode": "-", "lr_factor": 1.0, "seed": seed}
            )
            for n in args.dev_levels:
                for val_mode in args.val_modes:
                    for lr_factor in args.lr_factors:
                        jobs.append(
                            {
                                "arm": arm,
                                "n": n,
                                "val_mode": val_mode,
                                "lr_factor": lr_factor,
                                "seed": seed,
                            }
                        )
    return jobs


def stage_fit(args, pool) -> None:
    from tuberlens.probes.probe_factory import ProbeFactory

    from agentic_redteam.evaluation import seed_everything

    fit_dir = args.work_dir / "probes"
    fit_dir.mkdir(parents=True, exist_ok=True)

    jobs = _jobs(args)
    todo = [
        j
        for j in jobs
        if not (fit_dir / f"{_job_name(**j)}.pkl").exists()
    ]
    print(f"\nfit: {len(jobs)} job(s); {len(jobs) - len(todo)} already fitted, "
          f"{len(todo)} to go", flush=True)
    if args.dry_run:
        for j in todo:
            print(f"  {_job_name(**j)}")
        return
    if not todo:
        return

    for arm in args.arms:
        arm_jobs = [j for j in todo if j["arm"] == arm]
        if not arm_jobs:
            continue
        print(f"\n=== {arm} ===", flush=True)
        bundle = assemble(arm, pool, args)
        meta = bundle["meta"]

        # --- stage 1, once per (arm, seed) ------------------------------------------
        stage1: dict[int, object] = {}
        for seed in sorted({j["seed"] for j in arm_jobs}):
            t0 = time.time()
            seed_everything(seed)
            probe = ProbeFactory.build(
                probe_spec=meta["probe_spec"],
                train_dataset=bundle["stage1_train"],
                model_name=meta["model_name"],
                layer=meta["layer"],
                validation_dataset=bundle["stage1_val"],
                use_store=False,
                pos_class_label=meta["pos"],
                neg_class_label=meta["neg"],
                probe_description=meta["description"],
            )
            stage1[seed] = probe
            print(
                f"  stage1 seed={seed}: best_epoch={probe._classifier.best_epoch} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
            job = {"arm": arm, "n": 0, "val_mode": "-", "lr_factor": 1.0, "seed": seed}
            if any(_job_name(**j) == _job_name(**job) for j in arm_jobs):
                _save(fit_dir, job, probe, bundle, n_dev_fit=0, n_dev_val=0)

        # The red-team training tensor is the single largest object in the run and
        # nothing after this point reads it. Dropping it here is what lets the finetunes
        # run in a few GB instead of alongside ~8 GB of activations.
        bundle["stage1_train"] = None
        bundle["stage1_val"] = None
        _free()
        print(f"  released stage-1 training activations (RSS {_rss_gb():.1f} GB)",
              flush=True)

        # --- stage 2 ----------------------------------------------------------------
        val_full = bundle["val_full"]
        nondev = list(range(bundle["n_nondev_val"]))
        dev_train_pos = {k: i for i, k in enumerate(bundle["dev_train_keys"])}
        dev_val_pos = {
            k: bundle["n_nondev_val"] + i for i, k in enumerate(bundle["dev_val_keys"])
        }

        ft_jobs = [j for j in arm_jobs if j["n"] > 0]
        for done, job in enumerate(ft_jobs, 1):
            n = job["n"]
            level_keys = {
                _msg_key(m)
                for m in _labelled(dsr._dev_records(dsr._dev_rows(pool, n))).inputs
            }
            fit_idx = sorted(dev_train_pos[k] for k in level_keys if k in dev_train_pos)
            val_idx = sorted(dev_val_pos[k] for k in level_keys if k in dev_val_pos)
            if not fit_idx:
                print(f"  {_job_name(**job)}: no dev rows on the train side; skipped")
                continue

            if job["val_mode"] == "dev" and not _both_classes(bundle, val_idx):
                # roc_auc_score raises on a single-class y_true, so this variant simply
                # does not exist at this level. Recorded rather than quietly swapped for
                # a different validation set, which would put a differently-selected
                # model into the same column of the comparison table.
                print(
                    f"  {_job_name(**job)}: dev validation has {len(val_idx)} row(s) of "
                    f"one class — no AUROC is defined; skipped"
                )
                continue

            ft_train = bundle["dev_train_all"][fit_idx]
            if job["val_mode"] == "mixed":
                ft_val = val_full[nondev + val_idx]
            else:
                ft_val = val_full[val_idx]

            probe = copy.deepcopy(stage1[job["seed"]])
            orig = probe.hyper_params
            scaled = _scaled_args(orig, job["lr_factor"], args.ft_epochs, len(fit_idx))
            probe.hyper_params = scaled
            probe._classifier.training_args = scaled
            t0 = time.time()
            try:
                seed_everything(job["seed"])
                # initialize_model=False continues from the stage-1 weights instead of
                # re-initialising — the whole point of the sequential arm.
                probe.fit(ft_train, ft_val, initialize_model=False)
            finally:
                probe.hyper_params = orig
                probe._classifier.training_args = orig
            ft_best_epoch = probe._classifier.best_epoch
            _save(
                fit_dir,
                job,
                probe,
                bundle,
                n_dev_fit=len(fit_idx),
                n_dev_val=len(val_idx),
                ft_best_epoch=ft_best_epoch,
            )
            del ft_train, ft_val, probe
            _free()
            print(
                f"  [{done}/{len(ft_jobs)}] {_job_name(**job)}: "
                f"{len(fit_idx)} fit / {len(val_idx)} val rows, "
                f"best_epoch={ft_best_epoch} "
                f"({time.time() - t0:.0f}s, RSS {_rss_gb():.1f} GB)",
                flush=True,
            )

        del bundle, stage1, val_full
        _free()


def _both_classes(bundle: dict, val_idx: list[int]) -> bool:
    """Does this validation row selection contain both classes?"""
    labels = bundle["val_full"].other_fields["labels"]
    return len({labels[i] for i in val_idx}) == 2


def _save(fit_dir: Path, job: dict, probe, bundle: dict, **extra) -> None:
    name = _job_name(**job)
    with (fit_dir / f"{name}.pkl").open("wb") as f:
        pickle.dump(probe, f)
    meta = dict(job)
    meta.update(
        {
            "name": name,
            "mode": "finetune" if job["n"] > 0 else "stage1",
            "n_redteam": bundle["n_redteam"],
            "n_dev_offered": job["n"] * 4,
            "stage1_best_epoch": probe._classifier.best_epoch if job["n"] == 0 else None,
        }
    )
    meta.update(extra)
    (fit_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- #
# stage: eval
# --------------------------------------------------------------------------- #
def stage_eval(args) -> None:
    from agentic_redteam.evaluation import evaluate_probe

    fit_dir = args.work_dir / "probes"
    res_dir = args.work_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    pkls = sorted(fit_dir.glob("*.pkl"))
    todo = [p for p in pkls if not (res_dir / f"{p.stem}.csv").exists()]
    print(f"\neval: {len(pkls)} probe(s); {len(pkls) - len(todo)} already scored, "
          f"{len(todo)} to go", flush=True)
    if args.dry_run or not todo:
        return

    t_start = time.time()
    for done, pkl in enumerate(todo, 1):
        meta = json.loads((fit_dir / f"{pkl.stem}.json").read_text())
        t0 = time.time()
        df = evaluate_probe(
            probe_path=pkl,
            eval_dataset_dir=args.eval_dir,
            activations_cache_dir=args.cache_dir / "eval_activations",
            max_samples=None,
            seed=SPLIT_SEED,
            combine_consecutive_messages=COMBINE,
            convert_tool_to_assistant=CONVERT,
        )
        df = df.assign(
            **{
                k: v
                for k, v in meta.items()
                if k
                in {
                    "arm", "n", "val_mode", "lr_factor", "seed", "mode",
                    "n_redteam", "n_dev_offered", "n_dev_fit", "n_dev_val",
                    "ft_best_epoch",
                }
            }
        )
        df.to_csv(res_dir / f"{pkl.stem}.csv", index=False)
        r = df[df.dataset == "mean"].iloc[0]
        elapsed = time.time() - t_start
        print(
            f"  [{done}/{len(todo)}] {pkl.stem}: AUROC={r.auroc:.4f} "
            f"acc={r.accuracy:.4f} ({time.time() - t0:.0f}s, "
            f"ETA {elapsed / done * (len(todo) - done) / 60:.0f}m)",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# stage: analyze
# --------------------------------------------------------------------------- #
MIXED_SUMMARY = REPO_ROOT / "results/devsamples_kfold/dev_samples/dev_sample_summary.csv"


def stage_analyze(args) -> None:
    import pandas as pd

    res_dir = args.work_dir / "results"
    csvs = sorted(res_dir.glob("*.csv"))
    if not csvs:
        print("analyze: no result CSVs yet")
        return
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df.to_csv(args.work_dir / "finetune_all.csv", index=False)

    summary = (
        df.groupby(["arm", "n", "val_mode", "lr_factor", "dataset"])
        .agg(
            auroc=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            accuracy=("accuracy", "mean"),
            tpr_at_fpr=("tpr_at_fpr", "mean"),
            n_dev_fit=("n_dev_fit", "max"),
            n_dev_val=("n_dev_val", "max"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
        .round(4)
    )
    summary.to_csv(args.work_dir / "finetune_summary.csv", index=False)

    mean = summary[summary.dataset == "mean"]
    print("\n" + "=" * 92)
    print("SEQUENTIAL: stage 1 = base+red-team, stage 2 = dev only (eval AUROC, 4-split mean)")
    print("=" * 92)
    for arm in sorted(mean.arm.unique()):
        sub = mean[mean.arm == arm]
        base = sub[sub.n == 0]
        ref = float(base.auroc.iloc[0]) if len(base) else float("nan")
        print(f"\n{arm}   (stage-1 / N=0 AUROC = {ref:.4f})")
        print(f"  {'val':>6} {'lr×':>5} " + "".join(f"{f'N={n}':>18}" for n in args.dev_levels))
        for val_mode in args.val_modes:
            for lr in args.lr_factors:
                cells = []
                for n in args.dev_levels:
                    row = sub[
                        (sub.n == n) & (sub.val_mode == val_mode) & (sub.lr_factor == lr)
                    ]
                    if len(row):
                        a = float(row.auroc.iloc[0])
                        cells.append(f"{a:.4f} ({a - ref:+.4f})")
                    else:
                        cells.append("—")
                print(f"  {val_mode:>6} {lr:>5g} " + "".join(f"{c:>18}" for c in cells))

    # --- the comparison the run exists for ------------------------------------------
    if not MIXED_SUMMARY.exists():
        print(f"\n(no mixed-run summary at {MIXED_SUMMARY}; skipping the comparison)")
        return
    mixed = pd.read_csv(MIXED_SUMMARY)
    mixed = mixed[mixed.dataset == "mean"][["arm", "dev_per_split", "auroc", "accuracy"]]
    mixed = mixed.rename(columns={"dev_per_split": "n", "auroc": "mixed_auroc",
                                  "accuracy": "mixed_acc"})

    print("\n" + "=" * 92)
    print("MIXED vs SEQUENTIAL  (eval AUROC, 4-split mean)")
    print("=" * 92)
    for arm in sorted(mean.arm.unique()):
        m = mixed[mixed.arm == arm].set_index("n")
        print(f"\n{arm}")
        header = f"  {'N':>3} {'mixed':>8}"
        for val_mode in args.val_modes:
            for lr in args.lr_factors:
                header += f" {f'seq {val_mode}/{lr:g}':>16}"
        print(header)
        for n in [0] + list(args.dev_levels):
            mv = float(m.loc[n, "mixed_auroc"]) if n in m.index else float("nan")
            line = f"  {n:>3} {mv:>8.4f}"
            for val_mode in args.val_modes:
                for lr in args.lr_factors:
                    if n == 0:
                        row = mean[(mean.arm == arm) & (mean.n == 0)]
                    else:
                        row = mean[
                            (mean.arm == arm)
                            & (mean.n == n)
                            & (mean.val_mode == val_mode)
                            & (mean.lr_factor == lr)
                        ]
                    if len(row):
                        a = float(row.auroc.iloc[0])
                        line += f" {a:>8.4f} {a - mv:>+7.4f}"
                    else:
                        line += f" {'—':>16}"
            print(line)

    joined = mean.merge(mixed, on=["arm", "n"], how="left")
    joined["auroc_delta_vs_mixed"] = (joined.auroc - joined.mixed_auroc).round(4)
    joined.to_csv(args.work_dir / "finetune_vs_mixed.csv", index=False)
    print(f"\nwrote {args.work_dir / 'finetune_summary.csv'}")
    print(f"wrote {args.work_dir / 'finetune_vs_mixed.csv'}")

    print("\nper-split detail (AUROC):")
    piv = summary[summary.dataset != "mean"].pivot_table(
        index=["arm", "val_mode", "lr_factor", "dataset"], columns="n", values="auroc"
    ).round(4)
    print(piv.to_string())

    (args.work_dir / "finetune_done.json").write_text(
        json.dumps(
            {
                "n_jobs": len(csvs),
                "levels": args.dev_levels,
                "arms": args.arms,
                "val_modes": args.val_modes,
                "lr_factors": args.lr_factors,
                "seeds": args.seeds,
            },
            indent=2,
        )
    )


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=dsr.DEFAULT_CACHE,
                    help="Holds base_activations/ and eval_activations/")
    ap.add_argument("--base-data", type=Path, default=dsr.DEFAULT_BASE_DATA)
    ap.add_argument("--eval-dir", type=Path, default=dsr.DEFAULT_EVAL_DIR)
    ap.add_argument("--dev-dir", type=Path, default=dsr.DEFAULT_DEV_DIR)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--dev-levels", nargs="+", type=int, default=DEV_LEVELS,
                    help="Dev samples PER SPLIT for stage 2 (N=0 is stage 1 itself)")
    ap.add_argument("--val-modes", nargs="+", default=VAL_MODES,
                    choices=VAL_MODES,
                    help="What stage-2 early stopping watches")
    ap.add_argument("--lr-factors", nargs="+", type=float, default=LR_FACTORS,
                    help="Multipliers on the stage-2 lr and final_lr")
    ap.add_argument("--ft-epochs", type=int, default=None,
                    help="Stage-2 epoch budget (default: the probe's own)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="Weight-init seeds; the train/val split seed is fixed at 42 so "
                         "the activation caches stay hits")
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--model-name", default="google/gemma-3-27b-it")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--dev-order-seed", type=int, default=42)
    ap.add_argument("--stages", nargs="+", default=["fit", "eval", "analyze"],
                    choices=["fit", "eval", "analyze"])
    ap.add_argument("--allow-missing-acts", action="store_true",
                    help="Let `fit` compute activations that are not cached (loads the "
                         "27B model — normally a mistake)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    pool = dsr._dev_pool(args.dev_dir, args.dev_order_seed)
    if not pool:
        raise SystemExit(f"no dev sample JSONLs in {args.dev_dir}")

    print(f"work dir  : {args.work_dir}")
    print(f"cache     : {args.cache_dir}")
    print(f"arms      : {', '.join(args.arms)}  (iteration {args.iteration})")
    print(f"stage 2   : N {args.dev_levels} per split; val {args.val_modes}; "
          f"lr× {args.lr_factors}; seeds {args.seeds}")
    print("\nselection per level (train/val landing is deterministic):")
    for n in args.dev_levels:
        rows = dsr._dev_rows(pool, n)
        fit, val = dsr._fit_val_counts(dsr._dev_records(rows))
        print(f"  N={n:<3d} -> {len(rows):3d} rows  ({fit} fitted in stage 2, "
              f"{val} to validation)")
    print()

    if "fit" in args.stages:
        stage_fit(args, pool)
    if "eval" in args.stages:
        stage_eval(args)
    if "analyze" in args.stages and not args.dry_run:
        stage_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
