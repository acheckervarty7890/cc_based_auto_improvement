#!/usr/bin/env python
"""Cross-concept ceiling: how well can ONE probe do on all three concepts' eval data?

"Ceiling" here is the ceiling_analysis branch's definition — the best eval-set
performance this probe family (`linear_then_softmax` head on google/gemma-3-27b-it
layer 32) can reach *when it is trained on eval-distribution data* — estimated by
k-fold cross-validation **inside the eval sets themselves**: fit on the rows outside
fold k, early-stop against a fixed reserved dev slice, score fold k. Every eval row
gets exactly one out-of-fold score, and a ceiling probe differs from an ordinary probe
only in its training data.

This script asks that question of the three concepts **pooled**. Arms:

  within/<concept>  the per-concept ceiling — CV inside one concept's own eval splits.
                    The baseline the cross-concept numbers have to be read against.
  cross/native      CV over all three concepts' eval rows at once, each row labelled
                    positive iff it is its own concept's positive class.
  cross/aligned     the same pool with `instructions` INVERTED.

Why two cross orientations. `hu_ha` and `highstakes` both call the *problematic*
response positive (harmful, high-stakes), but `instructions` calls the *desirable* one
positive (assistant_follows_the_instruction). Pooling as-is therefore asks one direction
to place "harmful" and "correctly followed the instruction" on the same side, which is
not obviously the question anyone means; `aligned` flips instructions so all three
positives are the problematic response. Neither is privileged and the flip is not a
symmetry of the pooled task (it re-labels a third of the data, so the pooled AUROC does
not simply become 1 - x), so both are reported.

Sizing. The full pool is 6576 rows, and pooling pads everything to the longest split's
1024 tokens — 72 GB of fp16 activations, which neither fits the box nor the card. Each
split is therefore balanced-subsampled to `--per-split` rows (default 100, which is this
repo's own `--eval-max-samples` default), giving 1500 pooled rows = 16.5 GB, of which a
1200-row training fold is 13.2 GB and does fit the 24 GB card alongside the 3.3 GB
validation slice — so the fits are GPU-resident and cost minutes rather than hours.

No model is ever loaded: every eval and dev activation is read from the blobs already
under activations/{eval,dev}/<concept>/.

    .venv_claude/bin/python scripts/cross_concept_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Single probes, never ensembles (the ceiling_analysis convention), so fusion is
# irrelevant — but pin it anyway so a stray environment can't change what is fit.
os.environ.setdefault("PROBE_FUSED_ENSEMBLE", "0")

MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
COMBINE = True
CONVERT = True
SEED = 42
N_FOLDS = 5
FPR = 0.01
DEFAULT_ARCH = "linear_then_softmax"

RESULTS = REPO_ROOT / "results_ceiling"

# concept -> (pos_class_label, neg_class_label)
CONCEPTS = {
    "hu_ha": ("harmful_to_human", "not_harmful_to_human"),
    "highstakes": ("high-stakes", "low-stakes"),
    "instructions": (
        "assistant_follows_the_instruction",
        "assistant_does_not_follow_the_instruction",
    ),
}
# Concepts whose positive class is the *desirable* behaviour, and so get inverted in
# the `aligned` orientation.
INVERT_WHEN_ALIGNED = {"instructions"}


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------------------
def _balanced_indices(labels, n_per_class: int, key: str) -> list[int]:
    """Indices of a balanced subsample, drawn reproducibly from `key` alone.

    Seeded on the key rather than on a global RNG so the same split yields the same rows
    no matter what else the process has drawn, and so re-running one arm cannot shift
    another's data.
    """
    rng = random.Random(key)
    by_class: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)
    take = min([n_per_class] + [len(v) for v in by_class.values()])
    out: list[int] = []
    for y in sorted(by_class):
        out.extend(rng.sample(by_class[y], take))
    return sorted(out)


def _clean_dataset(ds, concept: str, split: str):
    """Rebuild a dataset carrying only the fields the pooling needs.

    ``_concatenate_consuming`` falls back to an intersection of columns, so the eval
    splits' provenance fields (which differ per concept) would be dropped anyway — but
    dropping them here makes that explicit and keeps `concept`/`split` present on every
    part, which is what the per-split reporting is keyed on.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    n = len(ds)
    return LabelledDataset(
        inputs=list(ds.inputs),
        ids=list(ds.ids),
        other_fields={
            "labels": list(ds.other_fields["labels"]),
            "concept": [concept] * n,
            "split": [split] * n,
            "activations": ds.other_fields["activations"],
            "attention_mask": ds.other_fields["attention_mask"],
            "input_ids": ds.other_fields["input_ids"],
        },
    )


def _load_split(kind: str, concept: str, jsonl: Path, n_per_class: int):
    """One split, subsampled, with its precomputed activations attached."""
    import torch
    from tuberlens.interfaces.dataset import LabelledDataset

    pos, neg = CONCEPTS[concept]
    ds = LabelledDataset.load_from(
        jsonl.resolve(),
        pos_class_label=pos,
        neg_class_label=neg,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
    )
    blob_path = REPO_ROOT / "activations" / kind / concept / f"{jsonl.stem}-acts_full.pt"
    if not blob_path.is_file():
        raise SystemExit(f"missing activation blob {blob_path}")
    blob = torch.load(blob_path, map_location="cpu", mmap=True, weights_only=False)
    if blob["activations"].shape[0] != len(ds):
        raise SystemExit(
            f"{blob_path.name}: blob has {blob['activations'].shape[0]} rows, "
            f"split has {len(ds)}"
        )
    if blob.get("model_name") != MODEL_NAME or int(blob.get("layer", -1)) != LAYER:
        raise SystemExit(
            f"{blob_path.name}: blob is {blob.get('model_name')} L{blob.get('layer')}, "
            f"expected {MODEL_NAME} L{LAYER}"
        )
    ds = ds.assign(
        activations=blob["activations"],
        attention_mask=blob["attention_mask"],
        input_ids=blob["input_ids"],
    )
    idx = _balanced_indices(
        ds.other_fields["labels"], n_per_class, f"{SEED}:{kind}:{concept}:{jsonl.stem}"
    )
    # Advanced indexing materialises only the chosen rows, so the mmapped blob is never
    # read in full.
    return _clean_dataset(ds[idx], concept, jsonl.stem)


def build_concept_pool(kind: str, concept: str, n_per_class: int, verbose: bool = True):
    """All of one concept's `kind` splits, subsampled and concatenated."""
    from agentic_redteam.retrain import _concatenate_consuming

    root = REPO_ROOT / ("eval_sets" if kind == "eval" else "dev_samples") / concept
    parts = []
    for jsonl in sorted(root.glob("*.jsonl")):
        part = _load_split(kind, concept, jsonl, n_per_class)
        if verbose:
            n_pos = sum(1 for y in part.other_fields["labels"] if int(y) == 1)
            print(f"    {kind:<4} {concept:<12} {jsonl.stem:<28} "
                  f"{len(part):>4} rows ({n_pos} pos)")
        parts.append(part)
    pool = _concatenate_consuming(parts)
    if verbose:
        seq = pool.other_fields["activations"].shape[1]
        gib = pool.other_fields["activations"].nelement() * 2 / 2**30
        print(f"    -> {concept} {kind} pool: {len(pool)} rows x {seq} tok ({gib:.1f} GiB)")
    return pool


# --------------------------------------------------------------------------------------
# cross-validation
# --------------------------------------------------------------------------------------
def _stratified_folds(pool, n_folds: int, key: str) -> list[int]:
    """Fold assignment per row, stratified by (concept, split, label).

    Stratifying on the split as well as the label keeps every fold's *composition*
    identical, so a fold is never easier merely because it drew more of a split the
    probe happens to be good at.
    """
    rng = random.Random(key)
    strata: dict[tuple, list[int]] = {}
    of = pool.other_fields
    for i in range(len(pool)):
        strata.setdefault(
            (of["concept"][i], of["split"][i], int(of["labels"][i])), []
        ).append(i)
    folds = [0] * len(pool)
    for stratum in sorted(strata):
        rows = strata[stratum][:]
        rng.shuffle(rows)
        for j, i in enumerate(rows):
            folds[i] = j % n_folds
    return folds


def spec_for(n_train: int):
    import math

    from tuberlens.interfaces.probes import ProbeSpec

    from agentic_redteam.retrain import _coerce_probe_spec

    base = _coerce_probe_spec(DEFAULT_ARCH)
    hp = dict(base.hyperparams)
    n_batches = max(1, math.ceil(n_train / int(hp.get("batch_size", 16))))
    hp["gradient_accumulation_steps"] = min(
        int(hp.get("gradient_accumulation_steps", 1)), n_batches
    )
    return ProbeSpec(name=base.name, hyperparams=hp)


def run_cv(pool, validation, arm: str, n_folds: int, verbose: bool = True) -> list[dict]:
    """One arm: k-fold CV over `pool`, early-stopping on `validation`. Returns row dicts."""
    import numpy as np
    from tuberlens.probes.probe_factory import ProbeFactory

    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.retrain import _to_device_for_fit

    folds = _stratified_folds(pool, n_folds, f"{SEED}:{arm}")
    scores = np.zeros(len(pool), dtype=float)
    of = pool.other_fields

    for k in range(n_folds):
        train_idx = [i for i, f in enumerate(folds) if f != k]
        test_idx = [i for i, f in enumerate(folds) if f == k]
        train, test = pool[train_idx], pool[test_idx]
        spec = spec_for(len(train))
        if verbose:
            print(f"  [{arm}] fold {k}: {len(train)} train / {len(test)} test / "
                  f"{len(validation)} val", flush=True)
        t0 = time.time()
        _to_device_for_fit([train, validation], verbose=verbose)
        seed_everything(SEED)
        probe = ProbeFactory.build(
            probe_spec=spec,
            train_dataset=train,
            model_name=MODEL_NAME,
            layer=LAYER,
            validation_dataset=validation,
            use_store=False,
            pos_class_label="positive",
            neg_class_label="negative",
            probe_description=f"cross-concept ceiling probe, arm {arm}, fold {k}",
        )
        scores[test_idx] = probe.predict_proba(test)
        if verbose:
            print(f"  [{arm}] fold {k} done in {time.time()-t0:.0f}s", flush=True)
        del train, test, probe
        _free_gpu()

    y = np.array([int(v) for v in of["labels"]], dtype=int)
    rows = []
    from tuberlens.evaluation import calculate_metrics

    def _row(scope: str, name: str, mask) -> dict:
        yy, ss = y[mask], scores[mask]
        if len(set(yy.tolist())) < 2:
            return {}
        m = calculate_metrics(yy, ss, FPR)
        return {"arm": arm, "scope": scope, "name": name, "n": int(mask.sum()), **m}

    concepts = np.array(of["concept"])
    splits = np.array(of["split"])
    for c in sorted(set(of["concept"])):
        for s in sorted(set(splits[concepts == c])):
            r = _row("split", f"{c}/{s}", (concepts == c) & (splits == s))
            if r:
                rows.append(r)
    for c in sorted(set(of["concept"])):
        r = _row("concept", c, concepts == c)
        if r:
            rows.append(r)
    r = _row("pooled", "ALL", np.ones(len(pool), dtype=bool))
    if r:
        rows.append(r)
    return rows


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-split", type=int, default=100,
                    help="Balanced eval rows per split (default 100; 50 per class)")
    ap.add_argument("--dev-per-concept", type=int, default=100,
                    help="Balanced dev rows per concept for the fixed validation slice")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--arm", action="append",
                    choices=["within", "cross_native", "cross_aligned"],
                    help="Arms to run (repeatable; default: all)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    arms = args.arm or ["within", "cross_native", "cross_aligned"]

    import numpy as np
    import pandas as pd
    from tuberlens.config import global_settings

    from agentic_redteam.retrain import _concatenate_consuming

    global_settings.PROBE_FUSED_ENSEMBLE = False
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / "cross_concept_ceiling.csv"
    done, frames = set(), []
    if out_csv.exists() and not args.force:
        prev = pd.read_csv(out_csv)
        frames.append(prev)
        done = set(prev["arm"])
        print(f"resuming: {sorted(done)} already in {out_csv.name}")

    print("assembling the fixed validation slice from dev_samples/ ...")
    dev_parts = [
        build_concept_pool("dev", c, max(1, args.dev_per_concept // 2))
        for c in CONCEPTS
    ]
    validation = _concatenate_consuming(dev_parts)
    print(f"  validation: {len(validation)} rows x "
          f"{validation.other_fields['activations'].shape[1]} tok")

    print("\nassembling the eval pools ...")
    per_concept = {
        c: build_concept_pool("eval", c, max(1, args.per_split // 2)) for c in CONCEPTS
    }

    # `within` first: the cross pool is built by CONSUMING the per-concept pools.
    if "within" in arms and "within" not in done:
        for c, pool in per_concept.items():
            frames.append(pd.DataFrame(run_cv(pool, validation, f"within/{c}", args.folds)))
            pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
    elif "within" in done:
        print("[skip] within")

    if "cross_native" in arms or "cross_aligned" in arms:
        cross = _concatenate_consuming([per_concept[c] for c in CONCEPTS])
        per_concept.clear()
        seq = cross.other_fields["activations"].shape[1]
        gib = cross.other_fields["activations"].nelement() * 2 / 2**30
        print(f"\ncross pool: {len(cross)} rows x {seq} tok ({gib:.1f} GiB)")

        if "cross_native" in arms and "cross_native" not in done:
            frames.append(pd.DataFrame(run_cv(cross, validation, "cross/native", args.folds)))
            pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)

        if "cross_aligned" in arms and "cross_aligned" not in done:
            of = cross.other_fields
            flipped = [
                (1 - int(y)) if of["concept"][i] in INVERT_WHEN_ALIGNED else int(y)
                for i, y in enumerate(of["labels"])
            ]
            aligned = cross.assign(labels=flipped)
            frames.append(pd.DataFrame(run_cv(aligned, validation, "cross/aligned", args.folds)))
            pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}")
        print(df[df.scope != "split"].pivot_table(
            index="name", columns="arm", values="auroc").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
