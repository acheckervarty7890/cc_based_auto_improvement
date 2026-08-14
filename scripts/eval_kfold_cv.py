#!/usr/bin/env python
"""k-fold cross-validation of a probe trained ON the human-harm eval set itself.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
Every other number in this repo is a probe trained on ``data/hu_harm_llama70b_50.jsonl``
(+ red-team data) and scored on ``eval_dataset_hu_ha/``, i.e. strictly out of
distribution. That leaves an open question the red-team results cannot answer on their
own: how much of the residual eval error is *concept difficulty* and how much is
*distribution shift*? This script answers the first half by removing the second — it
trains on the eval distribution and tests on held-out rows of the same distribution.

So the number produced here is a **ceiling**, not a comparable baseline. It is what the
eval data itself supports at this model/layer/architecture with ~690 training rows. A
pipeline probe scoring below it is losing something to shift; a ceiling that is itself
low says the concept is not linearly available at layer 32 and no amount of red-teaming
will fix that.

Base training data and red-team data are deliberately NOT used — see ``--help`` on
``--folds`` for the two fold geometries, both of which are eval-only.

TWO FOLD GEOMETRIES, BOTH REPORTED
----------------------------------
``pooled``  the four splits are concatenated and folded together, stratified on
            (split, label). Each fold's training set therefore contains rows from all
            four splits, and the held-out fold is scored both overall and broken down
            per split. This is "how well does the eval distribution predict itself".

``within``  k-fold run independently inside each split. Training data comes only from
            the split being tested, so this is the per-split ceiling with ~4/5 of that
            split's rows (110 for ai_dilemmas, 320 for balanced_refusal, ...). Lower
            than pooled where a split is small, and the gap between the two is itself
            informative: it says how much the splits transfer to each other.

Note ``eval_balanced_refusal`` (400 rows) is 46% of the pooled corpus, so a pooled
number is dominated by it. The per-split breakdown of the pooled arm is what to read if
you care about the other three.

VALIDATION SET
--------------
tuberlens' Adam-trained probes take a validation set (early stopping). It is carved out
of the k-1 *training* folds, never out of the held-out fold, by the same stratified
splitter — so the test fold is untouched by fitting in every sense. ``--val-size 0``
passes no validation set at all.

NO MODEL IS LOADED. Activations come from the eval cache — the same
``<split>-acts_full.pt`` blobs ``evaluate_probe`` uses, published on Kaggle. Fill the
cache first with ``scripts/fetch_kaggle_eval_activations.py`` (or ``--fetch-kaggle``).
Because the blobs are whole-split and row-ordered, fold membership is just indexing.

Typical use::

    python scripts/eval_kfold_cv.py --fetch-kaggle \
        --work-dir results/devsamples_kfold/kfold_cv

    # already cached
    python scripts/eval_kfold_cv.py --work-dir results/devsamples_kfold/kfold_cv

Outputs, under ``--work-dir``:

    folds/<geometry>__f<N>.csv   one file per fold, written AS THAT FOLD FINISHES
    kfold_cv_folds.csv           all of the above concatenated
    kfold_cv_summary.csv         mean/sd/min/max over folds, per geometry and scored split

The per-fold files are what make the stage resumable and what give a long run something
on disk to look at (and for the failsafe committer to checkpoint) before it ends. A fold
whose CSV exists is skipped on a re-run; delete the file to recompute it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

DEFAULT_EVAL_DIR = REPO_ROOT / "eval_dataset_hu_ha"
DEFAULT_EVAL_CACHE = (
    REPO_ROOT / "results_hu_harm_gemma27b_batch_ablation" / "eval_activations"
)
DEFAULT_PROBE = (
    REPO_ROOT / "probes" / "hu_harm_gemma27b_deepseekv4pro_batch" / "probe_iter0.pkl"
)
# The transforms the published eval blobs were computed under (both hu_harm configs'
# `eval:` section). They are not a free choice here: the cache is keyed by path alone,
# so loading the splits under different transforms would silently pair each row's
# activations with a different conversation.
COMBINE = True
CONVERT = True
CACHE_STEM = "acts_full.pt"
FPR = 0.01


# --------------------------------------------------------------------------- #
# loading: splits + their cached activations
# --------------------------------------------------------------------------- #
def _load_probe(path: Path):
    """Load the probe we inherit model/layer/labels/architecture from.

    Only its *metadata* is used — this script fits fresh probes. probe_iter0 is the
    initial probe (base data only), so taking it from either arm gives the same answer;
    it is simply the one file that carries the architecture the pipeline actually used.
    """
    with path.open("rb") as f:
        return pickle.load(f)


def _load_split(name: str, eval_dir: Path, cache_dir: Path, probe):
    """One eval split as a LabelledDataset with its cached activations attached.

    Raises rather than falling back to computing, because computing means loading
    gemma-3-27b and this script exists precisely to avoid that. A missing blob is a
    setup error (run the fetch), not something to paper over.
    """
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.model import LLMModel

    from agentic_redteam.kaggle_activations import _blob_header

    dataset = LabelledDataset.load_from(
        eval_dir / f"{name}.jsonl",
        pos_class_label=probe.pos_class_label,
        neg_class_label=probe.neg_class_label,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
    )
    blob = cache_dir / f"{name}-{Path(CACHE_STEM).stem}.pt"
    if not blob.exists():
        raise SystemExit(
            f"missing eval activations for split {name!r}: {blob}\n"
            f"Fill the cache first:\n"
            f"  python scripts/fetch_kaggle_eval_activations.py --concept hu_harm "
            f"--cache-dir {cache_dir}"
        )
    # Validate against the blob's own header, not against the loaded Activation:
    # LLMModel.load_activations DISCARDS the model_name/layer the blob was saved with,
    # so by the time it returns there is nothing left to check. These caches otherwise
    # load by path without validating their inputs, and this one was fetched from a
    # remote store rather than computed here, so it gets checked.
    header = _blob_header(blob)
    got_model, got_layer = header.get("model_name"), header.get("layer")
    if got_model not in (None, probe.model_name) or (
        got_layer is not None and int(got_layer) != int(probe.layer)
    ):
        raise SystemExit(
            f"{blob.name} was computed with {got_model} L{got_layer}, but the probe is "
            f"{probe.model_name} L{probe.layer}."
        )
    acts = LLMModel.load_activations(blob)
    if len(acts.activations) != len(dataset):
        raise SystemExit(
            f"{blob.name}: {len(acts.activations)} rows but split {name!r} has "
            f"{len(dataset)}. The blob does not describe this split — refusing to use it."
        )
    return dataset.assign(
        activations=acts.activations,
        attention_mask=acts.attention_mask,
        input_ids=acts.input_ids,
    )


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def _stratified_folds(strata: list, k: int, seed: int) -> list[list[int]]:
    """Assign every index to one of ``k`` folds, balanced within each stratum.

    Deals each stratum's (shuffled) members round-robin into the folds, starting at a
    rotating offset so that strata smaller than k do not all pile their single member
    into fold 0. The shuffle is from a local ``default_rng(seed)``, not the global RNG,
    so fold membership is reproducible regardless of what else ran first in the process.
    """
    rng = np.random.default_rng(seed)
    by_stratum: dict = defaultdict(list)
    for i, s in enumerate(strata):
        by_stratum[s].append(i)

    folds: list[list[int]] = [[] for _ in range(k)]
    for offset, key in enumerate(sorted(by_stratum, key=str)):
        members = np.array(by_stratum[key])
        rng.shuffle(members)
        for j, idx in enumerate(members):
            folds[(j + offset) % k].append(int(idx))
    return [sorted(f) for f in folds]


def _carve_validation(train_idx: list[int], strata: list, val_size: float, seed: int):
    """Split the training folds into (fit, validation), stratified the same way.

    Implemented as a 1/val_size-way stratified fold of the *training* indices with the
    first fold taken as validation, so the class and split balance of the validation set
    matches the fit set rather than being whatever a uniform draw happened to give.
    """
    if val_size <= 0 or len(train_idx) < 4:
        return train_idx, []
    k_val = max(2, int(round(1.0 / val_size)))
    sub = _stratified_folds([strata[i] for i in train_idx], k_val, seed + 1)
    val = sorted(train_idx[j] for j in sub[0])
    val_set = set(val)
    return [i for i in train_idx if i not in val_set], val


# --------------------------------------------------------------------------- #
# one fold
# --------------------------------------------------------------------------- #
def _fit(dataset, fit_idx, val_idx, probe_spec, probe, seed: int):
    from tuberlens.probes.probe_factory import ProbeFactory

    from agentic_redteam.evaluation import seed_everything

    # Re-seed immediately before the fit so the weight init depends only on `seed` and
    # the data, not on how far the global RNG advanced through earlier folds. Without
    # this, fold 4's probe differs from fold 0's for reasons that have nothing to do
    # with the data, and the fold spread stops being a data property.
    seed_everything(seed)
    return ProbeFactory.build(
        probe_spec=probe_spec,
        train_dataset=dataset[fit_idx],
        model_name=probe.model_name,
        layer=int(probe.layer),
        validation_dataset=dataset[val_idx] if val_idx else None,
        use_store=False,
        pos_class_label=probe.pos_class_label,
        neg_class_label=probe.neg_class_label,
        probe_description=getattr(probe, "description", None),
    )


def _score(fitted, dataset, idx: list[int], split_of: list[str]) -> list[dict]:
    """Metrics on the held-out fold: one row per source split plus an "all" rollup.

    A per-split row is emitted only when that split contributes both classes to this
    fold — AUROC is undefined on a single-class slice, and reporting accuracy alone
    there would silently mix a different metric into the mean.
    """
    from tuberlens.evaluation import calculate_metrics

    held = dataset[idx]
    y_pred = np.asarray(fitted.predict_proba(held))
    y_true = np.array([lab.to_int() for lab in held.labels])
    splits = [split_of[i] for i in idx]

    rows = [calculate_metrics(y_true, y_pred, fpr=FPR) | {"scored": "all", "n": len(idx)}]
    for name in sorted(set(splits)):
        m = np.array([s == name for s in splits])
        if len(set(y_true[m].tolist())) < 2:
            continue
        rows.append(
            calculate_metrics(y_true[m], y_pred[m], fpr=FPR)
            | {"scored": name, "n": int(m.sum())}
        )
    return rows


# --------------------------------------------------------------------------- #
# geometries
# --------------------------------------------------------------------------- #
def _fold_csv(work_dir: Path, label: str, fold: int) -> Path:
    """Per-fold result path. ``:`` is not portable in a filename, so it is rewritten."""
    return work_dir / "folds" / f"{label.replace(':', '__')}__f{fold}.csv"


def _backfill_folds(work_dir: Path) -> int:
    """Split a pre-existing ``kfold_cv_folds.csv`` into the per-fold files.

    Only for runs finished by the version of this script that wrote its results once, at
    the end. Those runs left no ``folds/`` directory, so without this a re-run would
    recompute all 25 folds even though every answer is already on disk — which would make
    picking up the per-fold change cost exactly the compute the change exists to protect.

    Conservative in both directions: it does nothing if ``folds/`` already holds anything
    (a newer run is authoritative over an older summary), and it never overwrites a file
    that exists. Returns the number of files written.
    """
    import pandas as pd

    combined = work_dir / "kfold_cv_folds.csv"
    folds_dir = work_dir / "folds"
    if not combined.is_file():
        return 0
    if folds_dir.is_dir() and any(folds_dir.glob("*.csv")):
        return 0

    df = pd.read_csv(combined)
    if not {"geometry", "fold"} <= set(df.columns):
        return 0
    written = 0
    for (label, fold), group in df.groupby(["geometry", "fold"]):
        path = _fold_csv(work_dir, str(label), int(fold))
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        group.to_csv(path, index=False)
        written += 1
    if written:
        print(f"backfilled {written} per-fold file(s) from {combined.name} — "
              f"these folds will be reused, not recomputed")
    return written


def _run_geometry(
    label: str, dataset, split_of: list[str], args, probe, probe_spec
) -> list[dict]:
    """k-fold over one corpus. ``split_of`` names each row's source split.

    Each fold's rows are written to their own CSV **as the fold finishes**, and a fold
    whose CSV already exists is skipped. Three things follow, all of which the first
    cloud run wanted and did not have: a 40-minute stage leaves a trace on disk while it
    runs (the failsafe committer has something to checkpoint, and `ls` answers "is it
    working?"), an interrupted stage resumes instead of restarting, and the summary at
    the end is assembled from the same files rather than from memory alone.
    """
    import time

    import pandas as pd

    strata = [f"{s}|{lab.to_int()}" for s, lab in zip(split_of, dataset.labels)]
    folds = _stratified_folds(strata, args.k, args.seed)
    out: list[dict] = []
    for f, test_idx in enumerate(folds):
        csv = _fold_csv(args.work_dir, label, f)
        if csv.exists():
            done = pd.read_csv(csv).to_dict("records")
            out += done
            roll = next(r for r in done if r["scored"] == "all")
            print(f"  {label:28s} fold {f + 1}/{args.k}  already done "
                  f"(AUROC={roll['auroc']:.4f})", flush=True)
            continue

        t0 = time.time()
        test_set = set(test_idx)
        train_idx = [i for i in range(len(dataset)) if i not in test_set]
        fit_idx, val_idx = _carve_validation(train_idx, strata, args.val_size, args.seed)
        fitted = _fit(dataset, fit_idx, val_idx, probe_spec, probe, args.seed)
        scored = [
            row | {"geometry": label, "fold": f,
                   "n_fit": len(fit_idx), "n_val": len(val_idx)}
            for row in _score(fitted, dataset, test_idx, split_of)
        ]
        csv.parent.mkdir(parents=True, exist_ok=True)
        # Written via a temp file + rename so a crash mid-write cannot leave a partial
        # CSV that the resume path above would then trust and skip.
        tmp = csv.with_suffix(".csv.tmp")
        pd.DataFrame(scored).to_csv(tmp, index=False)
        tmp.replace(csv)

        out += scored
        roll = scored[0]  # _score always emits the "all" rollup first
        print(
            f"  {label:28s} fold {f + 1}/{args.k}  fit={len(fit_idx):4d} "
            f"val={len(val_idx):4d} test={len(test_idx):4d}  "
            f"AUROC={roll['auroc']:.4f} acc={roll['accuracy']:.4f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    ap.add_argument("--eval-cache", type=Path, default=DEFAULT_EVAL_CACHE)
    ap.add_argument("--probe", type=Path, default=DEFAULT_PROBE,
                    help="Probe to inherit model/layer/labels/architecture from "
                         "(metadata only; fresh probes are fitted)")
    ap.add_argument("--k", type=int, default=5, help="Number of folds (default 5)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seeds both the fold assignment and the weight init")
    ap.add_argument("--val-size", type=float, default=0.2,
                    help="Fraction of the TRAINING folds held out for early stopping "
                         "(0 = fit with no validation set)")
    ap.add_argument("--geometries", nargs="+", default=["pooled", "within"],
                    choices=["pooled", "within"])
    ap.add_argument("--arch", default=None,
                    help="Override the architecture (a ProbeType name). Default: "
                         "inherit the probe's, so the ceiling is the ceiling for the "
                         "architecture the pipeline actually uses")
    ap.add_argument("--fetch-kaggle", action="store_true",
                    help="Fill --eval-cache from Kaggle before running")
    args = ap.parse_args(argv)
    # Before anything runs, not after: the per-fold CSVs are written as folds finish.
    args.work_dir.mkdir(parents=True, exist_ok=True)
    _backfill_folds(args.work_dir)

    if args.fetch_kaggle:
        import subprocess

        cmd = [sys.executable,
               str(REPO_ROOT / "scripts" / "fetch_kaggle_eval_activations.py"),
               "--concept", "hu_harm", "--cache-dir", str(args.eval_cache)]
        print(">>> " + " ".join(cmd), flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc

    from agentic_redteam.retrain import (
        _coerce_probe_spec,
        _concatenate_consuming,
        _infer_probe_spec,
    )

    probe = _load_probe(args.probe)
    probe_spec = (
        _coerce_probe_spec(args.arch) if args.arch else _infer_probe_spec(probe)
    )
    names = sorted(p.stem for p in args.eval_dir.glob("*.jsonl"))
    if not names:
        raise SystemExit(f"no eval split JSONLs in {args.eval_dir}")

    print(f"probe      : {probe.model_name} L{probe.layer} "
          f"({probe.pos_class_label} / {probe.neg_class_label})")
    print(f"arch       : {probe_spec.name.value if hasattr(probe_spec.name, 'value') else probe_spec.name}")
    print(f"splits     : {', '.join(names)}")
    print(f"folds      : k={args.k}, seed={args.seed}, val_size={args.val_size}")
    print()

    per_split = {n: _load_split(n, args.eval_dir, args.eval_cache, probe) for n in names}
    for n, d in per_split.items():
        pos = sum(lab.to_int() for lab in d.labels)
        print(f"  {n:24s} {len(d):4d} rows ({pos} positive / {len(d) - pos} negative)")
    print()

    rows: list[dict] = []
    # `within` runs FIRST because `pooled` consumes the per-split datasets — see below.
    if "within" in args.geometries:
        for n in names:
            d = per_split[n]
            rows += _run_geometry(f"within:{n}", d, [n] * len(d), args, probe, probe_spec)

    if "pooled" in args.geometries:
        # ~866 rows padded to the 1024-token cap at hidden 5376 / fp16 is ~9.5 GB of
        # activations. tuberlens' LabelledDataset.concatenate pads every part and *then*
        # torch.cats, holding inputs and output at once (~2x, ~19 GB); this repo's
        # _concatenate_consuming is byte-identical in output but fills the block slice by
        # slice, popping each part's fields as it copies, so the peak stays at ~1x. It
        # CONSUMES its inputs, which is why `within` had to run first and why split_of is
        # computed before the call.
        split_of = [n for n in names for _ in range(len(per_split[n]))]
        pooled = _concatenate_consuming([per_split[n] for n in names])
        per_split.clear()
        print(f"pooled corpus: {len(pooled)} rows")
        rows += _run_geometry("pooled", pooled, split_of, args, probe, probe_spec)
        del pooled

    df = pd.DataFrame(rows)[
        ["geometry", "fold", "scored", "n", "n_fit", "n_val",
         "auroc", "accuracy", "tpr_at_fpr", "fpr"]
    ]
    df.to_csv(args.work_dir / "kfold_cv_folds.csv", index=False)

    summary = (
        df.groupby(["geometry", "scored"])[["auroc", "accuracy", "tpr_at_fpr"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index().merge(
        df.groupby(["geometry", "scored"])["n"].sum().rename("n_total").reset_index(),
        on=["geometry", "scored"],
    )
    summary.to_csv(args.work_dir / "kfold_cv_summary.csv", index=False)

    print("\n" + "=" * 78)
    print(f"{args.k}-fold CV on the human-harm eval set (train and test both eval)")
    print("=" * 78)
    print(summary.to_string(index=False))
    print(f"\nwrote {args.work_dir / 'kfold_cv_folds.csv'}")
    print(f"wrote {args.work_dir / 'kfold_cv_summary.csv'}")
    (args.work_dir / "kfold_cv_done.json").write_text(
        json.dumps({"k": args.k, "seed": args.seed, "splits": names}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
