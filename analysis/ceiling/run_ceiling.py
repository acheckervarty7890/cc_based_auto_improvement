"""PART 1 — how high can this probe go on eval_sets/instructions?

Everything the red-teaming loop optimises is measured on the 7 instruction-following
eval splits, so "the ceiling" is: what AUROC does a gemma-3-27b L32
`linear_then_softmax` probe reach when it is allowed to train on eval-distribution
data instead of having to generalise to it?

Four reference points, all scored with tuberlens' own metric (per-split AUROC /
accuracy / TPR@1%FPR, macro-averaged over the 7 splits — the "mean" row of the
comparison CSVs):

  redteam_only  the run's own probe, refit here from base + the iteration-5 red-team
                snapshot. Sanity anchor: should land on the published iter5 numbers.
  cv_eval       5-fold CV *inside* the eval set: train on 4/5 of eval, predict the
                held-out fifth, pool the out-of-fold scores. The honest ceiling —
                what the architecture reaches given in-distribution labels and no
                test leakage.
  cv_eval_rt    same folds, but the training set also carries base + red-team. Shows
                whether red-team data still adds anything once real eval-distribution
                labels are available.
  oracle        fit on all 1302 eval rows and score those same rows. NOT a ceiling
                you could ever reach — it is the separability bound of the layer-32
                representation under this head, i.e. how much signal is there at all.

Early stopping in every condition uses the fixed 436-row dev set, which is disjoint
from eval, so no condition selects its checkpoint on its own test rows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import folds as F
import harness as H

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def fit_ensemble(train, val, n_members, tag, log):
    members, seeds = [], list(H.ENSEMBLE_SEEDS[:n_members])
    for i, s in enumerate(seeds):
        t0 = time.time()
        with H.Quiet() as q:
            members.append(H.fit_member(train, val, s))
        log(f"    [{tag}] member {i+1}/{n_members} seed={s} "
            f"{q.epochs_run()} epochs in {time.time()-t0:.0f}s")
    return H.ensemble_of(members, seeds), members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gptoss", choices=list(H.ARMS))
    ap.add_argument("--ensemble", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--conditions", default="redteam_only,cv_eval,cv_eval_rt,oracle")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    H.silence_tqdm()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else OUT / f"ceiling_{args.arm}.jsonl"
    log_path = out_path.with_suffix(".log")
    logf = log_path.open("a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    conditions = args.conditions.split(",")
    log(f"=== ceiling analysis | arm={args.arm} ensemble={args.ensemble} "
        f"folds={args.folds} conditions={conditions} ===")

    t0 = time.time()
    ev = H.load_eval_splits()
    dev = H.load_dev()
    base = H.load_base()
    rt = H.load_redteam(args.arm)
    log(f"loaded eval={sum(len(d) for d in ev.values())} dev={len(dev)} "
        f"base={len(base)} redteam={len(rt)} in {time.time()-t0:.0f}s")

    # base ∪ red-team, exactly the training set that produced probe_iter5.
    rt_train_cpu = H.concat(base, rt)
    del base, rt

    # Flatten eval into one indexable pool, remembering each row's split so the
    # per-split metric can be reassembled from pooled out-of-fold predictions.
    split_names = list(ev)
    row_split, row_local = [], []
    for name in split_names:
        row_split += [name] * len(ev[name])
        row_local += list(range(len(ev[name])))
    eval_pool = H.concat(*[ev[n] for n in split_names])
    y_pool = H.labels_of(eval_pool)
    row_split = np.array(row_split)
    log(f"eval pool: {len(eval_pool)} rows, seq={eval_pool.other_fields['activations'].shape[1]}")

    # Reload the per-split datasets: H.concat consumed the ones above.
    ev = H.load_eval_splits()

    strata = [f"{s}|{y}" for s, y in zip(row_split, y_pool)]
    fold_of = F.stratified_partition(strata, args.folds, seed=H.SEED)

    H.stage(dev, verbose=True)  # the validation set for every fit; park it once
    # NOTE the eval splits are deliberately NOT staged here: cv_eval_rt's training
    # set is ~16 GiB on its own, and holding eval's 4.9 GiB alongside it would push
    # past the card, at which point _to_device_for_fit silently leaves the TRAINING
    # set on the host and every fit gets ~100x slower. Scoring from the host costs
    # about a minute per full-eval pass, and the CV folds only score 1/5 of it.
    log(f"dev staged; {H.gpu_free_gib():.1f} GiB allocatable")

    def emit(rec):
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        per = rec["per_split"]
        log(f">>> {rec['condition']:14s} macro AUROC {rec['macro']['auroc']:.4f}  "
            f"acc {rec['macro']['accuracy']:.4f}  TPR@1%FPR {rec['macro']['tpr_at_fpr']:.4f} (<=1%: {rec['macro']['tpr_at_fpr_le']:.4f})")
        for k in split_names:
            log(f"      {k:32s} auroc {per[k]['auroc']:.4f}  tpr {per[k]['tpr_at_fpr']:.4f}")

    # ---------------------------------------------------------------- redteam_only
    if "redteam_only" in conditions:
        train = H.take(rt_train_cpu, range(len(rt_train_cpu)))
        H.stage(train, verbose=True)
        probe, _ = fit_ensemble(train, dev, args.ensemble, "redteam_only", log)
        per = H.score_splits(probe, ev)
        emit({"condition": "redteam_only", "arm": args.arm, "n_train": len(train),
              "per_split": per, "macro": H.macro(per)})
        del train, probe
        H.free()

    # ---------------------------------------------------------------- CV conditions
    for cond, with_rt in (("cv_eval", False), ("cv_eval_rt", True)):
        if cond not in conditions:
            continue
        oof = np.zeros(len(eval_pool))
        for k in range(args.folds):
            tr_idx = np.where(fold_of != k)[0]
            te_idx = np.where(fold_of == k)[0]
            parts = [H.take(eval_pool, tr_idx)]
            if with_rt:
                parts.append(H.take(rt_train_cpu, range(len(rt_train_cpu))))
            train = H.concat(*parts)
            H.stage(train, verbose=True)
            log(f"  [{cond}] fold {k+1}/{args.folds}: train={len(train)} test={len(te_idx)}"
                f" | {H.gpu_free_gib():.1f} GiB allocatable")
            probe, _ = fit_ensemble(train, dev, args.ensemble, f"{cond}/f{k}", log)
            held = H.take(eval_pool, te_idx)
            oof[te_idx] = np.asarray(probe.predict_proba(held))
            del train, probe, held
            H.free()
        per = {n: H.split_metrics(y_pool[row_split == n], oof[row_split == n])
               for n in split_names}
        emit({"condition": cond, "arm": args.arm, "folds": args.folds,
              "per_split": per, "macro": H.macro(per)})

    # ---------------------------------------------------------------- oracle
    if "oracle" in conditions:
        train = H.take(eval_pool, range(len(eval_pool)))
        H.stage(train, verbose=True)
        probe, _ = fit_ensemble(train, dev, args.ensemble, "oracle", log)
        per = H.score_splits(probe, ev)
        emit({"condition": "oracle", "arm": args.arm, "n_train": len(train),
              "per_split": per, "macro": H.macro(per)})
        del train, probe
        H.free()

    log(f"done in {(time.time()-t0)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
