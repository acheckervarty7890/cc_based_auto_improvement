#!/usr/bin/env python
"""How many in-distribution (dev) samples does red-team training data need to reach the ceiling?

For each concept, a **single** probe (never an ensemble) is trained on

    base training data  ∪  red-team successes  ∪  N dev samples

for 10 equidistant values of N from 0 to the size of the dev training pool, and scored on
the concept's eval splits. Two ways of adding the dev samples are compared:

  mixed     one fit over the union.
  finetune  fit on base ∪ red-team first, then continue training that same head on the N
            dev samples (tuberlens' own `initialize_model=False` hook).
  dev_only  supplementary control: the N dev samples alone, no red-team data — this is what
            says whether the curve is about the dev samples or about the combination.

Fixed across every point, every arm and the ceiling CV: the validation set (a reserved 25%
stratified slice of dev, never trained on), the architecture (`linear_then_softmax` with
tuberlens' default hyperparameters) and the fit seed. What moves is the training data.

The dev subsets are **nested** within a draw seed — the N=k subset is a prefix of the
N=k+1 subset — so each curve is a learning curve rather than 10 unrelated draws, and they
are stratified by (label, dev split) so composition does not drift with N. `--dev-seeds`
repeats the whole sweep with different draws; the spread across seeds is the error bar.

Results are appended to `results/sweep_<concept>.jsonl`, one row per (arm, seed, N), and a
re-run skips rows already there.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca_common as C  # noqa: E402

KEY_FIELDS = ("concept", "arm", "dev_seed", "n_dev")


def stratified_order(ds, keys, rng) -> list[int]:
    """A row order whose every prefix keeps the (label, key) composition of `ds`.

    Each group is shuffled, then every element is given the rank `(pos + 0.5) / len(group)`
    and all elements are sorted by it. A prefix of length N therefore holds ~N * |g| / |ds|
    of each group g, which is what makes the nested subsets comparable across N.
    """
    groups = C.stratified_indices(ds, keys)
    ranked = []
    for key in sorted(groups):
        members = list(rng.permutation(groups[key]))
        for pos, i in enumerate(members):
            ranked.append(((pos + 0.5) / len(members), str(key), int(i)))
    ranked.sort()
    return [i for _, _, i in ranked]


def free_gpu():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate(probe, eval_splits) -> dict:
    """Score every eval split, moving one split at a time onto the card."""
    from tuberlens.evaluation import calculate_metrics

    per_split = {}
    for name, ds in eval_splits.items():
        [ds_d], _ = C.to_device([ds])
        preds = np.asarray(probe.predict_proba(ds_d))
        per_split[name] = calculate_metrics(C.labels_array(ds), preds, fpr=0.01)
        del ds_d
        free_gpu()
    mean = {m: float(np.mean([v[m] for v in per_split.values()]))
            for m in ("auroc", "accuracy", "tpr_at_fpr")}
    return {"per_split": per_split, "mean": mean}


def run_concept(concept: C.Concept, args) -> None:
    log_path = C.RESULTS / f"sweep_{concept.name}.jsonl"
    done = C.done_keys(log_path, KEY_FIELDS) if args.resume else set()

    print(f"[{concept.name}] loading ...", flush=True)
    eval_splits = {k: C.trim(v) for k, v in C.load_eval_splits(concept).items()}
    dev_val, dev_pool, dev_keys = C.dev_partition(concept)
    redteam = C.load_redteam(concept)
    base = C.load_base(concept)
    rt_base = C.trim(C.pool([base, redteam]))
    points = C.sweep_points(len(dev_pool), args.n_points)
    print(f"[{concept.name}] red-team+base {len(rt_base)} rows "
          f"({C.nbytes(rt_base)/1e9:.2f} GB), dev pool {len(dev_pool)}, "
          f"validation {len(dev_val)}", flush=True)
    print(f"[{concept.name}] points: {points}", flush=True)

    [val_d], _ = C.to_device([dev_val])

    # Stage 1 of the finetune arm — and the N=0 point of the mixed arm — is the same fit:
    # base + red-team only. It depends on nothing that varies across points or draw seeds,
    # so it is fit once and deep-copied per finetune point.
    stage1 = None

    def get_stage1():
        nonlocal stage1
        if stage1 is None:
            t0 = time.time()
            [tr_d], gpu = C.to_device([rt_base])
            print(f"[{concept.name}] fitting stage 1 (red-team only, gpu={gpu}) ...",
                  flush=True)
            stage1 = C.fit_probe(tr_d, val_d, concept, seed=C.FIT_SEED)
            print(f"[{concept.name}] stage 1 fit in {time.time()-t0:.0f}s "
                  f"(val AUROC {C.val_auroc(stage1, val_d):.4f})", flush=True)
            del tr_d
            free_gpu()
        return stage1

    for dev_seed in args.dev_seeds:
        rng = np.random.default_rng(10_000 + dev_seed)
        order = stratified_order(dev_pool, dev_keys, rng)
        for n_dev in points:
            idx = sorted(order[:n_dev])
            dev_sub = dev_pool[idx] if n_dev else None
            for arm in args.arms:
                key = (concept.name, arm, dev_seed, n_dev)
                if key in done:
                    continue
                if arm == "dev_only" and n_dev == 0:
                    continue
                # every draw seed produces the same N=0 point; fit it once
                if n_dev == 0 and dev_seed != args.dev_seeds[0]:
                    continue
                t0 = time.time()
                extra = {}
                if arm == "mixed":
                    train = C.trim(C.pool([rt_base, dev_sub])) if dev_sub is not None else rt_base
                    [tr_d], gpu = C.to_device([train])
                    probe = C.fit_probe(tr_d, val_d, concept, seed=C.FIT_SEED)
                elif arm == "dev_only":
                    train = C.trim(dev_sub)
                    [tr_d], gpu = C.to_device([train])
                    probe = C.fit_probe(tr_d, val_d, concept, seed=C.FIT_SEED)
                elif arm == "finetune":
                    base_probe = get_stage1()
                    if n_dev == 0:
                        probe, extra = base_probe, {"checkpoint_kept": "stage1"}
                        train, tr_d, gpu = rt_base, None, None
                    else:
                        probe = copy.deepcopy(base_probe)
                        train = C.trim(dev_sub)
                        [tr_d], gpu = C.to_device([train])
                        probe, extra = C.finetune_probe(probe, tr_d, val_d, seed=C.FIT_SEED)
                else:
                    raise ValueError(arm)

                metrics = evaluate(probe, eval_splits)
                row = {
                    "concept": concept.name,
                    "arm": arm,
                    "dev_seed": dev_seed,
                    "n_dev": n_dev,
                    "n_train": len(train),
                    "n_val": len(dev_val),
                    "val_auroc": C.val_auroc(probe, val_d),
                    "best_epoch": getattr(probe._classifier, "best_epoch", None),
                    "fit_seconds": round(time.time() - t0, 1),
                    "mean": metrics["mean"],
                    "per_split": metrics["per_split"],
                    **extra,
                }
                C.append_jsonl(log_path, row)
                print(f"[{concept.name}] {arm} seed={dev_seed} N={n_dev}: "
                      f"eval AUROC {metrics['mean']['auroc']:.4f} "
                      f"val {row['val_auroc']:.4f} ({row['fit_seconds']}s)", flush=True)
                if arm != "finetune" or n_dev:
                    del probe
                del tr_d
                free_gpu()
            del dev_sub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=list(C.CONCEPTS))
    ap.add_argument("--arms", nargs="*", default=["mixed", "finetune", "dev_only"])
    ap.add_argument("--dev-seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    for name in args.concepts:
        run_concept(C.CONCEPTS[name], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
