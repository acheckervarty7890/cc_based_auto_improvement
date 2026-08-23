#!/usr/bin/env python
"""How many in-distribution (dev) samples does red-team training data need to reach the ceiling?

For each concept a **single** probe (never an ensemble) is trained on

    base training data  ∪  red-team successes  ∪  N dev samples

for 10 equidistant values of N from 0 to the size of the dev training pool, and scored on
the concept's eval splits. Three arms:

  mixed     one fit over the union.
  finetune  fit on base ∪ red-team first, then continue training that same head on the N
            dev samples (tuberlens' own `initialize_model=False` hook).
  dev_only  supplementary control: the N dev samples alone, no red-team data. This is what
            says whether a point's performance is about the dev samples or the combination.

Fixed across every point, every arm and the ceiling CV: the validation set (a reserved 25%
stratified slice of dev, never trained on), the architecture (`linear_then_softmax` with
tuberlens' default hyperparameters) and the fit seed. What moves is the training data.

The dev subsets are **nested** within a draw seed — the N=k subset is a prefix of the
N=k+1 subset — so each curve is a learning curve rather than 10 unrelated draws, and they
are stratified by (label, dev split) so composition does not drift with N. `--dev-seeds`
repeats the sweep with different draws; the spread across seeds is the error bar.

Results are appended to `results/sweep_<concept>.jsonl`, one row per (arm, seed, N); a
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


def stratified_order(labels, keys, rng) -> list[int]:
    """A row order whose every prefix keeps the (label, key) composition.

    Each group is shuffled, every element gets the rank `(pos + 0.5) / len(group)`, and all
    elements are sorted by it. A prefix of length N then holds ~N|g|/|all| of each group g,
    which is what makes the nested subsets comparable across N.
    """
    groups = C.stratified_indices(labels, keys)
    ranked = []
    for key in sorted(groups, key=str):
        members = list(rng.permutation(groups[key]))
        for pos, i in enumerate(members):
            ranked.append(((pos + 0.5) / len(members), str(key), int(i)))
    ranked.sort()
    return [i for _, _, i in ranked]


def evaluate(probe, eval_srcs, chunk: int = 64) -> dict:
    """Score every eval split, streaming through the memory maps a chunk at a time."""
    from tuberlens.evaluation import calculate_metrics

    per_split = {}
    for name, src in eval_srcs.items():
        y = C.source_labels(src)
        preds = C.score_source(probe, src, np.arange(len(src)), chunk=chunk)
        per_split[name] = calculate_metrics(y, preds, fpr=0.01)
    C.free_gpu()
    mean = {m: float(np.mean([v[m] for v in per_split.values()]))
            for m in ("auroc", "accuracy", "tpr_at_fpr")}
    return {"per_split": per_split, "mean": mean}


def run_concept(concept: C.Concept, args) -> None:
    # An unnormalized run at the default fit seed writes the plain filename it always did;
    # anything else writes its own, so a probe-design study never overwrites the sweep it is
    # measured against.
    suffix = "" if args.norm is None else f"__norm-{args.norm}"
    if args.fit_seed is not None:
        suffix += f"__fit{args.fit_seed}"
    log_path = C.RESULTS / f"sweep_{concept.name}{suffix}.jsonl"
    done = C.done_keys(log_path, KEY_FIELDS) if args.resume else set()

    eval_srcs = C.eval_sources(concept)
    dev_src, val_idx, pool_idx = C.dev_partition(concept)
    rt_src = C.redteam_source(concept)
    rt_all = list(range(len(rt_src)))

    val_d = C.ragged_from_parts([(dev_src, val_idx)])
    n_val = len(val_idx)
    points = C.sweep_points(len(pool_idx), args.n_points)
    print(f"[{concept.name}] base+red-team {len(rt_src)} rows "
          f"(max real length {int(rt_src.lengths().max())}), dev pool {len(pool_idx)}, "
          f"validation {n_val} (packed {val_d.nbytes/1e9:.2f} GB)", flush=True)
    print(f"[{concept.name}] points: {points}", flush=True)

    dev_labels = C.source_labels(dev_src)
    dev_keys = list(dev_src.dataset.other_fields["dev_split"])

    stage1 = {"probe": None}

    def get_stage1():
        if stage1["probe"] is None:
            t0 = time.time()
            train = C.ragged_from_parts([(rt_src, rt_all)])
            print(f"[{concept.name}] stage 1 fit (base+red-team only, packed "
                  f"{train.nbytes/1e9:.2f} GB) ...", flush=True)
            stage1["probe"] = C.fit(train, val_d, concept, seed=C.FIT_SEED)
            print(f"[{concept.name}] stage 1 fit in {time.time()-t0:.0f}s "
                  f"(val AUROC {C.ragged_val_auroc(stage1['probe'], val_d):.4f})", flush=True)
            del train
            C.free_gpu()
        return stage1["probe"]

    for dev_seed in args.dev_seeds:
        rng = np.random.default_rng(10_000 + dev_seed)
        order = stratified_order(dev_labels[pool_idx], [dev_keys[i] for i in pool_idx], rng)
        for n_dev in points:
            sel = sorted(pool_idx[i] for i in order[:n_dev])
            for arm in args.arms:
                if arm == "dev_only" and n_dev == 0:
                    continue
                # N=0 does not depend on the draw seed; fit it once
                if n_dev == 0 and dev_seed != args.dev_seeds[0]:
                    continue
                key = (concept.name, arm, dev_seed, n_dev)
                if key in done:
                    continue
                t0 = time.time()
                extra: dict = {}
                train = None
                if arm == "mixed":
                    train = C.ragged_from_parts([(rt_src, rt_all), (dev_src, sel)])
                    n_train = len(train)
                    probe = C.fit(train, val_d, concept, seed=C.FIT_SEED)
                elif arm == "dev_only":
                    train = C.ragged_from_parts([(dev_src, sel)])
                    n_train = len(train)
                    probe = C.fit(train, val_d, concept, seed=C.FIT_SEED)
                elif arm == "finetune":
                    base_probe = get_stage1()
                    if n_dev == 0:
                        probe = base_probe
                        n_train = len(rt_src)
                        extra = {"checkpoint_kept": "stage1"}
                    else:
                        probe = copy.deepcopy(base_probe)
                        train = C.ragged_from_parts([(dev_src, sel)])
                        n_train = len(rt_src) + len(train)
                        probe, extra = C.finetune(probe, train, val_d, seed=C.FIT_SEED)
                else:
                    raise ValueError(arm)
                if train is not None:
                    del train
                C.free_gpu()

                metrics = evaluate(probe, eval_srcs)
                row = {
                    "concept": concept.name,
                    "norm": C.NORM,
                    "fit_seed": C.FIT_SEED,
                    "arm": arm,
                    "dev_seed": dev_seed,
                    "n_dev": n_dev,
                    "n_train": n_train,
                    "n_val": n_val,
                    "val_auroc": C.ragged_val_auroc(probe, val_d),
                    "best_epoch": getattr(probe._classifier, "best_epoch", None),
                    "seconds": round(time.time() - t0, 1),
                    "mean": metrics["mean"],
                    "per_split": metrics["per_split"],
                    **extra,
                }
                C.append_jsonl(log_path, row)
                print(f"[{concept.name}] {arm} seed={dev_seed} N={n_dev}: "
                      f"eval AUROC {metrics['mean']['auroc']:.4f}  "
                      f"val {row['val_auroc']:.4f}  ({row['seconds']}s)", flush=True)
                if not (arm == "finetune" and n_dev == 0):
                    del probe
                C.free_gpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=list(C.CONCEPTS))
    ap.add_argument("--arms", nargs="*", default=["mixed", "finetune", "dev_only"])
    ap.add_argument("--dev-seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--fit-seed", type=int, default=None,
                    help="seed for the head's init and batch order (default 42). "
                         "--dev-seeds still governs which dev rows are drawn, so varying "
                         "THIS re-fits the same data — which is the only replication the "
                         "N=0 point can have, since no dev rows are drawn there. Suffixes "
                         "the output filename.")
    ap.add_argument("--norm", default=None,
                    help="normalization step in front of the head (see ca_norm.KINDS). "
                         "Omit for the unnormalized architecture every experiment used; "
                         "passing it also suffixes the output filename.")
    args = ap.parse_args()
    if args.fit_seed is not None:
        C.FIT_SEED = args.fit_seed
    if args.norm is not None:
        import ca_norm

        if args.norm not in ca_norm.KINDS:
            ap.error(f"--norm must be one of {ca_norm.KINDS}")
        C.NORM = args.norm
    for name in args.concepts:
        run_concept(C.CONCEPTS[name], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
