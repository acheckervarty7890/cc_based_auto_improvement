#!/usr/bin/env python
"""Do different-layer probes get the SAME samples wrong, or different ones?

`scripts/multilayer_refit.py` reports one AUROC per arm. Two probes at the same AUROC can
still be failing on disjoint halves of a split, and that difference is the whole question
behind stacking layers: if their errors are complementary there is something to combine,
and if they are the same errors there is not.

So this refits a set of arms under `multilayer_refit`'s exact protocol -- same base(50) +
couples(66), same 404-row validation set, same repo-pinned ENSEMBLE_SEEDS, sequential fits
-- and dumps **per-sample** probabilities on one eval split, then compares the arms row by
row.

Three things are reported, and they are not interchangeable:

* **Union coverage.** The fraction of rows classified correctly by AT LEAST ONE arm. This
  is an ORACLE: it assumes a selector that already knows which probe to trust on each row,
  which is exactly the thing you do not have. It is an upper bound on what any combination
  of these probes could achieve, not an achievable accuracy. It is reported because the gap
  between it and the best single arm is the measure of how complementary the errors are.
* **Majority vote** (and **mean-probability**), which ARE achievable, so they say what
  combining actually buys.
* **Mean score on correctly classified rows.** Raw mean probability over correct rows mixes
  the two classes -- a correct negative scores near 0 and a correct positive near 1, so
  their mean says more about class balance than about confidence. All three are printed:
  the mean over correct positives, over correct negatives, and the **margin**
  (`p` if the row is positive else `1 - p`), which is the one comparable across classes.

Classification is `p > 0.5`, matching tuberlens' own accuracy metric
(`evaluation.calculate_metrics`), so "correct" here means what it means everywhere else in
this repo.

    .venv_claude/bin/python scripts/probe_agreement.py --arm 32 --arm 40 --arm 48
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[], metavar="A[,B]",
                    help="Repeatable. A layer or comma-separated layer group, naming the "
                         "same arm multilayer_refit would fit.")
    ap.add_argument("--split", default="oig_omission")
    ap.add_argument("--eval-dir", type=Path, default=REPO / "eval_sets/instructions")
    ap.add_argument("--dev-dir", type=Path, default=REPO / "dev_samples/instructions")
    ap.add_argument("--acts-root", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/activations")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/probe_agreement.json")
    ap.add_argument("--threshold-source", choices=["fixed", "heldout_dev"], default="fixed",
                    help="fixed: classify at 0.5, tuberlens' own convention. heldout_dev: "
                         "choose each arm's threshold on the dev rows this protocol WITHHOLDS "
                         "-- the 16 oig_omission dev PAIRS that are neither trained on nor "
                         "validated on in the base+couples condition, so they are genuinely "
                         "unused and in-distribution for the split. Picking the threshold on "
                         "the eval split itself would be selection on the test set.")
    ap.add_argument("--ensemble-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    # multilayer_refit owns the blob-concat helpers and the arm protocol; importing them
    # rather than re-deriving keeps the two from drifting on what an "arm" is.
    import multilayer_refit as mr
    from agentic_redteam.ensemble import EnsembleProbe, fusion_enabled
    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.retrain import (
        _apply_message_transforms, _concatenate_consuming, _cpu_unpickle,
        _dev_lending_groups, _dev_lending_indices, _infer_probe_spec, _load_dev_dataset,
        _resolve_ensemble_seeds, _to_device_for_fit, stable_train_test_split,
    )
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage
    from tuberlens.probes.probe_factory import ProbeFactory

    groups = [[int(x) for x in a.split(",")] for a in args.arm]
    if not groups:
        raise SystemExit("pass --arm 32 --arm 40 ...")
    names = ["+".join(f"L{l}" for l in g) for g in groups]

    with mr.BP.open("rb") as f:
        bp = _cpu_unpickle(f)
    pos, neg = bp.pos_class_label, bp.neg_class_label
    spec = _infer_probe_spec(bp)
    seeds = _resolve_ensemble_seeds(args.seed, args.ensemble_size)
    C = V = True
    AR = args.acts_root
    print(f"[agree] split={args.split} arms={names} ensemble={len(seeds)} "
          f"fused={fusion_enabled()}", flush=True)

    base = LabelledDataset.load_from(
        mr.BASE_JSONL, pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    BASE_DS, _ = stable_train_test_split(base, test_size=0.0, split_field=None,
                                         seed=args.seed)
    rows = [json.loads(line) for line in mr.COUPLES_JSONL.open()]
    rt = LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]]
                for r in rows],
        ids=[r["id"] for r in rows],
        other_fields={"labels": [r["label"] for r in rows]})
    rt = _apply_message_transforms(rt, C, V)
    RT_DS, _ = stable_train_test_split(rt, test_size=0.0, split_field=None, seed=args.seed)
    dv, dfiles, dsizes = _load_dev_dataset(args.dev_dir, pos, neg, C, V, verbose=False)

    ev = LabelledDataset.load_from(
        args.eval_dir / f"{args.split}.jsonl", pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    y = np.array(ev.other_fields["labels"], dtype=int)

    P, HELD, VALS = {}, {}, {}
    for name, layers in zip(names, groups):
        train = _concatenate_consuming([
            mr._attach(BASE_DS[list(range(len(BASE_DS)))], AR, layers, "extra", "base"),
            mr._attach(RT_DS[list(range(len(RT_DS)))], AR, layers, "extra", "couples"),
        ])
        parts = []
        for f, n in zip(dfiles, dsizes):
            ds = LabelledDataset.load_from(
                f, pos_class_label=pos, neg_class_label=neg,
                combine_consecutive_messages=C, convert_tool_to_assistant=V)
            parts.append(mr._attach(ds, AR, layers, "dev_samples", f.stem))
        DEV_ALL = _concatenate_consuming(parts)
        g = _dev_lending_groups(DEV_ALL, dfiles, dsizes, mr.DEV_SPLIT, "pairs")
        lent, val_idx = _dev_lending_indices(len(DEV_ALL), mr.DEV_PAIRS, mr.DEV_PAIRS,
                                             args.seed, groups=g, verbose=False)
        VAL = DEV_ALL[val_idx]
        HELDOUT = DEV_ALL[lent] if lent else None   # never trained on, never validated on
        del DEV_ALL
        _to_device_for_fit([train, VAL], verbose=False)
        seed_everything(seeds[0])
        kw = dict(probe_spec=spec, train_dataset=train, model_name=bp.model_name,
                  layer=max(layers), validation_dataset=VAL, pos_class_label=pos,
                  neg_class_label=neg, probe_description=bp.description)
        be = getattr(ProbeFactory, "build_ensemble", None)
        if be is not None and fusion_enabled():
            members = be(seeds=list(seeds), verbose=False, **kw)
        else:
            members = []
            for s in seeds:
                seed_everything(s)
                members.append(ProbeFactory.build(use_store=False, **kw))
        probe = EnsembleProbe.from_members(members, list(seeds))
        del train
        torch.cuda.empty_cache()

        pv = np.asarray(probe.predict_proba(VAL), dtype=float)
        if pv.ndim == 2:
            pv = pv[:, -1]
        VALS[name] = (pv, np.array(VAL.other_fields["labels"], dtype=int))
        if HELDOUT is not None:
            ph = np.asarray(probe.predict_proba(HELDOUT), dtype=float)
            if ph.ndim == 2:
                ph = ph[:, -1]
            HELD[name] = (ph, np.array(HELDOUT.other_fields["labels"], dtype=int))
            del HELDOUT
        del VAL
        torch.cuda.empty_cache()
        ds = mr._attach(ev[list(range(len(ev)))], AR, layers, "eval_sets", args.split)
        p = np.asarray(probe.predict_proba(ds), dtype=float)
        if p.ndim == 2:
            p = p[:, -1]
        P[name] = p
        print(f"  {name:<10} AUROC {roc_auc_score(y, p):.4f}  "
              f"acc {(( p > 0.5).astype(int) == y).mean():.4f}", flush=True)
        del ds, probe, members
        torch.cuda.empty_cache()

    # ---- agreement ---------------------------------------------------------------------
    n = len(y)
    THR = {k: 0.5 for k in names}
    if args.threshold_source == "heldout_dev":
        for k in names:
            ph, yh = HELD[k]
            cands = np.unique(np.concatenate([[0.0], np.sort(ph), [1.0]]))
            # Ties are common because the scores saturate; take the median of the
            # accuracy-maximising thresholds rather than the first, so the cut sits in the
            # middle of the winning interval instead of on its edge.
            accs = np.array([((ph > t).astype(int) == yh).mean() for t in cands])
            THR[k] = float(np.median(cands[accs == accs.max()]))
            print(f"  {k:<10} held-out dev ({len(yh)} rows) threshold {THR[k]:.6g}  "
                  f"dev acc {((ph > THR[k]).astype(int) == yh).mean():.4f}")
    correct = {k: ((v > THR[k]).astype(int) == y) for k, v in P.items()}
    n_correct_by_row = np.sum([correct[k] for k in names], axis=0)

    out = {"split": args.split, "n_rows": int(n), "n_pos": int(y.sum()), "arms": names,
           "per_arm": {}, "overlap": {}, "combination": {},
           # Per-row scores, so the threshold can be re-examined offline. Accuracy at 0.5
           # and AUROC can disagree sharply when the scores saturate, and only the raw
           # scores show which is happening.
           # Scores are stored at FULL precision, deliberately. Rounding them (an earlier
           # version used round(x, 6)) destroys the ranking for saturating heads: the
           # linear_then_last probes put 91 of 114 eval scores below 1e-6, so six decimals
           # collapsed them all to exactly 0.0 -- 20 unique values out of 114 -- and AUROC
           # fell 0.7867 -> 0.6716 on stored scores that the fit-time metric had read
           # correctly. AUROC is rank-based, so precision loss near zero is not cosmetic.
           "labels": y.tolist(),
           "scores": {k: [float(x) for x in v] for k, v in P.items()},
           # Two label-bearing sets that are NOT the eval split, for choosing thresholds or
           # combination weights without selecting on the test set:
           #   heldout = the 32 oig_omission dev rows this protocol never trains or
           #             validates on -- clean, in-distribution, but small;
           #   val     = the 404-row validation set -- larger, but early stopping saw it,
           #             so it is mildly optimistic rather than fully independent.
           "heldout": {k: {"scores": [float(x) for x in v[0]],
                           "labels": v[1].tolist()} for k, v in HELD.items()},
           "val": {k: {"scores": [float(x) for x in v[0]],
                       "labels": v[1].tolist()} for k, v in VALS.items()}}

    print(f"\n=== {args.split}: {n} rows ({int(y.sum())} positive) ===")
    print(f"\n{'arm':<10}{'AUROC':>8}{'acc':>8}{'n_correct':>11}"
          f"{'mean p|correct pos':>20}{'mean p|correct neg':>20}{'mean margin':>13}")
    for k in names:
        p, c = P[k], correct[k]
        thr = THR[k]
        cp = c & (y == 1)
        cn = c & (y == 0)
        margin = np.where(y == 1, p, 1 - p)
        row = {"auroc": round(float(roc_auc_score(y, p)), 4), "threshold": round(thr, 6),
               "accuracy": round(float(c.mean()), 4),
               "n_correct": int(c.sum()),
               "mean_p_correct_positive": round(float(p[cp].mean()), 4) if cp.any() else None,
               "mean_p_correct_negative": round(float(p[cn].mean()), 4) if cn.any() else None,
               "mean_margin_correct": round(float(margin[c].mean()), 4),
               "mean_margin_incorrect": round(float(margin[~c].mean()), 4) if (~c).any() else None,
               "mean_p_all_correct_raw": round(float(p[c].mean()), 4)}
        out["per_arm"][k] = row
        print(f"{k:<10}{row['auroc']:>8.4f}{row['accuracy']:>8.4f}{row['n_correct']:>11}"
              f"{row['mean_p_correct_positive']:>20.4f}"
              f"{row['mean_p_correct_negative']:>20.4f}"
              f"{row['mean_margin_correct']:>13.4f}")

    print("\n--- do they get the SAME rows right? ---")
    print(f"{'# arms correct on a row':<26}" + "".join(f"{i:>8}" for i in range(len(names)+1)))
    hist = [int((n_correct_by_row == i).sum()) for i in range(len(names) + 1)]
    print(f"{'rows':<26}" + "".join(f"{h:>8}" for h in hist))
    print(f"{'% of split':<26}" + "".join(f"{100*h/n:>8.1f}" for h in hist))
    out["overlap"]["rows_by_n_arms_correct"] = hist

    print("\npairwise (rows where BOTH correct / ONLY first / ONLY second / NEITHER):")
    for a, b in itertools.combinations(names, 2):
        ca, cb = correct[a], correct[b]
        both = int((ca & cb).sum()); oa = int((ca & ~cb).sum())
        ob = int((~ca & cb).sum()); nei = int((~ca & ~cb).sum())
        jac = both / max(1, int((ca | cb).sum()))
        out["overlap"][f"{a}|{b}"] = {"both": both, f"only_{a}": oa, f"only_{b}": ob,
                                      "neither": nei, "jaccard_correct": round(jac, 4)}
        print(f"  {a:<10} vs {b:<10}  both {both:>4}  only-{a} {oa:>3}  "
              f"only-{b} {ob:>3}  neither {nei:>3}   Jaccard {jac:.3f}")

    union = int((n_correct_by_row >= 1).sum())
    inter = int((n_correct_by_row == len(names)).sum())
    best = max(int(correct[k].sum()) for k in names)
    maj = ((n_correct_by_row * 0 + np.sum([(P[k] > THR[k]).astype(int) for k in names], axis=0))
           > len(names) / 2).astype(int)
    maj_acc = float((maj == y).mean())
    meanp = np.mean([P[k] for k in names], axis=0)
    meanp_acc = float(((meanp > float(np.mean([THR[k] for k in names]))).astype(int) == y).mean())
    meanp_auroc = float(roc_auc_score(y, meanp))
    out["thresholds"] = {k: round(THR[k], 6) for k in names}
    out["threshold_source"] = args.threshold_source
    out["combination"] = {
        "union_correct_oracle": union, "union_pct_oracle": round(100 * union / n, 2),
        "intersection_correct": inter, "intersection_pct": round(100 * inter / n, 2),
        "best_single_correct": best, "best_single_pct": round(100 * best / n, 2),
        "majority_vote_accuracy": round(maj_acc, 4),
        "mean_probability_accuracy": round(meanp_acc, 4),
        "mean_probability_auroc": round(meanp_auroc, 4)}

    print(f"\n--- combining the {len(names)} probes ---")
    print(f"  union (>=1 arm correct)   {union:>4}/{n}  = {100*union/n:.2f}%   "
          f"<- ORACLE upper bound, needs a selector you do not have")
    print(f"  all {len(names)} correct           {inter:>4}/{n}  = {100*inter/n:.2f}%")
    print(f"  best single arm           {best:>4}/{n}  = {100*best/n:.2f}%")
    print(f"  majority vote  accuracy   {maj_acc:.4f}   (achievable)")
    print(f"  mean-probability accuracy {meanp_acc:.4f}  AUROC {meanp_auroc:.4f}  (achievable)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
