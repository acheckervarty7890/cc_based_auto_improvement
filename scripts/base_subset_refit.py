#!/usr/bin/env python
"""Refit on a SUBSET of the base rows chosen by the leave-one-out study, with controls.

`base_selection_study.py` ranks each base row by how the probe scores without it. This
takes the top-k of that ranking and refits.

**The ranking was computed on the eval split, so the top-k arm is selection on the test
set and its number is optimistically biased** -- it is not a held-out estimate of anything.
That is why the controls are not optional: `random` (a seeded draw of the same size) says
what k rows are worth at all, and `bottom` (the k least helpful) says whether the ranking
carries any signal beyond its own noise. Only the gaps between the three arms mean
anything; the top-k number on its own does not.

Class balance is reported per arm, since a selection made without regard to it can skew a
50/50 set and skew alone would move the score.

**Gradient accumulation has to be overridden here, and that is not a detail.**
`PytorchAdamClassifier.train` steps the optimizer only when
`(batch_idx + 1) % gradient_accumulation_steps == 0`, with no flush at the end of an epoch.
At the probe's own hyperparameters (batch_size 16, gradient_accumulation_steps 4) a training
set of fewer than 49 rows produces at most 3 batches, so the condition never fires and the
optimizer **never steps** -- the fit runs its epochs, early-stops on a frozen validation
AUROC, and returns the randomly initialized head. It is silent: no error, no warning, and a
plausible-looking AUROC near chance. A first version of this script hit exactly that, and
all three 30-row arms returned bit-identical untrained probes.

So every arm here runs at `--grad-accum 1`, INCLUDING the 50-row reference, since changing
the optimizer's step count would otherwise be confounded with changing the data. The
per-arm steps/epoch is printed, and an arm that would take zero steps refuses to run.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    RES = REPO / "results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
    ap.add_argument("--loo", type=Path, default=RES / "base_selection_oig_omission_base_only.json")
    ap.add_argument("-k", type=int, default=30)
    ap.add_argument("--base-probe", type=Path,
                    default=REPO / "probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl")
    ap.add_argument("--base-training-data", type=Path,
                    default=REPO / "data/instructions_llama70b_50.jsonl")
    ap.add_argument("--dev-data", type=Path, default=REPO / "dev_samples/instructions")
    ap.add_argument("--eval-dataset-dir", type=Path, default=REPO / "eval_sets/instructions")
    ap.add_argument("--base-activation-cache-dir", type=Path,
                    default=REPO / "results_instructions_gemma27b_shared/base_activations")
    ap.add_argument("--activations-cache-dir", type=Path,
                    default=REPO / "results_instructions_gemma27b_shared/eval_activations")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="gradient_accumulation_steps override; the probe's own 4 silently "
                         "disables training below 49 rows (see the module docstring)")
    ap.add_argument("--ensemble-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import torch
    from agentic_redteam.ensemble import EnsembleProbe, fusion_enabled
    from agentic_redteam.evaluation import evaluate_probe, seed_everything
    from agentic_redteam.retrain import (
        _base_activation_cache_paths, _cpu_unpickle, _dev_activation_cache_path,
        _infer_probe_spec, _load_dev_dataset, _resolve_ensemble_seeds, _to_device_for_fit,
        stable_train_test_split,
    )
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.interfaces.probes import ProbeSpec
    from tuberlens.model import LLMModel
    from tuberlens.probes.probe_factory import ProbeFactory

    out = args.out or (args.loo.parent / f"base_subset_refit_k{args.k}.json")
    with args.base_probe.open("rb") as f:
        bp = _cpu_unpickle(f)
    pos, neg = bp.pos_class_label, bp.neg_class_label
    spec = _infer_probe_spec(bp)
    hp = dict(spec.hyperparams)
    print(f"probe hyperparameters: batch_size {hp.get('batch_size')}, "
          f"gradient_accumulation_steps {hp.get('gradient_accumulation_steps')} "
          f"-> overridden to {args.grad_accum} for every arm")
    spec = ProbeSpec(name=spec.name,
                     hyperparams=hp | {"gradient_accumulation_steps": args.grad_accum})
    BATCH = int(hp.get("batch_size", 16))
    seeds = _resolve_ensemble_seeds(args.seed, args.ensemble_size)
    TS = 0.0; C = V = True

    base = LabelledDataset.load_from(
        Path(args.base_training_data), pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    btr, _ = stable_train_test_split(base, test_size=TS, split_field=None, seed=args.seed)
    btc, _ = _base_activation_cache_paths(
        args.base_activation_cache_dir, args.base_training_data, bp.model_name, bp.layer,
        args.seed, TS, None, C, V, 1.0)
    a = LLMModel.load_activations(btc)
    BASE = btr.assign(activations=a.activations, attention_mask=a.attention_mask,
                      input_ids=a.input_ids)
    names = {"positive": pos, "negative": neg}
    labels = [names.get(l.value, l.value) for l in BASE.labels]

    dv, dfiles, _sz = _load_dev_dataset(args.dev_data, pos, neg, C, V, verbose=False)
    da = LLMModel.load_activations(
        _dev_activation_cache_path(args.base_activation_cache_dir, dfiles, bp.model_name,
                                   bp.layer, C, V))
    VAL = dv.assign(activations=da.activations, attention_mask=da.attention_mask,
                    input_ids=da.input_ids)
    _to_device_for_fit([VAL], verbose=True)

    loo = json.load(args.loo.open())
    # delta > 0 means the row HURTS, so ascending delta is most-helpful first.
    order = [r["i"] for r in sorted(loo["rows"], key=lambda r: r["delta"])]
    rng = random.Random(args.seed)
    rand = sorted(rng.sample(range(len(BASE)), args.k))
    arms = {
        "all50": list(range(len(BASE))),
        f"top{args.k}": sorted(order[:args.k]),
        f"random{args.k}": rand,
        f"bottom{args.k}": sorted(order[-args.k:]),
    }

    def fit(idx, tag):
        p_n = sum(1 for i in idx if labels[i] == pos)
        n_batches = -(-len(idx) // BATCH)
        steps = n_batches // args.grad_accum
        if steps == 0:
            raise SystemExit(
                f"{tag}: {len(idx)} rows -> {n_batches} batches at batch_size {BATCH}, which is "
                f"fewer than --grad-accum {args.grad_accum}: the optimizer would never step and "
                f"the fit would silently return an untrained probe. Lower --grad-accum.")
        print(f"\n=== {tag}: {len(idx)} rows ({p_n} {pos.split('_')[0]}+ / {len(idx)-p_n} neg), "
              f"{n_batches} batches, {steps} optimizer steps/epoch ===", flush=True)
        train = BASE[list(idx)]
        _to_device_for_fit([train], verbose=False)
        seed_everything(seeds[0])
        be = getattr(ProbeFactory, "build_ensemble", None)
        if be is not None and fusion_enabled():
            members = be(probe_spec=spec, train_dataset=train, model_name=bp.model_name,
                         layer=bp.layer, seeds=list(seeds), validation_dataset=VAL,
                         pos_class_label=pos, neg_class_label=neg,
                         probe_description=bp.description, verbose=False)
        else:
            members = []
            for s in seeds:
                seed_everything(s)
                members.append(ProbeFactory.build(
                    probe_spec=spec, train_dataset=train, model_name=bp.model_name,
                    layer=bp.layer, validation_dataset=VAL, use_store=False,
                    pos_class_label=pos, neg_class_label=neg,
                    probe_description=bp.description))
        probe = EnsembleProbe.from_members(members, list(seeds))
        q = args.loo.parent / f"probe_basesubset_{tag}.pkl"
        with q.open("wb") as f:
            pickle.dump(probe, f)
        df = evaluate_probe(probe_path=q, eval_dataset_dir=args.eval_dataset_dir,
                            activations_cache_dir=args.activations_cache_dir, splits=None,
                            max_samples=None, seed=args.seed,
                            combine_consecutive_messages=C, convert_tool_to_assistant=V)
        df.to_csv(args.loo.parent / f"eval_basesubset_{tag}.csv", index=False)
        print(df.to_string(index=False), flush=True)
        del train, members, probe
        torch.cuda.empty_cache()
        return {r["dataset"]: round(float(r["auroc"]), 4) for _, r in df.iterrows()} | {
            "n_rows": len(idx), "n_pos": p_n}

    res = {t: fit(idx, t) for t, idx in arms.items()}
    out.write_text(json.dumps({"k": args.k, "seed": args.seed, "auroc": res}, indent=1))
    print("\n===== BASE SUBSET REFIT (AUROC) =====")
    keys = [s for s in res["all50"] if s not in ("mean", "n_rows", "n_pos")] + ["mean"]
    w = max(len(s) for s in keys)
    print(" " * (w + 2) + "".join(f"{t:>12}" for t in res))
    for s in keys:
        print(f"{s:<{w}}  " + "".join(f"{res[t][s]:>12.4f}" for t in res))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
