#!/usr/bin/env python
"""Fit on the 33 red-team couples ALONE -- no base data -- and score every split.

Every probe in this project so far has been base data, or base data plus couples. This
removes the base entirely, so the 66 rows the red team produced are the whole training set.
It is the cleanest statement of what those couples teach on their own: 0.797 is what the
base 50 alone reach, 0.714 is what base + couples reach, and this arm says whether the
couples carry a usable decision boundary at all or only distort one.

Run at both gradient-accumulation settings, because the two reference numbers live in
different regimes and neither can be quoted against the other:

* `gas4` is the probe's own hyperparameter, the setting base-only 0.797 and base+couples
  0.714 were produced under. At 66 rows that is 5 batches, so 1 optimizer step per epoch.
* `gas1` matches `base_subset_refit.py`'s arms (5 steps per epoch here).

`base_plus_couples` is refit under gas1 so the subset study's regime also has its own
reference. See `base_subset_refit.py`'s docstring for why fewer than 49 rows at gas4 does
not train at all; 66 rows is above that line, so both settings here are valid fits.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RES = REPO / "results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
PRB = REPO / "probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP = REPO / "probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE = REPO / "data/instructions_llama70b_50.jsonl"
DEV = REPO / "dev_samples/instructions"
BC = REPO / "results_instructions_gemma27b_shared/base_activations"
EC = REPO / "results_instructions_gemma27b_shared/eval_activations"
SEED = 42


def main() -> int:
    import torch
    from agentic_redteam.ensemble import EnsembleProbe, fusion_enabled
    from agentic_redteam.evaluation import evaluate_probe, seed_everything
    from agentic_redteam.retrain import (
        _activate_redteam_cached, _base_activation_cache_paths, _concatenate_consuming,
        _cpu_unpickle, _dev_activation_cache_path, _infer_probe_spec, _load_dev_dataset,
        _resolve_ensemble_seeds, _to_device_for_fit, stable_train_test_split,
    )
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage
    from tuberlens.interfaces.probes import ProbeSpec
    from tuberlens.model import LLMModel
    from tuberlens.probes.probe_factory import ProbeFactory

    with BP.open("rb") as f:
        bp = _cpu_unpickle(f)
    pos, neg = bp.pos_class_label, bp.neg_class_label
    spec0 = _infer_probe_spec(bp)
    hp = dict(spec0.hyperparams)
    BATCH = int(hp.get("batch_size", 16))
    seeds = _resolve_ensemble_seeds(SEED, 10)
    C = V = True; TS = 0.0

    base = LabelledDataset.load_from(Path(BASE), pos_class_label=pos, neg_class_label=neg,
                                     combine_consecutive_messages=C, convert_tool_to_assistant=V)
    btr, _ = stable_train_test_split(base, test_size=TS, split_field=None, seed=SEED)
    btc, _ = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, SEED, TS, None,
                                          C, V, 1.0)
    a = LLMModel.load_activations(btc)
    BASE_DS = btr.assign(activations=a.activations, attention_mask=a.attention_mask,
                         input_ids=a.input_ids)

    dv, dfiles, _sz = _load_dev_dataset(DEV, pos, neg, C, V, verbose=False)
    da = LLMModel.load_activations(_dev_activation_cache_path(BC, dfiles, bp.model_name,
                                                              bp.layer, C, V))
    VAL = dv.assign(activations=da.activations, attention_mask=da.attention_mask,
                    input_ids=da.input_ids)

    rows = [json.loads(l) for l in (PRB / "redteam_postprocessed_iter5.jsonl").open()]
    rt = LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]]
                for r in rows],
        ids=[r["id"] for r in rows],
        other_fields={"labels": [r["label"] for r in rows]})
    rtr, _ = stable_train_test_split(rt, test_size=TS, split_field=None, seed=SEED)
    RT = _activate_redteam_cached(rtr, BC, bp.model_name, bp.layer, C, V, lambda: None, False)
    n_pos = sum(1 for r in rows if r["label"] == "positive")
    print(f"{len(RT)} red-team rows ({n_pos} positive / {len(rows)-n_pos} negative) from "
          f"{len(rows)//2} couples; base {len(BASE_DS)}; dev {len(VAL)}", flush=True)
    _to_device_for_fit([VAL], verbose=True)

    def fit(tag, parts, gas):
        train = _concatenate_consuming(list(parts)) if len(parts) > 1 else parts[0]
        n_batches = -(-len(train) // BATCH)
        steps = n_batches // gas
        if steps == 0:
            raise SystemExit(f"{tag}: {len(train)} rows / {n_batches} batches < gas {gas}; "
                             "the optimizer would never step")
        print(f"\n=== {tag}: {len(train)} rows, {n_batches} batches, gas {gas}, "
              f"{steps} steps/epoch ===", flush=True)
        spec = ProbeSpec(name=spec0.name, hyperparams=hp | {"gradient_accumulation_steps": gas})
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
        q = RES / f"probe_rtonly_{tag}.pkl"
        with q.open("wb") as f:
            pickle.dump(probe, f)
        df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO / "eval_sets/instructions",
                            activations_cache_dir=EC, splits=None, max_samples=None, seed=SEED,
                            combine_consecutive_messages=C, convert_tool_to_assistant=V)
        df.to_csv(RES / f"eval_rtonly_{tag}.csv", index=False)
        print(df.to_string(index=False), flush=True)
        del train, members, probe
        torch.cuda.empty_cache()
        return {r["dataset"]: round(float(r["auroc"]), 4) for _, r in df.iterrows()}

    def copy_of(ds):
        # _concatenate_consuming consumes its inputs; list-indexing copies.
        return ds[list(range(len(ds)))]

    res = {}
    res["couples_gas4"] = fit("couples_gas4", [copy_of(RT)], 4)
    res["couples_gas1"] = fit("couples_gas1", [copy_of(RT)], 1)
    res["base_couples_gas1"] = fit("base_couples_gas1", [copy_of(BASE_DS), copy_of(RT)], 1)
    out = RES / "redteam_only_refit.json"
    out.write_text(json.dumps({"seed": SEED, "auroc": res}, indent=1))

    print("\n===== RED-TEAM COUPLES WITHOUT BASE DATA (AUROC) =====")
    keys = [s for s in res["couples_gas4"] if s != "mean"] + ["mean"]
    w = max(len(s) for s in keys)
    print(" " * (w + 2) + "".join(f"{t:>19}" for t in res))
    for s in keys:
        print(f"{s:<{w}}  " + "".join(f"{res[t][s]:>19.4f}" for t in res))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
