#!/usr/bin/env python
"""Does the MultiMax head help, at three training-set sizes?

Run:

    PROBE_FUSED_ENSEMBLE=0 .venv_claude/bin/python scripts/multimax_data_arms.py

MultiMax (arXiv:2601.11516 Section 3.2.1) replaces the pooling softmax with a hard
per-head max. `docs/probe_head_architecture_2026-08-26.md` tested five families of
readout change on this probe and found none that survived its own noise, with the
strong hint that the binding constraint is the 116 training rows rather than the
head's expressiveness — every arm that *added* parameters came back neutral or
negative. MultiMax at the paper's defaults adds ~549k of them. So the interesting
question is not "is MultiMax better" in the abstract but whether its answer changes
as the training set grows, which is what the three data conditions here vary:

    base            data/instructions_llama70b_50.jsonl                  50 rows
    base+couples    + the v3 run's 33 red-team couples                  116 rows
    base+couples+dev  + 16 lent `oig_omission` dev pairs                148 rows

The third is the condition `docs/dev_lending_and_pair_reshaping_2026-08-26.md` found
to be the only one that helps (0.797 -> 0.897 on `oig_omission` over its ladder), so
it is where a head with capacity to spare has the best chance of being fed enough.

**Four heads per condition**, because "MultiMax helps" and "a hard max helps" are
different claims and only one of them is about capacity:

    stock            LinearThenSoftmax, T=5 -- the control every reference uses
    linear_then_max  the same 5,377 parameters with the softmax swapped for a max
    multimax_paper   10 heads, MLP 2x100 (the paper's defaults)  ~549k parameters
    multimax_small   4 heads, MLP 2x16                            ~86k parameters

`linear_then_max` is the load-bearing control: it isolates the aggregation change
from the MLP-and-heads capacity that comes bundled with it in MultiMax.

**Everything apart from the head is held fixed**, matching the protocol of the
head-architecture document: gemma-3-27b-it L32, 10-member ensembles under the pinned
`ENSEMBLE_SEEDS`, the base probe's own training schedule (batch 16, gas 4, lr 5e-3,
200 epochs, patience 50), evaluation on all seven `eval_sets/instructions` splits at
full size. Only `probe_spec` differs between arms.

**Validation is identical in all twelve fits.** The 32 dev rows the third condition
lends to training are withheld from validation in *every* condition, not just the one
that trains on them — otherwise the third arm's probes would be early-stopped against
a different set than the first two and the checkpoints would not be comparable. This
is `_dev_lending_indices`' own rule, applied across arms rather than across
iterations.

All activations are cached, so no extraction model is loaded and nothing here needs
the network. Single seed per arm — the same governing caveat as the document this
extends.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
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
EVAL = REPO / "eval_sets/instructions"
SEED = 42

# The dev-lending reserve, in PAIRS, restricted to the split being read. 16 is the
# whole `oig_omission` dev pool (32 of the 436 dev rows) -- the saturation point the
# ladder in dev_lending_and_pair_reshaping reached.
DEV_PAIRS = 16
DEV_SPLIT = "oig_omission"

TARGET = "oig_omission"


def main() -> int:
    import torch
    from agentic_redteam.ensemble import EnsembleProbe, fusion_enabled
    from agentic_redteam.evaluation import evaluate_probe, seed_everything
    from agentic_redteam.retrain import (
        _activate_redteam_cached, _base_activation_cache_paths, _concatenate_consuming,
        _cpu_unpickle, _dev_activation_cache_path, _dev_lending_groups,
        _dev_lending_indices, _infer_probe_spec, _load_dev_dataset,
        _resolve_ensemble_seeds, _to_device_for_fit, stable_train_test_split,
    )
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType
    from tuberlens.model import LLMModel
    from tuberlens.probes.probe_factory import ProbeFactory

    with BP.open("rb") as f:
        bp = _cpu_unpickle(f)
    pos, neg = bp.pos_class_label, bp.neg_class_label
    stock_spec = _infer_probe_spec(bp)
    hp = dict(stock_spec.hyperparams)
    BATCH = int(hp.get("batch_size", 16))
    GAS = int(hp.get("gradient_accumulation_steps", 4))
    seeds = _resolve_ensemble_seeds(SEED, 10)
    C = V = True
    TS = 0.0

    # ---- base -------------------------------------------------------------
    base = LabelledDataset.load_from(
        Path(BASE), pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    btr, _ = stable_train_test_split(base, test_size=TS, split_field=None, seed=SEED)
    btc, _ = _base_activation_cache_paths(
        BC, BASE, bp.model_name, bp.layer, SEED, TS, None, C, V, 1.0)
    a = LLMModel.load_activations(btc)
    BASE_DS = btr.assign(activations=a.activations, attention_mask=a.attention_mask,
                         input_ids=a.input_ids)

    # ---- the 33 red-team couples -----------------------------------------
    rows = [json.loads(l) for l in (PRB / "redteam_postprocessed_iter5.jsonl").open()]
    rt = LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]]
                for r in rows],
        ids=[r["id"] for r in rows],
        other_fields={"labels": [r["label"] for r in rows]})
    rtr, _ = stable_train_test_split(rt, test_size=TS, split_field=None, seed=SEED)
    RT = _activate_redteam_cached(rtr, BC, bp.model_name, bp.layer, C, V, lambda: None, False)

    # ---- dev: split into the lent rows and the validation set -------------
    dv, dfiles, dsizes = _load_dev_dataset(DEV, pos, neg, C, V, verbose=False)
    da = LLMModel.load_activations(
        _dev_activation_cache_path(BC, dfiles, bp.model_name, bp.layer, C, V))
    DEV_ALL = dv.assign(activations=da.activations, attention_mask=da.attention_mask,
                        input_ids=da.input_ids)
    groups = _dev_lending_groups(DEV_ALL, dfiles, dsizes, DEV_SPLIT, "pairs")
    lent_idx, val_idx = _dev_lending_indices(
        len(DEV_ALL), DEV_PAIRS, DEV_PAIRS, SEED, groups=groups, verbose=True)
    LENT = DEV_ALL[lent_idx]
    VAL = DEV_ALL[val_idx]

    print(f"base {len(BASE_DS)} rows | couples {len(RT)} rows ({len(rows) // 2} couples) | "
          f"lent dev {len(LENT)} rows ({DEV_PAIRS} {DEV_SPLIT} pairs)")
    print(f"validation {len(VAL)} of {len(DEV_ALL)} dev rows "
          f"(the {len(LENT)} lent rows are withheld in EVERY arm)", flush=True)
    _to_device_for_fit([VAL], verbose=True)

    def copy_of(ds):
        # _concatenate_consuming consumes its inputs; list-indexing copies.
        return ds[list(range(len(ds)))]

    CONDITIONS = {
        "base": lambda: [copy_of(BASE_DS)],
        "base+couples": lambda: [copy_of(BASE_DS), copy_of(RT)],
        "base+couples+dev": lambda: [copy_of(BASE_DS), copy_of(RT), copy_of(LENT)],
    }
    ARMS = {
        "stock": ProbeSpec(name=ProbeType.linear_then_softmax, hyperparams=dict(hp)),
        "linear_then_max": ProbeSpec(name=ProbeType.linear_then_max, hyperparams=dict(hp)),
        "multimax_paper": ProbeSpec(
            name=ProbeType.multimax,
            hyperparams=hp | {"n_heads": 10, "mlp_layers": 2, "mlp_width": 100, "agg": "max"}),
        "multimax_small": ProbeSpec(
            name=ProbeType.multimax,
            hyperparams=hp | {"n_heads": 4, "mlp_layers": 2, "mlp_width": 16, "agg": "max"}),
    }

    def n_params(spec, embed_dim: int = 5376) -> int:
        from tuberlens.probes.probe_factory import _ADAM_ARCHITECTURES
        module = _ADAM_ARCHITECTURES[spec.name](embed_dim, **spec.hyperparams)
        return sum(p.numel() for p in module.parameters())

    print("\nhead parameter counts (embed 5376):")
    for tag, spec in ARMS.items():
        print(f"  {tag:<16} {n_params(spec):>9,}")

    def fit(cond: str, arm: str, spec: ProbeSpec) -> dict:
        train = _concatenate_consuming(CONDITIONS[cond]())
        n_batches = -(-len(train) // BATCH)
        steps = n_batches // GAS
        tag = f"{cond}__{arm}"
        print(f"\n=== {tag}: {len(train)} rows, {n_batches} batches, gas {GAS}, "
              f"{steps} steps/epoch ===", flush=True)
        if steps == 0:
            raise SystemExit(f"{tag}: the optimizer would never step")
        _to_device_for_fit([train], verbose=False)
        t0 = time.perf_counter()
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
        q = RES / f"probe_mm_{tag}.pkl"
        with q.open("wb") as f:
            pickle.dump(probe, f)
        df = evaluate_probe(probe_path=q, eval_dataset_dir=EVAL, activations_cache_dir=EC,
                            splits=None, max_samples=None, seed=SEED,
                            combine_consecutive_messages=C, convert_tool_to_assistant=V)
        df.to_csv(RES / f"eval_mm_{tag}.csv", index=False)
        out = {r["dataset"]: round(float(r["auroc"]), 4) for _, r in df.iterrows()}
        print(f"  fit+eval {time.perf_counter() - t0:.0f}s | {TARGET} {out[TARGET]:.4f} | "
              f"mean {out['mean']:.4f}", flush=True)
        del train, members, probe
        torch.cuda.empty_cache()
        return out

    res: dict[str, dict[str, dict]] = {}
    for cond in CONDITIONS:
        res[cond] = {}
        for arm, spec in ARMS.items():
            res[cond][arm] = fit(cond, arm, spec)
            (RES / "multimax_data_arms.json").write_text(
                json.dumps({"seed": SEED, "dev_pairs_lent": DEV_PAIRS,
                            "dev_split": DEV_SPLIT, "auroc": res}, indent=1))

    def table(metric: str) -> None:
        print(f"\n===== {metric} AUROC =====")
        w = max(len(c) for c in CONDITIONS)
        print(" " * (w + 2) + "".join(f"{a:>17}" for a in ARMS))
        for cond in CONDITIONS:
            print(f"{cond:<{w}}  " + "".join(
                f"{res[cond][a][metric]:>17.4f}" for a in ARMS))

    table(TARGET)
    table("mean")
    print(f"\nwrote {RES / 'multimax_data_arms.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
