#!/usr/bin/env python
"""Train on EVERY red-team attempt of the v3 arm -- finds and failures alike, no pairs.

`redteam_only_refit.py` and the `what_limits_the_instruction_probe` table both train on the
33 contrastive COUPLES that arm produced: 34 successful finds, each paired with a generated
opposite-class partner. That is 66 rows out of a rotation that actually submitted 373
conversations, and the 339 unsuccessful ones were thrown away -- even though the judge
labelled every single one of them on its own merits.

This script asks what those 373 rows are worth as supervision. No contrastive generation at
all: each attempt is trained under the judge's own label, exactly as `_records_to_labelled_dataset`
does for the successes. The class balance is whatever the rotation happened to produce
(a `false_positive` hunt writes mostly negative-truth conversations), which is part of what
is being measured, not something to correct.

Four arms, all at the probe's own `gradient_accumulation_steps` (the setting base-only 0.797
and base+couples 0.714 were fit under), 10 members, `dev_samples/instructions` as validation,
scored on all seven `eval_sets/instructions` splits at full size:

  attempts_only   373 attempts, no base data
  base_attempts   base 50 + 373 attempts
  scoped_only     327 attempts (the 46 the judge's scope check REJECTED dropped), no base
  base_scoped     base 50 + 327 attempts

The scoped variants exist because a rejected candidate is not the kind of conversation the
probe is scored on -- CLAUDE.md's convention is that it is never training data -- so including
it is a deliberate choice worth measuring rather than assuming.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RES = REPO / "results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
JSONL = RES / "gptoss120b_probing.jsonl"
BP = REPO / "probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE = REPO / "data/instructions_llama70b_50.jsonl"
DEV = REPO / "dev_samples/instructions"
BC = REPO / "results_instructions_gemma27b_shared/base_activations"
EC = REPO / "results_instructions_gemma27b_shared/eval_activations"
SEED = 42


def main() -> int:
    import gc

    import torch
    from agentic_redteam.ensemble import EnsembleProbe, fusion_enabled
    from agentic_redteam.evaluation import evaluate_probe, seed_everything
    from agentic_redteam.model_loading import load_extraction_model, unhook_model
    from agentic_redteam.retrain import (
        _activate_redteam_cached, _apply_message_transforms, _base_activation_cache_paths,
        _concatenate_consuming, _cpu_unpickle, _dev_activation_cache_path, _infer_probe_spec,
        _load_dev_dataset, _resolve_ensemble_seeds, _to_device_for_fit,
        stable_train_test_split,
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
    GAS = int(hp.get("gradient_accumulation_steps", 4))
    seeds = _resolve_ensemble_seeds(SEED, 10)
    C = V = True
    TS = 0.0

    # ---- the attempts -------------------------------------------------------------
    recs = [json.loads(l) for l in JSONL.open()]

    def canonical(r):
        # Same rule as retrain._records_to_labelled_dataset: the judge is the source of
        # truth, error_type is only the fallback for an unparseable verdict.
        if r["judge_label"] == r["pos_class_label"]:
            return "positive"
        if r["judge_label"] == r["neg_class_label"]:
            return "negative"
        return "negative" if r["error_type"] == "false_positive" else "positive"

    def build(rows, tag):
        ds = LabelledDataset(
            inputs=[[TLMessage(role=m["role"], content=m["content"])
                     for m in r["sample"]["messages"]] for r in rows],
            ids=[f"attempt-{tag}-{i}" for i in range(len(rows))],
            other_fields={"labels": [canonical(r) for r in rows]})
        return _apply_message_transforms(ds, C, V)

    scoped = [r for r in recs if not r.get("violated_constraint")]
    ALL_DS = build(recs, "all")
    SCOPED_DS = build(scoped, "scoped")
    for name, rows in (("all", recs), ("scoped", scoped)):
        n_pos = sum(1 for r in rows if canonical(r) == "positive")
        n_ok = sum(1 for r in rows if r["success"])
        print(f"{name}: {len(rows)} attempts -- {n_pos} positive / {len(rows) - n_pos} "
              f"negative, of which {n_ok} were successful finds", flush=True)

    # ---- (i) activations for every attempt ----------------------------------------
    loaded = {"model": None}

    def get_model():
        if loaded["model"] is None:
            print("Loading gemma-3-27b for activation extraction ...", flush=True)
            loaded["model"] = load_extraction_model(str(bp.model_name), bp.layer, verbose=True)
        return loaded["model"]

    def release_model():
        if loaded["model"] is None:
            return
        unhook_model(loaded["model"])
        loaded["model"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ALL_DS is a superset of SCOPED_DS, so activating it fills the per-conversation
    # cache for both; SCOPED_DS then loads entirely from disk with no model.
    ALL = _activate_redteam_cached(ALL_DS, BC, str(bp.model_name), bp.layer, C, V,
                                   get_model, True)
    release_model()
    SCOPED = _activate_redteam_cached(SCOPED_DS, BC, str(bp.model_name), bp.layer, C, V,
                                      lambda: None, False)
    print(f"activated: all {len(ALL)}, scoped {len(SCOPED)}", flush=True)

    # ---- base + dev ----------------------------------------------------------------
    base = LabelledDataset.load_from(Path(BASE), pos_class_label=pos, neg_class_label=neg,
                                     combine_consecutive_messages=C,
                                     convert_tool_to_assistant=V)
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
    print(f"base {len(BASE_DS)} rows; dev {len(VAL)} rows", flush=True)
    _to_device_for_fit([VAL], verbose=True)

    # ---- (ii) the fits ---------------------------------------------------------------
    def fit(tag, parts):
        train = _concatenate_consuming(list(parts)) if len(parts) > 1 else parts[0]
        n_batches = -(-len(train) // BATCH)
        steps = n_batches // GAS
        if steps == 0:
            raise SystemExit(f"{tag}: {len(train)} rows / {n_batches} batches < gas {GAS}")
        print(f"\n=== {tag}: {len(train)} rows, {n_batches} batches, gas {GAS}, "
              f"{steps} steps/epoch ===", flush=True)
        spec = ProbeSpec(name=spec0.name, hyperparams=hp)
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
        q = RES / f"probe_allatt_{tag}.pkl"
        with q.open("wb") as f:
            pickle.dump(probe, f)
        df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO / "eval_sets/instructions",
                            activations_cache_dir=EC, splits=None, max_samples=None,
                            seed=SEED, combine_consecutive_messages=C,
                            convert_tool_to_assistant=V)
        df.to_csv(RES / f"eval_allatt_{tag}.csv", index=False)
        print(df.to_string(index=False), flush=True)
        del train, members, probe
        gc.collect()
        torch.cuda.empty_cache()
        return {r["dataset"]: round(float(r["auroc"]), 4) for _, r in df.iterrows()}

    def copy_of(ds):
        # _concatenate_consuming consumes its inputs; list-indexing copies.
        return ds[list(range(len(ds)))]

    res = {}
    res["attempts_only"] = fit("attempts_only", [copy_of(ALL)])
    res["base_attempts"] = fit("base_attempts", [copy_of(BASE_DS), copy_of(ALL)])
    res["scoped_only"] = fit("scoped_only", [copy_of(SCOPED)])
    res["base_scoped"] = fit("base_scoped", [copy_of(BASE_DS), copy_of(SCOPED)])

    out = RES / "all_attempts_refit.json"
    out.write_text(json.dumps({
        "seed": SEED, "gas": GAS, "n_members": len(seeds),
        "n_attempts": len(recs), "n_scoped": len(scoped),
        "auroc": res}, indent=1))

    print("\n===== TRAINING ON EVERY ATTEMPT, NO CONTRASTIVE PAIRS (AUROC) =====")
    keys = [s for s in res["attempts_only"] if s != "mean"] + ["mean"]
    w = max(len(s) for s in keys)
    print(" " * (w + 2) + "".join(f"{t:>16}" for t in res))
    for s in keys:
        print(f"{s:<{w}}  " + "".join(f"{res[t][s]:>16.4f}" for t in res))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
