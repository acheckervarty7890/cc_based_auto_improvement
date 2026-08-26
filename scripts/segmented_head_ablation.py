"""Train probes with the k-segment readout head and score them against the standard one.

The head is agentic_redteam.segmented_head.LinearThenSegmentedSoftmax: steps 1-4 of the usual
probe unchanged (per-token linear scores, masked, softmax-weighted), then the sum at step 5 is
split into k positional buckets over the VALID span and a final Linear(k -> 1) combines them.
At initialization it is byte-for-byte the old head, so every arm starts from the same function.

Arms: the stock linear_then_softmax control, then k = 1, 2, 3, 4, plus k = 2 and 3 under
`segment_softmax` (weights renormalized within each bucket instead of competing globally).
k=1 isolates what the extra Linear(1->1) buys from what the segmentation buys.

Unlike scripts/segment_pooled_probe.py this changes only the HEAD -- the probe still receives
the full token sequence, so evaluate_probe scores it the ordinary way and the numbers are
directly comparable to every other fit in this project.

Same base 50, same 33 couples, same 436-row dev set, same 10 pinned ensemble seeds. All
activations are cached; no extraction model is loaded.
"""
import json, os, pickle, sys
from pathlib import Path
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
os.environ["PROBE_FUSED_ENSEMBLE"] = "0"   # vmap fusion assumes the stock head's parameter stack
from agentic_redteam.ensemble import EnsembleProbe
from agentic_redteam.evaluation import evaluate_probe, seed_everything
from agentic_redteam.retrain import (_activate_redteam_cached, _base_activation_cache_paths,
    _concatenate_consuming, _cpu_unpickle, _dev_activation_cache_path, _infer_probe_spec,
    _load_dev_dataset, _resolve_ensemble_seeds, _to_device_for_fit, stable_train_test_split)
from agentic_redteam.segmented_head import (LinearThenSegmentedSoftmax,
    verify_segmented_head_identity)
from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage
from tuberlens.model import LLMModel
from tuberlens.probes.probe_factory import ProbeFactory
from tuberlens.probes.pytorch_classifiers import PytorchAdamClassifier
from tuberlens.probes.pytorch_probes import PytorchProbe

RES = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
PRB = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP  = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE= REPO/"data/instructions_llama70b_50.jsonl"; DEV = REPO/"dev_samples/instructions"
BC  = REPO/"results_instructions_gemma27b_shared/base_activations"
EC  = REPO/"results_instructions_gemma27b_shared/eval_activations"
SEED = 42; C = V = True; TS = 0.0

print("verifying the head reproduces LinearThenSoftmax at init:")
verify_segmented_head_identity()

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(SEED, 10)
HP = dict(spec.hyperparams)
print(f"\nbase hyperparameters: {HP}")

base = LabelledDataset.load_from(Path(BASE), pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btr, _bva = stable_train_test_split(base, test_size=TS, split_field=None, seed=SEED)
btc, _bvc = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, SEED, TS, None, C, V, 1.0)
acts = LLMModel.load_activations(btc)
btr = btr.assign(activations=acts.activations, attention_mask=acts.attention_mask,
                 input_ids=acts.input_ids)
dv, dfiles, _sz = _load_dev_dataset(DEV, pos, neg, C, V, verbose=False)
dacts = LLMModel.load_activations(_dev_activation_cache_path(BC, dfiles, bp.model_name, bp.layer, C, V))
dv = dv.assign(activations=dacts.activations, attention_mask=dacts.attention_mask,
               input_ids=dacts.input_ids)
rt_rows = [json.loads(l) for l in (PRB/"redteam_postprocessed_iter5.jsonl").open()]
rt = LabelledDataset(
    inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rt_rows],
    ids=[r["id"] for r in rt_rows], other_fields={"labels": [r["label"] for r in rt_rows]})
rt = _activate_redteam_cached(rt, BC, bp.model_name, bp.layer, C, V, lambda: None, False)
TRAIN = _concatenate_consuming([btr, rt]); VAL = dv
print(f"train {len(TRAIN)} rows | validation {len(VAL)} rows")
_to_device_for_fit([TRAIN, VAL], verbose=True)


def fit(tag, k, seg_softmax=False, stock=False):
    print(f"\n=== {tag} ===", flush=True)
    members = []
    for s in seeds:
        seed_everything(s)
        if stock:
            members.append(ProbeFactory.build(probe_spec=spec, train_dataset=TRAIN,
                model_name=bp.model_name, layer=bp.layer, validation_dataset=VAL,
                use_store=False, pos_class_label=pos, neg_class_label=neg,
                probe_description=bp.description))
            continue
        hp = HP | {"n_segments": k, "segment_softmax": seg_softmax}
        probe = PytorchProbe(hyper_params=hp, model_name=bp.model_name, layer=bp.layer,
            pos_class_label=pos, neg_class_label=neg, description=bp.description,
            _classifier=PytorchAdamClassifier(training_args=hp,
                                              probe_architecture=LinearThenSegmentedSoftmax))
        probe.fit(TRAIN, VAL)
        members.append(probe)
    probe = EnsembleProbe.from_members(members, list(seeds))
    q = RES/f"probe_seghead_{tag}.pkl"
    with q.open("wb") as f: pickle.dump(probe, f)
    df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO/"eval_sets/instructions",
        activations_cache_dir=EC, splits=None, max_samples=None, seed=SEED,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    df.to_csv(RES/f"eval_seghead_{tag}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return {r["dataset"]: round(float(r["auroc"]), 4) for _, r in df.iterrows()}


ARMS = [("stock", 0, False, True), ("k1", 1, False, False), ("k2", 2, False, False),
        ("k3", 3, False, False), ("k4", 4, False, False),
        ("k2_segsm", 2, True, False), ("k3_segsm", 3, True, False)]
res = {t: fit(t, k, ss, st) for t, k, ss, st in ARMS}
json.dump({"seed": SEED, "auroc": res}, open(RES/"segmented_head_ablation.json", "w"), indent=1)

print("\n===== SEGMENTED-HEAD READOUT (AUROC) =====")
names = [s for s in res["stock"] if s != "mean"] + ["mean"]
w = max(len(s) for s in names)
print(" " * (w + 2) + "".join(f"{t:>11}" for t in res))
for s in names:
    print(f"{s:<{w}}  " + "".join(f"{res[t][s]:>11.4f}" for t in res))
