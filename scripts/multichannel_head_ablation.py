"""Train probes whose readout has p parallel scoring channels, and score them.

The head is agentic_redteam.segmented_head.MultiChannelLinearThenSoftmax: step 1 projects each
token to p scores instead of 1, each channel is softmax-pooled over the sequence independently,
and Linear(p -> 1) combines the p pooled values. At p=1 it is LinearThenSoftmax exactly, so
that arm is the control separating "more channels" from "one extra affine".

Rationale: the stock head is nearly a max over tokens along a single direction -- measured, one
token of 179 took 93% of the weight -- so one position and one direction carry the whole
verdict. p channels let the probe learn several things to look for and locate each of them
independently.

Same base 50, same 33 couples, same 436-row dev set, same 10 pinned ensemble seeds. Only the
head differs, so the probe still sees full token sequences and evaluate_probe scores it the
ordinary way; the numbers are directly comparable to every other fit in this project.
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
from agentic_redteam.segmented_head import (MultiChannelLinearThenSoftmax,
    verify_multichannel_head_identity)
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

print("verifying the head reproduces LinearThenSoftmax at p=1:")
verify_multichannel_head_identity()

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(SEED, 10)
HP = dict(spec.hyperparams)
print(f"\nbase hyperparameters: {HP}")

base = LabelledDataset.load_from(Path(BASE), pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btr, _ = stable_train_test_split(base, test_size=TS, split_field=None, seed=SEED)
btc, _bvc = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, SEED, TS, None, C, V, 1.0)
a = LLMModel.load_activations(btc)
btr = btr.assign(activations=a.activations, attention_mask=a.attention_mask, input_ids=a.input_ids)
dv, dfiles, _sz = _load_dev_dataset(DEV, pos, neg, C, V, verbose=False)
da = LLMModel.load_activations(_dev_activation_cache_path(BC, dfiles, bp.model_name, bp.layer, C, V))
dv = dv.assign(activations=da.activations, attention_mask=da.attention_mask, input_ids=da.input_ids)
rt_rows = [json.loads(l) for l in (PRB/"redteam_postprocessed_iter5.jsonl").open()]
rt = LabelledDataset(
    inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rt_rows],
    ids=[r["id"] for r in rt_rows], other_fields={"labels": [r["label"] for r in rt_rows]})
rt = _activate_redteam_cached(rt, BC, bp.model_name, bp.layer, C, V, lambda: None, False)
TRAIN = _concatenate_consuming([btr, rt]); VAL = dv
print(f"train {len(TRAIN)} rows | validation {len(VAL)} rows")
_to_device_for_fit([TRAIN, VAL], verbose=True)


def fit(tag, p, stock=False):
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
        hp = HP | {"n_channels": p}
        probe = PytorchProbe(hyper_params=hp, model_name=bp.model_name, layer=bp.layer,
            pos_class_label=pos, neg_class_label=neg, description=bp.description,
            _classifier=PytorchAdamClassifier(training_args=hp,
                                              probe_architecture=MultiChannelLinearThenSoftmax))
        probe.fit(TRAIN, VAL)
        members.append(probe)
    probe = EnsembleProbe.from_members(members, list(seeds))
    q = RES/f"probe_mch_{tag}.pkl"
    with q.open("wb") as f: pickle.dump(probe, f)
    df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO/"eval_sets/instructions",
        activations_cache_dir=EC, splits=None, max_samples=None, seed=SEED,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    df.to_csv(RES/f"eval_mch_{tag}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return {r["dataset"]: round(float(r["auroc"]), 4) for _, r in df.iterrows()}


# Default ladder, or the p values named on the command line (those arms only, merged into
# the existing JSON so the printed table always covers every p fit so far).
OUT = RES/"multichannel_head_ablation.json"
if len(sys.argv) > 1:
    ARMS = [(f"p{int(v)}", int(v), False) for v in sys.argv[1:]]
else:
    ARMS = [("stock", 0, True), ("p1", 1, False), ("p2", 2, False), ("p4", 4, False),
            ("p8", 8, False), ("p16", 16, False), ("p32", 32, False)]
res = json.load(OUT.open())["auroc"] if OUT.exists() else {}
for t, p, st in ARMS:
    res[t] = fit(t, p, st)
    json.dump({"seed": SEED, "auroc": res}, OUT.open("w"), indent=1)

order = sorted(res, key=lambda t: -1 if t == "stock" else int(t[1:]))
print("\n===== MULTI-CHANNEL READOUT (AUROC) =====")
names = [s for s in res[order[0]] if s != "mean"] + ["mean"]
w = max(len(s) for s in names)
print(" " * (w + 2) + "".join(f"{t:>10}" for t in order))
for s in names:
    print(f"{s:<{w}}  " + "".join(f"{res[t][s]:>10.4f}" for t in order))
