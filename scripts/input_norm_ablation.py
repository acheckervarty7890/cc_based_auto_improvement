"""LayerNorm on the activation vector before the probe's readout.

The head scores each token as w.x + b and then softmax-weights those scores by themselves.
Both halves are scale-sensitive: a token whose residual-stream vector is simply LARGER scores
higher along any direction, and the softmax reads the resulting scores against a fixed
temperature. Normalizing x over the embedding axis removes the first effect entirely and
standardizes the second.

That makes it two changes at once, which is why this runs at two temperatures rather than one:
LayerNorm shrinks the score scale, and the score scale is exactly what T is measured against.
Reporting a LayerNorm arm at T=5 alone would confound "normalization hurt" with "T is now
mistuned". T=10 is included because it beat T=5 on the target split at every channel count
below 16.

Both affine variants are run: the learnable one (2*5376 = 10,752 parameters, a 2x increase on
the p=1 head's parameter count) and the parameter-free one, since at 116 training rows the
affine may cost more in variance than it buys.

Controls are the existing identity arms at the same p and T. Same base 50, same 33 couples,
same 436-row dev set, same 10 pinned ensemble seeds; all activations cached.
"""
import json, os, pickle, sys
from pathlib import Path
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
os.environ["PROBE_FUSED_ENSEMBLE"] = "0"
from agentic_redteam.ensemble import EnsembleProbe
from agentic_redteam.evaluation import evaluate_probe, seed_everything
from agentic_redteam.retrain import (_activate_redteam_cached, _base_activation_cache_paths,
    _concatenate_consuming, _cpu_unpickle, _dev_activation_cache_path, _infer_probe_spec,
    _load_dev_dataset, _resolve_ensemble_seeds, _to_device_for_fit, stable_train_test_split)
from agentic_redteam.segmented_head import MultiChannelLinearThenSoftmax
from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage
from tuberlens.model import LLMModel
from tuberlens.probes.pytorch_classifiers import PytorchAdamClassifier
from tuberlens.probes.pytorch_probes import PytorchProbe

RES = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
PRB = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP  = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE= REPO/"data/instructions_llama70b_50.jsonl"; DEV = REPO/"dev_samples/instructions"
BC  = REPO/"results_instructions_gemma27b_shared/base_activations"
EC  = REPO/"results_instructions_gemma27b_shared/eval_activations"
SEED = 42; C = V = True; TS = 0.0

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(SEED, 10)
HP = dict(spec.hyperparams)

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
print(f"train {len(TRAIN)} rows | validation {len(VAL)} rows", flush=True)
_to_device_for_fit([TRAIN, VAL], verbose=True)

x = TRAIN.other_fields["activations"]; mask = TRAIN.other_fields["attention_mask"].bool()
vec = x[mask]
print(f"\nactivation vector norms over {vec.shape[0]} real tokens: "
      f"mean {vec.float().norm(dim=-1).mean():.1f}  "
      f"p5 {vec.float().norm(dim=-1).quantile(0.05):.1f}  "
      f"p95 {vec.float().norm(dim=-1).quantile(0.95):.1f}   "
      f"(LayerNorm maps every one of these to sqrt(embed) = {x.shape[-1] ** 0.5:.1f})", flush=True)


def fit(tag, p, norm, temperature):
    print(f"\n=== {tag} (p={p}, {norm}, T={temperature}) ===", flush=True)
    members = []
    for s in seeds:
        seed_everything(s)
        hp = HP | {"n_channels": p, "input_norm": norm, "temperature": temperature}
        probe = PytorchProbe(hyper_params=hp, model_name=bp.model_name, layer=bp.layer,
            pos_class_label=pos, neg_class_label=neg, description=bp.description,
            _classifier=PytorchAdamClassifier(training_args=hp,
                                              probe_architecture=MultiChannelLinearThenSoftmax))
        probe.fit(TRAIN, VAL)
        members.append(probe)
    probe = EnsembleProbe.from_members(members, list(seeds))
    q = RES/f"probe_ln_{tag}.pkl"
    with q.open("wb") as f: pickle.dump(probe, f)
    df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO/"eval_sets/instructions",
        activations_cache_dir=EC, splits=None, max_samples=None, seed=SEED,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    df.to_csv(RES/f"eval_ln_{tag}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return {r["dataset"]: round(float(r["auroc"]), 4) for _, r in df.iterrows()}


ARMS = [(f"p{p}_{short}_T{t}", p, norm, t)
        for p in (1, 2)
        for short, norm in (("ln", "layernorm"), ("lnna", "layernorm_noaffine"))
        for t in (5, 10)]

OUT = RES/"input_norm_ablation.json"
res = json.load(OUT.open())["auroc"] if OUT.exists() else {}
for tag, p, norm, t in ARMS:
    res[tag] = fit(tag, p, norm, t)
    json.dump({"seed": SEED, "auroc": res}, OUT.open("w"), indent=1)

# controls: identity at the same (p, T), from the ladder (T=5) and the temperature sweep (T=10)
ctl = {}
mch = json.load((RES/"multichannel_head_ablation.json").open())["auroc"]
for p in (1, 2):
    ctl[f"p{p}_none_T5"] = mch[f"p{p}"]
    f = RES/f"multichannel_activation_ablation_p{p}.json"
    if f.exists():
        ctl[f"p{p}_none_T10"] = json.load(f.open())["auroc"]["T10"]
res = ctl | res
print("\n===== INPUT LAYERNORM (AUROC) =====")
order = list(res); names = [s for s in res[order[0]] if s != "mean"] + ["mean"]
w = max(len(s) for s in names)
print(" " * (w + 2) + "".join(f"{t:>14}" for t in order))
for s in names:
    print(f"{s:<{w}}  " + "".join(f"{res[t][s]:>14.4f}" for t in order))
