"""Nonlinearity and temperature on the p-channel readout, at p=16.

Two questions, one run.

1. **Which activation on the per-token channel scores** -- and, separately, whether the softmax
   POOLING WEIGHTS should be computed from the activated scores (the literal proposal) or from
   the raw linear ones. The weights are softmax(z/T) of the scores themselves, so any activation
   that squashes the negative tail (gelu, silu, relu) replaces a denominator term of
   exp(-8/5)=0.20 with exp(0)=1 at every low-scoring position; the top token's share falls and
   pooling drifts from near-max toward near-mean, which this project already measured to be
   costly. Taking the weights from the raw scores decouples where to look from what to report.
   The `gelu_act` arm exists to measure that cost rather than assume it.

2. **Temperature.** Every fit in this ladder used T=5, inherited from the base probe's spec, and
   T controls the same max-vs-mean tradeoff directly, with one scalar and no new parameters.

Control for both is the existing identity p=16 arm (T=5) from multichannel_head_ablation.json.

Same base 50, same 33 couples, same 436-row dev set, same 10 pinned ensemble seeds; all
activations cached, so no extraction model is loaded.
"""
import json, os, pickle, sys
from pathlib import Path
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
os.environ["PROBE_FUSED_ENSEMBLE"] = "0"
import torch
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
# Channel count the whole sweep runs at (default 16, the band where the linear
# ladder was highest). Results and probes are named off it, so two p values never
# share a file.
P = int(sys.argv[1]) if len(sys.argv) > 1 else 16

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(SEED, 10)
HP = dict(spec.hyperparams)
print(f"p = {P}\nbase hyperparameters: {HP}")

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


def concentration(act, wf, temperature=5, n=24):
    """Mean share of the pooling weight taken by each channel's top token, at initialization.

    Direct measurement of the flattening argument: identity is the stock-head baseline, and any
    variant whose share is much lower has moved from near-max toward near-mean.
    """
    torch.manual_seed(0)
    x = TRAIN.other_fields["activations"][:n]
    mask = TRAIN.other_fields["attention_mask"][:n].bool()
    m = MultiChannelLinearThenSoftmax(x.shape[-1], temperature=temperature, n_channels=P,
                                      channel_activation=act, channel_weights_from=wf)
    m = m.to(device=x.device, dtype=x.dtype)
    with torch.no_grad():
        mm = mask.unsqueeze(-1)
        raw = m.linear(x)
        z = m.act(raw).masked_fill(~mm, 0)
        src = z if wf == "activated" else raw.masked_fill(~mm, 0)
        w = torch.softmax(src.masked_fill(~mm, float("-inf")).float() / temperature, dim=1)
    return float(w.max(dim=1).values.mean()), float(mask.sum(1).float().mean())


def fit(tag, act="identity", wf="activated", temperature=5):
    print(f"\n=== {tag} (p={P}, {act}, weights from {wf}, T={temperature}) ===", flush=True)
    members = []
    for s in seeds:
        seed_everything(s)
        hp = HP | {"n_channels": P, "channel_activation": act,
                   "channel_weights_from": wf, "temperature": temperature}
        probe = PytorchProbe(hyper_params=hp, model_name=bp.model_name, layer=bp.layer,
            pos_class_label=pos, neg_class_label=neg, description=bp.description,
            _classifier=PytorchAdamClassifier(training_args=hp,
                                              probe_architecture=MultiChannelLinearThenSoftmax))
        probe.fit(TRAIN, VAL)
        members.append(probe)
    probe = EnsembleProbe.from_members(members, list(seeds))
    q = RES/f"probe_mca_p{P}_{tag}.pkl"
    with q.open("wb") as f: pickle.dump(probe, f)
    df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO/"eval_sets/instructions",
        activations_cache_dir=EC, splits=None, max_samples=None, seed=SEED,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    df.to_csv(RES/f"eval_mca_p{P}_{tag}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return {r["dataset"]: round(float(r["auroc"]), 4) for _, r in df.iterrows()}


MCH = RES/"multichannel_head_ablation.json"
HAVE_CONTROL = MCH.exists() and f"p{P}" in json.load(MCH.open())["auroc"]

ARMS = [
    # The identity/T=5 control is the plain p-channel arm. Reuse it from the channel ladder
    # where that p was already fit (deterministic under the same pinned seeds); fit it here
    # for any p the ladder skipped, so every sweep has its own control either way.
    *([] if HAVE_CONTROL else [("identity", dict())]),
    ("gelu_raw",   dict(act="gelu",       wf="raw")),
    ("leaky_raw",  dict(act="leaky_relu", wf="raw")),
    ("tanh_raw",   dict(act="tanh",       wf="raw")),
    ("relu2_raw",  dict(act="relu2",      wf="raw")),
    ("gelu_act",   dict(act="gelu",       wf="activated")),
    ("T1",         dict(temperature=1)),
    ("T2",         dict(temperature=2)),
    ("T10",        dict(temperature=10)),
    ("T25",        dict(temperature=25)),
]

print("\n=== top-token weight share at init (identity T=5 is the stock baseline) ===")
conc = {}
for tag, kw in [("identity", {})] + ARMS:
    share, ln = concentration(kw.get("act", "identity"), kw.get("wf", "activated"),
                              kw.get("temperature", 5))
    conc[tag] = round(share, 4)
    print(f"  {tag:<11} top-token weight {share:.3f}   (mean valid length {ln:.0f})")

OUT = RES/f"multichannel_activation_ablation_p{P}.json"
res = json.load(OUT.open())["auroc"] if OUT.exists() else {}
for tag, kw in ARMS:
    res[tag] = fit(tag, **kw)
    json.dump({"seed": SEED, "p": P, "concentration": conc, "auroc": res}, OUT.open("w"), indent=1)

if HAVE_CONTROL:
    res = {"identity": json.load(MCH.open())["auroc"][f"p{P}"]} | res
order = list(res)
print(f"\n===== ACTIVATION + TEMPERATURE ON THE p={P} READOUT (AUROC) =====")
names = [s for s in res[order[0]] if s != "mean"] + ["mean"]
w = max(len(s) for s in names)
print(" " * (w + 2) + "".join(f"{t:>11}" for t in order))
for s in names:
    print(f"{s:<{w}}  " + "".join(f"{res[t][s]:>11.4f}" for t in order))
print("\ntop-token weight share at init: " + "  ".join(f"{t}={conc[t]:.3f}" for t in conc))
