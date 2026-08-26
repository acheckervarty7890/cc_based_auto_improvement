"""Does a couple's cosine distance predict whether it is worth training on?

Six fits, everything else held: base 50 plus the 10 couples whose two halves sit FURTHEST apart
in the probe's activation space, the 10 that sit CLOSEST, and a seeded random 10 as the control
that says whether any 10 couples behave this way.

Each selection is run twice, once ranked by the k=1 distance (the masked mean, the number the
ledger reports) and once by k=3 (adaptive pooling to three segments over the valid token span).
The two rankings are not the same, and k=3 is the more discriminating measure - it is the one
that showed these couples' differences are concentrated in the final third of the reply - so
whether the answer depends on which is used is itself worth knowing.

Reference points already measured, same base data and dev set: base alone 0.7975, all 33
couples 0.7135, the 23 the leave-one-out liked 0.7962, the 10 it disliked 0.6060.

Every activation is cached, so no extraction model is loaded.
"""
import json, pickle, random, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
from agentic_redteam.evaluation import evaluate_probe
from agentic_redteam.retrain import (_apply_message_transforms, _base_activation_cache_paths,
    _cpu_unpickle, _dev_activation_cache_path, _infer_probe_spec, _load_dev_dataset,
    _redteam_activation_cache_path, _resolve_ensemble_seeds,
    _train_with_cached_base_activations, stable_train_test_split)
from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

RES = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
PRB = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP  = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE= REPO/"data/instructions_llama70b_50.jsonl"; DEV = REPO/"dev_samples/instructions"
BC  = REPO/"results_instructions_gemma27b_shared/base_activations"
EC  = REPO/"results_instructions_gemma27b_shared/eval_activations"
SEED = 42; N = 10; C = V = True; TS = 0.0

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(SEED, 10)
base = LabelledDataset.load_from(Path(BASE), pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btr, bva = stable_train_test_split(base, test_size=TS, split_field=None, seed=SEED)
btc, bvc = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, SEED, TS, None, C, V, 1.0)
dv, dfiles, _sz = _load_dev_dataset(DEV, pos, neg, C, V, verbose=False)
dvc = _dev_activation_cache_path(BC, dfiles, bp.model_name, bp.layer, C, V)

rows = [json.loads(l) for l in (PRB/"redteam_postprocessed_iter5.jsonl").open()]
srcs = [r for r in rows if r["label"] == "negative"]
gens = [r for r in rows if r["label"] == "positive"]
assert len(srcs) == len(gens) == 33

def pooled(msgs, k):
    q = _redteam_activation_cache_path(BC, msgs, bp.model_name, bp.layer, C, V)
    d = torch.load(q, map_location="cpu", weights_only=False)
    a, m = d["activations"].float(), d["attention_mask"]
    L = int(m.sum(1)[0])
    span = a[0, :L].T.unsqueeze(0)
    return torch.nn.functional.adaptive_avg_pool1d(span, k).squeeze(0).T.reshape(-1).numpy()

ds_all = _apply_message_transforms(LabelledDataset(
    inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rows],
    ids=[r["id"] for r in rows], other_fields={"labels": [r["label"] for r in rows]}), C, V)
msgs_by_id = {r["id"]: m for r, m in zip(rows, ds_all.inputs)}

def distances(k):
    d = []
    for s, g in zip(srcs, gens):
        u, v = pooled(msgs_by_id[s["id"]], k), pooled(msgs_by_id[g["id"]], k)
        d.append(1.0 - float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))))
    return np.array(d)

def fit(tag, idx, note):
    picked = [x for i in sorted(idx) for x in (srcs[i], gens[i])]
    d = LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in picked],
        ids=[r["id"] for r in picked], other_fields={"labels": [r["label"] for r in picked]})
    rtr, rva = stable_train_test_split(d, test_size=TS, split_field=None, seed=SEED)
    print(f"\n=== {tag}: base 50 + {len(picked)} rows ({len(idx)} couples) - {note} ===", flush=True)
    p = _train_with_cached_base_activations(base_train=btr, base_val=bva, redteam_train=rtr,
        redteam_val=rva, dev_val=dv, model_name=bp.model_name, layer=bp.layer, probe_spec=spec,
        pos_class_label=pos, neg_class_label=neg, probe_description=bp.description,
        base_train_cache=btc, base_val_cache=bvc, dev_val_cache=dvc, redteam_cache_dir=BC,
        combine_consecutive_messages=C, convert_tool_to_assistant=V, seed=SEED,
        ensemble_seeds=seeds, verbose=False)
    q = Path("/tmp/pair_study")/f"cos_{tag}.pkl"; q.parent.mkdir(exist_ok=True)
    with q.open("wb") as f: pickle.dump(p, f)
    df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO/"eval_sets/instructions",
        activations_cache_dir=EC, splits=None, max_samples=None, seed=SEED,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    q.unlink(missing_ok=True)
    df.to_csv(RES/f"eval_cos_{tag}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return {r["dataset"]: round(float(r["auroc"]), 4) for _, r in df.iterrows()}

d1, d3 = distances(1), distances(3)
sel = {}
for k, d in (("k1", d1), ("k3", d3)):
    o = np.argsort(-d)
    sel[f"top10_{k}"] = sorted(int(i) for i in o[:N])
    sel[f"bot10_{k}"] = sorted(int(i) for i in o[-N:])
rnd = list(range(33)); random.Random(f"cosctl:{SEED}").shuffle(rnd)
sel["rand10"] = sorted(rnd[:N])
print("selections (couple indices, 0-32):")
for t, ix in sel.items():
    dd = d1 if t.endswith("k1") else (d3 if t.endswith("k3") else d1)
    print(f"  {t:<10} {ix}  mean cos(k1) {d1[ix].mean():.4f}  mean cos(k3) {d3[ix].mean():.4f}")
print(f"  overlap top10 k1 vs k3: {len(set(sel['top10_k1']) & set(sel['top10_k3']))}/10; "
      f"bot10 k1 vs k3: {len(set(sel['bot10_k1']) & set(sel['bot10_k3']))}/10")

res = {}
for tag in ("top10_k1", "bot10_k1", "top10_k3", "bot10_k3", "rand10"):
    res[tag] = fit(tag, sel[tag], {
        "top10_k1": "widest cosine gap, ranked on the masked mean",
        "bot10_k1": "tightest cosine gap, ranked on the masked mean",
        "top10_k3": "widest cosine gap, ranked on 3-segment pooling",
        "bot10_k3": "tightest cosine gap, ranked on 3-segment pooling",
        "rand10": "seeded random control"}[tag])
json.dump({"seed": SEED, "selections": sel,
           "cos_k1": [round(float(x), 4) for x in d1],
           "cos_k3": [round(float(x), 4) for x in d3], "auroc": res},
          open(RES/"cosine_extremes_ablation.json", "w"), indent=1)

print("\n===== COSINE-EXTREMES LADDER (AUROC) =====")
splits = [s for s in res["rand10"] if s != "mean"] + ["mean"]
w = max(len(s) for s in splits)
print(" " * (w + 2) + "".join(f"{t:>11}" for t in res))
for s in splits:
    print(f"{s:<{w}}  " + "".join(f"{res[t][s]:>11.4f}" for t in res))
