"""What do the couples' LABELS carry? Flip 10, 16 and all 33 of them and refit.

A couple is (find, generated partner) — the find labelled "does not follow", the partner
"follows". Flipping a couple swaps those two labels, leaving both conversations byte-identical.
Nothing about the text, the count, the class balance or the activations changes; only which
side of the boundary each row is claimed to be on. So this isolates the label from every other
property of the data, which no earlier ablation here does: the halves ablation changed the class
balance, the shortening and restyling arms changed the text.

The three flip sets are NESTED (10 ⊂ 16 ⊂ 33) off a seeded shuffle, so the results read as a
dose ladder on one sample rather than three unrelated draws.

Read against `base only` (no red-team data at all) and `0 flipped` (the untouched 33). If
flipping does little, the labels were not what the couples contributed. If flipping all 33 is
no worse than 0 flipped, they contributed nothing the probe could use.

Every activation is cached — flipping a label cannot change a conversation's activation — so
no extraction model is loaded and each arm costs one ensemble fit.
"""
import json, pickle, random, sys
from pathlib import Path
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
from agentic_redteam.evaluation import evaluate_probe
from agentic_redteam.retrain import (_base_activation_cache_paths, _cpu_unpickle,
    _dev_activation_cache_path, _infer_probe_spec, _load_dev_dataset, _resolve_ensemble_seeds,
    _train_with_cached_base_activations, stable_train_test_split)
from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

RES = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
PRB = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP  = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE= REPO/"data/instructions_llama70b_50.jsonl"; DEV = REPO/"dev_samples/instructions"
BC  = REPO/"results_instructions_gemma27b_shared/base_activations"
EC  = REPO/"results_instructions_gemma27b_shared/eval_activations"
SEED = 42

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(SEED, 10); TS = 0.0; C = V = True
base = LabelledDataset.load_from(Path(BASE), pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btr, bva = stable_train_test_split(base, test_size=TS, split_field=None, seed=SEED)
btc, bvc = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, SEED, TS, None, C, V, 1.0)
dv, dfiles, _sizes = _load_dev_dataset(DEV, pos, neg, C, V, verbose=False)
dvc = _dev_activation_cache_path(BC, dfiles, bp.model_name, bp.layer, C, V)

rows = [json.loads(l) for l in (PRB/"redteam_postprocessed_iter5.jsonl").open()]
srcs = [r for r in rows if r["label"] == "negative"]
gens = [r for r in rows if r["label"] == "positive"]
assert len(srcs) == len(gens) == 33, (len(srcs), len(gens))
order = list(range(33)); random.Random(f"labelflip:{SEED}").shuffle(order)

def ds(rows):
    return LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rows],
        ids=[r["id"] for r in rows],
        other_fields={"labels": [r["label"] for r in rows]})

def fit(tag, rt_rows, note):
    if rt_rows is None:
        rtr = rva = None; n = 0
    else:
        rtr, rva = stable_train_test_split(ds(rt_rows), test_size=TS, split_field=None, seed=SEED)
        n = len(rt_rows)
    print(f"\n=== {tag}: base 50 (25/25) + {n} red-team rows — {note} ===", flush=True)
    p = _train_with_cached_base_activations(base_train=btr, base_val=bva, redteam_train=rtr,
        redteam_val=rva, dev_val=dv, model_name=bp.model_name, layer=bp.layer, probe_spec=spec,
        pos_class_label=pos, neg_class_label=neg, probe_description=bp.description,
        base_train_cache=btc, base_val_cache=bvc, dev_val_cache=dvc, redteam_cache_dir=BC,
        combine_consecutive_messages=C, convert_tool_to_assistant=V, seed=SEED,
        ensemble_seeds=seeds, verbose=False)
    q = Path("/tmp/pair_study")/f"flip_{tag}.pkl"; q.parent.mkdir(exist_ok=True)
    with q.open("wb") as f: pickle.dump(p, f)
    df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO/"eval_sets/instructions",
        activations_cache_dir=EC, splits=None, max_samples=None, seed=SEED,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    q.unlink(missing_ok=True)
    df.to_csv(RES/f"eval_flip_{tag}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return {r["dataset"]: round(float(r["auroc"]), 4) for _, r in df.iterrows()}

def flipped(k):
    """The 66 rows with the first k couples of the shuffled order label-swapped."""
    hot = set(order[:k]); out = []
    for i, (s, g) in enumerate(zip(srcs, gens)):
        sl, gl = ("positive", "negative") if i in hot else ("negative", "positive")
        out.append({**s, "label": sl}); out.append({**g, "label": gl})
    return out

res = {}
res["base_only"] = fit("base_only", None, "no red-team data at all")
for k in (0, 10, 16, 33):
    tag = f"flip{k}"
    res[tag] = fit(tag, flipped(k), f"{k} of 33 couples label-swapped"
                   + (" (the untouched set)" if k == 0 else ""))
json.dump({"seed": SEED, "order": order, "auroc": res},
          open(RES/"label_flip_ablation.json", "w"), indent=1)

print("\n===== LABEL-FLIP LADDER (AUROC) =====")
splits = [s for s in res["flip0"] if s != "mean"] + ["mean"]
w = max(len(s) for s in splits)
print(" " * (w + 2) + "".join(f"{t:>12}" for t in res))
for s in splits:
    print(f"{s:<{w}}  " + "".join(f"{res[t][s]:>12.4f}" for t in res))
print(f"\nflipped couples (nested, seed {SEED}): first 10 = {sorted(order[:10])}")
print(f"                                       first 16 = {sorted(order[:16])}")
