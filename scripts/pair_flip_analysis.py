"""Per-row effect of the v3 run's helpful vs harmful couples on the oig_omission eval split.

Fits three probes on identical settings — base only, base + the 23 couples LOO called helpful,
base + the 10 it called harmful — and records every eval row's score under each, so a row can
be classified as gained (wrong under base, right after) or lost (right under base, wrong after).
Activations all come from existing caches; no extraction LLM is loaded.
"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, torch
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
sys.argv = ["flips_v3"]
spec = importlib.util.spec_from_file_location("pss", REPO/"scripts/pair_selection_study.py")
pss = importlib.util.module_from_spec(spec); spec.loader.exec_module(pss)
from agentic_redteam.retrain import (_base_activation_cache_paths, _cpu_unpickle,
    _dev_activation_cache_path, _infer_probe_spec, _load_dev_dataset,
    _resolve_ensemble_seeds, _train_with_cached_base_activations, stable_train_test_split)
from tuberlens.interfaces.dataset import LabelledDataset

RES = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
PRB = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP  = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE= REPO/"data/instructions_llama70b_50.jsonl"; DEV = REPO/"dev_samples/instructions"
BC  = REPO/"results_instructions_gemma27b_shared/base_activations"
SPL = REPO/"eval_sets/instructions/oig_omission.jsonl"
EVB = REPO/"results_instructions_gemma27b_shared/eval_activations/oig_omission-acts_full.pt"
OUT = RES/"flips_oig_omission.json"

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
ps = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(42, 10); TS = 0.0; C = V = True
THRESH = 0.5

base = LabelledDataset.load_from(BASE, pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btr, bva = stable_train_test_split(base, test_size=TS, split_field=None, seed=42)
btc, bvc = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, 42, TS, None, C, V, 1.0)
dv, dfiles, _sizes = _load_dev_dataset(DEV, pos, neg, C, V, verbose=False)
dvc = _dev_activation_cache_path(BC, dfiles, bp.model_name, bp.layer, C, V)
pairs = pss._load_pairs(PRB/"redteam_postprocessed_iter5.jsonl")

loo = json.load(open(RES/"pair_selection_oig_omission.json"))
helping = [p["i"] for p in loo["pairs"] if p["delta"] < 0]
hurting = [p["i"] for p in loo["pairs"] if p["delta"] > 0]

full = LabelledDataset.load_from(SPL, pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
blob = torch.load(EVB, map_location="cpu", weights_only=False)
evalds = full.assign(activations=blob["activations"], attention_mask=blob["attention_mask"],
                     input_ids=blob["input_ids"])
y = np.array(full.other_fields["labels"], dtype=int)

def fit_predict(keep, tag):
    rtr = rva = None
    if keep:
        rt = pss._dataset_from_rows([r for i in keep for r in pairs[i]])
        rtr, rva = stable_train_test_split(rt, test_size=TS, split_field=None, seed=42)
    print(f"  fitting {tag} (base 50 + {2*len(keep)} red-team rows)", flush=True)
    probe = _train_with_cached_base_activations(base_train=btr, base_val=bva, redteam_train=rtr,
        redteam_val=rva, dev_val=dv, model_name=bp.model_name, layer=bp.layer, probe_spec=ps,
        pos_class_label=pos, neg_class_label=neg, probe_description=bp.description,
        base_train_cache=btc, base_val_cache=bvc, dev_val_cache=dvc, redteam_cache_dir=BC,
        combine_consecutive_messages=C, convert_tool_to_assistant=V, seed=42,
        ensemble_seeds=seeds, verbose=False)
    p = np.asarray(probe.predict_proba(evalds))
    if p.ndim == 2: p = p[:, -1]
    from sklearn.metrics import roc_auc_score
    print(f"    {tag}: AUROC {roc_auc_score(y, p):.4f}  acc {(( p>=THRESH).astype(int)==y).mean():.3f}",
          flush=True)
    return p.astype(float)

scores = {"base": fit_predict([], "base"),
          "help": fit_predict(helping, f"help{len(helping)}"),
          "hurt": fit_predict(hurting, f"hurt{len(hurting)}")}
raw = [json.loads(l) for l in SPL.open()]
rows = []
for i, r in enumerate(raw):
    rows.append({"i": i, "orig": r["original_text"],
                 "label": r["labels"], "y": int(y[i]),
                 "inputs": json.loads(r["inputs"]) if isinstance(r["inputs"], str) else r["inputs"],
                 "j1": r.get("judge_1_reasoning", ""), "j2": r.get("judge_2_reasoning", ""),
                 "base": round(scores["base"][i], 4), "help": round(scores["help"][i], 4),
                 "hurt": round(scores["hurt"][i], 4)})
json.dump({"threshold": THRESH, "helping": helping, "hurting": hurting, "rows": rows},
          open(OUT, "w"), ensure_ascii=False)
for arm in ("help", "hurt"):
    ok_b = (scores["base"] >= THRESH).astype(int) == y
    ok_a = (scores[arm] >= THRESH).astype(int) == y
    print(f"{arm}: gained {int((~ok_b & ok_a).sum())}  lost {int((ok_b & ~ok_a).sum())}  "
          f"acc {ok_b.mean():.3f} -> {ok_a.mean():.3f}")
print("wrote", OUT)
