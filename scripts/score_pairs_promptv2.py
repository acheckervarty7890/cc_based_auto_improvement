"""Fit on base + the v2 (minimal-edit) pairs and score oig_omission, against the v1 controls."""
import json, pickle, sys
from pathlib import Path
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
from agentic_redteam.evaluation import evaluate_probe
from agentic_redteam.retrain import (_base_activation_cache_paths, _cpu_unpickle,
    _dev_activation_cache_path, _infer_probe_spec, _load_dev_dataset,
    _resolve_ensemble_seeds, _train_with_cached_base_activations, stable_train_test_split)
from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

BP  = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE= REPO/"data/instructions_llama70b_50.jsonl"; DEV=REPO/"dev_samples/instructions"
BC  = REPO/"results_instructions_gemma27b_shared/base_activations"
EC  = REPO/"results_instructions_gemma27b_shared/eval_activations"
SRC = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5/redteam_postprocessed_iter5.jsonl"
V2  = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5/pairs_promptv2.jsonl"

with BP.open("rb") as f: bp=_cpu_unpickle(f)
pos,neg=bp.pos_class_label,bp.neg_class_label; spec=_infer_probe_spec(bp)
seeds=_resolve_ensemble_seeds(42,10); TS=0.0; C=V=True

bd=LabelledDataset.load_from(Path(BASE),pos_class_label=pos,neg_class_label=neg,
    combine_consecutive_messages=C,convert_tool_to_assistant=V)
btr,bva=stable_train_test_split(bd,test_size=TS,split_field=None,seed=42)
btc,bvc=_base_activation_cache_paths(BC,BASE,bp.model_name,bp.layer,42,TS,None,C,V,1.0)
dv,dfiles=_load_dev_dataset(DEV,pos,neg,C,V,verbose=False)
dvc=_dev_activation_cache_path(BC,dfiles,bp.model_name,bp.layer,C,V)

rows=[json.loads(l) for l in SRC.open()]; n=len(rows)//2
finds=rows[:n]
v2=[json.loads(l) for l in V2.open()]
combined=finds+v2                       # 16 finds (negative) + 16 minimal-edit pairs (positive)
ds=LabelledDataset(
  inputs=[[TLMessage(role=m["role"],content=m["content"]) for m in r["inputs"]] for r in combined],
  ids=[r["id"] for r in combined],
  other_fields={"labels":[r["label"] for r in combined]})
rtr,rva=stable_train_test_split(ds,test_size=TS,split_field=None,seed=42)
print(f"training on base {len(bd)} + red-team {len(combined)} (16 finds + 16 v2 pairs)",flush=True)
probe=_train_with_cached_base_activations(base_train=btr,base_val=bva,redteam_train=rtr,
  redteam_val=rva,dev_val=dv,model_name=bp.model_name,layer=bp.layer,probe_spec=spec,
  pos_class_label=pos,neg_class_label=neg,probe_description=bp.description,
  base_train_cache=btc,base_val_cache=bvc,dev_val_cache=dvc,redteam_cache_dir=BC,
  combine_consecutive_messages=C,convert_tool_to_assistant=V,seed=42,ensemble_seeds=seeds,verbose=True)
q=Path("/tmp/pair_study/v2.pkl"); q.parent.mkdir(exist_ok=True)
with q.open("wb") as f: pickle.dump(probe,f)
df=evaluate_probe(probe_path=q,eval_dataset_dir=REPO/"eval_sets/instructions",
  activations_cache_dir=EC,splits=None,max_samples=None,seed=42,
  combine_consecutive_messages=C,convert_tool_to_assistant=V)
df.to_csv(REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5/eval_promptv2.csv",index=False)
print(df.to_string(index=False))
