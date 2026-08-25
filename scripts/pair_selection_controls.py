"""Controls for the LOO study: no pairs at all, and only the pairs LOO called helpful."""
import json,sys
from pathlib import Path
REPO=Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0,str(REPO/"src"))
sys.argv=["pair_selection_controls"]
import importlib.util
spec=importlib.util.spec_from_file_location("pss",REPO/"scripts/pair_selection_study.py")
pss=importlib.util.module_from_spec(spec); spec.loader.exec_module(pss)
# rebuild the same context main() does, then fit chosen subsets
res=json.load(open(REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5/pair_selection_oig_omission.json"))
helping=[p["i"] for p in res["pairs"] if p["delta"]<0]
hurting=[p["i"] for p in res["pairs"] if p["delta"]>0]
print("helping:",helping); print("hurting:",hurting)
import pickle
from agentic_redteam.evaluation import evaluate_probe
from agentic_redteam.retrain import (_base_activation_cache_paths,_cpu_unpickle,
  _dev_activation_cache_path,_infer_probe_spec,_load_dev_dataset,_resolve_ensemble_seeds,
  _train_with_cached_base_activations,stable_train_test_split)
from tuberlens.interfaces.dataset import LabelledDataset
BP=REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE=REPO/"data/instructions_llama70b_50.jsonl"; DEV=REPO/"dev_samples/instructions"
BCACHE=REPO/"results_instructions_gemma27b_shared/base_activations"
ECACHE=REPO/"results_instructions_gemma27b_shared/eval_activations"
with BP.open("rb") as f: bp=_cpu_unpickle(f)
pos,neg=bp.pos_class_label,bp.neg_class_label; spec_=_infer_probe_spec(bp)
seeds=_resolve_ensemble_seeds(42,10); TS=0.0; C=V=True
bd=LabelledDataset.load_from(Path(BASE),pos_class_label=pos,neg_class_label=neg,
    combine_consecutive_messages=C,convert_tool_to_assistant=V)
btr,bva=stable_train_test_split(bd,test_size=TS,split_field=None,seed=42)
btc,bvc=_base_activation_cache_paths(BCACHE,BASE,bp.model_name,bp.layer,42,TS,None,C,V,1.0)
dv,dfiles=_load_dev_dataset(DEV,pos,neg,C,V,verbose=False)
dvc=_dev_activation_cache_path(BCACHE,dfiles,bp.model_name,bp.layer,C,V)
pairs=pss._load_pairs(REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5/redteam_postprocessed_iter5.jsonl")
def score(keep,tag):
    rt_tr=rt_va=None
    if keep:
        rows=[r for i in keep for r in pairs[i]]
        rt=pss._dataset_from_rows(rows)
        rt_tr,rt_va=stable_train_test_split(rt,test_size=TS,split_field=None,seed=42)
    p=_train_with_cached_base_activations(base_train=btr,base_val=bva,
      redteam_train=rt_tr,redteam_val=rt_va,dev_val=dv,model_name=bp.model_name,layer=bp.layer,
      probe_spec=spec_,pos_class_label=pos,neg_class_label=neg,probe_description=bp.description,
      base_train_cache=btc,base_val_cache=bvc,dev_val_cache=dvc,redteam_cache_dir=BCACHE,
      combine_consecutive_messages=C,convert_tool_to_assistant=V,seed=42,ensemble_seeds=seeds,verbose=False)
    q=Path("/tmp/pair_study")/f"s_{tag}.pkl"; q.parent.mkdir(exist_ok=True)
    with q.open("wb") as f: pickle.dump(p,f)
    df=evaluate_probe(probe_path=q,eval_dataset_dir=REPO/"eval_sets/instructions",
      activations_cache_dir=ECACHE,splits=["oig_omission"],max_samples=None,seed=42,
      combine_consecutive_messages=C,convert_tool_to_assistant=V)
    q.unlink(missing_ok=True)
    a=float(df[df["dataset"]=="oig_omission"].iloc[0]["auroc"]); print(f"  {tag}: {a:.4f}",flush=True); return a
print("scoring controls:",flush=True)
r={}
r["base_only"]=score([],"base_only")
r["helping_8"]=score(helping,"helping_8")
r["hurting_8"]=score(hurting,"hurting_8")
r["all_16"]=res["baseline_auroc"]
json.dump(r,open(REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5/pair_selection_controls.json","w"),indent=1)
print(json.dumps(r,indent=1))
