"""Which half of a contrastive couple carries the effect?

Refit the D5 probe three more ways on the SAME 16 red-team finds:
  sources_only  - base + the 16 finds alone (all label `negative`), no partners
  gen_v1_only   - base + the 16 v1 (resemblance-prompt) partners alone (all `positive`)
  gen_v2_only   - base + the 16 v2 (minimal-edit) partners alone (all `positive`)
Everything else is held at the run's own settings. Eval on all 7 instruction splits.
"""
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
RES = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5"
SRC = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5/redteam_postprocessed_iter5.jsonl"
V2  = RES/"pairs_promptv2.jsonl"

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(42, 10); TS = 0.0; C = V = True

bd = LabelledDataset.load_from(Path(BASE), pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btr, bva = stable_train_test_split(bd, test_size=TS, split_field=None, seed=42)
btc, bvc = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, 42, TS, None, C, V, 1.0)
dv, dfiles, _sizes = _load_dev_dataset(DEV, pos, neg, C, V, verbose=False)
dvc = _dev_activation_cache_path(BC, dfiles, bp.model_name, bp.layer, C, V)

rows = [json.loads(l) for l in SRC.open()]; n = len(rows)//2
finds, gen_v1 = rows[:n], rows[n:]
gen_v2 = [json.loads(l) for l in V2.open()]
assert all(r["label"] == "negative" for r in finds)
assert all(r["label"] == "positive" for r in gen_v1 + gen_v2)

def ds_from(rs):
    return LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rs],
        ids=[r["id"] for r in rs],
        other_fields={"labels": [r["label"] for r in rs]})

def fit_and_score(rs, tag):
    d = ds_from(rs)
    rtr, rva = stable_train_test_split(d, test_size=TS, split_field=None, seed=42)
    lab = {}
    for r in rs: lab[r["label"]] = lab.get(r["label"], 0) + 1
    print(f"\n=== {tag}: base {len(bd)} + {len(rs)} red-team rows {lab} ===", flush=True)
    probe = _train_with_cached_base_activations(base_train=btr, base_val=bva, redteam_train=rtr,
        redteam_val=rva, dev_val=dv, model_name=bp.model_name, layer=bp.layer, probe_spec=spec,
        pos_class_label=pos, neg_class_label=neg, probe_description=bp.description,
        base_train_cache=btc, base_val_cache=bvc, dev_val_cache=dvc, redteam_cache_dir=BC,
        combine_consecutive_messages=C, convert_tool_to_assistant=V, seed=42,
        ensemble_seeds=seeds, verbose=False)
    q = Path("/tmp/pair_study")/f"h_{tag}.pkl"; q.parent.mkdir(exist_ok=True)
    with q.open("wb") as f: pickle.dump(probe, f)
    df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO/"eval_sets/instructions",
        activations_cache_dir=EC, splits=None, max_samples=None, seed=42,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    q.unlink(missing_ok=True)
    df.to_csv(RES/f"eval_half_{tag}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return {r["dataset"]: float(r["auroc"]) for _, r in df.iterrows()}

out = {}
out["sources_only"] = fit_and_score(finds,  "sources_only")
out["gen_v1_only"]  = fit_and_score(gen_v1, "gen_v1_only")
out["gen_v2_only"]  = fit_and_score(gen_v2, "gen_v2_only")
json.dump(out, open(RES/"half_ablation.json", "w"), indent=1)
print("\nwrote", RES/"half_ablation.json")
