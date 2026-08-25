"""Which half of a couple carries the effect — for the v3 run's 33, and for their shortened twins.

Four fits, everything else held: base + the finds alone, base + the generated partners alone,
at full length and compressed. Each single-sided set is one class only, so the 25/25 base
becomes 25/58 or 58/25; that skew is part of what is being measured, not a flaw to correct.
All activations are cached, so no extraction model is loaded.
"""
import json, pickle, sys
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

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(42, 10); TS = 0.0; C = V = True
base = LabelledDataset.load_from(Path(BASE), pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btr, bva = stable_train_test_split(base, test_size=TS, split_field=None, seed=42)
btc, bvc = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, 42, TS, None, C, V, 1.0)
dv, dfiles = _load_dev_dataset(DEV, pos, neg, C, V, verbose=False)
dvc = _dev_activation_cache_path(BC, dfiles, bp.model_name, bp.layer, C, V)

def load(path):
    rows = [json.loads(l) for l in Path(path).open()]
    return ([r for r in rows if r["label"] == "negative"],
            [r for r in rows if r["label"] == "positive"])
orig_s, orig_g = load(PRB/"redteam_postprocessed_iter5.jsonl")
short_s, short_g = load(RES/"shortened_paired.jsonl")

def ds(rows):
    return LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rows],
        ids=[r["id"] for r in rows],
        other_fields={"labels": [r["label"] for r in rows]})

def fit(rows, tag):
    rtr, rva = stable_train_test_split(ds(rows), test_size=TS, split_field=None, seed=42)
    lbl = rows[0]["label"]
    print(f"\n=== {tag}: base 50 (25/25) + {len(rows)} rows, all '{lbl}' ===", flush=True)
    p = _train_with_cached_base_activations(base_train=btr, base_val=bva, redteam_train=rtr,
        redteam_val=rva, dev_val=dv, model_name=bp.model_name, layer=bp.layer, probe_spec=spec,
        pos_class_label=pos, neg_class_label=neg, probe_description=bp.description,
        base_train_cache=btc, base_val_cache=bvc, dev_val_cache=dvc, redteam_cache_dir=BC,
        combine_consecutive_messages=C, convert_tool_to_assistant=V, seed=42,
        ensemble_seeds=seeds, verbose=False)
    q = Path("/tmp/pair_study")/f"h3_{tag}.pkl"; q.parent.mkdir(exist_ok=True)
    with q.open("wb") as f: pickle.dump(p, f)
    df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO/"eval_sets/instructions",
        activations_cache_dir=EC, splits=None, max_samples=None, seed=42,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    q.unlink(missing_ok=True)
    df.to_csv(RES/f"eval_half_{tag}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return {r["dataset"]: round(float(r["auroc"]), 4) for _, r in df.iterrows()}

out = {}
for tag, rows in (("orig_sources", orig_s), ("orig_partners", orig_g),
                  ("short_sources", short_s), ("short_partners", short_g)):
    out[tag] = fit(rows, tag)
json.dump(out, open(RES/"half_ablation_v3.json", "w"), indent=1)
print("\n=== oig_omission ===")
for k, v in out.items(): print(f"  {k:<16} {v['oig_omission']:.4f}   mean {v['mean']:.4f}")
