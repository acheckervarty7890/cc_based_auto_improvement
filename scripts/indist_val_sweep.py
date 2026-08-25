"""Every training set from today, refit with IN-DISTRIBUTION validation.

All the numbers so far early-stopped against the 436-row dev set — six of whose seven splits
are not the split being scored. This repeats the whole set with validation replaced by
oig_omission's own 32 dev rows, and nothing else changed, so the difference is attributable to
what the fit selects its checkpoint on.

Two things to hold in mind reading it: 32 validation rows is a thin signal for early stopping,
and those 32 rows are in-distribution for the eval split, so a gain here is partly a gain from
knowing what the target looks like — which is the point, but it is not free.

All activations come from caches; no extraction model is loaded.
"""
import json, pickle, sys
from pathlib import Path
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
SC = Path("/tmp/claude-1000/-workspace-cc-based-auto-improvement/0998e5c8-a752-408d-9842-7fe74f4434a1/scratchpad/devomi_cache")
from agentic_redteam.evaluation import evaluate_probe
from agentic_redteam.retrain import (_base_activation_cache_paths, _cpu_unpickle,
    _infer_probe_spec, _resolve_ensemble_seeds, _train_with_cached_base_activations,
    stable_train_test_split)
from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

RES = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
PRB = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP  = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE= REPO/"data/instructions_llama70b_50.jsonl"
DEVO= REPO/"dev_samples/instructions/oig_omission.jsonl"
BC  = REPO/"results_instructions_gemma27b_shared/base_activations"
EC  = REPO/"results_instructions_gemma27b_shared/eval_activations"

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(42, 10); TS = 0.0; C = V = True
base = LabelledDataset.load_from(Path(BASE), pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btr, bva = stable_train_test_split(base, test_size=TS, split_field=None, seed=42)
btc, bvc = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, 42, TS, None, C, V, 1.0)
devval = LabelledDataset.load_from(DEVO, pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
assert len(devval) == 32

def load(path):
    rows = [json.loads(l) for l in Path(path).open()]
    return ([r for r in rows if r["label"] == "negative"],
            [r for r in rows if r["label"] == "positive"])
orig_s, orig_g   = load(PRB/"redteam_postprocessed_iter5.jsonl")
short_s, short_g = load(RES/"shortened_paired.jsonl")
aug_s, aug_g     = load(RES/"variations_paired.jsonl")

CONFIGS = [("base_only",        []),
           ("orig_both",        orig_s + orig_g),
           ("orig_sources",     orig_s),
           ("orig_partners",    orig_g),
           ("short_both",       short_s + short_g),
           ("short_sources",    short_s),
           ("short_partners",   short_g),
           ("augmented_both",   aug_s + aug_g)]

def ds(rows):
    return LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rows],
        ids=[r["id"] for r in rows],
        other_fields={"labels": [r["label"] for r in rows]})

out = {}
for tag, rows in CONFIGS:
    rtr = rva = None
    if rows: rtr, rva = stable_train_test_split(ds(rows), test_size=TS, split_field=None, seed=42)
    print(f"\n=== {tag}: base 50 + {len(rows)} red-team rows, validated on 32 in-distribution "
          f"dev rows ===", flush=True)
    probe = _train_with_cached_base_activations(base_train=btr, base_val=bva, redteam_train=rtr,
        redteam_val=rva, dev_val=devval, model_name=bp.model_name, layer=bp.layer,
        probe_spec=spec, pos_class_label=pos, neg_class_label=neg,
        probe_description=bp.description, base_train_cache=btc, base_val_cache=bvc,
        dev_val_cache=SC/"omission_train.pt", redteam_cache_dir=BC,
        combine_consecutive_messages=C, convert_tool_to_assistant=V, seed=42,
        ensemble_seeds=seeds, verbose=False)
    q = Path("/tmp/pair_study")/f"iv_{tag}.pkl"; q.parent.mkdir(exist_ok=True)
    with q.open("wb") as f: pickle.dump(probe, f)
    df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO/"eval_sets/instructions",
        activations_cache_dir=EC, splits=None, max_samples=None, seed=42,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    q.unlink(missing_ok=True)
    df.to_csv(RES/f"eval_indistval_{tag}.csv", index=False)
    r = {x["dataset"]: round(float(x["auroc"]), 4) for _, x in df.iterrows()}
    r["acc_omission"] = round(float(df[df["dataset"]=="oig_omission"].iloc[0]["accuracy"]), 4)
    out[tag] = r
    print(f"  oig_omission {r['oig_omission']:.4f}   mean {r['mean']:.4f}", flush=True)

json.dump(out, open(RES/"indist_val_sweep.json", "w"), indent=1)
print("\n=== in-distribution validation (32 oig_omission dev rows) ===")
print(f"{'config':<18}{'oig_omission':>14}{'mean':>9}")
for k, v in out.items(): print(f"{k:<18}{v['oig_omission']:>14.4f}{v['mean']:>9.4f}")
