"""base 50 + the split's own 32 dev rows + the run's 33 couples.

The two halves of the story tested together: in-distribution data is worth +0.072 on this split
and the couples are worth -0.084. This asks whether the couples still poison a probe that has
good data available, or whether good data absorbs them.

Validation is the other six dev splits, as in the base+dev run, so the 32 omission dev rows are
training data here and nothing validates on the split. All activations are cached.
"""
import json, pickle, sys, tempfile
from pathlib import Path
import torch
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
SC = Path("/tmp/claude-1000/-workspace-cc-based-auto-improvement/0998e5c8-a752-408d-9842-7fe74f4434a1/scratchpad/devomi_cache")
from agentic_redteam.evaluation import evaluate_probe
from agentic_redteam.retrain import (_base_activation_cache_paths, _cpu_unpickle,
    _infer_probe_spec, _load_dev_dataset, _resolve_ensemble_seeds,
    _train_with_cached_base_activations, stable_train_test_split)
from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

RES = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
PRB = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP  = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE= REPO/"data/instructions_llama70b_50.jsonl"
DEVO= REPO/"dev_samples/instructions/oig_omission.jsonl"; DEV = REPO/"dev_samples/instructions"
BC  = REPO/"results_instructions_gemma27b_shared/base_activations"
EC  = REPO/"results_instructions_gemma27b_shared/eval_activations"
OUT = REPO/"results_instructions_gemma27b_shared/devomission_probe"
COUPLES = sys.argv[1] if len(sys.argv) > 1 else str(PRB/"redteam_postprocessed_iter5.jsonl")
TAG = sys.argv[2] if len(sys.argv) > 2 else "orig"

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(42, 10); C = V = True

base = LabelledDataset.load_from(Path(BASE), pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btr, _ = stable_train_test_split(base, test_size=0.0, split_field=None, seed=42)
devo = LabelledDataset.load_from(DEVO, pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btc, _bvc = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, 42, 0.0, None, C, V, 1.0)

tmp = Path(tempfile.mkdtemp())
for q in sorted(DEV.glob("*.jsonl")):
    if q.name != "oig_omission.jsonl": (tmp/q.name).symlink_to(q)
rest, _ = _load_dev_dataset(tmp, pos, neg, C, V, verbose=False)

a = torch.load(btc, map_location="cpu", weights_only=False)
b = torch.load(SC/"omission_train.pt", map_location="cpu", weights_only=False)
W = max(a["activations"].shape[1], b["activations"].shape[1])
def pad(d):
    n, s, h = d["activations"].shape
    if s == W: return d
    act = torch.zeros(n, W, h, dtype=d["activations"].dtype); act[:, :s] = d["activations"]
    am = torch.zeros(n, W, dtype=d["attention_mask"].dtype); am[:, :s] = d["attention_mask"]
    ii = torch.zeros(n, W, dtype=d["input_ids"].dtype); ii[:, :s] = d["input_ids"]
    return {"activations": act, "attention_mask": am, "input_ids": ii,
            "layer": d["layer"], "model_name": d["model_name"]}
a, b = pad(a), pad(b)
merged = tmp/"base_plus_devomission.pt"
torch.save({"activations": torch.cat([a["activations"], b["activations"]]),
            "attention_mask": torch.cat([a["attention_mask"], b["attention_mask"]]),
            "input_ids": torch.cat([a["input_ids"], b["input_ids"]]),
            "layer": a["layer"], "model_name": a["model_name"]}, merged)
train = LabelledDataset.concatenate([btr, devo])
tr, empty = stable_train_test_split(train, test_size=0.0, split_field=None, seed=42)

rows = [json.loads(l) for l in Path(COUPLES).open()]
rt = LabelledDataset(
    inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rows],
    ids=[r["id"] for r in rows], other_fields={"labels": [r["label"] for r in rows]})
rtr, rva = stable_train_test_split(rt, test_size=0.0, split_field=None, seed=42)
print(f"train {len(tr)} (50 base + 32 dev omission) + {len(rows)} couple rows [{TAG}]"
      f" | validation {len(rest)} (other six dev splits)", flush=True)
probe = _train_with_cached_base_activations(base_train=tr, base_val=empty, redteam_train=rtr,
    redteam_val=rva, dev_val=rest, model_name=bp.model_name, layer=bp.layer, probe_spec=spec,
    pos_class_label=pos, neg_class_label=neg, probe_description=bp.description,
    base_train_cache=merged, base_val_cache=tmp/"unused.pt", dev_val_cache=SC/"dev_rest.pt",
    redteam_cache_dir=BC, combine_consecutive_messages=C, convert_tool_to_assistant=V,
    seed=42, ensemble_seeds=seeds, verbose=False)
q = OUT/f"probe_base_dev_couples_{TAG}.pkl"
with q.open("wb") as f: pickle.dump(probe, f)
df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO/"eval_sets/instructions",
    activations_cache_dir=EC, splits=None, max_samples=None, seed=42,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
df.to_csv(OUT/f"eval_base_dev_couples_{TAG}.csv", index=False)
print(df.to_string(index=False))
