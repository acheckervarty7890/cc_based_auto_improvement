"""Base 50 + the split's own 32 dev rows, against base 50 alone — is it the DATA?

The grouped CV puts the achievable AUROC on oig_omission at ~0.91 with ~90 in-distribution
rows. Every arm trains on 50 out-of-distribution base rows plus red-team couples and lands
0.71-0.84. This asks the cheapest version of the question in between: does adding 32 rows
drawn from the split's own distribution move the base probe toward the ceiling, where red-team
couples have never managed to?

Validation is the other six dev splits, exactly as the real runs use. Activations are slices
of existing blobs, padded to a common width; no LLM is loaded.
"""
import pickle, sys, tempfile
from pathlib import Path
import torch
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
SC = Path("/tmp/claude-1000/-workspace-cc-based-auto-improvement/0998e5c8-a752-408d-9842-7fe74f4434a1/scratchpad/devomi_cache")
from agentic_redteam.evaluation import evaluate_probe
from agentic_redteam.retrain import (_base_activation_cache_paths, _cpu_unpickle,
    _infer_probe_spec, _load_dev_dataset, _resolve_ensemble_seeds,
    _train_with_cached_base_activations, stable_train_test_split)
from tuberlens.interfaces.dataset import LabelledDataset

BP = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE = REPO/"data/instructions_llama70b_50.jsonl"
DEVO = REPO/"dev_samples/instructions/oig_omission.jsonl"
DEV = REPO/"dev_samples/instructions"
BC = REPO/"results_instructions_gemma27b_shared/base_activations"
EC = REPO/"results_instructions_gemma27b_shared/eval_activations"
OUT = REPO/"results_instructions_gemma27b_shared/devomission_probe"

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(42, 10); C = V = True

base = LabelledDataset.load_from(BASE, pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btr, bva = stable_train_test_split(base, test_size=0.0, split_field=None, seed=42)
devo = LabelledDataset.load_from(DEVO, pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btc, _bvc = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, 42, 0.0, None, C, V, 1.0)

# validation: the other six dev splits, the blob the real runs validate against minus omission
tmp = Path(tempfile.mkdtemp())
for q in sorted(DEV.glob("*.jsonl")):
    if q.name != "oig_omission.jsonl": (tmp/q.name).symlink_to(q)
rest, _ = _load_dev_dataset(tmp, pos, neg, C, V, verbose=False)

a = torch.load(btc, map_location="cpu", weights_only=False)              # 50 x 135
b = torch.load(SC/"omission_train.pt", map_location="cpu", weights_only=False)  # 32 x 436
W = max(a["activations"].shape[1], b["activations"].shape[1])
def pad(d):
    n, s, h = d["activations"].shape
    if s == W: return d
    act = torch.zeros(n, W, h, dtype=d["activations"].dtype); act[:, :s] = d["activations"]
    am  = torch.zeros(n, W, dtype=d["attention_mask"].dtype);  am[:, :s]  = d["attention_mask"]
    ii  = torch.zeros(n, W, dtype=d["input_ids"].dtype);       ii[:, :s]  = d["input_ids"]
    return {"activations": act, "attention_mask": am, "input_ids": ii,
            "layer": d["layer"], "model_name": d["model_name"]}
a, b = pad(a), pad(b)
merged = tmp/"base_plus_devomission.pt"
torch.save({"activations": torch.cat([a["activations"], b["activations"]]),
            "attention_mask": torch.cat([a["attention_mask"], b["attention_mask"]]),
            "input_ids": torch.cat([a["input_ids"], b["input_ids"]]),
            "layer": a["layer"], "model_name": a["model_name"]}, merged)

train = LabelledDataset.concatenate([btr, devo])
assert len(train) == 82, len(train)
tr, empty = stable_train_test_split(train, test_size=0.0, split_field=None, seed=42)
print(f"train {len(tr)} (50 base + 32 dev oig_omission) | validation {len(rest)} (other six dev splits)",
      flush=True)
probe = _train_with_cached_base_activations(
    base_train=tr, base_val=empty, redteam_train=None, redteam_val=None, dev_val=rest,
    model_name=bp.model_name, layer=bp.layer, probe_spec=spec, pos_class_label=pos,
    neg_class_label=neg, probe_description=bp.description,
    base_train_cache=merged, base_val_cache=tmp/"unused.pt", dev_val_cache=SC/"dev_rest.pt",
    redteam_cache_dir=None, combine_consecutive_messages=C, convert_tool_to_assistant=V,
    seed=42, ensemble_seeds=seeds, verbose=False)
q = OUT/"probe_base_plus_devomission.pkl"
with q.open("wb") as f: pickle.dump(probe, f)
df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO/"eval_sets/instructions",
    activations_cache_dir=EC, splits=None, max_samples=None, seed=42,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
df.to_csv(OUT/"eval_base_plus_devomission.csv", index=False)
print(df.to_string(index=False))
