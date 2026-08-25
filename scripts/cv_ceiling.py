"""Grouped 5-fold CV on the oig_omission EVAL split: what is achievable on this split at all?

Every arm in the programme trains on out-of-distribution data (50 base rows + red-team
couples) and lands 0.71-0.84 here. This trains on the split ITSELF and scores held-out rows,
which upper-bounds what a layer-32 linear head over these activations can do on this data.

GROUPED, not row-level. All 57 couples in this split share `original_text` between their two
rows — one reply complete, one omitting — so a random row split would put a source's two
halves on both sides and leak. Folds are cut over the 57 SOURCES.

Validation is the split's 32 dev rows (verified disjoint from eval), so early stopping never
sees the held-out fold. Activations are slices of blobs that already exist; no LLM is loaded.
"""
import json, pickle, sys, tempfile
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
SC = Path("/tmp/claude-1000/-workspace-cc-based-auto-improvement/0998e5c8-a752-408d-9842-7fe74f4434a1/scratchpad/devomi_cache")
from agentic_redteam.retrain import (_cpu_unpickle, _infer_probe_spec, _resolve_ensemble_seeds,
    _train_with_cached_base_activations, stable_train_test_split)
from tuberlens.interfaces.dataset import LabelledDataset

BP = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
SPL = REPO/"eval_sets/instructions/oig_omission.jsonl"
DEV = REPO/"dev_samples/instructions/oig_omission.jsonl"
EVB = REPO/"results_instructions_gemma27b_shared/eval_activations/oig_omission-acts_full.pt"
OUT = REPO/"results_instructions_gemma27b_shared/devomission_probe"; OUT.mkdir(parents=True, exist_ok=True)
K, ENS = 5, 10

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(42, ENS); C = V = True

full = LabelledDataset.load_from(SPL, pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
dev = LabelledDataset.load_from(DEV, pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
raw = [json.loads(l) for l in SPL.open()]
assert len(full) == len(raw) == 114
groups = {}
for i, r in enumerate(raw): groups.setdefault(r["original_text"], []).append(i)
gkeys = sorted(groups)
assert len(gkeys) == 57, len(gkeys)
rng = np.random.RandomState(42); order = rng.permutation(len(gkeys))
folds = [[] for _ in range(K)]
for j, g in enumerate(order): folds[j % K].extend(groups[gkeys[g]])

blob = torch.load(EVB, map_location="cpu", weights_only=False)
y = np.array(full.other_fields["labels"], dtype=int)
labels = list(full.other_fields["labels"])

def subset(idx):
    d = LabelledDataset(inputs=[full.inputs[i] for i in idx],
                        ids=[str(i) for i in idx],
                        other_fields={"labels": [labels[i] for i in idx]})
    return d.assign(activations=blob["activations"][idx].clone(),
                    attention_mask=blob["attention_mask"][idx].clone(),
                    input_ids=blob["input_ids"][idx].clone())

tmp = Path(tempfile.mkdtemp())
oof = np.full(len(full), np.nan)
per_fold = []
for k in range(K):
    test_idx = sorted(folds[k])
    train_idx = sorted(i for j in range(K) if j != k for i in folds[j])
    tr_ds = LabelledDataset(inputs=[full.inputs[i] for i in train_idx], ids=[str(i) for i in train_idx],
                            other_fields={"labels": [labels[i] for i in train_idx]})
    fb = tmp/f"fold{k}.pt"
    torch.save({"activations": blob["activations"][train_idx].clone(),
                "attention_mask": blob["attention_mask"][train_idx].clone(),
                "input_ids": blob["input_ids"][train_idx].clone(),
                "layer": blob["layer"], "model_name": blob["model_name"]}, fb)
    tr, empty = stable_train_test_split(tr_ds, test_size=0.0, split_field=None, seed=42)
    print(f"\n=== fold {k}: train {len(train_idx)} rows / test {len(test_idx)} rows "
          f"({len(test_idx)//2} sources) ===", flush=True)
    probe = _train_with_cached_base_activations(
        base_train=tr, base_val=empty, redteam_train=None, redteam_val=None, dev_val=dev,
        model_name=bp.model_name, layer=bp.layer, probe_spec=spec, pos_class_label=pos,
        neg_class_label=neg, probe_description=bp.description,
        base_train_cache=fb, base_val_cache=tmp/f"fold{k}_val_unused.pt",
        dev_val_cache=SC/"omission_train.pt", redteam_cache_dir=None,
        combine_consecutive_messages=C, convert_tool_to_assistant=V, seed=42,
        ensemble_seeds=seeds, verbose=False)
    p = np.asarray(probe.predict_proba(subset(test_idx)))
    if p.ndim == 2: p = p[:, -1]
    oof[test_idx] = p
    a = roc_auc_score(y[test_idx], p)
    per_fold.append(a)
    print(f"  fold {k} AUROC {a:.4f}", flush=True)

pooled = roc_auc_score(y, oof)
res = {"split": "oig_omission", "folds": K, "grouped_by": "original_text", "n_sources": 57,
       "ensemble": ENS, "per_fold": [round(x, 4) for x in per_fold],
       "per_fold_mean": round(float(np.mean(per_fold)), 4),
       "pooled_oof_auroc": round(float(pooled), 4)}
json.dump(res, open(OUT/"cv_ceiling_oig_omission.json", "w"), indent=1)
print("\n" + json.dumps(res, indent=1))
