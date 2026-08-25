"""A 2D map of the activations behind every sample in the ledger.

Each conversation's layer-32 activation is [tokens, 5376]; the probe reads it as a
mask-weighted pool, so every sample is reduced here to its MASKED MEAN over real tokens —
one 5376-d vector — and the map is a linear projection of those. Two projections are offered:

  pca    PC1 vs PC2 of the pooled vectors, centered, with variance explained reported.
         Purely descriptive: the directions the data varies in, whatever they mean.

  probe  x = the BASE probe's own direction applied to the pooled vector (the mean of its 10
         members' unit weight vectors), y = PC1 of the component orthogonal to that direction.
         This is the more useful of the two: x is, up to the pooling approximation, the axis
         the classifier actually decides on, and y is the largest structure it ignores.

The pooling is an approximation of the head's own softmax-over-tokens aggregation, and is
stated as such rather than presented as the probe's exact input.

Nothing is extracted: base, red-team, dev and eval activations all come from existing caches.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
SP = Path(sys.argv[1])
from agentic_redteam.retrain import (_base_activation_cache_paths, _cpu_unpickle,
    _redteam_activation_cache_path, _apply_message_transforms)
from agentic_redteam.ensemble import iter_probe_members
from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

RES = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
PRB = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP  = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE= REPO/"data/instructions_llama70b_50.jsonl"
BC  = REPO/"results_instructions_gemma27b_shared/base_activations"
RTC = BC/"redteam_acts_google_gemma-3-27b-it_L32"
SPL = REPO/"eval_sets/instructions/oig_omission.jsonl"
EVB = REPO/"results_instructions_gemma27b_shared/eval_activations/oig_omission-acts_full.pt"
DEVB= Path("/tmp/claude-1000/-workspace-cc-based-auto-improvement/0998e5c8-a752-408d-9842-7fe74f4434a1/scratchpad/devomi_cache/omission_train.pt")
C = V = True

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label

def pool_blob(path_or_dict, idx=None):
    d = torch.load(path_or_dict, map_location="cpu", weights_only=False) if not isinstance(path_or_dict, dict) else path_or_dict
    a, m = d["activations"].float(), d["attention_mask"].float()
    if idx is not None: a, m = a[idx], m[idx]
    s = (a * m[..., None]).sum(1)
    n = m.sum(1).clamp(min=1)[:, None]
    return (s / n).numpy()

vecs, meta = [], []
def add(mat, kind, items):
    assert len(mat) == len(items), (kind, len(mat), len(items))
    for v, it in zip(mat, items):
        vecs.append(v); meta.append({"kind": kind, **it})

# ---- base training rows -----------------------------------------------------
base_ds = LabelledDataset.load_from(BASE, pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btc, _ = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, 42, 0.0, None, C, V, 1.0)
add(pool_blob(btc), "base",
    [{"y": int(l)} for l in base_ds.other_fields["labels"]])

# ---- the 33 couples ---------------------------------------------------------
loo = json.load(open(RES/"pair_selection_oig_omission.json"))
verdict = {}
for e in loo["pairs"]:
    verdict[e["sid"]] = ("hurt" if e["delta"] > 0 else "help", round(e["delta"], 4))
    verdict[e["gid"]] = verdict[e["sid"]]
rt = [json.loads(l) for l in (PRB/"redteam_postprocessed_iter5.jsonl").open()]
rt_ds = LabelledDataset(
    inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rt],
    ids=[r["id"] for r in rt],
    other_fields={"labels": [r["label"] for r in rt]})
rt_ds = _apply_message_transforms(rt_ds, C, V)
rt_vecs, rt_meta, missing = [], [], 0
for r, msgs in zip(rt, rt_ds.inputs):
    q = _redteam_activation_cache_path(RTC.parent, msgs, bp.model_name, bp.layer, C, V)
    if not Path(q).exists(): missing += 1; continue
    rt_vecs.append(pool_blob(q)[0])
    side, delta = verdict.get(r["id"], ("?", 0.0))
    rt_meta.append({"y": 1 if r["label"] == "positive" else 0, "side": side, "delta": delta,
                    "half": "partner" if r["label"] == "positive" else "find", "id": r["id"]})
print(f"red-team rows pooled: {len(rt_vecs)} (missing from cache: {missing})")
add(np.array(rt_vecs), "redteam", rt_meta)

# ---- the shortened couples, so the third tab has its own points ---------------
sh = [json.loads(l) for l in (RES/"shortened_paired.jsonl").open()]
sh_ds = LabelledDataset(
    inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in sh],
    ids=[r["id"] for r in sh],
    other_fields={"labels": [r["label"] for r in sh]})
sh_ds = _apply_message_transforms(sh_ds, C, V)
sh_vecs, sh_meta, sh_missing = [], [], 0
for r, msgs in zip(sh, sh_ds.inputs):
    q = _redteam_activation_cache_path(RTC.parent, msgs, bp.model_name, bp.layer, C, V)
    if not Path(q).exists(): sh_missing += 1; continue
    sh_vecs.append(pool_blob(q)[0])
    sh_meta.append({"y": 1 if r["label"] == "positive" else 0, "side": "short", "delta": None,
                    "half": "partner" if r["label"] == "positive" else "find", "id": r["id"]})
print(f"shortened rows pooled: {len(sh_vecs)} (missing: {sh_missing})")
add(np.array(sh_vecs), "redteam", sh_meta)

# ---- the eval split, tagged with what each arm did to it --------------------
flips = json.load(open(RES/"flips_oig_omission.json")); TH = flips["threshold"]
ev = flips["rows"]
def moved(r, k):
    ok_b = (r["base"] >= TH) == bool(r["y"]); ok_a = (r[k] >= TH) == bool(r["y"])
    return "same" if ok_b == ok_a else ("gained" if ok_a else "lost")
fd = json.load(open(SP/"flipdata.json"))
sh_state = {}
for st in ("gained", "lost"):
    for x in fd["arms"]["short"][st]: sh_state[x["i"]] = st
add(pool_blob(EVB), "eval",
    [{"y": r["y"], "i": r["i"], "base": r["base"], "help": moved(r, "help"),
      "hurt": moved(r, "hurt"), "short": sh_state.get(r["i"], "same")} for r in ev])

# ---- the split's own dev rows (the +0.072 that beat the whole programme) -----
dev_ds = LabelledDataset.load_from(REPO/"dev_samples/instructions/oig_omission.jsonl",
    pos_class_label=pos, neg_class_label=neg, combine_consecutive_messages=C,
    convert_tool_to_assistant=V)
add(pool_blob(DEVB), "dev", [{"y": int(l)} for l in dev_ds.other_fields["labels"]])

X = np.array(vecs, dtype=np.float64)
print("pooled matrix", X.shape)
Xc = X - X.mean(0, keepdims=True)

U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
ev_ratio = (S**2) / (S**2).sum()
pca = U[:, :2] * S[:2]
print(f"PCA variance explained: PC1 {ev_ratio[0]:.3f}  PC2 {ev_ratio[1]:.3f}")

w = np.mean([np.concatenate([p.detach().float().flatten().numpy()
                             for p in list(m._classifier.model.parameters())[:1]])
             for m in iter_probe_members(bp)], axis=0)
w = w[:X.shape[1]]; w = w / np.linalg.norm(w)
px = Xc @ w
R = Xc - np.outer(px, w)
Ur, Sr, _ = np.linalg.svd(R, full_matrices=False)
resid_ratio = (Sr**2)/(Sr**2).sum()
py = Ur[:, 0] * Sr[0]
print(f"probe axis: |w|=1, residual PC1 explains {resid_ratio[0]:.3f} of the leftover variance")

pts = []
for k, m in enumerate(meta):
    pts.append({**m, "px": round(float(pca[k, 0]), 3), "py": round(float(pca[k, 1]), 3),
                "wx": round(float(px[k]), 3), "wy": round(float(py[k]), 3)})
out = {"n": len(pts), "pca_var": [round(float(ev_ratio[0]), 4), round(float(ev_ratio[1]), 4)],
       "resid_var": round(float(resid_ratio[0]), 4), "points": pts}
json.dump(out, open(SP/"embed2d.json", "w"))
import collections
print(collections.Counter(p["kind"] for p in pts))
print("wrote", SP/"embed2d.json")
