"""Train the probe on SEGMENT-POOLED activations instead of the full token sequence.

The head is Linear(5376 -> 1) applied per position, then a temperature-5 softmax over positions.
Feeding it the full ~1000-token sequence lets it choose where to look; feeding it k adaptive
mean-pooled segments hands it a fixed, order-preserving summary instead. k=1 collapses the
conversation to a single masked mean, which is order-blind: a reply that answers three questions
and one that answers two and stops are the same bag of tokens averaged. k=2 and k=3 keep coarse
position, and the cosine study showed that is where the couples' structure lives - their halves
differ mainly in the FINAL third (0.0046 / 0.0197 / 0.0361 at k=3).

So this asks whether that structure is usable: does a probe that can only see "early / middle /
late" do better or worse than one given every token?

Pooling runs over the VALID span only (the mask still decides what is real; padding never enters
a segment) and is applied identically to the training data, the dev set and every eval split -
a probe trained on k segments must be scored on k segments.

k=1 is the control. It is NOT expected to reproduce the full-sequence baseline (0.7135): that
probe sees ~1000 positions and can attend to any of them, this one sees a single averaged
vector. The control's job is to show what collapsing to one segment costs, so k=2 and k=3 can be
read against it as well as against the full-sequence number.

Everything else is held: same base 50, same 33 couples, same 436-row dev set, same 10 pinned
ensemble seeds, same architecture and hyperparameters. Every activation is cached.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from agentic_redteam.ensemble import EnsembleProbe
from agentic_redteam.evaluation import seed_everything
from agentic_redteam.retrain import (_apply_message_transforms, _base_activation_cache_paths,
    _cpu_unpickle, _dev_activation_cache_path, _infer_probe_spec, _load_dev_dataset,
    _redteam_activation_cache_path, _resolve_ensemble_seeds, stable_train_test_split)
from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage
from tuberlens.probes.probe_factory import ProbeFactory

RES = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
PRB = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP  = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE= REPO/"data/instructions_llama70b_50.jsonl"; DEV = REPO/"dev_samples/instructions"
BC  = REPO/"results_instructions_gemma27b_shared/base_activations"
EC  = REPO/"results_instructions_gemma27b_shared/eval_activations"
EVD = REPO/"eval_sets/instructions"
SEED = 42; C = V = True; TS = 0.0

with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(SEED, 10)


def seg_pool(acts, mask, k):
    """Adaptive mean-pool each row's valid span to k segments. Returns (acts, mask) of width k."""
    lens = mask.sum(1).long()
    want = torch.arange(mask.shape[1])[None, :] < lens[:, None]
    assert bool(((mask > 0) == want).all()), "valid tokens are not a contiguous prefix"
    out = torch.empty(acts.shape[0], k, acts.shape[2], dtype=acts.dtype)
    for i, L in enumerate(lens.tolist()):
        span = acts[i, :max(L, 1)].float().T.unsqueeze(0)
        out[i] = torch.nn.functional.adaptive_avg_pool1d(span, k).squeeze(0).T.to(acts.dtype)
    return out, torch.ones(acts.shape[0], k, dtype=mask.dtype)


def assign_pooled(ds, blob, k):
    a, m = blob["activations"], blob["attention_mask"]
    assert len(a) == len(ds), (len(a), len(ds))
    pa, pm = seg_pool(a, m, k)
    return ds.assign(activations=pa, attention_mask=pm,
                     input_ids=torch.zeros(len(ds), k, dtype=torch.long))


def load_blob(p): return torch.load(p, map_location="cpu", weights_only=False)


def redteam_blob(rows):
    """One stacked blob for the couples, read from the per-conversation cache."""
    ds = _apply_message_transforms(LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rows],
        ids=[r["id"] for r in rows], other_fields={"labels": [r["label"] for r in rows]}), C, V)
    acts, masks = [], []
    for msgs in ds.inputs:
        d = load_blob(_redteam_activation_cache_path(BC, msgs, bp.model_name, bp.layer, C, V))
        acts.append(d["activations"][0]); masks.append(d["attention_mask"][0])
    w = max(a.shape[0] for a in acts)
    A = torch.zeros(len(acts), w, acts[0].shape[1], dtype=acts[0].dtype)
    M = torch.zeros(len(acts), w, dtype=masks[0].dtype)
    for i, (a, m) in enumerate(zip(acts, masks)):
        A[i, :a.shape[0]] = a; M[i, :m.shape[0]] = m
    return ds, {"activations": A, "attention_mask": M}


# ---- datasets (raw; activations attached per k below) ----
base = LabelledDataset.load_from(Path(BASE), pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btr, bva = stable_train_test_split(base, test_size=TS, split_field=None, seed=SEED)
btc, _bvc = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, SEED, TS, None, C, V, 1.0)
base_blob = load_blob(btc)
dv, dfiles, _sz = _load_dev_dataset(DEV, pos, neg, C, V, verbose=False)
dev_blob = load_blob(_dev_activation_cache_path(BC, dfiles, bp.model_name, bp.layer, C, V))
rt_rows = [json.loads(l) for l in (PRB/"redteam_postprocessed_iter5.jsonl").open()]
rt_ds, rt_blob = redteam_blob(rt_rows)
splits = {}
for f in sorted(EVD.glob("*.jsonl")):
    d = LabelledDataset.load_from(f, pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    splits[f.stem] = (d, load_blob(EC/f"{f.stem}-acts_full.pt"))
print(f"base {len(btr)} + red-team {len(rt_ds)} train | dev {len(dv)} val | "
      f"{len(splits)} eval splits", flush=True)


def run(k):
    print(f"\n=== k={k}: activations pooled to {k} segment{'s' if k > 1 else ''} ===", flush=True)
    tr = LabelledDataset.concatenate([assign_pooled(btr, base_blob, k),
                                      assign_pooled(rt_ds, rt_blob, k)])
    va = assign_pooled(dv, dev_blob, k)
    members = []
    for s in seeds:
        seed_everything(s)
        members.append(ProbeFactory.build(probe_spec=spec, train_dataset=tr,
            model_name=bp.model_name, layer=bp.layer, validation_dataset=va, use_store=False,
            pos_class_label=pos, neg_class_label=neg, probe_description=bp.description))
    probe = EnsembleProbe.from_members(members, list(seeds))
    rows = []
    for name, (d, blob) in splits.items():
        p = np.asarray(probe.predict_proba(assign_pooled(d, blob, k)))
        if p.ndim == 2: p = p[:, -1]
        y = np.asarray([int(l) for l in d.other_fields["labels"]])
        fv, tv, _ = roc_curve(y, p)
        rows.append({"dataset": name, "auroc": float(roc_auc_score(y, p)),
                     "accuracy": float(accuracy_score(y, p > 0.5)),
                     "tpr_at_fpr": float(tv[int(np.argmin(np.abs(fv - 0.01)))]), "fpr": 0.01})
    mean = {"dataset": "mean", "fpr": 0.01}
    for m in ("auroc", "accuracy", "tpr_at_fpr"):
        mean[m] = float(np.mean([r[m] for r in rows]))
    rows.append(mean)
    import pandas as pd
    df = pd.DataFrame(rows)[["dataset", "auroc", "accuracy", "tpr_at_fpr", "fpr"]]
    df.to_csv(RES/f"eval_segpool_k{k}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return {r["dataset"]: round(float(r["auroc"]), 4) for r in rows}


res = {f"k{k}": run(k) for k in (1, 2, 3)}
json.dump({"seed": SEED, "auroc": res}, open(RES/"segment_pooled_probe.json", "w"), indent=1)
print("\n===== SEGMENT-POOLED PROBE (AUROC) =====")
names = [s for s in res["k1"] if s != "mean"] + ["mean"]
w = max(len(s) for s in names)
print(" " * (w + 2) + "".join(f"{t:>10}" for t in res) + "     full-seq")
FULL = {"anthropic_harmless_refusal": 0.7260, "bbq_substitution": 0.9411,
        "hc_context_drift": 0.6470, "hc_contradiction": 0.8503, "mm_substitution": 0.9476,
        "oig_context_drift": 0.7299, "oig_omission": 0.7135, "mean": 0.7936}
for s in names:
    print(f"{s:<{w}}  " + "".join(f"{res[t][s]:>10.4f}" for t in res) + f"   {FULL[s]:>10.4f}")
print("\nfull-seq column: the same base 50 + 33 couples trained on the untouched token sequence.")
