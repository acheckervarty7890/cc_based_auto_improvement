"""How far apart are the two halves of a couple, in the space the probe reads?

A contrastive couple is meant to differ in the concept and as little else as possible, so the
distance between its halves is a direct measure of how well the "smallest edit" instruction
worked. This measures it where it matters: on the layer-32 activation each conversation is
scored from, pooled to one 5376-d vector by the masked mean (the same approximation
`embed_activations_2d.py` uses and states).

Reported as cosine DISTANCE, 1 - cos(find, partner): 0 = identical direction, 1 = orthogonal.

POOLING DEPTH. A single masked mean is order-blind: it cannot tell a reply that answers three
questions from one that answers two and stops, because both are the same bag of tokens averaged.
So the same distance is also computed at output length 2 and 3 — adaptive average pooling over
the VALID token span only (the mask still decides what is real; padding never enters a segment),
giving k contiguous segments of equal token count, concatenated into a k x 5376 vector. k=1 is
the masked mean exactly, which the script asserts rather than assumes. Per-segment distances are
reported too, since where in the conversation the halves diverge is the more interesting number
for a concept about a reply stopping early.

Three reference points make the number readable, and they are the reason this is worth
computing at all:
  * the eval split's OWN pairs, so the corpus states what this distance is supposed to look
    like. Note oig_omission is only PARTLY paired: its 114 rows carry 81 distinct requests, of
    which 33 appear twice (once each way) and 48 appear once. Only those 33 are measurable
    here, and they are the corpus's own answer to "how far apart should a couple be";
  * the dev split's 16 pairs, which unlike the eval split is fully paired (32 rows);
  * MISMATCHED couples - each find against another couple's partner - which is the distance
    with the pairing removed, i.e. what a couple has to beat to mean anything.

Every activation is cached, so nothing is extracted.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
from agentic_redteam.retrain import (_apply_message_transforms, _cpu_unpickle,
    _redteam_activation_cache_path)
from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

RES = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
PRB = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP  = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BC  = REPO/"results_instructions_gemma27b_shared/base_activations"
EVB = REPO/"results_instructions_gemma27b_shared/eval_activations/oig_omission-acts_full.pt"
DEVB= Path("/tmp/claude-1000/-workspace-cc-based-auto-improvement/"
           "0998e5c8-a752-408d-9842-7fe74f4434a1/scratchpad/devomi_cache/omission_train.pt")
C = V = True
with BP.open("rb") as f: bp = _cpu_unpickle(f)
pos, neg = bp.pos_class_label, bp.neg_class_label

def pool(d, k=1):
    """Adaptive average pool each row's VALID token span to `k` segments, concatenated.

    The activations are right-padded and every row's valid span is contiguous from 0 (asserted),
    so the span is a[:L] and the mask is applied by slicing rather than by weighting. k=1
    reproduces the masked mean exactly.
    """
    a, m = d["activations"].float(), d["attention_mask"]
    lens = m.sum(1).long()
    # Right-padded, valid span contiguous from 0 — checked, because the slicing below would
    # otherwise fold padding into a segment.
    want = torch.arange(m.shape[1])[None, :] < lens[:, None]
    assert bool(((m > 0) == want).all()), \
        "valid tokens are not a contiguous prefix; the slicing below would include padding"
    out = []
    for i, L in enumerate(lens.tolist()):
        span = a[i, :max(L, 1)].T.unsqueeze(0)                  # 1 x embed x L
        seg = torch.nn.functional.adaptive_avg_pool1d(span, k)  # 1 x embed x k
        out.append(seg.squeeze(0).T.reshape(-1).numpy())        # (k*embed,)
    return np.stack(out)

def cosd(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return 1.0 - (A * B).sum(1)

def couple_vecs(path, k=1):
    """Pooled (find, partner) matrices for a *_paired-style jsonl, in couple order."""
    rows = [json.loads(l) for l in Path(path).open()]
    ds = LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rows],
        ids=[r["id"] for r in rows],
        other_fields={"labels": [r["label"] for r in rows]})
    ds = _apply_message_transforms(ds, C, V)
    vec = {}
    for r, msgs in zip(rows, ds.inputs):
        q = _redteam_activation_cache_path(BC, msgs, bp.model_name, bp.layer, C, V)
        if not Path(q).exists(): raise SystemExit(f"missing activation cache for {r['id']}")
        vec[r["id"]] = pool(torch.load(q, map_location="cpu", weights_only=False), k)[0]
    s = [r["id"] for r in rows if r["label"] == "negative"]
    g = [r["id"] for r in rows if r["label"] == "positive"]
    return np.array([vec[i] for i in s]), np.array([vec[i] for i in g]), s, g

def corpus_pairs(jsonl, blob, k=1):
    """Pooled (neg, pos) matrices for a corpus paired by identical user turns."""
    rows = [json.loads(l) for l in Path(jsonl).open()]
    X = pool(torch.load(blob, map_location="cpu", weights_only=False), k)
    assert len(X) == len(rows), (len(X), len(rows))
    groups = {}
    for i, r in enumerate(rows):
        key = " ".join(m["content"] for m in json.loads(r["inputs"]) if m["role"] == "user")
        groups.setdefault(key, []).append(i)
    A, B = [], []
    for idx in groups.values():
        ng = [i for i in idx if rows[i]["labels"] == bp.neg_class_label]
        po = [i for i in idx if rows[i]["labels"] == bp.pos_class_label]
        for i, j in zip(ng, po): A.append(X[i]); B.append(X[j])
    return np.array(A), np.array(B)

def line(name, d, seg=None):
    extra = ""
    if seg is not None:
        extra = "   segments " + " ".join(f"{v:.4f}" for v in seg)
    print(f"{name:<36} n={len(d):3d}  mean {d.mean():.4f}  median {np.median(d):.4f}  "
          f"min {d.min():.4f}  max {d.max():.4f}{extra}")

def per_segment(A, B, k):
    """Cosine distance computed within each of the k segments separately."""
    e = A.shape[1] // k
    return [float(cosd(A[:, j * e:(j + 1) * e], B[:, j * e:(j + 1) * e]).mean()) for j in range(k)]

COUPLES = (("original", PRB/"redteam_postprocessed_iter5.jsonl"),
           ("shortened", RES/"shortened_paired.jsonl"),
           ("restyled", RES/"restyled_paired.jsonl"))
CORPORA = (("eval", REPO/"eval_sets/instructions/oig_omission.jsonl", EVB),
           ("dev", REPO/"dev_samples/instructions/oig_omission.jsonl", DEVB))
NAMES = {"original": "original 33", "shortened": "shortened 33", "restyled": "restyled 33",
         "eval": "eval oig_omission (33 of 57 paired)", "dev": "dev  oig_omission (16 pairs, all)"}

out, per_couple = {}, {}
for k in (1, 2, 3):
    print(f"\n{'='*94}\npooled to {k} segment{'s' if k > 1 else ''} over the valid token span"
          f"{'  (= the masked mean)' if k == 1 else ''}\n{'='*94}")
    res = {}
    for key, path in COUPLES:
        A, B, sid, gid = couple_vecs(path, k)
        d = cosd(A, B); seg = per_segment(A, B, k) if k > 1 else None
        line(NAMES[key], d, seg)
        if k == 1: per_couple[key] = [round(float(x), 4) for x in d]
        res[key] = {"n": len(d), "mean": round(float(d.mean()), 4),
                    "median": round(float(np.median(d)), 4),
                    "min": round(float(d.min()), 4), "max": round(float(d.max()), 4),
                    "segments": [round(v, 4) for v in seg] if seg else None}
        if key == "original":
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(A))
            perm = np.array([p if p != i else (p + 1) % len(A) for i, p in enumerate(perm)])
            dm = cosd(A, B[perm]); line("  (mismatched, pairing removed)", dm)
            res["original_mismatched"] = {"n": len(dm), "mean": round(float(dm.mean()), 4),
                                          "median": round(float(np.median(dm)), 4)}
    print()
    for key, jsonl, blob in CORPORA:
        A, B = corpus_pairs(jsonl, blob, k)
        d = cosd(A, B); seg = per_segment(A, B, k) if k > 1 else None
        line(NAMES[key], d, seg)
        res[key + "_corpus"] = {"n": len(d), "mean": round(float(d.mean()), 4),
                                "median": round(float(np.median(d)), 4),
                                "min": round(float(d.min()), 4), "max": round(float(d.max()), 4),
                                "segments": [round(v, 4) for v in seg] if seg else None}
    out[f"k{k}"] = res

print(f"\n{'='*94}\nMEAN COSINE DISTANCE BY POOLING DEPTH\n{'='*94}")
keys = ["original", "original_mismatched", "shortened", "restyled", "eval_corpus", "dev_corpus"]
w = max(len(x) for x in keys)
print(" " * (w + 2) + "".join(f"{'k=' + str(k):>10}" for k in (1, 2, 3)) + "     k3/k1")
for x in keys:
    v = [out[f"k{k}"][x]["mean"] for k in (1, 2, 3)]
    print(f"{x:<{w}}  " + "".join(f"{q:>10.4f}" for q in v) + f"   {v[2] / v[0]:>6.2f}x")

json.dump({"summary": out, "per_couple": per_couple},
          open(RES/"pair_cosine_distance.json", "w"), indent=1)
print(f"\nwrote {RES/'pair_cosine_distance.json'}")
