"""Attribute the gap between the refit red-team-only baseline and published iter5.

`run_ceiling.py`'s `redteam_only` condition rebuilds probe_iter5's training set
(base ∪ redteam_postprocessed_iter5) and refits it with the same 10 seeds and the
same 436-row dev validation set — yet it does not land exactly on the published
iter5 row. Two candidate causes:

  (a) the reconstructed training set is not what the run actually trained on, or
  (b) the fit is not bit-reproducible across processes.

Refitting the same thing twice separates them: identical twice => (a), different
=> (b). Also refits member-for-member against the pickled probe's own weights.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H

ARM = sys.argv[1] if len(sys.argv) > 1 else "gptoss"

H.silence_tqdm()
ev = H.load_eval_splits()
dev = H.load_dev()
train = H.concat(H.load_base(), H.load_redteam(ARM, 5))
for d in ev.values():
    H.stage(d)
H.stage(dev, train)
print(f"train={len(train)} dev={len(dev)}", flush=True)

with (H.ARMS[ARM] / "probe_iter5.pkl").open("rb") as f:
    published = pickle.load(f)
pub_members = list(getattr(published, "members", [published]))
# The eval activations are staged on the card, so the published heads have to be
# there too (the pickle may have been written from a CPU unpickle).
for m in pub_members:
    m._classifier.model.to(m._classifier.device)
print(f"published probe: {len(pub_members)} members", flush=True)

runs = []
for rep in range(2):
    members = []
    for i, s in enumerate(H.ENSEMBLE_SEEDS[:10]):
        with H.Quiet():
            members.append(H.fit_member(train, dev, s))
    per = H.score_splits(H.ensemble_of(members, list(H.ENSEMBLE_SEEDS[:10])), ev)
    runs.append((members, H.macro(per)))
    print(f"rep{rep}: macro AUROC {H.macro(per)['auroc']:.6f}", flush=True)

a, b = runs[0][0], runs[1][0]
same = all(
    all(torch.equal(p.cpu(), q.cpu()) for p, q in
        zip(x._classifier.model.state_dict().values(),
            y._classifier.model.state_dict().values()))
    for x, y in zip(a, b)
)
print(f"two refits bit-identical to each other: {same}")

matches = 0
for i, (mine, pub) in enumerate(zip(a, pub_members)):
    ms = mine._classifier.model.state_dict()
    ps = pub._classifier.model.state_dict()
    eq = all(torch.equal(ms[k].cpu(), ps[k].cpu()) for k in ms)
    d = max(float((ms[k].cpu().float() - ps[k].cpu().float()).abs().max()) for k in ms)
    matches += eq
    print(f"  member {i}: identical to published={eq} max|dw|={d:.4g}")
print(f"members matching the published probe: {matches}/{len(pub_members)}")

pub_per = H.score_splits(published, ev)
print(f"published macro AUROC (rescored here): {H.macro(pub_per)['auroc']:.6f}")
