#!/usr/bin/env python
"""Cosine distance from each training set to the eval set, in the probe's own space.

Every conversation is summarised by the **masked mean** of its layer-32 gemma-3-27b
residual activations — the same tensors the probe reads, pooled over real tokens only.
The probe's own aggregation (`LinearThenSoftmax`) reduces each token to a scalar before
pooling, so it offers no vector to compare; a masked mean of the 5376-d residual is the
standard summary of that space and is what this measures.

Three numbers per set, because they answer different questions:

  centroid    1 - cos(mean(X), mean(E))            — where the set sits on average
  pairwise    mean over (i, j) of 1 - cos(x_i, e_j) — how far a typical row is from a
                                                      typical eval row
  nearest     mean over i of min over j             — how close the set gets to eval at all

Reads only cached activations; nothing loads the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RUN_DIR = REPO / "results_gen_gemma27b_instructions_nemotron"
PROBE_DIR = REPO / "probes/gen_gemma27b_instructions_nemotron"
EVAL_CACHE = REPO / "cache_gen_gemma27b_instructions/eval_activations"
BASE_CACHE = REPO / "cache_gen_gemma27b_instructions/base_activations"
BASE_PROBE = PROBE_DIR / "probe_iter12.pkl"
ACCEPTED = RUN_DIR / "accepted_iter13.jsonl"
GENERATED = REPO / "data/instructions_like_accepted62.jsonl"
BASE_DATA = REPO / "data/instructions_llama70b_50.jsonl"

# The six rejected batches the poison curve consumed, in the order it added them.
POISON_KEYS = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3)]

SPLITS = [
    "anthropic_harmless_refusal", "bbq_substitution", "hc_context_drift",
    "hc_contradiction", "mm_substitution", "oig_context_drift", "oig_omission",
]


def pool(acts, mask):
    """Masked mean over the token axis → (n, hidden), float32."""
    import torch

    m = mask.to(torch.float32).unsqueeze(-1)
    return (acts.to(torch.float32) * m).sum(1) / m.sum(1).clamp(min=1)


def load_eval():
    import torch

    vecs, labels = [], []
    for s in SPLITS:
        blob = torch.load(EVAL_CACHE / f"{s}-acts_full.pt", map_location="cpu", weights_only=False)
        v = pool(blob["activations"], blob["attention_mask"])
        vecs.append(v)
        labels += [s] * v.shape[0]
        del blob
    return torch.cat(vecs), labels


def jsonl(path: Path) -> list[dict]:
    """Rows as {inputs: [{role, content}], labels}.

    tuberlens' on-disk `LabelledDataset` format stores `inputs` as a JSON-encoded
    string (that is how data/*.jsonl is written); the loop's own accepted/generated
    files store it as a list. Accept both.
    """
    rows = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            inputs = r["inputs"]
            if isinstance(inputs, str):
                inputs = json.loads(inputs)
            rows.append({"inputs": inputs, "labels": r["labels"]})
    return rows


def batch_rows(keys) -> list[dict]:
    latest = {}
    with (RUN_DIR / "batches.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            latest[(r["iteration"], r["batch_index"])] = r
    out = []
    for k in keys:
        out += [{"inputs": s["messages"], "labels": s["label"]} for s in latest[k]["samples"]]
    return out


def load_set(rows, meta):
    """Masked-mean vectors for a list of {inputs, labels} rows, from the per-sample cache."""
    import torch

    from agentic_redteam.retrain import (
        _apply_message_transforms,
        _dicts_to_labelled_dataset,
        _sample_activation_cache_path,
    )

    ds = _dicts_to_labelled_dataset(rows, meta["pos_class_label"], meta["neg_class_label"])
    ds = _apply_message_transforms(ds, True, True)
    vecs, missing = [], 0
    for dialogue in ds.inputs:
        p = _sample_activation_cache_path(
            BASE_CACHE, dialogue, meta["model_name"], meta["layer"], True, True
        )
        if not p.exists():
            missing += 1
            continue
        blob = torch.load(p, map_location="cpu", weights_only=False)
        vecs.append(pool(blob["activations"], blob["attention_mask"]))
        del blob
    if missing:
        print(f"    [warn] {missing} of {len(ds.inputs)} conversations not in the cache")
    return torch.cat(vecs) if vecs else None


def stats(X, E, labels):
    import torch

    Xn = torch.nn.functional.normalize(X, dim=1)
    En = torch.nn.functional.normalize(E, dim=1)
    sim = Xn @ En.T                       # (nx, ne) cosine similarity
    dist = 1.0 - sim
    centroid = 1.0 - torch.nn.functional.cosine_similarity(
        X.mean(0, keepdim=True), E.mean(0, keepdim=True)
    ).item()
    per_split = {}
    for s in SPLITS:
        idx = [i for i, l in enumerate(labels) if l == s]
        per_split[s] = dist[:, idx].mean().item()
    return {
        "n": X.shape[0],
        "centroid": centroid,
        "pairwise": dist.mean().item(),
        "nearest": dist.min(dim=1).values.mean().item(),
        "per_split": per_split,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RUN_DIR / "cosine_to_eval.json")
    args = ap.parse_args()

    import torch

    from agentic_redteam.retrain import read_probe_metadata

    meta = read_probe_metadata(BASE_PROBE)
    print(f"probe space: {meta['model_name']} layer {meta['layer']}\n")

    E, labels = load_eval()
    print(f"eval: {E.shape[0]} rows, {E.shape[1]}-d (masked mean of the layer-32 residual)")

    sets = {
        "62 accepted": jsonl(ACCEPTED),
        "107 generated": jsonl(GENERATED),
        "62 + 107": jsonl(ACCEPTED) + jsonl(GENERATED),
        "60 rejected": batch_rows(POISON_KEYS),
        "50 base (reference)": jsonl(BASE_DATA),
    }

    results = {}
    for name, rows in sets.items():
        print(f"\n{name}: {len(rows)} rows")
        X = load_set(rows, meta)
        if X is None:
            print("    no cached activations — skipped")
            continue
        results[name] = stats(X, E, labels)

    print(f"\n{'set':>22} {'n':>5} {'centroid':>9} {'pairwise':>9} {'nearest':>9}")
    for name, r in results.items():
        print(f"{name:>22} {r['n']:>5} {r['centroid']:>9.4f} {r['pairwise']:>9.4f} {r['nearest']:>9.4f}")

    print(f"\nmean pairwise cosine distance, per eval split")
    print(f"{'split':>27} " + " ".join(f"{n[:11]:>12}" for n in results))
    for s in SPLITS:
        print(f"{s:>27} " + " ".join(f"{r['per_split'][s]:>12.4f}" for r in results.values()))

    # How far apart the eval splits are from each other, for scale.
    En = torch.nn.functional.normalize(E, dim=1)
    print(f"\nfor scale — mean pairwise cosine distance WITHIN the eval set: "
          f"{(1.0 - En @ En.T).mean().item():.4f}")

    args.out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
