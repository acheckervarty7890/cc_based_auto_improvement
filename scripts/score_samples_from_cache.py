#!/usr/bin/env python
"""Score a run's generated conversations with an existing probe, straight from the cache.

Every generated conversation this project has produced already has its layer-32 activation
on disk, keyed by its own transformed content (``_sample_activation_cache_path``). So
scoring them needs no extraction and no LLM — which is the whole point of that cache, and
what ``evaluate_probe`` cannot do, because it reads a *path-keyed blob* per eval split and
would recompute a 30-row split from scratch.

Two things this gets right that a naive reconstruction does not:

* **The key is over ALREADY-TRANSFORMED messages.** ``_sample_activation_cache_path``
  hashes exactly what it is handed and only stringifies the combine/convert flags into the
  prefix. For a conversation whose roles already alternate the transforms are a no-op, so
  raw and transformed agree — but a malformed ``(user, user)`` conversation is merged into
  one tab-joined user turn, and keying the raw form points at a file that does not exist.
* **Rows are stored at their own width**, so they are padded to a common length before
  stacking, exactly as ``_concatenate_consuming`` does at merge time.

    .venv_claude/bin/python scripts/score_samples_from_cache.py \
        --run results_oig_omission_nemotron_topics_v1 \
        --probe probes/generated_scored/base_plus_eval_arm4.pkl --label arm4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
POS = "assistant_follows_the_instruction"
NEG = "assistant_does_not_follow_the_instruction"
COMBINE = CONVERT = True
CACHE = REPO / "cache_oig_omission/base_activations"
OUT = REPO / "results_generated_scored"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--batch", action="append", default=[], help="iteration:batch_index")
    args = ap.parse_args()

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from tuberlens.interfaces.dataset import LabelledDataset, Message
    from tuberlens.model import LLMModel

    from agentic_redteam.retrain import _sample_activation_cache_path, load_probe

    newest = {}
    for line in (args.run / "batches.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line); newest[(r["iteration"], r["batch_index"])] = r
    keys = ([tuple(int(x) for x in b.split(":")) for b in args.batch] if args.batch
            else sorted(newest))

    dialogues, labels, shapes = [], [], []
    for k in keys:
        for s in newest[k]["samples"]:
            dlg = [Message(role=m["role"], content=m["content"]) for m in s["messages"]]
            shapes.append(tuple(m.role for m in dlg))
            if CONVERT:
                dlg = LabelledDataset._convert_tool_to_assistant(dlg)
            if COMBINE:
                dlg = LabelledDataset._combine_consecutive_messages(dlg)
            dialogues.append(dlg)
            labels.append(s["label"])

    paths = [_sample_activation_cache_path(CACHE, d, MODEL_NAME, LAYER, COMBINE, CONVERT)
             for d in dialogues]
    missing = [p for p in paths if not p.exists()]
    print(f"{len(dialogues)} conversations · cache {len(paths) - len(missing)} hit, {len(missing)} miss")
    if missing:
        raise SystemExit(f"{len(missing)} conversation(s) not in the per-sample cache; "
                         "run the loop or warm_sample_activation_cache first")

    parts = [LLMModel.load_activations(p) for p in paths]
    width = max(a.activations.shape[1] for a in parts)
    print(f"row widths {min(a.activations.shape[1] for a in parts)}–{width}; padding to {width}")

    def pad(t, n, w):
        if t.shape[1] == w:
            return t
        z = torch.zeros((t.shape[0], w - t.shape[1], *t.shape[2:]), dtype=t.dtype)
        return torch.cat([t, z], dim=1)

    acts = torch.cat([pad(a.activations, len(parts), width) for a in parts], dim=0)
    mask = torch.cat([pad(a.attention_mask, len(parts), width) for a in parts], dim=0)
    ids = torch.cat([pad(a.input_ids, len(parts), width) for a in parts], dim=0)

    # LabelledDataset stores the canonical Label enum, not the probe's human-readable
    # class strings — load_from does this mapping on the way in, so do it here too.
    canon = ["positive" if l == POS else "negative" for l in labels]
    ds = LabelledDataset(inputs=dialogues, ids=[str(i) for i in range(len(dialogues))],
                         other_fields={"labels": canon})
    ds = ds.assign(activations=acts, attention_mask=mask, input_ids=ids)

    probe = load_probe(args.probe)
    p = np.asarray(probe.predict_proba(ds), dtype=float).reshape(-1)
    y = ds.labels_numpy()
    auroc = float(roc_auc_score(y, p))
    acc = float(((p >= 0.5).astype(int) == y).mean())

    print(f"\nprobe {args.probe.name} scored on {len(y)} generated rows")
    print(f"  AUROC {auroc:.4f}   accuracy {acc:.4f}")
    print(f"  mean score — labelled follows {p[y == 1].mean():.4f} | "
          f"labelled omits {p[y == 0].mean():.4f}   (higher = more 'follows')")
    bad = [s for s in shapes if s != ("user", "assistant") and s != ("user", "assistant") * 2]
    if bad:
        print(f"  note: {len(bad)} conversation(s) with irregular role shape, e.g. {bad[0]}")

    verdict = ("agrees with the generated labels" if auroc > 0.65 else
               "cannot separate the generated classes" if auroc >= 0.45 else
               "is ANTI-correlated with the generated labels")
    print(f"\n=> the probe {verdict}.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.label}.json").write_text(json.dumps({
        "label": args.label, "run": str(args.run), "probe": str(args.probe),
        "n": len(y), "auroc": auroc, "accuracy": acc,
        "mean_score_follows": float(p[y == 1].mean()),
        "mean_score_omits": float(p[y == 0].mean())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
