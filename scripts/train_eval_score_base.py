#!/usr/bin/env python
"""Train on the real eval split, then score the base training set. No extraction.

The reverse of every other probe here. It asks a narrow question: does a probe that has
learned omission from the real data recognise the concept in the 50-row base set every arm
started from? The base set is generic llama-written instruction-following data, not
omission data, so a low number here says the base set does not carry the concept the eval
split is scored on — which would explain why 50 base rows alone reach only ~0.798.

Everything is read from caches that already exist:

* eval 114 rows — the path-keyed Kaggle blob ``oig_omission-acts_full.pt``
* base 50 rows — the base-split blob written by an earlier fit
* dev 32 rows  — the dev blob, used as the fit's validation set (disjoint from both)

**Row alignment is the one real hazard.** A blob is a bare tensor; nothing in it records
which rows it holds or in what order, and ``LLMModel.load_activations`` validates nothing.
The base blob was written over ``stable_train_test_split(..., test_size=0.0)``'s train
side, so this reproduces that exact call rather than assuming file order, and then checks
row 0's stored ``input_ids`` against a fresh tokenisation before trusting the whole thing.

    set -a; . ./.env; set +a
    .venv_claude/bin/python scripts/train_eval_score_base.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
ARCH = "linear_then_softmax"
POS = "assistant_follows_the_instruction"
NEG = "assistant_does_not_follow_the_instruction"
SEED = 42
COMBINE = CONVERT = True

BASE = REPO / "data/instructions_llama70b_50.jsonl"
EVAL = REPO / "eval_sets/oig_omission/oig_omission.jsonl"
DEV = REPO / "dev_samples/oig_omission"
CACHE = REPO / "cache_oig_omission"
EVAL_BLOB = CACHE / "eval_activations/oig_omission-acts_full.pt"
OUT = REPO / "results_generated_scored"
DESC = ("A conversation where the user asks for several distinct things at once and the "
        "label is set by whether the assistant's reply addresses every part.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-size", type=int, default=5)
    args = ap.parse_args()

    import numpy as np
    from sklearn.metrics import roc_auc_score
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.model import LLMModel
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType
    from tuberlens.probes.probe_factory import ProbeFactory

    from agentic_redteam.ensemble import ENSEMBLE_SEEDS, EnsembleProbe
    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.retrain import (
        _base_activation_cache_paths, _dev_activation_cache_path, _load_dev_dataset,
        _to_device_for_fit, stable_train_test_split)

    load = lambda p: LabelledDataset.load_from(
        p.resolve(), pos_class_label=POS, neg_class_label=NEG,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT)

    # ---- base: reproduce the split the blob was written over -----------------------
    base_all = load(BASE)
    base_train, base_val = stable_train_test_split(base_all, test_size=0.0,
                                                   split_field=None, seed=SEED)
    base_blob, _ = _base_activation_cache_paths(
        CACHE / "base_activations", BASE, MODEL_NAME, LAYER, SEED, 0.0, None,
        COMBINE, CONVERT, 1.0)
    print(f"base  {len(base_train)} train / {len(base_val)} val rows · blob {base_blob.name}"
          f" · exists={base_blob.exists()}")
    if not base_blob.exists():
        raise SystemExit("base blob missing; run a fit on this base file first")
    ba = LLMModel.load_activations(base_blob)
    if ba.activations.shape[0] != len(base_train):
        raise SystemExit(f"base blob has {ba.activations.shape[0]} rows, split has {len(base_train)}")

    # alignment check: the blob's row 0 must be the tokenisation of base_train row 0
    from agentic_redteam.token_budget import count_tokens
    n0 = count_tokens(MODEL_NAME, [{"role": m.role, "content": m.content} for m in base_train.inputs[0]],
                      combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT)
    stored = int(ba.attention_mask[0].sum())
    print(f"  alignment check: row 0 tokenises to {n0}, blob row 0 masks {stored} "
          f"-> {'OK' if n0 == stored else 'MISMATCH'}")
    if n0 != stored:
        raise SystemExit("base blob row order does not match the reproduced split; refusing to score")
    base_train = base_train.assign(activations=ba.activations, attention_mask=ba.attention_mask,
                                   input_ids=ba.input_ids)

    # ---- eval as the TRAINING set ---------------------------------------------------
    ev = load(EVAL)
    ea = LLMModel.load_activations(EVAL_BLOB)
    print(f"eval  {len(ev)} rows · blob {EVAL_BLOB.name} · {ea.activations.shape[0]} rows")
    if ea.activations.shape[0] != len(ev):
        raise SystemExit("eval blob row count does not match the split")
    ev = ev.assign(activations=ea.activations, attention_mask=ea.attention_mask,
                   input_ids=ea.input_ids)

    # ---- dev as the fit's validation set ---------------------------------------------
    dev_ds, dev_files, _ = _load_dev_dataset(DEV, POS, NEG, COMBINE, CONVERT, verbose=False)
    dev_blob = _dev_activation_cache_path(CACHE / "base_activations", dev_files,
                                          MODEL_NAME, LAYER, COMBINE, CONVERT)
    da = LLMModel.load_activations(dev_blob)
    dev_ds = dev_ds.assign(activations=da.activations, attention_mask=da.attention_mask,
                           input_ids=da.input_ids)
    print(f"dev   {len(dev_ds)} rows · blob {dev_blob.name}")

    # ---- fit on eval, validate on dev -------------------------------------------------
    seeds = list(ENSEMBLE_SEEDS[: args.ensemble_size]) if args.ensemble_size > 1 else [SEED]
    spec = ProbeSpec(name=ProbeType(ARCH), hyperparams={})
    _to_device_for_fit([ev, dev_ds], verbose=False)
    t0 = time.monotonic()

    def build(s):
        seed_everything(s)
        return ProbeFactory.build(probe_spec=spec, train_dataset=ev, model_name=MODEL_NAME,
                                  layer=LAYER, validation_dataset=dev_ds, use_store=False,
                                  pos_class_label=POS, neg_class_label=NEG,
                                  probe_description=DESC)

    probe = (build(seeds[0]) if len(seeds) == 1 else
             EnsembleProbe([build(s) for s in seeds], member_seeds=seeds,
                           model_name=MODEL_NAME, layer=LAYER, description=DESC,
                           pos_class_label=POS, neg_class_label=NEG))
    print(f"\nfit on {len(ev)} eval rows, {len(seeds)} member(s), {(time.monotonic()-t0)/60:.1f} min")

    def score(ds, name):
        p = np.asarray(probe.predict_proba(ds), dtype=float).reshape(-1)
        y = ds.labels_numpy()
        a = float(roc_auc_score(y, p)); acc = float(((p >= 0.5).astype(int) == y).mean())
        print(f"  {name:22s} n={len(y):3d}  AUROC {a:.4f}  acc {acc:.4f}  "
              f"mean(pos) {p[y==1].mean():.4f}  mean(neg) {p[y==0].mean():.4f}")
        return {"n": len(y), "auroc": a, "accuracy": acc,
                "mean_pos": float(p[y == 1].mean()), "mean_neg": float(p[y == 0].mean())}

    print("\nprobe trained on the eval split, scored on:")
    res = {"base50": score(base_train, "base training set"),
           "dev32": score(dev_ds, "dev (its validation)"),
           "eval114_insample": score(ev, "eval (IN SAMPLE)")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train_eval_score_base.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {(OUT / 'train_eval_score_base.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
