#!/usr/bin/env python
"""Ceiling analysis for the re-cut oig_omission eval split, plus the two reference probes.

The 30/70 re-cut (``scripts/resplit_oig_omission.py``) produced a new 100-row eval split,
so every ceiling and every probe score measured on the old 114-row one is off a different
test set and cannot be carried over. This recomputes all of it:

* **ceiling** — 5-fold cross-validation *inside* the eval split. Each fold fits the same
  probe family on the other four folds and predicts the held-out one; the out-of-fold
  predictions are pooled and scored once over all 100 rows. This is the best this probe
  family does on this split when it is allowed eval-distribution training data, i.e. the
  bar that any amount of generated data is trying to reach.
* **base-trained probe** — fit on ``data/instructions_llama70b_50.jsonl`` (50 rows), the
  set every arm of the parent experiment started from.
* **dev-trained probe** — fit on the re-cut dev split (46 rows) instead.
* **base ∪ dev** — both (96 rows). Not asked for, but it is one more fit on activations
  that are already extracted and it is the honest "best reachable without touching eval"
  reference that makes the ceiling readable.

TWO METHOD POINTS THAT CHANGE THE NUMBER.

**The CV folds are grouped by user turn, not by row.** 34 of the eval split's 66 user
turns carry both labels — the same multi-part question answered in full and with a part
dropped. A row-wise fold would train on one half of such a pair and test on the other,
which is near-duplicate text with the opposite label, and the ceiling would come out
flattering and wrong. Whole user-turn groups are held out together.

**Early stopping needs a validation set that is not the thing being measured.** Every
ceiling fold validates on the dev split (disjoint from eval, never in any fold). The
base-trained probe validates on dev, exactly as the loop's iteration 0 does. The
dev-trained probe cannot — dev is its training data — so it validates on the base set,
which is the symmetric choice and is likewise disjoint from eval. That asymmetry is real
and is reported with the results rather than hidden: the dev-trained probe early-stops
against out-of-distribution rows.

Everything runs on activations extracted once (196 rows total) and cached; the fits
themselves load no model.

    set -a; . ./.env; set +a
    .venv_claude/bin/python scripts/ceiling_oig_omission_mixed.py
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
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
COMBINE = CONVERT = True
SEED = 42
N_FOLDS = 5

EVAL_JSONL = REPO / "eval_sets/oig_omission_mixed/oig_omission.jsonl"
DEV_JSONL = REPO / "dev_samples/oig_omission_mixed/oig_omission.jsonl"
BASE_JSONL = REPO / "data/instructions_llama70b_50.jsonl"
CACHE = REPO / "cache_oig_omission_mixed/analysis_activations"
OUT = REPO / "results_oig_omission_mixed"
TAG = "mixed"

DESC = (
    "A conversation where the user asks for several distinct things at once and the label "
    "is set by whether the assistant's reply addresses every part."
)


# ---------------------------------------------------------------- data helpers


def _load(path: Path):
    from tuberlens.interfaces.dataset import LabelledDataset

    return LabelledDataset.load_from(
        path.resolve(), pos_class_label=POS, neg_class_label=NEG,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
    )


def _subset(ds, idx: list[int]):
    """Row subset that carries the activation tensors through.

    ``LabelledDataset`` holds everything but ``inputs``/``ids`` in ``other_fields``, so a
    subset means indexing each column of full length and leaving anything else alone.
    Tensors and arrays take fancy indexing; plain lists are rebuilt.
    """
    n = len(ds)
    other = {}
    for k, v in ds.other_fields.items():
        if hasattr(v, "__len__") and len(v) == n:
            other[k] = v[idx] if hasattr(v, "shape") else [v[i] for i in idx]
        else:
            other[k] = v
    return type(ds)(
        inputs=[ds.inputs[i] for i in idx], ids=[ds.ids[i] for i in idx], other_fields=other
    )


def _user_turn(sample) -> str:
    """Grouping key: the first user message of a loaded conversation."""
    for m in sample:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "user":
            return m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
    return str(sample)


def _activate(ds, name: str, verbose: bool = True):
    """Attach activations, from a content-keyed blob when one exists.

    The key covers the rows' own content plus model/layer/transforms — the same discipline
    as every other cache here, since ``load_activations`` loads by path without checking
    what it is loading.
    """
    from tuberlens.model import LLMModel

    from agentic_redteam.model_loading import load_extraction_model, unhook_model

    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    for s in ds.inputs:
        h.update(json.dumps(s, sort_keys=True, default=str).encode())
    h.update(f"|{MODEL_NAME}|{LAYER}|{COMBINE}|{CONVERT}".encode())
    path = CACHE / f"{name}_{h.hexdigest()[:16]}.pt"

    if path.exists():
        print(f"[acts] {name}: cached ({len(ds)} rows) {path.name}")
        acts = LLMModel.load_activations(path)
    else:
        print(f"[acts] {name}: extracting {len(ds)} rows through {MODEL_NAME} ...")
        model = load_extraction_model(MODEL_NAME, LAYER, verbose=verbose)
        try:
            acts = model.get_activations(
                ds.inputs, layer=LAYER, show_progress=verbose, save_path=str(path)
            )
        finally:
            unhook_model(model)
            del model
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return ds.assign(
        activations=acts.activations, attention_mask=acts.attention_mask,
        input_ids=acts.input_ids,
    )


# ---------------------------------------------------------------- fitting


def _fit(train_ds, val_ds, seeds: list[int], verbose: bool = False):
    """Fit one probe (or a score-averaging ensemble) on pre-activated datasets."""
    from tuberlens.probes.probe_factory import ProbeFactory
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType

    from agentic_redteam.ensemble import EnsembleProbe
    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.retrain import _to_device_for_fit

    spec = ProbeSpec(name=ProbeType(ARCH), hyperparams={})
    _to_device_for_fit([train_ds, val_ds], verbose=False)

    def build(s: int):
        seed_everything(s)
        return ProbeFactory.build(
            probe_spec=spec, train_dataset=train_ds, model_name=MODEL_NAME, layer=LAYER,
            validation_dataset=val_ds, use_store=False,
            pos_class_label=POS, neg_class_label=NEG, probe_description=DESC,
        )

    if len(seeds) == 1:
        return build(seeds[0])
    return EnsembleProbe(
        [build(s) for s in seeds], member_seeds=list(seeds),
        model_name=MODEL_NAME, layer=LAYER, description=DESC,
        pos_class_label=POS, neg_class_label=NEG,
    )


def _auroc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if len(set(list(y))) == 2 else math.nan


def _score(probe, ds) -> tuple[float, float]:
    import numpy as np

    p = np.asarray(probe.predict_proba(ds), dtype=float).reshape(-1)
    y = ds.labels_numpy()
    acc = float(((p >= 0.5).astype(int) == y).mean())
    return _auroc(y, p), acc


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-size", type=int, default=5)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    # The same protocol has to be runnable on the SHIPPED 114-row split, or the
    # difference between this ceiling and the published 0.6725 confounds two changes at
    # once: the re-cut split and this analysis's grouped folds / per-split fits.
    ap.add_argument("--eval-jsonl", type=Path, default=EVAL_JSONL)
    ap.add_argument("--dev-jsonl", type=Path, default=DEV_JSONL)
    ap.add_argument("--tag", default=TAG, help="names the cached blobs and the output file")
    # The grouping key decides what leaks. `original_text` is the split's own provenance
    # column and couples ALL 114 rows into 57 sources; the first user turn only recovers
    # 33 of those couples and leaves 48 rows looking like singletons, so 24 couples can be
    # split across folds. Default to the column when the file carries one.
    ap.add_argument("--group-field", default="original_text",
                    help="JSONL column to cut folds over; 'user_turn' falls back to the first user message")
    args = ap.parse_args()

    from agentic_redteam.ensemble import ENSEMBLE_SEEDS

    seeds = list(ENSEMBLE_SEEDS[: args.ensemble_size]) if args.ensemble_size > 1 else [SEED]

    ev = _activate(_load(args.eval_jsonl), f"{args.tag}-eval")
    dv = _activate(_load(args.dev_jsonl), f"{args.tag}-dev")
    ba = _activate(_load(BASE_JSONL), "base50")
    print(f"\nloaded: eval {len(ev)} | dev {len(dv)} | base {len(ba)} rows"
          f" | ensemble {len(seeds)} member(s)\n")

    results: dict[str, dict] = {}

    # ---- reference probes -----------------------------------------------------
    for name, train, val, note in (
        ("base50", ba, dv, "trained on the 50-row base set, validated on dev"),
        ("devset", dv, ba, "trained on the dev split, validated on base (dev is its training data)"),
        ("base50+dev", None, None, "trained on both, validated on dev (in-sample validation — see note)"),
    ):
        if name == "base50+dev":
            from tuberlens.interfaces.dataset import LabelledDataset

            train = LabelledDataset.concatenate([ba, dv])
            val = dv
        t0 = time.monotonic()
        probe = _fit(train, val, seeds)
        auroc, acc = _score(probe, ev)
        results[name] = {"auroc": auroc, "accuracy": acc, "n_train": len(train), "note": note}
        print(f"{name:14s} n_train={len(train):>3}  eval AUROC {auroc:.4f}  acc {acc:.4f}"
              f"  [{(time.monotonic() - t0) / 60:.1f} min]  — {note}")

    # ---- ceiling: group-aware K-fold inside eval ------------------------------
    groups = collections.defaultdict(list)
    raw = [json.loads(l) for l in args.eval_jsonl.read_text().splitlines() if l.strip()]
    assert len(raw) == len(ev), f"{len(raw)} raw rows vs {len(ev)} loaded"
    use_col = args.group_field != "user_turn" and args.group_field in raw[0]
    for i, s in enumerate(ev.inputs):
        groups[raw[i][args.group_field] if use_col else _user_turn(s)].append(i)
    print(f"grouping folds by {args.group_field if use_col else 'user_turn'}: "
          f"{len(groups)} groups, sizes "
          f"{dict(collections.Counter(len(v) for v in groups.values()))}")
    keys = sorted(groups, key=lambda k: hashlib.sha256(f"cv:{SEED}:{k}".encode()).hexdigest())
    folds: list[list[int]] = [[] for _ in range(args.folds)]
    for j, k in enumerate(keys):                     # round-robin keeps folds even
        folds[j % args.folds].extend(groups[k])
    print(f"\nceiling: {args.folds}-fold CV over {len(ev)} rows in {len(keys)} user-turn groups"
          f"; fold sizes {[len(f) for f in folds]}")

    import numpy as np

    oof = np.zeros(len(ev), dtype=float)
    for fi, held in enumerate(folds):
        train_idx = [i for i in range(len(ev)) if i not in set(held)]
        tr, te = _subset(ev, train_idx), _subset(ev, sorted(held))
        probe = _fit(tr, dv, seeds)
        p = np.asarray(probe.predict_proba(te), dtype=float).reshape(-1)
        for slot, i in enumerate(sorted(held)):
            oof[i] = p[slot]
        print(f"  fold {fi + 1}/{args.folds}: train {len(tr):>3} → held-out {len(te):>2}"
              f"  fold AUROC {_auroc(te.labels_numpy(), p):.4f}")

    y = ev.labels_numpy()
    ceiling = _auroc(y, oof)
    ceil_acc = float(((oof >= 0.5).astype(int) == y).mean())
    results["ceiling"] = {
        "auroc": ceiling, "accuracy": ceil_acc, "folds": args.folds,
        "n_groups": len(keys), "note": "out-of-fold, group-aware CV inside the eval split",
    }

    print("\n" + "=" * 62)
    print(f"CEILING (pooled out-of-fold, {len(ev)} rows): {ceiling:.4f}   acc {ceil_acc:.4f}")
    for k in ("base50", "devset", "base50+dev"):
        r = results[k]
        print(f"  {k:14s} {r['auroc']:.4f}   gap to ceiling {ceiling - r['auroc']:+.4f}")
    print("=" * 62)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ceiling_{args.tag}.json").write_text(json.dumps(
        {"tag": args.tag, "eval_jsonl": str(args.eval_jsonl), "dev_jsonl": str(args.dev_jsonl),
         "eval_rows": len(ev), "dev_rows": len(dv), "base_rows": len(ba),
         "ensemble_size": len(seeds), "model": MODEL_NAME, "layer": LAYER,
         "results": results}, indent=2))
    print(f"\nwrote {(OUT / f'ceiling_{args.tag}.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
