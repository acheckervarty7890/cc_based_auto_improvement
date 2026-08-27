#!/usr/bin/env python
"""Which LAYER carries the instruction-following signal? Grouped CV over dev + eval.

Every probe in this project reads ``google/gemma-3-27b-it`` layer 32, and nothing has ever
varied that. This asks what is achievable *at all* from each of layers 16/24/32/40/48/56,
by the same measurement ``scripts/cv_ceiling.py`` makes at layer 32 on one split:

    train on the data itself, score held-out rows

which upper-bounds what a linear head over those activations can do — a CEILING, not a
transfer number. Comparing the ceilings tells you which layer's representation actually
separates the concept, independent of whatever out-of-distribution training data a real
arm happens to use.

**The pool is dev + eval combined**, per split. They are disjoint by construction (see
CLAUDE.md, "Eval dataset splits on disk") and identically shaped — the dev dir was cut from
the same source — so pooling them just buys ~50% more rows per split for the CV. Nothing
here is used as a held-out test of anything else afterwards; this is a measurement, not a
training run.

**GROUPED, not row-level.** Every one of these splits is built as contrastive pairs: one
source, answered once each way, two rows with opposite labels. A random row split puts a
pair's two halves on both sides of the fold boundary and leaks the answer outright.

The grouping key is a UNION of two things, joined by union-find:

* the conversation's USER turns (``retrain._dev_pair_key``, the key the dev-row lending
  ladder uses), and
* the split's own SOURCE column(s) — ``GROUP_COLUMNS`` below.

Neither alone is sufficient, and the second is not decoration. On ``oig_omission`` the two
halves of a pair carry *paraphrased* user turns ("What relevant legislation has he worked
on?" against "What legislation is he known for?"), so the user-turn key alone splits **24 of
57** sources across the fold boundary; ``original_text`` recovers them. Conversely
``anthropic_harmless_refusal`` ships no provenance columns at all, so there the user-turn key
is the only thing there is. Verified for every split by ``--report-groups``, which prints the
resulting group-size histogram and does no fitting.

Eval and dev sources are disjoint (checked: zero shared source values on all six splits that
have a source column), so a group never spans the two — pooling them cannot leak either.

**Three-way folds, so early stopping never sees the test rows.** ``linear_then_softmax`` is
fit by ``PytorchAdamClassifier``, which selects on a validation set. With K folds, fold k is
the test set, fold (k+1) % K is the validation set, and the remaining K-2 folds train. So
each fold trains on (K-2)/K of the pool, and no row is ever both selected on and scored.
The alternative — validating on a fixed dev set — is unavailable here, because the dev rows
are *in* the pool.

Two modes, and they answer different questions:

* ``per-split`` (default) — one CV per split, so the probe is fit and scored inside one
  failure mode. This is the per-split ceiling: how linearly separable is *this* distinction
  at this layer.
* ``pooled`` — one CV over all splits at once, folds still grouped, scored per split. This
  is the ceiling for a SINGLE probe that has to serve all seven failure modes — closer to
  what the programme actually trains, and the more honest "best layer" criterion if you
  intend to keep training one probe.

Both report pooled out-of-fold AUROC (every row scored exactly once, by the fold that held
it out), not a mean of per-fold AUROCs, so a fold that happens to be easy cannot be averaged
against one that is hard on a different scale.

Activations come from ``scripts/extract_multilayer_activations.py``; no LLM is loaded here.
The fit mirrors ``retrain._train_with_cached_base_activations``' tail exactly — same
``ProbeSpec`` read off a real probe, same repo-pinned ``ENSEMBLE_SEEDS``, same
``_to_device_for_fit`` staging, same ``ProbeFactory.build_ensemble`` — so a number here is
comparable to a number a real retrain produces.

Usage::

    .venv_claude/bin/python scripts/layer_cv_sweep.py \\
        --layers 16 24 32 40 48 56 --mode both \\
        --out results_instructions_gemma27b_layersweep/layer_cv.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, nargs="+", default=[16, 24, 32, 40, 48, 56])
    ap.add_argument("--acts-root", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/activations")
    ap.add_argument("--eval-dir", type=Path, default=REPO / "eval_sets/instructions")
    ap.add_argument("--dev-dir", type=Path, default=REPO / "dev_samples/instructions")
    ap.add_argument("--probe", type=Path,
                    default=REPO / "probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl")
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ensemble-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--combine-consecutive-messages", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--convert-tool-to-assistant", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--mode", choices=["per-split", "pooled", "both"], default="both")
    ap.add_argument("--report-groups", action="store_true",
                    help="Print each split's group-size histogram and exit. Fits nothing, "
                         "reads no activations.")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/layer_cv.json")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------------------
def _load_pool(split, layer, args, pos, neg, model_name):
    """dev + eval rows of one split, with layer-``layer`` activations attached."""
    import torch
    from agentic_redteam.retrain import _concatenate_consuming
    from tuberlens.interfaces.dataset import LabelledDataset

    parts = []
    for group, data_dir in (("eval_sets", args.eval_dir), ("dev_samples", args.dev_dir)):
        jsonl = data_dir / f"{split}.jsonl"
        if not jsonl.exists():
            continue
        ds = LabelledDataset.load_from(
            jsonl, pos_class_label=pos, neg_class_label=neg,
            combine_consecutive_messages=args.combine_consecutive_messages,
            convert_tool_to_assistant=args.convert_tool_to_assistant)
        blob_path = args.acts_root / f"L{layer}" / group / f"{split}-acts_full.pt"
        blob = torch.load(blob_path, map_location="cpu", weights_only=False)
        if blob["model_name"] != model_name or int(blob["layer"]) != layer:
            raise SystemExit(f"{blob_path}: wrong model/layer header")
        if blob["activations"].shape[0] != len(ds):
            raise SystemExit(f"{blob_path}: {blob['activations'].shape[0]} rows, "
                             f"{jsonl} has {len(ds)}")
        parts.append(ds.assign(activations=blob["activations"],
                               attention_mask=blob["attention_mask"],
                               input_ids=blob["input_ids"]))
        del blob
    if not parts:
        raise SystemExit(f"no data for split {split}")
    return _concatenate_consuming(parts) if len(parts) > 1 else parts[0]


# The column(s) naming the SOURCE a row was derived from, per split. A pair's two rows
# share their source, so rows sharing one of these values must land in the same fold.
# Chosen by profiling every provenance column of every split: these are the ones whose
# value-frequency histogram shows them identifying a source rather than a category
# (`bbq_substitution.category` has 9 values over 200 rows — a label, not a source).
# `anthropic_harmless_refusal` ships no provenance columns; there the user-turn key is all
# there is. Where two columns both identify the source (a document pair), both are used and
# union-find merges whatever they jointly connect.
GROUP_COLUMNS = {
    "anthropic_harmless_refusal": [],
    "bbq_substitution": ["context"],
    "hc_context_drift": ["turn1_doc", "turn2_doc"],
    "hc_contradiction": ["doc_a", "doc_b"],
    "mm_substitution": ["generated_content", "text"],
    "oig_context_drift": ["human_turn_1"],
    "oig_omission": ["original_text"],
}


class _Rows:
    """Minimal stand-in for a LabelledDataset: just ``inputs`` and a length.

    ``--report-groups`` inspects the fold cut without touching activations, and the only
    thing ``_grouped_folds`` reads off a dataset is those two.
    """

    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)


def _source_keys(split, eval_dir, dev_dir):
    """Per-row source keys, read from the RAW jsonl in the same order the loader sees.

    ``LabelledDataset.load_from`` keeps ``inputs``/``labels``, not the provenance columns,
    so these are read straight off the file. Row order is the file's order in both cases,
    which is what makes the index alignment valid; the caller asserts the counts match.
    """
    import json

    cols = GROUP_COLUMNS.get(split, [])
    keys = []
    for data_dir in (eval_dir, dev_dir):
        path = Path(data_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            row = json.loads(line)
            # Namespaced by column so two columns holding the same string cannot merge
            # groups by accident.
            keys.append([f"{split}\u0000{c}\u0000{json.dumps(row.get(c), sort_keys=True)}"
                         for c in cols if c in row])
    return keys


def _grouped_folds(dataset, k, seed, source_keys=None):
    """Fold assignment per row, cut over source-keyed GROUPS (see module docstring).

    Rows are unioned when they share the user-turn key OR any source key. Union-find,
    because the two relations are not nested: on ``mm_substitution`` some rows pair by
    ``generated_content`` and others by ``text``.
    """
    import numpy as np
    from agentic_redteam.retrain import _dev_pair_key

    n = len(dataset)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first = {}
    for i, row in enumerate(dataset.inputs):
        keys = [f"user\u0000{_dev_pair_key(row)}"]
        if source_keys is not None:
            keys += source_keys[i]
        for key in keys:
            if key in first:
                union(first[key], i)
            else:
                first[key] = i

    groups = OrderedDict()
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    keys = list(groups)
    order = np.random.RandomState(seed).permutation(len(keys))
    fold_of = np.empty(n, dtype=int)
    for rank, gi in enumerate(order):
        for row in groups[keys[gi]]:
            fold_of[row] = rank % k
    return fold_of, len(keys), [len(groups[k]) for k in keys]


def _subset(dataset, idx):
    # int() per element: these come from np.flatnonzero, and a list of np.int64 is not
    # the plain-int list LabelledDataset's advanced indexing expects.
    return dataset[[int(i) for i in idx]]


def _fit(train_ds, val_ds, *, spec, seeds, model_name, layer, pos, neg, desc):
    """The tail of ``retrain._train_with_cached_base_activations``, verbatim in behaviour."""
    from agentic_redteam.ensemble import EnsembleProbe, fusion_enabled
    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.retrain import _to_device_for_fit
    from tuberlens.probes.probe_factory import ProbeFactory

    _to_device_for_fit([train_ds, val_ds], verbose=False)
    kw = dict(probe_spec=spec, train_dataset=train_ds, model_name=model_name, layer=layer,
              validation_dataset=val_ds, pos_class_label=pos, neg_class_label=neg,
              probe_description=desc)
    if len(seeds) == 1:
        seed_everything(seeds[0])
        return ProbeFactory.build(use_store=False, **kw)
    build_ensemble = getattr(ProbeFactory, "build_ensemble", None)
    if build_ensemble is not None and fusion_enabled():
        seed_everything(seeds[0])
        members = build_ensemble(seeds=list(seeds), verbose=False, **kw)
    else:
        members = []
        for s in seeds:
            seed_everything(s)
            members.append(ProbeFactory.build(use_store=False, **kw))
    return EnsembleProbe.from_members(members, seeds)


def _proba(probe, dataset):
    import numpy as np
    p = np.asarray(probe.predict_proba(dataset))
    return p[:, -1] if p.ndim == 2 else p


# --------------------------------------------------------------------------------------
def _cv(dataset, *, k, seed, spec, seeds, model_name, layer, pos, neg, desc, tag,
        source_keys=None):
    """Grouped K-fold. test=fold k, val=fold (k+1)%K, train=the rest. Returns OOF scores."""
    import numpy as np

    fold_of, n_groups, _sizes = _grouped_folds(dataset, k, seed, source_keys)
    oof = np.full(len(dataset), np.nan)
    for f in range(k):
        test_idx = np.flatnonzero(fold_of == f)
        val_idx = np.flatnonzero(fold_of == (f + 1) % k)
        train_idx = np.flatnonzero((fold_of != f) & (fold_of != (f + 1) % k))
        t0 = time.time()
        probe = _fit(_subset(dataset, train_idx), _subset(dataset, val_idx),
                     spec=spec, seeds=seeds, model_name=model_name, layer=layer,
                     pos=pos, neg=neg, desc=desc)
        oof[test_idx] = _proba(probe, _subset(dataset, test_idx))
        print(f"      [{tag}] fold {f}: train {len(train_idx)} / val {len(val_idx)} / "
              f"test {len(test_idx)}  ({time.time()-t0:.0f}s)", flush=True)
        del probe
        gc.collect()
    assert not np.isnan(oof).any()
    return oof, n_groups


def main(argv=None) -> int:
    args = _parse_args(argv)
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from agentic_redteam.retrain import (_concatenate_consuming, _cpu_unpickle,
                                         _infer_probe_spec, _resolve_ensemble_seeds)

    with args.probe.open("rb") as f:
        bp = _cpu_unpickle(f)
    pos, neg, model_name, desc = (bp.pos_class_label, bp.neg_class_label,
                                  bp.model_name, bp.description)
    spec = _infer_probe_spec(bp)
    seeds = _resolve_ensemble_seeds(args.seed, args.ensemble_size)
    splits = args.splits or sorted(p.stem for p in args.eval_dir.glob("*.jsonl"))
    print(f"[cv] model={model_name} spec={spec.name} ensemble={len(seeds)} "
          f"folds={args.folds} splits={splits}")

    if args.report_groups:
        import collections
        from tuberlens.interfaces.dataset import LabelledDataset
        for split in splits:
            inputs, labels = [], []
            for d in (args.eval_dir, args.dev_dir):
                f = d / f"{split}.jsonl"
                if not f.exists():
                    continue
                ds = LabelledDataset.load_from(
                    f, pos_class_label=pos, neg_class_label=neg,
                    combine_consecutive_messages=args.combine_consecutive_messages,
                    convert_tool_to_assistant=args.convert_tool_to_assistant)
                inputs += list(ds.inputs)
                labels += list(ds.other_fields["labels"])
            sk = _source_keys(split, args.eval_dir, args.dev_dir)
            assert len(sk) == len(inputs), (split, len(sk), len(inputs))
            fold_of, n_groups, sizes = _grouped_folds(_Rows(inputs), args.folds,
                                                      args.seed, sk)
            per_fold = collections.Counter(fold_of.tolist())
            pos_fold = collections.Counter(f for f, l in zip(fold_of.tolist(), labels)
                                           if l == 1)
            print(f"  {split:28s} rows={len(inputs):4d} groups={n_groups:4d} "
                  f"size_hist={dict(sorted(collections.Counter(sizes).items()))} "
                  f"cols={GROUP_COLUMNS.get(split, [])}")
            print(f"  {'':28s} fold_rows={[per_fold[f] for f in range(args.folds)]} "
                  f"fold_pos={[pos_fold[f] for f in range(args.folds)]}")
        return 0

    results = {"model": model_name, "probe_spec": str(spec.name), "folds": args.folds,
               "ensemble_size": len(seeds), "seed": args.seed, "splits": splits,
               "per_split": {}, "pooled": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _save():
        args.out.write_text(json.dumps(results, indent=1))

    for layer in args.layers:
        print(f"\n================ LAYER {layer} ================", flush=True)

        # --- per-split CV ------------------------------------------------------------
        if args.mode in ("per-split", "both"):
            per = {}
            for split in splits:
                pool = _load_pool(split, layer, args, pos, neg, model_name)
                y = np.array(pool.other_fields["labels"], dtype=int)
                print(f"  -- {split}: {len(pool)} rows "
                      f"({int(y.sum())} pos) --", flush=True)
                sk = _source_keys(split, args.eval_dir, args.dev_dir)
                assert len(sk) == len(pool), (split, len(sk), len(pool))
                oof, n_groups = _cv(pool, k=args.folds, seed=args.seed, spec=spec,
                                    seeds=seeds, model_name=model_name, layer=layer,
                                    pos=pos, neg=neg, desc=desc, tag=f"L{layer}/{split}",
                                    source_keys=sk)
                auroc = float(roc_auc_score(y, oof))
                per[split] = {"n": len(pool), "n_groups": n_groups,
                              "oof_auroc": round(auroc, 4)}
                print(f"    -> {split} L{layer} OOF AUROC {auroc:.4f}", flush=True)
                del pool
                gc.collect(); torch.cuda.empty_cache()
            per["MEAN"] = round(float(np.mean([v["oof_auroc"] for v in per.values()])), 4)
            results["per_split"][str(layer)] = per
            print(f"  ==> LAYER {layer} per-split mean OOF AUROC {per['MEAN']:.4f}",
                  flush=True)
            _save()

        # --- pooled CV ---------------------------------------------------------------
        if args.mode in ("pooled", "both"):
            parts, origin, sk = [], [], []
            for split in splits:
                p = _load_pool(split, layer, args, pos, neg, model_name)
                origin += [split] * len(p)
                sk += _source_keys(split, args.eval_dir, args.dev_dir)
                parts.append(p)
            pool = _concatenate_consuming(parts)
            origin = np.array(origin)
            y = np.array(pool.other_fields["labels"], dtype=int)
            print(f"  -- POOLED: {len(pool)} rows, width "
                  f"{pool.other_fields['activations'].shape[1]} --",
                  flush=True)
            assert len(sk) == len(pool)
            oof, n_groups = _cv(pool, k=args.folds, seed=args.seed, spec=spec, seeds=seeds,
                                model_name=model_name, layer=layer, pos=pos, neg=neg,
                                desc=desc, tag=f"L{layer}/pooled", source_keys=sk)
            entry = {"n": len(pool), "n_groups": n_groups,
                     "oof_auroc_all": round(float(roc_auc_score(y, oof)), 4)}
            for split in splits:
                m = origin == split
                entry[split] = round(float(roc_auc_score(y[m], oof[m])), 4)
            entry["MEAN_per_split"] = round(
                float(np.mean([entry[s] for s in splits])), 4)
            results["pooled"][str(layer)] = entry
            print(f"  ==> LAYER {layer} pooled OOF AUROC {entry['oof_auroc_all']:.4f} "
                  f"(mean per split {entry['MEAN_per_split']:.4f})", flush=True)
            del pool
            gc.collect(); torch.cuda.empty_cache()
            _save()

    # --- summary ----------------------------------------------------------------------
    print("\n================ SUMMARY ================")
    if results["per_split"]:
        print("per-split CV (OOF AUROC):")
        hdr = ["layer"] + splits + ["MEAN"]
        print("  " + "  ".join(f"{h[:14]:>14}" for h in hdr))
        for layer in args.layers:
            r = results["per_split"].get(str(layer))
            if r:
                print("  " + "  ".join([f"{layer:>14}"] +
                      [f"{r[s]['oof_auroc']:>14.4f}" for s in splits] +
                      [f"{r['MEAN']:>14.4f}"]))
    if results["pooled"]:
        print("\npooled CV (one probe over all splits, OOF AUROC):")
        hdr = ["layer", "ALL"] + splits + ["MEAN"]
        print("  " + "  ".join(f"{h[:14]:>14}" for h in hdr))
        for layer in args.layers:
            r = results["pooled"].get(str(layer))
            if r:
                print("  " + "  ".join([f"{layer:>14}", f"{r['oof_auroc_all']:>14.4f}"] +
                      [f"{r[s]:>14.4f}" for s in splits] +
                      [f"{r['MEAN_per_split']:>14.4f}"]))
    _save()
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
