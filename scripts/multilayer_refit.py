#!/usr/bin/env python
"""Refit the probe on TWO layers' activations concatenated, and re-evaluate.

``scripts/layer_cv_sweep.py`` measures each layer's ceiling in isolation. This asks the
next question: do the best two layers carry *different* information, or the same
information twice? If different, a head reading both should beat either alone.

The features are concatenated along the EMBEDDING axis — layer A's 5376 dims followed by
layer B's, giving a 10752-dim input to the same ``LinearThenSoftmax`` head. Nothing else
about the head changes; ``LinearThenAgg`` takes its ``embed_dim`` from the data, so a
wider input is simply a wider ``nn.Linear``. Note this doubles the head's parameter count
(5,377 -> 10,753) on a 116-row training set, which is the reason the single-layer controls
below are not optional: a two-layer arm that wins by a hair is indistinguishable from one
that has merely been given more capacity to overfit, and the L32 control says which.

**The protocol is `scripts/multimax_data_arms.py`'s, unchanged apart from the features.**
Same training data (the 50 base rows + the v3 run's 33 red-team couples = 116 rows), same
10-member ensembles under the repo-pinned ``ENSEMBLE_SEEDS``, same base probe's schedule,
same evaluation on all seven full ``eval_sets/instructions`` splits. Crucially the same
**validation set**: the 32 ``oig_omission`` dev rows that that script lends to training in
its third condition are withheld from validation here too, so this run's L32 control should
reproduce ``eval_mm_base+couples__stock.csv`` (mean AUROC 0.7947) exactly. That
reproduction is the harness check — it is printed, and a mismatch means the comparison
below is not comparable to anything already measured.

**Scoring does not go through ``evaluate_probe``.** That path calls tuberlens'
``get_performances``, which extracts activations at ``probe.layer`` — a single int, which a
two-layer probe does not have. Splits are scored directly with ``predict_proba`` over the
pre-concatenated blobs instead, and the metric is ``roc_auc_score(y_true, y_pred)``, which
is precisely what ``get_performances`` computes (``tuberlens/evaluation.py:40``). The L32
reproduction above is what establishes the two paths agree.

For the same reason the pickles written here are **not** loadable by the normal pipeline:
their ``layer`` field names only the deepest layer, and feeding one to ``evaluate_probe`` or
``ProbeJudge`` would score it on 5376 features it was never fit on. They are written for
inspection, under names that say so.

Activations come from ``scripts/extract_multilayer_activations.py``. No LLM is loaded.

    .venv_claude/bin/python scripts/multilayer_refit.py --layers 40 48
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PRB = REPO / "probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP = REPO / "probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE_JSONL = REPO / "data/instructions_llama70b_50.jsonl"
COUPLES_JSONL = PRB / "redteam_postprocessed_iter5.jsonl"

# multimax_data_arms.py's reserve, reproduced so the validation set matches: 16 pairs is
# the whole oig_omission dev pool (32 of the 436 dev rows).
DEV_PAIRS = 16
DEV_SPLIT = "oig_omission"


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concat", action="append", default=[], metavar="A,B",
                    help="Repeatable. A comma-separated layer group to concatenate, e.g. "
                         "--concat 24,32 --concat 32,40. Order fixes the feature order. "
                         "More than one group is worth running when the CV leaves second "
                         "place unresolved — the runners-up here sit within 0.006 AUROC of "
                         "each other, which is not a gap that picks a layer.")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Shorthand for a single --concat group.")
    ap.add_argument("--controls", type=int, nargs="+", default=None,
                    help="Single-layer arms to fit alongside (default: each --layers "
                         "entry, plus 32 as the incumbent).")
    ap.add_argument("--acts-root", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/activations")
    ap.add_argument("--eval-dir", type=Path, default=REPO / "eval_sets/instructions")
    ap.add_argument("--dev-dir", type=Path, default=REPO / "dev_samples/instructions")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep")
    ap.add_argument("--ensemble-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-probes", action="store_true")
    return ap.parse_args(argv)


def _blob(acts_root, layer, group, name):
    import torch
    path = Path(acts_root) / f"L{layer}" / group / f"{name}-acts_full.pt"
    if not path.exists():
        raise SystemExit(f"missing activation blob: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _concat_layers(acts_root, layers, group, name, *, expect_rows=None):
    """One source's activations at several layers, concatenated on the EMBEDDING axis.

    The layers come from a single forward pass over a single tokenization, so the blobs
    share ``input_ids`` and ``attention_mask`` exactly — asserted rather than assumed,
    because a mismatch would silently glue together two different tokenizations and the
    fit would still run.
    """
    import torch

    first = _blob(acts_root, layers[0], group, name)
    acts = [first["activations"]]
    for layer in layers[1:]:
        b = _blob(acts_root, layer, group, name)
        assert torch.equal(b["input_ids"], first["input_ids"]), (group, name, layer)
        assert torch.equal(b["attention_mask"], first["attention_mask"]), (group, name, layer)
        acts.append(b["activations"])
    out = torch.cat(acts, dim=-1) if len(acts) > 1 else acts[0]
    if expect_rows is not None and out.shape[0] != expect_rows:
        raise SystemExit(f"{group}/{name}: blob has {out.shape[0]} rows, "
                         f"dataset has {expect_rows}")
    return out, first["attention_mask"], first["input_ids"]


def _attach(ds, acts_root, layers, group, name):
    a, m, i = _concat_layers(acts_root, layers, group, name, expect_rows=len(ds))
    return ds.assign(activations=a, attention_mask=m, input_ids=i)


def main(argv=None) -> int:
    args = _parse_args(argv)
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from agentic_redteam.ensemble import EnsembleProbe, fusion_enabled
    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.retrain import (
        _apply_message_transforms, _concatenate_consuming, _cpu_unpickle,
        _dev_lending_groups, _dev_lending_indices, _infer_probe_spec, _load_dev_dataset,
        _resolve_ensemble_seeds, _to_device_for_fit, stable_train_test_split,
    )
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage
    from tuberlens.probes.probe_factory import ProbeFactory

    with BP.open("rb") as f:
        bp = _cpu_unpickle(f)
    pos, neg = bp.pos_class_label, bp.neg_class_label
    spec = _infer_probe_spec(bp)
    seeds = _resolve_ensemble_seeds(args.seed, args.ensemble_size)
    C = V = True
    TS = 0.0
    AR = args.acts_root
    args.out_dir.mkdir(parents=True, exist_ok=True)

    groups = [[int(x) for x in g.split(",")] for g in args.concat]
    if args.layers:
        groups.append(list(args.layers))
    if not groups:
        raise SystemExit("pass --concat A,B (repeatable) or --layers A B")
    # Controls: every layer that appears in any group, plus 32 (the incumbent, and the
    # arm whose number reproduces an already-published one). Fitted once each, not once
    # per group.
    control_layers = (args.controls if args.controls is not None
                      else sorted({l for g in groups for l in g} | {32}))
    arms = {f"L{layer}": [layer] for layer in control_layers}
    for g in groups:
        arms["+".join(f"L{l}" for l in g)] = g

    # ---- base (50 rows) ---------------------------------------------------------------
    base = LabelledDataset.load_from(
        BASE_JSONL, pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    BASE_DS, _ = stable_train_test_split(base, test_size=TS, split_field=None,
                                         seed=args.seed)

    # ---- the 33 red-team couples (66 rows) --------------------------------------------
    rows = [json.loads(line) for line in COUPLES_JSONL.open()]
    rt = LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]]
                for r in rows],
        ids=[r["id"] for r in rows],
        other_fields={"labels": [r["label"] for r in rows]})
    rt = _apply_message_transforms(rt, C, V)
    RT_DS, _ = stable_train_test_split(rt, test_size=TS, split_field=None, seed=args.seed)

    # ---- dev: the same partition multimax_data_arms.py used ---------------------------
    # `_load_dev_dataset` concatenates the dir's *.jsonl in sorted() order, which is the
    # order the per-split blobs are re-concatenated in below; the row counts are asserted.
    dv, dfiles, dsizes = _load_dev_dataset(args.dev_dir, pos, neg, C, V, verbose=False)
    dev_stems = [f.stem for f in dfiles]

    # ---- eval splits ------------------------------------------------------------------
    eval_stems = sorted(p.stem for p in args.eval_dir.glob("*.jsonl"))
    eval_raw = {
        s: LabelledDataset.load_from(
            args.eval_dir / f"{s}.jsonl", pos_class_label=pos, neg_class_label=neg,
            combine_consecutive_messages=C, convert_tool_to_assistant=V)
        for s in eval_stems
    }

    print(f"[refit] base {len(BASE_DS)} | couples {len(RT_DS)} ({len(rows)//2} couples) | "
          f"dev {len(dv)} | eval {sum(len(d) for d in eval_raw.values())} rows "
          f"over {len(eval_stems)} splits")
    print(f"[refit] spec={spec.name} ensemble={len(seeds)} fused={fusion_enabled()}")
    print(f"[refit] arms: {', '.join(f'{k} {v}' for k, v in arms.items())}\n", flush=True)

    results, csv_rows = {}, {}

    for tag, layers in arms.items():
        t0 = time.perf_counter()
        print(f"=== ARM {tag}  layers {layers} ===", flush=True)

        train = _concatenate_consuming([
            _attach(BASE_DS[list(range(len(BASE_DS)))], AR, layers, "extra", "base"),
            _attach(RT_DS[list(range(len(RT_DS)))], AR, layers, "extra", "couples"),
        ])

        dev_parts = []
        for stem, n in zip(dev_stems, dsizes):
            ds = LabelledDataset.load_from(
                args.dev_dir / f"{stem}.jsonl", pos_class_label=pos, neg_class_label=neg,
                combine_consecutive_messages=C, convert_tool_to_assistant=V)
            assert len(ds) == n, (stem, len(ds), n)
            dev_parts.append(_attach(ds, AR, layers, "dev_samples", stem))
        DEV_ALL = _concatenate_consuming(dev_parts)
        assert len(DEV_ALL) == len(dv)
        groups = _dev_lending_groups(DEV_ALL, dfiles, dsizes, DEV_SPLIT, "pairs")
        lent_idx, val_idx = _dev_lending_indices(
            len(DEV_ALL), DEV_PAIRS, DEV_PAIRS, args.seed, groups=groups, verbose=False)
        VAL = DEV_ALL[val_idx]
        del DEV_ALL

        embed = train.other_fields["activations"].shape[-1]
        print(f"  train {len(train)} rows, embed {embed} | validation {len(VAL)} of "
              f"{len(dv)} dev rows ({len(lent_idx)} withheld)", flush=True)

        _to_device_for_fit([train, VAL], verbose=False)
        seed_everything(seeds[0])
        build_ensemble = getattr(ProbeFactory, "build_ensemble", None)
        kw = dict(probe_spec=spec, train_dataset=train, model_name=bp.model_name,
                  layer=max(layers), validation_dataset=VAL, pos_class_label=pos,
                  neg_class_label=neg,
                  probe_description=f"{bp.description} [layers {'+'.join(map(str, layers))}]")
        if build_ensemble is not None and fusion_enabled():
            members = build_ensemble(seeds=list(seeds), verbose=False, **kw)
        else:
            members = []
            for s in seeds:
                seed_everything(s)
                members.append(ProbeFactory.build(use_store=False, **kw))
        probe = EnsembleProbe.from_members(members, list(seeds))
        del train, VAL
        torch.cuda.empty_cache()

        # ---- score every eval split directly ------------------------------------------
        per = {}
        for s in eval_stems:
            ds = _attach(eval_raw[s][list(range(len(eval_raw[s])))], AR, layers,
                         "eval_sets", s)
            y = np.array(ds.other_fields["labels"], dtype=int)
            p = np.asarray(probe.predict_proba(ds), dtype=float)
            if p.ndim == 2:
                p = p[:, -1]
            per[s] = float(roc_auc_score(y, p))
            del ds
            torch.cuda.empty_cache()
        per["mean"] = float(np.mean([per[s] for s in eval_stems]))
        results[tag] = {"layers": layers, "embed": int(embed),
                        "auroc": {k: round(v, 4) for k, v in per.items()}}
        csv_rows[tag] = per

        if args.save_probes:
            # Named so it can't be mistaken for a pipeline probe: its `layer` field names
            # only the deepest layer, and evaluate_probe would score it on the wrong width.
            q = args.out_dir / f"probe_MULTILAYER_{tag}_NOT_PIPELINE_LOADABLE.pkl"
            with q.open("wb") as f:
                pickle.dump(probe, f)
        del probe, members
        torch.cuda.empty_cache()

        print(f"  -> mean {per['mean']:.4f} | " +
              " ".join(f"{s.split('_')[0]} {per[s]:.3f}" for s in eval_stems) +
              f"  ({time.perf_counter()-t0:.0f}s)\n", flush=True)

        (args.out_dir / "multilayer_refit.json").write_text(json.dumps(
            {"seed": args.seed, "ensemble_size": len(seeds), "spec": str(spec.name),
             "train": "base(50)+couples(66)", "dev_withheld_pairs": DEV_PAIRS,
             "dev_withheld_split": DEV_SPLIT, "arms": results}, indent=1))
        with (args.out_dir / f"eval_multilayer_{tag}.csv").open("w") as f:
            f.write("dataset,auroc\n")
            for s in eval_stems + ["mean"]:
                f.write(f"{s},{per[s]}\n")

    # ---- summary ----------------------------------------------------------------------
    print("\n================ eval AUROC (base+couples, 10-member ensemble) ============")
    w = max(len(t) for t in arms)
    print(f"{'arm':<{w}}  {'embed':>6}  " + "  ".join(f"{s[:16]:>16}" for s in eval_stems)
          + f"  {'MEAN':>8}")
    for tag in arms:
        r = csv_rows[tag]
        print(f"{tag:<{w}}  {results[tag]['embed']:>6}  "
              + "  ".join(f"{r[s]:>16.4f}" for s in eval_stems)
              + f"  {r['mean']:>8.4f}")

    ref = 0.7947
    if "L32" in csv_rows:
        got = csv_rows["L32"]["mean"]
        ok = abs(got - ref) < 5e-4
        print(f"\nharness check: L32 control mean {got:.4f} vs "
              f"eval_mm_base+couples__stock.csv {ref:.4f} -> "
              f"{'REPRODUCED' if ok else 'MISMATCH — see the docstring'}")
    print(f"\nwrote {args.out_dir / 'multilayer_refit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
