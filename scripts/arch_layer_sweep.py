#!/usr/bin/env python
"""Every tuberlens probe architecture, at every extracted layer, scored on one split.

The layer sweep so far has varied the LAYER with the head held at `linear_then_softmax`
(the incumbent), and `scripts/multimax_data_arms.py` varied the HEAD with the layer held at
32. This crosses the two: all 11 `ProbeType`s x all 6 extracted layers, 66 fits, one table.

**Hyperparameters are tuberlens' own `ProbeType.default_hyperparams`, untouched.** That is
not a shortcut -- it happens that the default for `linear_then_softmax` is byte-identical to
the spec `_infer_probe_spec` reads off the incumbent probe, so this protocol reproduces the
established L32 anchor (0.7119 on `oig_omission`) exactly, while giving every other head the
settings its author intended (`attention` and `multimax` want batch 128 / gas 1 /
final_lr 5e-4, not the linear heads' batch 16 / gas 4). Hand-normalising the schedule across
heads would have moved the anchor and privileged the incumbent's settings.

**Ensemble size is 10, except for the three closed-form architectures** (`sklearn`,
`difference_of_means`, `lda` -- `ensemble.DETERMINISTIC_ARCHS`), which are fit once. Ten
seeds there produce ten identical members and an average equal to a single probe, at ten
times the cost; one fit is exactly equivalent, not an approximation.

Everything else matches `scripts/multilayer_refit.py`: base(50) + couples(66) training,
the 404-row validation set with the 16 `oig_omission` dev pairs withheld, repo-pinned
ENSEMBLE_SEEDS, sequential fits.

Blobs are loaded **once per layer** and reused across all 11 heads, so the run costs 6 blob
loads rather than 66 -- loading, not fitting, is what dominates here.

Per-row scores are written for the eval split, the 32 withheld dev rows and the 404-row
validation set, in the schema `scripts/weighted_combine_sweep.py` consumes, so the
combination sweep can run straight off this file without refitting.

A head that fails to fit or score is recorded as an error and skipped rather than killing
the run -- some architectures may not accept these shapes.

    PROBE_FUSED_ENSEMBLE=0 .venv_claude/bin/python scripts/arch_layer_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, nargs="+", default=[16, 24, 32, 40, 48, 56])
    ap.add_argument("--archs", nargs="+", default=None, help="default: every ProbeType")
    ap.add_argument("--split", default="oig_omission")
    ap.add_argument("--eval-dir", type=Path, default=REPO / "eval_sets/instructions")
    ap.add_argument("--dev-dir", type=Path, default=REPO / "dev_samples/instructions")
    ap.add_argument("--acts-root", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/activations")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/arch_layer_sweep.json")
    ap.add_argument("--couples", type=Path, default=None,
                    help="Red-team couples JSONL (default: the v3 run's iter5, 33 couples). "
                         "Point at another run's redteam_postprocessed_iter{N}.jsonl to "
                         "study the couples accumulated up to that iteration.")
    ap.add_argument("--couples-name", default="couples",
                    help="Blob basename under <acts-root>/L*/extra/. MUST differ per "
                         "couples file or one set's activations would be read for another.")
    ap.add_argument("--dev-withhold-pairs", type=int, default=16,
                    help="oig_omission dev PAIRS withheld from validation (default 16 = the "
                         "whole pool, matching multimax_data_arms.py, and the source of the "
                         "held-out rows used for threshold/weight selection). Set 0 to "
                         "validate on all 436 dev rows, which is what the pipeline arms do.")
    ap.add_argument("--train", choices=["base", "base+couples"], default="base+couples",
                    help="Training condition, matching multimax_data_arms.py's names. "
                         "'base' is the 50 llama70b rows alone -- the probe with no "
                         "red-team data at all; 'base+couples' adds the v3 run's 33 "
                         "couples (116 rows). The validation set is identical in both, so "
                         "the checkpoints stay comparable.")
    ap.add_argument("--ensemble-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    import multilayer_refit as mr
    from agentic_redteam.ensemble import DETERMINISTIC_ARCHS, EnsembleProbe, fusion_enabled
    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.retrain import (
        _apply_message_transforms, _concatenate_consuming, _cpu_unpickle,
        _dev_lending_groups, _dev_lending_indices, _load_dev_dataset,
        _resolve_ensemble_seeds, _to_device_for_fit, stable_train_test_split,
    )
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType
    from tuberlens.probes.probe_factory import ProbeFactory

    archs = args.archs or [t.value for t in ProbeType]
    with mr.BP.open("rb") as f:
        bp = _cpu_unpickle(f)
    pos, neg = bp.pos_class_label, bp.neg_class_label
    C = V = True
    AR = args.acts_root
    print(f"[sweep] {len(archs)} archs x {len(args.layers)} layers = "
          f"{len(archs)*len(args.layers)} fits; split={args.split}; "
          f"train={args.train}; fused={fusion_enabled()}")
    print(f"[sweep] archs: {', '.join(archs)}\n", flush=True)

    # ---- data, built once; only the activations change per layer ----------------------
    base = LabelledDataset.load_from(
        mr.BASE_JSONL, pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    BASE_DS, _ = stable_train_test_split(base, test_size=0.0, split_field=None, seed=args.seed)
    couples_path = args.couples or mr.COUPLES_JSONL
    rows = [json.loads(line) for line in couples_path.open()]
    rt = LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]]
                for r in rows],
        ids=[r["id"] for r in rows], other_fields={"labels": [r["label"] for r in rows]})
    rt = _apply_message_transforms(rt, C, V)
    RT_DS, _ = stable_train_test_split(rt, test_size=0.0, split_field=None, seed=args.seed)
    dv, dfiles, dsizes = _load_dev_dataset(args.dev_dir, pos, neg, C, V, verbose=False)
    ev = LabelledDataset.load_from(
        args.eval_dir / f"{args.split}.jsonl", pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    y = np.array(ev.other_fields["labels"], dtype=int)

    # Scores are stored at FULL precision, deliberately. Rounding them (an earlier
    # version used round(x, 6)) destroys the ranking for saturating heads: the
    # linear_then_last probes put 91 of 114 eval scores below 1e-6, so six decimals
    # collapsed them all to exactly 0.0 -- 20 unique values out of 114 -- and AUROC
    # fell 0.7867 -> 0.6716 on stored scores that the fit-time metric had read
    # correctly. AUROC is rank-based, so precision loss near zero is not cosmetic.
    out = {"split": args.split, "train": args.train,
           "couples": str(couples_path), "n_couples_rows": len(rows),
           "dev_withhold_pairs": args.dev_withhold_pairs,
           "labels": y.tolist(), "arms": [],
           "scores": {}, "heldout": {}, "val": {}, "auroc": {}, "errors": {}}

    for layer in args.layers:
        layers = [layer]
        print(f"================ LAYER {layer} ================", flush=True)
        parts_tr = [mr._attach(BASE_DS[list(range(len(BASE_DS)))], AR, layers,
                               "extra", "base")]
        if args.train == "base+couples":
            parts_tr.append(mr._attach(RT_DS[list(range(len(RT_DS)))], AR, layers,
                                       "extra", args.couples_name))
        train0 = (_concatenate_consuming(parts_tr) if len(parts_tr) > 1 else parts_tr[0])
        parts = []
        for f, n in zip(dfiles, dsizes):
            ds = LabelledDataset.load_from(
                f, pos_class_label=pos, neg_class_label=neg,
                combine_consecutive_messages=C, convert_tool_to_assistant=V)
            parts.append(mr._attach(ds, AR, layers, "dev_samples", f.stem))
        DEV_ALL = _concatenate_consuming(parts)
        if args.dev_withhold_pairs > 0:
            g = _dev_lending_groups(DEV_ALL, dfiles, dsizes, mr.DEV_SPLIT, "pairs")
            lent, val_idx = _dev_lending_indices(
                len(DEV_ALL), args.dev_withhold_pairs, args.dev_withhold_pairs,
                args.seed, groups=g, verbose=False)
        else:
            # The pipeline arms' setting: the whole dev set validates, nothing is withheld.
            # There is then no clean set for choosing thresholds or fusion weights -- that
            # is the trade, and it is why the default withholds.
            lent, val_idx = [], list(range(len(DEV_ALL)))
        VAL0 = DEV_ALL[val_idx]
        HELD0 = DEV_ALL[lent] if lent else None
        del DEV_ALL
        EV0 = mr._attach(ev[list(range(len(ev)))], AR, layers, "eval_sets", args.split)
        y_val = np.array(VAL0.other_fields["labels"], dtype=int)
        y_held = (np.array(HELD0.other_fields["labels"], dtype=int)
                  if HELD0 is not None else np.array([], dtype=int))

        for arch in archs:
            name = f"{arch}@L{layer}"
            t0 = time.perf_counter()
            try:
                pt = ProbeType(arch)
                spec = ProbeSpec(name=pt, hyperparams=dict(pt.default_hyperparams))
                n_ens = 1 if arch in DETERMINISTIC_ARCHS else args.ensemble_size
                seeds = _resolve_ensemble_seeds(args.seed, n_ens)
                # Fresh copies: _concatenate_consuming/_to_device_for_fit mutate, and every
                # head must see the same starting tensors.
                train = train0[list(range(len(train0)))]
                VAL = VAL0[list(range(len(VAL0)))]
                _to_device_for_fit([train, VAL], verbose=False)
                kw = dict(probe_spec=spec, train_dataset=train, model_name=bp.model_name,
                          layer=layer, validation_dataset=VAL, pos_class_label=pos,
                          neg_class_label=neg, probe_description=bp.description)
                seed_everything(seeds[0])
                be = getattr(ProbeFactory, "build_ensemble", None)
                if len(seeds) > 1 and be is not None and fusion_enabled():
                    members = be(seeds=list(seeds), verbose=False, **kw)
                else:
                    members = []
                    for s in seeds:
                        seed_everything(s)
                        members.append(ProbeFactory.build(use_store=False, **kw))
                probe = (members[0] if len(members) == 1
                         else EnsembleProbe.from_members(members, list(seeds)))

                def sc(dataset):
                    p = np.asarray(probe.predict_proba(dataset), dtype=float)
                    return p[:, -1] if p.ndim == 2 else p

                pe, pv = sc(EV0), sc(VAL0)
                ph = sc(HELD0) if HELD0 is not None else np.array([])
                a = float(roc_auc_score(y, pe))
                out["arms"].append(name)
                out["scores"][name] = [float(x) for x in pe]
                out["heldout"][name] = {"scores": [float(x) for x in ph],
                                        "labels": y_held.tolist()}
                if HELD0 is None:
                    out["heldout"].pop(name, None)
                out["val"][name] = {"scores": [float(x) for x in pv],
                                    "labels": y_val.tolist()}
                out["auroc"][name] = round(a, 4)
                print(f"  {name:<34} n_ens {n_ens:>2}  AUROC {a:.4f}  "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
                del train, VAL, members, probe
            except Exception as exc:  # noqa: BLE001 - one bad head must not kill 65 fits
                out["errors"][name] = f"{type(exc).__name__}: {exc}"
                print(f"  {name:<34} FAILED  {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
            torch.cuda.empty_cache()
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(out, indent=1))

        del train0, VAL0, EV0
        HELD0 = None
        torch.cuda.empty_cache()

    # ---- table -------------------------------------------------------------------------
    print(f"\n================ {args.split} eval AUROC ================")
    print(f"{'architecture':<26}" + "".join(f"{'L'+str(l):>9}" for l in args.layers)
          + f"{'best':>9}")
    for arch in archs:
        cells = [out["auroc"].get(f"{arch}@L{l}") for l in args.layers]
        best = max([c for c in cells if c is not None], default=None)
        print(f"{arch:<26}"
              + "".join(f"{c:>9.4f}" if c is not None else f"{'--':>9}" for c in cells)
              + (f"{best:>9.4f}" if best is not None else f"{'--':>9}"))
    if out["auroc"]:
        top = sorted(out["auroc"].items(), key=lambda kv: -kv[1])[:10]
        print("\ntop 10 (arch@layer):")
        for k, v in top:
            print(f"  {k:<34}{v:.4f}")
    if out["errors"]:
        print(f"\n{len(out['errors'])} failed: {', '.join(out['errors'])}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
