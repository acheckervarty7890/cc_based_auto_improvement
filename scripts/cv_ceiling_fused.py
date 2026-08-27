#!/usr/bin/env python
"""Grouped CV ceiling on `oig_omission` for a single head AND for a fused combination.

`scripts/cv_ceiling.py` answers "what is achievable on this split at all" for one probe by
training on the split itself and scoring held-out rows. This does the same, but computes the
ceiling for **several components at once and then fuses their out-of-fold predictions**, so
the fused recipe gets a ceiling on the same footing as its parts.

Fusing OOF predictions is legitimate: each component's prediction for row *i* comes from a
fold that never saw row *i*, and the fusion weights are **fixed in advance** (they are the
recipe selected elsewhere), not fitted here. Nothing is selected on the held-out rows.

Protocol, matching `cv_ceiling.py` exactly so the single-head number is comparable to the
0.914 already on record:

* folds are cut over the split's 57 **sources** (`original_text`), never over rows -- the two
  rows of a source differ only in whether the reply is complete, so a row-level split leaks;
* validation is the split's own 32 **dev** rows, which are disjoint from the eval split, so
  early stopping never sees a held-out fold;
* 10-member ensembles under the repo-pinned `ENSEMBLE_SEEDS`;
* pooled out-of-fold AUROC (every row scored exactly once, by the fold that held it out),
  not a mean of per-fold AUROCs.

Activations come from `scripts/extract_multilayer_activations.py`; no LLM is loaded.

    PROBE_FUSED_ENSEMBLE=0 .venv_claude/bin/python scripts/cv_ceiling_fused.py
"""

from __future__ import annotations

import argparse, json, sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
LO = None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="oig_omission")
    ap.add_argument("--components", nargs="+",
                    default=["linear_then_last@L56", "linear_then_softmax@L32",
                             "linear_then_last@L16"],
                    help="arch@Llayer, in the order the --weights apply")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.25, 0.40, 0.35])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ensemble-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--acts-root", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/activations")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/cv_ceiling_fused.json")
    args = ap.parse_args(argv)
    assert len(args.weights) == len(args.components), "one weight per component"

    import numpy as np, torch
    from sklearn.metrics import roc_auc_score
    import multilayer_refit as mr
    from agentic_redteam.ensemble import EnsembleProbe, fusion_enabled
    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.retrain import (_cpu_unpickle, _resolve_ensemble_seeds,
                                         _to_device_for_fit)
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType
    from tuberlens.probes.probe_factory import ProbeFactory

    tiny, hi = np.finfo(float).tiny, np.nextafter(1.0, 0.0)
    def lg(p):
        p = np.clip(np.asarray(p, float), tiny, hi); return np.log(p) - np.log1p(-p)

    with mr.BP.open("rb") as f: bp = _cpu_unpickle(f)
    pos, neg = bp.pos_class_label, bp.neg_class_label
    seeds = _resolve_ensemble_seeds(args.seed, args.ensemble_size); C = V = True

    ev = LabelledDataset.load_from(REPO / f"eval_sets/instructions/{args.split}.jsonl",
        pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    dv = LabelledDataset.load_from(REPO / f"dev_samples/instructions/{args.split}.jsonl",
        pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    y = np.array(ev.other_fields["labels"], dtype=int)

    raw = [json.loads(l) for l in (REPO / f"eval_sets/instructions/{args.split}.jsonl").open()]
    assert len(raw) == len(ev)
    groups = OrderedDict()
    for i, r in enumerate(raw): groups.setdefault(r["original_text"], []).append(i)
    keys = list(groups)
    order = np.random.RandomState(args.seed).permutation(len(keys))
    fold_of = np.empty(len(ev), dtype=int)
    for rank, gi in enumerate(order):
        for row in groups[keys[gi]]: fold_of[row] = rank % args.folds
    print(f"[ceiling] {args.split}: {len(ev)} rows, {len(keys)} sources, {args.folds} folds, "
          f"validation = {len(dv)} dev rows, ensemble {len(seeds)}, fused={fusion_enabled()}")

    oof = {}
    for comp in args.components:
        arch, lay = comp.split("@L"); lay = int(lay)
        pt = ProbeType(arch); spec = ProbeSpec(name=pt, hyperparams=dict(pt.default_hyperparams))
        EV = mr._attach(ev[list(range(len(ev)))], args.acts_root, [lay], "eval_sets", args.split)
        DV = mr._attach(dv[list(range(len(dv)))], args.acts_root, [lay], "dev_samples", args.split)
        p = np.full(len(ev), np.nan)
        for k in range(args.folds):
            te = np.flatnonzero(fold_of == k); tr = np.flatnonzero(fold_of != k)
            train = EV[[int(i) for i in tr]]; val = DV[list(range(len(DV)))]
            _to_device_for_fit([train, val], verbose=False)
            members = []
            for s in seeds:
                seed_everything(s)
                members.append(ProbeFactory.build(probe_spec=spec, train_dataset=train,
                    model_name=bp.model_name, layer=lay, validation_dataset=val,
                    use_store=False, pos_class_label=pos, neg_class_label=neg,
                    probe_description=bp.description))
            probe = EnsembleProbe.from_members(members, list(seeds))
            q = np.asarray(probe.predict_proba(EV[[int(i) for i in te]]), dtype=float)
            p[te] = q[:, -1] if q.ndim == 2 else q
            del train, val, members, probe; torch.cuda.empty_cache()
        assert not np.isnan(p).any()
        oof[comp] = p
        print(f"  {comp:<28} pooled OOF ceiling {roc_auc_score(y, p):.4f}", flush=True)

    fused = sum(w * lg(oof[c]) for c, w in zip(args.components, args.weights))
    res = {"split": args.split, "n": len(ev), "sources": len(keys), "folds": args.folds,
           "ensemble": len(seeds), "seed": args.seed,
           "components": {c: round(float(roc_auc_score(y, oof[c])), 4) for c in args.components},
           "weights": dict(zip(args.components, args.weights)),
           "fused_ceiling": round(float(roc_auc_score(y, fused)), 4)}
    print(f"\n  FUSED ({' + '.join(f'{w:.2f}·{c}' for c,w in zip(args.components,args.weights))})"
          f"\n  pooled OOF ceiling {res['fused_ceiling']:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
