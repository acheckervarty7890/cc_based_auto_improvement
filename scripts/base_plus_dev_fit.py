#!/usr/bin/env python
"""base 50 + the lent `oig_omission` dev pairs, with NO red-team couples — the missing control.

`scripts/multimax_data_arms.py` runs three conditions — `base`, `base+couples`,
`base+couples+dev` — but never `base+dev`. That leaves its dev-lending arm without a control:
its 0.8643 could be the couples working once good data is present, or the dev rows doing all of
it. This runs the fourth cell.

**Protocol is `multimax_data_arms.py`'s exactly**, so the number is directly comparable to the
three it completes: the lent rows are the 16 `oig_omission` dev PAIRS `_dev_lending_indices`
selects, validation is the remaining 404 dev rows, the head is the base probe's own spec at
layer 32, and the ensemble is 10 members under the repo-pinned `ENSEMBLE_SEEDS` fit
sequentially. Scored on all 114 rows of `oig_omission`.

**Note there is an older, near-identical measurement** in
`scripts/train_base_plus_dev.py` (0.8695 against this script's 0.8717). It builds its
validation set by symlinking the six non-omission dev splits rather than by taking
`_dev_lending_indices`' complement — the same 404 rows by a different route — and merges the
lent rows into the base blob rather than concatenating a second dataset. The two differ by
0.0022, comfortably inside the ~0.013 row-order noise floor documented in
`docs/layer_and_head_sweep_2026-08-27.md`; they are the same measurement, not a disagreement.
Keep both: the pair is one of the few places where that floor has been observed directly on
two independent implementations.

All activations are read from the layer-sweep cache; no LLM is loaded.

    PROBE_FUSED_ENSEMBLE=0 .venv_claude/bin/python scripts/base_plus_dev_fit.py
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="oig_omission")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--dev-pairs", type=int, default=16)
    ap.add_argument("--ensemble-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--acts-root", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/activations")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results_instructions_gemma27b_layersweep/base_plus_dev_nocouples.json")
    args = ap.parse_args(argv)

    import numpy as np, torch
    from sklearn.metrics import roc_auc_score
    import multilayer_refit as mr
    from agentic_redteam.ensemble import EnsembleProbe, fusion_enabled
    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.retrain import (_concatenate_consuming, _cpu_unpickle,
        _dev_lending_groups, _dev_lending_indices, _infer_probe_spec, _load_dev_dataset,
        _resolve_ensemble_seeds, _to_device_for_fit, stable_train_test_split)
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.probes.probe_factory import ProbeFactory

    with mr.BP.open("rb") as f: bp = _cpu_unpickle(f)
    pos, neg = bp.pos_class_label, bp.neg_class_label
    spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(args.seed, args.ensemble_size)
    C = V = True; AR = args.acts_root; L = [args.layer]

    base = LabelledDataset.load_from(mr.BASE_JSONL, pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    BASE, _ = stable_train_test_split(base, test_size=0.0, split_field=None, seed=args.seed)

    dv, dfiles, dsizes = _load_dev_dataset(REPO / "dev_samples/instructions", pos, neg, C, V,
                                           verbose=False)
    parts = []
    for f, n in zip(dfiles, dsizes):
        ds = LabelledDataset.load_from(f, pos_class_label=pos, neg_class_label=neg,
            combine_consecutive_messages=C, convert_tool_to_assistant=V)
        assert len(ds) == n
        parts.append(mr._attach(ds, AR, L, "dev_samples", f.stem))
    DEV = _concatenate_consuming(parts)
    groups = _dev_lending_groups(DEV, dfiles, dsizes, args.split, "pairs")
    lent, val_idx = _dev_lending_indices(len(DEV), args.dev_pairs, args.dev_pairs, args.seed,
                                         groups=groups, verbose=False)
    LENT, VAL = DEV[lent], DEV[val_idx]
    del DEV

    train = _concatenate_consuming([
        mr._attach(BASE[list(range(len(BASE)))], AR, L, "extra", "base"),
        LENT[list(range(len(LENT)))]])
    print(f"[base+dev] train {len(train)} rows (base {len(BASE)} + lent {len(lent)}), "
          f"validation {len(VAL)}, NO couples; layer {args.layer}, fused={fusion_enabled()}")

    _to_device_for_fit([train, VAL], verbose=False)
    members = []
    for s in seeds:
        seed_everything(s)
        members.append(ProbeFactory.build(probe_spec=spec, train_dataset=train,
            model_name=bp.model_name, layer=args.layer, validation_dataset=VAL, use_store=False,
            pos_class_label=pos, neg_class_label=neg, probe_description=bp.description))
    probe = EnsembleProbe.from_members(members, list(seeds))

    ev = LabelledDataset.load_from(REPO / f"eval_sets/instructions/{args.split}.jsonl",
        pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    EV = mr._attach(ev[list(range(len(ev)))], AR, L, "eval_sets", args.split)
    y = np.array(EV.other_fields["labels"], dtype=int)
    p = np.asarray(probe.predict_proba(EV), dtype=float)
    if p.ndim == 2: p = p[:, -1]
    auroc = float(roc_auc_score(y, p))
    print(f"[base+dev] {args.split} AUROC {auroc:.4f}")
    del train, VAL, members, probe; torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"split": args.split, "layer": args.layer,
        "train_rows": len(BASE) + len(lent), "base_rows": len(BASE),
        "lent": len(lent), "lent_pairs": args.dev_pairs, "val": len(VAL),
        "ensemble": len(seeds), "seed": args.seed, "couples": None,
        "auroc": round(auroc, 4)}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
