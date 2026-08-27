#!/usr/bin/env python
"""Leave-one-out over the BASE training samples, scored on one eval split.

`scripts/pair_selection_study.py` asks which of a retrain's red-team couples the probe is
better for. This asks the same question of the 50 rows that every probe in this project is
built on -- the ones nothing has ever varied.

    delta_i = AUROC(without base row i) - AUROC(with all 50)

so **delta > 0 means row i was HURTING the split** (removing it helped) and delta < 0 means
it was helping, the same sign convention as the pair study: the question is "what is this
training sample doing to me", not "what does removing it do".

Two configurations, since the base data plays a different role in each:

* default -- base rows alone, so the row's own contribution is isolated. Baseline is the
  base-only probe (0.797 on oig_omission at the time of writing).
* ``--with-couples`` -- base rows plus the run's 33 red-team couples, the configuration
  every architecture experiment used. Baseline 0.714.

Fitting mirrors `retrain._train_with_cached_base_activations`'s tail exactly: the cached
base / dev / red-team activations are read from disk, merged with `_concatenate_consuming`,
staged on the GPU by `_to_device_for_fit`, and handed to `ProbeFactory.build_ensemble`
under the repo-pinned ENSEMBLE_SEEDS -- the same fused path a real retrain takes. The base
blob is loaded ONCE and subset by index per fit, so no extraction model is ever loaded and
each of the 51 fits costs a probe-head fit and nothing else.

The baseline is refit here rather than quoted, so every delta is measured against a probe
produced by this exact code path.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-probe", type=Path,
                    default=REPO / "probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl")
    ap.add_argument("--base-training-data", type=Path,
                    default=REPO / "data/instructions_llama70b_50.jsonl")
    ap.add_argument("--couples", type=Path,
                    default=REPO / "probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3/redteam_postprocessed_iter5.jsonl")
    ap.add_argument("--with-couples", action="store_true",
                    help="train on base + the 33 red-team couples (default: base alone)")
    ap.add_argument("--dev-data", type=Path, default=REPO / "dev_samples/instructions")
    ap.add_argument("--eval-dataset-dir", type=Path, default=REPO / "eval_sets/instructions")
    ap.add_argument("--split", default="oig_omission")
    ap.add_argument("--base-activation-cache-dir", type=Path,
                    default=REPO / "results_instructions_gemma27b_shared/base_activations")
    ap.add_argument("--activations-cache-dir", type=Path,
                    default=REPO / "results_instructions_gemma27b_shared/eval_activations")
    ap.add_argument("--ensemble-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/base_study"))
    args = ap.parse_args(argv)

    import torch
    from agentic_redteam.ensemble import EnsembleProbe, fusion_enabled
    from agentic_redteam.evaluation import evaluate_probe, seed_everything
    from agentic_redteam.retrain import (
        _activate_redteam_cached, _base_activation_cache_paths, _concatenate_consuming,
        _cpu_unpickle, _dev_activation_cache_path, _infer_probe_spec, _load_dev_dataset,
        _resolve_ensemble_seeds, _to_device_for_fit, stable_train_test_split,
    )
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage
    from tuberlens.model import LLMModel
    from tuberlens.probes.probe_factory import ProbeFactory

    tag = "with_couples" if args.with_couples else "base_only"
    out = args.out or (REPO / "results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
                       / f"base_selection_{args.split}_{tag}.json")
    args.workdir.mkdir(parents=True, exist_ok=True)

    with args.base_probe.open("rb") as f:
        bp = _cpu_unpickle(f)
    pos, neg = bp.pos_class_label, bp.neg_class_label
    spec = _infer_probe_spec(bp)
    seeds = _resolve_ensemble_seeds(args.seed, args.ensemble_size)
    TS = 0.0; C = V = True                      # --dev-data => base trains in FULL

    base = LabelledDataset.load_from(
        Path(args.base_training_data), pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=C, convert_tool_to_assistant=V)
    btr, _ = stable_train_test_split(base, test_size=TS, split_field=None, seed=args.seed)
    btc, _ = _base_activation_cache_paths(
        args.base_activation_cache_dir, args.base_training_data, bp.model_name, bp.layer,
        args.seed, TS, None, C, V, 1.0)
    a = LLMModel.load_activations(btc)
    BASE = btr.assign(activations=a.activations, attention_mask=a.attention_mask,
                      input_ids=a.input_ids)

    dv, dfiles, _sz = _load_dev_dataset(args.dev_data, pos, neg, C, V, verbose=False)
    da = LLMModel.load_activations(
        _dev_activation_cache_path(args.base_activation_cache_dir, dfiles, bp.model_name,
                                   bp.layer, C, V))
    VAL = dv.assign(activations=da.activations, attention_mask=da.attention_mask,
                    input_ids=da.input_ids)

    RT = None
    if args.with_couples:
        rows = [json.loads(l) for l in args.couples.open()]
        rt = LabelledDataset(
            inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]]
                    for r in rows],
            ids=[r["id"] for r in rows],
            other_fields={"labels": [r["label"] for r in rows]})
        rtr, _ = stable_train_test_split(rt, test_size=TS, split_field=None, seed=args.seed)
        RT = _activate_redteam_cached(rtr, args.base_activation_cache_dir, bp.model_name,
                                      bp.layer, C, V, lambda: None, False)

    n = len(BASE)
    # `other_fields["labels"]` holds ints once loaded; the `labels` property resolves them
    # back to the canonical Label enum. Render the probe's own class-label strings, so the
    # log and the JSON say which class a row is rather than "0"/"1".
    names = {"positive": pos, "negative": neg}
    labels = [names.get(l.value, l.value) for l in BASE.labels]
    print(f"{n} base rows ({labels.count(pos)} {pos} / {labels.count(neg)} {neg})"
          f"{f' + {len(RT)} red-team rows' if RT is not None else ''}; "
          f"scoring '{args.split}'; fused={fusion_enabled()}", flush=True)

    # The dev set is by far the largest tensor here and is identical for every fit, so it is
    # staged on the card once; the per-fit training set is small and rides along.
    _to_device_for_fit([VAL], verbose=True)

    def fit_and_score(keep: list[int], name: str) -> float:
        # _concatenate_consuming CONSUMES its inputs, so every fit gets fresh copies. Indexing
        # with a list is advanced indexing, which copies -- including on the GPU, so a staged
        # source stays staged.
        parts = [BASE[keep]] + ([RT[list(range(len(RT)))]] if RT is not None else [])
        train = _concatenate_consuming(parts) if len(parts) > 1 else parts[0]
        _to_device_for_fit([train], verbose=False)
        seed_everything(seeds[0])
        build_ensemble = getattr(ProbeFactory, "build_ensemble", None)
        if build_ensemble is not None and fusion_enabled():
            members = build_ensemble(
                probe_spec=spec, train_dataset=train, model_name=bp.model_name,
                layer=bp.layer, seeds=list(seeds), validation_dataset=VAL,
                pos_class_label=pos, neg_class_label=neg,
                probe_description=bp.description, verbose=False)
        else:
            members = []
            for s in seeds:
                seed_everything(s)
                members.append(ProbeFactory.build(
                    probe_spec=spec, train_dataset=train, model_name=bp.model_name,
                    layer=bp.layer, validation_dataset=VAL, use_store=False,
                    pos_class_label=pos, neg_class_label=neg,
                    probe_description=bp.description))
        probe = EnsembleProbe.from_members(members, list(seeds))
        p = args.workdir / f"probe_{name}.pkl"
        with p.open("wb") as f:
            pickle.dump(probe, f)
        df = evaluate_probe(
            probe_path=p, eval_dataset_dir=args.eval_dataset_dir,
            activations_cache_dir=args.activations_cache_dir, splits=[args.split],
            max_samples=None, seed=args.seed,
            combine_consecutive_messages=C, convert_tool_to_assistant=V)
        p.unlink(missing_ok=True)
        del train, members, probe
        torch.cuda.empty_cache()
        return float(df[df["dataset"] == args.split].iloc[0]["auroc"])

    allidx = list(range(n))
    baseline = fit_and_score(allidx, "all")
    print(f"baseline (all {n} base rows): {args.split} AUROC = {baseline:.4f}", flush=True)

    res = {"split": args.split, "config": tag, "n_base": n, "baseline_auroc": baseline,
           "ensemble_size": args.ensemble_size, "seed": args.seed, "rows": []}
    for i in allidx:
        auroc = fit_and_score([j for j in allidx if j != i], f"loo{i}")
        delta = auroc - baseline
        msgs = BASE.inputs[i]
        res["rows"].append(dict(
            i=i, id=BASE.ids[i], label=labels[i], auroc_without=auroc, delta=delta,
            n_messages=len(msgs),
            n_chars=sum(len(m.content or "") for m in msgs),
            user=next((m.content for m in msgs if m.role == "user"), None),
            assistant=next((m.content for m in msgs if m.role == "assistant"), None)))
        print(f"  [{i+1}/{n}] without row {i} ({labels[i][:24]}): {auroc:.4f} "
              f"delta {delta:+.4f}  ({'HURTS' if delta > 0 else 'helps'})", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))

    rows = sorted(res["rows"], key=lambda r: -r["delta"])
    print(f"\n=== most HARMFUL base rows (removing them helps '{args.split}') ===")
    for r in rows[:8]:
        print(f"  row {r['i']:>2} {r['label'][:32]:<34} delta {r['delta']:+.4f}  "
              f"{r['n_chars']:>5} chars")
    print(f"\n=== most HELPFUL base rows ===")
    for r in rows[-8:][::-1]:
        print(f"  row {r['i']:>2} {r['label'][:32]:<34} delta {r['delta']:+.4f}  "
              f"{r['n_chars']:>5} chars")
    pos_n = sum(1 for r in rows if r["delta"] > 0)
    print(f"\n{pos_n}/{n} rows hurt; delta range "
          f"{rows[-1]['delta']:+.4f} .. {rows[0]['delta']:+.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
