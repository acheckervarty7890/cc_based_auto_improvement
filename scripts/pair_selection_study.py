#!/usr/bin/env python
"""Leave-one-out study over a retrain's contrastive pairs, scored on one eval split.

A retrain trains on base data plus N red-team (find, generated-pair) couples. This asks
which of those couples the probe is actually better *for*: it fits the probe once on all
N, then once per couple with that couple held out, and reports the resulting AUROC on a
chosen eval split.

    delta_i = AUROC(without couple i) - AUROC(with all N)

so **delta > 0 means couple i was HURTING the split** (removing it helped) and delta < 0
means it was helping. The sign is deliberately this way round: the question being asked is
"what is this training sample doing to me", not "what does removing it do".

Both halves of a couple are held out together. They are one training decision — the
generator exists to mint an opposite-class partner for a find — so dropping only one side
would change the class balance as well as the content and confound the two.

Everything is fit through `retrain._train_with_cached_base_activations`, the same function
the real pipeline uses, so the probes are comparable to the run's own. Every activation is
read from the caches that run already populated (base blob, per-conversation red-team, dev
blob, eval blobs), so no extraction LLM is ever loaded: N+1 fits cost N+1 probe-head fits
and nothing else.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _load_pairs(path: Path):
    """The postprocessed dump, folded back into (find, generated-pair) couples.

    `_dump_labelled_dataset` writes the finds first and their generated partners second,
    in the same order, so couple i is (row i, row i + N). Verified against the user turn,
    which the generator is instructed to leave alone.
    """
    rows = [json.loads(l) for l in path.open()]
    n = len(rows) // 2
    finds, gens = rows[:n], rows[n:]
    assert all(r["label"] == "negative" for r in finds), "expected finds to be negative"
    assert all(r["label"] == "positive" for r in gens), "expected pairs to be positive"
    return list(zip(finds, gens))


def _dataset_from_rows(rows):
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

    return LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rows],
        ids=[r["id"] for r in rows],
        other_fields={"labels": [r["label"] for r in rows]},
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path,
                    default=REPO / "probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5/redteam_postprocessed_iter5.jsonl")
    ap.add_argument("--base-probe", type=Path,
                    default=REPO / "probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl")
    ap.add_argument("--base-training-data", type=Path, default=REPO / "data/instructions_llama70b_50.jsonl")
    ap.add_argument("--dev-data", type=Path, default=REPO / "dev_samples/instructions")
    ap.add_argument("--eval-dataset-dir", type=Path, default=REPO / "eval_sets/instructions")
    ap.add_argument("--split", default="oig_omission", help="eval split to score")
    ap.add_argument("--base-activation-cache-dir", type=Path,
                    default=REPO / "results_instructions_gemma27b_shared/base_activations")
    ap.add_argument("--activations-cache-dir", type=Path,
                    default=REPO / "results_instructions_gemma27b_shared/eval_activations")
    ap.add_argument("--ensemble-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO / "results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5/pair_selection_oig_omission.json")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/pair_study"))
    args = ap.parse_args(argv)

    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import (
        _base_activation_cache_paths,
        _cpu_unpickle,
        _dev_activation_cache_path,
        _infer_probe_spec,
        _load_dev_dataset,
        _resolve_ensemble_seeds,
        _train_with_cached_base_activations,
        stable_train_test_split,
    )
    from tuberlens.interfaces.dataset import LabelledDataset

    args.workdir.mkdir(parents=True, exist_ok=True)
    with args.base_probe.open("rb") as f:
        base_probe = _cpu_unpickle(f)
    model_name, layer = base_probe.model_name, base_probe.layer
    pos, neg = base_probe.pos_class_label, base_probe.neg_class_label
    probe_spec = _infer_probe_spec(base_probe)
    seeds = _resolve_ensemble_seeds(args.seed, args.ensemble_size)

    # --dev-data => the base data trains in FULL and the dev set is the whole validation
    # set, exactly as the run did. test_size 0.0 puts every base sample on the train side.
    TEST_SIZE = 0.0
    COMBINE = CONVERT = True

    # A Path routes to the jsonl loader; a str is read as a HuggingFace dataset name.
    base_dataset = LabelledDataset.load_from(
        Path(args.base_training_data), pos_class_label=pos, neg_class_label=neg,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT)
    base_train, base_val = stable_train_test_split(
        base_dataset, test_size=TEST_SIZE, split_field=None, seed=args.seed)
    base_train_cache, base_val_cache = _base_activation_cache_paths(
        args.base_activation_cache_dir, args.base_training_data, model_name, layer,
        args.seed, TEST_SIZE, None, COMBINE, CONVERT, 1.0)

    # Returns (dataset, files); the files are what the dev activation cache is keyed on.
    dev_val, dev_files = _load_dev_dataset(args.dev_data, pos, neg, COMBINE, CONVERT, verbose=False)
    dev_cache = _dev_activation_cache_path(
        args.base_activation_cache_dir, dev_files, model_name, layer, COMBINE, CONVERT)

    pairs = _load_pairs(args.pairs)
    print(f"{len(pairs)} couples ({2*len(pairs)} training rows); scoring split '{args.split}'", flush=True)

    def fit_and_score(keep: list[int], tag: str) -> float:
        rows = [r for i in keep for r in pairs[i]]
        rt = _dataset_from_rows(rows)
        rt_train, rt_val = stable_train_test_split(rt, test_size=TEST_SIZE, split_field=None, seed=args.seed)
        probe = _train_with_cached_base_activations(
            base_train=base_train, base_val=base_val,
            redteam_train=rt_train, redteam_val=rt_val, dev_val=dev_val,
            model_name=model_name, layer=layer, probe_spec=probe_spec,
            pos_class_label=pos, neg_class_label=neg,
            probe_description=base_probe.description,
            base_train_cache=base_train_cache, base_val_cache=base_val_cache,
            dev_val_cache=dev_cache,
            redteam_cache_dir=args.base_activation_cache_dir,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            seed=args.seed, ensemble_seeds=seeds, verbose=False)
        p = args.workdir / f"probe_{tag}.pkl"
        with p.open("wb") as f:
            pickle.dump(probe, f)
        df = evaluate_probe(
            probe_path=p, eval_dataset_dir=args.eval_dataset_dir,
            activations_cache_dir=args.activations_cache_dir, splits=[args.split],
            max_samples=None, seed=args.seed,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT)
        p.unlink(missing_ok=True)
        row = df[df["dataset"] == args.split].iloc[0]
        return float(row["auroc"])

    allidx = list(range(len(pairs)))
    base_auroc = fit_and_score(allidx, "all")
    print(f"baseline (all {len(pairs)} couples): {args.split} AUROC = {base_auroc:.4f}", flush=True)

    out = {"split": args.split, "n_pairs": len(pairs), "baseline_auroc": base_auroc, "pairs": []}
    for i in allidx:
        keep = [j for j in allidx if j != i]
        a = fit_and_score(keep, f"loo{i}")
        delta = a - base_auroc          # > 0 => removing it HELPED => the couple hurts
        find, gen = pairs[i]
        out["pairs"].append(dict(
            i=i, sid=find["id"], gid=gen["id"], auroc_without=a, delta=delta,
            user=find["inputs"][0]["content"],
            find_reply=next((m["content"] for m in find["inputs"] if m["role"] == "assistant"), None),
            gen_reply=next((m["content"] for m in gen["inputs"] if m["role"] == "assistant"), None)))
        print(f"  [{i+1}/{len(pairs)}] without couple {i}: {a:.4f}  delta {delta:+.4f}"
              f"  ({'HURTS' if delta > 0 else 'helps'})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
