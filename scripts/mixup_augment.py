#!/usr/bin/env python
"""Activation-space mixup: 100 synthetic rows averaged from same-label pairs.

Draws random same-label pairs from `62 accepted ∪ 107 generated`, averages their cached
layer-32 activations token-by-token at λ = 0.5, and adds the result as extra training
rows. Balanced 50 / 50 across the two classes.

**Alignment.** Each cached row is stored at its own true length with an all-ones mask, and
two conversations rarely share a length, so a pair is averaged over the first
`min(Ta, Tb)` positions and the synthetic row carries that length. Every position is then
a genuine average of two real token activations, and the mask stays all-ones — no
zero-padding is fabricated into the input. The cost is that the longer member's tail is
discarded; the script reports how much.

**How the rows reach the fit.** `retrain_probe` has no entry point for raw tensors, so each
synthetic row is given a placeholder two-turn conversation and its blob is written at that
conversation's own per-sample cache path. The fit then loads it as an ordinary cache hit
and the code path is identical to every other fit in this investigation. The placeholder
text is never tokenized — the script asserts every synthetic path exists before fitting, so
a cache miss (which would silently activate the placeholder text instead) fails loudly.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RUN_DIR = REPO / "results_gen_gemma27b_instructions_nemotron"
PROBE_DIR = REPO / "probes/gen_gemma27b_instructions_nemotron"
BASE_DATA = REPO / "data/instructions_llama70b_50.jsonl"
DEV_DATA = REPO / "dev_samples/instructions"
EVAL_DIR = REPO / "eval_sets/instructions"
BASE_CACHE = REPO / "cache_gen_gemma27b_instructions/base_activations"
EVAL_CACHE = REPO / "cache_gen_gemma27b_instructions/eval_activations"

BASE_PROBE = PROBE_DIR / "probe_iter12.pkl"
ACCEPTED = RUN_DIR / "accepted_iter13.jsonl"
GENERATED = REPO / "data/instructions_like_accepted62.jsonl"

REF = {"base+62": 0.8148, "base+62+107": 0.85043}
SEED = 42
COMBINE = True
CONVERT = True


def jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            inputs = r["inputs"]
            if isinstance(inputs, str):
                inputs = json.loads(inputs)
            rows.append({"inputs": inputs, "labels": r["labels"]})
    return rows


def placeholder(i: int) -> list[dict]:
    """A two-turn conversation whose only job is to key a cache entry.

    Distinct roles, so `combine_consecutive_messages` cannot merge them and the
    post-transform messages the fit hashes are the ones hashed here.
    """
    return [
        {"role": "user", "content": f"[mixup-{i:04d}] synthetic activation row"},
        {"role": "assistant", "content": f"[mixup-{i:04d}]"},
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="synthetic rows (split evenly across classes)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="mixup100")
    args = ap.parse_args()

    import torch

    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import (
        _apply_message_transforms,
        _dicts_to_labelled_dataset,
        _sample_activation_cache_path,
        read_probe_metadata,
        retrain_probe,
    )

    meta = read_probe_metadata(BASE_PROBE)
    pos, neg = meta["pos_class_label"], meta["neg_class_label"]
    model_name, layer = meta["model_name"], meta["layer"]

    real = jsonl(ACCEPTED) + jsonl(GENERATED)
    print(f"source pool: {len(real)} rows (62 accepted + 107 generated)")

    # Cache path per source row, via the post-transform messages the fit uses.
    ds = _apply_message_transforms(_dicts_to_labelled_dataset(real, pos, neg), COMBINE, CONVERT)
    paths = [
        _sample_activation_cache_path(BASE_CACHE, d, model_name, layer, COMBINE, CONVERT)
        for d in ds.inputs
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} source rows are not cached; run the fit once first")

    by_label: dict[str, list[Path]] = {pos: [], neg: []}
    for row, p in zip(real, paths):
        by_label[row["labels"]].append(p)
    print(f"  {pos}: {len(by_label[pos])}")
    print(f"  {neg}: {len(by_label[neg])}")

    rng = random.Random(args.seed)
    per_class = args.n // 2
    synth_rows, trunc_frac, lengths = [], [], []
    idx = 0
    for label in (pos, neg):
        pool = by_label[label]
        for _ in range(per_class):
            pa, pb = rng.sample(pool, 2)
            a = torch.load(pa, map_location="cpu", weights_only=False)
            b = torch.load(pb, map_location="cpu", weights_only=False)
            ta, tb = a["activations"].shape[1], b["activations"].shape[1]
            t = min(ta, tb)
            mixed = (
                a["activations"][:, :t].to(torch.float32)
                + b["activations"][:, :t].to(torch.float32)
            ) / 2.0
            blob = {
                "activations": mixed.to(a["activations"].dtype),
                "attention_mask": torch.ones(1, t, dtype=a["attention_mask"].dtype),
                "input_ids": a["input_ids"][:, :t].clone(),
                "layer": layer,
                "model_name": model_name,
            }
            msgs = placeholder(idx)
            out = _sample_activation_cache_path(
                BASE_CACHE,
                _apply_message_transforms(
                    _dicts_to_labelled_dataset([{"inputs": msgs, "labels": label}], pos, neg),
                    COMBINE, CONVERT,
                ).inputs[0],
                model_name, layer, COMBINE, CONVERT,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(blob, out)
            if not out.exists():
                raise SystemExit(f"failed to write {out}")
            synth_rows.append({"inputs": msgs, "labels": label})
            trunc_frac.append(1 - t / max(ta, tb))
            lengths.append(t)
            idx += 1
            del a, b, mixed

    print(f"\nbuilt {len(synth_rows)} synthetic rows ({per_class} per class)")
    print(f"  mixed length: median {int(statistics.median(lengths))} tokens "
          f"(min {min(lengths)}, max {max(lengths)})")
    print(f"  tail discarded from the longer member: median "
          f"{statistics.median(trunc_frac) * 100:.0f}%, mean {statistics.mean(trunc_frac) * 100:.0f}%")

    samples = real + synth_rows
    out_probe = PROBE_DIR / f"gen_{args.tag}.pkl"
    res = retrain_probe(
        samples=samples,
        base_probe_path=BASE_PROBE,
        base_training_data_path=BASE_DATA,
        new_probe_path=out_probe,
        dev_data_path=DEV_DATA,
        seed=SEED,
        base_data_fraction=1.0,
        base_activation_cache_dir=BASE_CACHE,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
        verbose=False,
    )
    print(f"\ntraining rows: {res.n_training_samples_total}")
    print(f"dev  mean: {res.dev_auroc['mean']:.5f}")

    df = evaluate_probe(
        out_probe, EVAL_DIR, EVAL_CACHE, max_samples=None, seed=SEED,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
    )
    print(df.to_string(index=False))
    csv = RUN_DIR / f"gen_{args.tag}_eval.csv"
    df.to_csv(csv, index=False)
    mean = float(df.loc[df["dataset"] == "mean", "auroc"].iloc[0])
    print(f"\neval mean: {mean:.5f}")
    for k, v in REF.items():
        print(f"  vs {k:<12} ({v:.4f}): {mean - v:+.5f}")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
