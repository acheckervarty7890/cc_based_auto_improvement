#!/usr/bin/env python
"""Measure the retrain-to-retrain AUROC noise floor of the cross-iteration-memo ablation.

Why this exists. The ablation's arms differ by ~0.01-0.06 mean eval AUROC per iteration,
and the memo arm is higher in 5 of 6 differentiated cells — which looks like a small
consistent effect. But nothing in the pipeline holds the probe *training* fixed:
`ProbeFactory` trains a `PytorchAdamClassifier` (LinearThenSoftmax, Adam, batch_size 16,
up to 200 epochs, **early stopping with patience 50**), and `seed_everything(--seed)` is
called once at CLI start, so each retrain draws from a different point in the RNG stream.
The recorded `best_epoch` of the 12 probes this ablation produced ranges from 8 to 43,
which is a strong hint that two retrains of the *same* data would not agree.

So: rebuild ONE arm-iteration's exact training inputs and retrain it `--repeats` times,
re-seeding immediately before each build so the only thing that varies is the training
RNG. Everything else is byte-identical — same base data, same red-team JSONL, same
`min_judge_confidence`, same content-deterministic split (`seed=--split-seed`, held FIXED
so the train/val partition cannot move), same cached activations, same eval splits.

The spread of the resulting AUROCs is the yardstick every arm-vs-arm AUROC claim in this
experiment has to clear.

Cost: no OpenRouter calls (the arm's contrastive cache is already complete, and its
`--contrastive-cache` is passed read-through) and, on a warm activation cache, no LLM load
either — only the probe head is retrained. Minutes, not hours.

Usage:
    scripts/retrain_noise_itermemo.py                       # R1 nomemo iter1, 5 repeats
    scripts/retrain_noise_itermemo.py --arm memo --run 2 --iteration 2 --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", choices=["1", "2"], default="1")
    p.add_argument("--arm", choices=["nomemo", "memo"], default="nomemo")
    p.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Which retrain to reproduce: probe_iter{N}.pkl (default: %(default)s)",
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--resample-successes",
        type=int,
        default=0,
        help="DATA-sampling noise mode. Probe training here is deterministic in (seed, data) "
        "— retrain.py re-seeds immediately before ProbeFactory.build — so repeating with a "
        "different RNG measures nothing (spread 0.0000, verified). The variation that "
        "actually separates the arms is WHICH successes the attacker happened to find. With "
        "N > 0, each repeat draws a random N-subset of the run's own successes (the seeded "
        "pre-history is always kept in full, as it is in both arms), so the spread of the "
        "resulting AUROCs is the data-sampling noise floor. Set N to about what one "
        "iteration yields (~10).",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="The run's --seed, i.e. the content-deterministic train/val split. Held FIXED "
        "across repeats so only the training RNG varies (default: %(default)s)",
    )
    args = p.parse_args(argv)

    from agentic_redteam.config import load_config
    from agentic_redteam.evaluation import evaluate_probe, seed_everything
    from agentic_redteam.retrain import retrain_probe

    sfx = "" if args.run == "1" else f"_run{args.run}"
    cfg_path = REPO / f"configs/itermemo_hs_llama1b_{args.arm}{sfx}.md"
    results = REPO / f"results_itermemo_{args.arm}{sfx}"
    probes = REPO / f"probes/itermemo_{args.arm}{sfx}"
    it = args.iteration

    config = load_config(cfg_path)

    # Iteration N's retrain started from the probe iteration N-1 produced (the warm-start
    # probe for N=1) and consumed the JSONL as it stood then. The JSONL is append-only and
    # rows carry `iteration`, so the state at that moment is recoverable exactly: keep the
    # rows from iterations < it (plus the seeded pre-history at -1).
    src_jsonl = results / "gptoss120b_probing.jsonl"
    kept = []
    import json

    for line in src_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r.get("iteration", -1)) < it:
            kept.append(line)

    # Drop successes whose contrastive pair is not already in the arm's cache, so the
    # repeats make ZERO OpenRouter calls. In practice this is a single record per arm: one
    # conversation's generation failed during the original run too (which is why that run's
    # postprocessed set is an even 2x the kept successes minus that pair), and a failed
    # generation is never cached, so it is retried — and re-fails — on every retrain. The
    # measurement only needs the training set held CONSTANT across repeats, which it is.
    from agentic_redteam.persistence import AttemptRecord
    from agentic_redteam.preprocessing import _cache_key, _guidance_fingerprint, _load_cache
    from agentic_redteam.retrain import _success_human_label

    cache = _load_cache(probes / "contrastive_cache.jsonl")
    pp = config.preprocessing
    pos, neg = "high-stakes", "low-stakes"
    uncached = 0
    if pp is not None:
        surviving = []
        for line in kept:
            r = json.loads(line)
            if not (r.get("success") and r.get("judge_confidence", 0) >= config.judge.confidence_threshold):
                surviving.append(line)  # non-successes never reach contrastive generation
                continue
            rec = AttemptRecord.from_jsonl_row(line)
            # Mirror _successes_to_dicts exactly — the cache key is computed off the same
            # {inputs, labels} shape the retrain feeds to generate_contrastive_dataset.
            msgs = [{"role": m.role, "content": m.content} for m in rec.sample.messages]
            cur = _success_human_label(rec, pos, neg)
            tgt = pos if cur == neg else neg
            key = _cache_key(msgs, tgt, _guidance_fingerprint(pp.concept_description, tgt, pp.label_guidance))
            if key in cache:
                surviving.append(line)
            else:
                uncached += 1
        kept = surviving
    if uncached:
        print(f"  dropped {uncached} success(es) with no cached contrastive pair "
              f"(kept identical across repeats)")

    base_probe = (
        REPO / "probes/itermemo_start/probe_start.pkl"
        if it == 1
        else probes / f"probe_iter{it - 1}.pkl"
    )

    print(f"reproducing {args.arm}{sfx} probe_iter{it}.pkl")
    print(f"  base probe   {base_probe.relative_to(REPO)}")
    print(f"  red-team     {len(kept)} rows (iterations < {it})")
    print(f"  split seed   {args.split_seed} (fixed)   repeats {args.repeats}\n")

    aurocs: list[float] = []
    per_split: dict[str, list[float]] = {}

    # Split the reconstruction into the fixed pre-history and the run's own successes, so
    # --resample-successes can vary only the latter (mirroring what differs between arms).
    prehistory = [l for l in kept if json.loads(l).get("iteration", -1) < 0]
    own = [l for l in kept if json.loads(l).get("iteration", -1) >= 0]
    own_succ = [l for l in own if json.loads(l).get("success")]
    own_fail = [l for l in own if not json.loads(l).get("success")]
    if args.resample_successes:
        print(f"  resampling {args.resample_successes} of {len(own_succ)} own successes per repeat")

    with tempfile.TemporaryDirectory(dir=REPO / "results_itermemo_shared") as tmp:
        tmpd = Path(tmp)
        jsonl = tmpd / "redteam.jsonl"
        jsonl.write_text("\n".join(kept) + "\n", encoding="utf-8")

        for k in range(args.repeats):
            if args.resample_successes:
                import random

                draw = random.Random(2000 + k).sample(
                    own_succ, min(args.resample_successes, len(own_succ))
                )
                jsonl.write_text(
                    "\n".join(prehistory + own_fail + draw) + "\n", encoding="utf-8"
                )
            # Re-seed so each repeat starts from a DIFFERENT, reproducible RNG state. The
            # split seed passed to retrain_probe is unchanged, so the data partition is
            # identical and only init/shuffling/early-stopping move.
            seed_everything(1000 + k)
            out = tmpd / f"probe_repeat{k}.pkl"
            retrain_probe(
                jsonl_path=[jsonl],
                base_probe_path=base_probe,
                base_training_data_path=REPO / "data/hs_ls_200.jsonl",
                new_probe_path=out,
                preprocessing=config.preprocessing,
                contrastive_cache_path=probes / "contrastive_cache.jsonl",
                min_judge_confidence=config.judge.confidence_threshold,
                test_size=0.2,
                seed=args.split_seed,
                base_activation_cache_dir=REPO / "results_itermemo_shared/base_activations",
                combine_consecutive_messages=True,
                convert_tool_to_assistant=True,
                verbose=False,
            )
            df = evaluate_probe(
                probe_path=out,
                eval_dataset_dir=REPO / "eval_datasets",
                activations_cache_dir=REPO / "results_itermemo_shared/eval_activations",
                max_samples=None,
                seed=args.split_seed,
                combine_consecutive_messages=True,
                convert_tool_to_assistant=True,
            )
            row = {str(r["dataset"]): float(r["auroc"]) for _, r in df.iterrows()}
            row.pop("mean", None)
            m = statistics.mean(row.values())
            aurocs.append(m)
            for kk, v in row.items():
                per_split.setdefault(kk, []).append(v)
            ep = None
            try:
                from agentic_redteam.probe_judge import _cpu_unpickle

                ep = _cpu_unpickle(out.open("rb"))._classifier.best_epoch
            except Exception:
                pass
            print(f"  repeat {k}: mean auroc {m:.4f}   best_epoch {ep}   "
                  + "  ".join(f"{s}={v:.3f}" for s, v in sorted(row.items())), flush=True)

    print(f"\n{'=' * 74}")
    print(f"RETRAIN NOISE, identical inputs, {args.repeats} repeats")
    print(f"{'=' * 74}")
    print(f"  mean auroc : min {min(aurocs):.4f}  max {max(aurocs):.4f}  "
          f"spread {max(aurocs) - min(aurocs):.4f}")
    if len(aurocs) > 1:
        print(f"               mean {statistics.mean(aurocs):.4f}  sd {statistics.stdev(aurocs):.4f}")
    for s, vals in sorted(per_split.items()):
        print(f"  {s:<12} min {min(vals):.4f}  max {max(vals):.4f}  spread {max(vals) - min(vals):.4f}")
    print("\n  Any arm-vs-arm AUROC gap smaller than this spread is not evidence of an effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
