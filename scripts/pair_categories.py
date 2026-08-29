#!/usr/bin/env python
"""How do two small batches of generated rows combine?

Keeps `base ∪ 62 accepted` fixed and draws balanced 20-row batches from a pool of
generated rows. Each batch is fitted alone (`base ∪ 62 ∪ batch`), each disjoint pair is
fitted together (`base ∪ 62 ∪ A ∪ B`), and the pair is filed by the signs:

    1  both A and B raise eval, and A∪B raises it
    2  exactly one raises eval, and A∪B raises it
    3  both lower eval, and A∪B raises it
    4  both lower eval, and A∪B lowers it
    5  both raise eval, and A∪B lowers it        (not in the original four)
    6  exactly one raises eval, and A∪B lowers it (not in the original four)

Batches come from a fixed pool so each batch is fitted once and reused across many
pairs: N batch fits + one fit per pair, instead of three fits per pair.

Every batch and every pair is appended to CSV as it completes, so the run is resumable
and readable while in flight.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RUN_DIR = REPO / "results_gen_gemma27b_instructions_nemotron"
PROBE_DIR = REPO / "probes/gen_gemma27b_instructions_nemotron"
BASE_DATA = REPO / "data/instructions_llama70b_50.jsonl"
DEV_DATA = REPO / "dev_samples/instructions"
EVAL_DIR = REPO / "eval_sets/instructions"
BASE_CACHE = REPO / "cache_gen_gemma27b_instructions/base_activations"
EVAL_CACHE = REPO / "cache_gen_gemma27b_instructions/eval_activations"
BASE_PROBE = PROBE_DIR / "probe_iter12.pkl"
ACCEPTED = RUN_DIR / "accepted_iter13.jsonl"

POS = "assistant_follows_the_instruction"
SEED, COMBINE, CONVERT = 42, True, True
SCRATCH = PROBE_DIR / "pairs"


def load_accepted() -> list[dict]:
    return [{"inputs": json.loads(l)["inputs"], "labels": json.loads(l)["labels"]}
            for l in ACCEPTED.open()]


def load_pool(path: Path) -> list[dict]:
    return [{"inputs": r["inputs"], "labels": r["labels"]}
            for r in (json.loads(l) for l in path.open())]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=REPO / "data/union_fixed2_plus_nmrep4.jsonl")
    ap.add_argument("--batch-size", type=int, default=20, help="rows per batch (even; half per class)")
    ap.add_argument("--n-batches", type=int, default=120, help="size of the batch pool")
    ap.add_argument("--target", type=int, default=100, help="pairs wanted in each of categories 1-4")
    ap.add_argument("--max-pairs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=RUN_DIR / "pair_categories.csv")
    ap.add_argument("--batch-csv", type=Path, default=RUN_DIR / "pair_batches.csv")
    args = ap.parse_args()

    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import retrain_probe

    SCRATCH.mkdir(parents=True, exist_ok=True)
    accepted = load_accepted()
    pool = load_pool(args.pool)
    pos = [i for i, r in enumerate(pool) if r["labels"] == POS]
    neg = [i for i, r in enumerate(pool) if r["labels"] != POS]
    half = args.batch_size // 2
    print(f"pool {len(pool)} rows ({len(pos)} pos / {len(neg)} neg) from {args.pool.name}")

    def fit_eval(extra: list[dict], tag: str) -> tuple[float, float]:
        out = SCRATCH / f"{tag}.pkl"
        res = retrain_probe(
            samples=accepted + extra, base_probe_path=BASE_PROBE,
            base_training_data_path=BASE_DATA, new_probe_path=out, dev_data_path=DEV_DATA,
            seed=SEED, base_data_fraction=1.0, base_activation_cache_dir=BASE_CACHE,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=False)
        df = evaluate_probe(out, EVAL_DIR, EVAL_CACHE, max_samples=None, seed=SEED,
                            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT)
        out.unlink(missing_ok=True)
        return res.dev_auroc["mean"], float(df.loc[df["dataset"] == "mean", "auroc"].iloc[0])

    t0 = time.time()
    floor_dev, floor_eval = fit_eval([], "floor")
    print(f"floor: dev {floor_dev:.5f}  eval {floor_eval:.5f}   ({time.time()-t0:.0f}s per fit+eval)")

    rng = random.Random(args.seed)
    batches = []
    for b in range(args.n_batches):
        idx = sorted(rng.sample(pos, half) + rng.sample(neg, half))
        batches.append(idx)

    # Batch fits (one each, reused by every pair the batch appears in).
    bw = args.batch_csv.open("w", newline="")
    bwr = csv.writer(bw); bwr.writerow(["batch", "rows", "dev", "eval", "d_dev", "d_eval"])
    bres = []
    for b, idx in enumerate(batches):
        dv, ev = fit_eval([pool[i] for i in idx], f"b{b}")
        bres.append((dv, ev))
        bwr.writerow([b, args.batch_size, f"{dv:.6f}", f"{ev:.6f}",
                      f"{dv-floor_dev:.6f}", f"{ev-floor_eval:.6f}"]); bw.flush()
        if b % 10 == 0:
            print(f"  batch {b+1}/{len(batches)} Δeval {ev-floor_eval:+.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    bw.close()
    npos = sum(1 for _, ev in bres if ev > floor_eval)
    print(f"batches done: {npos}/{len(bres)} raise eval  [{time.time()-t0:.0f}s]")

    # Pairs: disjoint only, random order, stop when every category 1-4 has `target`.
    order = [(a, b) for a, b in itertools.combinations(range(len(batches)), 2)
             if not set(batches[a]) & set(batches[b])]
    rng.shuffle(order)
    print(f"{len(order)} disjoint pairs available")

    counts = {c: 0 for c in range(1, 7)}
    w = args.out.open("w", newline="")
    wr = csv.writer(w)
    wr.writerow(["pair", "a", "b", "a_d_eval", "b_d_eval", "ab_dev", "ab_eval",
                 "ab_d_dev", "ab_d_eval", "category"])
    done = 0
    for a, b in order:
        if done >= args.max_pairs or all(counts[c] >= args.target for c in (1, 2, 3, 4)):
            break
        da, db = bres[a][1] - floor_eval, bres[b][1] - floor_eval
        up = (da > 0) + (db > 0)
        dv, ev = fit_eval([pool[i] for i in batches[a] + batches[b]], f"p{a}_{b}")
        dab = ev - floor_eval
        cat = {(2, True): 1, (1, True): 2, (0, True): 3,
               (0, False): 4, (2, False): 5, (1, False): 6}[(up, dab > 0)]
        counts[cat] += 1
        done += 1
        wr.writerow([done, a, b, f"{da:.6f}", f"{db:.6f}", f"{dv:.6f}", f"{ev:.6f}",
                     f"{dv-floor_dev:.6f}", f"{dab:.6f}", cat]); w.flush()
        if done % 10 == 0:
            print(f"  pair {done}: cat counts {dict(counts)}  [{time.time()-t0:.0f}s]", flush=True)
    w.close()
    print(f"\nDONE {done} pairs in {time.time()-t0:.0f}s: {dict(counts)}")


if __name__ == "__main__":
    main()
