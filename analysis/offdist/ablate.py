#!/usr/bin/env python
"""Phase 2: remove a group of red-team rows, refit, and score the eval splits.

The fit is the ceiling analysis's fit, unchanged — a single `linear_then_softmax` head at
seed 42, early-stopped on that study's reserved 25% dev validation slice — so `full` here
reproduces its N=0 point exactly and every condition is comparable to its curves. It is
**not** comparable to the run's published CSVs, which are 10-member ensembles.

**Removals are by contrastive pair, not by row.** The red-team set is exactly class
balanced because every conversation appears with a generated opposite-label partner.
Dropping single rows would move the class balance along with the flagged property, and the
refit would answer a different question. A pair is dropped when either half is flagged.

**Every condition is matched against random removal of the same size**, three seeds each.
Without that, "removing 30% of the training data changed the score" is not evidence about
*which* 30%. Sizes are deduplicated, so a topic removal and a quantile removal of the same
size share one control.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import od_common as O  # noqa: E402

sys.path.insert(0, str(O.REPO / "ceiling_analysis" / "scripts"))
import ca_common as C  # noqa: E402

KEY_FIELDS = ("arm", "condition", "seed")
N_BASE = 50  # base rows occupy pool indices 0..49 and are never removed


def load_flags(arm: O.Arm) -> list[dict]:
    path = O.RESULTS / f"flags_{arm.key}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} — run characterize.py first")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def pair_groups(flags: list[dict]) -> list[list[int]]:
    """Partition the red-team rows into removal units: a contrastive pair, or a lone row."""
    seen, groups = set(), []
    by_i = {f["i"]: f for f in flags}
    for f in flags:
        if f["i"] in seen:
            continue
        j = f.get("pair_with")
        if j is not None and j in by_i and j not in seen:
            groups.append([f["i"], j])
            seen.update((f["i"], j))
        else:
            groups.append([f["i"]])
            seen.add(f["i"])
    return groups


def group_score(groups, flags, field: str) -> np.ndarray:
    v = {f["i"]: f[field] for f in flags}
    return np.array([float(np.mean([v[i] for i in g])) for g in groups])


def conditions(flags, groups, args) -> dict[str, list[int]]:
    """condition name -> the red-team row indices to REMOVE."""
    out: dict[str, list[int]] = {"full": []}
    n_groups = len(groups)
    p = group_score(groups, flags, "p_redteam")
    order_hi = np.argsort(-p)          # most off-distribution first
    order_lo = np.argsort(p)           # most eval-like first

    for q in args.quantiles:
        k = max(1, int(round(n_groups * q / 100)))
        out[f"drop_most_offdist_{q}pct"] = [i for gi in order_hi[:k] for i in groups[gi]]
        out[f"drop_most_evallike_{q}pct"] = [i for gi in order_lo[:k] for i in groups[gi]]

    # The single strongest structural axis: assistant reply length (8.5x the eval mean).
    alen = group_score(
        groups, [{**f, "alen": f["structural"]["chars_assistant"]} for f in flags], "alen"
    )
    k = max(1, int(round(n_groups * 0.30)))
    out["drop_longest_assistant_30pct"] = [i for gi in np.argsort(-alen)[:k] for i in groups[gi]]

    inv = {f["i"] for f in flags if f["convention_inverted"]}
    if inv:
        out["drop_convention_inverted"] = [i for g in groups if inv & set(g) for i in g]

    for c in sorted({f["topic"] for f in flags}):
        members = {f["i"] for f in flags if f["topic"] == c}
        rows = [i for g in groups if members & set(g) for i in g]
        if 0 < len(rows) < len(flags):
            out[f"drop_topic_{c}"] = rows

    out["base_only"] = [f["i"] for f in flags]
    return out


def matched_random(groups, n_rows_removed: int, seed: int) -> list[int]:
    """Remove whole groups at random until at least as many rows are gone."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    picked, n = [], 0
    for gi in order:
        if n >= n_rows_removed:
            break
        picked.extend(groups[gi])
        n += len(groups[gi])
    return picked


def evaluate(probe, eval_srcs) -> dict:
    from tuberlens.evaluation import calculate_metrics

    per_split = {}
    for name, src in eval_srcs.items():
        y = C.source_labels(src)
        preds = C.score_source(probe, src, np.arange(len(src)), chunk=64)
        per_split[name] = calculate_metrics(y, preds, fpr=0.01)
    C.free_gpu()
    mean = {m: float(np.mean([v[m] for v in per_split.values()]))
            for m in ("auroc", "accuracy", "tpr_at_fpr")}
    return {"per_split": per_split, "mean": mean}


def run_arm(arm: O.Arm, args) -> None:
    concept = C.CONCEPTS[arm.concept]
    flags = load_flags(arm)
    groups = pair_groups(flags)
    conds = conditions(flags, groups, args)

    log_path = O.RESULTS / f"ablation_{arm.key}.jsonl"
    done = C.done_keys(log_path, KEY_FIELDS) if args.resume else set()

    eval_srcs = C.eval_sources(concept)
    dev_src, val_idx, _pool = C.dev_partition(concept)
    val_d = C.ragged_from_parts([(dev_src, val_idx)])
    rt_src = C.redteam_source(concept)
    n_rt = len(rt_src) - N_BASE
    assert n_rt == len(flags), f"{n_rt} red-team rows in the pool, {len(flags)} flagged"

    # one matched-random control per distinct removal size
    sizes = sorted({len(v) for k, v in conds.items() if v and k != "base_only"})
    jobs: list[tuple[str, int, list[int]]] = [(k, 0, v) for k, v in conds.items()]
    for n in sizes:
        for s in range(args.random_seeds):
            jobs.append((f"random_{n}", s, matched_random(groups, n, 1000 + s)))

    print(f"[{arm.key}] {len(flags)} red-team rows in {len(groups)} removal units; "
          f"{len(jobs)} fits", flush=True)

    for name, seed, drop in jobs:
        if (arm.key, name, seed) in done:
            continue
        keep_rt = sorted(set(range(len(flags))) - set(drop))
        keep = list(range(N_BASE)) + [N_BASE + i for i in keep_rt]
        t0 = time.time()
        train = C.ragged_from_parts([(rt_src, keep)])
        probe = C.fit(train, val_d, concept, seed=C.FIT_SEED)
        del train
        C.free_gpu()
        m = evaluate(probe, eval_srcs)
        row = {
            "arm": arm.key, "condition": name, "seed": seed,
            "n_removed": len(drop), "n_train": len(keep),
            "frac_removed": len(drop) / max(len(flags), 1),
            "val_auroc": C.ragged_val_auroc(probe, val_d),
            "seconds": round(time.time() - t0, 1),
            "mean": m["mean"], "per_split": m["per_split"],
        }
        C.append_jsonl(log_path, row)
        print(f"[{arm.key}] {name:32s} seed={seed} removed={len(drop):>4} "
              f"eval AUROC {m['mean']['auroc']:.4f} ({row['seconds']}s)", flush=True)
        del probe
        C.free_gpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(O.ARMS))
    ap.add_argument("--quantiles", nargs="*", type=int, default=[10, 20, 30, 50])
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    for key in args.arms:
        run_arm(O.ARMS[key], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
