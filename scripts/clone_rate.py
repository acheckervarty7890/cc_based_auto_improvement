#!/usr/bin/env python
"""Measure the "cloniness" of red-team successes in one or more run JSONLs.

Standalone — stdlib only (no venv / tuberlens needed). Point it at the arm
JSONLs produced by the clone-test configs and it reports, per arm:

  * successes / attempts (yield)
  * distinct successes (after EXACT dedup — the store already dedups by
    canonical text, this is a belt-and-suspenders pass)
  * clone rate  = fraction of distinct successes that sit in a near-duplicate
                  cluster (cluster size >= --min-cluster) at each threshold tau
  * effective diversity = #clusters / #distinct-successes  (1.0 = all unique)
  * largest cluster size
  * distinct-per-100-attempts  = the real "signal per unit compute" number

Similarity is difflib.SequenceMatcher ratio on the first user message
(truncated) — cheap and where the templating shows. Clustering is greedy
single-linkage above the threshold.

With >1 input, arms are ALSO compared at EQUAL N (every arm truncated to the
first `min(distinct successes)` successes), so a larger arm doesn't look clonier
just from volume.

Usage:
  python scripts/clone_rate.py \
      results_clone_test/arm0_baseline.jsonl \
      results_clone_test/arm1_reshuffle.jsonl \
      --tau 0.7 0.8 0.9 --min-cluster 2 --prefix 600
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path


def first_user(sample) -> str:
    msgs = sample if isinstance(sample, list) else sample.get("messages", sample)
    if not isinstance(msgs, list):
        return ""
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            return (m.get("content") or "").strip()
    return (msgs[0].get("content", "") if msgs and isinstance(msgs[0], dict) else "").strip()


def load_arm(path: Path):
    """Return (attempts, success_texts_in_order) — success_texts are exact-deduped."""
    attempts = 0
    successes = []
    seen = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            attempts += 1
            if not d.get("success"):
                continue
            t = first_user(d.get("sample"))
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            successes.append(t)
    return attempts, successes


def cluster(texts, tau: float, prefix: int):
    """Greedy single-linkage clustering by difflib ratio >= tau. Returns list of
    member-index lists."""
    trunc = [t[:prefix] for t in texts]
    used = [False] * len(texts)
    clusters = []
    for i in range(len(texts)):
        if used[i]:
            continue
        members = [i]
        used[i] = True
        for j in range(i + 1, len(texts)):
            if used[j]:
                continue
            if difflib.SequenceMatcher(None, trunc[i], trunc[j]).ratio() >= tau:
                members.append(j)
                used[j] = True
        clusters.append(members)
    return clusters


def metrics(texts, tau: float, prefix: int, min_cluster: int):
    n = len(texts)
    if n == 0:
        return dict(n=0, clone_rate=0.0, eff_div=0.0, n_clusters=0, largest=0, in_clones=0)
    clusters = cluster(texts, tau, prefix)
    in_clones = sum(len(c) for c in clusters if len(c) >= min_cluster)
    largest = max((len(c) for c in clusters), default=0)
    return dict(
        n=n,
        clone_rate=in_clones / n,
        eff_div=len(clusters) / n,
        n_clusters=len(clusters),
        largest=largest,
        in_clones=in_clones,
    )


def fmt_arm(name, attempts, successes, taus, prefix, min_cluster, cap=None):
    texts = successes[:cap] if cap is not None else successes
    print(f"\n{'=' * 68}\n{name}")
    print(f"  attempts={attempts}  successes(raw)={len(successes)}"
          + (f"  [capped to {len(texts)} for equal-N compare]" if cap is not None else ""))
    if attempts:
        print(f"  yield = {len(successes) / attempts:.1%} successes/attempt")
    # dist/100att = near-deduped distinct successes (=#clusters) per 100 attempts.
    # This is the headline decision metric: it rewards "fewer clones" AND
    # penalizes "found fewer real successes". Depends on tau, so it's per-row.
    print(f"  {'tau':>5} {'clone_rate':>11} {'eff_div':>8} {'#clusters':>10} {'largest':>8} {'dist/100att':>11}")
    for tau in taus:
        m = metrics(texts, tau, prefix, min_cluster)
        dpa = (m["n_clusters"] / attempts * 100) if attempts else 0.0
        print(f"  {tau:>5.2f} {m['clone_rate']:>10.1%} {m['eff_div']:>8.2f}"
              f" {m['n_clusters']:>10} {m['largest']:>8} {dpa:>11.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+", type=Path, help="one or more arm JSONL files")
    ap.add_argument("--tau", nargs="+", type=float, default=[0.7, 0.8, 0.9],
                    help="near-dup similarity thresholds (default 0.7 0.8 0.9)")
    ap.add_argument("--min-cluster", type=int, default=2,
                    help="min cluster size to count as clones (default 2)")
    ap.add_argument("--prefix", type=int, default=600,
                    help="chars of first-user message compared (default 600)")
    ap.add_argument("--no-equal-n", action="store_true",
                    help="skip the equal-N (truncated) comparison across arms")
    args = ap.parse_args(argv)

    arms = []
    for p in args.jsonl:
        if not p.exists():
            print(f"!! missing: {p}", file=sys.stderr)
            continue
        attempts, successes = load_arm(p)
        arms.append((p.name, attempts, successes))

    if not arms:
        print("no readable inputs", file=sys.stderr)
        return 1

    print("### FULL SETS (each arm at its own N) ###")
    for name, attempts, successes in arms:
        fmt_arm(name, attempts, successes, args.tau, args.prefix, args.min_cluster)

    if len(arms) > 1 and not args.no_equal_n:
        cap = min(len(s) for _, _, s in arms)
        print(f"\n\n### EQUAL-N COMPARISON (every arm truncated to first {cap} successes) ###")
        for name, attempts, successes in arms:
            fmt_arm(name, attempts, successes, args.tau, args.prefix, args.min_cluster, cap=cap)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
