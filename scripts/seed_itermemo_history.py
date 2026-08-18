#!/usr/bin/env python
"""Build the shared starting state for the cross-iteration-memo ablation.

Two artifacts, both extracted from ``experiment6_cloud`` (the deepseek-v4-pro
guidance arm on the high-stakes / llama-1b probe) so the ablation starts from a
probe that has already been hardened and a JSONL that already holds successes:

1. ``probes/itermemo_start/probe_start.pkl`` — that run's ``probe_iter3.pkl``,
   i.e. the probe after 3 red-team + retrain cycles. Deliberately NOT named
   ``probe_iter{N}.pkl`` and deliberately outside either arm's ``--probe-out-dir``:
   ``cli._latest_probe_iteration`` scans that dir for exactly that pattern to
   decide where ``--resume`` restarts, and a stray checkpoint there would make
   iteration 0 look already-finished.

2. ``data/itermemo_seed_hs_fp.jsonl`` — the last ``--rounds`` rounds of that run's
   final iteration, **renumbered into a pre-history**: ``iteration = -1`` and
   ``round = -N .. -1``.

Why renumber rather than copy verbatim. The seeded rows have to reach the
near-duplicate guard, the retrain's training set and the novelty metric (that is
the point — the run opens against a store that already contains winning
conversations), but they must NOT be mistaken for attempts *this* run made:

- ``JsonlStore.records_for_round(0)`` feeds the rolling round memo. Verbatim rows
  carrying ``round: 0`` would put 50 attempts against a *different, much weaker*
  probe (exp6's iteration-0 probe) into this run's first memo.
- ``JsonlStore.records_for_iteration(0)`` feeds the cross-iteration memo — the very
  thing under test. Verbatim rows carrying ``iteration: 0`` would make the memo
  arm's first memo a write-up of exp6's findings rather than its own.

Negative rounds are already the established convention for "not from a real round"
here: ``ViewSampler`` stamps ``round=-1`` on training seeds, and its only
round-based filter (``persistence_from_last_rounds``) is off by default. (The
ablation itself runs at ``view_limit: 0``, so nothing is rendered to the attacker
either way — but the numbering keeps the seed usable by a config that does show a
view.)

Note what the seeding *does* still change, identically in both arms: the retrain
reads **every** successful row in the JSONL, so these seeded successes join the
training set at every retrain. That is intended (a mid-run state has accumulated
successes) and, being identical across arms, cannot confound the memo comparison.

Usage:
    scripts/seed_itermemo_history.py                 # build both artifacts
    scripts/seed_itermemo_history.py --rounds 2      # how many trailing rounds to seed
    scripts/seed_itermemo_history.py --force         # overwrite existing artifacts
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SRC_BRANCH = "experiment6_cloud"
SRC_PROBE = "probes/hs_llama1b_deepseekv4pro_guidance/probe_iter3.pkl"
SRC_JSONL = "results_hs_llama1b_deepseekv4pro_guidance/deepseekv4pro_probing_fp.jsonl"

OUT_PROBE = REPO / "probes/itermemo_start/probe_start.pkl"
OUT_JSONL = REPO / "data/itermemo_seed_hs_fp.jsonl"


def _git_show(ref: str, path: str) -> bytes:
    """Read one blob out of a branch without checking it out."""
    proc = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{ref}:{path}"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"git show {ref}:{path} failed:\n{proc.stderr.decode(errors='replace')}"
        )
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Trailing rounds of the source run's final iteration to seed (default: %(default)s, "
        "matching the ablation's rounds-per-iteration so the pre-history has the same shape)",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing artifacts")
    args = p.parse_args(argv)

    for out in (OUT_PROBE, OUT_JSONL):
        if out.exists() and not args.force:
            print(f"ERROR: {out.relative_to(REPO)} exists — pass --force to overwrite.", file=sys.stderr)
            return 1

    # --- 1. the starting probe ---------------------------------------------------------
    OUT_PROBE.parent.mkdir(parents=True, exist_ok=True)
    OUT_PROBE.write_bytes(_git_show(SRC_BRANCH, SRC_PROBE))
    print(f"probe : {SRC_BRANCH}:{SRC_PROBE}")
    print(f"     -> {OUT_PROBE.relative_to(REPO)} ({OUT_PROBE.stat().st_size} bytes)")

    # --- 2. the seeded attempt history -------------------------------------------------
    rows = [
        json.loads(line)
        for line in _git_show(SRC_BRANCH, SRC_JSONL).decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        print(f"ERROR: no rows in {SRC_JSONL}", file=sys.stderr)
        return 1

    last_iter = max(int(r.get("iteration", -1)) for r in rows)
    iter_rounds = sorted({int(r["round"]) for r in rows if int(r.get("iteration", -1)) == last_iter})
    keep_rounds = iter_rounds[-args.rounds :]
    # Newest source round becomes round -1, the one before it -2, and so on: the seeded
    # history reads as the rounds immediately preceding this run's round 0.
    remap = {src: -(len(keep_rounds) - i) for i, src in enumerate(keep_rounds)}

    seeded = []
    for r in rows:
        if int(r.get("iteration", -1)) != last_iter or int(r["round"]) not in remap:
            continue
        r = dict(r)
        r["round"] = remap[int(r["round"])]
        r["iteration"] = -1
        seeded.append(r)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in seeded:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_succ = sum(1 for r in seeded if r.get("success"))
    per_round = Counter(r["round"] for r in seeded)
    print(f"\nhistory: {SRC_BRANCH}:{SRC_JSONL}")
    print(f"         source iteration {last_iter}, rounds {keep_rounds} -> {[remap[k] for k in keep_rounds]}")
    print(f"      -> {OUT_JSONL.relative_to(REPO)}")
    print(f"         {len(seeded)} attempts, {n_succ} successes "
          f"({', '.join(f'round {k}: {v}' for k, v in sorted(per_round.items()))})")
    print("\nBoth arms start from copies of these two files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
