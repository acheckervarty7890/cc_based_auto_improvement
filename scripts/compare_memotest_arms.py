#!/usr/bin/env python
"""Compare the memotest arms: recency view (A), reshuffled view (B), no view (C),
plus C2 — arm C re-run under the revised prompts (no success target disclosed to the
attacker, 200-word round memo), which is read against C rather than against A/B.

Each arm differs from arm A in exactly one config line — ``attacker.view_reshuffle``
for B, ``attacker.view_limit`` for C — so any difference is attributable to what the
attacker was shown of its own past attempts. Reports, per round and per arm:

  * success rate and mean probe score — did the attacker still find weaknesses?
  * pairwise diversity of the first user turn (difflib ratio, 3-gram Jaccard) — the
    quantity that collapsed in the experiment7 runs (0.30 → 0.72 across rounds)
  * which rounds the injected view actually drew from, replayed through the real
    ``ViewSampler`` against each arm's own store

Run with no arguments after both arms have finished.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import re
import tempfile
from pathlib import Path

from agentic_redteam.persistence import AttemptRecord, JsonlStore
from agentic_redteam.view_sampler import ViewSampler

# (label, jsonl, reshuffle, view_limit) — the last two mirror each arm's config so the
# view replay below reconstructs what that arm's attacker was actually shown.
ARMS = [
    ("A recency  ", Path("results_memotest/memotest_probing.jsonl"), False, 4),
    ("B reshuffle", Path("results_memotest_reshuffle/memotest_probing.jsonl"), True, 4),
    ("C no-view   ", Path("results_memotest_noview/memotest_probing.jsonl"), False, 0),
    # C2 is not a fourth arm: same config as C, re-run after the attacker stopped being
    # told the round's success target and the round memo dropped 460 -> 200 words. Read
    # it against C, not against A/B. Skipped automatically until the run exists.
    ("C2 newprompt", Path("results_memotest_noview_c2/memotest_probing.jsonl"), False, 0),
    # C3 = C2 with max_turns 5->10 and sessions_per_model 5->2 (fewer, longer sessions).
    # Read it against C2. Note its rounds hold ~20 submission slots instead of 25, so
    # compare rates rather than raw counts.
    ("C3 2x10turns", Path("results_memotest_noview_c3/memotest_probing.jsonl"), False, 0),
    # C4 = C2 with sessions_per_model 5->10 (concurrency raised to match), max_turns
    # still 5. With C2 and C3 these three form a fan-out sweep at a fixed total-ish
    # budget: 2x10, 5x5, 10x5 sessions x turns.
    ("C4 10x5turns", Path("results_memotest_noview_c4/memotest_probing.jsonl"), False, 0),
    # C6 = C4 with attacker.batch_submissions on and sessions_per_model 4: each session is
    # ONE call emitting all 5 conversations, so nothing it writes is informed by a verdict.
    # Read it against C4 — this is the feedback ablation, not another fan-out point, and
    # its per-round submission count is capped at 4x5=20 by construction. (C5 is the
    # repetition-clustering arm on branch cluster-repetition-memo, not listed here.)
    ("C6 batch-4x5", Path("results_memotest_noview_c6/memotest_probing.jsonl"), False, 0),
    # C7 = C6 with max_turns 5->10 and sessions_per_model 4->2: same 20 slots per round,
    # redistributed into fewer/wider batches. Read it against C6, and read that pair
    # against C2 vs C3 — the identical redistribution in the per-turn regime, where it
    # cost diversity (0.282 -> 0.314 at round 0).
    ("C7 batch-2x10", Path("results_memotest_noview_c7/memotest_probing.jsonl"), False, 0),
    # C8 = batch mode at 7 sessions x 3. With C6 and C7 this is a walk in the number of
    # INDEPENDENT STARTS (2, 4, 7) at a near-fixed ~20 slots per round, which is the
    # variable that governed round-level diversity in the per-turn arms too.
    ("C8 batch-7x3", Path("results_memotest_noview_c8/memotest_probing.jsonl"), False, 0),
]
WORD = re.compile(r"\w+")


def _first_user(rec: AttemptRecord) -> str:
    return next((m.content for m in rec.sample.messages if m.role == "user"), "")


def _diversity(texts: list[str]) -> tuple[float, float]:
    """(mean pairwise difflib ratio, mean pairwise 3-gram Jaccard). Higher = more similar."""
    if len(texts) < 2:
        return float("nan"), float("nan")
    ratios = [
        difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
        for a, b in itertools.combinations(texts, 2)
    ]
    grams = [
        set(zip(*[w[i:] for i in range(3)])) for w in (WORD.findall(t.lower()) for t in texts)
    ]
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(grams, 2) if a | b]
    return sum(ratios) / len(ratios), (sum(jac) / len(jac) if jac else float("nan"))


def _view_round_mix(jsonl: Path, reshuffle: bool, limit: int, interval: int) -> str:
    """Replay the injected view over the whole run; report the round mix of what was shown."""
    lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
    tmp = Path(tempfile.mkdtemp()) / "s.jsonl"
    seen: dict[int, int] = {}
    total = 0
    for n in range(max(2, limit), len(lines) + 1):
        tmp.write_text("\n".join(lines[:n]) + "\n")
        vs = ViewSampler(
            store=JsonlStore(path=tmp), reshuffle=reshuffle, balance=True,
            blend_seeds=True, reshuffle_interval=interval,
        )
        cur = json.loads(lines[n - 1])["round"]
        for item in vs.sample(only_successful=False, limit=limit, current_round=cur,
                              persistence_from_last_rounds=None):
            seen[item["round"]] = seen.get(item["round"], 0) + 1
            total += 1
    if not total:
        return "(nothing shown)"
    return "  ".join(f"r{r}={100 * c / total:.0f}%" for r, c in sorted(seen.items()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reshuffle-interval", type=int, default=20, help="must match the config")
    args = p.parse_args(argv)

    loaded = []
    for label, path, reshuffle, vlimit in ARMS:
        if not path.exists():
            print(f"{label}: {path} not found — run that arm first.")
            continue
        recs = [AttemptRecord.from_jsonl_row(l) for l in path.read_text().splitlines() if l.strip()]
        loaded.append((label, path, reshuffle, recs, vlimit))
    if len(loaded) < 2:
        print("\nNeed both arms to compare.")
        return 1

    print("\n=== outcome by round ===")
    print(f"{'arm':13} {'rnd':>3} {'n':>4} {'succ':>5} {'rate':>7} {'mean score':>11}")
    for label, _, _, recs, _ in loaded:
        for rd in sorted({r.round for r in recs}):
            sub = [r for r in recs if r.round == rd]
            s = sum(1 for r in sub if r.success)
            ms = sum(r.probe_score for r in sub) / len(sub)
            print(f"{label:13} {rd:>3} {len(sub):>4} {s:>5} {100*s/len(sub):>6.1f}% {ms:>11.3f}")

    print("\n=== first-user-turn similarity by round (LOWER = more diverse) ===")
    print(f"{'arm':13} {'rnd':>3} {'n':>4} {'difflib':>9} {'jaccard':>9}")
    for label, _, _, recs, _ in loaded:
        for rd in sorted({r.round for r in recs}):
            sub = [r for r in recs if r.round == rd]
            d, j = _diversity([_first_user(r) for r in sub])
            print(f"{label:13} {rd:>3} {len(sub):>4} {d:>9.3f} {j:>9.3f}")
    print("\n  experiment7 guidance FP for reference: r0 0.299 / r1 0.497 / r2 0.537 "
          "→ r5 0.703, r7 0.724 (difflib)")

    print("\n=== which rounds the injected view actually drew from ===")
    for label, path, reshuffle, _, vlimit in loaded:
        if vlimit <= 0:
            print(f"{label:13} (no view — view_limit 0, nothing injected)")
            continue
        mix = _view_round_mix(path, reshuffle, vlimit, args.reshuffle_interval)
        print(f"{label:13} {mix}")

    print("\n=== memo word counts / truncation ===")
    for label, path, _, _, _ in loaded:
        side = path.with_suffix(".summaries.jsonl")
        if not side.exists():
            print(f"{label}: no summaries sidecar")
            continue
        for line in side.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            t = d["text"]
            ok = t.rstrip().endswith((".", "!", "?", '"', "”", ")"))
            print(f"{label:13} round {d['round']}: {d['n_successes']}/{d['n_attempts']} succ | "
                  f"{len(t.split()):>4} words | {'complete' if ok else 'TRUNCATED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
