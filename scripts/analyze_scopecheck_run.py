#!/usr/bin/env python
"""Summarize a scope-check run: rejections per round, their tags, and what the memos said.

    python scripts/analyze_scopecheck_run.py [results_dir]

Reads the run's JSONL plus its `.summaries.jsonl` sidecar and prints, per round, how many
attempts were scored, how many were successes and how many the judge REJECTED as outside the
eval data's constraints (`violated_constraint`) — counted apart from failures, since a
rejected sample was never evidence about the classifier. Then dumps a few rejected and a few
accepted conversations, and every rolling memo with a note of whether it mentions the
rejections at all. No network, no GPU: it only reads the run's own files.
"""
import json, sys
from collections import Counter
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results_instructions_llama1b_scopecheck_test")
# The attempts log, not one of its sidecars: with a single error type the path carries no
# _fp/_fn suffix, and every sidecar shares the same stem.
SIDECARS = (".summaries.", ".runlog.", ".rounds_done.", ".iteration_memos.", ".prompts.")
cands = [f for f in sorted(D.glob("*.jsonl")) if not any(s in f.name for s in SIDECARS)]
if not cands:
    print(f"no attempts JSONL under {D}"); sys.exit(0)
jsonl = cands[0]

rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
print(f"{jsonl.name}: {len(rows)} attempts recorded\n")

print(f"{'round':>5} {'attempts':>9} {'success':>8} {'rejected':>9}  tags")
for rnd in sorted({r["round"] for r in rows}):
    sub = [r for r in rows if r["round"] == rnd]
    rej = [r for r in sub if r.get("violated_constraint")]
    tags = Counter(r["violated_constraint"] for r in rej)
    print(f"{rnd:>5} {len(sub):>9} {sum(1 for r in sub if r['success']):>8} {len(rej):>9}  "
          + ", ".join(f"{t}×{n}" for t, n in tags.most_common()))
allrej = [r for r in rows if r.get("violated_constraint")]
print(f"\ntotal: {len(rows)} attempts | {sum(1 for r in rows if r['success'])} successes | "
      f"{len(allrej)} rejected ({100*len(allrej)/max(1,len(rows)):.0f}%)")
print("tags overall:", Counter(r["violated_constraint"] for r in allrej).most_common())

print("\n--- a few rejected samples ---")
for r in allrej[:5]:
    print("=" * 90)
    print(f"round {r['round']} | probe {r['probe_score']:.3f} | judge {r['judge_label'].split('_')[-3:]} "
          f"| tag: {r['violated_constraint']}")
    print("  scope_reason:", r.get("scope_reason", "")[:200])
    for m in r["sample"]["messages"]:
        print(f"  [{m['role']}] {' '.join(m['content'].split())[:220]}")

print("\n--- a few ACCEPTED samples (in scope) ---")
for r in [x for x in rows if not x.get("violated_constraint")][:3]:
    print("=" * 90)
    print(f"round {r['round']} | probe {r['probe_score']:.3f} | success={r['success']} | judge={r['judge_label']}")
    for m in r["sample"]["messages"]:
        print(f"  [{m['role']}] {' '.join(m['content'].split())[:220]}")

summ = D / (jsonl.name + ".summaries.jsonl")
if summ.exists():
    print("\n--- rolling memos (did they pick the rejections up?) ---")
    for line in summ.read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        text = s["text"]
        hit = [w for w in ("reject", "constraint", "scope", "violated") if w in text.lower()]
        print("=" * 90)
        print(f"after round {s['round']} | {s['n_attempts']} attempts, {s['n_successes']} successes "
              f"| mentions: {hit or 'NONE'}")
        print(text[:2000])
runlog = D / (jsonl.name + ".runlog.jsonl")
if runlog.exists():
    ev = Counter(json.loads(l)["event"] for l in runlog.read_text().splitlines() if l.strip())
    print("\nrunlog events:", ev.most_common())
