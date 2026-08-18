#!/usr/bin/env python
"""Compare the two arms of the cross-iteration-memo ablation.

Standalone — stdlib only (no venv / tuberlens needed).

    ARM A  results_itermemo_nomemo/   cross_iteration_memos: false
    ARM B  results_itermemo_memo/     cross_iteration_memos: true

Four sections, in the order you should read them:

1. YIELD, per iteration. Iteration 0 is the control: neither arm has a prior memo
   there, so a large gap means something other than the memo differs between the
   arms and the rest of the comparison is not interpretable. Iterations 1 and 2 are
   the measurement — arm B's attackers open with a "## Lessons from previous
   iterations" block, arm A's do not.

2. NOVELTY of each iteration's successes. This is what the memo is *for*: it tells the
   next iteration's attackers that the previous iteration's winning strategies have
   since been trained against, so re-running them is wasted effort. So the metric is
   how far an iteration's successes sit from everything already known when it began
   (the seeded pre-history plus that arm's own earlier successes): max difflib ratio
   of each success's first user turn against that reference set. LOWER mean similarity
   and a LOWER near-duplicate fraction in arm B is the memo working — and the gap
   should widen from iteration 1 to 2 if the effect compounds.

   Note this is measured on successes that survived the near-duplicate GUARD, which
   is on in both arms at tau 0.8 — so the guard has already removed the crudest
   clones from both. What is left is whether the attacker went somewhere genuinely
   different, which the guard cannot enforce.

3. PROBE QUALITY, from each arm's eval comparison CSV. Whether the extra novelty (if
   any) actually bought a better probe.

4. MEMO AUDIT. Every memo either arm wrote — the cross-iteration memos (arm B only)
   and the per-round rolling memos (both arms) — with word counts and a
   truncated-mid-sentence check, since a memo guillotined by judge.max_tokens is fed
   back as the next prompt's input and the loss compounds.

Usage:
    scripts/compare_iteration_memo_arms.py                    # run 1
    scripts/compare_iteration_memo_arms.py --run 2            # the replicate
    scripts/compare_iteration_memo_arms.py --run both         # one report per run
    scripts/compare_iteration_memo_arms.py --tau 0.7 --show-memos
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def arms_for(run: str) -> list[tuple[str, Path, Path]]:
    """The (label, results-dir, probe-dir) triples for one run of the ablation.

    Run 2 is an independent replicate of run 1 — same configs knob for knob, different
    output paths — so the two are read with the same code and differ only by suffix.
    """
    sfx = "" if run == "1" else f"_run{run}"
    return [
        ("A nomemo", REPO / f"results_itermemo_nomemo{sfx}", REPO / f"probes/itermemo_nomemo{sfx}"),
        ("B memo", REPO / f"results_itermemo_memo{sfx}", REPO / f"probes/itermemo_memo{sfx}"),
    ]


ARMS = arms_for("1")  # rebound per --run in main(); the section fns read this module global
JSONL_NAME = "gptoss120b_probing.jsonl"
PREFIX = 600  # chars of the first user turn compared — matches JsonlStore._NEAR_DUP_PREFIX


def _first_user(sample) -> str:
    msgs = sample if isinstance(sample, list) else sample.get("messages", sample)
    for m in msgs or []:
        if m.get("role") == "user":
            return (m.get("content") or "")[:PREFIX]
    return ""


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _max_sim(text: str, reference: list[str]) -> float:
    """Highest difflib ratio between `text` and anything in `reference`.

    autojunk=False for the same reason JsonlStore._is_near disables it: difflib's
    autojunk heuristic derives its junk set from the SECOND argument, so ratio(a,b)
    != ratio(b,a) above 200 chars, and at our opener lengths it under-measures a
    genuine near-duplicate badly enough that the comparison stops meaning anything.
    """
    best = 0.0
    for other in reference:
        r = difflib.SequenceMatcher(None, text, other, autojunk=False).ratio()
        if r > best:
            best = r
    return best


def _complete(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", '"', "”", ")"))


def section_yield(loaded: dict[str, list[dict]]) -> None:
    print("=" * 78)
    print(" 1. YIELD PER ITERATION   (iteration -1 = seeded pre-history, identical by construction)")
    print("=" * 78)
    print(f"{'iter':>6} | {'arm':<10} {'attempts':>9} {'successes':>10} {'yield':>7}")
    print("-" * 78)
    iters = sorted({r.get("iteration", -1) for rows in loaded.values() for r in rows})
    for it in iters:
        for arm, rows in loaded.items():
            sel = [r for r in rows if r.get("iteration", -1) == it]
            if not sel:
                continue
            succ = sum(1 for r in sel if r.get("success"))
            print(f"{it:>6} | {arm:<10} {len(sel):>9} {succ:>10} {succ / len(sel) * 100:>6.1f}%")
        print("-" * 78)
    print("  iteration 0 = control (no prior memo in either arm) · iterations 1+ = differentiated\n")


def section_novelty(loaded: dict[str, list[dict]], tau: float) -> None:
    print("=" * 78)
    print(f" 2. NOVELTY OF EACH ITERATION'S SUCCESSES   (vs. everything known before it, tau={tau})")
    print("=" * 78)
    print(f"{'iter':>6} | {'arm':<10} {'n':>4} {'mean max-sim':>13} {'median':>8} {'>= tau':>9}   (lower = more novel)")
    print("-" * 78)
    # Iteration 0 has no memo in either arm, so its row is a second control: the arms should
    # look alike there. Every iteration >= 1 is differentiated in arm B.
    iters = sorted(
        {r.get("iteration", -1) for rows in loaded.values() for r in rows if r.get("iteration", -1) >= 0}
    )
    for it in iters:
        for arm, rows in loaded.items():
            reference = [
                _first_user(r["sample"])
                for r in rows
                if r.get("success") and r.get("iteration", -1) < it
            ]
            targets = [
                _first_user(r["sample"])
                for r in rows
                if r.get("success") and r.get("iteration", -1) == it
            ]
            if not targets or not reference:
                print(f"{it:>6} | {arm:<10} {len(targets):>4}   (nothing to compare)")
                continue
            sims = sorted(_max_sim(t, reference) for t in targets)
            mean = sum(sims) / len(sims)
            median = sims[len(sims) // 2]
            near = sum(1 for s in sims if s >= tau)
            print(
                f"{it:>6} | {arm:<10} {len(sims):>4} {mean:>13.3f} {median:>8.3f} "
                f"{near:>4} ({near / len(sims) * 100:>3.0f}%)"
            )
        print("-" * 78)
    print("  Reference set = seeded pre-history + this arm's own successes from earlier iterations.")
    print("  iteration 0 = control (no memo in either arm) · 1+ = differentiated.")
    print("  For within-arm templating instead, run: scripts/clone_rate.py on both JSONLs.\n")


def section_eval() -> None:
    print("=" * 78)
    print(" 3. PROBE QUALITY   (auroc per eval split, per retrained probe)")
    print("=" * 78)
    tables: dict[str, dict[tuple[str, str], float]] = {}
    for arm, results_dir, _ in ARMS:
        csv_path = results_dir / "gptoss120b_comparison.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            tables[arm] = {
                (row["round"], row["dataset"]): float(row["auroc"])
                for row in csv.DictReader(f)
            }
    if len(tables) < 2:
        print("  (need both arms' comparison CSVs — run the arms first)\n")
        return
    a, b = [t for t in tables.values()]
    arm_a, arm_b = list(tables)
    # The CLI writes its own aggregate row per probe; keep it out of the per-split listing
    # and out of the footer average, or it gets counted twice.
    keys = sorted(k for k in set(a) | set(b) if k[1] != "mean")
    print(f"{'probe':<10} {'split':<12} {arm_a:>10} {arm_b:>10} {'delta':>9}")
    print("-" * 78)
    for k in keys:
        va, vb = a.get(k), b.get(k)
        d = f"{vb - va:+.4f}" if va is not None and vb is not None else ""
        print(
            f"{k[0]:<10} {k[1]:<12} "
            f"{va if va is None else f'{va:.4f}':>10} {vb if vb is None else f'{vb:.4f}':>10} {d:>9}"
        )
    # Mean over splits, per probe, so a single number per iteration is visible.
    print("-" * 78)
    label = "MEAN(splits)"
    by_round: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for k in keys:
        if k in a and k in b:
            by_round[k[0]].append((a[k], b[k]))
    for rnd, pairs in sorted(by_round.items()):
        ma = sum(p[0] for p in pairs) / len(pairs)
        mb = sum(p[1] for p in pairs) / len(pairs)
        print(f"{rnd:<10} {label:<12} {ma:>10.4f} {mb:>10.4f} {mb - ma:>+9.4f}")
    print()


def section_memos(show: bool) -> None:
    print("=" * 78)
    print(" 4. MEMO AUDIT")
    print("=" * 78)
    for arm, results_dir, _ in ARMS:
        stem = results_dir / JSONL_NAME
        for kind, path in (
            ("cross-iteration", stem.with_suffix(".iteration_memos.jsonl")),
            ("rolling round", stem.with_suffix(".summaries.jsonl")),
        ):
            rows = _load(path)
            if not rows:
                # The nomemo arm has cross_iteration_memos off, so an absent sidecar there
                # is the expected result, not a missing file.
                expected = kind == "cross-iteration" and arm.endswith("nomemo")
                print(f"{arm:<10} {kind:<16} none{'  (expected — flag is off)' if expected else ''}")
                continue
            for d in rows:
                text = d.get("text", "")
                which = (
                    f"iter {d.get('iteration')}"
                    if kind == "cross-iteration"
                    else f"round {d.get('round')}"
                )
                print(
                    f"{arm:<10} {kind:<16} {which:<10} "
                    f"{d.get('n_successes', '?')}/{d.get('n_attempts', '?')} succeeded | "
                    f"{len(text.split()):>4} words | "
                    f"{'complete' if _complete(text) else 'TRUNCATED'}"
                )
                if show:
                    print("-" * 78)
                    print(text)
                    print("-" * 78)
    print("\n  Budgets: cross-iteration memo llm_judge._ITERATION_MEMO_WORD_BUDGET;")
    print("  rolling memo llm_judge._summary_word_budget(judge.max_tokens).")
    print("  TRUNCATED means judge.max_tokens cut the memo off mid-sentence — it is fed")
    print("  back as the next prompt's input, so the loss compounds.\n")


def report(run: str, tau: float, show_memos: bool) -> None:
    global ARMS
    ARMS = arms_for(run)

    print("#" * 78)
    print(f"#  RUN {run}   ({ARMS[0][1].name} vs {ARMS[1][1].name})")
    print("#" * 78 + "\n")

    loaded = {arm: _load(results_dir / JSONL_NAME) for arm, results_dir, _ in ARMS}
    missing = [arm for arm, rows in loaded.items() if not rows]
    if missing:
        print(f"WARNING: no attempts found for arm(s): {', '.join(missing)}\n")

    if any(loaded.values()):
        section_yield(loaded)
        section_novelty(loaded, tau)
    section_eval()
    section_memos(show_memos)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tau", type=float, default=0.8, help="Near-duplicate threshold (default: %(default)s)")
    p.add_argument("--show-memos", action="store_true", help="Print each memo's full text")
    p.add_argument(
        "--run",
        default="1",
        choices=["1", "2", "both"],
        help="Which run of the ablation to read (default: %(default)s). Run 2 is an "
        "independent replicate; 'both' prints one full report per run, so the arm gap "
        "in each can be read against the other.",
    )
    args = p.parse_args(argv)

    for run in (["1", "2"] if args.run == "both" else [args.run]):
        report(run, args.tau, args.show_memos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
