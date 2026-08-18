#!/usr/bin/env python
"""Replay one stored round through the summarization judge and print the memo.

Reads attempts back out of a red-team JSONL, feeds them to ``LLMJudge.summarize_round``
exactly as a live run would, and dumps both the rendered prompt and the memo that came
back. No probe, no GPU, no model load — the records already carry everything the
summarizer sees (probe score, probe label, judge label, judge reason, transcript).

Use it to check a prompt change against a round whose outcome you already know, e.g.
round 5 of the experiment7 guidance arm (1 success in 47, mean probe score 0.035):

    scripts/replay_round_memo.py --jsonl <path>/gptoss120b_probing_fp.jsonl --round 5

Add --no-hide-opposite to see the same round without the opposite-direction filter, or
--dry-run to render the prompt without spending a judge call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentic_redteam import llm_judge as L
from agentic_redteam.llm_judge import LLMJudge
from agentic_redteam.persistence import AttemptRecord


def _load_round(jsonl: Path, round_num: int | None) -> list[AttemptRecord]:
    records = [
        AttemptRecord.from_jsonl_row(line)
        for line in jsonl.read_text().splitlines()
        if line.strip()
    ]
    if round_num is None:
        return records
    return [r for r in records if r.round == round_num]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", type=Path, required=True, help="Red-team JSONL to read attempts from")
    p.add_argument("--round", type=int, default=None, help="Global round number (default: every row in the file)")
    p.add_argument("--error-type", default=None, help="Override the error_type on the records")
    p.add_argument("--model", default="openai/gpt-5.1", help="Judge model (default: %(default)s)")
    p.add_argument("--provider", default="openrouter", choices=["openrouter", "claude_sdk"])
    p.add_argument("--max-tokens", type=int, default=1024, help="Judge output budget; also sets the memo word budget")
    p.add_argument(
        "--no-hide-opposite",
        dest="hide_opposite",
        action="store_false",
        help="Keep opposite-direction misclassifications in the prompt (pre-fix behaviour)",
    )
    p.add_argument("--prior-summary", type=Path, default=None, help="File holding the prior memo (default: none)")
    p.add_argument("--out-dir", type=Path, default=Path("results_memotest/replay"))
    p.add_argument("--dry-run", action="store_true", help="Render the prompt but do not call the judge")
    args = p.parse_args(argv)

    records = _load_round(args.jsonl, args.round)
    if not records:
        print(f"No records for round {args.round} in {args.jsonl}", file=sys.stderr)
        return 1

    error_type = args.error_type or records[0].error_type
    pos, neg = records[0].pos_class_label, records[0].neg_class_label
    true_class = neg if error_type == "false_positive" else pos

    kept = L._drop_opposite_direction(records, error_type) if args.hide_opposite else records
    dropped = len(records) - len(kept)
    n_succ = sum(1 for r in kept if r.success)
    scores = [r.probe_score for r in kept]

    print(f"jsonl          : {args.jsonl}")
    print(f"round          : {args.round if args.round is not None else 'ALL'}")
    print(f"error_type     : {error_type}   (true class for a success: '{true_class}')")
    print(f"classes        : pos='{pos}'  neg='{neg}'")
    print(f"records        : {len(records)} loaded, {len(kept)} shown to the judge, {dropped} withheld")
    print(f"outcome        : {n_succ} success / {len(kept) - n_succ} fail")
    print(f"probe score    : mean {sum(scores)/len(scores):.3f}, min {min(scores):.3f}, max {max(scores):.3f}")
    print(f"word budget    : {L._summary_word_budget(args.max_tokens)} (from max_tokens={args.max_tokens})")
    print()

    prior = args.prior_summary.read_text() if args.prior_summary else ""

    judge = LLMJudge(
        model=args.model,
        system_prompt="",  # unused by summarize_round; classification prompt only
        pos_class_label=pos,
        neg_class_label=neg,
        provider=args.provider,
        max_tokens=args.max_tokens,
        hide_opposite_direction=args.hide_opposite,
    )

    # Capture the exact prompt on its way to the API.
    captured: dict[str, str] = {}
    real_call = LLMJudge._summarization_call

    def _capture(self, system, user_content, *, what):
        captured["system"] = system
        captured["user"] = user_content
        if args.dry_run:
            return "(dry run — judge not called)"
        return real_call(self, system, user_content, what=what)

    LLMJudge._summarization_call = _capture
    try:
        memo = judge.summarize_round(
            records,
            round_num=args.round if args.round is not None else 0,
            error_type=error_type,
            true_class_label=true_class,
            prior_summary=prior,
        )
    finally:
        LLMJudge._summarization_call = real_call

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"round{args.round}" if args.round is not None else "all"
    tag += "" if args.hide_opposite else "_nohide"
    paths = {
        "system": args.out_dir / f"{tag}_system.txt",
        "user": args.out_dir / f"{tag}_user.txt",
        "memo": args.out_dir / f"{tag}_memo.txt",
    }
    paths["system"].write_text(captured["system"])
    paths["user"].write_text(captured["user"])
    paths["memo"].write_text(memo)

    print(f"system prompt  : {len(captured['system']):>7,} chars  → {paths['system']}")
    print(f"user message   : {len(captured['user']):>7,} chars  → {paths['user']}")
    print(f"memo returned  : {len(memo):>7,} chars / {len(memo.split()):,} words → {paths['memo']}")
    if memo and not memo.rstrip().endswith((".", "!", "?", '"', "”", ")")):
        print("  ⚠ memo does not end on sentence punctuation — likely truncated by max_tokens")
    print()
    print("=" * 78)
    print(memo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
