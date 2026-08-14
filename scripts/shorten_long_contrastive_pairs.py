#!/usr/bin/env python
"""Re-generate the contrastive pairs that are longer than the activation window.

The problem
-----------
``LLMModel.get_activations`` pads or **truncates** every conversation to
``max_length=1024`` tokens (``tuberlens/model.py:433``: ``tensor[:, :max_len]``).
Anything past token 1024 never reaches the probe. For an *assistant-centric*
concept like ``harmful_to_human`` that is the worst possible cut: the label is a
property of the assistant's reply, which is exactly the tail that gets dropped —
so an over-long sample trains the probe on a premise with its verdict amputated.

The red-team *sources* are attacker-written and almost never that long (3 of 1656
rows in experiment11). The **generated contrastive pairs** are, because
``_generation_system_prompt`` only asks for "similar structure and length" and the
generator routinely elaborates well past its source:

    arm            generated  over 1024   median   max
    gptoss120b           389    34 (8.7%)    545  1664
    deepseekv4pro        439   71 (16.2%)    635  2014

What this script does
---------------------
For each over-length **generated** conversation in
``probes/<arm>/redteam_postprocessed_iter{N}.jsonl`` it re-asks the same
preprocessing LLM (same provider/model/system prompt as the retrain used) for the
same pair under an explicit word budget, and keeps re-asking — tightening the
budget from the measured words-per-token of each rejected attempt — until the
result fits under the cap or ``--max-attempts`` is spent. It then:

1. appends the corrected record to ``contrastive_cache.jsonl`` under the pair's
   **original cache key**, so a future retrain (whose key is derived from the
   *source* conversation + target label, not from the generated text) picks up the
   short version instead of re-generating a long one. ``_load_cache`` is
   last-write-wins, which is what makes appending an update;
2. rewrites the ``redteam_postprocessed_iter{N}.jsonl`` rows in place — same ids,
   same labels, same order, only the over-length conversations replaced (a
   ``.bak`` is kept unless ``--no-backup``);
3. writes a per-pair report to ``shortened_pairs_iter{N}.jsonl``.

Only the iteration(s) named by ``--iterations`` are rewritten (default 3, the last
one). Earlier snapshots are the record of what earlier probes actually trained on
and are deliberately left alone.

The regenerated pair is validated exactly as the retrain path would see it: the
config's message transforms are applied, then the conversation is rendered through
the probe model's chat template and tokenized the way ``tokenize_inputs`` does
(including its ``[:, 1:]`` first-token strip). A generation that is malformed
(``_is_well_formed_conversation``) is rejected and retried like any other failure.

After it finishes, the changed conversations have new content-addressed cache keys
(``retrain._redteam_activation_cache_path``), so their activations must be
recomputed and re-published:

    .venv_claude/bin/python scripts/publish_kaggle_redteam_activations.py sync --iterations 3

Requires ``OPENROUTER_API_KEY`` (the preprocessing model is ``openai/gpt-5.1`` via
OpenRouter in both arms). ``--dry-run`` needs no key: it only measures and reports.

Typical use::

    # what is over the cap, per arm, no LLM calls
    .venv_claude/bin/python scripts/shorten_long_contrastive_pairs.py --dry-run

    # fix iteration 3 for both arms
    .venv_claude/bin/python scripts/shorten_long_contrastive_pairs.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agentic_redteam.circuit_breaker import OpenRouterOutageError  # noqa: E402
from agentic_redteam.config import load_config  # noqa: E402
from agentic_redteam.preprocessing import (  # noqa: E402
    _ContrastiveLLM,
    _extract_messages,
    _generation_system_prompt,
    _generation_user_prompt,
    _is_well_formed_conversation,
    _short_label,
)
from publish_kaggle_redteam_activations import ARMS, Arm, _probe_metadata  # noqa: E402

# tuberlens' activation window. Anything past this is discarded at extraction time.
DEFAULT_LIMIT_TOKENS = 1024
# Aim below the cap, not at it: the tokenizer count here must match the extraction
# run's exactly, and a pair that lands on 1023 has no room for a tokenizer/template
# revision. Cheap insurance — the median pair is ~600 tokens.
DEFAULT_MARGIN_TOKENS = 64
DEFAULT_MAX_ATTEMPTS = 5
# Seed words-per-token for the first budget, before any attempt has been measured.
# Dense conversational English runs ~0.6-0.7; the loop replaces this with the ratio
# actually observed on the rejected attempt.
WORDS_PER_TOKEN = 0.62
# Never ask for less than this — below it the generator starts dropping the turns
# that carry the label rather than trimming prose.
MIN_WORD_BUDGET = 90
# Minimum tightening applied after a rejected attempt, on top of the measured ratio.
BUDGET_DECAY = 0.8

OUTAGE_EXIT_CODE = 3  # matches cli.OUTAGE_EXIT_CODE


# --------------------------------------------------------------------------------------
# Measuring a conversation the way extraction will
# --------------------------------------------------------------------------------------


class Measurer:
    """Token length of a conversation, as ``get_activations`` will see it.

    Mirrors ``tuberlens.model.tokenize_inputs``: render through the model's chat
    template with ``add_generation_prompt=False``, tokenize with the loader's
    ``default_tokenize_kwargs``, then drop the first token (``v[:, 1:]``). The
    config's message transforms are applied first, because the postprocessed dump
    — and hence the extraction input — is written *after* them.
    """

    def __init__(self, model_name: str, combine: bool, convert: bool) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._lock = threading.Lock()  # fast tokenizers are shared across the pool
        self.combine = combine
        self.convert = convert

    def transform(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Apply the config's loader transforms, as ``_apply_message_transforms`` does."""
        if not (self.combine or self.convert):
            return [{"role": m["role"], "content": m["content"]} for m in messages]
        from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

        msgs = [TLMessage(role=m["role"], content=m["content"]) for m in messages]
        if self.convert:
            msgs = LabelledDataset._convert_tool_to_assistant(msgs)
        if self.combine:
            msgs = LabelledDataset._combine_consecutive_messages(msgs)
        return [{"role": m.role, "content": m.content} for m in msgs]

    def n_tokens(self, messages: list[dict[str, str]]) -> int:
        dialogue = [{"role": m["role"], "content": m["content"]} for m in messages]
        with self._lock:
            text = self._tok.apply_chat_template(
                [dialogue], tokenize=False, add_generation_prompt=False
            )
            ids = self._tok(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=2**13,
                add_special_tokens=False,
            )["input_ids"]
        return int(ids.shape[1]) - 1  # tokenize_inputs strips the first token


def n_words(messages: list[dict[str, str]]) -> int:
    return sum(len(str(m.get("content", "")).split()) for m in messages)


def canon(messages: list[dict[str, str]]) -> str:
    """Content key for matching a dump row back to the cache record that produced it."""
    return json.dumps(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        sort_keys=True,
        ensure_ascii=False,
    )


# --------------------------------------------------------------------------------------
# The re-ask
# --------------------------------------------------------------------------------------


def _length_system_block(limit_tokens: int, budget_words: int, target_label: str) -> str:
    tgt = _short_label(target_label)
    return (
        "\n\nHARD LENGTH CONSTRAINT\n"
        f"The conversation you produce is read by a model that sees only its first "
        f"{limit_tokens} tokens; everything after that is discarded before it is used. "
        f'An over-long conversation therefore loses exactly the part that makes it "{tgt}".\n'
        f"Keep the WHOLE conversation — all messages combined — at or under "
        f"{budget_words} words.\n"
        "Keep the same number of turns and the class-defining behaviour; cut elaboration "
        "instead. No long enumerations, headers, procedures or boilerplate unless the "
        "original conversation had them. A short complete scenario is worth far more than "
        "a detailed one that gets cut off."
    )


def _retry_user_block(
    prev_tokens: int, prev_words: int, limit_tokens: int, budget_words: int, target_label: str
) -> str:
    tgt = _short_label(target_label)
    return (
        f"\n\nYour previous attempt was too long: it rendered to {prev_tokens} tokens "
        f"(about {prev_words} words), over the {limit_tokens}-token limit. Produce a NEW "
        f'"{tgt}" conversation for the same source, with the same premise and the same '
        f"number of turns, but at most {budget_words} words in total. Shorten the prose, "
        f'not the "{tgt}" behaviour. Reply with the same JSON object format.'
    )


@dataclass
class Outcome:
    """What happened to one over-length pair."""

    key: str
    row_index: int
    row_id: str
    label: str
    status: str  # fixed | reused_cache | still_long | failed
    before_tokens: int
    after_tokens: int | None = None
    attempts: int = 0
    messages: list[dict[str, str]] | None = None  # transformed, ready for the dump
    record: dict | None = None  # cache record to append (None when nothing changed)
    notes: list[str] = field(default_factory=list)


def shorten_one(
    *,
    llm: _ContrastiveLLM,
    measurer: Measurer,
    record: dict,
    key: str,
    row_index: int,
    row_id: str,
    label: str,
    before_tokens: int,
    limit_tokens: int,
    target_tokens: int,
    max_attempts: int,
) -> Outcome:
    """Re-ask for one pair until it fits, tightening the word budget each round."""
    source_messages = _extract_messages(
        {"inputs": record.get("original_messages", [])}, "inputs"
    )
    current_label = str(record.get("original_label", ""))
    target_label = str(record.get("labels", ""))

    # Start from whichever is tighter: the token cap converted to words, or a length
    # comparable to the source (the pair is supposed to look like its source, and the
    # sources here are short — a 200-token source has no business yielding 1600 tokens).
    budget = int(target_tokens * WORDS_PER_TOKEN)
    src_words = n_words(source_messages)
    if src_words:
        budget = min(budget, max(MIN_WORD_BUDGET, int(1.5 * src_words)))
    budget = max(MIN_WORD_BUDGET, budget)

    outcome = Outcome(
        key=key,
        row_index=row_index,
        row_id=row_id,
        label=label,
        status="failed",
        before_tokens=before_tokens,
    )
    best: tuple[int, list[dict[str, str]], dict] | None = None
    prev_tokens, prev_words = before_tokens, n_words(record.get("inputs", []))

    for attempt in range(1, max_attempts + 1):
        outcome.attempts = attempt
        system = (
            _generation_system_prompt(
                current_label,
                target_label,
                assistant_centric=llm.assistant_centric,
                concept_description=llm.concept_description,
                label_guidance=llm.label_guidance,
            )
            + _length_system_block(limit_tokens, budget, target_label)
        )
        user = _generation_user_prompt(
            source_messages, current_label, target_label
        ) + _retry_user_block(prev_tokens, prev_words, limit_tokens, budget, target_label)

        response = llm.call(system, user)
        if not response or "generated_messages" not in response:
            outcome.notes.append(f"attempt {attempt}: no parseable generation")
            continue
        new_messages = _extract_messages(
            {"inputs": response["generated_messages"]}, "inputs"
        )
        if not _is_well_formed_conversation(new_messages):
            outcome.notes.append(f"attempt {attempt}: malformed conversation, rejected")
            continue

        transformed = measurer.transform(new_messages)
        got = measurer.n_tokens(transformed)
        new_record = dict(record)
        new_record.update(
            {
                "inputs": new_messages,
                "generation_explanation": str(response.get("explanation", "")),
                "shortened_from_tokens": before_tokens,
                "shortened_to_tokens": got,
                "shortened_attempts": attempt,
                "shortened_word_budget": budget,
                "shortened_limit_tokens": limit_tokens,
            }
        )
        if best is None or got < best[0]:
            best = (got, transformed, new_record)
        if got <= limit_tokens:
            outcome.status = "fixed"
            outcome.after_tokens = got
            outcome.messages = transformed
            outcome.record = new_record
            return outcome

        outcome.notes.append(f"attempt {attempt}: {got} tokens at budget {budget}w")
        # Retarget off the words-per-token this attempt actually measured, but never
        # let an attempt end at a budget the last one already failed at: the generator
        # overshoots what it is told, so a ratio-only update can leave the budget flat
        # (it does whenever the attempt's ratio is above the seed) and burn every
        # remaining attempt at the same length. BUDGET_DECAY forces convergence.
        got_words = n_words(transformed)
        ratio = got_words / max(1, got)
        budget = max(
            MIN_WORD_BUDGET,
            min(int(target_tokens * ratio * 0.9), int(budget * BUDGET_DECAY)),
        )
        prev_tokens, prev_words = got, got_words

    if best is not None and best[0] < before_tokens:
        # Every attempt was over the cap, but the shortest one still loses less at
        # truncation than what is there now. Keep it and say so.
        outcome.status = "still_long"
        outcome.after_tokens = best[0]
        outcome.messages = best[1]
        outcome.record = best[2]
    return outcome


# --------------------------------------------------------------------------------------
# Per-arm driver
# --------------------------------------------------------------------------------------


def _load_cache_rows(cache_path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """``(last-wins key→record, canon(generated messages)→key over the WHOLE history)``.

    The canon index covers every line, not just the surviving one per key, so a
    re-run still recognises a dump row whose cache entry this script has already
    superseded — that is what makes an interrupted run resumable.
    """
    latest: dict[str, dict] = {}
    by_canon: dict[str, str] = {}
    if not cache_path.exists():
        return latest, by_canon
    with cache_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "key" not in row or "record" not in row:
                continue
            latest[row["key"]] = row["record"]
            by_canon.setdefault(canon(row["record"].get("inputs", [])), row["key"])
    return latest, by_canon


def process_arm(arm: Arm, args: argparse.Namespace) -> int:
    """Returns the number of rows still over the cap after processing this arm."""
    model_name, layer, pos_label, neg_label = _probe_metadata(arm)
    config = load_config(arm.config_path)
    if config.preprocessing is None:
        raise SystemExit(f"{arm.name}: config has no preprocessing: section.")

    combine = bool(config.eval.combine_consecutive_messages)
    convert = bool(config.eval.convert_tool_to_assistant)
    limit_tokens = args.limit_tokens
    target_tokens = max(1, limit_tokens - args.margin_tokens)

    print(f"\n{'=' * 86}\n{arm.name}: {arm.probe_dir}\n{'=' * 86}")
    print(f"probe          {model_name}  L{layer}   ({pos_label} / {neg_label})")
    print(f"transforms     combine={combine}  convert_tool_to_assistant={convert}")
    print(
        f"generator      {config.preprocessing.provider}/{args.model or config.preprocessing.model}"
        f"  max_tokens={config.preprocessing.max_tokens}"
        f"  assistant_centric={config.preprocessing.assistant_centric}"
    )
    print(f"cap            {limit_tokens} tokens (aiming for <= {target_tokens})")

    measurer = Measurer(model_name, combine, convert)
    cache_path = arm.probe_path / "contrastive_cache.jsonl"
    latest, by_canon = _load_cache_rows(cache_path)
    print(f"cache          {cache_path.name}: {len(latest)} keys, {len(by_canon)} distinct pairs")

    cache_lock = threading.Lock()
    still_over = 0

    for iteration in args.iterations:
        dump_path = arm.probe_path / f"redteam_postprocessed_iter{iteration}.jsonl"
        if not dump_path.exists():
            print(f"\n[iter{iteration}] {dump_path} not found — skipping.")
            continue
        rows = [json.loads(l) for l in dump_path.open(encoding="utf-8") if l.strip()]

        over: list[tuple[int, dict, int]] = []  # (row index, row, tokens)
        n_source_over = 0
        pending: list[dict] = []
        for i, row in enumerate(rows):
            tokens = measurer.n_tokens(row["inputs"])
            if tokens <= limit_tokens:
                continue
            key = by_canon.get(canon(row["inputs"]))
            if key is None:
                n_source_over += 1  # attacker-written source: not ours to regenerate
                print(
                    f"  [iter{iteration}] row {i} ({row.get('id')}, {row.get('label')}): "
                    f"{tokens} tokens, but it is a red-team SOURCE, not a generated pair "
                    "— left untouched."
                )
                continue
            over.append((i, row, tokens))
            pending.append({"key": key})

        print(
            f"\n[iter{iteration}] {len(rows)} rows: {len(over)} generated pairs over "
            f"{limit_tokens} tokens, {n_source_over} over-length sources (not fixable here)."
        )
        if args.dry_run:
            for i, row, tokens in over:
                print(f"    row {i:4d}  {row.get('id'):>14}  {row.get('label'):>8}  {tokens} tokens")
            still_over += len(over)
            continue
        if not over:
            continue

        # Anything already superseded by an earlier (interrupted) run is free.
        work: list[tuple[int, dict, int, str, dict]] = []
        outcomes: list[Outcome] = []
        for (i, row, tokens), p in zip(over, pending):
            key = p["key"]
            record = latest.get(key)
            if record is None:
                print(f"    row {i}: cache key {key[:12]} has no surviving record — skipping.")
                continue
            cached_msgs = measurer.transform(_extract_messages(record, "inputs"))
            cached_tokens = measurer.n_tokens(cached_msgs)
            if cached_tokens <= limit_tokens:
                outcomes.append(
                    Outcome(
                        key=key,
                        row_index=i,
                        row_id=str(row.get("id")),
                        label=str(row.get("label")),
                        status="reused_cache",
                        before_tokens=tokens,
                        after_tokens=cached_tokens,
                        messages=cached_msgs,
                        notes=["cache already holds a short version (earlier run)"],
                    )
                )
                continue
            work.append((i, row, tokens, key, record))

        if work:
            llm = _ContrastiveLLM(
                config.preprocessing.provider,
                args.model or config.preprocessing.model,
                config.preprocessing.max_tokens,
                assistant_centric=config.preprocessing.assistant_centric,
                concept_description=config.preprocessing.concept_description,
                label_guidance=config.preprocessing.label_guidance,
            )
            llm._ensure_client()  # initialize once before fan-out

            def _work(item):
                i, row, tokens, key, record = item
                outcome = shorten_one(
                    llm=llm,
                    measurer=measurer,
                    record=record,
                    key=key,
                    row_index=i,
                    row_id=str(row.get("id")),
                    label=str(row.get("label")),
                    before_tokens=tokens,
                    limit_tokens=limit_tokens,
                    target_tokens=target_tokens,
                    max_attempts=args.max_attempts,
                )
                if outcome.record is not None:
                    # Write through as each pair lands: a run stopped half way leaves a
                    # cache a re-run can pick up, instead of losing every call it paid for.
                    with cache_lock:
                        with cache_path.open("a", encoding="utf-8") as fh:
                            fh.write(
                                json.dumps(
                                    {"key": key, "record": outcome.record},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        latest[key] = outcome.record
                print(
                    f"    row {outcome.row_index:4d} {outcome.row_id:>14} "
                    f"{outcome.before_tokens:>5} -> "
                    f"{outcome.after_tokens if outcome.after_tokens is not None else '   -'} "
                    f"tokens  [{outcome.status}, {outcome.attempts} attempt(s)]"
                )
                return outcome

            workers = max(1, min(config.preprocessing.max_concurrent, len(work)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                outcomes.extend(pool.map(_work, work))

        # Rewrite the dump: same ids, labels and order — only the fixed rows change.
        n_changed = 0
        for outcome in outcomes:
            if outcome.messages is None:
                continue
            rows[outcome.row_index]["inputs"] = outcome.messages
            n_changed += 1
        if n_changed:
            if not args.no_backup:
                backup = dump_path.with_suffix(dump_path.suffix + ".bak")
                if not backup.exists():
                    shutil.copy2(dump_path, backup)
                    print(f"    backed up original dump to {backup.name}")
            with dump_path.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"    rewrote {n_changed} row(s) in {dump_path.name}")

        report_path = arm.probe_path / f"shortened_pairs_iter{iteration}.jsonl"
        with report_path.open("a", encoding="utf-8") as fh:
            for outcome in outcomes:
                fh.write(
                    json.dumps(
                        {
                            "arm": arm.name,
                            "iteration": iteration,
                            "cache_key": outcome.key,
                            "row_index": outcome.row_index,
                            "id": outcome.row_id,
                            "label": outcome.label,
                            "status": outcome.status,
                            "before_tokens": outcome.before_tokens,
                            "after_tokens": outcome.after_tokens,
                            "attempts": outcome.attempts,
                            "limit_tokens": limit_tokens,
                            "notes": outcome.notes,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        by_status: dict[str, int] = {}
        for outcome in outcomes:
            by_status[outcome.status] = by_status.get(outcome.status, 0) + 1
        leftover = sum(
            1 for o in outcomes if o.after_tokens is None or o.after_tokens > limit_tokens
        )
        still_over += leftover + n_source_over
        print(
            f"  [iter{iteration}] {by_status}; {leftover} generated pair(s) still over the "
            f"cap. Report appended to {report_path.name}"
        )

    return still_over


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        nargs="+",
        default=None,
        help=f"Which arm(s) to process (default all): {sorted(ARMS)}",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[3],
        help="Which redteam_postprocessed_iter{N} dumps to fix (default 3 — the last "
        "iteration). Earlier snapshots record what earlier probes trained on; only pass "
        "them if you intend to rewrite that history.",
    )
    parser.add_argument(
        "--limit-tokens",
        type=int,
        default=DEFAULT_LIMIT_TOKENS,
        help=f"Activation window a conversation must fit in (default {DEFAULT_LIMIT_TOKENS}, "
        "tuberlens' get_activations max_length).",
    )
    parser.add_argument(
        "--margin-tokens",
        type=int,
        default=DEFAULT_MARGIN_TOKENS,
        help=f"Aim this far under the cap when setting word budgets (default {DEFAULT_MARGIN_TOKENS}).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"LLM calls per over-length pair (default {DEFAULT_MAX_ATTEMPTS}). If none fits, "
        "the shortest attempt is kept when it beats the original, and reported as still_long.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the preprocessing model (default: the config's).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Measure and report only — no LLM calls, no writes (needs no API key).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not keep a .bak of a rewritten postprocessed dump.",
    )
    args = parser.parse_args(argv)

    if not args.arm or "all" in args.arm:
        arms = list(ARMS.values())
    else:
        unknown = [a for a in args.arm if a not in ARMS]
        if unknown:
            parser.error(f"unknown arm(s) {unknown}; known: {sorted(ARMS)}")
        arms = [ARMS[a] for a in args.arm]

    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error(
            "OPENROUTER_API_KEY is not set (both arms' preprocessing provider is "
            "openrouter). Use --dry-run to inspect without calling the LLM."
        )

    total_left = 0
    try:
        for arm in arms:
            total_left += process_arm(arm, args)
    except OpenRouterOutageError as exc:
        # Never swallowed (see CLAUDE.md): stop rather than silently leaving pairs long.
        # Everything already fixed is in the cache; re-run to continue.
        print(f"\nABORTED — OpenRouter is unusable: {exc}", file=sys.stderr)
        print("Fix credits/key and re-run; completed pairs are already cached.", file=sys.stderr)
        return OUTAGE_EXIT_CODE

    print(f"\n{'=' * 86}")
    if args.dry_run:
        print(f"DRY RUN: {total_left} generated pair(s) over the cap. Re-run without --dry-run.")
    else:
        print(f"Done. {total_left} conversation(s) still over the cap (see the reports above).")
        print(
            "Next: recompute + republish the activations for the changed conversations:\n"
            "  .venv_claude/bin/python scripts/publish_kaggle_redteam_activations.py sync "
            f"--iterations {' '.join(str(i) for i in args.iterations)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
