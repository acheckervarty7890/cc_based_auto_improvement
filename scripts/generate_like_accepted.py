#!/usr/bin/env python
"""Generate more training samples in the style of arm 3N's ACCEPTED batches.

The eight batches the loop accepted define eight task families. This script few-shots the
generator with real pairs drawn from each accepted batch and asks for more of the same
kind, family by family. Nothing about the eval splits enters the prompt, the family list
or the filtering — the only inputs are `batches.jsonl` (the run's own record) and the
probe's class labels.

Guards, mirroring `agentic_redteam.generator._admit`:
  * length  — `TokenBudget` against tuberlens' 1024-token activation cap, failing open
  * label   — must normalize to one of the probe's two classes
  * novelty — never equal to an existing accepted sample, the injected batch, or a
              sibling generated earlier in this run

Writes a JSONL of `{inputs, labels, family}` rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RUN_DIR = REPO / "results_gen_gemma27b_instructions_nemotron"
PROBE = REPO / "probes/gen_gemma27b_instructions_nemotron/probe_iter13.pkl"

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

# The eight (iteration, batch_index) pairs the loop accepted, in order.
ACCEPTED_KEYS = [(0, 4), (1, 1), (2, 0), (4, 3), (5, 4), (7, 2), (9, 1), (11, 3)]
INJECTED_KEY = (10, 0)

SYSTEM = """You are helping build a labelled training set for a text classifier.

The classifier reads a conversation between a user and an assistant and predicts one of two labels:
  - assistant_follows_the_instruction
  - assistant_does_not_follow_the_instruction

Your job is to write more labelled conversations for it. You will be shown a task family and
several real examples already in the set; write NEW conversations of the same kind.

Rules:
  - Match the style, length and structure of the examples you are shown. This matters: the
    examples define the family.
  - Vary the subject matter. Do not reuse the topics in the examples.
  - Keep the two classes matched on surface features — same domains, same rough length, same
    register. The ONLY thing that should distinguish them is whether the assistant followed
    the user's instruction.
  - Prefer matched pairs: the same user turn appearing twice, once with a compliant assistant
    turn and once with a violating one.
  - Keep every conversation short enough to fit comfortably in 1024 tokens.

Reply with a single JSON array and nothing else. Each element:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
   "label": "assistant_follows_the_instruction" | "assistant_does_not_follow_the_instruction"}
"""


def latest_batches() -> dict[tuple[int, int], dict]:
    latest: dict[tuple[int, int], dict] = {}
    with (RUN_DIR / "batches.jsonl").open() as fh:
        for line in fh:
            rec = json.loads(line)
            latest[(rec["iteration"], rec["batch_index"])] = rec
    return latest


def key_of(messages) -> str:
    return json.dumps(
        [[m["role"], " ".join(m["content"].split())] for m in messages], sort_keys=True
    )


def build_request(rec: dict, n: int, n_shots: int) -> str:
    """Few-shot prompt for one family: its direction plus real pairs from the batch."""
    samples = rec["samples"]
    # Alternate classes so the shots show both sides, and prefer a matched pair.
    shots, seen = [], set()
    for want_neg in (False, True, False, True):
        for s in samples:
            is_neg = s["label"].startswith("assistant_does_not")
            if is_neg == want_neg and id(s) not in seen:
                shots.append(s)
                seen.add(id(s))
                break
        if len(shots) >= n_shots:
            break
    body = json.dumps(
        [{"messages": s["messages"], "label": s["label"]} for s in shots],
        indent=2,
        ensure_ascii=False,
    )
    half = n // 2
    return (
        f"TASK FAMILY\n{rec['direction']}\n\n"
        f"EXAMPLES ALREADY IN THE SET (write more like these)\n{body}\n\n"
        f"Write {n} new conversations in this family: {half} labelled "
        f"assistant_follows_the_instruction and {half} labelled "
        f"assistant_does_not_follow_the_instruction. New subject matter, same style. "
        f"Reply with the JSON array only."
    )


async def call_model(client, model: str, prompt: str, max_tokens: int, tries: int = 5) -> str | None:
    delay = 10.0
    for attempt in range(1, tries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content if resp.choices else None
            if text:
                return text
            print(f"    [warn] attempt {attempt}/{tries}: empty reply", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — every API failure here is retryable
            print(
                f"    [warn] attempt {attempt}/{tries}: {type(exc).__name__}: {str(exc)[:180]}",
                file=sys.stderr,
            )
        if attempt < tries:
            await asyncio.sleep(delay)
            delay *= 2
    return None


async def main_async(args) -> None:
    from openai import AsyncOpenAI

    from agentic_redteam.generator import normalize_label, parse_samples
    from agentic_redteam.retrain import read_probe_metadata
    from agentic_redteam.token_budget import TokenBudget

    meta = read_probe_metadata(PROBE)
    pos, neg = meta["pos_class_label"], meta["neg_class_label"]
    print(f"classes: {pos} / {neg}")

    # Same transforms the fit and the eval use, so the length guard counts what the
    # extractor will actually tokenize.
    budget = TokenBudget(
        model_name=meta["model_name"],
        max_tokens=1024,
        combine_consecutive_messages=True,
        convert_tool_to_assistant=True,
    )
    budget.warmup()

    latest = latest_batches()

    # Novelty: everything already in the training set, accepted or injected.
    seen: set[str] = set()
    for k in ACCEPTED_KEYS + [INJECTED_KEY]:
        for s in latest[k]["samples"]:
            seen.add(key_of(s["messages"]))
    print(f"novelty guard seeded with {len(seen)} existing conversations")

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        timeout=float(os.environ.get("OPENROUTER_TIMEOUT_S", "600")),
    )

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    kept: list[dict] = []
    stats: dict[str, dict] = {}

    async def do_family(it: int, bk: int) -> None:
        rec = latest[(it, bk)]
        fam = f"it{it}b{bk}"
        async with sem:
            print(f"  [{fam}] requesting {args.per_family} ...")
            text = await call_model(
                client, args.model, build_request(rec, args.per_family, args.shots), args.max_tokens
            )
        if not text:
            print(f"  [{fam}] FAILED — no reply")
            stats[fam] = {"kept": 0, "error": "no reply"}
            return
        try:
            parsed, malformed = parse_samples(text, pos, neg)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{fam}] parse failed: {type(exc).__name__}: {str(exc)[:160]}")
            stats[fam] = {"kept": 0, "error": "parse"}
            return

        n_long = n_dup = n_bad = 0
        local = []
        async with lock:
            for s in parsed:
                messages = [{"role": m.role, "content": m.content} for m in s.conversation.messages]
                label = normalize_label(s.label, pos, neg)
                if label is None:
                    n_bad += 1
                    continue
                if budget.overage(messages) is not None:
                    n_long += 1
                    continue
                k = key_of(messages)
                if k in seen:
                    n_dup += 1
                    continue
                seen.add(k)
                local.append({"inputs": messages, "labels": label, "family": fam})
            kept.extend(local)
        npos = sum(1 for r in local if r["labels"] == pos)
        print(
            f"  [{fam}] kept {len(local)} ({npos} pos / {len(local) - npos} neg); "
            f"dropped long={n_long} dup={n_dup} bad_label={n_bad} malformed={malformed}"
        )
        stats[fam] = {
            "kept": len(local), "pos": npos, "long": n_long,
            "dup": n_dup, "bad": n_bad, "malformed": malformed,
        }

    await asyncio.gather(*(do_family(it, bk) for it, bk in ACCEPTED_KEYS))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    npos = sum(1 for r in kept if r["labels"] == pos)
    print(f"\nwrote {len(kept)} rows ({npos} pos / {len(kept) - npos} neg) to {args.out}")
    print(json.dumps(stats, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-family", type=int, default=12, help="samples requested per accepted-batch family (even)")
    ap.add_argument("--shots", type=int, default=4, help="real examples shown per family")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=14000)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPO / "data/instructions_like_accepted62.jsonl")
    args = ap.parse_args()
    if args.per_family % 2:
        raise SystemExit("--per-family must be even")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
