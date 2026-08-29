#!/usr/bin/env python
"""Cross the style of one rank slot with the vocabulary of the other.

Slot 1 (the family ranked #1 by Δdev in each of the seven same-prompt draws, pooled)
scores +0.0335 on eval; slot 8 scores −0.0480. Their vocabularies are one-sided but
their register is the same, so the open question is whether the gap rides on *what the
conversations are about* or on *how they are built*.

Two arms, each a **sample-by-sample rewrite**: every conversation in the source corpus is
sent to the generator on its own and comes back with its subject matter rebuilt on the
other side's words, keeping its shape, length, register and — critically — the same
relationship between the instruction and the reply. One call per conversation; nothing is
invented from scratch and no examples are shown.

  A  r1style_r8words — every slot-1 conversation rewritten onto slot-8 words
  B  r8style_r1words — every slot-8 conversation rewritten onto slot-1 words

If style carries the effect, A tracks slot 1 and B tracks slot 8. If vocabulary carries
it, they swap. If neither, both land in the noise — which is itself the answer.

Guards mirror `agentic_redteam.generator._admit`: length (`TokenBudget`, failing open),
label (must normalize to one of the probe's two classes), novelty (never equal to a shot,
to a conversation in either source corpus, or to a sibling generated in this run).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PROBE = REPO / "probes/gen_gemma27b_instructions_nemotron/probe_iter13.pkl"
SLOT1 = REPO / "data/union_slot1.jsonl"
SLOT8 = REPO / "data/union_slot8.jsonl"

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

# The five most one-sided content words on each side (rate per 10k, other side ~0):
# slot1: source 47, cloudsql 38, centrifuge 33, email 29, suggest 29
# slot8: primary 37, region 36, deploy 35, build 33, image 33
SLOT1_WORDS = ["source", "cloudsql", "centrifuge", "email", "suggest"]
SLOT8_WORDS = ["primary", "region", "deploy", "build", "image"]

SYSTEM = """You maintain a labelled training set for a text classifier.

The classifier reads a conversation between a user and an assistant and predicts one of two labels:
  - assistant_follows_the_instruction
  - assistant_does_not_follow_the_instruction

You will be given ONE conversation from the set, its label, and a short list of words.
Rewrite that conversation so its subject matter is built on those words. This is a rewrite,
not a new conversation.

Keep:
  - the same number of turns and the same speaker order
  - the same approximate length, per turn
  - the same register, tone and formatting (if the reply is a numbered list, keep a numbered
    list; if it is one terse sentence, keep one terse sentence)
  - above all, the same relationship between the user's instruction and the assistant's reply.
    If the assistant followed the instruction, your rewrite must follow it in the same way and
    for the same reason. If the assistant broke it, your rewrite must break it in the same way.
    The label must stay true of the rewrite.

Change only what the conversation is about. Use at least two of the given words naturally,
as part of the scenario rather than dropped in.

Reply with a single JSON object and nothing else:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
   "label": "assistant_follows_the_instruction" | "assistant_does_not_follow_the_instruction"}
"""


def key_of(messages) -> str:
    return json.dumps(
        [[m["role"], " ".join(m["content"].split())] for m in messages], sort_keys=True
    )


def read_corpus(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def build_request(row: dict, words: list[str]) -> str:
    body = json.dumps({"messages": row["inputs"], "label": row["labels"]},
                      indent=2, ensure_ascii=False)
    return (
        f"CONVERSATION TO REWRITE\n{body}\n\n"
        f"WORDS TO BUILD THE NEW SUBJECT MATTER ON\n"
        + "\n".join(f"  - {w}" for w in words)
        + "\n\nRewrite it. Same shape, same lengths, same label. Reply with the JSON object only."
    )


async def call_model(client, model, prompt, max_tokens, tries=5) -> str | None:
    delay = 10.0
    for attempt in range(1, tries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content if resp.choices else None
            if text:
                return text
            print(f"    [warn] attempt {attempt}/{tries}: empty reply", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — every API failure here is retryable
            print(f"    [warn] attempt {attempt}/{tries}: {type(exc).__name__}: {str(exc)[:160]}",
                  file=sys.stderr)
        if attempt < tries:
            await asyncio.sleep(delay)
            delay *= 2
    return None


ARMS = {
    "r1style_r8words": (SLOT1, SLOT8_WORDS),
    "r8style_r1words": (SLOT8, SLOT1_WORDS),
}


async def main_async(args) -> None:
    from openai import AsyncOpenAI

    from agentic_redteam.generator import normalize_label, parse_samples
    from agentic_redteam.retrain import read_probe_metadata
    from agentic_redteam.token_budget import TokenBudget

    meta = read_probe_metadata(PROBE)
    pos, neg = meta["pos_class_label"], meta["neg_class_label"]
    budget = TokenBudget(model_name=meta["model_name"], max_tokens=1024,
                         combine_consecutive_messages=True, convert_tool_to_assistant=True)
    budget.warmup()

    # Novelty against BOTH source corpora, so nothing is copied out of the shots.
    seen = {key_of(r["inputs"]) for p in (SLOT1, SLOT8) for r in read_corpus(p)}
    print(f"novelty guard seeded with {len(seen)} existing conversations")

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        timeout=float(os.environ.get("OPENROUTER_TIMEOUT_S", "600")),
    )
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    kept: dict[str, list[dict]] = {}

    stats: dict[str, Counter] = {}

    async def rewrite_one(arm: str, rep: int, row: dict) -> None:
        _source, words = ARMS[arm]
        tag = f"{arm}_rep{rep}"
        async with sem:
            text = await call_model(client, args.model, build_request(row, words), args.max_tokens)
        st = stats.setdefault(tag, Counter())
        if not text:
            st["no_reply"] += 1
            return
        try:
            parsed, _malformed = parse_samples(text, pos, neg)
        except Exception:  # noqa: BLE001
            st["parse"] += 1
            return
        if not parsed:
            st["empty"] += 1
            return
        s = parsed[0]  # one conversation in, one out
        messages = [{"role": m.role, "content": m.content} for m in s.conversation.messages]
        label = normalize_label(s.label, pos, neg)
        # The rewrite is supposed to preserve the compliance relation. A flipped label means
        # it did not, so the row is dropped rather than relabelled.
        if label is None:
            st["bad_label"] += 1
            return
        if label != row["labels"]:
            st["label_flip"] += 1
            return
        if len(messages) != len(row["inputs"]):
            st["shape"] += 1
            return
        if budget.overage(messages) is not None:
            st["too_long"] += 1
            return
        async with lock:
            k = key_of(messages)
            if k in seen:
                st["duplicate"] += 1
                return
            seen.add(k)
            kept.setdefault(tag, []).append(
                {"inputs": messages, "labels": label, "family": tag})
            st["kept"] += 1

    jobs = []
    for arm in ARMS:
        rows = read_corpus(ARMS[arm][0])
        for rep in range(1, args.reps + 1):
            jobs += [rewrite_one(arm, rep, row) for row in rows]
    print(f"{len(jobs)} rewrite calls ({args.reps} replications x 2 arms)")
    await asyncio.gather(*jobs)
    for tag in sorted(stats):
        print(f"  {tag}: " + "  ".join(f"{k}={v}" for k, v in sorted(stats[tag].items())))

    outdir = REPO / "data"
    for arm in ARMS:
        for rep in range(1, args.reps + 1):
            rows = kept.get(f"{arm}_rep{rep}", [])
            p = outdir / f"swap_{arm}_rep{rep}.jsonl"
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            npos = sum(1 for r in rows if r["labels"] == pos)
            print(f"wrote {p.name}: {len(rows)} rows ({npos} pos / {len(rows) - npos} neg)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=4, help="independent replications per arm")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
