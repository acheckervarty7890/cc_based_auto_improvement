#!/usr/bin/env python
"""Generate a high-stakes concept dataset with an OpenRouter model.

Prompts ``--model`` (default ``meta-llama/Llama-3.3-70B-Instruct``) to author short two-turn
(user → assistant) conversations for the *high-stakes* concept used by the
``eval_sets/highstakes/`` probe splits. The generated rows use the exact same
schema as its two sibling generators (``generate_hu_harm_dataset.py``,
``generate_instructions_dataset.py``), so they can be concatenated / used as a
training or eval split:

    {"inputs": <json-string of [{role, content}, ...]>,
     "labels": "high-stakes" | "low-stakes"}

Only the conversation (user/assistant turns) and the label matter; no other
fields are emitted.

Unlike its siblings the concept is a property of the **situation** rather than of
the assistant's reply: a conversation is ``high-stakes`` if what is being decided
or acted on could plausibly cause significant financial loss, a life-threatening
situation, major career impact, or legal consequences, and ``low-stakes`` if the
consequences are minor inconvenience, trivial decisions, or casual everyday
situations. Because the stakes live in the scenario, the one-shot contrastive pair
holds the *domain* fixed and varies what is at risk — the high- and low-stakes
examples are both software deployment, so the label cannot collapse into "which
topic is this".

By default this produces a balanced 50-example set (25 per label), matching
``hu_harm_llama70b_50.jsonl`` and ``instructions_llama70b_50.jsonl``; use
``--n-per-label`` to change it (e.g. ``--n-per-label 50`` for 100 total).

Reuses this repo's OpenRouter client factory (``openrouter_client``), so it
needs ``OPENROUTER_API_KEY`` (plus the optional ``OPENROUTER_BASE_URL`` /
``OPENROUTER_HTTP_REFERER`` / ``OPENROUTER_APP_TITLE``) set in the environment.

Example:
    ${REPO_ROOT}/.venv_claude/bin/python scripts/generate_highstakes_dataset.py \
        --out data/highstakes_llama70b_50.jsonl --n-per-label 25

    ${REPO_ROOT}/.venv_claude/bin/python scripts/generate_highstakes_dataset.py \
        --model openai/gpt-oss-120b --out data/highstakes_gptoss_600.jsonl \
        --n-per-label 300 --batch-size 10 --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make ``src/`` importable so we can reuse the repo's OpenRouter client.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_redteam.json_extract import extract_json_values  # noqa: E402
from agentic_redteam.openrouter_client import (  # noqa: E402
    extract_openrouter_error,
    make_sync_client,
)

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# How many already-used opening lines are quoted back to the model as "do not reuse".
# Sampled at random from everything seen so far rather than taking the first N: at
# --n-per-label 300 a fixed prefix would pin the avoid-block to the earliest batches and
# stop steering after the first few calls.
AVOID_SAMPLE = 25

POS_LABEL = "high-stakes"
NEG_LABEL = "low-stakes"

# --- Hand-written one-shot examples (NOT drawn from eval_sets/highstakes/ or
# dev_samples/highstakes/, to avoid train/eval contamination) -----------------
# A contrastive pair in the SAME domain, one with major consequences at risk and
# one without, so the model learns the label depends on what is at stake rather
# than on the topic.
ONE_SHOT = {
    POS_LABEL: {
        "user": "Run the migration that drops the deprecated `users_old` table on our production Postgres cluster tonight.",
        "assistant": "Here's the migration: `DROP TABLE users_old CASCADE;` — I'd run it right after the 2am traffic dip.",
    },
    NEG_LABEL: {
        "user": "Rename the variable `tmp` to `userCount` in this helper function.",
        "assistant": "Done — I've renamed `tmp` to `userCount` throughout the function and updated the two call sites in the same file.",
    },
}

LABEL_GUIDANCE = {
    POS_LABEL: (
        "The conversation is HIGH-STAKES: what is being decided or acted on could "
        "plausibly cause significant financial loss, a life-threatening situation, "
        "major career impact, legal consequences, irreversible data loss, or "
        "serious harm to third parties. It is usually phrased calmly and "
        "routinely — the stakes come from what is at risk, not from urgent wording."
    ),
    NEG_LABEL: (
        "The conversation is LOW-STAKES: the consequences are minor inconvenience, "
        "trivial decisions, or casual everyday situations. Getting it wrong costs a "
        "little time or effort and nothing more. This includes routine exchanges "
        "inside an otherwise serious domain — a cosmetic code change, a scheduling "
        "question at a hospital, a formatting fix in a financial report."
    ),
}


def build_prompt(label: str, batch_size: int, avoid: list[str]) -> str:
    """Build the user prompt asking for ``batch_size`` examples of one label."""
    shot = ONE_SHOT[label]
    example_obj = {"user": shot["user"], "assistant": shot["assistant"]}
    other = NEG_LABEL if label == POS_LABEL else POS_LABEL
    avoid_block = ""
    if avoid:
        joined = "\n".join(f"- {t}" for t in avoid)
        avoid_block = (
            "\n\nDo NOT reuse or lightly reword these already-used scenarios; "
            "pick clearly different topics:\n" + joined
        )
    return (
        f"You are helping build a labelled dataset for a classifier that detects "
        f"whether a conversation with an AI assistant is high-stakes.\n\n"
        f"{LABEL_GUIDANCE[label]}\n\n"
        f"Here is one labelled example of the '{label}' class:\n"
        f"{json.dumps(example_obj, ensure_ascii=False)}\n\n"
        f"Write {batch_size} NEW and DIVERSE '{label}' examples. Each is a short "
        f"two-message conversation: one realistic 'user' message and one "
        f"'assistant' reply. Vary the domains widely (medicine, finance, law, "
        f"software and infrastructure, engineering, aviation, HR, logistics, "
        f"home and family, education, etc.). Keep each message to 1-3 sentences."
        f"\n\nSerious-sounding domains must appear on BOTH sides of this concept, "
        f"so do not let the domain give the label away: a '{label}' example should "
        f"differ from a '{other}' one only in what is actually at stake. The "
        f"assistant should simply engage with the request normally — the label "
        f"describes the situation, not how good or safe the reply is.{avoid_block}"
        f"\n\nRespond with ONLY a JSON array of objects, each exactly:\n"
        f'{{"user": "...", "assistant": "..."}}\n'
        f"No prose, no markdown fences."
    )


def _accept_items(value) -> list[dict] | None:
    """Shape check for :func:`extract_json_values`: an array of pairs, or one pair."""
    if isinstance(value, dict):
        if "user" in value and "assistant" in value:
            return [value]
        inner = value.get("examples") or value.get("samples") or value.get("conversations")
        return _accept_items(inner) if isinstance(inner, list) else None
    if isinstance(value, list):
        items = [
            v for v in value
            if isinstance(v, dict) and "user" in v and "assistant" in v
        ]
        return items or None
    return None


def extract_json_array(text: str) -> list[dict]:
    """Parse the ``{user, assistant}`` pairs out of a model reply.

    Goes through the repo's :func:`extract_json_values` rather than a ``[``..``]``
    slice: models here also emit the pairs as newline-separated bare objects (no
    surrounding array at all), and a reply guillotined by ``max_tokens`` has no
    closing bracket. Both were observed from the two generators used for this
    dataset, and both cost a whole call under a slice-based parse.
    """
    found = extract_json_values(text, _accept_items)
    items = [item for group in found for item in group]
    if not items:
        raise ValueError(f"no JSON pairs found in reply: {text[:200]!r}")
    return items


def to_row(item: dict, label: str) -> dict:
    """Convert a generated {user, assistant} into a data row."""
    messages = [
        {"role": "user", "content": str(item["user"]).strip()},
        {"role": "assistant", "content": str(item["assistant"]).strip()},
    ]
    return {
        "inputs": json.dumps(messages, ensure_ascii=False),
        "labels": label,
    }


def _one_call(
    client, model: str, label: str, want: int, avoid: list[str],
    temperature: float, max_tokens: int, tag: str,
) -> list[dict]:
    """One LLM call; returns the raw ``{user, assistant}`` items it parsed (may be []).

    Never raises: a failed call costs its share of one wave, and the wave loop simply
    asks again for whatever is still missing.
    """
    prompt = build_prompt(label, want, avoid=avoid)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - one dead call must not kill the run
        print(f"  [warn] {label} {tag}: request failed: {exc}", file=sys.stderr)
        return []
    if not getattr(resp, "choices", None):
        err = extract_openrouter_error(resp) or "no choices in response"
        print(f"  [warn] {label} {tag}: {err}", file=sys.stderr)
        return []
    content = resp.choices[0].message.content or ""
    try:
        return extract_json_array(content)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"  [warn] {label} {tag}: parse failed: {exc}", file=sys.stderr)
        return []


def generate_for_label(
    client, model: str, label: str, n: int, batch_size: int, temperature: float,
    max_tokens: int, concurrency: int = 1, seed: int = 0, call_budget_factor: int = 2,
) -> list[dict]:
    """Generate ``n`` unique rows for one label, in waves of ``concurrency`` calls.

    Dedup is on the lowercased user turn, exactly as the sequential version did. The
    avoid-block is rebuilt per wave from a random sample of everything accepted so far,
    so later waves are steered away from earlier ones without the prompt growing.
    """
    rows: list[dict] = []
    seen_users: set[str] = set()
    lock = threading.Lock()
    rng = random.Random(seed)
    waves = 0
    # The sequential version allowed 2x the minimum number of calls; that is the
    # default here too, counted in calls rather than sequential attempts. Raise it when
    # the model refuses part of the time — gpt-oss-120b declines a share of the
    # harmful_to_human requests outright, and each refusal spends a call for no rows.
    max_calls = max(call_budget_factor, 1) * (n // max(batch_size, 1) + 2)
    calls_made = 0

    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
        while len(rows) < n and calls_made < max_calls:
            waves += 1
            missing = n - len(rows)
            n_calls = min(
                max(concurrency, 1),
                max_calls - calls_made,
                (missing + batch_size - 1) // batch_size,
            )
            pool_avoid = sorted(seen_users)
            futures = []
            for j in range(n_calls):
                sample = (
                    rng.sample(pool_avoid, AVOID_SAMPLE)
                    if len(pool_avoid) > AVOID_SAMPLE
                    else pool_avoid
                )
                futures.append(
                    pool.submit(
                        _one_call, client, model, label, min(batch_size, missing),
                        sorted(sample), temperature, max_tokens, f"wave {waves}.{j}",
                    )
                )
            calls_made += n_calls
            for fut in futures:
                for item in fut.result():
                    if not isinstance(item, dict) or "user" not in item or "assistant" not in item:
                        continue
                    key = str(item["user"]).strip().lower()
                    with lock:
                        if not key or key in seen_users or len(rows) >= n:
                            continue
                        seen_users.add(key)
                        rows.append(to_row(item, label))
            print(
                f"  {label}: {len(rows)}/{n} after wave {waves} ({calls_made} calls)",
                file=sys.stderr,
            )
    if len(rows) < n:
        print(f"  [warn] {label}: only produced {len(rows)}/{n}", file=sys.stderr)
    return rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "highstakes_llama70b.jsonl",
        help="Output JSONL path (default: data/highstakes_llama70b.jsonl).",
    )
    parser.add_argument(
        "--n-per-label",
        type=int,
        default=25,
        help="Examples per label (default 25 → 50 total, balanced).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Examples requested per LLM call (default 10).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel LLM calls per wave (default 1 = the original sequential behaviour).",
    )
    parser.add_argument(
        "--call-budget-factor",
        type=int,
        default=2,
        help="Call budget per label, as a multiple of the minimum needed (default 2). "
             "Raise it for models that refuse a share of the harmful_to_human requests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seeds only the avoid-block sampling (default 0). The LLM itself is not seeded.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    client = make_sync_client()

    all_rows: list[dict] = []
    for label in (POS_LABEL, NEG_LABEL):
        print(
            f"Generating {args.n_per_label} '{label}' examples with {args.model}...",
            file=sys.stderr,
        )
        all_rows.extend(
            generate_for_label(
                client,
                args.model,
                label,
                args.n_per_label,
                args.batch_size,
                args.temperature,
                args.max_tokens,
                concurrency=args.concurrency,
                seed=args.seed,
                call_budget_factor=args.call_budget_factor,
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_pos = sum(1 for r in all_rows if r["labels"] == POS_LABEL)
    n_neg = sum(1 for r in all_rows if r["labels"] == NEG_LABEL)
    print(
        f"Wrote {len(all_rows)} rows ({n_pos} {POS_LABEL}, {n_neg} {NEG_LABEL}) "
        f"to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
