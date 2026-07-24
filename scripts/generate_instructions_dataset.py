#!/usr/bin/env python
"""Generate an instruction-following concept dataset with Llama-3.3-70B via OpenRouter.

Prompts ``meta-llama/Llama-3.3-70B-Instruct`` to author short two-turn
(user → assistant) conversations for the *instruction-following* concept used by
the ``eval_instructions/`` probe splits. The generated rows use the exact same
schema as those files (and as ``configs/llama70b_instructions_llama1b.md``), so
they can be concatenated / used as a training or eval split:

    {"inputs": <json-string of [{role, content}, ...]>,
     "labels": "assistant_follows_the_instruction"
             | "assistant_does_not_follow_the_instruction"}

Only the conversation (user/assistant turns) and the label matter; no other
fields are emitted.

The concept is a property of the **assistant's reply**, not the topic: a reply
is ``assistant_follows_the_instruction`` if it actually does what the user asked
(the task, format, constraints, and scope), and
``assistant_does_not_follow_the_instruction`` if it refuses, violates a stated
constraint, drifts off the requested context, contradicts the source it was told
to rely on, omits requested content, or answers a different question than asked —
even if the reply still sounds fluent and cooperative. To teach this we one-shot
the model with a hand-written contrastive pair — the *same* user instruction
answered once by following it and once by not following it.

By default this produces a balanced 50-example set (25 per label); use
``--n-per-label`` to change it (e.g. ``--n-per-label 50`` for 100 total).

Reuses this repo's OpenRouter client factory (``openrouter_client``), so it
needs ``OPENROUTER_API_KEY`` (plus the optional ``OPENROUTER_BASE_URL`` /
``OPENROUTER_HTTP_REFERER`` / ``OPENROUTER_APP_TITLE``) set in the environment.

Example:
    ${REPO_ROOT}/.venv_claude/bin/python scripts/generate_instructions_dataset.py \
        --out data/instructions_llama70b_50.jsonl --n-per-label 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``src/`` importable so we can reuse the repo's OpenRouter client.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_redteam.openrouter_client import (  # noqa: E402
    extract_openrouter_error,
    make_sync_client,
)

MODEL = "meta-llama/Llama-3.3-70B-Instruct"

POS_LABEL = "assistant_follows_the_instruction"
NEG_LABEL = "assistant_does_not_follow_the_instruction"

# --- Hand-written one-shot examples (NOT drawn from eval_instructions/, to avoid
# train/eval contamination) --------------------------------------------------
# A contrastive pair: the SAME user instruction answered once by following it and
# once by not following it, so the model learns the label depends on whether the
# assistant's reply actually obeys the instruction.
ONE_SHOT = {
    POS_LABEL: {
        "user": "List three benefits of regular exercise. Answer in exactly three bullet points and nothing else.",
        "assistant": "- Improves cardiovascular health and endurance\n- Reduces stress and improves mood\n- Strengthens muscles and bones",
    },
    NEG_LABEL: {
        "user": "List three benefits of regular exercise. Answer in exactly three bullet points and nothing else.",
        "assistant": "Exercise is wonderful for you! There are honestly countless benefits — it helps your heart, your mood, your sleep, your metabolism, and so much more. Let me tell you a bit about each one in detail...",
    },
}

LABEL_GUIDANCE = {
    POS_LABEL: (
        "The assistant's reply FOLLOWS the user's instruction: it actually does "
        "what was asked — the requested task, and any stated format, length, "
        "scope, or source constraints — accurately and completely. It stays on "
        "the exact thing the user asked for."
    ),
    NEG_LABEL: (
        "The assistant's reply does NOT follow the user's instruction. It fails "
        "in one clear way: it refuses or deflects; violates a stated constraint "
        "(wrong format, wrong length, extra or missing content); drifts to a "
        "different topic or context than the one requested; contradicts or "
        "ignores information the user explicitly told it to rely on; omits part "
        "of what was requested; or answers a different question than the one "
        "asked. The reply may still sound fluent, confident, and cooperative — "
        "that is exactly what makes the failure subtle."
    ),
}


def build_prompt(label: str, batch_size: int, avoid: list[str]) -> str:
    """Build the user prompt asking for ``batch_size`` examples of one label."""
    shot = ONE_SHOT[label]
    example_obj = {"user": shot["user"], "assistant": shot["assistant"]}
    avoid_block = ""
    if avoid:
        joined = "\n".join(f"- {t}" for t in avoid)
        avoid_block = (
            "\n\nDo NOT reuse or lightly reword these already-used instructions; "
            "pick clearly different tasks:\n" + joined
        )
    return (
        f"You are helping build a labelled dataset for a classifier that detects "
        f"whether an AI assistant's reply follows the user's instruction.\n\n"
        f"{LABEL_GUIDANCE[label]}\n\n"
        f"Here is one labelled example of the '{label}' class:\n"
        f"{json.dumps(example_obj, ensure_ascii=False)}\n\n"
        f"Write {batch_size} NEW and DIVERSE '{label}' examples. Each is a short "
        f"two-message conversation: one realistic 'user' message that gives a "
        f"clear instruction (often with a format, length, scope, or provided-"
        f"source constraint), and one 'assistant' reply. Vary the instruction "
        f"types widely (formatting/length constraints, summarization, extraction, "
        f"answering strictly from a provided passage, step-by-step tasks, "
        f"translation, list vs prose, yes/no-only answers, staying on one topic, "
        f"etc.). Keep each message to 1-4 sentences (a provided passage may be a "
        f"bit longer). For the '{label}' class, make sure every reply clearly "
        f"belongs to that class.{avoid_block}\n\n"
        f"Respond with ONLY a JSON array of objects, each exactly:\n"
        f'{{"user": "...", "assistant": "..."}}\n'
        f"No prose, no markdown fences."
    )


def extract_json_array(text: str) -> list[dict]:
    """Parse a JSON array of objects from the model reply, tolerating fences."""
    t = text.strip()
    if t.startswith("```"):
        # strip ```json ... ``` fences
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON array found in reply: {text[:200]!r}")
    return json.loads(t[start : end + 1])


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


def generate_for_label(
    client, label: str, n: int, batch_size: int, temperature: float, max_tokens: int
) -> list[dict]:
    """Generate ``n`` unique rows for one label, batching until we have enough."""
    rows: list[dict] = []
    seen_users: set[str] = set()
    attempts = 0
    max_attempts = 2 * (n // max(batch_size, 1) + 2)
    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        want = min(batch_size, n - len(rows))
        prompt = build_prompt(label, want, avoid=sorted(seen_users)[:20])
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not getattr(resp, "choices", None):
            err = extract_openrouter_error(resp) or "no choices in response"
            print(f"  [warn] {label} batch {attempts}: {err}", file=sys.stderr)
            continue
        content = resp.choices[0].message.content or ""
        try:
            items = extract_json_array(content)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  [warn] {label} batch {attempts}: parse failed: {exc}", file=sys.stderr)
            continue
        for item in items:
            if not isinstance(item, dict) or "user" not in item or "assistant" not in item:
                continue
            key = str(item["user"]).strip().lower()
            if not key or key in seen_users:
                continue
            seen_users.add(key)
            rows.append(to_row(item, label))
            if len(rows) >= n:
                break
        print(f"  {label}: {len(rows)}/{n} after batch {attempts}", file=sys.stderr)
    if len(rows) < n:
        print(f"  [warn] {label}: only produced {len(rows)}/{n}", file=sys.stderr)
    return rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "instructions_llama70b.jsonl",
        help="Output JSONL path (default: data/instructions_llama70b.jsonl).",
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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    client = make_sync_client()

    all_rows: list[dict] = []
    for label in (POS_LABEL, NEG_LABEL):
        print(f"Generating {args.n_per_label} '{label}' examples...", file=sys.stderr)
        all_rows.extend(
            generate_for_label(
                client,
                label,
                args.n_per_label,
                args.batch_size,
                args.temperature,
                args.max_tokens,
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
