#!/usr/bin/env python
"""Generate a general-vs-specialized conversation dataset with Llama-3.3-70B via OpenRouter.

Prompts ``meta-llama/Llama-3.3-70B-Instruct`` to author short two-turn
(user → assistant) conversations for a *generality* concept. The generated rows
use the exact same schema as its three sibling generators
(``generate_highstakes_dataset.py``, ``generate_hu_harm_dataset.py``,
``generate_instructions_dataset.py``), so they can be concatenated / used as a
training or eval split:

    {"inputs": <json-string of [{role, content}, ...]>,
     "labels": "general" | "not_general"}

Only the conversation (user/assistant turns) and the label matter; no other
fields are emitted.

Like the high-stakes concept — and unlike the two assistant-centric ones — this
is a property of the **conversation itself**, not of the quality of the
assistant's reply: a conversation is ``general`` if it is a broad everyday
exchange any layperson could hold (cooking, travel, weather, hobbies, small
talk, common-sense questions), and ``not_general`` if it is a narrow
domain-specific or expert exchange that presumes professional training or
insider vocabulary (kernel debugging, clinical dosing, tax code, litigation
procedure). Both sides are ordinary, safe, cooperative conversations; nothing
here is about harm, stakes, or instruction violations. The one-shot contrastive
pair therefore holds the *subject area* fixed (houseplants / plant physiology)
and varies only the depth and vocabulary, so the label cannot collapse into
"which topic is this".

By default this produces a balanced 150-example set (75 per label), which is
larger than the 50-row sibling datasets; use ``--n-per-label`` to change it.

**Message length is held constant across the two classes** — see the length-parity
note by ``ONE_SHOT``. Every row must fall inside a word band that is stated
identically in both classes' prompts and re-checked at parse time, and the run
ends with a ``length_report`` audit that exits 2 if length alone still separates
the classes. ``--check-only`` runs that audit against an existing ``--out`` file
without generating anything (and without needing an API key).

Reuses this repo's OpenRouter client factory (``openrouter_client``), so it
needs ``OPENROUTER_API_KEY`` (plus the optional ``OPENROUTER_BASE_URL`` /
``OPENROUTER_HTTP_REFERER`` / ``OPENROUTER_APP_TITLE``) set in the environment.

Example:
    ${REPO_ROOT}/.venv_claude/bin/python scripts/generate_general_dataset.py \
        --out data/general_llama70b_150.jsonl --n-per-label 75
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

POS_LABEL = "general"
NEG_LABEL = "not_general"

# --- Length parity -----------------------------------------------------------
# The FIRST cut of this dataset was perfectly separable by message length alone
# (user turns: general 30-87 chars, not_general 86-203 — a single threshold at 86
# labelled 148/150 rows correctly, length-only AUROC 1.000). A probe trained on
# that learns "long user turn ⇒ specialized" and never sees the concept.
#
# The cause was the one-shot pair below: it held the *subject* fixed to stop the
# topic leaking, but its general user turn was 46 chars against 101 for
# not_general, and the model copied that ratio into every row.
#
# A single shared word band was the first fix and it was the wrong shape: the
# two classes have very different *natural* lengths (measured on that first run,
# general user turns ran 7-16 words against 12-33 for not_general), so a band
# tight enough to force parity — 12-28 — sat above where general lands and
# rejected every general row. Parity is therefore carried by an explicit
# per-example word TARGET instead, drawn from one schedule that both classes
# share row-for-row: row k of general and row k of not_general are asked for the
# same length, so the two distributions match by construction rather than by the
# model's own sense of how long an expert question should be.
#
# The bands below are back to being a loose sanity check around that schedule
# (targets ±~6 words), sized to admit both classes' natural ranges. They exist to
# catch a runaway row, not to do the parity work. ``length_report`` remains the
# backstop that fails the run if length still separates the classes.
USER_WORDS = (6, 32)
ASSISTANT_WORDS = (14, 50)

# Shared per-row length schedule, cycled by a row's index within its label.
# Spans both classes' measured natural ranges, so neither has to strain to hit it.
USER_TARGETS = (10, 14, 18, 22, 26)
ASSISTANT_TARGETS = (18, 24, 30, 36, 42)

# --- Hand-written one-shot examples ------------------------------------------
# A contrastive pair in the SAME subject area (plants) and at the SAME length,
# one an everyday question and one an expert one, so the model learns the label
# depends on how specialized the exchange is rather than on the topic or on how
# much text it is looking at.
ONE_SHOT = {
    POS_LABEL: {
        "user": "What's a good way to keep my basil alive indoors over the winter, and how often should I be watering it?",
        "assistant": "Give it the brightest windowsill you have and water when the top inch of soil dries out — basil hates sitting wet, and pinching the tips keeps it bushy.",
    },
    NEG_LABEL: {
        "user": "Is the interveinal chlorosis on my basil a magnesium deficiency, or is it iron lockout from a soil pH of 7.8?",
        "assistant": "Magnesium is mobile, so its deficiency shows on the older leaves first — interveinal chlorosis on the new growth at pH 7.8 points to iron lockout instead.",
    },
}


def _words(text: str) -> int:
    return len(text.split())


def _in_band(text: str, band: tuple[int, int]) -> bool:
    return band[0] <= _words(text) <= band[1]


# The one-shot pair is the single strongest length signal the model sees, so an
# edit that unbalances it would silently reintroduce the leak the band checks
# below are there to catch. Tie the two together instead.
for _side in ("user", "assistant"):
    _band = USER_WORDS if _side == "user" else ASSISTANT_WORDS
    _lengths = [_words(ONE_SHOT[lab][_side]) for lab in (POS_LABEL, NEG_LABEL)]
    assert all(_band[0] <= n <= _band[1] for n in _lengths), (
        f"one-shot {_side} turns {_lengths} fall outside the advertised band {_band}"
    )
    assert abs(_lengths[0] - _lengths[1]) <= 3, (
        f"one-shot {_side} turns differ by {abs(_lengths[0] - _lengths[1])} words "
        f"({_lengths}) — keep the two classes length-matched or the model copies the ratio"
    )

LABEL_GUIDANCE = {
    POS_LABEL: (
        "The conversation is GENERAL: a broad, everyday exchange that any "
        "layperson could hold without special training — casual questions, "
        "common-sense advice, hobbies, household matters, small talk, popular "
        "culture, travel, food, plans, opinions. The vocabulary is ordinary and "
        "the answer draws on widely shared knowledge."
    ),
    NEG_LABEL: (
        "The conversation is NOT GENERAL: a narrow, domain-specific or expert "
        "exchange that presumes professional training, technical vocabulary, or "
        "insider context — a specialist asking a specialist. Think jargon, "
        "specific standards, parameters, procedures, or named tools that a "
        "layperson would not recognise. It is still an ordinary, safe, helpful "
        "conversation; it is simply specialized."
    ),
}


def length_schedule(start_index: int, batch_size: int) -> list[tuple[int, int]]:
    """The (user, assistant) word targets for rows ``start_index .. +batch_size``.

    Keyed on the row's index within its own label, so row k of one class is asked
    for exactly the same lengths as row k of the other — which is what makes the
    two length distributions match without a band tight enough to reject a class.
    """
    return [
        (
            USER_TARGETS[(start_index + i) % len(USER_TARGETS)],
            ASSISTANT_TARGETS[(start_index + i) % len(ASSISTANT_TARGETS)],
        )
        for i in range(batch_size)
    ]


def build_prompt(
    label: str, batch_size: int, avoid: list[str], start_index: int = 0
) -> str:
    """Build the user prompt asking for ``batch_size`` examples of one label."""
    shot = ONE_SHOT[label]
    example_obj = {"user": shot["user"], "assistant": shot["assistant"]}
    other = NEG_LABEL if label == POS_LABEL else POS_LABEL
    schedule_block = "\n".join(
        f"  Example {i + 1}: user ~{u} words, assistant ~{a} words"
        for i, (u, a) in enumerate(length_schedule(start_index, batch_size))
    )
    avoid_block = ""
    if avoid:
        joined = "\n".join(f"- {t}" for t in avoid)
        avoid_block = (
            "\n\nDo NOT reuse or lightly reword these already-used scenarios; "
            "pick clearly different topics:\n" + joined
        )
    return (
        f"You are helping build a labelled dataset for a classifier that detects "
        f"whether a conversation with an AI assistant is a general everyday "
        f"exchange or a specialized domain-specific one.\n\n"
        f"{LABEL_GUIDANCE[label]}\n\n"
        f"Here is one labelled example of the '{label}' class:\n"
        f"{json.dumps(example_obj, ensure_ascii=False)}\n\n"
        f"Write {batch_size} NEW and DIVERSE '{label}' examples. Each is a short "
        f"two-message conversation: one realistic 'user' message and one "
        f"'assistant' reply. Vary the subject areas widely (health, money, food, "
        f"travel, technology, home and repairs, work, law, science, sport, "
        f"animals, education, arts, etc.)."
        f"\n\nLENGTH RULE — write each example to the word count assigned to it "
        f"below, within a word or two:\n{schedule_block}\n"
        f"These exact targets are used for every class in this dataset, so a "
        f"reader must not be able to guess the label from message length alone: a "
        f"specialized exchange is not longer than an everyday one, it is only more "
        f"technical. Do not pad an example to sound expert, and do not clip one to "
        f"sound casual — write the everyday questions out in full and keep the "
        f"technical ones to the length assigned."
        f"\n\nEvery subject area must be able to appear on BOTH sides of this "
        f"concept, so do not let the topic give the label away: a '{label}' "
        f"example should differ from a '{other}' one only in how specialized the "
        f"question and answer are, not in what they are about. Keep both sides "
        f"benign — no harm, no emergencies, no high-consequence decisions, and "
        f"the assistant simply answers the question normally.{avoid_block}"
        f"\n\nRespond with ONLY a JSON array of objects, each exactly:\n"
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
    # Insertion-ordered copy of ``seen_users``: at 75 rows per label the avoid
    # list is shown for ~8+ batches, and a sorted slice would pin it to the same
    # alphabetically-first scenarios all run. The most RECENT openers are what
    # the model is actually about to repeat.
    recent_users: list[str] = []
    attempts = 0
    # Roomier than the 2x the sibling generators use: out-of-band rows are
    # discarded here, so a batch no longer contributes ``batch_size`` rows.
    max_attempts = 4 * (n // max(batch_size, 1) + 2)
    n_out_of_band = 0
    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        want = min(batch_size, n - len(rows))
        prompt = build_prompt(
            label, want, avoid=recent_users[-20:], start_index=len(rows)
        )
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
            # A loose runaway check only — the parity work is done by the shared
            # target schedule in the prompt, and audited by ``length_report``.
            # Keep this band wide enough to admit both classes' natural lengths:
            # a band narrow enough to force parity rejects the shorter class
            # outright (12-28 words dropped every 'general' row it ever saw).
            if not _in_band(str(item["user"]), USER_WORDS) or not _in_band(
                str(item["assistant"]), ASSISTANT_WORDS
            ):
                n_out_of_band += 1
                continue
            seen_users.add(key)
            recent_users.append(key)
            rows.append(to_row(item, label))
            if len(rows) >= n:
                break
        print(
            f"  {label}: {len(rows)}/{n} after batch {attempts} "
            f"({n_out_of_band} dropped out-of-band)",
            file=sys.stderr,
        )
    if len(rows) < n:
        print(f"  [warn] {label}: only produced {len(rows)}/{n}", file=sys.stderr)
    return rows[:n]


def _auroc(pos: list[float], neg: list[float]) -> float:
    """Rank-based AUROC (Mann-Whitney), ties at 0.5. Kept dependency-free."""
    if not pos or not neg:
        return float("nan")
    wins = sum(
        1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg
    )
    return wins / (len(pos) * len(neg))


def length_report(rows: list[dict], max_length_auroc: float) -> bool:
    """Print the length-separability audit. Returns True if the set is clean.

    The first cut of this dataset scored 1.000 here — a probe trained on it would
    have learned message length. This is the check that would have caught it.
    """
    by_label: dict[str, list[tuple[int, int]]] = {POS_LABEL: [], NEG_LABEL: []}
    for row in rows:
        msgs = json.loads(row["inputs"])
        by_label[row["labels"]].append(
            (_words(msgs[0]["content"]), _words(msgs[1]["content"]))
        )

    print("\nLength audit (words):", file=sys.stderr)
    for i, side in ((0, "user"), (1, "assistant")):
        band = USER_WORDS if i == 0 else ASSISTANT_WORDS
        print(f"  {side} turn (band {band[0]}-{band[1]}):", file=sys.stderr)
        for label in (POS_LABEL, NEG_LABEL):
            vals = [v[i] for v in by_label[label]]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            print(
                f"    {label:<12} n={len(vals):<4} mean={mean:5.1f} "
                f"min={min(vals):<3} max={max(vals)}",
                file=sys.stderr,
            )

    ok = True
    for i, side in ((0, "user"), (1, "assistant")):
        auroc = _auroc(
            [float(v[i]) for v in by_label[NEG_LABEL]],
            [float(v[i]) for v in by_label[POS_LABEL]],
        )
        # 0.5 = length carries no signal; 1.0 = length alone separates the classes.
        skew = abs(auroc - 0.5) + 0.5
        verdict = "ok" if skew <= max_length_auroc else "LEAK"
        print(
            f"  {side}-length-only AUROC: {auroc:.3f} (separability {skew:.3f}, "
            f"limit {max_length_auroc:.3f}) [{verdict}]",
            file=sys.stderr,
        )
        ok = ok and skew <= max_length_auroc
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "general_llama70b_150.jsonl",
        help="Output JSONL path (default: data/general_llama70b_150.jsonl).",
    )
    parser.add_argument(
        "--n-per-label",
        type=int,
        default=75,
        help="Examples per label (default 75 → 150 total, balanced).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Examples requested per LLM call (default 10).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-length-auroc",
        type=float,
        default=0.65,
        help=(
            "Fail (exit 2) if message length alone separates the two classes "
            "above this, where 0.5 is no signal and 1.0 is perfect separation "
            "(default 0.65). The file is still written, for inspection."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Audit the existing --out file for length separability and exit; generate nothing.",
    )
    args = parser.parse_args()

    if args.check_only:
        rows = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"Auditing {len(rows)} rows from {args.out}", file=sys.stderr)
        if not length_report(rows, args.max_length_auroc):
            print("\n[FAIL] length separates the classes — see above.", file=sys.stderr)
            raise SystemExit(2)
        print("\n[ok] length carries no usable signal.", file=sys.stderr)
        return

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

    # Audited after writing, not before: a leaky set is worth inspecting rather
    # than discarding, and the exit code is what makes the failure noticeable.
    if not length_report(all_rows, args.max_length_auroc):
        print(
            "\n[FAIL] message length alone separates the classes — a probe trained "
            "on this would learn length, not the concept. The file was written; "
            "regenerate before using it.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print("\n[ok] length carries no usable signal.", file=sys.stderr)


if __name__ == "__main__":
    main()
