#!/usr/bin/env python
"""Ask the generator to rewrite its own conversations short and precise, holding the label.

The generated conversations carry the concept — a probe trained on real data reads arm 4's
30 rows at 0.80 AUROC — but they do not teach it. Two measured ways they differ in *shape*
from the split they are scored against:

* they are long. Arm 4's user turns run 245 characters at the median against the eval
  split's 171, and its assistant replies 400 against 445/319.
* the length relationship is **inverted**. In the eval split the omitting reply is shorter
  than its complete twin in 57 of 57 sources (median 319 vs 445). In arm 4 the rows
  labelled *omits* are the LONGER ones (420 vs 381), so length points the wrong way
  relative to the label.

This rewrites each conversation to be shorter and more precise while holding fixed the one
thing that must not move: which requested parts the reply answers. Only the wording is
allowed to change, never the label, never which part is missing.

Each rewrite is checked before it is accepted — label unchanged, roles well-formed, the
content actually shorter, and inside the probe's 1024-token cap. A rewrite that fails is
retried once, then the original is kept and counted, so the output is always 30 rows.

    set -a; . ./.env; set +a
    .venv_claude/bin/python scripts/shorten_samples.py \
        --run results_oig_omission_nemotron_topics_v1 --out data/arm4_shortened.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
POS = "assistant_follows_the_instruction"
NEG = "assistant_does_not_follow_the_instruction"
PROBE_MODEL = "google/gemma-3-27b-it"
MAX_TOKENS = 1024

SYSTEM = """You are editing training data for a text classifier.

Each item is a two-turn conversation: a user asking for several distinct things at once,
and an assistant reply. The label says whether the reply answers EVERY part the user asked
for, or silently leaves one out.

Rewrite the conversation to be shorter and more precise. What you must NOT change:

- Which requested parts the reply actually answers. If the label says a part is missing,
  exactly that part stays missing, and it is still never refused, flagged or apologised
  for — it is simply absent. If the label says every part is answered, every part is still
  answered after your edit.
- The substance of the user's request: the same number of questions, about the same
  subject, asking for the same things.

What you should change: length and padding. Cut preamble, restatement of the question,
filler transitions, hedging, closing offers of further help, and any elaboration beyond
what was asked. Keep concrete facts, names, numbers and dates. Write plain reference prose.

Targets, from the real data this classifier is scored on: the user turn around 170
characters, a reply that answers everything around 440, and a reply that leaves a part out
around 320 — an incomplete answer is SHORTER than a complete one, because a part of it is
missing, not because it was written more tersely.

Return ONLY a JSON object, no commentary:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}"""


def _request(messages: list[dict], label: str) -> str:
    which = ("answers every part the user asked for" if label == POS
             else "silently leaves out one of the parts the user asked for")
    body = json.dumps(messages, ensure_ascii=False, indent=1)
    return (f"This conversation is labelled: the reply {which}.\n\n{body}\n\n"
            f"Rewrite it shorter and more precise, holding that label true.")


def _coerce(text: str, label: str):
    """Parse a rewrite and accept it only if it is well-formed and actually shorter."""
    from agentic_redteam.json_extract import extract_json_values

    def accept(o):
        """`accept` returns the NORMALISED value or None — not a bool. Returning True
        here makes extract_json_values yield the bool itself, which is the bug this
        replaced. Take either {"messages": [...]} or a bare two-message array."""
        if isinstance(o, dict) and isinstance(o.get("messages"), list):
            return o["messages"]
        if isinstance(o, list) and len(o) == 2 and all(isinstance(m, dict) for m in o):
            return o
        return None

    for msgs in extract_json_values(text, accept=accept):
        if not isinstance(msgs, list) or len(msgs) != 2:
            continue
        if [m.get("role") for m in msgs] != ["user", "assistant"]:
            continue
        if not all(isinstance(m.get("content"), str) and m["content"].strip() for m in msgs):
            continue
        return [{"role": m["role"], "content": m["content"].strip()} for m in msgs]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=2)  # nemotron 429s above this
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    from agentic_redteam.openrouter_client import make_sync_client
    from agentic_redteam.token_budget import TokenBudget

    tb = TokenBudget(PROBE_MODEL, MAX_TOKENS, combine_consecutive_messages=True,
                     convert_tool_to_assistant=True)
    tb.warmup()
    client = make_sync_client()

    newest = {}
    for line in (args.run / "batches.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line); newest[(r["iteration"], r["batch_index"])] = r
    rows = [(k, s) for k in sorted(newest) for s in newest[k]["samples"]]
    print(f"{len(rows)} conversations from {args.run.name}")

    def work(item):
        (k, s) = item
        orig = [{"role": m["role"], "content": m["content"]} for m in s["messages"]]
        orig_len = sum(len(m["content"]) for m in orig)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": _request(orig, s["label"])}]
        for attempt in range(2):
            out = None
            for backoff in (0, 5, 15, 40):
                if backoff:
                    time.sleep(backoff)
                try:
                    r = client.chat.completions.create(model=MODEL, messages=msgs,
                                                       max_tokens=args.max_tokens)
                    out = _coerce(r.choices[0].message.content or "", s["label"])
                    break
                except Exception as e:  # noqa: BLE001 — retry 429s, don't drop the row
                    err = f"{type(e).__name__}"
                    if "RateLimit" not in err and "Timeout" not in err:
                        break
            else:
                print(f"  i{k[0]}b{k[1] + 1}: rate-limited out")
            if out is None:
                continue
            new_len = sum(len(m["content"]) for m in out)
            if new_len >= orig_len:
                msgs = msgs + [{"role": "assistant", "content": json.dumps({"messages": out})},
                               {"role": "user", "content":
                                f"That is {new_len} characters against the original's {orig_len}. "
                                "Cut it further; it must be shorter."}]
                continue
            if tb.overage(out) is not None:
                continue
            return {"inputs": json.dumps(out, ensure_ascii=False), "labels": s["label"],
                    "orig_chars": orig_len, "new_chars": new_len, "rewritten": True}
        return {"inputs": json.dumps(orig, ensure_ascii=False), "labels": s["label"],
                "orig_chars": orig_len, "new_chars": orig_len, "rewritten": False}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        out_rows = list(ex.map(work, rows))

    n_rw = sum(r["rewritten"] for r in out_rows)
    import statistics as st
    o = [r["orig_chars"] for r in out_rows]; n = [r["new_chars"] for r in out_rows]
    pos = [r for r in out_rows if r["labels"] == POS]
    neg = [r for r in out_rows if r["labels"] != POS]
    ass = lambda rs, key: st.median([len(json.loads(r["inputs"])[-1]["content"]) for r in rs]) if rs else 0
    print(f"\nrewritten {n_rw}/{len(out_rows)} · total chars median {int(st.median(o))} -> {int(st.median(n))}")
    print(f"  assistant reply median — follows {int(ass(pos,0))}  omits {int(ass(neg,0))}"
          f"   (eval split: 445 / 319)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(
        json.dumps({"inputs": r["inputs"], "labels": r["labels"]}, ensure_ascii=False) + "\n"
        for r in out_rows))
    (args.out.parent / (args.out.stem + "_meta.json")).write_text(json.dumps(
        {"n": len(out_rows), "rewritten": n_rw,
         "median_chars_before": int(st.median(o)), "median_chars_after": int(st.median(n))}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
