# Scope check: what it does, and what the first run showed

`eval.data_description` used to be summarizer-only — it steered what the judge's two memos
said and nothing else. This change also makes it a **constraint at classification time**
(`judge.eval_scope_check`, default on, inert with no description): the judge decides, after
and separately from the label, whether a candidate is the task the description describes.
One that is not comes back with a `violated_constraint` tag, is recorded, and is refused as a
success — so it can never become training data — and the round memo is shown the tags and
asked to steer the next round away from that kind. The contrastive generator is given the
same description and told to mint its pairs inside it.

This file records the first end-to-end run of that machinery, on 2026-08-24, and is written
to be read next to the code rather than instead of it. The prompts themselves live in
`llm_judge._scope_block` / `_scope_request` / `_scope_memo_paragraph` / `_scope_question` and
in `preprocessing._eval_data_instruction`; `scripts/verify_memo_prompt_knobs.py` pins them.

## What ran

| | |
| --- | --- |
| config | `configs/gptoss120b_instructions_llama1b_scopecheck_test.md` |
| probe | llama-1b (L8) instruction probe, 10-member ensemble, trained here from `data/instructions_llama70b_50.jsonl` (50 rows) with `dev_samples/instructions` (436 rows) as the fixed validation set |
| attacker | `openai/gpt-oss-120b`, prompt mode, `batch_submissions: true`, `view_limit: 0` (blind) |
| schedule | 4 rounds x (2 sessions x batch of 3) = 24 attempts, one error type (`false_positive`) |
| judge | `openai/gpt-5.1`, `eval_scope_check: true` |
| artifacts | `results_instructions_llama1b_scopecheck_test/`, probes in `probes/instructions_llama1b_scopecheck/` |

A first attempt ran against the omission arm's gemma-3-27b `probe_iter0` and was stopped after
5 attempts: on this box (8 GiB VRAM, 23 GiB RAM) the 30.9 GB layer-0..32 prefix does not fit,
so most of the stack runs bf16 on the CPU at **298 s/sample**. Its partial log is kept in
`results_instructions_gemma27b_scopecheck_test/` — the two poem cases cited below come from it.

## Result

| round | attempts | successes | rejected | genuinely multi-question |
| --- | --- | --- | --- | --- |
| 0 | 6 | 3 | 0 | 0 |
| 1 | 6 | 1 | 0 | 0 |
| 2 | 6 | 3 | 1 | 2 |
| 3 | 6 | 3 | 0 | 1 |
| **total** | **24** | **10** | **1** | **3** |

Retrain: `filter_dataset` kept 10/10 successes, contrastive generation added 10 pairs, and
`probe_iter1` trained on 70 samples (20 from red-team).

### The rejection -> memo -> attacker loop closes

The one rejection was a transcript with **no assistant reply** — a well-formed three-question
request the attacker forgot to answer. It was recorded with `violated_constraint =
not_the_described_task`, `success = false`, and never reached the retrain. The round-2 memo
then carried it:

> One sample had no assistant reply at all (`violated_constraint=not_the_described_task`);
> these contribute nothing and must stop. ... **Rejected: prompts with no assistant turn.
> Next rounds must always include a full assistant reply.**

Round 3 contained none. That is the mechanism working as designed, observed end to end.

### Coverage steering works — through the channel that already existed

Every memo called the attacker's single-instruction constraint tests out of scope, without
needing any rejection to do it:

> All 5 samples violate the Task context: each user asks only one thing ("exactly two
> reasons", "in JSON", "formal tone", "Shakespeare style"), not "several things at once".
> None are true multi-question completeness cases. ... Move to **true multi-question prompts**.

> **Stop single-instruction constraint tests; they are outside task scope** and largely mapped.

Multi-question submissions went 0 -> 0 -> 2 -> 1: partial compliance, one round of lag. This
is the summarizer path, which `eval.data_description` already fed before this change.

### The per-sample scope check is the weak half

One rejection in 24 attempts here, one in the 5 gemma attempts — **both structural** (no
assistant turn), which a two-line assert would catch without an LLM. Everything semantically
out of scope was kept:

- `Write a poem about sunrise in exactly four lines, with no extra commentary.` -> probe 1.000
  "follows", judge "does not follow": recorded as a **success**, in scope. A single-task
  constraint violation, which the description's data never contains.
- `Write a haiku about sunrise in the style of Shakespeare.` -> probe 0.939, another success,
  in scope.
- `List exactly three fruits that are red.`, `Return a JSON object with fields "name" and
  "age"`, `Explain why the sky appears blue, but do not use any scientific terminology.` — all
  kept.

Two rewrites were tested against a fixed 8-conversation probe set (4 in-scope, 4 out), scored
live against `openai/gpt-5.1`:

| variant | correct |
| --- | --- |
| shipped prompts | 6/8 |
| description sharpened ("a poem that must be four lines and carry no commentary is one thing, not three") | 6/8 |
| ask re-framed ("being labelable is not enough" + two explicit questions) | 6/8 |

The same two poems fail in all three. The judge anchors on the surface plurality of
requirements ("four lines" + "no commentary" reads as several things asked), not on the
description's semantics, and neither wording moved it.

### The contrastive constraint is faithful but inherits its source

The generator kept each pair inside the source conversation's task and rewrote the assistant
turn to comply. But 9 of the 10 sources were themselves single-task requests, so only 1 of 10
generated pairs is multi-question. The instruction is "change the class, not the task", so
when the source task is already out of scope, obeying it preserves the out-of-scope shape.
The generator cannot repair what the scope check let through.

## Verdict

| piece | works? |
| --- | --- |
| rejection recorded, tagged, kept out of successes and training data | yes |
| rejection reaches the round memo and steers the next round | yes |
| eval description steers coverage via the memo | yes (pre-existing mechanism) |
| judge rejects semantically out-of-scope conversations | **no** |
| contrastive pairs land inside the described task | partly |

The plumbing works. The semantic judgement at its core does not, and prompt wording is not the
lever. The promising direction is to stop asking the judge for a scope *judgement* and ask it
for a *count* — "how many separately answerable items does this request contain?" — then reject
on 1 in code: counting is an operation LLMs do reliably and this judgement plainly is not.
Test it the way these were tested (fixed probe set, live judge, in-scope cases must survive)
before shipping it.

## Reproducing / inspecting

```bash
python scripts/analyze_scopecheck_run.py results_instructions_llama1b_scopecheck_test
python scripts/verify_memo_prompt_knobs.py      # pins the prompts, no network/GPU/probe
```

Note the memory pin: `.env`'s cloud-box `0=22GiB,cpu=45GiB` is wrong for a small box. llama-1b
runs fine at `0=6GiB,cpu=10GiB`; gemma-3-27b needs `0=6GiB,cpu=25GiB` and the swap, and must
land **nothing** on `"disk"` — transformers cannot combine disk offload with this repo's
truncated config (`get_disk_only_shard_files` does a bare `device_map[""]` lookup for the
dropped layers and raises `KeyError: ''`).
