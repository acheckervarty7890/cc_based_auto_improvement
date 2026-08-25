# Scope check on gemma-3-27b: seven arms, and what actually fixed it

`docs/scope_check_run_2026-08-24.md` tested `judge.eval_scope_check` on a llama-1b probe at a
narrowed schedule and concluded the plumbing works while the semantic judgement at its core does
not. This file records seven arms run on 2026-08-25 against the probe it was always meant to
attack — the omission arm's 10-member gemma-3-27b (L32) `probe_iter0` — at experiment24_cloud's
own schedule, one retrain each.

The short version: the scope check *can* be improved a lot by prompt work, and it does not
matter much. What fixed the problem was showing the attacker what the eval data looks like,
which no amount of filtering substitutes for.

## Common setup

| | |
| --- | --- |
| probe | `probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl` — gemma-3-27b-it L32, 10-member ensemble |
| attacker | `openai/gpt-oss-120b`, prompt mode, `batch_submissions: true`, `view_limit: 0` (blind) |
| schedule | `sessions_per_model 3 x max_turns 5` = 15 attempts/round, 5 rounds, one error type (`false_positive`) |
| judge | `openai/gpt-5.1`, `eval_scope_check: true` |
| retrain | one (`--iterations 1`), base `data/instructions_llama70b_50.jsonl` (50 rows), dev `dev_samples/instructions` (436) |

Scoring ran at ~1.1 s/sample on a 24 GiB / 62 GiB box; eval and dev activations came from the
Kaggle blobs, so no LLM was loaded for either. Each red-team phase took under 5 minutes.

**"On-task" is an independent measurement, not the judge's verdict.** Every first user turn was
scored by a separate `openai/gpt-5.1` pass asking only *how many separately answerable items does
this request contain*, with formatting constraints explicitly excluded from the count. The
description's task is "asks more than one question", so a 1-item request is off-task.

## The arms

| arm | config suffix | what changed |
| --- | --- | --- |
| A | `_exp24` | baseline; the shipped prompts |
| B | `_scopefix` | "number of parts" clause removed from 3 prompts; description reworded ("things") |
| C | `_questions` | as B, description says "questions" throughout |
| D | `_tellattacker` | as C + **attacker shown the description** (`show_eval_data_description`) |
| E | `_armA_told` | arm A's *original* description + attacker shown it |
| F | `_nocue_told` | hybrid description, attacker shown it, **no** surface-cue sentence |
| G | `_cue_told` | as F **with** the surface-cue sentence |

## Result

| | A | B | C | D | E | F | G |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attempts | 71 | 75 | 73 | 75 | 75 | 73 | 73 |
| **on-task submissions** | 32% | 53% | 27% | **100%** | 90% | 98% | 98% |
| on-task, round 0 only | 20% | 0% | 7% | **100%** | **100%** | **100%** | **100%** |
| scope rejections | 3 | 10 | 13 | 8 | 10 | 9 | 12 |
| recall on off-task-with-a-reply | 1/47 | 3/35 | 11/52 | — | 0/7 | 0/1 | 0/1 |
| false rejects | 0 | 0 | 0 | 0 | **1** | 0 | 0 |
| finds | 44 | 18 | 30 | 11 | 18 | 14 | 9 |
| success rate (raw) | 62% | 24% | 41% | 15% | 24% | 19% | 12% |
| mean probe score of finds | 0.977 | 0.902 | 0.901 | 0.700 | 0.865 | 0.707 | 0.705 |
| finds at probe >= 0.99 | 82% | 56% | 57% | 9% | 39% | **0%** | **0%** |
| red-team samples trained on | 88 | 36 | 60 | 22 | 36 | 26 | 18 |
| mean eval AUROC delta | -0.030 | +0.002 | +0.004 | **+0.041** | -0.002 | +0.011 | -0.020 |

## 1. The clause: a real fix to the check, worth little

The check rejected 3 of 71 in arm A, catching 1 of 47 off-task conversations that had a reply.
The cause was a clause in **three** prompts — `llm_judge._scope_block` (classification SYSTEM
prompt, which said the number of parts "is never grounds for rejection"),
`llm_judge._scope_request`, and `preprocessing._eval_data_instruction` — telling the judge and
the generator that *how many parts the request has* is among the things a description leaves
open. For a description whose task **is** "asks more than one question", the count is
constitutive. The prompts were licensing exactly what the check exists to reject. Any one left
in place overrides the other two, so they were removed together.

Recall went **2% -> 9% -> 21%** across A/B/C, with zero on-task conversations wrongly rejected in
any of the three. Semantic (non-structural) catches went 1 -> 3 -> 11.

It did not help the data. Arm C's attacker produced *more* off-task material than the baseline
(73% vs 68%), so at 21% recall only 4 of its 30 finds were the described task. Memo steering
spiked and collapsed in every arm — on-task per round ran `20/36/69/20/21`, `0/87/100/53/27` and
`7/27/93/7/0` — with arm C ending at zero. **A better filter downstream of a blind generator
does not produce better training data.**

## 2. What actually fixed it: tell the attacker

`eval.data_description` was judge-side only. It reached the attacker laundered through a round
memo, and only from round 1, so **round 0 was always written with no idea what the eval data
looks like** — 20% / 0% / 7% on-task in A/B/C. `attacker.show_eval_data_description` (default
off) puts it in the attacker's system prompt under `## Probe under attack`.

Four arms, four descriptions, one result: **on-task goes to 90-100%, and round 0 to 100% in all
four.** Arm E is the sharpest test — it uses arm A's *own* loose description, the one containing
"how many things are asked ... are all free to vary", and simply showing it takes on-task from
32% to 90%. The 68-73% off-task rate was never a description-quality problem or an
attacker-capability problem. It was an information problem.

The scope check then has almost nothing left to do: **all 8 of arm D's rejections are transcripts
with no assistant reply**, every one a multi-question request the attacker forgot to answer. A
two-line `any(m["role"] == "assistant")` assert would catch them without an LLM.

## 3. The finds change kind, not just count

The raw success rate falls 62% -> 12-24%, and that fall is the improvement. The arms are not
finding the same thing:

- **Arm A's finds are constraint violations.** *"Translate into French and German, give only the
  translations"* -> assistant translates correctly and adds an English note. Probe 1.000. The
  failure is added text, not a dropped question — a failure mode the description explicitly
  excludes ("never a property of correctness", label turns only on completeness).
- **Arm D/F/G's are genuine omissions.** *"1. Chemical symbol for gold? 2. Who painted the Mona
  Lisa? 3. What year did the Titanic sink? 4. ..."* -> answers 1, 2, 4, silently drops 3. Probe
  0.506.

82% of arm A's finds sit at probe >= 0.99 (maximally confident and wrong); **0%** of F's and G's
do. Arm A's 62% was one easy out-of-distribution seam mined 44 times.

**A caveat on this repo's own metric.** The item count measures the *request*. It does not check
the description's second clause — that the **label must turn on completeness**. By that clause
arm A's constraint-violation finds are off-task too, so its 32% on-task figure is generous and
the real gap is wider than the table shows.

## 4. The surface cue (F vs G) does nothing useful

G adds one sentence naming the surface cue ("the incomplete reply is usually the shorter one").
F and G are otherwise byte-identical. They match on everything that sentence was meant to move:
99% on-task both, 100% round 0 both, mean probe 0.707 vs 0.705, 0 finds at >= 0.99 both.

The only differences are that G found **fewer** (9 vs 14) and wrote **longer** replies. Mean
assistant reply length in finds runs 116 -> 344 -> 853 -> 1287 -> 1523 chars across A/D/E/F/G:
told the incomplete reply is usually shorter, the attacker wrote longer ones, spending its budget
on elaborate conversations rather than on the boundary.

## 5. The eval cannot separate any of this

Mean AUROC delta across the seven arms: **-0.030 / +0.002 / +0.004 / +0.041 / -0.002 / +0.011 /
-0.020**. Four arms with 90-100% on-task data land at +0.041, -0.002, +0.011 and -0.020. That
spread, on essentially identical data quality and at retrain sizes of 22 / 36 / 26 / 18 red-team
samples, is noise.

`oig_omission` — the split the description names — falls in six of seven arms. Arm D is the one
that rises (+0.027), and with F and G at -0.046 and -0.074 on better-matched data, that is a coin
flip rather than a finding. Section 6 confirms it directly: re-running arm D's own config and code
put `oig_omission` at 0.765 against arm D's 0.824 — a 0.059 swing from the red-team draw alone,
larger than any arm-to-arm difference in this table.

**So arm D's +0.041 is not established.** The on-task and probe-score results are large,
consistent and replicated four times; the eval column is not.

## 6. Follow-up: it was the pairs all along

Seven arms varied what the attacker and judge were told and moved the eval by nothing
interpretable. A second pass varied what happens to the finds *after* they are collected, and
that did move it.

**The re-draw sets the noise floor.** Arm D was re-run at `--iterations 5`, same config, same
code: 75 attempts per iteration, 16 successes across five (8/4/2/0/2), retraining on the
cumulative set each time. Its iteration-1 `oig_omission` is **0.765** where arm D's was **0.824**.
Nothing differs but the draw. Every delta in section 5 is smaller than that.

**Leave-one-out over the 16 couples.** Refitting the probe 16 times, each time holding out one
couple (both halves together), and scoring `oig_omission`:

| training set | oig_omission |
| --- | --- |
| base only (50 rows) | 0.797 |
| base + all 16 couples | 0.750 |
| base + the 8 couples LOO called helpful | 0.818 |
| base + the 8 couples LOO called harmful | 0.727 |

The sign split is 8/8, though one couple on the harmful side has a delta of 1e-16 — numerically
no effect at all, so read it as 8 helping / 7 hurting / 1 inert.

The red-team data as collected **cost** 0.047. Selection recovers it and a little more, but
selection is fitted on the split it is scored on, so 0.818 is an upper bound, not a result. What
the study did give was a mechanism: the couples that hurt had generated replies 2-3x longer than
the find they came from. The pair was teaching length.

**A minimal-edit generation prompt** (`GENERATION_PROMPT_VERSION = 2`) replaced version 1's
"generate a similar-looking conversation of the other class" with "produce the smallest edit of
this conversation that flips the label", and made the copy-through explicit. Regenerating the same
16 finds with the same generator, changing only the prompt:

- mean find/partner assistant-length ratio **0.62 -> 0.90** (1.00 = same length)
- user turn preserved verbatim in **16/16** pairs
- `oig_omission` **0.750 -> 0.843**, beating the cherry-picked best-8 with no selection at all

**Which half carries it.** Refitting on one side of the couples only, everything else held:

| training set | rows | pos/neg | oig_omission |
| --- | --- | --- | --- |
| base only | 50 | 25/25 | 0.797 |
| + the 16 finds alone | 66 | 25/41 | **0.689** |
| + the 16 v1 partners alone | 66 | 41/25 | 0.779 |
| + the 16 v2 partners alone | 66 | 41/25 | 0.774 |
| + both halves, v1 | 82 | 41/41 | 0.750 |
| + both halves, v2 | 82 | 41/41 | **0.843** |

Neither half helps alone, and the finds alone are the single most damaging thing in this whole
document (-0.108). The v1 and v2 partners are indistinguishable *individually* (0.779 vs 0.774),
so the prompt's entire +0.093 comes from how the partner relates to its source. The v2 couple
beats its own better half by +0.069: superadditive, which is the mechanism contrastive pairs are
supposed to have and version 1 did not deliver. Caveat: the single-sided fits are class-skewed
(41/25 on a 66-row set), and the base is only 50 rows, so 16 additions is a 32% perturbation.

**Arms F and G re-run under v2** (`_nocue_told_v2`, `_cue_told_v2`) both moved `oig_omission` up,
+0.051 and +0.042. Read those as corroboration only — a full re-run re-draws the red-team phase,
so the finds differ as well as the pairs, and the re-draw alone is worth 0.059. The clean result
is the held-finds one above.

## Verdict

| claim | status |
| --- | --- |
| the clause caused the check to under-reject | yes — recall 2% -> 21%, zero false-reject cost |
| a better check yields better training data | **no** — arm C had the best recall and the worst data |
| showing the attacker the description fixes coverage | **yes** — 4/4 arms, 90-100%, round 0 at 100% |
| it needs the sharpened wording to work | no — arm E gets 90% on arm A's own loose description |
| naming the surface cue helps | no — F vs G identical, G found fewer |
| better data yields a better probe | **unproven at n=1 per arm** |
| minimal-edit contrastive pairs yield a better probe | **yes** — +0.093 on `oig_omission`, finds held fixed |
| the red-team finds themselves help | **no** — alone they cost 0.108; the value is in the couple |

## What to do next

1. **Repeats, not variants.** Seven arms sit between -0.030 and +0.041 with no relationship to
   data quality, and section 6 measures the draw-to-draw noise at 0.059. Three reseeded runs of
   one configuration would settle whether the retrain effect exists; another description variant
   would not.
2. **Reject no-assistant-turn transcripts in code.** It is now 8 of 8 rejections in arm D and
   needs no LLM.
3. **Fix the on-task metric** to check the label mechanism, not just the request's item count —
   the constraint-violation cases are currently counted as on-task.
4. Consider whether `eval_scope_check` earns its cost once the attacker is told. On arms D-G it
   catches only what an assert would.
5. **The single-variable F/G control**: hold F's and G's existing finds fixed and regenerate only
   their pairs under v2. That is the design that produced the clean +0.093 and it removes the
   re-draw confound from the F2/G2 numbers.
6. **Audit the other pair-generation paths** for the same failure. The length artifact was
   invisible in every eval column until it was measured directly.

## Follow-up

`docs/what_limits_the_instruction_probe_2026-08-25.md` takes the v3 run's 33 couples and asks why
red-team data costs this probe the split it is aimed at. Short version: the achievable AUROC on
`oig_omission` is **0.914**, volume / selection / validation are all ruled out, the attacker's
**finds** carry the damage while the generated partners do not — and the whole effect reverses
once the training set contains in-distribution data.

## Reproducing / inspecting

```bash
.venv_claude/bin/python scripts/analyze_scopecheck_run.py results_instructions_gemma27b_scopecheck_exp24_tellattacker
.venv_claude/bin/python scripts/verify_memo_prompt_knobs.py    # pins the prompts, no network/GPU/probe
```

Section 6 in order (each reads the cached activations; only the first needs the network):

```bash
.venv_claude/bin/python scripts/regenerate_pairs_promptv2.py   # same 16 finds, v2 prompt -> pairs_promptv2.jsonl
.venv_claude/bin/python scripts/pair_selection_study.py        # leave-one-out over the 16 couples
.venv_claude/bin/python scripts/pair_selection_controls.py     # base-only / helpful-8 / harmful-8 controls
.venv_claude/bin/python scripts/score_pairs_promptv2.py        # base + the 16 v2 couples, all 7 splits
.venv_claude/bin/python scripts/pair_half_ablation.py          # each half of the couples on its own
```

Each arm's `_item_counts.json` holds the per-attempt item counts the tables above are computed
from, alongside its JSONL, memos and comparison CSV.
