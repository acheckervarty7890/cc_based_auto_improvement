# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`agentic_redteam` is an agentic red-teaming and iterative-retraining toolkit
for [tuberlens](https://github.com/blandfort/tuberlens) activation probes. The
attacker and judge can each be driven by one of two providers, picked
per-section in the config:

- **`claude_sdk`** — Anthropic's [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
  (attacker) and Anthropic Python SDK (judge). Tools are exposed to the
  attacker via an in-process MCP server.
- **`openrouter`** — the official `openai` SDK pointed at
  [OpenRouter](https://openrouter.ai/), giving access to Claude, GPT, Gemini,
  Llama, Mistral, DeepSeek, etc. through one OpenAI-compatible endpoint. The
  attacker uses native OpenAI tool calls; the judge uses chat completions.
  No MCP machinery is used on this path.

Providers can be mixed within a single attacker rotation (e.g. one
`claude_sdk` model and several `openrouter` models in the same run), and the
two SDKs are imported lazily so a config that only uses one provider does not
need the other installed.

The end-to-end loop:

1. Load a pickled tuberlens probe.
2. Run an **attacker** in a submit-and-refine loop. It submits candidate
   conversations and reads back probe + judge verdicts to refine its strategy
   in-context. Whichever driver runs it, the attacker gets no shell, no
   filesystem and no web — only the three red-team tools (or, under
   `interface: prompt`, no tools at all). See `attacker.py` for the four
   drivers.
3. Each candidate is scored by the probe **and** independently classified by a
   Claude-based **human-style judge** (the judge picks one of the two class
   labels on the conversation's own merits, with no hint about what we are
   hoping for). A candidate counts as a successful red-team find only if the
   probe's predicted label and the judge's label **disagree** in the direction
   matching the configured `error_type` — e.g. for `error_type=false_positive`,
   the probe must predict the positive class and the judge must pick the
   negative class.
4. Every attempt is appended to a JSONL log.
5. With `attacker.round_summaries` enabled (the default), rounds run
   **sequentially**: at the end of each round the **judge** reads all of that
   round's attempts (successful and not) and folds them into a single **rolling
   strategy memo** — it rewrites and condenses the prior memo rather than
   appending, so the memo stays bounded no matter how many rounds run (the
   budget is derived in `llm_judge.py`). That memo is injected into the system prompt of every later round's
   attacker, which is always shown it and can still call `view_past_attempts`
   for specific conversations. The memo resets per iteration (and per error type).
6. With `attacker.cross_iteration_memos` enabled (default **off**), a second,
   **cross-iteration** memo bridges that reset: after each iteration's rotation
   finishes — and before the retrain — the judge writes a hand-off memo covering
   what was tried, what succeeded (and is therefore about to be trained against,
   so it should be treated as *patched*), and what remains unexplored. It is
   injected into the next iteration's attacker system prompts and rewritten
   (not appended) each iteration, so it stays bounded.
7. The retraining script converts JSONL successes into a tuberlens
   `LabelledDataset` — each sample labelled with the **judge's predicted
   class** (the judge is the source of truth; `error_type` is only used as a
   fallback for old rows missing `judge_label`). Optionally concatenates with
   a base training dataset, then trains a fresh probe with the same
   architecture and metadata as the original.

## Environment

The project's venv lives at:

```
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/
```

`tuberlens` is installed into it as an editable checkout under
`.venv_claude/src/tuberlens/`.
The other key packages: `anthropic` and `claude_agent_sdk` (used when
`provider: claude_sdk`), `openai` (used when `provider: openrouter` —
points at OpenRouter via `base_url`), `pyyaml` (config parser).

**Always invoke the venv's Python by absolute path** to avoid burning permission
prompts on `source .venv_claude/bin/activate` — the venv interpreter has its
`site-packages` baked into its own `sys.path`, so `source` adds nothing here:

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python -c "..."
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/pip install ...
```

Required environment variables:

- `ANTHROPIC_API_KEY` — only when any section uses `provider: claude_sdk`.
- `OPENROUTER_API_KEY` — only when any section uses `provider: openrouter`.
- Optional: `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`),
  `OPENROUTER_HTTP_REFERER`, `OPENROUTER_APP_TITLE` (sent as `HTTP-Referer` /
  `X-Title` for OpenRouter dashboard attribution).
- Optional circuit-breaker tuning (see `circuit_breaker.py`):
  `OPENROUTER_MAX_CONSECUTIVE_ERRORS` (default 10),
  `OPENROUTER_MAX_CONSECUTIVE_FATAL_ERRORS` (default 3),
  `OPENROUTER_MAX_CONNECTION_OUTAGE_S` (default 1800 — how long the network may
  stay down before the run aborts) and `OPENROUTER_CONNECTION_BACKOFF_S`
  (default `60,120,480` — retry intervals while it is down).
- Optional: `OPENROUTER_TIMEOUT_S` (default 60) — per-request wall-clock cap.
- Optional activation-extraction tuning (see `model_loading.py`):
  `AGENTIC_REDTEAM_TRUNCATE_LAYERS` (default on — load only layers `0..probe.layer`;
  set `0` to load the full model), `AGENTIC_REDTEAM_MAX_MEMORY`
  (e.g. `"0=21GiB,cpu=45GiB"` — pins accelerate's per-device budget so it can't fall
  back to disk offload; unset by default) and tuberlens' own `BATCH_SIZE` (default 1),
  which drives both `get_activations` and the red-team chunking in `retrain`.

## Common commands

Install in editable mode (re-run after dependency changes):

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/pip install -e .
```

Smoke-test imports:

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python -c \
  "import agentic_redteam; from agentic_redteam.attacker import run_redteam; print('ok')"
```

One round of red-teaming:

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python \
  scripts/run_redteam.py configs/example_config.md
```

Full iterative loop (train initial probe → red-team → retrain → optional eval, n times):

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python \
  scripts/iterative_retrain.py configs/example_config.md \
  --iterations 3 --base-training-data path/to/base.jsonl \
  --eval --eval-dataset-dir eval_datasets   # --eval is optional
```

`--base-training-data` is **required**: it trains the initial probe (unless
`probe.path` warm-starts from an existing one) and is concatenated with red-team
successes on every retrain. With no `probe.path`, the first probe is trained from
scratch using the `probe:` fields (`model`, `layer`, `pos_class_label`,
`neg_class_label`, `architecture`). `--eval` additionally scores the initial probe
and every retrained probe on the local eval splits and writes a cross-round
comparison CSV.

Validation is always derived by splitting the training data via this repo's
`stable_train_test_split` (`--test-size`, default 0.2; optional `--split-field`) —
there is no external validation-file flag. **The split is content-deterministic, not
RNG-based**: each sample's train-vs-val side is a pure function of its own content
(or its `split_field` value) plus `--seed`, so the base samples land identically
every iteration even as red-team successes accumulate. `--seed` (default 42) seeds
that split and the (reproducible) eval subsampling; `--eval-max-samples`
(default 100, `0` = full split) sets the balanced subsample size per eval split.
`--base-data-fraction` (default 1.0, range (0, 1]) ingests only a random fraction
of the **base** training data — selected by the same content-deterministic hash
(`stable_fraction_subsample`, namespaced `frac:{seed}` so it's independent of the
train/val split) and applied *before* the split, so the chosen subset is identical
every iteration, preserves class balance in expectation, and is folded into the
base activation cache key. Red-team successes are never subsampled.
When the config has a `preprocessing:` section, red-team successes are run through
`filter_dataset` + `generate_contrastive_dataset` before each retrain.

Because the base train/val split is fixed across iterations, the base training
split's activations are cached to disk (`--base-activation-cache-dir` flag, or
`output.base_activation_cache_dir` in config; default
`<probe-out-dir>/base_activation_cache`) and computed **once for the whole run** —
the initial training populates the cache and every retrain reuses it. The growing
red-team set is also cached in the same dir, but **per conversation** (the set
changes every iteration, so a whole-set blob like the base one would never hit):
a success first seen in iteration k is forwarded once and reused by every later
retrain, so each retrain only computes its *newly-seen* successes.
`--[no-]combine-consecutive-messages` /
`--[no-]convert-tool-to-assistant` (or the config `eval:` knobs) apply to **both the
training data and the eval splits**, so the probe trains and is scored on the same
message representation.

## Architecture

### `agentic_redteam/config.py`
Parses one markdown file with YAML frontmatter for runtime knobs and `# Attacker` /
`# Judge` sections for system prompts. Resolves all paths relative to the config
file. Frontmatter shape (see `configs/example_config.md` and
`configs/example_config_openrouter.md`):

```yaml
attacker:
  provider: claude_sdk | openrouter   # default provider for bare-string models
  models:
    - <bare-string>                   # inherits attacker.provider
    - {name: <model>, provider: claude_sdk | openrouter}  # per-model override
  max_turns: int
  batch_target: int
  rounds: int                         # fresh LLM sessions per model (default 1)
  concurrency: int                    # max parallel attacker sessions (default 1)
  sessions_per_model: int             # concurrent copies of EACH model launched within each round
                                      #   (default 1). >1 parallelizes the same model without duplicating
                                      #   it in `models` and without turning off round_summaries; still
                                      #   bounded by `concurrency`. All copies share the JsonlStore (dedup)
                                      #   and write with the same round number, so their attempts fold into
                                      #   that round's summary.
  persistence_from_last_rounds: int   # view_past_attempts window (default: all)
  view_reshuffle: bool                # view_past_attempts: periodic random reshuffle on/off (default true).
                                      #   when false, show most-recent success/fail attempts (recency),
                                      #   and use training seeds only as a fallback for the successful half
  view_reshuffle_interval: int        # view_past_attempts: redraw every N submissions (default 20; reshuffle=true only)
  view_balance: bool                  # view_past_attempts: ≈50/50 success/fail, total=limit (default true)
  view_training_seeds: bool           # view_past_attempts: blend true-class training examples (default true)
  near_dup_guard: bool                # default FALSE. Submit-time clone guard: a candidate whose first
                                      #   user turn is >= near_dup_threshold similar to any already-recorded
                                      #   SUCCESS is rejected BEFORE probe/judge run, so re-skinned winning
                                      #   templates are never scored or stored. Orthogonal to the view_* knobs.
  near_dup_threshold: float           # difflib ratio the guard rejects at (default 0.8; >= 1.0 disables)
  near_dup_broadcast: bool            # default FALSE. Show guard-rejected openers to ALL sessions as an
                                      #   "avoid these" prompt block. In-memory only, never written to JSONL.
  round_summaries: bool               # default true → rounds run SEQUENTIALLY; after each finished round the
                                      #   judge folds it into one bounded ROLLING memo (rewritten + condensed,
                                      #   not appended), injected into later rounds' attacker system prompts.
                                      #   false → legacy fully-concurrent scheduling, no memo. Models within a
                                      #   round are concurrent either way.
  cross_iteration_memos: bool         # default FALSE. true → after each iteration's rotation (before the
                                      #   retrain) the judge writes a hand-off memo — what was tried, what
                                      #   succeeded and is about to be trained against (⇒ treat as patched),
                                      #   what's unexplored — injected into the NEXT iteration's attacker
                                      #   system prompts. Persisted to <jsonl>.iteration_memos.jsonl, which
                                      #   is re-read at run start, so it crosses both the iteration boundary
                                      #   and a process restart (--resume). Independent of round_summaries.
  cross_iteration_memo_max_successes: int  # successes (most recent) shown to the judge when writing that
                                      #   memo (default 30; 0 = all — can make the judge prompt huge)
  interface: tools | prompt           # how the attacker is driven (default tools). "prompt" = classical
                                      #   no-tool prompting: the model gets NO tools; instead get_probe_info
                                      #   is baked into the system prompt and view_past_attempts is injected
                                      #   as text after every submission, and the model must output ONE
                                      #   candidate conversation per turn (fenced ```json array of {role,
                                      #   content}) which is scored through the same probe+judge path. Only
                                      #   supported for openrouter models — load_config raises if any model
                                      #   resolves to claude_sdk under interface: prompt.
  view_limit: int                     # prompt mode only: size of the view_past_attempts sample injected each
                                      #   turn (default 10). Mirrors the tools-mode view_past_attempts limit.
                                      #   <= 0 means inject NOTHING (note this is the opposite of
                                      #   ViewSampler.sample, where limit <= 0 means unlimited).
  capture_prompts: bool               # prompt mode only, default FALSE. Dump the verbatim message array of
                                      #   every API call to <jsonl>.prompts.jsonl (PromptTraceStore). Grows
                                      #   much faster than the JSONL.
  batch_submissions: bool             # prompt mode only, default FALSE. Ask for ALL `max_turns` candidate
                                      #   conversations in ONE reply instead of one per turn: the session
                                      #   makes a single API call, every conversation is scored, and it ends
                                      #   — the attacker never sees a probe/judge verdict. load_config RAISES
                                      #   if combined with interface: tools. See _run_openrouter_prompt_batch_model.
judge:
  provider: claude_sdk | openrouter
  model: <model>
  max_tokens: int                     # also caps the rolling memo's word budget
                                      #   (min(200, max(150, 0.45 × max_tokens)) — the
                                      #   200-word target governs at any max_tokens ≥ 512)
  hide_opposite_direction: bool       # default TRUE. Withhold misclassifications pointing the
                                      #   OTHER way from error_type (a false positive turned up
                                      #   during a false_negative hunt) from the round memo, so
                                      #   the judge doesn't write up weaknesses that are
                                      #   unactionable this rotation. Rows probe+judge AGREED on
                                      #   are kept. Affects summarize_round only.
probe:
  path: <path>                        # OPTIONAL: warm-start from an existing probe.
                                      # If omitted/missing, iterative_retrain_main trains
                                      # the first probe from --base-training-data using
                                      # the fields below.
  threshold: float
  error_type: false_positive | false_negative | [false_positive, false_negative]
  model: <tuberlens model key>        # from-scratch only (e.g. llama-1b)
  layer: int                          # from-scratch only
  pos_class_label: <str>              # from-scratch only; also loads base_training_data
  neg_class_label: <str>              # from-scratch only
  description: <str>                  # from-scratch only (optional)
  architecture: <ProbeType name>      # from-scratch only (optional; default linear_then_softmax)
preprocessing:                        # OPTIONAL: collation-style preprocessing of red-team
  provider: claude_sdk | openrouter   # successes before each retrain
  model: <model>                      # LLM for generate_contrastive_dataset
  max_concurrent: int                 # contrastive generation fan-out (default 50)
  max_tokens: int                     # per contrastive generation (default 2048)
  filter_percentile: float            # filter_dataset keep-threshold (default 0.8)
  assistant_centric: bool             # default false; true → prompt says the label is set by
                                      #   the assistant's reply, so edit the assistant turns
  concept_description: str            # OPTIONAL free text: what the concept IS. Injected into
                                      #   the generation prompt for both directions.
  label_guidance:                     # OPTIONAL {class label: free text}; shown when generating
    <pos_class_label>: |              #   TOWARD that label (keys are the raw probe labels; the
      - ...                           #   LABEL_SHORT alias is also accepted). Unknown keys are
    <neg_class_label>: |              #   warned about and ignored.
      - ...
eval:                                 # OPTIONAL: dataset message transforms applied to BOTH
  combine_consecutive_messages: bool  # training data AND eval splits (default false) — merge
  convert_tool_to_assistant: bool     # adjacent same-role msgs; rewrite tool→assistant (first)
  eval_max_samples: int               # balanced subsample per eval split (0 = full split). Unset (None)
                                      #   → the CLI's --eval-max-samples default; the flag overrides.
kaggle:                               # OPTIONAL: pull PRECOMPUTED eval activations from Kaggle
  owner: <kaggle username>            #   instead of extracting them (see kaggle_activations.py).
  eval_dataset_slug: <template>       #   slug + file templates, formatted with `split=<split stem>`
  eval_file_name: <template>          #   e.g. "{split}gemmaevalpt" / "{split}-gemmaeval.pt".
                                      #   Requires eval.eval_max_samples: 0 (validated at parse time).
output:   { jsonl_path, run_id,
            comparison_csv,             # OPTIONAL eval-output path (--eval); CLI --comparison-csv overrides
            activations_cache_dir,      # OPTIONAL eval activation cache (--eval); CLI --activations-cache-dir overrides
            base_activation_cache_dir } # OPTIONAL training (base-split) activation cache; CLI --base-activation-cache-dir overrides
```

Each attacker model entry can be a bare string (inherits `attacker.provider`)
or a dict `{name, provider}` to mix providers in one rotation. This is
represented at runtime as a list of `AttackerModel(name, provider)` —
`config.attacker.models` is **not** a list of strings; use `.model_names` if
you only need the names.

`error_type` drives everything downstream: it's both the target misprediction the
attacker is searching for and the implicit *true* class label (`negative` for
`false_positive`, `positive` for `false_negative`) that the judge confirms.
When `error_type` is a list (e.g. `[false_positive, false_negative]`), the CLI
runs red-teaming for each error type sequentially within every iteration, writing
to separate JSONL files (auto-suffixed `_fp` / `_fn`). The iterative retrain loop
is **interleaved**: each iteration attacks with all error types, then retrains
once on combined successes from all JSONL files.

`rounds` controls how many fresh LLM sessions each model gets per error type.
Each round is a new conversation context with up to `max_turns` tool calls.
`persistence_from_last_rounds` limits `view_past_attempts` to records from the
N most recent rounds (default: all rounds visible).

### `agentic_redteam/persistence.py`
`Conversation` (frozen tuple of `Message`s) and `JsonlStore`. The store dedups by
canonical text on append (no duplicate row for the same conversation), and
preloads any prior records on init so re-running against the same JSONL keeps
the success counter and dedup set warm. `append` returns **True if the row was
newly persisted**, False if it was a duplicate — `tools.py` uses that to
attribute the row to the submitting session. Each row carries
`{sample, probe_score, probe_predicts_positive, judge_label, judge_reason,
judge_confidence, success, attacker_model, run_id, round, iteration, error_type,
pos_class_label, neg_class_label}` — `judge_label` is the class label the judge
picked (human-readable, e.g. "high-stakes"), or `""` if the judge response was
unparseable, and `judge_confidence` is 1–10 (0 when missing/unparseable; it
feeds `retrain_probe(min_judge_confidence=)`, which the CLI supplies from
`judge.confidence_threshold`). `iteration` is the 0-based retrain-cycle index (the CLI threads it
through `run_redteam(..., iteration=)` → `ToolContext.current_iteration`); rows
written before this field existed read back as `-1`. Note `round` is the
*global* round number (`iteration * rounds + round_idx`), so `iteration` is now
explicit rather than only recoverable as `round // rounds`.

**Near-duplicate guard (`attacker.near_dup_guard`).** Exact-text dedup misses a
*re-skinned* winning template — same opening scenario, swapped nouns and numbers —
which is how a rotation talks itself into submitting one success fifty times. So
the store also compares the first `_NEAR_DUP_PREFIX` (600) chars of a candidate's
first user turn (`first_user_text`) against every persisted **success** by
`difflib.SequenceMatcher` ratio. Successes only: near-duplicates of past *failures*
don't inflate the clone rate, and blocking them would needlessly narrow exploration.

- `try_reserve_opener(conversation, threshold) → bool` is the form callers use. It
  checks the candidate against persisted successes **union the openers currently
  being scored**, and reserves it on success — one synchronous method with no
  `await` inside, so asyncio can't interleave two sessions between the check and
  the reserve. Callers **must** pair a True return with `release_opener` in a
  `finally`. `near_duplicate_success` is the same test without the in-flight set,
  and is racy under concurrency by construction.
- `_is_near` passes **`autojunk=False`**, which is load-bearing: difflib's autojunk
  heuristic (on above 200 chars) derives its junk set from the *second* argument, so
  `ratio(a,b) != ratio(b,a)`. At our ~250–400-char openers the guard's
  candidate-first order under-measured a genuine near-duplicate as 0.30 instead of
  0.84 and almost never fired. Disabling it also makes the guard measure exactly what
  `scripts/clone_rate.py` scores offline.
- `record_near_dup_reject` / `recent_near_dup_rejects(limit)` back
  `near_dup_broadcast`: an in-memory ring (most recent 200) of rejected openers,
  shared across the rotation so a rejection in **any** session steers every session.
  Deliberately **never persisted** — it is a within-run steering signal, and writing
  it would pollute both the scored-attempts dataset and the clone metric.

A `threshold >= 1.0` disables the guard. Both knobs default off, so existing configs
behave identically.

Also hosts `JsonlStore.records_for_round(round_num)` (all attempts for one global
round, used to summarize it), `JsonlStore.records_for_iteration(iteration,
only_successful=False)` (all attempts of one retrain cycle, used to write the
cross-iteration memo) and the rolling-memo storage: `RoundSummary`
(`{round, iteration, error_type, text, n_attempts, n_successes}`) plus
`SummaryStore`. A `SummaryStore` is built **once per `run_redteam` call** (i.e. per
`(iteration, error_type)`), so the memo **resets per iteration**. It holds a single
rolling memo string, not a list: `update()` *replaces* `current` with each round's
condensed memo (the judge folds the new round into the prior memo — see
`LLMJudge.summarize_round(prior_summary=...)`) and, if given a `path`, appends a
per-round snapshot to a JSONL sidecar (`<jsonl>.summaries.jsonl`) for diagnostics;
`current` feeds the latest memo back into the next update; `render()` wraps it as the
"## Strategy memo from earlier rounds" system-prompt block (or `""` before the first
memo exists). Because the judge rewrites-and-condenses instead of appending, the
memo stays bounded regardless of round count — it does **not** grow linearly.
The sidecar is diagnostics-only **except on resume**: constructed with
`SummaryStore(path, iteration=, error_type=, resume=True)` it seeds `current` from the
newest sidecar row matching that `(iteration, error_type)`, so a run restarting at
round 18 opens with the memo distilled from rounds 0..17 instead of an empty one.
Rows from other iterations are ignored, preserving the per-iteration reset; without
`iteration` (or with `resume=False`, the default) no load-back happens.

Round-level resume state lives in `RoundProgress`
(`{round, iteration, error_type, n_attempts, n_successes, completed_at}`) plus
`RoundProgressStore`, a set of finished `(iteration, error_type, round)` keys backed by
`<jsonl>.rounds_done.jsonl` and reloaded on init. `mark_done()` appends; `is_done()` /
`done_rounds()` query. A round is marked **only after its rolling-memo update**, which
keeps this store and the `SummaryStore` sidecar in lockstep — N rounds done ⟺ the
newest memo covers rounds 0..N-1 — so a resumed run skips exactly the rounds whose
findings the restored memo already reflects. Rounds are always recorded; whether they
are honoured is the caller's `resume` flag.

Finally, the **cross-iteration** memo storage: `IterationMemo`
(`{iteration, error_type, text, n_attempts, n_successes}`) plus `IterationMemoStore`
(built per `run_redteam` call when `attacker.cross_iteration_memos` is on). Unlike
`SummaryStore`, this store **reads its sidecar back on init**
(`<jsonl>.iteration_memos.jsonl`) — that is what carries the memo across the
iteration boundary (each iteration is a fresh `run_redteam` call) and across a
process restart / `--resume`. `update()` appends one memo per iteration;
`prior_text(iteration)` returns the newest memo from an iteration **strictly before**
`iteration` (so re-running an interrupted iteration never feeds it its own stale
memo); `render(iteration)` wraps it as the "## Lessons from previous iterations (the
probe has since been RETRAINED)" system-prompt block, or `""` at iteration 0. The
sidecar is per error type, since the JSONL path is.

`PromptTraceStore` (`<jsonl>.prompts.jsonl`, gated on `attacker.capture_prompts`, prompt
mode only) captures the **verbatim message array of every API call** — the one thing
nothing else records. The JSONL stores the conversation a turn *produced*, not the
messages that produced it, and carries no session or turn id, so with
`sessions_per_model > 1` the concurrent sessions interleave and the in-session context of
any submission after the first is unrecoverable after the fact. One row per API call:
`{session_id, turn, round, iteration, attacker_model, error_type, messages,
response_text, parsed, submission, submission_key, result}`, where `submission_key` is
the submitted conversation's canonical text and joins the row to its `AttemptRecord`.
Under `attacker.batch_submissions` one call yields several conversations off a single
prompt, so those rows carry plural **`submissions` / `submission_keys` / `results`** and
leave the singular fields null — one row per conversation would repeat the whole message
array once per batch member. A reader has to explode the list back out; the plural keys
are simply absent on per-turn rows, so older captures still read unchanged.

### `agentic_redteam/probe_judge.py`
Wraps a pickled tuberlens probe. Lazily loads the extraction LLM on first score
(heavy import). Exposes `score(conversation) → float`,
`evaluate(conversation) → (score, predicts_positive)`, and
`label_for(predicts_positive)` / `true_class_label` for turning a prediction into
one of the probe's own class-label strings. There is deliberately **no**
"is the probe already wrong?" predicate: that can only be decided against the
judge's label, so the judge always runs (see the conventions at the end).
The probe carries `pos_class_label`, `neg_class_label`, `description`,
`model_name` and `layer` as metadata, which everything downstream reads off the
loaded object — never duplicate these in code.

The model is loaded through `model_loading.load_extraction_model` (see below), which
carries `offload_buffers=True` and truncates the model to the layers the probe
actually reads. `release()` drops the loaded LLM and runs
`gc.collect()` + `torch.cuda.empty_cache()` — call it when a phase is done with
the probe (the attacker does, after each rotation) so the next phase reloads onto
a clean GPU. See "Free GPU memory between heavy phases" in the conventions for why.

### `agentic_redteam/model_loading.py`
`load_extraction_model(model_name, layer)` — the single loader for the frozen
extraction LLM, used by both `ProbeJudge._ensure_model` (red-team scoring) and
`retrain._get_model`. Beyond `offload_buffers=True` it does the one thing that
dominates wall-clock on a gemma-sized probe: **it loads only layers `0..layer`.**

tuberlens' `HookedModel.__enter__` already truncates the *executed* stack to
`original_layers[:layer+1]` (`tuberlens/model.py:144`) — but that happens inside the
context manager, long after `from_pretrained` has placed the whole model. For
`google/gemma-3-27b-it` at layer 32 that is 29 of 62 layers, **24 GB of bf16 weights**,
downloaded, dispatched and CPU/disk-offloaded without ever running a forward. Since
`device_map="auto"` fills the GPU in layer order and spills the rest, those dead
layers are exactly what push the *executed* tail off a 24 GB GPU and onto disk —
measured on a gemma-3-27b run, 6 of 8 loads reported "offloaded to the cpu **and
disk**" and extraction ran at 48–264 s/sample.

`_truncated_config` rebuilds the config with `num_hidden_layers = layer + 1`
(`text_config.num_hidden_layers` on multimodal checkpoints like gemma-3-*-it, the
top-level field otherwise), so only the executed prefix is instantiated and only its
weights are read out of the shards. **This is exact, not an approximation**: the stack
is causal, so layer `L`'s output is a function of layers `0..L` alone — verified
bit-identical on Llama-3.2-1B. That is why no activation cache key mentions truncation
and why blobs computed with and without it stay interchangeable. transformers logs an
"weights not used when initializing" warning for the dropped layers; that is expected.
It returns `None` (leave the model alone) when truncation is disabled, when the probe
reads the last layer anyway, or when the config exposes no layer count we recognise.

Two env knobs: `AGENTIC_REDTEAM_TRUNCATE_LAYERS=0` disables the truncation, and
`AGENTIC_REDTEAM_MAX_MEMORY` (e.g. `"0=21GiB,cpu=45GiB"`) pins accelerate's per-device
budget. The latter is unset by default, which keeps tuberlens' `max_memory=None` —
under which accelerate infers the budget from whatever is *free at load time* and
silently falls back to **disk** offload on a tight box. `extraction_batch_size()`
reads tuberlens' `BATCH_SIZE` (default 1) so the same env var drives both
`get_activations` and the red-team chunking in `retrain`.

### `agentic_redteam/llm_judge.py`
**Unbiased classifier** that works with either provider. When
`provider: claude_sdk` it uses the `anthropic` SDK directly; when
`provider: openrouter` it uses the `openai` SDK pointed at OpenRouter. In both
cases the judge is asked to pick one of the two class labels on the
conversation's own merits, with no hint about which label the caller is hoping
for. Expects strict JSON output (`{"label", "reason", "confidence"}`); parses
with code-fence stripping + brace extraction fallback; normalizes
case-insensitively against the probe's pos/neg class labels. Returns
`JudgeVerdict(label, reason, confidence)`.

The same judge also maintains the **rolling strategy memo** via
`summarize_round(records, *, round_num, error_type, true_class_label,
prior_summary="")`: it renders every attempt of the round (status, attacker model,
**`probe_score`**, probe vs. judge label, judge reason, and per-message-truncated
transcript) plus per-round aggregates (success rate, mean/min/max probe score, how
many samples the probe assigned to the positive class) and the `prior_summary` into
one user message, and asks the judge — under a dedicated system prompt built by
`_summary_system(max_tokens)`, not the classification one — to **rewrite and
condense** the prior memo with the new round's findings (merge duplicates, drop
superseded notes). So the memo is bounded, not cumulative. Reuses the same
`_call_anthropic` / `_call_openrouter` backends; returns `prior_summary` unchanged
for an empty round.

Three properties of that prompt are load-bearing and were each fixed after a
run pathology:

- **The word budget is a 200-word target under a `judge.max_tokens`-derived ceiling.**
  `_summary_word_budget(max_tokens) = min(_SUMMARY_TARGET_WORDS, max(150,
  int(max_tokens * _SUMMARY_WORDS_PER_TOKEN)))` — target 200, factor 0.45, so at the
  default 1024 tokens the ceiling is 460 and the **target governs: 200 words**. The
  two constraints are independent. The *ceiling* is physical: a budget the model
  cannot reach is worse than none — the fixed 700-word cap against a 1024-token
  ceiling truncated **all 48** memos of the experiment7 runs mid-sentence, always
  amputating the concluding strategy section, and since the memo is fed back as the
  next round's `prior_summary` the loss compounds (measured density for this
  dense-markdown register is ~0.61 words/token, so 0.45 leaves room to finish). The
  *target* is editorial: the memo is injected into every later round's attacker system
  prompt, and at 460 words it measured 3.4k chars against a 3.2k-char prompt — **54%
  of the whole system message**, crowding out the instructions it supplements. Raising
  `judge.max_tokens` no longer lengthens the memo; change `_SUMMARY_TARGET_WORDS` for
  that. The prompt's closing paragraph is tuned to the tighter budget: it tells the
  judge to write bullets, and to *drop* the weakest notes wholesale rather than
  compress every note into vagueness.
- **Unsuccessful samples are analyzed as first-class evidence.** The prompt asks
  what did *not* expose a weakness and how confidently (probe score near 0/1 = far
  from the boundary = strong evidence, near 0.5 = nearly flipped), which lines of
  investigation are exhausted, and explicitly licenses "this region has been
  characterized; examine a different one" as a conclusion. The earlier prompt asked
  only "what strategies work" / "what was most effective", which forces a
  recommendation even from a round with one success in 47 — the judge duly promoted
  that single sample to a template and the attackers converged on it.
- **Opposite-direction misclassifications are withheld** (`hide_opposite_direction`,
  default on) — see `_drop_opposite_direction`. Rows where probe and judge disagree
  the *other* way from `error_type` (a false positive found during a `false_negative`
  hunt) are dropped before rendering, and before the aggregates are computed, so the
  counts stay consistent. Rows they *agreed* on are kept: "the classifier handled
  this correctly" is evidence the memo needs whichever class it landed on. Without
  this, every experiment7 false-positive memo carried a "what reliably yields probe
  false negatives" section prescribing moves that were unwinnable in that rotation.
It also writes the **cross-iteration memo** via `summarize_iteration(successes, *,
iteration, error_type, true_class_label, round_memo="", prior_memo="", n_attempts=0,
max_successes=30)`, called once per rotation *before* the retrain. Under its own
`_ITERATION_SUMMARY_SYSTEM` prompt the judge is told the classifier is about to be
retrained on these misclassified samples, and asked for three things: failure modes
now covered by retraining, conversation types already handled correctly, and regions
of the input space not yet examined — folding `prior_memo` in by rewriting rather
than appending, capped at `_ITERATION_MEMO_WORD_BUDGET` (~900) words. Only the
`max_successes` most recent successes are rendered (0 = all); returns `prior_memo`
unchanged when the iteration produced neither successes nor a round memo.

**Both summarization prompts are written in neutral analyst register** — "analyze the
robustness of a text classifier", samples/misclassifications/evaluation cycles — never
as red-team/attacker coaching ("strategies that worked", "what the next attackers
should try"). This is not stylistic: the original adversarially-phrased round-summary
prompt drew refusals from the judge (`openai/gpt-5.1-chat` in every config)
and had to be rewritten. Keep any new summarization prompt in the same register, and
note that `summarize_iteration` is the more exposed of the two — its input is *only*
the successes, i.e. exactly the conversations the judge itself labelled
harmful/high-stakes.

**Refusal guard.** A refusal is a 200 with prose, not an exception, so it would
otherwise be stored as the memo and injected into later attackers' system prompts as
if it were findings. Both summarizers therefore route through
`LLMJudge._summarization_call(system, user_content, *, what)`:
`_looks_like_refusal` scans the first `_REFUSAL_SCAN_CHARS` (240) characters for a
leading refusal phrase from `_REFUSAL_MARKERS` (prefix-only, so a write-up quoting
"can't" mid-text doesn't trip it); on a hit the judge is re-asked once **in-context**
(original user turn + its refusal + `_REFUSAL_RETRY_NUDGE`, which restates that this
is a classifier-quality report over already-collected data). A second refusal raises
`JudgeRefusalError`, which `_summarize_round` / `_write_iteration_memo` deliberately
**do not swallow** (they log `summary_refused` / `iteration_memo_refused` and
re-raise) — the run stops rather than continuing on a missing or poisoned memo.
Ordinary transient errors are still swallowed as before.

### `agentic_redteam/openrouter_client.py`
Thin factory around `openai.OpenAI` / `openai.AsyncOpenAI` pointed at
OpenRouter (`https://openrouter.ai/api/v1`). Reads `OPENROUTER_API_KEY` and the
optional `OPENROUTER_BASE_URL` / `OPENROUTER_HTTP_REFERER` /
`OPENROUTER_APP_TITLE` env vars. Imports `openai` lazily.

### `agentic_redteam/circuit_breaker.py`
Process-global breaker that stops a run when OpenRouter is **durably** down.
Exists because every OpenRouter call site here is individually fault-tolerant
by design — a failed attacker round is logged and the rotation continues
(`attacker.run_one_model`), a failed contrastive generation drops one record
(`preprocessing`), a failed summarization is swallowed (`_summarize_round`) —
which is right for a blip but makes an exhausted balance or revoked key
*invisible*: the run grinds through every remaining round, retrains on nothing,
evals, and exits 0 with a plausible-looking comparison CSV. (This is exactly what
one memo-ablation sweep did: 300 + 200 model-rounds all failing 402, then 3
retrain/eval cycles on zero red-team data.)

Every OpenRouter call reports its outcome via `record_success()` /
`record_failure(detail, where=)`; N consecutive failures across **all** call
sites trip the breaker, which raises `OpenRouterOutageError` and keeps raising
it. `classify_error` sorts every failure into **three** classes, since they
deserve very different patience (fatal is checked first, so a 402 whose body
mentions a reset connection is still a drained balance):

- **transient** (429/5xx/empty-choices envelopes) — we *reached* OpenRouter and
  it answered badly; may recover, so `OPENROUTER_MAX_CONSECUTIVE_ERRORS`
  (default 10) are allowed in a row, retried on 2/4/8s backoff.
- **connection** (`APIConnectionError`/`APITimeoutError`, DNS failure, reset,
  network unreachable) — the request never completed a round trip. Retried on a
  **minutes**-long schedule (`OPENROUTER_CONNECTION_BACKOFF_S`, default
  `60,120,480`; last interval repeats) and bounded by **elapsed time, not a
  count**: `OPENROUTER_MAX_CONNECTION_OUTAGE_S` (default 1800 = 30 min).
- **fatal** (401/402/403, "Insufficient credits", "Invalid API key") — never
  recovers without human action, so only
  `OPENROUTER_MAX_CONSECUTIVE_FATAL_ERRORS` (default 3), and callers **skip
  their backoff sleeps** for these (`classify_error` / `is_fatal_error`).

**Why connection errors are timed, not counted.** A count threshold counts
*observations*, and with `attacker.sessions_per_model: 10` one network event is
observed by ten concurrent sessions at once — so the counter hit its limit of 10
on the first wave, **before any backoff sleep began**, and lengthening the
backoff alone would not have helped. That is how a ~2.5 min connection blip
killed a 10-hour run twice in one night — two `openrouter_outage` aborts, then 6h
and 30min of idle GPU awaiting a human restart. A wall-clock streak is immune to
that multiplication. So once a streak contains **any**
connection error it is governed by the outage clock; `_streak_started_at` is
stamped on the first failure after a success and cleared by `record_success`.

Correspondingly, the retry loops (`attacker._openrouter_create_with_retry`,
`LLMJudge._call_openrouter`, `_ContrastiveLLM.generate`) bound connection
retries by the **breaker**, not by `_OPENROUTER_MAX_ATTEMPTS` — they keep
probing until it trips. That is deliberate: a network back at minute 12 resumes
*that same round* mid-flight, whereas a fixed attempt cap would already have
abandoned the round and its remaining `max_turns`. Backoff sleeps go through
`breaker.sleep_sync` / `sleep_async`, which sleep in 5s chunks and re-check
`raise_if_tripped()`, so when one call site declares the outage terminal every
other site sleeping out an 8-minute backoff wakes and aborts (and interpreter
shutdown never has to join a thread parked for 8 minutes).

"Consecutive" is global and reset by *any* OpenRouter success, so a single
durably-broken model in a rotation whose siblings still work will **not** trip
it — the rotation genuinely can continue in that case. Counters are guarded by
a `threading.Lock`, not an asyncio primitive, because call sites live in the
asyncio attacker loop *and* in `preprocessing`'s `ThreadPoolExecutor` workers.
`raise_if_tripped()` is called at the top of `run_one_model` and before each
OpenRouter call, so once tripped the remaining schedule collapses without
touching the network. `reset()` / `snapshot()` exist so a test can drive the
breaker through all three classes without real network failures.

### `agentic_redteam/tools.py`
Hosts both the **provider-agnostic handlers** (`handle_submit_conversation`,
`handle_view_past_attempts`, `handle_get_probe_info` — all take a
`ToolContext` and return plain dicts) and **two surfaces** built on top of
them:

- `build_mcp_server(ctx)` — wraps the handlers as MCP tools via
  `claude_agent_sdk.create_sdk_mcp_server` + `@tool`. Used by the Claude SDK
  driver. `claude_agent_sdk` is imported inside this function so OpenRouter-
  only configs don't need it.
- `openai_tool_definitions()` + `dispatch_tool_call(ctx, name, args)` —
  OpenAI-format tool schemas and a direct dispatcher used by the OpenRouter
  driver.

Both surfaces share the same business logic; success classification,
deduplication, and JSONL persistence happen exactly once inside the handlers.

- `submit_conversation(messages)` — **always** runs both the probe and the
  Claude judge. Whether the probe predicted the wrong class can only be
  established by comparing its prediction to the judge's label, so there is
  no short-circuit. Computes `success` as: probe and judge labels disagree
  *and* the disagreement direction matches the configured `error_type`.
  Persists every attempt with the judge's label included, and increments
  `ctx.session_records` / `session_successes` only when `JsonlStore.append`
  reports the row as newly persisted (a sibling may have won the race).
  When `ctx.near_dup_guard` is on, `try_reserve_opener` runs **before** the
  probe and judge — a rejected candidate returns `near_duplicate=True` having
  cost no scoring — and the reservation is dropped in a `finally`.
- `view_past_attempts(only_successful, limit)` — delegates to the shared
  `ViewSampler` (see `view_sampler.py`) so later attacker models in a rotation
  can learn from earlier ones. The default (`only_successful=false`) view is a
  **balanced** ~50/50 mix of successful/unsuccessful attempts (total = `limit`,
  backfilling from the other side when one is short), **blended** with true-class
  training-set examples on the successful side (tagged `success=True`,
  `attacker_model="__training_seed__"`), and **periodically reshuffled** (a fresh
  seeded random draw every `attacker.view_reshuffle_interval` submissions, stable
  within an interval). Setting `attacker.view_reshuffle: false` turns off the random
  reshuffle entirely: the attacker is then shown the **most-recent** successful and
  unsuccessful attempts (recency, not a random draw), and training seeds are used
  **only as a fallback** for the successful half when there are no real successes yet
  (rather than always blended). There is **no judge-confidence filter** here — confidence
  gating lives only in the training path (`retrain_probe(min_judge_confidence=)`).
- `get_probe_info()` — returns probe metadata.

A `ToolContext` is the closure shared by all three tools — it holds the probe,
judge, store, run id, the currently-active round + attacker model, the shared
`view_sampler`, the near-dup knobs, this session's `session_id` +
`session_records` / `session_successes` counters, and the optional
`prompt_trace_store`. The attacker module updates `current_attacker_model` and
`current_round` before each model run so JSONL rows attribute correctly.
`confidence_threshold` is still recorded on the context but is **no longer used to
filter `view_past_attempts`** (it only feeds the training-path gate).

### `agentic_redteam/view_sampler.py`
`ViewSampler` — one shared instance per rotation (built in `run_redteam`) backing
`view_past_attempts`. Holds the shared `JsonlStore`, the true-class training seeds
(`load_true_class_seeds`, read from the base training JSONL, filtered to
`probe.true_class_label`), and the reshuffle/balance knobs from `attacker:`
(`view_reshuffle`, `view_reshuffle_interval`, `view_balance`, `view_training_seeds`).
When `view_reshuffle` is false, `sample()` skips the random draw and instead returns
the most-recent attempts per side, with seeds used only as a fallback for the
successful half. The reshuffle
RNG is keyed on `(rng_seed, interval_idx)` — independent of the global RNG, so the
draw is reproducible regardless of drift. The base training path is threaded in via
`run_redteam(base_training_data_path=)` ← `run_redteam_sync` ← the iterative CLI
(`args.base_training_data`); the one-shot `run_redteam_main` passes none, so seeds
degrade to empty.

Tool naming for the allow-list: `mcp__redteam_tools__<tool>`. The MCP server
name (`redteam_tools`) is exported as `MCP_SERVER_NAME`.

### `agentic_redteam/attacker.py`
Dispatcher + rotation. For each `AttackerModel` in `config.attacker.models`,
picks the driver by `model.provider` — except when `config.attacker.interface ==
"prompt"` and the model is `openrouter`, which routes to the prompt driver
instead (`run_one_model`):

- **`claude_sdk`** — `_run_claude_sdk_model` wraps `ClaudeSDKClient`. Critical
  sandbox configuration:

  ```python
  ClaudeAgentOptions(
      allowed_tools=allowed_tool_names(),                          # mcp__redteam_tools__*
      disallowed_tools=["Bash","Edit","Write","Read","Glob",...],  # block all built-ins
      permission_mode="bypassPermissions",                         # auto-approve MCP calls
      setting_sources=[],                                          # don't auto-load filesystem CLAUDE.md
  )
  ```

- **`openrouter`** — `_run_openrouter_model` drives `chat.completions.create`
  with `tools=openai_tool_definitions()` in a manual loop: read assistant
  message → record any `tool_calls` → dispatch each via `dispatch_tool_call`
  → append the result as a `role: "tool"` message → repeat until the assistant
  emits no tool calls, the batch target is hit, or `max_turns` is reached.
  No MCP server is constructed on this path.

- **`openrouter` + `interface: prompt`** — `_run_openrouter_prompt_model` drives
  the same model with **no tools** (`_openrouter_create_with_retry(..., tools=None)`).
  The system prompt is `_build_full_system_prompt(...)` (which already bakes in the
  probe metadata `get_probe_info` would return) plus `_PROMPT_MODE_INSTRUCTIONS`
  (output exactly ONE candidate conversation per turn as a fenced ```json array of
  `{role, content}`). Each turn: parse the reply with `_extract_conversation`
  (fenced block → balanced `[...]` → whole text; `_coerce_messages` validates
  role+content and also accepts a `{"messages": [...]}` wrapper); on parse failure,
  nudge and retry the turn; on success, score it through the same
  `dispatch_tool_call(ctx, "submit_conversation", ...)` path as tools mode, then feed
  back `_render_submission_feedback` (probe vs. judge verdict, duplicate /
  near-duplicate / error notes — no success count, see below) followed by a freshly
  injected `_render_injected_view` — `view_past_attempts` rendered as text,
  `attacker.view_limit` rows, since the model can't call it — and, under
  `near_dup_broadcast`, `_render_near_dup_rejects`. Both render blocks return `""` for
  `view_limit <= 0`, so a run configured to show the attacker nothing can't get past
  attempts back through the rejects channel.
  Assistant text is coerced to `""` before being appended so a
  null-content turn can't make the next request protocol-invalid. Respects
  `batch_target` (shared success counter) and `max_turns` (one submission per turn).
  This path is **openrouter-only** — `load_config` rejects `interface: prompt` with a
  `claude_sdk` model. No MCP server is constructed.

- **`openrouter` + `interface: prompt` + `batch_submissions: true`** —
  `_run_openrouter_prompt_batch_model`. Same system prompt, but with
  `_prompt_mode_batch_instructions(max_turns)` in place of `_PROMPT_MODE_INSTRUCTIONS`:
  the model is asked for **all `max_turns` conversations in one reply**, they are parsed
  by `_extract_conversations` (plural), each is scored through the same
  `dispatch_tool_call(ctx, "submit_conversation", ...)` path, and the session **ends** —
  so `max_turns` is a batch size, not a turn budget, and the attacker is **never shown a
  verdict**. That is the point of the mode: it isolates what in-context feedback does,
  since the per-turn loop is also how a session talks itself into mode collapse.

  `_extract_conversations` accepts N fenced blocks, one block holding an array of arrays,
  or a `{"conversations": [...]}` wrapper; each fenced block is parsed **independently**,
  so a batch whose last block was guillotined by `max_tokens` still yields every complete
  conversation before it. Exact duplicates within a reply are collapsed and the result is
  capped at `max_turns`.

  Two deliberate asymmetries with the per-turn loop. **`batch_target` is checked between
  calls, not between conversations** — a round can overshoot it by up to one batch per
  session, which is the right trade because the generation cost is already sunk and only
  the cheaper probe+judge scoring would be saved. And a reply **short of `max_turns` gets
  up to `_BATCH_MAX_FOLLOWUPS` (2) top-up asks**; the follow-up names only how many more
  conversations are wanted, never a verdict, or the session would stop being blind.
  `stop_reason` is one of `batch_complete` / `batch_short` / `batch_no_parse` /
  `target_reached`.

  Note this mode **does** tell the attacker a number — the batch size. That does not
  violate the "never state a quota" rule below, which is about `batch_target`'s success
  count: a batch size is a workload the model cannot produce without knowing, not a goal
  it can meet early and stop searching.

A fresh `ToolContext` is built per model run (with round/model labels set), but
all runs share the same `JsonlStore` so dedup and the success counter persist
across rotation. `run_redteam` builds one shared `ProbeJudge` for the whole
rotation and calls `probe.release()` once `asyncio.gather` finishes, freeing the
probe's LLM (gemma-sized) GPU memory before the next phase (retrain/eval) reloads
the base model — without it, two copies pile up and the retrain offload-thrashes.

**Round scheduling.** When `attacker.round_summaries` is on (default), `run_redteam`
runs rounds **sequentially** — for each round it launches that round's models
concurrently (bounded by `concurrency`), `await`s them, then calls
`_summarize_round` (judge folds the round into the rolling memo via
`summary_store.update`, passing `summary_store.current` as `prior_summary`) before
starting the next round. The final round is *not* summarized (nothing would consume
it). Each model run renders `ctx.summary_store.render()` into its system prompt at
session start, so sequential ordering guarantees round N sees the memo distilled
from rounds 0..N-1. `_summarize_round` swallows transient judge failures (logged to
the runlog as `summary_error`) so a summarization hiccup never aborts the rotation —
except a `JudgeRefusalError`, which is logged as `summary_refused` and re-raised. When
`round_summaries` is off, the legacy path launches **all** round×model sessions at
once with no memo. Note this trades throughput for the memo signal: with `rounds:
20, concurrency: 30` the legacy path runs all 20 rounds in parallel; sequential runs
them one at a time.

**Round-level resume.** `run_redteam(..., resume=)` makes a rotation restartable at
round granularity rather than only at the `(iteration, error_type)` phase boundary.
A `RoundProgressStore` on `<jsonl>.rounds_done.jsonl` is built on **every** run (so a
future restart always has checkpoints), but consulted only when `resume=True`; the
`SummaryStore` is then also constructed with `resume=True` so the rolling memo is
reloaded from `<jsonl>.summaries.jsonl` instead of starting empty. In the sequential
branch each round is marked done *after* `_summarize_round` returns, so progress and
memo can't diverge; the final round (never summarized) is marked straight after its
tasks. The legacy branch has no round boundary to checkpoint at, so it gives each
round its own task group, launches them **all** up front (the semaphore, not the await
order, governs concurrency — throughput is unchanged) and awaits the groups in round
order, marking each as it lands; a raising group cancels every sibling, matching
`_gather_or_cancel`. A round interrupted part-way is simply not marked and re-runs in
full — its attempts survive in the JSONL, and since they carry the same round number
they still count toward that round's summary. Skipped rounds contribute no
`ModelRunSummary`, so the CLI's per-error-type success count reports what *this*
process did, not the cumulative total. The iterative CLI passes
`resume=args.resume and i == start_iter` (only the resumed iteration can hold partial
progress) and logs a `round_skipped` runlog event per skipped round.

**Cross-iteration memo.** Independent of the above (it works with `round_summaries`
either on or off). When `attacker.cross_iteration_memos` is on, `run_redteam` builds
an `IterationMemoStore` on `<jsonl>.iteration_memos.jsonl`, threads it into every
`ToolContext`, and — after the whole rotation, before returning to the CLI's retrain
step — calls `_write_iteration_memo`: it gathers `store.records_for_iteration(iteration)`,
takes the successes plus the final rolling round memo, and asks the judge for the
hand-off memo (`summarize_iteration`), which is appended to the sidecar. Transient
judge failures are logged (`iteration_memo_error`) and swallowed; a `JudgeRefusalError`
(judge declined twice — see the refusal guard above) is logged and re-raised, stopping
the run. The prompt side is
`_prompt_memos(ctx)` → `(iteration_memo, round_memo)`, both passed to
`_build_full_system_prompt` (iteration memo first, round memo last as the more
immediate signal) by all four drivers. Note the CLI's phase-marker resume path skips
the whole rotation for an already-finished `(iteration, error_type)`, so that
iteration contributes no memo — the next one falls back to the newest earlier memo.

**`sessions_per_model`** multiplies the per-round fan-out: both the sequential and
legacy branches launch `sessions_per_model` tasks for *each* model (`for _ in
range(config.attacker.sessions_per_model)` in the round-task comprehension), so N>1
runs N independent concurrent sessions of the same model within a round **without**
duplicating it in `models` and **without** disabling `round_summaries` — the rounds
stay sequential and the memo is unaffected. All copies share the one `JsonlStore`
(dedup-by-canonical-text, so two siblings that hit the same conversation don't
double-write) and record the **same** `round`/`attacker_model`, so their attempts all
fold into that round's summary. Two consequences to plan for: (1) set `concurrency ≥
sessions_per_model × len(models)` or the copies queue on the semaphore instead of
running in parallel; (2) `batch_target` is checked against the **shared** success
counter (`ctx.store.success_count`), so N siblings collectively stop at ~`batch_target`
successes per round, not `N × batch_target` — it's a shared round budget, not
per-session. Note the baseline each session compares against is snapshotted when *it*
starts, so a session that queued on the semaphore gets a fresh budget of its own; keep
`concurrency ≥ sessions_per_model × len(models)` and the round stays at one budget.

**`batch_target` is enforced only programmatically — never told to the attacker.**
`_build_full_system_prompt` states what counts as a successful find but not how many
are wanted, `_render_submission_feedback` reports the verdict without a running count,
and the `submit_conversation` tool result carries no `successful_finds_so_far`. The
only stop signals the attacker can perceive are its own turn budget and the verdicts.
Enforcement lives in the OpenRouter driver loops, which check the shared counter after
each turn (after each *call*, in batch mode) and break with
`stop_reason="target_reached"`; the `claude_sdk` driver has no such check and is bounded
by `max_turns` alone. Don't reintroduce the quota into a prompt: an attacker given a
target treats it as a quota to satisfy and stops searching once it's met. The batch-size
number `batch_submissions` states is not this — see that driver above.

**`ModelRunSummary.new_successes` counts the session's own rows**, taken from
`ToolContext.session_records` / `session_successes` (incremented in
`handle_submit_conversation` only when `JsonlStore.append` actually persisted). It must
**not** be a delta on the shared store: siblings write concurrently, so a store delta
measured around one session also counts theirs, and since the caller sums the summaries
the error multiplies by the fan-out (with `sessions_per_model: 5`, a 30-success round
reported ~150). `_mark_round_done` sums these, so `rounds_done.jsonl` inherits the fix.

### `agentic_redteam/retrain.py`
Converts successful JSONL records into a tuberlens `LabelledDataset` — labelled
with the canonical enum value (`"positive"` / `"negative"`) corresponding to the
*true* class for the run's `error_type`. The base training dataset
(`LabelledDataset.load_from(path, pos_class_label, neg_class_label)`) and the
red-team set are **split independently** and combined per side at activation time
(see the caching paragraph below), not pre-concatenated. By default `_infer_probe_spec`
walks the loaded probe's `_classifier.probe_architecture` (or the SklearnProbe
shape, or the difference-of-means/LDA shape) to reconstruct the `ProbeSpec` so the
retrained probe matches the original's architecture and hyperparameters. Pass
`retrain_probe(..., probe_spec=...)` (a `ProbeSpec`, or a `ProbeType` name string
like `"linear_then_softmax"`) to instead train a **fresh** architecture; the CLIs
expose this as `--probe-arch` (bare flag → `DEFAULT_FRESH_PROBE_ARCH`,
`"linear_then_softmax"`; pass a name to override; omit to inherit). A name string
builds `ProbeSpec(name=ProbeType(name), hyperparams={})`, letting tuberlens fill
in the arch's default hyperparams (`_coerce_probe_spec` does this string→ProbeSpec
conversion, shared by `retrain_probe` and `train_initial_probe`).

`train_initial_probe(...)` trains the **first** probe from base training data alone
(no base probe to inherit from), so the caller supplies `model_name`, `layer`,
`pos_class_label`, `neg_class_label`, `probe_description`, and `probe_spec`
(defaulting to `DEFAULT_FRESH_PROBE_ARCH`). This mirrors tuberlens'
`collate_train_evaluate.train_high_stakes_probe` but with the concept passed in
rather than hardcoded.

Both `retrain_probe` and `train_initial_probe` derive the validation set with
`stable_train_test_split(dataset, test_size, split_field, seed)` — a
**content-deterministic** replacement for tuberlens' RNG-based
`create_train_test_split`. Each sample's train-vs-val side is
`sha256(seed : content)` (or the `split_field` value) thresholded at `test_size`,
independent of dataset size or order, so the base samples land identically every
iteration; class balance is preserved in expectation. There is no external
validation-file path.

**Activation caching (base-blob + red-team per-sample).** Because the base split is
fixed, the base train/val activations are cached on disk and reused across the whole
run. The red-team set grows every iteration, so it is cached at a **different
granularity**: per conversation (a single whole-set blob like the base one would get
a fresh key each iteration and never hit). `retrain_probe` / `train_initial_probe`
split base and red-team separately, then `_train_with_cached_base_activations`
re-hosts the tail of tuberlens' `train_probe`: it activates each sub-dataset (base via
tuberlens' `get_activations(save_path=...)` blob cache — a hit calls
`LLMModel.load_activations` and needs no model; the red-team set via
`_activate_redteam_cached`, which partitions by per-conversation cache hit, loads the
hits from disk, computes only the misses, and writes each new row back as its
own blob), merges per side with `_concatenate_consuming` (which pads +
concatenates the activation tensors), then calls `ProbeFactory.build` on the
pre-activated datasets. The heavy `LLMModel` loads **lazily** — a full cache hit with
no uncached red-team samples loads no model at all. `_base_activation_cache_paths`
keys the base cache on a hash of the base data file +
`model | layer | seed | test_size | split_field | combine | convert`;
`_redteam_activation_cache_path` keys each red-team blob on the conversation's own
(transformed) messages + `model | layer | combine | convert`. Per-conversation
caching is **correct across iterations because the underlying LLM is frozen** (only
the probe head is retrained), so a conversation's layer activation is identical
regardless of which iteration computes it — even when `preprocessing` keeps/drops
different records or mints new contrastive pairs each iteration, each is keyed by its
own final content. Since `get_activations` / `load_activations` load *by path without
validating inputs*, any change that would alter the activations changes the key (no
silent stale reuse). Both caches are disabled when `base_activation_cache_dir=None`.

**Misses are computed in chunks and written through per row.** `_activate_redteam_cached`
loops the miss set in chunks of `model_loading.extraction_batch_size()` (tuberlens'
`BATCH_SIZE`, default 1), saving each row's blob as soon as its chunk returns, rather
than one `get_activations` call over the whole miss set followed by a bulk save. Two
reasons, both from a 770-sample gemma-3-27b retrain:

- **Resumability.** The single-call form persists nothing until the last sample lands,
  so a crash at row 606 of 607 discarded ~25 h of forwards. Now the next attempt
  reloads everything already computed. This matters most where the cache does *not*
  survive the container (cloud boxes with no long-term store for red-team activations,
  unlike the Kaggle-published eval blobs) — within one run it is the difference between
  a retry costing minutes and costing the whole retrain again.
- **Width.** `get_activations` pads every row in a call to that call's max length,
  capped at 1024 (`tuberlens/model.py:433`). Over the whole miss set that is 1024 for
  essentially every row; per chunk it is the chunk's own max, and at chunk size 1 it is
  each row's true length. Real conversations average ~535 tokens, so this roughly halves
  both the cache's disk footprint (10.7 GB → ~5.6 GB at 970 rows) and resident RAM.
  `_concatenate_consuming` re-pads at merge, so the merged tensor is byte-identical
  either way — which is also why blobs written at different chunk sizes interoperate.

Progress is printed every `_REDTEAM_PROGRESS_EVERY` (10) rows with a running s/sample
and ETA, instead of tqdm — one bar per chunk would be hundreds of bars in the log.

**Host-RAM budget of a retrain.** This function's peak is what OOM-kills long runs —
observed as a SIGKILL (exit 137, no traceback) at the iteration-2 retrain of a
966-sample gemma-3-27b probe on a 60 GB box. Two things are held at once and both
are guarded:

- **The extraction model.** `LLMModel.load` uses `device_map="auto"` with
  `max_memory=None`, so accelerate hands the `"cpu"` device a budget equal to whatever
  RAM is free *at load time* — a gemma-sized model keeps multi-GB of CPU-offloaded
  shards resident for as long as it is referenced. `_train_with_cached_base_activations`
  therefore calls `_release_model()` (mirrors `ProbeJudge.release`) immediately after
  the last `_activate*` call and **before** the merge + `ProbeFactory.build`, which
  need no model.
- **The activations.** At hidden 5376 / fp16 / padded to `get_activations`' 1024-token
  cap that is 11 MB per sample, so ~10 GB resident for a 966-sample retrain.
  `LabelledDataset.concatenate` pads every part *then* `torch.cat`s, holding inputs and
  output simultaneously (~2x, ~19 GB). `_concatenate_consuming` is a drop-in
  replacement that is byte-identical in output but fills a `torch.empty` block slice by
  slice, popping each part's pad fields as it copies them, so peak stays at ~1x. It
  **consumes** its inputs — capture any `len()` you need before calling it — and falls
  back to `LabelledDataset.concatenate` for layouts it can't reproduce exactly
  (non-torch pad fields, mixed dtype/device/rank). `torch.empty` over `torch.zeros` is
  load-bearing: the allocation stays lazily faulted, so every byte must be written
  exactly once (real rows, then an explicit zero-fill of each part's pad region).

Neither is a full fix — the whole set is still materialized in RAM. Streaming it
(mmap-backed blobs, or a lazy `ActivationDataset` that pads per batch) needs tuberlens
changes; see the OOM analysis in the git history for this section.

**Training-time message transforms.** `combine_consecutive_messages` /
`convert_tool_to_assistant` apply to the training data too (not just eval): the
base data gets them via `load_from`, and the in-memory red-team set via
`_apply_message_transforms` (convert tool→assistant first, then combine, matching
`load_from` order). They're part of the activation cache key.

When `retrain_probe` is given a `preprocessing` config, the
red-team successes are first run through `_build_redteam_dataset`, which mirrors the
collation step of tuberlens' pipeline applied to the "extra" data: `filter_dataset`
(drop confounders) then `generate_contrastive_dataset` (add opposite-class pairs),
keyed off the probe's pos/neg labels. The contrastive pairs are cached to disk
(`contrastive_cache_path`) so successes accumulated across iterations aren't
re-generated. With no `preprocessing`, the plain `_records_to_labelled_dataset`
path (judge label → canonical class) is used unchanged. When given a
`postprocessed_out_path`, `retrain_probe` also dumps the resulting red-team
`LabelledDataset` (the postprocessed red-team samples **only** — base training
data excluded) to that JSONL via `_dump_labelled_dataset` (`{id, inputs, label}`
rows) before concatenation, giving a per-iteration snapshot of exactly what
red-team data trained each probe. The iterative CLI writes
`<probe-out-dir>/redteam_postprocessed_iter{N}.jsonl` per cycle.

### `agentic_redteam/preprocessing.py`
Ports the collation preprocessing of tuberlens' `collate_train_evaluate.py`,
generalized off the hard-coded `"high-stakes"`/`"low-stakes"` to arbitrary
`pos_class_label` / `neg_class_label`. `filter_dataset(records, pos_class_label,
filter_percentile)` fits a bag-of-words `LogisticRegression` (`BagOfWordsClassifier`)
and drops the records it predicts most confidently (top percentile); it's a no-op
when fewer than two classes are present. `generate_contrastive_dataset(...)` asks an
LLM (this repo's Anthropic / OpenRouter sync clients, fanned out over a
`ThreadPoolExecutor`, **not** litellm) to write an opposite-class version of each
conversation, returning originals + generated pairs. Generated pairs are cached to a
JSONL keyed by `sha256(source messages + target label [+ guidance fingerprint])` so
accumulating successes only pay for newly-seen conversations. `label_dataset` (LLM
relabel) is intentionally **not** ported — red-team data already carries judge labels.

**Concept detail in the generation prompt.** By default `_generation_system_prompt`
knows nothing about the concept beyond the two label strings.
`preprocessing.concept_description` (what the concept is, shown in both directions)
and `preprocessing.label_guidance` (a `{class label: free text}` mapping, shown when
generating *toward* that label) are injected verbatim, so a config can define the
concept as concretely as a hand-written per-concept prompt would — e.g. `high-stakes:
"- significant financial loss, life-threatening situations, legal consequences"`.
Guidance is looked up by the raw label first, then its `LABEL_SHORT` alias; keys
matching neither class label are warned about and ignored (a typo would otherwise
silently degrade the prompt back to its generic form). **Non-empty guidance is folded
into the contrastive cache key** via `_guidance_fingerprint` — the cache is loaded by
key without re-checking the prompt, so otherwise an edited description would silently
reuse pairs written under the old one. The fingerprint is `""` when neither knob is
set, so configs that don't use them keep byte-identical keys (existing caches still
hit), and it covers only the *target* label's guidance, so editing one class's text
doesn't invalidate the other direction's pairs.

### `agentic_redteam/evaluation.py`
`evaluate_probe(probe_path, eval_dataset_dir, activations_cache_dir, splits=None,
max_samples=100, seed=42, combine_consecutive_messages=False,
convert_tool_to_assistant=False)` scores one probe on local eval split JSONLs via
tuberlens `get_performances`, returning a per-split DataFrame. When `splits is None`
(the default) the splits are **auto-discovered** — every `*.jsonl` in `eval_dataset_dir`
is scored, keyed by its filename stem (there is no longer a hardcoded
`DEFAULT_EVAL_SPLITS` list; drop new eval JSONLs into a dir and they are picked up
without code changes). Each split is loaded with the probe's own pos/neg class labels,
so a split's `labels` strings must match them exactly. It calls `seed_everything(seed)` (ported from
tuberlens) and then balances each split to `max_samples` via
`subsample_balanced_subset(n_per_class=max_samples // 2)` (`max_samples=None` → full
split). Seeding before subsampling makes the subset identical across every probe
eval — which is what keeps the path-keyed activation cache correct, since
`get_activations` reloads by file path **without** checking the inputs match. The
cache filename embeds `max_samples`/`seed` (`acts_n{N}_seed{S}.pt`) so a different
subsample config can't silently reuse stale activations.
`combine_consecutive_messages` / `convert_tool_to_assistant` are tuberlens
`LabelledDataset` loader transforms forwarded into `load_from` for the eval splits
(merge adjacent same-role messages; rewrite `tool`→`assistant`, the latter applied
first). **Unlike tuberlens' `collate_train_evaluate.py`, where these are eval-time
only, this repo applies the same values to the training data as well** (see
`retrain.py`) so the probe trains and is scored on the same message representation.
Exposed via the config `eval:` section (`EvalConfig`) and overridable per-run by the
`--[no-]combine-consecutive-messages` / `--[no-]convert-tool-to-assistant` CLI flags.

`kaggle_source=` (a `KaggleActivationSource`, built by the CLI from the config
`kaggle:` section) pre-populates the activation cache from Kaggle before
`get_performances` runs — see below. It is rejected when `max_samples is not None`.

`_assign_cached_activations(eval_datasets, activations_save_path)` then attaches every
already-cached split's blob to its dataset **before** `get_performances` is called.
This is purely a fast path, but a load-bearing one: `get_performances` loads the
extraction model the moment it meets a split with no `activations` field
(`tuberlens/evaluation.py:75-77`) — *before* `get_activations` gets as far as checking
`save_path.exists()`. So a fully-cached eval (the normal case once `kaggle:` has
prefetched) still paid a multi-minute gemma-3-27b load it never used. Blobs are keyed
by the same path `get_performances` derives (`<dir>/<split>-<cache_stem>`); a split
whose blob is missing, unreadable, or the wrong row count is left alone and recomputed
exactly as before, so this can never mask a stale cache.

### `agentic_redteam/kaggle_activations.py`
`prefetch_eval_activations(cache_dir, eval_datasets, source, *, model_name, layer,
cache_stem)` downloads **precomputed** eval activations published on Kaggle into the
same path-keyed cache dir `evaluate_probe` already uses, writing each split to the
exact name `get_performances` derives (`<split>-acts_full.pt`). tuberlens'
`get_activations` checks `save_path.exists()` first, so the subsequent eval is a pure
cache hit and **no LLM is ever loaded** — the point being that full-split gemma-3-27b
activations are ~20 GB and hours of forward passes.

Deliberately **not** built on tuberlens' `get_activations(using_kaggle=True)`:
- **Addressing.** `LLMModel._get_kaggle_dataset_slug` derives the slug from the local
  `save_path` by stripping punctuation and truncating to the first 50 chars — our
  discriminating suffix is truncated away, so all four eval splits collapse to one
  slug. `KaggleActivationSource(owner, dataset_slug, file_name)` is explicit instead,
  with both fields `str.format`-ed on `split=<split stem>`.
- **Transfer volume.** tuberlens' `_download_from_kaggle` pulls and unzips the *whole*
  dataset to a temp dir per call. This fetches the one file, into a staging dir on the
  same filesystem as the cache, so landing it is a rename not a second copy.
- **Validation.** Every blob (downloaded *or* already cached) is checked against the
  probe's `model_name`/`layer` and the split's row count before it may be used;
  mismatches raise `KaggleActivationError`. `LLMModel.load_activations` discards the
  `model_name`/`layer` the blob was saved with, and this repo's caches otherwise load
  by path without validating inputs — acceptable for content-keyed blobs we computed,
  not for one fetched from a remote store. The header read uses `torch.load(...,
  mmap=True)`, so validating the 11 GB anthropic blob is instant.

Auth: `KAGGLE_CONFIG_DIR` must name the **directory** holding `kaggle.json` (the API
joins the filename on, and `os.makedirs` a wrong path, so pointing at the file fails
silently), or export `KAGGLE_API_TOKEN`. `_authenticate()` catches `SystemExit`
because `KaggleApi.authenticate()` ends in `exit(1)` when no credential resolves —
a `BaseException` an ordinary `except Exception` would let kill the run. A split that
cannot be fetched **raises rather than falling back** to computing it.

### Eval dataset splits on disk
Three eval-split directories ship in the repo, one per probe concept. `--eval-dataset-dir`
picks which one a run scores against; with `splits=None` (the default) `evaluate_probe`
auto-discovers every `<dir>/*.jsonl` as a split (keyed by filename stem). Every split
JSONL row is a tuberlens `LabelledDataset` record: `inputs` is a **JSON-encoded string**
of `[{role, content}, ...]` (parse it, don't treat it as text) and `labels` is the class
string; any other columns are provenance only.

- **`eval_datasets/`** — the **high-stakes** probe's splits:
  `anthropic` (1028), `mt` (278), `mts` (274), `toolace` (328).
  Labels are `high-stakes` / `low-stakes`, exactly class-balanced per split. Extra columns:
  `ids`, `scale_labels`, `scale_label_confidence`, and per-split provenance
  (`category`/`index` for anthropic, `medical_specialty`/`sample_name` for mt,
  `section_header` for mts, `original_system_prompts`/`modified_system_prompts` for toolace —
  toolace `inputs` always uses the *modified* system prompt). **Gotcha:**
  `label_explanation` here is boilerplate (`"Filled in based on scale_labels and
  scale_label_confidence"`) — the real human-readable rationale is in
  **`scale_label_explanation`**. Rows are **independently labelled**: there is no
  cross-class pairing (anthropic chosen/rejected `index` is unique and orthogonal to the
  stakes label; mt/mts/toolace ids are per-sample).
- **`eval_dataset_hu_ha/`** — the **harmful_to_human** probe's splits: `eval_ai_dilemmas`
  (136 rows), `eval_balanced_refusal` (400), `eval_daily_dilemmas` (196), `eval_ant_hh`
  (134). Labels are `harmful_to_human` / `not_harmful_to_human`, class-balanced. Slim
  schema: `inputs`, `labels`, `harm_explanation` (the real rationale here — no boilerplate
  field), `source`. Three of the four splits are **paired**: the same user prompt appears
  once with a harmful assistant response and once with a non-harmful one (ai_dilemmas
  68 pairs, daily_dilemmas 98, balanced_refusal 200 — 158 one-to-one plus 21 prompts
  carrying two of each class). `eval_ant_hh` is **not** paired (every user prompt is
  distinct). Pairing is a property of the data, not something the eval code uses —
  `evaluate_probe` scores every row independently regardless.
- **`eval_instructions/`** — the **instruction-following** probe's splits (a third
  assistant-centric concept: did the assistant's response *follow the user's
  instruction* or not?). Seven splits, each exactly class-balanced:
  `anthropic_harmless_refusal` (200), `bbq_substitution` (200), `hc_context_drift`
  (194), `hc_contradiction` (200), `mm_substitution` (200), `oig_context_drift` (194),
  `oig_omission` (114). Labels are `assistant_follows_the_instruction` /
  `assistant_does_not_follow_the_instruction` — the split names encode the *failure
  mode* on the negative side (refusal, context drift, contradiction of the provided
  source, omission of requested content, answer substitution). Slim schema: `inputs`,
  `labels`, `judge_1_reasoning`, `judge_2_reasoning` (the two rationales that produced
  the label), plus per-split provenance (e.g. `context`/`question`/`correct_answer`/
  `wrong_answer`/`category` for bbq, `query`/`doc_a`/`doc_b` for hc_contradiction,
  `turn1_doc`/`turn2_doc`/`*_polarity` for hc_context_drift, `text`/`generated_content`/
  `cosine_distance` for mm, `human_turn_*`/`bot_turn_*` or `human_turns`/`bot_turns`/
  `original_text` for the oig splits). **These files were converted in place from a
  raw `{conversation, follows_the_instruction: bool, ...}` form** to the standard
  `inputs` (JSON string) + `labels` schema. Attack this concept with
  `configs/llama70b_instructions_llama1b.md` (llama70b attacker → llama-1b probe).

### `agentic_redteam/cli.py`
Two entry points: `run_redteam_main` (one round against an existing probe) and
`iterative_retrain_main`. The latter runs the full pipeline: **(1)** obtain the
initial probe — warm-start from `config.probe.path` if it points to an existing
file, else `train_initial_probe` from `--base-training-data`; **(2)** red-team it
across all `error_types`; **(3)** `retrain_probe` on base data ∪ successes;
**(4)** optionally `evaluate_probe` (gated by `--eval`); then repeat 2–4 for
`--iterations` n. It rewrites `config.probe.path` to the freshest probe before
each round, and (with `--eval`) writes the cross-round comparison CSV. That CSV
path and the eval-activations cache dir each resolve by the precedence **CLI flag
(`--comparison-csv` / `--activations-cache-dir`) > config `output:`
(`comparison_csv` / `activations_cache_dir`) > `<results-dir>`-derived default**
(`<results-dir>/iter_run_comparison.csv`, `<results-dir>/eval_activations`); the
config paths resolve relative to the config file. It calls
`seed_everything(--seed)` up front, threads `config.preprocessing` +
`<probe-out-dir>/contrastive_cache.jsonl` + `--test-size` / `--split-field` / `--seed`
into the train/retrain calls. The base (training) activation cache dir resolves by
precedence **`--base-activation-cache-dir` flag > config `output.base_activation_cache_dir`
> `<probe-out-dir>/base_activation_cache` default**, and passes `--eval-max-samples` / `--seed` into
`evaluate_probe`. `_free_gpu()` (`gc.collect()` + `torch.cuda.empty_cache()`) is
called after the initial training, after each `_maybe_eval`, and after each retrain
so reserved GPU memory is returned between heavy phases (each tuberlens
`device_map="auto"` load re-infers its layer split from *free* GPU memory). The
`combine_consecutive_messages` / `convert_tool_to_assistant` config knobs are
resolved against the `--[no-]…` CLI flags (`BooleanOptionalAction`, default `None`
→ config value) and forwarded into **both** the train/retrain calls and
`evaluate_probe`.

**`--resume` (default on) is three-tiered**, coarsest first, all keyed off
`--probe-out-dir` / the JSONL sidecars so nothing extra needs threading through:

1. **Iteration** — `_latest_probe_iteration` finds the highest `probe_iter{N}.pkl`
   (written only once a *retrain* finishes) and starts at iteration N.
2. **`(iteration, error_type)` phase** — `redteam_done_iter{N}_{et}.marker`, written
   after each rotation returns, skips that rotation entirely (its successes are
   already in the append-only JSONL, so the retrain still sees them). Gated on
   `args.resume and i == start_iter` — **not** on a probe having been found, because
   a warm-started run that died inside iteration 0 has no `probe_iter{N}.pkl` yet
   while its markers are still valid (its input probe is `config.probe.path` either
   way).
3. **Round** — `run_redteam(resume=…)` skips individual finished rounds and restores
   the rolling memo (see "Round-level resume" above).

`--no-resume` ignores all three and re-runs from scratch (stale markers and progress
rows are not consulted, and new ones simply append). Note the eval comparison CSV does
**not** resume: `eval_results` is in-process and the CSV is rewritten at the end, so a
resumed run's CSV covers only the iterations that run actually executed.

## Conventions to preserve

- **Probe metadata is the source of truth.** Don't pass `pos_class_label` /
  `neg_class_label` / `description` separately — read them off the loaded probe.
- **The attacker must never get filesystem/shell access.** For the Claude SDK
  driver, always carry both `allowed_tools=` and `disallowed_tools=` plus
  `setting_sources=[]` when constructing `ClaudeAgentOptions`. For the
  OpenRouter driver, the only tools the model can see are the three exposed
  by `openai_tool_definitions()` — there is no analog of "built-in tools"
  there. Adding a new tool means **all three** of: appending to
  `allowed_tool_names()`, adding to `openai_tool_definitions()`, and writing
  a handler in `tools.HANDLERS`.
- **The judge always runs, and is unbiased.** Whether the probe predicted the
  wrong class can only be established by comparing the probe's prediction to
  the judge's label — there is no probe-prediction-only short-circuit. The
  judge is told the two candidate labels but is **not** told which one the
  caller is hoping for, so it acts as an independent classifier. `success` is
  computed in `tools.py` after both run.
- **`OpenRouterOutageError` must never be swallowed.** Every `except Exception`
  around an OpenRouter call (`run_one_model`, `_summarize_round`,
  `_write_iteration_memo`, `_ContrastiveLLM.generate`) is there to absorb a
  *single* failure so the run continues — so each one needs an explicit
  `except OpenRouterOutageError: raise` **before** it, mirroring how
  `JudgeRefusalError` is handled. New OpenRouter call sites must report to the
  breaker (`record_success` / `record_failure`) or the "consecutive" count
  silently under-counts. The CLI mains carry `@_exit_on_outage`, which turns
  the error into a clean message plus `OUTAGE_EXIT_CODE` (3) instead of a
  traceback; nothing is lost by stopping, since the JSONL is append-only and
  the phase markers and per-round progress sidecar let `--resume` continue from
  the round it stopped on.
- **JSONL is append-only and dedup-by-canonical-text.** `JsonlStore` rejects
  duplicate conversations silently — the agent's `submit_conversation`
  response surfaces `duplicate=True` so it can move on.
- **Tool functions return the `{"content": [{"type": "text", "text": ...}]}`
  shape exactly.** Anything else breaks the Claude Agent SDK's tool result
  streaming.
- **Load the extraction LLM through `model_loading.load_extraction_model`.** Never call
  `LLMModel.load` directly for activation extraction. It is the one place that knows the
  model only needs layers `0..probe.layer` — tuberlens truncates the *executed* stack
  inside `HookedModel` but places the whole model first, so a direct load dispatches
  ~24 GB of gemma-3-27b weights that never run a forward and pushes the executed tail
  onto disk. Truncation is exact (causal stack), so it does **not** belong in any
  activation cache key.
- **Free GPU memory between heavy phases.** Every tuberlens load uses
  `device_map="auto"` + `max_memory=None`, re-inferring the layer split from
  *free* GPU memory at load time; torch's caching allocator holds freed memory as
  reserved. So a model left resident from a previous phase forces the next load
  into CPU/disk offload (~5-10x slower). Release models and clear the cache
  between phases: `ProbeJudge.release()` after red-teaming, `cli._free_gpu()`
  after train/eval/retrain.
- **Activation caches load by path without validating inputs.** The eval cache
  (`evaluation.py`, key embeds `max_samples`/`seed`), the base-blob training cache
  (`retrain._base_activation_cache_paths`, key embeds the base file hash +
  `model`/`layer`/`seed`/`test_size`/`split_field`/transform flags), and the
  per-conversation red-team cache (`retrain._redteam_activation_cache_path`, key
  embeds the conversation's own messages + `model`/`layer`/transform flags) all
  rely on the **key** to prevent silent stale reuse. Anything new that changes
  which samples are selected or how they're tokenized must be folded into the key.
  Red-team caching is per-conversation, not a whole-set blob, **specifically so it
  survives the set growing each iteration** — don't "simplify" it to a single blob.
- **Use `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`** as
  the canonical model IDs for the rotation. Don't append date suffixes to opus
  or sonnet — only Haiku 4.5 currently requires the dated form.
