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
2. Run a Claude **attacker** (via `claude_agent_sdk.ClaudeSDKClient`) inside a
   tool-use loop. The attacker only has access to three in-process MCP tools —
   no shell, no filesystem, no web. It submits candidate conversations and
   reads back probe + judge verdicts to refine its strategy in-context.
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
   appending, so the memo stays bounded (~200 words) no matter how many rounds
   run. That memo is injected into the system prompt of every later round's
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
judge:
  provider: claude_sdk | openrouter
  model: <model>
  max_tokens: int
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
eval:                                 # OPTIONAL: dataset message transforms applied to BOTH
  combine_consecutive_messages: bool  # training data AND eval splits (default false) — merge
  convert_tool_to_assistant: bool     # adjacent same-role msgs; rewrite tool→assistant (first)
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
the success counter and dedup set warm. Each row carries
`{sample, probe_score, probe_predicts_positive, judge_label, judge_reason,
success, attacker_model, run_id, round, iteration, error_type, pos_class_label,
neg_class_label}` — `judge_label` is the class label the judge picked
(human-readable, e.g. "high-stakes"), or `""` if the judge response was
unparseable. `iteration` is the 0-based retrain-cycle index (the CLI threads it
through `run_redteam(..., iteration=)` → `ToolContext.current_iteration`); rows
written before this field existed read back as `-1`. Note `round` is the
*global* round number (`iteration * rounds + round_idx`), so `iteration` is now
explicit rather than only recoverable as `round // rounds`.

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
memo stays bounded (~200 words) regardless of round count — it does **not** grow
linearly. The sidecar is diagnostics-only — `render()` reflects only the latest memo.

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

### `agentic_redteam/probe_judge.py`
Wraps a pickled tuberlens probe. Lazily loads `tuberlens.model.LLMModel` on first
score (heavy import). Exposes `evaluate(conversation) → (score, predicts_positive)`
and `is_target_misprediction(predicts_positive)` so the tool layer can short-
circuit before invoking the (expensive) Claude judge when the probe is right.
The probe carries `pos_class_label`, `neg_class_label`, and `description` as
metadata, which everything downstream reads off the loaded object — never
duplicate these in code.

The model is loaded with `model_kwargs={"offload_buffers": True}` so that when
`device_map="auto"` offloads layers (e.g. gemma-3-27b), buffers offload too
instead of warning / risking OOM. `release()` drops the loaded LLM and runs
`gc.collect()` + `torch.cuda.empty_cache()` — call it when a phase is done with
the probe (the attacker does, after each rotation) so the next phase reloads onto
a clean GPU. This matters because every load re-infers the `device_map="auto"`
layer split from *free* GPU memory at load time, so a leftover copy forces the
next load into CPU/disk offload and ~5-10x slower forwards. See `cli._free_gpu`.

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
probe vs. judge label, judge reason, and per-message-truncated transcript) plus the
`prior_summary` into one user message and asks the judge — under a dedicated
`_SUMMARY_SYSTEM` prompt, not the classification one — to **rewrite and condense**
the prior memo with the new round's findings (merge duplicates, drop superseded
notes), capped at `_SUMMARY_WORD_BUDGET` (~200) words. So the memo is bounded, not
cumulative. Reuses the same `_call_anthropic` / `_call_openrouter` backends; returns
`prior_summary` unchanged for an empty round.

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
should try"). This is not stylistic: the original adversarially-phrased
`_SUMMARY_SYSTEM` drew refusals from the judge (`openai/gpt-5.1-chat` in every config)
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
  Persists every attempt with the judge's label included.
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
judge, store, run id, the currently-active round + attacker model, and the shared
`view_sampler`. The attacker module updates `current_attacker_model` and
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
  back `_render_submission_feedback` (probe vs. judge verdict, duplicate/error notes,
  running success count) followed by a freshly injected `_render_injected_view` —
  `view_past_attempts` rendered as text, `attacker.view_limit` rows, since the model
  can't call it. Assistant text is coerced to `""` before being appended so a
  null-content turn can't make the next request protocol-invalid. Respects
  `batch_target` (shared success counter) and `max_turns` (one submission per turn).
  This path is **openrouter-only** — `load_config` rejects `interface: prompt` with a
  `claude_sdk` model. No MCP server is constructed.

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
immediate signal) by all three drivers. Note the CLI's phase-marker resume path skips
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
per-session.

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
hits from disk, batch-computes only the misses, and writes each new row back as its
own blob), merges per side with `LabelledDataset.concatenate` (which pads +
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
JSONL keyed by `sha256(source messages + target label)` so accumulating successes
only pay for newly-seen conversations. `label_dataset` (LLM relabel) is intentionally
**not** ported — red-team data already carries judge labels.

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
- **JSONL is append-only and dedup-by-canonical-text.** `JsonlStore` rejects
  duplicate conversations silently — the agent's `submit_conversation`
  response surfaces `duplicate=True` so it can move on.
- **Tool functions return the `{"content": [{"type": "text", "text": ...}]}`
  shape exactly.** Anything else breaks the Claude Agent SDK's tool result
  streaming.
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
