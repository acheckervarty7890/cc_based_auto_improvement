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
5. The retraining script converts JSONL successes into a tuberlens
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

Validation is always derived by splitting the training data via tuberlens'
`create_train_test_split` (`--test-size`, default 0.2; optional `--split-field`) —
there is no external validation-file flag. `--seed` (default 42) seeds the
train/val split and the (reproducible) eval subsampling; `--eval-max-samples`
(default 100, `0` = full split) sets the balanced subsample size per eval split.
When the config has a `preprocessing:` section, red-team successes are run through
`filter_dataset` + `generate_contrastive_dataset` before each retrain.

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
  persistence_from_last_rounds: int   # view_past_attempts window (default: all)
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
output:   { jsonl_path, run_id }
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
success, attacker_model, run_id, round, error_type, pos_class_label,
neg_class_label}` — `judge_label` is the class label the judge picked
(human-readable, e.g. "high-stakes"), or `""` if the judge response was
unparseable.

### `agentic_redteam/probe_judge.py`
Wraps a pickled tuberlens probe. Lazily loads `tuberlens.model.LLMModel` on first
score (heavy import). Exposes `evaluate(conversation) → (score, predicts_positive)`
and `is_target_misprediction(predicts_positive)` so the tool layer can short-
circuit before invoking the (expensive) Claude judge when the probe is right.
The probe carries `pos_class_label`, `neg_class_label`, and `description` as
metadata, which everything downstream reads off the loaded object — never
duplicate these in code.

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
- `view_past_attempts(only_successful, limit)` — reads the shared store so
  later attacker models in a rotation can learn from earlier ones.
- `get_probe_info()` — returns probe metadata.

A `ToolContext` is the closure shared by all three tools — it holds the probe,
judge, store, run id, and the currently-active round + attacker model. The
attacker module updates `current_attacker_model` and `current_round` before
each model run so JSONL rows attribute correctly.

Tool naming for the allow-list: `mcp__redteam_tools__<tool>`. The MCP server
name (`redteam_tools`) is exported as `MCP_SERVER_NAME`.

### `agentic_redteam/attacker.py`
Dispatcher + rotation. For each `AttackerModel` in `config.attacker.models`,
picks the driver by `model.provider`:

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

A fresh `ToolContext` is built per model run (with round/model labels set), but
all runs share the same `JsonlStore` so dedup and the success counter persist
across rotation.

### `agentic_redteam/retrain.py`
Converts successful JSONL records into a tuberlens `LabelledDataset` — labelled
with the canonical enum value (`"positive"` / `"negative"`) corresponding to the
*true* class for the run's `error_type`. Optionally concatenates with a base
training dataset (`LabelledDataset.load_from(path, pos_class_label, neg_class_label)`)
via `LabelledDataset.concatenate([base, redteam])`. By default `_infer_probe_spec`
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

Both `retrain_probe` and `train_initial_probe` derive the validation set by
splitting the (post-concatenation) training data via tuberlens'
`create_train_test_split(test_size, split_field)` — there is no external
validation-file path. When `retrain_probe` is given a `preprocessing` config, the
red-team successes are first run through `_build_redteam_dataset`, which mirrors the
collation step of tuberlens' pipeline applied to the "extra" data: `filter_dataset`
(drop confounders) then `generate_contrastive_dataset` (add opposite-class pairs),
keyed off the probe's pos/neg labels. The contrastive pairs are cached to disk
(`contrastive_cache_path`) so successes accumulated across iterations aren't
re-generated. With no `preprocessing`, the plain `_records_to_labelled_dataset`
path (judge label → canonical class) is used unchanged.

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
max_samples=100, seed=42)` scores one probe on local eval split JSONLs (default
`DEFAULT_EVAL_SPLITS = ["anthropic", "mts"]`) via tuberlens `get_performances`,
returning a per-split DataFrame. It calls `seed_everything(seed)` (ported from
tuberlens) and then balances each split to `max_samples` via
`subsample_balanced_subset(n_per_class=max_samples // 2)` (`max_samples=None` → full
split). Seeding before subsampling makes the subset identical across every probe
eval — which is what keeps the path-keyed activation cache correct, since
`get_activations` reloads by file path **without** checking the inputs match. The
cache filename embeds `max_samples`/`seed` (`acts_n{N}_seed{S}.pt`) so a different
subsample config can't silently reuse stale activations.

### `agentic_redteam/cli.py`
Two entry points: `run_redteam_main` (one round against an existing probe) and
`iterative_retrain_main`. The latter runs the full pipeline: **(1)** obtain the
initial probe — warm-start from `config.probe.path` if it points to an existing
file, else `train_initial_probe` from `--base-training-data`; **(2)** red-team it
across all `error_types`; **(3)** `retrain_probe` on base data ∪ successes;
**(4)** optionally `evaluate_probe` (gated by `--eval`); then repeat 2–4 for
`--iterations` n. It rewrites `config.probe.path` to the freshest probe before
each round, and (with `--eval`) writes `results/iter_run_comparison.csv`. It calls
`seed_everything(--seed)` up front, threads `config.preprocessing` +
`<probe-out-dir>/contrastive_cache.jsonl` + `--test-size` / `--split-field` into the
train/retrain calls, and passes `--eval-max-samples` / `--seed` into `evaluate_probe`.

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
- **Use `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`** as
  the canonical model IDs for the rotation. Don't append date suffixes to opus
  or sonnet — only Haiku 4.5 currently requires the dated form.
