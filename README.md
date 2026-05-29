# agentic_redteam

Agentic red-teaming and iterative-retraining toolkit for
[tuberlens](https://github.com/blandfort/tuberlens) activation probes.

The attacker and judge can each be driven by one of two providers, picked
per-section in the config:

- **`claude_sdk`** — Anthropic's [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
  (attacker) and Anthropic Python SDK (judge). Tools are exposed to the
  attacker through an in-process MCP server.
- **`openrouter`** — the official `openai` SDK pointed at
  [OpenRouter](https://openrouter.ai/), giving access to Claude, GPT, Gemini,
  Llama, Mistral, DeepSeek, etc. through one OpenAI-compatible endpoint. The
  attacker uses native OpenAI tool calls; the judge uses chat completions. No
  MCP machinery is used on this path.

Providers can be **mixed within a single attacker rotation** (e.g. one
`claude_sdk` model and several `openrouter` models in the same run), and the
two SDKs are imported lazily, so a config that only uses one provider does not
need the other installed.

## What it does

1. Loads a pickled `tuberlens` probe and a markdown config.
2. Spins up a Claude/OpenRouter **attacker** inside a tool-use loop, exposing
   exactly three in-process tools — `submit_conversation`,
   `view_past_attempts`, `get_probe_info`. The attacker gets **no** shell,
   filesystem, or web access.
3. The attacker iteratively submits candidate conversations; each is scored by
   the probe **and** independently classified by a Claude-based, "human-style"
   **judge** (the judge picks one of the two class labels on the conversation's
   own merits, with no hint about what we are hoping for). A submission counts
   as a successful red-team find only if the probe's predicted label and the
   judge's label **disagree** in the direction matching the configured
   `error_type` (e.g. for `false_positive`, the probe predicts the positive
   class while the judge picks the negative class).
4. Every attempt is appended to a JSONL log with run/round/model metadata and
   the judge's label.
5. Optionally retrains a fresh probe (same architecture and metadata as the
   original, unless overridden) using the red-team successes — labelled with
   the **judge's** class — merged with a base training dataset, then repeats.
   When the config has a `preprocessing:` section, successes are first run
   through `filter_dataset` + `generate_contrastive_dataset` before each
   retrain. With `--eval`, every probe is also scored on local eval splits and
   a cross-round comparison CSV is written.

The attacker config supports rotating across multiple models — e.g.
`claude-sonnet-4-6`, `claude-opus-4-7`, `claude-haiku-4-5-20251001`, plus any
OpenRouter model — each running sequentially against the same shared JSONL log.

## Setup

```bash
# 1. Project venv at .venv_claude/ (already has tuberlens installed editable
#    at .venv_claude/src/tuberlens/). Install this package on top:
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/pip install -e .

# 2. Set the API key(s) for whichever provider(s) your config uses
export ANTHROPIC_API_KEY=sk-ant-...      # needed for any `provider: claude_sdk` section
export OPENROUTER_API_KEY=sk-or-...       # needed for any `provider: openrouter` section
```

Optional OpenRouter env vars: `OPENROUTER_BASE_URL` (default
`https://openrouter.ai/api/v1`), `OPENROUTER_HTTP_REFERER`,
`OPENROUTER_APP_TITLE` (sent as `HTTP-Referer` / `X-Title` for dashboard
attribution).

## Layout

```
src/agentic_redteam/
  config.py            # parse markdown+YAML-frontmatter config (attacker/judge/probe/preprocessing/output)
  persistence.py       # Conversation/Message + JSONL store with dedup
  probe_judge.py       # wraps a tuberlens probe → (score, predicts_positive)
  llm_judge.py         # unbiased LLM judge (claude_sdk or openrouter)
  openrouter_client.py # openai.OpenAI/AsyncOpenAI factory pointed at OpenRouter
  tools.py             # the three tools, as both MCP (claude_sdk) and OpenAI-format (openrouter)
  attacker.py          # provider dispatcher + model rotation
  preprocessing.py     # filter_dataset + generate_contrastive_dataset (collation-style)
  retrain.py           # train initial probe / retrain from JSONL successes (+ optional preprocessing)
  evaluation.py        # score a probe on local eval splits
  cli.py               # entry points: run_redteam, iterative_retrain
configs/
  example_config.md            # claude_sdk example: frontmatter + Attacker/Judge prompts
  example_config_openrouter.md # openrouter example
  dry_run_config.md            # minimal config for a quick smoke run
scripts/
  run_redteam.py
  iterative_retrain.py
```

## Usage

One round of red-teaming against an existing probe:

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python \
  scripts/run_redteam.py configs/example_config.md
```

Full iterative loop (train initial probe → red-team → retrain → optional eval,
n times):

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python \
  scripts/iterative_retrain.py configs/example_config.md \
    --iterations 3 \
    --base-training-data path/to/base_train.jsonl \
    --probe-out-dir probes/ \
    --eval --eval-dataset-dir eval_datasets   # --eval is optional
```

`--base-training-data` is **required**: it trains the initial probe (unless
`probe.path` in the config warm-starts from an existing one) and is
concatenated with red-team successes on every retrain. The validation set is
always derived by splitting the training data (`--test-size`, default 0.2;
optional `--split-field`) — there is no external validation-file flag.
`--probe-arch` overrides the architecture (bare flag → `linear_then_softmax`;
omit on retrains to inherit the current probe's). `--seed` (default 42) seeds
the train/val split and the reproducible eval subsampling; `--eval-max-samples`
(default 100, `0` = full split) sets the balanced subsample per eval split.

## Config file shape

A single markdown file: YAML frontmatter for runtime knobs
(`attacker` / `judge` / `probe` / optional `preprocessing` / `output`), then
`# Attacker` and `# Judge` sections holding the system prompts. Each attacker
model entry is either a bare string (inherits `attacker.provider`) or a dict
`{name, provider}` to mix providers in one rotation. `probe.error_type` drives
the whole loop and may be a single value or a list
(`[false_positive, false_negative]`). See `configs/example_config.md`,
`configs/example_config_openrouter.md`, and `configs/dry_run_config.md` for
canonical examples.
