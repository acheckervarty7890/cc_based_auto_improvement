# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`agentic_redteam` is a Claude-driven agentic red-teaming and iterative-retraining
toolkit for [tuberlens](https://github.com/blandfort/tuberlens) activation probes.
It replaces an earlier toolkit (`french-fries`) that used litellm-based attacker
and judge agents — both have been removed in favor of the **Claude Agent SDK** and
the Anthropic SDK.

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
The other key packages: `anthropic` (used by `llm_judge.py`),
`claude_agent_sdk` (used by `attacker.py`, `tools.py`), `pyyaml` (config
parser).

**Always invoke the venv's Python by absolute path** to avoid burning permission
prompts on `source .venv_claude/bin/activate` — the venv interpreter has its
`site-packages` baked into its own `sys.path`, so `source` adds nothing here:

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python -c "..."
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/pip install ...
```

`ANTHROPIC_API_KEY` must be set in the environment for both SDKs.

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

Full iterative loop:

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python \
  scripts/iterative_retrain.py configs/example_config.md \
  --iterations 3 --base-training-data path/to/base.jsonl
```

## Architecture

### `agentic_redteam/config.py`
Parses one markdown file with YAML frontmatter for runtime knobs and `# Attacker` /
`# Judge` sections for system prompts. Resolves all paths relative to the config
file. Frontmatter shape (see `configs/example_config.md`):

```yaml
attacker: { models: [...], max_turns, batch_target }
judge:    { model, max_tokens }
probe:    { path, threshold, error_type }   # error_type: false_positive | false_negative
output:   { jsonl_path, run_id }
```

`error_type` drives everything downstream: it's both the target misprediction the
attacker is searching for and the implicit *true* class label (`negative` for
`false_positive`, `positive` for `false_negative`) that the judge confirms.

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
A thin Claude wrapper using the `anthropic` SDK. **Unbiased classifier**: the
judge is asked to pick one of the two class labels on the conversation's own
merits, with no hint about which label the caller is hoping for. Expects
strict JSON output (`{"label": ..., "reason": ...}`); parses with code-fence
stripping + brace extraction fallback; normalizes case-insensitively against
the probe's pos/neg class labels. Returns `JudgeVerdict(label: str, reason: str)`.

### `agentic_redteam/tools.py`
Builds the MCP server and three tools via `claude_agent_sdk.create_sdk_mcp_server`
+ `@tool`:

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
Wraps the Claude Agent SDK loop. Critical configuration to keep the attacker
sandboxed:

```python
ClaudeAgentOptions(
    allowed_tools=allowed_tool_names(),                          # mcp__redteam_tools__*
    disallowed_tools=["Bash","Edit","Write","Read","Glob",...],  # block all built-ins
    permission_mode="bypassPermissions",                         # auto-approve MCP calls
    setting_sources=[],                                          # don't auto-load filesystem CLAUDE.md
)
```

Rotation works by instantiating a fresh `ClaudeSDKClient` per model (with a new
`ToolContext` round/model setup), all writing to the same shared `JsonlStore`.
The kickoff prompt is one user message; the agent then runs autonomously until
it returns a `ResultMessage`.

### `agentic_redteam/retrain.py`
Converts successful JSONL records into a tuberlens `LabelledDataset` — labelled
with the canonical enum value (`"positive"` / `"negative"`) corresponding to the
*true* class for the run's `error_type`. Optionally concatenates with a base
training dataset (`LabelledDataset.load_from(path, pos_class_label, neg_class_label)`)
via `LabelledDataset.concatenate([base, redteam])`. `_infer_probe_spec` walks
the loaded probe's `_classifier.probe_architecture` (or the SklearnProbe shape,
or the difference-of-means/LDA shape) to reconstruct the `ProbeSpec` so the
retrained probe matches the original's architecture and hyperparameters.

### `agentic_redteam/cli.py`
Two entry points: `run_redteam_main` (one round) and `iterative_retrain_main`
(red-team → retrain → next round). The iterative loop rewrites
`config.probe.path` between rounds so each round attacks the freshest probe.

## Conventions to preserve

- **Probe metadata is the source of truth.** Don't pass `pos_class_label` /
  `neg_class_label` / `description` separately — read them off the loaded probe.
- **The attacker must never get filesystem/shell access.** Always carry both
  `allowed_tools=` and `disallowed_tools=` plus `setting_sources=[]` when
  constructing `ClaudeAgentOptions`. Adding a new tool means appending to
  `allowed_tool_names()` *and* updating its definition in `tools.py`.
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
