# agentic_redteam

Claude-driven agentic red-teaming for [tuberlens](https://github.com/blandfort/tuberlens) probes,
plus an iterative retraining loop.

## What it does

1. Loads a pickled `tuberlens` probe and a markdown config.
2. Spins up a Claude attacker via the **Claude Agent SDK**, exposing three in-process
   MCP tools — `submit_conversation`, `view_past_attempts`, `get_probe_info`.
3. The attacker iteratively submits candidate conversations; each is scored by the
   probe **and** a Claude-based "human-style" judge. A submission counts as a
   successful red-team find only if the probe is wrong *and* the judge confirms
   the true class label.
4. Every attempt is appended to a JSONL log with run/round/model metadata.
5. Optionally retrains a fresh probe (same architecture and metadata as the
   original) using the red-team successes — labelled as the *true* class — merged
   with an optional base training dataset, and repeats.

The attacker config supports rotating across multiple Claude models
(e.g. `claude-sonnet-4-6`, `claude-opus-4-7`, `claude-haiku-4-5-20251001`); each
runs sequentially against the same shared JSONL log.

## Setup

```bash
# 1. Project venv at .venv_claude/ (already has tuberlens installed editable
#    at .venv_claude/src/tuberlens/). Install this package on top:
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/pip install -e .

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
```

The Claude Agent SDK and `anthropic` SDK both read `ANTHROPIC_API_KEY` from the env.

## Layout

```
src/agentic_redteam/
  config.py         # parse markdown+YAML-frontmatter config
  persistence.py    # Conversation/Message + JSONL store with dedup
  probe_judge.py    # wraps a tuberlens probe → score(Conversation)
  llm_judge.py      # Claude judge: confirms human label
  tools.py          # in-process MCP tools the attacker can call
  attacker.py       # ClaudeSDKClient loop with model rotation
  retrain.py        # JSONL successes → tuberlens.training.train_probe
  cli.py            # entry points: redteam, iterative-retrain
configs/
  example_config.md # frontmatter + Attacker/Judge prompts
scripts/
  run_redteam.py
  iterative_retrain.py
```

## Usage

One round of red-teaming:

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python \
  scripts/run_redteam.py configs/example_config.md
```

Full iterative loop (red-team → retrain → red-team the new probe → ...):

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python \
  scripts/iterative_retrain.py configs/example_config.md \
    --iterations 3 \
    --base-training-data path/to/base_train.jsonl \
    --validation-data path/to/val.jsonl \
    --probe-out-dir probes/
```

## Config file shape

YAML frontmatter for runtime knobs, then `# Attacker` and `# Judge` sections for
the system prompts. See `configs/example_config.md` for the canonical example.
