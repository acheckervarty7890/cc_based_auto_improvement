# agentic_redteam — generate → score → retrain → guide

Training-set curation for [tuberlens](https://github.com/blandfort/tuberlens)
activation probes, driven by two LLMs:

- a **generator** that writes batches of labelled conversations for the probe's
  concept (it labels its own samples, half per class, from the probe's `description`);
- a **judge** that reads how each batch moved the probe's dev-set AUROC and steers the
  generator toward what helped and away from what is exhausted.

The probe itself decides what gets kept: a batch enters the training set only if a
probe trained on it scores higher on the dev set than the current probe.

Both LLMs can be driven by `claude_sdk` (Anthropic Python SDK) or `openrouter` (the
`openai` SDK pointed at [OpenRouter](https://openrouter.ai/)); generator models may
mix providers. No tools, MCP or shell are involved — every call is a single chat
completion.

## The loop

For a current probe *P* with mean dev AUROC *A*, one iteration does:

1. **Directions.** Take the `n_batches` directions written for this iteration — the
   judge's, from the previous iteration; at iteration 0 the generator proposes them.
2. **Generate.** `n_batches` concurrent generator calls, batch *k* under direction *k*
   by `models[k % len(models)]`, each returning `batch_size` samples (`batch_size/2` per
   class). Over-long samples (`max_sample_tokens`, counted with the probe's tokenizer),
   malformed ones and duplicates of anything generated earlier in the run are dropped;
   a short batch gets up to `max_retries` in-context top-up asks.
3. **Warm the activation cache** for every new sample in one extraction-model load.
4. **Score each batch on its own.** Train a candidate probe on
   base ∪ accepted-so-far ∪ batch (pure cache hit — no model load), read its per-split
   dev AUROC, Δ = mean − *A*. Δ > `min_auroc_gain` ⇒ **accepted**;
   |Δ| ≤ `exhausted_gain` ⇒ flagged **exhausted** for the judge; Δ < 0 ⇒ harmful.
5. **Union retrain.** Train `probe_iter{i+1}.pkl` on base ∪ every accepted batch so
   far; its dev AUROC is the next baseline. Nothing accepted ⇒ the probe carries over.
6. **Judge.** Every batch (direction, sample excerpts, per-split Δ, verdict) goes to
   the judge, which rewrites a bounded rolling memo and writes the next iteration's
   `n_batches` directions.
7. Optional `--eval` on the eval splits.

`loop.iterations` (or `--iterations`) repeats this. Everything is resumable at batch
granularity from the sidecars in `output.run_dir`.

## Setup

```bash
# Project venv at .venv_claude/ (tuberlens installed editable at .venv_claude/src/tuberlens/)
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/pip install -e .

export ANTHROPIC_API_KEY=sk-ant-...      # any `provider: claude_sdk` section
export OPENROUTER_API_KEY=sk-or-...       # any `provider: openrouter` section
```

## Run

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python scripts/iterative_generate.py \
  configs/example_generate.md \
  --base-training-data data/highstakes_llama70b_50.jsonl \
  --probe-out-dir probes/generate_example \
  --eval --eval-dataset-dir eval_sets/highstakes      # --eval is optional
```

`--base-training-data` trains the initial probe (unless `probe.path` warm-starts one)
and is part of every retrain. A **dev set is required** (`validation.dev_data` or
`--dev-data`): it is both the fit's early-stopping validation set and the set every
batch's ΔAUROC is read on, and must be disjoint from the eval splits. The three shipped
dev sets are under `dev_samples/<concept>/`, the eval splits under `eval_sets/<concept>/`.

Outputs, per run:

| where | what |
| --- | --- |
| `<probe-out-dir>/probe_iter{N}.pkl` | the probe iteration N starts from (`probe_iter0` = initial) |
| `<probe-out-dir>/candidates/probe_iter{i}_batch{k}.pkl` | the per-batch candidate probes |
| `<run_dir>/batches.jsonl` | every batch: direction, samples, AUROC before/after, Δ, accepted/exhausted |
| `<run_dir>/guidance.jsonl` | the judge's memo + directions per iteration |
| `<run_dir>/auroc_history.csv` | one row per (iteration, batch): the ΔAUROC ledger |
| `<run_dir>/accepted_iter{N}.jsonl` | the accepted samples `probe_iter{N}` was trained with |
| `<run_dir>/runlog.jsonl` | lifecycle / error events |

## Config

A markdown file: YAML frontmatter + `# Generator` and `# Judge` sections holding the
two system prompts. See `configs/example_generate.md` for every key; the essentials:

```yaml
generator:
  provider: openrouter
  models: [meta-llama/llama-3.3-70b-instruct]   # batch k → models[k % len]
  n_batches: 5          # n
  batch_size: 20        # m (even)
  concurrency: 5
  max_sample_tokens: 1024
judge:
  provider: openrouter
  model: openai/gpt-5.1-chat
  memo_word_budget: 400
probe:                  # from-scratch fields; or `path:` to warm-start
  model: meta-llama/Llama-3.2-1B-Instruct
  layer: 8
  pos_class_label: high-stakes
  neg_class_label: low-stakes
  description: ...
loop:
  iterations: 3
  min_auroc_gain: 0.0
  exhausted_gain: 0.002
validation:
  dev_data: ../dev_samples/highstakes
output:
  run_dir: ../results/my_run
```

## Layout

```
src/agentic_redteam/
  config.py            # markdown+YAML config → LoopRunConfig
  generator.py         # batch generation + direction proposal (claude_sdk / openrouter)
  llm_judge.py         # rolling memo + next-iteration directions from scored batches
  json_extract.py      # forgiving JSON recovery from LLM replies
  persistence.py       # GeneratedSample / BatchRecord / BatchStore / GuidanceStore / RunLogger
  retrain.py           # train_initial_probe, retrain_probe (samples in memory), dev AUROC scoring,
                       #   activation caches (base blob, per-sample, dev blob), fit staging
  ensemble.py          # score-averaging deep ensembles (EnsembleProbe)
  evaluation.py        # evaluate_probe on eval splits (tuberlens get_performances)
  model_loading.py     # truncated extraction-model loader (layers 0..probe.layer)
  token_budget.py      # exact token counting against the 1024-token activation cap
  circuit_breaker.py   # process-wide OpenRouter outage detection
  openrouter_client.py # openai.OpenAI/AsyncOpenAI factory pointed at OpenRouter
  kaggle_activations.py# precomputed eval/dev activations from Kaggle
  cli.py               # iterative_generate_main
scripts/
  iterative_generate.py       # entry point
  verify_generation_loop.py   # end-to-end check with fake LLMs (--mode fake | real)
  verify_fit_staging.py, verify_ensemble_fusion.py   # pin the retrain fast paths
  extract_eval_activations.py, publish_kaggle_eval_activations.py
configs/example_generate.md
data/                  # base training sets
dev_samples/<concept>/ # dev splits (validation + ΔAUROC)
eval_sets/<concept>/   # eval splits
```

## Verifying

```bash
# Loop bookkeeping with fake LLMs and fake fits — no GPU, model or key needed
.venv_claude/bin/python scripts/verify_generation_loop.py --mode fake
# Same, with real llama-1b fits on a 50-row base set and a small dev cut
.venv_claude/bin/python scripts/verify_generation_loop.py --mode real
```
