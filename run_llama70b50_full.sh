#!/usr/bin/env bash
set -e

# Baseline (gpt-5.1-chat contrastive) counterparts of the grok runs in some.sh.
# Same base data (hu_harm_llama70b_50), same eval splits, same iterations — the
# only difference vs the *_grok configs is preprocessing.model.
#
# Run sequentially, NOT in parallel: the two configs deliberately share
#   base_activation_cache_dir: results_hu_harm_llama70b50/llama1b_base_activations
#   activations_cache_dir:     results_hu_harm_prompt_llama1b/llama1b_eval_activations
# so run 1 populates them and run 2 hits the cache. In parallel they would race
# on the same .pt files, and two probe LLMs resident at once forces the
# device_map="auto" split into CPU/disk offload.
#
# Usage:  nohup bash run_llama70b50_full.sh > logs/run_llama70b50_full.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# --- Llama-3.3-70B attacker -------------------------------------------------
.venv_claude/bin/python   scripts/iterative_retrain.py configs/llama70b_hu_harm_llama70b50.md   --iterations 3  --base-training-data data/hu_harm_llama70b_50.jsonl  --probe-out-dir probes/llama70b50_llama70b   --eval --eval-dataset-dir eval_dataset_hu_ha > logs/run_hh_llama70b50_llama70b.log 2>&1

# --- gpt-oss-120b attacker (attacker ablation) ------------------------------
.venv_claude/bin/python   scripts/iterative_retrain.py configs/gptoss120b_hu_harm_llama70b50.md   --iterations 3  --base-training-data data/hu_harm_llama70b_50.jsonl  --probe-out-dir probes/llama70b50_gptoss120b   --eval --eval-dataset-dir eval_dataset_hu_ha > logs/run_hh_llama70b50_gptoss120b.log 2>&1
