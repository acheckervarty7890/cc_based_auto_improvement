#!/usr/bin/env bash
set -e

# Guard runs (arm2 near-dup guard enabled in both configs). All outputs are
# repointed to *_guard dirs / 13July_* logs so the 8July results
# (probes/8July_hs_concept, results_hs_prompt/, 8July_run*) are NOT overwritten.

#.venv_claude/bin/python   scripts/iterative_retrain.py configs/gemma27_hs_prompt.md   --iterations 3  --base-training-data data/hs_ls_200.jsonl  --probe-out-dir probes/13July_hs_guard   --eval --eval-dataset-dir eval_datasets > 13July_run_hs_guard 2>&1

#.venv_claude/bin/python   scripts/iterative_retrain.py configs/gemma27_hu_harm_prompt.md   --iterations 3  --base-training-data data/hu_ha_200.jsonl  --probe-out-dir probes/15July_hh_guard   --eval --eval-dataset-dir eval_dataset_hu_ha > 15July_run_hh_guard 2>&1

# Grok-contrastive ablation (preprocessing.model = x-ai/grok-4.20; everything else identical
# to the gpt-5.1-chat contrastive runs). Base data is the 50-sample hu_harm_llama70b_50 set.
# Outputs go to fresh *_grok dirs and probes/, and logs to logs/, so nothing collides with the
# gpt-5.1-chat contrastive results. Fresh --probe-out-dir gives each a fresh contrastive_cache.jsonl
# so grok actually regenerates pairs (the cache key does not include the model).

.venv_claude/bin/python   scripts/iterative_retrain.py configs/llama70b_hu_harm_llama70b50_grok.md   --iterations 3  --base-training-data data/hu_harm_llama70b_50.jsonl  --probe-out-dir probes/llama70b50_grok_llama70b   --eval --eval-dataset-dir eval_dataset_hu_ha > logs/run_hh_llama70b50_grok_llama70b.log 2>&1

.venv_claude/bin/python   scripts/iterative_retrain.py configs/gptoss120b_hu_harm_llama70b50_grok.md   --iterations 3  --base-training-data data/hu_harm_llama70b_50.jsonl  --probe-out-dir probes/llama70b50_grok_gptoss120b   --eval --eval-dataset-dir eval_dataset_hu_ha > logs/run_hh_llama70b50_grok_gptoss120b.log 2>&1
