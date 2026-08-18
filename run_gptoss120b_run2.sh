#!/usr/bin/env bash
set -e

# Second run of the gpt-oss-120b attacker arm ONLY, fully isolated from the first run.
# Every write target is repointed to *_run2:
#   configs/gptoss120b_hu_harm_llama70b50_run2.md  -> results_hu_harm_llama70b50_gptoss120b_run2/
#   --probe-out-dir probes/llama70b50_gptoss120b_run2
#   logs/run_hh_llama70b50_gptoss120b_run2.log
# Nothing under results_hu_harm_llama70b50_gptoss120b/, probes/llama70b50_gptoss120b/
# or logs/run_hh_llama70b50_gptoss120b.log is read or written.
#
# A fresh --probe-out-dir also matters for two reasons beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it
#     would make the CLI skip red-teaming entirely and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, so contrastive pairs are
#     regenerated rather than replayed from the first run.
#
# Usage:
#   mkdir -p logs
#   nohup bash run_gptoss120b_run2.sh > logs/run_gptoss120b_run2.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# Refuse to clobber a previous run2.
for p in results_hu_harm_llama70b50_gptoss120b_run2 probes/llama70b50_gptoss120b_run2; do
    [ -e "$p" ] && { echo "ERROR: $p already exists â move it aside or bump the suffix." >&2; exit 1; }
done

.venv_claude/bin/python   scripts/iterative_retrain.py configs/gptoss120b_hu_harm_llama70b50_run2.md   --iterations 3  --base-training-data data/hu_harm_llama70b_50.jsonl  --probe-out-dir probes/llama70b50_gptoss120b_run2   --eval --eval-dataset-dir eval_sets/hu_ha > logs/run_hh_llama70b50_gptoss120b_run2.log 2>&1