#!/usr/bin/env bash
set -e

# Instruction-following concept: llama70b attacker -> llama-1b probe, fully isolated run.
# Trains the initial probe from scratch on the generated dataset, red-teams it, retrains,
# and evals against the eval_sets/instructions/ splits. Every write target is namespaced to
# this concept so nothing from the hu_harm / hs runs is read or written:
#   configs/llama70b_instructions_llama1b.md -> results_instructions_llama70b50/  (per config output:)
#   --probe-out-dir probes/instructions_llama70b50
#   logs/run_instructions_llama70b50.log
#
# A fresh --probe-out-dir matters for two reasons beyond overwriting:
#   - the dir holds redteam_done_iter*_*.marker resume markers; reusing a populated one
#     would make the CLI skip red-teaming entirely and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, so contrastive pairs are regenerated
#     rather than replayed from an earlier run.
#
# Usage:
#   mkdir -p logs
#   nohup bash run_instructions_llama70b50.sh > logs/run_instructions_llama70b50.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# Refuse to clobber an existing run.
for p in results_instructions_llama70b50 probes/instructions_llama70b50; do
    [ -e "$p" ] && { echo "ERROR: $p already exists — move it aside or bump the suffix." >&2; exit 1; }
done

.venv_claude/bin/python  scripts/iterative_retrain.py configs/llama70b_instructions_llama1b.md  --iterations 3  --base-training-data data/instructions_llama70b_50.jsonl  --probe-out-dir probes/instructions_llama70b50  --eval --eval-dataset-dir eval_sets/instructions > logs/run_instructions_llama70b50.log 2>&1
