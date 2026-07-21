#!/usr/bin/env bash
# Run the clone-test arms across several seeds and score them.
#   arm0 = recency baseline           (view_reshuffle: false)
#   arm1 = periodic reshuffle         (view_reshuffle: true)
#   arm2 = submit-time near-dup guard (block-only), on top of the recency baseline
#   arm3 = guard + broadcast          (also shows rejections to all sessions)
# Attacker+judge prompts are IDENTICAL across arms; each arm changes exactly one
# knob group, so arm{K-1}-vs-arm{K} isolates that knob (ladder: none→block→broadcast).
#
# Each (arm,seed) writes to its OWN jsonl (arm{arm}_seed{seed}.jsonl) so JsonlStore
# never preloads/dedups across seeds.
#
#   export OPENROUTER_API_KEY=...       # required (attacker, judge, 1B probe all via OpenRouter)
#   bash scripts/run_clone_seeds.sh                   # default: re-runs arm2+arm3 (autojunk + race fix); arm0/1 reused
#   ARMS="0 1 2 3" bash scripts/run_clone_seeds.sh    # full fresh sweep of all arms
#   SEEDS="42 43 44 45" bash scripts/run_clone_seeds.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv_claude/bin/python
BASE=data/hs_ls_200.jsonl
read -r -a SEEDS <<< "${SEEDS:-42 43 44}"
read -r -a ARMS  <<< "${ARMS:-2 3}"

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first}"
mkdir -p results_clone_test configs/_clone_seed_tmp results_clone_test/logs

# Tee ALL screen output (stdout+stderr) to a timestamped master log while still
# printing to the terminal. Override the path with LOG=... .
stamp="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-results_clone_test/logs/run_${stamp}.log}"
exec > >(tee -a "$LOG") 2>&1
echo "# logging to $LOG  (arms=${ARMS[*]}  seeds=${SEEDS[*]})"

for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    src="configs/clone_test_arm${arm}.md"
    tmp="configs/_clone_seed_tmp/arm${arm}_seed${seed}.md"
    # absolute path: tmp config lives in configs/_clone_seed_tmp/, so a relative
    # ../results_clone_test would resolve to configs/results_clone_test — use abs.
    out="$(pwd)/results_clone_test/arm${arm}_seed${seed}.jsonl"
    # per-seed config: unique jsonl_path + run_id, everything else untouched
    sed -e "s#jsonl_path: .*#jsonl_path: ${out}#" \
        -e "s#run_id: .*#run_id: clone_A${arm}_s${seed}#" \
        "$src" > "$tmp"
    # start fresh so the store doesn't append to a prior run's file
    rm -f results_clone_test/arm${arm}_seed${seed}.jsonl \
          results_clone_test/arm${arm}_seed${seed}.*.jsonl
    runlog="results_clone_test/logs/arm${arm}_seed${seed}_${stamp}.log"
    echo ">>> arm${arm}  seed${seed}  (log: $runlog)"
    # per-run log: tee this run's output to its own file AND up to the master log.
    "$PY" scripts/iterative_retrain.py "$tmp" \
      --iterations 1 --base-training-data "$BASE" \
      --probe-out-dir "probes/clone_test2_arm${arm}_seed${seed}" --seed "$seed" \
      2>&1 | tee "$runlog"
  done
done

echo
echo "########## per-seed clone_rate (all arms present) ##########"
for seed in "${SEEDS[@]}"; do
  files=()
  for arm in 0 1 2 3; do
    f="results_clone_test/arm${arm}_seed${seed}.jsonl"
    [ -f "$f" ] && files+=("$f")
  done
  echo "===================== SEED ${seed} ====================="
  "$PY" scripts/clone_rate.py "${files[@]}" --tau 0.7 0.8 0.9 --min-cluster 2 --prefix 600 --no-equal-n
done
