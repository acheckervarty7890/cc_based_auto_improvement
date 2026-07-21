#!/usr/bin/env bash
#
# Sweep the iterative-retrain pipeline over several base-data fractions, one
# single-iteration run per fraction, reusing all the configuration from
# configs/llama1b_hu_harm_prompt.md.
#
# Every run gets fully isolated result outputs (red-team JSONL, comparison CSV,
# probes, stdout log) so nothing overwrites or conflicts. The two activation
# caches are deliberately *shared*, which is correct because:
#   - eval activations depend only on the model + eval splits + seed (NOT the
#     base fraction), so all runs hit the same cache and skip recompute.
#   - base activations embed the fraction in their cache-file key
#     (_base_activation_cache_paths), so one dir holds every fraction's base
#     activations without collision.
#
# Runs are sequential (one GPU model load at a time). A failure in one fraction
# is reported but does not abort the rest of the sweep.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="$REPO/.venv_claude/bin/python"
BASE_CONFIG="$REPO/configs/llama1b_hu_harm_prompt.md"
BASE_DATA="$REPO/data/hu_ha_200.jsonl"
EVAL_DIR="$REPO/eval_dataset_hu_ha"

FRACTIONS=(0.2 0.4 0.6 0.8)

RESULTS_ROOT="$REPO/results_hu_harm_prompt_llama1b_fractions"
GEN_CONFIG_DIR="$RESULTS_ROOT/_configs"
LOG_DIR="$RESULTS_ROOT/_logs"
EVAL_ACTS="$RESULTS_ROOT/eval_activations"      # shared (fraction-independent)
BASE_ACTS="$RESULTS_ROOT/base_activations"      # shared (fraction is in cache key)
mkdir -p "$GEN_CONFIG_DIR" "$LOG_DIR" "$EVAL_ACTS" "$BASE_ACTS"

declare -a FAILED=()

for frac in "${FRACTIONS[@]}"; do
  tag="${frac/./_}"                              # 0.2 -> 0_2 (filesystem-friendly)
  out_dir="$RESULTS_ROOT/frac${tag}"
  probe_dir="$REPO/probes/15July_hh_llama1b_frac${tag}"
  gen_config="$GEN_CONFIG_DIR/llama1b_hu_harm_prompt_frac${tag}.md"
  log="$LOG_DIR/15July_run_hh_llama1b_frac${tag}.log"
  mkdir -p "$out_dir" "$probe_dir"

  # Per-fraction config: copy the base config and rewrite only its output: block
  # to absolute, fraction-isolated paths. Absolute paths mean the generated
  # config's location doesn't affect how they resolve.
  cp "$BASE_CONFIG" "$gen_config"
  sed -i \
    -e "s|^  jsonl_path:.*|  jsonl_path: ${out_dir}/probing.jsonl|" \
    -e "s|^  run_id:.*|  run_id: llama1b_frac${tag}|" \
    -e "s|^  comparison_csv:.*|  comparison_csv: ${out_dir}/comparison.csv|" \
    -e "s|^  activations_cache_dir:.*|  activations_cache_dir: ${EVAL_ACTS}|" \
    -e "s|^  base_activation_cache_dir:.*|  base_activation_cache_dir: ${BASE_ACTS}|" \
    "$gen_config"

  echo "===== base-data-fraction ${frac}  →  ${log} ====="
  start=$(date +%s)
  "$PY" scripts/iterative_retrain.py "$gen_config" \
    --iterations 1 \
    --base-training-data "$BASE_DATA" \
    --base-data-fraction "$frac" \
    --probe-out-dir "$probe_dir" \
    --eval --eval-dataset-dir "$EVAL_DIR" \
    > "$log" 2>&1
  status=$?
  elapsed=$(( $(date +%s) - start ))
  if [ "$status" -eq 0 ]; then
    echo "  fraction ${frac} done in ${elapsed}s → ${out_dir}/comparison.csv"
  else
    echo "  fraction ${frac} FAILED (exit ${status}) after ${elapsed}s — see ${log}"
    FAILED+=("$frac")
  fi
done

echo
if [ "${#FAILED[@]}" -eq 0 ]; then
  echo "All ${#FRACTIONS[@]} fraction runs complete. Results under $RESULTS_ROOT"
else
  echo "Done with failures for fractions: ${FAILED[*]}. See logs in $LOG_DIR"
  exit 1
fi
