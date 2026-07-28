#!/usr/bin/env bash
set -e

# deepseek-v4-flash attacker — cross-iteration-memo ablation. Two arms, IDENTICAL in
# every knob except attacker.cross_iteration_memos, run sequentially and fully isolated:
#
#   ARM A (nomemo): configs/deepseekv4_hu_harm_llama70b50_nomemo.md  cross_iteration_memos: false
#                   -> results_hu_harm_llama70b50_deepseekv4_nomemo/  probes/llama70b50_deepseekv4_nomemo
#   ARM B (memo):   configs/deepseekv4_hu_harm_llama70b50_memo.md    cross_iteration_memos: true
#                   -> results_hu_harm_llama70b50_deepseekv4_memo/    probes/llama70b50_deepseekv4_memo
#
# deepseek/deepseek-v4-flash is BOTH the attacker and the contrastive-pair generator
# (preprocessing.model). The judge is held fixed at openai/gpt-5.1-chat in both arms, so
# the only difference vs. the earlier attacker ablations is the attacker/contrastive model.
#
# ACTIVATIONS ARE COMPUTED FRESH. This is meant for a clean cloud box, so unlike
# run_gpt51_memo_ablation.sh there is NO symlink pre-seeding of the activation cache — the
# shared cache dir (results_hu_harm_llama70b50_deepseekv4/) starts empty. Arm A computes the
# base + eval activation blobs; arm B hits them, because those blobs depend only on the probe
# model / layer / seed / base data / eval splits / transforms — NOT on the attacker or on the
# memo flag. The redteam_acts_* per-conversation cache written into the same dir is
# content-keyed with a frozen LLM, so the two arms' distinct successes get distinct keys.
# That means arm A pays the full activation cost (~tens of minutes of GPU) and arm B mostly
# does not; budget wall-clock accordingly.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, so contrastive pairs are regenerated per arm.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_deepseekv4_memo_ablation.sh > logs/run_deepseekv4_memo_ablation.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start the failsafe alongside it. It
# already defaults to these two arms, in this order, and hands itself off from arm A to arm B
# when arm A's comparison CSV lands:
#   git checkout -b failsafe/deepseekv4-memo-ablation
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

SHARED_CACHE="results_hu_harm_llama70b50_deepseekv4"   # shared, attacker-independent activation cache

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in
# this list — it is meant to persist and grow across both arms and across re-runs.)
for p in \
    results_hu_harm_llama70b50_deepseekv4_nomemo results_hu_harm_llama70b50_deepseekv4_memo \
    probes/llama70b50_deepseekv4_nomemo probes/llama70b50_deepseekv4_memo ; do
    [ -e "$p" ] && { echo "ERROR: $p already exists — move it aside or bump the suffix." >&2; exit 1; }
done

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (starting empty — computed by arm A, reused by arm B)"

# --- run one arm ---------------------------------------------------------------------------
# Exit code the CLI uses for "OpenRouter is unusable" (cli.OUTAGE_EXIT_CODE).
OUTAGE_EXIT_CODE=3

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (log: $3)"
    local rc=0
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 3 \
        --base-training-data data/hu_harm_llama70b_50.jsonl \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_dataset_hu_ha \
        > "$3" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
        # The circuit breaker stopped the run: OpenRouter is out of credits or the
        # key is dead. Do NOT start the next arm — it would fail identically and
        # write a comparison CSV from probes trained on nothing.
        echo ">>> $(date -Is)  ABORTED $1 — OpenRouter unusable (exit $rc)." >&2
        tail -n 5 "$3" >&2
        echo ">>> Fix credits/key, then re-run with --resume to continue this arm." >&2
        exit "$rc"
    elif [ "$rc" -ne 0 ]; then
        echo ">>> $(date -Is)  FAILED  $1 (exit $rc) — see $3" >&2
        exit "$rc"
    fi
    echo ">>> $(date -Is)  DONE  $1"
}

run_arm configs/deepseekv4_hu_harm_llama70b50_nomemo.md probes/llama70b50_deepseekv4_nomemo logs/run_hh_llama70b50_deepseekv4_nomemo.log
run_arm configs/deepseekv4_hu_harm_llama70b50_memo.md   probes/llama70b50_deepseekv4_memo   logs/run_hh_llama70b50_deepseekv4_memo.log

echo ">>> $(date -Is)  both arms finished."
