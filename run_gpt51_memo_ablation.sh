#!/usr/bin/env bash
set -e

# gpt-5.1 attacker — cross-iteration-memo ablation. Two arms, IDENTICAL in every
# knob except attacker.cross_iteration_memos, run sequentially and fully isolated:
#
#   ARM A (nomemo): configs/gpt51_hu_harm_llama70b50_nomemo.md   cross_iteration_memos: false
#                   -> results_hu_harm_llama70b50_gpt51_nomemo/  probes/llama70b50_gpt51_nomemo
#   ARM B (memo):   configs/gpt51_hu_harm_llama70b50_memo.md     cross_iteration_memos: true
#                   -> results_hu_harm_llama70b50_gpt51_memo/    probes/llama70b50_gpt51_memo
#
# Both arms SHARE one activation cache (results_hu_harm_llama70b50_gpt51/) on purpose:
# base + eval activations depend only on the probe model / eval splits / seed / base data /
# transforms — NOT on the attacker or the memo flag — so the two arms hit the same blobs, and
# the redteam_acts_* per-conversation cache is content-keyed with a frozen LLM (distinct
# successes -> distinct keys, no collision). The cache is pre-seeded below with SYMLINKS to
# the already-computed blobs under results_hu_harm_llama70b50_grok/ so the first eval/train
# hits instead of recomputing.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, so contrastive pairs are regenerated per arm.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gpt51_memo_ablation.sh > logs/run_gpt51_memo_ablation.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

SHARED_CACHE="results_hu_harm_llama70b50_gpt51"          # shared, attacker-independent activation cache
SRC_CACHE="results_hu_harm_llama70b50_grok"             # existing correctly-keyed blobs to symlink from

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in
# this list — it is meant to persist and grow across both arms and across re-runs.)
for p in \
    results_hu_harm_llama70b50_gpt51_nomemo results_hu_harm_llama70b50_gpt51_memo \
    probes/llama70b50_gpt51_nomemo probes/llama70b50_gpt51_memo ; do
    [ -e "$p" ] && { echo "ERROR: $p already exists — move it aside or bump the suffix." >&2; exit 1; }
done

# --- pre-seed the shared activation cache with symlinks to existing blobs -------------------
mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
seed_links () {  # $1 = source dir, $2 = glob pattern, $3 = dest dir
    shopt -s nullglob
    local n=0 f
    for f in "$1"/$2; do
        ln -sfn "$PWD/$f" "$3/$(basename "$f")"
        n=$((n + 1))
    done
    shopt -u nullglob
    echo "    seeded $n symlink(s) into $3 from $1/$2"
}
echo ">>> pre-seeding shared activation cache from $SRC_CACHE"
seed_links "$SRC_CACHE/base_activations" 'base_acts_*_*.pt'    "$SHARED_CACHE/base_activations"
seed_links "$SRC_CACHE/eval_activations" 'eval_*-acts_full.pt' "$SHARED_CACHE/eval_activations"

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

run_arm configs/gpt51_hu_harm_llama70b50_nomemo.md probes/llama70b50_gpt51_nomemo logs/run_hh_llama70b50_gpt51_nomemo.log
run_arm configs/gpt51_hu_harm_llama70b50_memo.md   probes/llama70b50_gpt51_memo   logs/run_hh_llama70b50_gpt51_memo.log

echo ">>> $(date -Is)  both arms finished."
