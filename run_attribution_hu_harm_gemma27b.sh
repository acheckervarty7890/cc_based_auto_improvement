#!/usr/bin/env bash
#
# run_attribution_hu_harm_gemma27b.sh — attribute each red-team conversation's effect on
# the four hu_ha eval splits, for both arms of run_gemma27b_hu_harm_attacker_ablation_batch.sh.
#
# Answers: which (source, contrastive) pairs in probe_iter3's training set had NO effect on
# any eval split's AUROC, and which made one WORSE. Runs entirely off precomputed
# activations — nothing is ever pushed through gemma-3-27b, so a fresh box needs a GPU but
# not a big one.
#
# WHAT IT DOES, in order:
#   1. fetch    red-team iter3 blobs (both arms) + base split blobs + the four eval blobs,
#               all from Kaggle. ~6.6 GB downloaded, ~12.3 GB on disk. Skipped if present.
#   2. loo      leave-one-pair-out: 389 (gptoss) + 439 (deepseek) pairs x SEEDS seeds.
#               Resumable at pass granularity — see below.
#   3. verify   drop the flagged sets for real, retrain, and check the AUROC actually moves,
#               against a size-matched RANDOM control set.
#
# WHY IT IS SHAPED THIS WAY. The probe's seed-to-seed spread is enormous (sd 0.0233 on
# eval_balanced_refusal, range 0.078 over 10 refits of identical data) because validation
# AUROC saturates by epoch 4 and epoch selection then turns on one or two validation pairs
# out of 6624. No single pair out of ~400 is visible against that unpaired. So every pass
# trains a baseline column and up to K-1 drop columns TOGETHER from one shared
# initialisation and one shared shuffle stream, and the statistic is the paired difference,
# averaged over seeds. Raising SEEDS is the way to buy resolution: the error bar falls as
# 1/sqrt(SEEDS), and seeds already computed are never redone.
#
# RESUMABILITY. Each finished pass appends an fsync'd row to
# <attribution>/<arm>_iter3_loo_progress.jsonl, and a rerun of the identical command skips
# the passes already recorded. An unclean kill costs one pass (~30 s), not the run. Pair
# this with failsafe_commit_attribution.sh (see below) and a wiped container costs only
# whatever happened since the last push.
#
# COST at K=64 on an ~8 GB card: ~0.5 s per trained probe, ~30 s per pass, 7 passes per seed
# per arm. At the default 50 seeds that is roughly 3 h per arm for the sweep plus ~25 min
# per arm for the verification, so about 7 h total, plus the download.
#
# REQUIREMENTS
#   - the repo's venv at .venv_claude (see CLAUDE.md; invoke by absolute path)
#   - a CUDA GPU with >= 6 GB free. The packed train+val activations stay resident (~5.5 GB
#     peak at K=64). More VRAM means a larger --k, which is nearly free throughput.
#   - Kaggle credentials: KAGGLE_CONFIG_DIR pointing at the DIRECTORY holding kaggle.json
#     (the API joins the filename on, so pointing at the file itself fails), or
#     KAGGLE_API_TOKEN. Checked up front rather than 20 minutes into a download.
#   - NO OpenRouter key and NO Anthropic key: nothing here calls an LLM.
#
# USAGE
#   export KAGGLE_CONFIG_DIR=$HOME/.kaggle
#   nohup bash run_attribution_hu_harm_gemma27b.sh > logs/attribution_run.out 2>&1 &
#
#   # resume after any interruption — the identical command
#   nohup bash run_attribution_hu_harm_gemma27b.sh > logs/attribution_run.out 2>&1 &
#
# KNOBS (environment)
#   SEEDS=50           seeds per arm. More = tighter error bars; already-done seeds are kept.
#   K=64               probes trained per pass. Raise on a bigger card (24 GB -> ~256).
#   ARMS="a b"         which arms to process.
#   STAGES="fetch loo verify"   subset of stages to run.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY=.venv_claude/bin/python
export PYTHONUNBUFFERED=1   # logs go through `tee`; without this Python block-buffers
                           # and the log trails reality by kilobytes — which made a
                           # finished verify look stuck at 43/50.
SEEDS="${SEEDS:-50}"
K="${K:-64}"
ARMS="${ARMS:-gptoss120b deepseekv4pro}"
STAGES="${STAGES:-fetch loo verify}"
ATTRIB_DIR=results_hu_harm_gemma27b_batch_ablation/attribution

mkdir -p logs "$ATTRIB_DIR"

log() { echo ">>> $(date -Is)  $*"; }
have_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# --- preflight ------------------------------------------------------------------------
[ -x "$PY" ] || { echo "ERROR: no venv interpreter at $PY (see CLAUDE.md)" >&2; exit 1; }

if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: no CUDA device visible to torch. This needs a GPU (>= 6 GB free)." >&2
    exit 1
fi
"$PY" - <<'PYEOF'
import torch
free, total = torch.cuda.mem_get_info()
print(f">>> GPU: {torch.cuda.get_device_name(0)}  {free/1e9:.1f} GB free / {total/1e9:.1f} GB")
if free < 6e9:
    print("    WARNING: under 6 GB free. Lower K (e.g. K=16) or free the card.", flush=True)
PYEOF

if have_stage fetch; then
    if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
        kj="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
        [ -f "$kj" ] || kj="$HOME/.config/kaggle/kaggle.json"
        [ -f "$kj" ] || {
            echo "ERROR: no Kaggle credentials. Set KAGGLE_CONFIG_DIR to the DIRECTORY" >&2
            echo "       holding kaggle.json (not the file), or export KAGGLE_API_TOKEN." >&2
            exit 1; }
        log "kaggle credentials: $kj"
    fi
    "$PY" -c "import kaggle" 2>/dev/null || {
        echo "ERROR: the 'kaggle' package is missing: $PY -m pip install kaggle" >&2; exit 1; }
fi

# --- 1. fetch activations -------------------------------------------------------------
# Both fetchers validate what they download (model/layer, and row count for eval) and are
# no-ops for anything already on disk, so rerunning this stage is free.
if have_stage fetch; then
    log "fetching red-team iter3 + base activations from Kaggle"
    "$PY" scripts/publish_kaggle_redteam_activations.py restore --iterations 3

    log "fetching eval-split activations from Kaggle"
    "$PY" scripts/attribution_fetch_eval.py
fi

# --- 2. leave-one-pair-out ------------------------------------------------------------
if have_stage loo; then
    for arm in $ARMS; do
        log "LOO sweep: $arm ($SEEDS seeds, K=$K)"
        "$PY" scripts/attribution_loo.py --arm "$arm" --seeds "$SEEDS" --k "$K" \
            2>&1 | tee -a "logs/attribution_loo_${arm}.log"
    done
fi

# --- 3. verify by dropping the flagged sets -------------------------------------------
if have_stage verify; then
    for arm in $ARMS; do
        log "verification refits: $arm"
        "$PY" scripts/attribution_verify.py --arm "$arm" --seeds "$SEEDS" \
            2>&1 | tee -a "logs/attribution_verify_${arm}.log"
    done
fi

log "done. Results in $ATTRIB_DIR:"
ls -la "$ATTRIB_DIR"
echo
echo "Read <arm>_iter3_verify.json (or the verify log) first, and inside it compare every"
echo "variant against drop_random_control — a flagged set that does not beat a same-sized"
echo "random set by more than its error bar has identified nothing."
