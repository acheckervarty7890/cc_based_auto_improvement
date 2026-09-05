#!/usr/bin/env bash
set -e

# ARMS 10-12 — the remaining three attackers of the "the ATTACKER is shown the eval-data
# description too" condition. ARM 9 asked this of llama-3.3-70b; these three ask it of the other
# three attackers on the branch, so the arm6->arm9 shape can be checked for REPLICATION rather
# than read off a single rotation.
#
#   attacker          memo only   + desc to the judge's summarizers   + the attacker is shown it
#   ----------------  ----------  ---------------------------------   -------------------------
#   nemotron          arm 1       arm 2                               arm 10  <- this run
#   gpt-oss-120b      arm 3       arm 4                               arm 11  <- this run
#   llama-3.3-70b     arm 5       arm 6                               arm 9   (done: .8216/.8542/.8688)
#   deepseek-v4-pro   arm 7       arm 8                               arm 12  <- this run
#
# Each arm N here differs from its evaldesc sibling in EXACTLY ONE parsed key —
# attacker.show_eval_data_description — plus the three output paths. Verified by loading both
# configs and diffing every field of the parsed Config; judge.eval_scope_check stays false, so
# the labelling function does not move and every metric stays comparable within the triple.
#
# Usage:
#   set -a; . ./.env; set +a
#   nohup bash run_gemma27b_instructions_evaldesc_attacker_rest.sh > logs/run_arms10_12.out 2>&1 &
#
# Checkpointing: failsafe_commit.sh with these three stages, in this order —
#   nohup bash failsafe_commit.sh \
#     --config configs/nemotron_instructions_gemma27b_nmbase_evaldesc_attacker.md \
#     --probe-out-dir probes/instructions_gemma27b_nemotron_nmbase_evaldesc_attacker \
#     --log-file logs/run_instructions_gemma27b_nemotron_nmbase_evaldesc_attacker.log \
#     --config configs/gptoss120b_instructions_gemma27b_gobase_evaldesc_attacker.md \
#     --probe-out-dir probes/instructions_gemma27b_gptoss_gobase_evaldesc_attacker \
#     --log-file logs/run_instructions_gemma27b_gptoss_gobase_evaldesc_attacker.log \
#     --config configs/deepseekv4pro_instructions_gemma27b_dsbase_evaldesc_attacker.md \
#     --probe-out-dir probes/instructions_gemma27b_deepseekv4pro_dsbase_evaldesc_attacker \
#     --log-file logs/run_instructions_gemma27b_deepseekv4pro_dsbase_evaldesc_attacker.log \
#     > logs/failsafe_arms10_12.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# The three base files are this experiment's variable — check them BEFORE the hours-long first
# arm rather than letting train_initial_probe discover one missing at arm 12.
NEMOTRON_BASE="data/instructions_nemotron_50.jsonl"
GPTOSS_BASE="data/instructions_gptoss_50.jsonl"
DEEPSEEK_BASE="data/instructions_deepseekv4pro_50.jsonl"
for base in "$NEMOTRON_BASE" "$GPTOSS_BASE" "$DEEPSEEK_BASE"; do
    if [ ! -f "$base" ]; then
        echo "ERROR: base training data not found: $base" >&2
        echo "       It comes from the generator_experiment_1 branch:" >&2
        echo "       git show origin/generator_experiment_1:$base > $base" >&2
        exit 1
    fi
    echo ">>> base training data: $base ($(wc -l < "$base") rows)"
done

# Kaggle credentials for the precomputed eval AND dev activations (configs' `kaggle:` section).
# Checked HERE rather than at first use: the dev prefetch runs before iteration 0 trains, and an
# unauthenticated KaggleApi.authenticate() ends in exit(1), not an exception.
if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
    kaggle_json="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
    [ -f "$kaggle_json" ] || kaggle_json="$HOME/.config/kaggle/kaggle.json"
    if [ ! -f "$kaggle_json" ]; then
        echo "ERROR: no Kaggle credentials. Set KAGGLE_CONFIG_DIR to the DIRECTORY holding" >&2
        echo "       kaggle.json (not the file), or export KAGGLE_API_TOKEN." >&2
        exit 1
    fi
    echo ">>> kaggle credentials: $kaggle_json"
fi
.venv_claude/bin/python -c "import kaggle" 2>/dev/null || {
    echo "ERROR: the 'kaggle' package is not installed: .venv_claude/bin/pip install kaggle" >&2
    exit 1
}

# HuggingFace token. tuberlens' LLMModel.load calls hf_login(), which RAISES
# ValueError("No HuggingFace token found") when no token is set — even though the gemma weights
# are already in the local HF cache and no download happens.
if [ -z "${HF_TOKEN:-}" ] && [ -f hf_token.txt ]; then
    HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt)"
    export HF_TOKEN
fi
: "${HF_TOKEN:?export HF_TOKEN (or put it in hf_token.txt) — tuberlens hf_login() raises without one, even for a fully cached model}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo ">>> HF token: present (${#HF_TOKEN} chars)"

# --- weight download path ---------------------------------------------------------------------
# TRANSFER LAYER ONLY — cannot change a single number the run produces. GUARDED on the import:
# huggingface_hub RAISES when HF_HUB_ENABLE_HF_TRANSFER=1 and the package is missing.
if .venv_claude/bin/python -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
    echo ">>> hf_transfer: enabled (parallel shard download; transfer layer only)"
else
    echo ">>> hf_transfer: NOT installed — using the default single-threaded download."
fi

# --- pin the extraction model's memory budget -------------------------------------------------
# PLACEMENT ONLY. Sized for a 24 GiB card and ~57 GiB of host RAM. ADJUST IF THIS BOX IS
# DIFFERENT. Both vars deliberately: AGENTIC_REDTEAM_MAX_MEMORY is authoritative on this repo's
# load_extraction_model path, tuberlens' MAX_MEMORY reaches every OTHER tuberlens load
# (get_performances included, which this repo cannot pass model_kwargs to).
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
echo ">>> max_memory pinned: $AGENTIC_REDTEAM_MAX_MEMORY (placement only — does not change results)"

# --- ensemble fit/score path ------------------------------------------------------------------
# 0 = SEQUENTIAL, as in arms 1-9 and every earlier instruction experiment, so these probes are fit
# on the same path the arms they will be compared against were. Export PROBE_FUSED_ENSEMBLE=1 to
# take the ~3.8x fit speedup at the cost of exact comparability with those runs (4th-decimal
# AUROC drift from a changed reduction order).
export PROBE_FUSED_ENSEMBLE="${PROBE_FUSED_ENSEMBLE:-0}"
if [ "$PROBE_FUSED_ENSEMBLE" = "0" ]; then
    echo ">>> ensemble: SEQUENTIAL fit and scoring (PROBE_FUSED_ENSEMBLE=0 — matches arms 1-9, slower)"
else
    echo ">>> ensemble: FUSED fit and scoring (PROBE_FUSED_ENSEMBLE=$PROBE_FUSED_ENSEMBLE — faster; 4th-decimal drift vs those runs)"
fi
.venv_claude/bin/python - <<'PYEOF' || exit 1
import sys, os
from agentic_redteam.ensemble import fusion_enabled
want = os.environ.get("PROBE_FUSED_ENSEMBLE", "0") != "0"
got = fusion_enabled()
print(f">>> ensemble fusion_enabled() = {got} (wanted {want})")
sys.exit(0 if got == want else 1)
PYEOF

# Assert each arm really is its sibling plus ONE key, before spending hours on it.
.venv_claude/bin/python - <<'PYEOF' || exit 1
import dataclasses, pathlib, sys
from agentic_redteam.config import load_config
PAIRS = [
    ("configs/nemotron_instructions_gemma27b_nmbase_evaldesc.md",
     "configs/nemotron_instructions_gemma27b_nmbase_evaldesc_attacker.md"),
    ("configs/gptoss120b_instructions_gemma27b_gobase_evaldesc.md",
     "configs/gptoss120b_instructions_gemma27b_gobase_evaldesc_attacker.md"),
    ("configs/deepseekv4pro_instructions_gemma27b_dsbase_evaldesc.md",
     "configs/deepseekv4pro_instructions_gemma27b_dsbase_evaldesc_attacker.md"),
]
ALLOWED = {"attacker.show_eval_data_description", "output.comparison_csv",
           "output.jsonl_path", "output.run_id", "source_path"}
def flat(obj, prefix=""):
    out = {}
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            out.update(flat(getattr(obj, f.name), f"{prefix}{f.name}."))
    else:
        out[prefix.rstrip(".")] = repr(obj)
    return out
rc = 0
for a, b in PAIRS:
    fa, fb = flat(load_config(pathlib.Path(a))), flat(load_config(pathlib.Path(b)))
    diffs = {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}
    extra = diffs - ALLOWED
    if extra or "attacker.show_eval_data_description" not in diffs:
        print(f">>> CONFIG DIFF CHECK FAILED for {b}: unexpected {sorted(extra)}")
        rc = 1
    else:
        print(f">>> one-key check OK: {pathlib.Path(b).name}")
sys.exit(rc)
PYEOF

SHARED_CACHE="results_instructions_gemma27b_shared"   # shared, arm-independent activation cache
mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE"
echo ">>>   eval + dev blobs are warm if this box ran arm 9; otherwise they come from Kaggle."
echo ">>>   The three 50-row base blobs are per-attacker and are computed by each arm's iter0."

# No clobber guard on the per-arm output/probe dirs ON PURPOSE: --resume is on by default and is
# how a wiped container picks a run back up, which requires the existing probe-out-dir and its
# markers. To start genuinely fresh, move the per-arm dirs aside first.

# --- run one arm ------------------------------------------------------------------------------
OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is unusable"

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = base training data, $4 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (base: $3, log: $4)"
    local rc=0
    # --iterations 10, as in every arm on this branch: the contrast is about what crosses an
    # iteration boundary, so the number of boundaries must not vary.
    #
    # NOT passing --ensemble-size or --dev-data: both OVERRIDE the configs (flag > config) and
    # both are properties of the probe every arm shares, so they live in the configs where the
    # arms can be diffed. --test-size / --split-field are likewise absent: dev_data makes
    # retrain.py ignore them.
    #
    # NOTE the log is APPENDED, not truncated. Arm 9's runner used `>`, so restarting it after
    # the box died threw away iterations 0-8 of the log and they had to be recovered from a
    # git checkpoint. --resume makes a restart the normal case, so the log must survive one.
    echo "===== $(date -Is)  run_arm start (append) =====" >> "$4"
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 10 \
        --base-training-data "$3" \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_sets/instructions \
        >> "$4" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
        # The circuit breaker stopped the run: OpenRouter is out of credits or the key is dead.
        # Do NOT start the next arm — it would fail identically and write a comparison CSV from
        # probes trained on nothing.
        echo ">>> $(date -Is)  ABORTED $1 — OpenRouter unusable (exit $rc)." >&2
        tail -n 5 "$4" >&2
        echo ">>> Fix credits/key, then re-run with --resume to continue this arm." >&2
        exit "$rc"
    elif [ "$rc" -ne 0 ]; then
        echo ">>> $(date -Is)  FAILED  $1 (exit $rc) — see $4" >&2
        exit "$rc"
    fi
    echo ">>> $(date -Is)  DONE  $1"
}

run_arm configs/nemotron_instructions_gemma27b_nmbase_evaldesc_attacker.md \
        probes/instructions_gemma27b_nemotron_nmbase_evaldesc_attacker \
        "$NEMOTRON_BASE" \
        logs/run_instructions_gemma27b_nemotron_nmbase_evaldesc_attacker.log

run_arm configs/gptoss120b_instructions_gemma27b_gobase_evaldesc_attacker.md \
        probes/instructions_gemma27b_gptoss_gobase_evaldesc_attacker \
        "$GPTOSS_BASE" \
        logs/run_instructions_gemma27b_gptoss_gobase_evaldesc_attacker.log

run_arm configs/deepseekv4pro_instructions_gemma27b_dsbase_evaldesc_attacker.md \
        probes/instructions_gemma27b_deepseekv4pro_dsbase_evaldesc_attacker \
        "$DEEPSEEK_BASE" \
        logs/run_instructions_gemma27b_deepseekv4pro_dsbase_evaldesc_attacker.log

echo ">>> $(date -Is)  arms 10-12 finished."
