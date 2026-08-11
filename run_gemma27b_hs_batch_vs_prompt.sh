#!/usr/bin/env bash
set -e

# Prompt-vs-batch attacker experiment on the HIGH-STAKES concept with a
# google/gemma-3-27b-it (L32) probe. Two arms, run sequentially and fully isolated:
#
#   ARM 1 (gpt-oss-120b, per-turn prompt mode, view_limit 4)
#       configs/gptoss120b_hs_gemma27b_prompt.md
#       -> results_hs_gemma27b_gptoss120b_prompt/  probes/hs_gemma27b_gptoss120b_prompt
#
#   ARM 2 (nemotron-3-ultra-550b-a55b, BATCHED prompt mode, view_limit 0 = blind)
#       configs/nemotron3ultra_hs_gemma27b_batch.md
#       -> results_hs_gemma27b_nemotron3ultra_batch/  probes/hs_gemma27b_nemotron3ultra_batch
#
# The judge (openai/gpt-5.1), the preprocessing model (openai/gpt-5.1), the probe
# (gemma-3-27b-it L32), the base data (data/hs_ls_200.jsonl) and every scheduling knob are
# held fixed. Attempt volume is identical: 10 sessions x 5 conversations x 5 rounds per
# error type, both arms.
#
# THE ARMS ARE NOT A SINGLE-VARIABLE ABLATION — arm 2 changes the attacker model AND the
# submission mode AND the injected view together. The controlled pairs already exist:
#   arm 1 == experiment8_cloud's gptoss120b_hs_gemma27b_noguidance.md (only outputs repointed),
#   and arm 2 is knob-identical to experiment9_cloud's gptoss120b_hs_gemma27b_batch.md except
#   attacker.models. See the config headers.
#
# ACTIVATIONS. The shared cache dir (results_hs_gemma27b_batch_vs_prompt/) starts empty on a
# clean cloud box; arm 1 fills it and arm 2 hits it, because those blobs depend only on the
# probe model / layer / seed / base data / eval splits / transforms — NOT on the attacker or
# the submission mode. The redteam_acts_* per-conversation cache written into the same dir is
# content-keyed with a frozen LLM, so the two arms' distinct successes get distinct keys.
#
# The EVAL half is not computed at all: both configs carry a `kaggle:` section pointing at
# anku7890/{split}gemmaevalpt, so arm 1's first eval downloads ~20 GB of precomputed
# gemma-3-27b activations (validated against the probe's model/layer and each split's row
# count) straight into eval_activations/ instead of running full splits through a 27B model.
# That needs credentials — see the KAGGLE_CONFIG_DIR check below. The BASE split (~116 MB) is
# still computed locally by arm 1, as is every red-team conversation.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, keeping the two arms' provenance separate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_hs_batch_vs_prompt.sh > logs/run_gemma27b_hs_batch_vs_prompt.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it,
# pointed at these two arms in this order.

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# --- preflight: attacker model slugs are actually served -------------------------------------
# Arm 2's model is checked BEFORE arm 1 starts. Without this, a wrong slug surfaces only when
# arm 2 begins — many hours in — and then fails every round of the rotation while the circuit
# breaker patiently retries. OpenRouter's /models is unauthenticated, but the key is sent
# anyway so the same credential problem shows up here rather than mid-run.
check_model () {  # $1 = openrouter model slug
    .venv_claude/bin/python - "$1" <<'PY'
import json, os, sys, urllib.error, urllib.request

slug = sys.argv[1]
base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
req = urllib.request.Request(
    f"{base}/models",
    headers={"Authorization": "Bearer " + os.environ.get("OPENROUTER_API_KEY", "")},
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        ids = {m.get("id") for m in json.load(resp).get("data", [])}
except (urllib.error.URLError, OSError, ValueError) as exc:
    # Don't block the run on a flaky preflight — the breaker handles real outages.
    print(f"WARNING: could not reach {base}/models to verify '{slug}': {exc}", file=sys.stderr)
    sys.exit(0)

if slug in ids:
    print(f">>> openrouter model OK: {slug}")
    sys.exit(0)

print(f"ERROR: '{slug}' is not served by OpenRouter.", file=sys.stderr)
near = sorted(i for i in ids if slug.split("/")[0] in i)
if near:
    print("       models from the same provider:", file=sys.stderr)
    for i in near[:20]:
        print(f"         {i}", file=sys.stderr)
sys.exit(1)
PY
}

check_model openai/gpt-oss-120b
check_model nvidia/nemotron-3-ultra-550b-a55b

# --- preflight: Kaggle credentials for the precomputed eval activations -----------------------
# Checked HERE rather than at first use: the first eval is hours into arm 1, and an
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

SHARED_CACHE="results_hs_gemma27b_batch_vs_prompt"   # shared, arm-independent activation cache

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in
# this list — it is meant to persist and grow across both arms and across re-runs.)
for p in results_hs_gemma27b_gptoss120b_prompt probes/hs_gemma27b_gptoss120b_prompt \
         results_hs_gemma27b_nemotron3ultra_batch probes/hs_gemma27b_nemotron3ultra_batch; do
    [ -e "$p" ] && { echo "ERROR: $p already exists — move it aside or bump the suffix." >&2; exit 1; }
done

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (starting empty — computed by arm 1, reused by arm 2)"

# --- run one arm ------------------------------------------------------------------------------
# Exit code the CLI uses for "OpenRouter is unusable" (cli.OUTAGE_EXIT_CODE).
OUTAGE_EXIT_CODE=3

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (log: $3)"
    local rc=0
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 3 \
        --base-training-data data/hs_ls_200.jsonl \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_datasets \
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

run_arm configs/gptoss120b_hs_gemma27b_prompt.md      probes/hs_gemma27b_gptoss120b_prompt      logs/run_hs_gemma27b_gptoss120b_prompt.log
run_arm configs/nemotron3ultra_hs_gemma27b_batch.md   probes/hs_gemma27b_nemotron3ultra_batch   logs/run_hs_gemma27b_nemotron3ultra_batch.log

echo ">>> $(date -Is)  both arms finished."
