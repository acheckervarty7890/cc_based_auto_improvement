#!/usr/bin/env bash
set -e

# Cross-iteration memo experiment on the HIGH-STAKES concept with a google/gemma-3-27b-it
# (L32) probe and openai/gpt-oss-120b attacking in BATCH mode. Two arms, run sequentially and
# fully isolated:
#
#   ARM 1 (cross_iteration_memos ON,  NO contrastive label guidance)
#       configs/gptoss120b_hs_gemma27b_batch_itermemo.md
#       -> results_hs_gemma27b_gptoss_batch_itermemo/  probes/hs_gemma27b_gptoss_batch_itermemo
#
#   ARM 2 (cross_iteration_memos OFF, WITH concept_description + label_guidance)
#       configs/gptoss120b_hs_gemma27b_batch_guidance.md
#       -> results_hs_gemma27b_gptoss_batch_guidance/  probes/hs_gemma27b_gptoss_batch_guidance
#
# ############################################################################################
# THE ARMS DIFFER IN TWO KNOBS, IN OPPOSITE DIRECTIONS (memo on/guidance off vs memo
# off/guidance on). A delta between the two comparison CSVs is the SUM of both effects and
# cannot be attributed to either one. This is intentional — see the config headers for the
# controlled one-variable references (configs/itermemo_hs_llama1b_{memo,nomemo}.md for the
# memo; experiment6_cloud's guidance pair for the guidance).
# ############################################################################################
#
# Everything else is experiment12_cloud held fixed: judge openai/gpt-5.1, preprocessing model
# openai/gpt-5.1, probe gemma-3-27b-it L32 trained from scratch off data/hs_ls_200.jsonl, both
# error types, 3 iterations. Attempt volume is identical in both arms: 10 sessions x 5
# conversations (batch size) x 5 rounds per error type.
#
# vs experiment12's gpt-oss arm: that one ran PER-TURN prompt mode with view_limit 4. Here
# gpt-oss takes the BATCH shape exp12 gave to nemotron (batch_submissions + view_limit 0), so
# the attacker's entire input is its system prompt — which is what makes arm 1's
# "## Lessons from previous iterations" block the only cross-iteration signal in the run.
#
# ACTIVATIONS. The shared cache dir (results_hs_gemma27b_itermemo_shared/) starts empty on a
# clean cloud box; arm 1 fills it and arm 2 hits it, because those blobs depend only on the
# probe model / layer / seed / base data / eval splits / transforms — NOT on the memo knob,
# the guidance, or the attacker. The redteam_acts_* per-conversation cache written into the
# same dir is content-keyed with a frozen LLM, so the two arms' distinct successes and their
# distinct contrastive pairs get distinct keys.
#
# The CONTRASTIVE cache is NOT shared and cannot be: non-empty guidance is folded into the key
# by _guidance_fingerprint(), so arm 2 generates its own pairs. Each arm gets its own
# --probe-out-dir and therefore its own contrastive_cache.jsonl anyway.
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
#   nohup bash run_gemma27b_hs_itermemo.sh > logs/run_gemma27b_hs_itermemo.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start the failsafe commit script
# alongside it, pointed at these two arms in this order.

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# --- preflight: attacker model slug is actually served ---------------------------------------
# OpenRouter's /models is unauthenticated, but the key is sent anyway so the same credential
# problem shows up here rather than mid-run.
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

check_model openai/gpt-oss-120b   # attacker in BOTH arms
check_model openai/gpt-5.1        # judge + summarizer + contrastive generator in BOTH arms

# --- preflight: the two configs differ in exactly the two intended knobs -----------------------
# Cheap insurance against a stray edit: the arms share one attacker/judge prompt and one
# scheduling block, and a silent drift in either would make the pair uninterpretable on top of
# the confound it already carries.
.venv_claude/bin/python - <<'PY'
import sys
from agentic_redteam.config import load_config

a = load_config("configs/gptoss120b_hs_gemma27b_batch_itermemo.md")
b = load_config("configs/gptoss120b_hs_gemma27b_batch_guidance.md")

problems = []
if a.attacker.system_prompt != b.attacker.system_prompt:
    problems.append("attacker prompt differs between the arms")
if a.judge.system_prompt != b.judge.system_prompt:
    problems.append("judge prompt differs between the arms")

for k in ("model_names", "interface", "batch_submissions", "view_limit", "max_turns",
          "batch_target", "rounds", "concurrency", "sessions_per_model", "view_reshuffle",
          "near_dup_guard", "near_dup_threshold", "round_summaries", "capture_prompts",
          "cross_iteration_memo_max_successes"):
    va, vb = getattr(a.attacker, k), getattr(b.attacker, k)
    if va != vb:
        problems.append(f"attacker.{k}: {va!r} vs {vb!r}")

for k in ("model", "layer", "pos_class_label", "neg_class_label", "error_types"):
    va, vb = getattr(a.probe, k), getattr(b.probe, k)
    if va != vb:
        problems.append(f"probe.{k}: {va!r} vs {vb!r}")

for k in ("model", "provider", "max_tokens", "confidence_threshold", "hide_opposite_direction"):
    va, vb = getattr(a.judge, k), getattr(b.judge, k)
    if va != vb:
        problems.append(f"judge.{k}: {va!r} vs {vb!r}")

# The two knobs that are SUPPOSED to differ.
if not (a.attacker.cross_iteration_memos and not b.attacker.cross_iteration_memos):
    problems.append("cross_iteration_memos must be True in arm 1 and False in arm 2")
a_guided = bool(a.preprocessing.concept_description or a.preprocessing.label_guidance)
b_guided = bool(b.preprocessing.concept_description or b.preprocessing.label_guidance)
if a_guided or not b_guided:
    problems.append("label guidance must be ABSENT in arm 1 and PRESENT in arm 2")

if a.output.jsonl_path == b.output.jsonl_path:
    problems.append("the arms share output.jsonl_path — they would write into one another")

if problems:
    print("ERROR: the two arm configs are not in the intended relationship:", file=sys.stderr)
    for p in problems:
        print(f"       - {p}", file=sys.stderr)
    sys.exit(1)
print(">>> arm configs OK: differ in cross_iteration_memos + label guidance only")
PY

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

SHARED_CACHE="results_hs_gemma27b_itermemo_shared"   # shared, arm-independent activation cache

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in
# this list — it is meant to persist and grow across both arms and across re-runs.)
for p in results_hs_gemma27b_gptoss_batch_itermemo probes/hs_gemma27b_gptoss_batch_itermemo \
         results_hs_gemma27b_gptoss_batch_guidance probes/hs_gemma27b_gptoss_batch_guidance; do
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

run_arm configs/gptoss120b_hs_gemma27b_batch_itermemo.md  probes/hs_gemma27b_gptoss_batch_itermemo  logs/run_hs_gemma27b_gptoss_batch_itermemo.log
run_arm configs/gptoss120b_hs_gemma27b_batch_guidance.md  probes/hs_gemma27b_gptoss_batch_guidance  logs/run_hs_gemma27b_gptoss_batch_guidance.log

echo ">>> $(date -Is)  both arms finished."
