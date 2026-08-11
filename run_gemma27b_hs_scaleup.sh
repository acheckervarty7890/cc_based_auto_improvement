#!/usr/bin/env bash
set -e

# Budget-scaling experiment on the HIGH-STAKES concept with a google/gemma-3-27b-it (L32) probe
# and openai/gpt-oss-120b attacking in BATCH mode. Two arms, run sequentially and fully
# isolated:
#
#   ARM 1 (WIDER rounds: batch_target 30->60, sessions_per_model 10->20, concurrency 20)
#       configs/gptoss120b_hs_gemma27b_batch_target60.md
#       -> results_hs_gemma27b_gptoss_batch_target60/  probes/hs_gemma27b_gptoss_batch_target60
#
#   ARM 2 (DEEPER rotations: rounds 5->10, everything else at the base values)
#       configs/gptoss120b_hs_gemma27b_batch_rounds10.md
#       -> results_hs_gemma27b_gptoss_batch_rounds10/  probes/hs_gemma27b_gptoss_batch_rounds10
#
# BASE for both: experiment13_cloud's arm 1 with cross_iteration_memos turned OFF. Neither the
# cross-iteration memo nor contrastive label guidance is active in either arm, so the only
# things that differ from the base — and from each other — are the scheduling knobs above.
# Everything else is experiment12_cloud held fixed: judge openai/gpt-5.1, preprocessing model
# openai/gpt-5.1, probe gemma-3-27b-it L32 trained from scratch off data/hs_ls_200.jsonl, both
# error types, 3 iterations, max_turns 5, view_limit 0, near-dup guard at 0.8.
#
# THE ARMS SPEND THE SAME BUDGET IN DIFFERENT SHAPES — both reach ~500 conversations per error
# type per iteration (the base reached ~250), arm 1 by doubling the width of each round and arm
# 2 by doubling the number of rounds. What separates them is the rolling ROUND memo: rounds run
# sequentially under `round_summaries`, and at view_limit 0 in batch mode that memo is the only
# input that changes between rounds. So arm 1's extra conversations are written blind and in
# parallel (5 memo generations, as in the base) while arm 2's arrive after 5 more memo updates
# (9 per iteration — the final round is never summarized).
#
# EXPECT ARM 2 TO TAKE ROUGHLY TWICE ARM 1'S WALL CLOCK for the same volume: 10 sequential
# rounds against 5, each no wider than the base.
#
# A NOTE ON batch_target, so arm 1's result is not over-read: in batch mode
# _run_openrouter_prompt_batch_model checks batch_target only after a call comes back SHORT of
# max_turns. A session whose single reply carries all max_turns conversations breaks on
# `batch_complete` and never reaches the check. So batch_target does not cap a round here — it
# only suppresses top-up calls — and the 30->60 change is close to inert on its own. The real
# variable in arm 1 is sessions_per_model 10->20. They are raised together because a 60-success
# budget under the old 50-conversation round could not be reached even in principle.
#
# ACTIVATIONS. The shared cache dir (results_hs_gemma27b_scaleup_shared/) starts empty on a
# clean cloud box; arm 1 fills it and arm 2 hits it, because those blobs depend only on the
# probe model / layer / seed / base data / eval splits / transforms — NOT on any scheduling
# knob. The redteam_acts_* per-conversation cache written into the same dir is content-keyed
# with a frozen LLM, so the two arms' distinct successes get distinct keys.
#
# On a box that has ALREADY run experiment13 to completion, both output.*_cache_dir keys may be
# repointed at results_hs_gemma27b_itermemo_shared/ instead — the keys are identical, so that
# skips the ~20 GB Kaggle download and the base extraction. Never do that while that run is
# live: two writers can tear a blob.
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
#   nohup bash run_gemma27b_hs_scaleup.sh > logs/run_gemma27b_hs_scaleup.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start the failsafe commit script
# alongside it, pointed at these two arms in this order.

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# --- preflight: model slugs are actually served -----------------------------------------------
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

# --- preflight: the two configs differ in exactly the intended knobs --------------------------
# Cheap insurance against a stray edit: the arms share one attacker/judge prompt, one probe, one
# judge and one preprocessing block, and a silent drift in any of them would make the
# width-vs-depth comparison uninterpretable.
.venv_claude/bin/python - <<'PY'
import sys
from agentic_redteam.config import load_config

a = load_config("configs/gptoss120b_hs_gemma27b_batch_target60.md")   # wider
b = load_config("configs/gptoss120b_hs_gemma27b_batch_rounds10.md")   # deeper

problems = []
if a.attacker.system_prompt != b.attacker.system_prompt:
    problems.append("attacker prompt differs between the arms")
if a.judge.system_prompt != b.judge.system_prompt:
    problems.append("judge prompt differs between the arms")

# Held fixed across the arms.
for k in ("model_names", "interface", "batch_submissions", "view_limit", "max_turns",
          "view_reshuffle", "near_dup_guard", "near_dup_threshold", "round_summaries",
          "capture_prompts", "cross_iteration_memos"):
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

# The scheduling knobs that are SUPPOSED to differ.
if (a.attacker.batch_target, a.attacker.sessions_per_model, a.attacker.rounds) != (60, 20, 5):
    problems.append(
        "arm 1 must be batch_target 60 / sessions_per_model 20 / rounds 5, got "
        f"{a.attacker.batch_target} / {a.attacker.sessions_per_model} / {a.attacker.rounds}"
    )
if (b.attacker.batch_target, b.attacker.sessions_per_model, b.attacker.rounds) != (30, 10, 10):
    problems.append(
        "arm 2 must be batch_target 30 / sessions_per_model 10 / rounds 10, got "
        f"{b.attacker.batch_target} / {b.attacker.sessions_per_model} / {b.attacker.rounds}"
    )

# Neither arm may carry label guidance, and concurrency must not throttle the fan-out.
for name, c in (("arm 1", a), ("arm 2", b)):
    if c.preprocessing.concept_description or c.preprocessing.label_guidance:
        problems.append(f"{name} carries label guidance; both arms must have none")
    need = c.attacker.sessions_per_model * len(c.attacker.models)
    if c.attacker.concurrency < need:
        problems.append(
            f"{name}: concurrency {c.attacker.concurrency} < sessions_per_model x models "
            f"({need}) — the extra sessions would queue on the semaphore and each would get "
            "its own success budget"
        )

if a.output.jsonl_path == b.output.jsonl_path:
    problems.append("the arms share output.jsonl_path — they would write into one another")

if problems:
    print("ERROR: the two arm configs are not in the intended relationship:", file=sys.stderr)
    for p in problems:
        print(f"       - {p}", file=sys.stderr)
    sys.exit(1)
print(">>> arm configs OK: differ in the round-scheduling knobs only (wider vs deeper)")
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

SHARED_CACHE="results_hs_gemma27b_scaleup_shared"   # shared, arm-independent activation cache

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in
# this list — it is meant to persist and grow across both arms and across re-runs.)
for p in results_hs_gemma27b_gptoss_batch_target60 probes/hs_gemma27b_gptoss_batch_target60 \
         results_hs_gemma27b_gptoss_batch_rounds10 probes/hs_gemma27b_gptoss_batch_rounds10; do
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

run_arm configs/gptoss120b_hs_gemma27b_batch_target60.md  probes/hs_gemma27b_gptoss_batch_target60  logs/run_hs_gemma27b_gptoss_batch_target60.log
run_arm configs/gptoss120b_hs_gemma27b_batch_rounds10.md  probes/hs_gemma27b_gptoss_batch_rounds10  logs/run_hs_gemma27b_gptoss_batch_rounds10.log

echo ">>> $(date -Is)  both arms finished."
