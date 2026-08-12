#!/usr/bin/env bash
set -e

# ATTACKER-MODEL experiment on the INSTRUCTION-FOLLOWING concept with a google/gemma-3-27b-it
# (L32) probe, both arms attacking in BATCH mode. Two arms, run sequentially and fully
# isolated:
#
#   ARM 1 (attacker openai/gpt-oss-120b)
#       configs/gptoss120b_instructions_gemma27b_batch_target60.md
#       -> results_instructions_gemma27b_gptoss/   probes/instructions_gemma27b_gptoss
#
#   ARM 2 (attacker nvidia/nemotron-3-ultra-550b-a55b)
#       configs/nemotron_instructions_gemma27b_batch_target60.md
#       -> results_instructions_gemma27b_nemotron/ probes/instructions_gemma27b_nemotron
#
# THE ONLY VARIABLE IS THE ATTACKER MODEL. The scheduling shape is experiment15_cloud's arm 1
# held fixed in both: batch mode, batch_target 60, sessions_per_model 20, concurrency 20,
# rounds 5, max_turns 5 (= batch size), view_limit 0, near-dup guard 0.8, cross-iteration memos
# off, round memo on. Judge and contrastive generator are openai/gpt-5.1 in both. Probe is
# gemma-3-27b-it L32 trained from scratch off data/instructions_llama70b_50.jsonl, both error
# types, 3 iterations. The preflight below asserts the two configs are identical apart from
# attacker.models and the output paths, so a delta between the arms cannot come from anything
# else.
#
# ATTEMPT VOLUME per arm: 20 sessions x 5 (batch size) x 5 rounds = ~500 conversations per error
# type per iteration, x 2 error types x 3 iterations = ~3000. Each is scored by a gemma-3-27b
# forward pass — that, not the eval, dominates wall clock.
#
# NOTE ON VOLUME PARITY. In batch mode batch_target only suppresses top-up calls for sessions
# whose first reply came back SHORT of max_turns; a session that returns a full batch breaks on
# `batch_complete` and never reaches the check. So if one attacker reliably returns 5
# conversations and the other returns 3, the arms' attempt counts diverge even at identical
# knobs. Before reading a delta as attacker quality, check the stop_reason distribution
# (batch_complete / batch_short / batch_no_parse / target_reached) in each arm's
# <jsonl>.runlog.jsonl.
#
# ACTIVATIONS. results_instructions_gemma27b_shared/ starts EMPTY on a clean box. Arm 1 fills
# it and arm 2 hits it, because those blobs depend only on the probe model / layer / seed /
# base data / eval splits / transforms — NOT on the attacker. The redteam_acts_*
# per-conversation cache written into the same dir is content-keyed with a frozen LLM, so the
# two arms' distinct successes get distinct keys. This is why the arms MUST run sequentially:
# two live writers can tear a blob.
#
# NO KAGGLE PREFETCH IN THIS EXPERIMENT. Unlike the high-stakes runs, there are no published
# gemma-3-27b activation blobs for the eval_instructions splits, so neither config carries a
# `kaggle:` section and arm 1 computes all 1302 eval rows locally, once. That is a real one-off
# cost at 27B — expect the first --eval to be long. It is cached under
# results_instructions_gemma27b_shared/eval_activations as `<split>-acts_full.pt` and reused by
# every later iteration and by arm 2.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, keeping the two arms' provenance separate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_instructions_attackers.sh > logs/run_gemma27b_instructions.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start
# failsafe_commit_gemma27b_instructions.sh alongside it — it already defaults to these two arms
# in this order.

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

check_model openai/gpt-oss-120b                 # attacker, ARM 1
check_model nvidia/nemotron-3-ultra-550b-a55b   # attacker, ARM 2
check_model openai/gpt-5.1                      # judge + summarizer + contrastive generator, BOTH arms

# --- preflight: the two configs differ in EXACTLY the attacker model --------------------------
# This experiment's whole claim is "same setup, different attacker", so the check is stricter
# than the budget-scaling one: every attacker/judge/probe/preprocessing/eval field except
# attacker.models must be equal, and both system prompts must be byte-identical.
.venv_claude/bin/python - <<'PY'
import sys
from agentic_redteam.config import load_config

a = load_config("configs/gptoss120b_instructions_gemma27b_batch_target60.md")   # gpt-oss-120b
b = load_config("configs/nemotron_instructions_gemma27b_batch_target60.md")     # nemotron-3-ultra

problems = []
if a.attacker.system_prompt != b.attacker.system_prompt:
    problems.append("attacker prompt differs between the arms")
if a.judge.system_prompt != b.judge.system_prompt:
    problems.append("judge prompt differs between the arms")

# Everything on the attacker EXCEPT the model list is held fixed.
for k in ("interface", "batch_submissions", "view_limit", "max_turns", "batch_target", "rounds",
          "concurrency", "sessions_per_model", "view_reshuffle", "near_dup_guard",
          "near_dup_threshold", "round_summaries", "capture_prompts", "cross_iteration_memos"):
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

for k in ("provider", "model", "max_concurrent", "max_tokens", "filter_percentile",
          "assistant_centric", "concept_description", "label_guidance"):
    va, vb = getattr(a.preprocessing, k), getattr(b.preprocessing, k)
    if va != vb:
        problems.append(f"preprocessing.{k}: {va!r} vs {vb!r}")

for k in ("combine_consecutive_messages", "convert_tool_to_assistant", "eval_max_samples"):
    va, vb = getattr(a.eval, k), getattr(b.eval, k)
    if va != vb:
        problems.append(f"eval.{k}: {va!r} vs {vb!r}")

# The one thing that IS supposed to differ.
if a.attacker.model_names != ["openai/gpt-oss-120b"]:
    problems.append(f"arm 1 attacker must be openai/gpt-oss-120b, got {a.attacker.model_names}")
if b.attacker.model_names != ["nvidia/nemotron-3-ultra-550b-a55b"]:
    problems.append(f"arm 2 attacker must be nvidia/nemotron-3-ultra-550b-a55b, got {b.attacker.model_names}")

# This concept is assistant-centric; the contrastive generator must be told so in BOTH arms.
for name, c in (("arm 1", a), ("arm 2", b)):
    if not c.preprocessing.assistant_centric:
        problems.append(f"{name}: preprocessing.assistant_centric must be true for this concept")
    need = c.attacker.sessions_per_model * len(c.attacker.models)
    if c.attacker.concurrency < need:
        problems.append(
            f"{name}: concurrency {c.attacker.concurrency} < sessions_per_model x models "
            f"({need}) — the extra sessions would queue on the semaphore and each would get "
            "its own success budget"
        )
    if c.kaggle is not None:
        problems.append(
            f"{name} carries a `kaggle:` section — no gemma-3-27b eval activations are "
            "published for the eval_instructions splits, so it would fail at the first eval"
        )

# Isolation of the per-arm outputs; deliberate sharing of the activation caches.
if a.output.jsonl_path == b.output.jsonl_path:
    problems.append("the arms share output.jsonl_path — they would write into one another")
if a.output.comparison_csv == b.output.comparison_csv:
    problems.append("the arms share output.comparison_csv — the second would overwrite the first")
if a.output.activations_cache_dir != b.output.activations_cache_dir:
    problems.append("the arms should SHARE activations_cache_dir (identical keys; arm 1 fills it)")
if a.output.base_activation_cache_dir != b.output.base_activation_cache_dir:
    problems.append("the arms should SHARE base_activation_cache_dir (identical keys)")

if problems:
    print("ERROR: the two arm configs are not in the intended relationship:", file=sys.stderr)
    for p in problems:
        print(f"       - {p}", file=sys.stderr)
    sys.exit(1)
print(">>> arm configs OK: identical apart from attacker.models and the per-arm output paths")
PY

# --- preflight: the eval splits' labels match the probe's ---------------------------------------
# evaluate_probe loads each split with the probe's own pos/neg class label strings, so a split
# whose `labels` don't match exactly would silently score as one class. Cheap to check here;
# hours into the run is not.
.venv_claude/bin/python - <<'PY'
import json, pathlib, sys

POS, NEG = "assistant_follows_the_instruction", "assistant_does_not_follow_the_instruction"
bad = []
splits = sorted(pathlib.Path("eval_instructions").glob("*.jsonl"))
if not splits:
    print("ERROR: eval_instructions/ holds no *.jsonl splits", file=sys.stderr)
    sys.exit(1)
for p in splits:
    labels = {json.loads(line)["labels"] for line in p.open() if line.strip()}
    extra = labels - {POS, NEG}
    if extra:
        bad.append(f"{p.name}: unexpected labels {sorted(extra)}")
if bad:
    print("ERROR: eval split labels do not match the probe's class labels:", file=sys.stderr)
    for b in bad:
        print(f"       - {b}", file=sys.stderr)
    sys.exit(1)
print(f">>> eval_instructions OK: {len(splits)} splits, labels match the probe")
PY

SHARED_CACHE="results_instructions_gemma27b_shared"   # shared, arm-independent activation cache

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in
# this list — it is meant to persist and grow across both arms and across re-runs.)
for p in results_instructions_gemma27b_gptoss probes/instructions_gemma27b_gptoss \
         results_instructions_gemma27b_nemotron probes/instructions_gemma27b_nemotron; do
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
        --base-training-data data/instructions_llama70b_50.jsonl \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_instructions \
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

run_arm configs/gptoss120b_instructions_gemma27b_batch_target60.md  probes/instructions_gemma27b_gptoss    logs/run_instructions_gemma27b_gptoss.log
run_arm configs/nemotron_instructions_gemma27b_batch_target60.md    probes/instructions_gemma27b_nemotron  logs/run_instructions_gemma27b_nemotron.log

echo ">>> $(date -Is)  both arms finished."
