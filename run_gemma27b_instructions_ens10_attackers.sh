#!/usr/bin/env bash
set -e

# ATTACKER-MODEL experiment on the INSTRUCTION-FOLLOWING concept with a google/gemma-3-27b-it
# (L32) probe trained as a 10-MEMBER DEEP ENSEMBLE, both arms attacking in BATCH mode, over 5
# retrain iterations. Two arms, run sequentially and fully isolated:
#
#   ARM 1 (attacker openai/gpt-oss-120b)
#       configs/gptoss120b_instructions_gemma27b_ens10.md
#       -> results_instructions_gemma27b_ens10_gptoss/   probes/instructions_gemma27b_ens10_gptoss
#
#   ARM 2 (attacker nvidia/nemotron-3-ultra-550b-a55b)
#       configs/nemotron_instructions_gemma27b_ens10.md
#       -> results_instructions_gemma27b_ens10_nemotron/ probes/instructions_gemma27b_ens10_nemotron
#
# THE ONLY VARIABLE IS THE ATTACKER MODEL. This is experiment_instruction_cloud_1's
# run_gemma27b_instructions_attackers.sh with exactly FOUR changes, all applied to BOTH arms:
#
#   1. probe.ensemble_size: 10 — every train/retrain fits TEN linear_then_softmax probes on the
#      SAME activations under the repo-pinned ENSEMBLE_SEEDS[:10] and averages their
#      probabilities into one score (agentic_redteam/ensemble.py). The attacker, the judge, the
#      JSONL row and the eval all still see one score and one prediction. Only the FIT repeats:
#      the split, the extraction and the activation caches are shared across members, so member
#      k > 0 costs a probe-head fit, not another pass through gemma-3-27b.
#   2. --iterations 3 -> 5 below.
#   3. sessions_per_model 20 -> 10, concurrency 20 -> 10, batch_target 60 -> 30 in both
#      configs — experiment11_cloud's scheduling shape. max_turns stays 5 (it IS the batch
#      size here), so round volume halves to sessions_per_model x max_turns = 50. In batch
#      mode batch_target caps nothing (see the volume parity note below); it only stops
#      paying for TOP-UP calls from sessions whose first reply came back short.
#   4. A `kaggle:` section in both configs. Precomputed gemma-3-27b L32 activations for all
#      seven eval_instructions splits are now published at anku7890/{slug}-gemmaevalpt, so
#      the eval downloads ~1.45 GB of validated blobs (~4.9 GB on disk) instead of running
#      1302 rows through a 27B model. This needs Kaggle credentials — checked below, up
#      front — and disk for the unpacked blobs.
#
# Everything else is held at experiment_instruction_cloud_1's shape: batch mode, rounds 5,
# max_turns 5 (= batch size), view_limit 0,
# near-dup guard 0.8, cross-iteration memos off, round memo on. Judge and contrastive generator
# are openai/gpt-5.1 in both. Probe is gemma-3-27b-it L32 trained from scratch off
# data/instructions_llama70b_50.jsonl, both error types. The preflight below asserts the two
# configs are identical apart from attacker.models and the output paths — ensemble_size
# included — so a delta between the arms cannot come from anything else.
#
# ATTEMPT VOLUME per arm: 10 sessions x 5 (batch size) x 5 rounds = ~250 conversations per error
# type per iteration, x 2 error types x 5 iterations = ~2500 — the same per-round width as
# experiment11_cloud, over more iterations. Each is scored by a gemma-3-27b forward pass, and
# with the eval now served from Kaggle that is essentially the only thing loading the 27B model,
# so it dominates wall clock outright. Against experiment_instruction_cloud_1 that is 5/6 of the
# red-team volume per arm (half the round width, 5/3 the iterations) and none of its eval cost.
#
# NOTE ON VOLUME PARITY. In batch mode batch_target only suppresses top-up calls for sessions
# whose first reply came back SHORT of max_turns; a session that returns a full batch breaks on
# `batch_complete` and never reaches the check. So if one attacker reliably returns 5
# conversations and the other returns 3, the arms' attempt counts diverge even at identical
# knobs. Before reading a delta as attacker quality, check the stop_reason distribution
# (batch_complete / batch_short / batch_no_parse / target_reached) in each arm's
# <jsonl>.runlog.jsonl.
#
# ACTIVATIONS. results_instructions_gemma27b_shared/ is the SAME path
# experiment_instruction_cloud_1 used, on purpose: eval and base activations depend on the probe
# MODEL / layer / seed / base data / eval splits / transforms and NOT on the attacker or on the
# ensemble size (ensembling changes the fit, not the extraction), so blobs from that experiment
# are valid here. On a clean box it starts EMPTY, arm 1 fills it and arm 2 hits it. The
# redteam_acts_* per-conversation cache written into the same dir is content-keyed with a frozen
# LLM, so the two arms' distinct successes get distinct keys. This is why the arms MUST run
# sequentially: two live writers can tear a blob.
#
# KAGGLE PREFETCH — the change from experiment_instruction_cloud_1, which had none. Both configs
# now carry a `kaggle:` section pointing at anku7890/{slug}-gemmaevalpt, so arm 1's first --eval
# downloads the precomputed gemma-3-27b activations for all seven eval_instructions splits
# (~1.45 GB compressed / ~4.9 GB on disk, one file per split, named <split>-gemmaeval.pt) into
# results_instructions_gemma27b_shared/eval_activations as `<split>-acts_full.pt`. Each blob is
# validated against the probe's model_name/layer and the split's row count before use, and a
# split that cannot be fetched RAISES rather than falling back to local extraction. NO LLM is
# loaded for the eval at all — that is the point, and it removes what was the largest
# non-red-team cost of the predecessor. The BASE split is still computed locally by arm 1, as is
# every red-team conversation.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, keeping the two arms' provenance separate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_instructions_ens10_attackers.sh > logs/run_gemma27b_instructions_ens10.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start
# failsafe_commit_gemma27b_instructions_ens10.sh alongside it — it already defaults to these two
# arms in this order.

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

# Kaggle credentials for the precomputed eval activations (configs' `kaggle:` section).
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

check_model openai/gpt-oss-120b                 # attacker, ARM 1
check_model nvidia/nemotron-3-ultra-550b-a55b   # attacker, ARM 2
check_model openai/gpt-5.1                      # judge + summarizer + contrastive generator, BOTH arms

# --- preflight: the two configs differ in EXACTLY the attacker model --------------------------
# This experiment's whole claim is "same setup, different attacker", so the check is stricter
# than the budget-scaling one: every attacker/judge/probe/preprocessing/eval field except
# attacker.models must be equal, and both system prompts must be byte-identical. probe.
# ensemble_size is in that set AND is separately asserted to be 10 — an ensemble that silently
# came back to 1 in one arm would look like an attacker effect.
.venv_claude/bin/python - <<'PY'
import sys
from agentic_redteam.config import load_config
from agentic_redteam.ensemble import ENSEMBLE_SEEDS, MAX_ENSEMBLE_SIZE

EXPECTED_ENSEMBLE_SIZE = 10
EXPECTED_KAGGLE_OWNER = "anku7890"
EXPECTED_KAGGLE_SLUG = "{slug}-gemmaevalpt"
EXPECTED_KAGGLE_FILE = "{split}-gemmaeval.pt"

a = load_config("configs/gptoss120b_instructions_gemma27b_ens10.md")   # gpt-oss-120b
b = load_config("configs/nemotron_instructions_gemma27b_ens10.md")     # nemotron-3-ultra

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

for k in ("model", "layer", "pos_class_label", "neg_class_label", "error_types", "ensemble_size"):
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

if (a.kaggle is None) != (b.kaggle is None):
    problems.append("only one arm carries a `kaggle:` section — one would download eval "
                    "activations while the other extracted them locally")
elif a.kaggle is not None:
    for k in ("owner", "eval_dataset_slug", "eval_file_name"):
        va, vb = getattr(a.kaggle, k), getattr(b.kaggle, k)
        if va != vb:
            problems.append(f"kaggle.{k}: {va!r} vs {vb!r}")

# The one thing that IS supposed to differ.
if a.attacker.model_names != ["openai/gpt-oss-120b"]:
    problems.append(f"arm 1 attacker must be openai/gpt-oss-120b, got {a.attacker.model_names}")
if b.attacker.model_names != ["nvidia/nemotron-3-ultra-550b-a55b"]:
    problems.append(f"arm 2 attacker must be nvidia/nemotron-3-ultra-550b-a55b, got {b.attacker.model_names}")

# The new setting in this experiment. Checked per arm rather than only for equality, because
# "both arms are 1" would pass an equality check and quietly turn this into a rerun of
# experiment_instruction_cloud_1 at 5 iterations.
for name, c in (("arm 1", a), ("arm 2", b)):
    if c.probe.ensemble_size != EXPECTED_ENSEMBLE_SIZE:
        problems.append(
            f"{name}: probe.ensemble_size must be {EXPECTED_ENSEMBLE_SIZE} in this experiment, "
            f"got {c.probe.ensemble_size!r} (note the runner passes no --ensemble-size, so the "
            "config value is what is used)"
        )

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
    # Inverted from experiment_instruction_cloud_1, where a `kaggle:` section was a bug: the
    # blobs now exist, so a config MISSING the section would silently spend hours extracting
    # 1302 rows through gemma-3-27b instead of downloading them.
    if c.kaggle is None:
        problems.append(
            f"{name} carries NO `kaggle:` section — the eval would extract all 1302 "
            "eval_instructions rows locally through gemma-3-27b instead of downloading the "
            f"published blobs from {EXPECTED_KAGGLE_OWNER}/{EXPECTED_KAGGLE_SLUG}"
        )
    else:
        if c.kaggle.owner != EXPECTED_KAGGLE_OWNER:
            problems.append(f"{name}: kaggle.owner {c.kaggle.owner!r} != {EXPECTED_KAGGLE_OWNER!r}")
        if c.kaggle.eval_dataset_slug != EXPECTED_KAGGLE_SLUG:
            problems.append(
                f"{name}: kaggle.eval_dataset_slug {c.kaggle.eval_dataset_slug!r} != "
                f"{EXPECTED_KAGGLE_SLUG!r}"
            )
        if c.kaggle.eval_file_name != EXPECTED_KAGGLE_FILE:
            problems.append(
                f"{name}: kaggle.eval_file_name {c.kaggle.eval_file_name!r} != "
                f"{EXPECTED_KAGGLE_FILE!r}"
            )
        # Every eval_instructions stem contains an underscore, which Kaggle forbids in a
        # dataset slug — a {split}-based slug would name a dataset that cannot exist.
        if "{slug}" not in c.kaggle.eval_dataset_slug:
            problems.append(
                f"{name}: kaggle.eval_dataset_slug must use {{slug}}, not {{split}} — every "
                "eval_instructions split stem has an underscore and Kaggle slugs forbid them"
            )
    if c.eval.eval_max_samples != 0:
        problems.append(
            f"{name}: eval.eval_max_samples must be 0 to use precomputed full-split "
            f"activations, got {c.eval.eval_max_samples!r}"
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
print(f">>> deep ensemble ON in both arms: {EXPECTED_ENSEMBLE_SIZE} members per train/retrain "
      f"(max {MAX_ENSEMBLE_SIZE}), seeds {list(ENSEMBLE_SEEDS[:EXPECTED_ENSEMBLE_SIZE])}")
print(f">>> kaggle eval prefetch ON in both arms: {a.kaggle.owner}/"
      f"{a.kaggle.eval_dataset_slug} :: {a.kaggle.eval_file_name}")
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
# this list — it is meant to persist and grow across both arms, across re-runs, and across
# experiments on the same probe/splits.)
for p in results_instructions_gemma27b_ens10_gptoss probes/instructions_gemma27b_ens10_gptoss \
         results_instructions_gemma27b_ens10_nemotron probes/instructions_gemma27b_ens10_nemotron; do
    [ -e "$p" ] && { echo "ERROR: $p already exists — move it aside or bump the suffix." >&2; exit 1; }
done

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
if [ -n "$(ls -A "$SHARED_CACHE/eval_activations" 2>/dev/null)" ]; then
    echo ">>> activation cache: $SHARED_CACHE (eval blobs already present — no Kaggle download needed)"
else
    echo ">>> activation cache: $SHARED_CACHE (empty — arm 1's first eval downloads ~1.45 GB from Kaggle, ~4.9 GB unpacked)"
fi

# --- run one arm ------------------------------------------------------------------------------
# Exit code the CLI uses for "OpenRouter is unusable" (cli.OUTAGE_EXIT_CODE).
OUTAGE_EXIT_CODE=3

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (log: $3)"
    local rc=0
    # NOTE: no --ensemble-size flag. The size comes from probe.ensemble_size in the config
    # (flag > config > inherit-from-the-probe-being-retrained), which the preflight above
    # asserts is 10 in both arms.
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 5 \
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

run_arm configs/gptoss120b_instructions_gemma27b_ens10.md  probes/instructions_gemma27b_ens10_gptoss    logs/run_instructions_gemma27b_ens10_gptoss.log
run_arm configs/nemotron_instructions_gemma27b_ens10.md    probes/instructions_gemma27b_ens10_nemotron  logs/run_instructions_gemma27b_ens10_nemotron.log

echo ">>> $(date -Is)  both arms finished."
