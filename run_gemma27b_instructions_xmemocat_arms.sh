#!/usr/bin/env bash
set -e

# CATEGORY-STEERED MEMO experiment on the INSTRUCTION-FOLLOWING concept with a
# google/gemma-3-27b-it (L32) probe trained as a 10-MEMBER DEEP ENSEMBLE fit and scored
# SEQUENTIALLY, attacker in BATCH mode and BLIND (view_limit 0), cross-iteration memo ON at
# 150 words, over 5 retrain iterations. Two arms, run sequentially and fully isolated:
#
#   ARM 1 (attacker openai/gpt-oss-120b)
#       configs/gptoss120b_instructions_gemma27b_xmemocat.md
#       -> results_instructions_gemma27b_xmemocat_gptoss/   probes/instructions_gemma27b_xmemocat_gptoss
#
#   ARM 2 (attacker nvidia/nemotron-3-ultra-550b-a55b)
#       configs/nemotron_instructions_gemma27b_xmemocat.md
#       -> results_instructions_gemma27b_xmemocat_nemotron/ probes/instructions_gemma27b_xmemocat_nemotron
#
# THE ONLY VARIABLE BETWEEN THE ARMS IS attacker.models. This is
# experiment_instruction_cloud_5's run_gemma27b_instructions_xmemodesc_arms.sh with the same
# attacker pair and two changes, BOTH applied to BOTH arms — so they move the experiment, not
# the comparison inside it:
#
#   1. probe.description now ENUMERATES THE SIX CATEGORIES the negative class is made of —
#      unjustified refusal, biased answer, exaggeration, contradicting the source, context
#      drift, omission. Those six are the failure modes eval_sets/instructions and
#      dev_samples/instructions are actually built from (anthropic_harmless_refusal,
#      bbq_substitution, mm_substitution, hc_contradiction, hc_context_drift +
#      oig_context_drift, oig_omission), so the description names the shape of the data being
#      attacked instead of paraphrasing the concept. It reaches every attacker system prompt
#      and all three of the judge's prompts, verbatim.
#   2. BOTH SUMMARIZER PROMPTS are steered around that enumeration
#      (llm_judge._CATEGORY_COVERAGE_INSTRUCTION, shared by _summary_system and
#      _ITERATION_SUMMARY_SYSTEM_TEMPLATE, plus a new first question in each summarizer's user
#      turn). The judge must give every named category a status line — examined or not, robust
#      or fragile — INCLUDING the untouched ones, and close by naming which categories the next
#      samples should be drawn from, in priority order. The memo is the only thing a blind
#      attacker reads beyond its instructions, so that ordering is the steering signal. Without
#      it a rotation converges: the memo is written from the samples collected, the region
#      sampled most is written up most, and the memo feeds the next round.
#      The instruction is concept-agnostic and self-disabling — a description enumerating
#      nothing leaves both prompts as they were — so no existing config changes behaviour.
#   THE ARM VARIABLE, unchanged from cloud_5: openai/gpt-oss-120b (arm 1) vs
#   nvidia/nemotron-3-ultra-550b-a55b (arm 2).
#
# 1 + 2 are why this experiment's numbers are NOT comparable to cloud_5's — the search space,
# the labelling standard and the memo content all moved. Every OTHER knob is held at cloud_5's
# value, the memo word budgets included: the cross-iteration memo stays at 150 and the round
# memo at its 200-word repo target, so the roll-call has to earn its space inside the budget
# the memo already had rather than being handed more. Cloud_5's memos came out at 135-153 words
# against that 150 cap, i.e. the judge writes to the budget, so watch the first iteration's
# memos: if the six status lines crowd out the closing priority order — the part a blind
# attacker acts on — that is the signal to raise BOTH budgets, not just this one. probe_iter0 is the exception:
# no attacker touches it, so it should reproduce cloud_5's baseline exactly.
#
# PROBE_FUSED_ENSEMBLE=0 is exported below, as in cloud_5, so the ensemble is fit and scored
# SEQUENTIALLY (one ProbeFactory.build per seed, one predict_proba per member) — the path every
# earlier instruction experiment ran on. main's fused path is faster and moves AUROC only in the
# 4th decimal, but it is a different reduction order. The preflight asserts the setting took.
#
# Everything else is held at cloud_5's values: batch mode, view_limit 0, rounds 5, max_turns 5
# (= batch size), sessions_per_model 10, concurrency 10, batch_target 30, near-dup guard 0.8,
# round memo on, cross-iteration memo on with 30 successes shown, judge and contrastive
# generator openai/gpt-5.1, ensemble_size 10, validation.dev_data dev_samples/instructions, the
# kaggle eval+dev prefetch, both error types, probe gemma-3-27b-it L32 trained from scratch off
# data/instructions_llama70b_50.jsonl. The preflight below asserts the two configs are identical
# apart from attacker.models and the output paths — the memo knobs and the probe description
# included — and that _CATEGORY_COVERAGE_INSTRUCTION reaches both rendered summarizer prompts,
# so a delta between the arms cannot come from anything else.
#
# ATTEMPT VOLUME per arm: 10 sessions x 5 (batch size) x 5 rounds = ~250 conversations per
# error type per iteration, x 2 error types x 5 iterations = ~2500. Each is scored by a
# gemma-3-27b forward pass, and with both the eval and the dev set served from Kaggle that is
# essentially the only thing loading the 27B model, so it dominates wall clock outright.
#
# NOTE ON VOLUME PARITY. In batch mode batch_target only suppresses top-up calls for sessions
# whose first reply came back SHORT of max_turns; a session that returns a full batch breaks on
# `batch_complete` and never reaches the check. So the arms' attempt counts can diverge even at
# identical knobs — and here the arms run DIFFERENT models, which is exactly the thing that
# makes one return short batches. Before reading a delta as attacker quality, check the
# stop_reason distribution (batch_complete / batch_short / batch_no_parse / target_reached) in
# each arm's <jsonl>.runlog.jsonl.
#
# ACTIVATIONS. results_instructions_gemma27b_shared/ is the SAME path
# experiment_instruction_cloud_1, _3, _4 and _5 used, on purpose: eval, dev and base activations
# depend on the probe MODEL / layer / seed / base data / splits / transforms and NOT on the
# attacker knobs, the ensemble or the probe description, so blobs from those experiments are
# valid here. On a clean box it starts EMPTY, arm 1 fills it and arm 2 hits it. The
# redteam_acts_* per-conversation cache written into the same dir is content-keyed against a
# frozen LLM, so the two arms' distinct successes get distinct keys. This is why the arms MUST
# run sequentially: two live writers can tear a blob.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, keeping the two arms' provenance separate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_instructions_xmemocat_arms.sh > logs/run_gemma27b_instructions_xmemocat.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# SEQUENTIAL ensemble: one ProbeFactory.build per seed and one predict_proba per member, i.e.
# the pre-fusion path. ensemble.fusion_enabled() reads this single setting for BOTH halves, so
# the two can't end up half-reverted; the preflight asserts it took. Exported before any python
# starts because tuberlens' pydantic settings are populated at import.
export PROBE_FUSED_ENSEMBLE=0
echo ">>> ensemble fit/score path: PROBE_FUSED_ENSEMBLE=$PROBE_FUSED_ENSEMBLE (sequential)"

# Pin accelerate's per-device budget for EVERY extraction load. model_loading._resolve_max_memory
# gives this var precedence over tuberlens' own MAX_MEMORY, and the load line prints which source
# set it. UNPINNED accelerate infers the budget from whatever is FREE at load time and can
# silently fall back to CPU/disk offload — the load then logs "max_memory unpinned" and extraction
# crawls. 22 GiB of the 24 GiB card leaves room for the CUDA context and the forward's own
# activations; 45 GiB of the box's 62 GiB caps the CPU spill. Note this does NOT bound the
# probe-fit staging, which happens after the model is released and sizes itself from what is
# actually allocatable then (retrain._to_device_for_fit,
# AGENTIC_REDTEAM_FIT_STAGING_RESERVE_GIB). Export the var before launching to override.
: "${AGENTIC_REDTEAM_MAX_MEMORY:=0=22GiB,cpu=45GiB}"
export AGENTIC_REDTEAM_MAX_MEMORY
echo ">>> extraction memory budget: AGENTIC_REDTEAM_MAX_MEMORY=$AGENTIC_REDTEAM_MAX_MEMORY"

# --- preflight: model slugs are actually served -----------------------------------------------
# OpenRouter's /models is unauthenticated, but the key is sent anyway so the same credential
# problem shows up here rather than mid-run. Both ATTACKERS are checked — the experiment's
# variable is the model, so a dead slug in one arm silently turns this into a one-arm run.
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

# Kaggle credentials for the precomputed eval + dev activations (configs' `kaggle:` section).
# Checked HERE rather than at first use: the DEV prefetch runs before iteration 0 trains and an
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

check_model openai/gpt-oss-120b                # attacker, ARM 1
check_model nvidia/nemotron-3-ultra-550b-a55b  # attacker, ARM 2
check_model openai/gpt-5.1                     # judge + summarizer + contrastive generator, BOTH arms

# --- preflight: the two configs differ in EXACTLY attacker.models -----------------------------
# This experiment's whole claim is "same setup, different attacker", so the check is strict:
# every attacker/judge/probe/validation/preprocessing/eval/kaggle field except `models` must be
# equal, and both system prompts must be byte-identical. view_limit, the two cross-iteration
# memo knobs and the probe description are in that set AND are separately asserted per arm —
# "both arms sighted" or "both arms at the 900-word default" would pass an equality check and
# quietly turn this into a different experiment.
.venv_claude/bin/python - <<'PY'
import json
import sys
from pathlib import Path

from agentic_redteam.config import load_config
from agentic_redteam.ensemble import ENSEMBLE_SEEDS, MAX_ENSEMBLE_SIZE, fusion_enabled
from agentic_redteam.llm_judge import (
    _CATEGORY_COVERAGE_INSTRUCTION,
    _ITERATION_MEMO_WORD_BUDGET,
    _iteration_summary_system,
    _summary_system,
)

EXPECTED_ATTACKERS = (["openai/gpt-oss-120b"], ["nvidia/nemotron-3-ultra-550b-a55b"])
EXPECTED_ENSEMBLE_SIZE = 10
EXPECTED_MEMO_WORDS = 150
EXPECTED_VIEW_LIMIT = 0
EXPECTED_KAGGLE_OWNER = "anku7890"
EXPECTED_KAGGLE_SLUG = "{slug}-gemmaevalpt"
EXPECTED_KAGGLE_FILE = "{split}-gemmaeval.pt"
EXPECTED_DEV_SLUG = "{slug}-gemmadevpt"
EXPECTED_DEV_FILE = "{split}-gemmadev.pt"
EXPECTED_DEV_DIR = "dev_samples/instructions"
EVAL_DIR = "eval_sets/instructions"

a = load_config("configs/gptoss120b_instructions_gemma27b_xmemocat.md")   # gpt-oss-120b
b = load_config("configs/nemotron_instructions_gemma27b_xmemocat.md")     # nemotron-3-ultra

problems = []

# The ensemble path is a run-level env setting (PROBE_FUSED_ENSEMBLE), not a config field, so it
# cannot be asserted per arm — but it applies to both arms of this process, and getting it wrong
# silently changes the fit AND the scoring for the whole experiment.
if fusion_enabled():
    problems.append(
        "PROBE_FUSED_ENSEMBLE is on — this experiment runs the SEQUENTIAL ensemble path "
        "(one ProbeFactory.build per seed, one predict_proba per member). Export "
        "PROBE_FUSED_ENSEMBLE=0 before launching."
    )

if a.attacker.system_prompt != b.attacker.system_prompt:
    problems.append("attacker prompt differs between the arms")
if a.judge.system_prompt != b.judge.system_prompt:
    problems.append("judge prompt differs between the arms")

# Everything on the attacker EXCEPT `models` is held fixed — view_limit included, which is what
# makes this a different experiment from experiment_instruction_cloud_4.
for k in ("interface", "batch_submissions", "max_turns", "batch_target", "rounds",
          "concurrency", "sessions_per_model", "view_limit", "view_reshuffle",
          "view_reshuffle_interval", "view_balance", "view_training_seeds", "near_dup_guard",
          "near_dup_threshold", "near_dup_broadcast", "max_sample_tokens", "round_summaries",
          "capture_prompts", "cross_iteration_memos", "cross_iteration_memo_word_budget",
          "cross_iteration_memo_max_successes", "persistence_from_last_rounds"):
    va, vb = getattr(a.attacker, k), getattr(b.attacker, k)
    if va != vb:
        problems.append(f"attacker.{k}: {va!r} vs {vb!r}")

# `description` is in this list on purpose: it reaches the attacker's system prompt AND all
# three of the judge's prompts, so an arm carrying a different one is running a different task.
for k in ("model", "layer", "pos_class_label", "neg_class_label", "description", "error_types",
          "ensemble_size", "threshold"):
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
    problems.append("only one arm carries a `kaggle:` section — one would download activations "
                    "while the other extracted them locally")
elif a.kaggle is not None:
    for k in ("owner", "eval_dataset_slug", "eval_file_name", "dev_dataset_slug", "dev_file_name"):
        va, vb = getattr(a.kaggle, k), getattr(b.kaggle, k)
        if va != vb:
            problems.append(f"kaggle.{k}: {va!r} vs {vb!r}")

# The one thing that IS supposed to differ.
got = (a.attacker.model_names, b.attacker.model_names)
if got != EXPECTED_ATTACKERS:
    problems.append(
        f"attackers must be {EXPECTED_ATTACKERS[0]} in arm 1 and {EXPECTED_ATTACKERS[1]} in "
        f"arm 2, got {got} — with both equal there is no variable left and the two arms are "
        "the same run twice"
    )

# The probe description steers BOTH the attacker and the judge, and here it also supplies the
# category list the summarizers are told to roll-call — so a config that lost it, or reverted to
# cloud_5's prose form, is a different experiment rather than a cosmetic difference. Assert every
# category by name: dropping one silently removes it from the roll-call, and nothing downstream
# would report the omission.
EXPECTED_CATEGORIES = (
    "UNJUSTIFIED REFUSAL", "BIASED ANSWER", "EXAGGERATION",
    "CONTRADICTING THE SOURCE", "CONTEXT DRIFT", "OMISSION",
)
for phrase in ("FOLLOWS the instruction when",) + EXPECTED_CATEGORIES:
    if phrase not in (a.probe.description or ""):
        problems.append(
            f"probe.description is missing {phrase!r} — this experiment runs the "
            "SIX-CATEGORY description, which reaches the attacker's system prompt, all three of "
            "the judge's prompts, and (as the enumeration both summarizers organise their "
            "write-up around) the memos every later round and iteration reads"
        )

# The steering itself lives in the code, not the config: _CATEGORY_COVERAGE_INSTRUCTION is
# shared by the round summarizer and the cross-iteration one. Check it reaches BOTH rendered
# prompts — a refactor that wired it into only one would leave the round memo converging exactly
# as before while the header claims otherwise, and no artifact of the run would show it.
_probe_max_tokens = a.judge.max_tokens
for what, rendered in (
    ("round summarizer", _summary_system(_probe_max_tokens)),
    ("cross-iteration summarizer", _iteration_summary_system(a.attacker.cross_iteration_memo_word_budget)),
):
    if _CATEGORY_COVERAGE_INSTRUCTION not in rendered:
        problems.append(
            f"the {what}'s system prompt does not carry _CATEGORY_COVERAGE_INSTRUCTION — the "
            "category roll-call is what this experiment varies, so without it in both prompts "
            "this is a re-run of experiment_instruction_cloud_5 under a new name"
        )

for name, c in (("arm 1", a), ("arm 2", b)):
    if c.attacker.view_limit != EXPECTED_VIEW_LIMIT:
        problems.append(
            f"{name}: attacker.view_limit must be {EXPECTED_VIEW_LIMIT} — both arms are BLIND "
            f"here, got {c.attacker.view_limit!r}"
        )
    # Checked per arm, not just for equality: "both off" and "both at the 900-word default"
    # would each pass an equality check.
    if not c.attacker.cross_iteration_memos:
        problems.append(
            f"{name}: attacker.cross_iteration_memos must be true — with view_limit 0 it is the "
            "ONLY thing carrying anything across the iteration boundary, and with it off no "
            "<jsonl>.iteration_memos.jsonl is written at all"
        )
    if c.attacker.cross_iteration_memo_word_budget != EXPECTED_MEMO_WORDS:
        problems.append(
            f"{name}: attacker.cross_iteration_memo_word_budget must be {EXPECTED_MEMO_WORDS}, got "
            f"{c.attacker.cross_iteration_memo_word_budget!r} (unset means the repo default "
            f"{_ITERATION_MEMO_WORD_BUDGET}, which is 6x this experiment's budget)"
        )
    # A budget the judge cannot physically emit is worse than none: the memo is truncated
    # mid-sentence and fed back as the next iteration's prior_memo, so the loss compounds.
    # ~0.61 words/token measured for this dense-markdown register.
    reachable = int(c.judge.max_tokens * 0.61)
    if EXPECTED_MEMO_WORDS > reachable:
        problems.append(
            f"{name}: memo budget {EXPECTED_MEMO_WORDS} words exceeds what judge.max_tokens "
            f"{c.judge.max_tokens} can emit (~{reachable} words) — raise max_tokens or lower the budget"
        )
    if c.probe.ensemble_size != EXPECTED_ENSEMBLE_SIZE:
        problems.append(
            f"{name}: probe.ensemble_size must be {EXPECTED_ENSEMBLE_SIZE} in this experiment, "
            f"got {c.probe.ensemble_size!r} (the runner passes no --ensemble-size, so the config "
            "value is what is used)"
        )
    if not c.preprocessing.assistant_centric:
        problems.append(f"{name}: preprocessing.assistant_centric must be true for this concept")
    need = c.attacker.sessions_per_model * len(c.attacker.models)
    if c.attacker.concurrency < need:
        problems.append(
            f"{name}: concurrency {c.attacker.concurrency} < sessions_per_model x models "
            f"({need}) — the extra sessions would queue on the semaphore and each would get "
            "its own success budget"
        )
    if c.kaggle is None:
        problems.append(
            f"{name} carries NO `kaggle:` section — the eval would extract all 1302 "
            "eval_sets/instructions rows locally through gemma-3-27b, and the dev set another 436, "
            f"instead of downloading the published blobs from {EXPECTED_KAGGLE_OWNER}/…"
        )
    else:
        for field, want in (("owner", EXPECTED_KAGGLE_OWNER),
                            ("eval_dataset_slug", EXPECTED_KAGGLE_SLUG),
                            ("eval_file_name", EXPECTED_KAGGLE_FILE),
                            ("dev_dataset_slug", EXPECTED_DEV_SLUG),
                            ("dev_file_name", EXPECTED_DEV_FILE)):
            v = getattr(c.kaggle, field)
            if v != want:
                problems.append(f"{name}: kaggle.{field} {v!r} != {want!r}")
        # Every eval_sets/instructions stem contains an underscore, which Kaggle forbids in a
        # dataset slug — a {split}-based slug would name a dataset that cannot exist.
        for field in ("eval_dataset_slug", "dev_dataset_slug"):
            if "{slug}" not in getattr(c.kaggle, field):
                problems.append(
                    f"{name}: kaggle.{field} must use {{slug}}, not {{split}} — every "
                    "instructions split stem has an underscore and Kaggle slugs forbid them"
                )
    if c.eval.eval_max_samples != 0:
        problems.append(
            f"{name}: eval.eval_max_samples must be 0 to use precomputed full-split "
            f"activations, got {c.eval.eval_max_samples!r}"
        )
    if c.validation.dev_data is None:
        problems.append(
            f"{name}: validation.dev_data is not set — the fit would early-stop on a test_size "
            "slice, which experiment_instruction_cloud_3 exists to have replaced"
        )
    elif Path(c.validation.dev_data).resolve() != Path(EXPECTED_DEV_DIR).resolve():
        problems.append(
            f"{name}: validation.dev_data is {c.validation.dev_data}, expected {EXPECTED_DEV_DIR}"
        )
if a.validation.dev_data != b.validation.dev_data:
    problems.append("the arms early-stop against DIFFERENT dev sets — they are not comparable")

# The dev set must be disjoint from the eval splits, or the fit selects its checkpoint on the
# test set and every reported AUROC is optimistic. Verified on the actual rows, not assumed
# from the directory names, and done here because it is cheap and unrecoverable afterwards.
if a.validation.dev_data is not None:
    dev_dir = Path(a.validation.dev_data)

    def _keys(d):
        out = set()
        for f in sorted(Path(d).glob("*.jsonl")):
            for line in f.open():
                if line.strip():
                    out.add(json.dumps(json.loads(line)["inputs"], sort_keys=True))
        return out

    dev_keys, eval_keys = _keys(dev_dir), _keys(EVAL_DIR)
    if not dev_keys:
        problems.append(f"validation.dev_data {dev_dir} holds no rows")
    shared = dev_keys & eval_keys
    if shared:
        problems.append(
            f"{len(shared)} conversation(s) appear in BOTH {dev_dir} and {EVAL_DIR} — the probe "
            "would early-stop on rows it is then scored against"
        )
    else:
        print(f">>> dev/eval disjoint: {len(dev_keys)} dev rows, {len(eval_keys)} eval rows, "
              "no shared conversations")

# Isolation of the per-arm outputs; deliberate sharing of the activation caches.
if a.output.jsonl_path == b.output.jsonl_path:
    problems.append("the arms share output.jsonl_path — they would write into one another, and "
                    "the cross-iteration memo sidecar is derived from it")
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
print(f">>> attacker: arm 1 = {a.attacker.model_names[0]}, arm 2 = {b.attacker.model_names[0]}")
print(f">>> attacker is BLIND in both arms: view_limit {a.attacker.view_limit}")
print(f">>> cross-iteration memo ON in both arms @ {EXPECTED_MEMO_WORDS} words "
      f"(repo default {_ITERATION_MEMO_WORD_BUDGET}), "
      f"{a.attacker.cross_iteration_memo_max_successes} successes shown to the judge")
print(f">>> deep ensemble ON in both arms: {EXPECTED_ENSEMBLE_SIZE} members per train/retrain "
      f"(max {MAX_ENSEMBLE_SIZE}), seeds {list(ENSEMBLE_SEEDS[:EXPECTED_ENSEMBLE_SIZE])}, "
      f"fused={fusion_enabled()} (sequential fit and scoring)")
print(f">>> probe description ({len((a.probe.description or '').split())} words, "
      f"{len(EXPECTED_CATEGORIES)} named categories) reaches the attacker system prompt and all "
      "three judge prompts")
print(">>> category roll-call steering ON: _CATEGORY_COVERAGE_INSTRUCTION is in both the round "
      "and the cross-iteration summarizer system prompts")
print(f">>> kaggle eval prefetch ON in both arms: {a.kaggle.owner}/"
      f"{a.kaggle.eval_dataset_slug} :: {a.kaggle.eval_file_name}")
print(f">>> kaggle dev  prefetch ON in both arms: {a.kaggle.owner}/"
      f"{a.kaggle.dev_dataset_slug} :: {a.kaggle.dev_file_name}")
print(f">>> validation: held-out dev set {a.validation.dev_data} (base data and red-team "
      "successes train in full)")
PY

# --- preflight: the eval splits' labels match the probe's ---------------------------------------
# evaluate_probe loads each split with the probe's own pos/neg class label strings, so a split
# whose `labels` don't match exactly would silently score as one class. Cheap to check here;
# hours into the run is not.
.venv_claude/bin/python - <<'PY'
import json, pathlib, sys

POS, NEG = "assistant_follows_the_instruction", "assistant_does_not_follow_the_instruction"
bad = []
splits = sorted(pathlib.Path("eval_sets/instructions").glob("*.jsonl"))
if not splits:
    print("ERROR: eval_sets/instructions/ holds no *.jsonl splits", file=sys.stderr)
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
print(f">>> eval_sets/instructions OK: {len(splits)} splits, labels match the probe")
PY

SHARED_CACHE="results_instructions_gemma27b_shared"   # shared, arm-independent activation cache

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in
# this list — it is meant to persist and grow across both arms, across re-runs, and across
# experiments on the same probe/splits.)
#
# --resume turns the guard off, because on a resume those directories existing is the POINT:
# the CLI reads probe_iterN.pkl and the redteam_done_* markers out of the probe dir to pick up
# where it stopped, and re-running the red-team phases it already paid OpenRouter for would be
# the expensive mistake. The cross-iteration memo resumes with it — IterationMemoStore re-reads
# <jsonl>.iteration_memos.jsonl on init, so a restarted run does not lose the memo chain.
RESUME=0
for arg in "$@"; do
    case "$arg" in
        --resume) RESUME=1 ;;
        *) echo "ERROR: unknown argument $arg (only --resume is accepted)" >&2; exit 2 ;;
    esac
done

if [ "$RESUME" -eq 1 ]; then
    echo ">>> --resume: keeping existing per-arm outputs; each arm continues from its latest probe_iterN.pkl"
else
    for p in results_instructions_gemma27b_xmemocat_gptoss probes/instructions_gemma27b_xmemocat_gptoss \
             results_instructions_gemma27b_xmemocat_nemotron probes/instructions_gemma27b_xmemocat_nemotron; do
        [ -e "$p" ] && { echo "ERROR: $p already exists — move it aside, bump the suffix, or pass --resume." >&2; exit 1; }
    done
fi

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
if [ -n "$(ls -A "$SHARED_CACHE/eval_activations" 2>/dev/null)" ]; then
    echo ">>> activation cache: $SHARED_CACHE (eval blobs already present — no Kaggle download needed)"
else
    echo ">>> activation cache: $SHARED_CACHE (empty — arm 1 downloads ~1.45 GB of eval blobs (~4.9 GB unpacked) plus ~1.31 GB of dev blobs from Kaggle)"
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
        --eval --eval-dataset-dir eval_sets/instructions \
        > "$3" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
        # The circuit breaker stopped the run: OpenRouter is out of credits or the key is
        # dead. Do NOT start the next arm — it would fail identically and write a comparison
        # CSV from probes trained on nothing.
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

run_arm configs/gptoss120b_instructions_gemma27b_xmemocat.md  probes/instructions_gemma27b_xmemocat_gptoss    logs/run_instructions_gemma27b_xmemocat_gptoss.log
run_arm configs/nemotron_instructions_gemma27b_xmemocat.md    probes/instructions_gemma27b_xmemocat_nemotron  logs/run_instructions_gemma27b_xmemocat_nemotron.log

echo ">>> $(date -Is)  both arms finished."
