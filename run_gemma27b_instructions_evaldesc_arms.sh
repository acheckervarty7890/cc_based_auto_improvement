#!/usr/bin/env bash
set -e

# EVAL-DATA-DESCRIPTION experiment on the INSTRUCTION-FOLLOWING concept. Probe =
# google/gemma-3-27b-it (L32) trained as a 10-MEMBER DEEP ENSEMBLE, fit and scored SEQUENTIALLY,
# attacker in BATCH mode and BLIND (view_limit 0), cross-iteration memo ON at 150 words, over 5
# retrain iterations. Two arms, run sequentially and fully isolated:
#
#   ARM 1 (eval.data_description = eval_sets/instructions/oig_omission)
#       configs/gptoss120b_instructions_gemma27b_evaldesc_omission.md
#       -> results_instructions_gemma27b_evaldesc_omission/
#          probes/instructions_gemma27b_evaldesc_omission
#
#   ARM 2 (eval.data_description = eval_sets/instructions/oig_context_drift)
#       configs/gptoss120b_instructions_gemma27b_evaldesc_drift.md
#       -> results_instructions_gemma27b_evaldesc_drift/
#          probes/instructions_gemma27b_evaldesc_drift
#
# THE ONLY VARIABLE BETWEEN THE ARMS IS `eval.data_description` — the free text describing the
# conversation data the probe is SCORED on. It reaches the judge's two SUMMARIZERS only (one
# `## Task context` bullet in each user prompt, a coverage paragraph in each system prompt, one
# extra question in each). It does NOT reach the judge's classification prompt and it does NOT
# reach the attacker directly, so the labelling function does not move between the arms and the
# attacker's own instructions are byte-identical. What differs is what the MEMOS say.
#
# WHY THAT IS ENOUGH TO STEER ANYTHING. With `batch_submissions: true` and `view_limit: 0` a
# session sees no verdict on its own submissions and no sample of anyone else's. The rolling
# round memo and the cross-iteration memo are the ONLY things crossing into it. So the memo is
# the whole channel, and steering the memo is the only available way to steer coverage.
#
# ONE KIND, NOT SEVEN. eval_sets/instructions has seven splits, one per failure mode of the
# negative class. A description naming all seven would steer the memo to SPREAD over them. Each
# arm names exactly ONE, so there is nothing to spread over and the steering inverts from
# breadth to depth — how much of a round's evidence actually had that shape, and what within the
# shape is still untried. llm_judge's two coverage paragraphs and two coverage questions are
# written for that one-kind case.
#
# WHY THESE TWO SPLITS. oig_omission and oig_context_drift are the tightest pair in the eval
# set: same source corpus, same register, same short factual-QA material, both exactly
# class-balanced. What differs is where the failure lives, and therefore which surface cue runs
# with the label:
#
#   ARM 1  oig_omission        one turn; the reply answers some of the questions asked and
#                              silently drops the rest. Cue: QUANTITY — the incomplete reply is
#                              the shorter one, one answer short of the request.
#   ARM 2  oig_context_drift   two turns; the second reply answers the FIRST question again
#                              instead of the one just asked. Cue: REPETITION — median
#                              first/second-turn similarity 0.70 on the not-following side
#                              against 0.25 on the following side.
#
# So the pair asks whether a memo-writer pointed at a COUNTING cue steers the attacker somewhere
# different from one pointed at a REPETITION cue — and whether either steering reaches the split
# it names, the other OIG split, or all seven.
#
# THE EVAL IS UNCHANGED IN BOTH ARMS. The probe is still scored on all seven
# eval_sets/instructions splits and each arm's comparison CSV still reports all seven. That is
# what makes the cross-split effect readable: if describing one split lifts only that split,
# that is steering; if it lifts all seven, it is something else.
#
# THE CONTROL is experiment_instruction_cloud_4's ARM 1
# (configs/gptoss120b_instructions_gemma27b_xmemo.md, view_limit 0), which both arms are
# otherwise copies of: same attacker, same schedule, same one-line probe.description, same memo
# knobs at the same budgets, same base data, dev set, eval splits, transforms and five
# iterations. It sets no `eval.data_description`, so its six judge prompts are the pre-knob ones
# — which scripts/verify_memo_prompt_knobs.py pins independently.
#
# NOT comparable to experiment_instruction_cloud_5 or _6: _5 varied the attacker model and _6
# expanded probe.description into a six-category enumeration, which moves the ATTACKER prompt
# and the JUDGE'S CLASSIFICATION PROMPT — the labelling function itself. Both arms here keep
# cloud_4's bare one-line description on purpose.
#
# PROBE_FUSED_ENSEMBLE=0 is exported below, as in every earlier instruction experiment, so the
# ensemble is fit and scored SEQUENTIALLY (one ProbeFactory.build per seed, one predict_proba
# per member). main's fused path is faster and moves AUROC only in the 4th decimal, but it is a
# different reduction order. The preflight asserts the setting took.
#
# ATTEMPT VOLUME per arm: 10 sessions x 5 (batch size) x 5 rounds = ~250 conversations per error
# type per iteration, x 2 error types x 5 iterations = ~2500. Each is scored by a gemma-3-27b
# forward pass, and with both the eval and the dev set served from Kaggle that is essentially
# the only thing loading the 27B model, so it dominates wall clock outright.
#
# NOTE ON VOLUME PARITY. In batch mode batch_target only suppresses top-up calls for sessions
# whose first reply came back SHORT of max_turns; a session returning a full batch breaks on
# `batch_complete` and never reaches the check. Both arms run the SAME attacker here, so a
# volume difference is not a model effect — check the stop_reason distribution
# (batch_complete / batch_short / batch_no_parse / target_reached) in each arm's
# <jsonl>.runlog.jsonl before reading anything into it.
#
# ACTIVATIONS. results_instructions_gemma27b_shared/ is the SAME path
# experiment_instruction_cloud_1/_3/_4/_5/_6 used, on purpose: eval, dev and base activations
# depend on the probe MODEL / layer / seed / base data / splits / transforms and NOT on the
# attacker knobs, the ensemble, the probe description or the eval-data description, so blobs
# from those experiments are valid here. On a clean box it starts EMPTY, arm 1 fills it and arm
# 2 hits it. The redteam_acts_* per-conversation cache written into the same dir is
# content-keyed against a frozen LLM, so the two arms' distinct successes get distinct keys.
# This is why the arms MUST run sequentially: two live writers can tear a blob.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, keeping the two arms' provenance separate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_instructions_evaldesc_arms.sh > logs/run_evaldesc_arms.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it —
# its built-in stage list already points at these two arms, in this order:
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &

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
# silently fall back to CPU/disk offload — the load then logs "max_memory unpinned" and
# extraction crawls (measured 48-264 s/sample against ~2.8 s/sample resident). 22 GiB of the
# 24 GiB card leaves room for the CUDA context and the forward's own activations; 45 GiB of the
# box's ~62 GiB caps the CPU spill. ADJUST IF THIS BOX IS DIFFERENT. This does NOT bound the
# probe-fit staging, which happens after the model is released and sizes itself from what is
# actually allocatable then (retrain._to_device_for_fit).
: "${AGENTIC_REDTEAM_MAX_MEMORY:=0=22GiB,cpu=45GiB}"
export AGENTIC_REDTEAM_MAX_MEMORY
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"   # reaches get_performances too
echo ">>> extraction memory budget: AGENTIC_REDTEAM_MAX_MEMORY=$AGENTIC_REDTEAM_MAX_MEMORY (placement only)"

# HuggingFace token. tuberlens' LLMModel.load calls hf_login(), which RAISES
# ValueError("No HuggingFace token found") when no token is set — even though the gemma weights
# are already in the local HF cache and no download happens. Without this check the run gets all
# the way past the initial train and the iter0 eval before dying at the first red-team model
# load. (The token only has to EXIST; an expired one logs a warning and the cached load proceeds.)
if [ -z "${HF_TOKEN:-}" ] && [ -f hf_token.txt ]; then
    HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt)"
    export HF_TOKEN
fi
: "${HF_TOKEN:?export HF_TOKEN (or put it in hf_token.txt) — tuberlens hf_login() raises without one, even for a fully cached model}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo ">>> HF token: present (${#HF_TOKEN} chars)"

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
.venv_claude/bin/python -c "import kaggle" 2>/dev/null || {
    echo "ERROR: the 'kaggle' package is not installed: .venv_claude/bin/pip install kaggle" >&2
    exit 1
}

check_model openai/gpt-oss-120b   # attacker, BOTH arms
check_model openai/gpt-5.1        # judge + summarizer + contrastive generator, BOTH arms

# --- preflight: the two configs differ in EXACTLY eval.data_description ------------------------
# This experiment's whole claim is "same setup, different eval-data description", so the check is
# strict: every attacker/judge/probe/validation/preprocessing/eval/kaggle field EXCEPT
# `eval.data_description` must be equal, both system prompts must be byte-identical, and the two
# descriptions must both be set and must differ. view_limit, the memo knobs and the probe
# description are in that set AND are separately asserted per arm — "both arms sighted" or "both
# arms at the 900-word default" would pass an equality check and quietly become a different
# experiment.
.venv_claude/bin/python - <<'PY'
import inspect
import json
import sys
from pathlib import Path

from agentic_redteam.config import load_config
from agentic_redteam.ensemble import ENSEMBLE_SEEDS, MAX_ENSEMBLE_SIZE, fusion_enabled
from agentic_redteam.llm_judge import (
    DEFAULT_ITERATION_MEMO_WORD_BUDGET,
    _build_judge_request,
    _iteration_coverage_paragraph,
    _round_coverage_paragraph,
)
from agentic_redteam.persistence import Conversation, Message

EXPECTED_ATTACKER = ["openai/gpt-oss-120b"]
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

# The split each arm's description is ABOUT, and a phrase that can only come from having read
# that split. Asserted because the description is free text: an arm whose description was
# copy-pasted from its sibling, or trimmed until it no longer names the cue llm_judge's coverage
# paragraphs are written to pick up, would run and produce plausible numbers meaning nothing.
ARM_SPLITS = {
    "arm 1": ("oig_omission", "Completeness is doing all the work here"),
    "arm 2": ("oig_context_drift", "Repetition is doing all the work here"),
}

a = load_config("configs/gptoss120b_instructions_gemma27b_evaldesc_omission.md")
b = load_config("configs/gptoss120b_instructions_gemma27b_evaldesc_drift.md")

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

# EVERYTHING on the attacker is held fixed — there is no attacker-side variable in this
# experiment at all.
for k in ("interface", "batch_submissions", "max_turns", "batch_target", "rounds",
          "concurrency", "sessions_per_model", "view_limit", "view_reshuffle",
          "view_reshuffle_interval", "view_balance", "view_training_seeds", "near_dup_guard",
          "near_dup_threshold", "near_dup_broadcast", "max_sample_tokens", "round_summaries",
          "capture_prompts", "cross_iteration_memos", "cross_iteration_memo_word_budget",
          "cross_iteration_memo_max_successes", "persistence_from_last_rounds"):
    va, vb = getattr(a.attacker, k), getattr(b.attacker, k)
    if va != vb:
        problems.append(f"attacker.{k}: {va!r} vs {vb!r}")
if a.attacker.model_names != b.attacker.model_names:
    problems.append(f"attacker.models: {a.attacker.model_names} vs {b.attacker.model_names}")

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

# NOTE `data_description` is deliberately absent from this list — it is the variable.
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
if a.eval.data_description == b.eval.data_description:
    problems.append(
        "the two arms carry the SAME eval.data_description — with both equal there is no "
        "variable left and this is the same run twice"
    )

for name, c in (("arm 1", a), ("arm 2", b)):
    split, marker = ARM_SPLITS[name]
    desc = c.eval.data_description
    if not desc.strip():
        problems.append(
            f"{name}: eval.data_description is EMPTY — llm_judge keys the whole feature off "
            "truthiness, so this arm would send the pre-knob prompts and be the control, not an arm"
        )
    elif marker not in desc:
        problems.append(
            f"{name}: eval.data_description does not contain {marker!r} — that sentence names "
            f"the surface cue that runs with the label in {split}, and it is the clause "
            "llm_judge's coverage paragraphs are written to pick up. Without it the description "
            "still renders, but the memo cannot tell a finding about the concept from a finding "
            "about the artefact."
        )
    if not Path(EVAL_DIR, f"{split}.jsonl").exists():
        problems.append(f"{name}: {EVAL_DIR}/{split}.jsonl does not exist — the arm describes a "
                        "split the eval does not contain")

    if c.attacker.model_names != EXPECTED_ATTACKER:
        problems.append(f"{name}: attacker.models must be {EXPECTED_ATTACKER}, got "
                        f"{c.attacker.model_names}")
    if c.attacker.view_limit != EXPECTED_VIEW_LIMIT:
        problems.append(
            f"{name}: attacker.view_limit must be {EXPECTED_VIEW_LIMIT} — both arms are BLIND "
            "here, and that is what makes the memos the only channel the eval-data description "
            f"could steer. Got {c.attacker.view_limit!r}"
        )
    if not c.attacker.batch_submissions:
        problems.append(
            f"{name}: attacker.batch_submissions must be true — with the per-turn loop the "
            "attacker also sees its own verdicts, so the memo stops being the only channel"
        )
    # Checked per arm, not just for equality: "both off" and "both at the 900-word default"
    # would each pass an equality check.
    if not c.attacker.cross_iteration_memos:
        problems.append(
            f"{name}: attacker.cross_iteration_memos must be true — it is one of the TWO prompts "
            "the eval-data description steers, and with view_limit 0 it is the only thing "
            "carrying anything across the iteration boundary"
        )
    if not c.attacker.round_summaries:
        problems.append(
            f"{name}: attacker.round_summaries must be true — it is the OTHER of the two prompts "
            "the eval-data description steers, and turning it off also changes the round scheduling"
        )
    if c.attacker.cross_iteration_memo_word_budget != EXPECTED_MEMO_WORDS:
        problems.append(
            f"{name}: attacker.cross_iteration_memo_word_budget must be {EXPECTED_MEMO_WORDS}, got "
            f"{c.attacker.cross_iteration_memo_word_budget!r} (unset means the repo default "
            f"{DEFAULT_ITERATION_MEMO_WORD_BUDGET}, which is 6x this experiment's budget)"
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

# The description must actually REACH the two summarizers and NOT the classifier. That is the
# whole mechanism, it lives in llm_judge rather than in either config, and a refactor that broke
# it would leave both arms running and producing plausible numbers that mean nothing.
# (scripts/verify_memo_prompt_knobs.py pins this properly; this is the cheap launch-time echo.)
for name, c in (("arm 1", a), ("arm 2", b)):
    desc = c.eval.data_description
    if not _round_coverage_paragraph(desc) or not _iteration_coverage_paragraph(desc):
        problems.append(
            f"{name}: the eval-data description does not reach a summarizer's coverage "
            "paragraph — llm_judge is not wired as this experiment assumes"
        )
    # The classification prompt is built by a module-level function that is not even PASSED
    # the eval-data description — assert both halves of that: the signature has no parameter
    # for it, and the rendered prompt does not contain it.
    if "eval_data_description" in inspect.signature(_build_judge_request).parameters:
        problems.append(
            "_build_judge_request now takes an eval_data_description — the classification "
            "prompt must never learn about the test set, or the labelling function moves with "
            "the arm"
        )
    _, classification_system = _build_judge_request(
        Conversation(messages=(Message("user", "u"), Message("assistant", "a"))),
        c.judge.system_prompt,
        c.probe.pos_class_label,
        c.probe.neg_class_label,
        c.probe.description or "",
    )
    if desc.strip() and desc.strip().splitlines()[0] in classification_system:
        problems.append(
            f"{name}: the eval-data description LEAKED into the judge's classification prompt — "
            "describing the test set to the labeller moves the labelling function, and the two "
            "arms would no longer share one"
        )

# The dev set must be disjoint from the eval splits, or the fit selects its checkpoint on the
# test set and every reported AUROC is optimistic. Verified on the actual rows, not assumed from
# the directory names, and done here because it is cheap and unrecoverable afterwards.
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
print(">>> arm configs OK: identical apart from eval.data_description and the per-arm output paths")
for name, c in (("arm 1", a), ("arm 2", b)):
    split, _ = ARM_SPLITS[name]
    n_words = len(c.eval.data_description.split())
    print(f">>> {name}: eval.data_description describes {EVAL_DIR}/{split} ({n_words} words), "
          "summarizers only")
print(f">>> attacker (BOTH arms): {a.attacker.model_names[0]}, BLIND at view_limit "
      f"{a.attacker.view_limit}, batch_submissions={a.attacker.batch_submissions}")
print(f">>> round memo ON, cross-iteration memo ON @ {EXPECTED_MEMO_WORDS} words "
      f"(repo default {DEFAULT_ITERATION_MEMO_WORD_BUDGET}), "
      f"{a.attacker.cross_iteration_memo_max_successes} successes shown to the judge")
print(f">>> deep ensemble ON in both arms: {EXPECTED_ENSEMBLE_SIZE} members per train/retrain "
      f"(max {MAX_ENSEMBLE_SIZE}), seeds {list(ENSEMBLE_SEEDS[:EXPECTED_ENSEMBLE_SIZE])}, "
      f"fused={fusion_enabled()} (sequential fit and scoring)")
print(f">>> probe description ({len((a.probe.description or '').split())} words, bare definition) "
      "reaches the attacker system prompt and all three judge prompts — IDENTICAL in both arms")
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

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in this
# list — it is meant to persist and grow across both arms, across re-runs, and across experiments
# on the same probe/splits.)
#
# --resume turns the guard off, because on a resume those directories existing is the POINT: the
# CLI reads probe_iterN.pkl and the redteam_done_* markers out of the probe dir to pick up where
# it stopped, and re-running the red-team phases it already paid OpenRouter for would be the
# expensive mistake. The cross-iteration memo resumes with it — IterationMemoStore re-reads
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
    for p in results_instructions_gemma27b_evaldesc_omission probes/instructions_gemma27b_evaldesc_omission \
             results_instructions_gemma27b_evaldesc_drift    probes/instructions_gemma27b_evaldesc_drift; do
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
OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is unusable"

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (log: $3)"
    local rc=0
    # NOT passing --ensemble-size or --dev-data here on purpose. Both flags OVERRIDE the config
    # (precedence is flag > config), and both are properties of the probe both arms share, so
    # they live in the configs — probe.ensemble_size: 10 and validation.dev_data — where the arms
    # can be diffed against each other. Setting either here would silently mask a config edit.
    # --test-size / --split-field are likewise absent: dev_data makes retrain.py ignore them.
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 5 \
        --base-training-data data/instructions_llama70b_50.jsonl \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_sets/instructions \
        > "$3" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
        # The circuit breaker stopped the run: OpenRouter is out of credits or the key is dead.
        # Do NOT start the next arm — it would fail identically and write a comparison CSV from
        # probes trained on nothing.
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

# ARM 1 — oig_omission described
run_arm configs/gptoss120b_instructions_gemma27b_evaldesc_omission.md \
        probes/instructions_gemma27b_evaldesc_omission \
        logs/run_instructions_gemma27b_evaldesc_omission.log

# ARM 2 — oig_context_drift described
run_arm configs/gptoss120b_instructions_gemma27b_evaldesc_drift.md \
        probes/instructions_gemma27b_evaldesc_drift \
        logs/run_instructions_gemma27b_evaldesc_drift.log

echo ">>> $(date -Is)  both arms finished."
