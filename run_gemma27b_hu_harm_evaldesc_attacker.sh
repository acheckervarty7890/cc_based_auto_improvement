#!/usr/bin/env bash
set -e

# ARMS 5-6 — the two human-harm eval-description arms re-run with the description delivered to
# the ATTACKER as well as to the judge's two summarizers, and carrying the REWRITTEN description
# (commit d793fe5d), which recasts the four kinds in the shape the instruction arms use.
#
#   arm  attacker   memo only  + desc to the judge   + the attacker is shown it (this run)
#   ---  ---------  ---------  -------------------   ------------------------------------
#    5   llama70b   arm 1      arm 2                 configs/llama70b_hu_harm_gemma27b_l70base_evaldesc_attacker.md
#    6   nemotron   arm 3      arm 4                 configs/nemotron_hu_harm_gemma27b_nmbase_evaldesc_attacker.md
#
# WHAT MOVES vs THE SIBLING. Arm 5 differs from arm 2 in exactly one parsed key
# (attacker.show_eval_data_description) plus the three output paths — arm 2's CONFIG already
# carries the rewritten description, though its completed RUN predates the rewrite. Arm 6 differs
# from arm 4 in TWO: that key AND eval.data_description, because arm 4 still carries the earlier
# free-form text. Both new arms carry ONE identical description (asserted below), so arm 5 vs
# arm 6 is a clean attacker contrast.
#
# judge.eval_scope_check stays FALSE in both, so the description never reaches the judge's
# classification prompt, the labelling function does not move, and the eval numbers stay
# comparable to arms 1-4.
#
# Usage:
#   set -a; . ./.env; set +a
#   nohup bash run_gemma27b_hu_harm_evaldesc_attacker.sh > logs/run_huharm_arms5_6.out 2>&1 &
#
# Checkpointing: failsafe_commit.sh with these two stages, in this order —
#   nohup bash failsafe_commit.sh \
#     --config configs/llama70b_hu_harm_gemma27b_l70base_evaldesc_attacker.md \
#     --probe-out-dir probes/hu_harm_gemma27b_llama70b_l70base_evaldesc_attacker \
#     --log-file logs/run_hu_harm_gemma27b_llama70b_l70base_evaldesc_attacker.log \
#     --config configs/nemotron_hu_harm_gemma27b_nmbase_evaldesc_attacker.md \
#     --probe-out-dir probes/hu_harm_gemma27b_nemotron_nmbase_evaldesc_attacker \
#     --log-file logs/run_hu_harm_gemma27b_nemotron_nmbase_evaldesc_attacker.log \
#     > logs/failsafe_huharm_arms5_6.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

LLAMA70B_BASE="data/hu_harm_llama70b_50.jsonl"
NEMOTRON_BASE="data/hu_harm_nemotron_50.jsonl"
for base in "$LLAMA70B_BASE" "$NEMOTRON_BASE"; do
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

# HuggingFace token. tuberlens' LLMModel.load calls hf_login(), which RAISES without one even
# for a fully cached model, so the run would die at the first red-team model load.
if [ -z "${HF_TOKEN:-}" ] && [ -f hf_token.txt ]; then
    HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt)"
    export HF_TOKEN
fi
: "${HF_TOKEN:?export HF_TOKEN (or put it in hf_token.txt) — tuberlens hf_login() raises without one}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo ">>> HF token: present (${#HF_TOKEN} chars)"

# --- weight download path ---------------------------------------------------------------------
# TRANSFER LAYER ONLY. GUARDED on the import: huggingface_hub RAISES when
# HF_HUB_ENABLE_HF_TRANSFER=1 and the package is missing.
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
# 0 = SEQUENTIAL, as in arms 1-4 and experiment25/26, so these probes are fit on the same path
# the arms they will be compared against were.
export PROBE_FUSED_ENSEMBLE="${PROBE_FUSED_ENSEMBLE:-0}"
if [ "$PROBE_FUSED_ENSEMBLE" = "0" ]; then
    echo ">>> ensemble: SEQUENTIAL fit and scoring (PROBE_FUSED_ENSEMBLE=0 — matches arms 1-4, slower)"
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

# --- pre-flight: what actually differs between each new arm and its sibling --------------------
# Printed rather than merely asserted, because the two arms are NOT symmetric: arm 5 is a
# one-key change from arm 2, arm 6 is two keys from arm 4 (its sibling still carries the older
# free-form description). What IS asserted: the knob is on in both, the two NEW arms carry an
# IDENTICAL description, and nothing else drifts.
.venv_claude/bin/python - <<'PYEOF' || exit 1
import dataclasses, hashlib, pathlib, sys
from agentic_redteam.config import load_config
PAIRS = [
    ("configs/llama70b_hu_harm_gemma27b_l70base_evaldesc.md",
     "configs/llama70b_hu_harm_gemma27b_l70base_evaldesc_attacker.md"),
    ("configs/nemotron_hu_harm_gemma27b_nmbase_evaldesc.md",
     "configs/nemotron_hu_harm_gemma27b_nmbase_evaldesc_attacker.md"),
]
ALLOWED = {"attacker.show_eval_data_description", "eval.data_description",
           "output.comparison_csv", "output.jsonl_path", "output.run_id", "source_path"}
def flat(o, p=""):
    out = {}
    if dataclasses.is_dataclass(o):
        for f in dataclasses.fields(o):
            out.update(flat(getattr(o, f.name), f"{p}{f.name}."))
    else:
        out[p.rstrip(".")] = repr(o)
    return out
rc, descs = 0, []
for a, b in PAIRS:
    ca, cb = load_config(pathlib.Path(a)), load_config(pathlib.Path(b))
    fa, fb = flat(ca), flat(cb)
    diffs = {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}
    extra = diffs - ALLOWED
    if extra or "attacker.show_eval_data_description" not in diffs:
        print(f">>> CONFIG CHECK FAILED for {b}: unexpected {sorted(extra)}"); rc = 1
    else:
        moved = sorted(d for d in diffs if not d.startswith("output.") and d != "source_path")
        print(f">>> {pathlib.Path(b).name}: differs from its sibling in {moved} (+ output paths)")
    assert cb.attacker.show_eval_data_description, b
    descs.append(hashlib.md5(cb.eval.data_description.encode()).hexdigest())
if len(set(descs)) != 1:
    print(f">>> CONFIG CHECK FAILED: the two new arms carry DIFFERENT descriptions {descs}"); rc = 1
else:
    print(f">>> both new arms share one eval.data_description (md5 {descs[0]})")
sys.exit(rc)
PYEOF

SHARED_CACHE="results_hu_harm_gemma27b_batch_ablation"   # shared, arm-independent activation cache
mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE"
echo ">>>   On a clean box the four hu_ha eval blobs come FROM KAGGLE on the first eval, and the"
echo ">>>   dev blob + the two 50-row base blobs are computed once by each arm's iteration 0."

# No clobber guard on the per-arm output/probe dirs ON PURPOSE: --resume is on by default and is
# how a wiped container picks a run back up. To start genuinely fresh, move the dirs aside first.

# --- run one arm ------------------------------------------------------------------------------
OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is unusable"

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = base training data, $4 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (base: $3, log: $4)"
    local rc=0
    # --iterations 10, as in arms 1-4: the contrast is about what crosses an iteration boundary,
    # so the number of boundaries must not vary.
    #
    # NOT passing --ensemble-size or --dev-data: both OVERRIDE the configs (flag > config) and
    # both are properties of the probe every arm shares, so they live in the configs.
    #
    # The log is APPENDED, not truncated — --resume makes a restart the normal case, and a `>`
    # here is what cost the instruction arm 9 its first nine iterations of log.
    echo "===== $(date -Is)  run_arm start (append) =====" >> "$4"
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 10 \
        --base-training-data "$3" \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_sets/hu_ha \
        >> "$4" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
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

# llama70b first, matching arms 1-4's ordering: a box that dies half way then leaves one
# complete arm rather than two half-finished ones.
run_arm configs/llama70b_hu_harm_gemma27b_l70base_evaldesc_attacker.md \
        probes/hu_harm_gemma27b_llama70b_l70base_evaldesc_attacker \
        "$LLAMA70B_BASE" \
        logs/run_hu_harm_gemma27b_llama70b_l70base_evaldesc_attacker.log

run_arm configs/nemotron_hu_harm_gemma27b_nmbase_evaldesc_attacker.md \
        probes/hu_harm_gemma27b_nemotron_nmbase_evaldesc_attacker \
        "$NEMOTRON_BASE" \
        logs/run_hu_harm_gemma27b_nemotron_nmbase_evaldesc_attacker.log

echo ">>> $(date -Is)  human-harm arms 5-6 finished."
