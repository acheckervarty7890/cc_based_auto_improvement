#!/usr/bin/env bash
set -e

# ARMS 7-8 — the `+att` channel put on the remaining two human-harm attackers.
#
# Arms 5 and 6 showed the REWRITTEN eval-data description straight to the attacker as well as to
# the judge's summarizers, and DISAGREED: +0.035 net on llama-70b, -0.013 on nemotron, each
# surviving an eight-draw resampling floor (+2.3 and -2.1 sd). Two attackers cannot settle that,
# so these two put the identical configuration on the gpt-oss and deepseek bases — the two
# human-harm attackers that had a `+eval-desc` arm (experiment25 and experiment26) but no `+att`
# one. With them the human-harm `+att` row covers all four attackers, as instructions and
# high-stakes already do.
#
#   arm  attacker          base data                       memo   +desc   +att
#   ---  ----------------  ------------------------------  -----  ------  ----
#    5   llama-3.3-70b     hu_harm_llama70b_50.jsonl       arm 1  arm 2   done
#    6   nemotron-3-ultra  hu_harm_nemotron_50.jsonl       arm 3  arm 4   done
#    7   gpt-oss-120b      hu_harm_gptoss_50.jsonl         E25    E25     THIS RUN
#    8   deepseek-v4-pro   hu_harm_deepseekv4pro_50.jsonl  E26    E26     THIS RUN
#
# WHAT THE CONFIGS ARE. Each is ARM 5's config with the attacker and the three output paths
# changed and NOTHING else — asserted below against the parsed dataclass, plus an md5 on the two
# system-prompt sections. So the four `+att` arms are one recipe and their differences are
# attributable to the attacker and the base it wrote.
#
# WHAT THIS MEANS FOR THE E25/E26 COMPARISON — THREE KEYS MOVE, NOT ONE:
#   1. attacker.show_eval_data_description  false -> TRUE
#   2. eval.data_description   experiment25/26's earlier free-form text -> the REWRITTEN one
#   3. judge.eval_scope_check  experiment25/26 shipped it TRUE -> FALSE here
# (3) is the one that is easy to miss and the reason this runner prints it. With scope-check ON
# the description also reaches the judge's CLASSIFICATION prompt as a constraint, which moves the
# labelling function — and the judge's labels are what the retrain trains on. Every arm on
# `human_harm_last` pins it OFF; keeping experiment25/26's value would have made these arms'
# numbers incomparable to arms 1-6 on this branch. The clean comparison for arms 7-8 is therefore
# against ARMS 5 and 6, not against their own E25/E26 siblings.
#
# Usage:
#   set -a; . ./.env; set +a
#   nohup bash run_gemma27b_hu_harm_evaldesc_attacker_e25e26.sh > logs/run_huharm_arms7_8.out 2>&1 &
#
# Checkpointing: failsafe_commit.sh with these two stages, in this order —
#   nohup bash failsafe_commit.sh \
#     --config configs/gptoss120b_hu_harm_gemma27b_gptossbase_evaldesc_attacker.md \
#     --probe-out-dir probes/hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc_attacker \
#     --log-file logs/run_hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc_attacker.log \
#     --config configs/deepseekv4pro_hu_harm_gemma27b_dsbase_evaldesc_attacker.md \
#     --probe-out-dir probes/hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc_attacker \
#     --log-file logs/run_hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc_attacker.log \
#     > logs/failsafe_huharm_arms7_8.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

GPTOSS_BASE="data/hu_harm_gptoss_50.jsonl"
DEEPSEEK_BASE="data/hu_harm_deepseekv4pro_50.jsonl"

# The base files were brought onto this branch from experiment25/26. Their md5s are pinned so a
# silently different 50 rows cannot make arm 7's iteration-0 probe incomparable to E25's 0.8781
# (or arm 8's to E26's 0.8954) — the base activation cache is keyed on this file's bytes, so a
# changed file would also quietly mint a new cache key instead of failing.
declare -A BASE_MD5=(
  ["$GPTOSS_BASE"]="211d5659d76cbe967be7f09fe4cf23f4"
  ["$DEEPSEEK_BASE"]="fd0bb8b56385a4f895c01527d02c0fef"
)
for base in "$GPTOSS_BASE" "$DEEPSEEK_BASE"; do
    if [ ! -f "$base" ]; then
        echo "ERROR: base training data not found: $base" >&2
        echo "       git show origin/experiment25_gptoss_base_cloud:$base > $base   (or experiment26 for deepseek)" >&2
        exit 1
    fi
    got="$(md5sum "$base" | cut -d' ' -f1)"
    if [ "$got" != "${BASE_MD5[$base]}" ]; then
        echo "ERROR: $base md5 $got != expected ${BASE_MD5[$base]}" >&2
        echo "       This is NOT the file experiment25/26 trained on; iteration 0 would not reproduce." >&2
        exit 1
    fi
    echo ">>> base training data: $base ($(wc -l < "$base") rows, md5 ok)"
done

# Kaggle credentials for the precomputed eval AND dev activations (configs' `kaggle:` section).
# Checked HERE rather than at first use: an unauthenticated KaggleApi.authenticate() ends in
# exit(1), not an exception.
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

# HuggingFace token. tuberlens' LLMModel.load calls hf_login(), which RAISES without one even for
# a fully cached model, so the run would die at the first red-team model load.
if [ -z "${HF_TOKEN:-}" ] && [ -f hf_token.txt ]; then
    HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt)"
    export HF_TOKEN
fi
: "${HF_TOKEN:?export HF_TOKEN (or put it in hf_token.txt) — tuberlens hf_login() raises without one}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo ">>> HF token: present (${#HF_TOKEN} chars)"

if .venv_claude/bin/python -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
    echo ">>> hf_transfer: enabled (parallel shard download; transfer layer only)"
else
    echo ">>> hf_transfer: NOT installed — using the default single-threaded download."
fi

# --- pin the extraction model's memory budget --------------------------------------------------
# PLACEMENT ONLY. Sized for a 24 GiB card and ~57 GiB of host RAM.
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
echo ">>> max_memory pinned: $AGENTIC_REDTEAM_MAX_MEMORY (placement only — does not change results)"

# --- ensemble fit/score path -------------------------------------------------------------------
# 0 = SEQUENTIAL, as in arms 1-6, experiment25/26 and the 90/80 resampling grids, so these probes
# are fit on the same path as everything they will be compared against.
export PROBE_FUSED_ENSEMBLE="${PROBE_FUSED_ENSEMBLE:-0}"
if [ "$PROBE_FUSED_ENSEMBLE" = "0" ]; then
    echo ">>> ensemble: SEQUENTIAL fit and scoring (PROBE_FUSED_ENSEMBLE=0 — matches arms 1-6, slower)"
else
    echo ">>> ensemble: FUSED (PROBE_FUSED_ENSEMBLE=$PROBE_FUSED_ENSEMBLE — faster; 4th-decimal drift)"
fi
.venv_claude/bin/python - <<'PYEOF' || exit 1
import sys, os
from agentic_redteam.ensemble import fusion_enabled
want = os.environ.get("PROBE_FUSED_ENSEMBLE", "0") != "0"
got = fusion_enabled()
print(f">>> ensemble fusion_enabled() = {got} (wanted {want})")
sys.exit(0 if got == want else 1)
PYEOF

# --- pre-flight: each new arm must be ARM 5's recipe with a different attacker ------------------
# Asserted against the PARSED dataclass, so a stray edit to any knob fails here rather than
# surfacing as an unexplained gap ten hours later. The `# Attacker` / `# Judge` prose is checked
# separately by md5, since it is not part of the dataclass.
.venv_claude/bin/python - <<'PYEOF' || exit 1
import dataclasses, hashlib, pathlib, sys
from agentic_redteam.config import load_config
REF = "configs/llama70b_hu_harm_gemma27b_l70base_evaldesc_attacker.md"
NEW = ["configs/gptoss120b_hu_harm_gemma27b_gptossbase_evaldesc_attacker.md",
       "configs/deepseekv4pro_hu_harm_gemma27b_dsbase_evaldesc_attacker.md"]
ALLOWED = {"attacker.models", "output.comparison_csv", "output.jsonl_path",
           "output.run_id", "source_path"}
def flat(o, p=""):
    out = {}
    if dataclasses.is_dataclass(o):
        for f in dataclasses.fields(o):
            out.update(flat(getattr(o, f.name), f"{p}{f.name}."))
    else:
        out[p.rstrip(".")] = repr(o)
    return out
def prompts_md5(path):
    txt = pathlib.Path(path).read_text()
    return hashlib.md5(txt[txt.index("# Attacker"):].encode()).hexdigest()
ref = load_config(pathlib.Path(REF)); fr = flat(ref); rp = prompts_md5(REF)
rc, descs = 0, [hashlib.md5(ref.eval.data_description.encode()).hexdigest()]
for n in NEW:
    c = load_config(pathlib.Path(n)); fn = flat(c)
    diffs = {k for k in set(fr) | set(fn) if fr.get(k) != fn.get(k)}
    extra = diffs - ALLOWED
    name = pathlib.Path(n).name
    if extra or "attacker.models" not in diffs:
        print(f">>> CONFIG CHECK FAILED for {name}: unexpected {sorted(extra)}"); rc = 1
    else:
        print(f">>> {name}: arm 5's recipe with attacker={[m.name for m in c.attacker.models]} "
              f"(+ output paths), everything else identical")
    if prompts_md5(n) != rp:
        print(f">>> CONFIG CHECK FAILED for {name}: system prompts differ from arm 5's"); rc = 1
    assert c.attacker.show_eval_data_description, n
    assert c.judge.eval_scope_check is False, f"{n}: eval_scope_check must be pinned FALSE"
    descs.append(hashlib.md5(c.eval.data_description.encode()).hexdigest())
if len(set(descs)) != 1:
    print(f">>> CONFIG CHECK FAILED: the +att arms carry DIFFERENT descriptions {descs}"); rc = 1
else:
    print(f">>> all four +att arms share one eval.data_description (md5 {descs[0]})")
    print(">>> judge.eval_scope_check pinned FALSE in both — NOT experiment25/26's value (true)")
sys.exit(rc)
PYEOF

SHARED_CACHE="results_hu_harm_gemma27b_batch_ablation"   # shared, arm-independent activation cache
mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE"
echo ">>>   The four hu_ha eval blobs and the dev blob are already there from arms 1-6. Each of"
echo ">>>   these two arms computes its OWN 50-row base blob once (its base data is new to this"
echo ">>>   branch), keyed on that file's bytes, invalidating nothing."

OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is unusable"

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = base training data, $4 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (base: $3, log: $4)"
    local rc=0
    # The log is APPENDED, not truncated — --resume makes a restart the normal case, and a `>`
    # here is what cost instruction arm 9 its first nine iterations of log.
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

# gpt-oss first: it is the cheaper of the two per batch call (~1,600 reasoning tokens at 575 tok/s
# against deepseek's ~4,000 at 64), so a box that dies half way leaves the finished arm behind.
run_arm configs/gptoss120b_hu_harm_gemma27b_gptossbase_evaldesc_attacker.md \
        probes/hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc_attacker \
        "$GPTOSS_BASE" \
        logs/run_hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc_attacker.log

run_arm configs/deepseekv4pro_hu_harm_gemma27b_dsbase_evaldesc_attacker.md \
        probes/hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc_attacker \
        "$DEEPSEEK_BASE" \
        logs/run_hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc_attacker.log

echo ">>> $(date -Is)  human-harm arms 7-8 finished."
