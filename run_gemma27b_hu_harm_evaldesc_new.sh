#!/usr/bin/env bash
set -e

# ARMS 9-12 — the REWRITTEN human-harm eval description delivered JUDGE-SIDE ONLY, on all four
# attackers. Each is its `+att` sibling (arms 5-8) with ONE key flipped:
# attacker.show_eval_data_description true -> false.
#
# WHY. The +desc -> +att comparison is supposed to isolate the DELIVERY CHANNEL, which only works
# if the TEXT is held fixed — and on this concept it never has been:
#     arm 2    ran its config BEFORE commit d793fe5d rewrote the description
#     arm 4    still carries the earlier free-form text
#     E25/E26  ran the earlier text AND shipped judge.eval_scope_check ON
# So every existing human-harm +desc arm differs from its +att counterpart in two or three keys.
# These four make it one, on every attacker at once:
#
#   arm  attacker          judge-side (this run)  + shown to the attacker   base
#   ---  ----------------  ---------------------  -----------------------   ------------------------------
#    9   llama-3.3-70b     THIS RUN               arm 5                     hu_harm_llama70b_50.jsonl
#   11   gpt-oss-120b      THIS RUN               arm 7                     hu_harm_gptoss_50.jsonl
#   10   nemotron-3-ultra  THIS RUN               arm 6                     hu_harm_nemotron_50.jsonl
#   12   deepseek-v4-pro   THIS RUN               arm 8                     hu_harm_deepseekv4pro_50.jsonl
#
# Nothing already published is overwritten: each arm writes to its own `_evaldesc_new` results and
# probe directory, so arms 2, 4 and E25/E26 keep their own numbers for side-by-side reading.
#
# ITERATION 0 IS A FREE CONSISTENCY CHECK. Each arm shares its base activation blob with its +att
# sibling, so iteration 0 must reproduce that sibling's exactly — 0.8457 (llama70b), 0.8843
# (nemotron), 0.8781 (gpt-oss), 0.8954 (deepseek). A mismatch means the recipe drifted and the
# pair is not one key apart.
#
# Usage:
#   set -a; . ./.env; set +a
#   nohup bash run_gemma27b_hu_harm_evaldesc_new.sh > logs/run_huharm_arms9_12.out 2>&1 &
#
# Checkpointing: failsafe_commit.sh with these four stages, in the order below.

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

declare -A BASE_MD5=(
  ["data/hu_harm_llama70b_50.jsonl"]="88f74ec875b0adee4bd208604d8f9893"
  ["data/hu_harm_gptoss_50.jsonl"]="211d5659d76cbe967be7f09fe4cf23f4"
  ["data/hu_harm_nemotron_50.jsonl"]="f57affba4bed81a34227842977088930"
  ["data/hu_harm_deepseekv4pro_50.jsonl"]="fd0bb8b56385a4f895c01527d02c0fef"
)
for base in "${!BASE_MD5[@]}"; do
    [ -f "$base" ] || { echo "ERROR: base training data not found: $base" >&2; exit 1; }
    exp="${BASE_MD5[$base]}"; got="$(md5sum "$base" | cut -d' ' -f1)"
    if [ "$got" != "$exp" ]; then
        echo "ERROR: $base md5 $got != expected $exp" >&2
        echo "       This is NOT the file arms 5-8 trained on, so iteration 0 would not reproduce" >&2
        echo "       its sibling's and the base activation cache would quietly mint a new key." >&2
        exit 1
    fi
    echo ">>> base training data: $base ($(wc -l < "$base") rows, md5 ok)"
done

if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
    kaggle_json="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
    [ -f "$kaggle_json" ] || kaggle_json="$HOME/.config/kaggle/kaggle.json"
    [ -f "$kaggle_json" ] || { echo "ERROR: no Kaggle credentials (KAGGLE_CONFIG_DIR must name the DIRECTORY)." >&2; exit 1; }
    echo ">>> kaggle credentials: $kaggle_json"
fi
.venv_claude/bin/python -c "import kaggle" 2>/dev/null || { echo "ERROR: pip install kaggle" >&2; exit 1; }

if [ -z "${HF_TOKEN:-}" ] && [ -f hf_token.txt ]; then HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt)"; export HF_TOKEN; fi
: "${HF_TOKEN:?export HF_TOKEN — tuberlens hf_login() raises without one}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo ">>> HF token: present (${#HF_TOKEN} chars)"
.venv_claude/bin/python -c "import hf_transfer" 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" || true

export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
echo ">>> max_memory pinned: $AGENTIC_REDTEAM_MAX_MEMORY (placement only)"

# SEQUENTIAL fit and scoring, as in arms 1-8 and the resampling grids.
export PROBE_FUSED_ENSEMBLE="${PROBE_FUSED_ENSEMBLE:-0}"
echo ">>> ensemble: PROBE_FUSED_ENSEMBLE=$PROBE_FUSED_ENSEMBLE (0 = sequential, matches arms 1-8)"
.venv_claude/bin/python - <<'PYEOF' || exit 1
import sys, os
from agentic_redteam.ensemble import fusion_enabled
want = os.environ.get("PROBE_FUSED_ENSEMBLE", "0") != "0"
got = fusion_enabled()
print(f">>> ensemble fusion_enabled() = {got} (wanted {want})")
sys.exit(0 if got == want else 1)
PYEOF

# --- pre-flight: each arm must be its +att sibling with EXACTLY ONE key flipped ----------------
.venv_claude/bin/python - <<'PYEOF' || exit 1
import dataclasses, hashlib, pathlib, sys
from agentic_redteam.config import load_config
PAIRS = [
 ("configs/llama70b_hu_harm_gemma27b_l70base_evaldesc_attacker.md",
  "configs/llama70b_hu_harm_gemma27b_l70base_evaldesc_new.md"),
 ("configs/nemotron_hu_harm_gemma27b_nmbase_evaldesc_attacker.md",
  "configs/nemotron_hu_harm_gemma27b_nmbase_evaldesc_new.md"),
 ("configs/gptoss120b_hu_harm_gemma27b_gptossbase_evaldesc_attacker.md",
  "configs/gptoss120b_hu_harm_gemma27b_gptossbase_evaldesc_new.md"),
 ("configs/deepseekv4pro_hu_harm_gemma27b_dsbase_evaldesc_attacker.md",
  "configs/deepseekv4pro_hu_harm_gemma27b_dsbase_evaldesc_new.md"),
]
ALLOWED = {"attacker.show_eval_data_description", "output.comparison_csv",
           "output.jsonl_path", "output.run_id", "source_path"}
def flat(o, p=""):
    out = {}
    if dataclasses.is_dataclass(o):
        for f in dataclasses.fields(o):
            out.update(flat(getattr(o, f.name), f"{p}{f.name}."))
    else:
        out[p.rstrip(".")] = repr(o)
    return out
def pm(x):
    t = pathlib.Path(x).read_text()
    return hashlib.md5(t[t.index("# Attacker"):].encode()).hexdigest()
rc, descs = 0, set()
for a, b in PAIRS:
    ca, cb = load_config(pathlib.Path(a)), load_config(pathlib.Path(b))
    fa, fb = flat(ca), flat(cb)
    diffs = {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}
    extra = diffs - ALLOWED
    name = pathlib.Path(b).name
    if extra or "attacker.show_eval_data_description" not in diffs:
        print(f">>> CONFIG CHECK FAILED {name}: unexpected {sorted(extra)}"); rc = 1
    elif pm(a) != pm(b):
        print(f">>> CONFIG CHECK FAILED {name}: system prompts differ from its +att sibling"); rc = 1
    else:
        print(f">>> {name}: {pathlib.Path(a).name} with show_eval_data_description "
              f"{ca.attacker.show_eval_data_description} -> {cb.attacker.show_eval_data_description} "
              f"(+ output paths), everything else identical")
    if cb.attacker.show_eval_data_description is not False:
        print(f">>> CONFIG CHECK FAILED {name}: the knob must be FALSE in this arm"); rc = 1
    if cb.judge.eval_scope_check is not False:
        print(f">>> CONFIG CHECK FAILED {name}: eval_scope_check must stay FALSE"); rc = 1
    descs.add(hashlib.md5(cb.eval.data_description.encode()).hexdigest())
if len(descs) != 1:
    print(f">>> CONFIG CHECK FAILED: arms 9-12 carry DIFFERENT descriptions {descs}"); rc = 1
else:
    print(f">>> arms 9-12 share the rewritten description with arms 5-8 (md5 {descs.pop()})")
sys.exit(rc)
PYEOF

SHARED_CACHE="results_hu_harm_gemma27b_batch_ablation"
mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE — every blob these arms need (4 eval splits, the dev"
echo ">>>   blob, all four 50-row base blobs) is already there from arms 1-8, so iteration 0"
echo ">>>   loads no 27B model at all."

OUTAGE_EXIT_CODE=3

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = base training data, $4 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (base: $3, log: $4)"
    local rc=0
    echo "===== $(date -Is)  run_arm start (append) =====" >> "$4"
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 10 \
        --base-training-data "$3" \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_sets/hu_ha \
        >> "$4" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
        echo ">>> $(date -Is)  ABORTED $1 — OpenRouter unusable (exit $rc)." >&2
        tail -n 5 "$4" >&2; exit "$rc"
    elif [ "$rc" -ne 0 ]; then
        echo ">>> $(date -Is)  FAILED  $1 (exit $rc) — see $4" >&2; exit "$rc"
    fi
    echo ">>> $(date -Is)  DONE  $1"
}

# Cheapest attackers first, so a box that dies part-way leaves whole arms behind rather than
# halves: on arms 5-8 llama70b took 1h32m and gpt-oss 1h42m against nemotron's 3h37m.
run_arm configs/llama70b_hu_harm_gemma27b_l70base_evaldesc_new.md \
        probes/hu_harm_gemma27b_llama70b_l70base_evaldesc_new \
        data/hu_harm_llama70b_50.jsonl \
        logs/run_hu_harm_gemma27b_llama70b_l70base_evaldesc_new.log

run_arm configs/gptoss120b_hu_harm_gemma27b_gptossbase_evaldesc_new.md \
        probes/hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc_new \
        data/hu_harm_gptoss_50.jsonl \
        logs/run_hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc_new.log

run_arm configs/nemotron_hu_harm_gemma27b_nmbase_evaldesc_new.md \
        probes/hu_harm_gemma27b_nemotron_nmbase_evaldesc_new \
        data/hu_harm_nemotron_50.jsonl \
        logs/run_hu_harm_gemma27b_nemotron_nmbase_evaldesc_new.log

run_arm configs/deepseekv4pro_hu_harm_gemma27b_dsbase_evaldesc_new.md \
        probes/hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc_new \
        data/hu_harm_deepseekv4pro_50.jsonl \
        logs/run_hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc_new.log

echo ">>> $(date -Is)  human-harm arms 9-12 finished."
