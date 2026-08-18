#!/usr/bin/env bash
# Memo-prompt smoke test. Two stages:
#
#   A  replay  — feed experiment7's round 5 (1 success in 47) back through the rewritten
#                summarizer. No GPU, no probe, one judge call. Proves the prompt renders
#                and shows what the judge now says about a round that exposed nothing.
#   B  live    — 3 rounds against that same probe at half the original fan-out, which
#                produces TWO memo calls (the final round is never summarized).
#                ARM A: view_reshuffle false (recency view).
#   C  live    — ARM B: identical to stage B but view_reshuffle true. One-variable A/B.
#   D  live    — ARM C: identical to stage B but view_limit 0 (no past attempts shown).
#   E  live    — ARM C2: arm C re-run under the revised prompts (no success target told
#                to the attacker; 200-word round memo). Same config as D in every other
#                respect, but writes to results_memotest_noview_c2/ and its own viewer,
#                so arm C's results are left untouched for the comparison.
#   F  live    — ARM C3: arm C2 with max_turns 5->10 and sessions_per_model 5->2, i.e.
#                fewer, longer attacker sessions (2x10=20 slots/round vs 5x5=25). Own
#                results dir, log and viewer; C2 is left untouched.
#   G  live    — ARM C4: the opposite of C3. max_turns stays 5, sessions_per_model 5->10
#                (concurrency raised to match), maximizing INDEPENDENT STARTS per round.
#                Own results dir, log and viewer.
#   I  live    — ARM C6: C4 with attacker.batch_submissions on and sessions_per_model 4.
#                Each session is ONE call that emits all 5 conversations at once and ends,
#                so the attacker writes them BLIND — no per-submission verdict. Isolates
#                what the in-context feedback loop contributes to mode collapse.
#                (Stage letter h and the name C5 are taken: that is the repetition-
#                clustering arm on branch cluster-repetition-memo, whose results are still
#                in results_memotest_noview_c5/.)
#   J  live    — ARM C7: C6 with max_turns 5->10 and sessions_per_model 4->2. Same 20 slots
#                per round, redistributed into fewer/wider batches — the batch-mode analog
#                of C2 -> C3. Isolates how much of the leftover within-batch similarity is
#                the shared context itself, now that feedback is already gone.
#   K  live    — ARM C8: batch mode at max_turns 3, sessions_per_model 7 (21 slots/round).
#                Third point of the batch-shape sweep — 2, 4 and 7 independent starts at a
#                near-fixed budget (C7, C6, C8).
#
# Usage:  scripts/run_round5_memotest.sh [a|b|c|d|e|f|g|i|j|k|both|all]  (default: both)
#   both = a+b (original) · all = a+b+c+d (the original three arms; NOT e, which is a
#   re-run rather than a new arm) · each stage is fine on its own
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv_claude/bin/python"
STAGE="${1:-both}"

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"

# Round 5's stored attempts. Point this at wherever you have the guidance arm's FP JSONL;
# it also lives on origin/experiment7_cloud under results_hs_llama1b_gptoss120b_guidance/.
REPLAY_JSONL="${REPLAY_JSONL:-/tmp/claude-1000/-home-ankush-Documents-cc-based-auto-improvement/13995774-3f07-4e4f-a3ca-1a682ac14856/scratchpad/gptoss120b_probing_fp.jsonl}"

mkdir -p "$REPO/results_memotest" "$REPO/logs"

if [[ "$STAGE" == "a" || "$STAGE" == "both" || "$STAGE" == "all" ]]; then
  echo "=============================================================================="
  echo " STAGE A — replay round 5 through the new summarizer"
  echo "=============================================================================="
  if [[ ! -f "$REPLAY_JSONL" ]]; then
    echo "REPLAY_JSONL not found: $REPLAY_JSONL" >&2
    echo "Set REPLAY_JSONL=<path to gptoss120b_probing_fp.jsonl> and re-run." >&2
    exit 1
  fi

  # New prompt, with the opposite-direction filter on (43 of 47 rows shown).
  "$PY" "$REPO/scripts/replay_round_memo.py" \
    --jsonl "$REPLAY_JSONL" --round 5 \
    --out-dir "$REPO/results_memotest/replay" \
    2>&1 | tee "$REPO/logs/memotest_stageA.log"

  # Same round with the filter off, for a side-by-side. Comment out to save a judge call.
  "$PY" "$REPO/scripts/replay_round_memo.py" \
    --jsonl "$REPLAY_JSONL" --round 5 --no-hide-opposite \
    --out-dir "$REPO/results_memotest/replay" \
    2>&1 | tee -a "$REPO/logs/memotest_stageA.log"
fi

run_arm () {   # $1 = arm label, $2 = config, $3 = results dir
  echo "=============================================================================="
  echo " ARM $1 — 3 live rounds (⇒ 2 memo calls) — $2"
  echo "=============================================================================="
  # run_redteam_main never resumes (resume=False), so every invocation runs all 3 rounds.
  # But the JSONL is append-only and dedups by canonical text, so a second run starts with
  # the first run's conversations already blacklisted — wipe "$3" between runs
  # if you want a clean comparison.
  "$PY" "$REPO/scripts/run_redteam.py" "$2" 2>&1 | tee "$REPO/logs/memotest_arm$1.log"

  echo
  echo "--- memos written (one per summarized round) ---"
  RESULTS_DIR="$3" "$PY" - <<'EOF'
import json, os, pathlib
p = pathlib.Path(os.environ["RESULTS_DIR"]) / "memotest_probing.summaries.jsonl"
if not p.exists():
    print("no summaries sidecar — did round_summaries stay on?"); raise SystemExit
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    d = json.loads(line)
    text = d["text"]
    complete = text.rstrip().endswith((".", "!", "?", '"', "\u201d", ")"))
    print(f"\nround {d['round']} | {d['n_successes']}/{d['n_attempts']} succeeded "
          f"| {len(text.split())} words | {'complete' if complete else 'TRUNCATED'}")
    print("-" * 78)
    print(text)
EOF
}

if [[ "$STAGE" == "b" || "$STAGE" == "both" || "$STAGE" == "all" ]]; then
  run_arm A "$REPO/configs/round5_memotest.md" "$REPO/results_memotest"
fi

if [[ "$STAGE" == "c" || "$STAGE" == "all" ]]; then
  run_arm B "$REPO/configs/round5_memotest_reshuffle.md" "$REPO/results_memotest_reshuffle"
fi

if [[ "$STAGE" == "d" || "$STAGE" == "all" ]]; then
  run_arm C "$REPO/configs/round5_memotest_noview.md" "$REPO/results_memotest_noview"

  # Arm C runs with capture_prompts, so the verbatim per-turn prompts are on disk.
  echo
  echo "--- building prompt-trace viewer ---"
  "$PY" "$REPO/scripts/build_prompt_trace_viewer.py" \
    --prompts "$REPO/results_memotest_noview/memotest_probing.prompts.jsonl" \
    --out "$REPO/viewers/memotest_prompt_trace_viewer.html"
fi

if [[ "$STAGE" == "e" ]]; then
  # Refuse to clobber a finished C2. The JSONL is append-only and dedups by canonical
  # text, so re-running into a populated dir would start with the previous run's
  # conversations already blacklisted and its success counter warm — not a clean re-run.
  if [[ -e "$REPO/results_memotest_noview_c2/memotest_probing.jsonl" ]]; then
    echo "results_memotest_noview_c2/ already has a run in it." >&2
    echo "Move or delete it first if you want a clean C2:" >&2
    echo "  mv $REPO/results_memotest_noview_c2 $REPO/results_memotest_noview_c2.\$(date +%Y%m%d-%H%M)" >&2
    exit 1
  fi

  run_arm C2 "$REPO/configs/round5_memotest_noview_c2.md" "$REPO/results_memotest_noview_c2"

  echo
  echo "--- building prompt-trace viewer (separate file from arm C's) ---"
  "$PY" "$REPO/scripts/build_prompt_trace_viewer.py" \
    --prompts "$REPO/results_memotest_noview_c2/memotest_probing.prompts.jsonl" \
    --out "$REPO/viewers/memotest_prompt_trace_viewer_c2.html"

  echo
  echo "=============================================================================="
  echo " ARM C vs C2 (same config, revised prompts)"
  echo "=============================================================================="
  "$PY" "$REPO/scripts/compare_memotest_arms.py" || true
fi

if [[ "$STAGE" == "f" ]]; then
  if [[ -e "$REPO/results_memotest_noview_c3/memotest_probing.jsonl" ]]; then
    echo "results_memotest_noview_c3/ already has a run in it." >&2
    echo "Move or delete it first if you want a clean C3:" >&2
    echo "  mv $REPO/results_memotest_noview_c3 $REPO/results_memotest_noview_c3.\$(date +%Y%m%d-%H%M)" >&2
    exit 1
  fi

  run_arm C3 "$REPO/configs/round5_memotest_noview_c3.md" "$REPO/results_memotest_noview_c3"

  echo
  echo "--- building prompt-trace viewer (separate file from C/C2's) ---"
  "$PY" "$REPO/scripts/build_prompt_trace_viewer.py" \
    --prompts "$REPO/results_memotest_noview_c3/memotest_probing.prompts.jsonl" \
    --out "$REPO/viewers/memotest_prompt_trace_viewer_c3.html"

  echo
  echo "=============================================================================="
  echo " ARM C2 vs C3 (same prompts; 5x5 sessions vs 2x10)"
  echo "=============================================================================="
  "$PY" "$REPO/scripts/compare_memotest_arms.py" || true
fi

if [[ "$STAGE" == "g" ]]; then
  if [[ -e "$REPO/results_memotest_noview_c4/memotest_probing.jsonl" ]]; then
    echo "results_memotest_noview_c4/ already has a run in it." >&2
    echo "Move or delete it first if you want a clean C4:" >&2
    echo "  mv $REPO/results_memotest_noview_c4 $REPO/results_memotest_noview_c4.\$(date +%Y%m%d-%H%M)" >&2
    exit 1
  fi

  run_arm C4 "$REPO/configs/round5_memotest_noview_c4.md" "$REPO/results_memotest_noview_c4"

  echo
  echo "--- building prompt-trace viewer (separate file from C/C2/C3's) ---"
  "$PY" "$REPO/scripts/build_prompt_trace_viewer.py" \
    --prompts "$REPO/results_memotest_noview_c4/memotest_probing.prompts.jsonl" \
    --out "$REPO/viewers/memotest_prompt_trace_viewer_c4.html"

  echo
  echo "=============================================================================="
  echo " FAN-OUT SWEEP: C3 (2x10) vs C2 (5x5) vs C4 (10x5)"
  echo "=============================================================================="
  "$PY" "$REPO/scripts/compare_memotest_arms.py" || true
fi

if [[ "$STAGE" == "i" ]]; then
  if [[ -e "$REPO/results_memotest_noview_c6/memotest_probing.jsonl" ]]; then
    echo "results_memotest_noview_c6/ already has a run in it." >&2
    echo "Move or delete it first if you want a clean C6:" >&2
    echo "  mv $REPO/results_memotest_noview_c6 $REPO/results_memotest_noview_c6.\$(date +%Y%m%d-%H%M)" >&2
    exit 1
  fi

  run_arm C6 "$REPO/configs/round5_memotest_noview_c6.md" "$REPO/results_memotest_noview_c6"

  echo
  echo "--- building prompt-trace viewer (separate file from C/C2/C3/C4/C5's) ---"
  # Batch rows carry the whole batch on one row; the viewer explodes them back into one
  # entry per submission, badged "batch k/N".
  "$PY" "$REPO/scripts/build_prompt_trace_viewer.py" \
    --prompts "$REPO/results_memotest_noview_c6/memotest_probing.prompts.jsonl" \
    --summaries "$REPO/results_memotest_noview_c6/memotest_probing.summaries.jsonl" \
    --out "$REPO/viewers/memotest_prompt_trace_viewer_c6.html"

  echo
  echo "=============================================================================="
  echo " FEEDBACK ABLATION: C4 (10 sessions x 5 turns, per-turn verdicts)"
  echo "              vs C6 ( 4 sessions x 5-in-one, no verdicts at all)"
  echo "=============================================================================="
  "$PY" "$REPO/scripts/compare_memotest_arms.py" || true
fi

if [[ "$STAGE" == "j" ]]; then
  if [[ -e "$REPO/results_memotest_noview_c7/memotest_probing.jsonl" ]]; then
    echo "results_memotest_noview_c7/ already has a run in it." >&2
    echo "Move or delete it first if you want a clean C7:" >&2
    echo "  mv $REPO/results_memotest_noview_c7 $REPO/results_memotest_noview_c7.\$(date +%Y%m%d-%H%M)" >&2
    exit 1
  fi

  run_arm C7 "$REPO/configs/round5_memotest_noview_c7.md" "$REPO/results_memotest_noview_c7"

  echo
  echo "--- building prompt-trace viewer (own file; C6's is untouched) ---"
  "$PY" "$REPO/scripts/build_prompt_trace_viewer.py" \
    --prompts "$REPO/results_memotest_noview_c7/memotest_probing.prompts.jsonl" \
    --summaries "$REPO/results_memotest_noview_c7/memotest_probing.summaries.jsonl" \
    --out "$REPO/viewers/memotest_prompt_trace_viewer_c7.html"

  echo
  echo "=============================================================================="
  echo " BATCH SHAPE: C6 (4 sessions x 5-in-one) vs C7 (2 sessions x 10-in-one)"
  echo "   same 20 slots per round; compare against C2 vs C3, the per-turn analog"
  echo "=============================================================================="
  "$PY" "$REPO/scripts/compare_memotest_arms.py" || true
fi

if [[ "$STAGE" == "k" ]]; then
  if [[ -e "$REPO/results_memotest_noview_c8/memotest_probing.jsonl" ]]; then
    echo "results_memotest_noview_c8/ already has a run in it." >&2
    echo "Move or delete it first if you want a clean C8:" >&2
    echo "  mv $REPO/results_memotest_noview_c8 $REPO/results_memotest_noview_c8.\$(date +%Y%m%d-%H%M)" >&2
    exit 1
  fi

  run_arm C8 "$REPO/configs/round5_memotest_noview_c8.md" "$REPO/results_memotest_noview_c8"

  echo
  echo "--- building prompt-trace viewer (own file; C6's and C7's are untouched) ---"
  "$PY" "$REPO/scripts/build_prompt_trace_viewer.py" \
    --prompts "$REPO/results_memotest_noview_c8/memotest_probing.prompts.jsonl" \
    --summaries "$REPO/results_memotest_noview_c8/memotest_probing.summaries.jsonl" \
    --out "$REPO/viewers/memotest_prompt_trace_viewer_c8.html"

  echo
  echo "=============================================================================="
  echo " BATCH-SHAPE SWEEP at ~20 slots/round: independent starts 2 -> 4 -> 7"
  echo "   C7 (2 x 10)  vs  C6 (4 x 5)  vs  C8 (7 x 3)"
  echo "=============================================================================="
  "$PY" "$REPO/scripts/compare_memotest_arms.py" || true
fi

if [[ "$STAGE" == "all" || "$STAGE" == "c" || "$STAGE" == "d" ]]; then
  echo
  echo "=============================================================================="
  echo " A/B COMPARISON"
  echo "=============================================================================="
  "$PY" "$REPO/scripts/compare_memotest_arms.py" || true
fi
