#!/usr/bin/env bash
# Commit and push the instruction-following vintage sweep's progress every N seconds.
#
# The sweep is 80 real ProbeFactory refits and runs for hours on a box that can be
# reclaimed; its only durable state is the append-only progress sidecar
# (results_instructions_gemma27b_vintage/vintage_progress.jsonl) plus the rendered
# SUMMARY.md. This pushes both on a timer so whatever exists at any moment is on the
# remote and explains itself.
#
# What is committed is deliberately narrow: results* is gitignored wholesale (see
# .gitignore), so the sidecar/CSV/SUMMARY are force-added by name. The activation caches
# under results_instructions_gemma27b_shared/ are multi-GB and are never touched — and
# `git add -f` on explicit paths (rather than -A) is also what keeps the untracked
# kaggle/kaggle.json and HF_token.json credentials out of the history.
#
# Usage:
#   nohup bash scripts/checkpoint_vintage.sh > logs/checkpoint_vintage.log 2>&1 &

set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INTERVAL="${CHECKPOINT_INTERVAL_S:-1800}"   # 30 minutes
OUT_DIR="${VINTAGE_OUT_DIR:-results_instructions_gemma27b_vintage}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo ">>> checkpointing $OUT_DIR to origin/$BRANCH every ${INTERVAL}s"

while true; do
    # Re-render SUMMARY.md from the sidecar first, so the pushed summary always matches
    # the pushed rows rather than lagging them by one interval.
    HF_TOKEN="${HF_TOKEN:-}" .venv_claude/bin/python scripts/vintage_summary_md.py \
        >/dev/null 2>&1 || echo "    (summary render failed; committing rows anyway)"

    # One `git add` per path, because a single call with several pathspecs aborts the
    # WHOLE add when any one of them matches nothing — and the CSV/JSON outputs only
    # appear when an arm finishes, so for the first hours of the sweep the globs are
    # unmatched and a combined add would stage nothing at all while still reporting
    # "nothing new". (That is exactly what the first two polls of this script did.)
    for p in "$OUT_DIR"/vintage_progress.jsonl \
             "$OUT_DIR"/SUMMARY.md \
             "$OUT_DIR"/*.csv \
             "$OUT_DIR"/*.json \
             scripts/attribution_lib.py \
             scripts/attribution_refit.py \
             scripts/attribution_vintage.py \
             scripts/attribution_fetch_eval.py \
             scripts/vintage_summary_md.py \
             scripts/checkpoint_vintage.sh \
             scripts/publish_kaggle_redteam_activations.py \
             src/agentic_redteam/token_budget.py; do
        [ -f "$p" ] && git add -f "$p" 2>/dev/null
    done

    n=$(grep -c . "$OUT_DIR/vintage_progress.jsonl" 2>/dev/null || echo 0)
    if git diff --cached --quiet; then
        echo ">>> $(date -Is)  nothing new ($n fits recorded)"
    else
        git commit -q -m "vintage sweep: checkpoint @ $(date -Is) ($n fits)" \
            -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
        if git push -q origin "$BRANCH"; then
            echo ">>> $(date -Is)  pushed ($n fits)"
        else
            echo ">>> $(date -Is)  commit ok, PUSH FAILED — will retry next interval" >&2
        fi
    fi
    sleep "$INTERVAL"
done
