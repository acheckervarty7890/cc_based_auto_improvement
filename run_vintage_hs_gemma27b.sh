#!/usr/bin/env bash
set -e

# Red-team VINTAGE sweep for the experiment9 HIGH-STAKES gemma-3-27b arms.
#
# The high-stakes counterpart of experiment11_cloud's run_attribution_hu_harm_gemma27b.sh
# vintage stage. It answers: how much of probe_iter3's eval AUROC was already bought by
# the red-team data that existed at iteration 1?
#
# Each fit is a real ProbeFactory.build on base training data + one vintage of
# iteration-3 red-team pairs, scored on the four eval_datasets/ splits. Only set
# MEMBERSHIP varies across vintages — content, activations and hyperparameters are all
# iteration 3's — which is what makes the vintages comparable to each other in a way the
# committed probe_iter1/2/3 are not (those came from separate retrains with their own
# filter draws and contrastive generations).
#
# NO gemma-3-27b forward passes happen here. Everything runs off cached activations:
#   - eval blobs      -> scripts/attribution_fetch_eval.py                 (~20 GB, Kaggle)
#   - base + red-team -> publish_kaggle_hs_redteam_activations.py restore  (~3.4 GB, Kaggle)
# Both need KAGGLE_CONFIG_DIR (the DIRECTORY holding kaggle.json) or KAGGLE_API_TOKEN.
#
# OVER-LENGTH PAIRS ARE DROPPED (--drop-overlong pair). get_activations truncates at
# 1024 tokens; every row that hit that cap in either arm is an LLM-written contrastive
# counterpart, never an attacker-written source, and the tail it lost is the part
# carrying the opposite-class label. The affected pair is removed whole so the vintages
# stay exactly 50/50.
#
# Usage:
#   export KAGGLE_CONFIG_DIR=$PWD/kaggle
#   nohup bash run_vintage_hs_gemma27b.sh > logs/vintage_hs.log 2>&1 &
#
# Resumable: every finished (arm, vintage, seed) is fsync'd to
# results_hs_gemma27b_batch_ablation/vintage/vintage_progress.jsonl and skipped on a
# re-run, so a box that dies costs one fit rather than the sweep.

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

PY=.venv_claude/bin/python
CACHE=results_hs_gemma27b_batch_ablation
SEEDS=${SEEDS:-10}

# --- 1. activations ------------------------------------------------------------------
# Both stages are no-ops once the blobs are present, so this is safe to re-run.
if ! ls "$CACHE"/eval_activations/*-acts_full.pt >/dev/null 2>&1; then
    echo ">>> $(date -Is)  fetching eval activations (~20 GB) ..."
    $PY scripts/attribution_fetch_eval.py
fi
if ! ls "$CACHE"/base_activations/base_acts_*_train.pt >/dev/null 2>&1; then
    echo ">>> $(date -Is)  restoring base + iteration-3 red-team activations ..."
    $PY scripts/publish_kaggle_hs_redteam_activations.py restore \
        --iterations 3 --cache-dir "$CACHE/base_activations"
fi

# --- 2. what the over-length filter removes ------------------------------------------
echo ">>> $(date -Is)  token-length census"
$PY scripts/vintage_length_report.py --out "$CACHE/vintage/length_census.json"

# --- 3. the sweep --------------------------------------------------------------------
echo ">>> $(date -Is)  vintage sweep, $SEEDS seed(s) x 4 vintages x 2 arms"
$PY scripts/attribution_vintage.py --seeds "$SEEDS" --drop-overlong pair

# --- 4. the write-up -----------------------------------------------------------------
$PY scripts/vintage_summary_md.py
echo ">>> $(date -Is)  done — see $CACHE/vintage/SUMMARY.md"
