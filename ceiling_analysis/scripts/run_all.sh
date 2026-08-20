#!/usr/bin/env bash
# The whole analysis, end to end, on a fresh box.
#
# Every wait here is on **state** — the files the next step needs — never on "is a process
# with this name alive?". That distinction is not cosmetic: `pgrep -f <name>` also matches
# any shell whose command line merely MENTIONS the name, so a health-check loop grepping
# for the extractor kept its wait alive forever, and a download watchdog restarting its
# child made a process-based wait fire early and run the entire pipeline in 47 seconds
# against data that did not exist yet. A failing step aborts the chain rather than handing
# its absence to the next one.
#
# Prerequisites: KAGGLE_CONFIG_DIR pointing at the DIRECTORY holding kaggle.json, and
# HF_TOKEN in the environment (tuberlens' hf_login() raises without it).
#
#   ceiling_analysis/scripts/fetch_supervisor.sh &     # downloads, with retries
#   ceiling_analysis/scripts/run_all.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=./.venv_claude/bin/python
LOGS=ceiling_analysis/logs
SNAPDIR=$HOME/.cache/huggingface/hub/models--google--gemma-3-27b-it/snapshots
mkdir -p "$LOGS" ceiling_analysis/results

# No apostrophe in this message: the word of a ${var:?word} expansion is still quote-
# processed, so a bare ' in it opens a quote that never closes and the whole file fails
# to parse -- with the error reported at the next heredoc, nowhere near the real line.
: "${HF_TOKEN:?export HF_TOKEN, needed by the hf_login call in tuberlens}"

mark() { echo ">>> $(date -Is)  $1  $2" | tee -a "$LOGS/run_all.log"; }

run() {  # run <name> <cmd...> -- a failure stops the chain
  local name="$1"; shift
  mark START "$name"
  if "$@" > "$LOGS/$name.log" 2>&1; then
    mark DONE "$name"
  else
    mark "FAILED(rc=$?)" "$name"; mark ABORT chain; exit 1
  fi
}

wait_gemma() {  # all 12 shards on disk, not "the downloader exited"
  while true; do
    local d ok=1
    d=$(ls -d "$SNAPDIR"/*/ 2>/dev/null | head -1)
    for i in $(seq -w 1 12); do
      [ -s "$d/model-000$i-of-00012.safetensors" ] || ok=0
    done
    [ "$ok" = 1 ] && { mark DONE gemma_download; return 0; }
    sleep 120
  done
}

wait_blobs() {  # every published eval/dev blob for one concept is on disk
  local concept="$1"
  while true; do
    if $PY - "$concept" <<'EOF'
import sys
sys.path.insert(0, "ceiling_analysis/scripts")
import fetch_kaggle_activations as F
missing = [str(p) for _, _, _, p in F.targets(sys.argv[1])
           if not (p.is_file() and p.stat().st_size > 0)]
print(f"{sys.argv[1]}: {len(missing)} blobs missing", flush=True)
sys.exit(1 if missing else 0)
EOF
    then
      mark DONE "kaggle_blobs_$concept"; return 0
    fi
    sleep 120
  done
}

# --------------------------------------------------------------- 1. extraction (the GPU)
# Starts against whatever shards have landed; see extract_redteam_partial.py for why that
# is exact. Falls back to the ordinary extractor, which is a no-op once the cache is full.
run extract_partial  $PY ceiling_analysis/scripts/extract_redteam_partial.py
run extract_redteam  $PY ceiling_analysis/scripts/extract_redteam_activations.py \
    --concepts hu_ha highstakes

# --------------------------------------------------------------- 2. verification
# Needs the COMPLETE model: verify_batch_padding re-extracts cached rows with a fresh full
# load, which is also the check that the partial-shard load above was byte-identical.
mark WAIT gemma_download; wait_gemma
mark WAIT kaggle_blobs_hu_ha; wait_blobs hu_ha

run verify_batch_padding    $PY ceiling_analysis/scripts/verify_batch_padding.py
run verify_extraction_noise $PY ceiling_analysis/scripts/verify_extraction_noise.py
run verify_fast_fit         $PY ceiling_analysis/scripts/verify_fast_fit.py

# --------------------------------------------------------------- 3. hu_ha (small; first)
run ceiling_hu_ha $PY ceiling_analysis/scripts/run_ceiling.py --concepts hu_ha \
    --train-sizes 173 346 693 --add-dev-pool
run sweep_hu_ha   $PY ceiling_analysis/scripts/run_sweep.py --concepts hu_ha

# --------------------------------------------------------------- 4. highstakes
mark WAIT kaggle_blobs_highstakes; wait_blobs highstakes

run ceiling_highstakes $PY ceiling_analysis/scripts/run_ceiling.py --concepts highstakes \
    --train-sizes 881 1763 3526 --add-dev-pool
run sweep_highstakes   $PY ceiling_analysis/scripts/run_sweep.py --concepts highstakes

# --------------------------------------------------------------- 5. write-up
run make_report    $PY ceiling_analysis/scripts/make_report.py
run build_artifact $PY ceiling_analysis/scripts/build_artifact.py
mark ALLDONE chain
