#!/usr/bin/env bash
# Keep the ~135 GB of downloads alive: 16 published activation blobs from Kaggle plus the
# gemma-3-27b checkpoint. Run it in the background before run_all.sh.
#
# A plain `nohup hf download` is not enough on a flaky box. Observed on this one: the HF
# downloader **stopped moving bytes without exiting** — four `.incomplete` shards, the
# process alive, the network otherwise idle, and no retry. A watchdog that only restarts a
# *dead* child would have waited forever, so a stall (no cache growth for 5 minutes) is
# treated as a failure here. Resume is clean: the restarted download picks each shard up
# where it stopped.
#
# The Kaggle side is split into four streams, one per (concept, kind), because they write
# to different directories and so cannot collide on a staging dir; each retries on failure,
# and the fetch script skips whatever is already on disk.
#
# Prerequisites: KAGGLE_CONFIG_DIR pointing at the DIRECTORY holding kaggle.json, HF_TOKEN
# in the environment.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=./.venv_claude/bin/python
LOGS=ceiling_analysis/logs
HFCACHE=$HOME/.cache/huggingface
SNAPDIR=$HFCACHE/hub/models--google--gemma-3-27b-it/snapshots
mkdir -p "$LOGS"

: "${HF_TOKEN:?export HF_TOKEN}"
: "${KAGGLE_CONFIG_DIR:?export KAGGLE_CONFIG_DIR (the DIRECTORY holding kaggle.json)}"

log() { echo "$(date -Is) $*" >> "$LOGS/fetch_supervisor.log"; }

# ------------------------------------------------------------------ kaggle: retry per set
for spec in "hu_ha eval" "hu_ha dev" "highstakes eval" "highstakes dev"; do
  set -- $spec
  (
    c="$1"; k="$2"
    for attempt in $(seq 1 200); do
      if $PY ceiling_analysis/scripts/fetch_kaggle_activations.py --concepts "$c" --kinds "$k" \
            >> "$LOGS/fetch_kaggle_${c}_${k}.log" 2>&1; then
        log "kaggle $c/$k complete (attempt $attempt)"; exit 0
      fi
      log "kaggle $c/$k failed (attempt $attempt); retrying in 30s"
      sleep 30
    done
    log "kaggle $c/$k GAVE UP"
  ) &
done

# ------------------------------------------------ gemma: restart on death OR on a stall
(
  complete() {
    local d; d=$(ls -d "$SNAPDIR"/*/ 2>/dev/null | head -1)
    [ -n "$d" ] || return 1
    for i in $(seq -w 1 12); do
      [ -s "$d/model-000$i-of-00012.safetensors" ] || return 1
    done
    [ -s "$d/tokenizer.json" ]
  }
  while true; do
    if complete; then log "[gemma] all 12 shards present"; exit 0; fi
    if ! pgrep -f "hf download google/gemma" > /dev/null; then
      log "[gemma] downloader not running; starting"
      setsid nohup env HF_TOKEN="$HF_TOKEN" ./.venv_claude/bin/hf download google/gemma-3-27b-it \
          >> "$LOGS/fetch_gemma.log" 2>&1 < /dev/null &
      sleep 90
      continue
    fi
    before=$(du -sb "$HFCACHE" | cut -f1)
    sleep 300
    after=$(du -sb "$HFCACHE" | cut -f1)
    if [ "$after" -le "$before" ]; then
      log "[gemma] stalled at $after bytes for 5 min; restarting downloader"
      pkill -f "hf download google/gemma"
      sleep 5
    else
      log "[gemma] progress: $(( (after - before) / 300000 )) kB/s"
    fi
  done
) &

wait
log "all downloads complete"
