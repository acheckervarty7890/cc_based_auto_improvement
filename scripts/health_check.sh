#!/bin/bash
# One compact health snapshot of the generalization-test run: is the box actually being
# used the way this workload needs, or is it quietly degrading?
#
# The failure modes worth catching here are all silent ones:
#   * GPU idle while a fit/eval is running  -> activations fell back to host-resident
#   * swap in use                           -> the activation set outgrew RAM; everything crawls
#   * "disk" in an offload line             -> accelerate spilled weights to disk (48-264 s/sample)
#   * disk filling                          -> 75 GB of blobs + probes + caches on one volume
# Each prints a WARN line; otherwise the block is a plain status readout.
cd /workspace/cc_based_auto_improvement || exit 1

echo "--- health $(date -u '+%Y-%m-%d %H:%M:%SZ') ---"

read -r gpu_util gpu_used gpu_total < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | tr -d ',')
echo "gpu: ${gpu_util}% util, ${gpu_used}/${gpu_total} MiB"

read -r mem_total mem_used mem_avail < <(free -m | awk '/^Mem:/{print $2, $3, $7}')
swap_used=$(free -m | awk '/^Swap:/{print $3}')
echo "host: ${mem_used}/${mem_total} MiB used, ${mem_avail} MiB available, swap ${swap_used} MiB"

disk_avail=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
echo "disk: ${disk_avail} GiB free on /workspace"

proc=$(ps -eo pid,etime,args | grep "[g]eneralization_tests.py" | head -1)
if [ -n "$proc" ]; then
  echo "run: $(echo "$proc" | awk '{print "pid "$1", elapsed "$2", phase "$NF}')"
else
  echo "run: no generalization_tests.py process"
fi

for c in hu_ha instructions highstakes; do
  n_probes=$(ls results_generalization/$c/probes/*.pkl 2>/dev/null | wc -l)
  csv=results_generalization/$c/eval_results.csv
  n_eval=0
  [ -f "$csv" ] && n_eval=$(( $(wc -l < "$csv") - 1 ))
  [ "$n_probes" -gt 0 ] && echo "$c: ${n_probes}/16 probes, ${n_eval} eval rows"
done

# --- warnings ---------------------------------------------------------------------
if [ -n "$proc" ] && [ "$gpu_util" -lt 5 ] && [ "$gpu_used" -lt 1000 ]; then
  echo "WARN gpu idle (${gpu_util}%, ${gpu_used} MiB) while a run is active - check for host-resident fallback"
fi
if [ "$swap_used" -gt 1024 ]; then
  echo "WARN swap in use (${swap_used} MiB) - the activation set may have outgrown RAM"
fi
if [ "$disk_avail" -lt 60 ]; then
  echo "WARN only ${disk_avail} GiB free on /workspace"
fi
if grep -lqi "offloaded to the .*disk" logs/gen_*.log 2>/dev/null; then
  echo "WARN a log reports DISK offload - extraction would be ~20-100x slower than CPU offload"
fi
if grep -hE "Traceback|OutOfMemoryError|Killed" logs/gen_*.log 2>/dev/null | tail -1 | grep -q .; then
  echo "WARN a log contains a traceback/OOM/kill: $(grep -hE 'Traceback|OutOfMemoryError|Killed' logs/gen_*.log 2>/dev/null | tail -1 | cut -c1-120)"
fi
