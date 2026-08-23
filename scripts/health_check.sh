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
# Neither GPU idleness nor a quiet log means trouble. The eval phase reads 46 GB of
# cached blobs for the high-stakes splits, so the card sits at 0% for minutes at a time
# and a single torch.load of the 33 GB anthropic blob emits no output at all while it
# runs. The signal that holds across every phase is whether the process is still burning
# CPU. Warn only when nothing at all moved: no CPU time, no log growth, no GPU.
if [ -n "$proc" ]; then
  pid=$(echo "$proc" | awk '{print $1}')
  live_log=$(ls -t logs/gen_*.log 2>/dev/null | head -1)
  cpu0=$(awk '{print $14+$15}' /proc/"$pid"/stat 2>/dev/null || echo 0)
  size0=$(stat -c %s "$live_log" 2>/dev/null || echo 0)
  peak=0
  for _ in 1 2 3 4 5 6; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    [ "$u" -gt "$peak" ] && peak=$u
    sleep 2
  done
  cpu1=$(awk '{print $14+$15}' /proc/"$pid"/stat 2>/dev/null || echo 0)
  size1=$(stat -c %s "$live_log" 2>/dev/null || echo 0)
  if [ "$peak" -lt 5 ] && [ "$size1" -le "$size0" ] && [ "$cpu1" -le "$cpu0" ]; then
    echo "WARN stalled: pid $pid burned no CPU, gpu peak ${peak}%, and $live_log did not grow over 12s"
  fi
fi

# Swap alone is not pressure. Loading a 33 GB activation blob pushes the kernel to
# reclaim cold pages, so a GB or so of swap appears with tens of GB still free. What
# matters is available RAM running out, or swap large enough to mean real thrashing.
if [ "$mem_avail" -lt 8192 ]; then
  echo "WARN only ${mem_avail} MiB RAM available - a retrain OOM-kill looks like exit 137, no traceback"
elif [ "$swap_used" -gt 4096 ]; then
  echo "WARN ${swap_used} MiB swapped with ${mem_avail} MiB available - check for thrashing"
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
