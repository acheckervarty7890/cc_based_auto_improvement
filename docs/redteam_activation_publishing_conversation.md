# Session: extracting and publishing the hs / instructions red-team activations

Record of the 2026-08-14/15 session that generalized the Kaggle red-team publisher to cover
`experiment9_cloud` and `experiment_instruction_cloud_1`, and ran it on the WSL2 dev box until
the job was moved to a larger one. Tool output is abridged — progress bars and per-shard load
logs ran to hundreds of KB — but every number quoted below is copied from the run it came from.

The operational state and the restart instructions live in
[`redteam_activation_publishing_handoff.md`](redteam_activation_publishing_handoff.md); this
file is the reasoning, including the two measurements that were wrong on the first pass.

---

## The ask

> Look into the previous conversation in vintage conversation for reference. There are two
> branches: experiment9_cloud, and experiment_instruction_cloud_1, extract red teaming
> activation for both them and publish them on kaggle. This might take some time, when you run
> shell script monitors, also run a shell script to wake yourself after every 30 mins to check
> if everything is running fine.

Later, when asked what counted as in scope:

> only samples used for training

which is what the publisher already does — it reads `redteam_postprocessed_iter{N}.jsonl`, the
snapshot `retrain_probe` dumps after preprocessing and before the split, i.e. exactly the
red-team set that trained that iteration's probe. Not the raw attempt log, and no eval data.

And at the end:

> okay, finish the upload the save this conversation somewhere and other important things you
> think would useful to restart and commit and push in different branch. I am going to start
> this in different box with higher vram 24gb.

## What already existed

`scripts/publish_kaggle_redteam_activations.py` on `experiment11_cloud` published the
**harmful_to_human** arms, with the two arms hard-coded in a module-level `ARMS` dict and every
path resolved against `REPO_ROOT`. `experiment9_cloud` carried a *fork* of an earlier version of
that script (`publish_kaggle_hs_redteam_activations.py`) adapted to the high-stakes arms, adding
a slug-namespace guard; it had never been run — none of its datasets existed on Kaggle.

Two forks of one script, with the newer features (per-file restore, pagination) only in one and
the namespace guard only in the other, is not a base to add a third experiment to. So the fork's
guard was folded into the maintained script and the hard-coded arms became a registry:

```python
EXPERIMENTS = {"hu_harm": ..., "hs": ..., "instructions": ...}
```

Each `Experiment` carries its arms, its branch, its activation-cache dir and the Kaggle slug
namespace it owns. Two new flags: `--experiment` and `--experiment-root`, the latter because an
experiment's probes, configs, dumps and base data live **on its own branch** and the stage runs
from a different one (`git worktree add --detach` is enough — nothing is written there).

`hu_harm`'s plan was dry-run before and after and is byte-identical: same seven units, same
slugs, same archive names, so the existing published datasets stay addressable. The namespace
guard was tested by pointing an `hs` publish at the `hu-harm-` slug template; it refused, naming
the experiment it would have overwritten.

Scope resolved from the dumps: hs 1,470 unique conversations (648 + 822 across the two arms),
instructions 1,992 (968 + 1,024), 14 Kaggle datasets in total. `dataset_status` confirmed none
of the `hs-*` or `instructions-*` slugs existed, so nothing was at risk of being overwritten.

## Two things had to be fixed before anything ran at all

**1. The recorded memory pin was too small, and its failure mode is silent.** The first launch
failed every unit with `FAILED extracting <unit>: ''` — an empty exception message. Reproduced
directly:

```
KeyError: ''
  transformers/modeling_utils.py:5864 in get_disk_only_shard_files
    files_content[filename].append(device_map[weight_name])
```

`get_disk_only_shard_files` only runs when accelerate has decided to offload to **disk**. It
walks the checkpoint's `weight_map`, and our truncated config has dropped layers 33-61, so those
weight names have no `device_map` entry at any prefix; the name is trimmed to `""` and the lookup
raises. So layer truncation and disk offload are incompatible, and the symptom is a bare
`KeyError` with no hint of either.

The pin came from a memory note: `AGENTIC_REDTEAM_MAX_MEMORY="0=6GiB,cpu=23GiB"`. Measuring the
truncated model off the safetensors index gave **30.9 GB = 28.8 GiB** (27.25 GB of layers 0-32,
2.82 GB embeddings, 0.83 GB vision tower), against a 29 GiB budget — so accelerate had nowhere
to put the remainder except disk. `cpu=25GiB` loads with cpu+GPU only. The memory has been
corrected.

**2. Extraction was far slower than the note implied, and the first fix was wrong.** At
`BATCH_SIZE=1` the first measurement was 18 s/sample — ~17 h for 3,462 conversations. A second
at `BATCH_SIZE=4` gave 4.8 s/sample, which fit a tidy story: the model does not fit in VRAM, so
each forward streams the CPU-resident layers to the GPU and costs the same for one conversation
as for four. Batching was turned on, `_extract_conversations` gained length-sorted chunking so
the padding `get_activations` applies per call would not inflate the blobs, and the run started.

That story was wrong, and the ETA built on it (~7 h) was wrong with it. The 4.8 s/sample slice
happened to be short conversations. An A/B on the **same 20 conversations** settled it:

| leg | config | s/sample | blob size |
|---|---|---|---|
| A | `BATCH_SIZE=1` | 33.7 | 2.17 MB avg |
| B | `BATCH_SIZE=4`, same 20 | 68.7 | 2.64 MB avg |

Batching was **2× slower**, and the blobs bigger. Diagnosis, from `/proc/<pid>/io` and `vmstat`:
`read_bytes` climbing 16 MB/s while `rchar` stayed flat — page faults on mmapped weights, not
syscall reads — with the CPU 95% idle and sequential disk reads measured at 558 MB/s. The box
cannot cache all 28.8 GiB of weights (WSL2 pins the page cache near 23 GiB and holds ~6 GB free,
and there is no root to tune `swappiness` or drop caches), so each forward faults a slice back at
random-read speed, and a bigger batch's intermediates simply evict more of what the next forward
needs. **That fault traffic is the per-sample cost.**

A third leg tested a stub vision tower (0.83 GB less resident) — implemented as
`AGENTIC_REDTEAM_TRIM_VISION`, since text-only conversations never execute the tower. It is
**bit-identical**: three recomputed conversations matched the untrimmed blobs exactly on
`activations`, `attention_mask` and `input_ids`. But it timed 76.4 s/sample, *worse* than leg A,
and the three legs had degraded monotonically (33.7 → 68.7 → 76.4) regardless of what was being
varied — so the box's cache state, not the knob, was driving the measurements. Further 25-minute
A/Bs would not have produced a trustworthy answer, so the run restarted on the configuration with
the best first-principles case (smallest working set, smallest blobs, verified-identical trim) and
the long run became the measurement.

## What actually made it fast

The live run then showed a rate that *improved within a unit*: 116 → 107 → 86 → 67 s/sample
cumulative, with a marginal rate of 8.6 s/sample between the last two checkpoints. The expensive
part was never the forward — it was the **warm-up after each model load**, while the working set
faults in. And `sync` releases the model before every pack+upload by default, so that warm-up was
being paid **14 times**, once per unit. It also explains the benchmark legs: each measured 20-24
samples that were almost entirely warm-up.

`--no-release-between-units` fixed it. Sustained rate afterwards: **9.6 s/sample**, holding across
units, with per-unit uploads of ~90 s at 17 MB/s.

Three ETAs were given over the session — ~7 h (wrong, built on the 4.8 s/sample slice), ~31-68 h
(measured, but during the thrashing regime), and ~9-13 h (the warm rate, which held). The middle
one was corrected to the user explicitly rather than quietly revised.

## Where it stopped

Four of fourteen datasets are published: `hs-gemma27b-base` and the complete
`hs-gemma27b-gptoss120b-iter{1,2,3}` arm. During the last of those, Kaggle's upload throughput
collapsed from 17 MB/s to ~0.46 MB/s mid-flight — a 3.3 GB archive taking over half an hour. If
that persists it becomes the binding constraint on the remaining ~26 GB, and the answer would be
to publish one union dataset per arm plus a per-iteration membership manifest (same information,
~55% less volume), starting with `instructions`, which has not begun. hs is already half-published
in the per-iteration layout and should stay that way.

The job then moved to a box with 24 GB of VRAM, where the model nearly fits and the fault-bound
regime should not apply — which is exactly why the handoff document tells the next box to
**re-measure** `BATCH_SIZE` and `AGENTIC_REDTEAM_MAX_MEMORY` rather than inherit values tuned
against a constraint it does not have.
