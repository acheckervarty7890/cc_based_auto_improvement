# Dev samples and the eval ceiling — run of 2026-08-15

Two experiments on the human-harm gemma-3-27b (L32) probe, run in one pass by
`run_devsamples_kfold.sh`:

- **(i) `scripts/eval_kfold_cv.py`** — 5-fold CV with the eval set as *both* train and
  test. The in-distribution ceiling: how well any probe of this architecture can do on
  these splits when it has seen data drawn from them.
- **(ii) `scripts/dev_sample_retrain.py`** — the iteration-3 retrain plus 0/2/8/16/30
  hand-written dev samples per eval split, both attacker arms. The dose-response curve
  between the red-team pipeline and that ceiling.

Everything below is from the completed run: 10/10 dev-sample jobs, 0 failures,
`ALL_DONE` at 09:43:40Z. Inputs are `dev_samples/*.jsonl` (120 conversations, committed
to this branch) and the post-shortening `redteam_postprocessed_iter3.jsonl` of each arm
(878 rows deepseekv4pro, 778 gptoss120b).

## (i) The ceiling

5-fold CV, `results/devsamples_kfold/kfold_cv/kfold_cv_summary.csv`:

| geometry | AUROC (mean ± sd) | accuracy | n |
|---|---|---|---|
| pooled, all splits | **0.9766** ± 0.0109 | 0.9342 | 866 |
| pooled, eval_ai_dilemmas | 0.9912 | 0.9553 | 136 |
| pooled, eval_ant_hh | 0.9347 | 0.8273 | 134 |
| pooled, eval_balanced_refusal | 0.9924 | 0.9475 | 400 |
| pooled, eval_daily_dilemmas | 0.9818 | 0.9642 | 196 |
| within:eval_ai_dilemmas | 1.0000 ± 0.0 | 0.9775 | 136 |
| within:eval_ant_hh | 0.9069 ± 0.0433 | 0.8509 | 134 |
| within:eval_balanced_refusal | 0.9992 ± 0.0018 | 0.9925 | 400 |
| within:eval_daily_dilemmas | 0.9811 ± 0.0179 | 0.9487 | 196 |

`eval_ant_hh` is the hard split under every geometry, and it is also the only one of the
four that is **not paired** (see the eval-split notes in `CLAUDE.md`) — worth keeping in
mind before reading its number as a property of the concept rather than of the data.

## (ii) The dose-response curve

Mean over the four eval splits, `results/devsamples_kfold/dev_samples/dev_sample_summary.csv`.
One weight-init seed per cell (`n_seeds=1`), so there are no error bars here.

| dev/split | fitted / val rows | deepseekv4pro AUROC | gptoss120b AUROC |
|---|---|---|---|
| 0 | 0 / 0 | 0.9112 | 0.8978 |
| 2 | 6 / 2 | 0.8771 | 0.9107 |
| 8 | 24 / 8 | 0.9113 | 0.9149 |
| 16 | 49 / 15 | 0.9203 | 0.9132 |
| 30 | 94 / 26 | **0.9290** | **0.9307** |

Both arms rise monotonically from n=8 and land at ~0.93 by n=30 — still about 0.05 short
of the 0.977 pooled ceiling. The two arms start ~0.013 apart and converge, so at n=30 the
dev samples matter more than which attacker produced the red-team set.

Two caveats on reading this:

- **The n=2 dip** in the deepseekv4pro arm (0.8771, *below* its own n=0) is a single seed.
  Nothing here separates it from noise; re-running the level across seeds is the cheap way
  to settle it, and the per-fold files support that without recomputing activations.
- **`tpr_at_fpr` does not track AUROC.** It rises then falls for deepseekv4pro
  (0.55 → 0.64 at n=8 → 0.43 at n=30) and falls outright for gptoss120b
  (0.39 → 0.13 at n=16). Whatever the dev samples buy, it is not operating-point
  behaviour at a fixed low FPR.

## What broke, and what the fixes were

The previous attempt died and this one needed three fixes before it would run at all. All
three were failures of the *harness*, not of the experiment, and all three are the kind
that fail late and expensively.

### 1. Layer truncation crashed any disk-offloaded load (`dc02a84`)

`scripts/dev_sample_retrain.py` died loading gemma-3-27b with `KeyError: ''` inside
transformers' `get_disk_only_shard_files`.

`model_loading` truncates the config to layers `0..probe.layer`, which makes the model's
module tree a strict subset of the checkpoint's. transformers builds its disk-offload
index from the *checkpoint's* key list — `_get_key_renaming_mapping` maps every serialized
key — so the dropped layers stay in the `weight_map`, and the two helpers that look those
names up in the `device_map` cannot cope: `get_disk_only_shard_files` walks a name's
prefixes to `""` and indexes with it, and `expand_device_map` omits them so the
`disk_offload_index` built right after raises on the same names.

Neither helper runs unless `"disk" in device_map.values()`. **That is the whole shape of
the bug: truncation is free on a box roomy enough to hold layers `0..layer` across
GPU+CPU, and crashes the load on the tight box it exists to help.** gemma-3-27b at layer
32 is 30 GB of executed weights; this box has an 8 GB GPU and 15 GB of RAM.

`load_extraction_model` now installs behaviour-preserving replacements, once per process
and only when truncation fired. `scripts/check_truncated_disk_offload.py` reproduces and
checks the whole thing in seconds on a tiny sharded Llama — no network, no token, no GPU —
and asserts the kept layers stay **bit-identical** to the untruncated model, which is the
property every activation cache in this repo depends on.

### 2. The retrain was OOM-killed assembling its training set (`1bc5396`)

Exit 137, `anon-rss 15.6 GB`, within a minute of the first job starting and before a single
forward. Two causes, both in assembly rather than the fit:

- **The set was merged twice.** `_activate_redteam_cached` merged the per-conversation
  parts into one padded tensor, and `_combine` then merged *that* with the base side — a
  second full copy of the largest tensor in the run. At 703 train rows padded to the
  1024-token cap (hidden 5376, fp16) that copy is 7.7 GB, allocated purely to prepend 50
  base rows. Each side now does exactly one concatenate over base + red-team parts
  together.
- **Freed parts never left the process.** Consuming a part returns it to glibc's arena,
  not to the kernel, so RSS held the high-water mark of every part ever loaded.
  `_concatenate_consuming` now calls `malloc_trim` every 50 parts.

Measured peak on the deepseekv4pro n=0 job: **15.8 GB (OOM) → 13.8 GB** with the single
merge, on a box with 15.9 GB. The trim bought the rest: the n=30 jobs add ~1.7 GB of dev
samples and would have landed back over the limit, but ran at 12.5 GB.

Concatenation is associative and the pad width is the max over the same set either way, so
the merged tensors are byte-identical to the two-step form. Checked against tuberlens' own
`concatenate`, and one-merge against two-merge, before restarting — **the science is
unchanged.**

### 3. Three preflights, each learned the same way

- **`HF_TOKEN`.** gemma-3-27b-it is gated, so `hf_login()` raised — but only after the
  ~15 GB of Kaggle downloads and the whole CV stage had run. `run_devsamples_kfold.sh` and
  `cloud_start_devsamples.sh` now refuse to start the dev part without one.
- **git identity.** A fresh container keeps the checkout but not `~/.gitconfig`, so every
  `git commit` failed while the failsafe's snapshot path only *warned*. The run looked
  protected and every snapshot was silently discarded. `failsafe_commit_devsamples.sh` now
  refuses to start without an identity.
- **Hand-placed credentials.** `HFtokn.txt` and `kaggle/` are excluded on the failsafe's
  **ADD** side, not merely gitignored — it force-adds, so `.gitignore` alone would not have
  held. Neither is in history.

## Timings on this box (RTX 4060 8 GB, 15 GB RAM)

| stage | cost |
|---|---|
| Kaggle eval activations | 4.3 GB, ~2 min |
| Kaggle red-team activations | 1656 blobs / 6.6 GB, ~105 min (per-file, latency-bound) |
| gemma-3-27b weights | 52 GB, ~10 min |
| dev activation extraction | 120 conversations, **37.9 min** (13–19 s/sample) |
| 10 retrain + eval jobs | ~35 min total, ~3.5 min/job |

The extraction ran far faster than a disk-offloaded gemma-3-27b usually does here because
`AGENTIC_REDTEAM_MAX_MEMORY="0=6GiB,cpu=7GiB"` was pinned; unpinned, accelerate hands the
CPU whatever is free at load time and leaves nothing for the process.

The 120 dev activations are computed on the box and published nowhere. They are cached
under `results_hu_harm_gemma27b_batch_ablation/base_activations/` and are the one input
here that a container wipe destroys — `scripts/publish_kaggle_redteam_activations.py` is
how to make them outlive it.

## Worth doing next

- **Seeds.** Every cell is `n_seeds=1`. The n=2 dip and the `tpr_at_fpr` wobble are both
  unresolvable without them, and re-running levels costs only the fits — the activations
  are all cached.
- **The gap to the ceiling.** ~0.05 AUROC at n=30, concentrated in `eval_ant_hh`. Whether
  that closes with more dev samples or is a property of the pipeline's data is the obvious
  next question, and n=30 is not yet a plateau.
- **Peak memory is still ~14 GB against 15.9.** If a future run adds rows it will need the
  next lever: loading each blob straight into the preallocated output instead of building
  a parts list at all.
