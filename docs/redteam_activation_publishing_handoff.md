# Publishing the hs / instructions red-team activations — state and restart runbook

Written 2026-08-15, when the job moved off the WSL2 dev box (8 GB VRAM) to a 24 GB-VRAM box.
Everything below is what a fresh box needs to finish it.

## What the job is

Two gemma-3-27b runs wrote their activation caches on cloud boxes that were never synced back.
Only the probes, the postprocessed red-team dumps and the results JSONLs survived, so the
activations every retrain read have to be pushed through `google/gemma-3-27b-it` (layer 32)
again and published to Kaggle:

| experiment | branch | concept | arms | unique conversations |
|---|---|---|---|---|
| `hs` | `experiment9_cloud` | high-stakes | gptoss120b, deepseekv4pro | 1,470 |
| `instructions` | `experiment_instruction_cloud_1` | instruction-following | gptoss120b, nemotron | 1,992 |

Each arm publishes one **self-contained** dataset per retrain iteration (the iterations do not
nest — `filter_dataset` refits every cycle, so only exact per-iteration membership reproduces
what probe *k* trained on), plus one base-training-split dataset per experiment. 14 datasets,
all private, owned by `anku7890`.

The only conversations extracted are the ones that actually **trained** a probe:
`probes/<arm>/redteam_postprocessed_iter{N}.jsonl`, which `retrain_probe` dumps after
preprocessing and before the split. No raw attempt logs, no eval data.

## Done so far (2026-08-15 06:00 UTC)

Published and verified on Kaggle — a re-run **skips these**, it asks Kaggle what exists:

- `anku7890/hs-gemma27b-base`
- `anku7890/hs-gemma27b-gptoss120b-iter1`
- `anku7890/hs-gemma27b-gptoss120b-iter2`
- `anku7890/hs-gemma27b-gptoss120b-iter3`  ← the whole hs gptoss120b arm is complete

Still owed: hs deepseekv4pro iter1/2/3 (822 unique conversations), and all seven instructions
units (1,992). ~2,790 forwards remain, plus ~26 GB of archive uploads.

676 blobs sit in `results_hs_gemma27b_batch_ablation/base_activations/` on the old box. All but
~24 of them belong to already-published units, so there is nothing worth copying across —
`sync --restore-published` (the default) pulls them back from Kaggle instead.

## Restarting on the new box

```bash
git clone <repo> && cd cc_based_auto_improvement && git checkout redteam-activation-publishing
bash scripts/setup_env.sh          # or reuse an existing .venv_claude
export KAGGLE_CONFIG_DIR=$PWD/kaggle   # the DIRECTORY holding kaggle.json, not the file
mkdir -p logs && nohup bash run_publish_redteam_acts.sh > logs/publish_redteam_acts.out 2>&1 &
```

`run_publish_redteam_acts.sh` creates a read-only `git worktree` per experiment under `$WT_ROOT`
(default `~/wt`), because each experiment's probes, configs, dumps and base data live on **its
own branch** — this branch has none of them. It then runs `hs` and `instructions` sequentially.
Nothing is written into the worktrees.

Resume is driven by Kaggle, not local state: a unit whose dataset already exists is skipped and
its blobs are downloaded back into the cache so later overlapping units get hits. Kill and re-run
freely — every blob is written through as soon as it is computed.

Watch it with:

```bash
tail -f logs/publish_redteam_acts_hs.log | tr '\r' '\n' | grep -E "activations\]|>>> \[|done:|FAILED"
```

## Tuning for a 24 GB-VRAM box — read this before the first run

The truncated model (layers 0..32 + embeddings + vision tower) is **30.9 GB bf16 = 28.8 GiB**.
Two knobs in `run_publish_redteam_acts.sh` were tuned for a box that could not hold it, and
should be **re-measured** on one that nearly can:

- **`AGENTIC_REDTEAM_MAX_MEMORY`** (currently `0=6GiB,cpu=25GiB`). With 24 GB of VRAM, try
  something like `0=22GiB,cpu=8GiB`, which puts most of the stack on the GPU. **The total budget
  must exceed 28.8 GiB.** If it does not, accelerate falls back to *disk* offload, and layer
  truncation then dies with a bare `KeyError: ''` in transformers' `get_disk_only_shard_files`
  (the dropped layers are still in the checkpoint's `weight_map` but absent from the
  `device_map`). It surfaces only as `FAILED extracting <unit>: ''` — an empty message, no
  traceback. Also leave headroom on the GPU: at a 6 GiB weight pin, runtime peak was 7.4 GiB.
- **`BATCH_SIZE`** (currently 1). Batching *lost* on the old box — 33.7 s/sample at 1 against
  68.7 at 4, on the same 20 conversations — because a bigger working set evicted weight pages
  that then faulted back at ~16 MB/s of random reads, with the CPU 95% idle. Where the model
  stays resident that reason disappears and batching should win; `_extract_conversations` sorts
  the work by length before chunking so a batched run keeps most of the per-blob compactness of
  chunk size 1. Measure it: run `extract --limit N` on one arm at 1 and at 4-8.

Also relevant, both already set in the run script:

- `--no-release-between-units` keeps the model loaded across each pack+upload. On the old box a
  reload cost 100+ s/sample of cold warm-up before settling to ~9 s. On a box that loads fast
  this matters less, but it never hurts.
- `AGENTIC_REDTEAM_TRIM_VISION=1` instantiates a 1-layer stub in place of gemma-3's vision tower,
  which text-only conversations never execute. Worth 0.83 GB of resident weights. **Verified
  bit-identical**: three conversations recomputed with it matched the untrimmed blobs exactly on
  `activations`, `attention_mask` and `input_ids`. Opt-in, default off, implemented in
  `model_loading.py`.

Expected performance: the old box managed ~9.6 s/sample once warm, so ~2,790 conversations took
an estimated 7-8 h there. A box that holds the model should be materially faster, and the job
may become upload-bound instead.

## Gotchas that cost time here

- **Kaggle throttles uploads.** The first three archives went at ~17 MB/s; the fourth (3.3 GB)
  dropped to ~0.46 MB/s mid-flight and took over half an hour. If that persists, the remaining
  ~26 GB of archives dominate the run. It recovered on its own before; if it does not, see the
  volume-reduction option below.
- **`dataset_status` 403s for a dataset that does not exist**, and returns `ready` for one that
  does. `_remote_exists` treats any failure as "absent", so a resume never silently skips a unit
  that was not actually uploaded.
- **Kaggle's listing APIs are unreliable for this account** (`dataset_list_files` 403s,
  `dataset_list` omits real datasets) while downloads work fine. Never conclude a blob is missing
  from a listing.
- **Blobs are content-addressed**, so a name that matches is the same conversation. Restoring
  several iterations into one cache dir is safe, and blobs computed at different batch sizes
  interoperate (`_concatenate_consuming` re-pads at merge).

## If uploads become the bottleneck

The per-iteration datasets are self-contained, so a conversation shared by two iterations is
uploaded twice or three times: ~3,462 unique blobs become ~7,266 uploads, ~31 GB. The
volume-preserving alternative is **one union dataset per arm plus a per-iteration membership
manifest** — the manifest still records exactly which conversations trained which probe, and
upload volume drops ~55%. hs is already half-published in the per-iteration layout, so the
sensible place to switch is the `instructions` experiment, which has not started. This is not
implemented; `_iteration_units` in the publisher is where it would go.

## Files that matter

- `scripts/publish_kaggle_redteam_activations.py` — the publisher. `EXPERIMENTS` registers the
  three runs (`hu_harm`, `hs`, `instructions`), each with its arms, cache dir and Kaggle slug
  namespace. `_check_slug_namespace` refuses to publish outside a run's own namespace, so a
  `--dataset-slug` typo cannot overwrite a sibling experiment's datasets.
- `run_publish_redteam_acts.sh` — drives `hs` then `instructions`, with every tuning decision
  above recorded as a comment next to the export that sets it.
- `src/agentic_redteam/model_loading.py` — layer truncation, the `max_memory` pin and the vision
  trim.
- `CLAUDE.md` §`scripts/publish_kaggle_redteam_activations.py` — the durable description.
- `docs/redteam_activation_publishing_conversation.md` — the session that produced all of this,
  including the measurements that were wrong on the first pass and why.
