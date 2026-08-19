# Speeding up ensemble probe training

Measured investigation into why fitting probe heads — especially the `n` members of
a `--ensemble-size` deep ensemble — is slow, and what fixes it. All numbers below
were measured in-process against `tuberlens` at commit `60ea5ab` (the code as it
was before this work) on an RTX 4060 (8 GB) / 4 vCPU box, with synthetic
activations shaped like a real retrain: `seq_len=1024` (`get_activations`' cap),
`hidden=2048` (llama-1b), ragged lengths averaging ~535 tokens, `bfloat16`.

Reproduce the correctness half with:

```bash
.venv_claude/bin/python scripts/verify_ensemble_training.py --bench
```

## 1. Where the time actually went

A probe head is one `embed_dim -> 1` projection. It is tempting to assume the fit
is compute-bound; it is not. Instrumenting one epoch of the original
`PytorchAdamClassifier.train` (300 train / 600 validation samples,
`linear_then_softmax`, 5 epochs):

| phase | time | share |
| --- | ---: | ---: |
| `val_dataload` | 10.28 s | 58% |
| `train_dataload` | 5.30 s | 30% |
| `train_bwd` | 0.84 s | 5% |
| `train_fwd` | 0.83 s | 5% |
| `val_fwd` | 0.50 s | 3% |
| `val_auroc` | 0.02 s | <1% |

**87% of wall-clock is the DataLoader.** And an ensemble multiplies exactly that
part by `n`, because every member re-walks the same activations.

## 2. Why the DataLoader is slow (it is not collation)

`ActivationDataset.__getitems__` gathers per-sample tensors and lets
`default_collate` re-stack them, which looks like the obvious waste. It is not.
Timing one full pass over 600 samples (2.52 GB) at `batch_size=16`:

| variant | ms/pass | effective |
| --- | ---: | ---: |
| baseline (list split + `default_collate`) | 1795.6 | 1.40 GB/s |
| + identity collate (no relist/restack) | 1984.9 | 1.27 GB/s |
| + drop the unused `input_ids` | 1772.5 | 1.42 GB/s |
| manual permutation chunks, no DataLoader | 1787.0 | 1.41 GB/s |
| **pinned staging buffer + `non_blocking`** | **661.9** | **3.80 GB/s** |
| source tensor fully pinned (gather still on host) | 1650.8 | 1.52 GB/s |
| **activations resident on the GPU** | **21.0** | **119.67 GB/s** |
| sequential slices, batch 64 (validation-shaped) | 478.6 | 5.26 GB/s |
| sequential slices, batch 256 | 484.4 | 5.20 GB/s |

The cost is a **random gather into pageable host memory** followed by a pageable
host-to-device copy. Removing collation overhead changes nothing; changing *where
the memory lives* changes everything. Note also that pinning the source is not
enough — `source[indices]` gathers into a fresh pageable tensor — the gather has
to write *into* page-locked memory (`index_select(..., out=pinned)`).

## 3. What was implemented

### 3.1 `ActivationBatcher` (tuberlens `interfaces/activations.py`)

Replaces `DataLoader(ActivationDataset(...))` wherever a probe head reads
activations, and picks a placement instead of a collation strategy:

- **`resident`** — the whole tensor lives on the GPU for the fit; a batch is a
  device-side gather. One transfer for the entire run instead of one per batch per
  epoch per member. Chosen when it fits `PROBE_GPU_RESIDENT_FRACTION` (default
  0.6) of free device memory. Falls back to `staged` on `OutOfMemoryError`, since
  the budget is a prediction and another allocation can win the race.
- **`staged`** — activations stay on the host; each gather is written straight into
  a page-locked staging buffer with `index_select(out=)` and copied on a side
  stream, double-buffered so the next gather overlaps the current batch's compute.
- **`pageable`** — the original behaviour, kept for A/B and CPU-only boxes.

The free-memory check adds back `memory_reserved - memory_allocated`. Torch's
caching allocator holds freed blocks as *reserved*, so the driver reports them as
used; counting only driver-free memory made every fit after the first conclude the
GPU was full and silently take the slow path. That bug alone turned an early
measurement of the fused path from 8x faster into 0.6x *slower*.

### 3.2 Training-loop changes (tuberlens `probes/pytorch_classifiers.py`)

- The validation batcher is built **once per fit**, not once per epoch. Previously
  every epoch re-transferred the entire validation set — with `--dev-data
  dev_samples/highstakes` (1908 rows, 8.0 GB on llama-1b) that is 8 GB of PCIe
  traffic per epoch, per member.
- No-grad passes use `PROBE_EVAL_BATCH_SIZE` (default 64). The training batch size
  of 16 exists to shape the optimizer's gradient noise; an inference pass has no
  such constraint and 16 is pure per-batch overhead.
- The epoch loss is accumulated on-device and read once per epoch. `loss.item()`
  and `loss.isnan()` per batch each force a host/device sync — thousands of stalls
  for a number printed once. The NaN assert moved to the epoch mean, which a NaN
  survives to anyway.

### 3.3 Fused ensemble training (`_stack_and_train_ensemble_state`, `ProbeFactory.build_ensemble`)

`stack_module_state` puts the members' parameters on a leading dimension and
`torch.vmap` steps them together, so `n` microsecond-scale kernels become one.
Comparing ensemble strategies with the data path already fixed (so this is
compute-vs-compute), 5 members x 5 epochs:

| strategy | s/epoch | speedup |
| --- | ---: | ---: |
| sequential members | 0.947 | 1.00x |
| shared batch, python loop over members | 0.414 | 2.29x |
| vmap over stacked members, shared batch | 0.112 | 8.49x |
| **vmap, independent per-member permutations** | **0.128** | **7.42x** |

The last row is the one shipped: preserving each member's own batch order costs
almost nothing versus broadcasting one shared batch, and it keeps the fused fit
faithful to what a sequential fit would have done.

Members are unstacked back into ordinary independent probes, so the pickle,
`EnsembleProbe`, `ProbeJudge`, the eval and `_infer_probe_spec` are all unchanged.

### 3.4 Fused ensemble scoring (`EnsembleProbe._fused_proba`, `stacked_probs`)

`_mean_proba` used to call each member's `predict_proba`, walking the activations
`n` times. One vmapped pass over stacked weights reads them once: **4-5x** on a
5-member ensemble. This runs on every red-team submission and every eval split, not
only at retrain time. Returns `None` and falls back to the loop when the members
are not a stack of identical pytorch heads.

## 4. Results

5 members x 10 epochs, hidden 2048, versus the original loop:

| activations fit the GPU | loader only (members still sequential) | + fused ensemble |
| --- | ---: | ---: |
| yes — 1.3 GB train, 2.5 GB val | **12.7x** | **38.2x** |
| validation spills — 1.3 GB / 5.0 GB | 3.5x | 15.2x |
| neither fits — gemma-27b-sized | 2.7x | 4.0x |

Absolute, first row: 121.3 s → 9.6 s → 3.2 s.

The gain shrinks as the activations stop fitting because the staged path is still
transfer-bound; the fused trainer keeps a larger edge there because all members
share one validation pass.

## 5. Correctness

### The loader change is bit-identical

Not "close" — byte-for-byte the same probe, and all three placements train to
identical weights. Getting there required one non-obvious detail.

`ActivationBatcher` reproduces `DataLoader`'s permutation *and its RNG
consumption*. A shuffled DataLoader iteration draws **two** int64s off the ambient
RNG (`_BaseDataLoaderIter`'s worker base seed, then `RandomSampler.__iter__`'s
permutation seed); an **unshuffled** iteration still draws **one** (the base seed).
That unshuffled draw matters because the training loop iterates a `shuffle=False`
DataLoader over the validation set every epoch — so the draw lands *between* two
training shuffles and moves them. With the draw missing, the fits diverged
(`best_epoch` 7 vs 8, `max|Δp|` 7.8e-3) while looking for all the world like a
rounding difference. This is the failure mode the whole exercise is exposed to: a
"pure speedup" that quietly retrains every probe differently.

### The fused ensemble is equivalent, not identical

It preserves per member: seed and weight init, batch order (hence the
`(n, batch, seq, embed)` gather rather than one shared batch), gradient clipping
(`clip_grad_norm_` over the stacked tensors would take one norm across all members
and couple them), and early stopping — where a stopped member is **frozen** rather
than merely left running, since otherwise a later epoch could hand it a checkpoint
its sequential counterpart would never have reached.

What it does not preserve is floating-point association: vmap dispatches the
members' projections as one batched matmul. Measured against sequential fits on a
planted-signal task:

| | per-member held-out AUROC | ensemble AUROC |
| --- | --- | ---: |
| original (sequential) | 0.9629 / 0.9700 / 0.9682 / 0.9596 / 0.9610 | 0.9722 |
| optimized loader (sequential) | 0.9629 / 0.9700 / 0.9682 / 0.9596 / 0.9610 | 0.9722 |
| fused | 0.9629 / 0.9702 / 0.9667 / 0.9586 / 0.9602 | 0.9721 |

Per-member score correlation 0.997. Fused training remains fully reproducible from
`ENSEMBLE_SEEDS` alone — same seeds, same probe, on every run.

### Regression checks

`scripts/verify_ensemble_training.py` asserts all of the above (17 checks). There
is no test suite in either repo and this is a numerically sensitive path, so it is
worth running after touching the batcher, the training loop or the fused trainer.

## 6. Evaluated and rejected

- **Trimming batches to their own max length.** `get_activations` pads every row to
  1024 while real conversations average ~535 tokens, so most of what is transferred
  is padding, and a trimmed gather scales linearly with the span (1024 → 14.5 ms,
  600 → 8.9 ms, 400 → 5.8 ms). But trimming to the *batch's* max saves little — the
  max of 16 random draws already sits in the tail — and trimming to a
  length-**sorted** batch's max, which would nearly halve the staged path, reorders
  samples. That is harmless for validation but changes `linear_then_max` /
  `linear_then_rolling_max` results, whose max can legitimately land on a masked
  zero. Not worth the exactness footnote for a path the fused trainer already
  improves 4x. It also needs a *flat contiguous* pinned buffer: trimming into a
  sliced view of a full-width buffer makes the copy strided and lands at 26 ms,
  slower than not trimming at all.
- **A custom collate / dropping `input_ids`.** Correct in principle (`input_ids` is
  cast to `bfloat16` and never read during training) but worth ~1% — see §2.
- **DataLoader `num_workers > 0`.** The dataset is already-materialized tensors;
  workers would add IPC copies to a path whose problem is the copies.

## 7. Relation to experiment17

`experiment17_cloud` profiled the same bottleneck from the other end — a live
10-member gemma-27b run rather than a synthetic bench — and its diagnosis is
recorded in `experiment17_session_log.md` on that branch. This section reconciles
the two, and corrects two of its conclusions with measurements it could not make
at the time (its GPU was occupied by the run it was profiling).

### Confirmed independently

- **The fits dominate, and the extraction this repo is architected around caching
  is the cheap part.** experiment17 measured probe fits at 57% of iteration 0's
  wall-clock rising to >90%, against 12–23 min of gemma-27b extraction per retrain.
- **The head arithmetic is free; it is all data movement.** Its CPU benchmark had
  the full forward (variant E, 3.37 ms/sample) indistinguishable from merely
  touching the bytes (variant D, 3.23 ms/sample).
- **GPU residency is the main lever, and random gather on-device is nearly free.**
  It measured 83 ms/epoch for a resident random gather against 58 ms for
  contiguous reads and concluded, correctly, that exact per-sample shuffling is
  worth its 30% premium and should not be traded for block-shuffling. §2 here
  reaches the same place from the other direction (119 GB/s resident vs 1.4 GB/s
  pageable).
- **Determinism.** It ran a control that re-fit the initial probe and reproduced
  `probe_iter0.pkl` bit-for-bit across all 10 members, `best_epoch` included. That
  is what makes the bit-identity check in §5 a meaningful test rather than a
  hopeful one.

### Corrected: "the unbind/restack fix is worth approximately zero"

experiment17 measured the collate overhead at 2.2x on CPU, then — after its GPU
benchmark showed variant 2 (no unbind/restack) indistinguishable from variant 1 —
retracted it: "in the real path it's worth nothing, because the PCIe transfer
dominates so completely that the collate overhead disappears into it. My patch, as
written, would have bought approximately zero."

That is right about the *CPU-resident* path, and §2 here reproduces it (an
identity collate changed nothing). But its recommendation was residency **while
keeping the DataLoader**, and the retraction does not survive that change: once
the transfer is gone, the per-batch unbind-into-16-tuples and restack is what is
left. Measured on a full fit — forward, backward, AdamW, clipping, the per-epoch
validation pass and AUROC, none of which either side's microbenchmarks included:

| | hidden 2048 (300/600) | hidden 5376 (150/150) |
| --- | ---: | ---: |
| residency alone, DataLoader kept | 8.2x | 11.5x |
| residency + `ActivationBatcher` | 16.5x | 22.5x |
| **what the batcher adds on top** | **2.0x** | **2.0x** |

The overhead did not stop mattering; it stopped being *visible* behind a larger
cost, and removing that cost exposes it again.

### Corrected: "residency removes the reason to parallelize at all"

experiment17 listed fusing the heads under `vmap` as its third parallelism option,
estimated it at ~10x, and then set parallelism aside: residency "removes the
reason to parallelize at all", and fusing would cost diversity because "shuffle
order becomes shared across members".

Both halves are wrong, and in the same direction. Once residency lands the fit is
**launch-bound** — tens of microsecond-scale kernels per batch with the GPU idle
between them — which is precisely the regime fusing fixes, so residency does not
remove the reason to parallelize, it makes parallelism the *only* remaining lever.
And the diversity cost is avoidable: gathering `(n, batch, seq, embed)` keeps each
member's own permutation for 0.128 s/epoch against 0.112 for one shared batch, a
14% premium (§3.3). At experiment17's own ensemble size:

| 10 members x 25 epochs, 300 train / 600 val | wall-clock | vs original |
| --- | ---: | ---: |
| original tuberlens, 10 sequential fits | 595.0 s | 1.0x |
| experiment17's plan: residency, DataLoader kept | 74.0 s | 8.0x |
| this branch, loader only (still 10 sequential fits) | 34.9 s | 17.1x |
| **this branch, fused ensemble** | **9.7 s** | **61.1x** |

**7.6x beyond what residency alone would have delivered.**

### Its 113x is a data-path number, not a fit number

experiment17's headline was 113x (18.35 → 0.16 ms/sample) and the projection that
followed was "a 10-member retrain goes from 4.5 hours to roughly 3–5 minutes".
That benchmark ran a forward and backward through the head but no optimizer step,
no gradient clipping, no per-epoch validation pass over the 290-row dev set and no
AUROC — all of which the real loop does, and the validation pass alone was 58% of
an epoch in §1. Measured over a whole fit, the same change is 8–11x, not 113x. The
ratio is sound for what it measured; the extrapolation to wall-clock is roughly an
order of magnitude optimistic. Fusing is what actually closes most of that gap.

### Where the fix belongs

experiment17 proposed putting it in this repo — move the merged tensors to the GPU
in `_train_with_cached_base_activations` after `_release_model()`, and let
`ActivationDataset.__getitems__`'s existing `.to(self.device)` become a no-op.
That is the smallest possible change and it works, but it is unconditional, and
experiment17 had already identified the wall it runs into: at 11 MB/sample on
gemma-27b the activation set reaches ~16.6 GB by iteration 3 and ~26.5 GB by
iteration 4, past a 24 GB card. Its own note was that this would need "an OOM
guard falling back to CPU residency" — i.e. falling back to the 1.4 GB/s path that
started the investigation.

Putting the policy in tuberlens instead makes that fallback a real path rather
than a surrender: `staged` is 2.7x/4.0x where unconditional residency would be
1.0x, the choice is made per tensor against the actual budget, and every other
tuberlens caller (including `get_performances` during eval) inherits it.

### Not examined by experiment17

- **Scoring.** `EnsembleProbe._mean_proba` walked the activations once per member.
  That runs on every red-team submission and every eval split, not only at retrain
  time; fusing it is 4–5x (§3.4).
- **The free-memory trap.** `mem_get_info` excludes torch's cached-but-unused
  blocks, so a naive budget check takes the slow path on every fit after the first
  (§3.1). An unconditional `.to("cuda")` never meets this, which is why
  experiment17 would not have hit it — but any budgeted version does.
- **Per-epoch validation re-transfer** and the per-batch `.item()` sync (§3.2).

### Left unresolved

experiment17 measured fit cost scaling as `t_epoch ~ 2.3e-5 x N^1.97` — roughly
quadratic — and flagged that it could not explain it. Nothing here explains it
either: the transfer rates in §2 are flat across working sets from 2.5 to 5 GB, so
the data path is linear in bytes and the superlinearity must sit somewhere else,
most plausibly host-side gather locality on a working set growing into the tens of
GB. It was measured on a live run from log timings taken at different moments,
which is a noisy instrument, and it is moot under residency — so it was not chased.
