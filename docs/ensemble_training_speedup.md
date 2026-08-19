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
