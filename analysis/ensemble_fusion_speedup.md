# Ensemble probe fits: verifying `ensemble_speeduptest`, and what landed on `main`

Session of 2026-08-20, on the project box (RTX 3090 24 GB, 32 vCPU, 62 GB RAM, WSL2).

Two questions, in order:

1. The `ensemble_speeduptest` branch (this repo `fce08bf`, tuberlens `deb45d8`) claims a
   61x speedup on ensemble probe fitting. Does the fused ensemble really help?
2. If it does, land it on `main` in a form that holds in any configuration rather than
   only the one it was measured in.

Both were answered against **real data from this box** — no synthetic activations.

---

## 1. Method

Every training measurement uses the actual fit inputs of a real run, rebuilt from that
run's own caches, so no LLM is loaded and nothing is invented:

| case | train | validation | activations |
| --- | --- | --- | --- |
| `instructions` iteration 1 | 50 base + 178 red-team = 228 | 436-row dev set | 2.9 GiB |
| `instructions` iteration 5 | 50 base + 762 red-team = 812 | 436-row dev set | 8.8 GiB |
| `highstakes` iteration 3 | 50 base + 672 red-team = 722 | 1908-row dev set | **27 GiB** |

Sources: `base_acts_*_train.pt`, the per-conversation `redteam_acts_*` blobs keyed by
`_redteam_activation_cache_path`, and `dev_acts_*.pt`. The iteration-1 rebuild reproduces
the production log line exactly — *"Train/validation: 228 train, 436 validation"* and
*"Staged 2.9 GiB of activations on cuda"* — which is the check that the harness is fitting
what the run fitted. All rows were cache hits; a miss aborts rather than silently forwarding
through gemma-3-27b.

Probe: `linear_then_softmax`, gemma-3-27b layer 32 (hidden 5376), 10 members under the
repo-pinned `ENSEMBLE_SEEDS`, shipped hyperparameters (200 epochs, patience 50, batch 16)
unless a row says otherwise.

Arms differ **only** in the code under test — which tuberlens is on `PYTHONPATH`, which
checkout supplies `agentic_redteam`. The `main` arm imports `main`'s own
`_to_device_for_fit` rather than reimplementing its staging policy. Timing brackets the
fit alone, with `cuda.synchronize()` on both sides. Every run records per-member SHA-256
weight fingerprints, best epochs and dev AUROC, so equivalence is checked rather than
assumed.

### A measurement-integrity note, because it changed a headline

The first GPU arm of the session measured `main` at **68.9 s**. Re-measured under warm
conditions at the end of the session, the identical code and identical weights came in at
**43.8 s**. The first number was a cold-start artifact (first CUDA context, cold page cache
over a freshly written 2.9 GiB tensor) and inflated every speedup computed against it by
~1.6x.

All ratios below use the **43.8 s** control. The lesson is cheap and worth keeping: a
benchmark's first arm is not a baseline, and a run-to-run control at the end costs one
measurement.

---

## 2. Does the fused ensemble help? Yes — 2.5x to 4x, not 61x

Wall-clock for the whole 10-member fit. "main today" is `origin/main` with
`_to_device_for_fit`; "branch" is `ensemble_speeduptest`.

| case | main today | branch, per-member | branch, fused | fused vs main |
| --- | ---: | ---: | ---: | ---: |
| iteration 1, production settings | 43.8 s | 20.9 s | **18.1 s** | 2.4x |
| iteration 5, production settings | 101.7 s | 60.8 s | **29.0 s** | 3.5x |
| high-stakes, 10 x 3 epochs | 492.0 s | 360.3 s | **34.8 s** | 14.1x |
| iteration 1, fixed 25 epochs | 14.3 s | 12.7 s | **3.8 s** | 3.7x |

Scoring, the real 10-member `probe_iter5.pkl` over the run's own cached eval blobs
(194 rows), per-member loop vs one vmapped pass:

| split | members | loop | fused | speedup | max abs delta | AUROC loop -> fused | label flips |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| hc_context_drift | 5 | 0.800 s | 0.100 s | 8.0x | 3.9e-3 | 0.9650 -> 0.9652 | 0 |
| hc_context_drift | 10 | 1.638 s | 0.149 s | 11.0x | 3.7e-3 | 0.9713 -> 0.9714 | 0 |
| oig_context_drift | 5 | 0.712 s | 0.135 s | 5.3x | 4.7e-3 | 0.5821 -> 0.5823 | 0 |
| oig_context_drift | 10 | 1.537 s | 0.111 s | 13.9x | 3.1e-3 | 0.5869 -> 0.5865 | 0 |

### Why 61x does not reproduce

The branch's headline compares against **original tuberlens with host-resident
activations**, which `main` stopped doing in `_to_device_for_fit` (`6a928c8`, 2026-08-19
12:16 UTC — about an hour after the branch's write-up was committed; concurrent work, not
an error). That path measures 3.57 s/epoch here (10 members x 5 epochs = 178.5 s), so the
iteration-1 fit's 778 epochs would take ~46 min against `main`'s 43.8 s. Most of the
branch's claimed factor is therefore a fix `main` already has by another route.

The rest is the box, and it cuts both ways. The branch's own table puts
residency-with-DataLoader at 8.0x over original and its loader change at 17.1x — a 2.1x
gain for the loader work. Here that gain is **2.1x at production settings** (43.8 s ->
20.9 s) but only **1.1x at fixed 25 epochs** (14.3 s -> 12.7 s), because the branch's
batcher re-stages the activations once per member and needs many epochs to amortise an
8.5 s fixed cost. At 5 epochs it is a net loss: 9.3 s against `main`'s 3.5 s.

**Fusing is the part that survives**, because it is not a data-path trick: ten members
share one walk over the activations per epoch instead of taking ten. That is why its
advantage grows with the validation set — 2.4x at iteration 1, 14x on the high-stakes shape
whose 19.6 GiB dev set does not fit the card.

### Two effects that shape the win, both worth knowing

**Early stopping eats most of it.** A fused group runs until its *slowest* member stops.
At iteration 1 that is 2000 member-epochs fused against 778 sequential — 2.6x more
arithmetic — which is why a ~3x per-member-epoch advantage nets out at 2.4x. With fixed
epochs, where nothing stops early, the same shapes give 4.4x.

**The branch's spill-case win is partly an artifact of this box.** Its 14x came with a peak
of **39 GiB allocated on a 24 GB card**: its residency policy staged the 19.6 GiB dev set
anyway, which only worked because WSL2 pages GPU memory to host RAM. `main`'s capacity
check correctly declined the same tensor ("*19.6 GiB left on the host — it does not fit in
the 15.4 GiB allocatable*"). On a native Linux box that path OOMs into its staged fallback.

---

## 3. Correctness findings

### The loader change is bit-identical — verified, on real data

Host-resident, `main`-staged and batcher-resident fits produced **byte-identical weights
for all 10 members** at 5 epochs, 25 epochs, and at both iteration-1 and iteration-5
shapes. The RNG-consumption detail the branch documents (a shuffled DataLoader iteration
burns two int64s, an unshuffled one burns one) is real and its reproduction works.

### …except that the shipped `PROBE_EVAL_BATCH_SIZE=64` is not

In the 200-epoch production run, 9 of 10 members were bit-identical and one was not.
Member 6 peaked at 0.72948 (epoch 109) under the original eval batch of 16 and at 0.72949
(epoch 66) under 64. Re-running with `PROBE_EVAL_BATCH_SIZE=16` restored all ten
fingerprints exactly.

A larger no-grad batch changes the matmul reduction order, moving a validation AUROC in its
last bits; on a near-tie that flips which epoch is selected. The batcher is bit-identical;
the eval-batch change shipped alongside it is not, and the branch's write-up does not
separate them.

### The fused trainer silently repairs a checkpoint bug — so it is not "equivalent"

`PytorchAdamClassifier.train` records its best checkpoint as
`self.model.state_dict().copy()` — a *shallow* dict copy whose values are the live
parameter tensors the optimizer keeps updating in place. The closing `load_state_dict`
therefore restores the weights the fit **ended** on, not the ones it scored best with. At
the default `patience: 50` that is a probe fifty epochs past its selected checkpoint, and
`best_epoch` records an epoch whose weights were never kept.

Confirmed on all ten members of the real retrain — the returned AUROC equals the last
epoch's, never the recorded best:

| member | best epoch | max AUROC | last epoch's | returned |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 9 | 0.74118 | 0.71378 | 0.71378 |
| 2 | 12 | 0.79754 | 0.75386 | 0.75386 |
| 9 | 13 | 0.73470 | 0.69103 | 0.69103 |

The branch's `_stack_and_train_ensemble_state` clones instead, so its fused members really
do carry best-epoch weights — which is why they beat sequential on every member and lift
the ensemble's dev AUROC from 0.740 to 0.774. But that is the set early stopping selects
on. On the seven held-out eval splits the two ensembles are a wash, and are not small
variations of each other:

| split | sequential | fused | delta |
| --- | ---: | ---: | ---: |
| anthropic_harmless_refusal | 0.7938 | 0.5502 | **-0.244** |
| bbq_substitution | 0.9297 | 0.9515 | +0.022 |
| hc_context_drift | 0.7333 | 0.7584 | +0.025 |
| hc_contradiction | 0.9306 | 0.9137 | -0.017 |
| mm_substitution | 0.9480 | 0.9680 | +0.020 |
| oig_context_drift | 0.6531 | 0.7745 | +0.121 |
| oig_omission | 0.7473 | 0.7756 | +0.028 |
| **mean** | **0.8194** | **0.8131** | -0.006 |

Better on five splits, catastrophically worse on one, level on average. The fused path is
therefore a *different* probe, not a faster one and not a better one. The bug is worth
fixing; fixing it silently, on one of two code paths, inside a branch whose stated contract
is "a speedup, not a change of result", is the problem.

---

## 4. What landed on `main`

`main` `47ead49` (this repo) and `d02958a` (tuberlens `iterative_pipeline_2`, which is the
live editable checkout). The port keeps the mechanism and drops the coupling.

**The design difference.** The branch welds fusion to its own residency policy: an
`ActivationBatcher` that decides where activations live and stages a private copy *per
member*. That is why it is slower than `main` on short fits (9.3 s vs 3.5 s at 5 epochs —
it re-stages 2.9 GiB ten times) and why it reached 39 GiB on a 24 GB card. The new
`tuberlens/probes/fused_ensemble.py` takes no position on placement: it reads the tensors
wherever `_to_device_for_fit` left them and moves each batch exactly as `ActivationDataset`
does. A caller that has staged them pays no second copy; one that has not pays the transfer
it would have paid anyway; a CPU-only box works unchanged.

**Everything degrades instead of failing.** A closed-form architecture (sklearn / LDA /
difference-of-means, which a seed does not diversify anyway), a single member, a torch
without `torch.func`, `PROBE_FUSED_ENSEMBLE=0`, or an OOM — which halves the group size
first and only then gives up — all fall back to the per-member loop, which is slower and
produces the same thing. Members return as ordinary independent probes, so the pickle,
`EnsembleProbe` and every consumer are unchanged, and the call is reached through `getattr`
so a tuberlens predating it still trains ensembles the old way.

**Defaults preserve today's behaviour, exactly.** Verified on the real 10-member retrain:

| arm | fit | ensemble dev AUROC | weights identical to old `main` |
| --- | ---: | ---: | --- |
| old `main` (control, warm) | 43.8 s | 0.74002 | — |
| new, sequential | 42.4 s | 0.74002 | **yes, all 10 members** |
| new, `PROBE_FUSED_ENSEMBLE=0` | 42.1 s | 0.74002 | **yes, all 10 members** |
| new, fused | **17.8 s** | 0.74299 | no (bf16 reassociation, no systematic shift) |
| new, fused, groups of 4 | 16.4 s | 0.74299 | no (bit-identical to fused-all) |
| new, sequential + best-checkpoint | 46.3 s | 0.77207 | no (opt-in behaviour change) |
| new, fused + best-checkpoint | 25.3 s | 0.77484 | no (opt-in behaviour change) |

The last two rows are the point of the checkpoint handling: **both** trainers read
`PROBE_RESTORE_BEST_CHECKPOINT`, so the fused and per-member paths can never disagree about
what an early stop returns. The shallow copy is now a real clone, but restoring it is
opt-in — every probe in flight was trained the other way, and per §3 it is not a free win.

Also on the new code: the high-stakes spill case at 122.7 s against `main`'s 492.0 s
(4.0x), and fused scoring at 17.5x (11.1 s -> 0.63 s on a real eval split, AUROC unchanged
to the 4th decimal, no prediction flipped).

**Settings**, all defaulting to current behaviour:

- `PROBE_FUSED_ENSEMBLE` (default on) — turning it off can only cost speed.
- `PROBE_FUSED_MAX_MEMBERS` (default 0 = all). Splitting is bit-identical, so this is purely
  memory and scheduling. A fused step's `(members, batch, seq, embed)` gather cost 1.2 GiB
  of peak at 10 members against 0.34 GiB at 4; and since a group runs to its slowest member,
  smaller groups waste fewer epochs under early stopping (16.4 s vs 17.8 s) while costing
  more when nothing stops early (3.5 s vs 2.8 s). In the spill case grouping *hurts* —
  each group re-walks the spilled validation set — so one group is the right default.
- `PROBE_EVAL_BATCH_SIZE` (default 0 = training batch size) — see §3 for why it is off.
- `PROBE_PINNED_STAGING` (default on) — copies host-resident activations out through a
  page-locked buffer. Worth **6%** on a spilled validation set (122.7 s -> 115.7 s),
  bit-identical; far less than pinning a *random gather* buys, because these reads are
  already contiguous slices and the driver's own bounce does the same single copy.
- `PROBE_RESTORE_BEST_CHECKPOINT` (default off) — §3.

One repo-side change came out of the measurements: `_to_device_for_fit` staged activations
in fp16 while the fit runs in bf16, so every batch of every epoch of every member paid a
cast. Staging in the fit dtype costs nothing (both are 2 bytes) and is elementwise, hence
bit-identical. Worth ~3% sequentially and **1.4x fused** (24.7 s -> 17.8 s), since a fused
step moves ten members' worth of rows and pays the cast ten times over.

---

## 5. Open items

- **The checkpoint decision is unresolved, deliberately.** `PROBE_RESTORE_BEST_CHECKPOINT`
  defaults to off so runs in flight stay comparable. Whether to flip it needs a real
  arm-vs-arm comparison, not the single retrain in §3 — the eval-split evidence there is a
  wash with one large regression, which is a reason to measure properly rather than to
  choose either way.
- **`anthropic_harmless_refusal` collapsing 0.79 -> 0.55** under best-checkpoint selection
  is unexplained and is the single most interesting number in this session. It may be that
  the dev set's refusal rows and that split disagree, in which case early stopping on the
  dev set actively hurts that split.
- **Fusing does not help a single-probe run at all**, and most configs are `ensemble_size:
  1`. The wins here are for `ens10dev`-style runs.
- The branch's own `docs/ensemble_training_speedup.md` remains the more detailed account of
  the *data path*; its numbers are sound for what they measured (a 4-vCPU 8 GB box against
  original tuberlens) and do not transfer to this box or to today's `main`.
