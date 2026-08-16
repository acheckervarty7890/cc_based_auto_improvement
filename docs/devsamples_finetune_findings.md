# Mixing dev samples in vs. finetuning on them afterwards — run of 2026-08-16

The run of 2026-08-15 (`docs/devsamples_kfold_findings.md`) added N in-distribution dev
samples to the red-team training set and fitted **one** probe on the union. This run asks
whether the *order* matters: fit the probe on base + red-team first, then continue
training **that** probe on the dev rows alone.

- **`scripts/dev_sample_finetune.py`** (new) — the sequential arm. Stage 1 is bit-for-bit
  the mixed run's N=0 job; stage 2 calls `PytorchAdamClassifier.train(...,
  initialize_model=False)` on the dev rows.
- **`scripts/dev_sample_retrain.py`** (unchanged) — re-run on two additional weight-init
  seeds, because the effect being measured turned out to be the same size as this setup's
  seed noise.
- Driver: `run_devsamples_finetune.sh`. No model is loaded; everything runs off the
  activation caches the 2026-08-15 run left behind. 90 sequential probes + 20 new mixed
  jobs, ~2 h wall-clock.

Held identical between the two arms: attacker arms, the iteration-3 postprocessed dumps,
base data, activation caches, transforms, `stable_train_test_split` (so the *same* dev
rows are fitted and the same ones land in validation), and the weight-init seeds. The
only difference is when the dev gradient signal is applied.

## The answer

Eval AUROC, mean over the four `eval_dataset_hu_ha` splits, **3 seeds on both sides**.
Sequential column is its best variant (stage-2 validation = the mixed val set, lr ×1.0);
`delta` is sequential − mixed.

| N/split | rows fitted | deepseekv4pro mixed | sequential | Δ | gptoss120b mixed | sequential | Δ |
|---|---|---|---|---|---|---|---|
| 0  | 0  | 0.8837 ± 0.0254 | *(same probe)* | — | 0.9073 ± 0.0097 | *(same probe)* | — |
| 2  | 6  | 0.8807 ± 0.0219 | 0.8338 ± 0.0664 | **−0.047** | 0.9160 ± 0.0052 | 0.8722 ± 0.0254 | **−0.044** |
| 8  | 24 | 0.9154 ± 0.0035 | 0.8655 ± 0.0247 | **−0.050** | 0.9224 ± 0.0080 | 0.8918 ± 0.0084 | **−0.031** |
| 16 | 49 | 0.9196 ± 0.0043 | 0.9138 ± 0.0025 | −0.006 | 0.9184 ± 0.0049 | 0.9154 ± 0.0070 | −0.003 |
| 30 | 94 | 0.9292 ± 0.0009 | 0.9272 ± 0.0035 | −0.002 | 0.9250 ± 0.0086 | 0.9224 ± 0.0079 | −0.003 |
| *ceiling* | *~693* | **0.9750** | | | **0.9750** | | |

The ceiling row is the 5-fold CV of `scripts/eval_kfold_cv.py` (`pooled` geometry) — a probe
of this architecture trained on the eval distribution itself, so it is arm-independent and
is a bound, not a competitor. **It is the unweighted mean of the four per-split CV AUROCs,
not the 0.9766 headline** of `docs/devsamples_kfold_findings.md`: that figure pools all 866
held-out rows into one AUROC, and `eval_balanced_refusal` is 46% of them. Every other number
in this table is `evaluate_probe`'s `mean` row, which averages the four splits equally
(`tuberlens/evaluation.py:125`), so 0.9750 is the like-for-like value. The `within` geometry
gives 0.9718 by the same aggregation.

At N=30 both schedules are ~0.046–0.053 short of it, and the mixed-vs-sequential gap
(0.002–0.003) is ~5% of the distance still to go. **The order the dev samples arrive in is
a rounding error against what separates either schedule from a probe that has seen the
target distribution.**

**Mixing is better at every N and under every variant, and the margin is a function of how
much dev data there is.** At N=2 and N=8 the gap is 0.03–0.05, against per-cell seed sds of
0.004–0.025; the per-seed *paired* deltas agree in sign in 5 of 6 cases at N=2 and 6 of 6
at N=8, which is the comparison that matters since both arms share a seed. By N=16 it has fallen
to 0.003–0.006 and by N=30 to 0.002–0.003, which is inside the seed spread; the paired
deltas there change sign across seeds. So the two schedules converge once there is enough
in-distribution data, and the choice only matters in the regime where dev samples are
scarce — which is the regime anyone would actually be in.

Finetuning is not useless: it beats its own stage-1 starting point by +0.044
(deepseekv4pro) and +0.015 (gptoss120b) at N=30. It is simply dominated by mixing at
every point on the curve.

## Why sequential loses, per split

The 3-seed per-split AUROCs (deepseekv4pro, sequential mixed/×1.0 against mixed):

| split | mix N=0 | seq N=8 | mix N=8 | seq N=30 | mix N=30 | *ceiling* |
|---|---|---|---|---|---|---|
| eval_ai_dilemmas | 0.9220 | 0.9841 | 0.9981 | 0.9988 | 0.9990 | *0.9912* |
| eval_ant_hh | 0.7267 | 0.7107 | 0.7290 | **0.7791** | 0.7588 | *0.9347* |
| eval_balanced_refusal | 0.9015 | **0.7885** | 0.9472 | 0.9418 | 0.9685 | *0.9924* |
| eval_daily_dilemmas | 0.9846 | 0.9789 | 0.9872 | 0.9889 | 0.9905 | *0.9818* |

Against the ceiling column, three of the four splits are essentially solved by N=30 —
`eval_ai_dilemmas` and `eval_daily_dilemmas` are *above* their CV ceilings (both arms), and
`eval_balanced_refusal` is within 0.024 of it. **The entire 0.046 mean gap is
`eval_ant_hh`**, which sits 0.176 (mixed) / 0.156 (sequential) below a probe trained on its
own distribution and alone accounts for 96% of it. So "0.05 short of the ceiling" is not a diffuse shortfall spread over the
concept — it is one split, and it is the same split sequential training is the only thing
here that improves.

The damage is concentrated in `eval_balanced_refusal` — the split the stage-1 probe was
already good at — and it is worst at low N. Dev-only gradient steps have nothing holding
the earlier solution in place, so the probe moves toward the handful of dev rows and gives
up ground it already held. The mirror image shows up at N=30: sequential is *better* than
mixed on `eval_ant_hh` (0.7791 vs 0.7588 deepseekv4pro, 0.7925 vs 0.7615 gptoss120b), the
hard split both arms struggle with. So the trade is real and legible — dev-only finetuning
reallocates capacity toward the split the dev samples help most, at the cost of the splits
the red-team set was already covering. Mixing declines to make that trade and wins on the
mean.

`ft_best_epoch` corroborates it: under mixed validation the finetune early-stops at epoch
1–5 in 21 of 24 cells (the three exceptions are all at N=30, where there is enough dev data
to be worth more steps). Model selection sees the forgetting start immediately and takes
the earliest checkpoint it can.

## The two sequential knobs

Sequential training has two design choices the mixed design does not, so both were run as
a factorial rather than guessed. Neither rescues it (mean over all N>0 cells, both arms):

| stage-2 validation | lr × | AUROC |
|---|---|---|
| mixed val set | 1.0 | 0.8928 ± 0.0389 |
| mixed val set | 0.1 | 0.8984 ± 0.0223 |
| dev val rows only | 1.0 | 0.9060 ± 0.0239 |
| dev val rows only | 0.1 | 0.9020 ± 0.0158 |

- **What early stopping watches barely matters.** At matched (N, lr) the two agree to
  within 0.0001 at lr ×1.0 and 0.004 at ×0.1 — the finetune stops so early that both
  signals usually pick the same checkpoint. (`dev` validation does not exist at N=2: the deterministic
  split sends 2 rows there and both are positive, so no AUROC is defined. Those 6 jobs are
  recorded as skipped rather than quietly given a different validation set.)
- **A gentler lr trades where the damage lands.** ×0.1 is better at N=2 (it does less
  harm — gptoss120b 0.9120 vs 0.8722) and worse at N≥16 (it moves too little to collect
  the benefit). It never beats mixing.

## The larger finding: the 2026-08-15 curve was a single seed, and that was not enough

Stage 1 here is the *same probe* as the mixed run's N=0, so it is a direct measurement of
seed noise on the pipeline's own output:

| seed | deepseekv4pro | gptoss120b |
|---|---|---|
| 7 | 0.8611 | 0.9070 |
| 13 | 0.8788 | 0.9172 |
| 42 | **0.9112** | **0.8978** |
| mean ± sd | 0.8837 ± 0.0254 | 0.9073 ± 0.0097 |

Seed 42 — the only seed the previous run used — reproduces its published N=0 values
exactly (0.9112 / 0.8978), which is the check that the two scripts have not drifted. But
for deepseekv4pro it is the **top** of a 0.861–0.911 range. The previous write-up's
estimate of ~0.008 AUROC seed noise holds for gptoss120b and understates deepseekv4pro by
3×.

Re-running the mixed arm on all three seeds changes what its curve says:

| N | mixed, 1 seed (published) | mixed, 3 seeds |
|---|---|---|
| 0 | 0.9112 | 0.8837 ± 0.0254 |
| 2 | 0.8771 | 0.8807 ± 0.0219 |
| 8 | 0.9113 | 0.9154 ± 0.0035 |
| 16 | 0.9203 | 0.9196 ± 0.0043 |
| 30 | 0.9290 | 0.9292 ± 0.0009 |
| *ceiling* | *0.9750* | *0.9750* |

**The n=2 dip flagged as unresolved in the previous write-up is gone.** It was not a dip;
it was an inflated N=0 reference. The corrected curve rises from 0.884 to 0.929 and is
monotone from N=2 up. The direction of the earlier conclusion survives — dev samples help,
~0.93 at N=30, still ~0.046 short of the 0.9750 CV ceiling — but its shape at the
low end did not.

The sd column carries the other half of it: **dev samples do not only raise the mean, they
stabilise the probe.** deepseekv4pro's seed sd falls from 0.0254 at N=0 to 0.0009 at N=30,
a 28× reduction. A probe trained on red-team data alone is at the mercy of its
initialisation in a way that a probe with even 94 in-distribution rows is not.

`tpr_at_fpr` remains erratic across N under both schedules, as it did at one seed, so
three seeds do not rescue it either — whatever the dev samples buy, it is still not
operating-point behaviour at a fixed low FPR.

## What broke

**`gradient_accumulation_steps` silently made stage 2 a no-op.** `PytorchAdamClassifier.train`
steps the optimizer only when `(batch_idx + 1) % gradient_accumulation_steps == 0` and
zeroes the gradients at the top of every epoch. The pipeline's setting is 4, which is fine
for a ~750-row training set (47 batches/epoch, 11 steps) and fatal for a dev-only stage 2
of 6–94 rows (1–6 batches): at N=2 and N=8 the condition never fires and **the weights
never move**. The first smoke test returned a "finetuned" probe scoring 0.9112 — exactly
its stage-1 input, to four decimals.

This is the failure mode worth remembering: it does not raise, it produces a plausible
number, and that number reads as *"finetuning does nothing"* — the very hypothesis under
test. `_scaled_args` now clamps the accumulation steps to the number of batches the
stage-2 set actually produces, keeping the pipeline's effective batch size wherever the
data supports it and guaranteeing at least one step per epoch where it does not.

(Also: `PytorchAdamClassifier.wandb_project` binds `global_settings.WANDB_PROJECT` when the
class body runs, so mutating the setting later has no effect and an ambient
`WANDB_PROJECT` — even the empty string, which is not `None` and so still counts as
configured — makes every fit die on a missing API key. The script pops it before importing
tuberlens.)

## Reading list for a follow-up

- **Replay.** Sequential loses by forgetting; the standard fix is mixing a slice of the
  red-team set back into stage 2. That is a third schedule with its own ratio knob, and it
  is the obvious way to ask whether the N=8 gap is intrinsic to sequential training or
  just to *dev-only* sequential training.
- **`eval_ant_hh` is where the headroom is — all of it.** It sits at 0.73–0.79 under every
  schedule against a 0.9347 CV ceiling, and that one split accounts for 96% of the mean
  gap; the other three are at or above their ceilings by N=30. It is also the one split
  sequential training *improves*, and the only unpaired split (see `CLAUDE.md`), so part
  of the number is a property of the data. Any further work on this concept should be
  scored on `eval_ant_hh` alone — movement in the 4-split mean is now almost entirely
  movement in it.
- **Three seeds is the new floor, not a luxury.** Any future cell reported at one seed on
  the deepseekv4pro arm is reporting a draw from a ±0.025 distribution.

## Outputs

```
results/devsamples_kfold/finetune/finetune_all.csv        90 probes × 5 rows (4 splits + mean)
results/devsamples_kfold/finetune/finetune_summary.csv    grouped by arm/N/val_mode/lr
results/devsamples_kfold/finetune/finetune_vs_mixed.csv   the joined comparison
results/devsamples_kfold/finetune/probes/*.pkl,*.json     every fitted probe + its job metadata
results/devsamples_kfold/dev_samples/dev_sample_*.csv     the mixed arm, now at 3 seeds
```
