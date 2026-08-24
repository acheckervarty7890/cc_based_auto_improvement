# Concept probes trained on the llama8b synthetic cuts

_Generated 2026-08-24 03:49:44Z._

## What is being measured

Each probe is trained on its own concept's `data/<concept>_llama8b.jsonl` — ~50 balanced two-turn
conversations written by **meta-llama Llama-3.1-8B** to exhibit that concept's positive and
negative class — and scored on that concept's real eval splits at full size. So this
is a transfer question: how much of a concept does a 50-row synthetic cut teach a
probe? Model `google/gemma-3-27b-it`, layer 32, arch `linear_then_softmax`, seed 42,
activations precomputed.

| axis | values |
| --- | --- |
| probe config | `single` · `seq_ens10` (10 members, `PROBE_FUSED_ENSEMBLE=0`, fit sequentially under the repo-pinned `ENSEMBLE_SEEDS`) |
| validation | `dev` (the concept's `dev_samples/` set; all ~50 rows train) · `split` (0.2 of the training data via `stable_train_test_split`, leaving ~40 to train) |

4 probes per concept, 12 in total.

## Mean AUROC across eval splits

| arm | hu_ha | instructions | highstakes |
| --- | --- | --- | --- |
| `seq_ens10` / val=`dev` | 0.866 | 0.649 | 0.862 |
| `seq_ens10` / val=`split` | 0.855 | 0.719 | 0.841 |
| `single` / val=`dev` | 0.851 | 0.633 | 0.874 |
| `single` / val=`split` | 0.871 | 0.678 | 0.828 |

## hu_ha — AUROC per eval split

```
dataset             eval_ai_dilemmas  eval_ant_hh  eval_balanced_refusal  eval_daily_dilemmas   MEAN
config    val_mode                                                                                  
seq_ens10 dev                  0.938        0.726                  0.833                0.966  0.866
          split                0.912        0.755                  0.791                0.961  0.855
single    dev                  0.911        0.689                  0.831                0.970  0.851
          split                0.956        0.735                  0.820                0.975  0.871
```

## instructions — AUROC per eval split

```
dataset             anthropic_harmless_refusal  bbq_substitution  hc_context_drift  hc_contradiction  mm_substitution  oig_context_drift  oig_omission   MEAN
config    val_mode                                                                                                                                           
seq_ens10 dev                            0.738             0.709             0.572             0.694            0.668              0.617         0.546  0.649
          split                          0.721             0.789             0.616             0.875            0.809              0.638         0.588  0.719
single    dev                            0.782             0.735             0.541             0.531            0.662              0.627         0.549  0.633
          split                          0.737             0.867             0.620             0.611            0.717              0.629         0.567  0.678
```

## highstakes — AUROC per eval split

```
dataset             anthropic_hh_balanced  mt_balanced  mts_balanced  toolace_balanced   MEAN
config    val_mode                                                                           
seq_ens10 dev                       0.907        0.833         0.891             0.817  0.862
          split                     0.914        0.807         0.832             0.812  0.841
single    dev                       0.823        0.896         0.966             0.810  0.874
          split                     0.862        0.818         0.819             0.814  0.828
```

## Findings

### Mean AUROC by arm

| arm | hu_ha | highstakes | instructions |
| --- | --- | --- | --- |
| `single` / val=`dev` | 0.851 | **0.874** | 0.633 |
| `single` / val=`split` | **0.871** | 0.828 | 0.678 |
| `seq_ens10` / val=`dev` | 0.866 | 0.862 | 0.649 |
| `seq_ens10` / val=`split` | 0.855 | 0.841 | **0.719** |
| **concept mean** | **0.861** | **0.851** | **0.670** |

### 1. Fifty synthetic rows are enough for harm and stakes, and not for instruction-following

hu_ha (0.861) and highstakes (0.851) land within 0.01 of each other; instructions is
0.18 behind at 0.670. The gap is not an artifact of split count (7 vs 4) — it shows up
split by split. On highstakes, `anthropic_hh_balanced` (2984 rows, the largest split in
the project) runs 0.82–0.91; the same split never left 0.46 in the generalization
experiment, so this is the llama-8b cut teaching the actual concept rather than one
small split carrying a mean.

On instructions the weak splits are the ones whose negative class is a *drift* or an
*omission* — `oig_context_drift`, `oig_omission`, `hc_context_drift` sit at 0.54–0.64 in
every arm — while `bbq_substitution` (0.71–0.87) and `anthropic_harmless_refusal`
(0.72–0.78) do much better. A 50-row synthetic cut can show what a wrong answer or a
refusal looks like; it apparently does not show what quietly failing to use the provided
source looks like.

### 2. The 40-row `split` arms train correctly only because the accumulation is capped

`linear_then_softmax` defaults to `batch_size: 16, gradient_accumulation_steps: 4`, and
the trainer steps only on `(batch_idx + 1) % accumulation == 0` with no end-of-epoch
flush (`pytorch_classifiers.py:299-327`, unchanged since the tuberlens rename). Every
`split` arm here is 39–40 training rows = 3 batches/epoch, so uncapped, `optimizer.step()`
would never fire and all six probes would come back at their random initialization.

This is not hypothetical. `experiment16_cloud` ran the *same* shape — hu_harm,
gemma-3-27b L32, `linear_then_softmax`, `ensemble_size: 10`, base data
`data/hu_harm_llama70b_50.jsonl`, 0.2 `test_size` slice → 40 train / 10 val — and its
iteration-0 probe scored **0.336** mean AUROC, below chance on three of four splits
(0.180 / 0.232 / 0.453 / 0.477). `experiment17_cloud` is byte-identical except for
`validation.dev_data: dev_samples/hu_ha`, which puts all 50 rows in training = 4 batches,
and its iteration-0 probe scored **0.846**. The 0.51 gap between those two runs is the
bug, not the validation source. A systematically inverted probe is the signature of a
random direction; a merely weak probe sits at 0.5.

Capped, the equivalent arms here reach 0.678–0.871. Any run in this repo whose
iteration-0 training set was under 64 rows has the same defect.

### 3. `dev` vs `split` has no consistent winner, and on instructions `split` wins

| concept | dev | split | Δ |
| --- | --- | --- | --- |
| hu_ha | 0.858 | 0.863 | +0.005 |
| highstakes | 0.868 | 0.834 | −0.034 |
| instructions | 0.641 | 0.698 | +0.057 |

The two arms differ in two ways at once: the `dev` arm trains on ~10 more rows *and*
early-stops against hundreds of real conversations, while the `split` arm early-stops
against 10 synthetic ones (a set so small it saturates at AUROC 1.000 immediately, making
its stopping point close to arbitrary). The expectation is that `dev` wins. It does on
highstakes and it loses on instructions by more than it wins anywhere — which reads as the
llama-8b instructions cut and the real `dev_samples/instructions` set disagreeing about
the concept, so stopping on the real set stops at the wrong epoch for this training data.

### 4. The ensemble is a wash except in one cell

`seq_ens10` moves the mean by ≤ 0.02 against a single probe in five of six
(concept × validation) cells — expected when all ten members fit the same activations and
differ only in seed. The exception is instructions/`split` (0.678 → 0.719), which is also
the noisiest cell in the matrix (`hc_contradiction` alone swings 0.531 → 0.875 across
arms), so it is more plausibly variance than an ensemble effect.

## Reproducing

```bash
.venv_claude/bin/python scripts/concept_probes.py --generator llama8b --phase all
```

`prepare` extracts every requested generator's conversations in ONE model load and
slices each (concept, validation mode) train/val activation cache out of that master,
addressing rows by conversation content. The dev blobs and the activation cache dir
are shared across generators and with `results_generalization/` — base blobs are keyed
on the training file's own hash and dev blobs on the dev files' bytes, so neither can
collide. After `prepare`, no fit loads a model.
