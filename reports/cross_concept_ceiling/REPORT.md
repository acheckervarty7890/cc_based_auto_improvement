# Cross-concept ceiling

_Generated 2026-08-24 03:09:42Z._

## What is being measured

**Ceiling** is the `ceiling_analysis` branch's definition: the best eval-set
performance this probe family (`linear_then_softmax` head on
`google/gemma-3-27b-it` layer 32) can reach *when trained on eval-distribution
data*, estimated by 5-fold cross-validation **inside the eval sets themselves** —
fit on the rows outside fold k, early-stop against a fixed reserved dev slice,
score fold k. Every eval row gets exactly one out-of-fold score, so a ceiling probe
differs from an ordinary probe only in its training data.

This run asks that of the three concepts **pooled**:

| arm | training pool |
| --- | --- |
| `within/<concept>` | CV inside one concept's own eval splits — the per-concept ceiling, and the baseline the cross numbers must be read against |
| `cross/native` | all three concepts at once, each row positive iff it is its own concept's positive class |
| `cross/aligned` | the same pool with `instructions` **inverted**, so all three positives are the *problematic* response |

`hu_ha` and `highstakes` call the problematic response positive (harmful,
high-stakes); `instructions` calls the desirable one positive
(`assistant_follows_the_instruction`). Pooling as-is therefore asks one direction to
put "harmful" and "correctly followed the instruction" on the same side, which is
not obviously the intended question — hence both orientations. The flip is not a
symmetry of the pooled task (it re-labels a third of the rows), so `aligned` is not
`1 - native`.

Each eval split is balanced-subsampled to 100 rows (this repo's own
`--eval-max-samples` default), giving 1500 pooled rows; the full 6576-row pool
padded to the longest split's 1024 tokens would be 72 GB of fp16 activations and
fits neither the box nor the card. Validation is a fixed 300-row balanced slice of
`dev_samples/`, 100 per concept, identical for every fold and every arm. Single
probes, never ensembles, seed 42. No model is loaded — every activation comes from
the precomputed blobs.

## Pooled AUROC (all rows of the arm's own pool)

```
arm   cross/aligned  cross/native  within/highstakes  within/hu_ha  within/instructions
name                                                                                   
ALL           0.937         0.889              0.963         0.977                0.946
```

## AUROC by concept

```
arm           cross/aligned  cross/native  within/highstakes  within/hu_ha  within/instructions
name                                                                                           
highstakes            0.913         0.900              0.963           NaN                  NaN
hu_ha                 0.961         0.930                NaN         0.977                  NaN
instructions          0.935         0.862                NaN           NaN                0.946
```

## AUROC by eval split

```
arm                                      cross/aligned  cross/native  within/highstakes  within/hu_ha  within/instructions
name                                                                                                                      
highstakes/anthropic_hh_balanced                 0.897         0.858              0.987           NaN                  NaN
highstakes/mt_balanced                           0.959         0.946              0.974           NaN                  NaN
highstakes/mts_balanced                          0.967         0.970              0.959           NaN                  NaN
highstakes/toolace_balanced                      0.809         0.835              0.928           NaN                  NaN
hu_ha/eval_ai_dilemmas                           1.000         0.983                NaN         1.000                  NaN
hu_ha/eval_ant_hh                                0.867         0.847                NaN         0.895                  NaN
hu_ha/eval_balanced_refusal                      0.923         0.920                NaN         0.991                  NaN
hu_ha/eval_daily_dilemmas                        1.000         0.943                NaN         0.999                  NaN
instructions/anthropic_harmless_refusal          0.985         0.946                NaN           NaN                1.000
instructions/bbq_substitution                    0.977         0.922                NaN           NaN                0.985
instructions/hc_context_drift                    0.973         0.994                NaN           NaN                1.000
instructions/hc_contradiction                    0.924         0.891                NaN           NaN                0.966
instructions/mm_substitution                     0.995         0.977                NaN           NaN                0.980
instructions/oig_context_drift                   0.982         0.875                NaN           NaN                0.946
instructions/oig_omission                        0.552         0.477                NaN           NaN                0.630
```

## Accuracy and TPR@1%FPR by concept

```
                  accuracy                                                                    tpr_at_fpr                                                                
arm          cross/aligned cross/native within/highstakes within/hu_ha within/instructions cross/aligned cross/native within/highstakes within/hu_ha within/instructions
name                                                                                                                                                                    
highstakes           0.839        0.829             0.917          NaN                 NaN           0.0          0.0             0.508          NaN                 NaN
hu_ha                0.892        0.860               NaN         0.93                 NaN           0.0          0.0               NaN        0.675                 NaN
instructions         0.876        0.771               NaN          NaN               0.869           0.0          0.0               NaN          NaN               0.574
```

## Reproducing

```bash
.venv_claude/bin/python scripts/cross_concept_ceiling.py
```
