# Cross-concept ceiling

_Generated 2026-08-23 23:49:24Z._

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
arm   within/highstakes  within/hu_ha  within/instructions
name                                                      
ALL               0.963         0.977                0.946
```

## AUROC by concept

```
arm           within/highstakes  within/hu_ha  within/instructions
name                                                              
highstakes                0.963           NaN                  NaN
hu_ha                       NaN         0.977                  NaN
instructions                NaN           NaN                0.946
```

## AUROC by eval split

```
arm                                      within/highstakes  within/hu_ha  within/instructions
name                                                                                         
highstakes/anthropic_hh_balanced                     0.987           NaN                  NaN
highstakes/mt_balanced                               0.974           NaN                  NaN
highstakes/mts_balanced                              0.959           NaN                  NaN
highstakes/toolace_balanced                          0.928           NaN                  NaN
hu_ha/eval_ai_dilemmas                                 NaN         1.000                  NaN
hu_ha/eval_ant_hh                                      NaN         0.895                  NaN
hu_ha/eval_balanced_refusal                            NaN         0.991                  NaN
hu_ha/eval_daily_dilemmas                              NaN         0.999                  NaN
instructions/anthropic_harmless_refusal                NaN           NaN                1.000
instructions/bbq_substitution                          NaN           NaN                0.985
instructions/hc_context_drift                          NaN           NaN                1.000
instructions/hc_contradiction                          NaN           NaN                0.966
instructions/mm_substitution                           NaN           NaN                0.980
instructions/oig_context_drift                         NaN           NaN                0.946
instructions/oig_omission                              NaN           NaN                0.630
```

## Accuracy and TPR@1%FPR by concept

```
                      accuracy                                         tpr_at_fpr                                 
arm          within/highstakes within/hu_ha within/instructions within/highstakes within/hu_ha within/instructions
name                                                                                                              
highstakes               0.917          NaN                 NaN             0.508          NaN                 NaN
hu_ha                      NaN         0.93                 NaN               NaN        0.675                 NaN
instructions               NaN          NaN               0.869               NaN          NaN               0.574
```

## Reproducing

```bash
.venv_claude/bin/python scripts/cross_concept_ceiling.py
```
