# Generalization tests — probes trained on the general dataset

_Generated 2026-08-23 18:56:59Z. Regenerated automatically every 30 minutes while the run is in flight._

## What is being measured

Every probe below is trained on `data/general_llama70b_150_2.jsonl` — 150 two-turn
conversations labelled `general` (broad everyday exchanges) vs `not_general`
(narrow expert exchanges) — but carries a *concept's* class labels and is scored on
that concept's own eval splits. So each probe is fit on a distinction its eval set
does not measure. Model `google/gemma-3-27b-it`, layer 32, arch `linear_then_softmax`,
seed 42, full eval splits off precomputed activations.

| axis | values |
| --- | --- |
| data variant | `i_general_random` (50 rows from the `general` half, concept labels assigned at random, balanced 25/25) · `ii_notgeneral_random` (same from `not_general`) · `iii_general_pos` (all 150; general→positive) · `iii_general_neg` (all 150; inverted) |
| probe config | `single` · `seq_ens10` (10 members, `PROBE_FUSED_ENSEMBLE=0`, fit sequentially under the repo-pinned `ENSEMBLE_SEEDS`) |
| validation | `dev` (the concept's `dev_samples/` set) · `split` (0.2 of the training data via `stable_train_test_split`) |

16 probes per concept, 48 in total. The row selection and the random label
assignment are seeded per *variant*, not per concept, so all three concepts train on
the same conversations with the same positive/negative pattern.

## hu_ha — AUROC per eval split

```
dataset                                  eval_ai_dilemmas  eval_ant_hh  eval_balanced_refusal  eval_daily_dilemmas   MEAN
variant              config    val_mode                                                                                  
i_general_random     seq_ens10 dev                  0.555        0.481                  0.880                0.546  0.615
                               split                0.529        0.569                  0.802                0.485  0.596
                     single    dev                  0.583        0.467                  0.806                0.542  0.600
                               split                0.538        0.546                  0.827                0.499  0.602
ii_notgeneral_random seq_ens10 dev                  0.524        0.593                  0.803                0.456  0.594
                               split                0.530        0.604                  0.743                0.460  0.584
                     single    dev                  0.540        0.546                  0.797                0.489  0.593
                               split                0.556        0.612                  0.742                0.473  0.596
iii_general_neg      seq_ens10 dev                  0.520        0.493                  0.482                0.476  0.493
                               split                0.516        0.497                  0.494                0.480  0.497
                     single    dev                  0.513        0.492                  0.476                0.481  0.491
                               split                0.504        0.504                  0.484                0.478  0.492
iii_general_pos      seq_ens10 dev                  0.516        0.521                  0.710                0.503  0.563
                               split                0.508        0.539                  0.689                0.506  0.561
                     single    dev                  0.510        0.501                  0.657                0.506  0.543
                               split                0.511        0.496                  0.725                0.489  0.555
```

## instructions — AUROC per eval split

```
dataset                                  anthropic_harmless_refusal  bbq_substitution  hc_context_drift  hc_contradiction  mm_substitution  oig_context_drift  oig_omission   MEAN
variant              config    val_mode                                                                                                                                           
i_general_random     seq_ens10 dev                            0.865             0.377             0.475             0.459            0.421              0.622         0.503  0.532
                               split                          0.781             0.456             0.489             0.487            0.348              0.540         0.501  0.515
                     single    dev                            0.696             0.403             0.477             0.473            0.539              0.599         0.491  0.525
                               split                          0.793             0.449             0.488             0.480            0.326              0.553         0.485  0.511
ii_notgeneral_random seq_ens10 dev                            0.814             0.534             0.490             0.472            0.425              0.520         0.542  0.542
                               split                          0.772             0.556             0.494             0.487            0.445              0.509         0.529  0.542
                     single    dev                            0.777             0.526             0.493             0.475            0.454              0.536         0.513  0.539
                               split                          0.744             0.484             0.492             0.482            0.463              0.512         0.515  0.528
iii_general_neg      seq_ens10 dev                            0.531             0.506             0.507             0.516            0.529              0.497         0.490  0.511
                               split                          0.542             0.499             0.502             0.508            0.531              0.498         0.496  0.511
                     single    dev                            0.521             0.491             0.500             0.511            0.499              0.498         0.483  0.501
                               split                          0.524             0.510             0.500             0.511            0.509              0.509         0.493  0.508
iii_general_pos      seq_ens10 dev                            0.625             0.522             0.494             0.483            0.495              0.517         0.547  0.526
                               split                          0.609             0.514             0.497             0.487            0.496              0.503         0.547  0.522
                     single    dev                            0.598             0.513             0.496             0.486            0.504              0.511         0.537  0.521
                               split                          0.654             0.529             0.494             0.479            0.425              0.497         0.521  0.514
```

## highstakes — AUROC per eval split

```
dataset                                  anthropic_hh_balanced  mt_balanced  mts_balanced  toolace_balanced   MEAN
variant              config    val_mode                                                                           
i_general_random     seq_ens10 dev                       0.435        0.590         0.608             0.459  0.523
                               split                     0.460        0.635         0.640             0.415  0.537
                     single    dev                       0.399        0.571         0.540             0.419  0.482
                               split                     0.431        0.653         0.698             0.387  0.542
ii_notgeneral_random seq_ens10 dev                       0.569        0.374         0.649             0.523  0.529
                               split                     0.575        0.377         0.736             0.552  0.560
                     single    dev                       0.528        0.410         0.621             0.513  0.518
                               split                     0.614        0.342         0.693             0.549  0.549
iii_general_neg      seq_ens10 dev                       0.458        0.720         0.911             0.668  0.689
                               split                     0.455        0.713         0.915             0.655  0.684
                     single    dev                       0.464        0.736         0.906             0.649  0.689
                               split                     0.463        0.705         0.916             0.648  0.683
iii_general_pos      seq_ens10 dev                       0.409        0.467         0.188             0.397  0.365
                               split                     0.423        0.538         0.177             0.398  0.384
                     single    dev                       0.414        0.469         0.195             0.341  0.355
                               split                     0.376        0.516         0.214             0.349  0.363
```

## Findings

### Mean AUROC by variant (averaged over config and validation mode)

| variant | hu_ha | instructions | highstakes |
| --- | --- | --- | --- |
| `i_general_random` | 0.603 | 0.521 | 0.521 |
| `ii_notgeneral_random` | 0.592 | 0.538 | 0.539 |
| `iii_general_pos` | 0.556 | 0.521 | **0.367** |
| `iii_general_neg` | 0.493 | 0.508 | **0.686** |

### 1. A trainer bug had to be fixed before any of this was readable

`linear_then_softmax` defaults to `batch_size: 16, gradient_accumulation_steps: 4`,
and the trainer steps only on `(batch_idx + 1) % accumulation == 0` with no
end-of-epoch flush (`pytorch_classifiers.py:299-327`). The 50-row variants under a
0.2 split leave 39 training rows = 3 batches, so `optimizer.step()` never fired and
the probe was returned at its initialization — two variants with different data *and*
different labels produced a bitwise-identical probe (`be87fde7dde9`), `best_epoch=1`,
loss pinned at ln 2, validation AUROC exactly 0.5. Fixed by capping accumulation at
the batch count: a no-op for every arm that already had >= 4 batches (verified
bit-identical), changing only the degenerate cells.

### 2. Generality transfers to high-stakes, and not to the other two concepts

This is the headline, and it is concept-dependent in a way worth taking seriously.
On **highstakes**, the real general/not_general distinction is far from chance in
both directions: `iii_general_neg` (i.e. `not_general` -> high-stakes) reaches
**0.686** mean AUROC and `iii_general_pos` falls to **0.367**. Per split,
`mts_balanced` hits 0.906-0.916 against 0.177-0.214 inverted. The direction is
interpretable: specialized/expert conversations — clinical dosing, tax code,
litigation procedure — read as high-stakes, and everyday ones as low-stakes.

On **hu_ha** and **instructions** the same training data lands at 0.49-0.56, i.e.
at or below the random-label controls. So this is not a general "any direction
transfers" effect; generality genuinely overlaps with stakes and genuinely does not
overlap with harm or instruction-following.

### 3. On highstakes the label inversion mirrors; elsewhere it cannot

highstakes `iii_general_neg` 0.686 vs `iii_general_pos` 0.367 sum to ~1.05, close to
the 1.0 an exact mirror would give — the fit is nearly symmetric under a label flip
when there is real signal to flip. On hu_ha and instructions both directions sit near
0.5, where mirroring carries no information.

### 4. The random-label controls are not at chance

Variants i and ii are 50 conversations of a single generality class with labels
assigned at random, so they carry no signal by construction — yet they score 0.52-0.60
mean AUROC, and on hu_ha (0.59-0.60) they beat the real distinction (0.556). A
direction fit to 50 arbitrary labels in 5376 dimensions is not a uniform-random
direction, and it lands somewhere mildly predictive. Treat ~0.55 as the effective
noise floor for this setup rather than 0.50 — which is exactly what makes the
highstakes 0.686/0.367 result meaningful and the hu_ha/instructions numbers not.

### 5. One eval split usually carries the signal, and it is often the refusal split

hu_ha's `eval_balanced_refusal` runs 0.74-0.88 while its three siblings sit at
0.46-0.61; instructions' `anthropic_harmless_refusal` runs 0.70-0.87 against six
siblings near 0.5. On highstakes the outlier is `mts_balanced` (0.91) — but that split
is only 86 rows, so its confidence interval is wide. Note highstakes'
`anthropic_hh_balanced`, the largest split at 2984 rows, stays near 0.46 in every arm.

### 6. Neither the ensemble nor the validation source changes much

The 10-member sequential ensemble moves the mean by <= 0.02 against a single probe
almost everywhere — expected when members fit the same activations. The dev-set vs
0.2-split validation choice is likewise mostly within 0.02; the largest gap is
highstakes `i_general_random`/`single` (dev 0.482 vs split 0.542), which is a
random-label arm and so is noise.

## Reproducing

```bash
.venv_claude/bin/python scripts/generalization_tests.py --concept hu_ha --phase all
```

`prepare` extracts the 150 conversations once into a master blob and slices each
variant's train/val activation cache out of it, so the 8 (variant × validation)
combinations do not each trigger their own gemma-3-27b load; the dev blobs are
assembled from the per-split Kaggle downloads. After `prepare`, no fit loads a model.
