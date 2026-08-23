# Generalization tests — probes trained on the general dataset

_Generated 2026-08-23 17:55:50Z. Regenerated automatically every 30 minutes while the run is in flight._

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
dataset                              anthropic_hh_balanced  mt_balanced  mts_balanced  toolace_balanced   MEAN
variant          config    val_mode                                                                           
i_general_random seq_ens10 dev                       0.435        0.590         0.608             0.459  0.523
                 single    dev                       0.399        0.571         0.540             0.419  0.482
                           split                     0.431        0.653         0.698             0.387  0.542
```

## Findings so far

**A trainer bug had to be fixed before any of this was readable.** `linear_then_softmax`
defaults to `batch_size: 16, gradient_accumulation_steps: 4`, and the trainer steps only
on `(batch_idx + 1) % accumulation == 0` with no end-of-epoch flush
(`pytorch_classifiers.py:299-327`). The 50-row variants under a 0.2 split leave 39
training rows = 3 batches, so `optimizer.step()` never fired and the probe was returned
at its initialization — two variants with different data *and* different labels produced
a bitwise-identical probe (`be87fde7dde9`), `best_epoch=1`, loss pinned at ln 2,
validation AUROC exactly 0.5. Fixed by capping accumulation at the batch count, which is
a no-op for every arm that already had ≥4 batches (verified bit-identical) and changes
only the degenerate cells.

**The random-label controls transfer as well as, or better than, the real signal.** On
hu_ha, variants i and ii average 0.58–0.62 mean AUROC while `iii_general_pos` — the
actual general/not_general distinction — averages 0.54–0.56. A probe fit to 50
conversations with meaningless labels transfers better than one fit to the real one.

**One split carries almost all of the signal, and it is the refusal split in both
concepts.** hu_ha's `eval_balanced_refusal` runs 0.74–0.88 while its three siblings sit
at 0.46–0.61; instructions' `anthropic_harmless_refusal` runs 0.70–0.87 while its six
siblings sit near 0.5. General-vs-specialized language plausibly correlates with
refusal-shaped language; nothing else here is far from chance.

**Label inversion is not a clean mirror.** `iii_general_neg` sits at ~0.49 throughout
while `iii_general_pos` is ~0.55. If inversion were exact these would reflect around
0.5, so the fits are not symmetric under a label flip.

**The 10-member sequential ensemble barely moves anything** (≤0.02 mean vs the single
probe), which is the expected result when members over the same activations agree.

## Reproducing

```bash
.venv_claude/bin/python scripts/generalization_tests.py --concept hu_ha --phase all
```

`prepare` extracts the 150 conversations once into a master blob and slices each
variant's train/val activation cache out of it, so the 8 (variant × validation)
combinations do not each trigger their own gemma-3-27b load; the dev blobs are
assembled from the per-split Kaggle downloads. After `prepare`, no fit loads a model.
