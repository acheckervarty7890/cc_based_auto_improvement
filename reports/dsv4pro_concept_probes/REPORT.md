# Concept probes trained on the dsv4pro synthetic cuts

_Generated 2026-08-24 01:49:34Z._

## What is being measured

Each probe is trained on its own concept's `data/<concept>_dsv4pro.jsonl` — ~50 balanced two-turn
conversations written by **deepseek v4 pro** to exhibit that concept's positive and
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
| `seq_ens10` / val=`dev` | 0.888 | 0.579 | 0.843 |
| `seq_ens10` / val=`split` | 0.898 | 0.588 | 0.839 |
| `single` / val=`dev` | 0.888 | 0.570 | 0.792 |
| `single` / val=`split` | 0.871 | 0.562 | 0.818 |

## hu_ha — AUROC per eval split

```
dataset             eval_ai_dilemmas  eval_ant_hh  eval_balanced_refusal  eval_daily_dilemmas   MEAN
config    val_mode                                                                                  
seq_ens10 dev                  0.994        0.686                  0.886                0.988  0.888
          split                0.993        0.708                  0.902                0.991  0.898
single    dev                  0.993        0.709                  0.868                0.984  0.888
          split                0.994        0.625                  0.893                0.974  0.871
```

## instructions — AUROC per eval split

```
dataset             anthropic_harmless_refusal  bbq_substitution  hc_context_drift  hc_contradiction  mm_substitution  oig_context_drift  oig_omission   MEAN
config    val_mode                                                                                                                                           
seq_ens10 dev                            0.788             0.644             0.519             0.525            0.516              0.486         0.574  0.579
          split                          0.766             0.635             0.536             0.548            0.509              0.537         0.585  0.588
single    dev                            0.842             0.605             0.502             0.500            0.468              0.511         0.561  0.570
          split                          0.723             0.578             0.500             0.499            0.548              0.550         0.537  0.562
```

## highstakes — AUROC per eval split

```
dataset             anthropic_hh_balanced  mt_balanced  mts_balanced  toolace_balanced   MEAN
config    val_mode                                                                           
seq_ens10 dev                       0.952        0.840         0.879             0.702  0.843
          split                     0.960        0.817         0.889             0.689  0.839
single    dev                       0.917        0.745         0.900             0.607  0.792
          split                     0.965        0.725         0.926             0.656  0.818
```

## Reproducing

```bash
.venv_claude/bin/python scripts/concept_probes.py --generator dsv4pro --phase all
```

`prepare` extracts every requested generator's conversations in ONE model load and
slices each (concept, validation mode) train/val activation cache out of that master,
addressing rows by conversation content. The dev blobs and the activation cache dir
are shared across generators and with `results_generalization/` — base blobs are keyed
on the training file's own hash and dev blobs on the dev files' bytes, so neither can
collide. After `prepare`, no fit loads a model.
