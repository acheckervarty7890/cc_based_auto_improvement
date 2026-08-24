# Concept probes trained on the llama70b synthetic cuts

_Generated 2026-08-24 04:20:03Z._

## What is being measured

Each probe is trained on its own concept's `data/<concept>_llama70b_50.jsonl` — ~50 balanced two-turn
conversations written by **meta-llama Llama-3.3-70B** to exhibit that concept's positive and
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
| `seq_ens10` / val=`dev` | 0.846 | 0.771 | 0.918 |
| `seq_ens10` / val=`split` | 0.800 | 0.813 | 0.877 |
| `single` / val=`dev` | 0.852 | 0.778 | 0.900 |
| `single` / val=`split` | 0.855 | 0.825 | 0.895 |

## hu_ha — AUROC per eval split

```
dataset             eval_ai_dilemmas  eval_ant_hh  eval_balanced_refusal  eval_daily_dilemmas   MEAN
config    val_mode                                                                                  
seq_ens10 dev                  0.843        0.739                  0.878                0.923  0.846
          split                0.680        0.734                  0.881                0.905  0.800
single    dev                  0.861        0.737                  0.841                0.969  0.852
          split                0.836        0.719                  0.910                0.955  0.855
```

## instructions — AUROC per eval split

```
dataset             anthropic_harmless_refusal  bbq_substitution  hc_context_drift  hc_contradiction  mm_substitution  oig_context_drift  oig_omission   MEAN
config    val_mode                                                                                                                                           
seq_ens10 dev                            0.348             0.899             0.767             0.908            0.936              0.746         0.797  0.771
          split                          0.565             0.915             0.820             0.912            0.946              0.745         0.790  0.813
single    dev                            0.534             0.863             0.714             0.909            0.934              0.714         0.776  0.778
          split                          0.773             0.933             0.798             0.870            0.911              0.766         0.724  0.825
```

## highstakes — AUROC per eval split

```
dataset             anthropic_hh_balanced  mt_balanced  mts_balanced  toolace_balanced   MEAN
config    val_mode                                                                           
seq_ens10 dev                       0.958        0.893         0.954             0.868  0.918
          split                     0.961        0.778         0.899             0.869  0.877
single    dev                       0.947        0.854         0.942             0.856  0.900
          split                     0.957        0.816         0.944             0.861  0.895
```

## Findings

See `reports/concept_probes_summary.md` for the three-generator comparison and
`reports/cross_concept_ceiling/REPORT.md` for the ceiling these numbers should be read
against (hu_ha 0.977, highstakes 0.963, instructions 0.946).

llama-3.3-70B is the **strongest generator for highstakes (0.897) and instructions
(0.797)** and the weakest of the three for hu_ha (0.838). Its instructions cut is the
only one of the nine (generator x concept) cells that gets an instructions probe near its
ceiling — 0.797 against 0.946, where llama8b reaches 0.670 and dsv4pro 0.575.

Within this generator the four arms are tightly clustered (0.800–0.918); as everywhere in
this experiment, the choice of ensemble and of validation source moves the mean far less
than the choice of training data does.

## Reproducing

```bash
.venv_claude/bin/python scripts/concept_probes.py --generator llama70b --phase all
```

`prepare` extracts every requested generator's conversations in ONE model load and
slices each (concept, validation mode) train/val activation cache out of that master,
addressing rows by conversation content. The dev blobs and the activation cache dir
are shared across generators and with `results_generalization/` — base blobs are keyed
on the training file's own hash and dev blobs on the dev files' bytes, so neither can
collide. After `prepare`, no fit loads a model.
