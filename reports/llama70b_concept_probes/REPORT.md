# Concept probes trained on the llama70b synthetic cuts

_Generated 2026-08-23 21:49:13Z._

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

| arm | hu_ha |
| --- | --- |
| `seq_ens10` / val=`dev` | 0.846 |
| `seq_ens10` / val=`split` | 0.800 |
| `single` / val=`dev` | 0.852 |
| `single` / val=`split` | 0.855 |

## hu_ha — AUROC per eval split

```
dataset             eval_ai_dilemmas  eval_ant_hh  eval_balanced_refusal  eval_daily_dilemmas   MEAN
config    val_mode                                                                                  
seq_ens10 dev                  0.843        0.739                  0.878                0.923  0.846
          split                0.680        0.734                  0.881                0.905  0.800
single    dev                  0.861        0.737                  0.841                0.969  0.852
          split                0.836        0.719                  0.910                0.955  0.855
```

## instructions

_In progress — 0/4 probes fitted, no eval yet._

## highstakes

_In progress — 2/4 probes fitted, no eval yet._

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
