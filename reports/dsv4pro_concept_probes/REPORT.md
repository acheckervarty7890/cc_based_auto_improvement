# Concept probes trained on the dsv4pro synthetic cuts

_Generated 2026-08-23 21:49:13Z._

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

_No eval results yet._

## hu_ha

_In progress — 0/4 probes fitted, no eval yet._

## instructions

_In progress — 0/4 probes fitted, no eval yet._

## highstakes

_In progress — 0/4 probes fitted, no eval yet._

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
