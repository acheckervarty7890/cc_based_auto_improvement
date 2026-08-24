# Concept probes trained on the nemotron550b synthetic cuts

_Generated 2026-08-24 10:13:38Z._

## What is being measured

Each probe is trained on its own concept's `data/<concept>_nemotron550b.jsonl` — ~50 balanced two-turn
conversations written by **nvidia Nemotron 550B** to exhibit that concept's positive and
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
| `seq_ens10` / val=`dev` | 0.875 | 0.586 | 0.852 |
| `seq_ens10` / val=`split` | 0.878 | 0.621 | 0.849 |
| `single` / val=`dev` | 0.875 | 0.574 | 0.835 |
| `single` / val=`split` | 0.876 | 0.574 | 0.811 |

## hu_ha — AUROC per eval split

```
dataset             eval_ai_dilemmas  eval_ant_hh  eval_balanced_refusal  eval_daily_dilemmas   MEAN
config    val_mode                                                                                  
seq_ens10 dev                  0.963        0.724                  0.829                0.983  0.875
          split                0.958        0.752                  0.820                0.981  0.878
single    dev                  0.967        0.758                  0.798                0.979  0.875
          split                0.973        0.723                  0.826                0.981  0.876
```

## instructions — AUROC per eval split

```
dataset             anthropic_harmless_refusal  bbq_substitution  hc_context_drift  hc_contradiction  mm_substitution  oig_context_drift  oig_omission   MEAN
config    val_mode                                                                                                                                           
seq_ens10 dev                            0.752             0.636             0.518             0.522            0.533              0.598         0.546  0.586
          split                          0.698             0.694             0.557             0.599            0.643              0.612         0.546  0.621
single    dev                            0.700             0.679             0.518             0.529            0.495              0.557         0.543  0.574
          split                          0.773             0.611             0.500             0.503            0.527              0.580         0.524  0.574
```

## highstakes — AUROC per eval split

```
dataset             anthropic_hh_balanced  mt_balanced  mts_balanced  toolace_balanced   MEAN
config    val_mode                                                                           
seq_ens10 dev                       0.800        0.934         0.930             0.743  0.852
          split                     0.801        0.914         0.929             0.751  0.849
single    dev                       0.770        0.927         0.897             0.745  0.835
          split                     0.740        0.886         0.899             0.720  0.811
```

## Findings

See `reports/concept_probes_summary.md` for the four-generator comparison and
`reports/cross_concept_ceiling/REPORT.md` for the ceiling these numbers should be read
against (hu_ha 0.977, highstakes 0.963, instructions 0.946).

nvidia Nemotron 550B is the **second-strongest generator for hu_ha (0.876)** and third of
four on both `highstakes` (0.837) and `instructions` (0.589). It is the fourth generator
run through this matrix, and its value is mostly confirmatory: it lands adjacent to
dsv4pro on all three concepts and leaves the cross-concept ranking inversion intact — with
four generators, the hu_ha ordering is now the exact reverse of the highstakes and
instructions orderings, which are identical to each other.

Being the largest model in the rotation by parameter count buys nothing here. It finishes
behind the 70B model on two of three concepts and behind the 8B model on two of three.

Two details specific to this cut:

**Its `highstakes` arms spread wider than any other cell in the experiment** — 0.041
between `seq_ens10/dev` (0.852) and `single/split` (0.811), with the 10-member ensemble
ahead in *both* validation modes (0.852/0.849 vs 0.835/0.811). Everywhere else the
ensemble-vs-single and dev-vs-split choices have been ≤0.03 and directionless. It is still
a small effect next to the 0.074 spread between generators on this concept, but it is the
one cell where the ensemble looks like it is doing something rather than nothing.

**Its `instructions` probes are close to chance on five of seven splits** —
`hc_context_drift` 0.518, `hc_contradiction` 0.522, `mm_substitution` 0.533,
`oig_omission` 0.546, `oig_context_drift` 0.598 in the `seq_ens10/dev` arm. Only
`anthropic_harmless_refusal` (0.70–0.77) and `bbq_substitution` (0.61–0.69) carry signal.
Against a 0.946 ceiling this cut teaches the probe very little of the concept.

Its `hu_ha` cell is the tightest in the whole experiment: 0.875–0.878 across all four
arms, a spread of 0.003. Per split it follows the usual shape — `daily_dilemmas` 0.98 and
`ai_dilemmas` 0.96 near-saturated, `ant_hh` 0.72–0.76 holding the mean down.

### Cost

2 h 30 m end to end: 352 s to extract the 150 conversations in one gemma-3-27b load, 51
min of fits, the rest eval. As with every other generator, `highstakes` is 93% of the fit
time (2859 s of 3076 s) and its `seq_ens10/dev` arm alone is 2467 s — ten sequential
members each scoring the 1908-row / 21 GB dev set every epoch, which does not fit the
24 GB card and so cannot be staged by `_to_device_for_fit`.

## Reproducing

```bash
.venv_claude/bin/python scripts/concept_probes.py --generator nemotron550b --phase all
```

`prepare` extracts every requested generator's conversations in ONE model load and
slices each (concept, validation mode) train/val activation cache out of that master,
addressing rows by conversation content. The dev blobs and the activation cache dir
are shared across generators and with `results_generalization/` — base blobs are keyed
on the training file's own hash and dev blobs on the dev files' bytes, so neither can
collide. After `prepare`, no fit loads a model.
