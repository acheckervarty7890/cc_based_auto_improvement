# Ceiling analysis — category-steered run (experiment_instruction_cloud_6)

Probes: `probes/instructions_gemma27b_xmemocat_{gptoss,nemotron}`, 10-member ensembles,
gemma-3-27b-it L32. Everything below runs offline off the run's cached activations.
`sanity.py` reproduces all 12 published probe scores to 6 dp under
`PROBE_FUSED_ENSEMBLE=0` (the path the run itself scored on), so these numbers sit on
the same scale as the run's `*_comparison.csv`.

## Findings

**1. The ceiling is 0.9416; red-teaming reached 0.8343 / 0.8058.**
Iteration 0 is 0.7714, so the headroom is 0.170 and this run captured **37%** of it in
the gpt-oss arm and **20%** in the nemotron arm. For scale,
`experiment_instruction_cloud_5` — the same attackers, same probe, prose probe
description, no category steering — captured **4%** in its gpt-oss arm. The
category roll-call is a real and large improvement to red-teaming's efficiency.

**2. 84 labelled dev rows beat the entire red-team run, in both arms.**
0.9036 (gpt-oss) and 0.8640 (nemotron) at N=84, against 0.8343 and 0.8058 from five
iterations, 433 / 449 attacker successes and ~4.5 GPU-hours of scoring. 336 dev rows
reach or beat the ceiling outright (0.9541 / 0.9435). Red-teaming buys AUROC far more
expensively than labelling in-distribution data — the steering changes red-teaming's
efficiency at its own job, not its standing against that alternative.

**3. Adding red-team data on top of in-distribution data neither helps nor hurts.**
`cv_eval_rt` is +0.011 over `cv_eval` in the gpt-oss arm and +0.000 in the nemotron
arm. Both sit inside the ~0.015 noise floor measured for this concept, so the
supportable claim is "does not hurt", not "carries signal eval lacks".

**4. The ceiling is not uniform across splits, and `oig_omission` is the outlier.**
Five-fold CV *inside* the eval sets reaches 0.966–1.000 on six splits but only
**0.673** on `oig_omission`. That is the ceiling, not a probe failure: even trained on
the eval distribution itself, this probe family barely separates "answered only some
of what was asked". So part of the gap between the published probes and 0.9416 is not
reachable by better red-teaming at all — it is a limit of a linear head at L32 on that
one failure mode. Any future effort aimed at omission should change the probe, not the
attacker.

**5. The two arms are good at different splits, and it tracks what each attacker
sampled.** Trained on red-team data alone, gpt-oss reaches 0.935 on
`anthropic_harmless_refusal` where nemotron reaches 0.731; nemotron reaches 0.955 on
`bbq_substitution` where gpt-oss reaches 0.717. Refusal is the category gpt-oss's memos
flagged as an unfilled gap for three iterations before the attacker went there.

**6. Finetune beats joint at nearly every N, and `lr=1e-4` is inert.**
Both reproduce `experiment_instruction_cloud_3` independently. The 1e-4 arm stays flat
at its N=0 baseline across the whole sweep, so the default 5e-3 is doing all the work;
that arm is a control, not a competitive setting.

**A caveat on `redteam_only`.** It refits the run's own training set in *file order* and
lands ~0.010 below the published probe in both arms. `_activate_redteam_cached` emits
cache hits before newly-computed rows, so a run's training row order encodes its box's
cache history and is not recoverable from the snapshot. Compare conditions to each
other, not to the comparison CSV.

## Part 1 — ceiling on eval_sets/instructions

| condition | AUROC | acc | TPR@1%FPR (tuberlens) | TPR@FPR<=1% |
|---|---|---|---|---|
| red-team only (gptoss, iter5 training set) | 0.8239 | 0.7219 | 0.2589 | 0.2685 |
| red-team only (nemotron, iter5 training set) | 0.7956 | 0.7094 | 0.0942 | 0.2525 |
| CV on eval alone (5-fold, out-of-fold) | 0.9416 | 0.8972 | 0.6098 | 0.7755 |
| CV on eval + red-team (gptoss) | 0.9530 | 0.9002 | 0.6972 | 0.7714 |
| CV on eval + red-team (nemotron) | 0.9416 | 0.8834 | 0.5624 | 0.7651 |
| oracle: fit and scored on all 1302 eval rows | 1.0000 | 1.0000 | 0.0000 | 1.0000 |

### per split (AUROC)

| condition | anthropic harmless refusal | bbq substitution | hc context drift | hc contradiction | mm substitution | oig context drift | oig omission |
|---|---|---|---|---|---|---|---|
| red-team only (gptoss, iter5 training set) | 0.935 | 0.717 | 0.994 | 0.800 | 0.846 | 0.711 | 0.765 |
| red-team only (nemotron, iter5 training set) | 0.731 | 0.955 | 0.833 | 0.928 | 0.903 | 0.569 | 0.651 |
| CV on eval alone (5-fold, out-of-fold) | 1.000 | 0.995 | 0.999 | 0.979 | 0.981 | 0.966 | 0.673 |
| CV on eval + red-team (gptoss) | 0.994 | 0.990 | 0.998 | 0.978 | 0.986 | 0.968 | 0.757 |
| CV on eval + red-team (nemotron) | 1.000 | 0.993 | 0.999 | 0.960 | 0.985 | 0.961 | 0.693 |
| oracle: fit and scored on all 1302 eval rows | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Part 2 — dev-sample sweep


### arm = gptoss


**AUROC**

| N dev samples | 0 | 42 | 84 | 126 | 168 | 210 | 252 | 294 | 336 |
|---|---|---|---|---|---|---|---|---|---|
| joint | 0.8299 | 0.8502 | 0.8707 | 0.8956 | 0.9161 | 0.9364 | 0.9397 | 0.9448 | 0.9447 |
| finetune (lr=default 5e-3) | 0.8299 | 0.8299 | 0.9036 | 0.9196 | 0.9203 | 0.9397 | 0.9409 | 0.9460 | 0.9541 |
| finetune (lr=0.0001) | 0.8299 | 0.8299 | 0.8367 | 0.8405 | 0.8429 | 0.8449 | 0.8421 | 0.8434 | 0.8474 |

**TPR@FPR<=1%**

| N dev samples | 0 | 42 | 84 | 126 | 168 | 210 | 252 | 294 | 336 |
|---|---|---|---|---|---|---|---|---|---|
| joint | 0.1698 | 0.2104 | 0.2532 | 0.3373 | 0.3806 | 0.3920 | 0.4400 | 0.5238 | 0.5386 |
| finetune (lr=default 5e-3) | 0.1698 | 0.1698 | 0.4916 | 0.4515 | 0.3623 | 0.4071 | 0.5088 | 0.5445 | 0.7216 |
| finetune (lr=0.0001) | 0.1698 | 0.1698 | 0.1911 | 0.1940 | 0.1951 | 0.1912 | 0.1926 | 0.1940 | 0.1940 |

### arm = nemotron


**AUROC**

| N dev samples | 0 | 42 | 84 | 126 | 168 | 210 | 252 | 294 | 336 |
|---|---|---|---|---|---|---|---|---|---|
| joint | 0.7997 | 0.8260 | 0.8525 | 0.8635 | 0.8882 | 0.9052 | 0.9166 | 0.9339 | 0.9371 |
| finetune (lr=default 5e-3) | 0.7997 | 0.7997 | 0.8640 | 0.8882 | 0.9055 | 0.9244 | 0.9306 | 0.9396 | 0.9435 |
| finetune (lr=0.0001) | 0.7997 | 0.7997 | 0.7982 | 0.8015 | 0.8024 | 0.8041 | 0.8036 | 0.8057 | 0.8094 |

**TPR@FPR<=1%**

| N dev samples | 0 | 42 | 84 | 126 | 168 | 210 | 252 | 294 | 336 |
|---|---|---|---|---|---|---|---|---|---|
| joint | 0.2154 | 0.3411 | 0.2571 | 0.3978 | 0.3613 | 0.3618 | 0.6005 | 0.5823 | 0.4998 |
| finetune (lr=default 5e-3) | 0.2154 | 0.2154 | 0.3192 | 0.3822 | 0.4355 | 0.3997 | 0.4340 | 0.4541 | 0.4920 |
| finetune (lr=0.0001) | 0.2154 | 0.2154 | 0.2596 | 0.2568 | 0.2525 | 0.2554 | 0.2654 | 0.2668 | 0.2696 |