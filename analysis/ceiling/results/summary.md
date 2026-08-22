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