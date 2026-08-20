## Part 1 — ceiling on eval_sets/instructions

| condition | AUROC | acc | TPR@1%FPR (tuberlens) | TPR@FPR<=1% |
|---|---|---|---|---|
| red-team only (gptoss, iter5 training set) | 0.8272 | 0.7112 | 0.2226 | 0.2812 |
| red-team only (nemotron, iter5 training set) | 0.8586 | 0.7564 | 0.1447 | 0.2075 |
| CV on eval alone (5-fold, out-of-fold) | 0.9416 | 0.8972 | 0.6098 | 0.7755 |
| CV on eval + red-team (gptoss) | 0.9456 | 0.8901 | 0.3712 | 0.6958 |
| CV on eval + red-team (nemotron) | 0.9248 | 0.8669 | 0.4308 | 0.6270 |
| oracle: fit and scored on all 1302 eval rows | 1.0000 | 1.0000 | 0.0000 | 1.0000 |

### per split (AUROC)

| condition | anthropic harmless refusal | bbq substitution | hc context drift | hc contradiction | mm substitution | oig context drift | oig omission |
|---|---|---|---|---|---|---|---|
| red-team only (gptoss, iter5 training set) | 0.675 | 0.938 | 0.974 | 0.936 | 0.935 | 0.609 | 0.723 |
| red-team only (nemotron, iter5 training set) | 0.910 | 0.935 | 0.844 | 0.906 | 0.953 | 0.742 | 0.720 |
| CV on eval alone (5-fold, out-of-fold) | 1.000 | 0.995 | 0.999 | 0.979 | 0.981 | 0.966 | 0.673 |
| CV on eval + red-team (gptoss) | 0.998 | 0.990 | 0.999 | 0.954 | 0.985 | 0.957 | 0.736 |
| CV on eval + red-team (nemotron) | 0.999 | 0.991 | 0.997 | 0.957 | 0.989 | 0.952 | 0.589 |
| oracle: fit and scored on all 1302 eval rows | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Part 2 — dev-sample sweep


### arm = gptoss


**AUROC**

| N dev samples | 0 | 42 | 84 | 126 | 168 | 210 | 252 | 294 | 336 |
|---|---|---|---|---|---|---|---|---|---|
| joint | 0.8298 | 0.8376 | 0.8416 | 0.8781 | 0.9063 | 0.9201 | 0.9297 | 0.9368 | 0.9410 |
| finetune (lr=default 5e-3) | 0.8298 | 0.8298 | 0.8831 | 0.8990 | 0.9132 | 0.9254 | 0.9237 | 0.9405 | 0.9393 |
| finetune (lr=0.0001) | 0.8298 | 0.8298 | 0.8206 | 0.8169 | 0.8160 | 0.8140 | 0.8117 | 0.8117 | 0.8126 |

**TPR@FPR<=1%**

| N dev samples | 0 | 42 | 84 | 126 | 168 | 210 | 252 | 294 | 336 |
|---|---|---|---|---|---|---|---|---|---|
| joint | 0.2784 | 0.2705 | 0.2808 | 0.4813 | 0.3412 | 0.4077 | 0.3954 | 0.3857 | 0.4301 |
| finetune (lr=default 5e-3) | 0.2784 | 0.2784 | 0.3582 | 0.4977 | 0.4456 | 0.5712 | 0.5384 | 0.6393 | 0.4913 |
| finetune (lr=0.0001) | 0.2784 | 0.2784 | 0.2469 | 0.2365 | 0.2340 | 0.2283 | 0.2255 | 0.2297 | 0.2312 |

### arm = nemotron


**AUROC**

| N dev samples | 0 | 42 | 84 | 126 |
|---|---|---|---|---|
| joint | 0.8619 | 0.8462 | 0.8737 | 0.8885 |
| finetune (lr=default 5e-3) | 0.8619 | - | - | - |
| finetune (lr=0.0001) | 0.8619 | - | - | - |

**TPR@FPR<=1%**

| N dev samples | 0 | 42 | 84 | 126 |
|---|---|---|---|---|
| joint | 0.2415 | 0.2025 | 0.3093 | 0.4355 |
| finetune (lr=default 5e-3) | 0.2415 | - | - | - |
| finetune (lr=0.0001) | 0.2415 | - | - | - |