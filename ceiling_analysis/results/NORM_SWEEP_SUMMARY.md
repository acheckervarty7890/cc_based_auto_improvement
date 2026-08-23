# The dev sweep under layernorm

Arm **hu_ha_dd_gptoss120b**. A single probe is trained on base + red-team successes plus N dev rows and scored on the eval splits, for 10 values of N and three arms, exactly as in the main sweep — the only change is one normalization step in front of `LinearThenSoftmax`'s linear layer. The unnormalized curves are the existing `sweep_hu_ha_dd_gptoss120b.jsonl`, at the same fit seed. Each N>0 point is the mean over 3 dev draws; the unnormalized ceiling for this arm is 0.9844.

## What it does

**layernorm is ahead at 24 of the 24 informative points**, across all three arms, by +0.0072 on average. That is a larger and far more consistent effect than the ceiling study found at its top rung (+0.0020, 3/4 seeds) — which is what the ceiling study predicted would happen: the gain grows as the training set shrinks, and every point here trains on less eval-distribution data than the ceiling's smallest rung.

It also **tightens the spread across dev draws** — the normalized head's sd is 0.86x the unnormalized head's, averaged over the 21 points that have one. The `finetune` arm is where this is most visible (e.g. at N=97, 0.0116 -> 0.0021), and it is also the arm with the largest mean gain.

**None of this closes the gap to the ceiling.** The best sweep point moves from 0.9534 to 0.9573 against a ceiling of 0.9844. Normalization is a better-conditioned head, not more information.

## N = 0: base + red-team only

This is the point every write-up quotes as the red-team-only number, and `run_sweep.py` fits it once, so it had no error bar at all. Re-fit under 4 head seeds. The `none`/seed-42 re-run reproduces the existing sweep's N=0 row to 0.00e+00 — the control on the new code path.

| head seed | none | layernorm | diff |
|---|---|---|---|
| 7 | 0.9034 | 0.9121 | +0.0087 |
| 42 | 0.9164 | 0.9195 | +0.0031 |
| 1234 | 0.8990 | 0.9108 | +0.0119 |
| 20260823 | 0.8968 | 0.9217 | +0.0249 |
| **mean** | **0.9039** | **0.9160** | **+0.0121** |
| sd | 0.0088 | 0.0054 | 0.0093 |

Paired, layernorm wins 4/4. The unnormalized head's own spread across these seeds is 0.0197 — read the single-seed number against that before reading the difference.

## `mixed`

| N dev | none (mean +- sd) | layernorm (mean +- sd) | diff |
|---|---|---|---|
| 0 | 0.9164 +- 0.0000 | 0.9195 +- 0.0000 | +0.0031 |
| 24 | 0.9151 +- 0.0093 | 0.9227 +- 0.0078 | +0.0076 |
| 48 | 0.9221 +- 0.0029 | 0.9247 +- 0.0062 | +0.0026 |
| 73 | 0.9264 +- 0.0059 | 0.9283 +- 0.0049 | +0.0018 |
| 97 | 0.9300 +- 0.0026 | 0.9364 +- 0.0025 | +0.0063 |
| 121 | 0.9302 +- 0.0050 | 0.9342 +- 0.0050 | +0.0040 |
| 145 | 0.9342 +- 0.0070 | 0.9375 +- 0.0093 | +0.0033 |
| 170 | 0.9364 +- 0.0060 | 0.9368 +- 0.0040 | +0.0005 |
| 194 | 0.9350 +- 0.0096 | 0.9402 +- 0.0054 | +0.0052 |
| 218 | 0.9373 +- 0.0000 | 0.9415 +- 0.0000 | +0.0042 |

Mean difference over the 10 informative points **+0.0039**, positive at 10/10. Best point: none 0.9373, layernorm 0.9415.

## `finetune`

| N dev | none (mean +- sd) | layernorm (mean +- sd) | diff |
|---|---|---|---|
| 0 | 0.9164 | 0.9195 | _excluded: second stage never steps; identical to N=0_ |
| 24 | 0.9164 | 0.9195 | _excluded: second stage never steps; identical to N=0_ |
| 48 | 0.9164 | 0.9195 | _excluded: second stage never steps; identical to N=0_ |
| 73 | 0.9236 +- 0.0123 | 0.9350 +- 0.0101 | +0.0115 |
| 97 | 0.9231 +- 0.0116 | 0.9430 +- 0.0021 | +0.0199 |
| 121 | 0.9263 +- 0.0090 | 0.9393 +- 0.0020 | +0.0130 |
| 145 | 0.9304 +- 0.0124 | 0.9467 +- 0.0071 | +0.0163 |
| 170 | 0.9304 +- 0.0128 | 0.9374 +- 0.0031 | +0.0071 |
| 194 | 0.9386 +- 0.0058 | 0.9458 +- 0.0044 | +0.0072 |
| 218 | 0.9430 +- 0.0000 | 0.9488 +- 0.0000 | +0.0059 |

Mean difference over the 7 informative points **+0.0115**, positive at 7/7. Best point: none 0.9430, layernorm 0.9488.

## `dev_only`

| N dev | none (mean +- sd) | layernorm (mean +- sd) | diff |
|---|---|---|---|
| 24 | 0.5299 | 0.5223 | _excluded: no optimizer step fires_ |
| 48 | 0.5299 | 0.5223 | _excluded: no optimizer step fires_ |
| 73 | 0.9213 +- 0.0080 | 0.9344 +- 0.0010 | +0.0131 |
| 97 | 0.9303 +- 0.0037 | 0.9388 +- 0.0042 | +0.0084 |
| 121 | 0.9389 +- 0.0068 | 0.9454 +- 0.0098 | +0.0065 |
| 145 | 0.9411 +- 0.0116 | 0.9491 +- 0.0116 | +0.0080 |
| 170 | 0.9381 +- 0.0032 | 0.9462 +- 0.0061 | +0.0081 |
| 194 | 0.9466 +- 0.0046 | 0.9532 +- 0.0061 | +0.0066 |
| 218 | 0.9534 +- 0.0000 | 0.9573 +- 0.0000 | +0.0039 |

Mean difference over the 7 informative points **+0.0078**, positive at 7/7. Best point: none 0.9534, layernorm 0.9573.

