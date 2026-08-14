# Red-team vintage sweep — live status
_Updated 2026-08-14T18:26:15+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data plus one *vintage* of iteration-3 red-team pairs, scored on the four `eval_dataset_hu_ha` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**Read the sd, not just the mean.** `eval_balanced_refusal` moves by sd ~0.02-0.07 on seed alone, which is larger than most between-vintage gaps. A single-seed comparison of two vintages on that split means nothing; that is what this sweep exists to quantify. For a *paired* between-vintage contrast (common random numbers, far tighter error bars) see `*_gate.json` from `scripts/attribution_vintage_gate.py`.

## Progress: 32 fits recorded

- **deepseekv4pro**: v0×1, v1×1, v2×1, v3×1
- **gptoss120b**: v0×7, v1×7, v2×7, v3×7

## deepseekv4pro — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | ai_dilemmas | ant_hh | balanced_refusal | daily_dilemmas | mean |
|---|---|---|---|---|---|---|---|
| v0 | 0 | 1 | 0.4918 | 0.4840 | 0.6014 | 0.5426 | 0.5299 |
| v1 | 368 | 1 | 0.8179 | 0.7591 | 0.8434 | 0.9796 | 0.8500 |
| v2 | 706 | 1 | 0.9868 | 0.7367 | 0.8852 | 0.9797 | 0.8971 |
| v3 | 878 | 1 | 0.9813 | 0.7098 | 0.9688 | 0.9850 | 0.9112 |

## gptoss120b — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | ai_dilemmas | ant_hh | balanced_refusal | daily_dilemmas | mean |
|---|---|---|---|---|---|---|---|
| v0 | 0 | 7 | 0.4982 ± 0.0361 | 0.4684 ± 0.0795 | 0.3953 ± 0.1956 | 0.5167 ± 0.0285 | 0.4697 ± 0.0650 |
| v1 | 356 | 7 | 0.9398 ± 0.0199 | 0.7055 ± 0.0113 | 0.9185 ± 0.0106 | 0.9695 ± 0.0064 | 0.8833 ± 0.0076 |
| v2 | 546 | 7 | 0.9848 ± 0.0232 | 0.7430 ± 0.0155 | 0.9148 ± 0.0314 | 0.9889 ± 0.0047 | 0.9079 ± 0.0108 |
| v3 | 778 | 7 | 0.9940 ± 0.0043 | 0.7350 ± 0.0176 | 0.9296 ± 0.0428 | 0.9803 ± 0.0043 | 0.9097 ± 0.0138 |
