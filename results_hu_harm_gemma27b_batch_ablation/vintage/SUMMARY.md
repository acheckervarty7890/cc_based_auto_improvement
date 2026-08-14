# Red-team vintage sweep — live status
_Updated 2026-08-14T17:45:58+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data plus one *vintage* of iteration-3 red-team pairs, scored on the four `eval_dataset_hu_ha` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**Read the sd, not just the mean.** `eval_balanced_refusal` moves by sd ~0.02-0.07 on seed alone, which is larger than most between-vintage gaps. A single-seed comparison of two vintages on that split means nothing; that is what this sweep exists to quantify. For a *paired* between-vintage contrast (common random numbers, far tighter error bars) see `*_gate.json` from `scripts/attribution_vintage_gate.py`.

## Progress: 23 fits recorded

- **deepseekv4pro**: v0×1, v1×1, v2×1, v3×1
- **gptoss120b**: v0×5, v1×5, v2×5, v3×4

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
| v0 | 0 | 5 | 0.4987 ± 0.0431 | 0.4756 ± 0.0871 | 0.3529 ± 0.2089 | 0.5057 ± 0.0262 | 0.4582 ± 0.0756 |
| v1 | 356 | 5 | 0.9334 ± 0.0185 | 0.7078 ± 0.0075 | 0.9148 ± 0.0101 | 0.9672 ± 0.0060 | 0.8808 ± 0.0062 |
| v2 | 546 | 5 | 0.9922 ± 0.0089 | 0.7468 ± 0.0134 | 0.9262 ± 0.0083 | 0.9892 ± 0.0056 | 0.9136 ± 0.0054 |
| v3 | 778 | 4 | 0.9957 ± 0.0041 | 0.7417 ± 0.0111 | 0.9381 ± 0.0476 | 0.9811 ± 0.0040 | 0.9141 ± 0.0115 |
