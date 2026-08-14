# Red-team vintage sweep — live status
_Updated 2026-08-14T17:25:48+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data plus one *vintage* of iteration-3 red-team pairs, scored on the four `eval_dataset_hu_ha` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**Read the sd, not just the mean.** `eval_balanced_refusal` moves by sd ~0.02-0.07 on seed alone, which is larger than most between-vintage gaps. A single-seed comparison of two vintages on that split means nothing; that is what this sweep exists to quantify. For a *paired* between-vintage contrast (common random numbers, far tighter error bars) see `*_gate.json` from `scripts/attribution_vintage_gate.py`.

## Progress: 19 fits recorded

- **deepseekv4pro**: v0×1, v1×1, v2×1, v3×1
- **gptoss120b**: v0×4, v1×4, v2×4, v3×3

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
| v0 | 0 | 4 | 0.5116 ± 0.0369 | 0.4909 ± 0.0925 | 0.3980 ± 0.2113 | 0.5036 ± 0.0298 | 0.4760 ± 0.0742 |
| v1 | 356 | 4 | 0.9346 ± 0.0212 | 0.7109 ± 0.0033 | 0.9184 ± 0.0068 | 0.9668 ± 0.0069 | 0.8827 ± 0.0052 |
| v2 | 546 | 4 | 0.9904 ± 0.0091 | 0.7419 ± 0.0088 | 0.9247 ± 0.0087 | 0.9884 ± 0.0061 | 0.9113 ± 0.0020 |
| v3 | 778 | 3 | 0.9951 ± 0.0048 | 0.7386 ± 0.0113 | 0.9306 ± 0.0554 | 0.9827 ± 0.0030 | 0.9118 ± 0.0128 |
