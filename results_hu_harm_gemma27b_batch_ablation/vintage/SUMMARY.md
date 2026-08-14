# Red-team vintage sweep — live status
_Updated 2026-08-14T17:05:35+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data plus one *vintage* of iteration-3 red-team pairs, scored on the four `eval_dataset_hu_ha` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**Read the sd, not just the mean.** `eval_balanced_refusal` moves by sd ~0.02-0.07 on seed alone, which is larger than most between-vintage gaps. A single-seed comparison of two vintages on that split means nothing; that is what this sweep exists to quantify. For a *paired* between-vintage contrast (common random numbers, far tighter error bars) see `*_gate.json` from `scripts/attribution_vintage_gate.py`.

## Progress: 14 fits recorded

- **deepseekv4pro**: v0×1, v1×1, v2×1, v3×1
- **gptoss120b**: v0×3, v1×3, v2×2, v3×2

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
| v0 | 0 | 3 | 0.5107 ± 0.0451 | 0.4474 ± 0.0382 | 0.3927 ± 0.2585 | 0.5011 ± 0.0360 | 0.4630 ± 0.0851 |
| v1 | 356 | 3 | 0.9273 ± 0.0189 | 0.7105 ± 0.0038 | 0.9215 ± 0.0033 | 0.9642 ± 0.0056 | 0.8809 ± 0.0046 |
| v2 | 546 | 2 | 0.9860 ± 0.0126 | 0.7481 ± 0.0057 | 0.9203 ± 0.0068 | 0.9872 ± 0.0103 | 0.9104 ± 0.0026 |
| v3 | 778 | 2 | 0.9950 ± 0.0068 | 0.7382 ± 0.0160 | 0.9240 ± 0.0766 | 0.9839 ± 0.0032 | 0.9102 ± 0.0177 |
