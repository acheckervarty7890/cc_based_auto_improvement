# Red-team vintage sweep — live status
_Updated 2026-08-14T21:07:31+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data plus one *vintage* of iteration-3 red-team pairs, scored on the four `eval_dataset_hu_ha` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**Read the sd, not just the mean.** `eval_balanced_refusal` moves by sd ~0.02-0.07 on seed alone, which is larger than most between-vintage gaps. A single-seed comparison of two vintages on that split means nothing; that is what this sweep exists to quantify. For a *paired* between-vintage contrast (common random numbers, far tighter error bars) see `*_gate.json` from `scripts/attribution_vintage_gate.py`.

## Progress: 58 fits recorded

- **deepseekv4pro**: v0×5, v1×5, v2×4, v3×4
- **gptoss120b**: v0×10, v1×10, v2×10, v3×10

## deepseekv4pro — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | ai_dilemmas | ant_hh | balanced_refusal | daily_dilemmas | mean |
|---|---|---|---|---|---|---|---|
| v0 | 0 | 5 | 0.4987 ± 0.0431 | 0.4756 ± 0.0871 | 0.3529 ± 0.2089 | 0.5057 ± 0.0262 | 0.4582 ± 0.0756 |
| v1 | 368 | 5 | 0.8350 ± 0.0278 | 0.7289 ± 0.0226 | 0.8401 ± 0.0312 | 0.9665 ± 0.0202 | 0.8426 ± 0.0189 |
| v2 | 706 | 4 | 0.9565 ± 0.0282 | 0.7272 ± 0.0161 | 0.9035 ± 0.0201 | 0.9792 ± 0.0039 | 0.8916 ± 0.0043 |
| v3 | 878 | 4 | 0.9678 ± 0.0173 | 0.7307 ± 0.0219 | 0.9234 ± 0.0345 | 0.9871 ± 0.0024 | 0.9023 ± 0.0132 |

## gptoss120b — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | ai_dilemmas | ant_hh | balanced_refusal | daily_dilemmas | mean |
|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.5008 ± 0.0320 | 0.4501 ± 0.0727 | 0.4212 ± 0.2154 | 0.5268 ± 0.0389 | 0.4747 ± 0.0633 |
| v1 | 356 | 10 | 0.9429 ± 0.0186 | 0.7065 ± 0.0108 | 0.9072 ± 0.0265 | 0.9709 ± 0.0059 | 0.8819 ± 0.0090 |
| v2 | 546 | 10 | 0.9870 ± 0.0193 | 0.7445 ± 0.0156 | 0.9235 ± 0.0297 | 0.9889 ± 0.0039 | 0.9110 ± 0.0106 |
| v3 | 778 | 10 | 0.9922 ± 0.0079 | 0.7359 ± 0.0172 | 0.9335 ± 0.0358 | 0.9789 ± 0.0071 | 0.9101 ± 0.0114 |
