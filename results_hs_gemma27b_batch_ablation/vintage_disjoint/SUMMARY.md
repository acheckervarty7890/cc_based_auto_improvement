# Red-team vintage sweep — DISJOINT slices (experiment9, high-stakes)
_Updated 2026-08-16T18:16:38+00:00_

**What this measures.** The companion sweep in `../vintage/` is *cumulative*: vintage k holds every iteration-3 pair whose source success existed by iteration k, so v3 contains v2 contains v1. That conflates two things — a later vintage is both newer data and **more** data — so its flattening curve cannot say whether the later iterations' finds are individually weaker or merely redundant.

Here each vintage is fitted **alone** on top of the base training data: `v1`, then `v2-only` (v2 minus v1), then `v3-only` (v3 minus v2). Everything else is held at the cumulative sweep's values — same content, same cached activations, same hyperparameters, same ten seeds, same four `eval_datasets/` splits.

`v1` is identical under both memberships (nothing earlier to subtract), so its fits are the cumulative sweep's, not refits. The subtraction runs downward — v3 loses *cumulative* v2, then v2 loses *cumulative* v1 — so each slice is measured against the set the previous vintage actually held.

**Unpaired, so read the sigma.** Every gap below is quoted against the pooled seed sd of the two cells compared; only >= 2 sigma is treated as a result.

## deepseekv4pro — mean +/- sd over seeds (pipeline scale)

| slice | rows | seeds | anthropic | mt | mts | toolace | mean |
|---|---|---|---|---|---|---|---|
| v1 | 294 | 10 | 0.9282 ± 0.0142 | 0.9470 ± 0.0079 | 0.8857 ± 0.0144 | 0.7655 ± 0.0141 | 0.8816 ± 0.0063 |
| v2-only | 262 | 10 | 0.9617 ± 0.0016 | 0.9049 ± 0.0254 | 0.9440 ± 0.0149 | 0.8247 ± 0.0091 | 0.9088 ± 0.0086 |
| v3-only | 174 | 10 | 0.9472 ± 0.0051 | 0.9378 ± 0.0122 | 0.9335 ± 0.0103 | 0.7698 ± 0.0138 | 0.8971 ± 0.0067 |

## gptoss120b — mean +/- sd over seeds (pipeline scale)

| slice | rows | seeds | anthropic | mt | mts | toolace | mean |
|---|---|---|---|---|---|---|---|
| v1 | 228 | 10 | 0.9552 ± 0.0067 | 0.9599 ± 0.0117 | 0.9402 ± 0.0066 | 0.7520 ± 0.0213 | 0.9018 ± 0.0084 |
| v2-only | 214 | 10 | 0.9677 ± 0.0027 | 0.9656 ± 0.0061 | 0.9503 ± 0.0077 | 0.8145 ± 0.0111 | 0.9245 ± 0.0048 |
| v3-only | 140 | 10 | 0.9670 ± 0.0031 | 0.9755 ± 0.0040 | 0.9538 ± 0.0045 | 0.8320 ± 0.0122 | 0.9321 ± 0.0041 |

## The increment against the set that contains it

| arm | increment | rows | cumulative | rows | increment mean | cumulative mean | gap | sigma |
|---|---|---|---|---|---|---|---|---|
| deepseekv4pro | v2-only | 262 | v2 | 542 | 0.9088 ± 0.0086 | 0.9147 ± 0.0054 | -0.0058 | 0.6 |
| deepseekv4pro | v3-only | 174 | v3 | 716 | 0.8971 ± 0.0067 | 0.8912 ± 0.0064 | +0.0059 | 0.6 |
| gptoss120b | v2-only | 214 | v2 | 422 | 0.9245 ± 0.0048 | 0.9281 ± 0.0048 | -0.0036 | 0.5 |
| gptoss120b | v3-only | 140 | v3 | 562 | 0.9321 ± 0.0041 | 0.9326 ± 0.0032 | -0.0005 | 0.1 |

## The increments against each other

| arm | pair | rows | gap | sigma |
|---|---|---|---|---|
| deepseekv4pro | v2-only − v1 | 262 vs 294 | +0.0272 | 2.6 |
| deepseekv4pro | v3-only − v1 | 174 vs 294 | +0.0155 | 1.7 |
| deepseekv4pro | v3-only − v2-only | 174 vs 262 | -0.0117 | 1.1 |
| gptoss120b | v2-only − v1 | 214 vs 228 | +0.0227 | 2.3 |
| gptoss120b | v3-only − v1 | 140 vs 228 | +0.0302 | 3.2 |
| gptoss120b | v3-only − v2-only | 140 vs 214 | +0.0076 | 1.2 |

## Read-out

- **deepseekv4pro**: v1 0.8816 (294 rows) → v2-only 0.9088 (262 rows) → v3-only 0.8971 (174 rows).
  - v2-only (262 rows) is indistinguishable from the whole cumulative set (542 rows) — -0.0058, 0.6 sigma.
  - v3-only (174 rows) is indistinguishable from the whole cumulative set (716 rows) — +0.0059, 0.6 sigma.
- **gptoss120b**: v1 0.9018 (228 rows) → v2-only 0.9245 (214 rows) → v3-only 0.9321 (140 rows).
  - v2-only (214 rows) is indistinguishable from the whole cumulative set (422 rows) — -0.0036, 0.5 sigma.
  - v3-only (140 rows) is indistinguishable from the whole cumulative set (562 rows) — -0.0005, 0.1 sigma.
