# Red-team vintage sweep — HIGH-STAKES (experiment9)
_Updated 2026-08-15T14:50:38+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data (`data/hs_ls_200.jsonl`) plus one *vintage* of iteration-3 red-team pairs, scored on the four `eval_datasets/` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**Over-length pairs are dropped.** `get_activations` truncates at 1024 tokens, so a conversation at that width lost its tail — and for an LLM-written contrastive counterpart the tail is disproportionately the part carrying the opposite-class label. Every over-cap row in both arms is a *generated* row; not one attacker-written source overran. The affected pair is removed **whole**, because dropping the generated half alone would orphan its success and break the exact 50/50 balance that makes the vintages comparable.

- **deepseekv4pro**: 6 row(s) at the cap (6 generated, 0 attacker-written) → 6 pair(s), 12 row(s) removed from every vintage
- **gptoss120b**: 14 row(s) at the cap (14 generated, 0 attacker-written) → 14 pair(s), 28 row(s) removed from every vintage

**Read the sd, not just the mean.** These are unpaired refits with independent initialisations, so the seed-to-seed sd is the quantity that makes or breaks a single-seed reading — where it is comparable to a between-vintage gap, that gap is not evidence of anything.

## Progress: 70 fits recorded

- **deepseekv4pro**: v0×10, v1×10, v2×10, v3×10
- **gptoss120b**: v0×10, v1×10, v3×10

## deepseekv4pro — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic | mt | mts | toolace | mean |
|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.9531 ± 0.0044 | 0.8673 ± 0.0495 | 0.9436 ± 0.0094 | 0.7080 ± 0.0252 | 0.8680 ± 0.0121 |
| v1 | 294 | 10 | 0.9282 ± 0.0142 | 0.9470 ± 0.0079 | 0.8857 ± 0.0144 | 0.7655 ± 0.0141 | 0.8816 ± 0.0063 |
| v2 | 542 | 10 | 0.9641 ± 0.0025 | 0.9331 ± 0.0067 | 0.9425 ± 0.0160 | 0.8190 ± 0.0075 | 0.9147 ± 0.0054 |
| v3 | 716 | 10 | 0.9539 ± 0.0028 | 0.9127 ± 0.0097 | 0.9275 ± 0.0179 | 0.7708 ± 0.0093 | 0.8912 ± 0.0064 |

## gptoss120b — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic | mt | mts | toolace | mean |
|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.9531 ± 0.0044 | 0.8673 ± 0.0495 | 0.9436 ± 0.0094 | 0.7080 ± 0.0252 | 0.8680 ± 0.0121 |
| v1 | 228 | 10 | 0.9552 ± 0.0067 | 0.9599 ± 0.0117 | 0.9402 ± 0.0066 | 0.7520 ± 0.0213 | 0.9018 ± 0.0084 |
| v3 | 562 | 10 | 0.9682 ± 0.0019 | 0.9748 ± 0.0058 | 0.9653 ± 0.0039 | 0.8220 ± 0.0093 | 0.9326 ± 0.0032 |
