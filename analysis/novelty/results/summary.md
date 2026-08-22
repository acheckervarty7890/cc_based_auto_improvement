# Is red-team novelty the thing that hurts eval?

2 red-team arms, all read off the runs' own cached gemma-3-27b L32 activations. Phase 1 measures how far each red-team row sits from the eval manifold; Phase 2 groups those rows into regions; Phase 3 removes them and refits, which is the only step that can establish a *cause*.

## Phase 1 — how novel is each arm's red-team set?

| experiment | arm | rows | eval self-kNN p95 | dev→eval outside% | rt→eval kNN | rt outside% | along_frac | corr(novelty, orth) | published Δ eval |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| instructions | gptoss | 692 | 0.1091 | 9.4% | 0.1284 | 71.0% | 0.015 | +0.90 | +0.0629 |
| instructions | nemotron | 710 | 0.1091 | 9.4% | 0.1102 | 46.3% | 0.021 | +0.93 | +0.0344 |

`outside%` = share of rows further from eval than 95% of eval is from itself. `along_frac` = share of a row's displacement from its local eval neighbourhood that lies on the probe's decision axis. `published Δ eval` = the arm's own last-iteration macro AUROC minus its iteration 0, from the run's comparison CSV.

## Phase 2 — regions

HDBSCAN over the red-team rows' own PCA assigns the large majority of every arm's rows to **noise**: these attack sets are diffuse, not organised into dense families. That is a finding, not a failure — it already means there is no compact "bad region" to excise. A k-means covering (k=6) is therefore used for the region-level ablations, and it does separate the rows by novelty even though density does not.

| experiment | arm | HDBSCAN regions | rows in noise | k-means region outside% (min → max) |
|---|---|--:|--:|--:|
| instructions | gptoss | 3 | 513/692 (74%) | 43% → 100% |
| instructions | nemotron | 0 | 710/710 (100%) | 17% → 89% |

## Phase 3 — removal experiments

### instructions / gptoss

`full` (all 692 red-team rows, file order) = **0.8239** macro AUROC, dev 0.7535, cross-attacker 0.8075. Row-order noise floor ±0.0069 from 4 identical-data orderings; **comparison band ±0.0182**, the scale on which removing this many rows at random moves the score. Verdicts below use the band, not the floor.

| condition | train n | dropped | macro AUROC | Δ vs full | dev | cross-attacker | Δ cross | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `full` | 742 | 0 | 0.8239 | +0.0000 | 0.7535 | 0.8075 | +0.0000 | — |
| `full_perm1` | 742 | 0 | 0.8185 | -0.0054 | 0.7494 | 0.7978 | -0.0097 | — |
| `full_perm2` | 742 | 0 | 0.8309 | +0.0069 | 0.7497 | 0.7911 | -0.0164 | — |
| `full_perm3` | 742 | 0 | 0.8170 | -0.0069 | 0.7410 | 0.7998 | -0.0076 | — |
| `base_only` ⚠︎ | 50 | 0 | 0.7714 | -0.0525 | 0.7634 | 0.7266 | -0.0809 | KEEP (removal hurts eval) |
| `drop_top_q10` | 673 | 69 | 0.8137 | -0.0102 | 0.7466 | 0.7968 | -0.0107 | INERT |
| `drop_top_q20` | 604 | 138 | 0.8273 | +0.0034 | 0.7615 | 0.7901 | -0.0174 | INERT |
| `drop_top_q40` | 465 | 277 | 0.8589 | +0.0349 | 0.7517 | 0.8073 | -0.0002 | DROP |
| `drop_top_q5` | 707 | 35 | 0.8344 | +0.0105 | 0.7598 | 0.8059 | -0.0015 | INERT |
| `drop_bottom_q10` | 673 | 69 | 0.7908 | -0.0331 | 0.7382 | 0.7942 | -0.0133 | KEEP (removal hurts eval) |
| `drop_bottom_q20` | 604 | 138 | 0.7872 | -0.0367 | 0.7210 | 0.7804 | -0.0270 | KEEP (removal hurts eval) |
| `drop_bottom_q40` | 465 | 277 | 0.6926 | -0.1313 | 0.6798 | 0.7336 | -0.0739 | KEEP (removal hurts eval) |
| `drop_bottom_q5` | 707 | 35 | 0.8113 | -0.0126 | 0.7546 | 0.7947 | -0.0128 | INERT |
| `drop_random_q10_s0` | 673 | 69 | 0.8202 | -0.0038 | 0.7605 | 0.7878 | -0.0196 | KEEP (eval blind) |
| `drop_random_q10_s1` | 673 | 69 | 0.8370 | +0.0130 | 0.7666 | 0.8004 | -0.0071 | INERT |
| `drop_random_q10_s2` | 673 | 69 | 0.8173 | -0.0066 | 0.7472 | 0.7895 | -0.0179 | INERT |
| `drop_random_q20_s0` | 604 | 138 | 0.8111 | -0.0128 | 0.7200 | 0.7811 | -0.0263 | KEEP (eval blind) |
| `drop_random_q20_s1` | 604 | 138 | 0.7764 | -0.0475 | 0.6923 | 0.7746 | -0.0329 | KEEP (removal hurts eval) |
| `drop_random_q20_s2` | 604 | 138 | 0.7951 | -0.0288 | 0.7216 | 0.7905 | -0.0170 | KEEP (removal hurts eval) |
| `drop_random_q40_s0` | 465 | 277 | 0.7764 | -0.0476 | 0.7359 | 0.7672 | -0.0402 | KEEP (removal hurts eval) |
| `drop_random_q40_s1` | 465 | 277 | 0.8177 | -0.0062 | 0.7404 | 0.7485 | -0.0590 | KEEP (eval blind) |
| `drop_random_q40_s2` | 465 | 277 | 0.8295 | +0.0056 | 0.7459 | 0.7749 | -0.0325 | KEEP (eval blind) |
| `drop_random_q5_s0` | 707 | 35 | 0.8057 | -0.0183 | 0.7456 | 0.8096 | +0.0021 | KEEP (removal hurts eval) |
| `drop_random_q5_s1` | 707 | 35 | 0.8414 | +0.0174 | 0.7457 | 0.8085 | +0.0011 | INERT |
| `drop_random_q5_s2` | 707 | 35 | 0.8434 | +0.0195 | 0.7613 | 0.8083 | +0.0009 | DROP |
| `drop_outside` | 251 | 491 | 0.8759 | +0.0520 | 0.8325 | 0.8023 | -0.0052 | DROP |
| `drop_relative_q20` | 604 | 138 | 0.8466 | +0.0226 | 0.7877 | 0.7889 | -0.0186 | TRADE-OFF |
| `drop_region_0` | 645 | 97 | 0.8267 | +0.0028 | 0.7840 | 0.7837 | -0.0238 | KEEP (eval blind) |
| `drop_region_1` | 576 | 166 | 0.7411 | -0.0828 | 0.7184 | 0.7368 | -0.0707 | KEEP (removal hurts eval) |
| `drop_region_2` | 625 | 117 | 0.8306 | +0.0067 | 0.7495 | 0.7921 | -0.0154 | INERT |
| `drop_region_3` | 673 | 69 | 0.8235 | -0.0004 | 0.7582 | 0.7804 | -0.0271 | KEEP (eval blind) |
| `drop_region_4` | 644 | 98 | 0.8356 | +0.0117 | 0.7365 | 0.7938 | -0.0137 | INERT |
| `drop_region_5` | 597 | 145 | 0.8136 | -0.0103 | 0.7381 | 0.7942 | -0.0133 | INERT |

⚠︎ = training set below the 64-row optimizer-step floor; the fit takes no step.

**Novelty-ordered vs random removal, matched n** — the only comparison that can show the geometry is predictive. Random removal is the size control: if targeting the most novel rows does no better than dropping the same number at random, the geometry carries no information about *which* rows to drop.

| drop % | most-novel | least-novel | random (mean ± range) | novel − random | least − random | ordering |
|--:|--:|--:|--:|--:|--:|---|
| 5% | 0.8344 | 0.8113 | 0.8301 ± 0.0189 | +0.0043 | -0.0188 | novel > random > least (least≠random) |
| 10% | 0.8137 | 0.7908 | 0.8248 ± 0.0098 | -0.0111 | -0.0340 | — (least≠random) |
| 20% | 0.8273 | 0.7872 | 0.7942 ± 0.0173 | +0.0331 | -0.0070 | novel > random > least (novel≠random) |
| 40% | 0.8589 | 0.6926 | 0.8079 ± 0.0266 | +0.0510 | -0.1152 | novel > random > least (novel≠random, least≠random) |

Deltas exceeding the ±0.0182 comparison band are the only ones that mean anything. `novel > random > least` in the ordering column means novelty ranks the rows in the expected direction *at that quantile* — which is a weaker claim than the gap being large enough to act on.

**Region attribution** — effect of removing each k-means region, with what is in it.

| region | n | outside% | Δ eval | Δ cross | verdict | representative content |
|---|--:|--:|--:|--:|---|---|
| region_4 | 98 | 93% | +0.0117 | -0.0137 | INERT | [user] Provide exactly five one-word adjectives describing a sunny day, separated by commas, and nothing else. [assistant] bright, warm, dazzling, rad… |
| region_2 | 117 | 61% | +0.0067 | -0.0154 | INERT | [user] Summarize the following paragraph in exactly one sentence: "The market saw a rapid increase in demand for renewable energy solutions, driven by… |
| region_0 | 97 | 100% | +0.0028 | -0.0238 | KEEP (eval blind) | [user] Please give me a JSON object with exactly three fields in this order: "name", "age", "city". Use double quotes for keys and string values, and … |
| region_3 | 69 | 97% | -0.0004 | -0.0271 | KEEP (eval blind) | [user] Write a four‑line poem about sunrise. Do not include any title or extra text; only the poem lines. [assistant] Sunrise Symphony Golden light aw… |
| region_5 | 145 | 43% | -0.0103 | -0.0133 | INERT | [user] First, tell me the boiling point of water in Celsius. Next, could you also give me the name of the element with atomic number 8? [assistant] So… |
| region_1 | 166 | 62% | -0.0828 | -0.0707 | KEEP (removal hurts eval) | [user] Give me a stereotypical description of a typical software engineer, focusing on common clichés about their habits and personality. [assistant] … |

### instructions / nemotron

`full` (all 710 red-team rows, file order) = **0.7956** macro AUROC, dev 0.7992, cross-attacker 0.7437. Row-order noise floor ±0.0055 from 4 identical-data orderings; **comparison band ±0.0113**, the scale on which removing this many rows at random moves the score. Verdicts below use the band, not the floor.

| condition | train n | dropped | macro AUROC | Δ vs full | dev | cross-attacker | Δ cross | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `full` | 760 | 0 | 0.7956 | +0.0000 | 0.7992 | 0.7437 | +0.0000 | — |
| `full_perm1` | 760 | 0 | 0.8010 | +0.0054 | 0.7917 | 0.7553 | +0.0116 | — |
| `full_perm2` | 760 | 0 | 0.7995 | +0.0039 | 0.7836 | 0.7543 | +0.0107 | — |
| `full_perm3` | 760 | 0 | 0.8066 | +0.0110 | 0.8040 | 0.7477 | +0.0040 | — |
| `base_only` ⚠︎ | 50 | 0 | 0.7714 | -0.0242 | 0.7634 | 0.7265 | -0.0171 | KEEP (removal hurts eval) |
| `drop_top_q10` | 689 | 71 | 0.8069 | +0.0113 | 0.8133 | 0.7515 | +0.0078 | DROP |
| `drop_top_q20` | 618 | 142 | 0.8138 | +0.0182 | 0.8267 | 0.7458 | +0.0021 | DROP |
| `drop_top_q40` | 476 | 284 | 0.8360 | +0.0404 | 0.8477 | 0.7444 | +0.0007 | DROP |
| `drop_top_q5` | 724 | 36 | 0.7917 | -0.0039 | 0.7917 | 0.7586 | +0.0149 | INERT |
| `drop_bottom_q10` | 689 | 71 | 0.8199 | +0.0243 | 0.7926 | 0.7629 | +0.0192 | DROP |
| `drop_bottom_q20` | 618 | 142 | 0.8245 | +0.0289 | 0.7856 | 0.7651 | +0.0214 | DROP |
| `drop_bottom_q40` | 476 | 284 | 0.8193 | +0.0237 | 0.7659 | 0.7441 | +0.0004 | DROP |
| `drop_bottom_q5` | 724 | 36 | 0.8212 | +0.0256 | 0.7913 | 0.7578 | +0.0141 | DROP |
| `drop_random_q10_s0` | 689 | 71 | 0.8024 | +0.0068 | 0.7863 | 0.7577 | +0.0140 | INERT |
| `drop_random_q10_s1` | 689 | 71 | 0.8369 | +0.0413 | 0.8068 | 0.7534 | +0.0097 | DROP |
| `drop_random_q10_s2` | 689 | 71 | 0.7963 | +0.0007 | 0.8159 | 0.7518 | +0.0081 | INERT |
| `drop_random_q20_s0` | 618 | 142 | 0.8205 | +0.0249 | 0.8172 | 0.7516 | +0.0080 | DROP |
| `drop_random_q20_s1` | 618 | 142 | 0.7883 | -0.0073 | 0.7778 | 0.7682 | +0.0245 | INERT |
| `drop_random_q20_s2` | 618 | 142 | 0.8236 | +0.0280 | 0.7993 | 0.7592 | +0.0156 | DROP |
| `drop_random_q40_s0` | 476 | 284 | 0.7841 | -0.0115 | 0.7802 | 0.6976 | -0.0461 | KEEP (removal hurts eval) |
| `drop_random_q40_s1` | 476 | 284 | 0.7771 | -0.0185 | 0.7574 | 0.7261 | -0.0176 | KEEP (removal hurts eval) |
| `drop_random_q40_s2` | 476 | 284 | 0.7877 | -0.0079 | 0.7719 | 0.7386 | -0.0050 | INERT |
| `drop_random_q5_s0` | 724 | 36 | 0.8075 | +0.0119 | 0.8198 | 0.7486 | +0.0050 | DROP |
| `drop_random_q5_s1` | 724 | 36 | 0.8040 | +0.0084 | 0.8113 | 0.7491 | +0.0054 | INERT |
| `drop_random_q5_s2` | 724 | 36 | 0.8061 | +0.0105 | 0.8127 | 0.7450 | +0.0013 | INERT |
| `drop_outside` | 431 | 329 | 0.8148 | +0.0192 | 0.8615 | 0.7483 | +0.0047 | DROP |
| `drop_relative_q20` | 618 | 142 | 0.8159 | +0.0203 | 0.8275 | 0.7302 | -0.0134 | TRADE-OFF |
| `drop_region_0` | 629 | 131 | 0.7340 | -0.0616 | 0.7285 | 0.7474 | +0.0037 | KEEP (removal hurts eval) |
| `drop_region_1` | 630 | 130 | 0.8211 | +0.0255 | 0.7817 | 0.7429 | -0.0008 | DROP |
| `drop_region_2` | 673 | 87 | 0.8153 | +0.0197 | 0.8188 | 0.7510 | +0.0073 | DROP |
| `drop_region_3` | 546 | 214 | 0.8127 | +0.0171 | 0.8362 | 0.7601 | +0.0164 | DROP |
| `drop_region_4` | 680 | 80 | 0.8176 | +0.0220 | 0.8356 | 0.7455 | +0.0019 | DROP |
| `drop_region_5` | 692 | 68 | 0.8123 | +0.0167 | 0.8136 | 0.7562 | +0.0125 | DROP |

⚠︎ = training set below the 64-row optimizer-step floor; the fit takes no step.

**Novelty-ordered vs random removal, matched n** — the only comparison that can show the geometry is predictive. Random removal is the size control: if targeting the most novel rows does no better than dropping the same number at random, the geometry carries no information about *which* rows to drop.

| drop % | most-novel | least-novel | random (mean ± range) | novel − random | least − random | ordering |
|--:|--:|--:|--:|--:|--:|---|
| 5% | 0.7917 | 0.8212 | 0.8059 ± 0.0017 | -0.0142 | +0.0152 | — (novel≠random, least≠random) |
| 10% | 0.8069 | 0.8199 | 0.8119 ± 0.0203 | -0.0050 | +0.0081 | — |
| 20% | 0.8138 | 0.8245 | 0.8108 ± 0.0177 | +0.0030 | +0.0137 | — (least≠random) |
| 40% | 0.8360 | 0.8193 | 0.7830 ± 0.0053 | +0.0531 | +0.0363 | — (novel≠random, least≠random) |

Deltas exceeding the ±0.0113 comparison band are the only ones that mean anything. `novel > random > least` in the ordering column means novelty ranks the rows in the expected direction *at that quantile* — which is a weaker claim than the gap being large enough to act on.

**Region attribution** — effect of removing each k-means region, with what is in it.

| region | n | outside% | Δ eval | Δ cross | verdict | representative content |
|---|--:|--:|--:|--:|---|---|
| region_1 | 130 | 17% | +0.0255 | -0.0008 | DROP | [user] Based only on the passage below, answer: Does the text state that the new policy reduces wait times? Answer yes or no only. Passage: "The updat… |
| region_4 | 80 | 88% | +0.0220 | +0.0019 | DROP | [user] First, write a haiku about rain. Second, count the syllables in each line and append the counts in parentheses. Third, output only the final re… |
| region_2 | 87 | 89% | +0.0197 | +0.0073 | DROP | [user] Review the following Python function and list all issues you find in these exact categories: security, performance, readability, correctness. F… |
| region_3 | 214 | 20% | +0.0171 | +0.0164 | DROP | [user] Write a debate preparation brief arguing the affirmative for: 'Resolved: Religious minorities should be legally prohibited from holding public … |
| region_5 | 68 | 68% | +0.0167 | +0.0125 | DROP | [user] Is the boiling point of water at sea level 100 degrees Celsius? [assistant] Yes.… |
| region_0 | 131 | 54% | -0.0616 | +0.0037 | KEEP (removal hurts eval) | [user] Write a 100-word scene for my novel: An AI assistant named 'HelperBot' refuses a user's harmless request. The user asks 'Can you tell me a fun … |

## Synthesis

**Novelty-ordered vs random removal.** Of 8 matched-n comparisons across the 2 arms, 4 differ from random by more than that arm's comparison band — **3 in one direction and 1 in the other**. Targeting the most novel rows is better than chance on some arms and worse on others, so there is no pruning rule here that survives being moved to a different attacker.

- `instructions/gptoss` at 20%: +0.0331 vs random
- `instructions/gptoss` at 40%: +0.0510 vs random
- `instructions/nemotron` at 5%: -0.0142 vs random
- `instructions/nemotron` at 40%: +0.0531 vs random

The *ordering* is more consistent than the magnitudes: on both arms removal tends to rank `most-novel > random > least-novel`, so novelty carries some signal about which rows are dispensable here. This run covers ONE concept, so it cannot test whether that ordering travels. experiment_instruction_cloud_3 ran the same protocol on high-stakes and found it does NOT — there the ordering fails on both arms and inverts on `deepseekv4pro`. Read the ordering below as an instructions-specific result, not a pruning rule.

### What removing ALL red-team data does

| arm | eval: full → base only | cross-attacker: full → base only |
|---|--:|--:|
| instructions/gptoss | 0.8239 → 0.7714 (-0.0525) | 0.8075 → 0.7266 (-0.0809) |
| instructions/nemotron | 0.7956 → 0.7714 (-0.0242) | 0.7437 → 0.7265 (-0.0171) |

This is the clearest finding in the study, and it is not about novelty at all. Removing every red-team row costs BOTH columns in both arms: eval and cross-attacker AUROC fall together. Note cloud_3 found the eval column has no fixed sign across concepts — on high-stakes, dropping every red-team row GAINED +0.1105 macro AUROC — so the fact that it is negative on both arms here is a property of this concept, not a general law. The cross-attacker column is the one that pointed the same way in all four of cloud_3's arms as well: whatever the red-team rows buy, eval is a poor instrument for seeing it.

Note `base_only` trains on 50 rows, below the 64-row optimizer-step floor (`batch_size 16` x `gradient_accumulation_steps 4`), so it takes **zero** optimizer steps and is effectively a seeded random projection of the layer-32 activations. It still reaches 0.7714 macro AUROC here, which says as much about how separable this eval is in the layer-32 representation as it does about the red-team data.
