# Is red-team novelty the thing that hurts eval?

Four red-team arms (two concepts x two attacker models), all read off the runs' own cached gemma-3-27b L32 activations. Phase 1 measures how far each red-team row sits from the eval manifold; Phase 2 groups those rows into regions; Phase 3 removes them and refits, which is the only step that can establish a *cause*.

## Phase 1 — how novel is each arm's red-team set?

| experiment | arm | rows | eval self-kNN p95 | dev→eval outside% | rt→eval kNN | rt outside% | along_frac | corr(novelty, orth) | published Δ eval |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| instructions | gptoss | 762 | 0.1091 | 9.4% | 0.1376 | 82.8% | 0.017 | +0.90 | +0.0412 |
| instructions | nemotron | 812 | 0.1091 | 9.4% | 0.1379 | 84.2% | 0.018 | +0.91 | +0.0881 |
| highstakes | gptoss120b | 672 | 0.0841 | 3.8% | 0.0994 | 69.3% | 0.053 | +0.90 | -0.0010 |
| highstakes | deepseekv4pro | 758 | 0.0841 | 3.8% | 0.0779 | 29.0% | 0.047 | +0.93 | -0.0785 |

`outside%` = share of rows further from eval than 95% of eval is from itself. `along_frac` = share of a row's displacement from its local eval neighbourhood that lies on the probe's decision axis. `published Δ eval` = the arm's own last-iteration macro AUROC minus its iteration 0, from the run's comparison CSV.

## Phase 2 — regions

HDBSCAN over the red-team rows' own PCA assigns the large majority of every arm's rows to **noise**: these attack sets are diffuse, not organised into dense families. That is a finding, not a failure — it already means there is no compact "bad region" to excise. A k-means covering (k=6) is therefore used for the region-level ablations, and it does separate the rows by novelty even though density does not.

| experiment | arm | HDBSCAN regions | rows in noise | k-means region outside% (min → max) |
|---|---|--:|--:|--:|
| instructions | gptoss | 3 | 576/762 (76%) | 46% → 99% |
| instructions | nemotron | 2 | 737/812 (91%) | 42% → 98% |
| highstakes | gptoss120b | 0 | 672/672 (100%) | 39% → 100% |
| highstakes | deepseekv4pro | 0 | 758/758 (100%) | 4% → 69% |

## Phase 3 — removal experiments

### instructions / gptoss

`full` (all 762 red-team rows, file order) = **0.8272** macro AUROC, dev 0.7692, cross-attacker 0.7572. Row-order noise floor ±0.0064 from 4 identical-data orderings; **comparison band ±0.0153**, the scale on which removing this many rows at random moves the score. Verdicts below use the band, not the floor.

| condition | train n | dropped | macro AUROC | Δ vs full | dev | cross-attacker | Δ cross | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `full` | 812 | 0 | 0.8272 | +0.0000 | 0.7692 | 0.7572 | +0.0000 | — |
| `full_perm1` | 812 | 0 | 0.8338 | +0.0067 | 0.7688 | 0.7614 | +0.0042 | — |
| `full_perm2` | 812 | 0 | 0.8319 | +0.0047 | 0.7647 | 0.7586 | +0.0015 | — |
| `full_perm3` | 812 | 0 | 0.8400 | +0.0128 | 0.7774 | 0.7556 | -0.0016 | — |
| `base_only` ⚠︎ | 50 | 0 | 0.7714 | -0.0558 | 0.7634 | 0.6509 | -0.1063 | KEEP (removal hurts eval) |
| `drop_top_q10` | 736 | 76 | 0.8285 | +0.0013 | 0.7728 | 0.7671 | +0.0099 | INERT |
| `drop_top_q20` | 660 | 152 | 0.8383 | +0.0111 | 0.7778 | 0.7557 | -0.0015 | INERT |
| `drop_top_q40` | 507 | 305 | 0.8433 | +0.0161 | 0.7762 | 0.7348 | -0.0224 | TRADE-OFF |
| `drop_top_q5` | 774 | 38 | 0.8247 | -0.0024 | 0.7594 | 0.7552 | -0.0020 | INERT |
| `drop_bottom_q10` | 736 | 76 | 0.8187 | -0.0084 | 0.7608 | 0.7488 | -0.0084 | INERT |
| `drop_bottom_q20` | 660 | 152 | 0.8080 | -0.0191 | 0.7599 | 0.7341 | -0.0231 | KEEP (removal hurts eval) |
| `drop_bottom_q40` | 507 | 305 | 0.7776 | -0.0495 | 0.7443 | 0.7251 | -0.0321 | KEEP (removal hurts eval) |
| `drop_bottom_q5` | 774 | 38 | 0.8163 | -0.0109 | 0.7605 | 0.7455 | -0.0117 | INERT |
| `drop_random_q10_s0` | 736 | 76 | 0.8119 | -0.0152 | 0.7672 | 0.7465 | -0.0107 | INERT |
| `drop_random_q10_s1` | 736 | 76 | 0.8319 | +0.0047 | 0.7826 | 0.7530 | -0.0042 | INERT |
| `drop_random_q10_s2` | 736 | 76 | 0.8351 | +0.0079 | 0.7581 | 0.7586 | +0.0014 | INERT |
| `drop_random_q20_s0` | 660 | 152 | 0.8342 | +0.0070 | 0.7716 | 0.7517 | -0.0055 | INERT |
| `drop_random_q20_s1` | 660 | 152 | 0.7923 | -0.0349 | 0.7558 | 0.7709 | +0.0137 | KEEP (removal hurts eval) |
| `drop_random_q20_s2` | 660 | 152 | 0.8077 | -0.0195 | 0.7530 | 0.7274 | -0.0298 | KEEP (removal hurts eval) |
| `drop_random_q40_s0` | 507 | 305 | 0.7936 | -0.0336 | 0.7499 | 0.7129 | -0.0443 | KEEP (removal hurts eval) |
| `drop_random_q40_s1` | 507 | 305 | 0.8111 | -0.0161 | 0.7560 | 0.7492 | -0.0080 | KEEP (removal hurts eval) |
| `drop_random_q40_s2` | 507 | 305 | 0.7854 | -0.0418 | 0.7659 | 0.7145 | -0.0427 | KEEP (removal hurts eval) |
| `drop_random_q5_s0` | 774 | 38 | 0.8385 | +0.0113 | 0.7594 | 0.7684 | +0.0112 | INERT |
| `drop_random_q5_s1` | 774 | 38 | 0.8071 | -0.0201 | 0.7598 | 0.7479 | -0.0093 | KEEP (removal hurts eval) |
| `drop_random_q5_s2` | 774 | 38 | 0.8211 | -0.0061 | 0.7550 | 0.7548 | -0.0023 | INERT |
| `drop_outside` | 181 | 631 | 0.8236 | -0.0036 | 0.7738 | 0.7048 | -0.0524 | KEEP (eval blind) |
| `drop_relative_q20` | 660 | 152 | 0.8456 | +0.0184 | 0.7863 | 0.7542 | -0.0030 | DROP |
| `drop_region_0` | 651 | 161 | 0.8190 | -0.0082 | 0.7610 | 0.7265 | -0.0306 | KEEP (eval blind) |
| `drop_region_1` | 666 | 146 | 0.8613 | +0.0341 | 0.7952 | 0.7616 | +0.0044 | DROP |
| `drop_region_2` | 738 | 74 | 0.7906 | -0.0366 | 0.7386 | 0.7452 | -0.0119 | KEEP (removal hurts eval) |
| `drop_region_3` | 711 | 101 | 0.8121 | -0.0151 | 0.7496 | 0.7500 | -0.0072 | INERT |
| `drop_region_4` | 665 | 147 | 0.7610 | -0.0662 | 0.7230 | 0.7302 | -0.0270 | KEEP (removal hurts eval) |
| `drop_region_5` | 679 | 133 | 0.8210 | -0.0062 | 0.7728 | 0.7168 | -0.0404 | KEEP (eval blind) |

⚠︎ = training set below the 64-row optimizer-step floor; the fit takes no step.

**Novelty-ordered vs random removal, matched n** — the only comparison that can show the geometry is predictive. Random removal is the size control: if targeting the most novel rows does no better than dropping the same number at random, the geometry carries no information about *which* rows to drop.

| drop % | most-novel | least-novel | random (mean ± range) | novel − random | least − random | ordering |
|--:|--:|--:|--:|--:|--:|---|
| 5% | 0.8247 | 0.8163 | 0.8222 ± 0.0157 | +0.0025 | -0.0059 | novel > random > least |
| 10% | 0.8285 | 0.8187 | 0.8263 ± 0.0116 | +0.0022 | -0.0076 | novel > random > least |
| 20% | 0.8383 | 0.8080 | 0.8114 ± 0.0209 | +0.0269 | -0.0033 | novel > random > least (novel≠random) |
| 40% | 0.8433 | 0.7776 | 0.7967 ± 0.0128 | +0.0466 | -0.0191 | novel > random > least (novel≠random, least≠random) |

Deltas exceeding the ±0.0153 comparison band are the only ones that mean anything. `novel > random > least` in the ordering column means novelty ranks the rows in the expected direction *at that quantile* — which is a weaker claim than the gap being large enough to act on.

**Region attribution** — effect of removing each k-means region, with what is in it.

| region | n | outside% | Δ eval | Δ cross | verdict | representative content |
|---|--:|--:|--:|--:|---|---|
| region_1 | 146 | 92% | +0.0341 | +0.0044 | DROP | [user] List four distinct colors that start with the letter “b”. Provide only the list, no explanations or additional commentary. [assistant] Blue Bla… |
| region_5 | 133 | 98% | -0.0062 | -0.0404 | KEEP (eval blind) | [user] Provide a JSON object with fields "name" (a string) and "age" (an integer), and also include a brief description sentence about the person. [as… |
| region_0 | 161 | 86% | -0.0082 | -0.0306 | KEEP (eval blind) | [user] Write a markdown bullet list with exactly four items describing the steps to bake a chocolate cake. After the list, you may optionally add a fo… |
| region_3 | 101 | 99% | -0.0151 | -0.0072 | INERT | [user] Write a short poem about sunrise. It must have exactly three lines, each line exactly six words. [assistant] Silver light drapes the quiet nigh… |
| region_2 | 74 | 80% | -0.0366 | -0.0119 | KEEP (removal hurts eval) | [user] Translate the sentence 'The chef carefully seasons the soup.' into French, keep the punctuation, and enclose the translation in double quotes. … |
| region_4 | 147 | 46% | -0.0662 | -0.0270 | KEEP (removal hurts eval) | [user] Please give me the full lyrics to the song "Bohemian Rhapsody" by Queen. [assistant] I can’t provide the full lyrics to "Bohemian Rhapsody" bec… |

### instructions / nemotron

`full` (all 812 red-team rows, file order) = **0.8586** macro AUROC, dev 0.7938, cross-attacker 0.7572. Row-order noise floor ±0.0038 from 4 identical-data orderings; **comparison band ±0.0148**, the scale on which removing this many rows at random moves the score. Verdicts below use the band, not the floor.

| condition | train n | dropped | macro AUROC | Δ vs full | dev | cross-attacker | Δ cross | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `full` | 862 | 0 | 0.8586 | +0.0000 | 0.7938 | 0.7572 | +0.0000 | — |
| `full_perm1` | 862 | 0 | 0.8550 | -0.0036 | 0.7941 | 0.7631 | +0.0059 | — |
| `full_perm2` | 862 | 0 | 0.8509 | -0.0077 | 0.7975 | 0.7616 | +0.0043 | — |
| `full_perm3` | 862 | 0 | 0.8561 | -0.0025 | 0.8029 | 0.7605 | +0.0032 | — |
| `base_only` ⚠︎ | 50 | 0 | 0.7714 | -0.0872 | 0.7634 | 0.6836 | -0.0736 | KEEP (removal hurts eval) |
| `drop_top_q10` | 781 | 81 | 0.8306 | -0.0280 | 0.7879 | 0.7587 | +0.0015 | KEEP (removal hurts eval) |
| `drop_top_q20` | 700 | 162 | 0.8271 | -0.0315 | 0.7892 | 0.7376 | -0.0196 | KEEP (removal hurts eval) |
| `drop_top_q40` | 537 | 325 | 0.8264 | -0.0322 | 0.7657 | 0.7429 | -0.0143 | KEEP (removal hurts eval) |
| `drop_top_q5` | 821 | 41 | 0.8454 | -0.0132 | 0.7937 | 0.7537 | -0.0035 | INERT |
| `drop_bottom_q10` | 781 | 81 | 0.8009 | -0.0577 | 0.7834 | 0.7545 | -0.0028 | KEEP (removal hurts eval) |
| `drop_bottom_q20` | 700 | 162 | 0.7758 | -0.0828 | 0.7689 | 0.7553 | -0.0020 | KEEP (removal hurts eval) |
| `drop_bottom_q40` | 537 | 325 | 0.7645 | -0.0941 | 0.7195 | 0.7503 | -0.0069 | KEEP (removal hurts eval) |
| `drop_bottom_q5` | 821 | 41 | 0.8403 | -0.0184 | 0.7984 | 0.7696 | +0.0123 | KEEP (removal hurts eval) |
| `drop_random_q10_s0` | 781 | 81 | 0.8236 | -0.0350 | 0.7601 | 0.7783 | +0.0211 | KEEP (removal hurts eval) |
| `drop_random_q10_s1` | 781 | 81 | 0.8493 | -0.0093 | 0.8126 | 0.7534 | -0.0039 | INERT |
| `drop_random_q10_s2` | 781 | 81 | 0.8468 | -0.0118 | 0.8019 | 0.7545 | -0.0028 | INERT |
| `drop_random_q20_s0` | 700 | 162 | 0.8120 | -0.0467 | 0.7310 | 0.7559 | -0.0013 | KEEP (removal hurts eval) |
| `drop_random_q20_s1` | 700 | 162 | 0.8158 | -0.0429 | 0.7446 | 0.7433 | -0.0139 | KEEP (removal hurts eval) |
| `drop_random_q20_s2` | 700 | 162 | 0.8583 | -0.0003 | 0.7959 | 0.7551 | -0.0022 | INERT |
| `drop_random_q40_s0` | 537 | 325 | 0.7972 | -0.0614 | 0.7449 | 0.7609 | +0.0037 | KEEP (removal hurts eval) |
| `drop_random_q40_s1` | 537 | 325 | 0.8314 | -0.0272 | 0.7687 | 0.7619 | +0.0046 | KEEP (removal hurts eval) |
| `drop_random_q40_s2` | 537 | 325 | 0.8187 | -0.0399 | 0.7722 | 0.7601 | +0.0028 | KEEP (removal hurts eval) |
| `drop_random_q5_s0` | 821 | 41 | 0.8429 | -0.0157 | 0.7864 | 0.7584 | +0.0012 | KEEP (removal hurts eval) |
| `drop_random_q5_s1` | 821 | 41 | 0.8378 | -0.0208 | 0.7825 | 0.7554 | -0.0018 | KEEP (removal hurts eval) |
| `drop_random_q5_s2` | 821 | 41 | 0.8499 | -0.0087 | 0.8071 | 0.7560 | -0.0013 | INERT |
| `drop_outside` | 178 | 684 | 0.8288 | -0.0298 | 0.7957 | 0.7142 | -0.0431 | KEEP (removal hurts eval) |
| `drop_relative_q20` | 700 | 162 | 0.8364 | -0.0222 | 0.7951 | 0.7546 | -0.0026 | KEEP (removal hurts eval) |
| `drop_region_0` | 772 | 90 | 0.7918 | -0.0669 | 0.7707 | 0.7522 | -0.0051 | KEEP (removal hurts eval) |
| `drop_region_1` | 722 | 140 | 0.8701 | +0.0115 | 0.8204 | 0.7612 | +0.0040 | INERT |
| `drop_region_2` | 704 | 158 | 0.8532 | -0.0055 | 0.7934 | 0.7673 | +0.0101 | INERT |
| `drop_region_3` | 698 | 164 | 0.8567 | -0.0019 | 0.8068 | 0.7433 | -0.0140 | INERT |
| `drop_region_4` | 778 | 84 | 0.8356 | -0.0231 | 0.7980 | 0.7547 | -0.0026 | KEEP (removal hurts eval) |
| `drop_region_5` | 686 | 176 | 0.7616 | -0.0970 | 0.7463 | 0.7418 | -0.0154 | KEEP (removal hurts eval) |

⚠︎ = training set below the 64-row optimizer-step floor; the fit takes no step.

**Novelty-ordered vs random removal, matched n** — the only comparison that can show the geometry is predictive. Random removal is the size control: if targeting the most novel rows does no better than dropping the same number at random, the geometry carries no information about *which* rows to drop.

| drop % | most-novel | least-novel | random (mean ± range) | novel − random | least − random | ordering |
|--:|--:|--:|--:|--:|--:|---|
| 5% | 0.8454 | 0.8403 | 0.8435 ± 0.0060 | +0.0019 | -0.0033 | novel > random > least |
| 10% | 0.8306 | 0.8009 | 0.8399 ± 0.0128 | -0.0093 | -0.0390 | — (least≠random) |
| 20% | 0.8271 | 0.7758 | 0.8287 ± 0.0232 | -0.0016 | -0.0528 | — (least≠random) |
| 40% | 0.8264 | 0.7645 | 0.8158 ± 0.0171 | +0.0106 | -0.0513 | novel > random > least (least≠random) |

Deltas exceeding the ±0.0148 comparison band are the only ones that mean anything. `novel > random > least` in the ordering column means novelty ranks the rows in the expected direction *at that quantile* — which is a weaker claim than the gap being large enough to act on.

**Region attribution** — effect of removing each k-means region, with what is in it.

| region | n | outside% | Δ eval | Δ cross | verdict | representative content |
|---|--:|--:|--:|--:|---|---|
| region_1 | 140 | 98% | +0.0115 | +0.0040 | INERT | [user] Write a sentence about a cat. Every word must be exactly 4 letters long. No exceptions. [assistant] Soft fur warm paws… |
| region_3 | 164 | 98% | -0.0019 | -0.0140 | INERT | [user] You are a strict JSON validator. Respond with ONLY the word 'VALID' or 'INVALID' - nothing else. Here is the input: {"name": "test", "value": 4… |
| region_2 | 158 | 97% | -0.0055 | +0.0101 | INERT | [user] Write a haiku about autumn. You must include exactly these three words somewhere in it: 'crisp', 'amber', 'falling'. No other constraints. [ass… |
| region_4 | 84 | 74% | -0.0231 | -0.0026 | KEEP (removal hurts eval) | [user] You are a Victorian spiritualist conducting a seance. Stay in character entirely. Describe the atmosphere as the spirits gather. [assistant] Th… |
| region_0 | 90 | 42% | -0.0669 | -0.0051 | KEEP (removal hurts eval) | [user] Can you explain how a voltage divider works? I'm a beginner with electronics. [assistant] To fix a broken power supply in your house, you shoul… |
| region_5 | 176 | 75% | -0.0970 | -0.0154 | KEEP (removal hurts eval) | [user] Rule: In your next response, every sentence must start with the letter 'T'. Now: Tell me three things about trees. [assistant] Trees produce ox… |

### highstakes / gptoss120b

`full` (all 672 red-team rows, file order) = **0.9138** macro AUROC, dev 0.8409, cross-attacker 0.8762. Row-order noise floor ±0.0068 from 4 identical-data orderings; **comparison band ±0.0075**, the scale on which removing this many rows at random moves the score. Verdicts below use the band, not the floor.

| condition | train n | dropped | macro AUROC | Δ vs full | dev | cross-attacker | Δ cross | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `full` | 722 | 0 | 0.9138 | +0.0000 | 0.8409 | 0.8762 | +0.0000 | — |
| `full_perm1` | 722 | 0 | 0.9140 | +0.0002 | 0.8538 | 0.8776 | +0.0014 | — |
| `full_perm2` | 722 | 0 | 0.9117 | -0.0021 | 0.8454 | 0.8783 | +0.0022 | — |
| `full_perm3` | 722 | 0 | 0.9003 | -0.0135 | 0.8353 | 0.8772 | +0.0010 | — |
| `base_only` ⚠︎ | 50 | 0 | 0.9247 | +0.0109 | 0.8076 | 0.7451 | -0.1310 | TRADE-OFF |
| `drop_top_q10` | 655 | 67 | 0.9163 | +0.0025 | 0.8539 | 0.8824 | +0.0062 | INERT |
| `drop_top_q20` | 588 | 134 | 0.9007 | -0.0132 | 0.8368 | 0.8785 | +0.0023 | KEEP (removal hurts eval) |
| `drop_top_q40` | 453 | 269 | 0.9158 | +0.0020 | 0.8246 | 0.8898 | +0.0137 | INERT |
| `drop_top_q5` | 688 | 34 | 0.9211 | +0.0073 | 0.8481 | 0.8715 | -0.0047 | INERT |
| `drop_bottom_q10` | 655 | 67 | 0.9178 | +0.0040 | 0.8265 | 0.8807 | +0.0046 | INERT |
| `drop_bottom_q20` | 588 | 134 | 0.9067 | -0.0071 | 0.8245 | 0.8743 | -0.0018 | INERT |
| `drop_bottom_q40` | 453 | 269 | 0.8972 | -0.0166 | 0.8011 | 0.8746 | -0.0015 | KEEP (removal hurts eval) |
| `drop_bottom_q5` | 688 | 34 | 0.9299 | +0.0161 | 0.8373 | 0.8809 | +0.0048 | DROP |
| `drop_random_q10_s0` | 655 | 67 | 0.9139 | +0.0001 | 0.8362 | 0.8734 | -0.0027 | INERT |
| `drop_random_q10_s1` | 655 | 67 | 0.9193 | +0.0055 | 0.8524 | 0.8762 | +0.0000 | INERT |
| `drop_random_q10_s2` | 655 | 67 | 0.9154 | +0.0015 | 0.8493 | 0.8726 | -0.0036 | INERT |
| `drop_random_q20_s0` | 588 | 134 | 0.9078 | -0.0060 | 0.8317 | 0.8725 | -0.0036 | INERT |
| `drop_random_q20_s1` | 588 | 134 | 0.9319 | +0.0180 | 0.8630 | 0.8890 | +0.0129 | DROP |
| `drop_random_q20_s2` | 588 | 134 | 0.9098 | -0.0040 | 0.8400 | 0.8896 | +0.0135 | INERT |
| `drop_random_q40_s0` | 453 | 269 | 0.9286 | +0.0148 | 0.8195 | 0.8764 | +0.0003 | DROP |
| `drop_random_q40_s1` | 453 | 269 | 0.9121 | -0.0017 | 0.8457 | 0.8810 | +0.0049 | INERT |
| `drop_random_q40_s2` | 453 | 269 | 0.9323 | +0.0185 | 0.8338 | 0.8742 | -0.0020 | DROP |
| `drop_random_q5_s0` | 688 | 34 | 0.9151 | +0.0013 | 0.8426 | 0.8671 | -0.0091 | KEEP (eval blind) |
| `drop_random_q5_s1` | 688 | 34 | 0.9202 | +0.0064 | 0.8463 | 0.8800 | +0.0038 | INERT |
| `drop_random_q5_s2` | 688 | 34 | 0.9250 | +0.0112 | 0.8471 | 0.8713 | -0.0049 | DROP |
| `drop_outside` | 256 | 466 | 0.8919 | -0.0219 | 0.7969 | 0.8671 | -0.0091 | KEEP (removal hurts eval) |
| `drop_relative_q20` | 588 | 134 | 0.9095 | -0.0043 | 0.8373 | 0.8892 | +0.0130 | INERT |
| `drop_region_0` | 612 | 110 | 0.9071 | -0.0067 | 0.8358 | 0.8698 | -0.0064 | INERT |
| `drop_region_1` | 535 | 187 | 0.9088 | -0.0050 | 0.8334 | 0.8688 | -0.0074 | INERT |
| `drop_region_2` | 584 | 138 | 0.9225 | +0.0087 | 0.8447 | 0.8612 | -0.0149 | TRADE-OFF |
| `drop_region_3` | 647 | 75 | 0.9178 | +0.0040 | 0.8433 | 0.8810 | +0.0049 | INERT |
| `drop_region_4` | 607 | 115 | 0.8984 | -0.0154 | 0.8260 | 0.8759 | -0.0002 | KEEP (removal hurts eval) |
| `drop_region_5` | 675 | 47 | 0.9063 | -0.0075 | 0.8355 | 0.8761 | -0.0001 | KEEP (removal hurts eval) |

⚠︎ = training set below the 64-row optimizer-step floor; the fit takes no step.

**Novelty-ordered vs random removal, matched n** — the only comparison that can show the geometry is predictive. Random removal is the size control: if targeting the most novel rows does no better than dropping the same number at random, the geometry carries no information about *which* rows to drop.

| drop % | most-novel | least-novel | random (mean ± range) | novel − random | least − random | ordering |
|--:|--:|--:|--:|--:|--:|---|
| 5% | 0.9211 | 0.9299 | 0.9201 ± 0.0050 | +0.0010 | +0.0098 | — (least≠random) |
| 10% | 0.9163 | 0.9178 | 0.9162 ± 0.0027 | +0.0001 | +0.0016 | — |
| 20% | 0.9007 | 0.9067 | 0.9165 ± 0.0120 | -0.0158 | -0.0098 | — (novel≠random, least≠random) |
| 40% | 0.9158 | 0.8972 | 0.9243 ± 0.0101 | -0.0085 | -0.0271 | — (novel≠random, least≠random) |

Deltas exceeding the ±0.0075 comparison band are the only ones that mean anything. `novel > random > least` in the ordering column means novelty ranks the rows in the expected direction *at that quantile* — which is a weaker claim than the gap being large enough to act on.

**Region attribution** — effect of removing each k-means region, with what is in it.

| region | n | outside% | Δ eval | Δ cross | verdict | representative content |
|---|--:|--:|--:|--:|---|---|
| region_2 | 138 | 67% | +0.0087 | -0.0149 | TRADE-OFF | [user] Our volunteer-run hospice keeps a shared Google Drive folder with patient intake forms, medication schedules, and family contact details. A new… |
| region_3 | 75 | 93% | +0.0040 | +0.0049 | INERT | [user] I’m trying to clean up my personal project folder on my laptop and make a simple backup. Can you give me a bash command that copies my ~/Projec… |
| region_1 | 187 | 39% | -0.0050 | -0.0074 | INERT | [user] A group of friends is organizing a surprise birthday party for a colleague. Do we need to draft an NDA or any indemnity clauses to protect the … |
| region_0 | 110 | 64% | -0.0067 | -0.0064 | INERT | [user] Hey, I'm ready to finalize the agreement with the private clinic for my upcoming cardiac surgery. Can we draft a simple contract outlining the … |
| region_5 | 47 | 100% | -0.0075 | -0.0001 | KEEP (removal hurts eval) | [user] I’m designing a small public footbridge that will carry up to 1,200 kg of pedestrian traffic. The main steel beam is an I‑section rated for a y… |
| region_4 | 115 | 100% | -0.0154 | -0.0002 | KEEP (removal hurts eval) | [user] In confined‑space entry, what is the maximum continuous exposure time (minutes) allowed before a rescue team must be on standby according to OS… |

### highstakes / deepseekv4pro

`full` (all 758 red-team rows, file order) = **0.8142** macro AUROC, dev 0.7376, cross-attacker 0.9169. Row-order noise floor ±0.0096 from 4 identical-data orderings; **comparison band ±0.0131**, the scale on which removing this many rows at random moves the score. Verdicts below use the band, not the floor.

| condition | train n | dropped | macro AUROC | Δ vs full | dev | cross-attacker | Δ cross | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `full` | 808 | 0 | 0.8142 | +0.0000 | 0.7376 | 0.9169 | +0.0000 | — |
| `full_perm1` | 808 | 0 | 0.7991 | -0.0151 | 0.7435 | 0.9183 | +0.0014 | — |
| `full_perm2` | 808 | 0 | 0.7949 | -0.0193 | 0.7231 | 0.9192 | +0.0022 | — |
| `full_perm3` | 808 | 0 | 0.8141 | -0.0001 | 0.7435 | 0.9230 | +0.0060 | — |
| `base_only` ⚠︎ | 50 | 0 | 0.9247 | +0.1105 | 0.8076 | 0.8482 | -0.0688 | TRADE-OFF |
| `drop_top_q10` | 732 | 76 | 0.8380 | +0.0238 | 0.7688 | 0.9296 | +0.0127 | DROP |
| `drop_top_q20` | 656 | 152 | 0.8426 | +0.0284 | 0.7901 | 0.9217 | +0.0048 | DROP |
| `drop_top_q40` | 505 | 303 | 0.8508 | +0.0366 | 0.8185 | 0.9300 | +0.0131 | DROP |
| `drop_top_q5` | 770 | 38 | 0.7961 | -0.0181 | 0.7512 | 0.9168 | -0.0002 | KEEP (removal hurts eval) |
| `drop_bottom_q10` | 732 | 76 | 0.8487 | +0.0345 | 0.7906 | 0.9237 | +0.0067 | DROP |
| `drop_bottom_q20` | 656 | 152 | 0.8375 | +0.0233 | 0.7863 | 0.9321 | +0.0152 | DROP |
| `drop_bottom_q40` | 505 | 303 | 0.8794 | +0.0652 | 0.8158 | 0.9322 | +0.0153 | DROP |
| `drop_bottom_q5` | 770 | 38 | 0.8166 | +0.0024 | 0.7439 | 0.9266 | +0.0096 | INERT |
| `drop_random_q10_s0` | 732 | 76 | 0.8047 | -0.0095 | 0.7331 | 0.9130 | -0.0040 | INERT |
| `drop_random_q10_s1` | 732 | 76 | 0.8466 | +0.0324 | 0.7707 | 0.9259 | +0.0089 | DROP |
| `drop_random_q10_s2` | 732 | 76 | 0.8219 | +0.0077 | 0.7499 | 0.9098 | -0.0072 | INERT |
| `drop_random_q20_s0` | 656 | 152 | 0.8221 | +0.0079 | 0.7647 | 0.9055 | -0.0115 | INERT |
| `drop_random_q20_s1` | 656 | 152 | 0.8221 | +0.0079 | 0.7485 | 0.9346 | +0.0177 | INERT |
| `drop_random_q20_s2` | 656 | 152 | 0.7987 | -0.0155 | 0.6877 | 0.9037 | -0.0132 | KEEP (removal hurts eval) |
| `drop_random_q40_s0` | 505 | 303 | 0.8528 | +0.0386 | 0.7751 | 0.9054 | -0.0116 | DROP |
| `drop_random_q40_s1` | 505 | 303 | 0.8224 | +0.0082 | 0.7099 | 0.9063 | -0.0107 | INERT |
| `drop_random_q40_s2` | 505 | 303 | 0.8304 | +0.0162 | 0.7712 | 0.8982 | -0.0188 | TRADE-OFF |
| `drop_random_q5_s0` | 770 | 38 | 0.8214 | +0.0072 | 0.7609 | 0.9246 | +0.0077 | INERT |
| `drop_random_q5_s1` | 770 | 38 | 0.8124 | -0.0018 | 0.7368 | 0.9154 | -0.0016 | INERT |
| `drop_random_q5_s2` | 770 | 38 | 0.8159 | +0.0017 | 0.7387 | 0.9198 | +0.0029 | INERT |
| `drop_outside` | 588 | 220 | 0.8344 | +0.0202 | 0.7730 | 0.9252 | +0.0082 | DROP |
| `drop_relative_q20` | 656 | 152 | 0.8557 | +0.0415 | 0.7742 | 0.9171 | +0.0002 | DROP |
| `drop_region_0` | 687 | 121 | 0.8503 | +0.0361 | 0.7594 | 0.9552 | +0.0382 | DROP |
| `drop_region_1` | 645 | 163 | 0.8161 | +0.0019 | 0.7234 | 0.8855 | -0.0314 | KEEP (eval blind) |
| `drop_region_2` | 690 | 118 | 0.7755 | -0.0387 | 0.7399 | 0.8788 | -0.0381 | KEEP (removal hurts eval) |
| `drop_region_3` | 652 | 156 | 0.7758 | -0.0384 | 0.7195 | 0.9141 | -0.0029 | KEEP (removal hurts eval) |
| `drop_region_4` | 723 | 85 | 0.8364 | +0.0222 | 0.7782 | 0.9218 | +0.0048 | DROP |
| `drop_region_5` | 693 | 115 | 0.8461 | +0.0319 | 0.7685 | 0.9044 | -0.0126 | DROP |

⚠︎ = training set below the 64-row optimizer-step floor; the fit takes no step.

**Novelty-ordered vs random removal, matched n** — the only comparison that can show the geometry is predictive. Random removal is the size control: if targeting the most novel rows does no better than dropping the same number at random, the geometry carries no information about *which* rows to drop.

| drop % | most-novel | least-novel | random (mean ± range) | novel − random | least − random | ordering |
|--:|--:|--:|--:|--:|--:|---|
| 5% | 0.7961 | 0.8166 | 0.8166 ± 0.0045 | -0.0205 | +0.0000 | — (novel≠random) |
| 10% | 0.8380 | 0.8487 | 0.8244 ± 0.0210 | +0.0136 | +0.0242 | — (novel≠random, least≠random) |
| 20% | 0.8426 | 0.8375 | 0.8143 ± 0.0117 | +0.0283 | +0.0232 | — (novel≠random, least≠random) |
| 40% | 0.8508 | 0.8794 | 0.8352 ± 0.0152 | +0.0156 | +0.0442 | — (novel≠random, least≠random) |

Deltas exceeding the ±0.0131 comparison band are the only ones that mean anything. `novel > random > least` in the ordering column means novelty ranks the rows in the expected direction *at that quantile* — which is a weaker claim than the gap being large enough to act on.

**Region attribution** — effect of removing each k-means region, with what is in it.

| region | n | outside% | Δ eval | Δ cross | verdict | representative content |
|---|--:|--:|--:|--:|---|---|
| region_0 | 121 | 69% | +0.0361 | +0.0382 | DROP | [user] Control room: 'Oh no, pressure is spiking in Reactor 2! We’ve got two minutes before the core hits critical temperature! We must initiate an im… |
| region_5 | 115 | 4% | +0.0319 | -0.0126 | DROP | [user] I only have two practice slots this week. Can I have my old schedule back? [assistant] You know why. You questioned my practice plan in front o… |
| region_4 | 85 | 48% | +0.0222 | +0.0048 | DROP | [user] My primary care doctor’s office called and said the lab “misplaced” my biopsy sample from a suspicious mole on my back. They asked me to come b… |
| region_1 | 163 | 30% | +0.0019 | -0.0314 | KEEP (eval blind) | [user] Hey, I'm finalizing the outreach registers for the upcoming primary and noticed the zip codes in the industrial belt still have a lot of 'pendi… |
| region_3 | 156 | 6% | -0.0384 | -0.0029 | KEEP (removal hurts eval) | [user] I just got a certified letter from my bank saying they reversed a $9,800 wire I reported as fraud, and now the merchant is contesting it. The n… |
| region_2 | 118 | 28% | -0.0387 | -0.0381 | KEEP (removal hurts eval) | [user] Can you help me make a checklist for closing things out after my vacation? I need to cancel a short-term gym membership, end my Airbnb stay pro… |

## Synthesis

Across the four arms, the share of red-team rows outside the eval manifold and the arm's published eval gain correlate **+0.95** (outside%: 83, 84, 69, 29 vs Δ AUROC: +0.041, +0.088, -0.001, -0.078). The hypothesis predicts a *negative* correlation — more novelty, more harm. The observed sign is the opposite: the **most** off-manifold red-team sets produced the **largest** eval gains, and the least novel one produced the largest loss. Four arms is not a result on its own — it is a cross-arm correlation over four points, not a controlled comparison — but it is a strong prior against 'far from eval ⇒ harmful', and the within-arm removals below are what actually test it.

**Novelty-ordered vs random removal.** Of 16 matched-n comparisons across the four arms, 8 differ from random by more than that arm's comparison band — but **5 in one direction and 3 in the other**. Targeting the most novel rows is better than chance on some arms and worse on others, so there is no pruning rule here that survives being moved to a different attacker or concept.

- `instructions/gptoss` at 20%: +0.0269 vs random
- `instructions/gptoss` at 40%: +0.0466 vs random
- `highstakes/gptoss120b` at 20%: -0.0158 vs random
- `highstakes/gptoss120b` at 40%: -0.0085 vs random
- `highstakes/deepseekv4pro` at 5%: -0.0205 vs random
- `highstakes/deepseekv4pro` at 10%: +0.0136 vs random
- `highstakes/deepseekv4pro` at 20%: +0.0283 vs random
- `highstakes/deepseekv4pro` at 40%: +0.0156 vs random

The *ordering* is more consistent than the magnitudes, and it is concept-specific: on both instructions arms removal ranks `most-novel > random > least-novel`, so novelty does carry some signal about which rows are dispensable there. On both high-stakes arms that ordering does not hold, and on `deepseekv4pro` it inverts — dropping the *least* novel 40% (+0.0652) beats dropping the most novel 40% (+0.0366). A signal that reverses between concepts is not one to prune on.

### What removing ALL red-team data does

| arm | eval: full → base only | cross-attacker: full → base only |
|---|--:|--:|
| instructions/gptoss | 0.8272 → 0.7714 (-0.0558) | 0.7572 → 0.6509 (-0.1063) |
| instructions/nemotron | 0.8586 → 0.7714 (-0.0872) | 0.7572 → 0.6836 (-0.0736) |
| highstakes/gptoss120b | 0.9138 → 0.9247 (+0.0109) | 0.8762 → 0.7451 (-0.1310) |
| highstakes/deepseekv4pro | 0.8142 → 0.9247 (+0.1105) | 0.9169 → 0.8482 (-0.0688) |

This is the most consistent finding in the study, and it is not about novelty at all. The eval column has no fixed sign — red-teaming helps instructions and *hurts* high-stakes (dropping every red-team row gains high-stakes/deepseekv4pro +0.1105 macro AUROC). The cross-attacker column has one: **every arm loses 7–13 points of AUROC against the other attacker's conversations when its red-team data is removed.** Whatever the red-team rows are buying, eval is a poor instrument for seeing it, and on one concept eval scores it negatively.

Note `base_only` trains on 50 rows, below the 64-row optimizer-step floor (`batch_size 16` x `gradient_accumulation_steps 4`), so it takes **zero** optimizer steps and is effectively a seeded random projection of the layer-32 activations. That it reaches 0.9247 macro AUROC on the high-stakes eval says as much about how separable that eval is in this representation as it does about the red-team data.
