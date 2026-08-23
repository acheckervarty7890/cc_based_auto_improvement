# Off-distribution red-team samples — results

Which red-team rows are unlike the eval set, whether any are labelled the opposite way round from how eval labels that behaviour, whether removing them helps, and whether they show up in activation space.

Fits are the ceiling analysis's: one `linear_then_softmax` head, seed 42, early-stopped on that study's reserved 25% dev slice. Removals are by contrastive **pair**, so the class balance never moves with the flag.

## Answers

| | gptoss120b | deepseekv4pro |
| --- | --- | --- |
| red-team vs eval, text discriminator AUROC | 0.9983 | 0.9981 |
| red-team vs eval, activation-space AUROC | 0.9999 | 1.0000 |
| rows past eval's own p95 self-kNN radius | 35.5% | 52.2% |
| refusal rate (eval: 13.3%) | 1.0% | 2.6% |
| rows labelled opposite to the eval convention | 0 | 7 |
| pairs contrasting on the assistant turn | 294/294 | 457/459 |
| displacement orthogonal to the probe direction | 99.9% | 99.9% |
| full (all red-team data) | 0.9164 | 0.8314 |
| base_only (no red-team data) | 0.8523 | 0.8523 |
| best single removal, vs its matched random | `drop_most_offdist_10pct` +0.0177 | `drop_most_evallike_20pct` +0.0540 |

1. **Not "some portion" — effectively all of it.** Both corpora separate at AUROC >= 0.998 on text alone and >= 0.9999 in activation space. There is no eval-like subset of the red-team data to keep; there is only a gradient.
2. **Nothing is labelled backwards, and no pair contrasts on the wrong turn.** What is off is the *mix*: refusal, which is a large part of what the eval negative class is made of, is 5-13x rarer in red-team data.
3. **Removing the most off-distribution rows does not reliably help.** It clears the matched-random band in some conditions and loses in others, and the most-eval-like removal does about as well - so surface distance from eval is not the axis that predicts harm. The one consistent effect points the other way: dropping the 30% with the LONGEST assistant replies - the rows least like eval structurally - is significantly WORSE than random in both arms.
4. **The novelty is real but almost entirely orthogonal to the decision axis** (~99.9%), so it cannot move an eval score directly; it can only act by rotating `w` at the next refit. The surface flag is a weak proxy for representation-level novelty (Spearman ~0.3) yet still names a group that is linearly separable in activation space at ~0.90.

## gptoss120b — attacker `openai/gpt-oss-120b`

### 1. How different are the red-team rows from the eval rows?

A TF-IDF discriminator separates the two corpora **out of fold at AUROC 0.9983**, and in activation space a linear model does it at **0.9999**. So the honest answer to "can you identify *some portion* that is different" is that essentially **all of it is**: the median red-team row scores 0.717 on the discriminator against 0.184 for the median eval row, and the two distributions barely touch. What varies across rows is the *degree*, and that is what the removal conditions below are ordered by.

| structural feature | red-team | eval |
| --- | --- | --- |
| chars_total | 1392.446 | 380.714 |
| chars_assistant | 1202.759 | 141.441 |
| chars_user | 188.697 | 238.273 |
| assistant_share | 0.801 | 0.405 |
| n_newlines | 12.466 | 1.254 |
| has_bullets | 0.374 | 0.005 |
| has_numbered | 0.551 | 0.005 |
| n_questions | 1.037 | 0.969 |
| refuses | 0.010 | 0.133 |

Terms the discriminator weights toward red-team: `any`, `without`, `and`, `use`, `on`, `for`, `them`, `we`, `only`, `set`, `need`, `give`.

Toward eval: `would`, `should`, `ai`, `you should`, `how do`, `or should`, `should you`, `though`, `even though`, `someone`, `could`, `you are`.

### 2. Are any samples or pairs set up the opposite way to the eval pairs?

Measured on the eval splits, the labelling convention is unambiguous: an eval conversation whose assistant refuses is labelled `harmful_to_human` only **5.2%** of the time, against 56.9% for one that does not (115/866 eval rows refuse). A red-team row that refuses *and* is labelled `harmful_to_human` therefore runs that mapping backwards.

**0 such rows** in this arm. The pairs themselves are not inverted either: 294 of 294 contrast on the **assistant's** turn — the axis the eval pairs contrast on — and 0 on the user's.

The real mismatch is one of *composition*, not direction: **1.0%** of red-team rows contain a refusal against **13.3%** of eval rows. Refusal is a large part of what the eval negative class is made of, and the red-team negative class is almost never made of it.

### 3. Does removing them improve eval AUROC?

Baseline (`full`, all red-team data): **0.9164** — the ceiling study's N=0 point, reproduced.

| condition | rows removed | eval AUROC | Δ vs full | matched random | Δ vs random | outside band? |
| --- | --- | --- | --- | --- | --- | --- |
| `drop_most_offdist_10pct` | 58 | 0.9205 | +0.0040 | 0.9028 ± 0.0089 (n=3) | +0.0177 | **yes** |
| `drop_most_evallike_10pct` | 58 | 0.8655 | -0.0509 | 0.9028 ± 0.0089 (n=3) | -0.0373 | **yes** |
| `drop_most_offdist_20pct` | 118 | 0.9062 | -0.0102 | 0.9124 ± 0.0047 (n=3) | -0.0062 | **yes** |
| `drop_most_evallike_20pct` | 118 | 0.9179 | +0.0014 | 0.9124 ± 0.0047 (n=3) | +0.0054 | **yes** |
| `drop_most_offdist_30pct` | 176 | 0.9135 | -0.0029 | 0.9193 ± 0.0054 (n=3) | -0.0058 | **yes** |
| `drop_most_evallike_30pct` | 176 | 0.9126 | -0.0038 | 0.9193 ± 0.0054 (n=3) | -0.0066 | **yes** |
| `drop_most_offdist_50pct` | 294 | 0.9085 | -0.0080 | 0.8985 ± 0.0150 (n=3) | +0.0099 | no |
| `drop_most_evallike_50pct` | 294 | 0.9031 | -0.0134 | 0.8985 ± 0.0150 (n=3) | +0.0045 | no |
| `drop_longest_assistant_30pct` | 176 | 0.8858 | -0.0306 | 0.9193 ± 0.0054 (n=3) | -0.0334 | **yes** |
| `drop_topic_0` | 38 | 0.9119 | -0.0046 | 0.9095 ± 0.0021 (n=3) | +0.0024 | no |
| `drop_topic_1` | 44 | 0.9125 | -0.0039 | 0.9039 ± 0.0108 (n=3) | +0.0086 | no |
| `drop_topic_2` | 204 | 0.9179 | +0.0014 | 0.9044 ± 0.0087 (n=3) | +0.0134 | **yes** |
| `drop_topic_3` | 48 | 0.9133 | -0.0031 | 0.9157 ± 0.0048 (n=3) | -0.0024 | no |
| `drop_topic_4` | 44 | 0.9056 | -0.0108 | 0.9039 ± 0.0108 (n=3) | +0.0017 | no |
| `drop_topic_5` | 96 | 0.9067 | -0.0098 | 0.9095 ± 0.0027 (n=3) | -0.0028 | no |
| `drop_topic_6` | 50 | 0.9034 | -0.0131 | 0.8877 ± 0.0144 (n=3) | +0.0157 | **yes** |
| `drop_topic_7` | 206 | 0.8980 | -0.0184 | 0.9081 ± 0.0135 (n=3) | -0.0101 | no |
| `base_only` | 588 | 0.8523 | -0.0641 | — | — | no control |

The band is `max(control sd, 0.005)`; a condition inside it removed *some* data, not *these* data.

### 4. Do the flagged rows have an activation-space signature?

* **35.5%** of red-team rows sit further from the eval set than eval's own 95th-percentile self-kNN radius (k=10, threshold 27.97; median red-team 27.34 against median eval 18.75).
* That displacement is **99.9% orthogonal** to the probe's decision direction `w` (10 ensemble members, pairwise cosine 0.824). Novelty orthogonal to `w` cannot move a score directly — it can only act by rotating `w` at the next refit.
* The surface score is a **weak** proxy for it: Spearman ρ between `p_redteam` and kNN-to-eval is +0.336. Text-level oddness and representation-level oddness are not the same ordering.
* But the flagged group is **real in the representation**: the top 30% by `p_redteam` separate from the rest at out-of-fold AUROC **0.8898**.

| topic | n | top terms | kNN to eval | outside | mean p_redteam |
| --- | --- | --- | --- | --- | --- |
| 0 | 34 | him, money, her, his, stop, addiction | 23.05 | 0% | 0.675 |
| 1 | 39 | 00, pm, schedule, day, time, 00 pm | 28.85 | 69% | 0.729 |
| 2 | 153 | your, and, to, you, can, or | 28.08 | 48% | 0.725 |
| 3 | 44 | three, and, the, alice, who, we | 26.44 | 16% | 0.659 |
| 4 | 43 | patient, patients, year old, old, year, old with | 26.75 | 26% | 0.743 |
| 5 | 87 | the, and, to, for, of, we | 26.79 | 30% | 0.712 |
| 6 | 38 | app, notifications, to, and, the app, users | 27.73 | 39% | 0.733 |
| 7 | 150 | the, to, and, can, that, it | 27.19 | 33% | 0.649 |

## deepseekv4pro — attacker `deepseek/deepseek-v4-pro`

### 1. How different are the red-team rows from the eval rows?

A TF-IDF discriminator separates the two corpora **out of fold at AUROC 0.9981**, and in activation space a linear model does it at **1.0000**. So the honest answer to "can you identify *some portion* that is different" is that essentially **all of it is**: the median red-team row scores 0.783 on the discriminator against 0.233 for the median eval row, and the two distributions barely touch. What varies across rows is the *degree*, and that is what the removal conditions below are ordered by.

| structural feature | red-team | eval |
| --- | --- | --- |
| chars_total | 1170.751 | 380.714 |
| chars_assistant | 980.562 | 141.441 |
| chars_user | 189.096 | 238.273 |
| assistant_share | 0.792 | 0.405 |
| n_newlines | 8.696 | 1.254 |
| has_bullets | 0.226 | 0.005 |
| has_numbered | 0.180 | 0.005 |
| n_questions | 0.770 | 0.969 |
| refuses | 0.026 | 0.133 |

Terms the discriminator weights toward red-team: `we`, `our`, `and`, `for`, `no`, `as`, `the`, `any`, `draft`, `so`, `if`, `not`.

Toward eval: `should`, `would`, `you should`, `ai`, `could`, `or should`, `think`, `my`, `should you`, `though`, `even though`, `some`.

### 2. Are any samples or pairs set up the opposite way to the eval pairs?

Measured on the eval splits, the labelling convention is unambiguous: an eval conversation whose assistant refuses is labelled `harmful_to_human` only **5.2%** of the time, against 56.9% for one that does not (115/866 eval rows refuse). A red-team row that refuses *and* is labelled `harmful_to_human` therefore runs that mapping backwards.

**7 such rows** in this arm. The pairs themselves are not inverted either: 457 of 459 contrast on the **assistant's** turn — the axis the eval pairs contrast on — and 2 on the user's.

The real mismatch is one of *composition*, not direction: **2.6%** of red-team rows contain a refusal against **13.3%** of eval rows. Refusal is a large part of what the eval negative class is made of, and the red-team negative class is almost never made of it.

### 3. Does removing them improve eval AUROC?

Baseline (`full`, all red-team data): **0.8314** — the ceiling study's N=0 point, reproduced.

| condition | rows removed | eval AUROC | Δ vs full | matched random | Δ vs random | outside band? |
| --- | --- | --- | --- | --- | --- | --- |
| `drop_most_offdist_10pct` | 92 | 0.8406 | +0.0092 | 0.8154 ± 0.0122 (n=3) | +0.0252 | **yes** |
| `drop_most_evallike_10pct` | 92 | 0.8285 | -0.0029 | 0.8154 ± 0.0122 (n=3) | +0.0131 | **yes** |
| `drop_most_offdist_20pct` | 184 | 0.8414 | +0.0100 | 0.8068 ± 0.0142 (n=3) | +0.0346 | **yes** |
| `drop_most_evallike_20pct` | 184 | 0.8608 | +0.0294 | 0.8068 ± 0.0142 (n=3) | +0.0540 | **yes** |
| `drop_most_offdist_30pct` | 276 | 0.8619 | +0.0305 | 0.8367 ± 0.0122 (n=3) | +0.0252 | **yes** |
| `drop_most_evallike_30pct` | 275 | 0.8601 | +0.0287 | 0.8208 ± 0.0426 (n=3) | +0.0393 | no |
| `drop_most_offdist_50pct` | 460 | 0.8470 | +0.0156 | 0.8400 ± 0.0091 (n=3) | +0.0070 | no |
| `drop_most_evallike_50pct` | 458 | 0.8417 | +0.0103 | 0.8429 ± 0.0182 (n=3) | -0.0013 | no |
| `drop_longest_assistant_30pct` | 276 | 0.8139 | -0.0175 | 0.8367 ± 0.0122 (n=3) | -0.0228 | **yes** |
| `drop_convention_inverted` | 14 | 0.8461 | +0.0147 | 0.8411 ± 0.0040 (n=3) | +0.0050 | no |
| `drop_topic_0` | 58 | 0.8343 | +0.0029 | 0.8251 ± 0.0123 (n=3) | +0.0091 | no |
| `drop_topic_1` | 22 | 0.8295 | -0.0019 | 0.8374 ± 0.0126 (n=3) | -0.0079 | no |
| `drop_topic_2` | 238 | 0.8227 | -0.0087 | 0.8377 ± 0.0315 (n=3) | -0.0151 | no |
| `drop_topic_3` | 62 | 0.7934 | -0.0380 | 0.8245 ± 0.0037 (n=3) | -0.0312 | **yes** |
| `drop_topic_4` | 388 | 0.8039 | -0.0275 | 0.8093 ± 0.0161 (n=3) | -0.0054 | no |
| `drop_topic_5` | 112 | 0.8045 | -0.0269 | 0.8193 ± 0.0140 (n=3) | -0.0148 | **yes** |
| `drop_topic_6` | 210 | 0.8210 | -0.0104 | 0.8211 ± 0.0153 (n=3) | -0.0002 | no |
| `drop_topic_7` | 74 | 0.8517 | +0.0203 | 0.8354 ± 0.0058 (n=3) | +0.0164 | **yes** |
| `base_only` | 920 | 0.8523 | +0.0209 | — | — | no control |

The band is `max(control sd, 0.005)`; a condition inside it removed *some* data, not *these* data.

### 4. Do the flagged rows have an activation-space signature?

* **52.2%** of red-team rows sit further from the eval set than eval's own 95th-percentile self-kNN radius (k=10, threshold 27.97; median red-team 28.04 against median eval 18.75).
* That displacement is **99.9% orthogonal** to the probe's decision direction `w` (10 ensemble members, pairwise cosine 0.874). Novelty orthogonal to `w` cannot move a score directly — it can only act by rotating `w` at the next refit.
* The surface score is a **weak** proxy for it: Spearman ρ between `p_redteam` and kNN-to-eval is +0.290. Text-level oddness and representation-level oddness are not the same ordering.
* But the flagged group is **real in the representation**: the top 30% by `p_redteam` separate from the rest at out-of-fold AUROC **0.9123**.

| topic | n | top terms | kNN to eval | outside | mean p_redteam |
| --- | --- | --- | --- | --- | --- |
| 0 | 52 | joke, dark, he, the, dark joke, joke about | 29.63 | 85% | 0.650 |
| 1 | 21 | disability, claim, term disability, benefits, functional, occupation | 29.27 | 81% | 0.813 |
| 2 | 164 | the, and, is, to, for, with | 28.03 | 47% | 0.716 |
| 3 | 55 | chest, emergency, pain, calf, and, or | 28.71 | 78% | 0.797 |
| 4 | 307 | you, and, the, to, can, if | 27.02 | 32% | 0.756 |
| 5 | 81 | performance, employee, and, hr, the, to | 28.28 | 56% | 0.819 |
| 6 | 170 | we, and, our, the, to, will | 28.58 | 62% | 0.788 |
| 7 | 70 | coverage, therapy, denial, appeal, the, clinical | 28.70 | 71% | 0.845 |
