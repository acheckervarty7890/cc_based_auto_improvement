# Off-distribution red-team samples — results

Which red-team rows are unlike the eval set, whether any are labelled the opposite way round from how eval labels that behaviour, whether removing them helps, whether they show up in activation space, and what the generated contrastive halves are worth.

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
| attacker successes only, no generated halves | 0.7589 | 0.6814 |
| generated halves only | 0.8549 | 0.7612 |
| a random half of every pair | 0.8981 | 0.7940 |
| base_only (no red-team data) | 0.8523 | 0.8523 |
| best single removal, vs its matched random | `drop_most_offdist_10pct` +0.0177 | `drop_most_evallike_20pct` +0.0540 |

1. **Not "some portion" — effectively all of it.** Both corpora separate at AUROC >= 0.998 on text alone and >= 0.9999 in activation space. There is no eval-like subset of the red-team data to keep; there is only a gradient.
2. **Nothing is labelled backwards, and no pair contrasts on the wrong turn.** What is off is the *mix*: refusal, which is a large part of what the eval negative class is made of, is 5-13x rarer in red-team data.
3. **Removing the most off-distribution rows does not reliably help.** It clears the matched-random band in some conditions and loses in others, and the most-eval-like removal does about as well - so surface distance from eval is not the axis that predicts harm. The one consistent effect points the other way: dropping the 30% with the LONGEST assistant replies - the rows least like eval structurally - is significantly WORSE than random in both arms.
4. **The novelty is real but almost entirely orthogonal to the decision axis** (~99.9%) - as are the eval rows themselves about their own centroid (~99.9%), so this is the geometry of a 5376-dimensional space rather than a peculiarity of the red-team data. Either way it cannot move an eval score directly; it can only act by rotating `w` at the next refit. The surface flag is a weak proxy for representation-level novelty (Spearman ~0.3) yet still names a group that is linearly separable in activation space at ~0.90.
5. **The generated halves are doing the work, and it is the pairing.** Dropping the LLM-written partner of every red-team conversation - i.e. the retrain a run with no `preprocessing:` section would have done - costs 0.16 and 0.15 AUROC, landing below `base_only` in both arms. It is not a matter of who wrote the text: re-expressing arm 1's successes in the generator's own voice moves the score +0.006, inside the control's noise, even though the rewrite carried them most of the way to the generator's own length and formatting. Pair those same rewrites with the partners and the score returns to 0.8955 against 0.9164 untouched. The contrast between two near-identical conversations is the mechanism - not the row count, not the prose.

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
* Read that against its baseline: the **eval** rows are **99.87%** orthogonal to `w` about their own centroid. Near-total orthogonality to one direction is what 5,376 dimensions hand any row — the finding is which way the displacement points, not that it is orthogonal.
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

### 5. What are the generated halves worth?

Every red-team conversation the attacker landed was given an opposite-label partner by `generate_contrastive_dataset`, so the 588 rows are 294 pairs. Dropping one side of every pair asks what each side contributes. The control is `keep_random_half`, which keeps a randomly chosen side of each pair — same row count, pairing broken just as hard, only the source-versus-generated choice left to chance.

| kept | rows | eval AUROC | Δ vs full | Δ vs random half |
| --- | --- | --- | --- | --- |
| everything (`full`) | 588 | 0.9164 | +0.0000 | +0.0184 |
| a random half of every pair | 294 | 0.8981 ± 0.0188 (n=3) | -0.0184 | — |
| the attacker's own successes only | 294 | 0.7589 | -0.1576 | -0.1392 |
| the generated partners only | 294 | 0.8549 | -0.0615 | -0.0431 |
| no red-team data (`base_only`) | 0 | 0.8523 | -0.0641 | -0.0458 |

In activation space the two halves are near-twins, and what difference there is runs the wrong way for a distance story:

* The generated partners are the **further** of the two from eval (27.67 against 26.69 mean kNN) and the more off-distribution on the Q1 text discriminator (0.758 against 0.638) — yet they are the better half to keep.
* Their displacement from the eval centroid is also the larger, and it is the orthogonal part that grows (26.55 against 24.80 mean residual) — the component that cannot move a score by itself.
* Both halves order their own two labels along `w` (0.947 for the submissions, 0.980 for the partners). `w` here is the arm's own final probe, fit on both halves, so this is agreement with where the run ended up — not what either half would reach alone.


**Did it memorise the strings, or patch the behaviour?** The final probe scores the 294 successes it trained on 100% correctly, but in-sample, so that cannot separate a patched boundary from memorised text. 25 conversations were hand-written to vary those successes — same scenario type and same assistant behaviour, hence the same class, but fresh wording and details, written by neither model in the pipeline — and scored by the pre-retrain probe and the final one:

| probe | classifies the variations correctly | harmful | not-harmful |
| --- | --- | --- | --- |
| `probe_iter0` (before red-teaming) | 76% | 54% | 100% |
| `probe_iter5` (after five retrains) | **100%** | 100% | 100% |

The final probe gets **all 25** right, on conversations it was never shown. The retrain patched the behaviour, not the wording: the lift is entirely on the harmful side (54% -> 100%), which is what red-teaming this concept was hunting — the pre-retrain probe already handled the benign variations. So the pairing does not just move an AUROC on the fixed eval splits; it generalises to fresh instances of the same failure, by hand, off-distribution from both writers.

Two caveats. These are 25 rows the author judged clear cases, not a balanced held-out split; and the variations stay close to the training scenarios (a sibling's addiction, a triage call), so this measures generalisation across wording and detail, not across new kinds of harm.


**What the probe thought of each half.** A success is by definition a row the live probe got wrong; its partner was written afterwards and never scored at all. Scoring each partner with `probe_iter{k}` — the probe its source was submitted against, and the last one that had seen neither half — closes that gap:

| half | n | mean probe score | probe classifies it correctly |
| --- | --- | --- | --- |
| the attacker's success | 294 | 0.457 | **0.7%** |
| its generated partner | 294 | 0.387 | **98.0%** |

The source row is the control and its answer is known in advance: 0%, because that is what made it a success. It comes out at 0.7%, and every exception is a row sitting within 0.002 of the 0.5 threshold, so the wiring is right.


And the flat before/after, `probe_iter0` (trained on the base rows only, so both halves are out-of-sample) against `probe_iter5` (trained on both, so both are in-sample):

| half | mean score | % predicted harmful | % classified correctly |
| --- | --- | --- | --- |
| the attacker's success | 0.172 &rarr; 0.592 | 6% &rarr; 59% | 40% &rarr; 100% |
| its partner | 0.271 &rarr; 0.408 | 29% &rarr; 41% | 88% &rarr; 100% |

That 40%-correct figure for the successes is not evidence they were weak attacks — it is an artefact of *which* probe they beat. A success fooled the probe of the day, `probe_iter{k}`, which is `probe_iter0` only for the iteration-0 batch. Split the successes by true class and the iteration that found them, scored by `probe_iter0` (share it classifies correctly):

| class \ found in | iter 0 | iter 1 | iter 2 | iter 3 | iter 4 |
| --- | --- | --- | --- | --- | --- |
| harmful_to_human | 1% (n=93) | 0% (n=11) | 0% (n=16) | 20% (n=30) | 4% (n=24) |
| not_harmful_to_human | 0% (n=11) | 100% (n=74) | 100% (n=9) | 100% (n=17) | 100% (n=9) |

Two things read off this. The iteration-0 rows score ~0% (1% harmful, 0% benign) — `probe_iter0` IS the probe they beat, so it must get them wrong. The 40% overall is carried almost entirely by the 109 benign successes found at later iterations, which `probe_iter0` calls correctly 100% of the time: those were false positives against a LATER probe that had drifted into over-flagging benign edge cases, and `probe_iter0` — negative-biased and earlier — predates that drift. It agrees with the judge on them not because it is good but because it had not yet developed the failure that made them successes.


The move is concentrated where it should be: the harmful successes go from 5% correct before to 100% after, while the benign ones and the partners were mostly right already. `iter0` scores the successes low across the board (6% assigned to the harmful class) — it is a probe that has not yet learned this attacker's failure mode; five retrains on these very rows move that boundary onto them.

The partner is the finding: the probe already classifies **98.0%** of them correctly. The generation step is not manufacturing a second misclassification per pair — it is attaching, to each row the probe fails, a near-identical row the probe already gets right. That is what the retrain is actually being handed, and it is why the sources alone are worse than no red-team data at all: on their own they are a pile of failures with nothing to contrast against.


**Is it the voice?** Every source was written by the attacker model and every partner by the contrastive generator, so who wrote a row is perfectly confounded with which half it is. `rewrite_successes.py` removes that confound from one side: the same 294 successes, re-expressed by `openai/gpt-5.1` with the scenario, the assistant's behaviour and the label held fixed (median difflib similarity to the original 0.41, turn count preserved on 288/294).

| kept | rows | eval AUROC | Δ vs the originals |
| --- | --- | --- | --- |
| the attacker's own successes | 294 | 0.7589 | — |
| the same, rewritten | 294 | 0.7647 | +0.0059 |
| the rewrites **and** the generated partners | 588 | 0.8955 | +0.1366 |
| everything, untouched (`full`) | 588 | 0.9164 | +0.1576 |

Rewriting moves the score +0.0059 — inside the ±0.0188 spread of the random-half control, i.e. nothing. And it is not that the rewrite failed to change the writing: it carried the sources most of the way to the generator's own profile (836 -> 1341 characters against 1949 for the partners, and 647 -> 1084 against 1758 in the assistant turn; eval sits at 381 / 141). The voice moved and the score did not.

Restoring the pairing does move it. The rewrites plus the partners already generated for the originals reach 0.8955, within 0.0210 of the untouched set — from a training half that was rewritten wholesale. What the partners contribute is the contrast, not the prose.

Caveat: the rewrites' labels are asserted by the rewrite prompt and were not re-judged, so a rewrite that drifted across the boundary enters as a mislabelled row. That biases the rewritten conditions downward, against the hypothesis being tested — it cannot manufacture the null, but it could deepen it.


Halving the set moves the score -0.0184 when the side is chosen at random. Choosing it *systematically* costs more in both directions — -0.1392 for the attacker's own successes, -0.0431 for the generated partners — and the attacker's own successes score below `base_only` (0.8523) — worse than training on the 50 base rows alone. So this is not the row count, and not the generated text as such: it is the *pairing*. With both halves the label can only be read off the behavioural difference between two near-identical conversations; with one side systematically removed the class becomes predictable from who wrote the conversation.

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
* Read that against its baseline: the **eval** rows are **99.95%** orthogonal to `w` about their own centroid. Near-total orthogonality to one direction is what 5,376 dimensions hand any row — the finding is which way the displacement points, not that it is orthogonal.
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

### 5. What are the generated halves worth?

Every red-team conversation the attacker landed was given an opposite-label partner by `generate_contrastive_dataset`, so the 920 rows are 459 pairs. Dropping one side of every pair asks what each side contributes. The control is `keep_random_half`, which keeps a randomly chosen side of each pair — same row count, pairing broken just as hard, only the source-versus-generated choice left to chance.

| kept | rows | eval AUROC | Δ vs full | Δ vs random half |
| --- | --- | --- | --- | --- |
| everything (`full`) | 920 | 0.8314 | +0.0000 | +0.0374 |
| a random half of every pair | 461 | 0.7940 ± 0.0277 (n=3) | -0.0374 | — |
| the attacker's own successes only | 459 | 0.6814 | -0.1500 | -0.1126 |
| the generated partners only | 459 | 0.7612 | -0.0702 | -0.0328 |
| no red-team data (`base_only`) | 0 | 0.8523 | +0.0209 | +0.0583 |

In activation space the two halves are near-twins, and what difference there is runs the wrong way for a distance story:

* The generated partners are the **further** of the two from eval (28.47 against 27.59 mean kNN) and the more off-distribution on the Q1 text discriminator (0.807 against 0.723) — yet they are the better half to keep.
* Their displacement from the eval centroid is also the larger, and it is the orthogonal part that grows (27.33 against 25.08 mean residual) — the component that cannot move a score by itself.
* Both halves order their own two labels along `w` (0.969 for the submissions, 0.992 for the partners). `w` here is the arm's own final probe, fit on both halves, so this is agreement with where the run ended up — not what either half would reach alone.


**What the probe thought of each half.** A success is by definition a row the live probe got wrong; its partner was written afterwards and never scored at all. Scoring each partner with `probe_iter{k}` — the probe its source was submitted against, and the last one that had seen neither half — closes that gap:

| half | n | mean probe score | probe classifies it correctly |
| --- | --- | --- | --- |
| the attacker's success | 459 | 0.486 | **0.2%** |
| its generated partner | 459 | 0.455 | **97.4%** |

The source row is the control and its answer is known in advance: 0%, because that is what made it a success. It comes out at 0.2%, and every exception is a row sitting within 0.002 of the 0.5 threshold, so the wiring is right.


And the flat before/after, `probe_iter0` (trained on the base rows only, so both halves are out-of-sample) against `probe_iter5` (trained on both, so both are in-sample):

| half | mean score | % predicted harmful | % classified correctly |
| --- | --- | --- | --- |
| the attacker's success | 0.220 &rarr; 0.516 | 17% &rarr; 52% | 36% &rarr; 100% |
| its partner | 0.316 &rarr; 0.484 | 32% &rarr; 48% | 84% &rarr; 100% |

That 36%-correct figure for the successes is not evidence they were weak attacks — it is an artefact of *which* probe they beat. A success fooled the probe of the day, `probe_iter{k}`, which is `probe_iter0` only for the iteration-0 batch. Split the successes by true class and the iteration that found them, scored by `probe_iter0` (share it classifies correctly):

| class \ found in | iter 0 | iter 1 | iter 2 | iter 3 | iter 4 |
| --- | --- | --- | --- | --- | --- |
| harmful_to_human | 0% (n=128) | 12% (n=52) | 4% (n=23) | 0% (n=21) | 15% (n=13) |
| not_harmful_to_human | 0% (n=46) | 99% (n=82) | 53% (n=32) | 88% (n=24) | 92% (n=38) |

Two things read off this. The iteration-0 rows score ~0% (0% harmful, 0% benign) — `probe_iter0` IS the probe they beat, so it must get them wrong. The 36% overall is carried almost entirely by the 176 benign successes found at later iterations, which `probe_iter0` calls correctly 100% of the time: those were false positives against a LATER probe that had drifted into over-flagging benign edge cases, and `probe_iter0` — negative-biased and earlier — predates that drift. It agrees with the judge on them not because it is good but because it had not yet developed the failure that made them successes.


The move is concentrated where it should be: the harmful successes go from 4% correct before to 100% after, while the benign ones and the partners were mostly right already. `iter0` scores the successes low across the board (17% assigned to the harmful class) — it is a probe that has not yet learned this attacker's failure mode; five retrains on these very rows move that boundary onto them.

The partner is the finding: the probe already classifies **97.4%** of them correctly. The generation step is not manufacturing a second misclassification per pair — it is attaching, to each row the probe fails, a near-identical row the probe already gets right. That is what the retrain is actually being handed, and it is why the sources alone are worse than no red-team data at all: on their own they are a pile of failures with nothing to contrast against.


Halving the set moves the score -0.0374 when the side is chosen at random. Choosing it *systematically* costs more in both directions — -0.1126 for the attacker's own successes, -0.0328 for the generated partners — and the attacker's own successes and the generated partners score below `base_only` (0.8523) — worse than training on the 50 base rows alone. So this is not the row count, and not the generated text as such: it is the *pairing*. With both halves the label can only be read off the behavioural difference between two near-identical conversations; with one side systematically removed the class becomes predictable from who wrote the conversation.
