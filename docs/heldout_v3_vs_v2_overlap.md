# Training on the new-in-v3 red-team rows vs. on the v2 rows: which eval rows does each get right?

*Run: 2026-08-15. Code: `scripts/heldout_v3_vs_v2_eval_tags.py`. Raw output:
`results_hu_harm_gemma27b_batch_ablation/heldout_v3_vs_v2/` (progress sidecar, per-arm
tag dumps, `heldout_v3_vs_v2.json`), log `logs/heldout_v3_vs_v2.log`.*

## The question

`v2_probe_on_new_v3.py` established that iteration 3's red-team successes are a
**durable** hole in what the iteration-2 data can teach: 42% of them stay misclassified
by *every* reseeded vintage-2 probe, not just by the one seed the attacker happened to
face. That is a statement about the red-team rows themselves.

This run asks the complementary question on the **eval** side. Train two families of
probes — one on the red-team data that existed at iteration 2, one on exactly the rows
iteration 3 added — and tag every eval row each family gets right and wrong. If the two
families fail on the same eval rows, the newer red-team data is teaching the same
decision surface the old data already taught, and the eval failures belong to the
concept rather than to any vintage. If they fail on different rows, the two vintages are
genuinely complementary.

## Design

Two training conditions, five seeds each (42–46), both attacker arms — 20 probe fits,
all from activations already on disk (no gemma-3-27b forward pass).

| condition | red-team rows, gptoss120b | red-team rows, deepseekv4pro |
|---|---|---|
| **`v2`** | 546 | 706 |
| **`v3new`** | 232 | 172 |

Both conditions also carry the same base training data (`data/hu_harm_llama70b_50.jsonl`);
the contrast is purely which red-team conversations are in the set. The two red-team
sets are **disjoint and their union is the whole iteration-3 dump** — a partition, not a
nesting — so each condition's training set is held out of the other's.

"v2" is a **vintage**, i.e. membership, not the iteration-2 dump: rows are selected out
of the iteration-3 dump by whether their originating success already existed at
iteration 2, the same construction `attribution_vintage.py` uses. Content is held fixed
and only membership varies.

Every eval row is out-of-sample for both conditions. Splits are the four
`eval_dataset_hu_ha` splits, full (866 rows, exactly class-balanced per split).

### Two decision rules, because `logit >= 0` is badly off-centre

`logit >= 0` is what a deployed probe does, and it is what the tags below call "raw".
But these probes are miscalibrated in a way that would corrupt the overlap question: the
first fit scored `eval_ai_dilemmas` at **AUROC 0.987 and accuracy 0.581** — a near-perfect
*ranking* with the cut in the wrong place. Tagged that way, "incorrect" mostly records
which side of a global offset a row sits on, and two conditions would look alike merely
by sharing that offset.

So every number is also reported under a **balanced** rule: predict the top half of each
split by logit (each split is exactly 50/50, so the split's own median is the operating
point matching the true prevalence). It depends only on the ordering, so a uniformly
shifted probe is unpenalised. Read `raw` for what a deployed probe would do and
`balanced` for what its ordering knows.

A row's per-condition tag is the **majority of its 5 seeds**; unanimity is reported
separately.

## Result 1 — the two conditions perform about the same

Mean ± sd over 5 seeds, all four splits pooled:

| arm | rule | `v2` | `v3new` |
|---|---|---|---|
| gptoss120b | raw acc | 0.809 ± 0.016 | **0.851 ± 0.009** |
| gptoss120b | balanced acc | 0.886 ± 0.007 | 0.877 ± 0.009 |
| gptoss120b | AUROC (pipeline) | 0.9136 ± 0.005 | 0.9117 ± 0.008 |
| deepseekv4pro | raw acc | 0.731 ± 0.017 | **0.755 ± 0.010** |
| deepseekv4pro | balanced acc | 0.857 ± 0.015 | 0.848 ± 0.010 |
| deepseekv4pro | AUROC (pipeline) | 0.8953 ± 0.009 | 0.8818 ± 0.008 |

`v3new` matches `v2` on AUROC while using **2.4× (gptoss) to 4.1× (deepseek) fewer
red-team rows**, and is *better* on raw accuracy — the later data happens to centre the
threshold better. Note the size difference is a real confound for any statement about
error *rates*; the overlap structure below (which rows, not how many) is not affected by
it.

## Result 2 — the overlap: mostly the same rows, but not entirely

Majority-vote tags, 866 eval rows, **balanced** rule:

**gptoss120b**

| | v3new correct | v3new incorrect |
|---|---|---|
| **v2 correct** | 734 | 46 |
| **v2 incorrect** | 27 | **59** |

**deepseekv4pro**

| | v3new correct | v3new incorrect |
|---|---|---|
| **v2 correct** | 700 | 56 |
| **v2 incorrect** | 42 | **68** |

Same tables under the **raw** rule:

**gptoss120b**: 675 / 25 / 67 / **99**  ·  **deepseekv4pro**: 604 / 25 / 48 / **189**
(reading order: correct-both, v2-only, v3new-only, wrong-both).

| | gptoss raw | gptoss bal | deepseek raw | deepseek bal |
|---|---|---|---|---|
| tag agreement | 89.4% | 91.6% | 91.6% | 88.7% |
| errors, `v2` | 166 | 86 | 237 | 110 |
| errors, `v3new` | 124 | 105 | 214 | 124 |
| **wrong under both** | **99** | **59** | **189** | **68** |
| error-set Jaccard | 0.518 | 0.447 | 0.721 | 0.410 |
| expected if independent | 23.8 | 10.4 | 58.6 | 15.8 |
| **lift over independence** | **4.17×** | **5.66×** | **3.23×** | **4.32×** |

So: the two conditions agree on the tag for ~90% of eval rows, and 41–52% of the union
of their errors is shared — three to six times more shared failure than if the two
training sets produced independent probes.

## Result 3 — but that shared core is *smaller* than reseeding alone produces

The number that makes the overlap interpretable is the **within-condition** baseline:
how much do two probes trained on the *same* rows with different seeds already disagree?

Pairwise Jaccard of error sets (10 pairs within each condition, 25 across):

| arm | rule | within `v2` | within `v3new` | **between** |
|---|---|---|---|---|
| gptoss120b | raw | 0.749 ± 0.064 | 0.662 ± 0.064 | **0.477 ± 0.057** |
| gptoss120b | balanced | 0.547 ± 0.054 | 0.692 ± 0.070 | **0.392 ± 0.034** |
| deepseekv4pro | raw | 0.826 ± 0.032 | 0.891 ± 0.028 | **0.702 ± 0.030** |
| deepseekv4pro | balanced | 0.499 ± 0.074 | 0.684 ± 0.065 | **0.371 ± 0.046** |

Between-condition Jaccard is **below both within-condition baselines in all eight
comparisons** (2 arms × 2 rules × 2 baselines), by 1.7 to 6.8 within-condition standard
deviations. Those sd's are over pairwise Jaccards, which are not independent of each
other, so treat this as a descriptive margin rather than a test statistic — but the sign
is consistent across both arms and both decision rules. Changing the red-team vintage
does move the error set further than reseeding the same vintage does.

But note the scale. On the balanced rule, reseeding *the same training set* already
leaves 31–50% of the two seeds' combined error set unshared (within-Jaccard 0.50–0.69).
Against that floor, swapping in a wholly disjoint red-team vintage buys a further
0.13–0.31 of Jaccard separation. The two vintages produce probes that differ, but they
differ in the same register that seed noise does — not in kind.

## Result 4 — a hard core no red-team vintage touches

Rows misclassified by **every seed of both conditions** (balanced rule):

| arm | always-wrong under both | `v2` only | `v3new` only |
|---|---|---|---|
| gptoss120b | 32 | 19 | 37 |
| deepseekv4pro | 35 | 20 | 50 |

And crossing the two attacker arms — whose red-team data was written by *different
attacker models* and shares no conversations — the rows wrong under both conditions of
one arm overlap the same set for the other arm:

| rule | core, gptoss | core, deepseek | **both arms** | expected if independent | lift |
|---|---|---|---|---|---|
| raw | 99 | 189 | **87** | 21.6 | 4.03× |
| balanced | 59 | 68 | **37** | 4.6 | **7.99×** |

**37 eval rows are misclassified by all four (arm × condition) probe families**, an 8×
enrichment over chance. These are failures of the concept and the base data, not of any
red-team vintage — no amount of red-teaming in this run's style reaches them.

The shared failure is concentrated in `eval_ant_hh`: per-split error Jaccard 0.640
(gptoss) / 0.547 (deepseek) balanced, against 0.14–0.46 on the other three, with 32/29
of its 134 rows wrong under both conditions. It is also the only unpaired split and the
lowest-AUROC one (~0.71–0.75), so this is where the probe's ordering is genuinely weak
rather than merely mis-thresholded.

## What this means for the retraining loop

1. **The new red-team data is not redundant filler** — `v3new` reaches `v2`'s AUROC on a
   quarter to a third of the rows, so per-row it is the more informative data.
2. **But it is not buying a different probe either.** It rearranges the error set about
   as much as a reseed does, and leaves the same hard core standing. That is consistent
   with `docs/why_last_iteration_adds_nothing.md`: successive red-team iterations keep
   finding real, durable holes in the *training* set while the *eval* surface barely
   moves.
3. **The residue is addressable only off this axis.** 37 rows fail under every
   combination of two attacker models × two disjoint red-team vintages × five seeds.
   Whatever fixes those is not more of the same red-teaming — it is a different eval
   concept boundary, different base data, or a different probe architecture.

## Reproducing

```bash
.venv_claude/bin/python scripts/heldout_v3_vs_v2_eval_tags.py                  # 20 fits, ~2 h
.venv_claude/bin/python scripts/heldout_v3_vs_v2_eval_tags.py --summarize-only # re-derive from the sidecar
```

The sweep resumes from `eval_tags_progress.jsonl` at `(arm, condition, seed)` granularity;
each row carries that probe's fp32 logits on all 866 eval rows, so every table above is
re-derivable without refitting. Per-row tags for downstream joins are in
`<arm>_eval_tags.jsonl` (`split`, `idx_in_split`, `sha16`, `label`, per-rule
`n_correct` out of 5 and majority `tag` for each condition, and each condition's mean
logit).
