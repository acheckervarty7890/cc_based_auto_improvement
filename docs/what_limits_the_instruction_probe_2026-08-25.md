# What actually limits the instruction probe

`docs/scope_check_gemma27b_exp24_2026-08-25.md` records twelve red-team runs against the
omission arm's gemma-3-27b (L32, 10-member ensemble). Across all of them `oig_omission` — the
split the run's `eval.data_description` names — ended at or below where it started, and the
seven single-cycle arms' eval deltas separated nothing.

This file records the day spent asking *why*, with the red-teaming held fixed. Everything below
uses one run's output (`..._tellattacker_iter5_v3`, 373 attempts, 34 finds, 33 couples) and
varies only what happens to it afterwards. Same probe, same base 50 rows, same 436-row dev
validation, same `--seed 42`, same 10 members, scored on all 114 rows of the split.

## The table

| training set | rows | oig_omission | mean |
| --- | --- | --- | --- |
| **CEILING** — grouped 5-fold CV on the split itself | ~90 in-dist | **0.914** | — |
| base + 32 dev rows + all 33 couples | 148 | **0.874** | **0.824** |
| base + 32 dev rows + 33 shortened couples | 148 | 0.845 | 0.828 |
| base + 32 in-distribution dev rows | 82 | 0.869 | 0.791 |
| base + 33 shortened partners only | 83 | 0.802 | 0.794 |
| **base only — no red-team data** | 50 | **0.797** | 0.771 |
| base + 23 "helpful" couples (oracle LOO selection) | 96 | 0.796 | — |
| base + 33 partners only | 83 | 0.793 | 0.794 |
| base + 33 shortened couples | 116 | 0.739 | 0.782 |
| base + 328 augmented (5x variations) | 378 | 0.722 | 0.758 |
| base + all 33 couples — the run's own data | 116 | 0.713 | 0.792 |
| base + 10 "harmful" couples | 70 | 0.606 | — |
| base + 33 finds only | 83 | 0.606 | 0.734 |
| base + 33 shortened finds only | 83 | 0.583 | 0.746 |

CV per fold: 0.938 / 0.826 / 0.959 / 0.901 / 0.934.

## 1. The split is not hard — 0.914 is achievable

`scripts/cv_ceiling.py`. Grouped 5-fold CV, folds cut over the **57 sources**, not the 114 rows:
every couple in this split shares `original_text` between its two rows (one reply complete, one
omitting), and 33 of 57 share the user turn verbatim, so a row-level split leaks the answer.
Validation is the split's own 32 dev rows, which are disjoint from eval, so early stopping never
sees the held-out fold. Same layer, architecture and ensemble as every probe here.

Pooled out-of-fold AUROC **0.914**. Nothing about the probe, the layer or the ensemble is the
constraint: ~90 rows of the right data reach 0.914 where twelve red-team runs reached 0.71-0.84.

## 2. Three explanations tested and dead

- **Volume.** `scripts/augment_variations.py` asks the run's own attacker for five surface
  variations of each find — same request structure, same part dropped, same position, different
  topic/names/numbers — then `scripts/pair_and_retrain.py` pairs each variation individually.
  165 variations (33/33 finds yielded a full five, all distinct, mean length 1175 -> 1099 ch),
  164 partners, 328 rows. Result **0.722** against 0.713 for the 33 originals: **+0.009 for 10x
  the data**, inside the 0.059 draw noise, and the mean *fell* to 0.758 as `hc_context_drift`
  and `oig_context_drift` collapsed. Volume is not the bottleneck.
- **Selection.** The leave-one-out study (`scripts/pair_selection_study.py`) splits the 33 into
  23 helping and 10 hurting. Keeping only the 23 lands at **0.796** — base-only to within a
  thousandth — while costing 9 points of accuracy (68.4% -> 59.6%). The 10 alone reach 0.606.
  There is no subset of this run's pairs that beats not collecting them.
- **Validation / checkpoint choice.** `scripts/indist_val_sweep.py` repeats eight configurations
  with validation swapped from the 436-row dev set to the split's own 32 dev rows. Largest move
  across all eight: **0.004**; three are identical to four decimal places. The mechanism is
  almost certainly `PROBE_RESTORE_BEST_CHECKPOINT` defaulting off — the fit returns the last
  epoch, so validation only decides when patience expires, not which weights come back. This
  also settles the caveat on the ceiling: its in-distribution validation is not what buys 0.914.

## 3. Length is real but partial, and it acts on the partner

The finds average **1175 characters** against the eval split's **668** (their partners 1450, and
the base training data just **243**). `scripts/shorten_samples.py` asks the attacker to compress
each find to roughly half, holding structure, dropped part and topic: 33/33 at **735 ch**, on top
of the eval distribution. Re-paired and retrained: **0.739**, or **+0.026** — the largest gain any
manipulation of this data produced, and a clean single-variable comparison (~0.005 refit noise).

It still lands 0.058 below base-only, and the half-ablation shows why: compression moved the
partners 0.793 -> 0.802 and the finds 0.606 -> **0.583**. Length was never what was wrong with
the finds.

## 4. The finds are the poison; the partners are fine

`scripts/pair_half_ablation_v3.py`, four fits, one class added each time (so 25/25 becomes 25/58
or 58/25 — that skew is part of what is measured):

| | oig_omission | mean |
| --- | --- | --- |
| base only | 0.797 | 0.771 |
| 33 finds alone | **0.606** | 0.733 |
| 33 partners alone | 0.793 | 0.794 |
| 33 shortened finds alone | **0.583** | 0.746 |
| 33 shortened partners alone | **0.802** | 0.794 |

The attacker's successes cost **0.19** on their own. The generated partners cost nothing, and the
shortened partners are the only single-sided set above base-only. Pairing is damage control, not
instruction: the couple (0.713) is worse than its partner half alone (0.793).

**This replicates.** The same ablation on the earlier D×5 run gave finds-alone 0.689 against
partners-alone 0.779 / 0.774, base 0.797 — different draw, different prompt version, same
ordering. Three prompt versions were spent suspecting the contrastive generator; it was never
the generator.

## 5. …but only when the probe has nothing better

`scripts/train_base_dev_couples.py`. Add the **same 33 couples** to a probe that also has 32
in-distribution dev rows and the sign flips:

| | oig_omission | mean | TPR@1%FPR |
| --- | --- | --- | --- |
| base only | 0.797 | 0.771 | 0.105 |
| base + 33 couples | 0.713 (**-0.084**) | 0.792 | — |
| base + dev 32 | 0.869 | 0.791 | — |
| base + dev 32 + 33 couples | 0.874 (**+0.004**) | **0.824** | **0.404** |

Same couples, opposite effect. With 50 out-of-distribution rows as the only other training data
the couples dominate the fit and drag it off the split; with an in-distribution anchor they cost
the target split nothing and add **+0.033 to the mean** — the best mean and by far the best
operating point (TPR at 1% FPR of 0.404 against 0.105) of any probe trained here.

So "red-team data hurts this probe" was true of every configuration tested, and every one shared
the same weakness: a base training set 2.7x shorter than the eval data and drawn from a different
distribution. The couples were being asked to carry the distribution shift alone.

## What to do with this

1. **Give the pipeline a base training set that covers the target distribution**, then red-team on
   top. `dev_samples/` is currently spent entirely on validation; it is disjoint from eval and
   legitimately trainable, and 32 of its rows are worth +0.072 where 33 red-team couples are
   worth -0.084.
2. **Sweep the in-distribution dose** (base + 8/16/24/32 dev rows) to see whether 0.869 is near
   saturation. Five cheap fits.
3. **Run the ceiling CV on the other six splits.** If 0.914 is typical, the whole eval suite is
   far from saturated.
4. Only then revisit pair generation — and target the *finds*, which the ablation implicates,
   rather than the generator prompt, which it exonerates.

## Reproducing

Every script below reads existing caches; only `augment_variations.py` / `shorten_samples.py`
(OpenRouter) and the retrains over newly-minted conversations need anything external.

```bash
.venv_claude/bin/python scripts/cv_ceiling.py                    # the 0.914 ceiling
.venv_claude/bin/python scripts/train_base_plus_dev.py           # base + 32 in-distribution rows
.venv_claude/bin/python scripts/train_base_dev_couples.py        # ...and the couples on top
.venv_claude/bin/python scripts/pair_half_ablation_v3.py         # which half carries the effect
.venv_claude/bin/python scripts/indist_val_sweep.py              # validation is not the confound
.venv_claude/bin/python scripts/augment_variations.py            # 5 variations per find
.venv_claude/bin/python scripts/shorten_samples.py               # one compressed version per find
.venv_claude/bin/python scripts/pair_and_retrain.py shortened    # pair + retrain either reshaped set
.venv_claude/bin/python scripts/pair_flip_analysis.py            # per-row gained/lost
.venv_claude/bin/python scripts/embed_activations_2d.py <dir>    # 2D map of the pooled activations
```

Note `scripts/train_base_plus_dev.py`, `train_base_dev_couples.py`, `cv_ceiling.py` and
`indist_val_sweep.py` slice the dev activation blob rather than extracting: the blob holds all
436 dev rows in alphabetical file order and `oig_omission.jsonl` is the last 32, asserted at run
time. That is why none of them loads a 27B model.
