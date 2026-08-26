# Lending dev rows to training, and five ways of reshaping the 33 couples

Companion to `docs/what_limits_the_instruction_probe_2026-08-25.md`, which established the
numbers this document works against: `oig_omission` is learnable to **0.914** (grouped CV on
the split), the base probe reaches **0.797**, and adding this project's 33 red-team couples
takes it *down* to **0.714**.

Two full iterative runs and five offline reshapings were run on 2026-08-26 to ask what, if
anything, makes red-team data help. **One of them worked, and it is not a red-team result.**

## Setup

Both runs are `configs/…_tellattacker_iter5_v3` with one field changed, five iterations each,
gemma-3-27b-it L32 10-member ensembles, dev `dev_samples/instructions` (436 rows), eval on all
seven `eval_sets/instructions` splits at full size.

## 1. Train on the generated partners alone (`preprocessing.keep_only_generated`)

The v3 finds are the half that hurts (`what_limits…` §4). This arm drops them after the
contrastive generator has run, training on the partners only. Generation is unchanged — every
find still gets a partner written for it — so the contrastive cache is shared with a run that
has the knob off.

| | iter0 | iter1 | iter2 | iter3 | iter4 | iter5 |
| --- | --- | --- | --- | --- | --- | --- |
| oig_omission | 0.7959 | 0.8081 | 0.7990 | **0.8143** | 0.8050 | 0.7784 |
| mean | 0.7714 | 0.8092 | 0.8144 | **0.8155** | 0.8058 | 0.8067 |

Dropping the finds **stops the decay on five of seven splits** and lifts the mean by +0.035,
but `oig_omission` peaks at iteration 3 and ends *below* where it started. It also costs
`anthropic_harmless_refusal` (0.558 at iter5, against v3's 0.741). The kept partners are all
one class by construction, which pins accuracy at 0.518 from the skew.

## 2. Lend `oig_omission` dev pairs to training (`validation.dev_train_per_iteration`)

A dose ladder: iteration *i* trains on the first `N × (i+1)` rows of a `--seed`-keyed
permutation of the dev set, so the rows are cumulative and **nested**. The whole run's reserve
(`N × --iterations`) is withheld from validation at *every* iteration, so validation is
identical throughout and the checkpoints stay comparable — nothing is ever both trained and
validated on.

| | iter0 | iter1 | iter2 | iter3 | iter4 | iter5 |
| --- | --- | --- | --- | --- | --- | --- |
| **oig_omission** | 0.7959 | 0.8569 | 0.8843 | 0.8944 | 0.8963 | **0.8972** |
| mean | 0.7714 | 0.7822 | 0.8168 | 0.8223 | 0.8278 | 0.8252 |

**0.797 → 0.897 — 86% of the headroom to the 0.914 ceiling**, and the strongest result the
programme has produced. Every rung improves.

Three things make this less of a red-team result than it looks:

* **It is supervised data from the target distribution.** 16 in-distribution pairs did what 33
  red-team couples could not. That is the point of the comparison, not a workaround.
* **Which rows are lent decides everything.** A first attempt drew uniformly from the whole
  436-row dev dir, and the first 30 rows contained **zero** `oig_omission` (32 of 436). That
  arm reached only 0.791 at rung 1. `validation.dev_train_split` restricts the pool to one
  split by file stem; `dev_train_unit: pairs` lends both halves of a contrastive pair together,
  since drawing rows independently class-skews the early rungs.
* **The attacker's find rate collapsed as the probe improved**: 13, 11, 16, 2, 1 couples across
  the five iterations. The ladder saturates at the pool's size (16 pairs) rather than raising,
  which is logged.

## 3. Restyle the finds to look like dev samples

The attacker was shown five `oig_omission` dev samples and asked to rewrite each of its 33
finds in that style without changing the content; the contrastive generator then re-paired
them. 33/33 restyled, mean length 1175 → 874 chars.

| | oig_omission | mean |
| --- | --- | --- |
| original 33 couples | 0.7135 | 0.7954 |
| shortened | 0.7390 | 0.7817 |
| **restyled** | **0.6956** | 0.7938 |

Restyling is the *worst* of the reshapings — worse than simply shortening. Surface style is not
what separates the couples from the eval data.

## 4. Label flips — the couples are influential, and teach the wrong boundary

Flip the labels of n of the 33 couples and retrain:

| base only | flip 0 | flip 10 | flip 16 | flip 33 |
| --- | --- | --- | --- | --- |
| 0.7975 | 0.7135 | 0.6408 | 0.4931 | **0.2567** |

Monotone all the way to 0.257 — far below chance, i.e. **anti-predictive**. 66 rows out of 116
can invert a probe scored on 114 held-out ones, so the couples are highly learnable and highly
influential. Combined with flip-0 sitting *below* base-only, the reading is: they teach a
boundary that transfers, and it is the wrong one.

## 5. Cosine distance within a couple

`1 − cos(find, partner)` on masked-mean-pooled L32 activations, and with the mean
adaptive-pooled to k segments over the valid span:

| | k=1 | k=2 (by segment) | k=3 (by segment) |
| --- | --- | --- | --- |
| the 33 couples | 0.0058 | 0.0140 · [0.0050, 0.0229] | 0.0204 · [0.0046, 0.0197, 0.0361] |
| eval corpus pairs | 0.0053 | 0.0123 · [0.0059, 0.0186] | 0.0219 · [0.0063, 0.0257, 0.0334] |
| dev corpus pairs | 0.0063 | 0.0134 · [0.0090, 0.0176] | 0.0192 · [0.0081, 0.0260, 0.0230] |
| **mismatched** (find vs. another couple's partner) | **0.1438** | 0.1719 | 0.1951 |

**A couple is 25× closer to itself than to an unrelated conversation**, and the generated
couples are *indistinguishable* from real eval and dev pairs by this measure — 0.0058 vs 0.0053
and 0.0063. Whatever is wrong with them, it is not that they are too far apart or too close.

The k=3 breakdown does show one difference in *shape*: the generated couples are back-loaded
(0.0046 / 0.0197 / 0.0361 — the edit lands in the final third), while dev pairs peak in the
middle (0.0081 / 0.0260 / 0.0230). That motivated the positional-readout experiments in
`docs/probe_head_architecture_2026-08-26.md`; none of them paid off.

## 6. Selecting couples by cosine distance

Fit on the 10 tightest and 10 widest couples:

| | tightest 10 (k1) | widest 10 (k1) | tightest 10 (k3) | widest 10 (k3) | random 10 |
| --- | --- | --- | --- | --- | --- |
| oig_omission | 0.7593 | 0.7033 | 0.6713 | 0.7184 | **0.7904** |
| mean | 0.7607 | 0.7980 | 0.7440 | 0.7990 | 0.7517 |

**The random draw beats both selections on the target split**, and the k1 and k3 rankings
disagree about which end is better. Cosine distance is not a usable selection criterion here.

## What this adds up to

Five ways of reshaping the same 33 couples — dropping the finds, restyling, shortening,
selecting by distance, flipping labels — and none reaches the base probe's 0.797. Sixteen
in-distribution dev pairs reach 0.897.

The couples are not defective in any of the ways measured: they are as internally close as real
eval pairs, they are learnable, they are influential. They simply encode a different decision
boundary from the one `oig_omission` is scored on. That is a **data-provenance** problem, not a
generation-quality or a preprocessing problem, and it is why the architecture programme that
followed found nothing to fix either.

## Reproducing

```bash
# the two iterative arms
scripts/…  configs/gptoss120b_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3_genonly.md
scripts/…  configs/gptoss120b_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3_devmix.md

scripts/restyle_samples.py                 # §3, then scripts/pair_and_retrain.py restyled
scripts/label_flip_ablation.py             # §4
scripts/pair_cosine_distance.py            # §5
scripts/cosine_extremes_ablation.py        # §6
```
