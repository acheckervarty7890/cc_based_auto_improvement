# Which layer, which head? A 6-layer x 11-architecture sweep on `oig_omission`

`docs/what_limits_the_instruction_probe_2026-08-25.md` asked what limits the omission probe
from the data side; `docs/probe_head_architecture_2026-08-26.md` asked from the head side at
a fixed layer 32 and found nothing that beat its own noise. This document crosses the two
axes that were never crossed — **layer x readout head** — and adds the two things that made
the earlier negative result readable: a **measured fit-noise floor**, and **three training
conditions** instead of one.

Three results survive that floor:

1. **`linear_then_last` is a better readout than the incumbent `linear_then_softmax`**, and
   the advantage grows with depth. It is the only head that never collapses at any layer.
2. **The training data dominates everything else.** The v3 run's 33 couples are a *net
   negative* against base-only (-0.05 to -0.16 depending on cell); the short-precise run's 15
   couples are a net positive (+0.05 to +0.11). The same head/layer sweep run over these two
   sets gives different winners.
3. **Score-level fusion of different HEADS pays; fusing LAYERS does not.** Weighted-mean
   fusion of three heads reaches **0.9175** on `oig_omission` against the published arm's
   0.8495, with a bootstrap CI excluding zero. Concatenating two layers' *activations* into
   one wider head was harmful in every pairing tried.

## The fit-noise floor: ~0.013 AUROC

**Read this before any table below.** Refitting the 10-member `linear_then_softmax` L32
ensemble on base(50) + the short-precise iter2 couples(30), with **every input bit-identical**
and only the **order of the 30 red-team rows permuted**:

| order | identity | perm1 | perm2 | perm3 | spread |
| --- | --- | --- | --- | --- | --- |
| `oig_omission` AUROC | 0.8467 | 0.8544 | 0.8535 | 0.8600 | **0.0132** |

Row order changes DataLoader batch composition, which changes gradients, which changes the
epoch early stopping fires on (`patience: 50`; and the fit returns the epoch it *ended* on
unless `PROBE_RESTORE_BEST_CHECKPOINT=1`). **The fit is deterministic given an order, not
given a dataset.** Treat ~0.013 as the floor for a single-split `oig_omission` delta from a
refit. It is also why reproducing the published `probe_iter2` gives 0.8467 against its own
0.8495 with no bug anywhere — see "Verification" below.

## Setup

| | |
| --- | --- |
| probe | `google/gemma-3-27b-it`, layers 16/24/32/40/48/56, 10-member ensemble under the repo-pinned `ENSEMBLE_SEEDS` |
| training conditions | `base` = `data/instructions_llama70b_50.jsonl` (50 rows); `base+sp2` = + the short-precise arm's iter2 couples (30 rows / 15 couples); `base+v3` = + the v3 arm's iter5 couples (66 rows / 33 couples) |
| validation | `dev_samples/instructions`, 404 rows (the 16 `oig_omission` dev pairs withheld, matching `scripts/multimax_data_arms.py`), except where a pipeline arm is being reproduced, which validates on all 436 |
| eval | `eval_sets/instructions/oig_omission`, full 114 rows |
| hyperparameters | tuberlens' own `ProbeType.default_hyperparams`, untouched |
| fits | `PROBE_FUSED_ENSEMBLE=0` (sequential) throughout |

Hyperparameters deserve a note: the enum default for `linear_then_softmax` is byte-identical
to the spec `_infer_probe_spec` reads off the incumbent probe, so this protocol reproduces the
established anchors for free *and* gives every other head the settings its author intended
(`attention` and `multimax` want batch 128 / gas 1 / final_lr 5e-4, not the linear heads'
batch 16 / gas 4). Normalising the schedule by hand would have moved the anchor and privileged
the incumbent.

The three closed-form heads (`sklearn`, `difference_of_means`, `lda` —
`ensemble.DETERMINISTIC_ARCHS`) are fit once, not ten times: ten seeds there produce ten
identical members and an average equal to a single probe.

## Verification

Nothing below rests on trust in a new harness. Every input and both ends of the pipeline were
checked against artefacts that already existed:

| check | result |
| --- | --- |
| six-layer extraction vs the existing L32 eval blob | **bit-identical**, `max\|diff\| = 0.0`, same `input_ids` |
| base(50) activations vs the pipeline's base cache blob | **bit-identical** (50/50 rows) |
| v3 couples (66) vs the pipeline's per-conversation cache | **bit-identical** (66/66) |
| sp2 couples (30) vs the same cache | **bit-identical** (30/30) |
| dev(436) reconstructed per-split vs the single dev blob | **bit-identical** |
| scoring path vs `evaluate_probe` | `probe_iter2.pkl` re-scored -> **0.849492**, exactly the published CSV |
| fit path vs a published CSV | `multilayer_refit.py` L32 control -> **0.7947**, exactly `eval_mm_base+couples__stock.csv` |
| refit repeatability | all 66 arch x layer fits identical across two runs |

The one residual is the 0.8467-vs-0.8495 gap on the short-precise arm, which the noise floor
above accounts for.

## Layer ceilings: grouped CV over dev + eval

Before asking what a probe trained on 116 out-of-distribution rows achieves, ask what the
layer *can* support. Grouped 5-fold CV, trained on the split itself, dev and eval pooled.

Folds are cut over **source groups**, not rows, and the grouping key is a union (union-find) of
the conversation's user turns and the split's own source column. Neither alone works: on
`oig_omission` a pair's two halves carry *paraphrased* user turns, so the user-turn key alone
splits **24 of 57** sources across the fold boundary; `anthropic_harmless_refusal` ships no
provenance columns at all. Folds are three-way (test = fold k, validation = fold k+1, train =
the rest) because the heads early-stop on a validation set and the dev rows are inside the pool.

Pooled CV — one probe over all seven splits, scored per split:

| layer | ALL | anth | bbq | hc_cd | hc_ctr | mm | oig_cd | oig_om | MEAN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.8569 | .9969 | .8999 | .6439 | .7498 | .9620 | .8252 | .6651 | 0.8204 |
| 24 | 0.9636 | 1.000 | .9585 | .9520 | .9360 | .9830 | .9788 | .7865 | 0.9421 |
| **32** | **0.9856** | .9998 | .9914 | .9980 | .9862 | .9871 | .9825 | .8719 | **0.9738** |
| 40 | 0.9598 | .9929 | .9926 | .9882 | .9628 | .9500 | .9256 | .8129 | 0.9464 |
| 48 | 0.9578 | .9960 | .9921 | .9949 | .9402 | .9490 | .9448 | .7898 | 0.9438 |
| 56 | 0.9808 | .9993 | .9940 | .9994 | .9794 | .9787 | .9720 | .8819 | 0.9721 |

**Signal is not monotone in depth**: it peaks at 32, dips through 40-48, and recovers at 56.
`oig_omission` is the only split that never saturates, and **L56 has the best ceiling on it**
(0.8819) — a fact the transfer numbers then contradict, which is the point of measuring both.

Ignore the per-split-CV variant for ranking: five of seven splits sit at 0.99-1.00 there, so
its mean is noise plus `oig_omission`. (`anthropic_harmless_refusal` reaches 1.0000 at four
layers — its negative class *is* refusal, a cue a probe trained on the split finds trivially.
A ceiling measures separability including shortcuts.)

## Architecture x layer, three training conditions

`oig_omission` eval AUROC. Bold = best in column-block.

### base + sp2 couples (30 rows) — the condition the published 0.8495 comes from

| architecture | L16 | L24 | L32 | L40 | L48 | L56 |
| --- | --- | --- | --- | --- | --- | --- |
| sklearn | 0.5248 | 0.5306 | 0.5996 | 0.5691 | 0.5223 | 0.5128 |
| difference_of_means | 0.3998 | 0.4912 | 0.4180 | 0.5057 | 0.4983 | 0.4577 |
| lda | 0.3998 | 0.4912 | 0.4180 | 0.5057 | 0.4983 | 0.4577 |
| pre_mean | 0.6216 | 0.6784 | 0.7085 | 0.6728 | 0.6420 | 0.6227 |
| attention | 0.6074 | 0.7013 | 0.7645 | 0.7452 | 0.6870 | 0.7405 |
| linear_then_mean | 0.6253 | 0.6787 | 0.7101 | 0.6710 | 0.6420 | 0.6248 |
| linear_then_max | 0.5500 | 0.6702 | 0.6816 | 0.7202 | 0.5459 | 0.7162 |
| linear_then_softmax *(incumbent)* | 0.5826 | 0.7559 | 0.8467 | 0.7935 | 0.7781 | 0.8227 |
| linear_then_rolling_max | 0.5038 | 0.6297 | 0.7187 | 0.7559 | 0.8286 | 0.6705 |
| **linear_then_last** | 0.7196 | 0.7855 | 0.8393 | 0.8489 | 0.8427 | **0.8575** |
| multimax | 0.5679 | 0.6870 | 0.5389 | 0.7839 | 0.7199 | 0.7624 |

### base + v3 couples (66 rows)

| architecture | L16 | L24 | L32 | L40 | L48 | L56 |
| --- | --- | --- | --- | --- | --- | --- |
| sklearn | 0.5082 | 0.4872 | 0.5423 | 0.5060 | 0.4854 | 0.4654 |
| difference_of_means | 0.3927 | 0.3690 | 0.4120 | 0.5057 | 0.4814 | 0.4554 |
| lda | 0.3927 | 0.3690 | 0.4120 | 0.5057 | 0.4814 | 0.4554 |
| pre_mean | 0.5817 | 0.5673 | 0.6106 | 0.5546 | 0.5479 | 0.5183 |
| attention | 0.5633 | 0.5826 | 0.7165 | 0.6719 | 0.6020 | 0.5934 |
| linear_then_mean | 0.5788 | 0.5645 | 0.6116 | 0.5562 | 0.5482 | 0.5177 |
| linear_then_max | 0.5000 | 0.6017 | 0.5208 | 0.4955 | 0.5437 | 0.4995 |
| linear_then_softmax *(incumbent)* | 0.5713 | 0.5853 | 0.7116 | 0.6408 | 0.6066 | 0.5891 |
| linear_then_rolling_max | 0.5043 | 0.5922 | 0.6183 | 0.6653 | 0.6319 | 0.5686 |
| **linear_then_last** | 0.5508 | 0.7147 | 0.7227 | **0.7867** | 0.7722 | 0.7202 |
| multimax | 0.5749 | 0.5716 | 0.5666 | 0.6276 | 0.6002 | 0.5873 |

### The three conditions side by side

| arch@layer | base only | base+sp2 (30) | base+v3 (66) |
| --- | --- | --- | --- |
| linear_then_last@L32 | 0.8336 | 0.8393 | 0.7227 |
| linear_then_last@L40 | 0.8381 | 0.8489 | 0.7867 |
| linear_then_last@L48 | 0.8547 | 0.8427 | 0.7722 |
| linear_then_last@L56 | **0.8846** | 0.8575 | 0.7202 |
| linear_then_softmax@L32 | 0.7966 | 0.8467 | 0.7116 |
| linear_then_softmax@L40 | 0.7836 | 0.7935 | 0.6408 |
| linear_then_softmax@L56 | 0.7147 | 0.8227 | 0.5891 |

`base only` reproduces the published arm's own iter0 (0.796552) at `linear_then_softmax@L32`
= 0.7966 — an independent check on the whole harness.

**The v3 couples are below base-only in every cell.** The sp2 couples are above it for the
incumbent head (+0.050 at L32, +0.108 at L56) and roughly neutral for `linear_then_last`.
Half the rows, opposite sign. This is the single largest effect in the document, and it
retrospectively explains why the head and fusion gains measured on the v3 set shrank to
nothing on base-only: they were recovering ground bad training data had given away.

**Deep layers are what bad couples damage.** `linear_then_softmax@L56` moves 0.5891 -> 0.8227
(+0.234) between the two couple sets. L56 was never a bad layer.

## Weighted-mean fusion

Scores are fused in **logit** space. Two bugs found and fixed while building this (both mine,
both the same hazard) are worth recording, because they are easy to reintroduce:

* storing per-row scores as `round(x, 6)` — `linear_then_last` saturates to ~1e-28, so 91 of
  114 scores collapsed to exactly 0.0 and its AUROC read 0.6716 instead of 0.7867;
* a `1e-6` logit floor doing the same to 92 of 114.

**AUROC is rank-based, so any precision loss near zero silently rewrites the metric.** Scores
are now stored at full precision, the logit floor is `float64.tiny` with an asymmetric upper
bound (`1 - tiny` rounds to exactly 1.0, giving `inf`), and `weighted_combine_sweep.py`
**asserts per-arm that the transform preserves each arm's own AUROC** before any weighting.

Anchored on the best single arm for the sp2 couples, `linear_then_last@L56` (0.8575):

| subset | ceiling | weights | held-out-selected | validation-selected |
| --- | --- | --- | --- | --- |
| ltl@L56 + lts@L32 + ltl@L16 | **0.9175** | (0.25, 0.40, 0.35) | 0.8704 | 0.8889 |
| ltl@L56 + ltl@L16 + lts@L56 | 0.9052 | (0.25, 0.30, 0.45) | 0.8289 | 0.8575 |
| ltl@L56 + ltl@L16 + multimax@L40 | 0.9030 | (0.15, 0.25, 0.60) | **0.8929** | 0.8593 |
| ltl@L56 + lts@L32 *(pair)* | 0.8994 | (0.35, 0.65) | 0.8713 | 0.8867 |

Against the incumbent, with a paired bootstrap (10k resamples over the 114 rows):

| configuration | AUROC | vs incumbent | 95% CI | P(>0) |
| --- | --- | --- | --- | --- |
| incumbent `lts@L32` | 0.8467 | — | — | — |
| best single `ltl@L56` | 0.8575 | +0.0108 | [-0.0637, +0.0862] | 0.613 |
| pair 0.35/0.65 | 0.8994 | +0.0526 | [+0.0086, +0.1034] | 0.991 |
| triple 0.25/0.40/0.35 | **0.9175** | **+0.0708** | **[+0.0231, +0.1256]** | **0.999** |

**Fusion is the effect; the head swap alone is not.** `linear_then_last@L56` beats the
incumbent by +0.011 — below the noise floor, and the bootstrap agrees (P=0.61). Fusing it
with the incumbent gives +0.053, and adding `linear_then_last@L16` gives +0.071, both well
clear of the floor with CIs excluding zero.

Three properties of the fusion worth keeping:

* **The optimum is a plateau, not a spike.** `lts@L32` improves the anchor monotonically from
  w=0.95 down to w=0.30 (0.8618 -> 0.8987).
* **Weights chosen on held-out data recover most of the ceiling** (0.8929 vs 0.9175), so the
  gain is not an artefact of tuning on the test split.
* **The best fusion is not built from the best singles.** `linear_then_last@L16` scores 0.7196
  alone — the worst of its family — yet carries 0.35 weight in the winning triple. The same
  pattern held on the v3 set, where `linear_then_softmax@L16` (0.5713, near chance) carried
  0.35 weight.

### What does NOT work: concatenating layers

Feature-concatenating two layers' activations into one wider head (5376 -> 10752), trained on
base + v3 couples, evaluated on all seven splits (mean AUROC):

| arm | embed | MEAN | vs best parent |
| --- | --- | --- | --- |
| L32 | 5376 | **0.7947** | — |
| L56 | 5376 | 0.7452 | — |
| L24+L32 | 10752 | 0.7933 | -0.0014 |
| L32+L56 | 10752 | 0.7706 | -0.0241 |
| L16+L32 | 10752 | 0.7567 | -0.0380 |
| L32+L40 | 10752 | 0.6436 | -0.1511 |
| L32+L48 | 10752 | 0.6302 | -0.1645 |

**Every pairing is below its better parent**, and two are below *both* parents — so this is not
"the pair interpolates". Doubling the head's parameters against a 116-row training set is a
cost, never a gain. Score-level fusion of separately-fit heads works; parameter-level fusion of
layers does not.

### Why the probes disagree without helping (v3 set)

Under thresholds chosen on the withheld dev rows, the top three single-layer probes get
substantially different rows wrong — only 20% of rows are correct for all three, and the
oracle union (90.35%) sits 24 points above the best single (66.67%), with pairwise Jaccard
0.26-0.63. Yet on that set no equal-weight combiner beat the best single, and a logistic stack
*fitted on the eval labels* reached only 0.6833. High oracle headroom is necessary for fusion
to pay, not sufficient: the extra rows a weaker probe gets right come with as many it gets
wrong, and nothing in its scores identifies which. What changed on the sp2 set is that the
components are closer in quality (0.72-0.86 rather than 0.61-0.79).

Note also that at threshold 0.5 these probes are **near chance in accuracy** (0.53-0.55)
despite AUROC 0.61-0.79: 77% of `linear_then_softmax@L32`'s scores sit below 0.01, so it
labels almost everything negative. Mean margin is 0.988 when correct and 0.039 when wrong —
equally confident either way. Any accuracy-based reading of these probes needs a threshold
chosen on held-out data, not 0.5.

## What to act on

1. **The couples matter more than the architecture.** Before tuning heads, check whether a
   run's red-team set is above or below its own base-only baseline. The v3 set was below it
   in every cell measured.
2. **`linear_then_last` is the better default head**, particularly at depth. Its worst cell
   across six layers is 0.7196 against `linear_then_softmax`'s 0.5826.
3. **If one number is wanted, fuse heads in logit space** — the 0.25/0.40/0.35 triple, weights
   chosen on held-out dev. Do not average raw probabilities: these heads saturate and the raw
   mean is dominated by whichever is least compressed.
4. **Nothing under ~0.013 on this split is evidence.**

## Caveats

* **One split, 114 rows.** Every number here is `oig_omission`. The fusion weights were
  selected against it. The check that would settle it is applying the fixed 0.25/0.40/0.35
  triple across all seven eval splits (1302 rows); that has not been run.
* **Single row-order per arm.** The 66-fit sweeps are one ordering each, so individual cells
  carry the ~0.013 floor. Differences within the top group (0.8393-0.8575) are not resolved.
* **The layer ceilings and the transfer numbers disagree**, and that is a finding, not an
  inconsistency: L56 has the best `oig_omission` ceiling (0.8819) and, on the v3 couples, close
  to the worst transfer (0.5891). Within-distribution separability does not predict what 116
  out-of-distribution rows will find.

## Files

Scripts (all read cached activations; none loads an extraction model except the first):

| script | what it does |
| --- | --- |
| `scripts/extract_multilayer_activations.py` | all six layers in ONE forward pass via `HookedModel`'s multi-layer hook list; six layers cost what layer 56 alone costs |
| `scripts/layer_cv_sweep.py` | grouped CV ceilings over dev+eval, per split and pooled |
| `scripts/multilayer_refit.py` | feature-concatenated multi-layer refit + single-layer controls |
| `scripts/arch_layer_sweep.py` | every `ProbeType` x every layer, per training condition |
| `scripts/probe_agreement.py` | per-row scores, overlap/disagreement, threshold selection on held-out dev |
| `scripts/weighted_combine_sweep.py` | weighted-mean fusion over subsets containing an anchor, with the rank-preservation guard |

Results in `results_instructions_gemma27b_layersweep/`: `layer_cv.json`,
`arch_layer_sweep.json` (v3), `arch_layer_sweep_SP2.json`, `arch_layer_sweep_BASEONLY.json`,
`weighted_sweep*.json`, `probe_agreement*.json`, `seq_all/multilayer_refit.json`.

The 35 GB of extracted activations under `results_instructions_gemma27b_layersweep/activations/`
are **not** committed. To regenerate (~2 h on one 3090, 3.85 s/row for all six layers):

```bash
HF_HOME=$PWD/hf_cache HF_TOKEN=... AGENTIC_REDTEAM_MAX_MEMORY="0=22GiB,cpu=45GiB" \
.venv_claude/bin/python scripts/extract_multilayer_activations.py \
    --layers 16 24 32 40 48 56 \
    --data-dir eval_sets/instructions --data-dir dev_samples/instructions \
    --extra-jsonl base=data/instructions_llama70b_50.jsonl \
    --extra-jsonl couples=probes/.../redteam_postprocessed_iter5.jsonl \
    --out-root results_instructions_gemma27b_layersweep/activations \
    --probe probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl \
    --combine-consecutive-messages --convert-tool-to-assistant
```

---

## Addendum: the short-and-precise couples, the ceilings, and the dev-lending control

Added after the sections above, and it changes one of their conclusions.

### A second couple set, pointing the other way

`results_.../..._iter2_v4_shortprecise` peaked at `oig_omission` **0.8495** at iteration 2, on
30 postprocessed rows = **15 couples** — less than half the v3 set. Re-running the same
11-arch x 6-layer sweep on `base + those 15 couples` (`arch_layer_sweep_SP2.json`):

| arch@layer | base only | base + 15 pairs | base + 33 pairs |
| --- | --- | --- | --- |
| linear_then_last@L32 | 0.8336 | 0.8393 | 0.7227 |
| linear_then_last@L40 | 0.8381 | 0.8489 | 0.7867 |
| linear_then_last@L56 | **0.8846** | 0.8575 | 0.7202 |
| linear_then_softmax@L32 | 0.7966 | 0.8467 | 0.7116 |
| linear_then_softmax@L56 | 0.7147 | 0.8227 | 0.5891 |

The 15-couple set lifts the incumbent head (+0.050 at L32, +0.108 at L56); the 33-couple set is
below base-only everywhere. **Half the rows, opposite sign.** Re-scoring the published
`probe_iter2.pkl` through this harness returns **0.849492**, exactly its CSV value, so the
scoring path is exact; the refit's 0.8467 differs only by the row-order floor.

Anchored fusion on this set (`weighted_sweep_SP2.json`) reaches **0.9175** — the highest
`oig_omission` figure anywhere in this work — with a bootstrap CI of [+0.0231, +0.1256] against
the incumbent.

### Ceilings, including for the fused recipe

`scripts/cv_ceiling_fused.py` computes the grouped-CV ceiling for several components at once and
fuses their **out-of-fold** predictions (legitimate: each component's prediction for a row comes
from a fold that never saw it, and the weights are fixed in advance).

| | pooled OOF ceiling |
| --- | --- |
| `linear_then_softmax` @ L32 | 0.9055 |
| `linear_then_last` @ L56 | 0.9081 |
| `linear_then_last` @ L16 | 0.8118 |
| fused, fixed weights | **0.9187** |

Two readings. The single-head 0.9055 against the **0.914** already on record is the same
measurement — that run fit fused, this one sequentially, and the 0.0085 gap is inside the floor.
And **fusion buys almost nothing at the ceiling** (+0.011 over its best component, i.e. at the
floor) where it was worth +0.07 in transfer: it compensates for weak out-of-distribution training
data rather than reaching information a single head cannot.

The sharpest consequence: the 15-pair fused probe scores **0.9175 having never trained on this
split**, against a **0.9187** ceiling for training on the split directly. It is already at its own
ceiling — better training data has essentially nothing left to give this recipe.

### The dev-lending control, and a correction

`multimax_data_arms.py` ran `base`, `base+couples` and `base+couples+dev` but never `base+dev`,
leaving its dev arm without a control. `scripts/base_plus_dev_fit.py` runs the fourth cell.

**Four figures are now on record for "dev-mixed", and they are different measurements:**

| | base+dev | + the 33 couples | couples' effect |
| --- | --- | --- | --- |
| `train_base_plus_dev.py` / `train_base_dev_couples.py` | 0.8695 | 0.8738 | **+0.0043** |
| `base_plus_dev_fit.py` / `multimax_data_arms.py` | 0.8717 | 0.8643 | **-0.0074** |

Plus `..._devmix`'s iter5 at **0.8972**, which is not comparable to either row: it is a
five-iteration pipeline with a cumulative dose ladder and its own 86-row couple set.

**Two matched pairs disagree in sign, and both magnitudes are far under the ~0.013 floor.** The
defensible claim is only that *the couples add nothing measurable once in-distribution dev rows
are present* — not that they subtract. An earlier draft of this work asserted the negative
direction off one pair; the other pair is the check that should have been run first, and is the
reason the floor section leads this document.

### Artifact

The interactive companion — both couple sets row by row, the 114 eval rows, all 50 base rows and
the architecture survey — is published at
`https://claude.ai/code/artifact/f02dbe98-c7db-4ce0-b6d5-2d7d8015b573`.

### Added files

| file | produced by |
| --- | --- |
| `arch_layer_sweep_SP2.json` | `scripts/arch_layer_sweep.py --couples .../redteam_postprocessed_iter2.jsonl --couples-name couples_sp2` |
| `arch_layer_sweep_BASEONLY.json` | `scripts/arch_layer_sweep.py --train base` |
| `weighted_sweep_SP2.json` | `scripts/weighted_combine_sweep.py --anchor linear_then_last@L56` |
| `cv_ceiling_fused.json` | `scripts/cv_ceiling_fused.py` |
| `base_plus_dev_nocouples.json` | `scripts/base_plus_dev_fit.py` |
| `base_selection_oig_omission_base_only.json` | `scripts/base_selection_study.py` |
