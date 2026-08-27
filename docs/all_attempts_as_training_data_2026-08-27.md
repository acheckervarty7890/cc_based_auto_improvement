# Every attempt as training data, and what the seven-split mean was hiding

Third companion to `docs/what_limits_the_instruction_probe_2026-08-25.md` and
`docs/dev_lending_and_pair_reshaping_2026-08-26.md`, working against the same numbers on the
same run (`..._tellattacker_iter5_v3`, 373 attempts, 34 finds, 33 couples): `oig_omission` is
learnable to **0.914**, the base 50 rows reach **0.797**, and this run's 33 contrastive couples
take it down to **0.713**.

Every experiment so far has trained on the **finds** — 34 of 373 submissions, and only as
generated couples. The other 339 were thrown away, even though the judge labelled each one on
its own merits at submission time. This asks what those labels are worth. No contrastive
generation at all: each attempt trains under the judge's own label, exactly as
`retrain._records_to_labelled_dataset` does for the successes.

**The headline is a correction, not a result.** The seven-split mean rises, and that rise is an
artifact of one anti-predictive split. On the six splits where the base probe is actually
predictive, every red-team arm here loses to base-only.

## Setup

`scripts/all_attempts_refit.py`. One variable against the existing arms: what red-team rows
enter the training set.

* **Activations for all 373 attempts.** 33 were already in the per-conversation cache (the
  finds that survived `filter_dataset`); the other **340 were extracted** on gemma-3-27b-it
  L32 at ~1.2 s/sample and written through per row into
  `results_instructions_gemma27b_shared/base_activations`. Every submission of that rotation
  now has a cached activation, so any further study of this run's failures is a fit away.
* **Fits.** Spec inherited from `probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl`,
  10 fused members under `ENSEMBLE_SEEDS[:10]`, `gradient_accumulation_steps: 4` (the probe's
  own — the setting 0.797 and 0.713 were produced under), `dev_samples/instructions` (436 rows)
  as validation, `--seed 42`, scored on all seven `eval_sets/instructions` splits at full size.
  Fusion is on for every arm here *and* for the two reference arms, which come from
  `label_flip_ablation.py`'s `base_only` / `flip0`.
* **Class balance is whatever the rotation produced**: 86 positive / 287 negative over all 373
  (86 / 241 over the 327 in-scope). A `false_positive` hunt writes mostly negative-truth
  conversations. Nothing rebalances it — the skew is part of what is being measured.
* **The scoped arms** drop the 46 candidates the judge's scope check rejected. CLAUDE.md's
  convention is that a rejection is never training data, so including them is a choice worth
  measuring rather than assuming.

## The table

`mean(6)` drops `anthropic_harmless_refusal`; see §1 for why it has to be reported.

| training set | rows | oig_omission | mean(7) | **mean(6)** | refusal |
| --- | --- | --- | --- | --- | --- |
| **CEILING** — grouped 5-fold CV on the split | ~90 in-dist | **0.914** | — | — | — |
| **base only — no red-team data** | 50 | **0.797** | 0.772 | **0.842** | 0.348 |
| base + 327 scoped attempts | 377 | 0.685 | **0.809** | 0.827 | 0.698 |
| base + 373 attempts | 423 | 0.694 | 0.803 | 0.820 | 0.701 |
| base + 33 couples — the run's own data | 116 | 0.713 | 0.794 | 0.805 | 0.726 |
| 373 attempts alone — no base | 373 | 0.673 | 0.698 | 0.690 | 0.751 |
| 327 scoped attempts alone | 327 | 0.626 | 0.684 | 0.674 | 0.744 |
| 33 couples alone (`redteam_only_refit`, gas 4) | 66 | 0.644 | 0.673 | 0.660 | 0.751 |
| base + 33 finds only (`pair_half_ablation_v3`) | 83 | 0.606 | 0.734 | — | — |

## 1. The seven-split mean is not usable on this probe

Base-only scores **0.348** on `anthropic_harmless_refusal` — anti-predictive, not merely weak.
The 50-row base set carries no refusal data, so the probe's boundary is inverted there. Every
other base-only fit in the directory agrees (0.322 / 0.348 / 0.348 / 0.350 across
`eval_flip_base_only`, `eval_indistval_base_only`, `eval_basesubset_all50`,
`eval_mm_base__stock`), as does the v3 run's own iteration 0, so this is the probe, not a bad
draw.

Averaging a sub-chance AUROC into a mean understates the base probe by ~0.07, and **any arm
that adds refusal-shaped conversations gets that back for free**. For `base + 373 attempts` the
+0.031 mean(7) gain over base-only decomposes as **+0.050 from the refusal split alone**, minus
0.019 net across the other six:

| split | base + 373 attempts − base only |
| --- | --- |
| anthropic_harmless_refusal | **+0.353** |
| hc_context_drift | +0.059 |
| bbq_substitution | −0.001 |
| mm_substitution | −0.011 |
| hc_contradiction | −0.020 |
| oig_context_drift | −0.059 |
| **oig_omission** | **−0.104** |

Against base-only, on mean(6): base + 33 couples **−0.038**, base + 373 attempts **−0.023**,
base + 327 scoped **−0.015**. **No red-team arm in this programme beats not collecting the data**
once the inverted split is set aside. `74988235` already noted this for its own arm ("most of
the mean gain is anthropic_harmless_refusal recovering from a sub-chance 0.348"); it generalises
to every mean(7) quoted anywhere in these documents, including the 0.914-ceiling table's `mean`
column. Report mean(6) alongside mean(7), or fix the base probe's refusal blind spot so the
aggregate means something.

## 2. Without base data the attempts do not carry a boundary

373 attempts alone reach mean(6) 0.690 against base-only's 0.842, and accuracy sits at **0.507**
on essentially every split: the 3.3:1 skew shifts the whole score distribution, so the fitted
threshold is useless even where the ranking is not. They still beat the 66 couples alone (0.660,
`oig_omission` 0.673 vs 0.644), so 373 judge-labelled rows are worth more than 33 generated
couples — but neither is a training set.

## 3. The failures are worth more than the couples, and that is not the same as being worth collecting

The finds are the half that poisons: base + the 33 finds alone collapses `oig_omission` to
0.606. Adding the 339 **unsuccessful** attempts from the same rotation takes it to 0.694 and
lifts mean(6) from 0.805 (couples) to 0.820. The rows nothing was ever done with carry more of
what the rotation learned than the rows the whole pipeline exists to produce — the contrastive
generator, the couple structure and the scope check all operate on the 9% that succeeded.

But 0.820 is still below base-only's 0.842, and `oig_omission` — the split this run's
`eval.data_description` names, and the one it was steered at — goes **0.797 → 0.694**, worse
than base-only and no better than the couples' 0.713, against ~0.005 refit noise. Ten times the
red-team data does not move the target split, which is the same conclusion the 5x augmentation
study reached (`what_limits…` §2) by a different route. Only the 16 in-distribution
`oig_omission` dev pairs did (0.897).

## 4. Scope rejections cost almost nothing either way

Dropping all 46 rejected candidates moves mean(7) by +0.006, mean(6) by +0.007 and
`oig_omission` by −0.008 — all inside the noise. The scope check is doing its job as a filter on
what gets *recorded* as a success; as a filter on training data it is not where the loss is.

## Files

* `scripts/all_attempts_refit.py` — extraction + the four fits.
* `results_.../all_attempts_refit.json`, `eval_allatt_{attempts_only,base_attempts,scoped_only,base_scoped}.csv`,
  `probe_allatt_*.pkl`, `logs/all_attempts_refit.log`.
* Reference rows: `eval_flip_base_only.csv` / `eval_flip_flip0.csv` (`label_flip_ablation.py`),
  `redteam_only_refit.json` (`redteam_only_refit.py`).

## A note on fusion, for anyone comparing across documents

Every arm here and both reference arms were fit and scored with `PROBE_FUSED_ENSEMBLE` on
(the default). `multimax_data_arms.py`, `mlp_head_ablation.py`,
`multichannel_activation_ablation.py` and `input_norm_ablation.py` force it **off**, so their
"stock" baselines are not byte-comparable to `eval_flip_base_only`. The offset is small — the
same base-only fit measures 0.7714 sequential against 0.7719 fused, with every split within
0.002 except `hc_contradiction` (0.9078 vs 0.9099) — well under the 0.005 refit noise, but worth
knowing before a 0.002 is read as an effect.
