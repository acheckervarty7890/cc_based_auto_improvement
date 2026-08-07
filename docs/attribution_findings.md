# Attributing eval AUROC to individual red-team conversations (hu_harm, gemma-3-27b L32)

Working notes for the attribution of `probe_iter3`'s training set in both arms of
`run_gemma27b_hu_harm_attacker_ablation_batch.sh`. Three findings here are independent
of the attribution result itself and affect how any eval number from this pipeline
should be read.

## 1. `best_epoch` restoration does not work — the saved probe is the last epoch

`tuberlens/probes/pytorch_classifiers.py` snapshots the best-validation model with

```python
best_model_state = self.model.state_dict().copy()
```

`.copy()` on a `state_dict` is a **shallow dict copy**: the values are the live
parameter tensors. AdamW updates those in place, so the snapshot tracks the current
weights, and the closing `load_state_dict(best_model_state)` restores the *last*
epoch trained. Verified directly — the snapshot shares `data_ptr` with the parameter
and mutates under an in-place update.

Consequence: every probe under `probes/` holds the weights from the epoch at which
early stopping fired (`best_epoch + patience`, i.e. + 50), while its `best_epoch`
metadata reports a different epoch. Early stopping still bounds *training length*;
its model selection is a no-op.

This is not academic. A faithful reimplementation that returns the true best-epoch
snapshot scores 0.93 on `eval_ai_dilemmas` where the pipeline scores 0.998 — the
fully-trained model is the better one here, so the bug is currently helping. Fixing it
would change every probe this repo produces.

## 2. Seed noise dominates the between-iteration differences

Ten refits of the **identical** iter3 data, differing only in seed
(`gptoss120b_noisefloor.json`):

| split | mean | sd | range |
|---|---|---|---|
| eval_ai_dilemmas | 0.99843 | 0.0015 | 0.0049 |
| eval_ant_hh | 0.75353 | 0.0089 | 0.0277 |
| eval_balanced_refusal | 0.93102 | **0.0233** | **0.0778** |
| eval_daily_dilemmas | 0.98506 | 0.0059 | 0.0190 |

Seed alone moves `balanced_refusal` between 0.877 and 0.955. The gptoss arm's
committed iteration sequence on that split — 0.9664, 0.8354, 0.9378 — fits inside one
seed's spread, so those differences are not evidence of anything about the data.

The mechanism is upstream of the metric: validation AUROC is 0.985 by epoch 1 and
0.9999 by epoch 4, so consecutive epochs are separated by one or two validation pairs
out of 6624 and epoch selection is effectively a coin flip among near-ties. A 166-row
validation split is too small to choose between epochs of an already-saturated model.

**Any single-seed comparison from this pipeline needs a seed-averaged replicate before
it can be read as a data effect.**

## 3. The reported AUROC is computed on saturated probabilities

`get_performances` scores `predict_proba`, i.e. `sigmoid(logit)` in bf16, where
anything above logit ≈ 5.5 rounds to exactly 1.0 — 63/136, 34/134, 81/400 and 71/196
rows across the four splits. Those rows are mutually tied and sklearn credits 0.5 per
tied pair, so the published number is *not* the rank AUROC of the logits: on
`eval_ant_hh` it is 0.0057 lower.

Two consequences: a change that only reorders the confident block is invisible to the
reported metric, and any recomputation must replicate the saturation to be comparable
to the committed CSVs. `attribution_lib.auroc_pipeline` does (matching the CSVs to
≤1.5 rank-pairs on all four splits); `auroc_rank` gives the tie-free number, and both
are reported everywhere.

## Why influence functions were abandoned

The original plan was closed-form influence on a surrogate, verified by a few real
refits. Two measured properties killed it:

- **Perfect separation.** Both arms fit their training set exactly (train accuracy
  1.0000, zero misclassified rows, median |logit| ≈ 10). The Gauss-Newton Hessian's
  total weight is Σσ(1−σ) = 0.56 over 662 rows, so `H⁻¹` is pure damping and any
  ranking it produced would be an artifact of λ.
- **No stationary point.** Training stops at epoch 6 (gptoss) and 3 (deepseek), so θ*
  is a few steps from initialisation, not an optimum. Influence's premise fails.

The Jacobian machinery built for it survives in `attribution_features.py` (exact
derivatives through the softmax pooling, finite-difference checked to ~1e-4) and is
still useful for diagnostics. Note the tempting shortcut — freezing the pooling
weights and using Σₜpₜhₜ as the feature — reproduces the logit exactly at θ* but gets
the *derivative* wrong (cosine 0.958, 26% magnitude error), because it drops the term
where moving w moves the attention.

## What replaced it

Real leave-one-pair-out, made affordable by a GPU-resident reimplementation of the
trainer (`attribution_fasttrain.py`): activations packed ragged, K probes trained in
one pass since the parameter is a single 5376-vector. **0.49 s per probe at K=64**
against 131 s for a reference fit, with GPU memory flat in K.

Verified as the same procedure: bit-reproducible across runs (which required replacing
CUDA atomics with dense ops — atomic jitter was flipping which epoch got selected),
RNG-faithful to the reference (same init, same shuffle stream including the second
permutation `RandomSampler` draws and discards each epoch, clipping after every
micro-batch as the reference does), reproducing the last-epoch bug deliberately, and
distributionally matched over 10 seeds (mean 0.92135 vs 0.91701, comparable sds).

The unit of attribution is the **pair** — each red-team success plus the LLM-generated
opposite-class counterpart `preprocessing` mints for it — rejoined by re-deriving
`preprocessing._cache_key` against the arm's `contrastive_cache.jsonl` (389↔389 and
439↔439, zero orphans). Removal drops the pair from whichever side of the train/val
split each row landed on; 31% of pairs straddle it.

Given the noise floor, expect most individual pairs to come back indistinguishable
from zero. `attribution_verify.py` therefore reports a multiplicity-corrected set
(Benjamini-Hochberg) alongside the loose 2-SE one, and always drops a **size-matched
random control set** — a flagged set that does not beat that control by more than its
error bar has identified nothing.
