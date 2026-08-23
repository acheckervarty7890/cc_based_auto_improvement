# A normalization step in front of the probe head

`LinearThenSoftmax` reads the layer-32 residual stream raw — `nn.Linear(5376, 1)` straight onto the activation. This inserts one normalization step in front of that linear and changes nothing else: same 5 folds, same reserved 72-row dev validation slice, same seven hyperparameters, same training-size ladder. Arm **hu_ha_dd_gptoss120b**; out-of-fold mean eval AUROC over the 4 `eval_sets/hu_ha` splits (866 rows). The ceiling never touches red-team data, so the arm choice selects the eval/dev blobs and nothing else.

**Control.** `--norm none` runs the baseline architecture through the new subclass — an `nn.Identity`, no parameters, no RNG draws — and reproduces it exactly against the pre-existing `ceiling_hu_ha_dd_gptoss120b.json`, on every rung, split and metric. So what follows is the normalization, not the plumbing.

## What it does

1. **It helps, and the help is concentrated where the training data is scarce.** Paired over 4 head seeds, LayerNorm beats the raw input by +0.0064 at 173 rows/fold (4/4 seeds) and by +0.0020 at the top rung (3/4). At the top the paired sd (0.0023) is the same size as the difference, so the honest reading is: a clear gain at a few hundred in-distribution rows, fading to nothing once the head has ~900.
2. **It halves the run-to-run spread.** Across the same seeds the normalized head's AUROC sd is 0.49x the unnormalized head's, averaged over the four rungs — the same direction at every rung. For a *single* probe that is arguably the more useful property of the two; for this repo's 10-member ensembles it is partly redundant, since averaging members already cancels seed variance.
3. **Mean subtraction contributes nothing; the per-token rescaling is the whole effect.** RMSNorm and affine-free LayerNorm differ by at most 0.0003 at any rung, and the learnable affine adds little beyond either. What matters is dividing out each token's own magnitude.
4. **Per-feature standardization is actively harmful** (-0.0426 at the top rung — an order of magnitude outside the seed noise, and the only result here that does not need the paired test to be believed). Dividing each of the 5376 dimensions by its own std equalizes the feature geometry, which is what it was meant to do, and that is the problem: the dimensions with tiny variance are mostly noise, and this amplifies them to parity with the ones carrying the concept. The per-token norms apply **one scalar per token** and leave the relative feature scales intact, which is precisely the difference.
5. **AUROC is the only metric that moves.** Accuracy is a wash (<=0.005 either way at every rung). `tpr_at_fpr` looks worse for the normalized head, but its own across-seed sd is 0.10-0.22 — at 134-400 rows a 1% FPR threshold sits on one or two negatives — so that column is not readable at this split size, in either direction.

## All five variants, one head seed (42)

| normalization | 173 | 346 | 693 | 693+dev218 |
|---|---|---|---|---|
| none (baseline arch) | 0.9439 | 0.9605 | 0.9777 | 0.9844 |
| LayerNorm + affine | 0.9500 | 0.9652 | 0.9808 | 0.9859 |
| LayerNorm, no affine | 0.9477 | 0.9660 | 0.9804 | 0.9851 |
| RMSNorm + scale | 0.9476 | 0.9662 | 0.9802 | 0.9849 |
| per-feature standardize (frozen) | 0.9003 | 0.9294 | 0.9353 | 0.9418 |

Read this table for its *shape*, not its last digits — the next section shows the head seed alone moves the top rung by more than the gaps between the middle three rows. What is robust here is the sign and the ordering: the three per-token norms all beat the raw input at every rung, they land within 0.001 of each other, and the per-feature standardizer is far below all of them.

## Paired against the raw input, over 4 head seeds

`--fit-seed` re-fits the *same* folds and the *same* training rows under a different head init and batch order. Seeds 7, 42, 1234, 20260823. Each column is a matched pair, so the fold noise the two share cancels; `sd` is over the four per-seed differences.

**LayerNorm + affine** vs `none`:

| rows/fold | none (mean +- sd) | LayerNorm + affine (mean +- sd) | paired diff | sd | wins |
|---|---|---|---|---|---|
| 173 | 0.9413 +- 0.0030 | 0.9477 +- 0.0016 | **+0.0064** | 0.0028 | 4/4 |
| 346 | 0.9616 +- 0.0015 | 0.9651 +- 0.0007 | **+0.0035** | 0.0013 | 4/4 |
| 693 | 0.9818 +- 0.0039 | 0.9829 +- 0.0018 | **+0.0011** | 0.0042 | 3/4 |
| 693+dev218 | 0.9815 +- 0.0030 | 0.9835 +- 0.0016 | **+0.0020** | 0.0023 | 3/4 |

## Where the gain sits (LayerNorm + affine minus none, mean over 4 seeds)

| rows/fold | ai_dilemmas | ant_hh | balanced_refusal | daily_dilemmas |
|---|---|---|---|---|
| 173 | +0.0030 | +0.0193 | +0.0048 | -0.0013 |
| 346 | -0.0006 | +0.0138 | +0.0005 | +0.0003 |
| 693 | +0.0001 | +0.0024 | +0.0031 | -0.0014 |
| 693+dev218 | -0.0011 | +0.0097 | +0.0016 | -0.0021 |

## The other two metrics

| metric | rows/fold | none (mean +- sd) | LayerNorm + affine (mean +- sd) | paired diff |
|---|---|---|---|---|
| accuracy | 173 | 0.8781 +- 0.0048 | 0.8827 +- 0.0025 | +0.0045 |
| accuracy | 346 | 0.9150 +- 0.0043 | 0.9150 +- 0.0049 | -0.0000 |
| accuracy | 693 | 0.9298 +- 0.0032 | 0.9348 +- 0.0033 | +0.0050 |
| accuracy | 693+dev218 | 0.9409 +- 0.0032 | 0.9404 +- 0.0030 | -0.0005 |
| tpr_at_fpr | 173 | 0.5546 +- 0.1242 | 0.3825 +- 0.2110 | -0.1721 |
| tpr_at_fpr | 346 | 0.3921 +- 0.2197 | 0.3868 +- 0.1131 | -0.0053 |
| tpr_at_fpr | 693 | 0.4796 +- 0.1006 | 0.4746 +- 0.0049 | -0.0050 |
| tpr_at_fpr | 693+dev218 | 0.4578 +- 0.1762 | 0.3524 +- 0.1426 | -0.1054 |

## Is the ladder flat at the top? (`693` -> `693+dev218`)

| normalization | seeds where the top rung is the best rung | climb at seed 42 |
|---|---|---|
| none (baseline arch) | 1/4 | +0.0068 |
| LayerNorm + affine | 2/4 | +0.0050 |
| LayerNorm, no affine | 1/1 | +0.0047 |
| RMSNorm + scale | 1/1 | +0.0047 |
| per-feature standardize (frozen) | 1/1 | +0.0065 |

The ladder is still climbing on average, so every number above is a lower bound on its architecture's ceiling rather than a plateau — but it is not monotone seed to seed, which is a second reason not to read the top rung alone.

