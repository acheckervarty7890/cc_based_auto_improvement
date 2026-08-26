# Can a different readout head fix the omission probe?

`docs/what_limits_the_instruction_probe_2026-08-25.md` established that `oig_omission` is
learnable to **0.914** (grouped CV on the eval split itself), that the base probe reaches
**0.797**, and that adding this project's 33 red-team couples *drops* it to **0.714**. That
document ruled out several explanations on the data side. This one asks the remaining
question from the other side: is the limit the **readout architecture**?

Short answer: **no**. Five families of architectural change were tested and none produced a
gain that survives comparison with its own noise. Two *hyperparameter* settings did move
consistently — pooling temperature and an input LayerNorm — and both are single-seed and act
almost entirely on the mean rather than on the target split.

## Common setup

Every fit below is identical apart from the head:

| | |
| --- | --- |
| probe | gemma-3-27b-it L32, 10-member ensemble under the repo-pinned `ENSEMBLE_SEEDS` |
| train | base `data/instructions_llama70b_50.jsonl` (50) + the v3 run's 33 couples (66) = **116 rows** |
| validation | `dev_samples/instructions` (436), used whole |
| eval | all seven `eval_sets/instructions` splits, full (no subsampling) |
| references | stock head **0.7135** on `oig_omission`, **0.7954** mean; base-only 0.797; ceiling 0.914 |

`PROBE_FUSED_ENSEMBLE=0` throughout, so every member goes through
`PytorchAdamClassifier.train`. All activations were cached, so no extraction model was loaded
in any of these runs. **Every arm is a single seed** — that is the governing caveat for the
whole document.

## The stock head, and why it invites all of this

`LinearThenSoftmax` scores each token (`Linear(5376→1)`), softmax-weights those scores by
themselves at `T=5`, and sums. Measured on a real conversation, **one token of 179 carried 93%
of the weight**, so the head is nearly a max: one position, one direction, and the position is
discarded. Each family below relaxes one part of that.

## 1. Pooling the *inputs* into segments — decisive failure

Masked adaptive-mean-pool the activations to length k, then the stock head.

| k=1 | k=2 | k=3 | full sequence |
| --- | --- | --- | --- |
| 0.611 | 0.592 | 0.588 | **0.714** |

Destroying the head's ability to *select* costs far more than positional information buys.
This result is the reason every later family keeps the per-token scoring intact.

## 2. Segmenting the *sum* — `LinearThenSegmentedSoftmax`

Steps 1–4 unchanged; each token's contribution `z·w` is routed to one of k positional buckets
and `Linear(k→1)` combines them. Initializes at weight 1 / bias 0, so it reproduces the stock
head exactly at any k (asserted by `verify_segmented_head_identity`).

| stock | k1 | k2 | k3 | k4 | **k2 + per-segment softmax** | k3 + per-segment softmax |
| --- | --- | --- | --- | --- | --- | --- |
| 0.7135 | 0.7165 | 0.7098 | 0.7005 | 0.6968 | **0.7427** | 0.6602 |

One arm is up +0.029 and its k=3 sibling is down −0.053. No trend.

## 3. Parallel channels — `MultiChannelLinearThenSoftmax`

`Linear(5376→p)`, one softmax over the sequence **per channel**, `Linear(p→1)`. At p=1 it is
`LinearThenSoftmax` exactly (`verify_multichannel_head_identity`).

| p | 1 | 2 | 4 | 8 | 16 | 32 | 36 | 40 | 44 | 48 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oig_omission | 0.7122 | 0.7011 | 0.7045 | 0.6981 | 0.7501 | 0.7499 | 0.7428 | 0.7227 | **0.7707** | 0.7268 |
| mean | 0.7890 | 0.7982 | 0.7644 | 0.7655 | 0.7627 | 0.7566 | 0.7643 | 0.7488 | 0.7563 | 0.7412 |

p=44 is the best `oig_omission` number in this document. It is also **not a trend**: p ≤ 8 sits
at or below stock, then p ≥ 16 bounces in a ±0.025 band with p=40 and p=48 barely above stock.
A step function with nothing beneath it, on a 114-row split, at one seed. Meanwhile the mean
erodes monotonically with p (0.795 → 0.741), driven by `anthropic_harmless_refusal` collapsing
to 0.445 — at p=48 step 1 alone holds 258k parameters against 116 training rows.

## 4. A nonlinear score — `MLPThenSoftmax`

`Linear(5376→h) → GELU → Linear(h→1)` per token, then the stock pooling. Control `h1_id` is
h=1 with an identity activation, i.e. a rescaled single linear score.

| stock | h1_id (control) | h8 | h16 | h32 | h64 | h128 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.7135 | **0.7221** | 0.7088 | 0.7141 | 0.7128 | 0.6876 | 0.7122 |

**The best number in the row is the linear control.** Nothing from the hidden layer. It does
move two other splits coherently — `hc_contradiction` 0.839 → 0.900 and `bbq_substitution`
0.937 → 0.950, both monotone in h, both splits whose negative class is a contradiction against
supplied source material — so the capacity is used, just not on omission.

## 5. Nonlinear channels — 25 arms, all negative

`channel_activation` on the p-channel head, at p ∈ {1, 2, 4, 8, 16} × {gelu, leaky_relu, tanh,
relu², gelu with weights from the activated scores}.

**Not one of the 25 arms beat its own identity control on `oig_omission`.** The nearest was
p=12 `leaky_relu` at 0.7082 against a 0.7033 control (+0.005).

Two things worth recording:

* **`relu²` has a breaking point between p=4 and p=8.** 0.7061 / mean 0.7957 at p=2; 0.5111 /
  mean 0.5759 at p=8; 0.4832 / mean 0.5029 at p=16. Same activation, same data, same seeds.
  This is the clearest evidence in the document that the p=16 results are a
  capacity-times-variance effect rather than a property of any mechanism.
* **A prediction that failed.** Squashing the negative score tail should flatten the pooling
  softmax (a low-scoring token's denominator term goes from `exp(-8/5)=0.20` to `exp(0)=1`),
  so taking the weights from the *raw* pre-activation scores should have preserved selection
  and done better. At p=16 the opposite happened — `gelu_raw` 0.4923 vs `gelu_act` 0.5836 —
  and at p=2 the ordering reversed again. The flattening story is not supported. The
  `channel_weights_from` knob is kept because it is cheap and the arms are recorded, not
  because the mechanism was confirmed.

## 6. Temperature — the one architectural knob that moves consistently

`T` was never varied before this: every probe in the project inherited `temperature: 5` from
the base spec.

| oig_omission | T1 | T2 | T5 (default) | **T10** | T25 |
| --- | --- | --- | --- | --- | --- |
| p=1 | 0.5636 | 0.6978 | 0.7122 | 0.7316 | **0.7319** |
| p=2 | 0.6393 | 0.6433 | 0.7011 | **0.7310** | 0.7221 |
| p=4 | 0.4737 | 0.6131 | 0.7045 | **0.7205** | 0.7181 |
| p=8 | 0.6008 | 0.6171 | 0.6981 | **0.7153** | 0.6802 |
| p=16 | 0.4263 | 0.6183 | **0.7501** | 0.7045 | 0.6759 |

* **T=1 is the worst arm at every channel count**, without exception. Sharpening toward a hard
  argmax hurts — the same conclusion `relu²` reached by a different route.
* **T=10 beats the default T=5 at p = 1, 2, 4 and 8**, by +0.019 / +0.030 / +0.016 / +0.017.
  It fails only at p=16, the overfitting regime where everything is erratic.
* On the **mean** it is 3-of-4 rather than 4-of-4 (p=8 goes the other way: 0.7509 vs 0.7655).

## 7. Input LayerNorm

`input_norm` on the p-channel head, applied to the activation vector before step 1. Run at two
temperatures deliberately: LayerNorm rescales the scores, and the score scale is what `T` is
measured against, so a single-temperature test would confound "normalization hurt" with "`T` is
now mistuned".

Mean AUROC, matched pairs:

| | no LN | LN (affine) | LN (no affine) |
| --- | --- | --- | --- |
| p=1, T=5 | 0.7890 | **0.8029** | 0.8001 |
| p=1, T=10 | 0.8066 | **0.8172** | 0.8166 |
| p=2, T=5 | 0.7982 | **0.8038** | 0.8014 |
| p=2, T=10 | 0.8095 | 0.8125 | **0.8128** |

* **8 of 8 matched comparisons improve.** `p1_ln_T10` at **0.8172** is the best mean in this
  document; the top four arms overall are all LayerNorm + T=10.
* **The affine parameters are dead weight** — the two variants are within 0.003 everywhere, and
  the parameter-free one wins at p=2/T10. The learnable version adds 10,752 parameters, which
  *doubles* the p=1 head. Use `layernorm_noaffine`.
* **On `oig_omission` it does nothing**: 0.7215–0.7285 with LN against 0.7310–0.7316 without,
  at T=10. Marginally worse.
* **The entire mean gain is one split.** `anthropic_harmless_refusal` moves 0.792 → 0.855 at
  p=1/T10 and contributes ~0.009 of the ~0.011 average gain. That split swung from 0.735 to
  0.233 and back to 0.909 across the activation arms — it is the least stable measurement in
  the project.
* The two effects **compose without interaction** (LN ≈ +0.012 at either T; T=10 ≈ +0.014 at
  either normalization), which is mild evidence that the temperature effect is not merely a
  score-scale artefact.

The measured norm spread LayerNorm discards is narrow: over 27,668 real training tokens, vector
norms are mean 93.7, p5 80.8, p95 107.6 — a ±15% band, all mapped to `sqrt(5376) = 73.3`.

## What this adds up to

| lever | best `oig_omission` | vs stock 0.7135 |
| --- | --- | --- |
| parallel linear channels | 0.7707 (p=44) | +0.057, in a ±0.025 band with no trend |
| positional segments | 0.7427 (k=2, per-segment softmax) | +0.029, sibling arm −0.053 |
| **pooling temperature** | 0.7319 (T=25), 0.7316 (T=10) | **+0.020, same direction at 4 of 5 channel counts** |
| nonlinear score (MLP) | 0.7221 — the *linear* control | ~0 |
| input LayerNorm | 0.7285 | ~0 on target; +0.011 mean, 8 of 8 |
| nonlinear channels | 0.7082 (p=12 leaky) | negative in 25 of 25 arms |
| input segment-pooling | 0.611 | −0.10, decisive |

**The constraint on this probe is not the readout's expressiveness.** Every attempt to add
capacity — channels, buckets, hidden layers, nonlinearities — came back neutral or negative,
and the two things that did help (a temperature change and a normalization) add no capacity at
all. That is consistent with `what_limits_the_instruction_probe`'s finding that the limit is in
the *data*: 116 training rows cannot support more parameters than the stock head already has.

Nothing here closes the gap that matters. The best `oig_omission` in this document is 0.771 at
one seed; base-only is 0.797 and the ceiling is 0.914.

## What to do next

**Seed replication, before anything else.** 5 seeds × {stock/T5, stock/T10, LN-noaffine/T10} at
p=1 — 15 fits, roughly an hour — turns every number above into a mean with a spread. Two
outcomes, both useful: if T=10 and LayerNorm survive, they are a default change for every probe
this project trains; if `anthropic_harmless_refusal` swings ±0.06 across seeds, the LayerNorm
result dissolves and only the temperature one remains.

Do **not** build on p=44, k=2-segmented, or any single arm in section 3 or 5 without that check.

## Reproducing

All heads live in `src/agentic_redteam/segmented_head.py`, each with a verifier asserting it
reduces to `LinearThenSoftmax` at its control setting. Defaults are byte-identical to the stock
head — `input_norm=none`, `channel_activation=identity`, `channel_weights_from=activated` —
verified by tensor equality, so nothing here changes an existing config.

```bash
scripts/segmented_head_ablation.py                  # section 2
scripts/segment_pooled_probe.py                     # section 1
scripts/multichannel_head_ablation.py [p ...]       # section 3
scripts/mlp_head_ablation.py [h ...]                # section 4
scripts/multichannel_activation_ablation.py [p]     # sections 5-6, nine arms at that p
scripts/input_norm_ablation.py                      # section 7
```

Each writes `eval_*.csv` per arm and one summary JSON into
`results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3/`.
