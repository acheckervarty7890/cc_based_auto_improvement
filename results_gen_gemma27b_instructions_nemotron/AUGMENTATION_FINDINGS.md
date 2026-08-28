# Arm 3N — what the loop's accept/reject decisions were actually measuring

Follow-up to `10a227b0` ("thirteen iterations — eval peaks at iteration 8 and the loop
turns"). That commit recorded the trajectory; this one asks *why* the loop accepted 8 of
61 batches, and finds that the answer is not a property of the samples.

Everything below is on the arm 3N setup unchanged: `google/gemma-3-27b-it` layer 32,
`linear_then_softmax`, single probe, seed 42, base `data/instructions_llama70b_50.jsonl`,
dev `dev_samples/instructions` (436 rows), eval `eval_sets/instructions` (7 full splits,
1302 rows). Every fit inherits architecture and metadata from `probe_iter12.pkl`, which is
byte-identical to `probe_iter13.pkl` (iteration 12 accepted nothing).

## The ladder

| set | added | total rows | dev | eval |
|---|---:|---:|---:|---:|
| base + 62 accepted (= `probe_iter13`) | 62 | 112 | 0.8311 | 0.8148 |
| base + 62 + it10b0 | 72 | 122 | 0.8220 | 0.8108 |
| base + 62 + 107 generated | 169 | 219 | 0.8334 | 0.8504 |
| **base + 62 + it10b0 + 107** | 179 | 229 | 0.8451 | **0.8617** |
| base + 62 + it10b0 + 14 (dev-selected) | 86 | 136 | 0.8469 | 0.8526 |
| base + 62 + 107 + 100 mixup | 269 | 319 | 0.8082 | 0.8126 |

For reference: the run's own peak was `probe_iter8` at 0.8236, and the base probe at 0.7779.

## 1. No sample-level property separates accepted from rejected

All 61 scored batches were read (directions plus matched pos/neg pairs) and measured.
Seven candidate predictors, none of which separates the two sides:

| property | r vs Δ | accepted vs rejected |
|---|---:|---|
| pos/neg length ratio | +0.10 | median \|log ratio\| 0.16 vs 0.13 |
| pair minimality (token Jaccard) | −0.13 | median 0.21 vs 0.36 |
| grounding (answer ⊂ user turn) | +0.14 | median 0.21 vs 0.20 |
| cosine distance to eval | −0.04 | median 0.1617 vs 0.1700, ranges overlap |
| first-line label leak | — | 5 of 8 accepted leak, 3 don't — same as rejected |
| sample count, turn count | — | no separation |
| eval-split category | — | 3 of 8 accepted have no eval analogue at all |

The sharpest case is a near-twin pair. `it9b1` (accepted, Δ +0.0120) and `it10b0`
(rejected, Δ −0.0102) are both "reproduce the user's ordered checklist; the negative
perturbs it", and are indistinguishable on every measure taken: grounding 1.00 / 1.00,
Jaccard 1.00 / 0.99, length ratio 1.00 / 1.03, 5/5 matched pairs, 2 turns each.

## 2. The rejection of `it10b0` was a sample-size artifact

`scripts/ablation_twin_injection.py` injects `it10b0` into the accepted set. Its dev
verdict **replicates** — measured −0.0102 at iteration 10 against a 42-sample baseline,
−0.0091 here against 62 — so the dev set is reliably detecting something. But per split it
*repairs* what the loop had been destroying (`mm_substitution` +0.030, `hc_contradiction`
+0.025, `oig_omission` +0.019) and pays for it entirely on `anthropic_harmless_refusal`
(−0.086), the split supplying the loop's whole mean gain.

Then the sign flips with volume: the same ten rows are worth **+0.0113** on top of
`base + 62 + 107` (0.8504 → 0.8617). Ten rows of a family the probe had seen once look
like contradiction; twenty-four rows of it look like a rule.

## 3. Per-batch dev scoring anti-predicts the union

`scripts/generate_like_accepted.py` regenerates each of the eight accepted batches'
families — few-shot from four real pairs of that batch plus its recorded direction,
nothing about the eval splits in the prompt — giving 107 rows (`data/instructions_like_accepted62.jsonl`).
Guards mirror `Generator._admit`: 1024-token budget under the fit's transforms, label
normalization, novelty against all 72 rows already in play.

Scored one family at a time against the `62 + it10b0` floor (dev only, zero eval looks),
**seven of eight families are negative**:

```
it9b1  +0.02488     it2b0  -0.01256
it1b1  -0.00109     it0b4  -0.02809
it4b3  -0.00316     it7b2  -0.03165
it11b3 -0.00571     it5b4  -0.03169
```

All eight together are worth **+0.047** on eval. The loop scores every batch
independently against the same baseline, so a family that only pays off in combination is
rejected on sight — which fits an 8-in-61 acceptance rate better than any story about the
samples.

Dev also ranks the two unions backwards: the dev-selected probe has the higher dev
(0.8469 vs 0.8451) and the lower eval (0.8526 vs 0.8617).

## 4. Poisoning tolerance: 6 batches / 60 rows

`scripts/poison_curve.py` starts from `base + 62 + 107` and adds rejected batches ten rows
at a time in the run's chronological order (arbitrary w.r.t. harm; a worst-first ordering
would be faster). Five batches leaves it at 0.8184, still above target within noise; six
crosses to 0.7998.

The curve is not monotone — `it1b0`, scored at −0.0422 during the run, *raises* eval by
+0.0128 here, while `it1b3`, scored at a near-neutral −0.0020, costs −0.0186. Across the
six, `corr(Δ@scoring, Δnow) = −0.21` (n=6, weak, but the sign is backwards).

## 5. Cosine distance to eval explains nothing

`scripts/cosine_to_eval.py`, masked mean of the layer-32 residual, against all 1302 eval rows:

| set | n | centroid | pairwise | nearest |
|---|---:|---:|---:|---:|
| 50 base (llama) | 50 | 0.0351 | 0.1719 | 0.0987 |
| 62 accepted | 62 | 0.0325 | 0.1721 | 0.1184 |
| 107 generated | 107 | 0.0338 | 0.1726 | 0.1195 |
| 60 rejected (the poison batches) | 60 | 0.0373 | 0.1893 | 0.1324 |
| *within eval, for scale* | 1302 | — | *0.1485* | — |

The generated 107 sit exactly where the 62 sit — independent confirmation that
"generate more like the 62" landed where intended. The 60 rejected are farther, but the
per-batch control kills that reading: across all 61 batches `corr(distance, Δ) = −0.04`,
and the single closest batch to eval (`it3b0`, 0.1384) was rejected. Those particular 60
rows are all from iterations 0–1 (JSON extraction, algebra, micro-fiction, personas), two
of which are among the six farthest batches in the run. And the base 50 are the closest
set of all on nearest-neighbour (0.0987) while producing a 0.7779 probe.

## 6. Activation mixup at λ=0.5 does not work here

`scripts/mixup_augment.py` averages random same-label pairs from the 169 real rows
token-by-token and adds 100 synthetic rows (50/class), written into the per-sample cache
under placeholder conversation keys so the fit path is unchanged. Result: **0.8126**,
−0.0378 against the set it augments, wiping out the 107 rows' entire gain.

Read this as a verdict on *this* alignment, not on activation augmentation. Cached rows
are stored at their own true length (43–502 tokens, median 186), so a pair is averaged over
`min(Ta, Tb)` positions — discarding a median **40%** of the longer member, with 33 of 100
pairs losing more than half. These are averages of conversation *prefixes*, and the prefix
holds the user's instruction while the tail holds the assistant's compliance or violation.
`LinearThenSoftmax` also looks for a localised span of evidence, which token-wise mixing
blurs. Right-aligned averaging, or length-matched pairing, are the variants worth trying.

## Consequences for the loop

- **The mean is the wrong acceptance criterion.** It lets a batch buy 0.35 on the probe's
  weakest split while giving back 0.16 on its strongest. A per-split floor — reject if any
  split drops more than the noise band — would have caught the last three acceptances,
  which cost `oig_omission` 0.044–0.047 each.
- **`min_auroc_gain: 0.0` is too permissive** on a 436-row dev set whose noise floor is
  ~±0.005; three of the run's eight acceptances are inside it, and all three are the short
  connection-truncated batches that also carry the label defects (single-class `it2b0`,
  class-skewed `it1b1`).
- **Independent per-batch scoring cannot see compositional value.** Scoring a whole
  iteration's batches jointly, or accepting on a rolling union, would.
- **Batch size is the binding constraint.** Ten rows is too few to tell a new family from
  noise; the same family at 24 rows flips sign.

## Reproducing

```bash
V=.venv_claude/bin/python
$V scripts/ablation_twin_injection.py                    # base+62+it10b0
$V scripts/generate_like_accepted.py --per-family 12     # needs OPENROUTER_API_KEY
$V scripts/fit_with_generated.py --score-families        # dev-only per family
$V scripts/fit_with_generated.py --tag allin
$V scripts/fit_with_generated.py --families it9b1 --tag devsel
$V scripts/poison_curve.py --max-steps 30
$V scripts/cosine_to_eval.py
$V scripts/mixup_augment.py
```

Every activation involved is content-keyed and already cached under
`cache_gen_gemma27b_instructions/`, so only the generation step needs the network and
none of the fits load the 27B model.
