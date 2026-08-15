# Why the last red-team iteration adds no eval AUROC (hu_harm, gemma-3-27b L32)

The vintage sweep settled *that* iteration 3 adds nothing: `v3 − v2` is −0.001 (gptoss)
and −0.002 (deepseek) on the mean eval AUROC, 10 seeds each, against a seed sd of
0.011–0.027. This note is the *why*, and what to change.

Everything below runs off activations already on disk
(`scripts/why_iter3_null.py`, `scripts/why_iter3_addendum.py`; raw numbers in
`results_hu_harm_gemma27b_batch_ablation/vintage/why_iter3*.json`). No gemma-3-27b
forward was run.

## The instrument, and why it is not the pipeline trainer

The pipeline's own seed noise is as large as the effect being looked for, which is what
made the 80-fit sweep unable to resolve anything below ~0.01. So the measurements here
use **mean-pooled layer-32 activations plus an L2 logistic regression** — convex and
deterministic, so a difference of 0.003 is readable and every number reproduces exactly.

It is a *different head* from the deployed one (which pools with a softmax over its own
per-token logits), so its absolute AUROCs are not comparable to the committed CSVs — on
`eval_ai_dilemmas` the deployed head reaches 0.998 where this one reaches 0.64. Only
contrasts computed inside this note are quotable. Section 1 checks it reproduces the
pipeline's vintage *ordering* before anything else is read off it.

## 1. The null is in the data, not in the trainer's noise

| arm | v0 (base only) | v1 | v2 | v3 |
|---|---|---|---|---|
| gptoss120b | 0.7256 | 0.7774 | 0.7711 | **0.7744** |
| deepseekv4pro | 0.7256 | 0.7650 | 0.7715 | **0.7694** |

With seed noise removed entirely, `v3 − v2` is +0.0033 and −0.0021. The pipeline's null
was not a power problem. There is genuinely almost nothing there.

## 2. It is not saturation — no split is near its ceiling

Fitting a linear probe on **each eval split itself** (5-fold CV, same activations) gives
what a linear readout of layer 32 can do on that split:

| split | ceiling | v3 achieved (gptoss) | gap |
|---|---|---|---|
| eval_ai_dilemmas | 0.7997 | 0.6356 | +0.164 |
| eval_ant_hh | **0.9599** | 0.7721 | **+0.188** |
| eval_balanced_refusal | 0.9977 | 0.9126 | +0.085 |
| eval_daily_dilemmas | 0.9050 | 0.7773 | +0.128 |

`eval_ant_hh` is the split that never moves in any iteration of either arm (0.70–0.76
throughout the committed CSVs) — and it is the split with the *most* headroom. The
information is in the activations; the training data fails to expose it. A ceiling
measured *above* the achieved value is a lower bound on what is extractable, so this
conclusion is safe despite the pooling difference.

## 3. It is not that the boundary stopped moving

| | gptoss | deepseek |
|---|---|---|
| cos(w_v2, w_v3) | 0.8981 | 0.9335 |
| ‖ŵ_v3 − ŵ_v2‖ | 0.4515 | 0.3648 |

Iteration 3's data moves the decision boundary substantially. It just moves it somewhere
that does not help.

## 4. Alignment with what the eval splits need is tiny — and flat

Cosine between the fitted direction and each split's own best direction:

| vintage | gptoss (mean over splits) | deepseek |
|---|---|---|
| v1 | 0.1578 | 0.1400 |
| v2 | 0.1286 | 0.1375 |
| v3 | 0.1319 | 0.1247 |

Near-orthogonal, and **not improving with iterations**. This is the load-bearing number:
a 4th and 5th iteration of the same loop do not converge on the eval directions, because
nothing in the loop pushes them there.

## 5. The new pairs bring the training set no closer to eval

Nearest-neighbour cosine in activation space, gptoss:

- eval → v3: 0.9024–0.9305, versus eval → v2: 0.9018–0.9220 — **within 0.001–0.009**
- eval → its own split: 0.9336–0.9995

So the 116 added pairs do not move the training set toward the eval distribution, and the
eval rows are far more like each other than like anything the attacker wrote. Deepseek
replicates (eval→v3 vs eval→v2 differ by ≤0.002). Both sides of that comparison are
nearest-neighbour cosines against the same baseline, so it is like-for-like.

> **Retracted.** An earlier version of this section also read new-in-v3 → v2 = 0.9432
> (against v2's self-NN of 0.9728) as "the new pairs sit inside the old cloud". That does
> not follow — see §5a. The redundancy conclusion rests on §4 and §6, which do not depend
> on any cosine.

Note the near-dup guard reports **0% clones** at τ=0.8, and it is right: there is no
surface duplication. It compares difflib on the first 600 chars of the first user turn.

## 5a. Why "close to v2" and "a success against probe_iter2" are not in tension

The obvious objection to §5 is that these rows *were* red-team successes against
`probe_iter2` — at median score 0.777 (false positives) and 0.037 (false negatives), i.e.
confidently wrong, not marginal. `scripts/why_close_but_wrong.py` settles it: **cosine
proximity in this space does not imply the probe treats two conversations alike.**

### How the cosine is computed

For every conversation, take the cached `[T, 5376]` fp16 layer-32 activations, cast to
fp32, zero the padded positions, and average over the unpadded ones (`_pool` in
`why_iter3_null.py`) — one 5376-vector per conversation. Cosine is then the dot product of
the L2-normalised vectors, **with no mean-centering** (`unit()` in
`why_close_but_wrong.py`). Three properties of that choice matter:

- It is **mean pooling, not the probe's pooling.** The deployed head pools with a softmax
  over its own per-token logits, so its representation is a function of the weights being
  fitted — useless as a fixed feature space, but it does mean this metric is not the one
  the probe reads. That is the whole point of §5a.
- It is **uncentered**, so the residual stream's large shared component dominates every
  value and everything lands at 0.86–0.96. Absolute values are therefore not
  interpretable; only comparisons on the same scale are.
- Conversations over 1024 tokens were **truncated** by `get_activations`, so their tail is
  absent from the mean. The new-in-v3 sources are the longer ones (median 1364 chars).

| | gptoss | deepseek |
|---|---|---|
| random v2 pair, **same** label | 0.8749 | 0.8874 |
| random v2 pair, **opposite** label | 0.8616 | 0.8766 |
| a source vs **its own opposite-label counterpart** | **0.9364** | **0.9488** |
| new-in-v3 → nearest v2 row | 0.9432 | 0.9599 |

`preprocessing` mints every contrastive counterpart by editing its source toward the
*opposite* label, which puts two rows at cosine 0.94–0.95 on opposite sides of the
boundary **by construction** — 45% (gptoss) / 39% (deepseek) of those deliberately
opposite-label pairs are *closer* than the median new-row-to-v2 hop. So the hop to the
nearest v2 row is the same size as the hop to a row built to carry the opposite label.

### Does the uncentered metric drive this? (`why_close_but_wrong_centered.py`)

Partly, for one number. Rerunning under three representations — raw, grand-mean-centered,
and per-dimension z-scored — all L2-normalised:

| arm · rep | same-label | opp-label | Δ | own opposite-label pair | new → v2 NN | k-NN k=1 / 5 / 15 |
|---|---|---|---|---|---|---|
| gptoss · raw | 0.8749 | 0.8616 | 0.013 | 0.9364 | 0.9346 | 56.0 / 51.7 / 48.3% |
| gptoss · centered | 0.0720 | −0.0483 | **0.120** | 0.4937 | 0.4979 | 59.5 / 50.0 / 49.1% |
| gptoss · whitened | 0.0449 | −0.0249 | 0.070 | 0.5620 | 0.3931 | 55.2 / 67.2 / 62.1% |
| deepseek · raw | 0.8874 | 0.8766 | 0.011 | 0.9488 | 0.9528 | 77.9 / 66.3 / 58.1% |
| deepseek · centered | 0.0508 | −0.0484 | **0.099** | 0.5429 | 0.5303 | 75.6 / 69.8 / 68.6% |
| deepseek · whitened | 0.0269 | −0.0267 | 0.054 | 0.6099 | 0.4402 | 73.3 / 61.6 / 66.3% |

**Corrected:** "the entire label signal is worth ~0.013 of cosine" was a raw-cosine
artifact. Centered, same-label and opposite-label pairs separate cleanly around zero
(+0.072 vs −0.048), so the label is more visible than that number suggested.

**Unchanged, and cleaner centered:** the comparison the argument rests on. The new-row →
nearest-v2 hop (0.4979) is indistinguishable from the hop to a *deliberately
opposite-label* counterpart (0.4937); deepseek likewise, 0.5303 vs 0.5429. Whatever the
metric, being that close to a v2 row says nothing about which side of the boundary a row
belongs on. The k-NN columns move by a few points and keep their verdict: chance for
gptoss, weakly better than chance for deepseek — never the near-certainty that would be
needed for "close to v2" to imply "the v2 probe should have got it right".

### Three consequences measured directly

- **The nearest v2 neighbour of a new success has the same label only 56% (gptoss) /
  78% (deepseek) of the time.**
- **A k-NN fitted on v2 is just as wrong as the probe** on the new successes — 56% / 52% /
  48% at k = 1 / 5 / 15 for gptoss, i.e. chance on a class-balanced set (deepseek 78% /
  66% / 58%). The probe is not doing anything anomalous; proximity is simply not
  predictive here.
- **In the probe's own metric the hop is enormous.** Across that 0.94 cosine step
  `probe_iter2`'s logit moves by a median of 5.9 (gptoss) / 9.2 (deepseek), and **41% /
  60% of successes sit on the opposite side of the boundary from their nearest v2
  neighbour** — a median of 95 / 196 logits per unit of cosine distance.

Two structural reasons this is expected rather than surprising:

1. Both arms fit their training set **perfectly** (train accuracy 1.0000, median |logit| ≈
   10 — `attribution_findings.md`), 546 rows in 5376 dimensions. The hyperplane is
   massively underdetermined, so the probe's behaviour a short step off the training
   points is essentially unconstrained.
2. Being a success is defined by the *boundary*, not the neighbourhood, and the attacker
   submits ~250 candidates per rotation and keeps only the 40–80 that fell on the wrong
   side. By selection those are exactly the points where the boundary cuts through an
   otherwise well-covered region. Near-and-wrong is the definition of an adversarial
   example.

The redundancy in §6 is about **value per row**, not geometric duplication: a new pair is
a genuinely new error, and is worth exactly as much as re-drawing an old one.

## 6. The new pairs are interchangeable, not defective

Trained *alone* (plus base), against size-matched random draws from the earlier vintage:

| arm | v3-only | 3 × random same-size draw from v2 |
|---|---|---|
| gptoss (232 rows) | 0.7630 | 0.7702 ± 0.0049 |
| deepseek (172 rows) | 0.7683 | 0.7724 ± 0.0051 |

Equal value per row. Nothing is wrong with iteration 3's data; there is simply nothing
left for another draw from the same distribution to add.

## What the data actually is

The 116 new gptoss pairs are two narrow niches. The false positives are
violent-or-anatomical vocabulary with benign intent — basic self-defense moves, breaking
down a whole chicken, applying a tourniquet, safe rugby tackling, the Roman testudo
formation, filleting a fish. The false negatives are specific-risk-domain compliance — a
suicide method, a school-attack plan, DIY wisdom-tooth extraction, balcony parkour,
insulin dosing. Neither niche appears in the four eval splits; `eval_ant_hh` in
particular is short, blunt HH-style dialogue against a red-team median of 1364 source
characters.

## Why the loop cannot broaden on its own

The attacker in these arms is blind **by design** — that is what the ablation is for:

- `batch_submissions: true` — one API call, all 5 conversations in it, session ends. The
  attacker never sees a probe or judge verdict.
- `view_limit: 0` — no past attempts injected either.
- `cross_iteration_memos` is unset, i.e. **off** (no `.iteration_memos.jsonl` exists for
  either arm), and the rolling round memo resets per iteration.

So the iteration-3 generator is distributionally identical to the iteration-1 generator.
The only thing that changes across iterations is the *acceptance filter* ("is the current
probe wrong here?"). A fixed proposal distribution behind a narrowing filter mines one
niche deeper; it does not explore. Success rates show the oscillation rather than
convergence — gptoss FP 10.8% → 31.9% → 17.1%, FN 89.3% → 18.8% → 31.6% across the three
rotations: the loop chases whichever direction the previous retrain over-corrected.

## The metric may also be hiding what did change

Success is defined by the probe's **thresholded** prediction disagreeing with the judge.
The verdict is **threshold-free** AUROC. In the committed single-seed runs the last
iteration moves accuracy while AUROC is flat, and moves TPR@FPR=1% the other way:

| arm | | iter1 | iter2 | iter3 |
|---|---|---|---|---|
| gptoss | AUROC | 0.8908 | 0.8908 | 0.9207 |
| | accuracy | 0.7058 | 0.7996 | **0.8539** |
| | TPR@1%FPR | 0.5104 | 0.5608 | **0.3877** |
| deepseek | AUROC | 0.8687 | 0.8759 | 0.8880 |
| | accuracy | 0.6920 | 0.7095 | **0.7228** |
| | TPR@1%FPR | 0.3665 | 0.5027 | 0.4592 |

The vintage sweep recorded AUROC only, so it cannot see this. Single seed, so treat the
direction as a lead rather than a result.

## A design smell found on the way

`preprocessing` writes one LLM counterpart per success, so half the training set is
written by the attacker and half by `openai/gpt-5.1`. That provenance axis is **99.97%
(gptoss) / 99.93% (deepseek) linearly decodable** from the same activations (5-fold CV),
and because 71% of successes are false negatives, provenance predicts the *label* 69–70%
of the time. The fitted direction is not dominated by it here (cos 0.09 / 0.15; the
provenance direction alone scores 0.19–0.71 across the eval splits), but it is a large
free shortcut sitting in the training data.

## Ways forward

### Diagnosis — cheap, and worth doing regardless

1. **Use the deterministic instrument as the stopping rule.** The 80-fit seed sweep cost
   hours and could not resolve below 0.01; `why_iter3_null.py` resolves 0.003 in minutes.
   Run it after each iteration and stop when the marginal gain is under whatever you care
   about.
2. **Record accuracy / TPR@FPR / AUPRC next to AUROC** in the vintage sweep and in the
   gate's columns. The last iteration's real gain looks like calibration, which AUROC
   cannot see.
3. **Track `cos(w_vk, w_oracle[split])` per iteration** as a leading indicator. It is flat
   here; if a change to the loop makes it climb, that change is working — and it says so
   before the eval number does.
4. **Fix the noise floor at its source**: `best_epoch` restoration is a no-op (shallow
   `state_dict().copy()`, see `attribution_findings.md` §1) and validation is ~166 rows on
   a model saturated by epoch 4. Deep-copy the snapshot, enlarge/stratify validation, or
   average the head over seeds — until then no single-run iteration comparison is readable.

### Changes to the loop — this is where the diagnosis points

1. **Give the attacker cross-iteration state** — `attacker.cross_iteration_memos: true`.
   One line, and the memo explicitly tells the next iteration what is already patched.
   Note it changes what the blind-attacker ablation measures, so run it as a *new arm*,
   not as a patch to these two.
2. **Make the novelty guard semantic — but not by raw activation cosine.** The difflib
   guard only sees surface form. The obvious replacement (pooled-activation cosine to
   stored successes) is *wrong here*: §5a shows same-label and opposite-label pairs differ
   by only 0.013 of cosine, so a τ threshold would reject genuinely new errors as often as
   clones. Use something the label actually moves — distance along the probe's own
   readout direction, or a sentence-embedding of the scenario — and validate the metric
   against §5a's same/opposite-label separation before trusting it.
3. **Force coverage explicitly**: schedule the attacker over a harm/topic taxonomy, or
   rotate attacker models per iteration, so successive iterations must occupy different
   regions instead of re-mining the generator's favourite one.
4. **Select what enters training, not just what is a probe error.** `filter_dataset` drops
   the most bag-of-words-confident rows; nothing selects for coverage. An acquisition step
   preferring successes far from the existing training set in activation space is what
   turns "116 interchangeable pairs" into 116 that are not.
5. **Break the provenance shortcut** — have both halves of a pair share an authorship
   distribution (e.g. rewrite the source with the same model that writes the counterpart),
   so "who wrote it" stops correlating with the label.
6. **Rebalance base against red-team.** 778 red-team rows against 50 base rows is 94% of
   the training signal from one narrow attack distribution.
7. **Align the search objective with the verdict metric.** If AUROC is how you judge,
   define success by rank (this negative outranks the p-th percentile of known positives)
   rather than by threshold crossing; otherwise promote the threshold metrics to
   first-class and stop reading AUROC alone.

The smallest change that tests the diagnosis directly is **(loop 1) + (loop 2) as a new
arm, read out with (diagnosis 1) + (diagnosis 3)** — if the alignment trend starts
climbing, the diagnosis is right.
