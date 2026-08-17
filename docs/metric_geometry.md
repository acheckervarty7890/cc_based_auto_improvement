# Is there a geometry where "close" means "same label"? (hu_harm, gemma-3-27b L32)

`why_last_iteration_adds_nothing.md` ends with two loop fixes that both need a metric it
does not have. Loop-fix 2 wants a *semantic* novelty guard and explicitly rules out the
obvious choice — "pooled-activation cosine to stored successes is **wrong here**" — on the
strength of §5a, which found a new-in-v3 success sitting as close to the v2 training set
as a deliberately opposite-label counterpart sits to its own source. Loop-fix 4 wants an
acquisition rule preferring successes "far from the existing training set in activation
space", and has no validated way to measure far.

The note's own instruction is to "use something the label actually moves — distance along
the probe's own readout direction, or a sentence-embedding of the scenario — and validate
the metric against §5a's same/opposite-label separation before trusting it."

This is that validation, run over 21 candidate geometries, both arms, everything off
activations already on disk (no gemma-3-27b forward). Scripts:
`metric_geometry_features.py` (one pass over the 778/878 per-conversation blobs, caching
five poolings plus the probe's logit and weight-Jacobian), `metric_geometry.py` (the
sweep), `metric_two_factor_guard.py`, `metric_pooling_transfer.py`,
`metric_geometry_report.py`. Raw numbers in
`results_hu_harm_gemma27b_batch_ablation/vintage/metric_{geometry,two_factor_guard,pooling_transfer}.json`,
tables in `metric_geometry_tables.md`.

## The acceptance test

Every candidate is scored on the same measurements, and all fits see **only the v2 rows**,
so every number below is read off rows held out from the fit.

| column | question | good |
|---|---|---|
| `pairAUR` | does similarity predict *same label*, on held-out rows? | high |
| `scenAUR` | does it see *what the conversation is about* — is a source's own counterpart closer than an unrelated row? | high |
| `provAUR` | does it see *who wrote the row* (the doc's design smell)? | low |
| `hopPair` | is an opposite-label counterpart *farther* from its source than that source's nearest v2 row? | **low** |
| `durAUR` | does distance-to-v2 flag the ~40% of successes every reseeded v2 probe misses? | high |

`pairAUR`, `scenAUR` and `hopPair` are AUROCs rather than the note's raw cosine gaps
because AUROC is invariant to any monotone rescaling of the distance, and the raw gaps are
not — which turns out to matter (next section). `hopPair` is the paired form of §5a's
central comparison: same anchor rows on both sides, where the published version compared
all 389 couples against a maximum over 546 v2 rows and so inherited a set-size confound.

Two sanity checks passed before anything was read off the instrument. The baseline
`pool:mean` reproduces the published §5a numbers exactly — k-NN 56.0 / 51.7 / 48.3%,
nearest-neighbour label agreement 56.0%, on 778 rows / 546 in v2 / 116 successes / 44
durable. And the row counts, vintages and durable-hole labels are the same objects the
earlier scripts built.

## Result 0 — stretching a distance is cosmetic, and the sweep proves it

The direct answer to "can distances be expanded non-linearly so they look more separated":
yes, spectacularly, and it buys nothing whatsoever. `nl:expstretch` is exp(8·cos) on the
baseline — a strictly monotone transform. It takes §5a's same-label-minus-opposite-label
gap from **0.0133 to 151.6**, an inflation of about **11,000x**, and returns
**bit-identical values on every other column**: 0.540 / 0.900 / 0.615 / 0.770 / 0.482 /
0.419 / 0.684, the same figures as `pool:mean`, in both arms.

That is not a quirk of this transform. Nearest-neighbour identity, k-NN votes, and every
AUROC depend only on the *ordering* of distances, and a monotone map preserves orderings by
definition. So a stretched distance changes every plot and headline number while leaving
every decision — which row is nearest, which side of a threshold a candidate falls, what
gets selected into training — exactly as it was. Any real answer has to change the
geometry, the representation or the metric on it, not rescale it. Everything below changes
the geometry.

(This also re-derives §5a's own numbers as a check: the raw same/opposite means come back
as 0.8749 / 0.8616 for gptoss and 0.8874 / 0.8766 for deepseek, matching the published
table to four decimals.)

## Result 1 — a correction to §5a's chance line

§5a reads its k-NN numbers against 50%, on the stated grounds that "the new rows are
exactly class-balanced". The 232 new *rows* are, because each success is paired with an
opposite-label counterpart. But the k-NN test set there is the 116 **successes**, and
those are **70.7% positive** for gptoss and **82.6%** for deepseek — 71% / 83% of them are
false negatives, whose true label is the positive class.

So the constant classifier scores 70.7% / 82.6%, and the published 56 / 52 / 48% is not
"chance", it is far *below* chance. The conclusion §5a draws survives — proximity is not
predictive — and in fact gets stronger: in mean-pooled space, the nearest v2 neighbours of
a new success are actively *anti*-correlated with its label (balanced k-NN accuracy 0.383,
k-NN AUROC 0.419 at k=15). But the calibration sentence is wrong and the tables here
report balanced accuracy and AUROC instead.

## Result 2 — the metrics split in two, and the guard needs both halves

This is the finding. Plotting the 21 candidates on `pairAUR` against `scenAUR` gives two
disjoint clusters and an empty top-right corner:

| family | examples | `pairAUR` | `scenAUR` |
|---|---|---|---|
| **scenario-aware, label-blind** | `pool:mean`, `text:tfidf`, `text:minilm`, `pool:topz16`, `lin:whitened`, `lin:pcawhite`, `probe:wscaled`, `sup:winwhite` | 0.49–0.59 | 0.89–1.00 |
| **label-aware, scenario-blind** | `sup:lda`, `pool:last`, `probe:logit`, `probe:proj`, `probe:jac` | 0.59–0.76 | 0.29–0.65 |

**No candidate clears both bars, in either arm.** The best `pairAUR` among the metrics
that see the scenario (`scenAUR` ≥ 0.75) is `lin:whitened` at 0.588 / 0.563; the best
`scenAUR` among those that see the label (`pairAUR` ≥ 0.60) is `probe:jac` at 0.513 /
`pool:probe` at 0.721. Across the 20 distinct metrics the two columns are strongly
*anti*-correlated — Pearson r = **−0.795** (p < 0.0001) for gptoss and **−0.688**
(p = 0.0008) for deepseek, Spearman −0.744 / −0.573.

That is not a shortfall of effort, it is the shape of the problem. `preprocessing` mints
each counterpart by editing its source toward the opposite label, so scenario and label
are deliberately decorrelated in this corpus: a pair that shares everything except the
label is, by construction, the hardest possible case for a metric asked to track both.
A representation that averages over the whole conversation is dominated by the shared
scenario; a representation that isolates the boundary throws the scenario away.

**And a novelty guard needs both at once.** It must reject a re-skin (same scenario, same
label) and accept an opposite-label rewrite (same scenario, different label). A single
similarity threshold on a scenario-aware metric cannot do that — measurably: under
`pool:mean` the paired hop AUROC is 0.770 (gptoss) and 0.729 (deepseek), i.e. in ~75% of
cases a source's own opposite-label counterpart looks *more* similar than the nearest
training row of a genuinely new error. Any τ that would catch a clone drops the label
flips first. That is the doc's §5a result, now with a mechanism attached and generalised
to every geometry tested: **no single-metric threshold guard is available here**, in raw
cosine or in any of its 20 alternatives.

Result 5 below tests the way out that this implies.

## Result 3 — what each of the five suggestions actually bought

**Pooling (suggestion 3) — the one clear win, and not where it was expected.**
`pool:last` (the final unpadded token) is the only *representation* change that moves the
label columns substantially: `pairAUR` 0.650 / 0.756 against mean pooling's 0.540 / 0.534,
k-NN AUROC 0.869 / 0.733 against 0.419 / 0.580, and it is the top acquisition metric in
both arms. This is principled rather than lucky — the stack is causal, so the last
position is the only one whose residual stream has seen the entire conversation, which is
why last-token pooling is the standard readout for decoder classifiers. Mean pooling
dilutes a decisive turn across up to 1024 positions.

It survives the provenance control: restricted to the attacker-written half of v2, where
both sides share an author and the 99.9%-decodable authorship axis is unavailable, its
balanced k-NN accuracy is 0.781 / 0.561 against `pool:mean`'s 0.553 / 0.593. Its `provAUR`
is the highest in the sweep (0.723 / 0.700), so the control was necessary, and it passes.

But `pool:last` is *scenario-blind* (`scenAUR` 0.472 / 0.456 — at or below chance), so it
cannot be the novelty guard. What it is good for is the acquisition rule and, potentially,
the probe's own readout (Result 6).

`pool:probe` — the deployed head's own softmax-over-logits pooling — is worth a line of
its own: its k-NN AUROC is **0.138 / 0.377**, the most anti-predictive in the sweep. In
the representation the probe actually reads, a success's neighbours carry the *opposite*
label. That is close to tautological (successes are selected for being on the wrong side
of exactly that readout) and it is a compact restatement of why the loop's successes look
adversarial: they are.

**The probe's own readout (suggestion 1) — informative, but 1-D, and that disqualifies it
as a guard.** `probe:logit`, `probe:proj` and `sup:lda` all post `hopPair` ≈ 0.00, which
looks like a perfect pass. It is an artifact of dimensionality: with 546 training rows on
a line, every candidate has a near neighbour by pigeonhole, so "distance to the training
set" carries no information about content. `scenAUR` catches this precisely — 0.359 /
0.351 for `probe:logit`, 0.285 / 0.298 for `sup:lda`, i.e. a source's own counterpart is
*less* similar than a random unrelated row. This is exactly the sanity check the note asks
for, and it is why the verdict rule in `metric_geometry_report.py` requires `scenario`
before it will grant `guard`. The readout direction remains the right *gate* (Result 5),
just not the right *distance*.

`probe:jac` (the weight-gradient ds/dw — two rows close when they push the probe the same
way) is the most balanced single metric in the sweep, 0.629 / 0.656 label and 0.513 /
0.653 scenario, but it clears neither bar in both arms.

**Supervised metric learning (suggestion 2) — real, and confined to the label axis.**
`sup:lda` posts the highest `pairAUR` in the sweep (0.735 / 0.659) — genuinely
out-of-sample, since it is fitted on v2 and read on the new-in-v3 rows, with 5-fold CV
numbers in the JSON confirming it in-sample. So supervision does extract label structure
that the unsupervised geometries miss, which answers the "is it advisable" question in the
affirmative for the acquisition and readout uses. It does nothing for the guard: the
Fisher direction is 1-D and scenario-blind by construction. `sup:nca` and `sup:winwhite`
sit closer to the unsupervised control `unsup:pca64` than to `sup:lda`, so most of what
they do is the PCA projection rather than the supervision.

**Text embeddings (suggestion 4) — the clearest negative result.** `text:tfidf` and
`text:minilm` have the *highest* scenario scores in the sweep (0.997 / 0.986) and the
*lowest* label scores (0.542 / 0.509 — chance). A sentence embedding of these
conversations encodes the topic almost perfectly and the harm label not at all. That is
intuitive in hindsight: "how do I break down a whole chicken" and "how do I dismember a
body" share most of their lexical and semantic surface, and the entire red-team corpus is
built on exactly that kind of minimal pair. The note's suggestion to use "a
sentence-embedding of the scenario" is therefore right about what it measures and wrong
about what that buys — it is a good scenario detector and a useless label detector, so it
belongs on the scenario side of a two-factor guard, never alone.

**Manifold embeddings (suggestion 5) — confirmed not worth it.** `nl:tsne2` and `nl:umap`
both fail the label bar (0.531 / 0.531 and 0.554 / 0.527, i.e. chance) while producing the
visually separated clusters they always produce. `nl:tsne2` posts a striking-looking
balanced source-only k-NN accuracy of 0.920 on gptoss — and 0.593 on deepseek, so it does
not even replicate — on an embedding fitted *jointly on the rows it is scoring*, since
t-SNE has no out-of-sample transform. That makes it unusable as a submit-time guard
whatever its numbers (it would mean re-embedding the whole corpus per candidate), which the
report flags as `transductive`. UMAP does have an out-of-sample transform and is therefore
deployable in principle, and it still fails the label bar. The prior against these was
correct: they reorganise the picture without adding information.

## Result 4 — acquisition (loop-fix 4) does work, and is cheap

The one loop fix the sweep straightforwardly supports. "Distance to the v2 training set"
predicts which successes are **durable holes** — the ~40% that *every* one of 10 reseeded
vintage-2 probes misses, per §5b — well above chance, in both arms, in most geometries:

| metric | gptoss `durAUR` (CI95, source-only) | deepseek `durAUR` (CI95, source-only) |
|---|---|---|
| `pool:last` | 0.722 (0.62–0.82), src 0.719 | 0.771 (0.68–0.86), src 0.781 |
| `pool:last32` | 0.675 (0.57–0.77), src 0.657 | 0.767 (0.65–0.86), src 0.765 |
| `lin:whitened` | 0.654 (0.56–0.75), src 0.688 | 0.754 (0.65–0.85), src 0.701 |
| `pool:mean` (baseline) | 0.684 (0.58–0.78), src 0.680 | 0.739 (0.62–0.84), src 0.754 |
| `probe:wscaled` | 0.659 (0.55–0.76), src 0.648 | 0.727 (0.62–0.83), src 0.734 |

Every lower bound is clear of 0.5, though not by a wide margin at n = 116 / 86. It holds
under the provenance control (`src`, the same measurement against the attacker-written
half of v2 only, moves by ≤0.035), and it holds for the *baseline* metric
— so loop-fix 4 does not depend on any of this sweep's findings to be worth doing. §5b
proposed "keep a success only if k reseeded probes all miss it", which costs 10 probe-head
refits per iteration; the same selection is approximated at AUROC ~0.72–0.77 by a single
nearest-neighbour distance that costs nothing, and the two can be combined.

One honest qualification, from the reference predictors. Conversation length (`n_tokens`,
0.286 / 0.509) and the judge-side class (`label_is_positive`, 0.407 / 0.483) carry nothing,
so the distance is not a repackaging of those. But the probe's **own confidence at attack
time** (`abs_seq_logit`) reaches 0.708 on gptoss — competitive with `pool:last`'s 0.722 and
*better* than the mean-pooled baseline's 0.684 — while managing only 0.582 on deepseek.
So on gptoss an activation distance adds little over "how confidently was the probe wrong",
and on deepseek it adds a lot (0.771 vs 0.582). Both are free at submit time; the cheap
version of the rule should use them together, and the claim "activation distance is what
finds the durable holes" is only supported on one of the two arms.

## Result 5 — the two-factor guard, and why the obvious gate fails

Result 2 says the guard needs a scenario metric and a label metric together. The natural
construction is a conjunction:

> reject a candidate iff it is **scenario-similar** to a stored success **and** falls on
> the **same side** of the probe's boundary.

`metric_two_factor_guard.py` tests it, and the result is a clean negative with a clean
consequence.

**The gate must not use the probe's prediction.** Its soundness rests on the pairs it must
protect actually straddling the boundary. Over all couples they mostly do — 70.4% (gptoss)
/ 74.3% (deepseek). But restricted to the couples a guard actually meets, those built from
a **new success**, the straddle rate collapses to **8.6% / 17.4%**. The reason is
structural and worth stating plainly: a success *is* a conversation the probe gets wrong,
so the source's predicted side already equals its counterpart's true side, and the pair
lands on the same side of the boundary. The gate is blindest exactly where it is needed.
The judge's label separates 100% of couples by construction — but the judge runs *after*
the guard in the current pipeline (`tools.handle_submit_conversation` reserves the opener
before probe and judge precisely so a rejection costs no scoring), so using it means
inverting that order and paying a judge call per candidate, which is most of what the
guard was there to save.

**Activation cosine is not merely uninformative as a guard, it is destructive.** Because
uncentered pooled cosines all sit at 0.86–0.96, a threshold anywhere in the usual range
rejects everything. At τ=0.8, `pool:mean` rejects **100% of the genuinely new successes**
and 100% of the label-flipped counterparts, in both arms. Even at τ=0.95 it still drops
73–86% of the counterparts against 25–48% of the new rows — i.e. it preferentially
destroys the very rows that carry a flipped label. `pool:topz16` is worse. Had the
difflib guard been replaced with pooled-activation cosine at the existing τ=0.8, it would
have rejected the entire red-team yield.

**A text metric is the one that is safe.** `text:tfidf` sits at mean similarity 0.14–0.16
between unrelated conversations and 0.35–0.37 between a source and its own counterpart, so
at τ=0.8 it rejects **0% of new successes and 0–1.2% of counterparts**. That is the
profile a guard needs on the "must accept" side, and it is the direct vindication of the
note's instruction to use a sentence-embedding of the scenario — with the correction that
it works as a *scenario* detector at a high threshold, not as a label-aware distance.

What none of this measures is the half that matters most: the **catch** rate. This corpus
contains no clones to catch (the difflib guard reports 0% at τ=0.8, consistent with
`text:tfidf` firing on nothing here), so every column above is a false-rejection rate.
Establishing that a guard catches re-skins requires a set of deliberately re-skinned
successes, which has to be generated — that is the next experiment, and it is cheap.

## Result 6 — the pooling is worth more than the entire red-team loop

`metric_pooling_transfer.py` re-runs sections 1, 2 and 4 of `why_iter3_null.py` verbatim —
same L2 logistic regression, same C grid, same content-deterministic split, same vintages —
changing only which pooling the 5376-dim feature comes from. It reproduces the published
`mean` row **exactly** (v0 0.7256, v1 0.7774, v2 0.7711, v3 0.7744, ceiling 0.9156, align
0.1319), so the comparison below is like-for-like.

| arm | pooling | v0 (base only) | v1 | v2 | v3 | v3−v2 | ceiling | gap | align |
|---|---|---|---|---|---|---|---|---|---|
| gptoss | mean | 0.7256 | 0.7774 | 0.7711 | 0.7744 | +0.0033 | 0.9156 | 0.141 | 0.1319 |
| gptoss | **last** | **0.8440** | 0.9109 | 0.9292 | **0.9373** | +0.0081 | 0.9644 | **0.027** | 0.1113 |
| gptoss | probe\* | 0.9148 | 0.9009 | 0.8996 | 0.9101 | +0.0105 | 0.9850 | 0.075 | 0.0793 |
| deepseek | mean | 0.7256 | 0.7650 | 0.7715 | 0.7694 | −0.0021 | 0.9156 | 0.146 | 0.1247 |
| deepseek | **last** | **0.8440** | 0.9127 | 0.9035 | **0.9057** | +0.0023 | 0.9644 | **0.059** | 0.1127 |
| deepseek | probe\* | 0.9147 | 0.9140 | 0.8317 | 0.8781 | +0.0463 | 0.9850 | 0.107 | 0.0713 |

\* `probe` pooling uses `probe_iter2`'s frozen weights, and that probe was trained on base
plus the vintage-2 red-team data — so its "v0" row has already absorbed red-team
information through the pooling weights and is **not** a base-only number. `last` involves
no probe at all and is clean.

Three things fall out, and the first is the largest number in this whole line of work.

**Changing the pooling is worth about 2.4x the entire three-iteration red-team loop, and
it is free.** Under mean pooling the loop buys v0 → v3 = **+0.0488** (gptoss) / **+0.0438**
(deepseek). Switching mean → last, *with no red-team data at all*, buys **+0.1184** at v0.

But — and this is the part that would be lost by stopping at the headline — the loop does
not become redundant in the better representation, it becomes **more** productive: under
last pooling the same three iterations buy **+0.0933** (gptoss) and **+0.0617** (deepseek),
roughly double and 1.4x their mean-pooled gains. The two changes compound rather than
compete. The correct reading is not "pooling instead of red-teaming" but "the single
largest unclaimed gain is not in the data, and claiming it also makes the data worth more".

**§2's headroom is mostly the instrument, and what survives is one split.** The note's
transfer gap — ceiling minus achieved, the quantity that motivates "the information is in
the activations; the training data fails to expose it" — falls from 0.141 to **0.027**
(gptoss) and 0.146 to 0.059 (deepseek). Per split it essentially vanishes everywhere
except `eval_ant_hh`:

| split | gap, mean pooling | gap, last pooling |
|---|---|---|
| eval_ai_dilemmas | 0.164 | 0.007 |
| eval_balanced_refusal | 0.085 | 0.014 |
| eval_daily_dilemmas | 0.128 | 0.003 |
| **eval_ant_hh** | 0.188 | **0.084** |

`eval_ant_hh` is precisely the split the note singles out as never moving in any iteration
of either arm, and the only one whose *ceiling* falls under last pooling (0.9599 → 0.8688).
So the note's diagnosis narrows sharply and gets more useful: three quarters of the
apparent "the loop should be able to do better" headroom was mean pooling, and the genuine
remainder is concentrated in the one split whose data least resembles what the attacker
writes — short, blunt HH-style dialogue against a red-team median of 1364 source
characters. That is a coverage problem (loop-fix 3), not a metric problem.

**§4's load-bearing number survives, which matters.** The note rests its "a 4th and 5th
iteration will not converge on the eval directions" conclusion on the alignment
`cos(w_v3, w_oracle[split])` being tiny and flat. It is tiny in every pooling — 0.1319,
0.1113, 0.0793 (gptoss) and 0.1247, 0.1127, 0.0713 (deepseek) — so that conclusion is
robust to the representation and does not need re-deriving. So is the headline null:
v3 − v2 stays within ±0.01 under last pooling in both arms.

A caveat on scope. This is the note's *analysis instrument*, not the deployed probe, and
the two are different heads — the deployed one already pools with a softmax over its own
per-token logits, which is what the `probe` row proxies, and that pooling has the highest
ceiling of the three (0.9850). So the finding is "the measurement in §1–§4 was made in a
representation that costs ~0.16 AUROC of transfer", not "the deployed probe should switch
pooling". Whether the deployed head benefits is a separate experiment: train a probe with
last-token pooling and score it the normal way.

## Result 7 — §5 re-asked: do the new pairs move the training set toward eval?

§5 found they do not, in raw cosine. Re-asked with size matching (nearest-neighbour
similarity from each eval row to the 116/86 new successes versus to an equal-sized random
draw from v2, 20 draws), the answer is the same wherever the comparison is scale-meaningful:
under `pool:mean` the delta is +0.0014 / −0.0029 against a draw sd of 0.0025 / 0.0027 —
noise, replicating §5. `pool:probe` and `probe:wscaled` agree.

One consistent exception. Under `text:tfidf` the delta is **+0.0103 (sd 0.0023)** and
**+0.0085 (sd 0.0021)** — 4σ, same sign in both arms. So *lexically* the new pairs do move
the training set marginally toward the eval distribution, even though in activation space
they do not. The effect is small in absolute terms (against a baseline similarity of
0.14–0.16, about +7% relative) and it is the kind of thing that would show up if the new
rows are simply more lexically varied. It is worth one line, not a revision.

The euclidean-embedding metrics report deltas on arbitrary scales and are not comparable
here — and `nl:expstretch` reports a delta of +20.8, which is Result 0 arriving again to
make the point that raw distance deltas are not a currency.

## What to do

Ordered by measured value per unit of effort.

1. **Try last-token pooling before anything else in the loop.** It is the only change here
   worth more than the red-team loop itself (+0.118 at v0 against the loop's +0.049 over
   three iterations), it costs nothing, the activations are already on disk, and it makes
   the loop's own gain roughly double rather than replacing it. Two experiments, both
   cheap: re-run the note's diagnostics under it (done — Result 6), and train a *deployed*
   probe with last-token pooling and score it normally, which is the open question
   Result 6 cannot answer.
2. **Take loop-fix 4 (acquisition), and take the cheap version.** Distance to the training
   set predicts durable holes at AUROC 0.68–0.77 in both arms, under the baseline metric,
   robust to the provenance control. Combine it with `|logit|` at attack time, which is
   free and does most of the work on gptoss (0.708) and little on deepseek (0.582). This
   does not need any of the sweep's other findings to be worth doing.
3. **Build loop-fix 2's guard on text, not activations, and expect to gate it elsewhere.**
   `text:tfidf` at τ=0.8 rejects 0% of genuinely new successes and ≤1.2% of label flips;
   `pool:mean` at the same τ rejects 100% of everything. Before deploying it, generate a
   set of deliberately re-skinned successes and measure the catch rate — the one number
   this corpus cannot supply. Do **not** gate on the probe's predicted side: it separates
   only 8.6% / 17.4% of the pairs that matter (Result 5).
4. **Re-read §2's headroom as a coverage problem.** After the pooling fix, ~75% of the
   remaining transfer gap is `eval_ant_hh` alone. That is loop-fix 3 (force coverage over
   a taxonomy) pointed at one identifiable target, rather than a general call for better
   data.
5. **Do not spend more on metric search.** The scenario/label anti-correlation
   (r = −0.795 / −0.688 over 20 geometries, zero clearing both bars) is a property of a
   corpus built from minimal pairs, not of the geometries tried. A better single distance
   is not there to be found; a two-factor design or a supervised head is the way out.

And two things not to do, both now measured rather than argued: stretching a distance
non-linearly changes nothing that any decision depends on (Result 0), and manifold
embeddings buy no label information while t-SNE additionally cannot be deployed at all
(Result 3).

## Corrections to `why_last_iteration_adds_nothing.md`

- §5a's k-NN chance line of 50% is wrong. The test set is the 116/86 **successes**, which
  are 70.7% / 82.6% positive; the majority-class baseline is that, not 50%. The published
  56 / 52 / 48% is below chance, not at it. The conclusion is unaffected and strengthened.
- §5a's own-pair versus new→v2-NN comparison mixes 389 couples against a maximum over 546
  rows. The paired form (same anchors on both sides) is in `metric_geometry.json` as
  `hop_paired` and gives the same verdict, so this is a tightening rather than a change.
- §2's transfer gap is ~75% an artifact of mean pooling (Result 6). The sentence "the
  information is in the activations; the training data fails to expose it" should read
  "…the *pooling* fails to expose it", except on `eval_ant_hh`.
- Loop-fix 2's warning against pooled-activation cosine is correct and understated: at the
  existing τ=0.8 that guard would have rejected the entire red-team yield.

