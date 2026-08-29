# Does any of it replicate? — arm 3N, three independent generations

`AUGMENTATION_FINDINGS.md` reports single-draw numbers: one generation of 107 samples,
one fit per set, one eval. This document runs the same procedure again — twice — and asks
which of those numbers survive. The short answer is that the headline result does not,
the per-family ordering partly does, and a **ranked** acceptance rule beats the loop's
**thresholded** one in every draw.

Setup is arm 3N throughout, unchanged: `google/gemma-3-27b-it` layer 32,
`linear_then_softmax`, single probe, seed 42, base `data/instructions_llama70b_50.jsonl`,
dev `dev_samples/instructions` (436 rows), eval `eval_sets/instructions` (7 full splits,
1302 rows). Every fit inherits architecture and metadata from `probe_iter12.pkl`. The
floor is `base ∪ 62 accepted`, refit in-process for every experiment below:
**dev 0.83106 / eval 0.81481**, reproducing `probe_iter13` to the digit.

## 0. The 62 accepted samples do not contain the concept

The loop only ever trains `base ∪ accepted`, so the accepted rows have only ever been
measured as an increment. Trained alone (`scripts/fit_accepted_only.py`):

| set | rows | dev | eval |
|---|---:|---:|---:|
| base only | 50 | 0.7573 | 0.7779 |
| **62 accepted only** | **62** | **0.6250** | **0.6172** |
| base ∪ 62 | 112 | 0.8311 | 0.8148 |

Per split against `base ∪ 62`: −0.044 `anthropic_harmless_refusal`, −0.057
`bbq_substitution`, −0.291 `hc_context_drift`, −0.208 `hc_contradiction`, **−0.617
`mm_substitution`**, −0.143 `oig_context_drift`, −0.025 `oig_omission`.

`mm_substitution` at 0.2422 is *inverted*, not weak — 1−AUROC = 0.758, and dev agrees
(0.2647). The two splits that survive are the ones whose class boundary is blunt surface
behaviour (a refusal, an inserted stereotype). The accepted rows sharpen a boundary the
base data establishes; they do not reconstruct it. The +0.037 the loop bought is
regularization on top of on-distribution data, not a generator that learned the concept.

(62 rows is barely over the 49-row optimizer-step threshold, so this fit is genuinely
data-starved. It answers "what do the accepted rows carry alone", not "what could 62
well-chosen rows do".)

## 1. Vocabulary: the training data is a different genre from the eval set

`scripts/word_frequency.py` — message content only, lowercased, scikit-learn's
`ENGLISH_STOP_WORDS` plus contraction tails, tokens ≥ 3 chars, rates per 10k content
words. Six corpora plus the two eval-delta halves of the imitations.

Pairwise cosine of unigram rate vectors:

```
                base  accepted  imitated  eval_pos  eval_neg    poison       dev      eval
base           1.000     0.219     0.211     0.204     0.146     0.245     0.245     0.287
accepted       0.219     1.000     0.610     0.488     0.491     0.317     0.158     0.179
imitated       0.211     0.610     1.000     0.708     0.870     0.316     0.175     0.202
eval_pos       0.204     0.488     0.708     1.000     0.269     0.275     0.152     0.171
eval_neg       0.146     0.491     0.870     0.269     1.000     0.239     0.132     0.156
poison         0.245     0.317     0.316     0.275     0.239     1.000     0.200     0.230
dev            0.245     0.158     0.175     0.152     0.132     0.200     1.000     0.827
eval           0.287     0.179     0.202     0.171     0.156     0.230     0.827     1.000
```

- **dev ≈ eval (0.827, 59/100 top words shared).** The scoring set is a good proxy for
  the test set. Everything generated sits at 0.15–0.32 with 43–54% of its tokens outside
  eval's vocabulary (dev: 19.5%).
- **The generator drifted into software-engineering roleplay.** Most over-represented vs
  eval: `verify, int, calibration, deployment, dump, scores, python, log, blue, seconds`
  (accepted); `columns, intervals, kafka, output, nov, remote, grpc, fedex` (poison).
  Eval is medical/factual QA and bias substitution (`cancer, fever, selenium, steam`).
- **Accepted is no closer to eval than poison** (0.179 vs 0.230). Lexical distance does
  not separate the batches the loop kept from the ones it rejected.
- **The imitations copied style, not content** (0.610 to their source, against 0.317 for
  any other generated pair; 51.5% of their tokens come from accepted's vocabulary). They
  kept the register and swapped the props: `calibration/scores/plate/weight` out,
  `orders/customer/api/portal` in.
- **The eval-positive and eval-negative halves are nearly disjoint (0.269)** — `eval_pos`
  is literal code (`int` 121.0 vs 1.9, `str`, `def`, `len`), `eval_neg` is ops narrative
  (`verify`, `checklist`, `deployment`, `kubectl`, `cpu`). A clean lexical signature that
  turns out to be worthless for selection (§3).
- **Lexical distance from eval predicts nothing about transfer.** The imitations are the
  *furthest* corpus from eval by OOV share (50.3%) and were the most valuable addition
  ever measured.

## 2. Per-family eval scores of the 107 imitations

Each family added to `base ∪ 62` on its own (`like62_directions_results.csv`):

| family | rows | dev | Δdev | eval | Δeval |
|---|---:|---:|---:|---:|---:|
| it11b3 | 12 | 0.83254 | +0.0015 | 0.83580 | +0.0210 |
| it4b3 | 12 | 0.84098 | +0.0099 | 0.83455 | +0.0197 |
| it1b1 | 14 | 0.82998 | −0.0011 | 0.83035 | +0.0155 |
| it5b4 | 12 | 0.81188 | −0.0192 | 0.81676 | +0.0020 |
| it7b2 | 14 | 0.80190 | −0.0292 | 0.80284 | −0.0120 |
| it2b0 | 14 | 0.80413 | −0.0269 | 0.80071 | −0.0141 |
| it9b1 | 14 | 0.79725 | −0.0338 | 0.79361 | −0.0212 |
| it0b4 | 15 | 0.76981 | −0.0612 | 0.76606 | −0.0487 |
| **union** | **107** | **0.83344** | **+0.0024** | **0.85043** | **+0.0356** |

The parts do not predict the whole: the families average −0.0047 and sum to −0.0378, yet
together they gain +0.0356. Within the draw, corr(Δdev, Δeval) = **+0.987** — dev is not
misleading about any individual fit.

## 3. Selecting on the sign of the eval delta fails

| set | families | rows | dev | eval | Δeval |
|---|---|---:|---:|---:|---:|
| eval-positive | it11b3, it4b3, it1b1, it5b4 | 50 | 0.78600 | 0.81696 | +0.0022 |
| eval-negative | it7b2, it2b0, it9b1, it0b4 | 57 | 0.82553 | 0.81851 | +0.0037 |
| all | — | 107 | 0.83344 | 0.85043 | +0.0356 |

Grouping the four families that each *helped* eval gives +0.0022; the four that each
*hurt* give +0.0037. **Oracle selection on the test metric loses to no selection at all**,
and the harmful half comes out marginally ahead. The individual deltas carry no
information that survives pooling.

## 4. The +0.036 does not replicate

`scripts/generate_like_accepted.py` re-run at its defaults — same model, same eight
families, same four shots, temperature 1.0, nothing about eval in the prompt. Rep 3 adds
`--no-pairing-hint`, which **removes** the "prefer matched pairs" rule from the system
prompt without replacing it (the prompt then says nothing about pairing either way);
everything else is byte-identical.

| family | rep1 Δdev | rep1 Δeval | rep2 Δdev | rep2 Δeval | rep3 Δdev | rep3 Δeval |
|---|---:|---:|---:|---:|---:|---:|
| it11b3 | +0.0015 | +0.0210 | −0.0333 | −0.0185 | −0.0147 | −0.0076 |
| it4b3 | +0.0099 | +0.0197 | +0.0009 | +0.0265 | −0.0088 | +0.0154 |
| it1b1 | −0.0011 | +0.0155 | −0.0006 | +0.0217 | −0.0077 | +0.0097 |
| it5b4 | −0.0192 | +0.0020 | −0.0239 | −0.0158 | −0.0157 | −0.0022 |
| it7b2 | −0.0292 | −0.0120 | −0.0201 | −0.0098 | −0.0197 | +0.0019 |
| it2b0 | −0.0269 | −0.0141 | −0.0295 | −0.0259 | −0.0174 | +0.0037 |
| it9b1 | −0.0338 | −0.0212 | −0.0127 | −0.0047 | −0.0166 | −0.0082 |
| it0b4 | −0.0612 | −0.0487 | −0.0206 | −0.0158 | −0.0607 | −0.0430 |
| **union** | **+0.0024** | **+0.0356** | **−0.0211** | **−0.0165** | **−0.0305** | **−0.0073** |
| union eval | | 0.85043 | | 0.79834 | | 0.80756 |
| rows | 107 | | 105 | | 104 | |

**Three draws of the same procedure span 0.798–0.850 on eval, mean 0.8188 — just under
the 0.8148 floor.** The +0.036 was the upper tail. This also retires the volume
explanation: rep 2 and rep 3 make the same 112 → ~217 change to the training-set size and
lose.

~~What does replicate: **it0b4 is worst in all three draws** and **it1b1 and it4b3 are
eval-positive in all three**.~~ **Retracted — see §7.** Five further draws (reps 4-8) put
it0b4 at rank 1 in rep 4 and rank 2 in rep 7; across seven draws it is last only twice.
The three-draw pattern was an artifact.
Cross-draw correlations are weak — rep1~rep2 Δdev +0.34 / Δeval +0.49; rep1~rep3 +0.84 /
+0.78; rep2~rep3 +0.28 / +0.52. The prompt-modified draw resembles rep 1 more than the
identical-prompt draw does, which is the cleanest statement of how far draw noise
dominates.

**The pairing hint could not be shown to matter.** Removing it moved pairing from ~98% to
86.5% of rows (matched pairs 52 → 45), and the union landed inside the range the two
hinted draws already spanned. Note the shots are themselves drawn from paired accepted
batches, so the *signal* remains even when the *instruction* is gone.

Within each draw dev still tracks eval tightly (+0.987, +0.957, +0.937), and dev calls
every union correctly. Dev is not the failure point.

## 5. Harmful data composes; helpful data does not

Adding one more family to each draw's dev-positive set:

| policy | rep1 dev | rep1 eval | rep2 dev | rep2 eval |
|---|---:|---:|---:|---:|
| dev-positive only | 0.80740 | 0.83109 | 0.83194 | 0.84135 |
| + least dev-negative (it1b1 both) | 0.84842 | 0.84144 | 0.82946 | 0.82366 |
| + most dev-negative | 0.75477 | 0.77966 | 0.81417 | 0.81365 |

The *least* dev-negative family helps rep 1 and hurts rep 2 — no stable sign. The *most*
dev-negative family hurts both draws on both metrics, the only contrast in this document
that agrees across draws, and the damage tracks what the family did alone (rep 1: it0b4
scored −0.0487 alone, costs −0.0514 here; rep 2: it11b3 −0.0185 alone, costs −0.0277).

So dev-based acceptance earns its keep as a **filter against harm**, not as a search for
gain. The rejection half of the rule replicates; the acceptance half is a coin flip.

## 6. Rank, don't threshold

`base ∪ 62 ∪ (top-k families by Δdev)`, for each draw:

**Eval**

| k | rows (1/2/3) | rep1 | rep2 | rep3 | mean | spread | above floor |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | 38/37/45 | 0.84144 | 0.83052 | 0.83513 | **0.8357** | **0.0109** | **3/3** |
| 5 | 64/63/73 | 0.84022 | 0.82108 | 0.82269 | 0.8280 | 0.0191 | 3/3 |
| 7 | 92/91/89 | 0.83611 | 0.80157 | 0.81779 | 0.8185 | 0.0345 | 2/3 |
| 8 | 107/105/104 | 0.85043 | 0.79834 | 0.80756 | 0.8188 | 0.0521 | 1/3 |

**Dev**

| k | rep1 | rep2 | rep3 | mean | spread | above floor |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.84842 | 0.83796 | 0.83925 | **0.8419** | **0.0105** | **3/3** |
| 5 | 0.81947 | 0.82881 | 0.83829 | 0.8289 | 0.0188 | 1/3 |
| 7 | 0.83156 | 0.81450 | 0.83028 | 0.8254 | 0.0171 | 1/3 |
| 8 | 0.83344 | 0.80999 | 0.80059 | 0.8147 | 0.0328 | 1/3 |

Mean eval falls monotonically in k (0.8357 → 0.8280 → 0.8185 → 0.8188) and spread rises
monotonically (0.011 → 0.019 → 0.035 → 0.052). **k = 3 is the only setting above the
floor in all three draws on both metrics**, and it uses a third of the rows.

Compare the loop's actual rule (`min_auroc_gain: 0.0`) on the same three draws: it accepts
**2, 1 and 0** families. In rep 3 it banks nothing at all, while the top-3 of that same
draw is worth +0.020 eval. The per-draw mean Δdev is remarkably stable (−0.0200, −0.0175,
−0.0202), so the deltas look like a real per-family ordering sitting on a draw-level
offset — which an absolute threshold cannot see and a rank can.

Rep 2's top-5 is the stress test: it admits `it0b4`, worst family in the other two draws,
because rep 2's own ranking put it 5th. Result +0.0063 — still above the floor, but the
weakest k=5 cell, and it is what drags k=5 below k=3.

## 7. Seven draws: the ranking does not survive

Reps 4-8 add five more same-prompt draws (rep 3 excluded — it is the no-pairing-hint arm).
Δdev rank per family, 1 = highest:

| family | rep1 | rep2 | rep4 | rep5 | rep6 | rep7 | rep8 | mean rank | Δdev span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| it4b3 | 1 | 1 | 6 | 3 | 5 | 4 | 2 | 3.1 | 0.0221 |
| it1b1 | 3 | 2 | 7 | 1 | 4 | 5 | 1 | 3.3 | 0.0284 |
| it9b1 | 7 | 3 | 3 | 2 | 1 | 8 | 6 | 4.3 | 0.0544 |
| it0b4 | 8 | 5 | 1 | 8 | 3 | 2 | 4 | 4.4 | **0.0808** |
| it7b2 | 6 | 4 | 4 | 5 | 8 | 1 | 7 | 5.0 | 0.0375 |
| it11b3 | 2 | 8 | 5 | 6 | 2 | 7 | 5 | 5.0 | 0.0475 |
| it5b4 | 4 | 6 | 2 | 4 | 6 | 6 | 8 | 5.1 | 0.0375 |
| it2b0 | 5 | 7 | 8 | 7 | 7 | 3 | 3 | 5.7 | 0.0256 |

**Mean pairwise Spearman of the rankings is −0.007** (21 pairs, range −0.69 to +0.71) —
indistinguishable from reshuffling. it4b3 and it1b1 have the best mean ranks but are top-2
in only 3 of 7 draws each and fall to 6th/7th once. it0b4 swings 0.0808 in Δdev between
draws, 16× the single-fit noise floor. Three families occupy both rank 1 and rank 8 across
the seven.

Union eval across the seven: 0.8504, 0.7983, 0.7842, 0.8049, 0.8331, 0.8084, 0.8083 —
mean Δeval −0.0023, above floor 2/7, spread 0.066.

## 8. Pooled across draws, the family effects are large and clean

Pooling one family's rows from all seven draws (~85-105 rows) makes the effects legible:

| set | rows | dev | Δdev | eval | Δeval |
|---|---:|---:|---:|---:|---:|
| it1b1 alone | 101 | 0.86682 | +0.0358 | **0.85273** | **+0.0379** |
| it4b3 alone | 84 | 0.81942 | −0.0116 | 0.82940 | +0.0146 |
| it9b1 alone | 90 | 0.78626 | −0.0448 | **0.77415** | **−0.0407** |
| it4b3 + it1b1 | 185 | 0.85004 | +0.0190 | **0.85543** | **+0.0406** |
| + it9b1 | 275 | 0.80324 | −0.0278 | 0.79947 | −0.0153 |
| rank-1 family of each draw | 88 | 0.86567 | +0.0346 | 0.84834 | +0.0335 |
| rank-2 family of each draw | 85 | 0.82909 | −0.0020 | 0.81506 | +0.0003 |
| rank 1+2 | 173 | 0.86679 | +0.0357 | 0.84653 | +0.0317 |

Pooling is where the gain lives: every pooled set is at or above the floor, against 2/7 for
the single-draw unions. `it4b3 + it1b1` at 0.85543 is the best eval measured in this arm.

**it9b1 is poison** (−0.0407 alone; adding it to the winning pair costs 0.056), **and it had
the third-best mean Δdev rank of the eight families.** Averaging seven independent estimates
of a family's worth still gets its sign wrong. The signal is visible at ~90 pooled rows and
invisible at 12, and re-measuring the 12-row increment does not recover it.

## 9. The rank slots, fitted

Pooling by rank slot instead of by family (85-105 rows each, size-comparable):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Δeval | **+0.0335** | +0.0003 | +0.0043 | −0.0262 | +0.0016 | −0.0403 | −0.0106 | −0.0480 |
| Δdev | +0.0346 | −0.0020 | −0.0166 | −0.0377 | −0.0105 | −0.0935 | −0.0215 | −0.0731 |

corr(slot, Δeval) = −0.80, but it is a step function, not a gradient: slot 1 is the only
real gain, slots 2/3/5 sit exactly on the floor, and below that the order is scrambled
(slot 3 > slot 2, slot 5 > slot 4, slot 7 > slot 6). Slot 1 is 52 of its 88 rows (59%)
it1b1 + it4b3, against 29% for slot 2 — the argmax is *enriched* in the good families
rather than *identifying* good batches.

## 10. Nothing on the surface predicts the outcome

Across the eight size-matched slot corpora, none of these order the result:

- **Structure** — bulleted %, code-fence %, refusal %, constraint-word %, pairing %, user
  and assistant length. The only metric with any relationship is the neg/pos length ratio
  (corr +0.68 with Δeval), and slot 1 is the sole slot above 1.0 and the sole real gain, so
  it is one point driving the correlation. Slot 6 is the most code-heavy and most
  constraint-laden slot and the second-worst result.
- **Vocabulary distance to eval** (`word_frequency.py --slots`) — cosine to eval runs
  0.114-0.299 across slots with corr +0.48 to Δeval, and it misorders both ends: slot 5 is
  *closest* to eval (0.299) and scores +0.0016; slot 1 wins while sharing only 7 of eval's
  top 100 words. Pairwise cosine among the slots is 0.26-0.70 — they are one genre.
- Within a family, the two ends of a swing can look completely different (it0b4's best draw
  has 60-char answers and a 0.63 neg/pos ratio; its worst has 166-char answers and 1.08 —
  cosine 0.209 between them) or completely identical (it9b1's best and worst draws are both
  100% bulleted with a 1.000 ratio and cosine 0.536, and still differ by 0.054 in Δdev).

## 11. The effect travels with structure, not vocabulary

`scripts/generate_word_swap.py` rewrites every conversation of a slot corpus one at a time,
rebuilding its subject matter on the other slot's five most one-sided words while keeping
the turn count, per-turn length, register, formatting and the instruction/reply relation.
736 calls; 0 label flips, 0 shape violations, 0 duplicates. Four replications per arm:

| arm | rep1 | rep2 | rep3 | rep4 | mean | source |
|---|---:|---:|---:|---:|---:|---:|
| slot-1 conversations, slot-8 words | −0.0036 | +0.0247 | +0.0306 | +0.0229 | **+0.0186** | +0.0335 |
| slot-8 conversations, slot-1 words | −0.0158 | −0.0069 | −0.0369 | −0.0304 | **−0.0225** | −0.0480 |

**All four A replications beat all four B replications**, on eval and on dev, with no
overlap in range. Each arm keeps 55-60% of its source's effect after its vocabulary is
entirely replaced. The word-frequency contrast was a marker of which families landed in
which slot, not a cause.

## 12. The accepted 62 and the generated rows cover each other's failures

| set | rows | dev | eval |
|---|---:|---:|---:|
| base only | 50 | 0.7573 | 0.7779 |
| 62 accepted only | 62 | 0.6250 | 0.6172 |
| base ∪ slot1 (no accepted) | 138 | 0.80528 | 0.79124 |
| base ∪ 62 | 112 | 0.83106 | 0.81481 |
| base ∪ 62 ∪ slot1 | 200 | 0.86567 | 0.84834 |

slot1 is worth +0.0335 **on top of** the accepted 62 and −0.024 **instead of** them. Without
them it inverts `anthropic_harmless_refusal` (0.3922, i.e. 1−AUROC = 0.608) while improving
every other split — hc_contradiction 0.9256 vs 0.7452, mm_substitution 0.8741, oig_omission
0.7673. The accepted 62 alone invert `mm_substitution` (§0) and hold the refusal split up.
Neither set carries the concept; each covers the other's inverted split. That is why every
"replace the accepted rows" framing here fails and every "add to them" framing works.

## Consequences for the loop

> **Superseded in part by `SELECTION_FINDINGS.md`.** A second arm (8 draws imitating the
> loop's *rejected* near-miss batches) plus 16 pooled per-family fits show that (a) the
> per-draw and pooled measurements of the same family disagree in sign for 6 of 16
> families, so §7-§9's rank analysis was measuring noise on both ends; (b) ten selection
> rules built on measured ΔAUROC land at or below the floor, including "top-1 by Δdev";
> and (c) the best set found anywhere — eval **0.86254** — is two pooled families plus one
> arbitrary unselected draw.

- ~~**Replace the absolute `min_auroc_gain` threshold with a rank.**~~ **Weakened by §7-§9,
  retracted by `SELECTION_FINDINGS.md` §3.**
  At n=7 the Δdev ranking has ≈0 rank correlation between draws; top-3 selection worked at
  n=3 because slot 1 is enriched in the two good families, not because the ranking finds
  good batches. Taking the top **1** is the only cut that clearly beats the floor (+0.0335
  vs +0.0003 for the size-matched rank-2 set).
- **Trust dev to reject, not to accept.** Its bottom end replicates across draws and
  composes predictably; its top end does not.
- **A single accepted/rejected verdict is not a property of the batch.** Two draws of the
  same eight directions disagree on 2 of 8 signs and swing the union by 0.05.
- **Single-draw ladder entries need error bars.** Every row in `AUGMENTATION_FINDINGS.md`
  is n=1 against a between-draw spread of ~0.05. Only the two clearly-harmful arms in §7
  there (−0.038, −0.030) have margin to survive this.
- **Vocabulary diagnostics do not substitute for a fit.** Neither distance from eval nor
  the pos/neg lexical signature orders the data the way the probe does, and §11 shows why:
  swap a corpus' vocabulary wholesale and its value barely moves.
- **Measure families across draws, not batches within one.** A family's worth is legible at
  ~90 pooled rows and invisible at 12; seven averaged 12-row estimates still put the worst
  family third. If the loop is to select at all, it should pool a direction's output across
  iterations before judging it.
- **Track per-split inversions, not just the mean.** Both the accepted rows and the best
  generated set invert a split when trained alone (§0, §12); the mean hides it and the
  complementarity that makes the union work is only visible per split.

## Reproducing

```bash
V=.venv_claude/bin/python
$V scripts/fit_accepted_only.py                                    # §0
$V scripts/word_frequency.py --top 25                              # §1
$V scripts/fit_mixed_directions.py --generated data/instructions_like_accepted62.jsonl \
   --score-families --tag like62                                   # §2
$V scripts/fit_mixed_directions.py --generated data/instructions_like_accepted62.jsonl \
   --families it11b3,it4b3,it1b1,it5b4 --tag like62pos             # §3
$V scripts/generate_like_accepted.py --out data/instructions_like_accepted62_rep2.jsonl
$V scripts/generate_like_accepted.py --no-pairing-hint \
   --out data/instructions_like_accepted62_rep3.jsonl              # §4
$V scripts/fit_mixed_directions.py --generated data/instructions_like_accepted62_rep3.jsonl \
   --score-families --tag like62rep3
$V scripts/fit_mixed_directions.py --generated data/instructions_like_accepted62_rep2.jsonl \
   --families it4b3,it1b1,it9b1 --tag top3rep2                     # §6
```

```bash
# §7-§9 — five more draws, pooled families, the eight rank slots
for r in 4 5 6 7 8; do
  $V scripts/generate_like_accepted.py --out data/instructions_like_accepted62_rep$r.jsonl
  $V scripts/fit_mixed_directions.py --generated data/instructions_like_accepted62_rep$r.jsonl \
     --score-families --tag like62rep$r
done
$V scripts/fit_mixed_directions.py --generated data/union_it1b1.jsonl --tag u_it1b1
$V scripts/fit_mixed_directions.py --generated data/union_slot1.jsonl --tag u_slot1
$V scripts/word_frequency.py --slots --top 25 \
   --csv results_gen_gemma27b_instructions_nemotron/word_frequency_slots.csv   # §10
$V scripts/generate_word_swap.py                                              # §11
$V scripts/fit_base_plus.py data/union_slot1.jsonl                            # §12
```

Results CSVs: `like62*_directions_results.csv`, `top{3,5,7}rep{1,2,3}_*.csv`,
`u_{it1b1,it4b3,it9b1,fixed2,fixed3,rank1,rank2,rank12,slot1..slot8}_*.csv`,
`swap_*_directions_results.csv`, `word_frequency{,_slots}.{txt,csv}`. Pooled corpora:
`data/union_*.jsonl`, `data/swap_*.jsonl`. Probes: `cand_*`, `gen_u_*`, `baseplus_*`,
`gen_accepted_only.pkl`.
