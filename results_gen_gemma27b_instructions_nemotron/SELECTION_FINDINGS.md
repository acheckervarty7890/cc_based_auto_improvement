# Ten ways to choose training data, and the two that work

`REPLICATION_FINDINGS.md` established that a batch's measured ΔAUROC does not reproduce
across draws. This document asks the next question: **given many measurements, is there
any rule for assembling a good training set out of them?** Ten rules were tried. Eight
land at or below the floor. The two that work do not select on measurements at all.

Setup unchanged: `google/gemma-3-27b-it` layer 32, `linear_then_softmax`, single probe,
seed 42, base `data/instructions_llama70b_50.jsonl`, dev `dev_samples/instructions`,
eval `eval_sets/instructions` (7 splits, 1302 rows). Every fit inherits from
`probe_iter12.pkl`. Floor is `base ∪ 62 accepted`: **dev 0.83106 / eval 0.81481**.

## 1. A second arm: imitating the batches the loop REJECTED

`scripts/generate_like_accepted.py --keys nearmiss` few-shots the generator on the eight
**rejected** batches whose dev ΔAUROC came closest to zero (−0.0006 … −0.0069 of the 50
rejected batches carrying all ten samples). Everything else is identical to the accepted
arm — same prompt, 4 shots, 12 per family, temperature 1.0, same model. Eight draws.

| | mean union eval | mean Δeval | sd | above floor |
|---|---:|---:|---:|---:|
| **near-miss** (8 draws) | 0.81584 | **+0.0010** | 0.0198 | **5/8** |
| accepted (7 draws) | 0.81253 | −0.0023 | 0.0206 | 2/7 |

**Imitating rejected directions works at least as well as imitating accepted ones.** The
accepted arm's single best draw (+0.0356) is still the highest number, but on every
aggregate the near-miss arm is equal or better — including on dev (mean union dev 0.82259
vs 0.81095, Δdev −0.0085 vs −0.0201). The two source sets were separated by ~0.006 in Δdev
at scoring time, well inside the noise floor, so **the loop's accept/reject line ran
through the middle of a set of directions whose imitations behave the same.**

The near-miss families also *rank* stably where the accepted ones never did — mean pairwise
Spearman of the per-draw Δdev rankings **+0.554** (8 draws, 28 pairs) against **−0.007** for
the accepted arm. `it8b4` is last in 7 of 8 draws; `it1b3` is top-2 in 6 of 8.

## 2. Per-draw and pooled measurements of the same family disagree

Two ways to measure one family, over the same rows:

- **per-draw** — fit `base ∪ 62 ∪ (one 12-row batch)`, once per draw, average the results
  (120 fits: 8 families × 7 accepted draws + 8 × 8 near-miss draws)
- **pooled** — concatenate that family's batch from every draw (84–111 rows) and fit once
  (16 fits)

| family | arm | rows | mean per-draw Δeval | **pooled Δeval** |
|---|---|---:|---:|---:|
| it1b1 | acc | 101 | +0.0088 | **+0.0379** |
| it4b3 | acc | 84 | +0.0154 | +0.0146 |
| it6b1 | nm | 107 | −0.0060 | +0.0107 |
| it1b3 | nm | 104 | +0.0192 | +0.0086 |
| it5b0 | nm | 100 | +0.0127 | +0.0070 |
| it8b4 | nm | 111 | **−0.0256** | **+0.0006** |
| it2b0 | acc | 95 | −0.0086 | −0.0012 |
| it11b3 | acc | 88 | +0.0003 | −0.0037 |
| it12b3 | nm | 100 | +0.0064 | −0.0039 |
| it12b1 | nm | 98 | +0.0120 | −0.0080 |
| it0b4 | acc | 93 | −0.0036 | −0.0113 |
| it5b4 | acc | 92 | +0.0020 | −0.0206 |
| it5b1 | nm | 102 | −0.0039 | −0.0229 |
| it9b1 | acc | 90 | −0.0041 | −0.0407 |
| it9b0 | nm | 100 | +0.0137 | −0.0426 |
| it7b2 | acc | 85 | −0.0051 | **−0.0552** |

The two columns barely agree. `it8b4` is the most stable negative signal in the study
per-draw (last in 7 of 8 draws) and neutral pooled; `it7b2` is unremarkable per-draw and
the most harmful pooled set of all sixteen; `it9b0` averages +0.0137 per-draw and pools to
−0.0426. **A 12-row addition to a 112-row set is mostly fit noise; ~100 rows is a real
change.** Only the pooled column has predicted anything downstream.

This also retires §7-§9 of `REPLICATION_FINDINGS.md`: the near-miss arm's *stable* Δdev
ranking put `it9b0` 4th of 8 and `it8b4` last, and pooled those are −0.0426 and +0.0006.
A reproducible measurement is not a correct one.

## 3. Ten selection rules

Every row is `base ∪ 62 ∪ <set>`, scored on the 7 eval splits.

| # | rule | set | rows | Δdev | Δeval |
|---:|---|---|---:|---:|---:|
| 1 | dev threshold (`min_auroc_gain > 0`), 1 draw | 2 families | 24 | −0.0237 | +0.0163 |
| 2 | dev top-3, 1 draw | 3 families | 38 | +0.0174 | +0.0266 |
| 3 | sign of own eval Δ, 1 draw | 4 families | 50 | −0.0451 | +0.0022 |
| 4 | sign of own eval Δ (the harmful half) | 4 families | 57 | −0.0055 | +0.0037 |
| 5 | mean eval Δ > 0 over 7-8 draws (accepted) | 4 families | 365 | −0.0145 | +0.0026 |
| 6 | mean eval Δ > 0 over 7-8 draws (near-miss) | 5 families | 502 | −0.0518 | **−0.0168** |
| 7 | mean eval Δ > 0.01, both arms | 5 families | 486 | −0.0198 | −0.0015 |
| 8 | top-2 by mean Δdev rank, both arms | 4 families | 389 | +0.0045 | +0.0068 |
| 9 | every draw whose own union beat the floor | 5 draws | 512 | −0.0346 | −0.0043 |
| 10a | per-split champion families | 6 families | 591 | −0.0501 | **−0.0536** |
| 10b | per-split runner-up families | 5 families | 468 | −0.0649 | −0.0445 |
| 10c | per-split champion **draws** | 7 draws | 719 | −0.0179 | −0.0010 |

**The harder the selection, the worse the result.** Rule 10a is the most selective set
built anywhere in this study — the best family on each split, chosen from 120 fits — and
it is the worst result in the study. Rule 9 is the cleanest test of the principle: five
independent ~100-row draws, *each individually measured above the floor*, pooled to −0.0043.

Rule 10c is the sharpest illustration of why. Each chosen draw was the champion on one
split; in the union, five of seven splits score **below** the champion selected for them:

| split | floor | 7 win-draws | its champion, alone |
|---|---:|---:|---:|
| anthropic_harmless_refusal | 0.8867 | 0.7122 | 0.9158 (nm_rep5) |
| hc_context_drift | 0.8406 | 0.6331 | 0.8786 (acc_rep5) |
| bbq_substitution | 0.9751 | 0.9465 | 0.9689 (acc_rep2) |
| hc_contradiction | 0.7452 | 0.9365 | 0.9257 (nm_rep3) |
| mm_substitution | 0.8589 | 0.9376 | 0.9501 (nm_rep2) |
| oig_context_drift | 0.7420 | 0.7697 | 0.8212 (acc_rep1) |
| oig_omission | 0.6550 | 0.7615 | 0.8238 (nm_rep1) |
| mean | 0.8148 | 0.8139 | |

Each pick's strength on one split is another pick's collapse — `nm_rep5` wins refusal at
0.9158 and sits at 0.6790 on context-drift; `acc_rep5` wins context-drift at 0.8786 and is
the *worst* of all fifteen draws on refusal at 0.5990. Pooling them reconstructs the floor
with 719 rows.

## 4. What actually works

| set | rows | dev | Δdev | eval | Δeval |
|---|---:|---:|---:|---:|---:|
| it1b1 pooled (7 draws) | 101 | 0.86682 | +0.0358 | 0.85273 | +0.0379 |
| it1b1 + it4b3 pooled | 185 | 0.85004 | +0.0190 | 0.85543 | +0.0406 |
| **it1b1 + it4b3 + near-miss rep4 (whole draw)** | **285** | 0.82854 | −0.0025 | **0.86254** | **+0.0477** |

**0.86254 is the best result in this arm**, above the original run's 0.8617 peak. And it is
genuinely additive — the pair gives +0.0406, rep4 alone +0.0194, together +0.0477 — the
only time in this study that combining two positive sets improved on the better of them.
Per split, rep4 repairs precisely the pair's weakness: `anthropic_harmless_refusal`
0.7552 → 0.8715, while `oig_omission` rises above both parents to 0.8061.

Three additions to it have been tried and all failed: `it1b3` (−0.0240, via refusal and
mm_substitution), near-miss rep5 (−0.0248, via `hc_context_drift` 0.8022 → 0.6265), and
rep4+rep5 together (−0.0241). The set looks saturated.

Note rep4 and rep5 are the *same generator, same directions, 100 vs 98 rows,* +0.0194 vs
+0.0172 standalone — and one adds +0.0071 to the pair while the other subtracts −0.0248.

**Dev would reject the winner.** The best set's Δdev is −0.0025; the loop's acceptance rule
reads dev.

## 5. The splits fall into two groups

Per-split means over the 120 per-draw family fits, and over the 15 draw unions:

| split | floor | families above floor | best family | draws above floor | best draw |
|---|---:|---:|---|---:|---|
| anthropic_harmless_refusal | 0.8867 | **0/16** | it5b0 0.8799 | 3/15 | nm_rep5 0.9158 |
| bbq_substitution | 0.9751 | **0/16** | it7b2 0.9739 | **0/15** | acc_rep2 0.9689 |
| hc_context_drift | 0.8406 | **0/16** | it1b1 0.8288 | 2/15 | acc_rep5 0.8786 |
| hc_contradiction | 0.7452 | 13/16 | it6b1 0.8322 | 15/15 | nm_rep3 0.9257 |
| mm_substitution | 0.8589 | 12/16 | it12b3 0.9255 | 13/15 | nm_rep2 0.9501 |
| oig_context_drift | 0.7420 | 14/16 | it12b3 0.7924 | 8/15 | acc_rep1 0.8212 |
| oig_omission | 0.6550 | 16/16 | it12b1 0.7747 | 15/15 | nm_rep1 0.8238 |

**Generated data only improves splits the probe is already bad at.** The three splits with
the highest floors are improved by *no* family in 120 attempts; the four weakest are
improved by nearly everything. Every mean-level result in this study is a trade between
those two groups, which is why `it1b1` wins overall — it is 1st of 16 on `hc_context_drift`
and 2nd on `bbq_substitution`, i.e. the family that costs least on the protected splits
while still collecting the gains everyone gets.

The two arms specialise in opposite directions: near-miss families/draws win all four
improvable splits, accepted ones win two of the three unimprovable. Near-miss material
*teaches* what the probe doesn't know; accepted material *doesn't damage* what it does.

## Consequences

- **Do not select on per-batch ΔAUROC, at any threshold or rank.** Ten rules, eight at or
  below the floor. The measurements are real (corr(Δdev, Δeval) is +0.93 within a draw)
  but they do not compose: a set's value is not a function of its parts' measured values.
- **Pool a direction across draws before judging it.** Family effects are legible at ~100
  pooled rows and invisible at 12, and the per-draw and pooled columns disagree in sign for
  6 of 16 families.
- **Prefer whole unselected draws to curated mixtures.** The best set is two pooled
  families plus one arbitrary draw taken whole; every curated set of comparable or larger
  size is worse.
- **Read per-split, not the mean.** The mean hides that generated data cannot improve a
  strong split and reliably improves a weak one, and hides the split collapses (0.62, 0.54)
  that sink the curated sets.
- **The accept/reject boundary the loop drew is not meaningful.** Batches rejected at
  −0.006 produce imitations as good as batches accepted at +0.006.

## Reproducing

```bash
V=.venv_claude/bin/python
for r in 1 2 3 4 5 6 7 8; do
  $V scripts/generate_like_accepted.py --keys nearmiss \
     --out data/instructions_like_nearmiss_rep$r.jsonl
  $V scripts/fit_mixed_directions.py --generated data/instructions_like_nearmiss_rep$r.jsonl \
     --score-families --tag nearmiss_rep$r
done
$V scripts/fit_mixed_directions.py --generated data/union_it1b1.jsonl --tag u_it1b1     # §2
$V scripts/fit_mixed_directions.py --generated data/union_fixed2_plus_nmrep4.jsonl \
   --tag f2_nmrep4                                                                      # §4 winner
$V scripts/fit_mixed_directions.py --generated data/union_splitwinners.jsonl --tag s_splitwinners
$V scripts/fit_mixed_directions.py --generated data/union_splitwinreps.jsonl --tag s_winreps
```

Per-split matrices: `per_split_families.json` (129 probes × 7 splits),
`per_split_reps.json` (15 draw unions). Pooled corpora: `data/union_*.jsonl`.
Results CSVs: `nearmiss_rep*_directions_results.csv`, `p_*`, `s_*`, `t2_*`, `g_*`,
`f2_*`, `u_*`, `nm_posreps_*`, `trio_*`.
