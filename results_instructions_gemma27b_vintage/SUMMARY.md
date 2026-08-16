# Red-team vintage sweep — instruction-following concept
_Updated 2026-08-15T18:21:11+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data (`data/instructions_llama70b_50.jsonl`) plus one *vintage* of iteration-3 red-team pairs, scored on the seven `eval_instructions` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Setup.** Probe `google/gemma-3-27b-it` L32, labels `assistant_follows_the_instruction` / `assistant_does_not_follow_the_instruction`. Arms are the two attackers of `run_gemma27b_instructions_attackers.sh`: `gptoss120b` (`openai/gpt-oss-120b`) and `nemotron` (`nvidia/nemotron-3-ultra-550b-a55b`). All activations — base split, per-conversation red-team, and the seven eval splits — were pulled from Kaggle (`anku7890/instructions-gemma27b-*`, `anku7890/*-gemmaevalpt`); no gemma-3-27b forward pass runs here.

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**The over-1024-token filter.** `get_activations` pads *or truncates* every conversation to 1024 tokens, so a longer one is trained on from its opening alone. These runs predate `token_budget.py`, so nothing length-guarded the contrastive generator. Rows whose conversation exceeds the cap — **and** rows whose cached activation is stored truncated, which `get_activations` also produces for a short conversation that merely shared an extraction batch with an over-long one — are dropped together with their pair partner, keeping every vintage exactly 50/50. Per-arm counts are in `*_vintage.json` under `report.long_filter`.

**Read the sd, not just the mean.** These are independent `ProbeFactory` fits with independent initialisations, and seed alone moves some splits by more than the between-vintage gaps. A single-seed comparison of two vintages means nothing; quantifying that is what this sweep exists for.

## Progress: 80 fits recorded

- **gptoss120b**: v0×10, v1×10, v2×10, v3×10
- **nemotron**: v0×10, v1×10, v2×10, v3×10

## gptoss120b — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.4549 ± 0.2438 | 0.4951 ± 0.0629 | 0.4972 ± 0.0273 | 0.4962 ± 0.0317 | 0.5342 ± 0.1264 | 0.5226 ± 0.0629 | 0.4802 ± 0.0321 | 0.4972 ± 0.0349 |
| v1 | 434 | 10 | 0.4444 ± 0.0655 | 0.9097 ± 0.0203 | 0.7356 ± 0.0591 | 0.9297 ± 0.0141 | 0.8823 ± 0.0407 | 0.6749 ± 0.0360 | 0.7191 ± 0.0178 | 0.7565 ± 0.0186 |
| v2 | 674 | 10 | 0.6802 ± 0.1063 | 0.8930 ± 0.0230 | 0.7384 ± 0.1049 | 0.8998 ± 0.0221 | 0.8794 ± 0.0415 | 0.5852 ± 0.0319 | 0.7189 ± 0.0505 | 0.7707 ± 0.0233 |
| v3 | 858 | 10 | 0.8298 ± 0.0521 | 0.8908 ± 0.0300 | 0.7564 ± 0.0548 | 0.8980 ± 0.0193 | 0.8581 ± 0.0284 | 0.5872 ± 0.0260 | 0.7748 ± 0.0190 | 0.7993 ± 0.0228 |

## nemotron — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.4549 ± 0.2438 | 0.4951 ± 0.0629 | 0.4972 ± 0.0273 | 0.4962 ± 0.0317 | 0.5342 ± 0.1264 | 0.5226 ± 0.0629 | 0.4802 ± 0.0321 | 0.4972 ± 0.0349 |
| v1 | 458 | 10 | 0.6152 ± 0.1061 | 0.9670 ± 0.0048 | 0.8452 ± 0.0303 | 0.8692 ± 0.0120 | 0.9588 ± 0.0107 | 0.7379 ± 0.0195 | 0.6869 ± 0.0282 | 0.8115 ± 0.0166 |
| v2 | 630 | 10 | 0.8113 ± 0.0539 | 0.9542 ± 0.0098 | 0.8776 ± 0.0214 | 0.7661 ± 0.0498 | 0.8832 ± 0.0739 | 0.6985 ± 0.0283 | 0.6269 ± 0.0610 | 0.8025 ± 0.0346 |
| v3 | 926 | 10 | 0.7345 ± 0.0583 | 0.9174 ± 0.0238 | 0.7690 ± 0.0319 | 0.8416 ± 0.0532 | 0.9266 ± 0.0260 | 0.6750 ± 0.0297 | 0.6340 ± 0.0519 | 0.7854 ± 0.0222 |

## Findings

**Iteration 1's red-team data buys nearly all of the gain.** Both arms jump from chance to
~0.76-0.81 mean AUROC on iteration-1 pairs alone, and neither improves reliably after that:

| arm | v0 -> v1 | v1 -> v2 | v2 -> v3 | v1 -> v3 |
|---|---|---|---|---|
| gptoss120b | **+0.259** | +0.014 | +0.029 | +0.043 |
| nemotron | **+0.314** | -0.009 | -0.017 | **-0.026** |

Against a seed sd of ~0.017-0.035 on the mean, only gptoss120b's v1->v3 gap is even marginally
outside noise; every single-iteration step is inside it, and nemotron's later iterations are
flat to slightly negative — v3 trains on **twice** v1's data and scores about one sd *lower*.

**The aggregate hides a trade, not a plateau.** Later iterations are not adding nothing; they
are moving accuracy between splits. `anthropic_harmless_refusal` improves monotonically and
carries essentially all of the v1->v3 mean gain (gptoss 0.444 -> 0.680 -> 0.830; nemotron
0.615 -> 0.811 -> 0.735), while `oig_context_drift` *drops* at v2 and never recovers (gptoss
0.675 -> 0.585 -> 0.587; nemotron 0.738 -> 0.699 -> 0.675) and nemotron's `oig_omission` falls
0.687 -> 0.634. So iterations 2-3 chiefly reallocate the probe's capacity across failure modes.

**This is why the sweep exists: the committed single-seed CSVs are not readable as an ordering.**
`nemotron_comparison.csv` reports iter1/2/3 as 0.812 / 0.834 / 0.764, i.e. a peak at iteration 2.
Here v2 = 0.803 +/- 0.035 and v1 = 0.812 +/- 0.017 — that peak is inside seed noise and the
apparent iteration-2 advantage is an artifact of a single draw. Note also v0 on
`anthropic_harmless_refusal` has sd **0.244**: an untrained probe on that split is a coin flip
whose direction the seed picks, so any single-seed baseline there is meaningless.

**Attacker comparison.** nemotron's v1 beats gptoss120b's v1 by 0.055 (0.812 vs 0.757, ~3 sd) —
its first-iteration data is genuinely more useful. By v3 the arms converge (0.785 vs 0.799),
because nemotron gives the gain back over iterations 2-3 while gptoss120b slowly adds to it.

**v0 is identical across arms by construction** (base training data only, no red-team rows).
The two arms' v0 rows matching to the last digit is a consistency check on the two processes,
not a result.

## Held-out attack success: the v2 probes against the rows only v3 has

**What this measures.** Take the ten v2 probes above and put in front of them the pairs
that first appear at iteration 3 — rows held out of every v2 fit, on both sides of the
split. A misclassification is exactly the find the red-teamer was hunting, so per row the
**success rate is the fraction of the ten independently-initialised v2 probes that get it
wrong**. That separates a sample which beats *the architecture on this data* (10/10) from
one which beats *a single draw* (1/10) — the distinction a single-seed red-team run cannot
make, since it reports both as a find.

Thresholding follows `ProbeJudge.evaluate`: positive at `predict_proba >= 0.5`, i.e.
`logit >= 0` (`probe.threshold`'s default, which both arm configs use). The v2 probes are
**re-fit, not reloaded** — the sweep kept AUROC and dropped the probes — and every refit
asserts it reproduces the sweep's probe by re-scoring `oig_omission` and demanding the
recorded AUROC to the last bit. All 20 did.

| arm | held out of v2 | pairs | per-seed success rate |
|---|---|---|---|
| gptoss120b | 184 rows of 858 | 92 | 0.273 +/- 0.021 (min 0.239, max 0.310) |
| nemotron | 296 rows of 926 | 148 | 0.401 +/- 0.011 (min 0.385, max 0.416) |

### By row type — mean over the ten seeds

| arm | group | n | success | always (10/10) | never (0/10) |
|---|---|---|---|---|---|
| gptoss120b | held-out, all | 184 | 0.273 | 30 | 103 |
| gptoss120b | — attacker successes | 92 | **0.477** | 29 | 30 |
| gptoss120b | — generated counterparts | 92 | 0.070 | 1 | 73 |
| gptoss120b | — label `does_not_follow` | 92 | 0.420 | 25 | 36 |
| gptoss120b | — label `follows` | 92 | 0.127 | 5 | 67 |
| gptoss120b | v2's own rows, fit on | 562 | 0.003 | 0 | 553 |
| gptoss120b | v2's own rows, val side | 112 | 0.353 | 18 | 48 |
| nemotron | held-out, all | 296 | 0.401 | 77 | 132 |
| nemotron | — attacker successes | 148 | **0.707** | 74 | 21 |
| nemotron | — generated counterparts | 148 | 0.095 | 3 | 111 |
| nemotron | — label `does_not_follow` | 148 | 0.389 | 36 | 66 |
| nemotron | — label `follows` | 148 | 0.413 | 41 | 66 |
| nemotron | v2's own rows, fit on | 518 | 0.006 | 0 | 512 |
| nemotron | v2's own rows, val side | 112 | 0.265 | 14 | 61 |

### Distribution of the per-row success count

```
                gptoss120b (184)        nemotron (296)
 0/10 seeds        103                     132
 1/10                15                      15
 2/10                 6                      12
 3/10                 6                       9
 4/10                 4                       6
 5/10                 4                      10
 6/10                 5                       9
 7/10                 2                       3
 8/10                 3                       5
 9/10                 6                      18
10/10                30                      77
```

### Findings

**About half of iteration 3's genuinely-new finds were not probe-draw artifacts.** Of the
attacker's own success conversations that v2 never saw, **45 of 92 (gptoss120b) and 109 of
148 (nemotron) fool at least 5 of the 10 v2 probes**, and 29 / 74 fool all ten. So
iteration 3 was largely re-finding weaknesses the iteration-2 probe *really* has, not
exploiting the particular fit it happened to be handed.

**The rest is a seed lottery, and it is large.** 30 of 92 (gptoss) and 21 of 148
(nemotron) attacker successes fool **none** of the ten v2 probes. Those are conversations
that the probe of the day misclassified and a rerun of the same training with a different
initialisation would have classified correctly — reported as finds, trained against, and
not reproducible. Between the extremes the middle is thin (51 and 87 rows at 1-9 of 10),
so per-sample this is close to bimodal: a find is usually either robust or a coin flip,
rarely in between.

**Generated counterparts are nearly free.** The LLM-written opposite-class halves fail at
0.070 / 0.095 against probes that never saw them — 5-7x lower than the successes they were
written from. Only 1 of 92 and 3 of 148 fool all ten seeds; in just 1 and 6 pairs
respectively do *both* members beat a majority of seeds. They balance the labels, but as
held-out evidence about the probe they carry little.

**Memorisation is total, so the held-out rate is a real generalisation gap.** v2's own
*fitted* rows fail at 0.003 / 0.006 — the probe gets essentially every training row right.
Its *validation* rows (never fit; they act only through early stopping) fail at 0.353 /
0.265, which brackets the held-out numbers. Read against that reference, gptoss120b's
v3-only rows are no harder than v2's own unfitted rows (0.273 vs 0.353), whereas
nemotron's are harder (0.401 vs 0.265): its iteration-3 attacker found something its
iteration-2 data genuinely did not cover.

**And that is the tension with the AUROC table above.** nemotron's iteration-3 finds
transfer to the v2 probes far better than gptoss120b's (0.707 vs 0.477) — they are, by
this measure, the better attack — yet training on them *lowered* mean eval AUROC
(v3 0.785 vs v2 0.803). Beating the probe and improving the probe are not the same thing:
a batch of samples can be reliable probe failures and still not generalise to the eval
splits, either because they concentrate in a region those splits do not sample or because
they resemble each other more than they resemble anything else. Which of the two it is,
this measurement does not settle.

**Per-arm asymmetry.** gptoss120b's held-out failures are lopsided — 0.420 on rows labelled
`assistant_does_not_follow_the_instruction` against 0.127 on `follows`, i.e. its v2 probes
mostly miss violations rather than over-flag compliance. nemotron's are symmetric
(0.389 / 0.413).

**One caveat on "new at iteration 3":** it is a property of the training set, not of when
the attacker found the conversation. `filter_dataset` refits each cycle and drops a
different top-percentile, so a pair can be found early, dropped from the iteration-2 set,
and taken back at iteration 3 — 22 of gptoss120b's 92 pairs and 19 of nemotron's 148 came
from the first rotation. They are held out of every v2 fit either way, which is what the
number measures.

Per-row detail is in `holdout_success_rows.csv` (one row per held-out sample: label, pair
role, split side, success count, mean/min/max logit) and the group table in
`holdout_success_summary.csv`; `*_holdout_membership.json` carries the membership and row
provenance. The conversations themselves are in
`viewers/instructions_v3_only_pairs_viewer.html` (only these pairs, each annotated with
how many of the ten v2 probes it fools), and this section is also rendered standalone as
`viewers/instructions_v2_holdout_report.html`. Both are built by
`scripts/build_holdout_viewer.py`.

## Reproducing

```bash
# activations (all from Kaggle; no gemma-3-27b forward pass is needed anywhere)
KAGGLE_CONFIG_DIR=<dir with kaggle.json> \
  .venv_claude/bin/python scripts/publish_kaggle_redteam_activations.py restore \
  --experiment instructions --iterations 3 --base --no-per-file
KAGGLE_CONFIG_DIR=<dir with kaggle.json> \
  .venv_claude/bin/python scripts/attribution_fetch_eval.py

# the sweep (both arms in parallel; AGENTIC_FAST_ACTS is a bit-identical 3.5-4.5x speedup)
AGENTIC_FAST_ACTS=1 .venv_claude/bin/python scripts/attribution_vintage.py \
  --arm gptoss120b --seeds 10 --drop-long pair
AGENTIC_FAST_ACTS=1 .venv_claude/bin/python scripts/attribution_vintage.py \
  --arm nemotron --seeds 10 --drop-long pair
.venv_claude/bin/python scripts/attribution_vintage.py --summarize-only   # merge both arms

# held-out attack success of the v2 probes (same two-process pattern)
AGENTIC_FAST_ACTS=1 .venv_claude/bin/python scripts/vintage_holdout_success.py \
  --arm gptoss120b --seeds 10
AGENTIC_FAST_ACTS=1 .venv_claude/bin/python scripts/vintage_holdout_success.py \
  --arm nemotron --seeds 10
.venv_claude/bin/python scripts/vintage_holdout_success.py --summarize-only

# the HTML report + the viewer holding only the v3-only pairs
.venv_claude/bin/python scripts/build_holdout_viewer.py
```
