# Persistent eval errors

Branch: `experiment22_cloud`. Companion to `ceiling_analysis/` (how far the red-team-trained
probes are from what eval-distribution training could reach) and `analysis/offdist/` (which
*red-team* rows are responsible). This one turns the question around and asks about the
**eval** rows: which of them does every probe get wrong, and does that set depend on the
attacker model, the memo condition, or the experiment?

## The questions

1. Are the probes' errors concentrated on the same rows, or does each probe fail its own?
2. Does the always-wrong set change with the arm (attacker model, memo condition) or with
   the experiment?
3. Is it created by the red-team loop, or inherited from the probe every run starts from?
4. How much of it is the 0.5 threshold rather than a ranking failure?
5. Are those rows learnable at all from eval-distribution data?

## Scope

All **45 probes** the two experiments produced: experiment22's two arms
(`gptoss120b_datadesc`, `deepseekv4pro_datadesc`, `probe_iter0..5`) and experiment23's three
memo-ladder arms (`s3_control`, `s3_itermemo150`, `s3_evaldesc`, `probe_iter0..10`). Eval is
all four `eval_sets/hu_ha` splits, 866 rows, full splits. The unit of analysis is a
(probe, row) cell — 38 970 of them.

## Five decisions worth stating

**The 45 probes span two branches, and nothing is checked out.** experiment23's arms are
committed on `experiment23_cloud`; `pe_common.ensure_exp23_probes` reads their pickles out
of git into `probes_exp23/` (gitignored). A study that spans both branches cannot ask the
working tree to be on one of them, and the same function reads each run's published
comparison CSV from wherever that run lives.

**Scoring must reproduce the runs' own numbers before it may talk about rows.**
`score.py` rebuilds every published `(iteration, split)` AUROC and accuracy from its score
matrix and asserts the deviation — currently 4.5e-4 AUROC, and one accuracy cell, which is a
single row of the smallest split flipping. It runs under `PROBE_FUSED_ENSEMBLE=0` because
that is how the runs scored; fused scoring moves AUROC in the 4th decimal and flips a
handful of cells sitting within 1e-3 of 0.5. Harmless for a mean, not harmless for a claim
about *which* rows are always wrong.

**Two definitions of "misclassified", because they disagree by 5x.** `p > 0.5` is what the
comparison CSVs report accuracy under, and it gives 39 always-wrong rows. But these probes
call only 31% of eval rows positive against a 50% base rate, so 0.5 is doing much of the
work. Every eval split is exactly class balanced, which makes the per-probe, per-split
**median** score the balanced-accuracy-optimal rank cut; a row on the wrong side of *that*
is misranked rather than mis-thresholded, and only 8 rows qualify. Both are reported, and
the 8 are the ones with a semantic story.

**`probe_iter0` is excluded from every persistence intersection.** All five runs train their
first probe on the same 50 base samples under the same pinned ensemble seeds, so the five
`probe_iter0` pickles score identically to 0.0 — they are one probe. Leaving it inside an
arm's intersection makes "already wrong at iter0" true by construction. The reported
per-arm persistent sets are over the *retrained* probes only, which is why they are larger
(84–193) than the intersections that include it.

**Learnability is measured, not assumed.** "Every probe gets this row wrong" means something
different depending on whether any probe of this family *could*. Two in-distribution
controls answer it, both using `ca_common`'s fit unchanged so they land on the ceiling
study's curves: a probe trained on the dev pool alone, and 5-fold CV inside the eval set
plus that pool with every row scored out of fold. The ceiling study kept only aggregates,
so `controls.py` re-runs its fit to get the per-row scores.

## Layout

```
analysis/persistent/
  pe_common.py      run registry, git-resident probe staging, error-set definitions
  score.py          phase 1 - 45 probes x 866 rows -> results/scores.npz (+ reproduction check)
  controls.py       phase 2 - dev-only and ceiling out-of-fold per-row scores
  report.py         phase 3 - results/summary.json + results/SUMMARY.md
  build_artifact.py the same numbers as a standalone page
  probes_exp23/     staged from experiment23_cloud, gitignored
  results/          scores.npz, controls.{npz,json}, summary.json, SUMMARY.md, persistent.html
```

Run order:

```bash
PROBE_FUSED_ENSEMBLE=0 .venv_claude/bin/python analysis/persistent/score.py
.venv_claude/bin/python analysis/persistent/controls.py
.venv_claude/bin/python analysis/persistent/report.py
.venv_claude/bin/python analysis/persistent/build_artifact.py
```

Needs the ceiling analysis's `ceiling_acts/` prepared (its eval blobs are symlinks into
`results_hu_harm_gemma27b_batch_ablation/eval_activations/`). No LLM is loaded and no
activation is recomputed at any point; the whole pipeline is about two minutes, nearly all
of it the five control fits.

The written answer is in `results/SUMMARY.md`, and the same numbers as a standalone page at
https://claude.ai/code/artifact/10429880-0cd2-4bde-97da-96e5b86b5529
(regenerate with `build_artifact.py` and republish the same path to update it in place).
