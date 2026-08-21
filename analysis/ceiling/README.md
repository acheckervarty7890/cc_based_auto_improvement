# Ceiling analysis — instruction-following probe (gemma-3-27b L32)

Everything here runs **offline off the run's cached activations** — the 7 eval blobs,
the dev blob, the base blob and the 1,754 per-conversation red-team blobs. No LLM is
ever loaded, and a cache miss raises rather than silently forwarding a 27B model.

Report: `results/summary.md` · page: `ceiling_study.html`

## Layout

| file | what it does |
|---|---|
| `harness.py` | loaders (eval/dev/base/red-team), fit + finetune, metrics, GPU staging |
| `folds.py` | deterministic split- and class-stratified partitions |
| `sanity.py` | reproduces all 12 published probe scores from the comparison CSVs |
| `run_ceiling.py` | Part 1 — redteam_only / cv_eval / cv_eval_rt / oracle |
| `run_sweep.py` | Part 2 — the N-dev sweep, joint vs finetune (`--no-redteam` for the control) |
| `refit_check.py` | attributes the refit-vs-published gap |
| `report.py` | renders `results/summary.md` |
| `build_artifact.py` | renders `ceiling_study.html` (all figures read from `results/`) |
| `run_all.sh` | the four planned runs, sequentially (one GPU) |
| `push_progress.sh` | snapshot commit; adds only this dir + the run log |

## Headline numbers

| | macro AUROC |
|---|---|
| iteration 0 (base data only) | 0.7714 |
| best published probe (nemotron iter5) | 0.8595 |
| **ceiling** — 5-fold CV inside eval, within-fold | **0.9435** |
| trained on 436 dev rows, tested on all 1302 eval rows | 0.9613 |
| oracle (fit and scored on the same rows) | 1.0000 |

Red-teaming captured 51% of the iteration-0-to-ceiling headroom in the nemotron arm,
31% in the gpt-oss arm. 84 labelled dev rows alone beat the finished red-team probe.

## Method notes that matter

- **Metric** is tuberlens' own: per-split AUROC/accuracy/TPR, macro-averaged over the 7
  splits — the `mean` row of `*_comparison.csv`. `sanity.py` reproduces all 12 published
  probe scores to 6 dp, so anything here is directly comparable to the run.
- **Validation is fixed at every point of the sweep.** The 436 dev rows are cut once into
  a 100-row holdout (never trained on, in any condition) and a 336-row pool. A validation
  set that shrank as N grew would change how each point is selected.
- **Prefixes are stratified**, so `dev[:42]` is 6 rows from each split, half positive.
- **Noise floor 0.0146 macro AUROC**, measured from three row-orderings of identical data.
  Do not read smaller differences as effects.
- **`TPR@FPR<=1%`** is the corrected definition (best TPR at or below 1% FPR). tuberlens'
  `tpr_at_fixed_fpr_score` takes the *nearest* ROC point instead and returns 0.0 for a
  perfectly separating probe; both are recorded in the JSONL.

## Two upstream bugs found

1. `batch_size 16` × `gradient_accumulation_steps 4` ⇒ any training set under 64 rows
   takes **zero** optimizer steps. The fit runs its full epoch budget and changes nothing.
   Visible in the log as PyTorch's `lr_scheduler.step() before optimizer.step()` warning.
2. `tpr_at_fixed_fpr_score` reads the wrong ROC point (see above).

## Reproducibility finding

A retrain is **not** reproducible from its own snapshot files.
`_activate_redteam_cached` emits already-cached conversations first and appends
newly-computed ones, so the training set's row order encodes the box's cache history.
Refitting from `redteam_postprocessed_iter5.jsonl` in file order gives 0.8272; rebuilding
the cache-hit partition from blob mtimes (614 hits, 148 computed at iter5) reproduces the
published probe **bit-identically — 10/10 members, AUROC 0.812610 to 6 dp**.
