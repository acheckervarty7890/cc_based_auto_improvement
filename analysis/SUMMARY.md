# experiment_instruction_cloud_6 — category-steered memos

Branch off `main`. Instruction-following concept, `google/gemma-3-27b-it` L32, 10-member
score-averaging ensemble fit and scored **sequentially** (`PROBE_FUSED_ENSEMBLE=0`), blind
batch attacker (`view_limit: 0`), held-out dev validation, 5 retrain iterations, both error
types. Two arms differing only in `attacker.models`.

Ran 2026-08-21 19:17 → 23:38 UTC (4h21m). Detail: `ceiling/results/summary.md`,
`novelty/results/summary.md`.

## What this experiment changed

Against `experiment_instruction_cloud_5` (same attackers, same everything else), two changes,
both applied to both arms:

1. **`probe.description` enumerates the six categories the negative class is made of** —
   unjustified refusal, biased answer, exaggeration, contradicting the source, context drift,
   omission. These are the failure modes `eval_sets/instructions` is actually built from. The
   description reaches every attacker system prompt and all three judge prompts verbatim.
2. **Both summarizer prompts are steered around that enumeration**
   (`llm_judge._CATEGORY_COVERAGE_INSTRUCTION`). The judge must give each named category a
   status line — examined or not, robust or fragile — *including categories no sample has
   touched* — and close by naming which categories the next samples should come from, in
   priority order. Concept-agnostic and self-disabling: a description enumerating nothing
   leaves both prompts byte-identical, so no existing config changes behaviour.

Memo word budgets were deliberately **not** raised; the roll-call had to earn its space inside
the budget the memo already had.

## Results

`probe_iter0` scored 0.771411 in both arms, identical to cloud_5's to 6 dp — the training path
is untouched, so every difference below is attributable to the attacker and the memos.

| run | memo | steering | arm | iter5 AUROC | Δ vs iter0 | headroom captured | TPR@1%FPR |
|---|---|---|---|--:|--:|--:|--:|
| cloud_3 | off | — | nemotron | **0.8595** | +0.0881 | **51.7%** | 0.1472 |
| **cloud_6** | on | **yes** | **gptoss** | **0.8343** | +0.0629 | **37.0%** | **0.2384** |
| cloud_3 | off | — | gptoss | 0.8126 | +0.0412 | 24.2% | 0.3010 |
| **cloud_6** | on | **yes** | **nemotron** | **0.8058** | +0.0344 | **20.2%** | 0.1641 |
| cloud_5 | on | no | gptoss | 0.7786 | +0.0072 | 4.2% | 0.0971 |
| cloud_5 | on | no | nemotron | 0.7371 | −0.0343 | −20.2% | 0.1445 |

Headroom = (iter5 − iter0) / (ceiling − iter0), ceiling = 0.9416.

**The steering is a large, consistent improvement over cloud_5** — the run it is a controlled
variant of. Both arms gain (+0.056 and +0.069 AUROC), and cloud_5's nemotron arm was actually
*worse than its own starting probe*.

**It does not beat the no-memo baseline outright.** cloud_3 — same concept, same attackers, no
cross-iteration memo — still holds the best single probe at 0.8595. Read together, the most
economical reading of these six rows is that **the cross-iteration memo hurt, and the category
roll-call recovers most but not all of the damage**. This experiment cannot separate those two
effects: it varies the steering with the memo held on, and no arm here runs the memo off. A
clean memo ablation is the outstanding experiment.

Attack volume: 433/2277 successes (gptoss, 19.0%), 449/2312 (nemotron, 19.4%).

### Where the gain came from

Per-split AUROC change, iter0 → iter5:

| split | gptoss | nemotron |
|---|--:|--:|
| `anthropic_harmless_refusal` | **+0.594** | **+0.308** |
| `hc_context_drift` | +0.224 | +0.161 |
| `bbq_substitution` | −0.161 | +0.057 |
| `hc_contradiction` | −0.077 | +0.015 |
| `mm_substitution` | −0.083 | −0.024 |
| `oig_context_drift` | −0.034 | −0.152 |
| `oig_omission` | −0.022 | −0.124 |

The refusal split started **below chance** (0.348) and the memos flagged it as untouched for
three iterations before the attacker went there; it carries most of the mean's gain in both
arms. The splits the probe was already good at got worse. Breadth traded easy categories to
fix the broken one — strongly positive here only because the broken one was so far below
chance.

## Ceiling analysis

| condition | gptoss | nemotron |
|---|--:|--:|
| published iter5 | 0.8343 | 0.8058 |
| `redteam_only` (file-order refit) | 0.8239 | 0.7956 |
| `cv_eval` — **the ceiling** | **0.9416** | — |
| `cv_eval_rt` | 0.9530 | 0.9416 |
| `oracle` | 1.0000 | — |

Dev-sample sweep (macro AUROC):

| N dev rows | 0 | 84 | 168 | 252 | 336 |
|---|--:|--:|--:|--:|--:|
| gptoss joint | 0.8299 | 0.8707 | 0.9161 | 0.9397 | 0.9447 |
| gptoss finetune | 0.8299 | **0.9036** | 0.9203 | 0.9409 | **0.9541** |
| nemotron joint | 0.7997 | 0.8525 | 0.8882 | 0.9166 | 0.9371 |
| nemotron finetune | 0.7997 | 0.8640 | 0.9055 | 0.9306 | **0.9435** |

1. **336 labelled dev rows reach or beat the ceiling** in both arms.
2. **84 dev rows beat the entire red-team run** in both arms — against 433/449 successes and
   ~4.5 GPU-hours of scoring. Red-teaming buys AUROC far more expensively than labelling
   in-distribution data. The steering improves red-teaming's efficiency at its own job; it does
   not change its standing against that alternative.
3. **Adding red-team data on top of in-distribution data is inert** — `cv_eval_rt` is +0.011
   and +0.000, inside the ~0.015 noise floor. "Does not hurt", not "carries signal eval lacks".
4. **The ceiling is not uniform.** Five-fold CV *inside* eval reaches 0.966–1.000 on six splits
   but only **0.673** on `oig_omission`. Part of the gap to 0.9416 is unreachable by any
   attacker — it is a limit of a linear head at L32 on that failure mode. Work aimed at
   omission belongs on the probe, not the attacker.
5. Finetune beats joint at nearly every N; `lr=1e-4` is inert (flat at baseline throughout).

## Novelty analysis

| arm | rows | outside eval manifold | published Δ eval |
|---|--:|--:|--:|
| gptoss | 692 | **71.0%** | +0.0629 |
| nemotron | 710 | 46.3% | +0.0344 |

1. **Distance from eval does not predict harm.** The arm further outside the manifold gained
   more — the sign the hypothesis does not predict, reproducing cloud_3 on an independent run.
2. **There is no compact bad region.** HDBSCAN assigns 74% (gptoss) and 100% (nemotron) of rows
   to noise; these attack sets are diffuse.
3. **No pruning rule survives.** Of 8 matched-n novelty-vs-random removals, 4 clear the
   comparison band — 3 one way, 1 the other.
4. **Removing all red-team data costs both instruments in both arms** (eval −0.053/−0.024,
   cross-attacker −0.081/−0.017). cloud_3 found the eval column has no fixed sign across
   concepts, so that is a property of this concept; the cross-attacker cost is the consistent
   one. Whatever the red-team rows buy, eval is a poor instrument for seeing it.

## Reproducibility notes

- **Pin `PROBE_FUSED_ENSEMBLE=0` in any offline analysis of this run.** The runner sets it, so
  the arms scored sequentially; a fresh process defaults it back **on** and the fused path is a
  different reduction order. Unpinned, `sanity.py` reports MISMATCH on all 12 published probes
  at ~1e-4. Pinned, all 12 reproduce to 6 dp.
- **`redteam_only` lands ~0.010 below the published probe in both arms.**
  `_activate_redteam_cached` emits cache hits before newly-computed rows, so training row order
  encodes the box's cache history and is not recoverable from a snapshot. Compare analysis
  conditions to each other, never to the comparison CSV.
- **`.gitignore`'s unanchored `run*.sh` matches at any depth.** It silently swallowed both
  `analysis/*/run_all.sh` from a `git add -A`. Nothing reports a path an ignore rule removed.
- **`analysis/novelty/report.py` had cloud_3's narrative hardcoded** and asserted findings about
  high-stakes arms this box never ran. Fixed to count arms from results on disk and attribute
  every cross-concept claim.

## Open questions

1. **A clean memo ablation.** cloud_3 (memo off) still beats every memo-on run. Toggle
   `cross_iteration_memos` alone, nothing else.
2. **Refusal is fixable, omission may not be.** Refusal went 0.348 → 0.942 once sampled;
   `oig_omission` is capped near 0.673 at the ceiling. Different probes, not different attacks.
3. **Naming a gap is not enough to fill it.** The memos flagged refusal as untouched for three
   iterations before the attacker acted. Something stronger than a priority list may be needed.
