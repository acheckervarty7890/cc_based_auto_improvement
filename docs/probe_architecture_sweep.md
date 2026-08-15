# Does a different probe architecture reach the eval rows red-teaming cannot?

*Run: 2026-08-15. Code: `scripts/arch_sweep.py`, `scripts/nonlinear_ceiling.py`,
`src/agentic_redteam/probe_architectures.py`, checks in
`scripts/test_probe_architectures.py`. Raw output:
`results_hu_harm_gemma27b_batch_ablation/arch_sweep/` (progress sidecar `arch_progress.jsonl`,
`arch_sweep.json`, `nonlinear_ceiling.json`), logs `logs/arch_sweep.log`,
`logs/nonlinear_ceiling.log`.*

> **Status: §1 complete; §2 complete at one seed on both arms; §§3-6 pending more seeds.**
> §1 (ceiling and transfer, both arms) is finished and settled. §2 and §2b hold a full
> 8-architecture comparison on **both** attacker arms at seed 42 — enough for the ranking
> and the pooling result, which already replicate across arms, but **not** enough to
> separate the `eval_ant_hh` gain from a generalisation trade-off. That needs the residual
> test in §4, which requires at least two seeds. Everything is single-seed until this
> banner says otherwise: per-split seed noise in the comparable heldout run was sd
> 0.014-0.059.
>
> **To check on it or pick it up:**
> ```bash
> .venv_claude/bin/python scripts/arch_sweep.py --summarize-only   # read partial results
> tail -f logs/arch_sweep.log                                      # watch fits land
> nohup bash run_arch_sweep.sh 42 43 44 45 46 >> logs/arch_sweep.log 2>&1 &   # (re)start
> nohup bash failsafe_commit_arch.sh > logs/failsafe_arch.out 2>&1 &          # 30-min pushes
> ```
> Run `--summarize-only` **only while the sweep is idle** — see the warning in
> `run_arch_sweep.sh`. Everything resumes at
> `(arm, architecture, seed, variant, legacy)` granularity, so a restart costs at most the
> fit that was in flight.
>
> **Timing.** Fits vary by more than 10x. On deepseekv4pro the batch-16 heads take ~20 min
> and the two attention heads (batch 128) ~50; on gptoss120b every head lands in ~5 min,
> because that arm's validation split is perfectly separable and early stopping fires at
> epoch 3-5 (see §2). Budget ~3 h per seed for both arms, so ~15 h for the five.

## The question

`docs/heldout_v3_vs_v2_overlap.md` ends on a specific negative result. Thirty-one
`eval_dataset_hu_ha` rows are misclassified by **all eight** probe families it built — two
attacker arms x two disjoint red-team vintages x with and without the base training data,
five seeds each — a 19x enrichment over chance, with 21 of the 31 in `eval_ant_hh`. Its
conclusion:

> The residue is addressable only off this axis. […] Whatever fixes those is not more of
> the same red-teaming, and — per Result 5 — it is not the base training data either. It
> is a different eval concept boundary or a different probe architecture.

This run is the **probe architecture** half of that. The eval concept boundary is a
separate question and is not touched here.

## What was already known, and what it predicts

Two prior measurements constrain what to expect, and they point in opposite directions:

1. **Pooling has visible headroom.** `why_last_iteration_adds_nothing.md` notes in
   passing that the same layer-32 activations read out with *mean* pooling score 0.64 on
   `eval_ai_dilemmas` where the deployed *softmax* pooling reaches 0.998. How per-token
   evidence is aggregated is doing a large amount of work.
2. **Feature-wise capacity probably does not.** That doc's §2 fits a linear probe on each
   eval split *itself* (5-fold CV) to measure what a linear readout of these activations
   can do: **0.96 AUROC on `eval_ant_hh`** against 0.77 achieved. The signal is already
   linearly extractable; what fails is transfer from the training distribution, not the
   expressiveness of the head.

So the prediction going in is that **pooling changes should matter and MLP readouts
should not** — and the design below is built so that prediction can fail visibly rather
than being assumed.

A third constraint bounds how much capacity is safe to add: both arms already fit their
training sets perfectly (train accuracy 1.0000, ~600-900 rows in 5376 dimensions —
`attribution_findings.md`), so any added capacity buys variance unless it is small and
regularised.

## Design

### The architectures

Eight, crossing two axes so that reading down a column isolates pooling and reading
across a row isolates non-linearity. All are trained through one code path
(`probe_architectures.build_probe`), so no arm differs from another by anything but its
architecture.

| architecture | pooling | readout | notes |
|---|---|---|---|
| `linear_then_softmax` | weights each token by its **own** logit | linear | the deployed head — the baseline |
| `attention` | separate attention query | linear | `AttnLite`; decouples what is attended from what is scored |
| `pre_mean` | mean | linear | fixed, label-independent pooling |
| `difference_of_means` | mean | linear, closed form | mass-mean probing |
| `lda_shrinkage` | mean | linear, closed form, whitened | **new** — see below |
| `mlp_then_softmax` | weights each token by its own logit | MLP | **new** — deployed pooling, non-linear readout |
| `mean_then_mlp` | mean | MLP | **new** — non-linearity with pooling held fixed |
| `attention_then_mlp` | separate attention query | MLP | **new** — both axes at once |

The four marked **new** are implemented in `src/agentic_redteam/probe_architectures.py`
rather than in tuberlens, because `ProbeType` is a closed `str, Enum` that
`ProbeFactory.build` dispatches on with a `match` — a new architecture cannot be
registered from outside it. Forking tuberlens was the alternative, but its checkout lives
under `.venv_claude/src/` and is not committed with this repo, so the fork would not have
survived a fresh box.

The MLP heads are `Linear(5376 -> 64) -> GELU -> Linear(64 -> 1)` and **inherit their
optimizer settings unchanged** from their linear counterpart, so the head-to-head changes
the architecture and nothing else. `--sensitivity` then varies `hidden_dim` and
`weight_decay` separately, to check that any MLP result is not an artefact of that choice.

#### Which comparisons are clean, and which are confounded

tuberlens ships **different default optimizer settings per architecture**, so "same
trainer, different head" is not automatic. The actual defaults:

| architecture | batch_size | grad_accum | final_lr |
|---|---|---|---|
| `linear_then_softmax`, `mlp_then_softmax`, `pre_mean`, `mean_then_mlp` | 16 | 4 | 1e-4 |
| `attention`, `attention_then_mlp` | **128** | **1** | **5e-4** |

Two consequences, stated plainly because they bound what the table below can support:

- **Clean.** Any comparison *within* the batch-16 group is a pure architecture contrast —
  which covers both readout contrasts (`linear_then_softmax` ↔ `mlp_then_softmax`,
  `pre_mean` ↔ `mean_then_mlp`) **and**, importantly, the main pooling contrast
  (`linear_then_softmax` ↔ `pre_mean`: logit-weighted softmax vs. mean, identical
  settings). `attention` ↔ `attention_then_mlp` is likewise clean against each other.
- **Confounded.** Any comparison *across* the two groups — i.e. anything involving
  `attention` or `attention_then_mlp` against the other four — varies the optimizer
  settings as well as the architecture, and a difference cannot be attributed to pooling
  alone.

The confound is therefore contained: it touches 2 of the 8 architectures, and the
pooling question is still answerable cleanly through `linear_then_softmax` ↔ `pre_mean`.
`--normalize-hyperparams` re-runs the two attention heads under the batch-16 settings so
the decoupled-query pooling can also be compared cleanly; those fits are recorded as a
separate variant and reported in their own section rather than pooled into the main table.

### `lda_shrinkage` exists because tuberlens' `lda` is not LDA

`PytorchDifferenceOfMeansClassifier` declares a `use_lda` field and **never reads it**.
`ProbeType.lda` and `ProbeType.difference_of_means` therefore train the identical
mass-mean probe and return bit-identical logits (confirmed in
`scripts/test_probe_architectures.py`, which asserts the defect so a future fix surfaces
as a test failure rather than as a silently duplicated column).

Since the point of including a mass-mean estimator was to contrast it with a *whitened*
one — Marks & Tegmark (2023) report that the unwhitened difference of means transfers
better across distributions than whitened or logistic directions, and distribution shift
is exactly this pipeline's failure mode — a real LDA had to be written:
`w = S^-1 (mu_pos - mu_neg)` with Ledoit-Wolf shrinkage toward `(tr(S)/d) I`. Shrinkage is
not a refinement here but a requirement: the pooled covariance is 5376x5376 estimated
from ~750 samples, so the unshrunk estimate is singular and plain LDA is undefined.

### One fix applied to every arm: early stopping now restores

`docs/attribution_findings.md` §1 documents that `PytorchAdamClassifier` snapshots its
best epoch as `self.model.state_dict().copy()` — a shallow copy over live parameter
tensors, so the snapshot tracks training and the closing `load_state_dict` restores the
final epoch onto itself. Early stopping selects an epoch and then cannot return to it.

That defect had to be fixed before this comparison, not after: early stopping is the main
overfitting control for the higher-capacity heads, so leaving it broken would have
handicapped precisely the architectures under test. `build_probe` clones at the
`state_dict` boundary, which fixes it without touching tuberlens, and the fix is applied
to **every** Adam-trained arm so it cannot explain any difference *between* architectures.

Two consequences to keep in view:

- These AUROCs are **not comparable** to the committed comparison CSVs, which were
  produced with the defect in place.
- `--legacy-best-epoch` re-runs arms the old way so the size of that gap is measured
  rather than guessed. Its rows carry a distinct resume key, so the control cannot
  silently collapse into the main sweep.

### Protocol

- **Nothing is held out.** Every fit trains on the base training data plus the whole
  iteration-3 red-team dump (gptoss120b 778 red-team rows, deepseekv4pro 878), so within
  an arm the architecture is the only thing that varies. Contrast
  `heldout_v3_vs_v2_eval_tags.py`, where the training set was the variable.
- **5 seeds (42-46) x 2 attacker arms x 8 architectures = 80 fits**, all off activations
  already on disk — no gemma-3-27b forward pass is ever run.
- **Two decision rules.** `raw` is `logit >= 0`, what a deployed probe does. `balanced`
  predicts the top half of each split by logit; every hu_ha split is exactly 50/50, so the
  split's own median is the operating point matching the true prevalence. The second rule
  is necessary because these probes are badly off-centre — one heldout fit ranked
  `eval_ai_dilemmas` at AUROC 0.987 while scoring 0.581 accuracy — and architectures
  differ in how well-centred they leave the logits, so the raw rule alone would rank them
  partly by calibration.
- **Hard-core recovery is reported alongside mean AUROC, but it is a screen, not the
  primary readout** — a qualification the transfer results forced, and worth stating
  before §3 is read. The intent was sound: Result 3 of the heldout write-up showed that
  swapping a whole red-team vintage moves the error set about as much as a reseed does, so
  an architecture that only moves the mean has done what a reseed does, while one clearing
  core rows has reached something no vintage did. But the metric has two limits measured
  in §1. It is **noisy** — 31 rows, so a chance-level ranking scores 15.5 ± 2.8 and only a
  family clearing most of them is distinguishable. And it is **selection-biased**: the core
  was defined by linear heads trained on this same data, so any architecture in that family
  is expected to fail those rows regardless of merit. Concretely, in the table below the
  core column is close to uninformative for `linear_then_softmax`, `mlp_then_softmax`,
  `pre_mean` and `mean_then_mlp`, and carries what signal there is for
  `difference_of_means`, `lda_shrinkage` and the two attention-pooled heads. Read mean
  AUROC and the per-split table for everything else.

The core is joined by `(split, idx_in_split)`, not by content hash. The four hu_ha eval
splits hold 866 rows but only **825 distinct conversations** — 41 rows repeat, nearly all
across splits — so a content join pulls in each duplicate's twin and inflates the 31-row
core to 40. Positional joining reproduces the published 31 (balanced) and 71 (raw)
exactly.

## The falsification test, run first

`scripts/nonlinear_ceiling.py` re-runs §2's per-split ceiling measurement — 5-fold CV
fitting on each eval split itself, mean-pooled activations, identical folds and grid
selection — with an **RBF-SVM** and **gradient-boosted trees** alongside the linear
baseline, so the linear column reproduces §2 and the non-linear columns are read on the
same scale.

This bounds the whole Tier-1 half of the sweep before it is interpreted. If the
non-linear ceiling does not clear the linear one, then an MLP head that fails to help is
**confirming that bound**, not under-tuned, and no amount of hyperparameter search will
change it. If it does clear it, capacity is worth pursuing and the sensitivity grid is
where to look.

It also measures **transfer** separately — fit on the real base + red-team training set,
score the eval splits — because a model family can raise the in-domain ceiling and still
transfer worse, which under distribution shift is the common case rather than the
exception. Model selection there uses the pipeline's own validation side, never the eval
splits.

### A caveat this run discovered: the in-domain ceiling is only valid for models that cannot memorise

The gradient-boosted trees scored **0.148 AUROC** on `eval_ai_dilemmas` — not merely bad,
but far *below* chance, i.e. systematically inverted. That is a property of the eval data,
not of the model, and it invalidates the ungrouped ceiling protocol for any
high-capacity family.

Three of the four hu_ha splits are **prompt-paired**: the same user prompt appears twice,
once with a harmful assistant reply and once with a non-harmful one, so the label is
carried *entirely* by the assistant turn. Measured over the splits as loaded:

| split | rows | prompt-groups | rows in mixed-label groups | hard-core rows |
|---|---|---|---|---|
| `eval_ai_dilemmas` | 136 | 68, all pairs | **100%** | 0 |
| `eval_ant_hh` | 134 | 133 (132 singletons + 1 pair) | **0%** | 21 |
| `eval_balanced_refusal` | 400 | 179 (158 pairs + 21 quads) | **100%** | 7 |
| `eval_daily_dilemmas` | 196 | 98, all pairs | **100%** | 3 |

Under ungrouped 5-fold CV roughly 80% of a row's partners sit in the training fold. Since
the user prompt dominates a *mean-pooled* activation, a model with enough capacity to key
on the prompt predicts the partner's label — which is the opposite one — and is wrong
systematically rather than randomly. Hence far below 0.5.

Measured directly, keeping pairs in the same fold moves the tree score on
`eval_ai_dilemmas` from **0.1481 to 0.7777** — a +0.63 AUROC swing from nothing but the
fold assignment. Orientation was ruled out separately: the identical call path scores
1.0000 on synthetic separable data, so this is the data, not a sign error.

This does **not** invalidate the linear column: L2-regularised logistic regression cannot
memorise a prompt sharply enough for the effect to bite, and the linear numbers here
reproduce `why_last_iteration_adds_nothing.md` §2 **exactly** on every split measured
(0.7997, 0.9599, 0.9977). It does mean the tree column has to be read as a diagnostic of
pair-memorisation rather than as a ceiling, and that any future ceiling measurement over
these splits should group by prompt (`GroupKFold` on the user-turn hash) before trusting a
high-capacity model — `--group-by-prompt` does this.

Note also what the pairing implies about the concept itself, independently of any CV: on
those three splits the user prompt carries **no** label information by construction, so a
probe reading the prompt rather than the assistant's reply scores exactly chance there.
That is a property worth keeping in mind when reading any per-split number for an
assistant-centric concept.

`eval_ant_hh` — the one **unpaired** split, and the one holding 21 of the 31 hard-core
rows — is unaffected by this, which is convenient: it is the split the whole question
turns on.

## Results

*Partial — the sweep is still running; see the status banner. The ceiling section below
is complete for the splits listed and its headline finding is settled.*

### 1. The non-linear ceiling — no headroom where the hard core is

5-fold CV fitting on each eval split itself, mean-pooled layer-32 activations, best over
each family's grid. The linear column is the protocol `why_last_iteration_adds_nothing.md`
§2 used, and reproduces it **exactly**.

| split | linear | RBF-SVM | trees | best non-linear − linear | hard-core rows |
|---|---|---|---|---|---|
| `eval_ai_dilemmas` | 0.7997 | **0.8588** | 0.1481 † | **+0.0591** | 0 |
| `eval_ant_hh` | **0.9599** | 0.9595 | 0.9035 | **−0.0004** | **21** |
| `eval_balanced_refusal` | **0.9977** | 0.9969 | 0.9919 | **−0.0008** | 7 |
| `eval_daily_dilemmas` | 0.9050 | **0.9237** | 0.7188 † | **+0.0187** | 3 |

**Every linear value reproduces `why_last_iteration_adds_nothing.md` §2 exactly** — 0.7997,
0.9599, 0.9977, 0.9050, all four to four decimals. The non-linear columns are therefore
measured on the same instrument, not a re-derived one.

† depressed or inverted by prompt-pairing, not a ceiling — see the caveat above.

The tree column turns out to be a clean readout of the pairing mechanism rather than
noise. Ordered by how strong the split's real signal is, the damage is **monotone**:

| split | paired | linear ceiling | trees | trees − linear |
|---|---|---|---|---|
| `eval_ai_dilemmas` | 100% | 0.7997 (weakest) | 0.1481 | **−0.652** (inverted) |
| `eval_daily_dilemmas` | 100% | 0.9050 | 0.7188 | −0.186 |
| `eval_balanced_refusal` | 100% | 0.9977 (strongest) | 0.9919 | −0.006 |
| `eval_ant_hh` | **0%** | 0.9599 | 0.9035 | −0.056 (no pairing to exploit) |

So memorising the prompt wins exactly to the extent that learning the real signal is
hard. `eval_ai_dilemmas` has the weakest signal and inverts outright;
`eval_balanced_refusal` is nearly perfectly separable and is barely touched despite being
just as paired. `eval_ant_hh`, the unpaired split, sits outside the ordering entirely —
its trees are simply worse than its linear fit, which is the control that shows the
effect is pairing and not capacity.

**The headline.** On `eval_ant_hh` — the split holding 21 of the 31 hard-core rows, and the
one split whose CV is not compromised by pairing — the best non-linear ceiling is
**0.0004 below** the linear one. There is nothing for a higher-capacity readout to
extract there that a linear one cannot.

The two splits that between them hold **28 of the 31** core rows (`eval_ant_hh` 21,
`eval_balanced_refusal` 7) show **−0.0004** and **−0.0008**: no headroom at all. The two
that show headroom, `eval_ai_dilemmas` (+0.059) and `eval_daily_dilemmas` (+0.019), hold
**0 and 3** rows between them.

With only four splits this is a suggestive ordering rather than a fitted relationship, and
it is worth stating the load-bearing part on its own so it does not rest on the pattern:
**on the split carrying two thirds of the hard core, the non-linear readout ceiling is
zero.** The wider anticorrelation is consistent with that and adds nothing to it.

This is the falsification test the sweep was bounded by, and it did not falsify §2's
reading. The prediction it licenses: **the MLP heads should not recover hard-core rows.**
If they don't, they are confirming a measured bound rather than being under-tuned — and
no amount of hyperparameter search reaches those rows.

#### And transfer agrees: a non-linear readout buys nothing in practice

The ceiling is an upper bound measured in-domain. The transfer half asks the question a
deployed probe actually faces — fit on the real base + red-team training set, score the
eval splits — with model selection on the pipeline's own validation side, never on eval.
deepseekv4pro:

| arm | family | val AUROC | eval mean AUROC | hard-core rows recovered |
|---|---|---|---|---|
| deepseekv4pro | linear | 0.9942 | 0.7694 | 11 / 31 |
| deepseekv4pro | RBF-SVM | 0.9942 | **0.7698** | 12 / 31 |
| deepseekv4pro | trees | 0.9896 | 0.6993 | 17 / 31 |
| gptoss120b | linear | 1.0000 | 0.7744 | 9 / 31 |
| gptoss120b | RBF-SVM | 1.0000 | **0.7762** | 9 / 31 |
| gptoss120b | trees | 0.9995 | 0.7012 | 9 / 31 |
| *(random ranking, for reference)* | | | | *15.5 ± 2.8* |

**Both arms replicate.** RBF-SVM over linear: +0.0004 and +0.0018. Trees: −0.070 and
−0.073. The direction, the magnitude and the ordering are the same on two arms whose
red-team training data was written by different attacker models and shares no
conversations.

**The linear row is an exact reproduction of the prior work's, per split.** gptoss120b's
linear transfer scores `ai_dilemmas` 0.6356, `ant_hh` 0.7721, `balanced_refusal` 0.9126,
`daily_dilemmas` 0.7773, mean 0.7744 — every one of which matches
`why_last_iteration_adds_nothing.md`'s "v3 achieved (gptoss)" column and its §1 vintage
table to four decimals. Together with the four ceilings, that means **both halves of §1
here reproduce both halves of that investigation exactly**: the same features, pooling,
split and fit, independently re-derived. The non-linear columns are therefore additions to
that measurement rather than a parallel one, and a reader who trusts §2 of
`why_last_iteration_adds_nothing.md` can read these on the same footing.

**RBF-SVM against linear: +0.0004 mean AUROC and one extra core row out of 31.** So the
in-domain bound is not merely an upper bound that transfer fails to reach — there is
nothing there to reach. Trees are *worse* overall by 0.07, so no family here transfers
better than plain logistic regression.

**On core recovery, read the uncertainty before the numbers.** The balanced rule predicts
exactly half of each split positive, so a ranking uncorrelated with these 31 rows recovers
15.5 ± 2.8 of them (binomial, n=31). Measured against that, no family on either arm differs
from chance by more than 2.3 sd, and 31 rows cannot support a finer distinction.

The cross-arm replication settles it directly: **trees recover 17/31 on deepseekv4pro and
9/31 on gptoss120b** — the same model family, the same 31 rows, a swing of 8 rows (2.9 sd)
from nothing but which arm's training data was used. Whatever the deepseek 17 looked like,
it was noise. The honest reading is the weak one: no model family here does appreciably
better than chance on the core rows, and both smooth families trend somewhat worse.

An earlier revision of this section read 11/31 as showing the core rows are
"systematically mis-ranked". That over-read a 1.6-sd deviation; it is corrected here.

**And there is a selection effect that makes below-chance recovery the expected result,
not a discovery.** The 31 rows were *defined* as the ones every probe family in
`heldout_v3_vs_v2_overlap.md` got wrong — and those probes were mean-of-logits linear
heads trained on this same base + red-team data. A fresh linear model trained on that
same data is drawn from very nearly the same family, so of course it fails the rows
selected for being unreachable by it. gptoss120b's linear transfer recovers 9/31 (−2.3 sd)
and deepseekv4pro's 11/31 (−1.6 sd); both are what the selection predicts on its own.

The consequence for how this metric should be used: **core recovery is informative only
for a model far enough from the family that defined the core.** Trees are the only such
family here, and they land at chance. So the metric's one meaningful reading in this run
is "a genuinely different inductive bias does not clear the core either" — and even that
rests on a single family at +0.5 sd, which is why it is stated as a screen rather than a
result.

The other number on this table is solid and worth keeping: **val 0.9942 against an eval
mean of 0.769**. Both smooth families fit the red-team-heavy training distribution almost
perfectly and lose ~0.22 crossing to eval — the distribution gap in one line, and the same
story `why_last_iteration_adds_nothing.md` tells about the loop.

> **Scope — this bounds the readout axis only, not pooling.** Every family here is fitted
> on **mean-pooled** activations, so what is held fixed is the pooling and what varies is
> the readout. The result therefore says a non-linear *readout* has nothing extra to
> extract on `eval_ant_hh`; it says nothing whatever about whether a different *pooling*
> could. That distinction is not pedantic — the first partial sweep result already points
> the other way. At seed 42 on deepseekv4pro, `attention` (decoupled-query pooling, linear
> readout) scores **0.7740 on `eval_ant_hh` against `linear_then_softmax`'s 0.7149**, a
> +0.059 gain on precisely the split that matters, while losing 0.095 of mean AUROC
> everywhere else. One seed, and confounded by the batch-size difference documented above,
> so it is a lead rather than a result — but it is the reason §§2-4 are worth waiting for
> rather than being pre-empted by this section.

### 2. Per-architecture performance

**Partial — three of eight architectures, one seed, one arm (deepseekv4pro, seed 42).**
Reported now only because a pattern is visible across two of them; none of it is settled
until five seeds land. Per-split seed noise in the comparable heldout run was sd
0.014-0.059, so **no single-split difference below ~0.06 here is distinguishable from
reseeding.**

All eight architectures, ordered by mean AUROC:

| architecture | pooling | params | ai_dilemmas | `ant_hh` | balanced_refusal | daily_dilemmas | **mean** | best epoch |
|---|---|---|---|---|---|---|---|---|
| `linear_then_softmax` (deployed) | softmax | 5,377 | 0.9886 | 0.7149 | 0.9535 | 0.9878 | **0.9112** | 36 |
| `mlp_then_softmax` | softmax | 344,193 | 0.8804 | 0.7521 | 0.8328 | 0.9876 | 0.8632 | 17 |
| `attention` † | attn query | 10,754 | 0.6851 | **0.7740** | 0.9162 | 0.8877 | 0.8158 | 38 |
| `pre_mean` | mean | 5,377 | 0.6248 | **0.7744** | 0.8824 | 0.8189 | 0.7751 | 3 |
| `attention_then_mlp` † | attn query | 349,570 | 0.5993 | 0.7344 | 0.8771 | 0.8503 | 0.7653 | 8 |
| `mean_then_mlp` | mean | 344,193 | 0.6191 | 0.7719 | 0.8722 | 0.7809 | 0.7610 | 7 |
| `lda_shrinkage` | mean | 5,377 | 0.5229 | 0.7336 | 0.7623 | 0.6224 | 0.6603 | — |
| `difference_of_means` | mean | 5,377 | 0.5571 | 0.6072 | 0.7305 | 0.6901 | 0.6462 | — |

† optimizer settings differ (batch 128) — see the confound note above.

**The deployed head wins, by a wide margin.** Nothing here comes within 0.048 of its
0.9112, and the two closed-form estimators are 0.25 behind. Whatever the 31-row residue
needs, replacing the head with any of these seven costs a great deal elsewhere to get it.

**The clean pooling contrast.** `linear_then_softmax` vs `pre_mean` is the one pair that
differs *only* in pooling — same 5,377 parameters, same optimizer settings, same trainer.
Softmax-over-own-logits pooling is worth **+0.136 mean AUROC** over mean pooling
(0.9112 vs 0.7751). That is the largest single effect in the table, and it is the one
number here that is unambiguously about architecture rather than tuning. Note `pre_mean`'s
best epoch is 3: mean pooling saturates almost immediately and has nothing further to
learn. As an independent check, `pre_mean`'s 0.7751 sits close to §1's mean-pooled logistic
transfer (0.7694), which is the same function class fitted by a different optimizer.

**On the estimator axis**, the real LDA beats the unwhitened mass-mean (0.6603 vs 0.6462
mean, and 0.7336 vs 0.6072 on `ant_hh`) — the opposite of what Marks & Tegmark's
distribution-shift result would suggest, though at one seed and with both families far
behind every Adam-trained head, this is a weak signal at best. It does at least confirm
that writing a real LDA was worth doing: tuberlens' aliased `lda` would have reported the
`difference_of_means` row twice and hidden the difference entirely.

**The readout result is as §1 predicted.** `mlp_then_softmax` is the clean test — same
softmax pooling, same optimizer settings, 64x the parameters — and it **loses 0.048 mean
AUROC**. Best epoch falls monotonically with capacity across the whole table (36 → 17 → 8
as parameters go 5k → 344k → 350k): the bigger heads peak earlier and then overfit, which
is what they should do on 747 training rows, and what §1 said there was no signal to
justify. **Nothing beats the deployed head on mean AUROC.**

#### An arm-level asymmetry that affects how early stopping behaves

The two arms select their epochs very differently. On deepseekv4pro the best epochs are
36 and 38 for the two linear-readout heads; on gptoss120b the same architectures pick
**3 and 5**. §1's transfer measurement explains why: gptoss's validation split scores
**val AUROC = 1.0000** for every model family, against deepseek's 0.9942. Its validation
set is perfectly separable, so `best_val_auroc` cannot improve after the first few epochs,
early stopping latches onto whichever epoch first reaches 1.0, and everything after that
is unselected.

This is `why_last_iteration_adds_nothing.md`'s diagnosis item 4 — "validation is ~166 rows
on a model saturated by epoch 4" — now measured directly rather than inferred. Two
consequences for reading this sweep: the best-epoch *fix* matters **more** on gptoss, not
less (restoring epoch 3 rather than epoch 53 is a much larger intervention than restoring
36 rather than 86), and any per-architecture difference on that arm is selected by a
signal that ran out almost immediately. Cross-arm agreement is therefore the thing to
trust; a gptoss-only result is weakly selected by construction.

#### The `ant_hh` gain is probably a trade-off, not a win

Five of the seven alternatives score higher than the deployed head on `eval_ant_hh` — the
split holding 21 of the 31 core rows — which is the result this whole sweep was hoping for.
It should not be believed yet, for a reason visible in the table itself: **among the
Adam-trained heads, every architecture that gains on `ant_hh` is worse overall**, and the
ordering is close to inverse. The deployed head is best on mean AUROC and worst on
`ant_hh`; `pre_mean` and `attention` are 4th and 3rd on mean and best on `ant_hh`.

The two closed-form heads break the pattern by being worse *everywhere* including
`ant_hh`, which is a useful warning about the regression that tests this: a family that is
uniformly bad anchors the low end and pulls the fitted slope positive, masking a trade-off
operating among the stronger heads. `print_tradeoff_test` says so in its output.

There is a mundane mechanism that produces exactly this. `eval_ant_hh` is the split
*furthest* from the training distribution — short, blunt HH-style dialogue against a
red-team median of 1364 source characters (`why_last_iteration_adds_nothing.md`, "What the
data actually is"). A head that fits the training distribution harder does better on the
three near splits and worse on the far one; a head that fits it less hard trades the other
way. That is a generalisation trade-off driven by effective capacity, and it would look
identical to "this pooling understands harm better".

**The test that separates them**, once five seeds are in: regress `ant_hh` AUROC on mean
AUROC across all 8 architectures × 5 seeds × 2 arms. If the `ant_hh` gains lie on the
trade-off line, they are the mundane mechanism. An architecture that sits *above* that
line — better on `ant_hh` than its overall performance predicts — is the real thing. Until
that is run, no architecture here has been shown to help on the hard split.

### 2b. Both arms, seed 42 — the ranking replicates, and so does one gain

Seed 42 complete on both attacker arms. Mean AUROC, and each architecture's `eval_ant_hh`
delta against that arm's own `linear_then_softmax`:

| architecture | pooling | deepseek mean | gptoss mean | `ant_hh` Δ deepseek | `ant_hh` Δ gptoss |
|---|---|---|---|---|---|
| `linear_then_softmax` | softmax | **0.9112** | **0.9032** | — | — |
| `mlp_then_softmax` | softmax | 0.8632 | 0.8913 | +0.037 | +0.001 |
| `attention` | attn query | 0.8158 | 0.7789 | **+0.059** | **+0.052** |
| `pre_mean` | mean | 0.7751 | 0.7625 | **+0.060** | **+0.049** |
| `attention_then_mlp` | attn query | 0.7653 | 0.7751 | +0.020 | +0.005 |
| `mean_then_mlp` | mean | 0.7610 | 0.7668 | **+0.057** | **+0.051** |
| `lda_shrinkage` | mean | 0.6603 | 0.6916 | +0.019 | −0.077 |
| `difference_of_means` | mean | 0.6462 | 0.6453 | −0.108 | −0.095 |

**The ranking is arm-independent.** Top three and bottom two are identical on both arms;
only `pre_mean` / `attention_then_mlp` / `mean_then_mlp` shuffle in the middle, and on
gptoss those three sit within 0.013 of each other. Two arms whose red-team data was
written by different attacker models and shares no conversations produce the same ordering
— so the architecture effects here are properties of the head, not of one arm's data.

**And the `ant_hh` gain separates cleanly along the pooling axis.** Three architectures
replicate a gain of ~0.05 on both arms — `attention` (+0.059/+0.052), `pre_mean`
(+0.060/+0.049), `mean_then_mlp` (+0.057/+0.051) — and **all three abandon
softmax-over-own-logits pooling.** The one architecture that keeps that pooling and changes
only the readout, `mlp_then_softmax`, does **not** replicate (+0.037 then +0.001).

That is the pooling/readout split landing exactly where §1 predicted: a non-linear readout
has nothing to add, while *how the probe aggregates evidence across tokens* changes what it
sees on the split holding the hard core. It is also the first result in this document that
is about pooling at all — §1 could not speak to it, because it holds pooling fixed at mean
for every family.

**The caveat has not gone away.** Every one of those three costs 0.10-0.15 of mean AUROC to
buy ~0.05 on one split. One detail argues mildly *against* the pure trade-off reading,
though: the gains do not scale with the loss. `attention` gives up 0.095 of mean and
`mean_then_mlp` 0.150, yet both gain ~0.055 on `ant_hh` — if `ant_hh` were simply rising as
overall fit degrades, the bigger sacrifice should buy more. It does not; the gain looks
capped at ~0.05. That is what `print_tradeoff_test`'s residuals will quantify once more
seeds land, and it is the single most interesting open question in this run.

### 3. Hard-core recovery

### 4. Pooling vs. non-linearity

*Pending five seeds. The analysis that decides it — `print_tradeoff_test` in
`arch_sweep.py` — is implemented and runs automatically in `--summarize-only` once six or
more fits exist, so it does not depend on anyone remembering to do it. It regresses
`eval_ant_hh` AUROC on the mean of the other three splits across every fit and reports
each architecture's residual: on the line means the gain is the generalisation trade-off
described in §2, consistently above it means the architecture genuinely does better on the
hard split than its overall performance predicts.*

### 5. What the best-epoch fix was worth

### 6. Sensitivity to capacity and regularisation

## What this means for the retraining loop

*The readout-axis conclusions below are complete and supported by §1. Anything about
pooling waits on §§2-4.*

1. **Stop looking for a higher-capacity probe head.** Two independent measurements agree
   that a non-linear *readout* of layer 32 has nothing to add: zero in-domain ceiling
   headroom on the split holding two thirds of the hard core (−0.0004), and +0.0004 mean
   AUROC in actual transfer. `heldout_v3_vs_v2_overlap.md` offered two escapes from its
   31-row residue — a different eval concept boundary or a different probe architecture —
   and this closes the capacity half of the second. Whatever those rows need, more
   expressive feature extraction is not it.

2. **Fix the best-epoch restore before running any further probe comparison.**
   `docs/attribution_findings.md` §1 recorded the shallow-copy defect; this run measured
   what it costs. The single refit available so far puts the deployed architecture at
   0.9112 mean AUROC against the committed iteration-3 value of 0.8880 — **an order of
   magnitude larger than the `v3 − v2` vintage effect (−0.002) the vintage sweep spent 80
   fits trying to resolve.** Until it is fixed everywhere, every probe in this repo is its
   final epoch rather than its best, and cross-run comparisons inherit that noise. The fix
   is in `probe_architectures.build_probe`; porting it into `retrain.py`'s path is a
   separate, small change.

3. **Group by prompt before measuring anything with a high-capacity model on these eval
   splits.** Three of the four are prompt-paired with the label carried entirely by the
   assistant turn, and ungrouped CV *inverts* a model that can memorise — by 0.65 AUROC on
   the weakest split. The linear results published so far are unaffected, but the next
   person to try a richer model on this data will get a nonsense number and no error.

4. **Nothing reaches the hard core, but 31 rows cannot prove much either way.** Under the
   balanced rule a ranking uncorrelated with those rows recovers 15.5 ± 2.8 of them; the
   three transfer families land at 11, 12 and 17, i.e. within ±1.6 sd of chance. No family
   beats chance on the core, the two smooth ones trend slightly worse, and the set is far
   too small to separate them. Treat "core rows recovered" as a coarse screen — good for
   noticing an architecture that clears most of them, useless for ranking architectures
   that all sit near chance. The stronger evidence that these rows are a concept problem
   rather than a model problem remains `heldout_v3_vs_v2_overlap.md`'s own: they survived
   two attacker models × two disjoint red-team vintages × with and without the base data ×
   five seeds. Two thirds sit in `eval_ant_hh`, and reading those 21 conversations against
   the probe's concept description is the cheapest next step — an eval-side question, not
   a probe-side one.

## Reproducing

```bash
# 80 fits off cached activations; resumes at (arm, architecture, seed) granularity
.venv_claude/bin/python scripts/arch_sweep.py
.venv_claude/bin/python scripts/arch_sweep.py --summarize-only     # re-derive, no refits
.venv_claude/bin/python scripts/arch_sweep.py --sensitivity        # hidden_dim x weight_decay
.venv_claude/bin/python scripts/arch_sweep.py --legacy-best-epoch  # the control arm

# the ceiling / transfer measurement (run it ALONE — see the note below)
.venv_claude/bin/python scripts/nonlinear_ceiling.py

# seconds, no cached activations needed
.venv_claude/bin/python scripts/test_probe_architectures.py
```

**Run one of these at a time.** Both hold large activation sets resident, and this box's
cgroup limit is below its 31 GB of RAM: running the ceiling measurement alongside the
sweep OOM-killed the sweep at 13 GB anon-rss, silently and with no traceback. The sweep
checkpoints per fit and resumes cleanly, so the cost is one fit — but a process that
simply vanishes looks exactly like one still working.
