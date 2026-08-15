# Does a different probe architecture reach the eval rows red-teaming cannot?

*Run: 2026-08-15. Code: `scripts/arch_sweep.py`, `scripts/nonlinear_ceiling.py`,
`src/agentic_redteam/probe_architectures.py`, checks in
`scripts/test_probe_architectures.py`. Raw output:
`results_hu_harm_gemma27b_batch_ablation/arch_sweep/` (progress sidecar `arch_progress.jsonl`,
`arch_sweep.json`, `nonlinear_ceiling.json`), logs `logs/arch_sweep.log`,
`logs/nonlinear_ceiling.log`.*

> **Status: ceiling complete, sweep in progress.** §1 (the non-linear ceiling) is
> measured and its headline finding is settled. §§2-6 are filled in from
> `arch_sweep.json` as fits land — the sweep is ~32 h of wall clock and runs past any one
> session. Do not cite §§2-6 until this banner says they are complete.
>
> **To check on it or pick it up:**
> ```bash
> .venv_claude/bin/python scripts/arch_sweep.py --summarize-only   # read partial results
> tail -f logs/arch_sweep.log                                      # watch fits land
> nohup bash run_arch_sweep.sh 42 43 44 45 46 >> logs/arch_sweep.log 2>&1 &   # (re)start
> nohup bash failsafe_commit_arch.sh > logs/failsafe_arch.out 2>&1 &          # 30-min pushes
> ```
> Everything resumes at `(arm, architecture, seed, variant, legacy)` granularity, so a
> restart costs at most the fit that was in flight. Fits are ~20 min for the batch-16
> heads and ~55 min for the two attention heads (batch 128).

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
- **The primary readout is hard-core recovery, not mean AUROC.** Result 3 of the heldout
  write-up showed that swapping a whole red-team vintage moves the error set about as much
  as a reseed does. An architecture that only moves the mean has done the same thing; one
  that clears rows from the 31-row core has reached something no red-team vintage did.

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

### 2. Per-architecture performance

### 3. Hard-core recovery

### 4. Pooling vs. non-linearity

### 5. What the best-epoch fix was worth

### 6. Sensitivity to capacity and regularisation

## What this means for the retraining loop

*Pending.*

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
