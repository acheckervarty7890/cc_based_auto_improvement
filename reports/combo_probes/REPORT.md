# Pooled generators — one single probe per subset of the four synthetic cuts

_Generated 2026-08-24 11:25:31Z._

## What is being measured

Four attacker models each wrote ~50 balanced two-turn conversations per concept.
[`concept_probes_summary.md`](../concept_probes_summary.md) asked what each is worth
**alone**. This asks what they are worth **pooled**: every non-empty subset of

&nbsp;&nbsp;&nbsp;&nbsp;`llama8b` (`l8b`) · `llama70b` (`l70b`) · `dsv4pro` (`dsv`) · `nemotron550b` (`nem`)

is concatenated into one training set, giving 4 singles (~50 rows), 6 pairs (~100),
4 triples (~150) and the all-four pool (~200). Only **single** probes are fitted — no
ensembles — crossed with the two validation sources:

| axis | values |
| --- | --- |
| training data | the 15 non-empty subsets of the four generators |
| validation | `dev` (the concept's `dev_samples/` set is the whole validation set; every training row trains) · `split` (a 0.2 content-deterministic slice is held out instead) |

30 probes per concept, 90 in total. `google/gemma-3-27b-it`
layer 32, `linear_then_softmax`, seed 42, scored on full eval splits off precomputed
activations. Every cell below is mean AUROC over that concept's eval splits.

## Mean AUROC by pool and validation source

```
                           hu_ha  instructions  highstakes
combo            val_mode                                 
l8b              dev       0.851         0.633       0.874
                 split     0.871         0.678       0.828
l70b             dev       0.852         0.778       0.900
                 split     0.855         0.825       0.895
dsv              dev       0.888         0.570       0.792
                 split     0.871         0.562       0.818
nem              dev       0.875         0.574       0.835
                 split     0.876         0.574       0.811
l8b+l70b         dev       0.852         0.754       0.895
                 split     0.864         0.803       0.908
l8b+dsv          dev       0.887         0.617       0.868
                 split     0.884         0.682       0.870
l8b+nem          dev       0.856         0.581       0.896
                 split     0.875         0.619       0.894
l70b+dsv         dev       0.893         0.772       0.911
                 split     0.894         0.752       0.925
l70b+nem         dev       0.857         0.746       0.928
                 split     0.863         0.764       0.915
dsv+nem          dev       0.892         0.568       0.851
                 split     0.887         0.555       0.823
l8b+l70b+dsv     dev       0.895         0.775       0.932
                 split     0.891         0.780       0.927
l8b+l70b+nem     dev       0.856         0.773       0.936
                 split     0.868         0.766       0.934
l8b+dsv+nem      dev       0.881         0.646       0.889
                 split     0.890         0.686       0.899
l70b+dsv+nem     dev       0.883         0.767       0.928
                 split     0.888         0.765       0.933
l8b+l70b+dsv+nem dev       0.865         0.744       0.935
                 split     0.893         0.779       0.939
```

## Does pooling help? Mean AUROC by number of generators

```
                       hu_ha  instructions  highstakes
n_generators val_mode                                 
1            dev       0.867         0.639       0.850
             split     0.868         0.660       0.838
2            dev       0.873         0.673       0.891
             split     0.878         0.696       0.889
3            dev       0.879         0.740       0.921
             split     0.884         0.749       0.923
4            dev       0.865         0.744       0.935
             split     0.893         0.779       0.939
```

## Best pool vs best single generator

| concept | best single | AUROC | best pool of any size | AUROC | all four | ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| hu_ha | `dsv / dev` | 0.888 | `l8b+l70b+dsv / dev` | 0.895 | 0.893 | 0.977 |
| instructions | `l70b / split` | 0.825 | `l70b / split` | 0.825 | 0.779 | 0.946 |
| highstakes | `l70b / dev` | 0.900 | `l8b+l70b+dsv+nem / split` | 0.939 | 0.939 | 0.963 |

`ceiling` is the same concept's within-concept ceiling from
[`cross_concept_ceiling/`](../cross_concept_ceiling/REPORT.md) — a probe of this
family trained on eval-distribution data. It is a pooled-across-splits AUROC on a
balanced 100/class subsample, not a mean of per-split AUROCs, so read it as a
reference point rather than a directly comparable cell.

## What each generator adds to a pool that lacks it

Mean AUROC change from adding one generator to a pool, averaged over the seven
non-empty subsets of the other three — so every generator is scored against the
same seven baselines.

```
                       highstakes  hu_ha  instructions
generator    val_mode                                 
llama8b      dev            0.029 -0.007         0.016
             split          0.036  0.004         0.045
llama70b     dev            0.066 -0.004         0.163
             split          0.077  0.001         0.150
dsv4pro      dev            0.007  0.028         0.007
             split          0.019  0.022        -0.004
nemotron550b dev            0.027 -0.004        -0.011
             split          0.024  0.005        -0.021
```

## Is a pool more than the average of its parts?

For every pool of two or more, its AUROC minus the **mean** of its members'
own singleton AUROCs, and minus the **best** of them. If pooling only averaged
its inputs the first column would sit at zero; if it added coverage the second
would be positive.

| concept | val_mode | pool − mean(members) | pool > mean | pool − best(member) | pool > best |
| --- | --- | --- | --- | --- | --- |
| hu_ha | dev | +0.008 | 7/11 | -0.007 | 3/11 |
| hu_ha | split | +0.013 | 10/11 | +0.007 | 7/11 |
| instructions | dev | +0.065 | 9/11 | -0.016 | 1/11 |
| instructions | split | +0.063 | 9/11 | -0.040 | 2/11 |
| highstakes | dev | +0.056 | 11/11 | +0.020 | 9/11 |
| highstakes | split | +0.068 | 11/11 | +0.037 | 11/11 |

## Does a generator's solo score predict what it adds to a pool?

Correlation between the eight solo AUROCs above (4 generators x 2 validation
modes) and the eight marginal contributions.

| scope | pearson | spearman |
| --- | --- | --- |
| hu_ha | +0.716 | +0.762 |
| instructions | +0.967 | +0.786 |
| highstakes | +0.890 | +0.881 |
| **all concepts pooled** | +0.239 | +0.274 |

## All four vs the best single generator, split by split

The pooled means above average over splits, which can hide a pool that wins on
one split and loses on another. `val=split` shown; the best single is the one
with the highest mean AUROC in that concept.

**hu_ha** — best single `nem`, all four `l8b+l70b+dsv+nem`; all four wins on 3/4 splits.

```
dataset           eval_ai_dilemmas  eval_ant_hh  eval_balanced_refusal  eval_daily_dilemmas
combo                                                                                      
nem                          0.973        0.723                  0.826                0.981
l8b+l70b+dsv+nem             0.977        0.722                  0.884                0.989
```

**instructions** — best single `l70b`, all four `l8b+l70b+dsv+nem`; all four wins on 2/7 splits.

```
dataset           anthropic_harmless_refusal  bbq_substitution  hc_context_drift  hc_contradiction  mm_substitution  oig_context_drift  oig_omission
combo                                                                                                                                               
l70b                                   0.773             0.933             0.798             0.870            0.911              0.766         0.724
l8b+l70b+dsv+nem                       0.846             0.882             0.648             0.893            0.886              0.696         0.599
```

**highstakes** — best single `l70b`, all four `l8b+l70b+dsv+nem`; all four wins on 4/4 splits.

```
dataset           anthropic_hh_balanced  mt_balanced  mts_balanced  toolace_balanced
combo                                                                               
l70b                              0.957        0.816         0.944             0.861
l8b+l70b+dsv+nem                  0.975        0.943         0.970             0.867
```

## dev-set validation vs a 0.2 split

| concept | mean dev − split | mean abs diff | dev wins |
| --- | --- | --- | --- |
| hu_ha | -0.006 | 0.010 | 4/15 |
| instructions | -0.019 | 0.026 | 6/15 |
| highstakes | +0.003 | 0.013 | 8/15 |

## What the two validation sources cost

Every probe here is one `linear_then_softmax` head over cached activations,
so the only meaningful cost difference is that the `dev` arm scores the whole
`dev_samples/` set every epoch while `split` scores a ~20-row slice.

| concept | dev set | dev fit (median) | split fit (median) |
| --- | --- | --- | --- |
| hu_ha | 290 rows | 9s | 2s |
| instructions | 436 rows | 10s | 3s |
| highstakes | 1908 rows | 79s | 2s |

## hu_ha — AUROC per eval split

```
dataset                    eval_ai_dilemmas  eval_ant_hh  eval_balanced_refusal  eval_daily_dilemmas   MEAN
combo            val_mode                                                                                  
l8b              dev                  0.911        0.689                  0.831                0.970  0.851
                 split                0.956        0.735                  0.820                0.975  0.871
l70b             dev                  0.861        0.737                  0.841                0.969  0.852
                 split                0.836        0.719                  0.910                0.955  0.855
dsv              dev                  0.993        0.709                  0.868                0.984  0.888
                 split                0.994        0.625                  0.893                0.974  0.871
nem              dev                  0.967        0.758                  0.798                0.979  0.875
                 split                0.973        0.723                  0.826                0.981  0.876
l8b+l70b         dev                  0.884        0.746                  0.828                0.950  0.852
                 split                0.895        0.727                  0.877                0.959  0.864
l8b+dsv          dev                  0.982        0.728                  0.851                0.986  0.887
                 split                0.989        0.693                  0.876                0.981  0.884
l8b+nem          dev                  0.908        0.736                  0.819                0.962  0.856
                 split                0.978        0.697                  0.839                0.984  0.875
l70b+dsv         dev                  0.983        0.729                  0.875                0.987  0.893
                 split                0.985        0.703                  0.908                0.978  0.894
l70b+nem         dev                  0.873        0.731                  0.851                0.971  0.857
                 split                0.902        0.745                  0.838                0.967  0.863
dsv+nem          dev                  0.990        0.723                  0.872                0.984  0.892
                 split                0.993        0.714                  0.856                0.984  0.887
l8b+l70b+dsv     dev                  0.992        0.718                  0.882                0.988  0.895
                 split                0.975        0.729                  0.872                0.988  0.891
l8b+l70b+nem     dev                  0.895        0.715                  0.844                0.969  0.856
                 split                0.923        0.715                  0.855                0.979  0.868
l8b+dsv+nem      dev                  0.971        0.739                  0.830                0.985  0.881
                 split                0.985        0.738                  0.847                0.989  0.890
l70b+dsv+nem     dev                  0.956        0.739                  0.850                0.986  0.883
                 split                0.982        0.712                  0.876                0.983  0.888
l8b+l70b+dsv+nem dev                  0.911        0.707                  0.858                0.982  0.865
                 split                0.977        0.722                  0.884                0.989  0.893
```

## instructions — AUROC per eval split

```
dataset                    anthropic_harmless_refusal  bbq_substitution  hc_context_drift  hc_contradiction  mm_substitution  oig_context_drift  oig_omission   MEAN
combo            val_mode                                                                                                                                           
l8b              dev                            0.782             0.735             0.541             0.531            0.662              0.627         0.549  0.633
                 split                          0.737             0.867             0.620             0.611            0.717              0.629         0.567  0.678
l70b             dev                            0.534             0.863             0.714             0.909            0.934              0.714         0.776  0.778
                 split                          0.773             0.933             0.798             0.870            0.911              0.766         0.724  0.825
dsv              dev                            0.842             0.605             0.502             0.500            0.468              0.511         0.561  0.570
                 split                          0.723             0.578             0.500             0.499            0.548              0.550         0.537  0.562
nem              dev                            0.700             0.679             0.518             0.529            0.495              0.557         0.543  0.574
                 split                          0.773             0.611             0.500             0.503            0.527              0.580         0.524  0.574
l8b+l70b         dev                            0.805             0.796             0.643             0.833            0.903              0.698         0.604  0.754
                 split                          0.856             0.869             0.619             0.916            0.954              0.765         0.639  0.803
l8b+dsv          dev                            0.679             0.744             0.591             0.577            0.596              0.570         0.563  0.617
                 split                          0.790             0.792             0.720             0.679            0.616              0.583         0.596  0.682
l8b+nem          dev                            0.596             0.690             0.531             0.531            0.614              0.574         0.528  0.581
                 split                          0.735             0.708             0.548             0.567            0.617              0.613         0.549  0.619
l70b+dsv         dev                            0.779             0.909             0.687             0.843            0.842              0.699         0.647  0.772
                 split                          0.885             0.893             0.704             0.794            0.718              0.647         0.622  0.752
l70b+nem         dev                            0.502             0.866             0.756             0.788            0.914              0.762         0.634  0.746
                 split                          0.661             0.885             0.741             0.823            0.902              0.723         0.615  0.764
dsv+nem          dev                            0.696             0.639             0.524             0.516            0.536              0.520         0.545  0.568
                 split                          0.654             0.577             0.493             0.498            0.575              0.583         0.505  0.555
l8b+l70b+dsv     dev                            0.902             0.810             0.584             0.942            0.896              0.656         0.632  0.775
                 split                          0.897             0.881             0.618             0.890            0.851              0.697         0.624  0.780
l8b+l70b+nem     dev                            0.754             0.830             0.658             0.948            0.942              0.703         0.574  0.773
                 split                          0.774             0.824             0.554             0.915            0.982              0.722         0.588  0.766
l8b+dsv+nem      dev                            0.720             0.750             0.608             0.677            0.637              0.578         0.552  0.646
                 split                          0.754             0.775             0.570             0.824            0.691              0.630         0.557  0.686
l70b+dsv+nem     dev                            0.725             0.883             0.703             0.862            0.836              0.750         0.611  0.767
                 split                          0.715             0.891             0.756             0.895            0.798              0.665         0.632  0.765
l8b+l70b+dsv+nem dev                            0.819             0.873             0.563             0.873            0.829              0.657         0.596  0.744
                 split                          0.846             0.882             0.648             0.893            0.886              0.696         0.599  0.779
```

## highstakes — AUROC per eval split

```
dataset                    anthropic_hh_balanced  mt_balanced  mts_balanced  toolace_balanced   MEAN
combo            val_mode                                                                           
l8b              dev                       0.823        0.896         0.966             0.810  0.874
                 split                     0.862        0.818         0.819             0.814  0.828
l70b             dev                       0.947        0.854         0.942             0.856  0.900
                 split                     0.957        0.816         0.944             0.861  0.895
dsv              dev                       0.917        0.745         0.900             0.607  0.792
                 split                     0.965        0.725         0.926             0.656  0.818
nem              dev                       0.770        0.927         0.897             0.745  0.835
                 split                     0.740        0.886         0.899             0.720  0.811
l8b+l70b         dev                       0.964        0.828         0.903             0.884  0.895
                 split                     0.959        0.885         0.912             0.874  0.908
l8b+dsv          dev                       0.962        0.837         0.927             0.744  0.868
                 split                     0.961        0.837         0.922             0.761  0.870
l8b+nem          dev                       0.895        0.912         0.954             0.824  0.896
                 split                     0.926        0.902         0.937             0.814  0.894
l70b+dsv         dev                       0.980        0.856         0.960             0.850  0.911
                 split                     0.979        0.919         0.956             0.847  0.925
l70b+nem         dev                       0.956        0.956         0.949             0.850  0.928
                 split                     0.948        0.915         0.950             0.847  0.915
dsv+nem          dev                       0.938        0.871         0.905             0.689  0.851
                 split                     0.907        0.796         0.896             0.693  0.823
l8b+l70b+dsv     dev                       0.978        0.908         0.959             0.881  0.932
                 split                     0.976        0.917         0.953             0.863  0.927
l8b+l70b+nem     dev                       0.959        0.940         0.960             0.885  0.936
                 split                     0.949        0.945         0.966             0.877  0.934
l8b+dsv+nem      dev                       0.950        0.886         0.938             0.782  0.889
                 split                     0.960        0.894         0.942             0.801  0.899
l70b+dsv+nem     dev                       0.977        0.948         0.943             0.845  0.928
                 split                     0.973        0.948         0.961             0.849  0.933
l8b+l70b+dsv+nem dev                       0.974        0.960         0.957             0.850  0.935
                 split                     0.975        0.943         0.970             0.867  0.939
```

## Cross-check against the per-generator runs

A singleton pool's training file is byte-identical to that generator's own
cut, so its probe must reproduce the `single` arm of
[`concept_probes_summary.md`](../concept_probes_summary.md) exactly. Largest
absolute per-split AUROC difference over the 24 comparable arms: **0.00e+00**.

## Findings

### 1. Pooling is a hedge, not a gain

Across all 66 pools (11 multi-generator pools x 2 validation modes x 3 concepts), a
pool beats the **mean** of its own members' solo scores **57/66** times and beats the
**best** of them **33/66** — an exact coin flip. That is the whole result in one line:
concatenating cuts reliably protects you from having picked a bad generator, and does
not reliably beat having picked the good one.

Which makes the interesting question not "does pooling help" but "when does it do more
than average", and that answer is concept-dependent in the same way everything else in
this repo has turned out to be.

### 2. On high-stakes pooling genuinely adds; on instructions it genuinely dilutes

These two concepts sit at opposite ends, and the per-split tables show it is not an
averaging artefact:

- **highstakes** — all four beats the best single (`l70b`) on **4/4 splits**, by
  0.006–0.127, and the pool beats its best member in **20/22** cells. The mean AUROC
  is monotone in pool size in both validation modes (0.850 → 0.891 → 0.921 → 0.935 dev;
  0.838 → 0.889 → 0.923 → **0.939** split). The generators are finding different parts
  of the concept and the parts add up.
- **instructions** — all four *loses* to `l70b` alone on **5/7 splits**, and badly:
  `hc_context_drift` 0.798 → 0.648, `oig_omission` 0.724 → 0.599. The pool beats its
  best member in only **3/22** cells. Mean AUROC still rises with pool size
  (0.660 → 0.779 split), but only because it is climbing out of the hole the three
  weak generators dug — the ceiling of the exercise is `l70b` alone at 0.825, and no
  pool reaches it.
- **hu_ha** — nothing moves. Every one of the 30 cells lies in 0.851–0.895, a 0.044
  band, and the pool-size curve is not even monotone under `dev` validation
  (0.867 → 0.873 → 0.879 → 0.865). This concept was already the flattest across
  generators, and it is the flattest across pools too.

### 3. A generator's solo score predicts what it adds — but only within a concept

Correlation between the eight solo AUROCs and the eight marginal contributions is
+0.72 (hu_ha), +0.97 (instructions), +0.89 (highstakes) — and **+0.24** with all three
concepts pooled. So there is no such thing as a generally useful generator to add:
`llama70b` is worth +0.150/+0.163 on instructions and −0.004/+0.001 on hu_ha, while
`dsv4pro` is the only generator with a positive contribution on hu_ha (+0.022/+0.028)
and is worth +0.007/−0.004 on instructions. That is the same inversion
[`concept_probes_summary.md`](../concept_probes_summary.md) found in the solo scores,
reappearing in the pooled ones — which is a consistency check on both, since these are
different probes fit on different data.

### 4. 200 synthetic rows gets within 0.024 of the high-stakes ceiling

The best high-stakes pool scores **0.939** mean AUROC against a within-concept ceiling
of 0.963 — a probe of the same family trained on eval-distribution data. On
instructions the best cell is 0.825 against a 0.946 ceiling, and on hu_ha 0.895
against 0.977. So the pooling gain closed most of the remaining gap on exactly one of
the three concepts, and the two concepts where a 50-row cut was already near its
plateau stayed there.

### 5. The validation source is still the smallest term

Mean `dev − split` is −0.006 (hu_ha), −0.019 (instructions), +0.003 (highstakes), with
mean absolute differences of 0.010–0.026 — smaller than the spread across generators
(0.22) and smaller than the pooling effect on highstakes (0.10). It is also the only
real cost difference in this experiment: the `dev` arm scores the whole `dev_samples/`
set every epoch, which on high-stakes' 1908-row dev set is a 79 s median fit against
2 s for the 0.2 split. Paying 40x for a validation source worth ±0.02 is a poor trade
at this scale — but note the reason `--dev-data` exists is comparability across
iterations, not accuracy, and that reason is untouched by these numbers.

### 6. Caveats

Each cell is one probe, not a distribution: there is no seed replication here, and
[`ensemble_vs_single.md`](../ensemble_vs_single.md) measured single-probe seed-to-seed
movement of up to 0.055 AUROC on this same data. Differences below ~0.02 should be read
as noise, which is most of hu_ha. The pools are also confounded — a 4-generator pool has
both more diversity *and* 4x the rows of a single cut, and this design cannot separate
them; a size-matched control (200 rows from one generator) would be the way to.

## Reproducing

```bash
.venv_claude/bin/python scripts/combo_probes.py --phase all
.venv_claude/bin/python scripts/combo_probes_report.py
```

No model is loaded at any point. `prepare` assembles each pool's train/val activation
cache by addressing rows in the per-generator master blobs *by conversation content*,
which is sound because `stable_train_test_split` is content-deterministic — a
conversation falls on the same side of the train/val line in every pool it appears in.
The eval reads each split's activations once and scores all of that concept's probes
against it, rather than reloading 46 GB of high-stakes activations per probe.
