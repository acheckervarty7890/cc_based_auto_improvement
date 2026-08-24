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
