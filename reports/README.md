# Reports on this branch

Seven reports, all on `google/gemma-3-27b-it` layer 32 with `linear_then_softmax`
heads, seed 42, scored on the eval splits under `eval_sets/<concept>/` off precomputed
activations. Heavy artefacts (activations, fit caches, probe pickles) are gitignored; what
is published here is the numbers and the code that produced them.

| report | question | headline |
| --- | --- | --- |
| [`cross_concept_ceiling/`](cross_concept_ceiling/REPORT.md) | What is the *ceiling* — the best a probe of this family can do when trained on eval-distribution data — for each concept alone, and for all three pooled into one probe? | Per-concept ceilings 0.946–0.977. One shared direction reaches 0.913–0.961, so pooling costs only 0.011–0.050 AUROC — **but TPR@1%FPR falls to 0.000 on every concept.** `instructions/oig_omission` has a ceiling of 0.630 and is not linearly readable at this layer. |
| [`concept_probes_summary.md`](concept_probes_summary.md) | Four generating models each wrote ~50 balanced rows per concept. Does it matter which one? | Yes, and the ranking **inverts by concept**: across all four, hu_ha orders exactly opposite to highstakes and instructions (Spearman −1.00), which order identically to each other. dsv4pro best on hu_ha (0.887) / worst on instructions (0.575); llama70b the reverse (0.897 / 0.797). |
| [`llama8b_concept_probes/`](llama8b_concept_probes/REPORT.md) · [`llama70b_concept_probes/`](llama70b_concept_probes/REPORT.md) · [`dsv4pro_concept_probes/`](dsv4pro_concept_probes/REPORT.md) · [`nemotron550b_concept_probes/`](nemotron550b_concept_probes/REPORT.md) | Per-generator detail: `{single, seq_ens10} × {dev, split}` validation, 12 probes each. | Ensemble and validation source are the smallest terms everywhere — most cells span <0.03 across their four arms, against 0.22 between generators. |
| [`ensemble_vs_single.md`](ensemble_vs_single.md) | Across all 24 paired cells, what does the 10-member sequential ensemble buy over a single probe — on score, and on fit time? | **+0.010 AUROC mean for 7.4x the fit time**, sign unreliable (ensemble wins 16/24, range -0.055 to +0.051). It does not reduce sensitivity to the validation source either. |
| [`combo_probes/`](combo_probes/REPORT.md) · [page](combo_probes/pooling-generators.html) | Pool the four generators' cuts: one single probe per non-empty subset (15 pools x 2 validation sources x 3 concepts, 90 probes). Does more generators beat the best one? | **A pool beats the average of its members 57/66 times and the best of them 33/66** — a coin flip. Additive on highstakes (all four wins 4/4 splits, 0.939 vs a 0.963 ceiling), dilutive on instructions (loses 5/7 splits to `llama70b` alone), flat on hu_ha. |
| [`generalization_tests/`](generalization_tests/REPORT.md) | Probes trained on a general/not-general dataset under borrowed concept labels — a distinction the eval sets do not measure. | Generality transfers to high-stakes (0.686 one way, 0.367 inverted) and not to harm or instruction-following. Random-label controls sit at 0.52–0.60, so **~0.55 is the noise floor here, not 0.50**. |

## The one bug that touches every number above

`linear_then_softmax` defaults to `batch_size: 16, gradient_accumulation_steps: 4`, and
tuberlens' trainer steps only on `(batch_idx + 1) % accumulation == 0` with **no
end-of-epoch flush** (`pytorch_classifiers.py:299-327`, unchanged since the rename). A
training set yielding fewer than 4 batches therefore never calls `optimizer.step()`: the
fit burns its whole epoch budget, loss sits at ln 2, validation AUROC is exactly 0.5, and
the probe is returned at its random initialization.

Every `split` arm in these experiments is ~40 rows = 3 batches and would hit it exactly,
so `scripts/concept_probes.py:capped_spec` caps the accumulation at the batch count — a
no-op for any fit with ≥4 batches, verified bit-identical.

This is not hypothetical. `experiment16_cloud` ran that shape unguarded on
`data/hu_harm_llama70b_50.jsonl` (40 train / 10 val) and its iteration-0 probe scored
**0.336** mean AUROC, below chance on three of four splits.
`experiment17_cloud` is byte-identical except for `validation.dev_data`, which puts all 50
rows in training (4 batches), and scored **0.846**. That 0.51 gap is the bug, not the
validation source — and any run in this repo whose iteration-0 training set was under 64
rows has the same defect.

## Scripts

```bash
.venv_claude/bin/python scripts/concept_probes.py --generator llama70b --phase all
.venv_claude/bin/python scripts/cross_concept_ceiling.py
.venv_claude/bin/python scripts/generalization_tests.py --concept hu_ha --phase all
.venv_claude/bin/python scripts/concept_probes_report.py   # regenerate reports
.venv_claude/bin/python scripts/ceiling_report.py
.venv_claude/bin/python scripts/combo_probes.py --phase all
.venv_claude/bin/python scripts/combo_probes_report.py
```
