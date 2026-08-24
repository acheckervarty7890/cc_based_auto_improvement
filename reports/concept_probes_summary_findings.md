## Findings

### The generator ranking inverts between concepts

| generator | hu_ha | highstakes | instructions |
| --- | --- | --- | --- |
| llama70b | 0.838 | **0.897** | **0.797** |
| llama8b | 0.861 | 0.851 | 0.670 |
| dsv4pro | **0.887** | 0.823 | 0.575 |
| *ceiling* | *0.977* | *0.963* | *0.946* |

dsv4pro is the best generator for hu_ha and the worst for instructions; llama70b is
exactly the reverse. There is no "better generator" here, only better-matched ones — so a
result of the form "model X writes good synthetic probe-training data" does not survive
being asked *for which concept*.

Model size does not order the results either: the 8B model beats both larger models on
neither concept outright but sits between them on all three, and the 70B model beats the
deepseek model on two concepts while losing the third by 0.05.

### The spread between generators is larger than the gap to the ceiling

On `instructions`, llama70b reaches 0.797 against a 0.946 ceiling — a gap of 0.149 — while
dsv4pro reaches 0.575, a further 0.222 below llama70b. So on this concept, *which model
wrote the 50 rows* matters more than everything left on the table between the best cut and
training on eval-distribution data itself.

On `hu_ha` the picture is the opposite: all three generators land within 0.05 of each
other (0.838–0.887) and all are within 0.14 of the ceiling. A 50-row synthetic cut of harm
is close to sufficient regardless of who writes it; a 50-row cut of instruction-following
is not, and depends heavily on who writes it.

### Ensemble and validation source remain second-order everywhere

Across all 36 probes, no (generator, concept) cell shows a spread of more than 0.06 across
its four arms, and most are under 0.03 — against concept-level spreads of 0.22 between
generators and 0.28 between concepts. The `seq_ens10` vs `single` and `dev` vs `split`
choices are consistently the smallest terms in this experiment.

### The gradient-accumulation cap applies to all 36 probes

Every `split` arm here is ~40 training rows = 3 batches against `linear_then_softmax`'s
default `gradient_accumulation_steps: 4`, and the trainer has no end-of-epoch flush — so
uncapped, all eighteen `split` probes would have been returned at their random
initialization. `experiment16_cloud` ran that exact shape on `hu_harm_llama70b_50` and its
iteration-0 probe scored 0.336, below chance on three of four splits, against 0.846 for
the otherwise-identical dev-validated `experiment17_cloud`. See
`scripts/concept_probes.py:capped_spec`.
