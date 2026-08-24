## Findings

### 1. Per-concept ceilings are high, and one shared direction gets close to them

| | hu_ha | highstakes | instructions | pooled |
| --- | --- | --- | --- | --- |
| `within/<concept>` — the per-concept ceiling | **0.977** | **0.963** | **0.946** | — |
| `cross/aligned` | 0.961 | 0.913 | 0.935 | **0.937** |
| `cross/native` | 0.930 | 0.900 | 0.862 | 0.889 |

Trained on eval-distribution data, a layer-32 linear head reaches 0.946–0.977 on each
concept alone. Pooling all three into a *single* probe costs only **0.011–0.050** AUROC
once the label polarity is aligned — smallest on hu_ha (0.977 → 0.961), largest on
highstakes (0.963 → 0.913). Whatever these three concepts are, they are close enough at
layer 32 that one direction carries most of all three.

Read the other way: the cross-concept ceiling is a *ceiling*, not a promise. Any
single-probe result on these concepts should be compared to ~0.94, not to 1.0.

### 2. Label polarity is not a bookkeeping detail

`hu_ha` and `highstakes` call the *problematic* response positive (harmful, high-stakes);
`instructions` calls the *desirable* one positive
(`assistant_follows_the_instruction`). `cross/native` pools them as-labelled;
`cross/aligned` inverts instructions so all three positives are the problematic response.

`aligned` wins on **every** concept, not only the one it re-labels:

| concept | native | aligned | Δ |
| --- | --- | --- | --- |
| instructions | 0.862 | 0.935 | +0.073 |
| hu_ha | 0.930 | 0.961 | +0.031 |
| highstakes | 0.900 | 0.913 | +0.013 |

Forcing "harmful" and "correctly followed the instruction" onto the same side of one
direction does not merely fail on instructions — it drags the shared direction off the
two concepts whose labels were never touched. Note this is a genuine measurement, not an
identity: inverting a third of the rows is not a symmetry of the pooled task, so
`aligned` is not `1 - native` (0.937 and 0.889 do not sum to 1).

### 3. `oig_omission` is unreachable at this layer, and that reframes every result on it

| split | within | cross/aligned | cross/native |
| --- | --- | --- | --- |
| hu_ha/eval_ai_dilemmas | 1.000 | 1.000 | 0.983 |
| hu_ha/eval_daily_dilemmas | 0.999 | 1.000 | 0.943 |
| instructions/hc_context_drift | 1.000 | 0.973 | 0.994 |
| instructions/anthropic_harmless_refusal | 1.000 | 0.985 | 0.946 |
| highstakes/anthropic_hh_balanced | 0.987 | 0.897 | 0.858 |
| highstakes/toolace_balanced | 0.928 | 0.809 | 0.835 |
| hu_ha/eval_ant_hh | 0.895 | 0.867 | 0.847 |
| **instructions/oig_omission** | **0.630** | 0.552 | 0.477 |

Thirteen of fifteen splits have a within-concept ceiling of 0.89 or better and six are at
0.98+. `oig_omission` is the exception at **0.630** — trained on eval-distribution data of
its own concept, with early stopping on real dev data. "The assistant omitted requested
content" is therefore not linearly readable at layer 32, and no better training data will
fix it; only a different layer, architecture, or representation could.

That reframes the 0.55–0.59 the synthetic-cut probes score on this split: they are not
underperforming, they are already at the ceiling. It also means the `instructions` concept
mean carries one split that is structurally capped ~0.35 below its siblings.

### 4. `toolace_balanced` resists sharing more than it resists learning

highstakes' `toolace_balanced` has a healthy per-concept ceiling (0.928) but the largest
drop of any split when the probe is shared (0.809 aligned, −0.119). `anthropic_hh_balanced`
behaves the same way (0.987 → 0.897, −0.090). So the cost of a shared direction is not
spread evenly: it lands almost entirely on highstakes' two largest splits, which is also
why highstakes shows the biggest concept-level drop in finding 1.

## Method notes

**Validation slice.** A fixed 1092-row balanced draw from `dev_samples/` (up to 50 per
class from each of the 15 dev splits), identical for every fold and every arm — the
protocol's requirement is that the yardstick not move, which it does not. It is larger
than intended: `--dev-per-split` is applied per split, and an earlier draft of this
script's docstring described it as per concept. Nothing about the results changes, but a
1092-row validation set plus a 1188-row training fold exceeds the 24 GB card, so folds ran
~350 s each instead of the ~60 s a fully-staged fit would take.

**Eval subsampling.** Each of the 15 eval splits is balanced-subsampled to 100 rows,
this repo's own `--eval-max-samples` default, giving 1500 pooled rows. The full 6576-row
pool padded to the longest split's 1024 tokens is 72 GB of fp16 activations and fits
neither the host nor the card.

**No model is loaded anywhere in this analysis** — every activation comes from the
precomputed eval/dev blobs already on disk.
