# Cross-concept ceiling

_Generated 2026-08-23 21:09:13Z._

## What is being measured

**Ceiling** is the `ceiling_analysis` branch's definition: the best eval-set
performance this probe family (`linear_then_softmax` head on
`google/gemma-3-27b-it` layer 32) can reach *when trained on eval-distribution
data*, estimated by 5-fold cross-validation **inside the eval sets themselves** —
fit on the rows outside fold k, early-stop against a fixed reserved dev slice,
score fold k. Every eval row gets exactly one out-of-fold score, so a ceiling probe
differs from an ordinary probe only in its training data.

This run asks that of the three concepts **pooled**:

| arm | training pool |
| --- | --- |
| `within/<concept>` | CV inside one concept's own eval splits — the per-concept ceiling, and the baseline the cross numbers must be read against |
| `cross/native` | all three concepts at once, each row positive iff it is its own concept's positive class |
| `cross/aligned` | the same pool with `instructions` **inverted**, so all three positives are the *problematic* response |

`hu_ha` and `highstakes` call the problematic response positive (harmful,
high-stakes); `instructions` calls the desirable one positive
(`assistant_follows_the_instruction`). Pooling as-is therefore asks one direction to
put "harmful" and "correctly followed the instruction" on the same side, which is
not obviously the intended question — hence both orientations. The flip is not a
symmetry of the pooled task (it re-labels a third of the rows), so `aligned` is not
`1 - native`.

Each eval split is balanced-subsampled to 100 rows (this repo's own
`--eval-max-samples` default), giving 1500 pooled rows; the full 6576-row pool
padded to the longest split's 1024 tokens would be 72 GB of fp16 activations and
fits neither the box nor the card. Validation is a fixed 300-row balanced slice of
`dev_samples/`, 100 per concept, identical for every fold and every arm. Single
probes, never ensembles, seed 42. No model is loaded — every activation comes from
the precomputed blobs.

_No results yet._

## Reproducing

```bash
.venv_claude/bin/python scripts/cross_concept_ceiling.py
```
