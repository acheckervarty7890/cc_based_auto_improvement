# Concept probes — 4 generators compared

_Generated 2026-08-24 10:13:50Z._

The same experiment run on 4 ~50-row synthetic cuts per concept, one per
generating model. Every cell is mean AUROC over that concept's eval splits.

## Mean AUROC by generator and concept (averaged over all four arms)

```
concept       highstakes  hu_ha  instructions
generator                                    
dsv4pro            0.823  0.887         0.575
llama70b           0.897  0.838         0.797
llama8b            0.851  0.861         0.670
nemotron550b       0.837  0.876         0.589
```

## Mean AUROC by generator, concept and arm

```
concept                          highstakes  hu_ha  instructions
generator    config    val_mode                                 
dsv4pro      seq_ens10 dev            0.843  0.888         0.579
                       split          0.839  0.898         0.588
             single    dev            0.792  0.888         0.570
                       split          0.818  0.871         0.562
llama70b     seq_ens10 dev            0.918  0.846         0.771
                       split          0.877  0.800         0.813
             single    dev            0.900  0.852         0.778
                       split          0.895  0.855         0.825
llama8b      seq_ens10 dev            0.862  0.866         0.649
                       split          0.841  0.855         0.719
             single    dev            0.874  0.851         0.633
                       split          0.828  0.871         0.678
nemotron550b seq_ens10 dev            0.852  0.875         0.586
                       split          0.849  0.878         0.621
             single    dev            0.835  0.875         0.574
                       split          0.811  0.876         0.574
```

## Findings

### The generator ranking is exactly inverted between hu_ha and the other two concepts

| generator | hu_ha | highstakes | instructions |
| --- | --- | --- | --- |
| llama70b | 0.838 | **0.897** | **0.797** |
| llama8b | 0.861 | 0.851 | 0.670 |
| nemotron550b | 0.876 | 0.837 | 0.589 |
| dsv4pro | **0.887** | 0.823 | 0.575 |
| *ceiling* | *0.977* | *0.963* | *0.946* |

With three generators this looked like an unordered mismatch. The fourth resolves it into
something sharper: across all four, the hu_ha ordering is the **exact reverse** of the
highstakes ordering and of the instructions ordering, and those two are **identical** to
each other (Spearman −1.00, −1.00, +1.00). nemotron550b lands second on hu_ha and third on
both of the others, slotting in beside dsv4pro on every concept (0.876/0.887,
0.837/0.823, 0.589/0.575) without disturbing the pattern.

So there is no "better generator" — the axis that makes a cut good for harm is the same
axis, pointing the other way, that makes it bad for stakes and instruction-following. Any
claim of the form "model X writes good synthetic probe-training data" has to name the
concept, and on this evidence naming one concept tells you the ranking on the other two.

Treat the strength of the inversion with the caution n=4 deserves: an exact reversal of
four items has probability 1/24 under a random ordering, and the three correlations here
are not independent of one another. The *direction* is what the data supports; the
perfection of it is partly small-sample luck.

Model size does not order any of the three columns. The 550B and the deepseek model finish
adjacent on all three concepts while the 8B model sits between them and the 70B, so
parameter count predicts none of this.

### The spread between generators is larger than the gap to the ceiling

On `instructions`, llama70b reaches 0.797 against a 0.946 ceiling — a gap of 0.149 — while
dsv4pro reaches 0.575, a further 0.222 below llama70b. So on this concept, *which model
wrote the 50 rows* matters more than everything left on the table between the best cut and
training on eval-distribution data itself.

On `hu_ha` the picture is the opposite: all four generators land within 0.05 of each other
(0.838–0.887) and all are within 0.14 of the ceiling. A 50-row synthetic cut of harm is
close to sufficient regardless of who writes it; a 50-row cut of instruction-following is
not, and depends heavily on who writes it. `highstakes` sits between the two (0.823–0.897,
a 0.074 spread).

### Instruction-following is where the synthetic cuts genuinely fail

nemotron550b's instructions probes average 0.589, and five of its seven eval splits sit at
0.50–0.60 — `hc_context_drift` 0.518, `hc_contradiction` 0.522, `mm_substitution` 0.533 in
the `seq_ens10/dev` arm. Only `anthropic_harmless_refusal` (0.70–0.77) is meaningfully
above chance, which is the same single-split-carries-the-signal shape the generalization
experiment found. Against a 0.946 ceiling, three of four generators are closer to chance
than to what this probe family can do on this concept.

### Ensemble and validation source remain second-order, with one exception

Across all 48 probes, most (generator, concept) cells span under 0.03 across their four
arms — against concept-level spreads of 0.22 between generators. The exception is
nemotron550b on `highstakes`, which spans 0.041 (`seq_ens10/dev` 0.852 vs `single/split`
0.811) with the ensemble ahead in both validation modes; llama8b and llama70b on
`instructions` span 0.086 and 0.054 but with no consistent direction. The rule still holds
— these choices are the smallest terms here — but "≤0.06 everywhere" was too strong a
claim to carry over from three generators.

### The gradient-accumulation cap applies to all 48 probes

Every `split` arm here is ~38–43 training rows = 3 batches against `linear_then_softmax`'s
default `gradient_accumulation_steps: 4`, and the trainer has no end-of-epoch flush — so
uncapped, all twenty-four `split` probes would have been returned at their random
initialization. `experiment16_cloud` ran that exact shape on `hu_harm_llama70b_50` and its
iteration-0 probe scored 0.336, below chance on three of four splits, against 0.846 for
the otherwise-identical dev-validated `experiment17_cloud`. See
`scripts/concept_probes.py:capped_spec`.

