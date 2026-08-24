## Findings

See `reports/concept_probes_summary.md` for the four-generator comparison and
`reports/cross_concept_ceiling/REPORT.md` for the ceiling these numbers should be read
against (hu_ha 0.977, highstakes 0.963, instructions 0.946).

nvidia Nemotron 550B is the **second-strongest generator for hu_ha (0.876)** and third of
four on both `highstakes` (0.837) and `instructions` (0.589). It is the fourth generator
run through this matrix, and its value is mostly confirmatory: it lands adjacent to
dsv4pro on all three concepts and leaves the cross-concept ranking inversion intact — with
four generators, the hu_ha ordering is now the exact reverse of the highstakes and
instructions orderings, which are identical to each other.

Being the largest model in the rotation by parameter count buys nothing here. It finishes
behind the 70B model on two of three concepts and behind the 8B model on two of three.

Two details specific to this cut:

**Its `highstakes` arms spread wider than any other cell in the experiment** — 0.041
between `seq_ens10/dev` (0.852) and `single/split` (0.811), with the 10-member ensemble
ahead in *both* validation modes (0.852/0.849 vs 0.835/0.811). Everywhere else the
ensemble-vs-single and dev-vs-split choices have been ≤0.03 and directionless. It is still
a small effect next to the 0.074 spread between generators on this concept, but it is the
one cell where the ensemble looks like it is doing something rather than nothing.

**Its `instructions` probes are close to chance on five of seven splits** —
`hc_context_drift` 0.518, `hc_contradiction` 0.522, `mm_substitution` 0.533,
`oig_omission` 0.546, `oig_context_drift` 0.598 in the `seq_ens10/dev` arm. Only
`anthropic_harmless_refusal` (0.70–0.77) and `bbq_substitution` (0.61–0.69) carry signal.
Against a 0.946 ceiling this cut teaches the probe very little of the concept.

Its `hu_ha` cell is the tightest in the whole experiment: 0.875–0.878 across all four
arms, a spread of 0.003. Per split it follows the usual shape — `daily_dilemmas` 0.98 and
`ai_dilemmas` 0.96 near-saturated, `ant_hh` 0.72–0.76 holding the mean down.

### Cost

2 h 30 m end to end: 352 s to extract the 150 conversations in one gemma-3-27b load, 51
min of fits, the rest eval. As with every other generator, `highstakes` is 93% of the fit
time (2859 s of 3076 s) and its `seq_ens10/dev` arm alone is 2467 s — ten sequential
members each scoring the 1908-row / 21 GB dev set every epoch, which does not fit the
24 GB card and so cannot be staged by `_to_device_for_fit`.
