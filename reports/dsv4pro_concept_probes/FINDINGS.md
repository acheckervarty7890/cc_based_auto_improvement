## Findings

See `reports/concept_probes_summary.md` for the four-generator comparison and
`reports/cross_concept_ceiling/REPORT.md` for the ceiling these numbers should be read
against (hu_ha 0.977, highstakes 0.963, instructions 0.946).

deepseek v4 pro is the **strongest generator for hu_ha (0.887)** — within 0.09 of that
concept's ceiling — and the **weakest for instructions by a wide margin (0.575, against a
0.946 ceiling and llama70b's 0.797)**. Its highstakes cut is also the weakest of the four
(0.823).

That spread is the point: the generator ranking inverts between concepts, so "which model
writes better synthetic training data" has no answer independent of the concept being
probed.
