## Findings

See `reports/concept_probes_summary.md` for the four-generator comparison and
`reports/cross_concept_ceiling/REPORT.md` for the ceiling these numbers should be read
against (hu_ha 0.977, highstakes 0.963, instructions 0.946).

llama-3.3-70B is the **strongest generator for highstakes (0.897) and instructions
(0.797)** and the weakest of the four for hu_ha (0.838). Its instructions cut is the
only one of the nine (generator x concept) cells that gets an instructions probe near its
ceiling — 0.797 against 0.946, where llama8b reaches 0.670 and dsv4pro 0.575.

Within this generator the four arms are tightly clustered (0.800–0.918); as everywhere in
this experiment, the choice of ensemble and of validation source moves the mean far less
than the choice of training data does.
