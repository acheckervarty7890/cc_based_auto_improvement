# Every probe version of the hs and instructions runs, re-scored on full eval splits

All four probe versions (`probe_iter0..3`) of both arms of the **high-stakes**
(experiment9_cloud) and **instruction-following** (experiment_instruction_cloud_1) runs,
scored on their concept's full eval splits — 88 evaluations, 11 splits, no gemma-3-27b
forward passes. Produced by `scripts/eval_probe_versions.py`, summarized by
`scripts/eval_probe_versions_report.py`; raw rows in
`results_probe_versions/eval_rows.jsonl` and `probe_versions_eval.csv`.

`v0` is the probe trained on base data alone (never saw a red-team sample); `v1..v3` are
the successive retrains. `v0` is included because it is the only baseline the other three
mean anything against.

## The eval activations were on Kaggle for BOTH concepts

The high-stakes blobs are addressed by the `kaggle:` section both hs configs already
carry (`anku7890/{split}gemmaevalpt` → `{split}-gemmaeval.pt`).

The **instructions** blobs also exist, at `anku7890/{slug}-gemmaevalpt` →
`{split}-gemmaeval.pt` — the same templates the hu_ha splits use, `{slug}` because every
`eval_instructions` split stem contains an underscore and Kaggle slugs forbid those. All
seven download and validate.

Both instructions configs say otherwise. `gptoss120b_instructions_gemma27b_batch_target60.md`
and `nemotron_instructions_gemma27b_batch_target60.md` each carry a comment —

> NO `kaggle:` section — see the header. There are no published gemma-3-27b activation
> blobs for the eval_instructions splits, so arm 1 computes all 1302 rows locally

— and deliberately omit the section. That claim is **wrong**, and the reason is recorded
in `kaggle_activations.py`'s auth notes and in this repo's history: Kaggle's *listing*
endpoints (`dataset_list_files`, `dataset_list`) return 403 or silently omit datasets for
this account, while downloads by exact slug work fine. Absence from a listing was mistaken
for absence of the blob. The only reliable existence test is attempting the fetch.

Adding a `kaggle:` section to those two configs would let any future instructions run skip
the 1302-row local extraction; that edit is **not** made here, because changing a config
that a completed run is keyed to is a separate decision from measuring off it.

## Results

Mean AUROC over the concept's splits (unweighted, matching the `mean` row of the
pipeline's own comparison CSVs):

| concept | arm | v0 | v1 | v2 | v3 | v0→v3 | v2→v3 |
|---|---|---|---|---|---|---|---|
| high-stakes | gptoss120b | 0.8596 | 0.9004 | 0.9297 | **0.9308** | +0.071 | +0.001 |
| high-stakes | deepseekv4pro | 0.8595 | 0.8835 | **0.9141** | 0.9017 | +0.042 | −0.012 |
| instructions | gptoss | 0.4988 | 0.7537 | 0.7594 | **0.7976** | +0.299 | +0.038 |
| instructions | nemotron | 0.4988 | 0.8123 | **0.8338** | 0.7645 | +0.266 | −0.069 |

Three things read off this:

- **The instructions base probe is at chance.** 0.4988 mean AUROC, and `mm_substitution`
  at 0.347 is actively *anti*-correlated. On that concept red-teaming is not sharpening a
  working probe, it is building one: a single retrain moves it to 0.75–0.81. High-stakes
  starts from a competent 0.860 and the whole three-cycle gain is +0.04 to +0.07.
- **The last cycle is where the arms diverge, on both concepts.** One arm ends at its best,
  the other peaks at v2 and gives back 0.012 (hs/deepseek) or 0.069 (instructions/nemotron).
  This is the same shape as the hu_harm null documented in
  `docs/why_last_iteration_adds_nothing.md`, except here the v3 step is not uniformly
  ~0 — it is signed differently per arm, which a single-seed run cannot separate from
  seed noise. Treat the v2→v3 column as unresolved until it is refit across seeds the way
  the vintage sweep did.
- **Per-split, the gains are concentrated, not broad.** On hs almost all of v1's gain is
  `mt` (+0.14 at the first retrain); `toolace` is the hard split at every version and never
  clears 0.84. On instructions, `bbq_substitution` / `hc_contradiction` / `mm_substitution`
  go from chance to 0.93–0.98 at their best, while both `oig_*` splits and `anthropic_harmless_refusal`
  stay weak and swing erratically between versions.

Per-split AUROC grids are in
`results_probe_versions/{hs,instructions}_auroc_by_split.csv`.

## The numbers reproduce the original runs

The activations were downloaded rather than computed and the probes came off branches
untouched for weeks, so the run is only worth reading if it reproduces what the pipeline
measured in-flight. Compared against the four committed comparison CSVs, all 88 rows:

| concept / arm | rows | max \|Δ auroc\| | max \|Δ accuracy\| | max \|Δ tpr@1%\| |
|---|---|---|---|---|
| hs / gptoss120b | 16 | 2.8e−04 | 3.1e−03 | 7.3e−03 |
| hs / deepseekv4pro | 16 | 1.6e−04 | 3.6e−03 | 6.1e−03 |
| instructions / gptoss | 28 | 1.4e−03 | 5.0e−03 | 6.4e−17 |
| instructions / nemotron | 28 | 1.4e−03 | 0.0e+00 | 2.2e−01 |

Median |ΔAUROC| across all 88 rows is **6.4e−05**. The residual is float non-determinism
between one GPU and another: it reorders score ties, and AUROC depends on ordering.

The one outlier is `nemotron / anthropic_harmless_refusal / v3`, TPR 0.22 vs 0.0 — whose
AUROC agrees to 2e−04. `tpr_at_fixed_fpr_score` takes the ROC point whose FPR is *closest*
to 1%, and on a 200-row split that grid steps in whole percent, so a hair of score movement
jumps the argmin to the next plateau. One row in 88; every other TPR agrees to 7e−03.
**TPR@1%FPR on these split sizes is not a stable statistic** — read AUROC and accuracy.

## What makes the fetched blobs trustworthy

Beyond `_validate_blob`'s existing model/layer/row-count check, `--verify-tokens`
re-tokenizes one row per split through tuberlens' own `tokenize_inputs` and compares it to
the `input_ids` stored inside the blob. All eleven splits matched.

This is the only check that a published blob was computed under the *same message
transforms* the splits are loaded with here (`combine_consecutive_messages` +
`convert_tool_to_assistant`, both true in all four configs). Row count and probe metadata
match either way, so without it a blob computed under different transforms would sail
through validation and silently produce wrong scores. It needs the tokenizer only, not the
weights.

## Reproducing

The probes live on their experiments' branches, so the script's `EXPERIMENTS` table points
at detached worktrees:

```bash
git worktree add --detach /home/ubuntu/wt_exp9  origin/experiment9_cloud
git worktree add --detach /home/ubuntu/wt_instr origin/experiment_instruction_cloud_1

KAGGLE_CONFIG_DIR=$PWD/kaggle \
  .venv_claude/bin/python scripts/eval_probe_versions.py --verify-tokens
.venv_claude/bin/python scripts/eval_probe_versions_report.py
```

Results are checkpointed per (concept, split, arm, version), so an interrupted download or
eval resumes rather than repeating.

Two notes on cost. The blobs total ~25 GB and the `anthropic` split alone is 11.3 GB —
downloads dominate (~35 min of the ~75 min wall clock on the dev box at 5–15 MB/s; the rate
varies a lot by session, so measure rather than quote). And the loops are **split-major,
probe-minor** on purpose: `evaluate_probe` is probe-major, so scoring 8 probes that way
re-reads every blob 8 times and holds all four hs splits (~20 GB) at once, which is over
this box's cgroup ceiling. One split resident at a time peaks at 11.9 GB.
