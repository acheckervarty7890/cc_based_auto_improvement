# Do the instruction-following probes transfer to the other two concepts?
_Updated 2026-08-16T23:07:05+00:00_

**What this measures.** Every probe named in `SUMMARY.md` and `REDTEAM_ONLY.md` — 90 of them, five training sets x two arms x ten seeds — scored on the **other two concepts'** eval sets. The probes separate `assistant_follows_the_instruction` from `assistant_does_not_follow_the_instruction`; the question is whether the direction they learned also orders *high-stakes* and *harmful_to_human* labels.

**Why it costs no forward pass.** All three concepts' runs extract from the same frozen `google/gemma-3-27b-it` at layer 32 under the same two loader transforms, so a conversation's activation is the same tensor whichever concept's run computed it — only the probe head differs. The eight target blobs were pulled from Kaggle (`scripts/fetch_crossconcept_eval.py`) and each was validated against the probe's model name, layer and the split's row count before use.

**How to read the numbers.** AUROC is against each target concept's own positive class (`high-stakes`, `harmful_to_human`), on the pipeline scale. **0.5 is chance, and below 0.5 is signal pointing the other way, not failure** — a probe reading 0.19 against harm separates the harm labels as well as one reading 0.81, having learned a direction whose *follows-the-instruction* end is the harmful end. Nothing is thresholded: `probe.threshold` was calibrated for a different concept, so accuracy would only measure how two base rates happen to line up.

**`base_only` is the control.** It trains on the 50 base samples and no red-team rows, scores chance on its *own* concept (0.4972 ± 0.0349), and is fitted once because it does not depend on the arm.

## harmful_to_human (`eval_dataset_hu_ha/`) — AUROC vs `harmful_to_human`

| arm | training set | ai_dilemmas | ant_hh | balanced_refusal | daily_dilemmas | mean |
|---|---|---|---|---|---|---|
| shared | base only (no red-team) | 0.501 ± 0.032 | 0.450 ± 0.073 | 0.421 ± 0.215 | 0.527 ± 0.039 | **0.475 ± 0.063** |
| gptoss120b | v2 + base | 0.190 ± 0.044 | 0.295 ± 0.026 | 0.212 ± 0.067 | 0.050 ± 0.013 | **0.187 ± 0.023** |
| gptoss120b | v3 + base | 0.243 ± 0.062 | 0.287 ± 0.028 | 0.258 ± 0.055 | 0.074 ± 0.025 | **0.216 ± 0.026** |
| gptoss120b | v2 alone (no base) | 0.220 ± 0.071 | 0.308 ± 0.022 | 0.241 ± 0.043 | 0.055 ± 0.037 | **0.206 ± 0.033** |
| gptoss120b | v3only alone (no base) | 0.410 ± 0.069 | 0.445 ± 0.063 | 0.252 ± 0.177 | 0.318 ± 0.097 | **0.357 ± 0.092** |
| nemotron | v2 + base | 0.233 ± 0.082 | 0.360 ± 0.025 | 0.239 ± 0.114 | 0.073 ± 0.056 | **0.226 ± 0.064** |
| nemotron | v3 + base | 0.260 ± 0.099 | 0.409 ± 0.040 | 0.132 ± 0.082 | 0.058 ± 0.041 | **0.215 ± 0.038** |
| nemotron | v2 alone (no base) | 0.222 ± 0.050 | 0.367 ± 0.037 | 0.203 ± 0.082 | 0.060 ± 0.045 | **0.213 ± 0.044** |
| nemotron | v3only alone (no base) | 0.411 ± 0.042 | 0.493 ± 0.022 | 0.406 ± 0.081 | 0.374 ± 0.042 | **0.421 ± 0.022** |

## high-stakes (`eval_datasets/`) — AUROC vs `high-stakes`

| arm | training set | anthropic | mt | mts | toolace | mean |
|---|---|---|---|---|---|---|
| shared | base only (no red-team) | 0.523 ± 0.112 | 0.527 ± 0.166 | 0.519 ± 0.141 | 0.480 ± 0.092 | **0.512 ± 0.078** |
| gptoss120b | v2 + base | 0.278 ± 0.028 | 0.494 ± 0.044 | 0.532 ± 0.023 | 0.423 ± 0.018 | **0.432 ± 0.015** |
| gptoss120b | v3 + base | 0.277 ± 0.020 | 0.491 ± 0.064 | 0.488 ± 0.031 | 0.415 ± 0.013 | **0.418 ± 0.021** |
| gptoss120b | v2 alone (no base) | 0.238 ± 0.023 | 0.558 ± 0.035 | 0.539 ± 0.038 | 0.433 ± 0.014 | **0.442 ± 0.014** |
| gptoss120b | v3only alone (no base) | 0.444 ± 0.047 | 0.625 ± 0.025 | 0.416 ± 0.029 | 0.523 ± 0.028 | **0.502 ± 0.014** |
| nemotron | v2 + base | 0.310 ± 0.021 | 0.558 ± 0.039 | 0.543 ± 0.026 | 0.441 ± 0.010 | **0.463 ± 0.014** |
| nemotron | v3 + base | 0.430 ± 0.039 | 0.400 ± 0.064 | 0.548 ± 0.010 | 0.456 ± 0.009 | **0.458 ± 0.015** |
| nemotron | v2 alone (no base) | 0.322 ± 0.033 | 0.649 ± 0.043 | 0.530 ± 0.014 | 0.446 ± 0.022 | **0.487 ± 0.021** |
| nemotron | v3only alone (no base) | 0.476 ± 0.045 | 0.396 ± 0.059 | 0.464 ± 0.049 | 0.514 ± 0.022 | **0.462 ± 0.018** |

## Findings

**The harm transfer is strong, and inverted.** Every red-team-trained probe lands at 0.19-0.23 mean AUROC against `harmful_to_human`, 5-20 sd from chance, and on `eval_daily_dilemmas` it reaches 0.050-0.074 — near-perfect separation with the sign flipped. `base_only` sits at 0.475 ± 0.063, so the transfer is carried entirely by the red-team data, not by the base corpus or the architecture.

**The direction is semantically sensible.** The negative class in `eval_instructions` is failure-to-follow, whose split names spell out the modes: refusal, omission, contradiction, drift. Three of the four hu_ha splits are *paired* — the same user prompt with a harmful and a non-harmful assistant reply — and the non-harmful member is typically the refusal. So *follows the instruction* and *harmful* pick out the same conversations, and a probe trained on the first orders the second almost perfectly.

**The high-stakes transfer is weak in aggregate and lives in one split.** Mean AUROC is 0.42-0.50 across conditions, but that average hides opposite behaviour: `anthropic` inverts strongly (0.238-0.322 for the v2 conditions) while `mt` and `mts` sit at or slightly above chance and `toolace` just below. The pattern follows the refusal axis rather than the stakes axis — `anthropic` is the one hs split built from chosen/rejected assistant replies, the same comply-vs-refuse contrast the hu_ha splits carry, whereas medical-transcription (`mt`, `mts`) and tool-use system prompts (`toolace`) contain no such contrast. So what transfers is not "stakes" at all; it is the same refusal direction, showing up wherever a split happens to contain it.

**The v3only probes transfer least, on both concepts.** 0.357 / 0.421 against harm (vs ~0.21 for every other red-team condition) and 0.502 / 0.462 — flat chance — against high-stakes; nemotron's reaches 0.493 on `eval_ant_hh`, exactly chance. This is an independent signature of what `REDTEAM_ONLY.md` found by training on those rows alone: the pairs that first appear at iteration 3 encode something narrower than the earlier vintages. Narrow enough, it turns out, to lose the refusal axis that drives the whole transfer.

**Adding the base data changes nothing here either.** `v2 + base` and `v2 alone` differ by 0.019 (gptoss120b) and 0.013 (nemotron) on harm — the same non-effect the in-concept sweep measured, seen on a corpus neither probe was trained for.

## Files

- `crossconcept_progress.jsonl` — one row per (probe, split), append-only
- `crossconcept_auroc.csv` — the same, flat
- `crossconcept_fits.jsonl` — the 90 refits and their verification status
- `sweep_probes/` — the 90 probes themselves (~13 KB each), shared with `redteam_only_fits.py`

## Reproducing

```bash
KAGGLE_CONFIG_DIR=$PWD/kaggle .venv_claude/bin/python scripts/fetch_crossconcept_eval.py
AGENTIC_FAST_ACTS=1 .venv_claude/bin/python scripts/crossconcept_eval.py --stage fit     # skipped entirely if sweep_probes/ is populated
.venv_claude/bin/python scripts/crossconcept_eval.py --stage score
.venv_claude/bin/python scripts/crossconcept_summary_md.py
```
