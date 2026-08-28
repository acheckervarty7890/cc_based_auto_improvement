# Unsteered nemotron generation vs. the nemotron red-team loops — instruction-following

## What was run

`scripts/generate_instructions_dataset.py` (the unsteered one-shot generator: no memo, no
direction, no probe verdict, no judge) with `--model nvidia/nemotron-3-ultra-550b-a55b`,
200 rows, 100 per label → `data/instructions_nemotron_200.jsonl`.

Two probes were then fit and scored by `scripts/run_nemotron_instructions_arm.py`, pinned to
`scripts/run_control_arm.py`'s settings so nothing but the training data differs:
gemma-3-27b-it layer 32, `linear_then_softmax`, ensemble size 1, seed 42,
`combine_consecutive_messages` + `convert_tool_to_assistant` on, `dev_samples/instructions`
(436 rows) as the fit's sole validation set, full `eval_sets/instructions` splits.

* **nemotron200** — base 50 + all 200 nemotron rows (250 training rows)
* **nemotron150** — base 50 + the first 75 rows per label (200 rows), budget-matched to the
  llama-3.3-70b control arm

Every comparator below trains from the same `data/instructions_llama70b_50.jsonl` base and
scores the same seven full eval splits (row counts verified identical across branches), so
the mean-AUROC column is directly comparable.

## Headline

| arm | generator | steering | rows added | mean eval AUROC |
|---|---|---|---:|---:|
| base probe (50 rows only) | — | none | 0 | 0.7779 |
| control | llama-3.3-70b | none | 150 | 0.7375 |
| **nemotron150** | nemotron-3-ultra-550b | none | 150 | **0.7869** |
| **nemotron200** | nemotron-3-ultra-550b | none | 200 | **0.8070** |
| ΔAUROC loop, iter5 | llama-3.3-70b | judge memo + probe arbitration | 80 | 0.8200 |
| red team, 5 nemotron runs, final iter | nemotron-3-ultra-550b | attacker + judge + preprocessing | 710–928 | 0.7837 mean / 0.7643 median |
| red team, 5 nemotron runs, best iter | nemotron-3-ultra-550b | attacker + judge + preprocessing | 430–812 | 0.8077 mean / 0.8058 median |

Per-split numbers for every row: `instructions_comparison.csv`.

## Readings

1. **The generator identity, not the steering, carries most of the movement here.** Holding
   everything else fixed and swapping only the generator at the same 150-row budget moves the
   mean from 0.7375 to 0.7869 (**+0.049**). The llama control arm *damages* the base probe
   (−0.040); the nemotron control arm *helps* it (+0.009 at 150 rows, +0.029 at 200).

2. **200 unsteered nemotron rows match the red-team loops' best iteration.** 0.8070 sits on
   top of the five runs' best-iteration median (0.8058) and above their final-iteration mean
   (0.7837), for ~20 API calls against ~2500 attacker conversations + a gpt-5.1 judge + a
   gpt-5.1 preprocessing pass + 5 gemma-27b retrains *per arm*. Only one red-team arm
   (`ens10dev`, 0.8595) clearly beats it, and it does so with 812 added rows and a 10-member
   ensemble.

3. **The red-team loops are not monotone.** Three of the five peak mid-run and end lower
   (`batch60` 0.8337 → 0.7643; `xmemodesc` 0.7815 → 0.7371; `ens10` 0.7578 → 0.7518). The
   later rounds add hundreds of adversarially-shaped rows that cost eval AUROC. The unsteered
   arm has no such dynamics to manage — and its 150 → 200 step still gained (+0.020).

4. **`oig_omission` resists everything.** No arm anywhere beats the base probe's 0.7764; the
   best generated arm is `rt_ens10` iter4 at 0.7725, and nemotron200 reaches 0.7081. This
   reproduces the known result that the split's ceiling (~0.6725 out-of-fold on eval-
   distribution data, `experiment_instruction_cloud_3`) sits *below* what 50 base rows score.

5. **Where nemotron200's gain actually comes from.** `anthropic_harmless_refusal`
   0.5344 → 0.7649 (the base probe is at chance there) and `oig_context_drift`
   0.7142 → 0.7867. It gives back `hc_contradiction` (0.9089 → 0.7974) and `oig_omission`
   (0.7764 → 0.7081) — the same two splits the llama control arm loses, but by less.

## Caveats

* The four ensemble red-team arms (`ens10`, `ens10dev`, `xmemodesc`, `xmemocat`) use
  `ensemble_size: 10`; `batch60` and both arms here use a single probe. Their iter0 baseline
  is 0.7714, not 0.7779.
* Single-probe dev AUROC moves ~±0.005 between near-identical training sets. The +0.049
  generator delta is well outside that; the +0.020 for 150 → 200 rows is not a single-seed
  certainty.
* The generated set carries the label noise unsteered generation always carries — e.g. one
  `assistant_does_not_follow_the_instruction` row whose only violation is a four-word preamble
  on an "output only the maximum value" instruction. The control arm carries the same kind of
  noise by construction, so the comparison is fair; the absolute numbers are not a ceiling.
* `dev_auroc` and eval move in opposite directions for nemotron200 (dev 0.7664 vs the base
  probe's 0.7573 fit-time baseline, eval 0.8070). The dev set is 436 rows drawn from the same
  seven distributions; treat eval as the arbiter.

## Reproduce

```bash
set -a; . ./.env; set +a
export OPENROUTER_TIMEOUT_S=600
.venv_claude/bin/python scripts/generate_instructions_dataset.py \
    --model nvidia/nemotron-3-ultra-550b-a55b \
    --out data/instructions_nemotron_200.jsonl --n-per-label 100 \
    --batch-size 10 --max-tokens 8192
.venv_claude/bin/python scripts/run_nemotron_instructions_arm.py
```
