## Findings

### Mean AUROC by arm

| arm | hu_ha | highstakes | instructions |
| --- | --- | --- | --- |
| `single` / val=`dev` | 0.851 | **0.874** | 0.633 |
| `single` / val=`split` | **0.871** | 0.828 | 0.678 |
| `seq_ens10` / val=`dev` | 0.866 | 0.862 | 0.649 |
| `seq_ens10` / val=`split` | 0.855 | 0.841 | **0.719** |
| **concept mean** | **0.861** | **0.851** | **0.670** |

### 1. Fifty synthetic rows are enough for harm and stakes, and not for instruction-following

hu_ha (0.861) and highstakes (0.851) land within 0.01 of each other; instructions is
0.18 behind at 0.670. The gap is not an artifact of split count (7 vs 4) — it shows up
split by split. On highstakes, `anthropic_hh_balanced` (2984 rows, the largest split in
the project) runs 0.82–0.91; the same split never left 0.46 in the generalization
experiment, so this is the llama-8b cut teaching the actual concept rather than one
small split carrying a mean.

On instructions the weak splits are the ones whose negative class is a *drift* or an
*omission* — `oig_context_drift`, `oig_omission`, `hc_context_drift` sit at 0.54–0.64 in
every arm — while `bbq_substitution` (0.71–0.87) and `anthropic_harmless_refusal`
(0.72–0.78) do much better. A 50-row synthetic cut can show what a wrong answer or a
refusal looks like; it apparently does not show what quietly failing to use the provided
source looks like.

### 2. The 40-row `split` arms train correctly only because the accumulation is capped

`linear_then_softmax` defaults to `batch_size: 16, gradient_accumulation_steps: 4`, and
the trainer steps only on `(batch_idx + 1) % accumulation == 0` with no end-of-epoch
flush (`pytorch_classifiers.py:299-327`, unchanged since the tuberlens rename). Every
`split` arm here is 39–40 training rows = 3 batches/epoch, so uncapped, `optimizer.step()`
would never fire and all six probes would come back at their random initialization.

This is not hypothetical. `experiment16_cloud` ran the *same* shape — hu_harm,
gemma-3-27b L32, `linear_then_softmax`, `ensemble_size: 10`, base data
`data/hu_harm_llama70b_50.jsonl`, 0.2 `test_size` slice → 40 train / 10 val — and its
iteration-0 probe scored **0.336** mean AUROC, below chance on three of four splits
(0.180 / 0.232 / 0.453 / 0.477). `experiment17_cloud` is byte-identical except for
`validation.dev_data: dev_samples/hu_ha`, which puts all 50 rows in training = 4 batches,
and its iteration-0 probe scored **0.846**. The 0.51 gap between those two runs is the
bug, not the validation source. A systematically inverted probe is the signature of a
random direction; a merely weak probe sits at 0.5.

Capped, the equivalent arms here reach 0.678–0.871. Any run in this repo whose
iteration-0 training set was under 64 rows has the same defect.

### 3. `dev` vs `split` has no consistent winner, and on instructions `split` wins

| concept | dev | split | Δ |
| --- | --- | --- | --- |
| hu_ha | 0.858 | 0.863 | +0.005 |
| highstakes | 0.868 | 0.834 | −0.034 |
| instructions | 0.641 | 0.698 | +0.057 |

The two arms differ in two ways at once: the `dev` arm trains on ~10 more rows *and*
early-stops against hundreds of real conversations, while the `split` arm early-stops
against 10 synthetic ones (a set so small it saturates at AUROC 1.000 immediately, making
its stopping point close to arbitrary). The expectation is that `dev` wins. It does on
highstakes and it loses on instructions by more than it wins anywhere — which reads as the
llama-8b instructions cut and the real `dev_samples/instructions` set disagreeing about
the concept, so stopping on the real set stops at the wrong epoch for this training data.

### 4. The ensemble is a wash except in one cell

`seq_ens10` moves the mean by ≤ 0.02 against a single probe in five of six
(concept × validation) cells — expected when all ten members fit the same activations and
differ only in seed. The exception is instructions/`split` (0.678 → 0.719), which is also
the noisiest cell in the matrix (`hc_contradiction` alone swings 0.531 → 0.875 across
arms), so it is more plausibly variance than an ensemble effect.
