# High-stakes vintage probes scored OFF their own concept

Every row is one of the 80 committed `vintage/fits/*.pt` checkpoints — the probes behind `vintage/SUMMARY.md`, refitted nowhere — scored on an eval set it was never trained for. AUROC is against **that split's own positive class**, so 0.5 means the high-stakes score says nothing about the other concept and a value below 0.5 is a real anti-correlation, reported unflipped.

AUROC scale: `pipeline` (bf16 sigmoid then sklearn, as the pipeline reports it). The rank-faithful figures are in the CSVs alongside.

## Read-out

- **hu_harm / deepseekv4pro**: v0 0.5864 → v1 0.5233 → v2 0.5638 → v3 0.5715.
  - v1 moves -0.0631 against v0 (5.1σ) — the red-team data changed what this probe does off-concept.
  - strongest single split: `eval_ant_hh` at v2, 0.8505 ± 0.0112 (above chance by 0.3505).
  - same fits ON high-stakes, for reference: v0 0.8680 → v1 0.8816 → v2 0.9147 → v3 0.8912.
- **hu_harm / gptoss120b**: v0 0.5864 → v1 0.5637 → v2 0.5730 → v3 0.5708.
  - strongest single split: `eval_ant_hh` at v3, 0.8489 ± 0.0053 (above chance by 0.3489).
  - same fits ON high-stakes, for reference: v0 0.8680 → v1 0.9018 → v2 0.9281 → v3 0.9326.
- **instructions / deepseekv4pro**: v0 0.4672 → v1 0.5519 → v2 0.4810 → v3 0.4603.
  - v1 moves +0.0847 against v0 (2.2σ) — the red-team data changed what this probe does off-concept.
  - strongest single split: `bbq_substitution` at v1, 0.7557 ± 0.0156 (above chance by 0.2557).
  - same fits ON high-stakes, for reference: v0 0.8680 → v1 0.8816 → v2 0.9147 → v3 0.8912.
- **instructions / gptoss120b**: v0 0.4672 → v1 0.5173 → v2 0.4857 → v3 0.4885.
  - strongest single split: `oig_context_drift` at v1, 0.5711 ± 0.0097 (above chance by 0.0711).
  - same fits ON high-stakes, for reference: v0 0.8680 → v1 0.9018 → v2 0.9281 → v3 0.9326.

## hu_harm — harmful_to_human vs not_harmful_to_human

### deepseekv4pro

| vintage | rt rows | seeds | eval_ai_dilemmas | eval_ant_hh | eval_balanced_refusal | eval_daily_dilemmas | mean |
|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.5128 ± 0.0154 | 0.8387 ± 0.0161 | 0.4460 ± 0.0156 | 0.5482 ± 0.0319 | 0.5864 ± 0.0117 |
| v1 | 294 | 10 | 0.5000 ± 0.0044 | 0.8057 ± 0.0197 | 0.4112 ± 0.0161 | 0.3763 ± 0.0097 | 0.5233 ± 0.0042 |
| v2 | 542 | 10 | 0.4967 ± 0.0028 | 0.8505 ± 0.0112 | 0.4168 ± 0.0072 | 0.4910 ± 0.0040 | 0.5638 ± 0.0042 |
| v3 | 716 | 10 | 0.5054 ± 0.0056 | 0.8332 ± 0.0107 | 0.4494 ± 0.0174 | 0.4981 ± 0.0102 | 0.5715 ± 0.0062 |

### gptoss120b

| vintage | rt rows | seeds | eval_ai_dilemmas | eval_ant_hh | eval_balanced_refusal | eval_daily_dilemmas | mean |
|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.5128 ± 0.0154 | 0.8387 ± 0.0161 | 0.4460 ± 0.0156 | 0.5482 ± 0.0319 | 0.5864 ± 0.0117 |
| v1 | 228 | 10 | 0.5020 ± 0.0044 | 0.8345 ± 0.0215 | 0.4212 ± 0.0095 | 0.4971 ± 0.0117 | 0.5637 ± 0.0063 |
| v2 | 422 | 10 | 0.5112 ± 0.0074 | 0.8399 ± 0.0094 | 0.4298 ± 0.0089 | 0.5112 ± 0.0101 | 0.5730 ± 0.0056 |
| v3 | 562 | 10 | 0.5052 ± 0.0081 | 0.8489 ± 0.0053 | 0.4299 ± 0.0071 | 0.4995 ± 0.0052 | 0.5708 ± 0.0033 |

## instructions — assistant_follows_the_instruction vs assistant_does_not_follow_the_instruction

### deepseekv4pro

| vintage | rt rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.4451 ± 0.1852 | 0.4579 ± 0.0204 | 0.4396 ± 0.0294 | 0.4462 ± 0.0247 | 0.4393 ± 0.0419 | 0.5241 ± 0.0064 | 0.5180 ± 0.0078 | 0.4672 ± 0.0360 |
| v1 | 294 | 10 | 0.6273 ± 0.0603 | 0.7557 ± 0.0156 | 0.5027 ± 0.0041 | 0.5015 ± 0.0075 | 0.3950 ± 0.0155 | 0.5182 ± 0.0170 | 0.5631 ± 0.0043 | 0.5519 ± 0.0123 |
| v2 | 542 | 10 | 0.3918 ± 0.0234 | 0.5271 ± 0.0027 | 0.4957 ± 0.0012 | 0.4948 ± 0.0026 | 0.4292 ± 0.0085 | 0.5188 ± 0.0082 | 0.5094 ± 0.0062 | 0.4810 ± 0.0043 |
| v3 | 716 | 10 | 0.4039 ± 0.0553 | 0.5010 ± 0.0220 | 0.4951 ± 0.0019 | 0.4892 ± 0.0062 | 0.3245 ± 0.0212 | 0.4984 ± 0.0149 | 0.5100 ± 0.0047 | 0.4603 ± 0.0084 |

### gptoss120b

| vintage | rt rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.4451 ± 0.1852 | 0.4579 ± 0.0204 | 0.4396 ± 0.0294 | 0.4462 ± 0.0247 | 0.4393 ± 0.0419 | 0.5241 ± 0.0064 | 0.5180 ± 0.0078 | 0.4672 ± 0.0360 |
| v1 | 228 | 10 | 0.5657 ± 0.0530 | 0.5284 ± 0.0084 | 0.4858 ± 0.0070 | 0.4855 ± 0.0065 | 0.4542 ± 0.0111 | 0.5711 ± 0.0097 | 0.5305 ± 0.0081 | 0.5173 ± 0.0098 |
| v2 | 422 | 10 | 0.4535 ± 0.0253 | 0.4862 ± 0.0088 | 0.4875 ± 0.0037 | 0.4853 ± 0.0051 | 0.4505 ± 0.0111 | 0.5311 ± 0.0059 | 0.5058 ± 0.0037 | 0.4857 ± 0.0059 |
| v3 | 562 | 10 | 0.4376 ± 0.0261 | 0.5040 ± 0.0097 | 0.4957 ± 0.0015 | 0.4972 ± 0.0020 | 0.4484 ± 0.0137 | 0.5292 ± 0.0087 | 0.5073 ± 0.0060 | 0.4885 ± 0.0038 |

