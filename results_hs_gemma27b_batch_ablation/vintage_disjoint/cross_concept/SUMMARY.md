# High-stakes vintage probes scored OFF their own concept

Every row is one of the 80 committed `vintage/fits/*.pt` checkpoints — the probes behind `vintage/SUMMARY.md`, refitted nowhere — scored on an eval set it was never trained for. AUROC is against **that split's own positive class**, so 0.5 means the high-stakes score says nothing about the other concept and a value below 0.5 is a real anti-correlation, reported unflipped.

AUROC scale: `pipeline` (bf16 sigmoid then sklearn, as the pipeline reports it). The rank-faithful figures are in the CSVs alongside.

## Read-out

- **hu_harm / deepseekv4pro**: v0 0.5864 → v1 0.5233 → v2-only 0.6015 → v3-only 0.5704.
  - v1 moves -0.0631 against v0 (5.1σ) — the red-team data changed what this probe does off-concept.
  - strongest single split: `eval_ant_hh` at v0, 0.8387 ± 0.0161 (above chance by 0.3387).
  - same fits ON high-stakes, for reference: v0 0.8680 → v1 0.8816 → v2 0.9147 → v3 0.8912.
- **hu_harm / gptoss120b**: v0 0.5864 → v1 0.5637 → v2-only 0.5900 → v3-only 0.5739.
  - strongest single split: `eval_ant_hh` at v0, 0.8387 ± 0.0161 (above chance by 0.3387).
  - same fits ON high-stakes, for reference: v0 0.8680 → v1 0.9018 → v2 0.9281 → v3 0.9326.
- **instructions / deepseekv4pro**: v0 0.4672 → v1 0.5519 → v2-only 0.4467 → v3-only 0.4564.
  - v1 moves +0.0847 against v0 (2.2σ) — the red-team data changed what this probe does off-concept.
  - strongest single split: `bbq_substitution` at v1, 0.7557 ± 0.0156 (above chance by 0.2557).
  - same fits ON high-stakes, for reference: v0 0.8680 → v1 0.8816 → v2 0.9147 → v3 0.8912.
- **instructions / gptoss120b**: v0 0.4672 → v1 0.5173 → v2-only 0.4374 → v3-only 0.5095.
  - strongest single split: `anthropic_harmless_refusal` at v2-only, 0.2766 ± 0.0414 (below chance by 0.2234).
  - same fits ON high-stakes, for reference: v0 0.8680 → v1 0.9018 → v2 0.9281 → v3 0.9326.

## hu_harm — harmful_to_human vs not_harmful_to_human

### deepseekv4pro

| vintage | rt rows | seeds | eval_ai_dilemmas | eval_ant_hh | eval_balanced_refusal | eval_daily_dilemmas | mean |
|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.5128 ± 0.0154 | 0.8387 ± 0.0161 | 0.4460 ± 0.0156 | 0.5482 ± 0.0319 | 0.5864 ± 0.0117 |
| v1 | 294 | 10 | 0.5000 ± 0.0044 | 0.8057 ± 0.0197 | 0.4112 ± 0.0161 | 0.3763 ± 0.0097 | 0.5233 ± 0.0042 |
| v2-only | 262 | 10 | 0.5721 ± 0.0355 | 0.8162 ± 0.0096 | 0.4476 ± 0.0110 | 0.5701 ± 0.0255 | 0.6015 ± 0.0155 |
| v3-only | 174 | 10 | 0.5310 ± 0.0079 | 0.7977 ± 0.0081 | 0.4383 ± 0.0118 | 0.5145 ± 0.0100 | 0.5704 ± 0.0060 |

### gptoss120b

| vintage | rt rows | seeds | eval_ai_dilemmas | eval_ant_hh | eval_balanced_refusal | eval_daily_dilemmas | mean |
|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.5128 ± 0.0154 | 0.8387 ± 0.0161 | 0.4460 ± 0.0156 | 0.5482 ± 0.0319 | 0.5864 ± 0.0117 |
| v1 | 228 | 10 | 0.5020 ± 0.0044 | 0.8345 ± 0.0215 | 0.4212 ± 0.0095 | 0.4971 ± 0.0117 | 0.5637 ± 0.0063 |
| v2-only | 214 | 10 | 0.5318 ± 0.0141 | 0.8233 ± 0.0094 | 0.4366 ± 0.0115 | 0.5685 ± 0.0101 | 0.5900 ± 0.0048 |
| v3-only | 140 | 10 | 0.5380 ± 0.0100 | 0.8275 ± 0.0111 | 0.4184 ± 0.0032 | 0.5116 ± 0.0044 | 0.5739 ± 0.0034 |

## instructions — assistant_follows_the_instruction vs assistant_does_not_follow_the_instruction

### deepseekv4pro

| vintage | rt rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.4451 ± 0.1852 | 0.4579 ± 0.0204 | 0.4396 ± 0.0294 | 0.4462 ± 0.0247 | 0.4393 ± 0.0419 | 0.5241 ± 0.0064 | 0.5180 ± 0.0078 | 0.4672 ± 0.0360 |
| v1 | 294 | 10 | 0.6273 ± 0.0603 | 0.7557 ± 0.0156 | 0.5027 ± 0.0041 | 0.5015 ± 0.0075 | 0.3950 ± 0.0155 | 0.5182 ± 0.0170 | 0.5631 ± 0.0043 | 0.5519 ± 0.0123 |
| v2-only | 262 | 10 | 0.2535 ± 0.0446 | 0.4691 ± 0.0095 | 0.4685 ± 0.0082 | 0.4742 ± 0.0084 | 0.4631 ± 0.0246 | 0.5162 ± 0.0057 | 0.4825 ± 0.0074 | 0.4467 ± 0.0083 |
| v3-only | 174 | 10 | 0.4130 ± 0.0613 | 0.4607 ± 0.0126 | 0.4851 ± 0.0041 | 0.4776 ± 0.0055 | 0.3301 ± 0.0135 | 0.5068 ± 0.0155 | 0.5215 ± 0.0057 | 0.4564 ± 0.0095 |

### gptoss120b

| vintage | rt rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.4451 ± 0.1852 | 0.4579 ± 0.0204 | 0.4396 ± 0.0294 | 0.4462 ± 0.0247 | 0.4393 ± 0.0419 | 0.5241 ± 0.0064 | 0.5180 ± 0.0078 | 0.4672 ± 0.0360 |
| v1 | 228 | 10 | 0.5657 ± 0.0530 | 0.5284 ± 0.0084 | 0.4858 ± 0.0070 | 0.4855 ± 0.0065 | 0.4542 ± 0.0111 | 0.5711 ± 0.0097 | 0.5305 ± 0.0081 | 0.5173 ± 0.0098 |
| v2-only | 214 | 10 | 0.2766 ± 0.0414 | 0.4171 ± 0.0153 | 0.4559 ± 0.0154 | 0.4656 ± 0.0107 | 0.4275 ± 0.0201 | 0.5237 ± 0.0137 | 0.4957 ± 0.0037 | 0.4374 ± 0.0103 |
| v3-only | 140 | 10 | 0.5146 ± 0.0542 | 0.5013 ± 0.0095 | 0.4929 ± 0.0031 | 0.4944 ± 0.0034 | 0.4711 ± 0.0117 | 0.5774 ± 0.0061 | 0.5145 ± 0.0055 | 0.5095 ± 0.0105 |

