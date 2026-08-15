# Red-team vintage sweep — instruction-following concept
_Updated 2026-08-15T15:38:57+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data (`data/instructions_llama70b_50.jsonl`) plus one *vintage* of iteration-3 red-team pairs, scored on the seven `eval_instructions` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Setup.** Probe `google/gemma-3-27b-it` L32, labels `assistant_follows_the_instruction` / `assistant_does_not_follow_the_instruction`. Arms are the two attackers of `run_gemma27b_instructions_attackers.sh`: `gptoss120b` (`openai/gpt-oss-120b`) and `nemotron` (`nvidia/nemotron-3-ultra-550b-a55b`). All activations — base split, per-conversation red-team, and the seven eval splits — were pulled from Kaggle (`anku7890/instructions-gemma27b-*`, `anku7890/*-gemmaevalpt`); no gemma-3-27b forward pass runs here.

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**The over-1024-token filter.** `get_activations` pads *or truncates* every conversation to 1024 tokens, so a longer one is trained on from its opening alone. These runs predate `token_budget.py`, so nothing length-guarded the contrastive generator. Rows whose conversation exceeds the cap — **and** rows whose cached activation is stored truncated, which `get_activations` also produces for a short conversation that merely shared an extraction batch with an over-long one — are dropped together with their pair partner, keeping every vintage exactly 50/50. Per-arm counts are in `*_vintage.json` under `report.long_filter`.

**Read the sd, not just the mean.** These are independent `ProbeFactory` fits with independent initialisations, and seed alone moves some splits by more than the between-vintage gaps. A single-seed comparison of two vintages means nothing; quantifying that is what this sweep exists for.

## Progress: 32 fits recorded

- **gptoss120b**: v0×5, v1×5, v2×4, v3×4
- **nemotron**: v0×4, v1×4, v2×3, v3×3

## gptoss120b — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 5 | 0.3700 ± 0.2337 | 0.5190 ± 0.0677 | 0.5055 ± 0.0336 | 0.5024 ± 0.0429 | 0.5094 ± 0.1305 | 0.4990 ± 0.0660 | 0.4777 ± 0.0349 | 0.4833 ± 0.0241 |
| v1 | 434 | 5 | 0.4314 ± 0.0843 | 0.9148 ± 0.0126 | 0.7435 ± 0.0785 | 0.9231 ± 0.0142 | 0.8882 ± 0.0182 | 0.6837 ± 0.0232 | 0.7235 ± 0.0210 | 0.7583 ± 0.0194 |
| v2 | 674 | 4 | 0.7343 ± 0.0568 | 0.8856 ± 0.0334 | 0.7717 ± 0.1050 | 0.9044 ± 0.0191 | 0.8672 ± 0.0544 | 0.5910 ± 0.0256 | 0.7132 ± 0.0297 | 0.7811 ± 0.0247 |
| v3 | 858 | 4 | 0.8361 ± 0.0454 | 0.9087 ± 0.0313 | 0.7691 ± 0.0797 | 0.9027 ± 0.0168 | 0.8753 ± 0.0217 | 0.6052 ± 0.0216 | 0.7858 ± 0.0125 | 0.8119 ± 0.0193 |

## nemotron — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 4 | 0.3997 ± 0.2588 | 0.5028 ± 0.0661 | 0.4939 ± 0.0247 | 0.4874 ± 0.0308 | 0.4721 ± 0.1157 | 0.4915 ± 0.0737 | 0.4836 ± 0.0372 | 0.4759 ± 0.0202 |
| v1 | 458 | 4 | 0.6470 ± 0.0428 | 0.9674 ± 0.0049 | 0.8425 ± 0.0269 | 0.8651 ± 0.0064 | 0.9638 ± 0.0093 | 0.7519 ± 0.0171 | 0.6921 ± 0.0357 | 0.8186 ± 0.0058 |
| v2 | 630 | 3 | 0.8572 ± 0.0251 | 0.9443 ± 0.0043 | 0.8871 ± 0.0095 | 0.7879 ± 0.0533 | 0.9351 ± 0.0369 | 0.7122 ± 0.0060 | 0.6826 ± 0.0563 | 0.8295 ± 0.0138 |
| v3 | 926 | 3 | 0.7150 ± 0.0294 | 0.9064 ± 0.0375 | 0.7673 ± 0.0367 | 0.7937 ± 0.0563 | 0.9331 ± 0.0131 | 0.6962 ± 0.0253 | 0.6343 ± 0.0316 | 0.7780 ± 0.0171 |
