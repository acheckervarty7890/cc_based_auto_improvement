# Red-team vintage sweep — instruction-following concept
_Updated 2026-08-15T15:08:35+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data (`data/instructions_llama70b_50.jsonl`) plus one *vintage* of iteration-3 red-team pairs, scored on the seven `eval_instructions` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Setup.** Probe `google/gemma-3-27b-it` L32, labels `assistant_follows_the_instruction` / `assistant_does_not_follow_the_instruction`. Arms are the two attackers of `run_gemma27b_instructions_attackers.sh`: `gptoss120b` (`openai/gpt-oss-120b`) and `nemotron` (`nvidia/nemotron-3-ultra-550b-a55b`). All activations — base split, per-conversation red-team, and the seven eval splits — were pulled from Kaggle (`anku7890/instructions-gemma27b-*`, `anku7890/*-gemmaevalpt`); no gemma-3-27b forward pass runs here.

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**The over-1024-token filter.** `get_activations` pads *or truncates* every conversation to 1024 tokens, so a longer one is trained on from its opening alone. These runs predate `token_budget.py`, so nothing length-guarded the contrastive generator. Rows whose conversation exceeds the cap — **and** rows whose cached activation is stored truncated, which `get_activations` also produces for a short conversation that merely shared an extraction batch with an over-long one — are dropped together with their pair partner, keeping every vintage exactly 50/50. Per-arm counts are in `*_vintage.json` under `report.long_filter`.

**Read the sd, not just the mean.** These are independent `ProbeFactory` fits with independent initialisations, and seed alone moves some splits by more than the between-vintage gaps. A single-seed comparison of two vintages means nothing; quantifying that is what this sweep exists for.

## Progress: 22 fits recorded

- **gptoss120b**: v0×3, v1×3, v2×3, v3×2
- **nemotron**: v0×3, v1×3, v2×3, v3×2

## gptoss120b — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 3 | 0.4443 ± 0.2975 | 0.4905 ± 0.0752 | 0.4917 ± 0.0297 | 0.4874 ± 0.0377 | 0.4770 ± 0.1412 | 0.5173 ± 0.0644 | 0.4667 ± 0.0187 | 0.4821 ± 0.0193 |
| v1 | 434 | 3 | 0.3924 ± 0.0529 | 0.9201 ± 0.0143 | 0.7249 ± 0.0850 | 0.9230 ± 0.0100 | 0.8920 ± 0.0225 | 0.6870 ± 0.0320 | 0.7271 ± 0.0204 | 0.7523 ± 0.0247 |
| v2 | 674 | 3 | 0.7528 ± 0.0530 | 0.9003 ± 0.0196 | 0.7907 ± 0.1199 | 0.9122 ± 0.0135 | 0.8462 ± 0.0423 | 0.5915 ± 0.0313 | 0.7270 ± 0.0135 | 0.7887 ± 0.0238 |
| v3 | 858 | 2 | 0.8264 ± 0.0315 | 0.9277 ± 0.0100 | 0.7344 ± 0.0400 | 0.9131 ± 0.0001 | 0.8762 ± 0.0373 | 0.5963 ± 0.0328 | 0.7777 ± 0.0049 | 0.8074 ± 0.0081 |

## nemotron — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 3 | 0.4443 ± 0.2975 | 0.4905 ± 0.0752 | 0.4917 ± 0.0297 | 0.4874 ± 0.0377 | 0.4770 ± 0.1412 | 0.5173 ± 0.0644 | 0.4667 ± 0.0187 | 0.4821 ± 0.0193 |
| v1 | 458 | 3 | 0.6343 ± 0.0422 | 0.9681 ± 0.0058 | 0.8374 ± 0.0304 | 0.8664 ± 0.0072 | 0.9658 ± 0.0103 | 0.7503 ± 0.0206 | 0.7039 ± 0.0329 | 0.8180 ± 0.0070 |
| v2 | 630 | 3 | 0.8572 ± 0.0251 | 0.9443 ± 0.0043 | 0.8871 ± 0.0095 | 0.7879 ± 0.0533 | 0.9351 ± 0.0369 | 0.7122 ± 0.0060 | 0.6826 ± 0.0563 | 0.8295 ± 0.0138 |
| v3 | 926 | 2 | 0.7130 ± 0.0414 | 0.8960 ± 0.0464 | 0.7612 ± 0.0498 | 0.7619 ± 0.0168 | 0.9259 ± 0.0056 | 0.6986 ± 0.0354 | 0.6218 ± 0.0325 | 0.7683 ± 0.0049 |
