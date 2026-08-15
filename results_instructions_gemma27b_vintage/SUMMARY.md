# Red-team vintage sweep — instruction-following concept
_Updated 2026-08-15T17:10:02+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data (`data/instructions_llama70b_50.jsonl`) plus one *vintage* of iteration-3 red-team pairs, scored on the seven `eval_instructions` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Setup.** Probe `google/gemma-3-27b-it` L32, labels `assistant_follows_the_instruction` / `assistant_does_not_follow_the_instruction`. Arms are the two attackers of `run_gemma27b_instructions_attackers.sh`: `gptoss120b` (`openai/gpt-oss-120b`) and `nemotron` (`nvidia/nemotron-3-ultra-550b-a55b`). All activations — base split, per-conversation red-team, and the seven eval splits — were pulled from Kaggle (`anku7890/instructions-gemma27b-*`, `anku7890/*-gemmaevalpt`); no gemma-3-27b forward pass runs here.

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**The over-1024-token filter.** `get_activations` pads *or truncates* every conversation to 1024 tokens, so a longer one is trained on from its opening alone. These runs predate `token_budget.py`, so nothing length-guarded the contrastive generator. Rows whose conversation exceeds the cap — **and** rows whose cached activation is stored truncated, which `get_activations` also produces for a short conversation that merely shared an extraction batch with an over-long one — are dropped together with their pair partner, keeping every vintage exactly 50/50. Per-arm counts are in `*_vintage.json` under `report.long_filter`.

**Read the sd, not just the mean.** These are independent `ProbeFactory` fits with independent initialisations, and seed alone moves some splits by more than the between-vintage gaps. A single-seed comparison of two vintages means nothing; quantifying that is what this sweep exists for.

## Progress: 60 fits recorded

- **gptoss120b**: v0×9, v1×9, v2×8, v3×8
- **nemotron**: v0×7, v1×7, v2×6, v3×6

## gptoss120b — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 9 | 0.4217 ± 0.2334 | 0.5049 ± 0.0581 | 0.4996 ± 0.0278 | 0.4984 ± 0.0328 | 0.5238 ± 0.1295 | 0.5102 ± 0.0522 | 0.4778 ± 0.0332 | 0.4909 ± 0.0304 |
| v1 | 434 | 9 | 0.4423 ± 0.0691 | 0.9082 ± 0.0209 | 0.7332 ± 0.0622 | 0.9311 ± 0.0142 | 0.8747 ± 0.0349 | 0.6675 ± 0.0290 | 0.7223 ± 0.0154 | 0.7542 ± 0.0181 |
| v2 | 674 | 8 | 0.7112 ± 0.0938 | 0.8919 ± 0.0259 | 0.7538 ± 0.1109 | 0.8962 ± 0.0207 | 0.8715 ± 0.0424 | 0.5878 ± 0.0353 | 0.7147 ± 0.0549 | 0.7753 ± 0.0229 |
| v3 | 858 | 8 | 0.8362 ± 0.0541 | 0.8950 ± 0.0323 | 0.7608 ± 0.0593 | 0.9003 ± 0.0208 | 0.8599 ± 0.0319 | 0.5915 ± 0.0276 | 0.7790 ± 0.0190 | 0.8033 ± 0.0235 |

## nemotron — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 7 | 0.4447 ± 0.2296 | 0.5085 ± 0.0605 | 0.5012 ± 0.0315 | 0.4988 ± 0.0364 | 0.5037 ± 0.1359 | 0.5052 ± 0.0581 | 0.4851 ± 0.0312 | 0.4925 ± 0.0291 |
| v1 | 458 | 7 | 0.6109 ± 0.1237 | 0.9674 ± 0.0049 | 0.8449 ± 0.0370 | 0.8678 ± 0.0125 | 0.9589 ± 0.0104 | 0.7434 ± 0.0171 | 0.6830 ± 0.0304 | 0.8109 ± 0.0200 |
| v2 | 630 | 6 | 0.8244 ± 0.0572 | 0.9519 ± 0.0095 | 0.8745 ± 0.0228 | 0.7677 ± 0.0532 | 0.8914 ± 0.0684 | 0.6991 ± 0.0225 | 0.6407 ± 0.0644 | 0.8071 ± 0.0352 |
| v3 | 926 | 6 | 0.6999 ± 0.0377 | 0.9091 ± 0.0271 | 0.7727 ± 0.0322 | 0.8262 ± 0.0595 | 0.9248 ± 0.0318 | 0.6785 ± 0.0279 | 0.6043 ± 0.0446 | 0.7737 ± 0.0176 |
