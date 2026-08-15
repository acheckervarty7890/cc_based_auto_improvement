# Red-team vintage sweep — instruction-following concept
_Updated 2026-08-15T16:39:40+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data (`data/instructions_llama70b_50.jsonl`) plus one *vintage* of iteration-3 red-team pairs, scored on the seven `eval_instructions` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Setup.** Probe `google/gemma-3-27b-it` L32, labels `assistant_follows_the_instruction` / `assistant_does_not_follow_the_instruction`. Arms are the two attackers of `run_gemma27b_instructions_attackers.sh`: `gptoss120b` (`openai/gpt-oss-120b`) and `nemotron` (`nvidia/nemotron-3-ultra-550b-a55b`). All activations — base split, per-conversation red-team, and the seven eval splits — were pulled from Kaggle (`anku7890/instructions-gemma27b-*`, `anku7890/*-gemmaevalpt`); no gemma-3-27b forward pass runs here.

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**The over-1024-token filter.** `get_activations` pads *or truncates* every conversation to 1024 tokens, so a longer one is trained on from its opening alone. These runs predate `token_budget.py`, so nothing length-guarded the contrastive generator. Rows whose conversation exceeds the cap — **and** rows whose cached activation is stored truncated, which `get_activations` also produces for a short conversation that merely shared an extraction batch with an over-long one — are dropped together with their pair partner, keeping every vintage exactly 50/50. Per-arm counts are in `*_vintage.json` under `report.long_filter`.

**Read the sd, not just the mean.** These are independent `ProbeFactory` fits with independent initialisations, and seed alone moves some splits by more than the between-vintage gaps. A single-seed comparison of two vintages means nothing; quantifying that is what this sweep exists for.

## Progress: 49 fits recorded

- **gptoss120b**: v0×7, v1×7, v2×7, v3×6
- **nemotron**: v0×6, v1×6, v2×5, v3×5

## gptoss120b — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 7 | 0.4447 ± 0.2296 | 0.5085 ± 0.0605 | 0.5012 ± 0.0315 | 0.4988 ± 0.0364 | 0.5037 ± 0.1359 | 0.5052 ± 0.0581 | 0.4851 ± 0.0312 | 0.4925 ± 0.0291 |
| v1 | 434 | 7 | 0.4331 ± 0.0769 | 0.9137 ± 0.0139 | 0.7271 ± 0.0704 | 0.9287 ± 0.0154 | 0.8857 ± 0.0177 | 0.6738 ± 0.0255 | 0.7230 ± 0.0174 | 0.7550 ± 0.0188 |
| v2 | 674 | 7 | 0.7074 ± 0.1007 | 0.8901 ± 0.0275 | 0.7306 ± 0.0965 | 0.8959 ± 0.0223 | 0.8755 ± 0.0441 | 0.5849 ± 0.0371 | 0.7171 ± 0.0589 | 0.7716 ± 0.0220 |
| v3 | 858 | 6 | 0.8275 ± 0.0450 | 0.8994 ± 0.0365 | 0.7700 ± 0.0660 | 0.9050 ± 0.0153 | 0.8620 ± 0.0372 | 0.6016 ± 0.0240 | 0.7763 ± 0.0177 | 0.8060 ± 0.0245 |

## nemotron — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 6 | 0.4145 ± 0.2358 | 0.5080 ± 0.0662 | 0.4991 ± 0.0340 | 0.4980 ± 0.0398 | 0.4819 ± 0.1348 | 0.5082 ± 0.0631 | 0.4814 ± 0.0325 | 0.4844 ± 0.0217 |
| v1 | 458 | 6 | 0.6549 ± 0.0454 | 0.9686 ± 0.0042 | 0.8542 ± 0.0302 | 0.8648 ± 0.0108 | 0.9583 ± 0.0112 | 0.7456 ± 0.0177 | 0.6811 ± 0.0328 | 0.8182 ± 0.0055 |
| v2 | 630 | 5 | 0.8390 ± 0.0499 | 0.9502 ± 0.0096 | 0.8743 ± 0.0255 | 0.7700 ± 0.0592 | 0.9059 ± 0.0655 | 0.7043 ± 0.0208 | 0.6554 ± 0.0598 | 0.8141 ± 0.0343 |
| v3 | 926 | 5 | 0.7111 ± 0.0291 | 0.9133 ± 0.0281 | 0.7758 ± 0.0349 | 0.8140 ± 0.0575 | 0.9361 ± 0.0175 | 0.6793 ± 0.0312 | 0.6171 ± 0.0355 | 0.7781 ± 0.0155 |
