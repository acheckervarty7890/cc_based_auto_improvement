# Red-team vintage sweep — instruction-following concept
_Updated 2026-08-15T14:38:13+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data (`data/instructions_llama70b_50.jsonl`) plus one *vintage* of iteration-3 red-team pairs, scored on the seven `eval_instructions` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Setup.** Probe `google/gemma-3-27b-it` L32, labels `assistant_follows_the_instruction` / `assistant_does_not_follow_the_instruction`. Arms are the two attackers of `run_gemma27b_instructions_attackers.sh`: `gptoss120b` (`openai/gpt-oss-120b`) and `nemotron` (`nvidia/nemotron-3-ultra-550b-a55b`). All activations — base split, per-conversation red-team, and the seven eval splits — were pulled from Kaggle (`anku7890/instructions-gemma27b-*`, `anku7890/*-gemmaevalpt`); no gemma-3-27b forward pass runs here.

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**The over-1024-token filter.** `get_activations` pads *or truncates* every conversation to 1024 tokens, so a longer one is trained on from its opening alone. These runs predate `token_budget.py`, so nothing length-guarded the contrastive generator. Rows whose conversation exceeds the cap — **and** rows whose cached activation is stored truncated, which `get_activations` also produces for a short conversation that merely shared an extraction batch with an over-long one — are dropped together with their pair partner, keeping every vintage exactly 50/50. Per-arm counts are in `*_vintage.json` under `report.long_filter`.

**Read the sd, not just the mean.** These are independent `ProbeFactory` fits with independent initialisations, and seed alone moves some splits by more than the between-vintage gaps. A single-seed comparison of two vintages means nothing; quantifying that is what this sweep exists for.

## Progress: 14 fits recorded

- **gptoss120b**: v0×2, v1×2, v2×2, v3×1
- **nemotron**: v0×2, v1×2, v2×2, v3×1

## gptoss120b — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 2 | 0.6155 ± 0.0356 | 0.4612 ± 0.0784 | 0.4758 ± 0.0161 | 0.4694 ± 0.0298 | 0.4019 ± 0.0776 | 0.5545 ± 0.0018 | 0.4708 ± 0.0245 | 0.4927 ± 0.0085 |
| v1 | 434 | 2 | 0.4226 ± 0.0112 | 0.9273 ± 0.0099 | 0.7476 ± 0.1065 | 0.9191 ± 0.0105 | 0.9034 ± 0.0150 | 0.7032 ± 0.0219 | 0.7277 ± 0.0288 | 0.7644 ± 0.0186 |
| v2 | 674 | 2 | 0.7833 ± 0.0047 | 0.8998 ± 0.0277 | 0.7386 ± 0.1114 | 0.9045 ± 0.0033 | 0.8346 ± 0.0527 | 0.5811 ± 0.0364 | 0.7291 ± 0.0183 | 0.7816 ± 0.0288 |
| v3 | 858 | 1 | 0.8042 | 0.9348 | 0.7627 | 0.9131 | 0.8499 | 0.5731 | 0.7742 | 0.8017 |

## nemotron — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 2 | 0.6155 ± 0.0356 | 0.4612 ± 0.0784 | 0.4758 ± 0.0161 | 0.4694 ± 0.0298 | 0.4019 ± 0.0776 | 0.5545 ± 0.0018 | 0.4708 ± 0.0245 | 0.4927 ± 0.0085 |
| v1 | 458 | 2 | 0.6515 ± 0.0422 | 0.9700 ± 0.0066 | 0.8446 ± 0.0392 | 0.8642 ± 0.0086 | 0.9648 ± 0.0143 | 0.7557 ± 0.0259 | 0.6951 ± 0.0414 | 0.8209 ± 0.0071 |
| v2 | 630 | 2 | 0.8607 ± 0.0345 | 0.9419 ± 0.0010 | 0.8908 ± 0.0098 | 0.7732 ± 0.0663 | 0.9230 ± 0.0431 | 0.7145 ± 0.0062 | 0.6510 ± 0.0188 | 0.8222 ± 0.0076 |
| v3 | 926 | 1 | 0.7423 | 0.8631 | 0.7964 | 0.7500 | 0.9298 | 0.6736 | 0.5988 | 0.7649 |
