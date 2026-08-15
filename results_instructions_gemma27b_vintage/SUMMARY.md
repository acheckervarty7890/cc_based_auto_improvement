# Red-team vintage sweep — instruction-following concept
_Updated 2026-08-15T18:11:56+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data (`data/instructions_llama70b_50.jsonl`) plus one *vintage* of iteration-3 red-team pairs, scored on the seven `eval_instructions` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Setup.** Probe `google/gemma-3-27b-it` L32, labels `assistant_follows_the_instruction` / `assistant_does_not_follow_the_instruction`. Arms are the two attackers of `run_gemma27b_instructions_attackers.sh`: `gptoss120b` (`openai/gpt-oss-120b`) and `nemotron` (`nvidia/nemotron-3-ultra-550b-a55b`). All activations — base split, per-conversation red-team, and the seven eval splits — were pulled from Kaggle (`anku7890/instructions-gemma27b-*`, `anku7890/*-gemmaevalpt`); no gemma-3-27b forward pass runs here.

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**The over-1024-token filter.** `get_activations` pads *or truncates* every conversation to 1024 tokens, so a longer one is trained on from its opening alone. These runs predate `token_budget.py`, so nothing length-guarded the contrastive generator. Rows whose conversation exceeds the cap — **and** rows whose cached activation is stored truncated, which `get_activations` also produces for a short conversation that merely shared an extraction batch with an over-long one — are dropped together with their pair partner, keeping every vintage exactly 50/50. Per-arm counts are in `*_vintage.json` under `report.long_filter`.

**Read the sd, not just the mean.** These are independent `ProbeFactory` fits with independent initialisations, and seed alone moves some splits by more than the between-vintage gaps. A single-seed comparison of two vintages means nothing; quantifying that is what this sweep exists for.

## Progress: 79 fits recorded

- **gptoss120b**: v0×10, v1×10, v2×10, v3×10
- **nemotron**: v0×10, v1×10, v2×10, v3×9

## gptoss120b — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.4549 ± 0.2438 | 0.4951 ± 0.0629 | 0.4972 ± 0.0273 | 0.4962 ± 0.0317 | 0.5342 ± 0.1264 | 0.5226 ± 0.0629 | 0.4802 ± 0.0321 | 0.4972 ± 0.0349 |
| v1 | 434 | 10 | 0.4444 ± 0.0655 | 0.9097 ± 0.0203 | 0.7356 ± 0.0591 | 0.9297 ± 0.0141 | 0.8823 ± 0.0407 | 0.6749 ± 0.0360 | 0.7191 ± 0.0178 | 0.7565 ± 0.0186 |
| v2 | 674 | 10 | 0.6802 ± 0.1063 | 0.8930 ± 0.0230 | 0.7384 ± 0.1049 | 0.8998 ± 0.0221 | 0.8794 ± 0.0415 | 0.5852 ± 0.0319 | 0.7189 ± 0.0505 | 0.7707 ± 0.0233 |
| v3 | 858 | 10 | 0.8298 ± 0.0521 | 0.8908 ± 0.0300 | 0.7564 ± 0.0548 | 0.8980 ± 0.0193 | 0.8581 ± 0.0284 | 0.5872 ± 0.0260 | 0.7748 ± 0.0190 | 0.7993 ± 0.0228 |

## nemotron — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.4549 ± 0.2438 | 0.4951 ± 0.0629 | 0.4972 ± 0.0273 | 0.4962 ± 0.0317 | 0.5342 ± 0.1264 | 0.5226 ± 0.0629 | 0.4802 ± 0.0321 | 0.4972 ± 0.0349 |
| v1 | 458 | 10 | 0.6152 ± 0.1061 | 0.9670 ± 0.0048 | 0.8452 ± 0.0303 | 0.8692 ± 0.0120 | 0.9588 ± 0.0107 | 0.7379 ± 0.0195 | 0.6869 ± 0.0282 | 0.8115 ± 0.0166 |
| v2 | 630 | 10 | 0.8113 ± 0.0539 | 0.9542 ± 0.0098 | 0.8776 ± 0.0214 | 0.7661 ± 0.0498 | 0.8832 ± 0.0739 | 0.6985 ± 0.0283 | 0.6269 ± 0.0610 | 0.8025 ± 0.0346 |
| v3 | 926 | 9 | 0.7232 ± 0.0488 | 0.9155 ± 0.0243 | 0.7726 ± 0.0316 | 0.8366 ± 0.0539 | 0.9256 ± 0.0274 | 0.6784 ± 0.0293 | 0.6278 ± 0.0509 | 0.7828 ± 0.0219 |
