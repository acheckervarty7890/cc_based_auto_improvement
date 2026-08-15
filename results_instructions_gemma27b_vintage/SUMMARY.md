# Red-team vintage sweep — instruction-following concept
_Updated 2026-08-15T16:09:24+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data (`data/instructions_llama70b_50.jsonl`) plus one *vintage* of iteration-3 red-team pairs, scored on the seven `eval_instructions` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Setup.** Probe `google/gemma-3-27b-it` L32, labels `assistant_follows_the_instruction` / `assistant_does_not_follow_the_instruction`. Arms are the two attackers of `run_gemma27b_instructions_attackers.sh`: `gptoss120b` (`openai/gpt-oss-120b`) and `nemotron` (`nvidia/nemotron-3-ultra-550b-a55b`). All activations — base split, per-conversation red-team, and the seven eval splits — were pulled from Kaggle (`anku7890/instructions-gemma27b-*`, `anku7890/*-gemmaevalpt`); no gemma-3-27b forward pass runs here.

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**The over-1024-token filter.** `get_activations` pads *or truncates* every conversation to 1024 tokens, so a longer one is trained on from its opening alone. These runs predate `token_budget.py`, so nothing length-guarded the contrastive generator. Rows whose conversation exceeds the cap — **and** rows whose cached activation is stored truncated, which `get_activations` also produces for a short conversation that merely shared an extraction batch with an over-long one — are dropped together with their pair partner, keeping every vintage exactly 50/50. Per-arm counts are in `*_vintage.json` under `report.long_filter`.

**Read the sd, not just the mean.** These are independent `ProbeFactory` fits with independent initialisations, and seed alone moves some splits by more than the between-vintage gaps. A single-seed comparison of two vintages means nothing; quantifying that is what this sweep exists for.

## Progress: 41 fits recorded

- **gptoss120b**: v0×6, v1×6, v2×6, v3×5
- **nemotron**: v0×5, v1×5, v2×4, v3×4

## gptoss120b — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 6 | 0.4145 ± 0.2358 | 0.5080 ± 0.0662 | 0.4991 ± 0.0340 | 0.4980 ± 0.0398 | 0.4819 ± 0.1348 | 0.5082 ± 0.0631 | 0.4814 ± 0.0325 | 0.4844 ± 0.0217 |
| v1 | 434 | 6 | 0.4422 ± 0.0800 | 0.9168 ± 0.0123 | 0.7363 ± 0.0724 | 0.9274 ± 0.0164 | 0.8892 ± 0.0165 | 0.6772 ± 0.0262 | 0.7225 ± 0.0190 | 0.7588 ± 0.0174 |
| v2 | 674 | 6 | 0.6976 ± 0.1065 | 0.8927 ± 0.0291 | 0.7439 ± 0.0984 | 0.9025 ± 0.0151 | 0.8709 ± 0.0464 | 0.5909 ± 0.0368 | 0.7104 ± 0.0616 | 0.7727 ± 0.0239 |
| v3 | 858 | 5 | 0.8387 ± 0.0397 | 0.9104 ± 0.0274 | 0.7770 ± 0.0712 | 0.9064 ± 0.0167 | 0.8755 ± 0.0188 | 0.6082 ± 0.0199 | 0.7798 ± 0.0173 | 0.8137 ± 0.0172 |

## nemotron — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 5 | 0.3700 ± 0.2337 | 0.5190 ± 0.0677 | 0.5055 ± 0.0336 | 0.5024 ± 0.0429 | 0.5094 ± 0.1305 | 0.4990 ± 0.0660 | 0.4777 ± 0.0349 | 0.4833 ± 0.0241 |
| v1 | 458 | 5 | 0.6608 ± 0.0481 | 0.9681 ± 0.0045 | 0.8456 ± 0.0243 | 0.8680 ± 0.0085 | 0.9599 ± 0.0117 | 0.7461 ± 0.0197 | 0.6869 ± 0.0331 | 0.8193 ± 0.0053 |
| v2 | 630 | 4 | 0.8597 ± 0.0211 | 0.9465 ± 0.0055 | 0.8852 ± 0.0086 | 0.7903 ± 0.0438 | 0.9326 ± 0.0305 | 0.7133 ± 0.0054 | 0.6742 ± 0.0489 | 0.8288 ± 0.0113 |
| v3 | 926 | 4 | 0.7056 ± 0.0305 | 0.9110 ± 0.0320 | 0.7798 ± 0.0390 | 0.7954 ± 0.0461 | 0.9299 ± 0.0124 | 0.6819 ± 0.0353 | 0.6186 ± 0.0408 | 0.7746 ± 0.0155 |
