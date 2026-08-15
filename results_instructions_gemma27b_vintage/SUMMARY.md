# Red-team vintage sweep — instruction-following concept
_Updated 2026-08-15T17:41:17+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data (`data/instructions_llama70b_50.jsonl`) plus one *vintage* of iteration-3 red-team pairs, scored on the seven `eval_instructions` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Setup.** Probe `google/gemma-3-27b-it` L32, labels `assistant_follows_the_instruction` / `assistant_does_not_follow_the_instruction`. Arms are the two attackers of `run_gemma27b_instructions_attackers.sh`: `gptoss120b` (`openai/gpt-oss-120b`) and `nemotron` (`nvidia/nemotron-3-ultra-550b-a55b`). All activations — base split, per-conversation red-team, and the seven eval splits — were pulled from Kaggle (`anku7890/instructions-gemma27b-*`, `anku7890/*-gemmaevalpt`); no gemma-3-27b forward pass runs here.

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**The over-1024-token filter.** `get_activations` pads *or truncates* every conversation to 1024 tokens, so a longer one is trained on from its opening alone. These runs predate `token_budget.py`, so nothing length-guarded the contrastive generator. Rows whose conversation exceeds the cap — **and** rows whose cached activation is stored truncated, which `get_activations` also produces for a short conversation that merely shared an extraction batch with an over-long one — are dropped together with their pair partner, keeping every vintage exactly 50/50. Per-arm counts are in `*_vintage.json` under `report.long_filter`.

**Read the sd, not just the mean.** These are independent `ProbeFactory` fits with independent initialisations, and seed alone moves some splits by more than the between-vintage gaps. A single-seed comparison of two vintages means nothing; quantifying that is what this sweep exists for.

## Progress: 70 fits recorded

- **gptoss120b**: v0×10, v1×10, v2×10, v3×9
- **nemotron**: v0×8, v1×8, v2×8, v3×7

## gptoss120b — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 10 | 0.4549 ± 0.2438 | 0.4951 ± 0.0629 | 0.4972 ± 0.0273 | 0.4962 ± 0.0317 | 0.5342 ± 0.1264 | 0.5226 ± 0.0629 | 0.4802 ± 0.0321 | 0.4972 ± 0.0349 |
| v1 | 434 | 10 | 0.4444 ± 0.0655 | 0.9097 ± 0.0203 | 0.7356 ± 0.0591 | 0.9297 ± 0.0141 | 0.8823 ± 0.0407 | 0.6749 ± 0.0360 | 0.7191 ± 0.0178 | 0.7565 ± 0.0186 |
| v2 | 674 | 10 | 0.6802 ± 0.1063 | 0.8930 ± 0.0230 | 0.7384 ± 0.1049 | 0.8998 ± 0.0221 | 0.8794 ± 0.0415 | 0.5852 ± 0.0319 | 0.7189 ± 0.0505 | 0.7707 ± 0.0233 |
| v3 | 858 | 9 | 0.8364 ± 0.0506 | 0.8920 ± 0.0315 | 0.7616 ± 0.0555 | 0.8999 ± 0.0195 | 0.8589 ± 0.0300 | 0.5896 ± 0.0265 | 0.7764 ± 0.0194 | 0.8021 ± 0.0222 |

## nemotron — mean ± sd over seeds (pipeline scale)

| vintage | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 8 | 0.4602 ± 0.2170 | 0.5125 ± 0.0571 | 0.5014 ± 0.0292 | 0.5009 ± 0.0342 | 0.5059 ± 0.1260 | 0.5056 ± 0.0538 | 0.4842 ± 0.0290 | 0.4958 ± 0.0285 |
| v1 | 458 | 8 | 0.6135 ± 0.1147 | 0.9678 ± 0.0046 | 0.8450 ± 0.0343 | 0.8676 ± 0.0116 | 0.9570 ± 0.0110 | 0.7380 ± 0.0221 | 0.6866 ± 0.0299 | 0.8108 ± 0.0185 |
| v2 | 630 | 8 | 0.8111 ± 0.0595 | 0.9549 ± 0.0110 | 0.8726 ± 0.0205 | 0.7745 ± 0.0496 | 0.8874 ± 0.0790 | 0.6976 ± 0.0306 | 0.6340 ± 0.0655 | 0.8046 ± 0.0373 |
| v3 | 926 | 7 | 0.7107 ± 0.0447 | 0.9134 ± 0.0273 | 0.7780 ± 0.0326 | 0.8335 ± 0.0577 | 0.9271 ± 0.0296 | 0.6805 ± 0.0260 | 0.6178 ± 0.0541 | 0.7802 ± 0.0236 |
