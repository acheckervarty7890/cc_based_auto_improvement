# Red-team vintage sweep — instruction-following concept
_Updated 2026-08-15T18:21:11+00:00_

**What this measures.** Each row is a real `ProbeFactory` refit on the base training data (`data/instructions_llama70b_50.jsonl`) plus one *vintage* of iteration-3 red-team pairs, scored on the seven `eval_instructions` splits from cached activations. Only set membership varies between vintages — content, activations and every hyperparameter are iteration 3's — so the vintages are directly comparable to each other in a way the original `probe_iter1/2/3` are not (those came from separate retrains with their own filter draws and contrastive generations).

**Setup.** Probe `google/gemma-3-27b-it` L32, labels `assistant_follows_the_instruction` / `assistant_does_not_follow_the_instruction`. Arms are the two attackers of `run_gemma27b_instructions_attackers.sh`: `gptoss120b` (`openai/gpt-oss-120b`) and `nemotron` (`nvidia/nemotron-3-ultra-550b-a55b`). All activations — base split, per-conversation red-team, and the seven eval splits — were pulled from Kaggle (`anku7890/instructions-gemma27b-*`, `anku7890/*-gemmaevalpt`); no gemma-3-27b forward pass runs here.

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs

**The over-1024-token filter.** `get_activations` pads *or truncates* every conversation to 1024 tokens, so a longer one is trained on from its opening alone. These runs predate `token_budget.py`, so nothing length-guarded the contrastive generator. Rows whose conversation exceeds the cap — **and** rows whose cached activation is stored truncated, which `get_activations` also produces for a short conversation that merely shared an extraction batch with an over-long one — are dropped together with their pair partner, keeping every vintage exactly 50/50. Per-arm counts are in `*_vintage.json` under `report.long_filter`.

**Read the sd, not just the mean.** These are independent `ProbeFactory` fits with independent initialisations, and seed alone moves some splits by more than the between-vintage gaps. A single-seed comparison of two vintages means nothing; quantifying that is what this sweep exists for.

## Progress: 80 fits recorded

- **gptoss120b**: v0×10, v1×10, v2×10, v3×10
- **nemotron**: v0×10, v1×10, v2×10, v3×10

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
| v3 | 926 | 10 | 0.7345 ± 0.0583 | 0.9174 ± 0.0238 | 0.7690 ± 0.0319 | 0.8416 ± 0.0532 | 0.9266 ± 0.0260 | 0.6750 ± 0.0297 | 0.6340 ± 0.0519 | 0.7854 ± 0.0222 |

## Findings

**Iteration 1's red-team data buys nearly all of the gain.** Both arms jump from chance to
~0.76-0.81 mean AUROC on iteration-1 pairs alone, and neither improves reliably after that:

| arm | v0 -> v1 | v1 -> v2 | v2 -> v3 | v1 -> v3 |
|---|---|---|---|---|
| gptoss120b | **+0.259** | +0.014 | +0.029 | +0.043 |
| nemotron | **+0.314** | -0.009 | -0.017 | **-0.026** |

Against a seed sd of ~0.017-0.035 on the mean, only gptoss120b's v1->v3 gap is even marginally
outside noise; every single-iteration step is inside it, and nemotron's later iterations are
flat to slightly negative — v3 trains on **twice** v1's data and scores about one sd *lower*.

**The aggregate hides a trade, not a plateau.** Later iterations are not adding nothing; they
are moving accuracy between splits. `anthropic_harmless_refusal` improves monotonically and
carries essentially all of the v1->v3 mean gain (gptoss 0.444 -> 0.680 -> 0.830; nemotron
0.615 -> 0.811 -> 0.735), while `oig_context_drift` *drops* at v2 and never recovers (gptoss
0.675 -> 0.585 -> 0.587; nemotron 0.738 -> 0.699 -> 0.675) and nemotron's `oig_omission` falls
0.687 -> 0.634. So iterations 2-3 chiefly reallocate the probe's capacity across failure modes.

**This is why the sweep exists: the committed single-seed CSVs are not readable as an ordering.**
`nemotron_comparison.csv` reports iter1/2/3 as 0.812 / 0.834 / 0.764, i.e. a peak at iteration 2.
Here v2 = 0.803 +/- 0.035 and v1 = 0.812 +/- 0.017 — that peak is inside seed noise and the
apparent iteration-2 advantage is an artifact of a single draw. Note also v0 on
`anthropic_harmless_refusal` has sd **0.244**: an untrained probe on that split is a coin flip
whose direction the seed picks, so any single-seed baseline there is meaningless.

**Attacker comparison.** nemotron's v1 beats gptoss120b's v1 by 0.055 (0.812 vs 0.757, ~3 sd) —
its first-iteration data is genuinely more useful. By v3 the arms converge (0.785 vs 0.799),
because nemotron gives the gain back over iterations 2-3 while gptoss120b slowly adds to it.

**v0 is identical across arms by construction** (base training data only, no red-team rows).
The two arms' v0 rows matching to the last digit is a consistency check on the two processes,
not a result.

## Reproducing

```bash
# activations (all from Kaggle; no gemma-3-27b forward pass is needed anywhere)
KAGGLE_CONFIG_DIR=<dir with kaggle.json> \
  .venv_claude/bin/python scripts/publish_kaggle_redteam_activations.py restore \
  --experiment instructions --iterations 3 --base --no-per-file
KAGGLE_CONFIG_DIR=<dir with kaggle.json> \
  .venv_claude/bin/python scripts/attribution_fetch_eval.py

# the sweep (both arms in parallel; AGENTIC_FAST_ACTS is a bit-identical 3.5-4.5x speedup)
AGENTIC_FAST_ACTS=1 .venv_claude/bin/python scripts/attribution_vintage.py \
  --arm gptoss120b --seeds 10 --drop-long pair
AGENTIC_FAST_ACTS=1 .venv_claude/bin/python scripts/attribution_vintage.py \
  --arm nemotron --seeds 10 --drop-long pair
.venv_claude/bin/python scripts/attribution_vintage.py --summarize-only   # merge both arms
```
