# Red-team data on its own — no base training data
_Updated 2026-08-16T09:33:34+00:00_

**What this measures.** The vintage sweep (`SUMMARY.md`) always trains on `data/instructions_llama70b_50.jsonl` **plus** a vintage of iteration-3 red-team pairs, which is what the pipeline does. Each row here is the same real `ProbeFactory` refit with the **base training data removed entirely** — from the fit *and* from validation — so the probe sees red-team conversations and nothing else. Two conditions, ten seeds each, per arm:

- **`v2`** — every iteration-3 pair whose source success already existed at iteration 2.
- **`v3only`** — only the pairs that first appear at iteration 3 (`keep[3] − keep[2]`).

`v3only` is exactly the set `vintage_holdout_success.py` puts *in front of* the v2 probes; here it is what the probe is trained *from*.

**Setup.** Identical to the vintage sweep in every other respect: membership is derived per source success so pairs stay atomic and each set stays 50/50, the over-1024-token filter is on (`--drop-long pair`), each row's train/val side is the run's own content-deterministic split, all activations come off disk (no gemma-3-27b forward pass), and AUROC is the pipeline scale — bf16 `sigmoid` then sklearn, as `get_performances` reports it.

**One caveat on reading these as a training recipe.** With the base data gone the validation set is red-team rows only, so early stopping is judged on the same distribution the probe is fitted on. A high number here means *these rows alone separate the concept well enough to generalise to the eval splits*; it is not a claim that dropping the base data is a better way to train.

## Progress

- **gptoss120b**: v2×10, v3only×10
- **nemotron**: v2×10, v3only×10

## gptoss120b — mean ± sd over seeds (pipeline scale)

| training set | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| base only (no red-team) | 0 | 10 | 0.4549 ± 0.2438 | 0.4951 ± 0.0629 | 0.4972 ± 0.0273 | 0.4962 ± 0.0317 | 0.5342 ± 0.1264 | 0.5226 ± 0.0629 | 0.4802 ± 0.0321 | 0.4972 ± 0.0349 |
| v2 + base | 674 | 10 | 0.6802 ± 0.1063 | 0.8930 ± 0.0230 | 0.7384 ± 0.1049 | 0.8998 ± 0.0221 | 0.8794 ± 0.0415 | 0.5852 ± 0.0319 | 0.7189 ± 0.0505 | 0.7707 ± 0.0233 |
| v3 + base | 858 | 10 | 0.8298 ± 0.0521 | 0.8908 ± 0.0300 | 0.7564 ± 0.0548 | 0.8980 ± 0.0193 | 0.8581 ± 0.0284 | 0.5872 ± 0.0260 | 0.7748 ± 0.0190 | 0.7993 ± 0.0228 |
| **v2, no base** | 674 | 10 | 0.6834 ± 0.0898 | 0.8973 ± 0.0226 | 0.7082 ± 0.0758 | 0.8771 ± 0.0306 | 0.8340 ± 0.0356 | 0.5869 ± 0.0434 | 0.6999 ± 0.0423 | 0.7553 ± 0.0211 |
| **v3only, no base** | 184 | 10 | 0.5660 ± 0.1829 | 0.8597 ± 0.0659 | 0.7786 ± 0.0653 | 0.9001 ± 0.0484 | 0.9038 ± 0.0325 | 0.6938 ± 0.0584 | 0.6418 ± 0.0809 | 0.7634 ± 0.0208 |

## nemotron — mean ± sd over seeds (pipeline scale)

| training set | rows | seeds | anthropic_harmless_refusal | bbq_substitution | hc_context_drift | hc_contradiction | mm_substitution | oig_context_drift | oig_omission | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| base only (no red-team) | 0 | 10 | 0.4549 ± 0.2438 | 0.4951 ± 0.0629 | 0.4972 ± 0.0273 | 0.4962 ± 0.0317 | 0.5342 ± 0.1264 | 0.5226 ± 0.0629 | 0.4802 ± 0.0321 | 0.4972 ± 0.0349 |
| v2 + base | 630 | 10 | 0.8113 ± 0.0539 | 0.9542 ± 0.0098 | 0.8776 ± 0.0214 | 0.7661 ± 0.0498 | 0.8832 ± 0.0739 | 0.6985 ± 0.0283 | 0.6269 ± 0.0610 | 0.8025 ± 0.0346 |
| v3 + base | 926 | 10 | 0.7345 ± 0.0583 | 0.9174 ± 0.0238 | 0.7690 ± 0.0319 | 0.8416 ± 0.0532 | 0.9266 ± 0.0260 | 0.6750 ± 0.0297 | 0.6340 ± 0.0519 | 0.7854 ± 0.0222 |
| **v2, no base** | 630 | 10 | 0.7780 ± 0.0543 | 0.9585 ± 0.0162 | 0.8462 ± 0.0380 | 0.7777 ± 0.0528 | 0.9075 ± 0.0434 | 0.6880 ± 0.0131 | 0.6026 ± 0.0502 | 0.7941 ± 0.0185 |
| **v3only, no base** | 296 | 10 | 0.4004 ± 0.0621 | 0.7124 ± 0.0284 | 0.6624 ± 0.0487 | 0.6339 ± 0.0571 | 0.7920 ± 0.0444 | 0.6097 ± 0.0195 | 0.5816 ± 0.0216 | 0.6275 ± 0.0217 |

## What the base data is worth

| arm | v2 + base | v2 alone | delta |
|---|---|---|---|
| gptoss120b | 0.7707 | 0.7553 | -0.0154 |
| nemotron | 0.8025 | 0.7941 | -0.0085 |

The 50 base samples are worth roughly nothing on top of the red-team set — both deltas are inside one seed sd. That is consistent with the sweep's `v0` row, where the base data **alone** scores 0.4972 ± 0.0349, i.e. chance: this concept's eval signal comes from the red-team conversations, not from `instructions_llama70b_50`.

## What iteration 3's own rows are worth

| arm | rows | v3only alone | v2 alone | v3 + base |
|---|---|---|---|---|
| gptoss120b | 184 | 0.7634 | 0.7553 | 0.7993 |
| nemotron | 296 | 0.6275 | 0.7941 | 0.7854 |

**The two arms split completely here, and in the direction the AUROC sweep did not predict.** gptoss120b's 92 iteration-3 pairs — 184 rows, a quarter of its training set — reach the same eval AUROC on their own as its whole 674-row v2 set. nemotron's 148 pairs are *more* rows (296) and score 0.166 lower than its own v2 set, about eight seed sd: taken alone they do not span the concept.

**That resolves the tension left open in `SUMMARY.md`.** The held-out measurement found nemotron's iteration-3 finds fool the v2 probes far more reliably than gptoss120b's (0.707 vs 0.477 per-seed success) and yet *lower* mean eval AUROC when trained on. The two candidate explanations there were "they concentrate in a region the eval splits do not sample" and "they resemble each other more than they resemble anything else"; both predict what is measured here, and the alternative — that they carry broad signal which merely fails to add to what v2 already has — does not. Being a reliable probe failure and being useful training data are different properties, and nemotron's third rotation bought the first without the second.

**Per split, the iteration-3 rows are a different concept, not a weaker one (gptoss120b).** Trained on those 184 rows alone the probe *beats* every base-inclusive condition on `oig_context_drift` (0.694 vs 0.585/0.587) and `mm_substitution` (0.904 vs 0.879/0.858), and loses badly on `anthropic_harmless_refusal` (0.566 ± 0.183 — the sd says the seeds disagree about the direction) and `oig_omission` (0.642 vs 0.719/0.775). This is the same reallocation the vintage sweep saw across v1→v3, seen without the earlier rows present to mask it.

**nemotron's iteration-3 rows are anti-correlated on one split.** `anthropic_harmless_refusal` scores 0.400 ± 0.062 — below chance at every seed, not a draw that happened to land low. Whatever those 148 pairs encode about refusal is the opposite of what that split labels.

## Files

- `redteam_only_progress.jsonl` — one row per fit (append-only, resumable)
- `redteam_only_auroc.csv` — per fit × split
- `redteam_only_summary.csv` — mean/sd/min/max per (arm, condition, split)

## Reproducing

```bash
AGENTIC_FAST_ACTS=1 .venv_claude/bin/python scripts/redteam_only_fits.py \
  --arm gptoss120b nemotron --conditions v2 v3only --seeds 10
.venv_claude/bin/python scripts/redteam_only_summary_md.py
```
