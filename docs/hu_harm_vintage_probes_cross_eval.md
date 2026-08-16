# The hu_harm vintage probes, re-fitted and scored on two other concepts

The 80 probes behind `results_hu_harm_gemma27b_batch_ablation/vintage/SUMMARY.md` were
refitted (that sweep recorded metrics only and discarded the probes), kept on disk, and
scored on the **high-stakes** and **instruction-following** eval sets as well as their own.

- Script: `scripts/vintage_probes_cross_eval.py` (`fit` → `score` → `report`)
- Probes: `results_hu_harm_vintage_cross_eval/probes/{arm}_v{vintage}_s{seed}.pkl` (80 × ~13 KB)
- Rows: `results_hu_harm_vintage_cross_eval/cross_eval_rows.csv` (1200 = 15 splits × 80 probes)

## What the 80 probes are

2 attacker arms × 4 vintages × 10 seeds (42–51). Every one is a `harmful_to_human` probe
on `google/gemma-3-27b-it` layer 32, `linear_then_softmax` — same architecture, same
activations, same base data (`data/hu_harm_llama70b_50.jsonl`, 50 rows → 42 train / 8 val).

Vintages are **not** `probe_iter0..3`. All four carry *iteration 3's* content and
hyperparameters; only membership varies. Vintage k = every iteration-3 pair whose
originating red-team success already existed at iteration k.

| vintage | gptoss120b rows | deepseekv4pro rows |
|---|---|---|
| v0 (base data only) | 0 | 0 |
| v1 | 356 | 368 |
| v2 | 546 | 706 |
| v3 (full iter-3 set) | 778 | 878 |

Membership is imported from `attribution_vintage.vintages()` rather than reimplemented, so
these are the same sets by construction. The row counts reproduce SUMMARY.md exactly.

## The refits reproduce SUMMARY.md to 1e−4

| arm | vintage | new mean ± sd | SUMMARY mean ± sd | Δmean |
|---|---|---|---|---|
| deepseekv4pro | v0 | 0.4746 ± 0.0634 | 0.4747 ± 0.0633 | −0.0001 |
| deepseekv4pro | v1 | 0.8418 ± 0.0140 | 0.8419 ± 0.0140 | −0.0001 |
| deepseekv4pro | v2 | 0.8851 ± 0.0242 | 0.8852 ± 0.0242 | −0.0001 |
| deepseekv4pro | v3 | 0.8827 ± 0.0271 | 0.8828 ± 0.0270 | −0.0001 |
| gptoss120b | v0 | 0.4746 ± 0.0634 | 0.4747 ± 0.0633 | −0.0001 |
| gptoss120b | v1 | 0.8819 ± 0.0090 | 0.8819 ± 0.0090 | 0.0000 |
| gptoss120b | v2 | 0.9109 ± 0.0106 | 0.9110 ± 0.0106 | −0.0001 |
| gptoss120b | v3 | 0.9101 ± 0.0113 | 0.9101 ± 0.0114 | 0.0000 |

Both the means **and the seed sds** match to four decimals. I expected agreement within the
sd column, not this — but `refit` calls `seed_everything(seed)` before `ProbeFactory.build`,
so a fit is fully determined by its seed and these are the *same* probes, not merely
statistically equivalent ones. The residual 1e−4 is the fused scoring pass (below), not the
training. That makes the transfer numbers below attributable to the same probes SUMMARY.md
describes.

## Transfer: mean AUROC over each eval set's splits (pipeline scale, mean ± sd over 10 seeds)

| eval set | arm | v0 | v1 | v2 | v3 |
|---|---|---|---|---|---|
| **hu_ha** (own concept) | deepseekv4pro | 0.4746 ± .063 | 0.8418 ± .014 | **0.8851** ± .024 | 0.8827 ± .027 |
| | gptoss120b | 0.4746 ± .063 | 0.8819 ± .009 | **0.9109** ± .011 | 0.9101 ± .011 |
| **hs** (high-stakes) | deepseekv4pro | 0.5122 ± .079 | 0.5715 ± .028 | 0.6362 ± .020 | **0.6586** ± .012 |
| | gptoss120b | 0.5122 ± .079 | 0.6486 ± .019 | 0.6753 ± .008 | **0.6756** ± .020 |
| **instructions** | deepseekv4pro | 0.4969 ± .035 | **0.2901** ± .023 | 0.3362 ± .029 | 0.3081 ± .022 |
| | gptoss120b | 0.4969 ± .035 | 0.2767 ± .035 | 0.2776 ± .017 | **0.2673** ± .020 |

AUROC is computed against **each eval directory's own positive class** (`high-stakes`,
`assistant_follows_the_instruction`), not the probe's. A value below 0.5 is a real result —
the harm score ranking that concept's *negative* class higher — and nothing here flips it.

### High-stakes: real but modest positive transfer

v0 sits at chance (0.512 ± 0.079 — the base-only probe is 42 training rows and has no signal
on any concept). Harm red-team data lifts it to **0.66–0.68**, monotonically in vintage, and
the seed sd collapses from 0.079 to 0.008–0.020: the probes are not just better on average,
they are *consistently* better. That is about half the on-concept gain (0.47 → 0.91).

Per split, the transfer is concentrated: `mt` 0.53 → 0.77 and `anthropic` 0.52 → 0.72 on the
gptoss arm, while `toolace` barely moves (0.48 → 0.57). `toolace` is also the hardest split
for the purpose-built high-stakes probes, so this is the same split resisting both.

### Instruction-following: strong *negative* transfer, and it is not noise

v0 is at chance (0.497). Every red-team vintage pushes it to **0.27–0.34**, i.e. the harm
probe systematically scores instruction-*violating* responses as more harmful. Inverted, that
is 0.66–0.73 discriminative power — the probe is genuinely informative about this concept,
with the sign reversed.

The effect is concentrated in exactly the splits where violating the instruction is itself
harm-laden (gptoss arm, v3):

| split | AUROC | what the negative class is |
|---|---|---|
| `bbq_substitution` | 0.101 | answers with a stereotype-consistent wrong answer |
| `hc_context_drift` | 0.107 | drifts off the provided source |
| `hc_contradiction` | 0.147 | contradicts the provided source |
| `mm_substitution` | 0.181 | substitutes different content |
| `anthropic_harmless_refusal` | 0.373 | refuses |
| `oig_context_drift` | 0.470 | drifts off topic |
| `oig_omission` | 0.492 | omits requested content |

The plausible reading — stated as interpretation, not measurement — is that the probe is
still doing its job: a stereotype-consistent answer (bbq) or a claim contradicting the source
(hc) is *both* an instruction violation and plausibly harmful, so a harm detector ranks it
high. The two `oig_*` splits, whose failure modes are drifting and omitting rather than
asserting something bad, sit at chance — which is what that reading predicts.

**The practical caution:** a harm probe deployed as a general "response quality" signal would
be actively misleading here. It does not merely fail to detect instruction-following, it
ranks it backwards, and confidently — the sd is 0.02.

## Method notes

**No gemma-3-27b forward passes.** All three concepts' probes are L32 on the same model, and
an activation blob is a function of the conversation/model/layer, not the probe — so the
already-cached hs and instructions eval blobs are directly reusable, and all three runs load
their splits under the same two message transforms.

**Two phases, for memory.** `fit` holds the assembly (base + all iteration-3 red-team
activations, ~19 GB RSS) and touches no eval blob; `score` drops it and walks splits outer,
probes inner. Scoring inside the fit loop would have held 19 GB alongside the 11.3 GB
`anthropic` blob, over this box's cgroup ceiling (a process was OOM-killed at 24.3 GB the
day before).

**One fused pass per split.** `_classifier.logits` re-reads the whole activation tensor per
probe — ~4 min for `anthropic`, so ~5.3 h for 80 probes on that split alone.
`multi_probe_logits` evaluates all 80 heads in one traversal (`h @ W` with W of shape
(5376, 80), then each head's softmax pooling along the token axis). `anthropic` took **26 s**
for all 80. Whole scoring phase: ~70 s for 1200 evaluations.

**Precision is load-bearing, and cost me a false start.** The first fused pass ran fp32 and
disagreed with tuberlens by 1.3e−01 in logit space. These probes are stored *and evaluated*
in **bf16**, whose ULP at these logit magnitudes is ~0.1; `attribution_lib`'s docstring
records that an fp32 recomputation is a different scale, not a more precise one, and moves
`eval_ant_hh` by ~0.006. SUMMARY.md came from the bf16 path, so `multi_probe_logits` defaults
to bf16. Verified against `_classifier.logits` on all four hu_ha splits: max |ΔAUROC| 5.6e−4
(bf16) — well inside the seed sd of 0.012–0.07, and visible as the 1e−4 residual in the
reproduction table.

**Cost.** 80 fits, 6.02 h on the dev box (deepseekv4pro ~22 min per seed-curve, gptoss120b
~15). Scoring, ~70 s.

## Reproducing

```bash
.venv_claude/bin/python scripts/vintage_probes_cross_eval.py fit          # ~6 h, resumable
.venv_claude/bin/python scripts/vintage_probes_cross_eval.py score --verify-forward
.venv_claude/bin/python scripts/vintage_probes_cross_eval.py report
```

Both phases checkpoint per unit and resume, so an interruption costs one fit rather than the
run. The hs and instructions eval blobs must be present — `scripts/eval_probe_versions.py`
fetches them from Kaggle (see `docs/probe_versions_hs_instructions_eval.md`).
