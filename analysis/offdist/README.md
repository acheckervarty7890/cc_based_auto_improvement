# Off-distribution red-team samples

Branch: `experiment22_cloud`. Companion to `ceiling_analysis/`, which asked *how far* the
red-team-trained probes are from what eval-distribution training could reach. This asks
*which rows* are responsible.

## The questions

1. Can a portion of the red-team samples be identified as generally different from the eval
   set, by words / topic / structure?
2. Are some samples or pairs set up the **opposite** way to the eval pairs?
3. Does removing them improve eval AUROC?
4. Do they carry a specific signal in activation space?

## Scope and settings

Both experiment22 arms' final post-processed sets — the rows the last retrain actually
trained on: `hu_ha_dd_gptoss120b` (588 rows, 294 contrastive pairs) and
`hu_ha_dd_deepseekv4pro` (920 rows, 459 pairs). Eval is all four `eval_sets/hu_ha` splits
(866 rows).

Fits are the **ceiling analysis's** fit, reused unchanged: a single `linear_then_softmax`
head at seed 42, early-stopped on that study's reserved 25% dev validation slice. So `full`
here reproduces its N=0 point to four decimals (0.9164 / 0.8314) and every condition is
comparable to its curves. These are **not** comparable to the run's published comparison
CSVs, which are 10-member ensembles.

## Three decisions worth stating

**Removals are by contrastive pair, not by row.** The post-processed set is exactly class
balanced because every conversation appears with a generated opposite-label partner
(`generate_contrastive_dataset`). Dropping single rows would move the class balance along
with the flagged property and the refit would answer a different question. A pair is dropped
when either half is flagged. Pairs are recovered from the run's own
`contrastive_cache.jsonl` by re-deriving `_cache_key`, so nothing is regenerated — 294/294
and 459/460 pairs resolve.

**Every condition is matched against random removal of the same size**, three seeds. Without
it, "removing 30% changed the score" says nothing about *which* 30%. The control is keyed on
the size it was **asked** for, not the size it achieved: `matched_random` removes whole pairs
until it has removed at least the target, so one seed can land on 275 where another lands on
276, and keying on the achieved count scatters a 3-seed control into three 1-seed buckets
that each report a standard deviation of zero — a zero-width noise band, which marks
essentially everything as an effect.

**The eval labelling convention is measured, not assumed.** The refusal detector extends
`llm_judge._REFUSAL_MARKERS` with a regex for the softer forms an assistant uses and a judge
does not, and keeps `_strip_quoted_spans` verbatim — this concept's data is *about* refusal,
so a quoted refusal phrase is common and a plain substring scan reads the citation as the
utterance (the failure CLAUDE.md records). Its precision is then reported against the eval
labels: an eval conversation whose assistant refuses is labelled `harmful_to_human` 5.2% of
the time against 56.9% for one that does not. That measurement is what makes "opposite to the
eval convention" a threshold rather than a premise.

## Layout

```
analysis/offdist/
  od_common.py     arms, loading, pair recovery, refusal + structural features
  characterize.py  phase 1 - structure, TF-IDF discriminator, topics, convention
  ablate.py        phase 2 - removal conditions + matched random controls
  actsig.py        phase 3 - pooled activations, kNN novelty, probe-direction split
  report.py        results/SUMMARY.md
  results/         flags_*.jsonl, surface_*.json, ablation_*.jsonl, actsig_*.{json,npz}
```

Run order: `characterize.py` → `ablate.py` → `actsig.py` → `report.py`. Needs
`PROBE_FUSED_ENSEMBLE=0` and the ceiling analysis's `ceiling_acts/` prepared; no LLM is
loaded at any point and no activation is recomputed.

The written answer is in `results/SUMMARY.md`.
