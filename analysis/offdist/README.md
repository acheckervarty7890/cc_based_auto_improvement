# Off-distribution red-team samples

Branch: `experiment22_cloud`. Companion to `ceiling_analysis/`, which asked *how far* the
red-team-trained probes are from what eval-distribution training could reach. This asks
*which rows* are responsible — and, at the end, which *half* of each row's pair is.

## The questions

1. Can a portion of the red-team samples be identified as generally different from the eval
   set, by words / topic / structure?
2. Are some samples or pairs set up the **opposite** way to the eval pairs?
3. Does removing them improve eval AUROC?
4. Do they carry a specific signal in activation space?
5. What are the generated contrastive halves worth — i.e. what would the retrain have
   scored with no `preprocessing:` section at all? And is the answer about *who wrote*
   each half, or about the pairing itself?

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

## Four decisions worth stating

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

**Question 5 gets a different control, because it breaks the pairing on purpose.**
`drop_generated` keeps the conversations the attacker landed and drops the partners
`generate_contrastive_dataset` wrote for them; `drop_sources` is its mirror. Matching those
against `matched_random` would be wrong twice over — it removes whole pairs, so it preserves
both the pairing and the exact class balance, which are the two things being removed. They
are matched against `keep_random_half` instead: one randomly chosen side of every pair, same
row count, pairing broken just as hard, only the source-versus-generated choice left to
chance. Which half a row is comes from `characterize.py`'s `pair_role`, set from
`od_common.recover_pairs`, so the provenance is read off the run's own contrastive cache
rather than guessed from the text.

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
  ablate.py        phase 2 - removal conditions + matched random controls, and the
                   pair-provenance conditions of question 5 + their random-half control
  actsig.py        phase 3 - pooled activations, kNN novelty, probe-direction split,
                   per-eval-split and per-pair-role geometry
  rewrite_successes.py  question 5b - ask the run's own contrastive generator (gpt-5.1)
                   to re-express arm 1's attacker successes in its own voice, label held
                   fixed. Subclasses preprocessing._ContrastiveLLM so the breaker, the
                   outage retry schedule and the JSON parse are the run's own, not a
                   second implementation
  rewrite_ablate.py  extract those rewrites' activations into the shared per-conversation
                   cache, then refit and score them beside drop_generated
  pair_probe_scores.py  question 5c - score each GENERATED partner with the probe its
                   source was submitted against (the only half of a pair that never got a
                   verdict), with the source side as a control that must read 0%
  handwritten_check.py  question 6 - 25 hand-written variations of arm-1 successes
                   (results/handwritten_gptoss120b.jsonl), scored by probe_iter0 and the
                   final probe_iter5, to tell memorised strings from a patched boundary
  before_after_scores.py  every success and partner scored by probe_iter0 (before any
                   red-team training, both halves out-of-sample) vs probe_iter5 (after,
                   both in-sample) — the flat before/after, split by half and true class
  report.py        results/SUMMARY.md
  results/         flags_*.jsonl, surface_*.json, ablation_*.jsonl, actsig_*.{json,npz},
                   rewritten_*.jsonl + rewrite_cache_*.jsonl + rewrite_stats_*.json,
                   pair_probe_scores_*.json,
                   handwritten_*.jsonl + handwritten_scores_*.json, before_after_scores_*.json
```

Run order: `characterize.py` → `ablate.py` → `actsig.py` → `report.py`, with
`rewrite_successes.py` → `rewrite_ablate.py` as an optional branch after `ablate.py`
(the only step that needs an API key, and the only one that loads the 27B model —
its 294 new conversations have no cached activations by construction). Needs
`PROBE_FUSED_ENSEMBLE=0` and the ceiling analysis's `ceiling_acts/` prepared; no LLM is
loaded at any point and no activation is recomputed.

The written answer is in `results/SUMMARY.md`, and the same numbers as a standalone page at
https://claude.ai/code/artifact/e1399fd2-3adf-4b1c-bb0d-17d3a90b3a21 (regenerate with
`build_artifact.py` and republish the same path to update it in place).
