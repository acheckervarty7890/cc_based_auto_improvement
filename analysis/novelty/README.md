# Red-team novelty — is "too far from eval" what hurts eval scores?

The question this answers: red-team rows are known to sit in a different distribution
from the eval sets. Are there **regions** so far outside eval space that they hurt the
probe's eval score — such that removing them would improve it, or at worst leave it
unchanged because eval has no support there?

Everything runs **offline off the runs' cached activations** — the eval blobs, the dev
blob, the base blob and the per-conversation red-team blobs, for both concepts. No LLM
is ever loaded; a cache miss raises rather than silently forwarding a 27B model.

Report: `results/summary.md`

## Scope

Four arms — two concepts x two attacker models — which is what makes the cross-arm
comparison possible at all:

| concept | probe | arms | eval | red-team rows |
|---|---|---|--:|--:|
| instructions | gemma-3-27b L32, **10-member ensemble** | gptoss, nemotron | 1302 rows / 7 splits | 762 / 812 |
| highstakes | gemma-3-27b L32, single probe | gptoss120b, deepseekv4pro | 4408 rows / 4 splits | 672 / 758 |

## Layout

| file | what it does |
|---|---|
| `experiments.py` | the four arms and every path/parameter, all verified against the on-disk cache hashes |
| `loaders.py` | eval/dev/base/red-team loaders for both concepts; streaming mask-weighted pooling |
| `pool.py` | Phase 0 — every activation blob to one pooled vector per row |
| `novelty.py` | Phase 1 — six novelty scores + the probe-direction decomposition |
| `regions.py` | Phase 2 — HDBSCAN regions (and a k-means covering, because HDBSCAN finds little) |
| `ablate.py` | Phase 3 — remove, refit, score. The causal test |
| `report.py` | renders `results/summary.md` |
| `build_artifact.py` | renders `results/novelty_study.html` (all figures read from `results/`) |
| `run_all.sh` | the whole pipeline in order (one GPU) |

## Headline numbers

| | |
|---|--:|
| red-team rows outside the eval manifold | 29% – 84% depending on the arm |
| correlation, arm novelty vs the arm's published eval gain | **+0.95** (the sign the hypothesis does *not* predict) |
| share of a row's off-manifold displacement lying on the probe's decision axis | 0.017 – 0.053 |
| matched-n comparisons where novelty-targeted removal beat random | 5 of 16 — and it *lost* in 3 |
| cost of removing all red-team data, cross-attacker AUROC | **−0.069 to −0.131, in all four arms** |

Three conclusions, in order of how much weight they bear:

1. **Distance from eval does not predict harm.** The most off-manifold arms produced the
   largest eval gains. Within arms, targeting the most novel rows beats matched-n random
   removal on the two instructions arms and loses on high-stakes/deepseekv4pro. A signal
   that reverses between concepts is not a pruning rule.
2. **"Eval didn't move" is not permission to delete.** On both instructions arms,
   `drop_outside` — removing every row past the calibrated threshold — leaves eval inside
   the noise band while cross-attacker AUROC falls 0.043–0.052. Eval has no support where
   those rows live.
3. **The one consistent effect is the trade, not the geometry.** Removing all red-team data
   moves eval in both directions by concept (instructions −0.06/−0.09, high-stakes
   **+0.01/+0.11**) but costs 7–13 points of cross-attacker AUROC everywhere.

Two things that reframe the high-stakes "regression": its iteration-0 probe trains on 50
rows, under the 64-row optimizer-step floor, so it takes zero optimizer steps and is a
seeded random projection — which already scores **0.9247** macro AUROC on that eval. And
HDBSCAN finds no density clusters in either high-stakes arm: there is no compact bad region
to excise.

## Method notes that matter

- **Parameters are recovered, not assumed.** The base and dev activation blob names are
  content hashes of their inputs, so the run's real settings could be read back off
  disk: `seed=42` (the CLI default, *not* `retrain_probe`'s `seed=0` default),
  `test_size=0.0`, `combine_consecutive_messages=True`, `convert_tool_to_assistant=True`.
  All four arms hit 100% of their per-conversation red-team blobs under those transforms.

- **Pooling is a proxy, deliberately.** The probe pools with a softmax over per-token
  linear scores; the novelty scores use a flat mask-weighted mean. Using the probe's own
  pooling would make the analysis circular — a row would count as "novel" partly because
  the probe already mishandles it. The probe's view enters separately and explicitly, as
  the `along`/`orth` decomposition.

- **The novelty threshold is calibrated, not chosen.** `outside` means "further from eval
  than 95% of eval is from itself", so the cut comes from the eval set's own dispersion.
  Dev sits at 9.4% / 3.8% outside by that measure, which is what makes it a fair
  reference point for "inside the manifold".

- **Row order is a real variable.** `_activate_redteam_cached` emits cache hits before
  newly-computed rows, so a run's training row order encodes its box's cache history and
  is not recoverable from the snapshot (see `../ceiling/README.md`). Every condition here
  uses snapshot file order, so conditions are mutually comparable but none reproduces the
  published probe. **Compare conditions to `full`, never to the comparison CSV.** As a
  check, `full` for instructions/gptoss reproduces the ceiling study's file-order refit
  exactly (0.8272), and `base_only` reproduces the published iteration-0 score (0.7714).

- **High-stakes validates on a 400-row stratified subsample of dev.** Its dev blob is
  19.6 GiB; staged whole it fills the card and leaves the training set to be copied over
  PCIe every epoch, which measured at **over 6 minutes for a single member** — 20+ hours
  for the study. Subsampled, both the validation and training sets stay resident and a
  fit takes ~18 s. Validation here only selects the best epoch for a 5376-parameter
  linear head, which a few hundred split- and class-stratified rows do well. Every
  high-stakes condition validates on the *same* rows, so conditions remain comparable to
  one another — which is the only comparison this study makes. Instructions is unaffected
  and uses all 436 dev rows.

- **The noise floor is measured per arm**, from refits of identical data in permuted row
  order, and every delta is read against it. Differences inside the floor are reported as
  inert rather than as small effects.

- **The 64-row optimizer-step floor** (`batch_size 16` x `gradient_accumulation_steps 4`)
  means `base_only` (50 rows) takes *zero* optimizer steps on both concepts. It is
  flagged, not hidden.

## Why cross-attacker AUROC is reported alongside eval

The trap in the original question is the "at worst unchanged" branch. If removing a
far-out region leaves eval flat, that is **not** evidence the region was useless — it can
equally mean eval has no support there and simply cannot see what the region defends
against. Since each concept has two attacker arms sharing one activation cache, each
arm's refits are also scored on the *other* arm's red-team rows, which that arm never
trained on. A region whose removal leaves eval flat but drags cross-attacker AUROC down
is carrying robustness the eval set is blind to, and dropping it would be a silent
regression.

## Running it

```bash
bash analysis/novelty/run_all.sh          # everything, in order
```

Or one phase at a time (`--experiment instructions|highstakes`, `--arm` optional):

```bash
.venv_claude/bin/python analysis/novelty/pool.py     --experiment instructions
.venv_claude/bin/python analysis/novelty/novelty.py  --experiment instructions
.venv_claude/bin/python analysis/novelty/regions.py  --experiment instructions --method kmeans -k 6
.venv_claude/bin/python analysis/novelty/ablate.py   --experiment instructions
.venv_claude/bin/python analysis/novelty/report.py
```

`highstakes` is the heavy one: 46 GiB of eval blobs and a 19.6 GiB dev blob against a
24 GiB card and 62 GiB of host RAM. Pooling streams in row chunks; the ablation stages
the dev set on the GPU and leaves the training set host-resident (the ordering
`retrain.py` measured as cheapest), and scores each eval split once for all conditions
rather than once per condition.
