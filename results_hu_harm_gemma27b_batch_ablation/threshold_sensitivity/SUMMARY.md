# Threshold sensitivity of the red-team success rate on a vintage increment

**What this measures.** `v2_probe_on_new_v3.py` asked how often the rows iteration 3 added
are misclassified by the ten reseeded vintage-2 probes, at the deployed operating point.
`scripts/vintage_threshold_sensitivity.py` asks how that rate moves as the operating point
moves, and runs the same question one cycle earlier (`--vintage 1`: v1 probes on the rows
v2 adds). Probes are the **saved** fits in `results_hu_harm_vintage_cross_eval/probes/`,
not fresh refits; at `--vintage 2` the committed refit logits are loaded as a cross-check
and agree at **r = 1.0000, 100% sign agreement**. Everything runs off cached activations —
no gemma-3-27b forward pass.

The sweep is over the **logit** threshold `tau`, reported alongside `t = sigmoid(tau)`;
`tau = 0` is the deployed `predict_proba >= 0.5`. Logit space is used because bf16
`predict_proba` saturates to exactly 1.0 above logit ~5.5, so a probability grid cannot
separate rows a logit grid can. Direction is fixed by the judge's label: a
`false_positive` hunt succeeds iff `logit >= tau`, a `false_negative` hunt iff
`logit < tau`. **The two directions are monotone opposites, so no single threshold can
improve both** — which is why nothing here is reported pooled only. `eval acc` is accuracy
over the four `eval_dataset_hu_ha` splits (866 rows, each split exactly 50/50), mean over
the ten probes; on a balanced split that already *is* balanced accuracy.

## Deployed point (t = 0.5)

| increment | arm | rows | FP/FN | anchor wrong | pooled | FP-hunt | FN-hunt | eval acc |
|---|---|---|---|---|---|---|---|---|
| v1→v2 | gptoss120b | 97 | 60/37 | 87.6% | 70.3% | 81.3% | 52.4% | 74.8% |
| v1→v2 | deepseekv4pro | 179 | 115/64 | 100.0% | 92.4% | 88.6% | 99.2% | 69.1% |
| v2→v3 | gptoss120b | 116 | 34/82 | 91.4% | 71.0% | 84.4% | 65.5% | 80.4% |
| v2→v3 | deepseekv4pro | 86 | 15/71 | 80.2% | 65.8% | 73.3% | 64.2% | 72.4% |

`anchor wrong` is `probe_iter{k}.pkl` — the deployed probe these attacks actually beat —
on the same rows. It is a check on the reconstruction (labels, pairing, cached
activations, prediction rule), not a result; it falls short of 100% where
`filter_dataset` re-admitted older successes that never faced that probe.

## Can the pooled rate be raised? (paired over the same ten fits)

| increment | arm | tau* | pooled at tau* | paired Δ | sd | t | seeds ↑ | eval acc at tau* |
|---|---|---|---|---|---|---|---|---|
| v1→v2 | gptoss120b | −0.50 | 71.1% | +0.8pp | 3.3 | 0.79 | 5/10 | 75.6% |
| v1→v2 | deepseekv4pro | −3.25 | 96.5% | +4.1pp | 4.4 | 2.92 | 7/10 | 74.4% |
| v2→v3 | gptoss120b | +1.25 | 72.9% | +1.9pp | 8.1 | 0.74 | 6/10 | 78.8% |
| v2→v3 | deepseekv4pro | +28.75 | 82.4% | +16.6pp | 7.0 | 7.49 | 10/10 | 50.0% |

The contrast is **paired** — the same fit scored at two cuts — because the across-seed sd
(4–7 points) is far larger than the effect and would hide it either way. Two of the four
gains are inside noise. The largest is **degenerate**: `tau = +28.75` is the "answer
negative for everything" end, where eval accuracy is 50.0%, i.e. chance, and 82.4% is
just 71/86, the FN-hunt share of that cohort. Only deepseekv4pro's v1→v2 row is both real
and non-degenerate, and it *also* improves eval accuracy (69.1% → 74.4%).

## Best directional rate at no eval-accuracy cost

| increment | arm | tau | FP-hunt | eval acc | FN-hunt best (tau) | eval-optimal tau | eval acc there |
|---|---|---|---|---|---|---|---|
| v1→v2 | gptoss120b | −6.50 | 100.0% | 75.3% | 52.4% (0) | −5.00 | 81.7% |
| v1→v2 | deepseekv4pro | −7.75 | 100.0% | 69.4% | 99.2% (0) | −5.00 | 75.3% |
| v2→v3 | gptoss120b | −7.00 | 100.0% | 80.6% | 65.5% (0) | −4.75 | 86.4% |
| v2→v3 | deepseekv4pro | −8.75 | 89.3% | 75.1% | 64.2% (0) | −6.75 | 83.1% |

## What holds across both increments

- The deployed cut sits **5–8 logits below the eval optimum** in all four cells (per-seed
  eval-optimal tau: −5.03, −5.53, −5.42, −7.22), so recalibrating buys 5–7 points of eval
  accuracy. These probes **under-predict the positive class**, and that single fact
  explains the rest of the table: false-positive attacks are near-free, false-negative
  attacks are hard, and the rows that get misclassified are the positive-labelled ones —
  including the generated counterparts, which the v1 probes get wrong 30.6% / 27.2% of the
  time (their increment is >60% positive-labelled) against 0.7% / 0.6% for the v2 probes
  (theirs is >70% negative-labelled).
- **The false-positive direction has free headroom**: 100% (three cells) or 89.3% at eval
  accuracy equal to or better than deployed. The **false-negative direction is already at
  its no-cost maximum at t = 0.5** — raising tau helps it but always costs accuracy, so
  `tau = 0` is the constrained optimum for that direction in every cell.
- Local sensitivity is mild: −3.7 to +6.6 pp of success per unit of logit threshold, with
  9–27% of (row, seed) scores within 1 logit of the boundary. The headline rate is not
  knife-edge; it takes a move the size of the calibration error to change the picture.

## What differs between them

- **The FP/FN mix flips.** Iteration 1's rotations found mostly false positives (60/37,
  115/64); iteration 2's found mostly false negatives (34/82, 15/71). So the pooled curve
  slopes the other way and its optimum sits slightly negative at v1→v2 and slightly
  positive at v2→v3 — the deployed threshold is near-optimal in both, approached from
  opposite sides. The pooled rate is therefore partly a fact about the cohort, which is
  why `direction_balanced` (mean of the two directions, i.e. the pooled rate at a 50/50
  mix) is carried in the JSON alongside it.
- **deepseekv4pro's v1 probes are far more vulnerable than its v2 probes** — 92.4% vs
  65.8% pooled — driven by a total false-negative hole: 99.2% wrong, FN-hunt logits at
  p90 = −3.69 with only 0.8% of scores on the positive side, and sensitivity of +0.9 pp
  per logit. Those rows are nowhere near the boundary, so **no threshold reaches them**;
  that hole is in the weights, not in the cut.

## Files

- `threshold_sensitivity_v{k}_to_v{k+1}.json` — full 241-point curves per cohort
  (`source_all`, `source_fp_hunt`, `source_fn_hunt`, `source_found_at_vintage`,
  `generated`, `attacker_best_direction`, `direction_balanced`), per-split eval accuracy
  curves, per-seed rates at each operating point, local sensitivity, and the anchor check.
- `threshold_curves_v{k}_to_v{k+1}.csv` — the same curves flattened for plotting
  (`arm, tau, prob_threshold, cohort, rate`).

Reproduce with:

```bash
.venv_claude/bin/python scripts/vintage_threshold_sensitivity.py --vintage 2
.venv_claude/bin/python scripts/vintage_threshold_sensitivity.py --vintage 1
```

Each pass is ~30 s on a 15 GB / 8 GB-GPU box.
