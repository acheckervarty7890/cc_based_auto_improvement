# Pooling both attackers' red-team data — HIGH-STAKES (experiment9)
_Updated 2026-08-16T22:48:23+00:00_

**What this measures.** The sweep in `../vintage/` fits each attacker arm separately. This one adds a third arm, **`combined`**, whose training set at each vintage is the base training data (`data/hs_ls_200.jsonl`) plus the *union* of both arms' iteration-3 red-team pairs of that vintage. Vintages are cumulative, so `combined` v3 is the base data plus every iteration-3 pair either attacker produced — the "v1 + v2 + v3, both attackers" set.

Everything except the row set is held at the single-arm sweep's values: the same per-conversation cached activations, the same content-deterministic train/val split, the same `ProbeSpec` (verified identical between the two arms' `probe_iter3.pkl`), the same ten seeds, the same four `eval_datasets/` splits, the same `--drop-overlong pair`. So the `combined` row at vintage k is comparable to the two single-arm rows at vintage k, and the only thing that differs is which conversations are in the training set.

**Vintages**
- `v0` — base training data only, no red-team rows
- `v1` — iter-3 pairs whose source success existed at iteration 1
- `v2` — …existed at iteration 2
- `v3` — all iteration-3 pairs — i.e. v1 + v2 + v3

**v0 is one fit, not three.** With no red-team rows the training set does not depend on the arm, so v0 is reported once. The combined sweep fits it anyway as a cross-check: it reproduces the single-arm sweep's v0 AUROC **exactly** (max |Δ| = 0 across every seed and split), which is what licenses reading the `combined` rows as continuous with the others — the pooled indexing and the disk-backed assembly this sweep needs are not perturbing the fit path.

**Read the sd, not just the mean.** These are unpaired refits with independent initialisations. Every gap below is quoted against the pooled seed sd of the two cells compared, and only >= 2 sigma is treated as a result.

## Progress: 30 combined fits recorded

- **combined**: v1×10, v2×10, v3×10

## Eval AUROC — mean ± sd over seeds (pipeline scale)

| vintage | arm | rows | seeds | anthropic | mt | mts | toolace | mean |
|---|---|---|---|---|---|---|---|---|
| v0 | _base only_ | 0 | 10 | 0.9531 ± 0.0044 | 0.8673 ± 0.0495 | 0.9436 ± 0.0094 | 0.7080 ± 0.0252 | 0.8680 ± 0.0121 |
| v1 | deepseekv4pro | 294 | 10 | 0.9282 ± 0.0142 | 0.9470 ± 0.0079 | 0.8857 ± 0.0144 | 0.7655 ± 0.0141 | 0.8816 ± 0.0063 |
| v1 | gptoss120b | 228 | 10 | 0.9552 ± 0.0067 | 0.9599 ± 0.0117 | 0.9402 ± 0.0066 | 0.7520 ± 0.0213 | 0.9018 ± 0.0084 |
| v1 | **combined** | 522 | 10 | 0.9362 ± 0.0147 | 0.9695 ± 0.0050 | 0.9300 ± 0.0044 | 0.7624 ± 0.0186 | 0.8995 ± 0.0053 |
| v2 | deepseekv4pro | 542 | 10 | 0.9641 ± 0.0025 | 0.9331 ± 0.0067 | 0.9425 ± 0.0160 | 0.8190 ± 0.0075 | 0.9147 ± 0.0054 |
| v2 | gptoss120b | 422 | 10 | 0.9643 ± 0.0040 | 0.9701 ± 0.0060 | 0.9625 ± 0.0037 | 0.8156 ± 0.0103 | 0.9281 ± 0.0048 |
| v2 | **combined** | 964 | 10 | 0.9635 ± 0.0013 | 0.9512 ± 0.0092 | 0.9506 ± 0.0069 | 0.8262 ± 0.0058 | 0.9229 ± 0.0042 |
| v3 | deepseekv4pro | 716 | 10 | 0.9539 ± 0.0028 | 0.9127 ± 0.0097 | 0.9275 ± 0.0179 | 0.7708 ± 0.0093 | 0.8912 ± 0.0064 |
| v3 | gptoss120b | 562 | 10 | 0.9682 ± 0.0019 | 0.9748 ± 0.0058 | 0.9653 ± 0.0039 | 0.8220 ± 0.0093 | 0.9326 ± 0.0032 |
| v3 | **combined** | 1278 | 10 | 0.9596 ± 0.0031 | 0.9450 ± 0.0054 | 0.9399 ± 0.0089 | 0.7993 ± 0.0101 | 0.9109 ± 0.0054 |

## Pooled against each arm alone

| vintage | arm | its rows | arm mean | pooled rows | combined mean | gap | sigma | verdict |
|---|---|---|---|---|---|---|---|---|
| v1 | deepseekv4pro | 294 | 0.8816 ± 0.0063 | 522 | 0.8995 ± 0.0053 | +0.0179 | 2.2 | **above** |
| v1 | gptoss120b | 228 | 0.9018 ± 0.0084 | 522 | 0.8995 ± 0.0053 | -0.0023 | 0.2 | indistinguishable |
| v2 | deepseekv4pro | 542 | 0.9147 ± 0.0054 | 964 | 0.9229 ± 0.0042 | +0.0082 | 1.2 | indistinguishable |
| v2 | gptoss120b | 422 | 0.9281 ± 0.0048 | 964 | 0.9229 ± 0.0042 | -0.0052 | 0.8 | indistinguishable |
| v3 | deepseekv4pro | 716 | 0.8912 ± 0.0064 | 1278 | 0.9109 ± 0.0054 | +0.0197 | 2.3 | **above** |
| v3 | gptoss120b | 562 | 0.9326 ± 0.0032 | 1278 | 0.9109 ± 0.0054 | -0.0216 | 3.4 | **below** |

`sigma` is the gap over the pooled seed sd of the two cells. "indistinguishable" means the seed noise covers it — not that the two training sets are equivalent, only that this sweep cannot separate them.

## Read-out

- **combined**: v1 0.8995 → v2 0.9229 → v3 0.9109; best is **v2**.
