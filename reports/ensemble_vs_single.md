# Ensemble vs single probe — what the 10 members buy, and what they cost

_Generated 2026-08-24 10:25:59Z by `scripts/ensemble_vs_single.py` from the existing run outputs. Nothing is refitted._

24 paired cells: 4 generators x 3 concepts x 2 validation modes. `seq_ens10` is a 10-member score-averaging deep ensemble fit sequentially (`PROBE_FUSED_ENSEMBLE=0`); `single` is one probe. Every score is mean AUROC over that concept's eval splits.

## Headline

| | |
| --- | --- |
| mean delta (ens - single) | **+0.0095** |
| median delta | +0.0127 |
| range | -0.055 to +0.051 |
| ensemble wins | 16/24 |
| cells moving more than 0.02 | 8/24 |
| total fit time, single | 1150 s = 19.2 min |
| total fit time, seq_ens10 | 8554 s = 142.6 min |
| cost ratio | **7.4x** |

About +0.01 AUROC for 7x the fit time, with the sign unreliable — a third of cells go the wrong way. Against the effects this experiment actually measures (0.22 between generators on `instructions`, 0.28 between concepts), ensembling is noise.

## Scores

```
config                              seq_ens10  single  delta
generator    concept      val_mode                          
dsv4pro      highstakes   dev           0.843   0.792  0.051
                          split         0.839   0.818  0.021
             hu_ha        dev           0.888   0.888  0.000
                          split         0.898   0.871  0.027
             instructions dev           0.579   0.570  0.009
                          split         0.588   0.562  0.026
llama70b     highstakes   dev           0.918   0.900  0.019
                          split         0.877   0.895 -0.018
             hu_ha        dev           0.846   0.852 -0.006
                          split         0.800   0.855 -0.055
             instructions dev           0.771   0.778 -0.006
                          split         0.813   0.825 -0.012
llama8b      highstakes   dev           0.862   0.874 -0.012
                          split         0.841   0.828  0.013
             hu_ha        dev           0.866   0.851  0.015
                          split         0.855   0.871 -0.017
             instructions dev           0.649   0.633  0.017
                          split         0.719   0.678  0.041
nemotron550b highstakes   dev           0.852   0.835  0.017
                          split         0.849   0.811  0.038
             hu_ha        dev           0.875   0.875 -0.001
                          split         0.878   0.876  0.002
             instructions dev           0.586   0.574  0.012
                          split         0.621   0.574  0.047
```

### By concept

```
                mean     min     max
concept                             
highstakes    0.0160 -0.0180  0.0509
hu_ha        -0.0043 -0.0548  0.0272
instructions  0.0167 -0.0118  0.0473
```

### By validation mode

```
            mean     min     max
val_mode                        
dev       0.0095 -0.0115  0.0509
split     0.0094 -0.0548  0.0473
```

## Fit wall-clock (seconds)

```
config                              seq_ens10  single  ratio
generator    concept      val_mode                          
dsv4pro      highstakes   dev          2651.0   302.0    8.8
                          split          12.0     2.0    6.0
             hu_ha        dev            77.0    15.0    5.1
                          split          12.0     1.0   12.0
             instructions dev            55.0     8.0    6.9
                          split          13.0     2.0    6.5
llama70b     highstakes   dev          2137.0   266.0    8.0
                          split          13.0     1.0   13.0
             hu_ha        dev            54.0     7.0    7.7
                          split          11.0     1.0   11.0
             instructions dev            88.0    10.0    8.8
                          split          13.0     1.0   13.0
llama8b      highstakes   dev           588.0    95.0    6.2
                          split          10.0     1.0   10.0
             hu_ha        dev            67.0    16.0    4.2
                          split          12.0     1.0   12.0
             instructions dev            64.0     9.0    7.1
                          split          12.0     1.0   12.0
nemotron550b highstakes   dev          2467.0   377.0    6.5
                          split          13.0     2.0    6.5
             hu_ha        dev            77.0    19.0    4.1
                          split          13.0     1.0   13.0
             instructions dev            82.0    11.0    7.5
                          split          13.0     1.0   13.0
```

The ratio is sub-linear rather than 10x because `_to_device_for_fit` stages the activations, and the dev blob is read from disk, **once per `build_ensemble` call** rather than once per member — the single-probe fit pays that fixed cost against one fit, the ensemble amortizes it over ten. It is largest on the `split` arms, where the fits themselves are 1-2 s and the fixed cost dominates both sides.

## Two caveats before reading the delta column as an ensemble effect

**The single probe is not member 0 of the ensemble.** `retrain._resolve_ensemble_seeds` carves out `n == 1` and returns `[--seed]` (42), while `n > 1` uses the repo-pinned `ENSEMBLE_SEEDS[:10]`. The two sides are different draws, so each cell's delta mixes averaging with a change of seed, and some of the +-0.05 scatter is seed noise.

**It does not buy stability either.** Mean `|dev - split|` per (generator, concept) is 0.0246 for the ensemble against 0.0202 for the single probe — if anything slightly worse, though within noise. The usual argument for a deep ensemble, that averaging damps sensitivity to arbitrary choices, does not show up here.

```
             mean     max
config                   
seq_ens10  0.0246  0.0704
single     0.0202  0.0472
```

## Where it looks least like noise

The `split` arms of `highstakes` and `instructions` — the cells with only ~40 training rows, where a single fit is least stable and averaging has the most to fix — average +0.019, 6/8 positive. That is roughly double the overall mean, and it is the only slice with a defensible story behind it. It is still not a clean result: llama70b moves the wrong way on both of its cells, so even here the effect does not hold for every generator.

```
config                              seq_ens10  single  delta
generator    concept      val_mode                          
dsv4pro      highstakes   split         0.839   0.818  0.021
             instructions split         0.588   0.562  0.026
llama70b     highstakes   split         0.877   0.895 -0.018
             instructions split         0.813   0.825 -0.012
llama8b      highstakes   split         0.841   0.828  0.013
             instructions split         0.719   0.678  0.041
nemotron550b highstakes   split         0.849   0.811  0.038
             instructions split         0.621   0.574  0.047
```
