## Sizes

| arm | rows | v2 | new-in-v3 | successes | durable | couples |
|---|---|---|---|---|---|---|
| gptoss120b | 778 | 546 | 232 | 116 | 44 | 389 |
| deepseekv4pro | 878 | 706 | 172 | 86 | 36 | 439 |

## Acceptance test

`pairAUR` = AUROC of similarity predicting *same label*, on the new-in-v3 rows (held out from every fit). `scenAUR` = AUROC separating a source's own counterpart from an unrelated row — whether the metric sees content at all. `provAUR` = the same test for *authorship*, the doc's design smell. `hopPair` = AUROC separating a source's own opposite-label counterpart from that same source's nearest v2 row; **low is good** (the counterpart must look farther). `kNN15aur` is AUROC over the fraction of positive neighbours, because the success sets are 71% / 83% positive rather than class-balanced. `durAUR` = AUROC of distance-to-v2 predicting a durable hole.


### gptoss120b

| metric | dim | pairAUR | scenAUR | provAUR | hopPair | nnBal | kNN15aur | durAUR (CI95) |
|---|---|---|---|---|---|---|---|---|
| `pool:mean` | 5376 | 0.540 | 0.900 | 0.615 | 0.770 | 0.482 | 0.419 | 0.684 (0.58–0.78) |
| `pool:last` | 5376 | 0.650 | 0.472 | 0.723 | 0.144 | 0.720 | 0.869 | 0.722 (0.62–0.82) |
| `pool:last32` | 5376 | 0.584 | 0.686 | 0.673 | 0.342 | 0.469 | 0.528 | 0.675 (0.57–0.77) |
| `pool:probe` | 5376 | 0.588 | 0.630 | 0.649 | 0.483 | 0.396 | 0.138 | 0.667 (0.56–0.76) |
| `pool:topz16` | 5376 | 0.508 | 0.998 | 0.490 | 0.992 | 0.456 | 0.455 | 0.606 (0.50–0.71) |
| `lin:centered` | 5376 | 0.565 | 0.916 | 0.708 | 0.729 | 0.507 | 0.379 | 0.639 (0.55–0.74) |
| `lin:whitened` | 5376 | 0.588 | 0.980 | 0.685 | 0.863 | 0.436 | 0.528 | 0.654 (0.56–0.75) |
| `lin:pcawhite` | 128 | 0.511 | 0.936 | 0.512 | 0.876 | 0.552 | 0.365 | 0.640 (0.54–0.73) |
| `unsup:pca64` | 64 | 0.559 | 0.831 | 0.617 | 0.583 | 0.617 | 0.463 | 0.682 (0.58–0.79) |
| `probe:logit` | 1 | 0.594 | 0.359 | 0.586 | 0.006 | 0.188 | 0.114 | 0.330 (0.22–0.44) |
| `probe:proj` | 1 | 0.584 | 0.460 | 0.528 | 0.009 | 0.316 | 0.152 | 0.504 (0.38–0.61) |
| `probe:wscaled` | 5376 | 0.544 | 0.892 | 0.614 | 0.743 | 0.518 | 0.477 | 0.659 (0.55–0.76) |
| `probe:jac` | 5376 | 0.629 | 0.513 | 0.634 | 0.319 | 0.352 | 0.229 | 0.578 (0.46–0.68) |
| `sup:lda` | 1 | 0.735 | 0.285 | 0.609 | 0.000 | 0.620 | 0.615 | 0.578 (0.47–0.68) |
| `sup:winwhite` | 128 | 0.564 | 0.904 | 0.538 | 0.821 | 0.546 | 0.489 | 0.641 (0.53–0.73) |
| `sup:nca` | 32 | 0.584 | 0.739 | 0.675 | 0.268 | 0.773 | 0.598 | 0.574 (0.47–0.68) |
| `text:tfidf` | 20000 | 0.542 | 0.997 | 0.590 | 0.963 | 0.369 | 0.211 | 0.661 (0.56–0.76) |
| `text:minilm` | 384 | 0.509 | 0.986 | 0.540 | 0.885 | 0.342 | 0.235 | 0.516 (0.40–0.64) |
| `nl:expstretch` | 5376 | 0.540 | 0.900 | 0.615 | 0.770 | 0.482 | 0.419 | 0.684 (0.58–0.78) |
| `nl:tsne2` | 2 | 0.531 | 0.703 | 0.665 | 0.518 | 0.629 | 0.472 | 0.693 (0.60–0.79) |
| `nl:umap` | 16 | 0.554 | 0.789 | 0.551 | 0.104 | 0.415 | 0.592 | 0.555 (0.46–0.66) |

### deepseekv4pro

| metric | dim | pairAUR | scenAUR | provAUR | hopPair | nnBal | kNN15aur | durAUR (CI95) |
|---|---|---|---|---|---|---|---|---|
| `pool:mean` | 5376 | 0.534 | 0.926 | 0.611 | 0.729 | 0.840 | 0.580 | 0.739 (0.62–0.84) |
| `pool:last` | 5376 | 0.756 | 0.456 | 0.700 | 0.052 | 0.591 | 0.733 | 0.771 (0.68–0.86) |
| `pool:last32` | 5376 | 0.629 | 0.658 | 0.654 | 0.158 | 0.559 | 0.695 | 0.767 (0.65–0.86) |
| `pool:probe` | 5376 | 0.612 | 0.721 | 0.588 | 0.512 | 0.532 | 0.377 | 0.608 (0.48–0.73) |
| `pool:topz16` | 5376 | 0.485 | 0.993 | 0.510 | 0.974 | 0.643 | 0.464 | 0.658 (0.56–0.77) |
| `lin:centered` | 5376 | 0.538 | 0.943 | 0.690 | 0.714 | 0.766 | 0.554 | 0.702 (0.59–0.80) |
| `lin:whitened` | 5376 | 0.563 | 0.988 | 0.673 | 0.848 | 0.785 | 0.349 | 0.754 (0.65–0.85) |
| `lin:pcawhite` | 128 | 0.515 | 0.973 | 0.521 | 0.856 | 0.796 | 0.677 | 0.689 (0.57–0.80) |
| `unsup:pca64` | 64 | 0.537 | 0.894 | 0.608 | 0.386 | 0.861 | 0.639 | 0.722 (0.62–0.82) |
| `probe:logit` | 1 | 0.616 | 0.351 | 0.553 | 0.006 | 0.415 | 0.359 | 0.411 (0.29–0.54) |
| `probe:proj` | 1 | 0.530 | 0.427 | 0.559 | 0.002 | 0.371 | 0.267 | 0.483 (0.36–0.62) |
| `probe:wscaled` | 5376 | 0.534 | 0.924 | 0.621 | 0.723 | 0.833 | 0.589 | 0.727 (0.62–0.83) |
| `probe:jac` | 5376 | 0.656 | 0.653 | 0.567 | 0.308 | 0.423 | 0.491 | 0.591 (0.46–0.71) |
| `sup:lda` | 1 | 0.659 | 0.298 | 0.620 | 0.000 | 0.620 | 0.577 | 0.558 (0.44–0.67) |
| `sup:winwhite` | 128 | 0.552 | 0.954 | 0.541 | 0.791 | 0.803 | 0.682 | 0.692 (0.58–0.81) |
| `sup:nca` | 32 | 0.562 | 0.713 | 0.668 | 0.053 | 0.594 | 0.580 | 0.699 (0.58–0.80) |
| `text:tfidf` | 20000 | 0.540 | 0.999 | 0.591 | 0.956 | 0.631 | 0.304 | 0.698 (0.58–0.81) |
| `text:minilm` | 384 | 0.510 | 0.986 | 0.546 | 0.799 | 0.631 | 0.564 | 0.688 (0.59–0.80) |
| `nl:expstretch` | 5376 | 0.534 | 0.926 | 0.611 | 0.729 | 0.840 | 0.580 | 0.739 (0.62–0.84) |
| `nl:tsne2` | 2 | 0.531 | 0.754 | 0.587 | 0.305 | 0.924 | 0.413 | 0.723 (0.62–0.82) |
| `nl:umap` | 16 | 0.527 | 0.750 | 0.567 | 0.063 | 0.492 | 0.421 | 0.590 (0.47–0.71) |

## Verdicts (both arms must clear each bar)

`label` pairAUR ≥ 0.6; `guard` hopPair ≤ 0.25; `acquisition` durAUR ≥ 0.65 with CI clear of 0.5.

| metric | verdict |
|---|---|
| `pool:mean` | scenario + **acquisition** |
| `pool:last` | label + **acquisition** |
| `pool:last32` | **acquisition** |
| `pool:probe` | — |
| `pool:topz16` | scenario |
| `lin:centered` | scenario |
| `lin:whitened` | scenario + **acquisition** |
| `lin:pcawhite` | scenario |
| `unsup:pca64` | scenario + **acquisition** |
| `probe:logit` | — |
| `probe:proj` | — |
| `probe:wscaled` | scenario + **acquisition** |
| `probe:jac` | label |
| `sup:lda` | label |
| `sup:winwhite` | scenario |
| `sup:nca` | — |
| `text:tfidf` | scenario + **acquisition** |
| `text:minilm` | scenario |
| `nl:expstretch` | scenario + **acquisition** |
| `nl:tsne2` | **acquisition** + (transductive — not deployable) |
| `nl:umap` | scenario |

## §5a's own columns, for continuity with the published note

Raw similarity means. These are *scale-dependent* — `nl:expstretch` moves them while changing nothing else — so they are here to line up against the note, not to be compared across metrics.


### gptoss120b

| metric | same-label | opp-label | Δ | own counterpart | new→v2 NN | frac own closer | kNN 1/5/15 (raw acc) |
|---|---|---|---|---|---|---|---|
| `pool:mean` | 0.8749 | 0.8616 | 0.0133 | 0.9364 | 0.9346 | 0.73 | 56.0% / 51.7% / 48.3% |
| `pool:last` | 0.7210 | 0.6283 | 0.0926 | 0.6757 | 0.8666 | 0.09 | 75.0% / 78.4% / 78.4% |
| `pool:last32` | 0.8222 | 0.7991 | 0.0230 | 0.8509 | 0.9020 | 0.39 | 57.8% / 53.4% / 49.1% |
| `pool:probe` | 0.8125 | 0.7526 | 0.0600 | 0.8318 | 0.9050 | 0.51 | 41.4% / 19.8% / 20.7% |
| `pool:topz16` | 0.9132 | 0.9123 | 0.0009 | 0.9966 | 0.9554 | 0.98 | 47.4% / 49.1% / 50.9% |
| `lin:centered` | 0.0655 | -0.0552 | 0.1207 | 0.4938 | 0.5159 | 0.70 | 59.5% / 49.1% / 45.7% |
| `lin:whitened` | 0.0367 | -0.0328 | 0.0694 | 0.5632 | 0.4196 | 0.79 | 54.3% / 62.9% / 56.9% |
| `lin:pcawhite` | -15.6967 | -15.7433 | 0.0465 | -9.6671 | -10.0045 | 0.85 | 50.0% / 41.4% / 30.2% |
| `unsup:pca64` | -28.4980 | -30.9230 | 2.4250 | -18.8663 | -13.6411 | 0.60 | 69.0% / 60.3% / 50.0% |
| `probe:logit` | -5.4493 | -20.7018 | 15.2524 | -17.0487 | -0.1568 | 0.00 | 18.1% / 9.5% / 11.2% |
| `probe:proj` | -5.4221 | -9.0421 | 3.6200 | -7.8875 | -0.0244 | 0.00 | 36.2% / 25.9% / 24.1% |
| `probe:wscaled` | 0.8088 | 0.7918 | 0.0170 | 0.9010 | 0.9015 | 0.71 | 58.6% / 56.0% / 50.9% |
| `probe:jac` | 0.6906 | 0.5710 | 0.1197 | 0.6613 | 0.8661 | 0.25 | 32.8% / 23.3% / 30.2% |
| `sup:lda` | -0.6830 | -4.0886 | 3.4056 | -3.5422 | -0.0354 | 0.00 | 62.1% / 63.8% / 63.8% |
| `sup:winwhite` | -14.1668 | -15.1428 | 0.9759 | -9.7341 | -9.2015 | 0.81 | 57.8% / 50.9% / 45.7% |
| `sup:nca` | -27.2551 | -30.4467 | 3.1917 | -20.6098 | -11.0175 | 0.28 | 77.6% / 62.9% / 58.6% |
| `text:tfidf` | 0.0438 | 0.0356 | 0.0082 | 0.3496 | 0.1559 | 0.95 | 44.8% / 49.1% / 44.0% |
| `text:minilm` | 0.1840 | 0.1608 | 0.0232 | 0.6924 | 0.5275 | 0.90 | 42.2% / 38.8% / 32.8% |
| `nl:expstretch` | 1180.7190 | 1029.1620 | 151.5570 | 1846.8364 | 1809.3982 | 0.73 | 56.0% / 51.7% / 48.3% |
| `nl:tsne2` | -26.2508 | -34.4640 | 8.2133 | -19.6534 | -4.9510 | 0.54 | 64.7% / 51.7% / 50.9% |
| `nl:umap` | -6.1816 | -6.9469 | 0.7652 | -2.5476 | -0.3272 | 0.15 | 52.6% / 47.4% / 47.4% |

### deepseekv4pro

| metric | same-label | opp-label | Δ | own counterpart | new→v2 NN | frac own closer | kNN 1/5/15 (raw acc) |
|---|---|---|---|---|---|---|---|
| `pool:mean` | 0.8874 | 0.8766 | 0.0108 | 0.9488 | 0.9528 | 0.73 | 77.9% / 66.3% / 58.1% |
| `pool:last` | 0.7256 | 0.6303 | 0.0953 | 0.6578 | 0.8412 | 0.05 | 62.8% / 60.5% / 61.6% |
| `pool:last32` | 0.8380 | 0.8169 | 0.0211 | 0.8565 | 0.9108 | 0.09 | 66.3% / 64.0% / 70.9% |
| `pool:probe` | 0.8100 | 0.7705 | 0.0394 | 0.8679 | 0.9160 | 0.59 | 48.8% / 36.0% / 29.1% |
| `pool:topz16` | 0.9056 | 0.9053 | 0.0002 | 0.9885 | 0.9540 | 0.97 | 62.8% / 69.8% / 68.6% |
| `lin:centered` | 0.0486 | -0.0506 | 0.0992 | 0.5447 | 0.5452 | 0.67 | 74.4% / 67.4% / 68.6% |
| `lin:whitened` | 0.0242 | -0.0296 | 0.0539 | 0.6111 | 0.4586 | 0.84 | 73.3% / 61.6% / 60.5% |
| `lin:pcawhite` | -15.7043 | -15.8474 | 0.1431 | -8.6362 | -9.3826 | 0.86 | 66.3% / 46.5% / 22.1% |
| `unsup:pca64` | -26.7379 | -28.7620 | 2.0241 | -16.8725 | -11.7244 | 0.43 | 81.4% / 69.8% / 62.8% |
| `probe:logit` | -6.4835 | -20.4516 | 13.9680 | -18.5490 | -0.0538 | 0.01 | 33.7% / 26.7% / 26.7% |
| `probe:proj` | -4.1631 | -7.3982 | 3.2351 | -6.4767 | -0.0131 | 0.00 | 39.5% / 31.4% / 29.1% |
| `probe:wscaled` | 0.8671 | 0.8546 | 0.0125 | 0.9385 | 0.9426 | 0.69 | 76.7% / 69.8% / 58.1% |
| `probe:jac` | 0.6841 | 0.6060 | 0.0781 | 0.7506 | 0.8678 | 0.35 | 39.5% / 38.4% / 37.2% |
| `sup:lda` | -0.9526 | -4.0304 | 3.0778 | -3.6897 | -0.0082 | 0.00 | 54.7% / 52.3% / 48.8% |
| `sup:winwhite` | -14.4222 | -15.1469 | 0.7247 | -8.7935 | -8.6266 | 0.79 | 67.4% / 58.1% / 46.5% |
| `sup:nca` | -28.3088 | -31.5790 | 3.2702 | -23.0559 | -10.2080 | 0.06 | 72.1% / 72.1% / 69.8% |
| `text:tfidf` | 0.0441 | 0.0379 | 0.0062 | 0.3977 | 0.1616 | 0.98 | 65.1% / 60.5% / 59.3% |
| `text:minilm` | 0.1634 | 0.1484 | 0.0150 | 0.6814 | 0.5498 | 0.81 | 47.7% / 47.7% / 55.8% |
| `nl:expstretch` | 1270.2828 | 1144.9646 | 125.3182 | 2018.5103 | 2059.7659 | 0.73 | 77.9% / 66.3% / 58.1% |
| `nl:tsne2` | -33.5021 | -37.3400 | 3.8379 | -17.9311 | -4.4162 | 0.41 | 91.9% / 81.4% / 67.4% |
| `nl:umap` | -4.5590 | -4.9904 | 0.4314 | -2.3971 | -0.3637 | 0.02 | 64.0% / 67.4% / 72.1% |

## Class balance of the k-NN test set

| arm | positive rate (successes) | majority-class baseline |
|---|---|---|
| gptoss120b | 70.7% | 70.7% |
| deepseekv4pro | 82.6% | 82.6% |

## Acquisition baselines that need no metric

| arm | predictor | AUROC (CI95) |
|---|---|---|
| gptoss120b | `abs_seq_logit` | 0.708 (0.61–0.80) |
| gptoss120b | `neg_abs_seq_logit` | 0.292 (0.20–0.39) |
| gptoss120b | `n_tokens` | 0.286 (0.20–0.39) |
| gptoss120b | `label_is_positive` | 0.407 (0.33–0.50) |
| deepseekv4pro | `abs_seq_logit` | 0.582 (0.46–0.70) |
| deepseekv4pro | `neg_abs_seq_logit` | 0.418 (0.30–0.54) |
| deepseekv4pro | `n_tokens` | 0.509 (0.41–0.63) |
| deepseekv4pro | `label_is_positive` | 0.483 (0.41–0.57) |

## Does the new vintage move the training set toward eval?

Size-matched nearest-neighbour similarity from each eval row to the 116/86 new-in-v3 successes, versus to an equal-sized random draw from v2 (20 draws). §5 asked this in raw cosine; a metric that passed the label bar asking it again is the point.

| arm | metric | split | eval→new-v3 | eval→v2 (size-matched) | Δ |
|---|---|---|---|---|---|
| gptoss120b | `pool:mean` | ai_dilemmas | 0.9301 | 0.9171 | +0.0129 |
| gptoss120b | `pool:mean` | ant_hh | 0.8867 | 0.8949 | -0.0082 |
| gptoss120b | `pool:mean` | balanced_refusal | 0.9013 | 0.9012 | +0.0001 |
| gptoss120b | `pool:mean` | daily_dilemmas | 0.9138 | 0.9130 | +0.0008 |
| gptoss120b | `pool:probe` | ai_dilemmas | 0.8329 | 0.8285 | +0.0045 |
| gptoss120b | `pool:probe` | ant_hh | 0.8308 | 0.8397 | -0.0089 |
| gptoss120b | `pool:probe` | balanced_refusal | 0.8574 | 0.8707 | -0.0133 |
| gptoss120b | `pool:probe` | daily_dilemmas | 0.8061 | 0.8117 | -0.0056 |
| gptoss120b | `lin:centered` | ai_dilemmas | 0.5529 | 0.4508 | +0.1021 |
| gptoss120b | `lin:centered` | ant_hh | 0.3198 | 0.3536 | -0.0338 |
| gptoss120b | `lin:centered` | balanced_refusal | 0.4318 | 0.4274 | +0.0044 |
| gptoss120b | `lin:centered` | daily_dilemmas | 0.4063 | 0.3913 | +0.0151 |
| gptoss120b | `lin:whitened` | ai_dilemmas | 0.2865 | 0.2108 | +0.0757 |
| gptoss120b | `lin:whitened` | ant_hh | 0.1531 | 0.1934 | -0.0403 |
| gptoss120b | `lin:whitened` | balanced_refusal | 0.2150 | 0.2369 | -0.0219 |
| gptoss120b | `lin:whitened` | daily_dilemmas | 0.2008 | 0.1933 | +0.0075 |
| gptoss120b | `lin:pcawhite` | ai_dilemmas | -10.7940 | -11.0672 | +0.2732 |
| gptoss120b | `lin:pcawhite` | ant_hh | -13.8963 | -13.6325 | -0.2638 |
| gptoss120b | `lin:pcawhite` | balanced_refusal | -13.4642 | -13.3463 | -0.1179 |
| gptoss120b | `lin:pcawhite` | daily_dilemmas | -11.0109 | -10.7803 | -0.2306 |
| gptoss120b | `unsup:pca64` | ai_dilemmas | -14.3450 | -18.0779 | +3.7329 |
| gptoss120b | `unsup:pca64` | ant_hh | -18.7701 | -19.5734 | +0.8032 |
| gptoss120b | `unsup:pca64` | balanced_refusal | -18.1681 | -19.7300 | +1.5619 |
| gptoss120b | `unsup:pca64` | daily_dilemmas | -17.4724 | -18.5977 | +1.1253 |
| gptoss120b | `probe:proj` | ai_dilemmas | -1.1583 | -0.4891 | -0.6692 |
| gptoss120b | `probe:proj` | ant_hh | -0.2268 | -0.1672 | -0.0596 |
| gptoss120b | `probe:proj` | balanced_refusal | -0.4878 | -0.2647 | -0.2231 |
| gptoss120b | `probe:proj` | daily_dilemmas | -0.5785 | -0.3845 | -0.1940 |
| gptoss120b | `probe:wscaled` | ai_dilemmas | 0.9065 | 0.8842 | +0.0222 |
| gptoss120b | `probe:wscaled` | ant_hh | 0.8465 | 0.8582 | -0.0117 |
| gptoss120b | `probe:wscaled` | balanced_refusal | 0.8709 | 0.8683 | +0.0026 |
| gptoss120b | `probe:wscaled` | daily_dilemmas | 0.8806 | 0.8790 | +0.0017 |
| gptoss120b | `sup:lda` | ai_dilemmas | -0.0155 | -0.2044 | +0.1889 |
| gptoss120b | `sup:lda` | ant_hh | -0.1294 | -0.1331 | +0.0037 |
| gptoss120b | `sup:lda` | balanced_refusal | -0.0656 | -0.1327 | +0.0671 |
| gptoss120b | `sup:lda` | daily_dilemmas | -0.0324 | -0.1313 | +0.0989 |
| gptoss120b | `sup:winwhite` | ai_dilemmas | -9.6037 | -10.4601 | +0.8564 |
| gptoss120b | `sup:winwhite` | ant_hh | -12.6356 | -12.6524 | +0.0168 |
| gptoss120b | `sup:winwhite` | balanced_refusal | -12.2052 | -12.3559 | +0.1507 |
| gptoss120b | `sup:winwhite` | daily_dilemmas | -10.0640 | -10.1041 | +0.0401 |
| gptoss120b | `sup:nca` | ai_dilemmas | -12.7200 | -15.7327 | +3.0127 |
| gptoss120b | `sup:nca` | ant_hh | -16.4197 | -17.2020 | +0.7822 |
| gptoss120b | `sup:nca` | balanced_refusal | -15.4313 | -17.4850 | +2.0536 |
| gptoss120b | `sup:nca` | daily_dilemmas | -15.0219 | -17.0602 | +2.0383 |
| gptoss120b | `text:tfidf` | ai_dilemmas | 0.1023 | 0.0755 | +0.0267 |
| gptoss120b | `text:tfidf` | ant_hh | 0.0929 | 0.0890 | +0.0039 |
| gptoss120b | `text:tfidf` | balanced_refusal | 0.0904 | 0.0857 | +0.0047 |
| gptoss120b | `text:tfidf` | daily_dilemmas | 0.0930 | 0.0873 | +0.0057 |
| gptoss120b | `nl:expstretch` | ai_dilemmas | 1706.3315 | 1538.1866 | +168.1449 |
| gptoss120b | `nl:expstretch` | ant_hh | 1221.8789 | 1313.6868 | -91.8079 |
| gptoss120b | `nl:expstretch` | balanced_refusal | 1369.8505 | 1373.0823 | -3.2318 |
| gptoss120b | `nl:expstretch` | daily_dilemmas | 1500.1953 | 1490.0223 | +10.1730 |
| gptoss120b | `nl:umap` | ai_dilemmas | -0.4970 | -0.5144 | +0.0174 |
| gptoss120b | `nl:umap` | ant_hh | -0.9399 | -0.6015 | -0.3384 |
| gptoss120b | `nl:umap` | balanced_refusal | -0.6168 | -0.5420 | -0.0748 |
| gptoss120b | `nl:umap` | daily_dilemmas | -0.6282 | -0.5847 | -0.0435 |
| deepseekv4pro | `pool:mean` | ai_dilemmas | 0.9203 | 0.9227 | -0.0023 |
| deepseekv4pro | `pool:mean` | ant_hh | 0.8843 | 0.8902 | -0.0059 |
| deepseekv4pro | `pool:mean` | balanced_refusal | 0.8858 | 0.8958 | -0.0100 |
| deepseekv4pro | `pool:mean` | daily_dilemmas | 0.9223 | 0.9155 | +0.0067 |
| deepseekv4pro | `pool:probe` | ai_dilemmas | 0.8404 | 0.8384 | +0.0020 |
| deepseekv4pro | `pool:probe` | ant_hh | 0.8215 | 0.8296 | -0.0081 |
| deepseekv4pro | `pool:probe` | balanced_refusal | 0.8136 | 0.8284 | -0.0148 |
| deepseekv4pro | `pool:probe` | daily_dilemmas | 0.8356 | 0.8139 | +0.0217 |
| deepseekv4pro | `lin:centered` | ai_dilemmas | 0.4204 | 0.4405 | -0.0201 |
| deepseekv4pro | `lin:centered` | ant_hh | 0.2590 | 0.3033 | -0.0443 |
| deepseekv4pro | `lin:centered` | balanced_refusal | 0.3175 | 0.3688 | -0.0513 |
| deepseekv4pro | `lin:centered` | daily_dilemmas | 0.3481 | 0.3585 | -0.0103 |
| deepseekv4pro | `lin:whitened` | ai_dilemmas | 0.1974 | 0.1958 | +0.0016 |
| deepseekv4pro | `lin:whitened` | ant_hh | 0.1235 | 0.1722 | -0.0487 |
| deepseekv4pro | `lin:whitened` | balanced_refusal | 0.1313 | 0.2061 | -0.0749 |
| deepseekv4pro | `lin:whitened` | daily_dilemmas | 0.2329 | 0.1850 | +0.0479 |
| deepseekv4pro | `lin:pcawhite` | ai_dilemmas | -10.9721 | -11.3575 | +0.3853 |
| deepseekv4pro | `lin:pcawhite` | ant_hh | -14.5742 | -14.5703 | -0.0040 |
| deepseekv4pro | `lin:pcawhite` | balanced_refusal | -14.2680 | -14.2372 | -0.0308 |
| deepseekv4pro | `lin:pcawhite` | daily_dilemmas | -11.2108 | -11.8892 | +0.6784 |
| deepseekv4pro | `unsup:pca64` | ai_dilemmas | -16.9437 | -17.7639 | +0.8202 |
| deepseekv4pro | `unsup:pca64` | ant_hh | -19.5269 | -19.9939 | +0.4670 |
| deepseekv4pro | `unsup:pca64` | balanced_refusal | -20.6600 | -20.3166 | -0.3434 |
| deepseekv4pro | `unsup:pca64` | daily_dilemmas | -15.6599 | -18.6960 | +3.0361 |
| deepseekv4pro | `probe:proj` | ai_dilemmas | -0.1725 | -0.1322 | -0.0403 |
| deepseekv4pro | `probe:proj` | ant_hh | -0.2193 | -0.1655 | -0.0539 |
| deepseekv4pro | `probe:proj` | balanced_refusal | -0.4180 | -0.3013 | -0.1168 |
| deepseekv4pro | `probe:proj` | daily_dilemmas | -0.8942 | -0.5181 | -0.3761 |
| deepseekv4pro | `probe:wscaled` | ai_dilemmas | 0.9091 | 0.9097 | -0.0006 |
| deepseekv4pro | `probe:wscaled` | ant_hh | 0.8696 | 0.8764 | -0.0068 |
| deepseekv4pro | `probe:wscaled` | balanced_refusal | 0.8750 | 0.8831 | -0.0082 |
| deepseekv4pro | `probe:wscaled` | daily_dilemmas | 0.9094 | 0.9008 | +0.0086 |
| deepseekv4pro | `sup:lda` | ai_dilemmas | -0.0229 | -0.0729 | +0.0500 |
| deepseekv4pro | `sup:lda` | ant_hh | -0.0967 | -0.0788 | -0.0178 |
| deepseekv4pro | `sup:lda` | balanced_refusal | -0.1157 | -0.0777 | -0.0380 |
| deepseekv4pro | `sup:lda` | daily_dilemmas | -0.0204 | -0.0790 | +0.0585 |
| deepseekv4pro | `sup:winwhite` | ai_dilemmas | -10.0155 | -10.7492 | +0.7337 |
| deepseekv4pro | `sup:winwhite` | ant_hh | -13.3561 | -13.5687 | +0.2126 |
| deepseekv4pro | `sup:winwhite` | balanced_refusal | -13.1411 | -13.2708 | +0.1297 |
| deepseekv4pro | `sup:winwhite` | daily_dilemmas | -10.3545 | -11.1701 | +0.8156 |
| deepseekv4pro | `sup:nca` | ai_dilemmas | -13.7844 | -16.1583 | +2.3739 |
| deepseekv4pro | `sup:nca` | ant_hh | -16.6692 | -17.3888 | +0.7196 |
| deepseekv4pro | `sup:nca` | balanced_refusal | -18.7031 | -18.4164 | -0.2868 |
| deepseekv4pro | `sup:nca` | daily_dilemmas | -13.6781 | -16.9527 | +3.2746 |
| deepseekv4pro | `text:tfidf` | ai_dilemmas | 0.0771 | 0.0728 | +0.0043 |
| deepseekv4pro | `text:tfidf` | ant_hh | 0.0888 | 0.0827 | +0.0060 |
| deepseekv4pro | `text:tfidf` | balanced_refusal | 0.0834 | 0.0780 | +0.0054 |
| deepseekv4pro | `text:tfidf` | daily_dilemmas | 0.0973 | 0.0792 | +0.0181 |
| deepseekv4pro | `nl:expstretch` | ai_dilemmas | 1577.7095 | 1608.1157 | -30.4062 |
| deepseekv4pro | `nl:expstretch` | ant_hh | 1214.2230 | 1269.8315 | -55.6085 |
| deepseekv4pro | `nl:expstretch` | balanced_refusal | 1234.4440 | 1321.8323 | -87.3883 |
| deepseekv4pro | `nl:expstretch` | daily_dilemmas | 1604.6952 | 1522.2140 | +82.4812 |
| deepseekv4pro | `nl:umap` | ai_dilemmas | -2.5135 | -0.5802 | -1.9334 |
| deepseekv4pro | `nl:umap` | ant_hh | -1.7603 | -0.6086 | -1.1517 |
| deepseekv4pro | `nl:umap` | balanced_refusal | -2.4920 | -0.5532 | -1.9388 |
| deepseekv4pro | `nl:umap` | daily_dilemmas | -0.9085 | -0.6832 | -0.2253 |
