# RQ4: Copy gate ablation (500 test sentences, Shakespeare -> modern)

- **Style score** = Shakespeare probability from the style classifier (lower = more modern; the Shakespeare input scores 0.70).
- **BLEU 95% CI** = bootstrap confidence interval (300 resamples). If two intervals overlap a lot, the difference may be luck.

## A. Ablation

| System | BLEU | BLEU 95% CI | Anchors kept % | Kept-word rate % | Style score |
|---|---|---|---|---|---|
| Copy input (no change) | 19.53 | 17.5 - 21.7 | 100.00 | 100.00 | 0.72 |
| Dictionary baseline | 21.70 | 19.5 - 24.2 | 100.00 | 99.95 | 0.60 |
| LSTM without copy gate | 14.93 | 12.8 - 17.3 | 63.61 | 72.86 | 0.37 |
| LSTM with copy gate, no dropout | 17.32 | 15.3 - 19.4 | 68.19 | 78.29 | 0.48 |
| LSTM with copy gate | 19.22 | 17.0 - 21.3 | 72.24 | 83.26 | 0.45 |
| Human reference | - | - | - | - | 0.29 |

## B. Were archaic words modernised?

A hit = the dictionary meaning of the archaic word appears in the output (thou -> you). The dictionary baseline is 100% by construction, but it ignores context (it turns *art* into *are* even when *art* means painting).

| archaic word (times seen) | Dictionary baseline | LSTM without copy gate | LSTM with copy gate, no dropout | LSTM with copy gate |
|---|---|---|---|---|
| **all archaic words** | **100.0%** | **64.6%** | **58.1%** | **58.1%** |
| thou -> you (90) | 100% | 97% | 97% | 93% |
| thy -> your (49) | 100% | 84% | 80% | 86% |
| thee -> you (25) | 100% | 88% | 84% | 88% |
| art -> are (20) | 100% | 80% | 85% | 60% |
| hath -> has (11) | 100% | 82% | 55% | 45% |
| hast -> have (11) | 100% | 55% | 27% | 45% |
| hence -> from here (8) | 100% | 0% | 0% | 12% |
| dost -> do (8) | 100% | 38% | 12% | 50% |
| ay -> yes (8) | 100% | 50% | 75% | 62% |
| doth -> does (7) | 100% | 0% | 0% | 0% |
| nay -> no (6) | 100% | 83% | 83% | 83% |
| wilt -> will (6) | 100% | 0% | 0% | 17% |

## C1. Copy probability by word-map label

If the copy gate is a faithful explanation, KEPT words should be copied (high value) and changed words generated (low value).

| label | mean copy probability | words |
|---|---|---|
| kept | 0.454 | 4016 |
| added | 0.314 | 477 |
| reworded | 0.213 | 505 |
| swapped | 0.030 | 142 |

## C2. Attention as alignment

For 647 changed output words (swapped or reworded), the attention peak pointed at the input word they replaced **60.7%** of the time (a random guess over the input words would be about 9%).

## C3. Confidence calibration

Each output word has a confidence (its softmax probability). We call a word *correct* if it appears in the human reference. A well-calibrated model is right about 80% of the time when it says 80%. This tells us whether the app's 'unsure' underline can be trusted.

| confidence range | words | actually correct % |
|---|---|---|
| 0.0 - 0.2 | 866 | 33.9 |
| 0.2 - 0.4 | 1388 | 51.7 |
| 0.4 - 0.6 | 1227 | 64.5 |
| 0.6 - 0.8 | 1054 | 72.0 |
| 0.8 - 1.0 | 605 | 83.5 |

## D. BLEU by source sentence length (tokens)

| system | 1-5 | 6-10 | 11-20 | 21+ |
|---|---|---|---|---|
| Copy input (no change) | 24.0 (n=118) | 22.5 (n=193) | 20.6 (n=126) | 14.2 (n=63) |
| Dictionary baseline | 27.5 (n=118) | 25.2 (n=193) | 23.1 (n=126) | 15.4 (n=63) |
| LSTM without copy gate | 24.6 (n=118) | 16.7 (n=193) | 17.6 (n=126) | 8.1 (n=63) |
| LSTM with copy gate, no dropout | 24.0 (n=118) | 19.4 (n=193) | 19.4 (n=126) | 10.9 (n=63) |
| LSTM with copy gate | 26.5 (n=118) | 20.9 (n=193) | 21.4 (n=126) | 13.8 (n=63) |

## E. Overfitting and dropout

See `rq4_overfitting.png`: without dropout the training loss keeps falling while the validation loss turns upward after a few epochs (the model memorises the training sentences). Dropout (randomly switching off 30% of values during training) narrows that gap. Compare the two copy-gate rows in table A.