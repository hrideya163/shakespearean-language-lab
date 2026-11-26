# RQ4: Copy gate ablation (500 test sentences, Shakespeare -> modern)

- **Style score** = Shakespeare probability from the style classifier (lower = more modern; the Shakespeare input scores 0.70).
- **BLEU 95% CI** = bootstrap confidence interval (300 resamples). If two intervals overlap a lot, the difference may be luck.

## A. Ablation

| System | BLEU | BLEU 95% CI | Anchors kept % | Kept-word rate % | Style score |
|---|---|---|---|---|---|
| Copy input (no change) | 19.53 | 17.5 - 21.7 | 100.00 | 100.00 | 0.72 |
| Dictionary baseline | 21.70 | 19.5 - 24.2 | 100.00 | 99.95 | 0.60 |
| LSTM without copy gate | 17.76 | 15.9 - 19.7 | 64.15 | 75.99 | 0.38 |
| LSTM with copy gate + extended vocabulary | 19.38 | 17.3 - 21.3 | 68.73 | 85.66 | 0.48 |
| LSTM with copy gate, no dropout | 18.32 | 16.3 - 20.3 | 72.24 | 85.41 | 0.51 |
| LSTM with copy gate | 19.97 | 18.1 - 22.0 | 75.47 | 86.79 | 0.49 |
| Human reference | - | - | - | - | 0.29 |

## B. Were archaic words modernised?

A hit = the dictionary meaning of the archaic word appears in the output (thou -> you). The dictionary baseline is 100% by construction, but it ignores context (it turns *art* into *are* even when *art* means painting).

| archaic word (times seen) | Dictionary baseline | LSTM without copy gate | LSTM with copy gate + extended vocabulary | LSTM with copy gate, no dropout | LSTM with copy gate |
|---|---|---|---|---|---|
| **all archaic words** | **100.0%** | **67.3%** | **60.5%** | **58.4%** | **58.7%** |
| thou -> you (90) | 100% | 99% | 92% | 97% | 87% |
| thy -> your (49) | 100% | 84% | 96% | 90% | 96% |
| thee -> you (25) | 100% | 88% | 96% | 88% | 100% |
| art -> are (20) | 100% | 90% | 85% | 80% | 80% |
| hath -> has (11) | 100% | 91% | 73% | 27% | 45% |
| hast -> have (11) | 100% | 64% | 36% | 36% | 45% |
| hence -> from here (8) | 100% | 0% | 12% | 12% | 12% |
| dost -> do (8) | 100% | 88% | 0% | 0% | 25% |
| ay -> yes (8) | 100% | 50% | 75% | 75% | 62% |
| doth -> does (7) | 100% | 0% | 14% | 0% | 0% |
| nay -> no (6) | 100% | 83% | 83% | 83% | 83% |
| wilt -> will (6) | 100% | 17% | 0% | 0% | 0% |

## C1. Copy probability by word-map label

If the copy gate is a faithful explanation, KEPT words should be copied (high value) and changed words generated (low value).

| label | mean copy probability | words |
|---|---|---|
| kept | 0.467 | 4209 |
| added | 0.321 | 244 |
| reworded | 0.200 | 455 |
| swapped | 0.029 | 155 |

## C2. Attention as alignment

For 610 changed output words (swapped or reworded), the attention peak pointed at the input word they replaced **67.7%** of the time (a random guess over the input words would be about 9%).

## C3. Confidence calibration

Each output word has a confidence (its softmax probability). We call a word *correct* if it appears in the human reference. A well-calibrated model is right about 80% of the time when it says 80%. This tells us whether the app's 'unsure' underline can be trusted.

| confidence range | words | actually correct % |
|---|---|---|
| 0.0 - 0.2 | 735 | 34.8 |
| 0.2 - 0.4 | 1253 | 46.4 |
| 0.4 - 0.6 | 1242 | 63.1 |
| 0.6 - 0.8 | 1153 | 70.4 |
| 0.8 - 1.0 | 680 | 81.9 |

## D. BLEU by source sentence length (tokens)

| system | 1-5 | 6-10 | 11-20 | 21+ |
|---|---|---|---|---|
| Copy input (no change) | 24.0 (n=118) | 22.5 (n=193) | 20.6 (n=126) | 14.2 (n=63) |
| Dictionary baseline | 27.5 (n=118) | 25.2 (n=193) | 23.1 (n=126) | 15.4 (n=63) |
| LSTM without copy gate | 26.0 (n=118) | 19.5 (n=193) | 19.4 (n=126) | 11.9 (n=63) |
| LSTM with copy gate + extended vocabulary | 25.6 (n=118) | 21.3 (n=193) | 21.1 (n=126) | 14.2 (n=63) |
| LSTM with copy gate, no dropout | 24.6 (n=118) | 20.2 (n=193) | 20.2 (n=126) | 12.9 (n=63) |
| LSTM with copy gate | 26.3 (n=118) | 21.8 (n=193) | 21.7 (n=126) | 14.9 (n=63) |

## E. Overfitting and dropout

See `rq4_overfitting.png`: without dropout the training loss keeps falling while the validation loss turns upward after a few epochs (the model memorises the training sentences). Dropout (randomly switching off 30% of values during training) narrows that gap. Compare the two copy-gate rows in table A.