# Decoding settings tuned on the validation set

300 validation sentences, average BLEU of both directions (beam search 5, no repeats).

| coverage weight \ hint bonus | 0.0 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| 0.0 | 15.92 | 15.81 | 15.75 | 15.63 |
| 0.2 | 16.36 | 16.31 | 16.29 | 16.36 |
| 0.4 | 16.56 | 16.51 | 16.53 | 16.38 |
| 0.8 | 16.47 | 16.41 | 16.48 | 16.40 |

Best: {'coverage_weight': 0.4, 'hint_bonus': 0.0} (BLEU 16.56). Set COVERAGE_WEIGHT and HINT_BONUS in lab/translator/decode.py.