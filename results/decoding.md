# Decoding study (same trained model, different ways of picking words)

500 test sentences per direction. Rows 1-4 each add one step; row 4 is the final system. Rows 5 and 6 add one more idea to row 4 (both were rejected on the validation set or here).

## Shakespeare -> modern

| decoding | BLEU | loops % | length ratio | anchors kept % | ms / sentence |
|---|---|---|---|---|---|
| 1. greedy | 19.23 | 14.6 | 0.94 | 72.2 | 35 |
| 2. + no repeats | 19.20 | 1.0 | 0.91 | 71.7 | 31 |
| 3. + beam search (5) | 19.18 | 0.4 | 0.85 | 66.8 | 57 |
| 4. + coverage penalty (final) | 19.97 | 0.6 | 0.93 | 75.5 | 62 |
| 5. final + dictionary hints | 19.97 | 0.6 | 0.93 | 75.5 | 64 |
| 6. final + clause splitting | 18.72 | 8.6 | 1.12 | 82.2 | 78 |

## Modern -> Shakespeare

| decoding | BLEU | loops % | length ratio | anchors kept % | ms / sentence |
|---|---|---|---|---|---|
| 1. greedy | 17.62 | 11.8 | 0.86 | 82.6 | 32 |
| 2. + no repeats | 17.39 | 0.8 | 0.85 | 81.7 | 33 |
| 3. + beam search (5) | 17.31 | 0.0 | 0.83 | 83.1 | 58 |
| 4. + coverage penalty (final) | 17.93 | 0.2 | 0.87 | 87.3 | 61 |
| 5. final + dictionary hints | 18.07 | 0.2 | 0.87 | 86.4 | 63 |
| 6. final + clause splitting | 18.37 | 5.0 | 0.99 | 86.9 | 72 |
