# RQ2: Inside the LSTM generator

Analysed 20,000 validation characters, 256 cells.

## Cell most correlated with each feature

| feature | cell | correlation |
|---|---|---|
| position_in_line | #53 | +0.35 |
| inside_word | #10 | +0.62 |
| capital_letter | #77 | +0.31 |
| after_punctuation | #129 | +0.44 |
| end_of_line | #34 | +0.66 |

## Average forget gate by character type

Values near 1 mean the LSTM keeps its memory, and lower values mean it clears part of it.

| character type | mean forget gate | count |
|---|---|---|
| sentence end (. ! ?) | 0.618 | 426 |
| other punctuation | 0.657 | 533 |
| space | 0.678 | 3486 |
| newline | 0.697 | 408 |
| letter | 0.737 | 15147 |

## Gate saturation

| gate | mean value | units closed > 50% of the time | units open > 50% of the time |
|---|---|---|---|
| input | 0.831 | 0 | 220 |
| forget | 0.721 | 20 | 166 |
| output | 0.404 | 142 | 50 |

## Causal ablation: switching one cell off

Loss is measured only on the predictions linked to the feature (in nats). If the top cell really matters, switching it off should hurt much more than switching off a random cell. 'No' means the information is probably stored redundantly in several cells, so removing one does not matter.

| feature | cell | predictions | normal loss | increase (top cell off) | increase (random cell off, avg of 10) | causal effect? |
|---|---|---|---|---|---|---|
| position_in_line | #53 | 408 | 0.055 | +0.0033 | +0.0003 | yes |
| inside_word | #10 | 15146 | 1.673 | +0.0102 | +0.0023 | yes |
| capital_letter | #77 | 639 | 3.170 | +0.0011 | -0.0015 | no |
| after_punctuation | #129 | 815 | 0.225 | -0.0001 | -0.0005 | no |
| end_of_line | #34 | 408 | 0.055 | +0.0000 | -0.0001 | no |