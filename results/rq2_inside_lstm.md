# RQ2: Inside the LSTM generator

Model: `lstm_long`.

Analysed 20,000 validation characters, 256 cells.

## Cell most correlated with each feature

| feature | cell | correlation |
|---|---|---|
| position_in_line | #53 | +0.30 |
| inside_word | #220 | +0.63 |
| capital_letter | #77 | +0.33 |
| after_punctuation | #129 | +0.37 |
| end_of_line | #184 | +0.43 |

## Average forget gate by character type

Values near 1 mean the LSTM keeps its memory, and lower values mean it clears part of it.

| character type | mean forget gate | count |
|---|---|---|
| sentence end (. ! ?) | 0.576 | 426 |
| other punctuation | 0.620 | 533 |
| space | 0.649 | 3486 |
| newline | 0.686 | 408 |
| letter | 0.710 | 15147 |

## Gate saturation

| gate | mean value | units closed > 50% of the time | units open > 50% of the time |
|---|---|---|---|
| input | 0.795 | 1 | 215 |
| forget | 0.693 | 27 | 169 |
| output | 0.399 | 160 | 49 |

## Causal ablation: switching one cell off

Loss is measured only on the predictions linked to the feature (in nats). If the top cell really matters, switching it off should hurt much more than switching off a random cell. 'No' means the information is probably stored redundantly in several cells, so removing one does not matter.

| feature | cell | predictions | normal loss | increase (top cell off) | increase (random cell off, avg of 10) | causal effect? |
|---|---|---|---|---|---|---|
| position_in_line | #53 | 408 | 0.060 | +0.0040 | +0.0003 | yes |
| inside_word | #220 | 15146 | 1.623 | +0.0099 | +0.0061 | no |
| capital_letter | #77 | 639 | 3.288 | +0.0050 | +0.0010 | yes |
| after_punctuation | #129 | 815 | 0.240 | -0.0014 | +0.0004 | no |
| end_of_line | #184 | 408 | 0.060 | -0.0015 | -0.0006 | no |