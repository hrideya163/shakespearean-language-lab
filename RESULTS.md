# Results: Inside the LSTM

Findings of the five research questions, with the numbers from the final run. Every number here can be
reproduced with `python run_all.py` (the per-experiment reports and all figures are written to `results/`).

**Setup in one line:** character-level generators (hidden size 256, 3,000 steps of 64 × 100 characters) and a
word-level BiLSTM encoder–decoder translator with attention and a copy gate (6.4 M parameters), trained on
CPU using the 18,395-pair Shakespeare ↔ modern corpus. Seed 42 everywhere.

---

## RQ1: Why LSTM? RNN vs GRU vs LSTM

| model | hidden | recurrent weights | test bpc ↓ | memory gain (bpc) | gradient 10 back | gradient 50 back |
|---|---|---|---|---|---|---|
| RNN | 256 | 82 k | 2.322 | 0.35 | 12.5 % | 0.001 % |
| RNN (matched size) | 542 | 329 k | 2.230 | 0.40 | 22.8 % | 0.033 % |
| LSTM | 256 | 329 k | 2.227 | 0.36 | 9.8 % | 0.007 % |
| GRU | 256 | 247 k | 2.176 | 0.42 | 16.1 % | 0.382 % |
| GRU (matched size) | 300 | 329 k | **2.160** | **0.44** | 16.5 % | 0.316 % |

**Long training budget (9,000 steps, same number of weights):**

| model | best valid bpc at 3,000 steps | best valid bpc at 9,000 steps | training loss 3k → 9k |
|---|---|---|---|
| RNN (matched size) | 2.144 | GRU_LONG_RNN | 1.44 → 1.44 (flat) |
| LSTM | 2.132 | **2.069** | 1.49 → 1.35 |
| GRU (matched size) | 2.065 | GRU_LONG_GRU | GRU_LONG_TRAIN |

**Findings**

1. **The vanishing gradient is real and measurable** (`rq1_gradient_flow.png`). Fifty characters back,
   the plain RNN's gradient is 0.001 % of its size at the current step, so it effectively cannot learn
   from it. The GRU keeps 0.3–0.4 %, about 300× more.
2. **Gating wins, but not in the order the textbook suggests.** With the same number of weights the GRU
   is best and learns fastest. After 3,000 steps the LSTM only ties the size-matched RNN.
3. **The LSTM's advantage shows with more training.** With 3× the budget the size-matched RNN stops
   improving (its *training* loss is flat, so it has hit its capacity rather than overfitting), while the
   LSTM keeps learning and pulls ahead by 0.07 bpc. The LSTM learns more slowly than the GRU but can learn
   more than the RNN.
4. **Honest caveat:** the LSTM's gradient measured at 50 steps back is small too (0.007 %). The cell
   state can carry information, but a 3,000-step model has not learned to use very long context. The
   memory curve gains for all models are 0.35–0.44 bpc, mostly from the first ~15 characters (`rq1_memory_curve.png`).

Figures: `rq1_learning_curves.png`, `rq1_long_budget.png`, `rq1_memory_curve.png`, `rq1_gradient_flow.png`.

---

## RQ2: What do individual LSTM cells learn?

| feature | most correlated cell | correlation | switching it off hurts more than a random cell? |
|---|---|---|---|
| inside a word | #10 | +0.62 | **yes** (+0.0102 vs +0.0023 nats) |
| position in line | #53 | +0.35 | **yes** (+0.0033 vs +0.0003) |
| end of line | #34 | +0.66 | no |
| after punctuation | #129 | +0.44 | no |
| capital letter | #77 | +0.31 | no |

**Findings**

1. **Some cells are interpretable *and* causal.** Cell #10 tracks "am I inside a word". Switching off
   that single cell (1 of 256) raises the loss on within-word letters 4× more than switching off a random
   cell. The coloured text in `rq2_inside_lstm.html` shows it switching on and off at word boundaries.
2. **Correlation is not causation.** The cell most correlated with *end of line* (+0.66) has no causal
   effect when removed, so that information must be stored redundantly across several cells.
3. **The forget gate resets at boundaries.** Mean forget value: 0.74 on letters, 0.70 on newlines,
   0.68 on spaces, 0.62 after sentence ends. The model clears memory most at the end of a sentence, as
   intuition suggests.
4. **Gates specialise** (`rq2_gate_saturation.png`). The input gate is almost always open (mean 0.83),
   while the output gate is mostly closed (0.40, 142 of 256 units closed most of the time). The LSTM
   writes a lot to memory but reveals only a little at each step.
5. **The character embedding learns what a character *does*, not what it looks like**
   (`rq2_embeddings.png`, 2-D PCA). Letters form one central cloud. Punctuation and whitespace sit on the
   edges, and the sentence enders `. ! ?` sit next to each other. Upper/lower-case pairs (a/A, e/E,
   o/O) land close together, although the model was never told they are the same letter. Vowels and
   consonants do *not* separate clearly.

---

## RQ3: Temperature and generation quality

| temperature | real words % | distinct-2 % | repetition % | style score | archaic words % |
|---|---|---|---|---|---|
| 0.2 | 99.2 | 29.4 | 26.0 | 0.82 | 0.4 |
| 0.5 | 96.8 | 73.4 | 7.6 | 0.77 | 1.2 |
| **0.8** | 89.3 | **91.9** | **2.8** | **0.76** | 2.6 |
| 1.0 | 80.5 | 94.6 | 1.2 | 0.71 | 1.8 |
| 1.3 | 64.9 | 99.2 | 0.5 | 0.68 | 1.9 |
| *real Shakespeare* | *94.7* | *90.5* | *2.2* | *0.75* | *6.6* |

**Findings**

1. **Temperature trades correctness for diversity**, monotonically, in every metric.
2. **T ≈ 0.8 matches real Shakespeare best** in diversity, repetition and style score. Lower
   temperatures sound *more* Shakespearean to the classifier than real Shakespeare (0.82 vs 0.75), because
   they repeat the most typical phrases ("my lord", "the world").
3. **Rare words are under-used.** Even at the best temperature the model produces archaic words at less
   than half the real rate (2.6 % vs 6.6 %). A small model learns frequent patterns first. The same
   effect shows up in the translator (RQ4 B).

---

## RQ4: Does the copy gate help, and is it a faithful explanation?

500 test sentences, Shakespeare → modern:

| system | BLEU (95 % CI) | names/numbers/negations kept % | style score (lower = more modern) |
|---|---|---|---|
| Copy input (no change) | 19.5 (17.5–21.7) | 100 | 0.72 |
| Dictionary baseline | 21.7 (19.5–24.2) | 100 | 0.60 |
| LSTM without copy gate | 14.9 (12.8–17.3) | 64 | **0.37** |
| LSTM + copy gate, no dropout | 17.3 (15.3–19.4) | 68 | 0.48 |
| **LSTM + copy gate + dropout** | 19.2 (17.0–21.3) | 72 | 0.45 |
| *human reference* | – | – | *0.29* |

**Findings**

1. **The copy gate is the biggest single improvement:** +4.3 BLEU, and 8 percentage points more meaning
   anchors kept. The confidence intervals do not overlap. Its validation perplexity is 20.8 vs 24.6
   without it.
2. **Dropout is the second** (+1.9 BLEU). Without it the model overfits after epoch 3: training loss
   keeps falling while validation loss rises (`rq4_overfitting.png`).
3. **BLEU alone would pick the wrong system.** Copying the input unchanged scores as high as our model,
   because Shakespeare and modern English share most words. The style score shows what BLEU misses: our
   model's output reads as clearly more modern (0.45) than the dictionary baseline's (0.60), which only
   swaps words. Lower BLEU at higher modernity is a **faithfulness vs. fluency trade-off**: the no-copy
   model is the most modern (0.37) but loses a third of names, numbers and negations.
4. **The copy gate is a faithful built-in explanation.** Mean copy probability is 0.45 for words that were
   kept, 0.21 for reworded words and 0.03 for swapped words (thou → you). The model's internal switch agrees
   with what actually happened to each word.
5. **Attention works as alignment.** For changed words, the attention peak lands on the input word they
   replaced **60.7 %** of the time (chance ≈ 9 %).
6. **Confidence is usable** (`rq4_calibration.png`). Accuracy rises steadily with confidence
   (34 % → 84 %) and is close to calibrated above 0.6. Low confidence is *under*-confident, partly because
   "correct" means "appears in the reference", a lenient test.
7. **Weaknesses are clear and explainable.** Long sentences (21+ tokens) drop to 13.8 BLEU. Rare archaic
   words (doth, wilt, hence) are almost never modernised, which is the same rare-word effect as RQ3.

---

## RQ5: Can we detect meaning drift?

1,068 candidate translations: 600 clean and 468 deliberately corrupted (negation flipped, name swapped,
number changed).

| drift type | n | rule-based guard detects | model likelihood AUROC (95 % CI) | model prefers the correct version (paired) |
|---|---|---|---|---|
| negation | 338 | 91.7 % | 0.61 (0.58–0.65) | 90.5 % |
| name | 104 | 87.5 % | 0.60 (0.54–0.65) | 72.1 % |
| number | 26 | 76.9 % | 0.82 (0.75–0.87) | 100 % |

The guard raises false alarms on 17.2 % of clean translations.

**Findings**

1. **The model "knows" the right meaning but can't flag it on its own.** Given both versions of the same
   sentence, the translator prefers the correct one 90.5 % of the time for negations. As a standalone
   detector (AUROC) it is only 0.61, because how likely a sentence is depends far more on the sentence
   than on the error. **Comparison beats absolute scores.**
2. **The two detectors complement each other.** The model cannot see most name swaps (rare names
   become `<unk>`, so both versions get *identical* scores), while the rule-based guard catches 87.5 % of
   them. The guard catches negations well too, but at a 17 % false-alarm cost.
3. **The model is fooled by fluent negation flips.** "…a tear that has been washed off yet" scores higher
   than the correct "hasn't been". Removing a word makes the sentence shorter and more common.

---

## Overall conclusions

- **Gating matters, and you can see why:** we measured the vanishing gradient directly, and the gated
  models both learn more and keep improving longer. In our setting the GRU is the most efficient cell.
- **LSTMs are inspectable:** single cells have identifiable, causal jobs. Gates behave in interpretable
  ways (forgetting at sentence ends, writing freely, revealing little).
- **Architecture choices are explanations:** the copy gate improves quality *and* gives a faithful
  word-level account of what the model changed.
- **Metrics disagree, so report several:** BLEU, style score, anchors kept and paired likelihood each
  tell part of the story, and BLEU on its own would be misleading here.

## Limitations

One random seed per model (bootstrap CIs cover test-set noise, not training noise). Small models trained
on CPU for minutes to an hour. Synthetic drift benchmark. "Correct word" in calibration means it appears
in the reference. The style score comes from a simple classifier.
