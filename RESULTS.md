# Results: Inside the LSTM

Findings of the five research questions, with the numbers from the final run. Every number here can be
reproduced with `python run_all.py` (the per-experiment reports and all figures are written to `results/`).

**Setup in one line:** character-level generators (hidden size 256, 3,000 steps of 64 × 100 characters) and a
word-level BiLSTM encoder–decoder translator with attention, a copy gate and beam-search decoding (6.4 M parameters), trained on
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

| model | test bpc after 3,000 steps | test bpc after 9,000 steps | training loss 3k → 9k | gradient 50 back, 3k → 9k |
|---|---|---|---|---|
| RNN (matched size) | 2.230 | 2.221 (best checkpoint at step 3,750) | 1.44 → 1.48 (stuck) | 0.03 % → 0.18 % |
| LSTM | 2.227 | **2.166** | 1.49 → 1.35 | 0.007 % → 0.12 % |
| GRU (matched size) | **2.160** | **2.162** (no gain) | 1.36 → 1.32 | 0.32 % → 2.3 % |

**Findings**

1. **The vanishing gradient is real and measurable** (`rq1_gradient_flow.png`). Fifty characters back,
   the plain RNN's gradient is 0.001 % of its size at the current step, so it effectively cannot learn
   from it. The GRU keeps 0.3–0.4 %, about 300× more.
2. **Gating wins, but not in the order the textbook suggests.** With the same number of weights the GRU
   is best and learns fastest. After 3,000 steps the LSTM only ties the size-matched RNN.
3. **The LSTM is a slow starter, not a weaker model** (`rq1_long_budget.png`). With 3× the budget the
   size-matched RNN stops improving at about step 2,000 and then slowly gets worse. Its *training* loss
   doesn't fall either, so this is a learning limit, not overfitting. The LSTM keeps improving until about
   step 7,000 and **catches up with the GRU** (test 2.166 vs 2.162). The GRU had already reached that level
   at 3,000 steps. In short: GRU = fast, LSTM = slow but gets there, RNN = stuck.
4. **Gradient flow is learned, not built in.** After longer training every model passes back much more
   gradient (the LSTM: 0.007 % → 0.12 % at 50 steps back). The gates *can* keep the signal alive, but the
   model has to learn to keep them open. Gradient size alone doesn't decide quality, though. The long RNN
   passes back slightly more gradient than the long LSTM but still can't use it, because without gates it
   has no way to choose *what* to keep.
5. **Most context benefit is short-range.** The memory curves (`rq1_memory_curve.png`) drop sharply over
   the first ~15 characters and are roughly flat after that, for every model (gain 0.35–0.44 bpc). At
   this size, the models mostly use word-level context.

Figures: `rq1_learning_curves.png`, `rq1_long_budget.png`, `rq1_memory_curve.png`, `rq1_gradient_flow.png`.

---

## RQ2: What do individual LSTM cells learn?

Analysed on the best LSTM (`lstm_long`, 9,000 steps). It was trained with the same seed as the
3,000-step model, so its first 3,000 steps are identical, and we can see which cells keep their job.

| feature | most correlated cell | correlation | switching it off hurts more than a random cell? | same cell at 3,000 steps? |
|---|---|---|---|---|
| position in line | #53 | +0.30 | **yes** (+0.0040 vs +0.0003 nats, 13×) | yes |
| capital letter | #77 | +0.33 | **yes** (+0.0050 vs +0.0010, 5×) | yes (was not causal yet) |
| inside a word | #220 | +0.63 | no (+0.0099 vs +0.0061) | no (was #10, causal) |
| after punctuation | #129 | +0.37 | no | yes |
| end of line | #184 | +0.43 | no | no (was #34) |

**Findings**

1. **Some cells are interpretable *and* causal.** Cell #53 tracks the position in the line. Switching
   off that single cell (1 of 256) hurts the newline predictions 13× more than switching off a random
   cell. The coloured text in `rq2_inside_lstm.html` (and the web app's neuron viewer) shows it ramping
   up along each line.
2. **Jobs are stable, but they reorganise.** Three of the five top cells are the same after 3× more
   training. The capital-letter cell *became* causal, while "inside a word" went from one causal
   cell (#10) to being spread over many cells. Training both sharpens some cells and spreads out others.
3. **Correlation is not causation.** The strongest end-of-line cell (+0.43) has no effect when removed,
   so that information must be stored redundantly across several cells.
4. **The forget gate resets at boundaries.** Mean forget value: 0.71 on letters, 0.69 on newlines,
   0.65 on spaces, 0.58 after sentence ends. The model clears memory most at the end of a sentence, as
   intuition suggests.
5. **Gates specialise** (`rq2_gate_saturation.png`). The input gate is almost always open (mean 0.80),
   while the output gate is mostly closed (0.40, 160 of 256 units closed most of the time). The LSTM
   writes a lot to memory but reveals only a little at each step.
6. **The character embedding learns what a character *does*, not what it looks like**
   (`rq2_embeddings.png`, 2-D PCA). Letters form one central cloud. Punctuation and whitespace sit on the
   edges, and the sentence enders `? !` sit next to each other. Upper/lower-case pairs (a/A, e/E, q/Q)
   land close together, although the model was never told they are the same letter. Vowels and
   consonants do *not* separate clearly.

---

## RQ3: Sampling: temperature, spelling filter and model size

Model: the 9,000-step LSTM (`lstm_long`). 16 samples of 300 characters per row.

**A. Temperature (plain sampling)**

| temperature | real words % | distinct-2 % | repetition % | style score | archaic words % |
|---|---|---|---|---|---|
| 0.2 | 98.5 | 31.0 | 25.6 | 0.84 | 0.6 |
| 0.5 | 98.4 | 75.2 | 7.5 | 0.77 | 1.6 |
| **0.8** | 92.8 | **92.4** | **3.2** | **0.78** | **3.8** |
| 1.0 | 86.3 | 95.9 | 1.9 | 0.74 | 3.2 |
| 1.3 | 69.7 | 98.3 | 0.6 | 0.66 | 3.1 |
| *real Shakespeare* | *94.7* | *90.5* | *2.2* | *0.75* | *6.6* |

**B. Sampling method (temperature 0.7, the app's default)**

| method | real words % | distinct-2 % | repetition % | style score |
|---|---|---|---|---|
| plain | 94.9 | 88.0 | 3.6 | 0.78 |
| top-p 0.9 | 97.0 | 83.5 | 4.0 | 0.76 |
| **spelling filter (app)** | **99.7** | **87.9** | **3.4** | 0.73 |
| spelling filter + top-p 0.9 | 99.7 | 82.4 | 4.9 | 0.76 |

**C. Model size (temperature 0.7, spelling filter)**

| model | test bpc | real words % | distinct-2 % | repetition % | archaic words % |
|---|---|---|---|---|---|
| LSTM, 256 units (9,000 steps) | 2.166 | 99.7 | 87.9 | 3.4 | 3.1 |
| **LSTM, 512 units (best checkpoint, step 2,250)** | **2.131** | 99.3 | **88.8** | **2.9** | 2.2 |

**Findings**

1. **Temperature trades correctness for diversity**, monotonically, in every metric. T ≈ 0.8 matches
   real Shakespeare best in diversity, repetition and style score. Low temperatures sound *more*
   Shakespearean to the classifier than real Shakespeare (0.84 vs 0.75), because they repeat the most
   typical phrases ("my lord", "the world").
2. **A character model invents words, and a simple filter fixes that.** Plain sampling makes up about 1
   word in 20 ("knowshorting", "cideation"), which is what makes the output read as gibberish. The
   **spelling filter** only lets the model continue a word towards a real word from the training text.
   It reaches 99.7 % real words *without* making the text less varied or more repetitive. (The remaining
   0.3 % are spelling variants the simple word counter splits differently.)
3. **Top-p sampling is not a free win here.** Cutting the unlikely tail removes some invented words, but
   makes the text more repetitive and less varied, further from real Shakespeare. So the app uses the
   spelling filter only.
4. **Bigger helps, but overfits quickly.** The 512-unit LSTM reaches the best test score (2.131 bpc)
   after only 2,250 steps. After that its training loss keeps falling (1.33 → 1.10) while validation gets
   worse (2.04 → 2.17): with ~940 k characters of text there is not enough data for a model this size.
   Keeping the best checkpoint (early stopping) is essential. Its text is a little more varied and less
   repetitive, so the app uses it by default.
5. **Rare words are under-used.** At every setting the model produces archaic words at about half the
   real rate (2–4 % vs 6.6 %). A small model learns frequent patterns first. The same effect shows up in
   the translator (RQ4).
6. **The honest limit:** the filter guarantees real words and the LSTM produces plausible short phrases
   ("Is it not the tyrant of your house."), but sentences don't carry meaning over many words. That needs
   far more data and parameters than a laptop CPU allows: it is the gap between a character-level LSTM and
   a large language model.

---

## RQ4: Copy gate, unknown words and decoding

All translators below use the same final decoding (beam search, no repeats, coverage penalty).
500 test sentences, Shakespeare → modern:

| system | BLEU (95 % CI) | names/numbers/negations kept % | style score (lower = more modern) |
|---|---|---|---|
| Copy input (no change) | 19.5 (17.5–21.7) | 100 | 0.72 |
| Dictionary baseline | 21.7 (19.5–24.2) | 100 | 0.60 |
| LSTM without copy gate | 17.8 (15.9–19.7) | 64 | **0.38** |
| LSTM + copy gate, no dropout | 18.3 (16.3–20.3) | 72 | 0.51 |
| LSTM + copy gate + extended vocabulary | 19.4 (17.3–21.3) | 69 | 0.48 |
| **LSTM + copy gate + dropout (final)** | **20.0** (18.1–22.0) | **75** | 0.49 |
| *human reference* | – | – | *0.29* |

On the full test set (1,462 sentences, `experiments/evaluate.py`): copy input 21.6, dictionary 23.9, **our LSTM 23.1**.

**Decoding study** (`experiments/rq4_decoding.py`, same trained model, Shakespeare → modern):

| decoding | BLEU | outputs stuck in a loop | names/numbers/negations kept |
|---|---|---|---|
| greedy | 19.2 | 14.6 % | 72 % |
| + no repeats | 19.2 | 1.0 % | 72 % |
| + beam search (5) | 19.2 | 0.4 % | 67 % |
| **+ coverage penalty (final)** | **20.0** | **0.6 %** | **76 %** |
| final + dictionary hints *(rejected)* | 20.0 | 0.6 % | 76 % |
| final + clause splitting *(rejected)* | 18.7 | 8.6 % | 82 % |

The other direction (modern → Shakespeare) shows the same pattern: greedy 17.6 BLEU with 11.8 % loops,
final 17.9 BLEU with 0.2 % loops. The one mixed result: in this direction clause splitting
raised BLEU to 18.4 but brought loops back (5 %), so it stays off. A stricter no-repeat rule (also no word repeated two positions back, to stop "why, why") cost
0.9 BLEU on validation and was rejected too. The coverage weight (0.4) was chosen on the validation set
(`decoding_tuning.md`), where dictionary hints never helped.

**Findings**

1. **Decoding matters as much as the model.** With greedy decoding, 1 in 7 translations gets stuck
   repeating itself ("agree, agree, agree"). A two-line rule (no word twice in a row, no repeated
   3-word phrase) removes almost all loops. Beam search alone drops content (fewer anchors kept),
   because shorter outputs are more likely. The **coverage penalty** fixes that: candidates that ignored
   an input word rank lower, giving +0.8 BLEU and +9 points of names, numbers and negations kept.
2. **The copy gate is the most important part of the model:** +2.2 BLEU and 11 points more anchors kept
   than the same model without it. Dropout adds another +1.7 BLEU. Without it the model overfits after
   epoch 3 (`rq4_overfitting.png`).
3. **A textbook improvement that didn't work.** The pointer-generator *extended vocabulary* (See et al.,
   2017) gives unknown words like new names their own temporary ids, so the copy gate can copy them. Here
   it was *worse* (69 % vs 75 % anchors, 61 % vs 70 % of names). Our simpler fallback already copies
   them: when the model writes `<unk>`, we take the input word its attention points at. The harder
   training target only made the model less accurate.
4. **BLEU alone would pick the wrong system.** Copying the input unchanged scores almost as high as our
   model, because Shakespeare and modern English share most words. The style score shows what BLEU
   misses: our output reads clearly more modern (0.49) than the dictionary baseline's (0.60), which only
   swaps words. The no-copy model is the most modern (0.38) but loses a third of the names, numbers and
   negations: a **faithfulness vs. fluency trade-off**.
5. **The copy gate is a faithful built-in explanation.** Mean copy probability is 0.47 for words that
   were kept, 0.20 for reworded words and 0.03 for swapped words (thou → you). The model's internal switch
   agrees with what actually happened to each word.
6. **Attention works as alignment.** For changed words, the attention peak lands on the input word they
   replaced **67.7 %** of the time (chance ≈ 9 %).
7. **Confidence is usable** (`rq4_calibration.png`). Accuracy rises steadily with confidence
   (35 % → 82 %) and is close to calibrated above 0.6. Low confidence is *under*-confident, partly because
   "correct" means "appears in the reference", a lenient test.
8. **Weaknesses are clear and explainable.** Long sentences (21+ tokens) drop to 14.9 BLEU. Rare archaic
   words (doth, wilt, hence) are almost never modernised, which is the same rare-word effect as in RQ3.

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

- **Gating matters, and you can see why:** we measured the vanishing gradient directly. Given time,
  both gated cells reach the same quality, while the plain RNN stays stuck with the same number of weights.
  The GRU gets there 3× faster.
- **Small sampling choices change how good a model looks:** a spelling filter turns "Shakespearean
  gibberish" into real words without retraining, and the translator's loops disappear with a decoding rule.
- **LSTMs are inspectable:** single cells have identifiable, causal jobs. Gates behave in interpretable
  ways (forgetting at sentence ends, writing freely, revealing little).
- **Architecture choices are explanations:** the copy gate improves quality *and* gives a faithful
  word-level account of what the model changed.
- **How you decode matters as much as what you train:** the same model goes from 1 in 7 sentences stuck
  in a loop to almost none, using rules that fit in a few lines.
- **Negative results are results:** the extended vocabulary, dictionary hints and clause splitting were
  measured and rejected. Settings were chosen on the validation set, never the test set.
- **Metrics disagree, so report several:** BLEU, style score, anchors kept and paired likelihood each
  tell part of the story, and BLEU on its own would be misleading here.

## Limitations

One random seed per model (bootstrap CIs cover test-set noise, not training noise). Small models trained
on CPU for minutes to an hour. Synthetic drift benchmark. "Correct word" in calibration means it appears
in the reference. The style score comes from a simple classifier.
