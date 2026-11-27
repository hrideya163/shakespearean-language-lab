# Presenter's guide: explaining the project

A short guide for presenting this project and answering questions about it. Section 2 explains the ideas,
section 3 walks through the code file by file, and sections 4 and 5 cover the results and likely questions.

---

## 1. The 60-second version

We built two small neural networks from scratch and studied them like a scientist studies an organism:

1. A **character-level LSTM generator** that writes Shakespeare-like text one letter at a time.
   We wrote the LSTM equations ourselves (`lab/generator/cells.py`) so we can record every gate.
2. An **LSTM translator** (Shakespearean ↔ modern English) with attention and a copy gate.

Then we ran five experiments: why LSTMs beat plain RNNs, what individual cells learn, how temperature
controls generation, whether the copy gate helps and explains the translation, and whether we can detect
when a translation changes the meaning.

---

## 2. Core concepts in plain words

### Recurrent network (RNN)
Reads a sequence one step at a time and keeps a **hidden state** h (a vector of numbers), its "memory".
`h_t = tanh(W · [x_t, h_{t-1}] + b)`  (see `RNNCell` in `lab/generator/cells.py`)

**Problem:** during training the error signal is multiplied by the weights at every step backwards. It
shrinks (or explodes) exponentially, so the network can't learn from things far in the past. This is the
**vanishing gradient problem**. We *measure* it in RQ1 (`rq1_gradient_flow.png`).

### LSTM
Adds a separate memory, the **cell state c**, and three **gates** (numbers between 0 and 1 from a sigmoid):

| gate | question it answers |
|---|---|
| forget gate f | how much of the old memory to keep? |
| input gate i | how much new information to write? |
| output gate o | how much of the memory to show? |

```
c_t = f * c_{t-1} + i * g        (g = tanh(...) is the new candidate information)
h_t = o * tanh(c_t)
```
Because the memory is updated by **adding** and multiplying by f (not squashed through tanh every
step), the error signal can travel back much further. Trick: we start the forget gate's bias at 1, so the
model remembers by default (`LSTMCell.__init__`).

### GRU
A simpler gated cell with two gates (update and reset) and no separate cell state. It often performs
about as well as an LSTM with fewer weights.

### Language model and bits-per-character
The generator predicts the **next character**. Loss = cross-entropy. **Bits per character (bpc)** = loss
in base 2. Roughly, it's how many yes/no questions you'd need on average to guess the next letter.
Lower is better.

### Temperature
Before sampling we divide the scores by T. T < 1 makes the model confident and repetitive. T > 1 makes it
adventurous and error-prone. (`lab/generator/sample.py`)

### Spelling filter
A character-level model spells one letter at a time, so it can invent words ("knowshorting"). The
spelling filter knows every word of the training text and every beginning of a word. At each step it
crosses out letters that cannot lead to a real word, and only lets a word end (space, punctuation) when it
is a complete real word. The LSTM still makes every choice among the letters that remain. In RQ3 this
raised real words from 95% to 99.7% without making the text less varied. We also tried **top-p**
sampling (drop the unlikely tail); it made the text more repetitive, so it is off. (`SpellingFilter` in
`lab/generator/sample.py`. The app's "Only real words" switch shows the raw model when unticked.)

### Encoder–decoder with attention
- **Encoder**: a bidirectional LSTM reads the whole input sentence.
- **Decoder**: an LSTM writes the output word by word.
- **Attention**: at each output word, the decoder scores every input word, turns the scores into
  weights with softmax, and takes a weighted average. The weights show *where it is looking*.

### Copy gate
A learned switch `p_copy` between 0 and 1:
`final = (1 - p_copy) · generate_distribution + p_copy · copy_distribution`
The copy distribution is just the attention weights placed on the input words. Most words in a
Shakespeare line stay the same in modern English, so copying is very useful. The gate value tells us
whether each word was copied or rewritten.

### Unknown words and the extended vocabulary (a negative result)
The vocabulary is built from the training plays only. *Romeo and Juliet* is in the test set, so "Romeo"
is an unknown word (`<unk>`). Our translator handles this simply: when it writes `<unk>`, we replace it
by the input word its attention points at (`Translator.words` in `lab/translator/decode.py`).
We also tried the textbook fix, the **extended vocabulary** of pointer-generator networks (See et al.,
2017): each unknown input word gets a temporary id, so the copy gate can copy it directly
(`python -m lab.translator.train --extended`). It did **not** help. It kept fewer names (61% vs 70%), because the
simple fallback already copies them and the harder training target made the model worse. Reporting
ideas that failed is part of the study.

### Decoding: beam search, no repeats, coverage
The model only gives probabilities. **Decoding** decides which words to write (`lab/translator/decode.py`):
- **Greedy**: always the single most likely word. Fast, but a small mistake can't be undone, and
  small models get stuck in loops ("agree, agree, agree").
- **Beam search**: keep the 5 most likely partial translations and extend all of them, then pick the
  best finished one (by *average* log-probability, otherwise short outputs win).
- **No repeats**: forbid the previous word again and any 3-word phrase that already appeared.
- **Coverage penalty**: add up the attention each input word received. If a word got almost none, the
  translation probably left it out, so it ranks lower.
- **Tuned on validation, measured on test**: the coverage weight was picked on the validation set
  (`python -m experiments.rq4_decoding --tune`). Two other ideas, dictionary hints and clause splitting, were measured and
  rejected.

### Teacher forcing
During training the decoder is fed the *correct* previous word, not its own guess. That makes
training faster and more stable.

### Bootstrap confidence interval
We resample the test set with replacement 300–1000 times and recompute the metric. The middle 95% of
the results is the interval. Overlapping intervals mean the difference might be luck. (`lab/stats.py`)

### AUROC
For a detector that gives a score: the probability that a random *bad* example gets a higher
"suspicious" score than a random *good* one. 0.5 means guessing, 1.0 means perfect.

---

## 3. Code walkthrough (what to say about each file)

The code has no comments on purpose: names say what things do, and this section is the explanation.
Read the files in this order.

### `lab/config.py`
Every number we chose: model sizes, learning rates, batch sizes, maximum sentence length, file paths.
Changing an experiment means changing one value here.

### `lab/data.py` (words → numbers)
- `tokenize` splits text with one regular expression that keeps archaic contractions together
  (`'tis`, `o'er`, `speak'st`). `detokenize` glues tokens back into readable text.
- `Vocab` maps words to ids (`encode`) and back (`decode`). Words seen fewer than `MIN_FREQ` times
  become `<unk>`. The first ids are the special tokens: padding, unknown, start, end, and the two
  **direction tags** `<to_modern>` / `<to_shakes>`: one model translates both ways and the tag says which.
- `make_examples` turns every sentence pair into two training examples, one per direction.
- `source_oovs` / `encode_extended` are only used by the extended-vocabulary ablation.

### `lab/generator/cells.py` (the heart of the project)
Three hand-written cells with the same interface: `init_state` gives an empty memory, and `forward(x, state)`
returns the new hidden state, the new memory and a dictionary of gate values (so we can record them).
- `RNNCell`: `h = tanh(W [x, h])`. No gates.
- `GRUCell`: update gate `z` and reset gate `r`; `h = (1 - z) * h + z * n`.
- `LSTMCell`: one linear layer produces four blocks, split with `chunk(4)` into input gate `i`,
  forget gate `f`, candidate `g` and output gate `o`. Then `c = f * c + i * g` and `h = o * tanh(c)`.
  The forget-gate bias starts at 1 so the cell remembers by default.

### `lab/generator/model.py`
- `CharVocab`: characters ↔ ids.
- `CharLM`: embedding → one cell, applied character by character in a `for` loop → linear layer giving
  a score for every possible next character. `record=True` stores the gates at every step (used by the
  visualizer and RQ2); `ablate` switches chosen memory units off (RQ2's causal test).
- `load_shakespeare_text` and `load_generator` load the data and a trained model.

### `lab/generator/train.py`
Standard training loop: take a random batch of 100-character windows, predict each next character,
cross-entropy loss, backpropagation, **gradient clipping** (stops exploding gradients), Adam step.
Every 250 steps it measures validation bits per character and saves the model only if it improved
(**early stopping** by checkpoint).

### `lab/generator/sample.py`
`generate` reads the prompt, then repeatedly: scores → softmax with **temperature** → remove what the
filters forbid → sample one character → feed it back in. `SpellingFilter` keeps the set of real words
and of all word beginnings; `allowed` returns, for every character, whether it may come next.
`top_p_filter` is the rejected nucleus-sampling option.

### `lab/translator/model.py`
- `Encoder`: embedding → bidirectional LSTM over the input sentence (packed so padding is ignored); the
  final forward and backward states are combined into the decoder's starting state.
- `Decoder.step` writes **one** word: LSTM step → attention scores (dot product with every input word)
  → softmax → context vector → probability of every vocabulary word (`generate_probs`) → copy
  probabilities (the attention weights placed on the input words, `scatter_add`) → the copy gate mixes
  the two: `(1 - p_copy) * generate_probs + p_copy * copy_probs`.
- `Seq2Seq.forward` uses **teacher forcing**: at every step the decoder gets the correct previous word.

### `lab/translator/train.py`
`collate` pads a batch to equal length. `run_epoch` does one pass over the data (training or validation).
`train` loops over epochs with **early stopping**: stop after `PATIENCE` epochs without improvement.
`checkpoint_for` picks the file name for the normal model and each ablation.

### `lab/translator/decode.py` (how a translation is chosen)
- `Translator.decode` is **beam search**: keep `BEAM_SIZE` partial translations (`Hypothesis`), run the
  decoder for all of them at once, add every possible next word, keep the best `BEAM_SIZE`.
- `mask` applies the rules before choosing: never write special tokens, no empty translation,
  `blocked_words` (no word twice in a row, no repeated three-word phrase), optional dictionary hints.
- `Hypothesis.score` ranks finished translations: average log-probability plus the **coverage penalty**.
- `words` turns ids back into words; a written `<unk>` becomes the input word the attention points at.
- `translate` puts it together and returns the words plus everything the app shows: confidence, copy
  probability and attention for each word, and the top three beam candidates.
- `score` measures how likely the model finds a given translation (used in RQ5).
- `GREEDY` is the plain one-candidate setting, used for comparisons.

### `lab/explain.py`, `lab/lexicon.py`, `lab/style.py`, `lab/stats.py`
- `lexicon.py`: about 150 archaic words with their modern meaning, and the word-swapping baseline.
- `explain.py`: `word_map` aligns input and output words (kept / swapped / reworded / added / removed)
  with Python's `SequenceMatcher`; `find_anchors` finds names, numbers and negations; `meaning_guard`
  warns when one of them disappears.
- `style.py`: TF-IDF of character 2–4-grams + logistic regression → "how Shakespearean is this text?"
- `stats.py`: `bootstrap_ci`, resample the test set 1,000 times → 95 % confidence interval.

### `experiments/`
One script per research question. Each loads trained models, measures something, and writes a Markdown
report and figures to `results/`. `run_all.py` runs training and all experiments in order.

### `server.py` and `frontend/`
- `server.py` (FastAPI) has three endpoints. `/api/generate` returns the text plus, for every character,
  the model's top guesses, which ones the spelling filter blocked, the gate averages and the whole memory.
  `/api/translate` returns the translation with word map, warnings, attention, copy gate and beam
  candidates. `/results/...` serves the figures.
- `frontend/src/pages/` has one script per page; `frontend/src/viz/` draws the visualizer:
  `cell.js` (SVG cell diagram whose gate boxes fill up to the gate value), `memory.js` (canvas heat map of
  all memory units over time), `timeline.js` (gates over time), `alignment.js` (attention curves between
  input and output words). `utils.js` holds the colour scales and the tooltip.

---

## 4. What each experiment shows (and how to say it)

| RQ | One-sentence claim to present | Evidence |
|---|---|---|
| RQ1 | "We measured the vanishing gradient directly: 50 characters back the plain RNN keeps 0.001 % of its gradient. With the same number of weights the GRU learns fastest. With 3× more training the LSTM catches up with it, while the RNN gets stuck." | gradient-flow plot, long-budget plot, test bpc table (incl. size-matched models) |
| RQ2 | "Individual LSTM cells learn interpretable jobs, like tracking the position in the line or capital letters, and switching some of them off specifically hurts the related predictions. The same cells keep their jobs as training continues." | correlation table, coloured-text HTML, ablation table, embedding map |
| RQ3 | "Temperature trades correctness for diversity. A character model invents words, but a spelling filter makes 99.7% of words real without losing variety. A bigger LSTM is better but overfits quickly." | temperature, sampling-method and model-size tables, plot |
| RQ4 | "The copy gate is the most important part of the translator and a faithful explanation: kept words have high copy probability, changed words low. How we decode matters as much: blocking repeats and a coverage penalty took loops from 14.6% to 0.6%." | ablation table with CIs, copy-probability table, alignment %, decoding study, calibration plot |
| RQ5 | "Rule-based and model-based drift detectors complement each other. The guard catches name swaps that the model can't see (rare names are unknown words), and the model's likelihood catches some negation flips." | drift table, paired test, failure examples |

---

## 5. Questions you may be asked

**Q: Why not just use ChatGPT?**
A: Our goal is to *understand* how recurrent models learn language, not just to produce text. An LLM
can't be opened up like this. We still include an LLM as a baseline in `experiments/evaluate.py`.

**Q: Isn't an LSTM outdated?**
A: Transformers replaced LSTMs for large-scale work, but LSTMs are small, run on a CPU, and their
gates make them unusually easy to inspect. The concepts we study (vanishing gradients, gating,
attention, copying) are the foundations transformers were built on.

**Q: Is the comparison between RNN, GRU and LSTM fair?**
A: We compare them twice: at the same hidden size, and at the same number of weights (`_matched`
models). All get the same training steps, data and seed.

**Q: How do you know a cell really does what the correlation suggests?**
A: Correlation isn't causation, so we also *switch the cell off* and compare the damage with switching
off random cells (`experiments/rq2_inside_lstm.py`). Some cells pass this test and some don't. The ones that don't
probably share their job with other cells (redundancy).

**Q: Your BLEU is lower than the dictionary baseline. Isn't the model bad?**
A: Shakespeare and modern lines share most of their words, so even copying the input unchanged scores
19.5 BLEU. Our model scores 20.0 and the dictionary 21.7, with overlapping confidence intervals (on the
full test set: 21.6 / 23.1 / 23.9). BLEU rewards keeping words. The style classifier shows our output is
clearly more modern (0.49) than the dictionary's (0.60), which only swaps words and keeps the old sentence
structure. Within our own models BLEU is still useful for comparisons: the copy gate adds +2.2 (and keeps
11 points more names, numbers and negations), dropout +1.7, and the coverage penalty +0.8.

**Q: Why did the translator repeat words ("agree, agree, agree") in the first version?**
A: Greedy decoding: small seq2seq models often fall into loops, and 14.6% of greedy outputs did. Blocking
repeated words and 3-word phrases, plus beam search, brings that to 0.6%. We measured each step
(`experiments/rq4_decoding.py`), which is a nice example of how much decoding matters, separately from training.

**Q: Why is the drift benchmark synthetic?**
A: Real meaning errors are rare and expensive to label. Controlled corruptions (negation, names,
numbers) give us thousands of labelled examples, and we know exactly what changed. The limitation is
that real drift can be subtler, as stated in the plan.

**Q: What would you do with more time or compute?**
A: Run several random seeds (we have one per model), use subword units so rare names and archaic words
aren't unknown or rare, train larger models on a GPU, and collect a small human evaluation.

**Q: Why does the LSTM lose to the GRU if LSTMs are "better"?**
A: The GRU has fewer gates, so with a small budget it learns faster. When we trained 3× longer, the
LSTM caught up with the GRU (test bpc 2.166 vs 2.162), while the size-matched RNN stopped improving after
about 2,000 steps. Which gated cell wins depends on budget and task. Gating versus no gating is the bigger difference.

**Q: Your model prefers the correct sentence 90 % of the time, but its AUROC is only 0.61. How?**
A: AUROC compares scores across *different* sentences, and some sentences are just harder to predict
than others. The paired test compares two versions of the *same* sentence, so that difficulty cancels
out. Comparison is a much stronger signal than an absolute score.
