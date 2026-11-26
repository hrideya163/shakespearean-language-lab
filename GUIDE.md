# Presenter's guide: explaining the project

A short guide for presenting this project and answering questions about it. Every section points to the file
where the idea lives in code.

---

## 1. The 60-second version

We built two small neural networks from scratch and studied them like a scientist studies an organism:

1. A **character-level LSTM generator** that writes Shakespeare-like text one letter at a time.
   We wrote the LSTM equations ourselves (`cells.py`) so we can record every gate.
2. An **LSTM translator** (Shakespearean ↔ modern English) with attention and a copy gate.

Then we ran five experiments: why LSTMs beat plain RNNs, what individual cells learn, how temperature
controls generation, whether the copy gate helps and explains the translation, and whether we can detect
when a translation changes the meaning.

---

## 2. Core concepts in plain words

### Recurrent network (RNN)
Reads a sequence one step at a time and keeps a **hidden state** h (a vector of numbers), its "memory".
`h_t = tanh(W · [x_t, h_{t-1}] + b)`  (see `RNNCell` in `cells.py`)

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
adventurous and error-prone. (`generate.py`)

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

### Extended vocabulary (pointer-generator)
The vocabulary is built from the training plays only. *Romeo and Juliet* is in the test set, so "Romeo"
is an unknown word (`<unk>`). Fix: each unknown **input** word gets a temporary id after the vocabulary,
and the copy distribution puts probability on those ids. The model can now copy a word it has never
seen. Its embedding layer still sees `<unk>`, but the copy gate points at the right word.
(`source_oovs` / `encode_extended` in `data.py`)

### Decoding: beam search, no repeats, coverage
The model only gives probabilities. **Decoding** decides which words to write (`translate.py`):
- **Greedy**: always the single most likely word. Fast, but a small mistake can't be undone, and
  small models get stuck in loops ("agree, agree, agree").
- **Beam search**: keep the 5 most likely partial translations and extend all of them, then pick the
  best finished one (by *average* log-probability, otherwise short outputs win).
- **No repeats**: forbid the previous word again and any 3-word phrase that already appeared.
- **Coverage penalty**: add up the attention each input word received. If a word got almost none, the
  translation probably left it out, so it ranks lower.

### Teacher forcing
During training the decoder is fed the *correct* previous word, not its own guess. That makes
training faster and more stable.

### Bootstrap confidence interval
We resample the test set with replacement 300–1000 times and recompute the metric. The middle 95% of
the results is the interval. Overlapping intervals mean the difference might be luck. (`stats.py`)

### AUROC
For a detector that gives a score: the probability that a random *bad* example gets a higher
"suspicious" score than a random *good* one. 0.5 means guessing, 1.0 means perfect.

---

## 3. What each experiment shows (and how to say it)

| RQ | One-sentence claim to present | Evidence |
|---|---|---|
| RQ1 | "We measured the vanishing gradient directly: 50 characters back the plain RNN keeps 0.001 % of its gradient. With the same number of weights the GRU learns fastest. With 3× more training the LSTM catches up with it, while the RNN gets stuck." | gradient-flow plot, long-budget plot, test bpc table (incl. size-matched models) |
| RQ2 | "Individual LSTM cells learn interpretable jobs, like tracking position in the line or being inside a word, and switching some of them off specifically hurts the related predictions." | correlation table, coloured-text HTML, ablation table, embedding map |
| RQ3 | "Temperature trades correctness for diversity. Around T = 0.8 the generated text's statistics best match real Shakespeare, but the model under-uses rare archaic words." | temperature table and plot |
| RQ4 | "The copy gate gives a free, faithful explanation: kept words have high copy probability, changed words low. Attention points at the replaced word far above chance, and confidence is roughly calibrated." | ablation table with CIs, copy-probability table, alignment %, calibration plot |
| RQ5 | "Rule-based and model-based drift detectors complement each other. The guard catches name swaps that the model can't see (rare names are unknown words), and the model's likelihood catches some negation flips." | drift table, paired test, failure examples |

---

## 4. Questions you may be asked

**Q: Why not just use ChatGPT?**
A: Our goal is to *understand* how recurrent models learn language, not just to produce text. An LLM
can't be opened up like this. We still include an LLM as a baseline in `evaluate.py`.

**Q: Isn't an LSTM outdated?**
A: Transformers replaced LSTMs for large-scale work, but LSTMs are small, run on a CPU, and their
gates make them unusually easy to inspect. The concepts we study (vanishing gradients, gating,
attention, copying) are the foundations transformers were built on.

**Q: Is the comparison between RNN, GRU and LSTM fair?**
A: We compare them twice: at the same hidden size, and at the same number of weights (`_matched`
models). All get the same training steps, data and seed.

**Q: How do you know a cell really does what the correlation suggests?**
A: Correlation isn't causation, so we also *switch the cell off* and compare the damage with switching
off random cells (`exp_inside_lstm.py`). Some cells pass this test and some don't. The ones that don't
probably share their job with other cells (redundancy).

**Q: Your BLEU is lower than the dictionary baseline. Isn't the model bad?**
A: Shakespeare and modern lines share most of their words, so even copying the input unchanged scores
19.5 BLEU. Our model scores 19.2 and the dictionary 21.7, with overlapping confidence intervals. BLEU
rewards keeping words. The style classifier shows our output is clearly more modern (0.45) than the
dictionary's (0.60), which only swaps words and keeps the old sentence structure. Within our own models,
BLEU is still useful: the copy gate adds +4.3 and dropout +1.9, and those differences are significant.

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
about 2,000 steps. Which gated
cell wins depends on budget and task. Gating versus no gating is the bigger difference.

**Q: Your model prefers the correct sentence 90 % of the time, but its AUROC is only 0.61. How?**
A: AUROC compares scores across *different* sentences, and some sentences are just harder to predict
than others. The paired test compares two versions of the *same* sentence, so that difficulty cancels
out. Comparison is a much stronger signal than an absolute score.
