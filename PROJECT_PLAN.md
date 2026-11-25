# Shakespearean Language Lab
### Inside the LSTM: a study of how recurrent networks learn, generate and transform Shakespearean English

---

## 1. Summary

We build two working LSTM systems from scratch in PyTorch and use them as **objects of study**:

1. **An LSTM text generator.** It writes new Shakespearean text one character at a time. It uses our own
   hand-written LSTM cell, so every gate (input, forget, output) and the cell state can be recorded.
2. **An LSTM translator.** It is an encoder–decoder with attention and a copy gate, and it rewrites Shakespearean English into
   modern English and back. It shows which words were kept, swapped or reworded, and warns when the meaning may have changed.

The project is not "yet another Shakespeare generator". It is a set of **five small, measurable
experiments** that explain *why* LSTMs work on this task, *what* they learn internally, and *where* they fail.
The systems are only the tools for that.

**Central question:**
> What does a compact LSTM learn about Shakespearean language, and can its internal signals (gates,
> attention, copy gate, probabilities) make its generation and translation controllable and checkable?

## 2. Why this is not just "what an LLM already does"

| | General LLM (ChatGPT, etc.) | This project |
|---|---|---|
| Goal | Produce good text | **Understand and measure** how a recurrent model produces text |
| Model | Billions of parameters, closed or too large to inspect | ≈ 1–10 M parameters, every gate recorded and plotted |
| Explanations | Self-reported, can be wrong | Read directly from the model's internals (gates, attention, copy gate) |
| Meaning checks | None built in | Meaning guard + likelihood-based drift detector, tested on a benchmark |
| Reproducible | Output changes run to run | Deterministic translation; fixed seeds |

We include an LLM as a **baseline** in the translation experiment, to show honestly what a compact model
can and can't do. We don't claim to beat it at fluent writing.

Prior work we build on (and cite): Karpathy et al. (2015), *Visualizing and Understanding Recurrent
Networks* (interpretable LSTM cells); Jhamtani et al. (2017), *Shakespearizing Modern Language Using
Copy-Enriched Seq2Seq Models* (dataset + copy mechanism); See et al. (2017), pointer-generator networks.
Our contribution is applying these ideas together to one domain, plus the new drift benchmark and the joint analysis.
Using an LSTM on Shakespeare is not presented as novel.

## 3. The two systems

### A. LSTM text generator (character level)
- Input: a prompt such as `ROMEO:` or `My lord,`. Output: new Shakespearean text, sampled with a
  **temperature** setting (low = safe and repetitive, high = creative and noisy).
- Model: embedding → **hand-written recurrent cell** → linear layer → softmax over characters.
- The same code can build three cell types (**vanilla RNN, GRU, LSTM**), so we can compare them fairly.
- Trained on the Shakespeare side of the corpus (about 18k lines, about 1 MB of text).

### B. LSTM translator (word level)
- BiLSTM encoder + LSTM decoder + attention + **copy gate** (a learned switch between copying an input word and
  generating a new one). One model handles both directions, using a direction tag.
- Outputs: the translation, plus per-word confidence, attention weights, copy probability, a
  colour-coded **word map** (kept / swapped / reworded / added / removed), and **meaning-guard**
  warnings when names, numbers or negations disappear.

Both systems are available in a small web app (FastAPI + JavaScript), including a **neuron viewer** that
colours generated text by the activation of a chosen LSTM cell.

## 4. Research questions and experiments

| # | Question | Experiment (script) | Measured by |
|---|---|---|---|
| RQ1 | **Why LSTM?** Does the gated cell beat a vanilla RNN and match a GRU? | Train RNN, GRU and LSTM generators with the same hidden size **and** with the same number of weights (`exp_compare_cells.py`) | Validation and **test** bits-per-character, learning curves, **memory curve** (loss vs. context seen), **gradient flow** (the vanishing gradient problem measured directly) |
| RQ2 | **What does the LSTM learn?** Are there cells that track line position, word boundaries, capital letters or sentence ends? When does it "forget"? | Record gates and cells over validation text and correlate each cell with simple text features (`exp_inside_lstm.py`) | Top cell per feature, forget gate per character type, **gate saturation**, **causal ablation** (switch a cell off and compare with random cells), **2-D map of learned character embeddings** |
| RQ3 | **How controllable is generation?** What does temperature do to quality and style? | Generate samples at 5 temperatures (`exp_temperature.py`) | % real words, diversity (distinct bigrams), repetition, **Shakespeare-style score** from a style classifier |
| RQ4 | **Does the copy gate help translation, and is it a faithful trace?** | Train the translator with and without the copy gate (`exp_copy_gate.py`) | BLEU with **95% bootstrap confidence intervals**, kept-word rate, anchors kept, **archaic-word accuracy** per word; copy probability for kept vs. changed words; **attention-as-alignment accuracy** |
| RQ5 | **Can meaning drift be detected automatically?** | New **drift benchmark**: corrupt correct translations (flip a negation, swap a name, change a number) and test two detectors (`exp_drift.py`) | Guard detection / false alarms; likelihood AUROC with confidence interval; **paired test** ("did the model prefer the correct version?"); failure examples |

Plus the **LLM baseline** (`evaluate.py`): the same 200 test lines are translated by an LLM (twice), then
compared on BLEU, anchors kept, kept-word rate, consistency, speed and model size.

### What makes the results interesting (expected findings to check)
- RQ1: the RNN should plateau earlier. LSTM and GRU should use long context better (their memory curve keeps falling).
- RQ2: like Karpathy et al., we expect a few cells to track position-in-line and word boundaries, and the forget gate to
  drop after punctuation.
- RQ4: the copy gate should keep more words unchanged and keep names. Copy probability should be high for
  kept words and low for modernised ones, which makes it a free, faithful explanation.
- RQ5: a rule-based guard catches obvious drift, and the model's likelihood catches some subtler cases. We report where each fails.

## 5. Machine-learning concepts covered

| Concept | Where |
|---|---|
| Character and word tokenisation, vocabularies | `data.py`, `char_lm.py` |
| Embeddings | both models |
| Vanilla RNN, GRU and LSTM **equations written by hand** | `cells.py` |
| Vanishing gradients, gates, forget-gate bias trick | `cells.py`, RQ1 |
| Language modelling, cross-entropy, perplexity, bits-per-character | `char_lm.py`, `train_generator.py` |
| Sampling and temperature | `generate.py`, RQ3 |
| Encoder–decoder, bidirectional LSTM, attention, copy mechanism | `translator_model.py` |
| Teacher forcing, greedy decoding, early stopping, gradient clipping | `train_translator.py`, `translate.py` |
| Logistic regression on character n-grams (style classifier) | `style_classifier.py` |
| Evaluation: BLEU, AUROC, ablation studies, baselines | `evaluate.py`, `exp_*.py` |
| Statistical confidence: bootstrap confidence intervals, held-out test set | `stats.py`, `exp_copy_gate.py`, `exp_drift.py` |
| Vanishing gradients measured with backpropagation | `exp_compare_cells.py` |
| Interpretability: correlation, causal ablation, PCA of embeddings | `exp_inside_lstm.py` |
| Reproducibility: fixed seeds, one-command pipeline | `config.py`, `run_all.py` |

## 6. Data

- **Parallel corpus** (Jhamtani et al. 2017 / Xu et al. 2012): 18,395 train / 1,218 valid / 1,462 test
  aligned Shakespeare–modern line pairs. Downloaded by `download_data.py`.
- The **generator** trains on the Shakespeare side only. The **style classifier** learns Shakespeare-vs-modern from both sides.
- A small **archaic dictionary** (`lexicon.py`) for the glossary, word map labels and the dictionary baseline.
- The **drift benchmark** is generated automatically from the test set (`exp_drift.py`).

## 7. Code structure (flat and beginner-friendly)

```
config.py               all settings
download_data.py        downloads the corpus
data.py                 word tokeniser, vocabulary, loading pairs
lexicon.py              archaic dictionary + dictionary baseline
explain.py              word map, meaning guard, glossary, difficulty

cells.py                hand-written RNN / GRU / LSTM cells (gates exposed)
char_lm.py              character-level language model + character data
train_generator.py      trains the generator (choose --cell rnn|gru|lstm)
generate.py             sampling with temperature + gate recording

translator_model.py     encoder-decoder with attention and copy gate
train_translator.py     trains the translator (--no_copy for the ablation)
translate.py            greedy translation + likelihood scoring
evaluate.py             baseline vs LSTM vs LLM table

style_classifier.py     Shakespeare-vs-modern logistic regression
stats.py                bootstrap confidence intervals
run_all.py              reproduces the whole study with one command
exp_compare_cells.py    RQ1
exp_inside_lstm.py      RQ2
exp_temperature.py      RQ3
exp_copy_gate.py        RQ4
exp_drift.py            RQ5

server.py               FastAPI backend
frontend/               Vite + plain JavaScript web app (Generator tab, Translator tab)
```

## 8. Timeline (8 weeks)

| Week | Work |
|---|---|
| 1 | Data, tokenisers, hand-written cells, style classifier |
| 2 | Generator training; RQ1 (RNN vs GRU vs LSTM) |
| 3 | RQ2 (inside the LSTM) + RQ3 (temperature) |
| 4–5 | Translator training with and without copy gate; RQ4 |
| 6 | Drift benchmark, RQ5; LLM baseline collection |
| 7 | Web app (generator, neuron viewer, translator), figures |
| 8 | Report, slides, demo |

## 9. Limitations

- Small models and a small dataset, so the generator's text is Shakespeare-*like*, not coherent over long passages.
- The style classifier and meaning guard are simple. They are measuring tools, not ground truth.
- The drift benchmark uses synthetic corruptions. Real drift can be subtler.
- The LLM comparison is a small, manually collected sample.

## 10. Pitch for the professor

> We study how compact recurrent networks learn Shakespearean English, using two systems we built from
> scratch in PyTorch: a character-level text generator with hand-written RNN/GRU/LSTM cells, and a
> word-level LSTM translator with attention and a copy gate. Five experiments answer concrete
> questions: why gating beats a vanilla RNN (perplexity and memory curves), which internal cells track
> linguistic features (gate and cell analysis), how temperature trades quality for creativity (style
> classifier and diversity metrics), whether the copy gate improves translation and acts as a faithful
> explanation (ablation), and whether meaning drift can be detected automatically (a new
> corruption benchmark scored by AUROC). An LLM is used as a reference baseline. The deliverable is a
> working generator and translator with a neuron viewer, and a report of measured findings about what
> the LSTM learns and where it fails.
