# Shakespearean Language Lab
### Inside the LSTM: generating and transforming Shakespearean English with small recurrent networks

Two working systems built from scratch in PyTorch, plus five experiments that study them:

- **Text generator:** a character-level language model with **hand-written RNN / GRU / LSTM cells**
  (`cells.py`), so every gate can be recorded and visualised.
- **Translator:** a Shakespeare ↔ modern English encoder–decoder with attention and a **copy gate**,
  plus a word map, meaning guard and confidence flags.

- **[PROJECT_PLAN.md](PROJECT_PLAN.md)**: research questions and the pitch
- **[RESULTS.md](RESULTS.md)**: findings of all five experiments, with numbers and figures
- **[GUIDE.md](GUIDE.md)**: presenter's guide (concepts in plain words, likely questions)

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd frontend; npm install; cd ..
python download_data.py
```

## Train

```powershell
python train_generator.py --cell lstm      # ~8 min on a laptop CPU
python train_generator.py --cell gru       # needed for RQ1
python train_generator.py --cell rnn       # needed for RQ1
python train_generator.py --cell gru --hidden 300 --name gru_matched   # same #weights as LSTM
python train_generator.py --cell rnn --hidden 542 --name rnn_matched   # same #weights as LSTM
python style_classifier.py                 # Shakespeare-vs-modern classifier (seconds)
python train_translator.py                 # translator with copy gate
python train_translator.py --no_copy       # ablation for RQ4
python train_translator.py --dropout 0     # no regularisation, to show overfitting (RQ4)
```

Quick checks:

```powershell
python generate.py --cell lstm_large --prompt "My lord, " --temperature 0.7
python generate.py --no_filter          # raw model, without the spelling filter
python translate.py "Wherefore art thou Romeo?"
python translate.py --greedy "Wherefore art thou Romeo?"   # old decoding, for comparison
```

## Run the experiments

Everything (training + experiments) in one go: `python run_all.py`
(or `python run_all.py --skip_train` to only redo the analyses).

| Script | Research question | Output in `results/` |
|---|---|---|
| `python exp_compare_cells.py` | RQ1: RNN vs GRU vs LSTM | `rq1_compare_cells.md`, learning-curve + memory-curve plots |
| `python exp_inside_lstm.py` | RQ2: what do LSTM cells track? | `rq2_inside_lstm.md`, `rq2_inside_lstm.html` |
| `python exp_temperature.py` | RQ3: temperature, spelling filter vs top-p, model size | `rq3_temperature.md`, plot |
| `python exp_copy_gate.py` | RQ4: copy gate and extended vocabulary ablation; is the trace faithful? | `rq4_copy_gate.md`, calibration + overfitting plots |
| `python exp_decoding.py` | RQ4: greedy vs beam search, repeat blocking, coverage | `decoding.md`, `decoding.png` |
| `python exp_drift.py` | RQ5: can meaning drift be detected? | `rq5_drift.md`, `drift_benchmark.tsv` |
| `python evaluate.py` | LLM baseline comparison | `results.md` |

### LLM baseline

```powershell
python evaluate.py --write_llm_input
```

Paste `results/llm_input.txt` into an LLM with this prompt:

> Translate each of the following Shakespearean lines into plain modern English.
> Reply with exactly one translated line per input line, in the same order, with no numbering or extra text.

Save the answers as `results/llm_run1.txt`. Repeat in a new chat, save the second answers as `results/llm_run2.txt`, then run:

```powershell
python evaluate.py --llm results/llm_run1.txt results/llm_run2.txt
```

## Tests

```powershell
python -m pytest tests
```

These check the measuring tools (tokeniser, word map, meaning guard, bootstrap) that the experiments rely on,
plus the decoding rules and the extended vocabulary.

## Web app

Run these in two terminals:

```powershell
uvicorn server:app --port 8000
```

```powershell
cd frontend; npm run dev
```

Open http://localhost:5173. The app has three pages:

- **Generator:** write text with the LSTM, GRU or RNN (an "Only real words" switch turns the spelling
  filter on or off; invented words are underlined), then step through it character by character
  (click, arrow keys or ▶ Play). For each character the **visualizer** shows a live diagram of the cell
  with its gate values, the next-character probabilities the model sampled from, a heat map of the whole
  memory over time, and the gates over time. The text can be coloured by confidence, a gate, or one of
  the cells found in RQ2.
- **Translator:** translation with a word map, meaning guard and glossary. The explanation shows an
  **alignment diagram** (where attention looked), the copy gate for every word, the beam-search
  candidates and an attention heat map. A switch compares beam search with greedy decoding.
- **Research:** the findings of RQ1-RQ5 with their figures, and diagrams of both models.

## Files

| File | What it does |
|---|---|
| `config.py` | all settings |
| `download_data.py` | downloads the corpus |
| `data.py` | word tokeniser, vocabulary, loading pairs |
| `lexicon.py` | archaic dictionary + dictionary baseline |
| `explain.py` | word map, meaning guard, glossary, difficulty |
| `cells.py` | hand-written RNN, GRU and LSTM cells |
| `char_lm.py` | character-level language model |
| `train_generator.py` / `generate.py` | train the generator / sample text |
| `translator_model.py` | encoder–decoder with attention and copy gate |
| `train_translator.py` | train the translator |
| `translate.py` | decoding (beam search, repeat blocking, coverage) + likelihood scoring |
| `style_classifier.py` | logistic-regression style classifier |
| `stats.py` | bootstrap confidence intervals |
| `run_all.py` | runs the whole study in order |
| `evaluate.py` | baseline vs LSTM vs LLM |
| `exp_*.py` | the five experiments |
| `server.py` | FastAPI backend |
| `frontend/` | Vite + plain JavaScript web app (`src/viz/` = the visualizer components) |

## Data credit

Parallel corpus from Jhamtani et al. (2017), *Shakespearizing Modern Language Using Copy-Enriched
Sequence-to-Sequence Models*, and Xu et al. (2012), *Paraphrasing for Style*.
