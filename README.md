# Shakespearean Language Lab
### Inside the LSTM: generating and transforming Shakespearean English with small recurrent networks

Two working systems built from scratch in PyTorch, plus five experiments that study them:

- **Text generator:** a character-level language model with **hand-written RNN / GRU / LSTM cells**
  (`lab/generator/cells.py`), so every gate can be recorded and visualised.
- **Translator:** a Shakespeare ↔ modern English encoder–decoder with attention and a **copy gate**,
  decoded with beam search, plus a word map, meaning guard and confidence flags.

Documents:

- **[PROJECT_PLAN.md](PROJECT_PLAN.md)**: research questions and the pitch
- **[RESULTS.md](RESULTS.md)**: findings of all five experiments, with numbers and figures
- **[GUIDE.md](GUIDE.md)**: presenter's guide: concepts in plain words, a walkthrough of the code, likely questions

## Project structure

```
lab/                        the library: everything the models need
  config.py                 all settings (sizes, learning rates, paths)
  data.py                   word tokeniser, vocabulary, loading the parallel corpus
  lexicon.py                archaic word dictionary + dictionary baseline
  explain.py                word map, meaning guard, glossary, difficulty
  stats.py                  bootstrap confidence intervals
  style.py                  Shakespeare-vs-modern style classifier
  generator/
    cells.py                hand-written RNN, GRU and LSTM cells
    model.py                character-level language model
    train.py                training loop
    sample.py               sampling with temperature and the spelling filter
  translator/
    model.py                encoder-decoder with attention and copy gate
    train.py                training loop with early stopping
    decode.py               beam search, repeat blocking, coverage penalty, likelihood scoring
experiments/                one script per research question
  rq1_cells.py              RNN vs GRU vs LSTM
  rq2_inside_lstm.py        what single LSTM cells track
  rq3_sampling.py           temperature, spelling filter, model size
  rq4_copy_gate.py          copy gate, extended vocabulary and dropout ablations
  rq4_decoding.py           greedy vs beam search and the decoding rules
  rq5_drift.py              detecting meaning drift
  evaluate.py               baselines vs LSTM (vs LLM)
tests/                      tests for the text tools, decoding rules and spelling filter
server.py                   FastAPI backend for the web app
frontend/                   Vite + plain JavaScript web app
  src/pages/                generator, translator and research pages
  src/viz/                  the visualizer: cell diagram, memory heat map, gate timeline, alignment
run_all.py                  reproduces the whole study
download_data.py            downloads the corpus
results/                    reports and figures written by the experiments
```

All commands are run from the project folder.

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
python -m lab.generator.train --cell lstm --hidden 512 --steps 9000 --name lstm_large
python -m lab.generator.train --cell lstm --steps 9000 --name lstm_long
python -m lab.style
python -m lab.translator.train
```

The study trains more models (RNN, GRU, size-matched and ablation models); `python run_all.py` trains all of them
and runs every experiment (a few hours on a laptop CPU). `python run_all.py --skip_train` only re-runs the experiments.

Quick checks:

```powershell
python -m lab.generator.sample --prompt "My lord, "
python -m lab.generator.sample --no_filter
python -m lab.translator.decode "Wherefore art thou Romeo?"
python -m lab.translator.decode --greedy "Wherefore art thou Romeo?"
```

## Experiments

| Command | Research question | Output in `results/` |
|---|---|---|
| `python -m experiments.rq1_cells` | RQ1: RNN vs GRU vs LSTM | `rq1_compare_cells.md`, learning-curve, memory and gradient plots |
| `python -m experiments.rq2_inside_lstm` | RQ2: what do LSTM cells track? | `rq2_inside_lstm.md`, `rq2_inside_lstm.html` |
| `python -m experiments.rq3_sampling` | RQ3: temperature, spelling filter vs top-p, model size | `rq3_temperature.md`, plot |
| `python -m experiments.rq4_copy_gate` | RQ4: copy gate, extended vocabulary, dropout; is the trace faithful? | `rq4_copy_gate.md`, calibration + overfitting plots |
| `python -m experiments.rq4_decoding` | RQ4: greedy vs beam search, repeat blocking, coverage (`--tune` picks settings on validation) | `decoding.md`, `decoding.png` |
| `python -m experiments.rq5_drift` | RQ5: can meaning drift be detected? | `rq5_drift.md`, `drift_benchmark.tsv` |
| `python -m experiments.evaluate` | baselines vs LSTM (vs LLM) | `results.md` |

### LLM baseline

```powershell
python -m experiments.evaluate --write_llm_input
```

Paste `results/llm_input.txt` into an LLM with this prompt:

> Translate each of the following Shakespearean lines into plain modern English.
> Reply with exactly one translated line per input line, in the same order, with no numbering or extra text.

Save the answers as `results/llm_run1.txt`. Repeat in a new chat, save the second answers as `results/llm_run2.txt`, then run:

```powershell
python -m experiments.evaluate --llm results/llm_run1.txt results/llm_run2.txt
```

## Tests

```powershell
python -m pytest tests
```

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

## Data credit

Parallel corpus from Jhamtani et al. (2017), *Shakespearizing Modern Language Using Copy-Enriched
Sequence-to-Sequence Models*, and Xu et al. (2012), *Paraphrasing for Style*.
