"""FastAPI backend for the Shakespearean Language Lab web app.

    uvicorn server:app --reload --port 8000

Endpoints used by the frontend (frontend/ folder):
    GET  /api/status      which trained models are available
    POST /api/generate    text generator + everything the visualizer shows for each character:
                          the probabilities it was sampled from, the gate values and the memory
    POST /api/translate   translator + word map, meaning guard, attention, copy gate, beam candidates
    GET  /results/...     the figures and reports written by the experiments
"""
import json
import os

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from char_lm import load_generator
from data import detokenize, tokenize
from explain import (change_summary, difficulty, glossary, meaning_guard,
                     restore_name_capitals, word_map)
from generate import generate
from lexicon import baseline_translate
from style_classifier import shakespeare_score

app = FastAPI(title="Shakespearean Language Lab")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])
os.makedirs(config.RESULTS_DIR, exist_ok=True)
app.mount("/results", StaticFiles(directory=config.RESULTS_DIR), name="results")

# ---------------------------------------------------------------- models
try:
    from translate import Translator
    translator = Translator()
except FileNotFoundError as error:
    print("No trained translator, using the dictionary baseline:", error)
    translator = None

# Generators offered in the app, best first (name of the checkpoint -> label)
GENERATORS = {
    "lstm_long": "LSTM · 9k steps (best)",
    "gru_matched_long": "GRU · 9k steps",
    "rnn_matched_long": "Vanilla RNN · 9k steps",
    "lstm": "LSTM · 3k steps",
    "gru": "GRU · 3k steps",
    "rnn": "Vanilla RNN · 3k steps",
}
MAX_UNITS = 256            # memory units sent to the visualizer (enough for the heat map)
generators = {}            # loaded on first use: {"lstm_long": (model, vocab), ...}


def get_generator(name):
    if name not in GENERATORS:
        raise HTTPException(status_code=404, detail=f"unknown model {name}")
    if name not in generators:
        try:
            generators[name] = load_generator(name)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error))
    return generators[name]


def interesting_units(model_name):
    """Cells found by exp_inside_lstm.py for this model: {"inside_word (cell #220)": 220, ...}"""
    path = os.path.join(config.RESULTS_DIR, "interesting_units.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        found = json.load(f)
    if found.get("model") != model_name:
        return {}
    return {f"{name.replace('_', ' ')} (cell #{v['unit']})": v["unit"]
            for name, v in found["features"].items()}


def rounded(values, digits=3):
    return [round(v, digits) for v in values]


# ---------------------------------------------------------------- endpoints
@app.get("/api/status")
def status():
    return {
        "translator": "lstm" if translator else "baseline",
        "generators": [{"id": name, "label": label} for name, label in GENERATORS.items()
                       if os.path.exists(config.generator_path(name))],
    }


class GenerateRequest(BaseModel):
    prompt: str = "ROMEO:"
    length: int = Field(300, ge=10, le=800)
    temperature: float = Field(0.7, gt=0, le=2.0)
    model: str = "lstm_long"


@app.post("/api/generate")
def generate_text(request: GenerateRequest):
    model, vocab = get_generator(request.model)
    text, recorded = generate(model, vocab, request.prompt, request.length,
                              request.temperature, record=True)
    cell = type(model.cell).__name__.replace("Cell", "").lower()       # "lstm", "gru" or "rnn"
    gate_names = [g for g in ("input", "forget", "output", "update", "reset") if g in recorded[0]]

    steps = []
    for ch, r in zip(text, recorded):
        top = r["probs"].topk(8)
        steps.append({
            "p": round(float(r["probs"][vocab.index[ch]]), 4),          # probability of the chosen char
            "top": [[vocab.chars[i], round(p, 4)] for p, i in zip(top.values.tolist(), top.indices.tolist())],
            "gates": {g: round(float(r[g].mean()), 3) for g in gate_names},
        })

    # memory heat map: the LSTM's cell state (squashed to -1..1), otherwise the hidden state
    memory_key = "cell" if cell == "lstm" else "hidden"
    memory = [rounded(torch.tanh(r[memory_key][:MAX_UNITS]).tolist() if memory_key == "cell"
                      else r[memory_key][:MAX_UNITS].tolist(), 2) for r in recorded]
    neurons = {label: [m[unit] for m in memory] for label, unit in interesting_units(request.model).items()
               if unit < MAX_UNITS}

    lines = [line for line in text.split("\n") if len(line.strip()) > 10]
    style = shakespeare_score(lines) if lines else []
    return {
        "prompt": request.prompt,
        "text": text,
        "cell": cell,
        "hidden_size": model.cell.hidden_dim,
        "gate_names": gate_names,
        "steps": steps,                    # one entry per generated character
        "memory_name": "cell state c" if cell == "lstm" else "hidden state h",
        "memory": memory,                  # (characters x units), values -1..1
        "neurons": neurons,                # {name: one value per character} for the neuron viewer
        "style_score": sum(style) / len(style) if style else None,
    }


class TranslateRequest(BaseModel):
    text: str = Field(..., max_length=3000)
    direction: str = "to_modern"        # "to_modern" or "to_shakes"
    decoding: str = "beam"              # "beam" (final system) or "greedy" (for comparison)


def translate_line(line, direction, decoding):
    if translator:
        options = {} if decoding == "beam" else dict(beam_size=1, block_repeats=False, coverage_weight=0, hint_bonus=0)
        return translator.translate(line, direction, **options)
    tokens = tokenize(line, lower=False)
    out = restore_name_capitals(tokens, baseline_translate([t.lower() for t in tokens], direction))
    return {"source_tokens": tokens, "output_tokens": out, "text": detokenize(out),
            "confidences": None, "copy_probs": None, "attention": None, "alternatives": []}


@app.post("/api/translate")
def translate(request: TranslateRequest):
    all_tokens = tokenize(request.text, lower=False)
    level, ratio = difficulty(all_tokens)
    lines = []
    for line in request.text.splitlines():
        if not line.strip():
            continue
        result = translate_line(line, request.direction, request.decoding)
        if result is None:
            continue
        items = word_map(result["source_tokens"], result["output_tokens"])
        _, _, warnings = meaning_guard(result["source_tokens"], result["output_tokens"])
        if result["attention"]:
            result["attention"] = [rounded(row) for row in result["attention"]]
        lines.append({**result, "original": line, "items": items,
                      "counts": change_summary(items), "warnings": warnings})
    return {
        "model": "lstm" if translator else "baseline",
        "difficulty": level,
        "archaic_ratio": ratio,
        "glossary": glossary(all_tokens),
        "lines": lines,
    }
