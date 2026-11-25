"""FastAPI backend for the Shakespearean Language Lab web app.

    uvicorn server:app --reload --port 8000

Endpoints used by the frontend (frontend/ folder):
    GET  /api/status      which trained models are available
    POST /api/generate    LSTM text generator + neuron values for the neuron viewer
    POST /api/translate   LSTM translator + word map, meaning guard, glossary
"""
import json
import os

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# ---------------------------------------------------------------- models
try:
    from translate import Translator
    translator = Translator()
except FileNotFoundError as error:
    print("No trained translator, using the dictionary baseline:", error)
    translator = None

generators = {}            # loaded on first use: {"lstm": (model, vocab), ...}


def get_generator(cell):
    if cell not in generators:
        try:
            generators[cell] = load_generator(cell)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error))
    return generators[cell]


def interesting_units():
    """Cells found by exp_inside_lstm.py, e.g. {"position_in_line": {"unit": 17, ...}}"""
    path = os.path.join(config.RESULTS_DIR, "interesting_units.json")
    if os.path.exists(path):
        with open(path) as f:
            return {f"{name} (cell #{v['unit']})": v["unit"] for name, v in json.load(f).items()}
    return {f"cell #{u}": u for u in range(4)}


# ---------------------------------------------------------------- endpoints
@app.get("/api/status")
def status():
    return {
        "translator": "lstm" if translator else "baseline",
        "generators": [c for c in ["lstm", "gru", "rnn"] if os.path.exists(config.generator_path(c))],
    }


class GenerateRequest(BaseModel):
    prompt: str = "ROMEO:"
    length: int = Field(300, ge=10, le=1000)
    temperature: float = Field(0.8, gt=0, le=2.0)
    cell: str = "lstm"


@app.post("/api/generate")
def generate_text(request: GenerateRequest):
    model, vocab = get_generator(request.cell)
    text, recorded = generate(model, vocab, request.prompt, request.length,
                              request.temperature, record=True)
    neurons, gate_name = {}, None
    if request.cell == "lstm":
        cells = torch.stack([torch.tanh(r["cell"]) for r in recorded])      # (T, H)
        neurons = {name: cells[:, unit].tolist() for name, unit in interesting_units().items()}
        gate_name = "forget"
    elif request.cell == "gru":
        gate_name = "update"
    gate = [float(r[gate_name].mean()) for r in recorded] if gate_name else []
    lines = [line for line in text.split("\n") if len(line.strip()) > 10]
    style = shakespeare_score(lines) if lines else []
    return {
        "prompt": request.prompt,
        "text": text,
        "neurons": neurons,                 # {name: one value per generated character}
        "gate_name": gate_name,
        "gate": gate,                       # mean gate value per generated character
        "style_score": sum(style) / len(style) if style else None,
    }


class TranslateRequest(BaseModel):
    text: str
    direction: str = "to_modern"        # "to_modern" or "to_shakes"


def translate_line(line, direction):
    if translator:
        return translator.translate(line, direction)
    tokens = tokenize(line, lower=False)
    out = restore_name_capitals(tokens, baseline_translate([t.lower() for t in tokens], direction))
    return {"source_tokens": tokens, "output_tokens": out, "text": detokenize(out),
            "confidences": None, "copy_probs": None, "attention": None}


@app.post("/api/translate")
def translate(request: TranslateRequest):
    all_tokens = tokenize(request.text, lower=False)
    level, ratio = difficulty(all_tokens)
    lines = []
    for line in request.text.splitlines():
        if not line.strip():
            continue
        result = translate_line(line, request.direction)
        if result is None:
            continue
        items = word_map(result["source_tokens"], result["output_tokens"])
        _, _, warnings = meaning_guard(result["source_tokens"], result["output_tokens"])
        lines.append({**result, "original": line, "items": items,
                      "counts": change_summary(items), "warnings": warnings})
    return {
        "model": "lstm" if translator else "baseline",
        "difficulty": level,
        "archaic_ratio": ratio,
        "glossary": glossary(all_tokens),
        "lines": lines,
    }
