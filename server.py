import json
import os
import re

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lab import config
from lab.data import detokenize, tokenize
from lab.explain import change_summary, difficulty, glossary, meaning_guard, restore_name_capitals, word_map
from lab.generator.model import load_generator, load_shakespeare_text
from lab.generator.sample import SpellingFilter, generate
from lab.lexicon import baseline_translate
from lab.style import shakespeare_score
from lab.translator.decode import GREEDY, Translator

GENERATORS = {
    "lstm_large": "LSTM · 512 units (largest)",
    "lstm_long": "LSTM · 256 units, 9k steps",
    "gru_matched_long": "GRU · 9k steps",
    "rnn_matched_long": "Vanilla RNN · 9k steps",
    "lstm": "LSTM · 3k steps",
    "gru": "GRU · 3k steps",
    "rnn": "Vanilla RNN · 3k steps",
}
GATES = ("input", "forget", "output", "update", "reset")
MAX_UNITS = 256
WORD = r"[A-Za-z']+"

app = FastAPI(title="Shakespearean Language Lab")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
os.makedirs(config.RESULTS_DIR, exist_ok=True)
app.mount("/results", StaticFiles(directory=config.RESULTS_DIR), name="results")

try:
    translator = Translator()
except FileNotFoundError as error:
    print("No trained translator, using the dictionary baseline:", error)
    translator = None

generators = {}


def get_generator(name):
    if name not in GENERATORS:
        raise HTTPException(status_code=404, detail=f"unknown model {name}")
    if name not in generators:
        try:
            model, vocab = load_generator(name)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error))
        generators[name] = (model, vocab, SpellingFilter(load_shakespeare_text("train"), vocab))
    return generators[name]


def interesting_units(model_name):
    path = os.path.join(config.RESULTS_DIR, "interesting_units.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        found = json.load(f)
    if found.get("model") != model_name:
        return {}
    return {f"{name.replace('_', ' ')} (cell #{v['unit']})": v["unit"] for name, v in found["features"].items()}


def rounded(values, digits=3):
    return [round(v, digits) for v in values]


def describe_step(char, record, vocab, gate_names):
    probs, blocked = record["probs"], record["blocked"]
    top = probs.topk(8)
    return {
        "p": round(float(probs[vocab.index[char]]), 4),
        "top": [[vocab.chars[i], round(p, 4), bool(blocked[i])] for p, i in zip(top.values.tolist(), top.indices.tolist())],
        "blocked": round(float(probs[blocked].sum()), 4),
        "gates": {g: round(float(record[g].mean()), 3) for g in gate_names},
    }


def memory_trace(records, cell_type):
    if cell_type == "lstm":
        return [rounded(torch.tanh(r["cell"][:MAX_UNITS]).tolist(), 2) for r in records]
    return [rounded(r["hidden"][:MAX_UNITS].tolist(), 2) for r in records]


def invented_words(text, spelling):
    finished = re.sub(WORD + "$", "", text)
    return sorted({w for w in re.findall(WORD, finished) if w not in spelling.words})


@app.get("/api/status")
def status():
    return {
        "translator": "lstm" if translator else "baseline",
        "generators": [{"id": name, "label": label} for name, label in GENERATORS.items()
                       if os.path.exists(config.generator_path(name))],
    }


class GenerateRequest(BaseModel):
    prompt: str = "My lord, "
    length: int = Field(300, ge=10, le=800)
    temperature: float = Field(0.7, gt=0, le=2.0)
    model: str = "lstm_large"
    spelling: bool = True


@app.post("/api/generate")
def generate_text(request: GenerateRequest):
    model, vocab, spelling = get_generator(request.model)
    text, records = generate(model, vocab, request.prompt, request.length, request.temperature,
                             record=True, spelling=spelling if request.spelling else None)
    gate_names = [g for g in GATES if g in records[0]]
    memory = memory_trace(records, model.cell_type)
    neurons = {label: [m[unit] for m in memory] for label, unit in interesting_units(request.model).items()
               if unit < MAX_UNITS}
    lines = [line for line in text.split("\n") if len(line.strip()) > 10]
    style = shakespeare_score(lines) if lines else []
    return {
        "prompt": request.prompt,
        "text": text,
        "cell": model.cell_type,
        "hidden_size": model.cell.hidden_dim,
        "gate_names": gate_names,
        "steps": [describe_step(ch, r, vocab, gate_names) for ch, r in zip(text, records)],
        "memory_name": "cell state c" if model.cell_type == "lstm" else "hidden state h",
        "memory": memory,
        "neurons": neurons,
        "style_score": sum(style) / len(style) if style else None,
        "unknown_words": invented_words(text, spelling),
    }


class TranslateRequest(BaseModel):
    text: str = Field(..., max_length=3000)
    direction: str = "to_modern"
    decoding: str = "beam"


def translate_line(line, direction, decoding):
    if translator:
        return translator.translate(line, direction, **({} if decoding == "beam" else GREEDY))
    tokens = tokenize(line, lower=False)
    out = restore_name_capitals(tokens, baseline_translate([t.lower() for t in tokens], direction))
    return {"source_tokens": tokens, "output_tokens": out, "text": detokenize(out),
            "confidences": None, "copy_probs": None, "attention": None, "alternatives": []}


@app.post("/api/translate")
def translate(request: TranslateRequest):
    all_tokens = tokenize(request.text, lower=False)
    level, ratio = difficulty(all_tokens)
    lines = []
    for line in filter(str.strip, request.text.splitlines()):
        result = translate_line(line, request.direction, request.decoding)
        if result is None:
            continue
        items = word_map(result["source_tokens"], result["output_tokens"])
        _, _, warnings = meaning_guard(result["source_tokens"], result["output_tokens"])
        if result["attention"]:
            result["attention"] = [rounded(row) for row in result["attention"]]
        lines.append({**result, "original": line, "items": items, "counts": change_summary(items), "warnings": warnings})
    return {
        "model": "lstm" if translator else "baseline",
        "difficulty": level,
        "archaic_ratio": ratio,
        "glossary": glossary(all_tokens),
        "lines": lines,
    }
