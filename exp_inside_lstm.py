"""RQ2: What does the LSTM learn?  Look for cells that track simple text features.

    python exp_inside_lstm.py

We run the trained LSTM generator over validation text, record its cell state
and forget gate at every character, and compute the correlation between every
cell (256 of them) and some features we can compute ourselves:

    position_in_line   how far we are into the current line
    inside_word        1 if the current character is a letter
    capital_letter     1 if the current character is upper case
    after_punctuation  1 right after . , ! ? ; :
    end_of_line        1 if the NEXT character is a newline (needs prediction!)

Outputs (in results/):
    rq2_inside_lstm.md      top cell per feature + forget gate by character type
    rq2_inside_lstm.html    validation text coloured by each top cell
    interesting_units.json  used by the web app's neuron viewer
"""
import html
import json
import os

import torch

import config
from char_lm import load_generator, load_shakespeare_text

N_CHARS = 20_000


def text_features(text):
    """One list of numbers per feature, one number per character."""
    feats = {"position_in_line": [], "inside_word": [], "capital_letter": [],
             "after_punctuation": [], "end_of_line": []}
    pos = 0
    for i, ch in enumerate(text):
        pos = 0 if ch == "\n" else pos + 1
        feats["position_in_line"].append(float(pos))
        feats["inside_word"].append(1.0 if ch.isalpha() else 0.0)
        feats["capital_letter"].append(1.0 if ch.isupper() else 0.0)
        feats["after_punctuation"].append(1.0 if i > 0 and text[i - 1] in ".,!?;:" else 0.0)
        feats["end_of_line"].append(1.0 if i + 1 < len(text) and text[i + 1] == "\n" else 0.0)
    return {name: torch.tensor(values) for name, values in feats.items()}


def correlation(units, feature):
    """Pearson correlation between every unit (columns of units) and one feature."""
    u = units - units.mean(dim=0)
    f = feature - feature.mean()
    return (u * f[:, None]).sum(0) / (u.norm(dim=0) * f.norm() + 1e-8)


def char_type(ch):
    if ch == "\n":
        return "newline"
    if ch == " ":
        return "space"
    if ch.isalpha():
        return "letter"
    if ch in ".!?":
        return "sentence end (. ! ?)"
    return "other punctuation"


def coloured_html(text, values):
    """Blue = negative activation, red = positive (like Karpathy et al. 2015)."""
    spans = []
    for ch, v in zip(text, values):
        v = max(-1.0, min(1.0, v))
        colour = f"rgba(220,40,40,{v:.2f})" if v > 0 else f"rgba(40,90,220,{-v:.2f})"
        shown = "↵<br>" if ch == "\n" else html.escape(ch)
        spans.append(f"<span style='background:{colour}'>{shown}</span>")
    return "".join(spans)


@torch.no_grad()
def main():
    model, vocab = load_generator("lstm")
    text = load_shakespeare_text("valid")[:N_CHARS]
    ids = torch.tensor([vocab.encode(text)])
    text = vocab.decode(ids[0].tolist())                 # same characters the model saw
    _, _, recorded = model(ids, record=True)

    cells = torch.tanh(torch.stack([r["cell"][0] for r in recorded]))   # (T, H), squashed to -1..1
    forget = torch.stack([r["forget"][0] for r in recorded])            # (T, H)

    # 1. top unit per feature
    feats = text_features(text)
    top = {}
    for name, feature in feats.items():
        corr = correlation(cells, feature)
        unit = int(corr.abs().argmax())
        top[name] = {"unit": unit, "correlation": float(corr[unit])}

    # 2. average forget gate by character type (low value = the cell is being cleared)
    by_type = {}
    mean_forget = forget.mean(dim=1)
    for ch, f in zip(text, mean_forget.tolist()):
        by_type.setdefault(char_type(ch), []).append(f)

    # 3. write the report
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    lines = ["# RQ2: Inside the LSTM generator", "",
             f"Analysed {len(text):,} validation characters, {cells.shape[1]} cells.", "",
             "## Cell most correlated with each feature", "",
             "| feature | cell | correlation |", "|---|---|---|"]
    lines += [f"| {name} | #{t['unit']} | {t['correlation']:+.2f} |" for name, t in top.items()]
    lines += ["", "## Average forget gate by character type", "",
              "Values near 1 mean the LSTM keeps its memory, and lower values mean it clears part of it.", "",
              "| character type | mean forget gate | count |", "|---|---|---|"]
    for name, values in sorted(by_type.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        lines.append(f"| {name} | {sum(values) / len(values):.3f} | {len(values)} |")
    with open(os.path.join(config.RESULTS_DIR, "rq2_inside_lstm.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    page = ["<html><head><meta charset='utf-8'><title>Inside the LSTM</title></head>",
            "<body style='font-family:monospace;max-width:900px;margin:auto'>"]
    snippet = slice(0, 1500)
    for name, t in top.items():
        values = cells[snippet, t["unit"]].tolist()
        if t["correlation"] < 0:
            values = [-v for v in values]                 # flip so red = feature present
        page.append(f"<h3>{name}: cell #{t['unit']} (r = {t['correlation']:+.2f})</h3>")
        page.append(f"<p>{coloured_html(text[snippet], values)}</p>")
    page.append("</body></html>")
    with open(os.path.join(config.RESULTS_DIR, "rq2_inside_lstm.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(page))

    with open(os.path.join(config.RESULTS_DIR, "interesting_units.json"), "w") as f:
        json.dump(top, f, indent=2)

    print("\n".join(lines))
    print("\nWritten: results/rq2_inside_lstm.md, rq2_inside_lstm.html, interesting_units.json")


if __name__ == "__main__":
    main()
