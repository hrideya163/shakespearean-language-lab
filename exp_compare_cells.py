"""RQ1: Why LSTM?  Compare vanilla RNN, GRU and LSTM text generators.

First train the three models (same size, same number of steps):
    python train_generator.py --cell lstm
    python train_generator.py --cell gru
    python train_generator.py --cell rnn
Then run:
    python exp_compare_cells.py

Outputs (in results/):
    rq1_compare_cells.md     table + sample text
    rq1_learning_curves.png  validation bits-per-character during training
    rq1_memory_curve.png     loss at each position of a 200-character window

The MEMORY CURVE: every window starts with an empty memory. At position 1 the
model knows nothing, and at position 150 it has read 150 characters. If a model
uses long context well, its loss keeps dropping as the position grows. A model
that forgets quickly flattens out early.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import config
from char_lm import load_generator, load_shakespeare_text
from generate import generate

CELLS = ["rnn", "gru", "lstm"]
WINDOW = 200


@torch.no_grad()
def memory_curve(model, vocab, text):
    """Average loss (in bits) at each position 0..WINDOW-1, over many windows."""
    data = torch.tensor(vocab.encode(text))
    n = (len(data) - 1) // WINDOW
    n = min(n, 300)
    x = data[:n * WINDOW].view(n, WINDOW)
    y = data[1:n * WINDOW + 1].view(n, WINDOW)
    logits, _, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return (loss.view(n, WINDOW).mean(dim=0) / math.log(2)).tolist()


def smooth(values, k=5):
    return [sum(values[max(0, i - k):i + k + 1]) / len(values[max(0, i - k):i + k + 1])
            for i in range(len(values))]


def main():
    valid_text = load_shakespeare_text("valid")
    rows, curves, logs, samples = [], {}, {}, {}
    for cell in CELLS:
        try:
            model, vocab = load_generator(cell)
        except FileNotFoundError as error:
            print("skipping", cell, "-", error)
            continue
        ckpt = torch.load(config.generator_path(cell), map_location="cpu")
        log = ckpt["log"]
        logs[cell] = log
        curve = memory_curve(model, vocab, valid_text)
        curves[cell] = curve
        best = min(entry["valid_bpc"] for entry in log)
        rows.append({
            "cell": cell.upper(),
            "params (M)": sum(p.numel() for p in model.parameters()) / 1e6,
            "best valid bpc": best,
            "bpc at chars 1-10": sum(curve[:10]) / 10,
            "bpc at chars 150-200": sum(curve[150:]) / len(curve[150:]),
            "train minutes": log[-1]["minutes"],
        })
        samples[cell], _ = generate(model, vocab, "\n", 300, 0.8, seed=1)

    if not rows:
        print("No trained generators found.")
        return
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # table
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else r[c] for c in cols) + " |")
    report = ["# RQ1: RNN vs GRU vs LSTM (character-level generator)", "",
              "bpc = bits per character on validation text (lower is better).", "", *lines, "",
              "## Samples (temperature 0.8)", ""]
    for cell, text in samples.items():
        report += [f"### {cell.upper()}", "```", text.strip(), "```", ""]
    with open(os.path.join(config.RESULTS_DIR, "rq1_compare_cells.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    # plots
    plt.figure(figsize=(6, 4))
    for cell, log in logs.items():
        plt.plot([e["step"] for e in log], [e["valid_bpc"] for e in log], marker="o", label=cell.upper())
    plt.xlabel("training step"); plt.ylabel("validation bits per character")
    plt.title("Learning curves"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "rq1_learning_curves.png"), dpi=120)

    plt.figure(figsize=(6, 4))
    for cell, curve in curves.items():
        plt.plot(range(1, WINDOW + 1), smooth(curve), label=cell.upper())
    plt.xlabel("position in window (characters of context seen)"); plt.ylabel("bits per character")
    plt.title("Memory curve: does more context help?"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "rq1_memory_curve.png"), dpi=120)

    print("\n".join(lines))
    print("\nWritten: results/rq1_compare_cells.md, rq1_learning_curves.png, rq1_memory_curve.png")


if __name__ == "__main__":
    main()
