import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from lab import config
from lab.generator.model import load_generator, load_shakespeare_text
from lab.generator.sample import generate

MODELS = ["rnn", "gru", "lstm", "rnn_matched", "gru_matched",
          "lstm_long", "rnn_matched_long", "gru_matched_long"]
WINDOW = 200
GRAD_WINDOW = 100


@torch.no_grad()
def bits_per_char(model, vocab, text, max_chars=100_000):
    data = torch.tensor(vocab.encode(text[:max_chars]))
    n = (len(data) - 1) // config.GEN_SEQ_LEN
    x = data[:n * config.GEN_SEQ_LEN].view(n, -1)
    y = data[1:n * config.GEN_SEQ_LEN + 1].view(n, -1)
    logits, _, _ = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item() / math.log(2)


@torch.no_grad()
def memory_curve(model, vocab, text):
    data = torch.tensor(vocab.encode(text))
    n = min((len(data) - 1) // WINDOW, 300)
    x = data[:n * WINDOW].view(n, WINDOW)
    y = data[1:n * WINDOW + 1].view(n, WINDOW)
    logits, _, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return (loss.view(n, WINDOW).mean(dim=0) / math.log(2)).tolist()


def gradient_flow(model, vocab, text, n_windows=64):
    data = torch.tensor(vocab.encode(text))
    x = data[:n_windows * GRAD_WINDOW].view(n_windows, GRAD_WINDOW)
    y_last = data[GRAD_WINDOW:(n_windows + 1) * GRAD_WINDOW:GRAD_WINDOW][:n_windows]

    emb = model.embedding(x).detach().requires_grad_(True)
    state = model.cell.init_state(n_windows, x.device)
    for t in range(GRAD_WINDOW):
        h, state, _ = model.cell(emb[:, t], state)
    loss = F.cross_entropy(model.out(h), y_last)
    loss.backward()
    size = emb.grad.norm(dim=2).mean(dim=0)
    return list(reversed(size.tolist()))


def read_log(name):
    path = os.path.join(config.RESULTS_DIR, f"generator_{name}_log.csv")
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def smooth(values, k=5):
    return [sum(values[max(0, i - k):i + k + 1]) / len(values[max(0, i - k):i + k + 1])
            for i in range(len(values))]


def main():
    valid_text = load_shakespeare_text("valid")
    test_text = load_shakespeare_text("test")
    rows, curves, grads, logs, samples = [], {}, {}, {}, {}
    for name in MODELS:
        try:
            model, vocab = load_generator(name)
        except FileNotFoundError:
            print("skipping", name, "(not trained)")
            continue
        if not os.path.exists(os.path.join(config.RESULTS_DIR, f"generator_{name}_log.csv")):
            print("skipping", name, "(still training)")
            continue
        log = read_log(name)
        logs[name], curves[name] = log, memory_curve(model, vocab, valid_text)
        grads[name] = gradient_flow(model, vocab, valid_text)
        g = grads[name]
        rows.append({
            "model": name.upper(),
            "hidden": model.cell.hidden_dim,
            "recurrent weights (k)": sum(p.numel() for p in model.cell.parameters()) / 1e3,
            "best valid bpc": min(entry["valid_bpc"] for entry in log),
            "test bpc": bits_per_char(model, vocab, test_text),
            "memory gain (bpc)": sum(curves[name][:10]) / 10 - sum(curves[name][150:]) / 50,
            "gradient 10 back (%)": 100 * g[10] / g[0],
            "gradient 50 back (%)": 100 * g[50] / g[0],
            "steps": int(log[-1]["step"]),
            "train minutes": log[-1]["minutes"],
        })
        samples[name], _ = generate(model, vocab, "\n", 300, 0.8, seed=1)
        print(f"done {name}")

    if not rows:
        print("No trained generators found.")
        return
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])
                                       for c in cols) + " |")
    report = ["# RQ1: RNN vs GRU vs LSTM (character-level generator)", "",
              "- **bpc** = bits per character (lower is better).",
              "- **memory gain** = bpc on the first 10 characters of a window minus bpc on characters "
              "150-200: how much reading more context helps.",
              "- **gradient N back** = size of the gradient reaching an input N characters in the "
              "past, as % of the gradient at the most recent input (vanishing gradients -> near 0).",
              "- `_MATCHED` models have about the same number of recurrent weights as the LSTM.", "",
              *lines, "", "## Samples (temperature 0.8)", ""]
    for name, text in samples.items():
        report += [f"### {name.upper()}", "```", text.strip(), "```", ""]
    with open(os.path.join(config.RESULTS_DIR, "rq1_compare_cells.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    short = {n: log for n, log in logs.items() if not n.endswith("_long")}
    long_ = {n: log for n, log in logs.items() if n.endswith("_long")}
    plt.figure(figsize=(6, 4))
    for name, log in short.items():
        plt.plot([e["step"] for e in log], [e["valid_bpc"] for e in log], marker="o", label=name.upper())
    plt.xlabel("training step"); plt.ylabel("validation bits per character")
    plt.title("Learning curves"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "rq1_learning_curves.png"), dpi=120)

    if long_:
        plt.figure(figsize=(6, 4))
        for name, log in long_.items():
            plt.plot([e["step"] for e in log], [e["valid_bpc"] for e in log], label=name.upper())
        plt.xlabel("training step"); plt.ylabel("validation bits per character")
        plt.title("Long training budget (same number of weights)"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, "rq1_long_budget.png"), dpi=120)

    plt.figure(figsize=(6, 4))
    for name, curve in curves.items():
        plt.plot(range(1, WINDOW + 1), smooth(curve), label=name.upper())
    plt.xlabel("position in window (characters of context seen)"); plt.ylabel("bits per character")
    plt.title("Memory curve: does more context help?"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "rq1_memory_curve.png"), dpi=120)

    plt.figure(figsize=(6, 4))
    for name, g in grads.items():
        plt.semilogy(range(len(g)), [v / g[0] for v in g], label=name.upper())
    plt.xlabel("how many characters back"); plt.ylabel("gradient size (relative, log scale)")
    plt.title("Gradient flow: the vanishing gradient problem"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "rq1_gradient_flow.png"), dpi=120)

    print("\n".join(lines))
    print("\nWritten: results/rq1_compare_cells.md + 3 plots")


if __name__ == "__main__":
    main()
