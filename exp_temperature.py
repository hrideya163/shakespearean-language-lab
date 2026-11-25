"""RQ3: How does temperature control the generator?

    python exp_temperature.py

For each temperature we generate several samples and measure:
    real words %     share of generated words that exist in the training text
    distinct-2 %     share of different word pairs (higher = more varied)
    repetition %     share of words that repeat the word 1-3 positions earlier
    style score      Shakespeare probability from the style classifier (0..1)

Outputs: results/rq3_temperature.md and results/rq3_temperature.png
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from char_lm import load_generator, load_shakespeare_text
from generate import generate
from style_classifier import shakespeare_score

TEMPERATURES = [0.2, 0.5, 0.8, 1.0, 1.3]
PROMPTS = ["\n", "My lord,", "ROMEO:", "I pray thee", "What", "O, ", "Good morrow", "Come, sir"]
SAMPLES_PER_PROMPT = 2
LENGTH = 300


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def measure(texts, known_words):
    all_words = [w for t in texts for w in words(t)]
    bigrams = [b for t in texts for b in zip(words(t), words(t)[1:])]
    repeats = sum(1 for t in texts for i, w in enumerate(words(t))
                  if w in words(t)[max(0, i - 3):i])
    lines = [line for t in texts for line in t.split("\n") if len(line.strip()) > 10]
    return {
        "real words %": 100 * sum(w in known_words for w in all_words) / max(len(all_words), 1),
        "distinct-2 %": 100 * len(set(bigrams)) / max(len(bigrams), 1),
        "repetition %": 100 * repeats / max(len(all_words), 1),
        "style score": sum(shakespeare_score(lines)) / max(len(lines), 1),
    }


def main():
    model, vocab = load_generator("lstm")
    known_words = set(words(load_shakespeare_text("train")))
    results, examples = {}, {}
    for temp in TEMPERATURES:
        texts = []
        for p_i, prompt in enumerate(PROMPTS):
            for s in range(SAMPLES_PER_PROMPT):
                text, _ = generate(model, vocab, prompt, LENGTH, temp, seed=100 * p_i + s)
                texts.append(text)
        results[temp] = measure(texts, known_words)
        examples[temp] = PROMPTS[1] + texts[2]
        print(f"T={temp}: " + ", ".join(f"{k} {v:.2f}" for k, v in results[temp].items()))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    cols = list(next(iter(results.values())).keys())
    lines = ["# RQ3: Effect of temperature on the LSTM generator", "",
             f"{len(PROMPTS) * SAMPLES_PER_PROMPT} samples of {LENGTH} characters per temperature.", "",
             "| temperature | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for temp, m in results.items():
        lines.append(f"| {temp} | " + " | ".join(f"{m[c]:.2f}" for c in cols) + " |")
    lines += ["", "## Examples", ""]
    for temp, text in examples.items():
        lines += [f"### T = {temp}", "```", text.strip(), "```", ""]
    with open(os.path.join(config.RESULTS_DIR, "rq3_temperature.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, col in zip(axes, cols):
        ax.plot(TEMPERATURES, [results[t][col] for t in TEMPERATURES], marker="o")
        ax.set_title(col); ax.set_xlabel("temperature"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "rq3_temperature.png"), dpi=120)
    print("Written: results/rq3_temperature.md, rq3_temperature.png")


if __name__ == "__main__":
    main()
