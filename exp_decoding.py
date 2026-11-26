"""Decoding study: how much does the way we PICK words matter, with the same trained model?

The model gives a probability for every next word. Decoding decides which word to write.
We add the steps of translate.py one at a time and measure each one:

  1. greedy             always take the single most likely word
  2. + no repeats       forbid the previous word again and repeated 3-word phrases
  3. + beam search      keep the 5 best partial translations instead of 1
  4. + coverage         prefer candidates that paid attention to every input word
  5. + clause splitting translate long inputs clause by clause (turned out to hurt,
                        so the final system is row 4)

Metrics (both directions, test set):
  BLEU             overlap with the human reference
  loops %          outputs that repeat a word twice in a row or a 3-word phrase
  length ratio     output length / reference length (1.0 = same length)
  anchors kept %   names, numbers and negations of the input still present
  ms               average time per sentence on the CPU

    python exp_decoding.py            # 500 test sentences per direction
Output: results/decoding.md, results/decoding.png
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data import read_split, tokenize
from evaluate import score
from translate import Translator

SETTINGS = [
    ("1. greedy", dict(beam_size=1, block_repeats=False, coverage_weight=0, split=False)),
    ("2. + no repeats", dict(beam_size=1, block_repeats=True, coverage_weight=0, split=False)),
    ("3. + beam search (5)", dict(beam_size=5, block_repeats=True, coverage_weight=0, split=False)),
    ("4. + coverage penalty (final)", dict(beam_size=5, block_repeats=True, split=False)),
    ("5. + clause splitting", dict(beam_size=5, block_repeats=True, split=True)),
]


def has_loop(tokens):
    """True if a word repeats immediately or a 3-word phrase appears twice."""
    if any(a == b for a, b in zip(tokens, tokens[1:])):
        return True
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    return len(trigrams) != len(set(trigrams))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    translator = Translator()
    pairs = read_split("test")[:args.n]

    report = ["# Decoding study (same trained model, different ways of picking words)", "",
              f"{len(pairs)} test sentences per direction. Each row adds one step to the row above.", ""]
    bleu_by_direction = {}
    for direction, title in [("to_modern", "Shakespeare -> modern"), ("to_shakes", "Modern -> Shakespeare")]:
        sources_text = [s if direction == "to_modern" else m for s, m in pairs]
        references = [tokenize(m if direction == "to_modern" else s) for s, m in pairs]
        sources_cased = [tokenize(s, lower=False) for s in sources_text]

        report += [f"## {title}", "",
                   "| decoding | BLEU | loops % | length ratio | anchors kept % | ms / sentence |",
                   "|---|---|---|---|---|---|"]
        bleu_by_direction[title] = []
        for name, options in SETTINGS:
            start = time.time()
            outputs = [[w.lower() for w in translator.translate(s, direction, **options)["output_tokens"]]
                       for s in sources_text]
            ms = 1000 * (time.time() - start) / len(outputs)
            m = score(sources_cased, references, outputs)
            loops = 100 * sum(has_loop(o) for o in outputs) / len(outputs)
            ratio = sum(len(o) for o in outputs) / sum(len(r) for r in references)
            report.append(f"| {name} | {m['BLEU']:.2f} | {loops:.1f} | {ratio:.2f} | "
                          f"{m['Anchors kept %']:.1f} | {ms:.0f} |")
            bleu_by_direction[title].append(m["BLEU"])
            print(direction, name, f"BLEU {m['BLEU']:.2f}  loops {loops:.1f}%")
        report.append("")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "decoding.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    plt.figure(figsize=(7, 4))
    labels = [name.split(". ")[1] for name, _ in SETTINGS]
    for title, values in bleu_by_direction.items():
        plt.plot(range(len(values)), values, marker="o", label=title)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("BLEU"); plt.title("Decoding: what each step adds (same model)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "decoding.png"), dpi=120)
    print("\n".join(report))
    print("Written: results/decoding.md, results/decoding.png")


if __name__ == "__main__":
    main()
