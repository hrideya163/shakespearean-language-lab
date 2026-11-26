"""Decoding study: how much does the way we PICK words matter, with the same trained model?

The model gives a probability for every next word. Decoding decides which word to write.
We add the steps of translate.py one at a time and measure each one:

  1. greedy             always take the single most likely word
  2. + no repeats       forbid the previous word again and repeated 3-word phrases
  3. + beam search      keep the 5 best partial translations instead of 1
  4. + coverage         prefer candidates that paid attention to every input word
  = final system (rows 1-4). Two more ideas, each added on top of row 4, did not help:
  5. + dictionary hints small bonus for the dictionary's suggestion (thou -> you)
  6. + clause splitting translate long inputs clause by clause

Metrics (both directions, TEST set):
  BLEU             overlap with the human reference
  loops %          outputs that repeat a word twice in a row or a 3-word phrase
  length ratio     output length / reference length (1.0 = same length)
  anchors kept %   names, numbers and negations of the input still present
  ms               average time per sentence on the CPU

The two settings (coverage weight, hint bonus) are chosen on the VALIDATION set, never on
the test set, otherwise the test score would be optimistic:
    python exp_decoding.py --tune     # grid search on 300 validation sentences
    python exp_decoding.py            # final study on 500 test sentences per direction
Output: results/decoding.md, results/decoding.png (and results/decoding_tuning.md)
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

OFF = dict(coverage_weight=0, hint_bonus=0)
SETTINGS = [
    ("1. greedy", dict(beam_size=1, block_repeats=False, **OFF)),
    ("2. + no repeats", dict(beam_size=1, block_repeats=True, **OFF)),
    ("3. + beam search (5)", dict(beam_size=5, block_repeats=True, **OFF)),
    ("4. + coverage penalty (final)", dict(beam_size=5, block_repeats=True, coverage_weight=0.4, hint_bonus=0)),
    ("5. final + dictionary hints", dict(beam_size=5, block_repeats=True, coverage_weight=0.4, hint_bonus=1.0)),
    ("6. final + clause splitting", dict(beam_size=5, block_repeats=True, coverage_weight=0.4, hint_bonus=0,
                                         split=True)),
]
DIRECTIONS = [("to_modern", "Shakespeare -> modern"), ("to_shakes", "Modern -> Shakespeare")]


def has_loop(tokens):
    """True if a word repeats immediately or a 3-word phrase appears twice."""
    if any(a == b for a, b in zip(tokens, tokens[1:])):
        return True
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    return len(trigrams) != len(set(trigrams))


def run(translator, pairs, direction, options):
    """Translate every pair in one direction; return the metrics."""
    sources_text = [s if direction == "to_modern" else m for s, m in pairs]
    references = [tokenize(m if direction == "to_modern" else s) for s, m in pairs]
    start = time.time()
    outputs = [[w.lower() for w in translator.translate(s, direction, **options)["output_tokens"]]
               for s in sources_text]
    m = score([tokenize(s, lower=False) for s in sources_text], references, outputs)
    m["ms"] = 1000 * (time.time() - start) / len(outputs)
    m["loops"] = 100 * sum(has_loop(o) for o in outputs) / len(outputs)
    m["ratio"] = sum(len(o) for o in outputs) / sum(len(r) for r in references)
    return m


def tune(translator, n):
    """Grid search of coverage weight x hint bonus on the validation set (average BLEU of both directions)."""
    pairs = read_split("valid")[:n]
    weights, bonuses = [0.0, 0.2, 0.4, 0.8], [0.0, 0.5, 1.0, 2.0]
    lines = ["# Decoding settings tuned on the validation set", "",
             f"{len(pairs)} validation sentences, average BLEU of both directions (beam search 5, no repeats).", "",
             "| coverage weight \\ hint bonus | " + " | ".join(str(b) for b in bonuses) + " |",
             "|---" * (len(bonuses) + 1) + "|"]
    best = (-1, None)
    for w in weights:
        cells = []
        for b in bonuses:
            options = dict(coverage_weight=w, hint_bonus=b)
            bleu = sum(run(translator, pairs, d, options)["BLEU"] for d, _ in DIRECTIONS) / 2
            cells.append(f"{bleu:.2f}")
            best = max(best, (bleu, options), key=lambda x: x[0])
            print(f"coverage {w}  hint {b}  BLEU {bleu:.2f}", flush=True)
        lines.append(f"| {w} | " + " | ".join(cells) + " |")
    lines += ["", f"Best: {best[1]} (BLEU {best[0]:.2f}). Set COVERAGE_WEIGHT and HINT_BONUS in translate.py."]
    with open(os.path.join(config.RESULTS_DIR, "decoding_tuning.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--tune", action="store_true", help="choose settings on the validation set")
    args = parser.parse_args()
    translator = Translator()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    if args.tune:
        return tune(translator, min(args.n, 300))

    pairs = read_split("test")[:args.n]
    report = ["# Decoding study (same trained model, different ways of picking words)", "",
              f"{len(pairs)} test sentences per direction. Rows 1-4 each add one step; row 4 is the final "
              "system. Rows 5 and 6 add one more idea to row 4 (both were rejected on the validation set "
              "or here).", ""]
    bleu_by_direction = {}
    for direction, title in DIRECTIONS:
        report += [f"## {title}", "",
                   "| decoding | BLEU | loops % | length ratio | anchors kept % | ms / sentence |",
                   "|---|---|---|---|---|---|"]
        bleu_by_direction[title] = []
        for name, options in SETTINGS:
            m = run(translator, pairs, direction, options)
            report.append(f"| {name} | {m['BLEU']:.2f} | {m['loops']:.1f} | {m['ratio']:.2f} | "
                          f"{m['Anchors kept %']:.1f} | {m['ms']:.0f} |")
            bleu_by_direction[title].append(m["BLEU"])
            print(direction, name, f"BLEU {m['BLEU']:.2f}  loops {m['loops']:.1f}%", flush=True)
        report.append("")

    with open(os.path.join(config.RESULTS_DIR, "decoding.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    plt.figure(figsize=(7, 4))
    labels = [name.split(". ")[1].replace(" (final)", "").replace("final ", "") for name, _ in SETTINGS]
    for title, values in bleu_by_direction.items():
        plt.plot(range(len(values)), values, marker="o", label=title)
    plt.axvline(3, color="grey", linestyle="--", linewidth=1)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("BLEU"); plt.title("Decoding: what each step adds (same model, dashed = final)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "decoding.png"), dpi=120)
    print("\n".join(report))
    print("Written: results/decoding.md, results/decoding.png")


if __name__ == "__main__":
    main()
