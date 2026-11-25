"""RQ4: Does the copy gate help translation, and is it a faithful explanation?

First train both translators:
    python train_translator.py              -> checkpoints/translator.pt
    python train_translator.py --no_copy    -> checkpoints/translator_nocopy.pt
Then run:
    python exp_copy_gate.py

Part A  ABLATION      compare baseline / no-copy / copy on the test set:
                      BLEU (with a 95% bootstrap confidence interval), anchors kept,
                      kept-word rate, style score (lower = more modern)
Part B  ARCHAIC WORDS for every archaic word in the input (thou, hath, 'tis ...),
                      was its modern meaning produced? Overall and for the most common words.
Part C  TRACE         for the copy model:
                      - average copy probability per word-map label (kept vs changed)
                      - attention as alignment: for each changed word, does the model's
                        attention point at the input word it replaced?

Output: results/rq4_copy_gate.md
"""
import argparse
import os
from collections import Counter

import sacrebleu

import config
from data import detokenize, read_split, tokenize
from evaluate import score
from explain import word_map
from lexicon import baseline_translate, meaning_of
from stats import bootstrap_ci
from style_classifier import shakespeare_score
from translate import Translator


def bleu_ci(outputs, references):
    def metric(idx):
        return sacrebleu.corpus_bleu([" ".join(outputs[i]) for i in idx],
                                     [[" ".join(references[i]) for i in idx]], tokenize="none").score
    return bootstrap_ci(metric, len(outputs), n_resamples=300)


def archaic_hits(sources, outputs):
    """Per archaic word: [times translated correctly, times seen]."""
    counts = {}
    for src, out in zip(sources, outputs):
        out_set = set(out)
        for word in src:
            meaning = meaning_of(word)
            if meaning is None:
                continue
            hit = all(m in out_set for m in meaning.lower().split())
            entry = counts.setdefault(word.lower(), [0, 0])
            entry[0] += hit
            entry[1] += 1
    return counts


def attention_alignment(results):
    """For changed words (swapped / reworded): does attention point at the replaced input word?"""
    correct = total = 0
    for r in results:
        src_lower = [t.lower() for t in r["source_tokens"]]
        out_i = 0
        for item in word_map(r["source_tokens"], r["output_tokens"]):
            for _ in item["out"]:
                if item["type"] in ("swapped", "reworded") and item["src"].lower() in src_lower:
                    attn = r["attention"][out_i]
                    total += 1
                    correct += attn.index(max(attn)) == src_lower.index(item["src"].lower())
                out_i += 1
    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of test sentences")
    args = parser.parse_args()

    pairs = read_split("test")[:args.n]
    sources_text = [s for s, _ in pairs]
    sources_cased = [tokenize(s, lower=False) for s in sources_text]
    sources = [[t.lower() for t in s] for s in sources_cased]
    references = [tokenize(m) for _, m in pairs]

    systems = {"Dictionary baseline": [baseline_translate(s) for s in sources]}
    results_by_model = {}
    for name, path in [("LSTM without copy gate", config.TRANSLATOR_NOCOPY_PATH),
                       ("LSTM with copy gate", config.TRANSLATOR_PATH)]:
        if not os.path.exists(path):
            print("skipping", name, "- missing", path)
            continue
        translator = Translator(path)
        results = [translator.translate(s) for s in sources_text]
        results_by_model[name] = results
        systems[name] = [[w.lower() for w in r["output_tokens"]] for r in results]
        print("translated with", name)

    # ---------------- Part A: ablation table
    rows = {}
    for name, outputs in systems.items():
        rows[name] = score(sources_cased, references, outputs)
        low, high = bleu_ci(outputs, references)
        rows[name]["BLEU 95% CI"] = f"{low:.1f} - {high:.1f}"
        styles = shakespeare_score([detokenize(o) for o in outputs])
        rows[name]["Style score"] = sum(styles) / len(styles)
    rows["Human reference"] = {"Style score": sum(shakespeare_score(
        [detokenize(r) for r in references])) / len(references)}
    src_style = sum(shakespeare_score(sources_text)) / len(sources_text)

    cols = ["BLEU", "BLEU 95% CI", "Anchors kept %", "Kept-word rate %", "Style score"]
    lines = [f"# RQ4: Copy gate ablation ({len(pairs)} test sentences, Shakespeare -> modern)", "",
             "- **Style score** = Shakespeare probability from the style classifier "
             f"(lower = more modern; the Shakespeare input scores {src_style:.2f}).",
             "- **BLEU 95% CI** = bootstrap confidence interval (300 resamples). If two intervals "
             "overlap a lot, the difference may be luck.", "",
             "## A. Ablation", "",
             "| System | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in rows.items():
        cells = [m[c] if isinstance(m.get(c), str) else (f"{m[c]:.2f}" if c in m else "-") for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    # ---------------- Part B: archaic word accuracy
    hits = {name: archaic_hits(sources, outputs) for name, outputs in systems.items()}
    freq = Counter({w: c[1] for w, c in hits["Dictionary baseline"].items()})
    names = list(systems.keys())
    lines += ["", "## B. Were archaic words modernised?", "",
              "A hit = the dictionary meaning of the archaic word appears in the output "
              "(thou -> you). The dictionary baseline is 100% by construction, but it ignores context "
              "(it turns *art* into *are* even when *art* means painting).", "",
              "| archaic word (times seen) | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    overall = []
    for name in names:
        correct = sum(c[0] for c in hits[name].values())
        seen = sum(c[1] for c in hits[name].values())
        overall.append(f"**{100 * correct / max(seen, 1):.1f}%**")
    lines.append("| **all archaic words** | " + " | ".join(overall) + " |")
    for word, n in freq.most_common(12):
        cells = [f"{100 * hits[name][word][0] / n:.0f}%" for name in names]
        lines.append(f"| {word} -> {meaning_of(word)} ({n}) | " + " | ".join(cells) + " |")

    # ---------------- Part C: is the trace faithful?
    if "LSTM with copy gate" in results_by_model:
        results = results_by_model["LSTM with copy gate"]
        by_type = {}
        for r in results:
            out_i = 0
            for item in word_map(r["source_tokens"], r["output_tokens"]):
                for _ in item["out"]:
                    by_type.setdefault(item["type"], []).append(r["copy_probs"][out_i])
                    out_i += 1
        lines += ["", "## C1. Copy probability by word-map label", "",
                  "If the copy gate is a faithful explanation, KEPT words should be copied "
                  "(high value) and changed words generated (low value).", "",
                  "| label | mean copy probability | words |", "|---|---|---|"]
        for label, values in sorted(by_type.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            lines.append(f"| {label} | {sum(values) / len(values):.3f} | {len(values)} |")

        correct, total = attention_alignment(results)
        lines += ["", "## C2. Attention as alignment", "",
                  f"For {total} changed output words (swapped or reworded), the attention peak "
                  f"pointed at the input word they replaced **{100 * correct / max(total, 1):.1f}%** "
                  f"of the time (a random guess over the input words would be about "
                  f"{100 / (sum(len(s) for s in sources) / len(sources)):.0f}%)."]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "rq4_copy_gate.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nWritten: results/rq4_copy_gate.md")


if __name__ == "__main__":
    main()
