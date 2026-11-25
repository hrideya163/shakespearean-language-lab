"""RQ4: Does the copy gate help translation, and is it a faithful explanation?

First train both translators:
    python train_translator.py              -> checkpoints/translator.pt
    python train_translator.py --no_copy    -> checkpoints/translator_nocopy.pt
Then run:
    python exp_copy_gate.py

Part A (ablation): compare both models on the test set
    BLEU, anchors kept %, kept-word rate %, style score (lower = more modern)
Part B (trace): for the copy model, average copy probability of output words that
    the word map labels "kept" vs words that were swapped/reworded/added.
    If the copy gate is a faithful explanation, kept words should have a much
    higher copy probability.

Output: results/rq4_copy_gate.md
"""
import argparse
import os

import config
from data import detokenize, read_split, tokenize
from evaluate import score
from explain import word_map
from lexicon import baseline_translate
from style_classifier import shakespeare_score
from translate import Translator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of test sentences")
    args = parser.parse_args()

    pairs = read_split("test")[:args.n]
    sources_text = [s for s, _ in pairs]
    sources_cased = [tokenize(s, lower=False) for s in sources_text]
    references = [tokenize(m) for _, m in pairs]

    systems = {"Dictionary baseline": [baseline_translate([t.lower() for t in s]) for s in sources_cased]}
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

    rows = {}
    for name, outputs in systems.items():
        rows[name] = score(sources_cased, references, outputs)
        styles = shakespeare_score([detokenize(o) for o in outputs])
        rows[name]["Style score"] = sum(styles) / len(styles)
    rows["Human reference"] = {"Style score": sum(shakespeare_score(
        [detokenize(r) for r in references])) / len(references)}

    cols = ["BLEU", "Anchors kept %", "Kept-word rate %", "Style score"]
    lines = [f"# RQ4: Copy gate ablation ({len(pairs)} test sentences, Shakespeare -> modern)", "",
             "Style score = Shakespeare probability from the style classifier (lower = more modern).", "",
             "| System | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.2f}" if c in m else "-" for c in cols) + " |")

    # Part B: is the copy gate a faithful trace?
    if "LSTM with copy gate" in results_by_model:
        by_type = {}
        for r in results_by_model["LSTM with copy gate"]:
            out_i = 0
            for item in word_map(r["source_tokens"], r["output_tokens"]):
                for _ in item["out"]:
                    by_type.setdefault(item["type"], []).append(r["copy_probs"][out_i])
                    out_i += 1
        lines += ["", "## Copy probability by word-map label", "",
                  "| label | mean copy probability | words |", "|---|---|---|"]
        for label, values in sorted(by_type.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            lines.append(f"| {label} | {sum(values) / len(values):.3f} | {len(values)} |")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "rq4_copy_gate.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nWritten: results/rq4_copy_gate.md")


if __name__ == "__main__":
    main()
