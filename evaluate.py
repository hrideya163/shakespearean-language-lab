"""Compare the dictionary baseline, our LSTM and (optionally) an LLM on the same test sentences.

Step 1 (optional, for the LLM comparison):
    python evaluate.py --write_llm_input
  -> creates results/llm_input.txt. Paste it into ChatGPT (or another LLM) with the prompt
     from the README, and save the answers, one per line, as results/llm_run1.txt.
     Repeat in a NEW chat and save the answers as results/llm_run2.txt (to measure consistency).

Step 2:
    python evaluate.py --llm results/llm_run1.txt results/llm_run2.txt
  -> prints a results table and writes results/results.md
"""
import argparse
import os
import time

import sacrebleu

import config
from data import read_split, tokenize
from explain import meaning_guard
from lexicon import baseline_translate

PUNCT = set(".,!?;:'\"()-")


def kept_word_rate(src_tokens, ref_tokens, hyp_tokens):
    """Words that appear in BOTH the source and the human reference should stay unchanged.
    Returns (how many of those the system kept, how many there were)."""
    should_keep = {t for t in src_tokens if t not in PUNCT} & set(ref_tokens)
    return len(should_keep & set(hyp_tokens)), len(should_keep)


def score(sources_cased, references, outputs):
    """outputs: list of token lists (lowercase)."""
    bleu = sacrebleu.corpus_bleu([" ".join(o) for o in outputs],
                                 [[" ".join(r) for r in references]], tokenize="none").score
    a_found = a_kept = k_kept = k_total = 0
    for src, ref, out in zip(sources_cased, references, outputs):
        found, kept, _ = meaning_guard(src, out)
        a_found, a_kept = a_found + found, a_kept + kept
        kk, kt = kept_word_rate([t.lower() for t in src], ref, out)
        k_kept, k_total = k_kept + kk, k_total + kt
    return {
        "BLEU": bleu,
        "Anchors kept %": 100 * a_kept / max(a_found, 1),
        "Kept-word rate %": 100 * k_kept / max(k_total, 1),
    }


def consistency(run_a, run_b):
    same = sum(a == b for a, b in zip(run_a, run_b))
    return 100 * same / len(run_a)


def read_lines(path, n):
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) != n:
        raise ValueError(f"{path} has {len(lines)} lines but {n} test sentences were used")
    return [tokenize(line) for line in lines]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of test sentences")
    parser.add_argument("--sample", action="store_true", help="use the tiny sample dataset")
    parser.add_argument("--write_llm_input", action="store_true")
    parser.add_argument("--llm", nargs="*", default=[], help="LLM answer files (1 or 2 runs)")
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    data_dir = config.SAMPLE_DIR if args.sample else config.CORPUS_DIR
    pairs = read_split("test", data_dir)[:args.n]
    sources_text = [s for s, _ in pairs]
    sources_cased = [tokenize(s, lower=False) for s in sources_text]
    references = [tokenize(m) for _, m in pairs]

    if args.write_llm_input:
        path = os.path.join(config.RESULTS_DIR, "llm_input.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(sources_text) + "\n")
        print(f"Wrote {len(pairs)} lines to {path}")
        return

    rows = {}

    # 1. dictionary baseline
    baseline = [baseline_translate([t.lower() for t in s]) for s in sources_cased]
    rows["Dictionary baseline"] = score(sources_cased, references, baseline)

    # 2. our LSTM (measured twice to check consistency, timed on CPU)
    from translate import Translator
    translator = Translator()
    start = time.perf_counter()
    lstm_run1 = [[w.lower() for w in translator.translate(s)["output_tokens"]] for s in sources_text]
    ms_per_sentence = 1000 * (time.perf_counter() - start) / len(pairs)
    lstm_run2 = [[w.lower() for w in translator.translate(s)["output_tokens"]] for s in sources_text]
    rows["Our LSTM"] = score(sources_cased, references, lstm_run1)
    rows["Our LSTM"]["Consistency %"] = consistency(lstm_run1, lstm_run2)
    rows["Our LSTM"]["ms / sentence (CPU)"] = ms_per_sentence
    rows["Our LSTM"]["Model size MB"] = os.path.getsize(config.TRANSLATOR_PATH) / 1e6
    n_params = sum(p.numel() for p in translator.model.parameters())

    # 3. LLM answers (if provided)
    if args.llm:
        llm_runs = [read_lines(path, len(pairs)) for path in args.llm]
        rows["LLM"] = score(sources_cased, references, llm_runs[0])
        if len(llm_runs) > 1:
            rows["LLM"]["Consistency %"] = consistency(llm_runs[0], llm_runs[1])

    # results table
    columns = ["BLEU", "Anchors kept %", "Kept-word rate %", "Consistency %",
               "ms / sentence (CPU)", "Model size MB"]
    lines = ["| System | " + " | ".join(columns) + " |",
             "|---" * (len(columns) + 1) + "|"]
    for system, metrics in rows.items():
        cells = [f"{metrics[c]:.1f}" if c in metrics else "-" for c in columns]
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    examples = ["", "## Examples", ""]
    for i in range(min(10, len(pairs))):
        examples += [f"**Original:** {sources_text[i]}  ",
                     f"**Human:** {pairs[i][1]}  ",
                     f"**Baseline:** {' '.join(baseline[i])}  ",
                     f"**LSTM:** {' '.join(lstm_run1[i])}  "]
        if args.llm:
            examples.append(f"**LLM:** {' '.join(llm_runs[0][i])}  ")
        examples.append("")

    report = (f"# Results on {len(pairs)} test sentences (Shakespeare -> modern)\n\n{table}\n\n"
              f"LSTM parameters: {n_params / 1e6:.2f} M\n" + "\n".join(examples))
    path = os.path.join(config.RESULTS_DIR, "results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(table)
    print("\nFull report written to", path)


if __name__ == "__main__":
    main()
