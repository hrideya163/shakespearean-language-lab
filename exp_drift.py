"""RQ5: Can meaning drift be detected automatically?

    python exp_drift.py

1. BUILD A DRIFT BENCHMARK from the test set. For each (Shakespeare, modern) pair:
     clean     the correct modern translation                     -> label 0
     negation  "not" added or removed  (meaning reversed)         -> label 1
     name      a character name swapped for another name          -> label 1
     number    a number changed                                   -> label 1
2. TEST TWO DETECTORS on every (Shakespeare line, candidate translation):
     meaning guard   rule-based check from explain.py (flag = any warning)
     LSTM likelihood how probable our translator finds the candidate
                     (a drifted candidate should look less likely)
3. REPORT detection rate / false alarms for the guard and AUROC for the
   likelihood (0.5 = random guessing, 1.0 = perfect).

Outputs: results/rq5_drift.md, results/drift_benchmark.tsv
"""
import argparse
import os
import random

from sklearn.metrics import roc_auc_score

import config
from data import detokenize, read_split, tokenize
from explain import find_anchors, meaning_guard
from translate import Translator

AUXILIARIES = {"is", "are", "am", "was", "were", "will", "would", "can", "could", "shall",
               "should", "do", "does", "did", "have", "has", "had", "must", "may", "might"}
CONTRACTIONS = {"don't": "do", "doesn't": "does", "didn't": "did", "can't": "can",
                "won't": "will", "isn't": "is", "aren't": "are", "wasn't": "was",
                "weren't": "were", "haven't": "have", "hasn't": "has", "wouldn't": "would",
                "couldn't": "could", "shouldn't": "should"}
NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def flip_negation(tokens):
    for i, t in enumerate(tokens):
        low = t.lower()
        if low in ("not", "never"):
            return tokens[:i] + tokens[i + 1:]                      # remove the negation
        if low in CONTRACTIONS:
            return tokens[:i] + [CONTRACTIONS[low]] + tokens[i + 1:]
    for i, t in enumerate(tokens):
        if t.lower() in AUXILIARIES:
            return tokens[:i + 1] + ["not"] + tokens[i + 1:]        # add a negation
    return None


def swap_name(tokens, name_pool, rng):
    names = find_anchors(tokens)["names"]
    if not names:
        return None
    old = names[0]
    new = rng.choice([n for n in name_pool if n != old])
    return [new if t == old else t for t in tokens]


def change_number(tokens, rng):
    for i, t in enumerate(tokens):
        low = t.lower()
        if low in NUMBERS:
            return tokens[:i] + [rng.choice([n for n in NUMBERS if n != low])] + tokens[i + 1:]
        if low.isdigit():
            return tokens[:i] + [str(int(low) + rng.randint(1, 9))] + tokens[i + 1:]
    return None


def build_benchmark(pairs, rng):
    name_pool = sorted({n for _, m in pairs for n in find_anchors(tokenize(m, lower=False))["names"]})
    rows = []
    for shakes, modern in pairs:
        ref = tokenize(modern, lower=False)
        rows.append((shakes, detokenize(ref), "clean"))
        for kind, corrupted in [("negation", flip_negation(ref)),
                                ("name", swap_name(ref, name_pool, rng) if len(name_pool) > 1 else None),
                                ("number", change_number(ref, rng))]:
            if corrupted:
                rows.append((shakes, detokenize(corrupted), kind))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600, help="number of test pairs to corrupt")
    args = parser.parse_args()
    rng = random.Random(config.SEED)

    pairs = read_split("test")[:args.n]
    benchmark = build_benchmark(pairs, rng)
    translator = Translator()

    scored = []
    for shakes, candidate, kind in benchmark:
        _, _, warnings = meaning_guard(tokenize(shakes, lower=False), tokenize(candidate))
        avg_lp, min_lp = translator.score(shakes, candidate)
        scored.append({"kind": kind, "guard": bool(warnings), "avg": avg_lp, "min": min_lp,
                       "shakes": shakes, "candidate": candidate})

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "drift_benchmark.tsv"), "w", encoding="utf-8") as f:
        f.write("type\tshakespeare\tcandidate\n")
        for s in scored:
            f.write(f"{s['kind']}\t{s['shakes']}\t{s['candidate']}\n")

    clean = [s for s in scored if s["kind"] == "clean"]
    lines = ["# RQ5: Meaning drift detection", "",
             f"Benchmark built from {len(pairs)} test pairs: {len(scored)} candidates.", "",
             f"Meaning guard false alarms on clean translations: "
             f"{100 * sum(s['guard'] for s in clean) / len(clean):.1f}%", "",
             "| drift type | examples | guard detection % | AUROC avg log-prob | AUROC min log-prob |",
             "|---|---|---|---|---|"]
    for kind in ["negation", "name", "number"]:
        drifted = [s for s in scored if s["kind"] == kind]
        if not drifted:
            continue
        labels = [0] * len(clean) + [1] * len(drifted)
        # lower likelihood = more suspicious, so the drift score is the NEGATIVE log-prob
        auc_avg = roc_auc_score(labels, [-s["avg"] for s in clean + drifted])
        auc_min = roc_auc_score(labels, [-s["min"] for s in clean + drifted])
        detected = 100 * sum(s["guard"] for s in drifted) / len(drifted)
        lines.append(f"| {kind} | {len(drifted)} | {detected:.1f} | {auc_avg:.3f} | {auc_min:.3f} |")

    with open(os.path.join(config.RESULTS_DIR, "rq5_drift.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nWritten: results/rq5_drift.md, results/drift_benchmark.tsv")


if __name__ == "__main__":
    main()
