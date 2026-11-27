import argparse
import os
import random

from sklearn.metrics import roc_auc_score

from lab import config
from lab.data import detokenize, read_split, tokenize
from lab.explain import find_anchors, meaning_guard
from lab.stats import bootstrap_ci
from lab.translator.decode import Translator

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
            return tokens[:i] + tokens[i + 1:]
        if low in CONTRACTIONS:
            return tokens[:i] + [CONTRACTIONS[low]] + tokens[i + 1:]
    for i, t in enumerate(tokens):
        if t.lower() in AUXILIARIES:
            return tokens[:i + 1] + ["not"] + tokens[i + 1:]
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
    for pair_id, (shakes, modern) in enumerate(pairs):
        ref = tokenize(modern, lower=False)
        rows.append((pair_id, shakes, detokenize(ref), "clean"))
        for kind, corrupted in [("negation", flip_negation(ref)),
                                ("name", swap_name(ref, name_pool, rng) if len(name_pool) > 1 else None),
                                ("number", change_number(ref, rng))]:
            if corrupted:
                rows.append((pair_id, shakes, detokenize(corrupted), kind))
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
    for pair_id, shakes, candidate, kind in benchmark:
        _, _, warnings = meaning_guard(tokenize(shakes, lower=False), tokenize(candidate))
        avg_lp, min_lp = translator.score(shakes, candidate)
        scored.append({"pair": pair_id, "kind": kind, "guard": bool(warnings), "avg": avg_lp, "min": min_lp,
                       "shakes": shakes, "candidate": candidate})

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "drift_benchmark.tsv"), "w", encoding="utf-8") as f:
        f.write("type\tshakespeare\tcandidate\n")
        for s in scored:
            f.write(f"{s['kind']}\t{s['shakes']}\t{s['candidate']}\n")

    clean = [s for s in scored if s["kind"] == "clean"]
    clean_by_pair = {s["pair"]: s for s in clean}
    lines = ["# RQ5: Meaning drift detection", "",
             f"Benchmark built from {len(pairs)} test pairs: {len(scored)} candidates "
             f"({len(clean)} clean, {len(scored) - len(clean)} drifted).", "",
             f"Meaning guard false alarms on clean translations: "
             f"**{100 * sum(s['guard'] for s in clean) / len(clean):.1f}%**", "",
             "| drift type | examples | guard detection % | AUROC (avg log-prob) | 95% CI | "
             "AUROC (min log-prob) | model prefers correct version % |",
             "|---|---|---|---|---|---|---|"]
    fooled = []
    for kind in ["negation", "name", "number"]:
        drifted = [s for s in scored if s["kind"] == kind]
        if not drifted:
            continue
        both = clean + drifted
        labels = [0] * len(clean) + [1] * len(drifted)
        auc_avg = roc_auc_score(labels, [-s["avg"] for s in both])
        auc_min = roc_auc_score(labels, [-s["min"] for s in both])

        def auc_of(idx):
            sample_labels = [labels[i] for i in idx]
            if len(set(sample_labels)) < 2:
                return 0.5
            return roc_auc_score(sample_labels, [-both[i]["avg"] for i in idx])
        low, high = bootstrap_ci(auc_of, len(both), n_resamples=300)

        prefers = [clean_by_pair[s["pair"]]["avg"] > s["avg"] for s in drifted]
        detected = 100 * sum(s["guard"] for s in drifted) / len(drifted)
        lines.append(f"| {kind} | {len(drifted)} | {detected:.1f} | {auc_avg:.3f} | "
                     f"{low:.2f} - {high:.2f} | {auc_min:.3f} | {100 * sum(prefers) / len(prefers):.1f} |")
        fooled += [(s, clean_by_pair[s["pair"]]) for s, ok in zip(drifted, prefers) if not ok][:3]

    lines += ["", "## Where the likelihood was fooled (it preferred the corrupted version)", ""]
    for bad, good in fooled:
        lines += [f"- **{bad['kind']}**: *{bad['shakes']}*  ",
                  f"  correct: {good['candidate']} (avg log-prob {good['avg']:.2f})  ",
                  f"  corrupted: {bad['candidate']} (avg log-prob {bad['avg']:.2f})"]

    with open(os.path.join(config.RESULTS_DIR, "rq5_drift.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nWritten: results/rq5_drift.md, results/drift_benchmark.tsv")


if __name__ == "__main__":
    main()
