import html
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from lab import config
from lab.generator.model import load_generator, load_shakespeare_text

N_CHARS = 20_000


def text_features(text):
    feats = {"position_in_line": [], "inside_word": [], "capital_letter": [],
             "after_punctuation": [], "end_of_line": []}
    pos = 0
    for i, ch in enumerate(text):
        pos = 0 if ch == "\n" else pos + 1
        feats["position_in_line"].append(float(pos))
        feats["inside_word"].append(1.0 if ch.isalpha() else 0.0)
        feats["capital_letter"].append(1.0 if ch.isupper() else 0.0)
        feats["after_punctuation"].append(1.0 if i > 0 and text[i - 1] in ".,!?;:" else 0.0)
        feats["end_of_line"].append(1.0 if i + 1 < len(text) and text[i + 1] == "\n" else 0.0)
    return {name: torch.tensor(values) for name, values in feats.items()}


def correlation(units, feature):
    u = units - units.mean(dim=0)
    f = feature - feature.mean()
    return (u * f[:, None]).sum(0) / (u.norm(dim=0) * f.norm() + 1e-8)


def char_type(ch):
    if ch == "\n":
        return "newline"
    if ch == " ":
        return "space"
    if ch.isalpha():
        return "letter"
    if ch in ".!?":
        return "sentence end (. ! ?)"
    return "other punctuation"


def coloured_html(text, values):
    spans = []
    for ch, v in zip(text, values):
        v = max(-1.0, min(1.0, v))
        colour = f"rgba(220,40,40,{v:.2f})" if v > 0 else f"rgba(40,90,220,{-v:.2f})"
        shown = "↵<br>" if ch == "\n" else html.escape(ch)
        spans.append(f"<span style='background:{colour}'>{shown}</span>")
    return "".join(spans)


def target_mask(name, inputs, targets):
    if name in ("end_of_line", "position_in_line"):
        return [t == "\n" for t in targets]
    if name == "inside_word":
        return [t.isalpha() for t in targets]
    if name == "capital_letter":
        return [t.isupper() for t in targets]
    if name == "after_punctuation":
        return [i in ".,!?;:" for i in inputs]
    raise ValueError(name)


CHUNK = 200


@torch.no_grad()
def masked_loss(model, ids, mask, ablate=None):
    n = (ids.size(1) - 1) // CHUNK
    x = ids[0, :n * CHUNK].view(n, CHUNK)
    y = ids[0, 1:n * CHUNK + 1].view(n, CHUNK)
    logits, _, _ = model(x, ablate=ablate)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return loss[torch.tensor(mask[:n * CHUNK])].mean().item()


def ablation_study(model, ids, text, top, n_random=10):
    rng = random.Random(config.SEED)
    hidden = model.cell.hidden_dim
    rows = []
    for name, t in top.items():
        mask = target_mask(name, text[:-1], text[1:])
        base = masked_loss(model, ids, mask)
        top_increase = masked_loss(model, ids, mask, ablate=[t["unit"]]) - base
        random_units = rng.sample([u for u in range(hidden) if u != t["unit"]], n_random)
        rand_increase = sum(masked_loss(model, ids, mask, ablate=[u]) - base
                            for u in random_units) / n_random
        rows.append((name, t["unit"], sum(mask), base, top_increase, rand_increase))
    return rows


def gate_saturation(recorded, gate_names=("input", "forget", "output")):
    result = {}
    for g in gate_names:
        values = torch.stack([r[g][0] for r in recorded])
        result[g] = ((values < 0.1).float().mean(0), (values > 0.9).float().mean(0))
    return result


def char_group(ch):
    if ch in "aeiouAEIOU":
        return "vowel"
    if ch.isalpha():
        return "upper-case consonant" if ch.isupper() else "lower-case consonant"
    if ch in " \n":
        return "space / newline"
    return "punctuation / other"


def embedding_map(model, vocab, path):
    weights = model.embedding.weight.detach()
    centred = weights - weights.mean(dim=0)
    _, _, v = torch.linalg.svd(centred, full_matrices=False)
    points = centred @ v[:2].T
    colours = {"vowel": "tab:red", "lower-case consonant": "tab:blue",
               "upper-case consonant": "tab:cyan", "space / newline": "tab:green",
               "punctuation / other": "tab:gray"}
    plt.figure(figsize=(7, 6))
    for group, colour in colours.items():
        idx = [i for i, ch in enumerate(vocab.chars) if char_group(ch) == group]
        plt.scatter(points[idx, 0], points[idx, 1], c=colour, label=group, s=25)
    for i, ch in enumerate(vocab.chars):
        label = {"\n": "\\n", " ": "space"}.get(ch, ch)
        plt.annotate(label, (points[i, 0], points[i, 1]), fontsize=8, xytext=(3, 3),
                     textcoords="offset points")
    plt.title("Character embeddings learned by the LSTM (PCA to 2-D)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120)


MODEL = "lstm_long" if os.path.exists(config.generator_path("lstm_long")) else "lstm"


@torch.no_grad()
def main():
    model, vocab = load_generator(MODEL)
    print("analysing", MODEL)
    text = load_shakespeare_text("valid")[:N_CHARS]
    ids = torch.tensor([vocab.encode(text)])
    text = vocab.decode(ids[0].tolist())
    _, _, recorded = model(ids, record=True)

    cells = torch.tanh(torch.stack([r["cell"][0] for r in recorded]))
    forget = torch.stack([r["forget"][0] for r in recorded])

    feats = text_features(text)
    top = {}
    for name, feature in feats.items():
        corr = correlation(cells, feature)
        unit = int(corr.abs().argmax())
        top[name] = {"unit": unit, "correlation": float(corr[unit])}

    by_type = {}
    mean_forget = forget.mean(dim=1)
    for ch, f in zip(text, mean_forget.tolist()):
        by_type.setdefault(char_type(ch), []).append(f)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print("running ablations ...")
    ablations = ablation_study(model, ids, text, top)
    saturation = gate_saturation(recorded)
    embedding_map(model, vocab, os.path.join(config.RESULTS_DIR, "rq2_embeddings.png"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (g, (closed, opened)) in zip(axes, saturation.items()):
        ax.scatter(closed, opened, s=8, alpha=0.6)
        ax.plot([0, 1], [1, 0], "k--", linewidth=0.5)
        ax.set_xlabel("fraction of time closed (< 0.1)"); ax.set_ylabel("fraction of time open (> 0.9)")
        ax.set_title(f"{g} gate (one dot per unit)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "rq2_gate_saturation.png"), dpi=120)

    lines = ["# RQ2: Inside the LSTM generator", "", f"Model: `{MODEL}`.", "",
             f"Analysed {len(text):,} validation characters, {cells.shape[1]} cells.", "",
             "## Cell most correlated with each feature", "",
             "| feature | cell | correlation |", "|---|---|---|"]
    lines += [f"| {name} | #{t['unit']} | {t['correlation']:+.2f} |" for name, t in top.items()]
    lines += ["", "## Average forget gate by character type", "",
              "Values near 1 mean the LSTM keeps its memory, and lower values mean it clears part of it.", "",
              "| character type | mean forget gate | count |", "|---|---|---|"]
    for name, values in sorted(by_type.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        lines.append(f"| {name} | {sum(values) / len(values):.3f} | {len(values)} |")
    lines += ["", "## Gate saturation", "",
              "| gate | mean value | units closed > 50% of the time | units open > 50% of the time |",
              "|---|---|---|---|"]
    for g, (closed, opened) in saturation.items():
        mean = torch.stack([r[g][0] for r in recorded]).mean().item()
        lines.append(f"| {g} | {mean:.3f} | {int((closed > 0.5).sum())} | {int((opened > 0.5).sum())} |")
    lines += ["", "## Causal ablation: switching one cell off", "",
              "Loss is measured only on the predictions linked to the feature (in nats). "
              "If the top cell really matters, switching it off should hurt much more than "
              "switching off a random cell. 'No' means the information is probably stored "
              "redundantly in several cells, so removing one does not matter.", "",
              "| feature | cell | predictions | normal loss | increase (top cell off) | "
              "increase (random cell off, avg of 10) | causal effect? |", "|---|---|---|---|---|---|---|"]
    for name, unit, n, base, top_inc, rand_inc in ablations:
        causal = "yes" if top_inc > max(3 * abs(rand_inc), 0.001) else "no"
        lines.append(f"| {name} | #{unit} | {n} | {base:.3f} | {top_inc:+.4f} | {rand_inc:+.4f} | "
                     f"{causal} |")
    with open(os.path.join(config.RESULTS_DIR, "rq2_inside_lstm.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    page = ["<html><head><meta charset='utf-8'><title>Inside the LSTM</title></head>",
            "<body style='font-family:monospace;max-width:900px;margin:auto'>"]
    snippet = slice(0, 1500)
    for name, t in top.items():
        values = cells[snippet, t["unit"]].tolist()
        if t["correlation"] < 0:
            values = [-v for v in values]
        page.append(f"<h3>{name}: cell #{t['unit']} (r = {t['correlation']:+.2f})</h3>")
        page.append(f"<p>{coloured_html(text[snippet], values)}</p>")
    page.append("</body></html>")
    with open(os.path.join(config.RESULTS_DIR, "rq2_inside_lstm.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(page))

    with open(os.path.join(config.RESULTS_DIR, "interesting_units.json"), "w") as f:
        json.dump({"model": MODEL, "features": top}, f, indent=2)

    print("\n".join(lines))
    print("\nWritten: results/rq2_inside_lstm.md, rq2_inside_lstm.html, rq2_gate_saturation.png, "
          "rq2_embeddings.png, interesting_units.json")


if __name__ == "__main__":
    main()
