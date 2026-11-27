import argparse
import re

import torch

from lab.generator.model import load_generator, load_shakespeare_text

WORD_CHARS = re.compile(r"[A-Za-z']")


class SpellingFilter:
    def __init__(self, text, vocab):
        words = set(re.findall(r"[A-Za-z']+", text))
        words |= {w.lower() for w in words} | {w.capitalize() for w in words}
        self.words = words
        self.prefixes = {w[:i] for w in words for i in range(1, len(w) + 1)}
        self.vocab = vocab
        self.is_word_char = torch.tensor([bool(WORD_CHARS.match(c)) for c in vocab.chars])
        self.cache = {}

    def allowed(self, current_word):
        if current_word not in self.cache:
            letters = torch.tensor([(current_word + c) in self.prefixes for c in self.vocab.chars])
            can_end = current_word == "" or current_word in self.words
            self.cache[current_word] = torch.where(self.is_word_char, letters, torch.tensor(can_end))
        return self.cache[current_word]


def current_word(text):
    match = re.search(r"[A-Za-z']+$", text)
    return match.group(0) if match else ""


def top_p_filter(probs, p):
    sorted_probs, order = probs.sort(descending=True)
    keep = sorted_probs.cumsum(0) - sorted_probs < p
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[order[keep]] = True
    return mask


@torch.no_grad()
def generate(model, vocab, prompt="\n", length=300, temperature=0.8, record=False, seed=None,
             spelling=None, top_p=None):
    if seed is not None:
        torch.manual_seed(seed)
    prompt_ids = vocab.encode(prompt) or vocab.encode("\n")
    logits, state, _ = model(torch.tensor([prompt_ids]))
    next_logits = logits[0, -1]
    text = vocab.decode(prompt_ids)

    generated, recorded = [], []
    for _ in range(length):
        probs = torch.softmax(next_logits / max(temperature, 1e-3), dim=0)
        allowed = torch.ones_like(probs, dtype=torch.bool)
        if spelling is not None:
            allowed &= spelling.allowed(current_word(text))
        if top_p is not None:
            allowed &= top_p_filter(probs * allowed, top_p)
        if not allowed.any():
            allowed[:] = True
        filtered = probs * allowed
        char_id = torch.multinomial(filtered / filtered.sum(), 1).item()
        generated.append(char_id)
        text += vocab.chars[char_id]
        logits, state, gates = model(torch.tensor([[char_id]]), state, record=record)
        next_logits = logits[0, -1]
        if record:
            recorded.append({"probs": probs, "blocked": ~allowed,
                             **{name: value[0] for name, value in gates[0].items()}})
    return vocab.decode(generated), recorded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm_large", help="checkpoint name, e.g. lstm_large, lstm_long, gru")
    parser.add_argument("--prompt", default="My lord, ")
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=None, help="e.g. 0.9 (off by default)")
    parser.add_argument("--no_filter", action="store_true", help="plain sampling, may invent words")
    args = parser.parse_args()
    model, vocab = load_generator(args.model)
    spelling = None if args.no_filter else SpellingFilter(load_shakespeare_text("train"), vocab)
    text, _ = generate(model, vocab, args.prompt, args.length, args.temperature,
                       spelling=spelling, top_p=args.top_p)
    print(args.prompt + text)


if __name__ == "__main__":
    main()
