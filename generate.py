"""Generate Shakespearean text with a trained character-level model.

  python generate.py --prompt "My lord, " --temperature 0.7
  python generate.py --cell gru --prompt "My lord,"
  python generate.py --no_filter       # plain sampling, may invent words

How the next character is chosen (SAMPLING), step by step:
  1. TEMPERATURE  divide the model's scores by T before the softmax.
                  T < 1: safer and more repetitive.  T > 1: more creative, more mistakes.
  2. SPELLING FILTER (optional)  a character-level model can invent words ("knowshorting").
                  We only allow characters that keep the current word on the way to a REAL word
                  from the training text, and only allow the word to end when it is a real word.
                  The model still makes every choice; the filter only removes misspellings.
  3. TOP-P (optional)  keep only the most likely characters that together have probability p
                  (e.g. 0.9) and drop the long tail of unlikely ones ("nucleus sampling").
Experiment RQ3 (exp_temperature.py) measures what each option changes. The spelling filter
removes invented words without making the text less varied; top-p made it more repetitive,
so the app uses the spelling filter only.
"""
import argparse
import re

import torch

WORD_CHARS = re.compile(r"[A-Za-z']")


class SpellingFilter:
    """Knows every word of the training text and every beginning of a word ("pr", "pri", ...)."""

    def __init__(self, text, vocab):
        words = set(re.findall(r"[A-Za-z']+", text))
        words |= {w.lower() for w in words} | {w.capitalize() for w in words}
        self.words = words
        self.prefixes = {w[:i] for w in words for i in range(1, len(w) + 1)}
        self.vocab = vocab
        self.is_word_char = torch.tensor([bool(WORD_CHARS.match(c)) for c in vocab.chars])
        self.cache = {}

    def allowed(self, current_word):
        """True/False for every character: may it come next after `current_word`?"""
        if current_word not in self.cache:
            # a letter is allowed if the longer word is still the start of a real word
            letters = torch.tensor([(current_word + c) in self.prefixes for c in self.vocab.chars])
            # a space or punctuation (ending the word) is allowed if the word is complete
            can_end = current_word == "" or current_word in self.words
            self.cache[current_word] = torch.where(self.is_word_char, letters, torch.tensor(can_end))
        return self.cache[current_word]


def current_word(text):
    """The unfinished word at the end of the text: 'My lo' -> 'lo'."""
    match = re.search(r"[A-Za-z']+$", text)
    return match.group(0) if match else ""


def top_p_filter(probs, p):
    """Keep the smallest set of characters whose probabilities add up to at least p."""
    sorted_probs, order = probs.sort(descending=True)
    keep = sorted_probs.cumsum(0) - sorted_probs < p        # always keeps the most likely one
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[order[keep]] = True
    return mask


@torch.no_grad()
def generate(model, vocab, prompt="\n", length=300, temperature=0.8, record=False, seed=None,
             spelling=None, top_p=None):
    """Feed the prompt, then sample one character at a time.

    spelling: a SpellingFilter (or None)   top_p: e.g. 0.9 (or None)
    Returns the generated text (without the prompt) and, if record=True, a list with, for
    every generated character: the model's probabilities ("probs", after temperature), which
    characters the filters blocked ("blocked") and the gate values / hidden state after reading it."""
    if seed is not None:
        torch.manual_seed(seed)
    prompt_ids = vocab.encode(prompt) or vocab.encode("\n")
    # 1. read the prompt to build up the hidden state
    logits, state, _ = model(torch.tensor([prompt_ids]))
    next_logits = logits[0, -1]
    text = vocab.decode(prompt_ids)

    generated, recorded = [], []
    for _ in range(length):
        # 2. scores -> probabilities (with temperature), then remove what the filters forbid
        probs = torch.softmax(next_logits / max(temperature, 1e-3), dim=0)
        allowed = torch.ones_like(probs, dtype=torch.bool)
        if spelling is not None:
            allowed &= spelling.allowed(current_word(text))
        if top_p is not None:
            allowed &= top_p_filter(probs * allowed, top_p)
        if not allowed.any():                      # nothing left (e.g. a strange prompt): no filter
            allowed[:] = True
        filtered = probs * allowed
        char_id = torch.multinomial(filtered / filtered.sum(), 1).item()
        generated.append(char_id)
        text += vocab.chars[char_id]
        # 3. feed the sampled character back in
        logits, state, gates = model(torch.tensor([[char_id]]), state, record=record)
        next_logits = logits[0, -1]
        if record:
            recorded.append({"probs": probs, "blocked": ~allowed,
                             **{name: value[0] for name, value in gates[0].items()}})
    return vocab.decode(generated), recorded


def main():
    from char_lm import load_generator, load_shakespeare_text
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="lstm_long", help="checkpoint name, e.g. lstm_long, gru, rnn")
    parser.add_argument("--prompt", default="My lord, ")
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=None, help="e.g. 0.9 (off by default)")
    parser.add_argument("--no_filter", action="store_true", help="plain sampling, may invent words")
    args = parser.parse_args()
    model, vocab = load_generator(args.cell)
    spelling = None if args.no_filter else SpellingFilter(load_shakespeare_text("train"), vocab)
    text, _ = generate(model, vocab, args.prompt, args.length, args.temperature,
                       spelling=spelling, top_p=args.top_p)
    print(args.prompt + text)


if __name__ == "__main__":
    main()
