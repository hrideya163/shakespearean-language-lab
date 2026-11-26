"""Generate Shakespearean text with a trained character-level model.

  python generate.py --prompt "ROMEO:" --temperature 0.8 --length 300
  python generate.py --cell gru --prompt "My lord,"
"""
import argparse

import torch


@torch.no_grad()
def generate(model, vocab, prompt="\n", length=300, temperature=0.8, record=False, seed=None):
    """Feed the prompt, then sample one character at a time.

    temperature < 1: sharper distribution  -> safer, more repetitive text
    temperature > 1: flatter distribution  -> more creative, more mistakes

    Returns the generated text (without the prompt) and, if record=True, a list with,
    for every generated character: the probabilities it was sampled from ("probs")
    and the gate values / hidden state after reading it."""
    if seed is not None:
        torch.manual_seed(seed)
    prompt_ids = vocab.encode(prompt) or vocab.encode("\n")
    # 1. read the prompt to build up the hidden state
    logits, state, _ = model(torch.tensor([prompt_ids]))
    next_logits = logits[0, -1]

    generated, recorded = [], []
    for _ in range(length):
        # 2. turn scores into probabilities (with temperature) and sample a character
        probs = torch.softmax(next_logits / max(temperature, 1e-3), dim=0)
        char_id = torch.multinomial(probs, 1).item()
        generated.append(char_id)
        # 3. feed the sampled character back in
        logits, state, gates = model(torch.tensor([[char_id]]), state, record=record)
        next_logits = logits[0, -1]
        if record:
            recorded.append({"probs": probs, **{name: value[0] for name, value in gates[0].items()}})
    return vocab.decode(generated), recorded


def main():
    from char_lm import load_generator
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=["lstm", "gru", "rnn"], default="lstm")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    model, vocab = load_generator(args.cell)
    text, _ = generate(model, vocab, args.prompt, args.length, args.temperature)
    print(args.prompt + text)


if __name__ == "__main__":
    main()
