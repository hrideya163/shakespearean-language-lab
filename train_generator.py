"""Train the character-level text generator.

  python train_generator.py                 # LSTM (default)
  python train_generator.py --cell gru      # or rnn, for the comparison experiment (RQ1)
  python train_generator.py --steps 500     # shorter run
"""
import argparse
import csv
import math
import os
import random
import time

import torch
import torch.nn.functional as F

import config
from char_lm import CharLM, CharVocab, load_shakespeare_text


def random_batch(data, batch_size, seq_len):
    """Pick random slices of text. Input = characters 0..n-1, target = characters 1..n."""
    starts = [random.randint(0, len(data) - seq_len - 2) for _ in range(batch_size)]
    x = torch.stack([data[s:s + seq_len] for s in starts])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y


@torch.no_grad()
def validation_loss(model, data, seq_len, max_chars=50_000):
    """Average loss on the validation text, read in consecutive chunks."""
    model.eval()
    data = data[:max_chars]
    n_chunks = (len(data) - 1) // seq_len
    x = data[:n_chunks * seq_len].view(n_chunks, seq_len)
    y = data[1:n_chunks * seq_len + 1].view(n_chunks, seq_len)
    logits, _, _ = model(x)
    model.train()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()


def train(cell="lstm", steps=config.GEN_STEPS, data_dir=config.CORPUS_DIR, verbose=True):
    """Train one generator and save it. Returns the training log (list of dicts)."""
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    train_text = load_shakespeare_text("train", data_dir)
    valid_text = load_shakespeare_text("valid", data_dir)
    vocab = CharVocab.build(train_text)
    train_data = torch.tensor(vocab.encode(train_text))
    valid_data = torch.tensor(vocab.encode(valid_text))

    model_args = dict(vocab_size=len(vocab), emb_dim=config.GEN_EMB_DIM,
                      hid_dim=config.GEN_HID_DIM, cell_type=cell, dropout=config.GEN_DROPOUT)
    model = CharLM(**model_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.GEN_LEARNING_RATE)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[{cell}] {len(train_text):,} training characters, vocab {len(vocab)}, "
              f"{n_params / 1e6:.2f} M parameters")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    log, best, start, running = [], float("inf"), time.time(), 0.0
    for step in range(1, steps + 1):
        x, y = random_batch(train_data, config.GEN_BATCH_SIZE, config.GEN_SEQ_LEN)
        logits, _, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
        optimizer.step()
        running += loss.item()

        if step % config.GEN_EVAL_EVERY == 0 or step == steps:
            n = config.GEN_EVAL_EVERY if step % config.GEN_EVAL_EVERY == 0 else step % config.GEN_EVAL_EVERY
            train_loss, running = running / n, 0.0
            valid = validation_loss(model, valid_data, config.GEN_SEQ_LEN)
            # bits per character = loss in base 2 (lower is better; ~1.5 is decent for small models)
            entry = {"step": step, "train_loss": train_loss, "valid_loss": valid,
                     "valid_bpc": valid / math.log(2), "minutes": (time.time() - start) / 60}
            log.append(entry)
            saved = ""
            if valid < best:
                best = valid
                torch.save({"model_state": model.state_dict(), "model_args": model_args,
                            "chars": vocab.chars, "log": log}, config.generator_path(cell))
                saved = "  (saved)"
            if verbose:
                print(f"[{cell}] step {step:5d} | train loss {train_loss:.3f} | "
                      f"valid bpc {entry['valid_bpc']:.3f} | {entry['minutes']:.1f} min{saved}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, f"generator_{cell}_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        writer.writeheader()
        writer.writerows(log)
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=["lstm", "gru", "rnn"], default="lstm")
    parser.add_argument("--steps", type=int, default=config.GEN_STEPS)
    parser.add_argument("--sample", action="store_true", help="use the tiny sample dataset")
    args = parser.parse_args()
    data_dir = config.SAMPLE_DIR if args.sample else config.CORPUS_DIR
    train(args.cell, args.steps, data_dir)


if __name__ == "__main__":
    main()
