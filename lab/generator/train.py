import argparse
import csv
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from lab import config
from lab.generator.model import CharLM, CharVocab, load_shakespeare_text


def random_batch(data, batch_size, seq_len):
    starts = [random.randint(0, len(data) - seq_len - 2) for _ in range(batch_size)]
    x = torch.stack([data[s:s + seq_len] for s in starts])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y


@torch.no_grad()
def validation_loss(model, data, seq_len, max_chars=50_000):
    model.eval()
    data = data[:max_chars]
    n_chunks = (len(data) - 1) // seq_len
    x = data[:n_chunks * seq_len].view(n_chunks, seq_len)
    y = data[1:n_chunks * seq_len + 1].view(n_chunks, seq_len)
    logits, _, _ = model(x)
    model.train()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()


def train(cell="lstm", steps=config.GEN_STEPS, data_dir=config.CORPUS_DIR, hidden=config.GEN_HID_DIM, name=None):
    name = name or cell
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    train_text = load_shakespeare_text("train", data_dir)
    valid_text = load_shakespeare_text("valid", data_dir)
    vocab = CharVocab.build(train_text)
    train_data = torch.tensor(vocab.encode(train_text))
    valid_data = torch.tensor(vocab.encode(valid_text))

    model_args = dict(vocab_size=len(vocab), emb_dim=config.GEN_EMB_DIM,
                      hid_dim=hidden, cell_type=cell, dropout=config.GEN_DROPOUT)
    model = CharLM(**model_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.GEN_LEARNING_RATE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] {len(train_text):,} training characters, vocab {len(vocab)}, {n_params / 1e6:.2f} M parameters")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    log, recent_losses, best, start = [], [], float("inf"), time.time()
    for step in range(1, steps + 1):
        x, y = random_batch(train_data, config.GEN_BATCH_SIZE, config.GEN_SEQ_LEN)
        logits, _, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
        optimizer.step()
        recent_losses.append(loss.item())

        if step % config.GEN_EVAL_EVERY == 0 or step == steps:
            train_loss = sum(recent_losses) / len(recent_losses)
            recent_losses = []
            valid = validation_loss(model, valid_data, config.GEN_SEQ_LEN)
            entry = {"step": step, "train_loss": train_loss, "valid_loss": valid,
                     "valid_bpc": valid / math.log(2), "minutes": (time.time() - start) / 60}
            log.append(entry)
            saved = ""
            if valid < best:
                best = valid
                torch.save({"model_state": model.state_dict(), "model_args": model_args, "chars": vocab.chars},
                           config.generator_path(name))
                saved = "  (saved)"
            print(f"[{name}] step {step:5d} | train loss {train_loss:.3f} | "
                  f"valid bpc {entry['valid_bpc']:.3f} | {entry['minutes']:.1f} min{saved}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, f"generator_{name}_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        writer.writeheader()
        writer.writerows(log)
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=["lstm", "gru", "rnn"], default="lstm")
    parser.add_argument("--steps", type=int, default=config.GEN_STEPS)
    parser.add_argument("--hidden", type=int, default=config.GEN_HID_DIM, help="hidden state size")
    parser.add_argument("--name", default=None, help="checkpoint name (default: the cell type)")
    parser.add_argument("--sample", action="store_true", help="use the tiny sample dataset")
    args = parser.parse_args()
    data_dir = config.SAMPLE_DIR if args.sample else config.CORPUS_DIR
    train(args.cell, args.steps, data_dir, hidden=args.hidden, name=args.name)


if __name__ == "__main__":
    main()
