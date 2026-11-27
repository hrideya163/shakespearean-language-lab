import argparse
import csv
import math
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lab import config
from lab.data import PAD, SPECIALS, Vocab, make_examples, read_split, tokenize
from lab.lexicon import ARCHAIC
from lab.translator.model import Seq2Seq

PAD_IDX = SPECIALS.index(PAD)


def pad(sequences, length):
    return torch.tensor([s + [PAD_IDX] * (length - len(s)) for s in sequences])


def collate(batch):
    sources, sources_ext, targets = zip(*batch)
    src_lens = torch.tensor([len(s) for s in sources])
    max_src, max_tgt = int(src_lens.max()), max(len(t) for t in targets)
    return pad(sources, max_src), pad(sources_ext, max_src), src_lens, pad(targets, max_tgt)


def loss_fn(probs, tgt):
    log_probs = torch.log(probs + 1e-9)
    return F.nll_loss(log_probs.reshape(-1, log_probs.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=PAD_IDX)


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for batch_i, (src, src_ext, src_lens, tgt) in enumerate(loader, start=1):
            loss = loss_fn(model(src, src_ext, src_lens, tgt), tgt)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
                optimizer.step()
                if batch_i % 100 == 0:
                    print(f"    batch {batch_i}/{len(loader)} | loss {total / (batch_i - 1):.3f}", flush=True)
            total += loss.item()
    return total / len(loader)


def build_data(data_dir, min_freq, extended):
    train_pairs = read_split("train", data_dir)
    valid_pairs = read_split("valid", data_dir)
    vocab = Vocab.build([tokenize(s) for pair in train_pairs for s in pair], min_freq, extra_words=ARCHAIC.keys())
    return vocab, make_examples(train_pairs, vocab, extended), make_examples(valid_pairs, vocab, extended)


def checkpoint_for(args):
    if args.no_copy:
        return config.TRANSLATOR_NOCOPY_PATH, "translator_nocopy"
    if args.extended:
        return config.TRANSLATOR_EXT_PATH, "translator_ext"
    if args.dropout == 0:
        return config.TRANSLATOR_NODROPOUT_PATH, "translator_nodropout"
    return config.TRANSLATOR_PATH, "translator"


def train(model, train_loader, valid_loader, epochs, patience, save):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    history, best_valid, bad_epochs = [], float("inf"), 0
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        valid_loss = run_epoch(model, valid_loader)
        history.append((epoch, train_loss, valid_loss))
        print(f"epoch {epoch:2d} | train loss {train_loss:.3f} | valid loss {valid_loss:.3f} "
              f"| valid perplexity {math.exp(min(valid_loss, 20)):.1f}")
        if valid_loss < best_valid:
            best_valid, bad_epochs = valid_loss, 0
            save()
            print("  saved best model")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  no improvement for {patience} epochs -> early stopping")
                break
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="use the tiny sample dataset")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no_copy", action="store_true", help="turn the copy gate off (ablation)")
    parser.add_argument("--dropout", type=float, default=config.DROPOUT)
    parser.add_argument("--extended", action="store_true", help="temporary ids for unknown words (ablation)")
    args = parser.parse_args()

    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    data_dir = config.SAMPLE_DIR if args.sample else config.CORPUS_DIR
    epochs = args.epochs or (60 if args.sample else config.EPOCHS)
    patience = epochs if args.sample else config.PATIENCE
    extended = args.extended and not args.no_copy

    vocab, train_ex, valid_ex = build_data(data_dir, 1 if args.sample else config.MIN_FREQ, extended)
    print(f"vocab={len(vocab)}  train={len(train_ex)}  valid={len(valid_ex)} examples")
    train_loader = DataLoader(train_ex, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ex, batch_size=config.BATCH_SIZE, collate_fn=collate)

    model_args = dict(vocab_size=len(vocab), emb_dim=config.EMB_DIM, hid_dim=config.HID_DIM, pad_idx=PAD_IDX,
                      use_copy=not args.no_copy, dropout=args.dropout)
    model = Seq2Seq(**model_args)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    checkpoint_path, run_name = checkpoint_for(args)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    def save():
        torch.save({"model_state": model.state_dict(), "model_args": model_args,
                    "vocab": vocab.itos, "extended": extended}, checkpoint_path)

    history = train(model, train_loader, valid_loader, epochs, patience, save)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(config.RESULTS_DIR, f"{run_name}_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss"])
        writer.writerows(history)
    print("Training log written to", log_path)


if __name__ == "__main__":
    main()
