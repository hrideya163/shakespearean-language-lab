"""Train the LSTM translator.

  python train_translator.py              # full corpus (run download_data.py first)
  python train_translator.py --no_copy    # ablation: same model without the copy gate (RQ4)
  python train_translator.py --dropout 0  # turn regularisation off (to see overfitting)
  python train_translator.py --extended   # ablation: pointer-generator extended vocabulary
  python train_translator.py --sample     # tiny built-in dataset, just to check everything runs
"""
import argparse
import csv
import math
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config
from data import PAD, SPECIALS, Vocab, make_examples, read_split, tokenize
from lexicon import ARCHAIC
from translator_model import Seq2Seq

PAD_IDX = SPECIALS.index(PAD)


class TranslationDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch):
    """Pad a list of (src, src_ext, tgt) id lists into tensors of equal length."""
    src_lens = torch.tensor([len(s) for s, _, _ in batch])
    max_src = int(src_lens.max())
    max_tgt = max(len(t) for _, _, t in batch)
    src = torch.tensor([s + [PAD_IDX] * (max_src - len(s)) for s, _, _ in batch])
    src_ext = torch.tensor([e + [PAD_IDX] * (max_src - len(e)) for _, e, _ in batch])
    tgt = torch.tensor([t + [PAD_IDX] * (max_tgt - len(t)) for _, _, t in batch])
    return src, src_ext, src_lens, tgt


def loss_fn(probs, tgt):
    """Cross-entropy on the mixed (generate + copy) probabilities."""
    log_probs = torch.log(probs + 1e-9)                        # (B, T-1, V)
    return F.nll_loss(log_probs.reshape(-1, log_probs.size(-1)),
                      tgt[:, 1:].reshape(-1), ignore_index=PAD_IDX)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for batch_i, (src, src_ext, src_lens, tgt) in enumerate(loader, start=1):
            src, src_ext, tgt = src.to(device), src_ext.to(device), tgt.to(device)
            probs = model(src, src_ext, src_lens, tgt)
            loss = loss_fn(probs, tgt)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
                optimizer.step()
                if batch_i % 100 == 0:
                    print(f"    batch {batch_i}/{len(loader)} | loss {total / (batch_i - 1 or 1):.3f}",
                          flush=True)
            total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="use the tiny sample dataset")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no_copy", action="store_true", help="turn the copy gate off (ablation)")
    parser.add_argument("--dropout", type=float, default=config.DROPOUT)
    parser.add_argument("--extended", action="store_true",
                        help="temporary ids for unknown words (pointer-generator ablation)")
    args = parser.parse_args()

    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = config.SAMPLE_DIR if args.sample else config.CORPUS_DIR
    min_freq = 1 if args.sample else config.MIN_FREQ
    epochs = args.epochs or (60 if args.sample else config.EPOCHS)
    patience = epochs if args.sample else config.PATIENCE

    # 1. data
    train_pairs = read_split("train", data_dir)
    valid_pairs = read_split("valid", data_dir)
    all_tokens = [tokenize(s) for pair in train_pairs for s in pair]
    vocab = Vocab.build(all_tokens, min_freq, extra_words=ARCHAIC.keys())
    extended = args.extended and not args.no_copy   # without a copy gate there is nothing to extend
    train_ex = make_examples(train_pairs, vocab, extended)
    valid_ex = make_examples(valid_pairs, vocab, extended)
    print(f"device={device}  vocab={len(vocab)}  train={len(train_ex)}  valid={len(valid_ex)} examples")

    train_loader = DataLoader(TranslationDataset(train_ex), batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(TranslationDataset(valid_ex), batch_size=config.BATCH_SIZE,
                              collate_fn=collate)

    # 2. model
    model_args = dict(vocab_size=len(vocab), emb_dim=config.EMB_DIM,
                      hid_dim=config.HID_DIM, pad_idx=PAD_IDX,
                      use_copy=not args.no_copy, dropout=args.dropout)
    model = Seq2Seq(**model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params / 1e6:.2f} M")

    # 3. training loop with early stopping
    if args.no_copy:
        checkpoint_path, run_name = config.TRANSLATOR_NOCOPY_PATH, "translator_nocopy"
    elif args.extended:
        checkpoint_path, run_name = config.TRANSLATOR_EXT_PATH, "translator_ext"
    elif args.dropout == 0:
        checkpoint_path, run_name = config.TRANSLATOR_NODROPOUT_PATH, "translator_nodropout"
    else:
        checkpoint_path, run_name = config.TRANSLATOR_PATH, "translator"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    history = []
    best_valid, bad_epochs = float("inf"), 0
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        valid_loss = run_epoch(model, valid_loader, device)
        history.append((epoch, train_loss, valid_loss))
        print(f"epoch {epoch:2d} | train loss {train_loss:.3f} | valid loss {valid_loss:.3f} "
              f"| valid perplexity {math.exp(min(valid_loss, 20)):.1f}")

        if valid_loss < best_valid:
            best_valid, bad_epochs = valid_loss, 0
            torch.save({"model_state": model.state_dict(), "model_args": model_args,
                        "vocab": vocab.itos, "extended": extended}, checkpoint_path)
            print("  saved best model")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("  no improvement for", patience, "epochs -> early stopping")
                break

    log_path = os.path.join(config.RESULTS_DIR, f"{run_name}_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss"])
        writer.writerows(history)
    print("Training log written to", log_path)


if __name__ == "__main__":
    main()
