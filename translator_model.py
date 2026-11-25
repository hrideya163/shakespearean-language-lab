"""LSTM encoder-decoder with attention and a copy gate.

Big picture:
  ENCODER  reads the whole input sentence with a bidirectional LSTM.
  DECODER  writes the output one word at a time. At each step it
           1. updates its LSTM state,
           2. uses ATTENTION to decide which input words matter right now,
           3. uses a COPY GATE (a number between 0 and 1) to decide whether to
              copy a word straight from the input or generate a new word.

Why a copy gate? Most words in a Shakespeare line stay the same in modern
English ("I have betrayed him" -> only "thou"/"hast" change). Copying makes it
easy for the model to keep names and ordinary words unchanged, and the gate
value tells the student whether each word was kept or rewritten.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        # The decoder is one-directional, so we squash forward+backward states into one.
        self.bridge_h = nn.Linear(2 * hid_dim, hid_dim)
        self.bridge_c = nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, src, src_lens):
        emb = self.embedding(src)                                   # (B, S, E)
        # "Packing" tells the LSTM to ignore the padding at the end of short sentences.
        packed = pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=src.size(1))
        # h, c: (2, B, H) -> forward and backward final states
        h = torch.tanh(self.bridge_h(torch.cat([h[0], h[1]], dim=1)))  # (B, H)
        c = torch.tanh(self.bridge_c(torch.cat([c[0], c[1]], dim=1)))  # (B, H)
        return outputs, h, c                                        # outputs: (B, S, 2H)


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx, use_copy=True):
        super().__init__()
        self.use_copy = use_copy       # False = ablation without the copy gate (experiment RQ4)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm_cell = nn.LSTMCell(emb_dim, hid_dim)
        self.attn_proj = nn.Linear(2 * hid_dim, hid_dim, bias=False)   # makes encoder states comparable to h
        self.combine = nn.Linear(hid_dim + 2 * hid_dim, hid_dim)
        self.out = nn.Linear(hid_dim, vocab_size)
        self.copy_gate = nn.Linear(hid_dim + 2 * hid_dim + emb_dim, 1)

    def step(self, prev_token, h, c, enc_outputs, enc_keys, src, src_mask):
        """Produce ONE output word distribution.

        Returns:
          probs   (B, V) probability of every vocabulary word
          h, c    new LSTM state
          attn    (B, S) attention weights over the input words
          p_copy  (B,)   probability of copying instead of generating
        """
        emb = self.embedding(prev_token)                            # (B, E)
        h, c = self.lstm_cell(emb, (h, c))                          # (B, H)

        # --- attention: score every input word against the decoder state ---
        scores = torch.bmm(enc_keys, h.unsqueeze(2)).squeeze(2)     # (B, S)
        scores = scores.masked_fill(~src_mask, -1e9)                # ignore padding
        attn = F.softmax(scores, dim=1)                             # (B, S), sums to 1
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)  # (B, 2H) weighted input summary

        # --- option 1: GENERATE a word from the whole vocabulary ---
        hidden = torch.tanh(self.combine(torch.cat([h, context], dim=1)))
        generate_probs = F.softmax(self.out(hidden), dim=1)         # (B, V)

        # --- option 2: COPY an input word (use attention weights as probabilities) ---
        copy_probs = torch.zeros_like(generate_probs).scatter_add(1, src, attn)  # (B, V)

        # --- copy gate mixes the two options ---
        p_copy = torch.sigmoid(self.copy_gate(torch.cat([h, context, emb], dim=1)))  # (B, 1)
        if not self.use_copy:
            p_copy = torch.zeros_like(p_copy)
        probs = (1 - p_copy) * generate_probs + p_copy * copy_probs
        return probs, h, c, attn, p_copy.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx, use_copy=True):
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = Encoder(vocab_size, emb_dim, hid_dim, pad_idx)
        self.decoder = Decoder(vocab_size, emb_dim, hid_dim, pad_idx, use_copy)

    def encode(self, src, src_lens):
        enc_outputs, h, c = self.encoder(src, src_lens)
        enc_keys = self.decoder.attn_proj(enc_outputs)
        src_mask = src != self.pad_idx
        return enc_outputs, enc_keys, src_mask, h, c

    def forward(self, src, src_lens, tgt):
        """Training with TEACHER FORCING: at each step the decoder is fed the
        correct previous word (from tgt), not its own guess.
        Returns probabilities of shape (B, T-1, V) for predicting tgt[:, 1:]."""
        enc_outputs, enc_keys, src_mask, h, c = self.encode(src, src_lens)
        all_probs = []
        for t in range(tgt.size(1) - 1):
            probs, h, c, _, _ = self.decoder.step(tgt[:, t], h, c, enc_outputs, enc_keys, src, src_mask)
            all_probs.append(probs)
        return torch.stack(all_probs, dim=1)
