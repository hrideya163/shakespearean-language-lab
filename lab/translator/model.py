import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lab.data import SPECIALS, UNK

UNK_IDX = SPECIALS.index(UNK)


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.bridge_h = nn.Linear(2 * hid_dim, hid_dim)
        self.bridge_c = nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, src, src_lens):
        emb = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=src.size(1))
        h = torch.tanh(self.bridge_h(torch.cat([h[0], h[1]], dim=1)))
        c = torch.tanh(self.bridge_c(torch.cat([c[0], c[1]], dim=1)))
        return outputs, h, c


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx, use_copy=True, dropout=0.0):
        super().__init__()
        self.use_copy = use_copy
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm_cell = nn.LSTMCell(emb_dim, hid_dim)
        self.attn_proj = nn.Linear(2 * hid_dim, hid_dim, bias=False)
        self.combine = nn.Linear(hid_dim + 2 * hid_dim, hid_dim)
        self.out = nn.Linear(hid_dim, vocab_size)
        self.copy_gate = nn.Linear(hid_dim + 2 * hid_dim + emb_dim, 1)

    def step(self, prev_token, h, c, enc_outputs, enc_keys, src_ext, src_mask):
        prev_token = prev_token.masked_fill(prev_token >= self.vocab_size, UNK_IDX)
        emb = self.dropout(self.embedding(prev_token))
        h, c = self.lstm_cell(emb, (h, c))

        scores = torch.bmm(enc_keys, h.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~src_mask, -1e9)
        attn = F.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)

        hidden = torch.tanh(self.combine(torch.cat([h, context], dim=1)))
        generate_probs = F.softmax(self.out(self.dropout(hidden)), dim=1)
        n_extra = max(0, int(src_ext.max()) + 1 - self.vocab_size)
        generate_probs = F.pad(generate_probs, (0, n_extra))

        copy_probs = torch.zeros_like(generate_probs).scatter_add(1, src_ext, attn)

        p_copy = torch.sigmoid(self.copy_gate(torch.cat([h, context, emb], dim=1)))
        if not self.use_copy:
            p_copy = torch.zeros_like(p_copy)
        probs = (1 - p_copy) * generate_probs + p_copy * copy_probs
        return probs, h, c, attn, p_copy.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx, use_copy=True, dropout=0.0):
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = Encoder(vocab_size, emb_dim, hid_dim, pad_idx, dropout)
        self.decoder = Decoder(vocab_size, emb_dim, hid_dim, pad_idx, use_copy, dropout)

    def encode(self, src, src_lens):
        enc_outputs, h, c = self.encoder(src, src_lens)
        enc_keys = self.decoder.attn_proj(enc_outputs)
        src_mask = src != self.pad_idx
        return enc_outputs, enc_keys, src_mask, h, c

    def forward(self, src, src_ext, src_lens, tgt):
        enc_outputs, enc_keys, src_mask, h, c = self.encode(src, src_lens)
        all_probs = []
        for t in range(tgt.size(1) - 1):
            probs, h, c, _, _ = self.decoder.step(tgt[:, t], h, c, enc_outputs, enc_keys, src_ext, src_mask)
            all_probs.append(probs)
        return torch.stack(all_probs, dim=1)
