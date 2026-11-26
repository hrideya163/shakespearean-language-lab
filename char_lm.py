"""Character-level language model: predicts the next character from the previous ones.

Text in:  "To be or not to b"   ->   most likely next character: "e"
"""
import os

import torch
import torch.nn as nn

import config
from cells import CELLS
from data import detokenize


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_shakespeare_text(split, data_dir=config.CORPUS_DIR):
    """All Shakespeare lines of one split as one long string (one line per sentence)."""
    path = os.path.join(data_dir, f"{split}.original.nltktok")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python download_data.py")
    with open(path, encoding="utf-8") as f:
        # The corpus is pre-tokenised ("Madam , he's well ."), so join it back into normal text.
        lines = [detokenize(line.split()) for line in f if line.strip()]
    return "\n".join(lines) + "\n"


class CharVocab:
    def __init__(self, chars):
        self.chars = list(chars)
        self.index = {ch: i for i, ch in enumerate(self.chars)}

    @classmethod
    def build(cls, text):
        return cls(sorted(set(text)))

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return [self.index[ch] for ch in text if ch in self.index]   # skip unknown characters

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CharLM(nn.Module):
    """embedding -> hand-written recurrent cell (rnn / gru / lstm) -> linear -> next-char scores"""

    def __init__(self, vocab_size, emb_dim, hid_dim, cell_type="lstm", dropout=0.0):
        super().__init__()
        self.cell_type = cell_type
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.cell = CELLS[cell_type](emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hid_dim, vocab_size)

    def forward(self, x, state=None, record=False, ablate=None):
        """x: (B, T) character ids.
        Returns logits (B, T, V), the final state, and (if record=True) a list with
        the gate values and the hidden state h of every time step.
        ablate: optional list of hidden units to switch off (set to 0) at every step.
                Used in exp_inside_lstm.py to test what a unit is responsible for."""
        batch, steps = x.shape
        if state is None:
            state = self.cell.init_state(batch, x.device)
        emb = self.embedding(x)
        outputs, recorded = [], []
        for t in range(steps):                       # one character at a time
            h, state, gates = self.cell(emb[:, t], state)
            if ablate:
                state = tuple(s.index_fill(1, torch.tensor(ablate), 0.0) for s in state)
                h = state[0]                         # h is always the first part of the state
            outputs.append(h)
            if record:
                recorded.append({"hidden": h.detach(), **{name: v.detach() for name, v in gates.items()}})
        logits = self.out(self.dropout(torch.stack(outputs, dim=1)))
        return logits, state, recorded


def load_generator(name="lstm", device="cpu"):
    """Load a trained generator by name (lstm, gru, rnn, rnn_matched, ...). Returns (model, vocab)."""
    path = config.generator_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python train_generator.py --cell {name}")
    ckpt = torch.load(path, map_location=device)
    model = CharLM(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, CharVocab(ckpt["chars"])
