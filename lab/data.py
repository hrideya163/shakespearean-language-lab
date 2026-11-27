import os
import re
from collections import Counter

from lab import config

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*'?|'[A-Za-z]+|\d+|[^\sA-Za-z\d]")


def tokenize(text, lower=True):
    text = text.replace("’", "'").replace("‘", "'")
    tokens = TOKEN_PATTERN.findall(text)
    return [t.lower() for t in tokens] if lower else tokens


def detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r" ([.,!?;:)\]])", r"\1", text)
    text = re.sub(r"([(\[]) ", r"\1", text)
    text = re.sub(r"\bi\b", "I", text)
    return text[:1].upper() + text[1:]


PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
TO_MODERN, TO_SHAKES = "<to_modern>", "<to_shakes>"
SPECIALS = [PAD, UNK, SOS, EOS, TO_MODERN, TO_SHAKES]


class Vocab:
    def __init__(self, words):
        self.itos = list(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq, extra_words=()):
        counts = Counter(t for tokens in token_lists for t in tokens)
        words = [w for w, c in counts.most_common() if c >= min_freq]
        words += [w for w in extra_words if w not in counts or counts[w] < min_freq]
        return cls(SPECIALS + [w for w in words if w not in SPECIALS])

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def index(self, word):
        return self.stoi[word]


def read_split(split, data_dir=config.CORPUS_DIR):
    def read(side):
        path = os.path.join(data_dir, f"{split}.{side}.nltktok")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run: python download_data.py")
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f]

    pairs = list(zip(read("original"), read("modern")))
    return [(s, m) for s, m in pairs if s and m]


def encode_source(tokens, direction, vocab):
    tag = TO_MODERN if direction == "to_modern" else TO_SHAKES
    return [vocab.index(tag)] + vocab.encode(tokens)


def encode_target(tokens, vocab):
    return [vocab.index(SOS)] + vocab.encode(tokens) + [vocab.index(EOS)]


def source_oovs(tokens, vocab):
    return list(dict.fromkeys(t for t in tokens if t not in vocab.stoi))


def encode_extended(tokens, vocab, oovs):
    unk = vocab.stoi[UNK]
    return [vocab.stoi[t] if t in vocab.stoi else
            len(vocab) + oovs.index(t) if t in oovs else unk for t in tokens]


def make_example(src_tokens, tgt_tokens, direction, vocab, extended=True):
    src = encode_source(src_tokens, direction, vocab)
    if not extended:
        return src, src, encode_target(tgt_tokens, vocab)
    oovs = source_oovs(src_tokens, vocab)
    src_ext = src[:1] + encode_extended(src_tokens, vocab, oovs)
    tgt_ext = [vocab.index(SOS)] + encode_extended(tgt_tokens, vocab, oovs) + [vocab.index(EOS)]
    return src, src_ext, tgt_ext


def make_examples(pairs, vocab, extended=True):
    examples = []
    for shakes, modern in pairs:
        s, m = tokenize(shakes), tokenize(modern)
        if len(s) > config.MAX_LEN or len(m) > config.MAX_LEN:
            continue
        examples.append(make_example(s, m, "to_modern", vocab, extended))
        examples.append(make_example(m, s, "to_shakes", vocab, extended))
    return examples
