import argparse
import os
from collections import Counter
from dataclasses import dataclass, field

import torch

from lab import config
from lab.data import (EOS, PAD, SOS, TO_MODERN, TO_SHAKES, UNK, Vocab, detokenize, encode_extended,
                      encode_source, source_oovs, tokenize)
from lab.explain import restore_name_capitals
from lab.lexicon import MODERN_TO_ARCHAIC, meaning_of
from lab.translator.model import Seq2Seq

BEAM_SIZE = 5
COVERAGE_WEIGHT = 0.4
HINT_BONUS = 0.0
GREEDY = dict(beam_size=1, block_repeats=False, coverage_weight=0, hint_bonus=0)

SENTENCE_END = {".", "!", "?", ";"}
CLAUSE_END = {",", ":", ";"}
LONG_PIECE = 16
MIN_CLAUSE = 6


def split_clauses(tokens):
    sentences, start = [], 0
    for i, tok in enumerate(tokens):
        if tok in SENTENCE_END:
            sentences.append((start, i + 1))
            start = i + 1
    if start < len(tokens):
        sentences.append((start, len(tokens)))

    pieces = []
    for s, e in sentences:
        if e - s <= LONG_PIECE:
            pieces.append((s, e))
            continue
        cut = s
        for i in range(s, e):
            long_enough = i + 1 - cut >= MIN_CLAUSE and e - (i + 1) >= MIN_CLAUSE // 2
            if tokens[i] in CLAUSE_END and long_enough:
                pieces.append((cut, i + 1))
                cut = i + 1
        pieces.append((cut, e))
    return pieces


def blocked_words(tokens):
    blocked = set(tokens[-1:])
    last_two = tokens[-2:]
    for j in range(len(tokens) - 2):
        if tokens[j:j + 2] == last_two:
            blocked.add(tokens[j + 2])
    return blocked


@dataclass
class Hypothesis:
    tokens: list
    log_prob: float
    h: torch.Tensor
    c: torch.Tensor
    confidences: list = field(default_factory=list)
    copy_probs: list = field(default_factory=list)
    attention: list = field(default_factory=list)

    def score(self, coverage_weight=0.0):
        average = self.log_prob / max(len(self.tokens), 1)
        if not coverage_weight or not self.attention:
            return average
        received = torch.tensor(self.attention).sum(dim=0).clamp(1e-3, 1.0)
        return average + coverage_weight * received.log().mean().item()


class Translator:
    def __init__(self, checkpoint_path=config.TRANSLATOR_PATH):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"{checkpoint_path} not found. Run: python -m lab.translator.train")
        ckpt = torch.load(checkpoint_path)
        self.vocab = Vocab(ckpt["vocab"])
        self.model = Seq2Seq(**ckpt["model_args"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.extended = ckpt.get("extended", False)
        self.never_write = [self.vocab.index(t) for t in (PAD, SOS, TO_MODERN, TO_SHAKES)]

    def encode(self, source, direction):
        src_ids = encode_source(source, direction, self.vocab)
        src_ext = src_ids
        if self.extended:
            src_ext = src_ids[:1] + encode_extended(source, self.vocab, source_oovs(source, self.vocab))
        src_ext = torch.tensor([src_ext])
        return (src_ext, *self.model.encode(torch.tensor([src_ids]), torch.tensor([len(src_ids)])))

    def dictionary_hints(self, source, direction):
        hints = Counter()
        for word in source:
            meaning = meaning_of(word) if direction == "to_modern" else MODERN_TO_ARCHAIC.get(word)
            for m in (meaning or "").lower().split():
                if m in self.vocab.stoi:
                    hints[self.vocab.stoi[m]] += 1
        return hints

    def mask(self, log_probs, beams, step, block_repeats, hints, hint_bonus):
        log_probs[:, self.never_write] = float("-inf")
        if step == 0:
            log_probs[:, self.vocab.index(EOS)] = float("-inf")
        for i, beam in enumerate(beams):
            if block_repeats:
                log_probs[i, list(blocked_words(beam.tokens))] = float("-inf")
            written = Counter(beam.tokens)
            for word, needed in hints.items():
                if written[word] < needed:
                    log_probs[i, word] += hint_bonus
        return log_probs

    @torch.no_grad()
    def decode(self, source, direction, beam_size=BEAM_SIZE, block_repeats=True,
               coverage_weight=COVERAGE_WEIGHT, hint_bonus=HINT_BONUS):
        src_ext, enc_outputs, enc_keys, src_mask, h, c = self.encode(source, direction)
        hints = self.dictionary_hints(source, direction) if hint_bonus else Counter()
        eos = self.vocab.index(EOS)
        beams, finished = [Hypothesis([], 0.0, h, c)], []

        for step in range(int(1.5 * len(source)) + 4):
            k = len(beams)
            prev = torch.tensor([b.tokens[-1] if b.tokens else self.vocab.index(SOS) for b in beams])
            probs, h, c, attn, p_copy = self.model.decoder.step(
                prev, torch.cat([b.h for b in beams]), torch.cat([b.c for b in beams]),
                enc_outputs.expand(k, -1, -1), enc_keys.expand(k, -1, -1), src_ext.expand(k, -1),
                src_mask.expand(k, -1))
            log_probs = self.mask(torch.log(probs + 1e-9), beams, step, block_repeats, hints, hint_bonus)
            totals = torch.tensor([b.log_prob for b in beams]).unsqueeze(1) + log_probs

            new_beams = []
            for total, flat in zip(*totals.view(-1).topk(beam_size)):
                if total == float("-inf"):
                    continue
                i, word = divmod(flat.item(), log_probs.size(1))
                parent = beams[i]
                hyp = Hypothesis(parent.tokens + [word], total.item(), h[i:i + 1], c[i:i + 1],
                                 parent.confidences + [probs[i, word].item()],
                                 parent.copy_probs + [p_copy[i].item()],
                                 parent.attention + [attn[i, 1:].tolist()])
                (finished if word == eos else new_beams).append(hyp)
            beams = new_beams
            if len(finished) >= beam_size or not beams:
                break
        return sorted(finished or beams, key=lambda hyp: hyp.score(coverage_weight), reverse=True)

    def words(self, hyp, source, block_repeats=True):
        oovs = source_oovs(source, self.vocab)
        words, positions = [], []
        for t, token in enumerate(hyp.tokens):
            if token == self.vocab.index(EOS):
                break
            word = oovs[token - len(self.vocab)] if token >= len(self.vocab) else self.vocab.itos[token]
            if word == UNK:
                attn = hyp.attention[t]
                word = source[attn.index(max(attn))]
            if block_repeats and words and word == words[-1]:
                continue
            words.append(word)
            positions.append(t)
        return words, positions

    def translate(self, text, direction="to_modern", beam_size=BEAM_SIZE, block_repeats=True, split=False,
                  coverage_weight=COVERAGE_WEIGHT, hint_bonus=HINT_BONUS):
        source_cased = tokenize(text, lower=False)
        source = [t.lower() for t in source_cased]
        if not source:
            return None

        pieces = split_clauses(source) if split else [(0, len(source))]
        output, confidences, copy_probs, attention, alternatives = [], [], [], [], []
        for start, end in pieces:
            piece, piece_cased = source[start:end], source_cased[start:end]
            hyps = self.decode(piece, direction, beam_size, block_repeats, coverage_weight, hint_bonus)
            best = hyps[0]
            words, positions = self.words(best, piece, block_repeats)
            if split and len(pieces) > 1 and words and words[-1] in ".!?" and piece[-1] in CLAUSE_END:
                words[-1] = piece[-1]
            output += words
            confidences += [best.confidences[t] for t in positions]
            copy_probs += [best.copy_probs[t] for t in positions]
            attention += [[0.0] * start + best.attention[t] + [0.0] * (len(source) - end) for t in positions]
            options = [{"text": detokenize(restore_name_capitals(piece_cased, self.words(h, piece, block_repeats)[0])),
                        "score": round(h.score(coverage_weight), 3)} for h in hyps[:3]]
            alternatives.append({"source": detokenize(piece_cased), "options": options})

        output = restore_name_capitals(source_cased, output)
        for i in range(1, len(output)):
            if output[i - 1] in ".!?" and output[i][:1].islower() and output[i] != "i":
                output[i] = output[i][:1].upper() + output[i][1:]
        return {
            "source_tokens": source_cased,
            "output_tokens": output,
            "text": detokenize(output),
            "confidences": confidences,
            "copy_probs": copy_probs,
            "attention": attention,
            "alternatives": alternatives,
        }

    @torch.no_grad()
    def score(self, source_text, candidate_text, direction="to_modern"):
        source = tokenize(source_text)
        src_ext, enc_outputs, enc_keys, src_mask, h, c = self.encode(source, direction)
        oovs = source_oovs(source, self.vocab) if self.extended else []
        targets = encode_extended(tokenize(candidate_text), self.vocab, oovs) + [self.vocab.index(EOS)]
        token = torch.tensor([self.vocab.index(SOS)])
        log_probs = []
        for target in targets:
            probs, h, c, _, _ = self.model.decoder.step(token, h, c, enc_outputs, enc_keys, src_ext, src_mask)
            log_probs.append(torch.log(probs[0, target] + 1e-9).item())
            token = torch.tensor([target])
        return sum(log_probs) / len(log_probs), min(log_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--to_shakes", action="store_true", help="modern -> Shakespearean")
    parser.add_argument("--greedy", action="store_true", help="greedy decoding, for comparison")
    args = parser.parse_args()
    result = Translator().translate(args.text, "to_shakes" if args.to_shakes else "to_modern",
                                    **(GREEDY if args.greedy else {}))
    print(result["text"])
    for word, conf, cp in zip(result["output_tokens"], result["confidences"], result["copy_probs"]):
        print(f"  {word:15s} confidence={conf:.2f}  copied={cp:.2f}")


if __name__ == "__main__":
    main()
