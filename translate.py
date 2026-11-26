"""Load the trained model and translate sentences.

  python translate.py "Wherefore art thou Romeo?"
  python translate.py --to_shakes "Why are you Romeo?"
  python translate.py --greedy "Wherefore art thou Romeo?"     # compare with greedy decoding

How a translation is produced (DECODING), step by step:
  1. BEAM SEARCH  instead of keeping only the single most likely word at each step
             (greedy), keep the BEAM_SIZE most likely partial translations and extend
             all of them. A word that looks best now can lead to a bad sentence later.
             Beam search can recover from that.
  2. NO REPEATS  forbid the same word twice in a row and any three-word phrase that
             already appeared. Small seq2seq models often get stuck in loops
             ("agree, agree, agree"); this simple rule stops that.
  3. COVERAGE  when choosing among the finished candidates, penalise translations that
             never paid attention to some input word (they probably left it out).
             This is the "coverage penalty" of Google's 2016 translation system.
  4. LENGTH CAP  the output may be at most 1.5x the input length + 4 words.
The decoding experiment (exp_decoding.py) measures what each of these steps adds.
We also tried translating long inputs clause by clause (split=True). It made results
WORSE (the model needs the whole sentence as context), so it is off by default.
"""
import argparse
import os
from dataclasses import dataclass, field

import torch

import config
from data import (EOS, PAD, SOS, TO_MODERN, TO_SHAKES, UNK, Vocab, detokenize, encode_extended, encode_source,
                  source_oovs, tokenize)
from explain import restore_name_capitals
from translator_model import Seq2Seq

BEAM_SIZE = 5
SENTENCE_END = {".", "!", "?", ";"}
CLAUSE_END = {",", ":", ";"}
LONG_PIECE = 16          # pieces longer than this are split again at commas
MIN_CLAUSE = 6           # ...but never into clauses shorter than this
COVERAGE_WEIGHT = 0.4    # how strongly to penalise ignored input words (0 = off)


def split_clauses(tokens):
    """Cut a token list into short pieces: after . ! ? ; always, and after , : inside long
    sentences. Returns a list of (start, end) index pairs."""
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
    """Word ids that may NOT come next, so the output never repeats itself:
    the previous word again ("betrayed betrayed"), or a word that would repeat a
    three-word phrase already written ("agree , agree , agree")."""
    blocked = set()
    if tokens:
        blocked.add(tokens[-1])
    if len(tokens) >= 2:
        last_two = tokens[-2:]
        for j in range(len(tokens) - 2):
            if tokens[j:j + 2] == last_two:
                blocked.add(tokens[j + 2])
    return blocked


@dataclass
class Hypothesis:
    """One partial translation kept in the beam."""
    tokens: list            # word ids written so far
    log_prob: float         # sum of log-probabilities of those words
    h: torch.Tensor         # decoder LSTM state after the last word
    c: torch.Tensor
    confidences: list = field(default_factory=list)
    copy_probs: list = field(default_factory=list)
    attention: list = field(default_factory=list)

    def score(self, coverage_weight=0.0):
        """Average log-probability per word, minus a coverage penalty.
        - Without dividing by the length, beam search would prefer short translations
          (fewer words = fewer penalties).
        - Coverage: add up the attention each input word received over the whole output.
          A word that got (almost) none was probably left out: log(min(total, 1)) is 0 for
          a covered word and very negative for an ignored one."""
        avg = self.log_prob / max(len(self.tokens), 1)
        if not coverage_weight or not self.attention:
            return avg
        received = torch.tensor(self.attention).sum(dim=0).clamp(1e-3, 1.0)
        return avg + coverage_weight * received.log().mean().item()


class Translator:
    def __init__(self, checkpoint_path=config.TRANSLATOR_PATH, device="cpu"):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"{checkpoint_path} not found. Run: python train_translator.py")
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.vocab = Vocab(ckpt["vocab"])
        self.model = Seq2Seq(**ckpt["model_args"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = device

    # ------------------------------------------------------------------ decoding one piece
    @torch.no_grad()
    def decode(self, source, direction, beam_size=BEAM_SIZE, block_repeats=True,
               coverage_weight=COVERAGE_WEIGHT):
        """Translate one short list of (lower-case) tokens.
        beam_size=1 and block_repeats=False gives plain greedy decoding.
        Returns the finished hypotheses, best first."""
        src_ids = encode_source(source, direction, self.vocab)
        src = torch.tensor([src_ids], device=self.device)
        # unknown input words get temporary ids, so they can be copied (see data.py)
        src_ext = torch.tensor([src_ids[:1] + encode_extended(source, self.vocab, source_oovs(source, self.vocab))],
                               device=self.device)
        enc_outputs, enc_keys, src_mask, h, c = self.model.encode(src, torch.tensor([len(src_ids)]))
        sos, eos = self.vocab.index(SOS), self.vocab.index(EOS)
        max_len = int(1.5 * len(source)) + 4

        beams = [Hypothesis([], 0.0, h, c)]
        finished = []
        for step in range(max_len):
            k = len(beams)
            # run the decoder once for all k hypotheses (as a batch of size k)
            prev = torch.tensor([b.tokens[-1] if b.tokens else sos for b in beams], device=self.device)
            probs, h, c, attn, p_copy = self.model.decoder.step(
                prev, torch.cat([b.h for b in beams]), torch.cat([b.c for b in beams]),
                enc_outputs.expand(k, -1, -1), enc_keys.expand(k, -1, -1),
                src_ext.expand(k, -1), src_mask.expand(k, -1))
            log_probs = torch.log(probs + 1e-9)
            if block_repeats:
                for i, b in enumerate(beams):
                    for word in blocked_words(b.tokens):
                        log_probs[i, word] = float("-inf")
            for special in (PAD, SOS, TO_MODERN, TO_SHAKES):  # never write these
                log_probs[:, self.vocab.index(special)] = float("-inf")
            if step == 0:
                log_probs[:, eos] = float("-inf")           # no empty translations

            # every (hypothesis, next word) pair is a candidate; keep the best beam_size
            totals = torch.tensor([b.log_prob for b in beams]).unsqueeze(1) + log_probs
            top = totals.view(-1).topk(beam_size)
            new_beams = []
            for total, flat in zip(top.values.tolist(), top.indices.tolist()):
                if total == float("-inf"):
                    continue
                i, word = divmod(flat, log_probs.size(1))
                parent = beams[i]
                hyp = Hypothesis(parent.tokens + [word], total, h[i:i + 1], c[i:i + 1],
                                 parent.confidences + [probs[i, word].item()],
                                 parent.copy_probs + [p_copy[i].item()],
                                 parent.attention + [attn[i, 1:].tolist()])  # drop the direction tag
                (finished if word == eos else new_beams).append(hyp)
            beams = new_beams
            if len(finished) >= beam_size or not beams:
                break
        finished = finished or beams                        # hit the length cap
        return sorted(finished, key=lambda hyp: hyp.score(coverage_weight), reverse=True)

    def _words(self, hyp, source):
        """Turn a hypothesis into words. A temporary id means "copy unknown input word
        number k". A plain <unk> is replaced by the input word the model was looking at."""
        n = len(hyp.tokens) - (1 if hyp.tokens and hyp.tokens[-1] == self.vocab.index(EOS) else 0)
        oovs = source_oovs(source, self.vocab)
        words = []
        for t in range(n):
            token = hyp.tokens[t]
            word = oovs[token - len(self.vocab)] if token >= len(self.vocab) else self.vocab.itos[token]
            if word == UNK:
                attn = hyp.attention[t]
                word = source[max(range(len(attn)), key=attn.__getitem__)]
            words.append(word)
        return words, n

    # ------------------------------------------------------------------ full translation
    def translate(self, text, direction="to_modern", beam_size=BEAM_SIZE,
                  block_repeats=True, split=False, coverage_weight=COVERAGE_WEIGHT):
        source_cased = tokenize(text, lower=False)       # keep capitals for name detection
        source = [t.lower() for t in source_cased]
        if not source:
            return None

        pieces = split_clauses(source) if split else [(0, len(source))]
        output, confidences, copy_probs, attention, alternatives = [], [], [], [], []
        for start, end in pieces:
            piece = source[start:end]
            hyps = self.decode(piece, direction, beam_size, block_repeats, coverage_weight)
            best = hyps[0]
            words, n = self._words(best, piece)
            # a piece cut at a comma should not end with a full stop
            if split and len(pieces) > 1 and words and words[-1] in ".!?" and piece[-1] in CLAUSE_END:
                words[-1] = piece[-1]
            output += words
            confidences += best.confidences[:n]
            copy_probs += best.copy_probs[:n]
            # attention over the WHOLE input: zeros outside this piece
            for row in best.attention[:n]:
                attention.append([0.0] * start + row + [0.0] * (len(source) - end))
            alternatives.append({
                "source": detokenize(source_cased[start:end]),
                "options": [{"text": detokenize(restore_name_capitals(source_cased[start:end], self._words(h, piece)[0])),
                             "score": round(h.score(coverage_weight), 3)}
                            for h in hyps[:3]],
            })

        output = restore_name_capitals(source_cased, output)
        for i in range(1, len(output)):                  # capital letter after . ! ?
            if output[i - 1] in ".!?" and output[i][:1].islower() and output[i] != "i":
                output[i] = output[i][:1].upper() + output[i][1:]
        return {
            "source_tokens": source_cased,
            "output_tokens": output,
            "text": detokenize(output),
            "confidences": confidences,   # probability of each output word
            "copy_probs": copy_probs,     # how much each output word was copied (0..1)
            "attention": attention,       # rows = output words, columns = input words
            "pieces": [[s, e] for s, e in pieces],
            "alternatives": alternatives, # top beam-search candidates for each piece
        }

    @torch.no_grad()
    def score(self, source_text, candidate_text, direction="to_modern"):
        """How likely does the model find this candidate translation? (used in RQ5)
        We feed the candidate word by word (like teacher forcing) and collect the
        probability the model gives to each word. Returns (average, minimum) log-probability."""
        source = tokenize(source_text)
        candidate = tokenize(candidate_text)
        src_ids = encode_source(source, direction, self.vocab)
        src = torch.tensor([src_ids], device=self.device)
        oovs = source_oovs(source, self.vocab)
        src_ext = torch.tensor([src_ids[:1] + encode_extended(source, self.vocab, oovs)], device=self.device)
        enc_outputs, enc_keys, src_mask, h, c = self.model.encode(src, torch.tensor([len(src_ids)]))
        targets = encode_extended(candidate, self.vocab, oovs) + [self.vocab.index(EOS)]
        token = torch.tensor([self.vocab.index(SOS)], device=self.device)
        log_probs = []
        for target in targets:
            probs, h, c, _, _ = self.model.decoder.step(token, h, c, enc_outputs, enc_keys, src_ext, src_mask)
            log_probs.append(torch.log(probs[0, target] + 1e-9).item())
            token = torch.tensor([target], device=self.device)
        return sum(log_probs) / len(log_probs), min(log_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--to_shakes", action="store_true", help="modern -> Shakespearean")
    parser.add_argument("--greedy", action="store_true", help="old decoding, for comparison")
    args = parser.parse_args()
    options = dict(beam_size=1, block_repeats=False, split=False, coverage_weight=0) if args.greedy else {}
    result = Translator().translate(args.text, "to_shakes" if args.to_shakes else "to_modern", **options)
    print(result["text"])
    for word, conf, cp in zip(result["output_tokens"], result["confidences"], result["copy_probs"]):
        print(f"  {word:15s} confidence={conf:.2f}  copied={cp:.2f}")


if __name__ == "__main__":
    main()
