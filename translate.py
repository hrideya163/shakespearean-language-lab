"""Load the trained model and translate sentences.

  python translate.py "Wherefore art thou Romeo?"
  python translate.py --to_shakes "Why are you Romeo?"
"""
import argparse
import os

import torch

import config
from data import EOS, SOS, UNK, Vocab, detokenize, encode_source, tokenize
from translator_model import Seq2Seq
from explain import restore_name_capitals


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

    @torch.no_grad()
    def translate(self, text, direction="to_modern"):
        """Greedy decoding: at each step pick the most likely word.
        Greedy decoding is deterministic, so the same input always gives the same output."""
        source_cased = tokenize(text, lower=False)       # keep capitals for name detection
        source = [t.lower() for t in source_cased]
        if not source:
            return None

        src_ids = encode_source(source, direction, self.vocab)
        src = torch.tensor([src_ids], device=self.device)
        enc_outputs, enc_keys, src_mask, h, c = self.model.encode(src, torch.tensor([len(src_ids)]))

        token = torch.tensor([self.vocab.index(SOS)], device=self.device)
        eos, unk = self.vocab.index(EOS), self.vocab.index(UNK)
        output, confidences, copy_probs, attention = [], [], [], []

        for _ in range(2 * len(source) + 5):
            probs, h, c, attn, p_copy = self.model.decoder.step(
                token, h, c, enc_outputs, enc_keys, src, src_mask)
            confidence, idx = probs.max(dim=1)
            if idx.item() == eos:
                break
            attn = attn[0, 1:]                               # drop the direction tag position
            word = self.vocab.itos[idx.item()]
            if idx.item() == unk:                            # unknown word -> copy the input word
                word = source[int(attn.argmax())]            # the model was looking at
            output.append(word)
            confidences.append(confidence.item())
            copy_probs.append(p_copy.item())
            attention.append(attn.tolist())
            token = idx

        output = restore_name_capitals(source_cased, output)
        return {
            "source_tokens": source_cased,
            "output_tokens": output,
            "text": detokenize(output),
            "confidences": confidences,   # probability of each output word
            "copy_probs": copy_probs,     # how much each output word was copied (0..1)
            "attention": attention,       # rows = output words, columns = input words
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
        enc_outputs, enc_keys, src_mask, h, c = self.model.encode(src, torch.tensor([len(src_ids)]))
        targets = self.vocab.encode(candidate) + [self.vocab.index(EOS)]
        token = torch.tensor([self.vocab.index(SOS)], device=self.device)
        log_probs = []
        for target in targets:
            probs, h, c, _, _ = self.model.decoder.step(token, h, c, enc_outputs, enc_keys, src, src_mask)
            log_probs.append(torch.log(probs[0, target] + 1e-9).item())
            token = torch.tensor([target], device=self.device)
        return sum(log_probs) / len(log_probs), min(log_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--to_shakes", action="store_true", help="modern -> Shakespearean")
    args = parser.parse_args()
    result = Translator().translate(args.text, "to_shakes" if args.to_shakes else "to_modern")
    print(result["text"])
    for word, conf, cp in zip(result["output_tokens"], result["confidences"], result["copy_probs"]):
        print(f"  {word:15s} confidence={conf:.2f}  copied={cp:.2f}")


if __name__ == "__main__":
    main()
