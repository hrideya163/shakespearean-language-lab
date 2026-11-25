"""Small tests for the non-ML parts (run with:  python -m pytest tests).

They check that the tools used to MEASURE the models behave as intended,
so the experiment results can be trusted.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import Vocab, detokenize, tokenize                      # noqa: E402
from explain import difficulty, find_anchors, meaning_guard, word_map  # noqa: E402
from lexicon import baseline_translate, meaning_of                # noqa: E402
from stats import bootstrap_ci                                    # noqa: E402


def test_tokenizer_keeps_archaic_contractions():
    assert tokenize("'Tis o'er, speak'st thou?") == ["'tis", "o'er", ",", "speak'st", "thou", "?"]


def test_detokenize_fixes_spacing():
    assert detokenize(["you", "are", "romeo", "!"]) == "You are romeo!"


def test_vocab_unknown_words():
    vocab = Vocab.build([["a", "a", "b"]], min_freq=2)
    assert vocab.decode(vocab.encode(["a", "b"])) == ["a", "<unk>"]


def test_lexicon():
    assert meaning_of("Thou") == "you"
    assert meaning_of("speak'st") == "speak"
    assert meaning_of("table") is None
    assert baseline_translate(["'tis", "thy", "name"]) == ["it", "is", "your", "name"]


def test_word_map_labels():
    src = tokenize("Thou hast betrayed me", lower=False)
    out = ["you", "have", "betrayed", "me"]
    assert [i["type"] for i in word_map(src, out)] == ["swapped", "swapped", "kept", "kept"]


def test_word_map_multiword_swap():
    items = word_map(["'tis", "true"], ["it", "is", "true"])
    assert items[0] == {"type": "swapped", "src": "'tis", "out": ["it", "is"]}


def test_anchors():
    anchors = find_anchors(tokenize("I know not where Portia is, nor her two rings.", lower=False))
    assert anchors["names"] == ["Portia"]
    assert anchors["numbers"] == ["2"]
    assert anchors["negation"]


def test_meaning_guard_catches_lost_negation_and_name():
    src = tokenize("I know not where Portia is.", lower=False)
    _, _, warnings = meaning_guard(src, ["i", "know", "where", "she", "is", "."])
    assert len(warnings) == 2


def test_meaning_guard_accepts_contraction():
    src = tokenize("I know not.", lower=False)
    _, _, warnings = meaning_guard(src, ["i", "don't", "know", "."])
    assert warnings == []


def test_difficulty():
    assert difficulty(tokenize("The cat sat on the mat."))[0] == "Easy"
    assert difficulty(tokenize("Wherefore art thou?"))[0] == "Hard"


def test_bootstrap_interval_contains_mean():
    values = list(range(100))
    low, high = bootstrap_ci(lambda idx: sum(values[i] for i in idx) / len(idx), len(values))
    assert low < 49.5 < high
