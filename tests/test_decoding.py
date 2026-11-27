"""Tests for the decoding rules and the extended vocabulary (run with:  python -m pytest tests)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import EOS, SOS, Vocab, encode_extended, make_example, source_oovs  # noqa: E402
from translate import blocked_words, split_clauses                         # noqa: E402

VOCAB = Vocab.build([["you", "are", "why", "?", "art", "thou", "wherefore"]], min_freq=1)


def test_no_immediate_repeat():
    assert 7 in blocked_words([3, 7])


def test_no_repeated_three_word_phrase():
    # "agree , agree ," must not continue with "agree": that would repeat "agree , agree"
    agree, comma = 10, 11
    assert agree in blocked_words([agree, comma, agree, comma])


def test_nothing_blocked_at_start():
    assert blocked_words([]) == set()


def test_split_keeps_short_sentences_whole():
    tokens = "thou art a villain , and i know it .".split()
    assert split_clauses(tokens) == [(0, len(tokens))]


def test_split_long_sentence_at_commas():
    tokens = ("one two three four five six seven , eight nine ten eleven twelve thirteen , "
              "fourteen fifteen sixteen seventeen .").split()
    pieces = split_clauses(tokens)
    assert len(pieces) > 1
    assert pieces[0][0] == 0 and pieces[-1][1] == len(tokens)          # nothing lost


def test_unknown_words_get_temporary_ids():
    oovs = source_oovs(["wherefore", "art", "thou", "romeo", "?"], VOCAB)
    assert oovs == ["romeo"]
    assert encode_extended(["romeo"], VOCAB, oovs) == [len(VOCAB)]


def test_target_can_copy_unknown_source_word():
    src, src_ext, tgt = make_example(["wherefore", "art", "thou", "romeo", "?"],
                                     ["why", "are", "you", "romeo", "?"], "to_modern", VOCAB)
    assert src[4] == VOCAB.index("<unk>")            # the encoder still sees <unk>
    assert src_ext[4] == len(VOCAB)                  # but the copy gate can point at "romeo"
    assert tgt[4] == len(VOCAB)                      # and the target asks for exactly that
    assert tgt[0] == VOCAB.index(SOS) and tgt[-1] == VOCAB.index(EOS)


def test_no_copy_ablation_uses_plain_unk():
    _, src_ext, tgt = make_example(["romeo"], ["romeo"], "to_modern", VOCAB, extended=False)
    assert max(src_ext) < len(VOCAB) and max(tgt) < len(VOCAB)


# ---------------------------------------------------------------- generator spelling filter
class _CharVocab:
    chars = ["a", "b", "t", " ", "."]


def test_spelling_filter_only_allows_real_words():
    from generate import SpellingFilter, current_word
    sf = SpellingFilter("bat tab", _CharVocab())
    allowed = dict(zip(_CharVocab.chars, sf.allowed("ba").tolist()))
    assert allowed["t"] and not allowed["b"]          # "bat" is a word, "bab..." is not
    assert not allowed[" "] and not allowed["."]      # "ba" is not a complete word
    assert dict(zip(_CharVocab.chars, sf.allowed("bat").tolist()))[" "]
    assert current_word("My lo") == "lo" and current_word("My lord, ") == ""
