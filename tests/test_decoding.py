from lab.data import EOS, SOS, Vocab, encode_extended, make_example, source_oovs
from lab.generator.sample import SpellingFilter, current_word
from lab.translator.decode import blocked_words, split_clauses

VOCAB = Vocab.build([["you", "are", "why", "?", "art", "thou", "wherefore"]], min_freq=1)


def test_no_immediate_repeat():
    assert 7 in blocked_words([3, 7])


def test_no_repeated_three_word_phrase():
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
    assert pieces[0][0] == 0 and pieces[-1][1] == len(tokens)


def test_unknown_words_get_temporary_ids():
    oovs = source_oovs(["wherefore", "art", "thou", "romeo", "?"], VOCAB)
    assert oovs == ["romeo"]
    assert encode_extended(["romeo"], VOCAB, oovs) == [len(VOCAB)]


def test_target_can_copy_unknown_source_word():
    src, src_ext, tgt = make_example(["wherefore", "art", "thou", "romeo", "?"],
                                     ["why", "are", "you", "romeo", "?"], "to_modern", VOCAB)
    assert src[4] == VOCAB.index("<unk>")
    assert src_ext[4] == len(VOCAB)
    assert tgt[4] == len(VOCAB)
    assert tgt[0] == VOCAB.index(SOS) and tgt[-1] == VOCAB.index(EOS)


def test_no_copy_ablation_uses_plain_unk():
    _, src_ext, tgt = make_example(["romeo"], ["romeo"], "to_modern", VOCAB, extended=False)
    assert max(src_ext) < len(VOCAB) and max(tgt) < len(VOCAB)


class TinyCharVocab:
    chars = ["a", "b", "t", " ", "."]


def test_spelling_filter_only_allows_real_words():
    sf = SpellingFilter("bat tab", TinyCharVocab())
    allowed = dict(zip(TinyCharVocab.chars, sf.allowed("ba").tolist()))
    assert allowed["t"] and not allowed["b"]
    assert not allowed[" "] and not allowed["."]
    assert dict(zip(TinyCharVocab.chars, sf.allowed("bat").tolist()))[" "]
    assert current_word("My lo") == "lo" and current_word("My lord, ") == ""
