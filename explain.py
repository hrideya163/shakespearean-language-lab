"""Explain a translation to the student (no machine learning here, just simple rules):

  word_map()      which words were kept, swapped, reworded, added or removed
  find_anchors()  names, numbers and negations: words that carry the meaning
  meaning_guard() warns when one of those anchors is missing from the output
  difficulty()    how archaic a passage is

Because the word map is computed by comparing the REAL input with the REAL
output, it is always accurate. An LLM that describes its own edits can't promise that.
"""
from difflib import SequenceMatcher

from lexicon import is_archaic, meaning_of

# ---------------------------------------------------------------------------
# 1. Word map
# ---------------------------------------------------------------------------
# Each item: {"type": ..., "src": input word or None, "out": list of output words}
#   kept      same word in input and output
#   swapped   archaic <-> modern dictionary pair      (thou -> you)
#   reworded  changed to a word that is not in the dictionary
#   added     new output word
#   removed   input word that disappeared


def _is_swap(src_word, out_words):
    meaning = meaning_of(src_word)                         # thou -> "you"
    if meaning and meaning.lower().split() == out_words:
        return True
    if len(out_words) == 1:                                # you -> thou (other direction)
        back = meaning_of(out_words[0])
        return back is not None and back.lower().split() == [src_word]
    return False


def word_map(src_tokens, out_tokens):
    src = [t.lower() for t in src_tokens]
    out = [t.lower() for t in out_tokens]
    items = []
    matcher = SequenceMatcher(a=src, b=out, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                items.append({"type": "kept", "src": src_tokens[i1 + k], "out": [out_tokens[j1 + k]]})
        elif op == "delete":
            for k in range(i1, i2):
                items.append({"type": "removed", "src": src_tokens[k], "out": []})
        elif op == "insert":
            for k in range(j1, j2):
                items.append({"type": "added", "src": None, "out": [out_tokens[k]]})
        else:  # "replace": pair up the changed words from left to right
            j = j1
            for i in range(i1, i2):
                meaning = meaning_of(src[i])
                n = len(meaning.split()) if meaning else 1
                if j < j2 and _is_swap(src[i], out[j:j + n]):
                    items.append({"type": "swapped", "src": src_tokens[i], "out": out_tokens[j:j + n]})
                    j += n
                elif j < j2:
                    items.append({"type": "reworded", "src": src_tokens[i], "out": [out_tokens[j]]})
                    j += 1
                else:
                    items.append({"type": "removed", "src": src_tokens[i], "out": []})
            for k in range(j, j2):
                items.append({"type": "added", "src": None, "out": [out_tokens[k]]})
    return items


def change_summary(items):
    """Count each type of change, e.g. {'kept': 5, 'swapped': 2, ...}"""
    counts = {"kept": 0, "swapped": 0, "reworded": 0, "added": 0, "removed": 0}
    for item in items:
        counts[item["type"]] += 1
    return counts


# ---------------------------------------------------------------------------
# 2. Meaning anchors + guard
# ---------------------------------------------------------------------------
NEGATIONS = {"not", "no", "never", "nay", "ne'er", "naught", "nought", "nothing",
             "none", "nor", "neither", "nobody", "cannot", "n't"}
NUMBER_WORDS = {"one": "1", "two": "2", "twain": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
                "eleven": "11", "twelve": "12", "twenty": "20", "hundred": "100",
                "thousand": "1000"}
NOT_NAMES = {"i", "o", "oh", "god", "lord", "lady", "sir", "madam", "king", "queen", "duke",
             "my", "good", "sweet", "come", "nay", "ay", "yes", "what", "how"}


def _is_negation(word):
    word = word.lower()
    return word in NEGATIONS or word.endswith("n't")


def _number(word):
    word = word.lower()
    return word if word.isdigit() else NUMBER_WORDS.get(word)


def find_anchors(tokens_cased):
    """tokens_cased must keep the original capital letters (used to spot names)."""
    names, numbers = [], []
    for i, tok in enumerate(tokens_cased):
        sentence_start = i == 0 or tokens_cased[i - 1] in {".", "!", "?", ":", ";", '"'}
        if (tok[:1].isupper() and tok.isalpha() and not sentence_start
                and tok.lower() not in NOT_NAMES and not is_archaic(tok)):
            names.append(tok)
        if _number(tok):
            numbers.append(_number(tok))
    negation = any(_is_negation(t) for t in tokens_cased)
    return {"names": names, "numbers": numbers, "negation": negation}


def restore_name_capitals(source_cased, output):
    """The model works in lowercase; put capitals back on names (romeo -> Romeo)."""
    names = {n.lower(): n for n in find_anchors(source_cased)["names"]}
    return [names.get(w, w) for w in output]


def meaning_guard(src_tokens_cased, out_tokens):
    """Return (anchors_found, anchors_kept, warnings)."""
    anchors = find_anchors(src_tokens_cased)
    out = [t.lower() for t in out_tokens]
    out_numbers = {_number(t) for t in out if _number(t)}
    found = kept = 0
    warnings = []

    for name in anchors["names"]:
        found += 1
        if name.lower() in out:
            kept += 1
        else:
            warnings.append(f"The name '{name}' is missing from the translation.")
    for num in anchors["numbers"]:
        found += 1
        if num in out_numbers:
            kept += 1
        else:
            warnings.append(f"The number '{num}' is missing from the translation.")

    out_negation = any(_is_negation(t) for t in out)
    if anchors["negation"]:
        found += 1
        if out_negation:
            kept += 1
        else:
            warnings.append("The original is NEGATIVE (not / never / nay...) but the translation "
                            "is not. The meaning may be reversed!")
    elif out_negation:
        warnings.append("The translation added a negation that is not in the original. "
                        "The meaning may be reversed!")
    return found, kept, warnings


# ---------------------------------------------------------------------------
# 3. Difficulty meter + glossary
# ---------------------------------------------------------------------------
def glossary(tokens):
    """{archaic word: modern meaning} for every archaic word in the passage."""
    return {t.lower(): meaning_of(t) for t in tokens if is_archaic(t)}


def difficulty(tokens):
    """Share of archaic words -> ('Easy' | 'Medium' | 'Hard', ratio)."""
    words = [t for t in tokens if t[:1].isalpha() or t.startswith("'")]
    if not words:
        return "Easy", 0.0
    ratio = sum(is_archaic(w) for w in words) / len(words)
    level = "Easy" if ratio < 0.05 else "Medium" if ratio < 0.15 else "Hard"
    return level, ratio
