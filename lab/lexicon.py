ARCHAIC = {
    "thou": "you", "thee": "you", "thy": "your", "thine": "yours", "ye": "you",
    "thyself": "yourself",
    "art": "are", "wert": "were", "wast": "were", "beest": "are",
    "hast": "have", "hath": "has", "hadst": "had",
    "dost": "do", "doth": "does", "didst": "did",
    "wilt": "will", "shalt": "shall", "canst": "can", "couldst": "could",
    "wouldst": "would", "shouldst": "should", "mayst": "may", "mightst": "might",
    "knowest": "know", "knoweth": "knows", "sayest": "say", "saith": "says",
    "goest": "go", "goeth": "goes", "cometh": "comes", "maketh": "makes",
    "speaketh": "speaks", "loveth": "loves", "seest": "see", "seeth": "sees",
    "durst": "dared", "bade": "told", "spake": "spoke", "wot": "know",
    "'tis": "it is", "'twas": "it was", "'twere": "it would be", "'twill": "it will",
    "o'er": "over", "ne'er": "never", "e'er": "ever", "e'en": "even",
    "'gainst": "against", "th'": "the", "i'": "in", "o'": "of", "an't": "if it",
    "wherefore": "why", "whence": "from where", "hence": "from here",
    "thence": "from there", "hither": "here", "thither": "there",
    "whither": "where", "yonder": "over there", "yon": "that",
    "wherein": "in which", "whereof": "of which", "whereon": "on which",
    "ere": "before", "anon": "soon", "oft": "often", "morrow": "morning",
    "whilst": "while", "betimes": "early",
    "ay": "yes", "aye": "yes", "nay": "no", "naught": "nothing", "nought": "nothing",
    "aught": "anything",
    "prithee": "please", "pray": "please", "methinks": "I think",
    "forsooth": "truly", "verily": "truly", "sooth": "truth", "troth": "faith",
    "alack": "alas", "fie": "shame", "marry": "indeed", "zounds": "damn",
    "perchance": "perhaps", "haply": "perhaps", "mayhap": "perhaps",
    "belike": "probably", "fain": "gladly", "withal": "as well",
    "sirrah": "sir", "knave": "rascal", "wench": "girl", "coz": "cousin",
    "liege": "lord", "nuncle": "uncle", "kinsman": "relative", "visage": "face",
    "afeard": "afraid", "bootless": "useless", "twain": "two", "nigh": "near",
    "betwixt": "between", "hie": "hurry", "beseech": "beg", "hap": "luck",
}

MODERN_TO_ARCHAIC = {
    "you": "thou", "your": "thy", "yours": "thine", "yourself": "thyself",
    "before": "ere", "often": "oft", "why": "wherefore", "never": "ne'er",
    "over": "o'er", "soon": "anon", "please": "prithee",
}


def meaning_of(word):
    word = word.lower()
    if word in ARCHAIC:
        return ARCHAIC[word]
    if word.endswith("'st") and len(word) > 4:
        return word[:-3]
    return None


def is_archaic(word):
    return meaning_of(word) is not None


def baseline_translate(tokens, direction="to_modern"):
    out = []
    for t in tokens:
        if direction == "to_modern":
            meaning = meaning_of(t)
            out.extend(meaning.lower().split() if meaning else [t])
        else:
            out.append(MODERN_TO_ARCHAIC.get(t, t))
    return out
