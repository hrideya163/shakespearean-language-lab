"""A simple Shakespeare-vs-modern style classifier, used as a MEASURING TOOL:
  - RQ3: how Shakespearean is the generated text at each temperature?
  - RQ4: how modern is the translator's output?

Model: logistic regression on character n-grams (pieces of 2-4 characters such as
"th", "eth ", "thou"). Simple, fast and easy to explain.

  python style_classifier.py      # trains it and prints test accuracy
"""
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import config
from data import detokenize, read_split


def load_examples(split):
    texts, labels = [], []
    for shakes, modern in read_split(split):
        texts += [detokenize(shakes.split()), detokenize(modern.split())]
        labels += [1, 0]                       # 1 = Shakespeare, 0 = modern
    return texts, labels


def train():
    x_train, y_train = load_examples("train")
    x_test, y_test = load_examples("test")
    clf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True),
        LogisticRegression(max_iter=1000),
    )
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    joblib.dump(clf, config.STYLE_CLASSIFIER_PATH)
    print(f"Style classifier test accuracy: {accuracy:.1%}  (saved to {config.STYLE_CLASSIFIER_PATH})")
    return clf


_classifier = None


def shakespeare_score(texts):
    """Probability (0..1) that each text is Shakespearean. Trains the classifier if needed."""
    global _classifier
    if _classifier is None:
        if os.path.exists(config.STYLE_CLASSIFIER_PATH):
            _classifier = joblib.load(config.STYLE_CLASSIFIER_PATH)
        else:
            _classifier = train()
    return _classifier.predict_proba(texts)[:, 1].tolist()


if __name__ == "__main__":
    train()
