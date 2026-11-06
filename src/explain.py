"""Explain which words in the input pushed the prediction.

Two methods, chosen automatically from the model type:

* Linear (baseline TF-IDF + Logistic Regression): each word's score is its
  TF-IDF value times the model's weight for that word and label. The label's
  raw score (logit) is exactly the sum of these plus a constant, so this is exact.
* Leave-one-word-out (any other model, e.g. DistilBERT): remove each word,
  predict again, and use the drop in the label's probability as the score.

Example::

    python -m src.explain --model baseline "I have a high fever, chills and sweating every few days"
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.predict import MODEL_NAMES, Predictor, validate_text
from src.safety import DISCLAIMER, check_red_flags

MAX_POSITIVE = 5
MAX_NEGATIVE = 3
MIN_EFFECT = 1e-4  # scores closer to zero than this count as "no effect"

# Common words hidden from the leave-one-out (BERT) display only. Removing them
# mostly breaks the grammar rather than removing a symptom, so their scores are
# noise for a reader. Negations ("no", "not", "never", "without", "can't") are
# deliberately NOT listed, because they change meaning.
FILLER_WORDS: frozenset[str] = frozenset({
    "i", "i'm", "i've", "i'd", "me", "my", "mine", "myself", "you", "your", "we", "our",
    "he", "him", "his", "she", "her", "it", "it's", "its", "they", "them", "their",
    "a", "an", "the", "and", "or", "but", "so", "of", "to", "in", "on", "at", "for",
    "with", "from", "by", "about", "as", "into", "than", "then", "up", "out", "over",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "get", "gets", "got", "getting",
    "feel", "feels", "feeling", "felt",
    "when", "while", "if", "that", "this", "these", "those", "there", "here",
    "what", "which", "who", "also", "just", "very", "really", "quite", "too",
    "every", "some", "any", "all", "lot", "lots", "much", "many", "more", "most",
})

_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def select_top(scores: dict[str, float], max_positive: int = MAX_POSITIVE,
               max_negative: int = MAX_NEGATIVE,
               min_effect: float = MIN_EFFECT) -> list[tuple[str, float]]:
    """Keep the strongest positive and negative words, sorted by score (highest first).

    Positive words (towards the label) come first, strongest first; negative
    words (away from the label) come last, with the strongest push away at the end.
    """
    pos = sorted(((w, s) for w, s in scores.items() if s >= min_effect),
                 key=lambda ws: (-ws[1], ws[0]))[:max_positive]
    neg = sorted(((w, s) for w, s in scores.items() if s <= -min_effect),
                 key=lambda ws: (ws[1], ws[0]))[:max_negative]
    return pos + neg[::-1]


def _label_index(predictor: Predictor, label: str) -> int:
    """Column of ``label`` in the model's probability output."""
    if label not in predictor.labels:
        raise ValueError(f"Unknown label '{label}'. Known labels: {', '.join(predictor.labels)}")
    return predictor.labels.index(label)


def normalize_word(token: str) -> str:
    """Lowercase a token and strip punctuation from its ends ("Fever," -> "fever")."""
    return _EDGE_PUNCT.sub("", token).lower()


# --------------------------------------------------------------------------- #
# Method 1: exact linear contributions (baseline)
# --------------------------------------------------------------------------- #
def linear_parts(model: Any) -> tuple[Any, Any] | None:
    """Return (vectorizer, classifier) if ``model`` is a vectorizer + linear-model pipeline."""
    steps = getattr(model, "named_steps", None)
    if not steps:
        return None
    parts = list(steps.values())
    vectorizer, clf = parts[0], parts[-1]
    if hasattr(vectorizer, "transform") and hasattr(vectorizer, "get_feature_names_out") \
            and hasattr(clf, "coef_"):
        return vectorizer, clf
    return None


def _coef_row(clf: Any, class_index: int) -> np.ndarray:
    """Weights for one class. Binary models store one row (for class 1); class 0 is its negative."""
    coef = np.asarray(clf.coef_)
    if coef.shape[0] == 1:
        return coef[0] if class_index == 1 else -coef[0]
    return coef[class_index]


def linear_scores(vectorizer: Any, clf: Any, text: str, class_index: int) -> dict[str, float]:
    """Score every word in ``text``: TF-IDF value x weight for the class."""
    row = vectorizer.transform([text])
    if hasattr(row, "tocsr"):
        row = row.tocsr()
        indices, values = row.indices, row.data
    else:  # dense array
        dense = np.asarray(row)[0]
        indices = np.flatnonzero(dense)
        values = dense[indices]
    names = vectorizer.get_feature_names_out()
    weights = _coef_row(clf, class_index)
    return {str(names[j]): float(v * weights[j]) for j, v in zip(indices, values)}


# --------------------------------------------------------------------------- #
# Method 2: leave-one-word-out (BERT or any black-box model)
# --------------------------------------------------------------------------- #
def leave_one_out_scores(model: Any, text: str, class_index: int) -> dict[str, float]:
    """Drop in the class probability when each distinct word is removed.

    Repeated words are removed together, so each word gets one score. All
    variants are predicted in a single ``predict_proba`` call (the model batches them).
    """
    tokens = text.split()
    keys = [normalize_word(t) for t in tokens]
    words = [w for w in dict.fromkeys(keys) if w]  # unique, in order, skip pure punctuation
    if not words:
        return {}
    variants = [" ".join(t for t, k in zip(tokens, keys) if k != w) for w in words]
    probs = np.asarray(model.predict_proba([text, *variants]))[:, class_index]
    return {w: float(probs[0] - p) for w, p in zip(words, probs[1:])}


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
def explain_method(predictor: Predictor) -> str:
    """Name of the method :func:`explain` will use for this predictor."""
    return "linear" if linear_parts(predictor.model) else "leave-one-out"


def explain(predictor: Predictor, text: str, label: str,
            max_positive: int = MAX_POSITIVE,
            max_negative: int = MAX_NEGATIVE,
            hide_filler: bool = True) -> list[tuple[str, float]]:
    """Return the words that most influenced ``label`` for ``text``.

    The result is a list of (word, score) sorted from the strongest push
    towards the label to the strongest push away: at most ``max_positive``
    positive and ``max_negative`` negative words. Words with no effect are left out.

    Baseline scores are logit contributions; leave-one-out scores are
    probability drops (0 to 1). For leave-one-out only, words in
    :data:`FILLER_WORDS` are hidden (``hide_filler``); the other words' scores
    are unchanged. The baseline output never hides anything.
    """
    cleaned = validate_text(text)
    class_index = _label_index(predictor, label)
    parts = linear_parts(predictor.model)
    if parts:
        scores = linear_scores(*parts, cleaned, class_index)
    else:
        scores = leave_one_out_scores(predictor.model, cleaned, class_index)
        if hide_filler:
            scores = {w: s for w, s in scores.items() if w not in FILLER_WORDS}
    return select_top(scores, max_positive, max_negative)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def format_explanation(label: str, confidence: float, words: list[tuple[str, float]],
                       method: str) -> str:
    """Plain-text report of the top prediction and its important words."""
    unit = "logit contribution" if method == "linear" else "drop in probability if removed"
    lines = [f"Top prediction: {label} ({confidence:.1%})",
             f"Important words ({method}; score = {unit}):"]
    if not words:
        lines.append("  (no single word had a measurable effect)")
    for word, score in words:
        direction = "towards" if score > 0 else "away from"
        lines.append(f"  {score:+.4f}  {word:<15} pushes {direction} {label}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m src.explain``. Returns the exit code."""
    parser = argparse.ArgumentParser(description="Show which words drove the top prediction.")
    parser.add_argument("text", help="symptom description in plain English (use quotes)")
    parser.add_argument("--model", choices=MODEL_NAMES, default="baseline",
                        help="which model to use (default: baseline)")
    args = parser.parse_args(argv)

    for flag in check_red_flags(args.text):  # safety first, before any model
        prefix = "!!! EMERGENCY" if flag.level == "emergency" else "! URGENT"
        print(f"{prefix} ({flag.name}): {flag.message}")
    print(DISCLAIMER)

    try:
        predictor = Predictor(args.model)
        top = predictor.predict(args.text).top
        words = explain(predictor, args.text, top.name)
    except (ValueError, FileNotFoundError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    print(format_explanation(top.name, top.confidence, words, explain_method(predictor)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
