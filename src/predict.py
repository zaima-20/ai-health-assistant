"""Single predict(text) interface for either model, returning the top predictions.

Example (command line)::

    python -m src.predict --model bert "I have a high fever, chills and sweating every few days"

The user's text is only held in memory for the prediction; it is never
written to a file or log.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.config import CONFIDENCE_THRESHOLD, MODELS_DIR, TOP_K
from src.safety import DISCLAIMER, assess, check_red_flags

MODEL_NAMES: tuple[str, ...] = ("baseline", "bert")
DEFAULT_PATHS: dict[str, Path] = {
    "baseline": MODELS_DIR / "baseline.joblib",
    "bert": MODELS_DIR / "bert",
}
MIN_WORDS = 3


class ProbabilityModel(Protocol):
    """Anything with a scikit-learn-style ``predict_proba`` (baseline, BERT, or a test fake)."""

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n_texts, n_labels) array of probabilities."""
        ...


@dataclass(frozen=True)
class Condition:
    """One predicted condition and the model's confidence in it (0 to 1)."""

    name: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    """The top conditions (highest confidence first) and the low-confidence flag."""

    conditions: list[Condition]
    low_confidence: bool

    @property
    def top(self) -> Condition:
        """The single most likely condition."""
        return self.conditions[0]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def validate_text(text: str, min_words: int = MIN_WORDS) -> str:
    """Return the trimmed text, or raise ValueError if it is empty or too short."""
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        raise ValueError("Please describe your symptoms. The text is empty.")
    if len(cleaned.split()) < min_words:
        raise ValueError(
            f"Please describe your symptoms in a bit more detail (at least {min_words} "
            "words), for example: 'I have a fever, a sore throat and a headache'."
        )
    return cleaned


def get_labels(model: Any) -> list[str]:
    """Return the label names in the same order as the model's probability columns.

    scikit-learn pipelines keep them in ``classes_``; :class:`BertClassifier`
    keeps them in ``labels``.
    """
    if hasattr(model, "labels"):
        return list(model.labels)
    if hasattr(model, "classes_"):
        return [str(c) for c in model.classes_]
    raise TypeError("The model has neither 'labels' nor 'classes_', so labels are unknown.")


def top_k_conditions(probs: Sequence[float], labels: Sequence[str],
                     k: int = TOP_K) -> list[Condition]:
    """Return the ``k`` most likely conditions, highest confidence first."""
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (len(labels),):
        raise ValueError(f"Got {probs.shape[0] if probs.ndim else 0} probabilities "
                         f"for {len(labels)} labels.")
    order = np.argsort(-probs, kind="stable")[:k]
    return [Condition(labels[i], float(probs[i])) for i in order]


def load_model(model_name: str, path: Path | None = None) -> ProbabilityModel:
    """Load the saved ``baseline`` or ``bert`` model from disk.

    Raises FileNotFoundError with instructions if the model has not been trained.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(MODEL_NAMES)}.")
    path = Path(path) if path is not None else DEFAULT_PATHS[model_name]

    if model_name == "baseline":
        if not path.is_file():
            raise FileNotFoundError(
                f"No baseline model found at {path}. Train it first with: "
                "python -m src.baseline"
            )
        from src.baseline import load_model as load_baseline

        return load_baseline(path)

    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"No DistilBERT model found in {path}. Train it first with "
            "'python -m src.bert_model' (or the Colab notebook and copy the folder "
            "into models/bert/), or use the baseline model instead: --model baseline"
        )
    from src.bert_model import load_bert  # heavy import (torch), only when needed

    return load_bert(path)


# --------------------------------------------------------------------------- #
# Main interface
# --------------------------------------------------------------------------- #
class Predictor:
    """Load one model once, then call :meth:`predict` as often as needed.

    Args:
        model_name: ``"baseline"`` or ``"bert"``.
        model_path: optional custom location of the saved model.
        model: an already-loaded model (used by tests); skips loading from disk.
        top_k: how many conditions to return.
        threshold: below this top confidence, ``low_confidence`` is True.
    """

    def __init__(
        self,
        model_name: str = "baseline",
        model_path: Path | None = None,
        model: ProbabilityModel | None = None,
        top_k: int = TOP_K,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        """Load the model (unless one is passed in) and read its label list."""
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose one of: {', '.join(MODEL_NAMES)}."
            )
        self.model_name = model_name
        self.model = model if model is not None else load_model(model_name, model_path)
        self.labels = get_labels(self.model)
        self.top_k = top_k
        self.threshold = threshold

    def predict(self, text: str) -> PredictionResult:
        """Return the top conditions for ``text`` and whether confidence is low.

        Raises ValueError if the text is empty or shorter than three words.
        """
        cleaned = validate_text(text)
        probs = self.model.predict_proba([cleaned])[0]
        conditions = top_k_conditions(probs, self.labels, self.top_k)
        return PredictionResult(conditions, conditions[0].confidence < self.threshold)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def format_result(result: PredictionResult) -> str:
    """Render a result as plain text with percentages."""
    lines = [f"{i}. {c.name:<35} {c.confidence:6.1%}"
             for i, c in enumerate(result.conditions, start=1)]
    lines.append(f"low_confidence: {result.low_confidence}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m src.predict``. Returns the exit code."""
    parser = argparse.ArgumentParser(description="Predict likely conditions from symptoms.")
    parser.add_argument("text", help="symptom description in plain English (use quotes)")
    parser.add_argument("--model", choices=MODEL_NAMES, default="baseline",
                        help="which model to use (default: baseline)")
    args = parser.parse_args(argv)

    # Safety first: the red-flag check runs before (and independently of) the model.
    for flag in check_red_flags(args.text):  # emergency flags come first
        prefix = "!!! EMERGENCY" if flag.level == "emergency" else "! URGENT"
        print(f"{prefix} ({flag.name}): {flag.message}")
    print(DISCLAIMER)

    try:
        result = Predictor(args.model).predict(args.text)
    except (ValueError, FileNotFoundError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    print(format_result(result))
    report = assess(args.text, result)
    if report.confidence_message:
        print(report.confidence_message)
    return 0

if __name__ == "__main__":
    sys.exit(main())
