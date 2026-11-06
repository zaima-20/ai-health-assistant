"""Tests for src.predict with tiny fake models (no real model loading, no training)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from src import predict
from src.predict import Condition, Predictor, PredictionResult

LABELS = ["cold", "flu", "malaria", "migraine", "allergy"]
TEXT = "I have a fever and chills"


class FakeBaseline:
    """Looks like a scikit-learn pipeline: labels in ``classes_``, fixed probabilities."""

    def __init__(self, probs: Sequence[float], labels: Sequence[str] = LABELS) -> None:
        self.classes_ = np.array(labels)
        self.probs = np.asarray(probs, dtype=float)
        self.seen: list[list[str]] = []

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        self.seen.append(list(texts))
        return np.tile(self.probs, (len(texts), 1))


class FakeBert(FakeBaseline):
    """Looks like BertClassifier: labels in ``labels`` instead of ``classes_``."""

    def __init__(self, probs: Sequence[float], labels: Sequence[str] = LABELS) -> None:
        super().__init__(probs, labels)
        del self.classes_
        self.labels = list(labels)


UNSORTED = [0.05, 0.10, 0.60, 0.20, 0.05]  # malaria > migraine > flu


# --------------------------------------------------------------------------- #
# Sorting and top-k
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fake_cls, name", [(FakeBaseline, "baseline"), (FakeBert, "bert")])
def test_predict_returns_top3_sorted(fake_cls: type, name: str) -> None:
    result = Predictor(name, model=fake_cls(UNSORTED)).predict(TEXT)
    assert isinstance(result, PredictionResult)
    assert [c.name for c in result.conditions] == ["malaria", "migraine", "flu"]
    assert [c.confidence for c in result.conditions] == pytest.approx([0.60, 0.20, 0.10])
    assert result.top == Condition("malaria", pytest.approx(0.60))


def test_confidences_are_floats_between_0_and_1() -> None:
    result = Predictor("baseline", model=FakeBaseline(UNSORTED)).predict(TEXT)
    for c in result.conditions:
        assert isinstance(c.confidence, float)
        assert 0.0 <= c.confidence <= 1.0


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_top_k_controls_number_of_results(k: int) -> None:
    result = Predictor("baseline", model=FakeBaseline(UNSORTED), top_k=k).predict(TEXT)
    assert len(result.conditions) == k


def test_default_top_k_comes_from_config() -> None:
    result = Predictor("baseline", model=FakeBaseline(UNSORTED)).predict(TEXT)
    assert len(result.conditions) == predict.TOP_K


def test_top_k_larger_than_label_count_returns_all() -> None:
    result = Predictor("baseline", model=FakeBaseline(UNSORTED), top_k=50).predict(TEXT)
    assert len(result.conditions) == len(LABELS)


def test_top_k_conditions_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        predict.top_k_conditions([0.5, 0.5], LABELS)


def test_model_receives_cleaned_text() -> None:
    fake = FakeBaseline(UNSORTED)
    Predictor("baseline", model=fake).predict("  I   have\n a fever  ")
    assert fake.seen == [["I have a fever"]]


# --------------------------------------------------------------------------- #
# Low-confidence flag
# --------------------------------------------------------------------------- #
def test_low_confidence_false_when_top_above_threshold() -> None:
    result = Predictor("baseline", model=FakeBaseline(UNSORTED), threshold=0.5).predict(TEXT)
    assert result.low_confidence is False


def test_low_confidence_true_when_top_below_threshold() -> None:
    probs = [0.30, 0.25, 0.20, 0.15, 0.10]
    result = Predictor("baseline", model=FakeBaseline(probs), threshold=0.5).predict(TEXT)
    assert result.low_confidence is True


def test_confidence_exactly_at_threshold_is_not_low() -> None:
    probs = [0.50, 0.20, 0.10, 0.10, 0.10]
    result = Predictor("baseline", model=FakeBaseline(probs), threshold=0.5).predict(TEXT)
    assert result.low_confidence is False


def test_default_threshold_comes_from_config() -> None:
    assert Predictor("baseline", model=FakeBaseline(UNSORTED)).threshold == \
        predict.CONFIDENCE_THRESHOLD


# --------------------------------------------------------------------------- #
# Short or empty input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
def test_empty_input_raises_value_error(text: str | None) -> None:
    with pytest.raises(ValueError, match="empty"):
        Predictor("baseline", model=FakeBaseline(UNSORTED)).predict(text)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["fever", "high fever", "  high   fever  "])
def test_short_input_raises_value_error(text: str) -> None:
    fake = FakeBaseline(UNSORTED)
    with pytest.raises(ValueError, match="at least 3 words"):
        Predictor("baseline", model=fake).predict(text)
    assert fake.seen == []  # the model was never called


def test_three_words_is_enough() -> None:
    result = Predictor("baseline", model=FakeBaseline(UNSORTED)).predict("fever and chills")
    assert result.top.name == "malaria"


# --------------------------------------------------------------------------- #
# Model loading errors
# --------------------------------------------------------------------------- #
def test_missing_bert_folder_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as err:
        Predictor("bert", model_path=tmp_path / "no_such_bert")
    message = str(err.value)
    assert "python -m src.bert_model" in message
    assert "--model baseline" in message


def test_bert_folder_without_config_counts_as_missing(tmp_path: Path) -> None:
    (tmp_path / "bert").mkdir()
    with pytest.raises(FileNotFoundError, match="No DistilBERT model"):
        Predictor("bert", model_path=tmp_path / "bert")


def test_missing_baseline_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="python -m src.baseline"):
        Predictor("baseline", model_path=tmp_path / "missing.joblib")


def test_unknown_model_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        Predictor("gpt", model=FakeBaseline(UNSORTED))


def test_model_without_labels_raises() -> None:
    class NoLabels:
        def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
            return np.zeros((len(texts), 2))

    with pytest.raises(TypeError):
        Predictor("baseline", model=NoLabels())


def test_model_loads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_load(name: str, path: Path | None = None) -> FakeBaseline:
        calls.append(name)
        return FakeBaseline(UNSORTED)

    monkeypatch.setattr(predict, "load_model", fake_load)
    p = Predictor("baseline")
    for _ in range(3):
        p.predict(TEXT)
    assert calls == ["baseline"]


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def test_cli_prints_percentages_and_flag(monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(predict, "load_model", lambda name, path=None: FakeBert(UNSORTED))
    assert predict.main(["--model", "bert", TEXT]) == 0
    out = capsys.readouterr().out
    assert "not a medical diagnosis" in out
    assert "malaria" in out and "60.0%" in out
    assert "low_confidence: False" in out


def test_cli_short_text_prints_error(monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(predict, "load_model", lambda name, path=None: FakeBaseline(UNSORTED))
    assert predict.main(["fever"]) == 1
    assert "at least 3 words" in capsys.readouterr().err
