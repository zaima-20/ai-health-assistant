"""Tests for src.evaluate using tiny hand-made label lists."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import evaluate

LABELS = ["cold", "flu", "malaria"]


def test_perfect_predictions() -> None:
    y = ["cold", "flu", "malaria", "flu"]
    m = evaluate.compute_metrics(y, y, LABELS)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["n_samples"] == 4
    assert m["per_class"]["flu"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2}


def test_known_values() -> None:
    y_true = ["cold", "cold", "flu", "flu"]
    y_pred = ["cold", "flu", "flu", "flu"]
    m = evaluate.compute_metrics(y_true, y_pred, LABELS)
    assert m["accuracy"] == 0.75
    # cold F1 = 2/3, flu F1 = 0.8, malaria absent -> 0; macro = (0.6667 + 0.8 + 0) / 3
    assert m["per_class"]["cold"]["f1"] == pytest.approx(0.6667, abs=1e-4)
    assert m["per_class"]["flu"]["f1"] == pytest.approx(0.8, abs=1e-4)
    assert m["per_class"]["malaria"]["support"] == 0
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 3, abs=1e-4)


def test_top_confusions_sorted_and_excludes_correct() -> None:
    y_true = ["cold", "cold", "cold", "flu", "malaria"]
    y_pred = ["flu", "flu", "cold", "malaria", "malaria"]
    result = evaluate.top_confusions(y_true, y_pred, LABELS, n=5)
    assert result == [
        {"true": "cold", "predicted": "flu", "count": 2},
        {"true": "flu", "predicted": "malaria", "count": 1},
    ]
    assert len(evaluate.top_confusions(y_true, y_pred, LABELS, n=1)) == 1


def test_plot_confusion_matrix_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "cm.png"
    evaluate.plot_confusion_matrix(["cold", "flu"], ["cold", "cold"], LABELS, out, "Test")
    assert out.exists() and out.stat().st_size > 0


def test_save_and_load_metrics(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "metrics.json"
    metrics = {"accuracy": 0.5, "macro_f1": 0.4}
    evaluate.save_metrics(metrics, path)
    assert evaluate.load_metrics(path) == metrics
