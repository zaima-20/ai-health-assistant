"""Tests for src.baseline using a tiny fake dataset (fits in milliseconds)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import baseline

WORDS = {
    "flu": ["fever chills body aches", "high fever and aching muscles", "chills fever tired"],
    "cold": ["runny nose sneezing", "sneezing and blocked nose", "sore throat runny nose"],
    "rash": ["itchy red rash on skin", "red itchy skin patches", "skin rash itching"],
}


def _frame(repeats: int) -> pd.DataFrame:
    rows = [
        {"text": f"{text} day {i}", "label": label}
        for label, texts in WORDS.items()
        for text in texts
        for i in range(repeats)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def train_df() -> pd.DataFrame:
    return _frame(2)


@pytest.fixture
def val_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["fever and chills", "runny nose", "itchy rash"],
            "label": ["flu", "cold", "rash"],
        }
    )


def test_build_pipeline_settings() -> None:
    model = baseline.build_pipeline((1, 2), C=10.0, seed=7)
    assert model.named_steps["tfidf"].ngram_range == (1, 2)
    assert model.named_steps["clf"].C == 10.0
    assert model.named_steps["clf"].random_state == 7


def test_train_baseline_predicts_known_labels(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> None:
    model, seconds = baseline.train_baseline(train_df, C=10.0)
    assert seconds >= 0
    assert list(model.classes_) == ["cold", "flu", "rash"]
    assert list(model.predict(val_df["text"])) == ["flu", "cold", "rash"]
    probs = model.predict_proba(val_df["text"])
    assert probs.shape == (3, 3)
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])


def test_grid_search_covers_every_setting(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> None:
    results = baseline.grid_search(train_df, val_df, [(1, 1), (1, 2)], [0.1, 1.0])
    assert [(tuple(r["ngram_range"]), r["C"]) for r in results] == [
        ((1, 1), 0.1), ((1, 1), 1.0), ((1, 2), 0.1), ((1, 2), 1.0)
    ]
    assert all(0.0 <= r["val_macro_f1"] <= 1.0 for r in results)
    assert "val macro-F1" in baseline.format_grid_table(results)


def test_pick_best_uses_macro_f1_and_breaks_ties_by_order() -> None:
    results = [
        {"ngram_range": [1, 1], "C": 0.1, "val_macro_f1": 0.8},
        {"ngram_range": [1, 1], "C": 1.0, "val_macro_f1": 0.9},
        {"ngram_range": [1, 2], "C": 1.0, "val_macro_f1": 0.9},
    ]
    assert baseline.pick_best(results)["C"] == 1.0
    assert baseline.pick_best(results)["ngram_range"] == [1, 1]


def test_save_and_load_round_trip(
    train_df: pd.DataFrame, val_df: pd.DataFrame, tmp_path: Path
) -> None:
    model, _ = baseline.train_baseline(train_df)
    path = tmp_path / "m.joblib"
    baseline.save_model(model, path)
    loaded = baseline.load_model(path)
    assert list(loaded.predict(val_df["text"])) == list(model.predict(val_df["text"]))
