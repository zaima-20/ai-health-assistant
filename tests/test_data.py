"""Tests for src.data using tiny fake DataFrames (no downloads)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import data


@pytest.fixture
def fake_train() -> pd.DataFrame:
    """Twenty rows, two labels with 10 rows each, plus messy spacing and case."""
    texts = [f"  symptom   text {i} " for i in range(20)]
    labels = [" Flu " if i % 2 == 0 else "common cold" for i in range(20)]
    labels[1] = "Common Cold "  # same label, different spelling
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def clean_train(fake_train: pd.DataFrame) -> pd.DataFrame:
    """The fake train set after cleaning."""
    return data.clean_dataframe(fake_train)


def test_clean_text_strips_and_collapses_spaces() -> None:
    assert data.clean_text("  I have   a\theadache \n") == "I have a headache"


def test_clean_label_lowercases_and_strips() -> None:
    assert data.clean_label("  Common  Cold ") == "common cold"


def test_clean_dataframe_merges_label_spellings(clean_train: pd.DataFrame) -> None:
    assert list(clean_train.columns) == ["text", "label"]
    assert set(clean_train["label"]) == {"flu", "common cold"}
    assert clean_train["text"].iloc[0] == "symptom text 0"


@pytest.fixture
def messy_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A train set with duplicates and test leakage, plus a small test set."""
    train = pd.DataFrame(
        {
            "text": ["a", "b", "a", "c", "d", "c", "e"],
            "label": ["flu", "cold", "flu", "flu", "cold", "flu", "cold"],
        }
    )
    test = pd.DataFrame({"text": ["c", "z"], "label": ["flu", "cold"]})
    return train, test


def test_remove_train_duplicates_keeps_first_copy() -> None:
    train = pd.DataFrame({"text": ["a", "b", "a"], "label": ["flu", "cold", "cold"]})
    test = pd.DataFrame({"text": ["z"], "label": ["flu"]})
    cleaned, stats = data.remove_train_duplicates(train, test)
    assert cleaned["text"].tolist() == ["a", "b"]
    assert cleaned["label"].tolist() == ["flu", "cold"]  # first copy wins
    assert stats["removed_duplicates"] == 1
    assert stats["removed_test_overlap"] == 0


def test_remove_train_duplicates_drops_test_overlap(
    messy_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = messy_pair
    cleaned, stats = data.remove_train_duplicates(train, test)
    assert cleaned["text"].tolist() == ["a", "b", "d", "e"]
    assert stats == {
        "rows_before": 7,
        "removed_duplicates": 2,
        "removed_test_overlap": 1,
        "rows_after": 4,
    }
    assert data.find_duplicates(cleaned)["extra_rows"] == 0
    assert data.find_overlap(cleaned, test)["shared_texts"] == 0


def test_remove_train_duplicates_leaves_inputs_unchanged(
    messy_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = messy_pair
    train_before, test_before = train.copy(), test.copy()
    data.remove_train_duplicates(train, test)
    pd.testing.assert_frame_equal(test, test_before)
    pd.testing.assert_frame_equal(train, train_before)


def test_summary_includes_cleaning_stats(
    messy_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = messy_pair
    cleaned, stats = data.remove_train_duplicates(train, test)
    summary = data.summarize(cleaned.iloc[:2], cleaned.iloc[2:], test, cleaned, stats)
    assert summary["train_cleaning"] == stats
    assert "removed 2 duplicates, 1 also in test" in data.format_summary(summary)


def test_make_splits_sizes_and_stratification(clean_train: pd.DataFrame) -> None:
    train, val = data.make_splits(clean_train, val_size=0.2, seed=0)
    assert len(train) == 16 and len(val) == 4
    assert val["label"].value_counts().to_dict() == {"flu": 2, "common cold": 2}
    assert set(train["text"]).isdisjoint(set(val["text"]))


def test_make_splits_is_reproducible(clean_train: pd.DataFrame) -> None:
    a = data.make_splits(clean_train, val_size=0.2, seed=42)[1]
    b = data.make_splits(clean_train, val_size=0.2, seed=42)[1]
    pd.testing.assert_frame_equal(a, b)


def test_make_splits_does_not_modify_input(clean_train: pd.DataFrame) -> None:
    before = clean_train.copy()
    data.make_splits(clean_train, val_size=0.2, seed=0)
    pd.testing.assert_frame_equal(clean_train, before)


def test_label_round_trip(clean_train: pd.DataFrame, tmp_path: Path) -> None:
    labels = data.build_label_list(clean_train)
    assert labels == ["common cold", "flu"]

    path = tmp_path / "labels.json"
    data.save_labels(labels, path)
    assert data.load_labels(path) == labels

    l2i = data.label_to_id(labels)
    i2l = data.id_to_label(labels)
    assert l2i == {"common cold": 0, "flu": 1}
    assert all(i2l[l2i[label]] == label for label in labels)


def test_encode_labels_rejects_unknown() -> None:
    labels = ["common cold", "flu"]
    assert data.encode_labels(pd.Series(["flu", "common cold"]), labels) == [1, 0]
    with pytest.raises(KeyError):
        data.encode_labels(pd.Series(["malaria"]), labels)


def test_find_duplicates_counts_without_removing() -> None:
    df = pd.DataFrame(
        {
            "text": ["a", "a", "a", "b", "b", "c"],
            "label": ["flu", "flu", "flu", "flu", "cold", "flu"],
        }
    )
    result = data.find_duplicates(df)
    assert result == {
        "duplicated_texts": 2,
        "rows_involved": 5,
        "extra_rows": 3,
        "conflicting_labels": 1,
    }
    assert len(df) == 6


def test_find_overlap_detects_leakage() -> None:
    train = pd.DataFrame({"text": ["a", "b", "c"], "label": ["flu", "cold", "flu"]})
    test = pd.DataFrame({"text": ["a", "b", "z"], "label": ["flu", "flu", "cold"]})
    assert data.find_overlap(train, test) == {
        "shared_texts": 2,
        "rows_in_second": 2,
        "rows_in_second_same_label": 1,
    }


def test_summarize_reports_counts_and_lengths(clean_train: pd.DataFrame) -> None:
    train, val = data.make_splits(clean_train, val_size=0.2, seed=0)
    test = pd.DataFrame(
        {"text": ["symptom text 0", "one two three four"], "label": ["flu", "malaria"]}
    )
    summary = data.summarize(train, val, test, full_train_df=clean_train)

    assert summary["rows"] == {"train": 16, "validation": 4, "test": 2}
    assert summary["num_labels"] == 2
    assert summary["labels_only_in_test"] == ["malaria"]
    assert summary["rows_per_label"]["train"] == {"common cold": 8, "flu": 8}
    assert summary["text_length_words"]["test"] == {"min": 3, "mean": 3.5, "max": 4}
    assert summary["overlap_official_train_vs_test"]["shared_texts"] == 1
    assert "Rows: train=16" in data.format_summary(summary)


def test_plot_label_distribution_writes_png(clean_train: pd.DataFrame, tmp_path: Path) -> None:
    train, val = data.make_splits(clean_train, val_size=0.2, seed=0)
    summary = data.summarize(train, val, clean_train.head(4))
    out = tmp_path / "dist.png"
    data.plot_label_distribution(summary, out)
    assert out.exists() and out.stat().st_size > 0
