"""Tests for src.display: safe highlighting and reading saved test scores."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.display import highlight_html, load_test_scores, scores_table_markdown


# --------------------------------------------------------------------------- #
# highlight_html
# --------------------------------------------------------------------------- #
def test_escapes_html_and_attributes() -> None:
    out = highlight_html('<i x="1">fever</i>', {"fever": 1.0})
    assert "<i" not in out
    assert "&lt;i x=&quot;1&quot;&gt;" in out
    assert ">fever</span>" in out


@pytest.mark.parametrize("evil", [
    "<script>alert(1)</script> fever",
    '<img src=x onerror="alert(1)"> fever',
    "fever <a href='javascript:alert(1)'>x</a>",
])
def test_no_user_tags_survive(evil: str) -> None:
    out = highlight_html(evil, {"fever": 0.5, "script": 0.5, "img": 0.5})
    body = out.replace('<span style="', "").replace("</span>", "").replace(
        '<div style="line-height: 2; white-space: pre-wrap;">', "").replace("</div>", "")
    assert "<" not in body.replace("&lt;", "")


def test_highlighted_word_inside_html_is_still_escaped() -> None:
    out = highlight_html("<script>", {"script": 1.0})
    assert "<script>" not in out
    assert "&lt;" in out and ">script</span>" in out


def test_ampersand_and_quotes_escaped() -> None:
    out = highlight_html('chills & "sweating"', {})
    assert "&amp;" in out and "&quot;sweating&quot;" in out


def test_positive_green_negative_red() -> None:
    out = highlight_html("fever and sneezing", {"fever": 1.0, "sneezing": -0.5})
    assert 'rgba(34,160,70,0.80)' in out   # strongest word: full colour, green
    assert 'rgba(220,50,50,0.53)' in out   # half as strong: lighter, red
    assert 'title="+1.0000">fever</span>' in out
    assert 'title="-0.5000">sneezing</span>' in out
    assert "and" in out and ">and</span>" not in out


def test_matching_ignores_case_and_punctuation() -> None:
    out = highlight_html("FEVER, chills!", {"fever": 1.0, "chills": 0.5})
    assert ">FEVER</span>," in out
    assert ">chills</span>!" in out


def test_keeps_spacing_and_newlines() -> None:
    out = highlight_html("fever\n\n  chills", {})
    assert "fever\n\n  chills" in out
    assert "white-space: pre-wrap" in out


def test_no_scores_means_no_highlight() -> None:
    out = highlight_html("I feel fine", {})
    assert "<span" not in out and "I feel fine" in out


# --------------------------------------------------------------------------- #
# load_test_scores
# --------------------------------------------------------------------------- #
def _write(dir_: Path, name: str, data: object) -> None:
    (dir_ / name).write_text(json.dumps(data), encoding="utf-8")


def test_reads_test_section_only(tmp_path: Path) -> None:
    _write(tmp_path, "baseline_metrics.json",
           {"validation": {"accuracy": 0.5}, "test": {"accuracy": 0.9, "macro_f1": 0.8,
                                                      "n_samples": 10}})
    scores = load_test_scores(tmp_path)
    assert list(scores) == ["Baseline (TF-IDF + Logistic Regression)"]
    assert scores["Baseline (TF-IDF + Logistic Regression)"]["accuracy"] == 0.9


def test_skips_missing_broken_or_untested_files(tmp_path: Path) -> None:
    (tmp_path / "baseline_metrics.json").write_text("{not json", encoding="utf-8")
    _write(tmp_path, "bert_metrics.json", {"validation": {"accuracy": 0.9}})
    assert load_test_scores(tmp_path) == {}
    assert load_test_scores(tmp_path / "missing_dir") == {}


def test_real_reports_match_files() -> None:
    from src.config import REPORTS_DIR

    scores = load_test_scores()
    for name, file in (("Baseline (TF-IDF + Logistic Regression)", "baseline_metrics.json"),
                       ("DistilBERT", "bert_metrics.json")):
        path = REPORTS_DIR / file
        if path.exists():
            expected = json.loads(path.read_text(encoding="utf-8"))["test"]
            assert scores[name]["accuracy"] == expected["accuracy"]


def test_scores_table_markdown() -> None:
    table = scores_table_markdown({"M": {"accuracy": 0.9387, "macro_f1": 0.9392,
                                         "n_samples": 212},
                                   "Broken": {"n_samples": 5}})
    assert "| M | 93.9% | 0.939 | 212 |" in table
    assert "| Broken | n/a | n/a | 5 |" in table
    assert table.startswith("| Model | Accuracy | Macro-F1 | Test rows |")
