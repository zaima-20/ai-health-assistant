"""UI tests for app.py with Streamlit's AppTest and fake predictors (no real models)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import predict
from src.config import CRISIS_HELPLINE_TEXT
from src.explain import normalize_word
from src.safety import DISCLAIMER, WARNING_CONTEXT_NOTE

APP = str(Path(__file__).resolve().parent.parent / "app.py")
LABELS = ["common cold", "dengue", "malaria", "typhoid"]
ORDINARY = "I have a high fever, chills and sweating every few days"
CONDITION_HEADINGS = ("1. Malaria", "2. Dengue", "3. Typhoid")


class FakeModel:
    """Malaria is likely when 'chills'/'sweating' appear; 'sneezing' pushes towards a cold."""

    classes_ = np.array(LABELS)
    WEIGHTS = {"chills": [0, 0.5, 2.0, 0.3], "sweating": [0, 0.5, 1.5, 0.3],
               "fever": [0, 1.0, 1.0, 0.8], "sneezing": [2.5, 0, -1, 0]}

    def __init__(self, flat: bool = False) -> None:
        self.flat = flat

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for t in texts:
            logits = np.array([0.0, 0.2, 0.0, 0.1])
            if not self.flat:
                for w in t.split():
                    logits += self.WEIGHTS.get(normalize_word(w), 0.0)
            e = np.exp(logits - logits.max())
            out.append(e / e.sum())
        return np.array(out)


@pytest.fixture(autouse=True)
def fake_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every test uses the fake model and a fresh model cache."""
    monkeypatch.setattr(predict, "load_model", lambda name, path=None: FakeModel())
    bert_dir = tmp_path / "bert"
    bert_dir.mkdir()
    (bert_dir / "config.json").write_text("{}")
    monkeypatch.setitem(predict.DEFAULT_PATHS, "bert", bert_dir)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def run_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def analyse(text: str) -> AppTest:
    at = run_app()
    at.text_area(key="symptoms").input(text)
    next(b for b in at.button if b.label == "Analyse").click()
    at.run()
    assert not at.exception, at.exception
    return at


def markdown_values(block) -> list[str]:  # noqa: ANN001
    return [m.value for m in block.markdown]


def has_conditions(block) -> bool:  # noqa: ANN001
    values = markdown_values(block)
    return all(any(h in v for v in values) for h in CONDITION_HEADINGS)


def expander(at: AppTest, label: str):  # noqa: ANN201
    return next((e for e in at.expander if e.label == label), None)


# --------------------------------------------------------------------------- #
# Disclaimer and layout
# --------------------------------------------------------------------------- #
def test_disclaimer_shows_on_first_load() -> None:
    at = run_app()
    assert at.title[0].value == "AI Health Assistant"
    assert any(DISCLAIMER in w.value for w in at.warning)


@pytest.mark.parametrize("text", [ORDINARY, "I have crushing chest pain", "fever"])
def test_disclaimer_shows_after_every_analysis(text: str) -> None:
    at = analyse(text)
    assert any(DISCLAIMER in w.value for w in at.warning)


def test_model_choice_defaults_to_baseline() -> None:
    at = run_app()
    radio = at.sidebar.radio[0]
    assert radio.options == ["Baseline (fast)", "DistilBERT"]
    assert radio.value == "Baseline (fast)"


def test_missing_bert_offers_only_baseline(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: Path) -> None:
    monkeypatch.setitem(predict.DEFAULT_PATHS, "bert", tmp_path / "nothing_here")
    at = run_app()
    assert at.sidebar.radio[0].options == ["Baseline (fast)"]
    assert any("not available" in i.value for i in at.sidebar.info)


def test_example_button_fills_text_area() -> None:
    at = run_app()
    next(b for b in at.button if b.label == "Fever and chills").click()
    at.run()
    assert "high fever, chills" in at.text_area(key="symptoms").value


def test_text_area_limit() -> None:
    at = run_app()
    assert at.text_area(key="symptoms").proto.max_chars == 1000


def test_about_shows_real_test_scores() -> None:
    import json

    from src.config import REPORTS_DIR

    at = run_app()
    about = expander(at, "About this app")
    assert about is not None
    text = "\n".join(markdown_values(about))
    base = json.loads((REPORTS_DIR / "baseline_metrics.json").read_text())["test"]
    assert f"{base['accuracy']:.1%}" in text
    assert "rewritten by a language model" in text and "wheeze" in text


# --------------------------------------------------------------------------- #
# Ordinary input
# --------------------------------------------------------------------------- #
def test_ordinary_input_shows_conditions_without_banners() -> None:
    at = analyse(ORDINARY)
    assert not at.error
    assert not [w for w in at.warning if DISCLAIMER not in w.value]
    assert has_conditions(at)
    assert expander(at, "Show possible conditions anyway") is None
    assert len(at.get("progress")) == 3


def test_condition_card_shows_disease_info() -> None:
    at = analyse(ORDINARY)
    text = "\n".join(markdown_values(at))
    assert "Who to see:" in text
    assert "https://www.nhs.uk/conditions/malaria/" in text


@pytest.mark.parametrize("reviewed, note_shown", [(False, True), (True, False)])
def test_unreviewed_note_follows_reviewed_flag(monkeypatch: pytest.MonkeyPatch,
                                               reviewed: bool, note_shown: bool) -> None:
    import dataclasses

    from src import disease_info

    real = disease_info.get_disease_info
    monkeypatch.setattr(disease_info, "get_disease_info",
                        lambda label: dataclasses.replace(real(label), reviewed=reviewed))
    at = analyse(ORDINARY)
    assert any("not yet been reviewed" in c.value for c in at.caption) is note_shown


def test_warning_sign_headings_follow_source_levels() -> None:
    # Top 3 here: malaria (urgent only), dengue (emergency only), typhoid (emergency + urgent).
    at = analyse(ORDINARY)
    values = markdown_values(at)
    assert values.count("**Get emergency help now if**") == 2   # dengue, typhoid
    assert values.count("**See a doctor urgently if**") == 2    # malaria, typhoid
    assert "**See a doctor if**" not in values
    joined = "\n".join(values)
    assert "- Vomiting blood, or vomit that looks like ground coffee" in joined  # typhoid
    emergency_blocks = [values[i + 1] for i, v in enumerate(values)
                        if v == "**Get emergency help now if**"]
    typhoid_block = next(b for b in emergency_blocks if "Black or dark red stool" in b)
    assert "Feeling unwell after traveling" not in typhoid_block


def test_routine_signs_heading(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    from src import disease_info

    real = disease_info.get_disease_info

    def with_routine(label: str):  # noqa: ANN202
        info = real(label)
        if label != "malaria":
            return info
        return dataclasses.replace(info, warning_level="routine", emergency_signs=(),
                                   urgent_signs=(), routine_signs=("Sign A", "Sign B"))

    monkeypatch.setattr(disease_info, "get_disease_info", with_routine)
    values = markdown_values(analyse(ORDINARY))
    i = values.index("**See a doctor if**")
    assert values[i + 1] == "- Sign A\n- Sign B"


def test_low_confidence_shows_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predict, "load_model", lambda name, path=None: FakeModel(flat=True))
    at = analyse("I feel a bit unwell today")
    assert any("general physician" in i.value.lower() for i in at.info)


def test_short_text_shows_friendly_message() -> None:
    at = analyse("fever")
    assert any("at least 3 words" in w.value for w in at.warning)


# --------------------------------------------------------------------------- #
# Emergency
# --------------------------------------------------------------------------- #
def test_emergency_banner_hides_predictions() -> None:
    at = analyse("I have crushing chest pain and chills and sweating")
    assert any("Seek emergency care now" in e.value for e in at.error)
    hidden = expander(at, "Show possible conditions anyway")
    assert hidden is not None
    assert hidden.proto.expanded is False
    assert has_conditions(hidden)
    # every condition heading on the page is inside the collapsed expander
    outside = [v for v in markdown_values(at) if v not in markdown_values(hidden)]
    assert not any(h in v for v in outside for h in CONDITION_HEADINGS)


def test_emergency_banner_is_above_the_disclaimer_and_input() -> None:
    at = analyse("I passed out at work and have chills")
    main = list(at.main)
    types = [getattr(node, "type", "") for node in main]
    error_pos = next(i for i, n in enumerate(main) if any(
        "emergency" in getattr(c, "value", "").lower() for c in getattr(n, "error", [])))
    assert error_pos < types.index("text_area")


def test_self_harm_shows_helpline_prominently() -> None:
    at = analyse("I want to end my life and I feel tired")
    errors = [e.value for e in at.error]
    assert any("crisis helpline" in e.lower() for e in errors)
    assert any(e.startswith("###") and CRISIS_HELPLINE_TEXT in e for e in errors)


def test_emergency_banner_even_when_text_too_short_for_model() -> None:
    at = analyse("seizure")
    assert any("Seizures" in e.value for e in at.error)
    assert any("at least 3 words" in w.value for w in at.warning)


# --------------------------------------------------------------------------- #
# Urgent
# --------------------------------------------------------------------------- #
def test_urgent_shows_warning_and_predictions() -> None:
    at = analyse("I feel short of breath with a fever and chills")
    assert not at.error
    assert any("See a doctor today" in w.value for w in at.warning)
    assert expander(at, "Show possible conditions anyway") is None
    assert has_conditions(at)


# --------------------------------------------------------------------------- #
# Note that risk labels are general when warning signs are present
# --------------------------------------------------------------------------- #
def _note_infos(block) -> list:  # noqa: ANN001
    return [i for i in block.info if i.value == WARNING_CONTEXT_NOTE]


def test_urgent_shows_risk_label_note_above_cards() -> None:
    at = analyse("I feel short of breath with a fever and chills")
    assert len(_note_infos(at)) == 1
    def walk(node):  # noqa: ANN001, ANN202 - every element in on-screen order
        children = getattr(node, "children", None)
        if children is None:
            yield node
        else:
            for child in children.values():
                yield from walk(child)

    values = [str(getattr(n, "value", "")) for n in walk(at.main)]
    note_pos = values.index(WARNING_CONTEXT_NOTE)
    card_pos = next(i for i, v in enumerate(values) if v.startswith("#### 1."))
    assert note_pos < card_pos


def test_emergency_shows_risk_label_note_inside_expander() -> None:
    at = analyse("I have crushing chest pain and chills and sweating")
    hidden = expander(at, "Show possible conditions anyway")
    assert len(_note_infos(hidden)) == 1


def test_no_risk_label_note_without_warning_signs() -> None:
    at = analyse(ORDINARY)
    assert _note_infos(at) == []


# --------------------------------------------------------------------------- #
# Explanation and HTML escaping
# --------------------------------------------------------------------------- #
def test_explanation_highlights_words() -> None:
    at = analyse(ORDINARY)
    html_blocks = [v for v in markdown_values(at) if "<span" in v]
    assert html_blocks
    assert "rgba(34,160,70" in html_blocks[0]  # green: pushes towards malaria
    assert ">chills</span>" in html_blocks[0]


def test_html_in_input_is_escaped() -> None:
    evil = ('<script>alert("x")</script><img src=x onerror=alert(1)> '
            "I have chills & sweating <b>badly</b>")
    at = analyse(evil)
    shown = [v for v in markdown_values(at) if "white-space: pre-wrap" in v]
    assert shown, "explanation block not rendered"
    block = shown[0]
    assert "<script>" not in block and "<img" not in block and "<b>" not in block
    assert "&lt;script&gt;" in block and "&lt;img" in block and "&lt;b&gt;" in block
    assert "&amp;" in block
    assert ">chills</span>" in block  # highlighting still works


# --------------------------------------------------------------------------- #
# Results stay on screen, and are cleared when the text or model changes
# --------------------------------------------------------------------------- #
def test_results_stay_after_an_unrelated_rerun() -> None:
    at = analyse(ORDINARY)
    at.run()  # e.g. another click elsewhere; Analyse is not pressed this time
    assert has_conditions(at)
    assert any("<span" in v for v in markdown_values(at))


def test_emergency_banner_stays_after_rerun() -> None:
    at = analyse("I have crushing chest pain and chills")
    at.run()
    assert any("Seek emergency care now" in e.value for e in at.error)
    assert expander(at, "Show possible conditions anyway") is not None


def test_results_cleared_when_text_changes() -> None:
    at = analyse(ORDINARY)
    at.text_area(key="symptoms").input("Now I have a different problem entirely").run()
    assert "last_result" not in at.session_state
    assert not any(h in v for v in markdown_values(at) for h in CONDITION_HEADINGS)


def test_results_cleared_when_model_changes() -> None:
    at = analyse(ORDINARY)
    at.sidebar.radio[0].set_value("DistilBERT").run()
    assert "last_result" not in at.session_state
    assert not any(h in v for v in markdown_values(at) for h in CONDITION_HEADINGS)


def test_results_cleared_when_example_chosen() -> None:
    at = analyse("I have crushing chest pain and chills")
    next(b for b in at.button if b.label == "Itchy skin rash").click().run()
    assert "last_result" not in at.session_state
    assert not at.error


def test_stored_result_does_not_contain_the_text() -> None:
    marker = "zebramarker"
    at = analyse(f"I have chills and sweating {marker} every night")
    stored = at.session_state["last_result"]
    assert not hasattr(stored, "text")
    assert marker not in repr(stored)
    assert stored.prediction is not None and stored.words


# --------------------------------------------------------------------------- #
# Privacy
# --------------------------------------------------------------------------- #
def test_user_text_is_never_printed(capsys: pytest.CaptureFixture[str]) -> None:
    secret = "zebra-unicorn-marker with chills and sweating"
    analyse(secret)
    captured = capsys.readouterr()
    assert "zebra-unicorn-marker" not in captured.out
    assert "zebra-unicorn-marker" not in captured.err


@pytest.mark.parametrize("path", [APP, str(Path(APP).parent / "src" / "display.py")])
def test_source_has_no_logging_or_file_writes(path: str) -> None:
    source = Path(path).read_text(encoding="utf-8")
    for forbidden in ("print(", "logging", "open(", "write_text", "write_bytes", "to_csv",
                      "json.dump", "pickle", "joblib", "st.cache_data"):
        assert forbidden not in source, forbidden
