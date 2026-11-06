"""Tests for src.safety: red flags, confidence message, disclaimer and the combined report."""

from __future__ import annotations

import re

import pytest

from src import config, predict, safety
from src.predict import Condition, PredictionResult
from src.safety import assess, check_confidence, check_red_flags

# Each category with at least two different wordings that must trigger it.
TRIGGERS: dict[str, list[str]] = {
    "chest_pain": [
        "I have chest pain when I walk upstairs",
        "There is a crushing pressure in my chest",
        "My chest feels tight and heavy",
        "Sharp pains in the chest since this morning",
    ],
    "breathing": [
        "I can't breathe properly and I'm wheezing",
        "I feel short of breath all the time",
        "I have difficulty breathing at night",
        "I cannot breathe when I lie down",
        "Shortness of breath after a few steps",
    ],
    "fainting": [
        "I passed out at work yesterday",
        "I fainted twice this week",
        "I keep blacking out when I stand up",
        "My father lost consciousness for a minute",
    ],
    "bleeding": [
        "I have severe bleeding from a cut on my leg",
        "The wound won't stop bleeding",
        "I have been vomiting blood since last night",
        "I'm coughing up blood",
    ],
    "confusion_headache": [
        "My mother seems confused and doesn't know where she is",
        "I have the worst headache of my life",
        "I got a sudden severe headache an hour ago",
        "He is disoriented and sleepy",
    ],
    "stroke": [
        "One side of my face is drooping",
        "My dad has slurred speech and arm weakness",
        "I suddenly feel weak on one side and can't lift my arm",
        "She is slurring her words",
    ],
    "seizure": [
        "My son had a seizure this morning",
        "I had convulsions after the fever",
        "She has been having seizures all day",
    ],
    "self_harm": [
        "I want to die and I can't cope anymore",
        "I have been having suicidal thoughts",
        "Sometimes I think about hurting myself",
        "I've been cutting myself again",
        "I just want to end my life",
    ],
}

# Ordinary symptom texts copied from the dataset (gretelai/symptom_to_diagnosis).
# None of them describes an emergency sign, so none should trigger.
DATASET_ORDINARY_TEXTS: list[str] = [
    "I've noticed that my skin is more sensitive now than it used to be. It's red and itchy. "
    "I have silvery scales on my back and elbows.",
    "I have a rash on my chest and back and it itches. I also have dry skin that sometimes "
    "flakes and leaves marks on my body.",
    "I have varicose veins in my legs. They are swollen and painful.",
    "I have been having headaches for a while now. They are very painful and last for a long "
    "time. I also get nausea, vomiting, and sensitivity to light and sound.",
    "I have been feeling really sick. My body hurts a lot and I have no appetite. I have also "
    "developed rashes on my arms and face. The back of my eyes hurt a lot.",
    "I've been having a lot of pain in my neck and back. I've also been having trouble with my "
    "balance and coordination. I've been coughing a lot and my limbs feel weak.",
    "I have a rash on my face that is getting worse. It is red, inflamed, and has blisters "
    "that are bleeding clear pus. It is really painful.",
    "I have been having trouble with my muscles and joints. My neck is really tight and my "
    "muscles feel weak. I have swollen joints and it is hard to move around without becoming "
    "stiff. It is also really uncomfortable to walk.",
    "I've been feeling really itchy all over, and I've been throwing up. I've also lost a lot "
    "of weight, and I'm always tired. My skin and my pee are both yellow, and my stomach hurts.",
    "I have been urinating blood. I sometimes feel sick to my stomach when I urinate. I often "
    "feel like I have a fever.",
    "I've been feeling really run down and weak. My throat is sore and I've been coughing a "
    "lot. I've also been having chills and a fever.",
    "I have a high fever and a headache. I also have a lot of pain in my belly. I can't go to "
    "work because of this.",
    "I have red spots on my arms and legs that are really itchy. I can't sleep at night "
    "because of the itching. I also have a headache and a fever.",
    "I have a rash on my skin that is itchy and has a different color than the rest of my "
    "skin. I also have some firm pimples or breakouts on my skin.",
    # Near-misses: words that look like red flags but are not.
    "I have a rash on my arms and neck that bleeds and hurts when I scratch it.",
    "They are very noticeable and make me self-conscious.",
    "I've been having periodic coughing fits, terrible back pain, and limb weakness.",
    "They are usually on one side of my head and are very painful.",
    "I have a high fever and a severe headache. My body aches and I feel chills every night.",
    "I have a chronic nagging pain in my throat and bad breath.",
]


def _keys(text: str) -> set[str]:
    return {f.key for f in check_red_flags(text)}


def _result(top: float) -> PredictionResult:
    return PredictionResult([Condition("flu", top), Condition("cold", 1 - top)],
                            low_confidence=top < config.CONFIDENCE_THRESHOLD)


# --------------------------------------------------------------------------- #
# Red flags
# --------------------------------------------------------------------------- #
def test_all_eight_categories_exist() -> None:
    assert {c.key for c in safety.RED_FLAG_CATEGORIES} == set(TRIGGERS)


@pytest.mark.parametrize("key, text", [(k, t) for k, ts in TRIGGERS.items() for t in ts])
def test_category_triggers(key: str, text: str) -> None:
    assert key in _keys(text)


@pytest.mark.parametrize("key", TRIGGERS)
def test_every_category_has_at_least_two_triggering_wordings(key: str) -> None:
    assert sum(key in _keys(t) for t in TRIGGERS[key]) >= 2


@pytest.mark.parametrize("text", DATASET_ORDINARY_TEXTS)
def test_ordinary_dataset_texts_do_not_trigger(text: str) -> None:
    assert check_red_flags(text) == []


@pytest.mark.parametrize("text", ["CHEST PAIN", "Chest Pain", "chest pain", "cHeSt PaIn"])
def test_matching_ignores_case(text: str) -> None:
    assert _keys(text) == {"chest_pain"}


@pytest.mark.parametrize("text", ["I can’t breathe", "I cant breathe", "I can not breathe"])
def test_apostrophe_and_cant_spellings(text: str) -> None:
    assert "breathing" in _keys(text)


def test_whole_phrases_only() -> None:
    # "seizure" inside another word, or "suicide" as part of a longer word, should not count
    assert check_red_flags("I am fine, just a mild rash") == []
    assert "fainting" not in _keys("I feel self-conscious about my skin")
    assert "bleeding" not in _keys("the cut bled a little")


def test_negation_is_not_detected_on_purpose() -> None:
    # A false alarm is acceptable; missing an emergency is not.
    assert "chest_pain" in _keys("I have no chest pain")


def test_multiple_categories_in_one_text() -> None:
    assert _keys("I have chest pain and I'm short of breath, then I passed out") == {
        "chest_pain", "breathing", "fainting"}


def test_red_flag_reports_matched_phrases_and_message() -> None:
    [flag] = check_red_flags("Sudden chest pain")
    assert flag.matched_phrases == ("chest pain",)
    assert "emergency" in flag.message.lower()


def test_empty_or_none_text_has_no_flags() -> None:
    assert check_red_flags("") == []
    assert check_red_flags(None) == []  # type: ignore[arg-type]


def test_short_text_still_checked() -> None:
    # Two words is too short for the model, but the safety check must still fire.
    assert _keys("chest pain") == {"chest_pain"}


# --------------------------------------------------------------------------- #
# Crisis message
# --------------------------------------------------------------------------- #
def test_crisis_message_points_to_emergency_services_and_helpline() -> None:
    [flag] = check_red_flags("I feel suicidal")
    msg = flag.message.lower()
    assert "emergency services" in msg
    assert "crisis helpline" in msg
    assert "right away" in msg
    assert config.CRISIS_HELPLINE_TEXT in flag.message


def test_crisis_message_has_no_made_up_number() -> None:
    [flag] = check_red_flags("I want to kill myself")
    assert not re.search(r"\d", flag.message)
    assert not re.search(r"\d", config.CRISIS_HELPLINE_TEXT)
    assert "[verify helpline]" in config.CRISIS_HELPLINE_TEXT


def test_no_message_contains_a_phone_number() -> None:
    for cat in safety.RED_FLAG_CATEGORIES:
        assert not re.search(r"\d", cat.message), cat.key


# --------------------------------------------------------------------------- #
# Confidence check, disclaimer, combined report
# --------------------------------------------------------------------------- #
def test_check_confidence_low() -> None:
    msg = check_confidence(_result(0.3))
    assert msg is not None and "general physician" in msg


def test_check_confidence_high_or_missing() -> None:
    assert check_confidence(_result(0.9)) is None
    assert check_confidence(None) is None


def test_disclaimer_matches_spec_exactly() -> None:
    assert safety.DISCLAIMER == ("This is an educational tool, not a medical diagnosis. "
                                 "Always consult a qualified doctor.")


def test_assess_emergency_and_low_confidence() -> None:
    report = assess("I have chest pain and feel dizzy", _result(0.3))
    assert report.is_emergency is True
    assert [f.key for f in report.red_flags] == ["chest_pain"]
    assert report.confidence_message == safety.LOW_CONFIDENCE_MESSAGE
    assert report.disclaimer == safety.DISCLAIMER


def test_assess_ordinary_and_confident() -> None:
    report = assess(DATASET_ORDINARY_TEXTS[0], _result(0.9))
    assert report.is_emergency is False
    assert report.red_flags == []
    assert report.confidence_message is None
    assert report.disclaimer == safety.DISCLAIMER


def test_assess_without_prediction() -> None:
    report = assess("I passed out")
    assert report.is_emergency is True
    assert report.confidence_message is None


# --------------------------------------------------------------------------- #
# The predict command runs the red-flag check first
# --------------------------------------------------------------------------- #
class _FakeModel:
    classes_ = ["cold", "flu"]

    def predict_proba(self, texts):  # noqa: ANN001, ANN201
        import numpy as np
        return np.array([[0.2, 0.8]] * len(texts))


def test_cli_prints_emergency_before_prediction(monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(predict, "load_model", lambda name, path=None: _FakeModel())
    assert predict.main(["I have crushing chest pain and sweating"]) == 0
    out = capsys.readouterr().out
    assert out.index("URGENT") < out.index(safety.DISCLAIMER) < out.index("flu")


def test_cli_red_flag_runs_even_if_model_missing(monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    def missing(name: str, path=None):  # noqa: ANN001, ANN202
        raise FileNotFoundError("no model")

    monkeypatch.setattr(predict, "load_model", missing)
    assert predict.main(["--model", "bert", "I want to die tonight"]) == 1
    out = capsys.readouterr().out
    assert "crisis helpline" in out.lower()


def test_cli_red_flag_runs_for_too_short_text(monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(predict, "load_model", lambda name, path=None: _FakeModel())
    assert predict.main(["seizure"]) == 1
    captured = capsys.readouterr()
    assert "URGENT (Seizures)" in captured.out
    assert "at least 3 words" in captured.err
