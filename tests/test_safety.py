"""Tests for src.safety: two-level red flags, confidence message, disclaimer and report."""

from __future__ import annotations

import re

import numpy as np
import pytest

from src import config, predict, safety
from src.predict import Condition, PredictionResult
from src.safety import EMERGENCY, NONE, URGENT, assess, check_confidence, check_red_flags

# --------------------------------------------------------------------------- #
# Examples
# --------------------------------------------------------------------------- #
EMERGENCY_EXAMPLES: dict[str, list[str]] = {
    "severe_chest_pain": [
        "I have crushing chest pain",
        "There is a crushing pressure in my chest",
        "Severe chest pain since an hour ago",
        "I have chest pain that spreads to my left arm",
        "Chest discomfort that goes up my neck and into my jaw",
        "Pain in my chest radiating to my back",
    ],
    "severe_breathing": [
        "I can't breathe properly and I'm wheezing",
        "I cannot breathe when I lie down",
        "He is gasping for air",
        "I'm struggling to breathe",
        "Her lips are turning blue",
        "I am unable to breathe",
    ],
    "fainting": [
        "I passed out at work yesterday",
        "I fainted twice this week",
        "I keep blacking out when I stand up",
        "My father lost consciousness for a minute",
        "She collapsed in the kitchen",
        "He was unconscious for a few seconds",
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
    "bleeding": [
        "I have severe bleeding from a cut on my leg",
        "The wound won't stop bleeding",
        "I have been vomiting blood since last night",
        "I'm coughing up blood",
    ],
    "sudden_headache": [
        "I have the worst headache of my life",
        "I got a sudden severe headache an hour ago",
        "A thunderclap headache came on out of nowhere",
    ],
    "self_harm": [
        "I want to die and I can't cope anymore",
        "I have been having suicidal thoughts",
        "Sometimes I think about hurting myself",
        "I've been cutting myself again",
        "I just want to end my life",
    ],
}

URGENT_EXAMPLES: dict[str, list[str]] = {
    "chest_pain": [
        "I have chest pain when I walk upstairs",
        "My chest feels tight and heavy",
        "Sharp pains in the chest since this morning",
        "I have some chest discomfort",
    ],
    "breathing": [
        "I feel short of breath all the time",
        "I have difficulty breathing at night",
        "Shortness of breath after a few steps",
        "I get out of breath easily",
        "I'm having trouble breathing",
        "I feel like I'm choking when I eat",
    ],
    "near_fainting": [
        "I feel faint when I stand up",
        "I feel like I'm going to pass out",
        "I was feeling faint all morning",
    ],
    "confusion": [
        "My mother seems confused and doesn't know where she is",
        "He is disoriented and sleepy",
        "I've had some confusion today",
    ],
}

# Every example from the previous (single-level) test, with its new level.
PREVIOUS_EXAMPLES: list[tuple[str, str]] = [
    ("I have chest pain when I walk upstairs", URGENT),
    ("There is a crushing pressure in my chest", EMERGENCY),
    ("My chest feels tight and heavy", URGENT),
    ("Sharp pains in the chest since this morning", URGENT),
    ("I can't breathe properly and I'm wheezing", EMERGENCY),
    ("I feel short of breath all the time", URGENT),
    ("I have difficulty breathing at night", URGENT),
    ("I cannot breathe when I lie down", EMERGENCY),
    ("Shortness of breath after a few steps", URGENT),
    ("I passed out at work yesterday", EMERGENCY),
    ("I fainted twice this week", EMERGENCY),
    ("I keep blacking out when I stand up", EMERGENCY),
    ("My father lost consciousness for a minute", EMERGENCY),
    ("I have severe bleeding from a cut on my leg", EMERGENCY),
    ("The wound won't stop bleeding", EMERGENCY),
    ("I have been vomiting blood since last night", EMERGENCY),
    ("I'm coughing up blood", EMERGENCY),
    ("My mother seems confused and doesn't know where she is", URGENT),
    ("I have the worst headache of my life", EMERGENCY),
    ("I got a sudden severe headache an hour ago", EMERGENCY),
    ("He is disoriented and sleepy", URGENT),
    ("One side of my face is drooping", EMERGENCY),
    ("My dad has slurred speech and arm weakness", EMERGENCY),
    ("I suddenly feel weak on one side and can't lift my arm", EMERGENCY),
    ("She is slurring her words", EMERGENCY),
    ("My son had a seizure this morning", EMERGENCY),
    ("I had convulsions after the fever", EMERGENCY),
    ("She has been having seizures all day", EMERGENCY),
    ("I want to die and I can't cope anymore", EMERGENCY),
    ("I have been having suicidal thoughts", EMERGENCY),
    ("Sometimes I think about hurting myself", EMERGENCY),
    ("I've been cutting myself again", EMERGENCY),
    ("I just want to end my life", EMERGENCY),
]

# Ordinary symptom texts copied from the dataset (gretelai/symptom_to_diagnosis).
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


def _flags(text: str) -> dict[str, str]:
    """Map matched category key -> level."""
    return {f.key: f.level for f in check_red_flags(text)}


def _result(top: float) -> PredictionResult:
    return PredictionResult([Condition("flu", top), Condition("cold", 1 - top)],
                            low_confidence=top < config.CONFIDENCE_THRESHOLD)


# --------------------------------------------------------------------------- #
# Categories and levels
# --------------------------------------------------------------------------- #
def test_category_levels() -> None:
    levels = {c.key: c.level for c in safety.RED_FLAG_CATEGORIES}
    assert {k for k, v in levels.items() if v == EMERGENCY} == set(EMERGENCY_EXAMPLES)
    assert {k for k, v in levels.items() if v == URGENT} == set(URGENT_EXAMPLES)


@pytest.mark.parametrize("key, text",
                         [(k, t) for k, ts in EMERGENCY_EXAMPLES.items() for t in ts])
def test_emergency_examples(key: str, text: str) -> None:
    report = assess(text)
    assert report.level == EMERGENCY
    assert report.is_emergency
    assert _flags(text).get(key) == EMERGENCY


@pytest.mark.parametrize("key, text", [(k, t) for k, ts in URGENT_EXAMPLES.items() for t in ts])
def test_urgent_examples(key: str, text: str) -> None:
    report = assess(text)
    assert report.level == URGENT
    assert not report.is_emergency
    assert _flags(text) == {key: URGENT}


@pytest.mark.parametrize("key", [*EMERGENCY_EXAMPLES, *URGENT_EXAMPLES])
def test_every_category_has_at_least_two_wordings(key: str) -> None:
    examples = {**EMERGENCY_EXAMPLES, **URGENT_EXAMPLES}[key]
    assert sum(key in _flags(t) for t in examples) >= 2


@pytest.mark.parametrize("text, level", PREVIOUS_EXAMPLES)
def test_previous_examples_keep_triggering(text: str, level: str) -> None:
    assert assess(text).level == level


@pytest.mark.parametrize("text", DATASET_ORDINARY_TEXTS)
def test_ordinary_dataset_texts_do_not_trigger(text: str) -> None:
    assert check_red_flags(text) == []
    assert assess(text).level == NONE


# --------------------------------------------------------------------------- #
# Severe vs ordinary chest pain
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "chest pain that gets worse when I lie down",
    "I have a burning sensation in my chest",  # heartburn wording: not a red flag at all
])
def test_plain_or_heartburn_chest_is_not_emergency(text: str) -> None:
    assert assess(text).level in (URGENT, NONE)


def test_severity_must_be_in_the_same_sentence() -> None:
    assert assess("I have chest pain. My headache is severe.").level == URGENT
    assert assess("I have chest pain and it is severe.").level == EMERGENCY


def test_go_back_is_not_spreading() -> None:
    assert assess("I have chest pain when I go back to work").level == URGENT


# --------------------------------------------------------------------------- #
# Emergency replaces the milder warning for the same sign
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, kept, dropped", [
    ("I can't breathe and I'm short of breath", "severe_breathing", "breathing"),
    ("Crushing chest pain and my chest feels tight", "severe_chest_pain", "chest_pain"),
    ("I felt faint and then passed out", "fainting", "near_fainting"),
])
def test_emergency_replaces_urgent_for_same_sign(text: str, kept: str, dropped: str) -> None:
    flags = _flags(text)
    assert flags.get(kept) == EMERGENCY
    assert dropped not in flags


def test_mixed_levels_emergency_listed_first() -> None:
    flags = check_red_flags("I'm confused and I had a seizure")
    assert [f.key for f in flags] == ["seizure", "confusion"]
    assert assess("I'm confused and I had a seizure").level == EMERGENCY


# --------------------------------------------------------------------------- #
# Matching rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["CHEST PAIN", "Chest Pain", "chest pain", "cHeSt PaIn"])
def test_matching_ignores_case(text: str) -> None:
    assert _flags(text) == {"chest_pain": URGENT}


@pytest.mark.parametrize("text", ["CRUSHING CHEST PAIN", "Crushing Chest Pain"])
def test_emergency_matching_ignores_case(text: str) -> None:
    assert assess(text).level == EMERGENCY


@pytest.mark.parametrize("text", ["I can’t breathe", "I cant breathe", "I can not breathe"])
def test_apostrophe_and_cant_spellings(text: str) -> None:
    assert _flags(text) == {"severe_breathing": EMERGENCY}


def test_whole_phrases_only() -> None:
    assert check_red_flags("I am fine, just a mild rash") == []
    assert check_red_flags("I feel self-conscious about my skin") == []
    assert check_red_flags("the cut bled a little") == []


def test_negation_is_not_detected_on_purpose() -> None:
    # A false alarm is acceptable; missing an emergency is not.
    assert assess("I have no chest pain").level == URGENT
    assert assess("I have not passed out").level == EMERGENCY


def test_red_flag_reports_matched_phrases_and_message() -> None:
    [flag] = check_red_flags("Sudden chest pain")
    assert flag.matched_phrases == ("chest pain",)
    assert flag.message == safety.URGENT_MESSAGE


def test_empty_or_none_text_has_no_flags() -> None:
    assert check_red_flags("") == []
    assert check_red_flags(None) == []  # type: ignore[arg-type]
    assert assess("").level == NONE


def test_short_text_still_checked() -> None:
    assert assess("seizure").level == EMERGENCY
    assert assess("chest pain").level == URGENT


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
def test_emergency_and_urgent_messages() -> None:
    assert "seek emergency care now" in safety.EMERGENCY_MESSAGE.lower()
    assert "see a doctor today, or sooner if it gets worse" in safety.URGENT_MESSAGE.lower()


def test_crisis_message_points_to_emergency_services_and_helpline() -> None:
    [flag] = check_red_flags("I feel suicidal")
    assert flag.level == EMERGENCY
    msg = flag.message.lower()
    assert "emergency services" in msg
    assert "crisis helpline" in msg
    assert "right away" in msg
    assert config.CRISIS_HELPLINE_TEXT in flag.message


def _without_helpline(text: str) -> str:
    """Remove the owner-verified helpline text, the only place digits are allowed."""
    return text.replace(config.CRISIS_HELPLINE_TEXT, "")


def test_crisis_message_digits_only_from_helpline() -> None:
    [flag] = check_red_flags("I want to kill myself")
    assert config.CRISIS_HELPLINE_TEXT.strip()
    assert flag.message.endswith(config.CRISIS_HELPLINE_TEXT)
    assert not re.search(r"\d", _without_helpline(flag.message))


def test_no_message_contains_digits_except_helpline() -> None:
    for cat in safety.RED_FLAG_CATEGORIES:
        assert not re.search(r"\d", _without_helpline(cat.message)), cat.key
    for msg in (safety.EMERGENCY_MESSAGE, safety.URGENT_MESSAGE,
                safety.LOW_CONFIDENCE_MESSAGE, safety.DISCLAIMER):
        assert not re.search(r"\d", msg)
    assert not re.search(r"\d", _without_helpline(safety.CRISIS_MESSAGE))


def _all_safety_messages() -> list[str]:
    """Every user-facing text in src.safety: module-level strings and category messages."""
    texts = [v for k, v in vars(safety).items() if k.isupper() and isinstance(v, str)]
    return texts + [cat.message for cat in safety.RED_FLAG_CATEGORIES]


def test_no_safety_message_contains_gp() -> None:
    messages = _all_safety_messages()
    assert safety.LOW_CONFIDENCE_MESSAGE in messages
    assert safety.WARNING_CONTEXT_NOTE in messages
    for msg in messages:
        assert not re.search(r"\bGP\b", msg), msg


def test_low_confidence_message_says_general_physician() -> None:
    assert "general physician for a proper assessment" in safety.LOW_CONFIDENCE_MESSAGE


def test_warning_context_note_text_and_no_digits() -> None:
    assert safety.WARNING_CONTEXT_NOTE == (
        "Your description includes warning signs. The risk labels below describe each "
        "condition in general, not your situation. Follow the warning above.")
    assert not re.search(r"\d", safety.WARNING_CONTEXT_NOTE)


def test_digit_check_would_catch_a_number_outside_the_helpline() -> None:
    bad = safety.EMERGENCY_MESSAGE + " Call 999."
    assert re.search(r"\d", _without_helpline(bad))


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
    report = assess("Crushing chest pain and I feel dizzy", _result(0.3))
    assert report.level == EMERGENCY
    assert [f.key for f in report.red_flags] == ["severe_chest_pain"]
    assert report.confidence_message == safety.LOW_CONFIDENCE_MESSAGE
    assert report.disclaimer == safety.DISCLAIMER


def test_assess_urgent_and_confident() -> None:
    report = assess("I have chest pain and feel dizzy", _result(0.9))
    assert report.level == URGENT
    assert [f.key for f in report.red_flags] == ["chest_pain"]
    assert report.confidence_message is None


def test_assess_ordinary_and_confident() -> None:
    report = assess(DATASET_ORDINARY_TEXTS[0], _result(0.9))
    assert report.level == NONE
    assert report.red_flags == []
    assert report.confidence_message is None


# --------------------------------------------------------------------------- #
# The predict command runs the red-flag check first
# --------------------------------------------------------------------------- #
class _FakeModel:
    classes_ = ["cold", "flu"]

    def predict_proba(self, texts):  # noqa: ANN001, ANN201
        return np.array([[0.2, 0.8]] * len(texts))


def test_cli_prints_emergency_then_urgent_before_prediction(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(predict, "load_model", lambda name, path=None: _FakeModel())
    assert predict.main(["I feel confused and I have crushing chest pain"]) == 0
    out = capsys.readouterr().out
    assert "!!! EMERGENCY (Severe or spreading chest pain)" in out
    assert "! URGENT (Confusion or disorientation)" in out
    assert (out.index("!!! EMERGENCY") < out.index("! URGENT")
            < out.index(safety.DISCLAIMER) < out.index("flu"))


def test_cli_urgent_only(monkeypatch: pytest.MonkeyPatch,
                         capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(predict, "load_model", lambda name, path=None: _FakeModel())
    assert predict.main(["I get short of breath when I climb stairs"]) == 0
    out = capsys.readouterr().out
    assert "! URGENT (Shortness of breath" in out
    assert "EMERGENCY" not in out


def test_cli_red_flag_runs_even_if_model_missing(monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    def missing(name: str, path=None):  # noqa: ANN001, ANN202
        raise FileNotFoundError("no model")

    monkeypatch.setattr(predict, "load_model", missing)
    assert predict.main(["--model", "bert", "I want to die tonight"]) == 1
    out = capsys.readouterr().out
    assert "!!! EMERGENCY (Suicidal thoughts or self-harm)" in out
    assert "crisis helpline" in out.lower()


def test_cli_red_flag_runs_for_too_short_text(monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(predict, "load_model", lambda name, path=None: _FakeModel())
    assert predict.main(["seizure"]) == 1
    captured = capsys.readouterr()
    assert "!!! EMERGENCY (Seizures)" in captured.out
    assert "at least 3 words" in captured.err
