"""Safety checks: red-flag detection and low-confidence handling.

The red-flag check is a plain phrase list, run on the raw text BEFORE any
model. It deliberately does not try to detect negation ("no chest pain" still
triggers): a false alarm is better than missing a real emergency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.config import CRISIS_HELPLINE_TEXT

if TYPE_CHECKING:  # avoid a circular import at runtime
    from src.predict import PredictionResult

DISCLAIMER = ("This is an educational tool, not a medical diagnosis. "
              "Always consult a qualified doctor.")

EMERGENCY_MESSAGE = ("Your description mentions a possible emergency sign. "
                     "Seek emergency care now: call your local emergency number "
                     "or go to the nearest emergency department.")

LOW_CONFIDENCE_MESSAGE = ("The model is not confident about this prediction. "
                          "Please see a general physician (GP) for a proper assessment.")

CRISIS_MESSAGE = ("If you are thinking about suicide or harming yourself, please contact "
                  "your local emergency services or a crisis helpline right away. "
                  "You do not have to face this alone. " + CRISIS_HELPLINE_TEXT)


@dataclass(frozen=True)
class RedFlagCategory:
    """One kind of emergency sign: its phrases and the message to show."""

    key: str
    name: str
    message: str
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class RedFlag:
    """A category that matched the text, with the phrases that matched."""

    key: str
    name: str
    message: str
    matched_phrases: tuple[str, ...]


@dataclass(frozen=True)
class SafetyReport:
    """Everything the UI needs to show around a prediction."""

    is_emergency: bool
    red_flags: list[RedFlag] = field(default_factory=list)
    confidence_message: str | None = None
    disclaimer: str = DISCLAIMER


# --------------------------------------------------------------------------- #
# Phrase lists (lowercase; "can't" also matches "cant", "cannot", "can not";
# a trailing "s" is allowed on every phrase, e.g. "chest pain" -> "chest pains")
# --------------------------------------------------------------------------- #
RED_FLAG_CATEGORIES: tuple[RedFlagCategory, ...] = (
    RedFlagCategory(
        "chest_pain", "Chest pain or pressure", EMERGENCY_MESSAGE,
        ("chest pain", "pain in my chest", "pain in the chest", "pains in my chest",
         "pains in the chest", "chest hurt",
         "chest is hurting", "hurts in my chest", "chest pressure", "pressure in my chest",
         "pressure in the chest", "chest tightness", "tight chest", "tightness in my chest",
         "chest feels tight", "chest is tight", "chest feels heavy", "heavy chest",
         "crushing chest", "chest discomfort", "discomfort in my chest", "squeezing in my chest"),
    ),
    RedFlagCategory(
        "breathing", "Difficulty breathing", EMERGENCY_MESSAGE,
        ("difficulty breathing", "difficulty in breathing", "trouble breathing",
         "problems breathing", "breathing problem", "hard to breathe", "hard time breathing",
         "struggling to breathe", "struggle to breathe", "can't breathe", "unable to breathe",
         "short of breath", "shortness of breath", "out of breath", "breathless",
         "can't catch my breath", "can't get enough air", "gasping for air",
         "breathing is getting harder", "choking"),
    ),
    RedFlagCategory(
        "fainting", "Fainting or loss of consciousness", EMERGENCY_MESSAGE,
        ("fainted", "fainting", "feel faint", "feeling faint", "passed out", "pass out",
         "passing out", "blacked out", "black out", "blacking out", "lost consciousness",
         "loss of consciousness", "losing consciousness", "unconscious", "collapsed"),
    ),
    RedFlagCategory(
        "bleeding", "Severe bleeding", EMERGENCY_MESSAGE,
        ("severe bleeding", "heavy bleeding", "bleeding heavily", "bleeding a lot",
         "bleeding badly", "won't stop bleeding", "bleeding won't stop", "bleeding that won't stop",
         "can't stop the bleeding", "uncontrolled bleeding", "losing a lot of blood",
         "vomiting blood", "throwing up blood", "bloody vomit", "blood in my vomit",
         "coughing up blood", "cough up blood"),
    ),
    RedFlagCategory(
        "confusion_headache", "Confusion or sudden severe headache", EMERGENCY_MESSAGE,
        ("confused", "confusion", "disoriented", "disorientation",
         "sudden severe headache", "sudden and severe headache", "severe sudden headache",
         "sudden intense headache", "worst headache", "thunderclap headache"),
    ),
    RedFlagCategory(
        "stroke", "Stroke signs (face drooping, arm weakness, slurred speech)",
        EMERGENCY_MESSAGE + " Stroke signs need help immediately: note the time they started.",
        ("face drooping", "face is drooping", "drooping face", "face droop", "facial droop",
         "droopy face", "side of my face is drooping", "arm weakness", "weakness in one arm",
         "one arm is weak", "weak on one side", "weakness on one side",
         "numbness on one side", "one side of my body", "can't lift my arm",
         "slurred speech", "speech is slurred", "slurring", "can't speak", "trouble speaking",
         "difficulty speaking"),
    ),
    RedFlagCategory(
        "seizure", "Seizures", EMERGENCY_MESSAGE,
        ("seizure", "convulsion", "convulsing", "having a fit", "epileptic fit"),
    ),
    RedFlagCategory(
        "self_harm", "Suicidal thoughts or self-harm", CRISIS_MESSAGE,
        ("suicidal", "suicide", "kill myself", "killing myself", "want to die",
         "wanna die", "wish i was dead", "wish i were dead", "better off dead",
         "end my life", "ending my life", "take my own life", "no reason to live",
         "self-harm", "self harm", "harm myself", "harming myself", "hurt myself",
         "hurting myself", "cut myself", "cutting myself"),
    ),
)

_CANT = re.compile(r"\b(?:can ?not|cant)\b")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, unify apostrophes and spellings of "can't", collapse whitespace."""
    text = str(text or "").lower().replace("’", "'").replace("‘", "'")
    text = _SPACES.sub(" ", text)
    return _CANT.sub("can't", text)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Whole-phrase pattern: word boundaries, flexible spaces/hyphens, optional plural 's'."""
    words = [re.escape(w) for w in re.split(r"[ -]", phrase)]
    return re.compile(r"(?<![\w'])" + r"[\s-]+".join(words) + r"s?(?![\w'])")


_COMPILED: tuple[tuple[RedFlagCategory, tuple[tuple[str, re.Pattern[str]], ...]], ...] = tuple(
    (cat, tuple((p, _phrase_pattern(p)) for p in cat.phrases)) for cat in RED_FLAG_CATEGORIES
)


# --------------------------------------------------------------------------- #
# Public checks
# --------------------------------------------------------------------------- #
def check_red_flags(text: str) -> list[RedFlag]:
    """Return every emergency category whose phrases appear in ``text``.

    Matching is case-insensitive on whole phrases. Works on any text, even one
    too short for the model, so it can run before the model.
    """
    clean = normalize(text)
    flags: list[RedFlag] = []
    for cat, patterns in _COMPILED:
        hits = tuple(phrase for phrase, pat in patterns if pat.search(clean))
        if hits:
            flags.append(RedFlag(cat.key, cat.name, cat.message, hits))
    return flags


def check_confidence(result: PredictionResult | None) -> str | None:
    """Return a 'see a general physician' message if the prediction is low-confidence."""
    if result is not None and result.low_confidence:
        return LOW_CONFIDENCE_MESSAGE
    return None


def assess(text: str, result: PredictionResult | None = None) -> SafetyReport:
    """Combine the red-flag check, the confidence check and the disclaimer.

    ``result`` may be None when no prediction was made (for example, text too short).
    """
    flags = check_red_flags(text)
    return SafetyReport(
        is_emergency=bool(flags),
        red_flags=flags,
        confidence_message=check_confidence(result),
        disclaimer=DISCLAIMER,
    )
