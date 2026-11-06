"""Safety checks: red-flag detection and low-confidence handling.

The red-flag check runs on the raw text BEFORE any model. It has two levels:

* ``"emergency"``: seek emergency care now.
* ``"urgent"``: see a doctor today, or sooner if it gets worse.

It deliberately does not try to detect negation ("no chest pain" still
triggers): a false alarm is better than missing a real emergency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from src.config import CRISIS_HELPLINE_TEXT

if TYPE_CHECKING:  # avoid a circular import at runtime
    from src.predict import PredictionResult

Level = Literal["emergency", "urgent", "none"]
EMERGENCY: Level = "emergency"
URGENT: Level = "urgent"
NONE: Level = "none"

DISCLAIMER = ("This is an educational tool, not a medical diagnosis. "
              "Always consult a qualified doctor.")

EMERGENCY_MESSAGE = ("Your description mentions a possible emergency sign. "
                     "Seek emergency care now: call your local emergency number "
                     "or go to the nearest emergency department.")

URGENT_MESSAGE = ("Your description mentions a warning sign. See a doctor today, "
                  "or sooner if it gets worse. If it becomes severe, seek emergency care.")

LOW_CONFIDENCE_MESSAGE = ("The model is not confident about this prediction. "
                          "Please see a general physician (GP) for a proper assessment.")

CRISIS_MESSAGE = ("If you are thinking about suicide or harming yourself, please contact "
                  "your local emergency services or a crisis helpline right away. "
                  "You do not have to face this alone. " + CRISIS_HELPLINE_TEXT)


@dataclass(frozen=True)
class RedFlagCategory:
    """One kind of warning sign: its level, phrases and the message to show.

    ``replaces`` names an urgent category that is dropped when this one matches
    (e.g. "can't breathe" replaces the milder "short of breath" warning).
    """

    key: str
    name: str
    level: Level
    message: str
    phrases: tuple[str, ...]
    replaces: str | None = None


@dataclass(frozen=True)
class RedFlag:
    """A category that matched the text, with the phrases that matched."""

    key: str
    name: str
    level: Level
    message: str
    matched_phrases: tuple[str, ...]


@dataclass(frozen=True)
class SafetyReport:
    """Everything the UI needs to show around a prediction."""

    level: Level
    red_flags: list[RedFlag] = field(default_factory=list)
    confidence_message: str | None = None
    disclaimer: str = DISCLAIMER

    @property
    def is_emergency(self) -> bool:
        """True when at least one emergency-level red flag matched."""
        return self.level == EMERGENCY


# --------------------------------------------------------------------------- #
# Phrase lists (lowercase; "can't" also matches "cant", "cannot", "can not";
# a trailing "s" is allowed on every phrase, e.g. "chest pain" -> "chest pains")
# --------------------------------------------------------------------------- #
CHEST_PAIN_PHRASES: tuple[str, ...] = (
    "chest pain", "pain in my chest", "pain in the chest", "pains in my chest",
    "pains in the chest", "chest hurt", "chest is hurting", "hurts in my chest",
    "chest pressure", "pressure in my chest", "pressure in the chest", "chest tightness",
    "tight chest", "tightness in my chest", "chest feels tight", "chest is tight",
    "chest feels heavy", "heavy chest", "chest discomfort", "discomfort in my chest",
    "squeezing in my chest",
)

# A chest-pain phrase becomes an EMERGENCY when, in the same sentence, it is
# described as severe or as spreading to the arm, jaw or back.
CHEST_SEVERITY_WORDS: tuple[str, ...] = (
    "crushing", "crushed", "severe", "severely", "intense", "excruciating", "unbearable",
    "worst",
)
CHEST_SPREAD_VERBS: tuple[str, ...] = (
    "spread", "spreads", "spreading", "radiates", "radiating", "goes", "going", "moves",
    "moving", "travels", "traveling", "travelling", "shoots", "shooting", "extends",
    "extending",
)
CHEST_SPREAD_TARGETS: tuple[str, ...] = ("arm", "arms", "jaw", "back")

RED_FLAG_CATEGORIES: tuple[RedFlagCategory, ...] = (
    # ----------------------------- EMERGENCY ------------------------------- #
    RedFlagCategory(
        "severe_chest_pain", "Severe or spreading chest pain", EMERGENCY, EMERGENCY_MESSAGE,
        (),  # matched by _severe_chest_hits(): chest phrase + severity/spread wording
        replaces="chest_pain",
    ),
    RedFlagCategory(
        "severe_breathing", "Severe difficulty breathing", EMERGENCY, EMERGENCY_MESSAGE,
        ("can't breathe", "unable to breathe", "gasping for air", "struggling to breathe",
         "struggle to breathe", "lips turning blue", "lips are blue", "lips are turning blue",
         "lips have turned blue", "lips went blue", "blue lips"),
        replaces="breathing",
    ),
    RedFlagCategory(
        "fainting", "Fainting or loss of consciousness", EMERGENCY, EMERGENCY_MESSAGE,
        ("passed out", "passing out", "fainted", "fainting", "blacked out", "blacking out",
         "lost consciousness", "loss of consciousness", "losing consciousness",
         "unconscious", "collapsed", "collapsing"),
        replaces="near_fainting",
    ),
    RedFlagCategory(
        "stroke", "Stroke signs (face drooping, arm weakness, slurred speech)", EMERGENCY,
        EMERGENCY_MESSAGE + " Stroke signs need help immediately: note the time they started.",
        ("face drooping", "face is drooping", "drooping face", "face droop", "facial droop",
         "droopy face", "side of my face is drooping", "arm weakness", "weakness in one arm",
         "one arm is weak", "weak on one side", "weakness on one side",
         "numbness on one side", "one side of my body", "can't lift my arm",
         "slurred speech", "speech is slurred", "slurring", "can't speak", "trouble speaking",
         "difficulty speaking"),
    ),
    RedFlagCategory(
        "seizure", "Seizures", EMERGENCY, EMERGENCY_MESSAGE,
        ("seizure", "convulsion", "convulsing", "having a fit", "epileptic fit"),
    ),
    RedFlagCategory(
        "bleeding", "Severe bleeding", EMERGENCY, EMERGENCY_MESSAGE,
        ("severe bleeding", "heavy bleeding", "bleeding heavily", "bleeding a lot",
         "bleeding badly", "won't stop bleeding", "bleeding won't stop",
         "bleeding that won't stop", "can't stop the bleeding", "uncontrolled bleeding",
         "losing a lot of blood", "vomiting blood", "throwing up blood", "bloody vomit",
         "blood in my vomit", "coughing up blood", "cough up blood"),
    ),
    RedFlagCategory(
        "sudden_headache", "Sudden severe headache", EMERGENCY, EMERGENCY_MESSAGE,
        ("sudden severe headache", "sudden and severe headache", "severe sudden headache",
         "sudden intense headache", "worst headache", "thunderclap headache"),
    ),
    RedFlagCategory(
        "self_harm", "Suicidal thoughts or self-harm", EMERGENCY, CRISIS_MESSAGE,
        ("suicidal", "suicide", "kill myself", "killing myself", "want to die",
         "wanna die", "wish i was dead", "wish i were dead", "better off dead",
         "end my life", "ending my life", "take my own life", "no reason to live",
         "self-harm", "self harm", "harm myself", "harming myself", "hurt myself",
         "hurting myself", "cut myself", "cutting myself"),
    ),
    # ------------------------------- URGENT -------------------------------- #
    RedFlagCategory(
        "chest_pain", "Chest pain, tightness or discomfort", URGENT, URGENT_MESSAGE,
        CHEST_PAIN_PHRASES,
    ),
    RedFlagCategory(
        "breathing", "Shortness of breath or difficulty breathing", URGENT, URGENT_MESSAGE,
        ("short of breath", "shortness of breath", "out of breath", "breathless",
         "difficulty breathing", "difficulty in breathing", "trouble breathing",
         "problems breathing", "breathing problem", "hard to breathe", "hard time breathing",
         "breathing is getting harder", "can't catch my breath", "can't get enough air",
         "choking"),
    ),
    RedFlagCategory(
        "near_fainting", "Feeling faint", URGENT, URGENT_MESSAGE,
        ("feel faint", "feeling faint", "felt faint", "going to pass out",
         "gonna pass out", "about to pass out", "might pass out"),
    ),
    RedFlagCategory(
        "confusion", "Confusion or disorientation", URGENT, URGENT_MESSAGE,
        ("confused", "confusion", "disoriented", "disorientation"),
    ),
)

_CANT = re.compile(r"\b(?:can ?not|cant)\b")
_SPACES = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"[.!?;\n]+")


def normalize(text: str) -> str:
    """Lowercase, unify apostrophes and spellings of "can't", collapse whitespace."""
    text = str(text or "").lower().replace("’", "'").replace("‘", "'")
    text = _SPACES.sub(" ", text)
    return _CANT.sub("can't", text)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Whole-phrase pattern: word boundaries, flexible spaces/hyphens, optional plural 's'."""
    words = [re.escape(w) for w in re.split(r"[ -]", phrase)]
    return re.compile(r"(?<![\w'])" + r"[\s-]+".join(words) + r"s?(?![\w'])")


def _alternation(words: tuple[str, ...]) -> str:
    return "|".join(re.escape(w) for w in words)


_COMPILED: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    cat.key: tuple((p, _phrase_pattern(p)) for p in cat.phrases) for cat in RED_FLAG_CATEGORIES
}
_CHEST_PATTERNS = _COMPILED["chest_pain"]
_SEVERITY = re.compile(rf"\b({_alternation(CHEST_SEVERITY_WORDS)})\b")
_SPREAD = re.compile(
    rf"\b(?:{_alternation(CHEST_SPREAD_VERBS)})\b[^.!?;]{{0,40}}?"
    r"\b(?:to|into|down|up|through|towards?|in)\s+(?:[\w']+\s+){0,3}?"
    rf"(?:{_alternation(CHEST_SPREAD_TARGETS)})\b"
)


def _severe_chest_hits(clean: str) -> tuple[str, ...]:
    """Chest-pain phrases that share a sentence with a severity or spreading description."""
    hits: list[str] = []
    for sentence in _SENTENCE_END.split(clean):
        chest = [p for p, pat in _CHEST_PATTERNS if pat.search(sentence)]
        if not chest:
            continue
        severity = _SEVERITY.search(sentence)
        spread = _SPREAD.search(sentence)
        if severity:
            hits += [f"{p} + {severity.group(1)}" for p in chest]
        if spread:
            hits += [f"{p} + spreading ({spread.group(0)})" for p in chest]
    return tuple(dict.fromkeys(hits))


# --------------------------------------------------------------------------- #
# Public checks
# --------------------------------------------------------------------------- #
def check_red_flags(text: str) -> list[RedFlag]:
    """Return every warning category found in ``text``, emergency ones first.

    Matching is case-insensitive on whole phrases. When an emergency category
    matches, the milder urgent category it ``replaces`` is dropped. Works on any
    text, even one too short for the model, so it can run before the model.
    """
    clean = normalize(text)
    matched: list[RedFlag] = []
    for cat in RED_FLAG_CATEGORIES:
        if cat.key == "severe_chest_pain":
            hits = _severe_chest_hits(clean)
        else:
            hits = tuple(p for p, pat in _COMPILED[cat.key] if pat.search(clean))
        if hits:
            matched.append(RedFlag(cat.key, cat.name, cat.level, cat.message, hits))

    replaced = {cat.replaces for cat in RED_FLAG_CATEGORIES
                if cat.replaces and any(f.key == cat.key for f in matched)}
    flags = [f for f in matched if f.key not in replaced]
    return sorted(flags, key=lambda f: f.level != EMERGENCY)  # stable: emergency first


def overall_level(flags: list[RedFlag]) -> Level:
    """The most serious level among ``flags`` ("none" if there are none)."""
    levels = {f.level for f in flags}
    if EMERGENCY in levels:
        return EMERGENCY
    return URGENT if URGENT in levels else NONE


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
        level=overall_level(flags),
        red_flags=flags,
        confidence_message=check_confidence(result),
        disclaimer=DISCLAIMER,
    )
