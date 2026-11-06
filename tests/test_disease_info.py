"""Tests for src.disease_info: loading, validation, lookup and review tracking."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from src import disease_info as di
from src.disease_info import DiseaseInfoError, find_forbidden_text, is_allowed_source, validate

LABELS = ["cold", "flu"]


def _entry(label: str, **changes: Any) -> dict[str, Any]:
    entry = {
        "label": label,
        "description": f"A short description of {label}.",
        "doctor_type": "General physician",
        "risk_level": "low",
        "risk_reason": "Usually mild.",
        "self_care": ["Rest.", "Drink plenty of fluids.", "See a doctor if it gets worse."],
        "warning_level": "urgent",
        "emergency_signs": [],
        "urgent_signs": ["Trouble breathing", "Fever that lasts longer than four days"],
        "routine_signs": [],
        "source": "https://www.nhs.uk/conditions/example/",
        "reviewed": False,
    }
    entry.update(changes)
    return entry


def _write(tmp_path: Path, entries: Any, labels: list[str] = LABELS) -> tuple[Path, Path]:
    info_path, labels_path = tmp_path / "disease_info.json", tmp_path / "labels.json"
    info_path.write_text(json.dumps(entries), encoding="utf-8")
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    return info_path, labels_path


@pytest.fixture
def good_files(tmp_path: Path) -> tuple[Path, Path]:
    return _write(tmp_path, [_entry("cold"), _entry("flu", reviewed=True)])


# --------------------------------------------------------------------------- #
# The real file
# --------------------------------------------------------------------------- #
def test_real_file_is_valid_and_covers_every_label() -> None:
    info = di.load_disease_info()
    labels = json.loads(di.LABELS_PATH.read_text(encoding="utf-8"))
    assert len(labels) == 22
    assert set(info) == set(labels)


def test_real_file_has_no_todo_left() -> None:
    raw = di.DISEASE_INFO_PATH.read_text(encoding="utf-8")
    assert "TODO" not in raw


# --------------------------------------------------------------------------- #
# Loading and lookup
# --------------------------------------------------------------------------- #
def test_load_returns_dataclasses(good_files: tuple[Path, Path]) -> None:
    info = di.load_disease_info(*good_files)
    assert set(info) == {"cold", "flu"}
    cold = info["cold"]
    assert isinstance(cold, di.DiseaseInfo)
    assert cold.self_care == ("Rest.", "Drink plenty of fluids.", "See a doctor if it gets worse.")
    assert cold.reviewed is False


def test_get_disease_info(good_files: tuple[Path, Path]) -> None:
    assert di.get_disease_info("flu", *good_files).label == "flu"


def test_get_disease_info_unknown_label(good_files: tuple[Path, Path]) -> None:
    with pytest.raises(KeyError, match="measles"):
        di.get_disease_info("measles", *good_files)


def test_unreviewed_entries(good_files: tuple[Path, Path]) -> None:
    assert [e.label for e in di.unreviewed_entries(*good_files)] == ["cold"]


def test_invalid_file_raises_with_all_problems(tmp_path: Path) -> None:
    files = _write(tmp_path, [_entry("cold", risk_level="extreme", source="http://x.com")])
    with pytest.raises(DiseaseInfoError) as err:
        di.load_disease_info(*files)
    msg = str(err.value)
    assert "risk_level" in msg and "source" in msg and "flu: no entry" in msg


# --------------------------------------------------------------------------- #
# Validation: labels
# --------------------------------------------------------------------------- #
def test_valid_entries_have_no_errors() -> None:
    assert validate([_entry("cold"), _entry("flu")], LABELS) == []


def test_missing_label_entry() -> None:
    assert validate([_entry("cold")], LABELS) == ["flu: no entry"]


def test_duplicate_label_entry() -> None:
    errors = validate([_entry("cold"), _entry("cold"), _entry("flu")], LABELS)
    assert errors == ["cold: 2 entries (expected exactly one)"]


def test_entry_for_unknown_label() -> None:
    errors = validate([_entry("cold"), _entry("flu"), _entry("measles")], LABELS)
    assert errors == ["measles: entry for a label not in labels.json"]


def test_top_level_must_be_a_list() -> None:
    assert validate({"cold": _entry("cold")}, LABELS)


# --------------------------------------------------------------------------- #
# Validation: fields
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", di.REQUIRED_FIELDS)
def test_missing_field(field: str) -> None:
    entry = _entry("cold")
    del entry[field]
    assert any(f"missing field '{field}'" in e for e in validate([entry, _entry("flu")], LABELS))


@pytest.mark.parametrize("field", ["description", "doctor_type", "risk_reason", "source"])
@pytest.mark.parametrize("value", ["", "   ", None, 3])
def test_empty_text_field(field: str, value: Any) -> None:
    errors = validate([_entry("cold", **{field: value}), _entry("flu")], LABELS)
    assert any(f"'{field}' must be a non-empty string" in e for e in errors)


@pytest.mark.parametrize("field, items", [
    ("self_care", ["Rest.", "Fluids."]),                 # too few
    ("self_care", ["a.", "b.", "c.", "d.", "e."]),       # too many
    ("urgent_signs", ["a", "b", "c", "d", "e"]),         # too many
])
def test_list_length(field: str, items: list[str]) -> None:
    errors = validate([_entry("cold", **{field: items}), _entry("flu")], LABELS)
    assert any(f"'{field}' must have" in e for e in errors)


# --------------------------------------------------------------------------- #
# Validation: warning_level and the two sign lists
# --------------------------------------------------------------------------- #
def test_warning_level_is_required() -> None:
    entry = _entry("cold")
    del entry["warning_level"]
    errors = validate([entry, _entry("flu")], LABELS)
    assert any("missing field 'warning_level'" in e for e in errors)


@pytest.mark.parametrize("level", ["", "Emergency", "high", "non-urgent", None])
def test_bad_warning_level(level: Any) -> None:
    errors = validate([_entry("cold", warning_level=level), _entry("flu")], LABELS)
    assert any("warning_level must be one of" in e for e in errors)


def test_emergency_entry_is_valid() -> None:
    entry = _entry("cold", warning_level="emergency",
                   emergency_signs=["Struggling to breathe"], urgent_signs=["Fever"])
    assert validate([entry, _entry("flu")], LABELS) == []


def test_level_must_be_emergency_when_emergency_signs_exist() -> None:
    entry = _entry("cold", warning_level="urgent",
                   emergency_signs=["Struggling to breathe"], urgent_signs=["Fever"])
    errors = validate([entry, _entry("flu")], LABELS)
    assert any("should be 'emergency'" in e for e in errors)


def test_level_must_be_urgent_without_emergency_signs() -> None:
    errors = validate([_entry("cold", warning_level="emergency"), _entry("flu")], LABELS)
    assert any("should be 'urgent'" in e for e in errors)


def test_routine_entry_is_valid() -> None:
    entry = _entry("cold", warning_level="routine", urgent_signs=[],
                   routine_signs=["A rash that does not improve", "Symptoms that come back"])
    assert validate([entry, _entry("flu")], LABELS) == []


@pytest.mark.parametrize("em, ur, ro, expected", [
    ([], [], ["a", "b"], "routine"),
    ([], ["a"], ["b"], "urgent"),
    (["a"], [], ["b"], "emergency"),
    (["a"], ["b"], ["c"], "emergency"),
])
def test_level_is_most_serious_non_empty_list(em: list[str], ur: list[str], ro: list[str],
                                              expected: str) -> None:
    good = _entry("cold", warning_level=expected, emergency_signs=em, urgent_signs=ur,
                  routine_signs=ro)
    assert validate([good, _entry("flu")], LABELS) == []
    for wrong in set(di.WARNING_LEVELS) - {expected}:
        bad = dict(good, warning_level=wrong)
        errors = validate([bad, _entry("flu")], LABELS)
        assert any(f"should be '{expected}'" in e for e in errors)


@pytest.mark.parametrize("em, ur, ro", [
    ([], ["Only one"], []),                        # too few in total
    (["a", "b"], ["c", "d"], ["e"]),               # too many in total
    ([], [], []),                                   # none at all
])
def test_signs_total(em: list[str], ur: list[str], ro: list[str]) -> None:
    level = "emergency" if em else "urgent" if ur else "routine"
    entry = _entry("cold", warning_level=level, emergency_signs=em, urgent_signs=ur,
                   routine_signs=ro)
    errors = validate([entry, _entry("flu")], LABELS)
    assert any("items together" in e for e in errors)


@pytest.mark.parametrize("field", ["emergency_signs", "urgent_signs", "routine_signs"])
def test_sign_lists_must_hold_strings(field: str) -> None:
    errors = validate([_entry("cold", **{field: "Rash"}), _entry("flu")], LABELS)
    assert any(f"'{field}' must be a list" in e for e in errors)


def test_old_warning_signs_field_is_rejected() -> None:
    errors = validate([_entry("cold", warning_signs=["Rash", "Fever"]), _entry("flu")],
                      LABELS)
    assert any("unknown field 'warning_signs'" in e for e in errors)


def test_loaded_entry_has_both_lists(good_files: tuple[Path, Path]) -> None:
    info = di.load_disease_info(*good_files)["cold"]
    assert info.warning_level == "urgent"
    assert info.emergency_signs == ()
    assert info.urgent_signs == ("Trouble breathing", "Fever that lasts longer than four days")
    assert info.routine_signs == ()


# Decided from each source page's own headings:
# "Call 999 / go to A&E / emergency room" -> emergency, "Urgent advice" -> urgent,
# "Non-urgent advice: See a GP" -> routine. Dengue: WHO "seek care immediately" -> emergency.
# Values: (warning_level, #emergency, #urgent, #routine)
REAL_LEVELS = {
    "allergy": ("emergency", 4, 0, 0), "bronchial asthma": ("emergency", 3, 0, 0),
    "drug reaction": ("emergency", 4, 0, 0), "migraine": ("emergency", 4, 0, 0),
    "peptic ulcer disease": ("emergency", 4, 0, 0), "pneumonia": ("emergency", 4, 0, 0),
    "psoriasis": ("emergency", 3, 0, 0), "hypertension": ("emergency", 3, 1, 0),
    "typhoid": ("emergency", 3, 1, 0), "urinary tract infection": ("emergency", 1, 3, 0),
    "dengue": ("emergency", 4, 0, 0), "malaria": ("urgent", 0, 4, 0),
    "jaundice": ("urgent", 0, 3, 0), "gastroesophageal reflux disease": ("urgent", 0, 1, 3),
    "varicose veins": ("urgent", 0, 1, 2), "fungal infection": ("routine", 0, 0, 3),
    "impetigo": ("routine", 0, 0, 4),
}


@pytest.mark.parametrize("label, expected", REAL_LEVELS.items())
def test_real_file_warning_levels(label: str, expected: tuple[str, int, int, int]) -> None:
    info = di.load_disease_info()[label]
    assert (info.warning_level, len(info.emergency_signs), len(info.urgent_signs),
            len(info.routine_signs)) == expected


def test_real_file_reviewed_values_are_booleans() -> None:
    assert all(isinstance(i.reviewed, bool) for i in di.load_disease_info().values())


@pytest.mark.parametrize("value", [[], "Rest.", ["Rest.", "", "Sleep."], None])
def test_list_field_must_hold_non_empty_strings(value: Any) -> None:
    errors = validate([_entry("cold", self_care=value), _entry("flu")], LABELS)
    assert any("self_care" in e for e in errors)


def test_reviewed_must_be_boolean() -> None:
    errors = validate([_entry("cold", reviewed="no"), _entry("flu")], LABELS)
    assert any("'reviewed' must be true or false" in e for e in errors)


def test_unknown_field() -> None:
    errors = validate([_entry("cold", dosage="x"), _entry("flu")], LABELS)
    assert any("unknown field 'dosage'" in e for e in errors)


@pytest.mark.parametrize("level", RISK := di.RISK_LEVELS)
def test_allowed_risk_levels(level: str) -> None:
    assert validate([_entry("cold", risk_level=level), _entry("flu")], LABELS) == []


@pytest.mark.parametrize("level", ["Low", "medium", "severe", ""])
def test_bad_risk_level(level: str) -> None:
    assert validate([_entry("cold", risk_level=level), _entry("flu")], LABELS)


# --------------------------------------------------------------------------- #
# Validation: source URL
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", [
    "https://www.nhs.uk/conditions/asthma/",
    "https://medlineplus.gov/asthma.html",
    "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
    "https://www.cdc.gov/malaria/index.html",
])
def test_allowed_sources(url: str) -> None:
    assert is_allowed_source(url)


@pytest.mark.parametrize("url", [
    "http://www.nhs.uk/conditions/asthma/",          # not https
    "https://example.com/asthma",                    # not an allowed site
    "https://nhs.uk.evil.com/asthma",                # look-alike host
    "https://fakenhs.uk/asthma",                     # suffix without a dot
    "www.nhs.uk/conditions/asthma/",                 # no scheme
    "",
])
def test_rejected_sources(url: str) -> None:
    assert not is_allowed_source(url)


# --------------------------------------------------------------------------- #
# Validation: dosage wording and digits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "Take two tablets.", "Use a medicine at the right dose.", "Take it twice a day.",
    "Use it three times a day.", "Up to five hundred mg.", "Drink ten ml.",
    "This cures it.", "Ask for a prescription.", "Take it times daily.",
])
def test_dosage_wording_is_rejected(text: str) -> None:
    assert find_forbidden_text(text)


@pytest.mark.parametrize("text", ["Fever lasting 4 days", "Rest for 2 weeks"])
def test_digits_are_rejected(text: str) -> None:
    assert "contains a digit" in find_forbidden_text(text)


@pytest.mark.parametrize("text", [
    "Drink plenty of fluids.", "Fever that lasts longer than four days",
    "Avoid foods that trigger your symptoms.", "Ask a pharmacist for advice.",
])
def test_ordinary_text_is_allowed(text: str) -> None:
    assert find_forbidden_text(text) == []


@pytest.mark.parametrize("text", [
    "Call NHS one-one-one.", "Call 111 for advice.", "Go to A&E now.", "Go to A & E.",
    "Ask your nhs doctor.", "See a GP urgently.",
])
def test_uk_specific_wording_is_rejected(text: str) -> None:
    assert any("UK-specific" in p for p in find_forbidden_text(text))


@pytest.mark.parametrize("text", [
    "Go to the nearest emergency department.", "Call your local emergency number.",
    "Contact a general physician.", "Ask a pharmacist for advice.",
])
def test_general_wording_is_allowed(text: str) -> None:
    assert find_forbidden_text(text) == []


def test_source_url_on_nhs_is_still_allowed() -> None:
    # The rule applies to user-facing text, not to the source link.
    assert validate([_entry("cold"), _entry("flu")], LABELS) == []
    assert "nhs.uk" in _entry("cold")["source"]


@pytest.mark.parametrize("field, value", [
    ("doctor_type", "General physician (GP)"),
    ("description", "Call NHS one-one-one if worried."),
    ("description", "Treat with 500mg daily."),
    ("risk_reason", "Has a cure."),
    ("self_care", ["Rest.", "Take tablets.", "Fluids."]),
    ("urgent_signs", ["Fever for 3 days", "Rash"]),
    ("emergency_signs", ["Call A&E now", "Rash"]),
    ("routine_signs", ["See your GP", "Rash"]),
])
def test_forbidden_text_in_any_field(field: str, value: Any) -> None:
    assert validate([_entry("cold", **{field: value}), _entry("flu")], LABELS)


def test_real_entries_pass_text_checks() -> None:
    raw = json.loads(di.DISEASE_INFO_PATH.read_text(encoding="utf-8"))
    for entry in raw:
        assert di.validate_entry(copy.deepcopy(entry)) == [], entry["label"]
