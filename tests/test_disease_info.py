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
        "warning_signs": ["Trouble breathing", "Fever that lasts longer than four days"],
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
    ("warning_signs", ["Only one"]),                     # too few
    ("warning_signs", ["a", "b", "c", "d", "e"]),        # too many
])
def test_list_length(field: str, items: list[str]) -> None:
    errors = validate([_entry("cold", **{field: items}), _entry("flu")], LABELS)
    assert any(f"'{field}' must have" in e for e in errors)


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
    ("warning_signs", ["Fever for 3 days", "Rash"]),
])
def test_forbidden_text_in_any_field(field: str, value: Any) -> None:
    assert validate([_entry("cold", **{field: value}), _entry("flu")], LABELS)


def test_real_entries_pass_text_checks() -> None:
    raw = json.loads(di.DISEASE_INFO_PATH.read_text(encoding="utf-8"))
    for entry in raw:
        assert di.validate_entry(copy.deepcopy(entry)) == [], entry["label"]
