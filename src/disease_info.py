"""Load and validate the hand-curated disease_info.json file.

Each entry is written only from its ``source`` page (NHS, MedlinePlus, WHO or
CDC) and starts with ``"reviewed": false`` until the project owner checks it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.config import DATA_DIR

DISEASE_INFO_PATH: Path = DATA_DIR / "disease_info.json"
LABELS_PATH: Path = DATA_DIR / "labels.json"

RISK_LEVELS: tuple[str, ...] = ("low", "moderate", "high")
ALLOWED_DOMAINS: tuple[str, ...] = ("medlineplus.gov", "nhs.uk", "who.int", "cdc.gov")
TEXT_FIELDS: tuple[str, ...] = ("label", "description", "doctor_type", "risk_level",
                                "risk_reason", "source")
LIST_FIELDS: dict[str, tuple[int, int]] = {
    "self_care": (3, 4),
    "emergency_signs": (0, 4),  # source: ambulance / emergency department ("Call 999 ...")
    "urgent_signs": (0, 4),     # source: urgent doctor visit ("Urgent advice: ... NHS 111")
    "routine_signs": (0, 4),    # source: non-urgent doctor visit ("Non-urgent advice: See a GP")
}
SIGN_FIELDS: tuple[str, ...] = ("emergency_signs", "urgent_signs", "routine_signs")
SIGNS_TOTAL: tuple[int, int] = (2, 4)  # all three sign lists together
# Most serious first; warning_level is the most serious level that has any signs.
WARNING_LEVELS: tuple[str, ...] = ("emergency", "urgent", "routine")
REQUIRED_FIELDS: tuple[str, ...] = (*TEXT_FIELDS, *LIST_FIELDS, "warning_level", "reviewed")

# Medicine-dose wording that must never appear in user-facing text.
DOSAGE_PATTERN = re.compile(
    r"\b(?:mg|mcg|ml|milligrams?|millilit(?:re|er)s?|tablets?|capsules?|pills?|doses?|"
    r"dosage|prescriptions?|cures?|cured)\b"
    r"|\b(?:once|twice|thrice|\w+ times) (?:a|per) day\b|\btimes daily\b",
    re.IGNORECASE,
)
DIGIT_PATTERN = re.compile(r"\d")

# UK-only services; the app is for a general audience, so use general wording
# ("contact a doctor", "call your local emergency number") instead.
REGION_PATTERN = re.compile(r"\bNHS\b|\b111\b|\bA\s*&\s*E\b|\bGP\b", re.IGNORECASE)


class DiseaseInfoError(ValueError):
    """Raised when disease_info.json is missing entries or has invalid content."""


@dataclass(frozen=True)
class DiseaseInfo:
    """Curated information about one condition."""

    label: str
    description: str
    doctor_type: str
    risk_level: str
    risk_reason: str
    self_care: tuple[str, ...]
    warning_level: str
    emergency_signs: tuple[str, ...]
    urgent_signs: tuple[str, ...]
    routine_signs: tuple[str, ...]
    source: str
    reviewed: bool


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def is_allowed_source(url: str) -> bool:
    """True if ``url`` is https and its host is (a subdomain of) an allowed site."""
    if not isinstance(url, str) or not url.startswith("https://"):
        return False
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def find_forbidden_text(text: str) -> list[str]:
    """Return a list of problems (dosage wording, UK-only services, digits) in ``text``."""
    problems: list[str] = []
    match = DOSAGE_PATTERN.search(text)
    if match:
        problems.append(f"dosage/cure wording '{match.group(0)}'")
    match = REGION_PATTERN.search(text)
    if match:
        problems.append(f"UK-specific wording '{match.group(0)}' (use general wording)")
    if DIGIT_PATTERN.search(text):
        problems.append("contains a digit")
    return problems


def _validate_warning_level(entry: dict[str, Any], name: str) -> list[str]:
    """Check warning_level and that it agrees with the three sign lists.

    warning_level must be the most serious level whose list is not empty
    (emergency > urgent > routine), and the lists must hold 2 to 4 signs together.
    """
    errors: list[str] = []
    level = entry.get("warning_level")
    if "warning_level" in entry and level not in WARNING_LEVELS:
        errors.append(f"{name}: warning_level must be one of {', '.join(WARNING_LEVELS)}")
    lists = [entry.get(f) for f in SIGN_FIELDS]
    if not all(isinstance(lst, list) for lst in lists):
        return errors
    total = sum(len(lst) for lst in lists)
    lo, hi = SIGNS_TOTAL
    if not lo <= total <= hi:
        errors.append(f"{name}: emergency_signs + urgent_signs + routine_signs must have "
                      f"{lo} to {hi} items together, has {total}")
    if level in WARNING_LEVELS and total:
        expected = next(lvl for lvl, lst in zip(WARNING_LEVELS, lists) if lst)
        if level != expected:
            errors.append(f"{name}: warning_level is '{level}' but should be '{expected}' "
                          "(the most serious level that has signs)")
    return errors


def validate_entry(entry: Any) -> list[str]:
    """Return every problem found in one entry (empty list if it is valid)."""
    if not isinstance(entry, dict):
        return ["entry is not a JSON object"]
    name = entry.get("label", "<no label>")
    errors: list[str] = []

    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    errors += [f"{name}: missing field '{f}'" for f in missing]
    extra = sorted(set(entry) - set(REQUIRED_FIELDS))
    errors += [f"{name}: unknown field '{f}'" for f in extra]

    for f in TEXT_FIELDS:
        if f in entry and (not isinstance(entry[f], str) or not entry[f].strip()):
            errors.append(f"{name}: '{f}' must be a non-empty string")
    for f, (lo, hi) in LIST_FIELDS.items():
        if f not in entry:
            continue
        items = entry[f]
        if not isinstance(items, list) or not all(isinstance(i, str) and i.strip()
                                                  for i in items):
            errors.append(f"{name}: '{f}' must be a list of non-empty strings")
        elif not lo <= len(items) <= hi:
            errors.append(f"{name}: '{f}' must have {lo} to {hi} items, has {len(items)}")
    if "reviewed" in entry and not isinstance(entry["reviewed"], bool):
        errors.append(f"{name}: 'reviewed' must be true or false")
    errors += _validate_warning_level(entry, name)

    if entry.get("risk_level") not in RISK_LEVELS and "risk_level" in entry:
        errors.append(f"{name}: risk_level must be one of {', '.join(RISK_LEVELS)}")
    if "source" in entry and not is_allowed_source(entry["source"]):
        errors.append(f"{name}: source must be an https:// URL from "
                      f"{', '.join(ALLOWED_DOMAINS)}")

    texts: list[tuple[str, str]] = []
    for f in ("description", "doctor_type", "risk_reason"):
        if isinstance(entry.get(f), str):
            texts.append((f, entry[f]))
    for f in LIST_FIELDS:
        if isinstance(entry.get(f), list):
            texts += [(f, t) for t in entry[f] if isinstance(t, str)]
    for f, text in texts:
        errors += [f"{name}: {f} {p}: {text!r}" for p in find_forbidden_text(text)]
    return errors


def validate(entries: Any, labels: list[str]) -> list[str]:
    """Return every problem in the whole file; each label must have exactly one entry."""
    if not isinstance(entries, list):
        return ["disease_info.json must contain a JSON list of entries"]
    errors: list[str] = []
    for entry in entries:
        errors += validate_entry(entry)

    names = [e.get("label") for e in entries if isinstance(e, dict)]
    for label in labels:
        count = names.count(label)
        if count == 0:
            errors.append(f"{label}: no entry")
        elif count > 1:
            errors.append(f"{label}: {count} entries (expected exactly one)")
    errors += [f"{n}: entry for a label not in labels.json"
               for n in dict.fromkeys(names) if n not in labels]
    return errors


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_disease_info(path: Path = DISEASE_INFO_PATH,
                      labels_path: Path = LABELS_PATH) -> dict[str, DiseaseInfo]:
    """Load, validate and return ``{label: DiseaseInfo}``.

    Raises DiseaseInfoError listing every problem if the file is invalid.
    """
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    errors = validate(entries, labels)
    if errors:
        raise DiseaseInfoError(f"{path} has {len(errors)} problem(s):\n  "
                               + "\n  ".join(errors))
    return {
        e["label"]: DiseaseInfo(
            label=e["label"], description=e["description"], doctor_type=e["doctor_type"],
            risk_level=e["risk_level"], risk_reason=e["risk_reason"],
            self_care=tuple(e["self_care"]), warning_level=e["warning_level"],
            emergency_signs=tuple(e["emergency_signs"]),
            urgent_signs=tuple(e["urgent_signs"]),
            routine_signs=tuple(e["routine_signs"]),
            source=e["source"], reviewed=e["reviewed"],
        )
        for e in entries
    }


@lru_cache(maxsize=None)
def _cached(path: Path, labels_path: Path) -> dict[str, DiseaseInfo]:
    """Load the file once per path pair."""
    return load_disease_info(path, labels_path)


def get_disease_info(label: str, path: Path = DISEASE_INFO_PATH,
                     labels_path: Path = LABELS_PATH) -> DiseaseInfo:
    """Return the entry for ``label``; raises KeyError for an unknown label."""
    info = _cached(Path(path), Path(labels_path))
    if label not in info:
        raise KeyError(f"No disease info for '{label}'.")
    return info[label]


def unreviewed_entries(path: Path = DISEASE_INFO_PATH,
                       labels_path: Path = LABELS_PATH) -> list[DiseaseInfo]:
    """Return the entries whose ``reviewed`` flag is still false, in label order."""
    info = _cached(Path(path), Path(labels_path))
    return [info[k] for k in sorted(info) if not info[k].reviewed]


def main() -> None:
    """Validate the file and print a short summary (``python -m src.disease_info``)."""
    info = load_disease_info()
    pending = unreviewed_entries()
    print(f"OK: {len(info)} valid entries. Not yet reviewed: {len(pending)}")
    for entry in pending:
        print(f"  - {entry.label}  ({entry.source})")


if __name__ == "__main__":
    main()
