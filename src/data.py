"""Load the dataset, make train/validation/test splits, and encode labels."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    DATASET_NAME,
    RANDOM_SEED,
    REPORTS_DIR,
    VALIDATION_SIZE,
)

TEXT_COL = "text"
LABEL_COL = "label"
LABELS_PATH: Path = DATA_DIR / "labels.json"
SUMMARY_PATH: Path = REPORTS_DIR / "data_summary.json"
DISTRIBUTION_PLOT_PATH: Path = REPORTS_DIR / "label_distribution.png"

# Raw column names in the Hugging Face dataset
_RAW_TEXT_COL = "input_text"
_RAW_LABEL_COL = "output_text"
_WHITESPACE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Trim the text and collapse runs of whitespace into single spaces.

    Case is kept as is, because the tokenizers decide how to handle it.
    """
    return _WHITESPACE.sub(" ", str(text)).strip()


def clean_label(label: str) -> str:
    """Lowercase a label and trim/collapse its spaces so each label has one spelling."""
    return _WHITESPACE.sub(" ", str(label)).strip().lower()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with cleaned ``text`` and ``label`` columns only."""
    out = pd.DataFrame(
        {
            TEXT_COL: df[TEXT_COL].map(clean_text),
            LABEL_COL: df[LABEL_COL].map(clean_label),
        }
    )
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Loading and splitting
# --------------------------------------------------------------------------- #
def load_raw(dataset_name: str = DATASET_NAME) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (or read from cache) the dataset and return cleaned (train, test).

    Each DataFrame has exactly two columns: ``text`` and ``label``.
    """
    from datasets import load_dataset  # imported here so tests never need it

    ds = load_dataset(dataset_name)
    frames: list[pd.DataFrame] = []
    for split in ("train", "test"):
        raw = ds[split].to_pandas()
        raw = raw.rename(columns={_RAW_TEXT_COL: TEXT_COL, _RAW_LABEL_COL: LABEL_COL})
        frames.append(clean_dataframe(raw))
    return frames[0], frames[1]


def remove_train_duplicates(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean the official train set before splitting; the test set is only read.

    Rule 1: drop exact duplicate texts within train, keeping the first copy.
    Rule 2: drop train rows whose text also appears in the test set (leakage).

    Returns the cleaned train DataFrame and how many rows each rule removed.
    """
    rows_before = len(train_df)
    deduped = train_df.drop_duplicates(subset=TEXT_COL, keep="first")
    removed_duplicates = rows_before - len(deduped)

    test_texts = set(test_df[TEXT_COL])
    no_leak = deduped[~deduped[TEXT_COL].isin(test_texts)]
    removed_test_overlap = len(deduped) - len(no_leak)

    stats = {
        "rows_before": rows_before,
        "removed_duplicates": removed_duplicates,
        "removed_test_overlap": removed_test_overlap,
        "rows_after": len(no_leak),
    }
    return no_leak.reset_index(drop=True), stats


def make_splits(
    train_df: pd.DataFrame,
    val_size: float = VALIDATION_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the official train set into (train, validation), stratified by label.

    The official test set is not an input here, so it can never be changed or
    used for a decision.
    """
    train_part, val_part = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df[LABEL_COL],
    )
    return train_part.reset_index(drop=True), val_part.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def build_label_list(df: pd.DataFrame) -> list[str]:
    """Return the sorted list of unique labels in ``df``."""
    return sorted(df[LABEL_COL].unique().tolist())


def save_labels(labels: list[str], path: Path = LABELS_PATH) -> None:
    """Write the label list to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def load_labels(path: Path = LABELS_PATH) -> list[str]:
    """Read the label list from a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def label_to_id(labels: list[str]) -> dict[str, int]:
    """Map each label name to its integer id (its position in ``labels``)."""
    return {label: i for i, label in enumerate(labels)}


def id_to_label(labels: list[str]) -> dict[int, str]:
    """Map each integer id back to its label name."""
    return dict(enumerate(labels))


def encode_labels(series: pd.Series, labels: list[str]) -> list[int]:
    """Convert a Series of label names to integer ids. Unknown labels raise KeyError."""
    mapping = label_to_id(labels)
    return [mapping[label] for label in series]


# --------------------------------------------------------------------------- #
# Data check
# --------------------------------------------------------------------------- #
def _word_counts(df: pd.DataFrame) -> pd.Series:
    """Number of whitespace-separated words in each text."""
    return df[TEXT_COL].str.split().str.len()


def _length_stats(df: pd.DataFrame) -> dict[str, float]:
    """Min, mean and max word counts for a split."""
    counts = _word_counts(df)
    return {
        "min": int(counts.min()),
        "mean": round(float(counts.mean()), 1),
        "max": int(counts.max()),
    }


def find_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Report exact duplicate texts inside one DataFrame (nothing is removed).

    ``extra_rows`` is how many rows would go if each text were kept once.
    ``conflicting_labels`` counts duplicated texts that carry different labels.
    """
    dup_mask = df[TEXT_COL].duplicated(keep=False)
    dups = df[dup_mask]
    groups = dups.groupby(TEXT_COL)[LABEL_COL].nunique()
    return {
        "duplicated_texts": int(groups.shape[0]),
        "rows_involved": int(dup_mask.sum()),
        "extra_rows": int(df[TEXT_COL].duplicated().sum()),
        "conflicting_labels": int((groups > 1).sum()),
    }


def find_overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, int]:
    """Report texts that appear in both ``a`` and ``b`` (possible data leakage)."""
    shared = set(a[TEXT_COL]) & set(b[TEXT_COL])
    b_rows = b[b[TEXT_COL].isin(shared)]
    a_labels = a.groupby(TEXT_COL)[LABEL_COL].agg(set)
    same_label = sum(
        1 for text, label in zip(b_rows[TEXT_COL], b_rows[LABEL_COL]) if label in a_labels[text]
    )
    return {
        "shared_texts": len(shared),
        "rows_in_second": int(b_rows.shape[0]),
        "rows_in_second_same_label": same_label,
    }


def summarize(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_train_df: pd.DataFrame | None = None,
    cleaning_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the data-check summary as a JSON-ready dict.

    ``full_train_df`` is the (cleaned) train set before the validation split; it
    is used for the duplicate and train-vs-test leakage checks. If omitted, it
    is rebuilt by joining ``train_df`` and ``val_df``. ``cleaning_stats`` is the
    output of :func:`remove_train_duplicates` and is included as-is.
    """
    if full_train_df is None:
        full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    splits = {"train": train_df, "validation": val_df, "test": test_df}

    labels = build_label_list(full_train_df)
    per_label = {
        name: df[LABEL_COL].value_counts().reindex(labels, fill_value=0).astype(int).to_dict()
        for name, df in splits.items()
    }
    balance = {}
    for name, counts in per_label.items():
        values = list(counts.values())
        balance[name] = {
            "min": min(values),
            "max": max(values),
            "max_to_min_ratio": round(max(values) / min(values), 2) if min(values) else None,
        }

    test_only_labels = sorted(set(test_df[LABEL_COL]) - set(labels))
    return {
        "train_cleaning": cleaning_stats,
        "rows": {name: int(len(df)) for name, df in splits.items()},
        "num_labels": len(labels),
        "labels_only_in_test": test_only_labels,
        "rows_per_label": per_label,
        "balance": balance,
        "text_length_words": {name: _length_stats(df) for name, df in splits.items()},
        "duplicates_in_official_train": find_duplicates(full_train_df),
        "duplicates_in_test": find_duplicates(test_df),
        "overlap_official_train_vs_test": find_overlap(full_train_df, test_df),
        "overlap_train_vs_validation": find_overlap(train_df, val_df),
    }


def format_summary(summary: dict[str, Any]) -> str:
    """Turn the summary dict into a readable plain-text report."""
    lines: list[str] = []
    c = summary.get("train_cleaning")
    if c:
        lines.append(
            f"Train cleaning: {c['rows_before']} rows -> {c['rows_after']}"
            f" (removed {c['removed_duplicates']} duplicates,"
            f" {c['removed_test_overlap']} also in test)"
        )
    rows = summary["rows"]
    lines.append(
        f"Rows: train={rows['train']}  validation={rows['validation']}  test={rows['test']}"
    )
    lines.append(f"Labels: {summary['num_labels']}")
    if summary["labels_only_in_test"]:
        lines.append(f"WARNING labels only in test: {summary['labels_only_in_test']}")

    lines.append("\nRows per label (train / validation / test):")
    per = summary["rows_per_label"]
    width = max(len(label) for label in per["train"])
    for label in per["train"]:
        lines.append(
            f"  {label:<{width}}  {per['train'][label]:>4} {per['validation'][label]:>4}"
            f" {per['test'][label]:>4}"
        )
    lines.append("\nBalance (rows per label):")
    for name, b in summary["balance"].items():
        lines.append(
            f"  {name:<10} min={b['min']}  max={b['max']}  max/min={b['max_to_min_ratio']}"
        )

    lines.append("\nText length in words:")
    for name, s in summary["text_length_words"].items():
        lines.append(f"  {name:<10} min={s['min']}  mean={s['mean']}  max={s['max']}")

    d = summary["duplicates_in_official_train"]
    lines.append(
        f"\nDuplicates in official train: {d['duplicated_texts']} texts appear more than once"
        f" ({d['rows_involved']} rows, {d['extra_rows']} extra copies,"
        f" {d['conflicting_labels']} with conflicting labels)"
    )
    d = summary["duplicates_in_test"]
    lines.append(
        f"Duplicates in test: {d['duplicated_texts']} texts ({d['extra_rows']} extra copies)"
    )
    o = summary["overlap_official_train_vs_test"]
    lines.append(
        f"Leakage official train vs test: {o['shared_texts']} shared texts,"
        f" {o['rows_in_second']} test rows affected ({o['rows_in_second_same_label']} same label)"
    )
    o = summary["overlap_train_vs_validation"]
    lines.append(
        f"Overlap train vs validation: {o['shared_texts']} shared texts,"
        f" {o['rows_in_second']} validation rows affected"
    )
    return "\n".join(lines)


def plot_label_distribution(summary: dict[str, Any], path: Path = DISTRIBUTION_PLOT_PATH) -> None:
    """Save a horizontal stacked bar chart of rows per label, one segment per split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = summary["rows_per_label"]
    labels = sorted(per["train"], key=lambda lab: sum(per[s][lab] for s in per))
    series = [
        ("train", "Train", "#2a78d6"),
        ("validation", "Validation", "#eb6834"),
        ("test", "Test", "#1baf7a"),
    ]
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=surface)
    ax.set_facecolor(surface)
    left = [0] * len(labels)
    for key, name, color in series:
        values = [per[key][lab] for lab in labels]
        ax.barh(labels, values, left=left, color=color, label=name, height=0.7,
                edgecolor=surface, linewidth=1.5)
        left = [a + b for a, b in zip(left, values)]
    for y, total in enumerate(left):
        ax.text(total + 0.5, y, str(total), va="center", fontsize=8, color=muted)

    ax.set_title("Rows per label by split", loc="left", color=ink, fontsize=13, pad=30)
    ax.set_xlabel("Rows", color=muted)
    ax.tick_params(colors=muted, labelsize=9, length=0)
    ax.xaxis.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              labelcolor=ink, handlelength=1, borderaxespad=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Command-line entry point: python -m src.data
# --------------------------------------------------------------------------- #
def main() -> None:
    """Load data, split it, save labels, and write the data-check report and chart."""
    raw_train, test = load_raw()
    full_train, cleaning_stats = remove_train_duplicates(raw_train, test)
    train, val = make_splits(full_train)

    labels = build_label_list(full_train)
    save_labels(labels)

    summary = summarize(
        train, val, test, full_train_df=full_train, cleaning_stats=cleaning_stats
    )
    print(format_summary(summary))

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_label_distribution(summary)
    print(f"\nSaved {LABELS_PATH.relative_to(DATA_DIR.parent)}, "
          f"{SUMMARY_PATH.relative_to(REPORTS_DIR.parent)}, "
          f"{DISTRIBUTION_PLOT_PATH.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
