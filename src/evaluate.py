"""Evaluation metrics: accuracy, macro-F1, per-class report and confusion matrix."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: list[str]
) -> dict[str, Any]:
    """Return accuracy, macro-F1 and a per-class precision/recall/F1/support report.

    ``labels`` fixes the class order, so every class appears in the report even
    if it never shows up in ``y_true`` or ``y_pred``.
    """
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(report[label]["precision"]), 4),
            "recall": round(float(report[label]["recall"]), 4),
            "f1": round(float(report[label]["f1-score"]), 4),
            "support": int(report[label]["support"]),
        }
        for label in labels
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4
        ),
        "n_samples": len(y_true),
        "per_class": per_class,
    }


def top_confusions(
    y_true: Sequence[str], y_pred: Sequence[str], labels: list[str], n: int = 10
) -> list[dict[str, Any]]:
    """Return the ``n`` most frequent (true -> predicted) mistakes, largest first."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = [
        {"true": labels[i], "predicted": labels[j], "count": int(cm[i, j])}
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda p: (-p["count"], p["true"], p["predicted"]))
    return pairs[:n]


def plot_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: list[str],
    path: Path,
    title: str = "Confusion matrix",
) -> None:
    """Save a confusion-matrix heatmap (rows = true label, columns = predicted)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    cmap = LinearSegmentedColormap.from_list("seq_blue", [surface, "#2a78d6", "#0d3a73"])

    size = max(6.0, 0.42 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(size + 1, size), facecolor=surface)
    ax.set_facecolor(surface)
    vmax = max(int(cm.max()), 1)
    image = ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            dark_cell = cm[i, j] > 0.55 * vmax
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color=surface if dark_cell else ink,
                    fontweight="normal" if i == j else "bold")

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks, labels, rotation=55, ha="right", rotation_mode="anchor")
    ax.set_yticks(ticks, labels)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    ax.set_xticks(ticks - 0.5, minor=True)
    ax.set_yticks(ticks - 0.5, minor=True)
    ax.grid(which="minor", color=grid, linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Predicted label", color=muted)
    ax.set_ylabel("True label", color=muted)
    ax.set_title(title, loc="left", color=ink, fontsize=12, pad=10)

    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=muted, labelsize=8, length=0)
    bar.set_label("Rows", color=muted)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """Write a metrics dict to a JSON file, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_metrics(path: Path) -> dict[str, Any]:
    """Read a metrics JSON file back into a dict."""
    return json.loads(path.read_text(encoding="utf-8"))
