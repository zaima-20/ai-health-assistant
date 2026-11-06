"""Display helpers for the Streamlit app: safe word highlighting and saved test scores.

Nothing here stores or logs the user's text; ``highlight_html`` only turns it
into escaped HTML for the page.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from src.config import REPORTS_DIR
from src.explain import normalize_word

WORD_PATTERN = re.compile(r"[\w']+")
POSITIVE_RGB = "34,160,70"   # green: pushes towards the label
NEGATIVE_RGB = "220,50,50"   # red: pushes away from the label

METRICS_FILES: dict[str, str] = {
    "Baseline (TF-IDF + Logistic Regression)": "baseline_metrics.json",
    "DistilBERT": "bert_metrics.json",
}


def highlight_html(text: str, scores: dict[str, float]) -> str:
    """Return ``text`` as safe HTML with scored words highlighted.

    Words are matched to ``scores`` after :func:`normalize_word` (lowercase,
    edge punctuation removed). Stronger scores get stronger colour. Every
    piece of the user's text goes through ``html.escape`` before it is placed
    in HTML, so anything the user types is shown as text and never runs.
    """
    biggest = max((abs(s) for s in scores.values()), default=0.0) or 1.0
    parts: list[str] = []
    last = 0
    for match in WORD_PATTERN.finditer(text):
        parts.append(html.escape(text[last:match.start()]))
        word = match.group(0)
        score = scores.get(normalize_word(word))
        if score is None:
            parts.append(html.escape(word))
        else:
            alpha = 0.25 + 0.55 * abs(score) / biggest
            rgb = POSITIVE_RGB if score > 0 else NEGATIVE_RGB
            parts.append(f'<span style="background: rgba({rgb},{alpha:.2f}); '
                         f'border-radius: 4px; padding: 0 3px;" '
                         f'title="{score:+.4f}">{html.escape(word)}</span>')
        last = match.end()
    parts.append(html.escape(text[last:]))
    return ('<div style="line-height: 2; white-space: pre-wrap;">'
            + "".join(parts) + "</div>")


def load_test_scores(reports_dir: Path = REPORTS_DIR) -> dict[str, dict[str, Any]]:
    """Return ``{model name: test metrics}`` from ``reports/*_metrics.json``.

    Only real, saved numbers are returned. Files that are missing, unreadable
    or have no ``test`` section are skipped.
    """
    scores: dict[str, dict[str, Any]] = {}
    for name, file in METRICS_FILES.items():
        try:
            data = json.loads((Path(reports_dir) / file).read_text(encoding="utf-8"))
            test = data["test"]
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if isinstance(test, dict):
            scores[name] = test
    return scores


def scores_table_markdown(scores: dict[str, dict[str, Any]]) -> str:
    """Markdown table of accuracy, macro-F1 and test rows for each model."""
    rows = [
        f"| {name} | {s['accuracy']:.1%} | {s['macro_f1']:.3f} | {s.get('n_samples', '?')} |"
        if isinstance(s.get("accuracy"), (int, float)) and isinstance(s.get("macro_f1"),
                                                                      (int, float))
        else f"| {name} | n/a | n/a | {s.get('n_samples', '?')} |"
        for name, s in scores.items()
    ]
    return "| Model | Accuracy | Macro-F1 | Test rows |\n|---|---|---|---|\n" + "\n".join(rows)
