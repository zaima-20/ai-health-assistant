"""TF-IDF + Logistic Regression baseline: train, save and load."""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from src.data import LABEL_COL, TEXT_COL, build_label_list
from src.evaluate import compute_metrics, plot_confusion_matrix, save_metrics, top_confusions

MODEL_PATH: Path = MODELS_DIR / "baseline.joblib"
METRICS_PATH: Path = REPORTS_DIR / "baseline_metrics.json"
CONFUSION_VAL_PATH: Path = REPORTS_DIR / "baseline_confusion_val.png"
CONFUSION_TEST_PATH: Path = REPORTS_DIR / "baseline_confusion_test.png"

NGRAM_OPTIONS: list[tuple[int, int]] = [(1, 1), (1, 2)]
C_OPTIONS: list[float] = [0.1, 1.0, 10.0]
MAX_ITER = 5000


def build_pipeline(
    ngram_range: tuple[int, int] = (1, 1), C: float = 1.0, seed: int = RANDOM_SEED
) -> Pipeline:
    """Create an untrained TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=ngram_range)),
            ("clf", LogisticRegression(C=C, max_iter=MAX_ITER, random_state=seed)),
        ]
    )


def train_baseline(
    train_df: pd.DataFrame,
    ngram_range: tuple[int, int] = (1, 1),
    C: float = 1.0,
    seed: int = RANDOM_SEED,
) -> tuple[Pipeline, float]:
    """Fit a pipeline on ``train_df`` and return it with the training time in seconds."""
    model = build_pipeline(ngram_range, C, seed)
    start = time.perf_counter()
    model.fit(train_df[TEXT_COL], train_df[LABEL_COL])
    return model, time.perf_counter() - start


def grid_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ngram_options: list[tuple[int, int]] = NGRAM_OPTIONS,
    c_options: list[float] = C_OPTIONS,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Train one model per setting on train and score it on validation only."""
    labels = build_label_list(train_df)
    results: list[dict[str, Any]] = []
    for ngram_range, C in itertools.product(ngram_options, c_options):
        model, seconds = train_baseline(train_df, ngram_range, C, seed)
        metrics = compute_metrics(val_df[LABEL_COL], model.predict(val_df[TEXT_COL]), labels)
        results.append(
            {
                "ngram_range": list(ngram_range),
                "C": C,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "train_seconds": round(seconds, 3),
            }
        )
    return results


def pick_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the setting with the highest validation macro-F1.

    Ties go to the earlier (simpler) setting in grid order.
    """
    return max(results, key=lambda r: r["val_macro_f1"])


def format_grid_table(results: list[dict[str, Any]]) -> str:
    """Render grid results as a plain-text table."""
    lines = [f"{'n-grams':<9}{'C':>6}  {'val acc':>8}  {'val macro-F1':>12}  {'secs':>6}"]
    for r in results:
        ngram = f"({r['ngram_range'][0]},{r['ngram_range'][1]})"
        lines.append(
            f"{ngram:<9}{r['C']:>6g}  {r['val_accuracy']:>8.4f}  {r['val_macro_f1']:>12.4f}"
            f"  {r['train_seconds']:>6.2f}"
        )
    return "\n".join(lines)


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    """Save a trained pipeline with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load a pipeline saved with :func:`save_model`."""
    return joblib.load(path)


def main() -> None:
    """Grid-search on validation, retrain the best setting, evaluate once on test, save all."""
    from src.data import load_raw, make_splits, remove_train_duplicates

    raw_train, test = load_raw()
    full_train, _ = remove_train_duplicates(raw_train, test)
    train, val = make_splits(full_train)
    labels = build_label_list(full_train)

    results = grid_search(train, val)
    print("Grid search (scored on validation):")
    print(format_grid_table(results))
    best = pick_best(results)
    ngram_range, C = tuple(best["ngram_range"]), best["C"]
    print(f"\nBest: n-grams={ngram_range}, C={C:g} (val macro-F1 {best['val_macro_f1']:.4f})")

    model, seconds = train_baseline(train, ngram_range, C)
    val_pred = model.predict(val[TEXT_COL])
    test_pred = model.predict(test[TEXT_COL])  # the single evaluation on test

    val_metrics = compute_metrics(val[LABEL_COL], val_pred, labels)
    test_metrics = compute_metrics(test[LABEL_COL], test_pred, labels)
    val_metrics["top_confusions"] = top_confusions(val[LABEL_COL], val_pred, labels)
    test_metrics["top_confusions"] = top_confusions(test[LABEL_COL], test_pred, labels)

    save_model(model)
    save_metrics(
        {
            "model": "tfidf_logreg",
            "best_params": {"ngram_range": list(ngram_range), "C": C, "max_iter": MAX_ITER},
            "random_seed": RANDOM_SEED,
            "train_rows": len(train),
            "train_seconds": round(seconds, 3),
            "grid_search": results,
            "validation": val_metrics,
            "test": test_metrics,
        },
        METRICS_PATH,
    )
    plot_confusion_matrix(val[LABEL_COL], val_pred, labels, CONFUSION_VAL_PATH,
                          "Baseline confusion matrix (validation)")
    plot_confusion_matrix(test[LABEL_COL], test_pred, labels, CONFUSION_TEST_PATH,
                          "Baseline confusion matrix (test)")

    print(f"Training time: {seconds:.2f} s")
    for name, m in (("Validation", val_metrics), ("Test", test_metrics)):
        print(f"{name:<11} accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}"
              f"  (n={m['n_samples']})")
    print("\nTop test confusions (true -> predicted):")
    for c in test_metrics["top_confusions"]:
        print(f"  {c['count']}  {c['true']} -> {c['predicted']}")
    print("\nLowest test F1:")
    for label, s in sorted(test_metrics["per_class"].items(), key=lambda kv: kv[1]["f1"])[:5]:
        print(f"  {s['f1']:.3f}  {label}")


if __name__ == "__main__":
    main()
