"""DistilBERT fine-tuning, saving and loading."""

from __future__ import annotations

import argparse
import copy
import json
import random
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import BASE_MODEL_NAME, MODELS_DIR, RANDOM_SEED, REPORTS_DIR

BERT_DIR: Path = MODELS_DIR / "bert"
METRICS_PATH: Path = REPORTS_DIR / "bert_metrics.json"
CONFUSION_VAL_PATH: Path = REPORTS_DIR / "bert_confusion_val.png"
CONFUSION_TEST_PATH: Path = REPORTS_DIR / "bert_confusion_test.png"

DEFAULT_LR = 5e-5
BATCH_SIZE = 16
MAX_EPOCHS = 8
WEIGHT_DECAY = 0.01
PATIENCE = 2
MAX_LENGTH = 128
WARMUP_FRACTION = 0.1
SMOKE_TRAIN_ROWS = 40
SMOKE_VAL_ROWS = 20


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU and GPU).

    GPU kernels can still be slightly non-deterministic, so two GPU runs may
    differ in the last decimal places.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(prefer_gpu: bool = True) -> torch.device:
    """Return the CUDA device if one is available and wanted, otherwise the CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_label_maps(labels: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    """Return (id2label, label2id) for the model config, using the label list order."""
    id2label = dict(enumerate(labels))
    label2id = {label: i for i, label in id2label.items()}
    return id2label, label2id


class EarlyStopper:
    """Track a score that should go up; signal a stop after ``patience`` bad epochs."""

    def __init__(self, patience: int = PATIENCE) -> None:
        """Start with no best score and zero epochs without improvement."""
        self.patience = patience
        self.best_score = float("-inf")
        self.bad_epochs = 0

    def update(self, score: float) -> bool:
        """Record a new score. Return True if it is a new best."""
        if score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        """True once the score has not improved for ``patience`` epochs in a row."""
        return self.bad_epochs >= self.patience


def _linear_warmup_decay(warmup_steps: int, total_steps: int):  # noqa: ANN202
    """Learning-rate multiplier: rise linearly during warmup, then fall linearly to 0."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return factor


def _make_loader(
    texts: Sequence[str],
    label_ids: Sequence[int],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    seed: int,
) -> DataLoader:
    """Shuffled DataLoader that tokenizes each batch (dynamic padding)."""

    def collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        batch_texts, batch_labels = zip(*batch)
        enc = tokenizer(list(batch_texts), truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt")
        enc = dict(enc)
        enc["labels"] = torch.tensor(batch_labels, dtype=torch.long)
        return enc

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(list(zip(texts, label_ids)), batch_size=batch_size, shuffle=True,
                      collate_fn=collate, generator=generator)


def _predict_proba(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """Return an (n_texts, n_labels) array of class probabilities."""
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            enc = tokenizer(batch, truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in dict(enc).items()}
            logits = model(**enc).logits
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    n_labels = len(model.config.id2label)
    return np.concatenate(chunks) if chunks else np.zeros((0, n_labels))


# --------------------------------------------------------------------------- #
# Inference wrapper (CPU-friendly)
# --------------------------------------------------------------------------- #
class BertClassifier:
    """A fine-tuned model plus tokenizer with a scikit-learn-like predict_proba()."""

    def __init__(self, model: Any, tokenizer: Any, device: torch.device | str = "cpu",
                 max_length: int = MAX_LENGTH, batch_size: int = 32) -> None:
        """Wrap a model and tokenizer; the model is moved to ``device``."""
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    @property
    def labels(self) -> list[str]:
        """Label names in model output order."""
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        return [id2label[i] for i in sorted(id2label)]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n_texts, n_labels) probability array, columns in ``labels`` order."""
        return _predict_proba(self.model, self.tokenizer, list(texts), self.device,
                              self.batch_size, self.max_length)


def save_bert(model: Any, tokenizer: Any, path: Path = BERT_DIR) -> None:
    """Save the fine-tuned model and its tokenizer to a folder."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_bert(path: Path = BERT_DIR, device: str = "cpu") -> BertClassifier:
    """Load a saved model and tokenizer from ``path`` (CPU by default)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return BertClassifier(model, tokenizer, device=device)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    """Everything produced by :func:`train_bert`."""

    model: Any
    tokenizer: Any
    best_epoch: int
    epochs_run: int
    train_seconds: float
    history: list[dict[str, Any]] = field(default_factory=list)


def train_bert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    labels: list[str],
    lr: float = DEFAULT_LR,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    max_length: int = MAX_LENGTH,
    seed: int = RANDOM_SEED,
    device: torch.device | None = None,
    model_name: str = BASE_MODEL_NAME,
) -> TrainResult:
    """Fine-tune ``model_name`` with early stopping on validation macro-F1.

    The weights from the best validation epoch are restored before returning.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.data import LABEL_COL, TEXT_COL, encode_labels
    from src.evaluate import compute_metrics

    device = device or choose_device()
    set_seed(seed)
    id2label, label2id = build_label_maps(labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    ).to(device)

    loader = _make_loader(train_df[TEXT_COL].tolist(), encode_labels(train_df[LABEL_COL], labels),
                          tokenizer, batch_size, max_length, seed)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    total_steps = len(loader) * max_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _linear_warmup_decay(int(WARMUP_FRACTION * total_steps), total_steps)
    )

    stopper = EarlyStopper(patience)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        probs = _predict_proba(model, tokenizer, val_df[TEXT_COL].tolist(), device,
                               max_length=max_length)
        preds = [labels[i] for i in probs.argmax(axis=1)]
        m = compute_metrics(val_df[LABEL_COL], preds, labels)
        improved = stopper.update(m["macro_f1"])
        history.append({"epoch": epoch, "train_loss": round(total_loss / len(loader), 4),
                        "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"]})
        print(f"epoch {epoch}: train_loss={history[-1]['train_loss']:.4f} "
              f"val_acc={m['accuracy']:.4f} val_macro_f1={m['macro_f1']:.4f}"
              f"{'  <- best' if improved else ''}", flush=True)
        if improved:
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stopper.should_stop:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break
    seconds = time.perf_counter() - start

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model, tokenizer, best_epoch, len(history), seconds, history)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``--lr`` and one optional mode: ``--smoke``, ``--skip-test`` or ``--test-only``."""
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on symptom texts.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"learning rate (default {DEFAULT_LR}); ignored with --test-only")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="1 epoch on 40 train / 20 val rows; no test set; real results "
                           "are not overwritten")
    mode.add_argument("--skip-test", action="store_true",
                      help="train, evaluate on validation, save model and validation metrics; "
                           "never touch the test set")
    mode.add_argument("--test-only", action="store_true",
                      help="load models/bert/ and evaluate it once on the test set")
    args = parser.parse_args(argv)
    if args.lr <= 0:
        parser.error("--lr must be positive")
    return args


def smoke_subsets(
    train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small random samples of train and validation for a quick code check."""
    return (
        train_df.sample(n=min(SMOKE_TRAIN_ROWS, len(train_df)), random_state=seed),
        val_df.sample(n=min(SMOKE_VAL_ROWS, len(val_df)), random_state=seed),
    )


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """The same train/validation/test splits and label list the baseline uses."""
    from src.data import (
        LABELS_PATH,
        build_label_list,
        load_labels,
        load_raw,
        make_splits,
        remove_train_duplicates,
    )

    raw_train, test = load_raw()
    full_train, _ = remove_train_duplicates(raw_train, test)
    train, val = make_splits(full_train)
    labels = build_label_list(full_train)
    if LABELS_PATH.exists() and load_labels(LABELS_PATH) != labels:
        raise ValueError(f"{LABELS_PATH} does not match the dataset; rerun python -m src.data")
    return train, val, test, labels


def run_smoke(lr: float) -> None:
    """Check the whole pipeline runs, writing only to a temporary folder."""
    train, val, _, labels = _load_splits()
    small_train, small_val = smoke_subsets(train, val)
    result = train_bert(small_train, small_val, labels, lr=lr, max_epochs=1)
    with tempfile.TemporaryDirectory() as tmp:
        save_bert(result.model, result.tokenizer, Path(tmp))
        clf = load_bert(Path(tmp))
        probs = clf.predict_proba(small_val["text"].head(2).tolist())
    assert probs.shape == (2, len(labels)), probs.shape
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    print(f"Smoke test OK: trained {len(small_train)} rows for 1 epoch in "
          f"{result.train_seconds:.1f} s; save -> load -> predict_proba gave shape "
          f"{probs.shape}. No files in models/ or reports/ were written.")


def _evaluate_split(
    clf: BertClassifier, df: pd.DataFrame, labels: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """Predict ``df`` and return (metrics with top confusions, predicted labels)."""
    from src.data import LABEL_COL, TEXT_COL
    from src.evaluate import compute_metrics, top_confusions

    preds = [labels[i] for i in clf.predict_proba(df[TEXT_COL].tolist()).argmax(axis=1)]
    metrics = compute_metrics(df[LABEL_COL], preds, labels)
    metrics["top_confusions"] = top_confusions(df[LABEL_COL], preds, labels)
    return metrics, preds


def _print_scores(name: str, m: dict[str, Any]) -> None:
    """Print one line of accuracy and macro-F1."""
    print(f"{name:<11} accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}"
          f"  (n={m['n_samples']})")


def run_train(lr: float, evaluate_test: bool = True) -> None:
    """Train, evaluate on validation, save model and metrics.

    With ``evaluate_test=False`` (``--skip-test``) the test set is never
    predicted; run :func:`run_test_only` later on the chosen model.
    """
    from src.data import LABEL_COL
    from src.evaluate import plot_confusion_matrix, save_metrics

    train, val, test, labels = _load_splits()
    device = choose_device()
    print(f"Device: {device}  |  train={len(train)} val={len(val)}  |  lr={lr}"
          f"{'' if evaluate_test else '  |  test set skipped'}")
    result = train_bert(train, val, labels, lr=lr, device=device)

    clf = BertClassifier(result.model, result.tokenizer, device=device)
    val_metrics, val_pred = _evaluate_split(clf, val, labels)
    metrics: dict[str, Any] = {
        "model": "distilbert",
        "base_model": BASE_MODEL_NAME,
        "params": {"lr": lr, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                   "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
                   "max_length": MAX_LENGTH, "warmup_fraction": WARMUP_FRACTION},
        "random_seed": RANDOM_SEED,
        "device": str(device),
        "train_rows": len(train),
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "train_seconds": round(result.train_seconds, 1),
        "history": result.history,
        "validation": val_metrics,
    }
    if evaluate_test:
        test_metrics, test_pred = _evaluate_split(clf, test, labels)
        metrics["test"] = test_metrics

    save_bert(result.model, result.tokenizer, BERT_DIR)
    save_metrics(metrics, METRICS_PATH)
    plot_confusion_matrix(val[LABEL_COL], val_pred, labels, CONFUSION_VAL_PATH,
                          "DistilBERT confusion matrix (validation)")
    if evaluate_test:
        plot_confusion_matrix(test[LABEL_COL], test_pred, labels, CONFUSION_TEST_PATH,
                              "DistilBERT confusion matrix (test)")

    print(f"\nBest epoch: {result.best_epoch} of {result.epochs_run} run  |  "
          f"training time {result.train_seconds:.1f} s")
    _print_scores("Validation", val_metrics)
    if evaluate_test:
        _print_scores("Test", metrics["test"])
    print(f"Saved {BERT_DIR}, {METRICS_PATH}")


def run_test_only() -> None:
    """Load the saved model from ``models/bert/`` and evaluate it once on the test set.

    The test results are added to the existing ``bert_metrics.json``. It refuses
    to run if that file already has test results, so the test set is used once.
    """
    from src.data import LABEL_COL
    from src.evaluate import load_metrics, plot_confusion_matrix, save_metrics

    if not (BERT_DIR / "config.json").exists():
        raise FileNotFoundError(f"No saved model in {BERT_DIR}; train with --skip-test first.")
    metrics = load_metrics(METRICS_PATH) if METRICS_PATH.exists() else {}
    if "test" in metrics:
        raise RuntimeError(f"{METRICS_PATH} already has test results; the test set is "
                           "evaluated only once.")

    _, _, test, labels = _load_splits()
    clf = load_bert(BERT_DIR, device=str(choose_device()))
    if clf.labels != labels:
        raise ValueError("The saved model's labels do not match data/labels.json.")

    test_metrics, test_pred = _evaluate_split(clf, test, labels)
    metrics.setdefault("model", "distilbert")
    metrics["test"] = test_metrics
    save_metrics(metrics, METRICS_PATH)
    plot_confusion_matrix(test[LABEL_COL], test_pred, labels, CONFUSION_TEST_PATH,
                          "DistilBERT confusion matrix (test)")

    if "validation" in metrics:
        _print_scores("Validation", metrics["validation"])
    _print_scores("Test", test_metrics)
    print(f"Updated {METRICS_PATH}")
    print(json.dumps(test_metrics["top_confusions"][:5], indent=1))


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for ``python -m src.bert_model``."""
    args = parse_args(argv)
    if args.smoke:
        run_smoke(args.lr)
    elif args.test_only:
        run_test_only()
    else:
        run_train(args.lr, evaluate_test=not args.skip_test)


if __name__ == "__main__":
    main()
