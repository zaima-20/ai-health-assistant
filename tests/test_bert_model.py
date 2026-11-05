"""Tests for src.bert_model with fake models and tokenizers (no download, no training)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src import bert_model

LABELS = ["cold", "flu", "malaria"]


class FakeTokenizer:
    """Turns each text into a fixed-size tensor; the first token is the word count."""

    def __call__(self, texts, truncation=True, max_length=128, padding=True,
                 return_tensors="pt"):
        ids = torch.tensor([[len(t.split()), 0] for t in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeModel(torch.nn.Module):
    """Predicts label index (word count - 1) with a large logit."""

    def __init__(self) -> None:
        super().__init__()
        id2label, label2id = bert_model.build_label_maps(LABELS)
        self.config = SimpleNamespace(id2label=id2label, label2id=label2id)

    def forward(self, input_ids, attention_mask):
        logits = torch.zeros(input_ids.shape[0], len(LABELS))
        target = (input_ids[:, 0] - 1).clamp(0, len(LABELS) - 1)
        logits[torch.arange(len(target)), target] = 10.0
        return SimpleNamespace(logits=logits)


# ----- label mapping -------------------------------------------------------- #
def test_build_label_maps_round_trip() -> None:
    id2label, label2id = bert_model.build_label_maps(LABELS)
    assert id2label == {0: "cold", 1: "flu", 2: "malaria"}
    assert label2id == {"cold": 0, "flu": 1, "malaria": 2}
    assert all(id2label[label2id[x]] == x for x in LABELS)


def test_classifier_labels_follow_id_order_even_with_string_keys() -> None:
    model = FakeModel()
    model.config.id2label = {"2": "malaria", "0": "cold", "1": "flu"}  # as loaded from JSON
    clf = bert_model.BertClassifier(model, FakeTokenizer())
    assert clf.labels == LABELS


# ----- device choice -------------------------------------------------------- #
def test_choose_device_uses_gpu_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert bert_model.choose_device().type == "cuda"
    assert bert_model.choose_device(prefer_gpu=False).type == "cpu"


def test_choose_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bert_model.choose_device().type == "cpu"


# ----- argument parsing ----------------------------------------------------- #
def test_parse_args_defaults() -> None:
    args = bert_model.parse_args([])
    assert args.lr == 5e-5
    assert args.smoke is False


def test_parse_args_custom_values() -> None:
    args = bert_model.parse_args(["--lr", "3e-5", "--smoke"])
    assert args.lr == 3e-5
    assert args.smoke is True


def test_parse_args_rejects_bad_lr() -> None:
    with pytest.raises(SystemExit):
        bert_model.parse_args(["--lr", "0"])


def test_parse_args_skip_test_and_test_only() -> None:
    args = bert_model.parse_args(["--lr", "3e-5", "--skip-test"])
    assert args.skip_test and not args.test_only and args.lr == 3e-5
    args = bert_model.parse_args(["--test-only"])
    assert args.test_only and not args.skip_test


@pytest.mark.parametrize(
    "flags", [["--skip-test", "--test-only"], ["--smoke", "--test-only"], ["--smoke", "--skip-test"]]
)
def test_parse_args_modes_are_exclusive(flags: list[str]) -> None:
    with pytest.raises(SystemExit):
        bert_model.parse_args(flags)


def test_main_dispatches_to_the_right_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(bert_model, "run_smoke", lambda lr: calls.append(("smoke", lr)))
    monkeypatch.setattr(bert_model, "run_test_only", lambda: calls.append(("test_only",)))
    monkeypatch.setattr(bert_model, "run_train",
                        lambda lr, evaluate_test: calls.append(("train", lr, evaluate_test)))
    bert_model.main(["--smoke"])
    bert_model.main(["--test-only"])
    bert_model.main(["--skip-test", "--lr", "3e-5"])
    bert_model.main([])
    assert calls == [("smoke", 5e-5), ("test_only",), ("train", 3e-5, False),
                     ("train", 5e-5, True)]


# ----- prediction on CPU with a fake model ---------------------------------- #
def test_predict_proba_shape_order_and_sums() -> None:
    clf = bert_model.BertClassifier(FakeModel(), FakeTokenizer(), device="cpu", batch_size=2)
    texts = ["one", "two words", "three words here", "one"]
    probs = clf.predict_proba(texts)
    assert probs.shape == (4, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert [clf.labels[i] for i in probs.argmax(axis=1)] == ["cold", "flu", "malaria", "cold"]


def test_predict_proba_empty_input() -> None:
    clf = bert_model.BertClassifier(FakeModel(), FakeTokenizer())
    assert clf.predict_proba([]).shape == (0, 3)


# ----- early stopping, seeds, smoke subsets --------------------------------- #
def test_early_stopper_keeps_best_and_stops_after_patience() -> None:
    stopper = bert_model.EarlyStopper(patience=2)
    improved = [stopper.update(s) for s in [0.5, 0.7, 0.6, 0.7]]
    assert improved == [True, True, False, False]
    assert stopper.best_score == 0.7
    assert stopper.should_stop


def test_early_stopper_resets_after_improvement() -> None:
    stopper = bert_model.EarlyStopper(patience=2)
    for s in [0.5, 0.4, 0.6]:
        stopper.update(s)
    assert stopper.bad_epochs == 0 and not stopper.should_stop


def test_set_seed_makes_torch_reproducible() -> None:
    bert_model.set_seed(123)
    a = torch.rand(3)
    bert_model.set_seed(123)
    assert torch.equal(a, torch.rand(3))


def test_linear_warmup_decay_shape() -> None:
    f = bert_model._linear_warmup_decay(warmup_steps=2, total_steps=10)
    assert f(0) == 0.5 and f(1) == 1.0
    assert f(2) == 1.0 and f(10) == 0.0
    assert f(6) == pytest.approx(0.5)


# ----- --skip-test and --test-only (fake model, fake data, temp paths) ------- #
# Word count picks the label in FakeModel: 1 word -> cold, 2 -> flu, 3 -> malaria.
VAL = pd.DataFrame({"text": ["a", "a b", "a b c"], "label": ["cold", "flu", "malaria"]})
TEST = pd.DataFrame({"text": ["test one", "test two words", "x"],
                     "label": ["flu", "flu", "cold"]})


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Point all output paths at tmp_path and replace data, training and saving."""
    monkeypatch.setattr(bert_model, "BERT_DIR", tmp_path / "bert")
    monkeypatch.setattr(bert_model, "METRICS_PATH", tmp_path / "bert_metrics.json")
    monkeypatch.setattr(bert_model, "CONFUSION_VAL_PATH", tmp_path / "cm_val.png")
    monkeypatch.setattr(bert_model, "CONFUSION_TEST_PATH", tmp_path / "cm_test.png")
    monkeypatch.setattr(bert_model, "choose_device", lambda prefer_gpu=True: torch.device("cpu"))
    monkeypatch.setattr(bert_model, "_load_splits", lambda: (VAL, VAL, TEST, LABELS))
    monkeypatch.setattr(
        bert_model, "train_bert",
        lambda *a, **k: bert_model.TrainResult(FakeModel(), FakeTokenizer(), 2, 4, 1.5,
                                               [{"epoch": 1}]),
    )

    def fake_save(model, tokenizer, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")

    monkeypatch.setattr(bert_model, "save_bert", fake_save)
    monkeypatch.setattr(bert_model, "load_bert",
                        lambda path, device="cpu": bert_model.BertClassifier(FakeModel(),
                                                                             FakeTokenizer()))
    seen: list[str] = []
    original = bert_model.BertClassifier.predict_proba

    def spy(self, texts):
        seen.extend(texts)
        return original(self, texts)

    monkeypatch.setattr(bert_model.BertClassifier, "predict_proba", spy)
    return tmp_path, seen


def test_skip_test_never_touches_test_set(fake_env) -> None:
    tmp_path, seen = fake_env
    bert_model.run_train(5e-5, evaluate_test=False)
    metrics = json.loads((tmp_path / "bert_metrics.json").read_text())

    assert not set(seen) & set(TEST["text"])  # no test text was ever predicted
    assert "test" not in metrics
    assert metrics["validation"]["accuracy"] == 1.0
    assert metrics["best_epoch"] == 2 and metrics["params"]["lr"] == 5e-5
    assert (tmp_path / "bert" / "config.json").exists()
    assert (tmp_path / "cm_val.png").exists()
    assert not (tmp_path / "cm_test.png").exists()


def test_test_only_adds_test_results_once(fake_env) -> None:
    tmp_path, _ = fake_env
    bert_model.run_train(3e-5, evaluate_test=False)
    bert_model.run_test_only()
    metrics = json.loads((tmp_path / "bert_metrics.json").read_text())

    assert metrics["validation"]["accuracy"] == 1.0  # validation results kept
    assert metrics["params"]["lr"] == 3e-5
    assert metrics["test"]["n_samples"] == 3
    assert metrics["test"]["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert (tmp_path / "cm_test.png").exists()

    with pytest.raises(RuntimeError, match="only once"):
        bert_model.run_test_only()


def test_test_only_needs_a_saved_model(fake_env) -> None:
    with pytest.raises(FileNotFoundError):
        bert_model.run_test_only()


def test_test_only_rejects_label_mismatch(fake_env, monkeypatch: pytest.MonkeyPatch) -> None:
    bert_model.run_train(5e-5, evaluate_test=False)
    monkeypatch.setattr(bert_model, "_load_splits",
                        lambda: (VAL, VAL, TEST, ["cold", "flu", "typhoid"]))
    with pytest.raises(ValueError, match="labels"):
        bert_model.run_test_only()


def test_smoke_subsets_sizes_and_reproducible() -> None:
    train = pd.DataFrame({"text": [f"t{i}" for i in range(100)], "label": ["flu"] * 100})
    val = pd.DataFrame({"text": [f"v{i}" for i in range(30)], "label": ["flu"] * 30})
    a_train, a_val = bert_model.smoke_subsets(train, val)
    b_train, _ = bert_model.smoke_subsets(train, val)
    assert len(a_train) == 40 and len(a_val) == 20
    pd.testing.assert_frame_equal(a_train, b_train)
