"""Tests for src.explain with tiny fake models (no real model loading, no training)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src import explain as ex
from src import predict
from src.explain import explain, select_top
from src.predict import Predictor

LABELS = ["cold", "malaria", "migraine"]
TEXT = "I have chills and a fever and sweating at night"


# --------------------------------------------------------------------------- #
# Fake linear pipeline (looks like TF-IDF + LogisticRegression)
# --------------------------------------------------------------------------- #
class FakeVectorizer:
    """Every known word present in the text gets TF-IDF value 1.0 (or ``value``)."""

    def __init__(self, vocab: Sequence[str], value: float = 1.0) -> None:
        self.vocab = list(vocab)
        self.value = value

    def transform(self, texts: Sequence[str]) -> csr_matrix:
        rows = np.zeros((len(texts), len(self.vocab)))
        for r, t in enumerate(texts):
            words = {ex.normalize_word(w) for w in t.split()}
            for j, v in enumerate(self.vocab):
                if v in words:
                    rows[r, j] = self.value
        return csr_matrix(rows)

    def get_feature_names_out(self) -> np.ndarray:
        return np.array(self.vocab)


class FakeClassifier:
    def __init__(self, coef: np.ndarray, classes: Sequence[str]) -> None:
        self.coef_ = np.asarray(coef, dtype=float)
        self.classes_ = np.array(classes)


class FakePipeline:
    def __init__(self, vocab: Sequence[str], coef: np.ndarray,
                 classes: Sequence[str] = LABELS, value: float = 1.0) -> None:
        self.named_steps = {"tfidf": FakeVectorizer(vocab, value),
                            "clf": FakeClassifier(coef, classes)}
        self.classes_ = np.array(classes)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        x = self.named_steps["tfidf"].transform(texts).toarray()
        logits = x @ self.named_steps["clf"].coef_.T
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


VOCAB = ["chills", "fever", "sweating", "night", "sneezing", "headache"]
#                     chills fever sweat night sneeze headache
COEF = np.array([[-1.0, 0.2, -0.3, 0.0, 2.0, 0.0],    # cold
                 [3.0, 0.5, 1.0, 0.0, -1.5, -0.2],     # malaria
                 [-0.5, -0.1, 0.0, 0.0, 0.0, 2.5]])    # migraine


def linear_predictor(**kw) -> Predictor:  # noqa: ANN003
    return Predictor("baseline", model=FakePipeline(VOCAB, COEF, **kw))


# --------------------------------------------------------------------------- #
# Fake black-box model (for leave-one-word-out)
# --------------------------------------------------------------------------- #
class FakeBlackBox:
    """Softmax over per-word weights; no named_steps, so explain must use leave-one-out."""

    def __init__(self, weights: dict[str, list[float]], labels: Sequence[str] = LABELS) -> None:
        self.weights = {w: np.asarray(v, dtype=float) for w, v in weights.items()}
        self.labels = list(labels)
        self.calls: list[int] = []

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(len(texts))
        out = []
        for t in texts:
            logits = np.zeros(len(self.labels))
            for w in t.split():
                logits += self.weights.get(ex.normalize_word(w), 0.0)
            e = np.exp(logits - logits.max())
            out.append(e / e.sum())
        return np.array(out)


BB_WEIGHTS = {"chills": [-1, 4, -1], "fever": [0, 1, 0], "sweating": [0, 1.5, 0],
              "night": [0, 0, 0.5], "sneezing": [3, -2, 0]}


def bb_predictor(weights: dict[str, list[float]] = BB_WEIGHTS) -> Predictor:
    return Predictor("bert", model=FakeBlackBox(weights))


# --------------------------------------------------------------------------- #
# Method selection
# --------------------------------------------------------------------------- #
def test_method_selection() -> None:
    assert ex.explain_method(linear_predictor()) == "linear"
    assert ex.explain_method(bb_predictor()) == "leave-one-out"


# --------------------------------------------------------------------------- #
# Strongly linked word ranks first
# --------------------------------------------------------------------------- #
def test_linear_strong_word_ranks_first() -> None:
    words = explain(linear_predictor(), TEXT, "malaria")
    assert words[0][0] == "chills"
    assert words[0][1] == pytest.approx(3.0)


def test_leave_one_out_strong_word_ranks_first() -> None:
    words = explain(bb_predictor(), TEXT, "malaria")
    assert words[0][0] == "chills"
    assert words[0][1] > 0


def test_ranking_depends_on_label() -> None:
    text = "I have sneezing and chills today"
    assert explain(linear_predictor(), text, "cold")[0][0] == "sneezing"
    assert explain(bb_predictor(), text, "cold")[0][0] == "sneezing"


# --------------------------------------------------------------------------- #
# Signs and sorting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make", [linear_predictor, bb_predictor])
def test_scores_sorted_highest_first(make) -> None:  # noqa: ANN001
    text = "chills fever sweating sneezing headache night"
    words = explain(make(), text, "malaria")
    scores = [s for _, s in words]
    assert scores == sorted(scores, reverse=True)


def test_negative_words_push_away() -> None:
    words = dict(explain(linear_predictor(), "chills fever sneezing headache", "malaria"))
    assert words["sneezing"] == pytest.approx(-1.5)
    assert words["headache"] == pytest.approx(-0.2)
    assert words["chills"] > 0


def test_leave_one_out_negative_word() -> None:
    words = dict(explain(bb_predictor(), "chills fever sneezing all day", "malaria"))
    assert words["sneezing"] < 0


def test_linear_score_is_tfidf_times_weight() -> None:
    p = Predictor("baseline", model=FakePipeline(VOCAB, COEF, value=0.25))
    words = dict(explain(p, TEXT, "malaria"))
    assert words["chills"] == pytest.approx(0.25 * 3.0)
    assert words["sweating"] == pytest.approx(0.25 * 1.0)


# --------------------------------------------------------------------------- #
# Limits: at most 5 positive and 3 negative words
# --------------------------------------------------------------------------- #
MANY_WORDS = [f"w{i}" for i in range(12)]
MANY_COEF = np.array([[0.0] * 12,
                      [6, 5, 4, 3, 2, 1, 0.5, -1, -2, -3, -4, -5],
                      [0.0] * 12])
MANY_TEXT = " ".join(MANY_WORDS)


def test_linear_limits() -> None:
    p = Predictor("baseline", model=FakePipeline(MANY_WORDS, MANY_COEF))
    words = explain(p, MANY_TEXT, "malaria")
    pos = [w for w, s in words if s > 0]
    neg = [w for w, s in words if s < 0]
    assert pos == ["w0", "w1", "w2", "w3", "w4"]
    assert neg == ["w9", "w10", "w11"]  # the three strongest, strongest push away last


def test_leave_one_out_limits() -> None:
    weights = {w: [0, float(c), 0] for w, c in zip(MANY_WORDS, MANY_COEF[1])}
    words = explain(bb_predictor(weights), MANY_TEXT, "malaria")
    assert sum(s > 0 for _, s in words) <= 5
    assert sum(s < 0 for _, s in words) <= 3
    assert len(words) == 8


def test_custom_limits() -> None:
    p = Predictor("baseline", model=FakePipeline(MANY_WORDS, MANY_COEF))
    words = explain(p, MANY_TEXT, "malaria", max_positive=2, max_negative=1)
    assert [w for w, _ in words] == ["w0", "w1", "w11"]


def test_fewer_words_than_limits() -> None:
    words = explain(linear_predictor(), "I have chills today", "malaria")
    assert words == [("chills", pytest.approx(3.0))]


# --------------------------------------------------------------------------- #
# Words with no effect are ignored
# --------------------------------------------------------------------------- #
def test_linear_ignores_zero_weight_and_unknown_words() -> None:
    words = dict(explain(linear_predictor(), "chills at night zzz unknownword", "malaria"))
    assert "night" not in words  # weight 0 for malaria
    assert "zzz" not in words and "unknownword" not in words


def test_leave_one_out_ignores_no_effect_words() -> None:
    words = dict(explain(bb_predictor(), "I have chills and fever", "malaria"))
    assert set(words) == {"chills", "fever"}


def test_select_top_drops_tiny_scores() -> None:
    assert select_top({"a": 0.5, "b": 1e-9, "c": -1e-9, "d": -0.2}) == [("a", 0.5), ("d", -0.2)]


# --------------------------------------------------------------------------- #
# Leave-one-out details
# --------------------------------------------------------------------------- #
def test_leave_one_out_uses_one_batched_call() -> None:
    p = bb_predictor()
    explain(p, TEXT, "malaria")
    assert p.model.calls == [1 + len(set(TEXT.lower().split()))]


def test_leave_one_out_merges_repeats_and_strips_punctuation() -> None:
    words = dict(explain(bb_predictor(), "Chills, chills and more CHILLS!", "malaria"))
    assert list(words) == ["chills"]


def test_leave_one_out_score_is_probability_drop() -> None:
    p = bb_predictor()
    text = "chills fever today"
    full = p.model.predict_proba([text])[0][1]
    without = p.model.predict_proba(["fever today"])[0][1]
    assert dict(explain(p, text, "malaria"))["chills"] == pytest.approx(full - without)


# --------------------------------------------------------------------------- #
# Filler words: hidden for leave-one-out only, scores unchanged
# --------------------------------------------------------------------------- #
FILLER_TEXT = "I have chills and fever every few days"
FILLER_WEIGHTS = {"i": [0, -1, 0], "have": [0, 2, 0], "and": [0, -0.5, 0],
                  "every": [0, -2, 0], "chills": [0, 3, 0], "fever": [0, 1, 0],
                  "days": [0, 0.5, 0]}


def test_leave_one_out_hides_filler_words() -> None:
    words = dict(explain(bb_predictor(FILLER_WEIGHTS), FILLER_TEXT, "malaria"))
    assert set(words) == {"chills", "fever", "days"}
    assert not set(words) & ex.FILLER_WORDS


def test_hiding_filler_keeps_scores_unchanged() -> None:
    p = bb_predictor(FILLER_WEIGHTS)
    shown = dict(explain(p, FILLER_TEXT, "malaria"))
    all_words = dict(explain(p, FILLER_TEXT, "malaria", hide_filler=False))
    assert {"i", "have", "and", "every"} <= set(all_words)  # they do have an effect
    for word, score in shown.items():
        assert score == all_words[word]


def test_hidden_filler_frees_a_slot_for_a_content_word() -> None:
    # moderate weights so probabilities don't saturate at 1.0
    weights = {"have": [0, 1.5, 0], "chills": [0, 1, 0], "fever": [0, 0.3, 0]}
    p = bb_predictor(weights)
    unfiltered = explain(p, "have chills fever", "malaria", max_positive=1, hide_filler=False)
    assert unfiltered[0][0] == "have"  # the filler word would take the only slot...
    words = explain(p, "have chills fever", "malaria", max_positive=1)
    assert words[0][0] == "chills"     # ...but hidden, the next content word gets it


def test_negations_are_not_filler() -> None:
    assert not {"no", "not", "never", "without", "can't"} & ex.FILLER_WORDS


def test_baseline_never_hides_filler_words() -> None:
    vocab = ["have", "every", "chills"]
    coef = np.array([[0, 0, 0], [0.7, -0.4, 3.0], [0, 0, 0]])
    p = Predictor("baseline", model=FakePipeline(vocab, coef))
    words = explain(p, FILLER_TEXT, "malaria")
    assert words == [("chills", 3.0), ("have", 0.7), ("every", -0.4)]
    assert explain(p, FILLER_TEXT, "malaria", hide_filler=False) == words


# --------------------------------------------------------------------------- #
# Exactness against real scikit-learn objects (weights set by hand, no training)
# --------------------------------------------------------------------------- #
def test_linear_scores_sum_to_decision_function() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    vec = TfidfVectorizer().fit(["chills fever sweating", "sneezing headache fever", "night"])
    clf = LogisticRegression()
    n = len(vec.get_feature_names_out())
    rng = np.random.default_rng(0)
    clf.coef_ = rng.normal(size=(3, n))
    clf.intercept_ = rng.normal(size=3)
    clf.classes_ = np.array(LABELS)
    pipe = Pipeline([("tfidf", vec), ("clf", clf)])

    text = "chills and fever with sweating"
    for k, label in enumerate(LABELS):
        scores = ex.linear_scores(vec, clf, text, k)
        expected = clf.decision_function(vec.transform([text]))[0][k] - clf.intercept_[k]
        assert sum(scores.values()) == pytest.approx(expected)
    assert explain(Predictor("baseline", model=pipe), text, "malaria")


def test_binary_classifier_coef_sign() -> None:
    p = Predictor("baseline", model=FakePipeline(["chills"], np.array([[2.0]]),
                                                 classes=["cold", "malaria"]))
    assert explain(p, "I have chills today", "malaria") == [("chills", 2.0)]
    assert explain(p, "I have chills today", "cold") == [("chills", -2.0)]


# --------------------------------------------------------------------------- #
# Errors and command line
# --------------------------------------------------------------------------- #
def test_unknown_label() -> None:
    with pytest.raises(ValueError, match="Unknown label"):
        explain(linear_predictor(), TEXT, "measles")


def test_short_text() -> None:
    with pytest.raises(ValueError, match="at least 3 words"):
        explain(bb_predictor(), "chills", "malaria")


def test_cli_prints_top_prediction_and_words(monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(predict, "load_model",
                        lambda name, path=None: FakePipeline(VOCAB, COEF))
    assert ex.main(["--model", "baseline", TEXT]) == 0
    out = capsys.readouterr().out
    assert "not a medical diagnosis" in out
    assert "Top prediction: malaria" in out
    assert "chills" in out and "+3.0000" in out


def test_cli_shows_red_flags_first(monkeypatch: pytest.MonkeyPatch,
                                   capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(predict, "load_model",
                        lambda name, path=None: FakePipeline(VOCAB, COEF))
    assert ex.main(["I have chills and I passed out"]) == 0
    out = capsys.readouterr().out
    assert out.index("!!! EMERGENCY") < out.index("Top prediction")
