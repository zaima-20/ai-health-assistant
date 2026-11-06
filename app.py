"""Streamlit user interface for the AI Health Assistant (UI only, no business logic).

Run with:  streamlit run app.py

Privacy: the symptom text lives only in this browser session's memory (the text
box). It is never written to a file, logged, printed or cached. The last result
kept in ``st.session_state`` holds predictions, the safety report and word
scores only, never the text itself, and it is cleared when the text or model changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src import predict
from src.config import CRISIS_HELPLINE_TEXT
from src.disease_info import DiseaseInfo, get_disease_info
from src.display import highlight_html, load_test_scores, scores_table_markdown
from src.explain import explain
from src.predict import Condition, PredictionResult, Predictor
from src.safety import (
    DISCLAIMER,
    EMERGENCY,
    URGENT,
    WARNING_CONTEXT_NOTE,
    SafetyReport,
    assess,
)

MAX_CHARS = 1000
RESULT_KEY = "last_result"
MODEL_OPTIONS: dict[str, str] = {"Baseline (fast)": "baseline", "DistilBERT": "bert"}
EXAMPLES: dict[str, str] = {
    "Fever and chills": "I have had a high fever, chills and heavy sweating every few days "
                        "since I came back from a trip.",
    "Itchy skin rash": "I have red, itchy patches of dry, flaky skin on my elbows and knees "
                       "that sometimes crack.",
    "Burning chest after meals": "I get a burning feeling in my throat after eating, a sour "
                                 "taste in my mouth and I burp a lot, especially at night.",
}
RISK_COLOURS = {"low": "green", "moderate": "orange", "high": "red"}
EMERGENCY_SIGNS_HEADING = "Get emergency help now if"
URGENT_SIGNS_HEADING = "See a doctor urgently if"
ROUTINE_SIGNS_HEADING = "See a doctor if"


@dataclass(frozen=True)
class AnalysisResult:
    """What is kept between reruns. Deliberately has no field for the user's text."""

    report: SafetyReport
    prediction: PredictionResult | None
    words: tuple[tuple[str, float], ...]
    message: str | None  # friendly error, e.g. "please write at least 3 words"


# --------------------------------------------------------------------------- #
# Cached models and session state
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def get_predictor(model_name: str) -> Predictor:
    """Load a model once and reuse it for every request."""
    return Predictor(model_name)


def bert_available() -> bool:
    """True if a trained DistilBERT model folder exists."""
    return (predict.DEFAULT_PATHS["bert"] / "config.json").is_file()


def clear_result() -> None:
    """Forget the last result (called when the text or the model changes)."""
    st.session_state.pop(RESULT_KEY, None)


def set_example(text: str) -> None:
    """Button callback: put an example into the text area and clear old results."""
    st.session_state["symptoms"] = text
    clear_result()


def analyse_text(text: str, model_name: str) -> AnalysisResult:
    """Safety check FIRST, then the model and the explanation."""
    report = assess(text)  # runs before any model
    try:
        predictor = get_predictor(model_name)
        prediction = predictor.predict(text)
        words = tuple(explain(predictor, text, prediction.top.name))
    except (ValueError, FileNotFoundError) as err:
        return AnalysisResult(report, None, (), str(err))
    full = assess(text, prediction)
    return AnalysisResult(full, prediction, words, None)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_banners(report: SafetyReport) -> None:
    """Red emergency and orange urgent messages."""
    if report.level == EMERGENCY:
        for flag in (f for f in report.red_flags if f.level == EMERGENCY):
            st.error(f"**🚨 {flag.name}.** {flag.message}")
            if flag.key == "self_harm":
                st.error(f"### 📞 {CRISIS_HELPLINE_TEXT}")
    for flag in (f for f in report.red_flags if f.level == URGENT):
        st.warning(f"**{flag.name}.** {flag.message}", icon="🟠")


def render_condition(rank: int, condition: Condition) -> None:
    """One card: name, confidence bar, risk label and curated disease info."""
    info: DiseaseInfo | None
    try:
        info = get_disease_info(condition.name)
    except KeyError:
        info = None

    with st.container(border=True):
        st.markdown(f"#### {rank}. {condition.name.title()}")
        st.progress(min(max(condition.confidence, 0.0), 1.0),
                    text=f"Model confidence: {condition.confidence:.1%}")
        if info is None:
            st.caption("No curated information is available for this condition.")
            return
        st.badge(f"{info.risk_level.title()} risk",
                 color=RISK_COLOURS.get(info.risk_level, "gray"))
        st.caption(info.risk_reason)
        st.write(info.description)
        st.markdown(f"**Who to see:** {info.doctor_type}")
        left, right = st.columns(2)
        with left:
            st.markdown("**General self-care**")
            st.markdown("\n".join(f"- {tip}" for tip in info.self_care))
        with right:
            # Each sign sits under the level its source page gives it.
            if info.emergency_signs:
                st.markdown(f"**{EMERGENCY_SIGNS_HEADING}**")
                st.markdown("\n".join(f"- {sign}" for sign in info.emergency_signs))
            if info.urgent_signs:
                st.markdown(f"**{URGENT_SIGNS_HEADING}**")
                st.markdown("\n".join(f"- {sign}" for sign in info.urgent_signs))
            if info.routine_signs:
                st.markdown(f"**{ROUTINE_SIGNS_HEADING}**")
                st.markdown("\n".join(f"- {sign}" for sign in info.routine_signs))
        st.markdown(f"Source: [{info.source}]({info.source})")
        if not info.reviewed:
            st.caption("ℹ️ This information has not yet been reviewed.")


def render_predictions(result: AnalysisResult, text: str) -> None:
    """Low-confidence note, condition cards and the highlighted explanation."""
    assert result.prediction is not None
    if result.report.confidence_message:
        st.info(result.report.confidence_message)
    st.subheader("Possible conditions")
    if result.report.level in (EMERGENCY, URGENT):
        st.info(WARNING_CONTEXT_NOTE, icon="⚠️")
    for rank, condition in enumerate(result.prediction.conditions, start=1):
        render_condition(rank, condition)

    top = result.prediction.top.name
    st.subheader("Why the model suggested this")
    st.markdown(highlight_html(text, dict(result.words)), unsafe_allow_html=True)
    st.caption(f"Green words pushed the prediction towards **{top}**; red words pushed it "
               "away. Stronger colour means a stronger effect.")


def render_result(result: AnalysisResult, text: str, banner) -> None:  # noqa: ANN001
    """Banners at the top, then predictions (hidden in an expander for emergencies)."""
    with banner:
        render_banners(result.report)
    if result.message:
        st.warning(result.message)
    if result.prediction is None:
        return
    if result.report.level == EMERGENCY:
        with st.expander("Show possible conditions anyway", expanded=False):
            render_predictions(result, text)
    else:
        render_predictions(result, text)


def render_about() -> None:
    """Dataset, real test scores and known limitations."""
    with st.expander("About this app"):
        st.markdown(
            "**Dataset:** [gretelai/symptom_to_diagnosis]"
            "(https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) "
            "(Apache 2.0): 1,065 symptom descriptions covering 22 conditions, "
            "with an official test set of 212 descriptions."
        )
        scores = load_test_scores()
        if scores:
            st.markdown("**Scores on the held-out test set:**")
            st.markdown(scores_table_markdown(scores))
        else:
            st.caption("Test scores are not available yet.")
        st.markdown(
            "**Known limitations**\n"
            "- The descriptions were rewritten by a language model to sound like patients, "
            "so results on real patients would be lower.\n"
            "- It only knows these 22 conditions; anything else is forced into one of them.\n"
            "- Words it never saw in training (for example *wheeze*) may be ignored or "
            "misread.\n"
            "- The emergency check uses simple phrase matching: it can give false alarms, "
            "and it cannot understand negation (\"no chest pain\" still triggers it).\n"
            "- Disease information comes from public health websites and is general, not "
            "personal medical advice."
        )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AI Health Assistant", page_icon="🩺", layout="centered")
st.title("AI Health Assistant")
banner = st.container()  # emergency / urgent banners go here, at the very top
st.warning(DISCLAIMER, icon="⚠️")

with st.sidebar:
    st.header("Settings")
    has_bert = bert_available()
    options = list(MODEL_OPTIONS) if has_bert else ["Baseline (fast)"]
    choice = st.radio("Model", options, index=0, key="model_choice", on_change=clear_result)
    if not has_bert:
        st.info("The DistilBERT model is not available on this machine, so only the "
                "baseline model can be used. Train it with `python -m src.bert_model`.")
    st.caption("Your text is not saved, logged or stored anywhere.")

st.text_area("Describe your symptoms in your own words", key="symptoms",
             max_chars=MAX_CHARS, height=140, on_change=clear_result,
             placeholder="For example: I have had a sore throat, a runny nose and "
                         "sneezing for three days.")
st.caption("Try an example:")
for col, (name, example) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
    col.button(name, on_click=set_example, args=(example,), width="stretch")

if st.button("Analyse", type="primary"):
    st.session_state[RESULT_KEY] = analyse_text(st.session_state.get("symptoms", "") or "",
                                                MODEL_OPTIONS[choice])

if RESULT_KEY in st.session_state:
    render_result(st.session_state[RESULT_KEY], st.session_state.get("symptoms", "") or "",
                  banner)

render_about()
