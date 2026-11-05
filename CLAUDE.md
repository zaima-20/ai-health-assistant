# AI Health Assistant: Project Spec

This file tells Claude Code what we are building. Read it before every task.

## Goal

A Streamlit app where a user describes their symptoms in plain English, and the app:
1. Predicts the most likely conditions (top 3, with confidence)
2. Shows, for each condition: a short description, which type of doctor to see, a risk level, general self-care tips, and warning signs that need urgent care
3. Explains which words in the description drove the prediction

This is a portfolio project for learning NLP. It is NOT a medical device and must never present itself as a diagnosis.

## Safety rules (most important, never break these)

- Always show a clear disclaimer: "This is an educational tool, not a medical diagnosis. Always consult a qualified doctor."
- Red-flag check runs BEFORE the model. If the text mentions emergency signs (for example chest pain, difficulty breathing, fainting, severe bleeding, confusion, stroke signs, suicidal thoughts), show an urgent "seek emergency care now" message at the top, whatever the model predicts.
- Never give medicine names with doses, prescriptions, or "cures". Self-care tips must be general (rest, fluids, see a doctor if...).
- If the top prediction's confidence is below a threshold (default 0.5), say the app is not confident and recommend seeing a general physician.
- Risk level comes from the curated disease info file plus the red-flag check, never from the model alone.
- Do not store the user's symptom text anywhere (no files, no logs, no database).
- Disease info (description, doctor type, risk, self-care, warning signs) lives in a hand-curated JSON file. Each disease entry has a `source` URL from a reliable public health site (for example WHO, NHS, MedlinePlus, CDC). The user will check every entry before publishing.

## Dataset

- Hugging Face `gretelai/symptom_to_diagnosis` (Apache 2.0 licence)
- 1,065 rows: 853 train, 212 test. Columns: `input_text` (symptom description), `output_text` (diagnosis)
- 22 diagnoses, one label per row (multi-class classification)
- The text was rewritten by an LLM to sound like patients, so results on real patients would be lower. Say this in the README.
- Keep the official test split untouched for final evaluation. Make a validation split from train (stratified, fixed random seed).

## Tech stack

- Python 3.11+
- pandas, scikit-learn (baseline), Hugging Face `datasets` and `transformers`, PyTorch (DistilBERT)
- Streamlit for the UI (no HTML/CSS/JS frameworks)
- matplotlib for plots
- pytest for tests
- Google Colab (free GPU) for transformer training if the laptop has no GPU

## Project structure

```
ai-health-assistant/
├── app.py                    # Streamlit UI only, no business logic
├── src/
│   ├── __init__.py
│   ├── config.py             # paths, random seed, thresholds, model names
│   ├── data.py               # load dataset, make train/val/test splits, label encoding
│   ├── baseline.py           # TF-IDF + Logistic Regression: train, save, load
│   ├── bert_model.py         # DistilBERT fine-tuning, save, load
│   ├── evaluate.py           # accuracy, macro-F1, per-class report, confusion matrix
│   ├── predict.py            # one predict(text) interface for either model, returns top 3
│   ├── explain.py            # which words pushed the prediction
│   ├── safety.py             # red-flag detection, confidence check
│   └── disease_info.py       # load and validate disease_info.json
├── data/
│   └── disease_info.json     # hand-curated info for the 22 diseases, with sources
├── notebooks/
│   └── train_bert_colab.ipynb  # optional: train DistilBERT on Colab GPU
├── models/                   # saved models (gitignored, too large)
├── reports/                  # metrics JSON + confusion matrix images (committed)
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Phases

1. Setup: folders, requirements, config
2. Data: load, explore, split, save label list
3. Baseline: TF-IDF + Logistic Regression, evaluate
4. Transformer: fine-tune DistilBERT, evaluate, compare with baseline
5. Prediction interface: one `predict()` for either model, top 3 with confidence
6. Safety: red flags, low-confidence handling, disclaimer
7. Disease info: curated JSON with sources, validation
8. Explainability: important words for each prediction
9. Streamlit app
10. README, results table, GitHub, blog post

## Rules for Claude Code

- Work in small steps; one phase or module per task unless asked otherwise
- Fix the random seed everywhere so results are reproducible
- Report real numbers only; never invent metrics. Save metrics to reports/
- Add type hints and docstrings to every function
- Write pytest tests for data, safety, disease_info and predict (use tiny fake models in tests; no training in tests)
- After changing code, run the tests
- Keep `app.py` thin: UI only, logic lives in `src/`
- Explain what you changed in plain words when done
