# AI Health Assistant

A symptom-to-condition classifier comparing a TF-IDF + Logistic Regression baseline with fine-tuned DistilBERT.

> **Disclaimer:** This is an educational project, not a medical diagnosis tool. It must not be used to make health decisions. Always consult a qualified doctor.

## Status: work in progress

**Done**
- Data loading, cleaning and leakage removal (duplicate texts in train dropped; train texts that also appear in the test set dropped)
- Stratified train/validation split with a fixed random seed; the official test set is kept untouched
- Baseline: TF-IDF + Logistic Regression, tuned on the validation set and evaluated once on the test set

**Next**
- Fine-tuned DistilBERT and comparison with the baseline
- Safety checks (emergency red flags, low-confidence warning, disclaimer)
- Explainability (which words drove each prediction)
- Streamlit app

## Dataset

[`gretelai/symptom_to_diagnosis`](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) on Hugging Face (Apache 2.0 licence).

- 22 conditions, one label per symptom description
- The descriptions were rewritten by an LLM to sound like patients, so the wording is cleaner and more consistent than real patient text. Results on real patients would very likely be lower than the numbers below.

After cleaning, the official train set went from 853 to 848 rows (4 duplicates and 1 test-overlapping text removed):

| Split | Rows |
|---|---|
| Train | 720 |
| Validation | 128 |
| Test (official, untouched) | 212 |

## Baseline results so far

TF-IDF (single words) + Logistic Regression (C = 10), chosen by validation macro-F1 from a small grid. Training takes about 0.2 seconds on a laptop CPU.

| Split | Accuracy | Macro-F1 |
|---|---|---|
| Validation (128 rows) | 0.9453 | 0.9480 |
| Test (212 rows) | 0.9387 | 0.9392 |

Full metrics, the grid-search table and confusion matrices are in [`reports/`](reports/).

## How to run

Requires Python 3.11+.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Load, clean and split the data; writes data/labels.json and reports/data_summary.json
python -m src.data

# 4. Train and evaluate the baseline; writes models/baseline.joblib and reports/baseline_*
python -m src.baseline

# 5. Run the tests
pytest
```
