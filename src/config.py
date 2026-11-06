"""Project-wide settings: paths, random seed, thresholds and model names."""

from pathlib import Path
from typing import Final

# Paths (built relative to the project folder, so they work on any machine)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"

# Reproducibility
RANDOM_SEED: Final[int] = 42

# Data splitting: share of the training data held out for validation
VALIDATION_SIZE: Final[float] = 0.15

# Prediction and safety
CONFIDENCE_THRESHOLD: Final[float] = 0.5
TOP_K: Final[int] = 3

# Shown with the suicidal-thoughts / self-harm red flag.
# TODO(owner): replace the placeholder with a verified helpline for your audience.
CRISIS_HELPLINE_TEXT: Final[str] = "Crisis helpline: [verify helpline]"

# Dataset and model names
DATASET_NAME: Final[str] = "gretelai/symptom_to_diagnosis"
BASE_MODEL_NAME: Final[str] = "distilbert-base-uncased"
