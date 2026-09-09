"""Shared paths and constants for the essay-scoring pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OOF_DIR = ARTIFACTS_DIR / "oof"
MODELS_DIR = ARTIFACTS_DIR / "models"
FOLDS_DIR = ARTIFACTS_DIR / "folds"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
FOLD_ASSIGNMENTS_PATH = FOLDS_DIR / "fold_assignments.csv"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
NEAR_DUPLICATE_CANDIDATES_PATH = REPORTS_DIR / "near_duplicate_candidates.csv"

ID_COLUMN = "essay_id"
TEXT_COLUMN = "full_text"
TARGET_COLUMN = "score"
SCORE_MIN = 1
SCORE_MAX = 6
RANDOM_SEED = 42
N_SPLITS = 5
