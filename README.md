# Learning Agency Lab Automated Essay Scoring 2

Local solution workspace for the AI Coding Gym MLE-bench challenge. The task is
to predict an integer essay score from 1 to 6 from `full_text`. Submissions are
evaluated with quadratic weighted kappa (QWK), where larger mistakes receive a
larger penalty.

## Environment

The shared Python 3.12 virtual environment lives one directory above this
repository:

```bash
source ../.venv/bin/activate
uv pip install -r requirements.txt
```

## Data

Downloaded data is intentionally ignored by Git and is stored in `data/raw/`:

- `train.csv`: 15,576 labeled essays (`essay_id`, `full_text`, `score`)
- `test.csv`: 1,731 unlabeled essays (`essay_id`, `full_text`)
- `sample_submission.csv`: required submission IDs and column order
- `description.md`: original competition description

The local integrity check found no missing texts, duplicate IDs, exact duplicate
essays, or train/test ID overlap. Scores are valid integers from 1 through 6.

## Project layout

```text
.
├── data/raw/          # downloaded challenge data (ignored)
├── src/               # reusable EDA, training, and inference code
├── reports/           # reproducible EDA tables, report, and figures
├── artifacts/folds/   # immutable local validation-fold mapping (ignored)
├── artifacts/oof/     # out-of-fold predictions (ignored)
├── artifacts/models/  # trained models (ignored)
├── submissions/       # generated submission CSV files
├── requirements.txt
└── README.md
```

## Reproduce the EDA

From this repository, run:

```bash
../.venv/bin/python -m src.eda
```

The main outputs are `reports/eda_report.md`, `reports/eda_summary.json`,
`reports/near_duplicate_candidates.csv`, and
`reports/figures/eda_overview.png`.

## Create the fixed validation folds

After EDA, generate the deterministic grouped five-fold mapping:

```bash
../.venv/bin/python -m src.make_folds
```

The assignment is saved to `artifacts/folds/fold_assignments.csv`. Once present,
the command verifies it and refuses to silently overwrite a different mapping.
The audit is written to `reports/validation_report.md` and
`reports/validation_summary.json`. The official metric helper is
`src.metrics.quadratic_weighted_kappa`.

## Train the TF-IDF baseline

Train fold-local word and character TF-IDF Ridge models with:

```bash
../.venv/bin/python -m src.train_tfidf
```

This writes complete OOF predictions to `artifacts/oof/tfidf_ridge_oof.csv`,
averaged test predictions to `artifacts/oof/tfidf_ridge_test.csv`, a detailed
report to `reports/tfidf_baseline_report.md`, and the fixed-threshold submission
to `submissions/tfidf_ridge_fixed.csv`.

## Blend explicit essay-style features

After the TF-IDF baseline exists, train a fold-local gradient-boosting model on
length and style features, blend its continuous predictions with TF-IDF, and
calibrate five ordered score thresholds with:

```bash
../.venv/bin/python -m src.train_tfidf_style_blend
```

This writes its audit artifacts under `artifacts/oof/`, a validation report to
`reports/tfidf_style_blend_report.md`, and the calibrated submission to
`submissions/tfidf_style_blend_calibrated.csv`.

## Planned workflow

1. Explore distributions, text lengths, and class imbalance.
2. Establish five-fold stratified validation with the official QWK metric.
3. Train a word-and-character TF-IDF regression baseline.
4. Calibrate five ordered score thresholds using out-of-fold predictions.
5. Add statistical features and, if useful, a Transformer model.
6. Validate the final CSV against `sample_submission.csv` and submit it through
   the AI Coding Gym CLI.

## Fair-use constraint

Use only labels in the downloaded `train.csv`. Do not recover hidden test labels
from the original Kaggle training data by matching essay IDs or text; that would
be target leakage rather than a model solution.
