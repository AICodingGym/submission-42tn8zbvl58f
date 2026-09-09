"""Train a leakage-safe word + character TF-IDF Ridge baseline."""

from __future__ import annotations

import gc
import hashlib
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import FeatureUnion

try:
    from src.config import (
        FOLD_ASSIGNMENTS_PATH,
        ID_COLUMN,
        N_SPLITS,
        OOF_DIR,
        RANDOM_SEED,
        REPORTS_DIR,
        SAMPLE_SUBMISSION_PATH,
        SCORE_MAX,
        SCORE_MIN,
        SUBMISSIONS_DIR,
        TARGET_COLUMN,
        TEST_PATH,
        TEXT_COLUMN,
        TRAIN_PATH,
    )
    from src.metrics import DEFAULT_THRESHOLDS, apply_ordered_thresholds
    from src.metrics import quadratic_weighted_kappa
except ModuleNotFoundError:  # Support `python src/train_tfidf.py`.
    from config import (  # type: ignore[no-redef]
        FOLD_ASSIGNMENTS_PATH,
        ID_COLUMN,
        N_SPLITS,
        OOF_DIR,
        RANDOM_SEED,
        REPORTS_DIR,
        SAMPLE_SUBMISSION_PATH,
        SCORE_MAX,
        SCORE_MIN,
        SUBMISSIONS_DIR,
        TARGET_COLUMN,
        TEST_PATH,
        TEXT_COLUMN,
        TRAIN_PATH,
    )
    from metrics import (  # type: ignore[no-redef]
        DEFAULT_THRESHOLDS,
        apply_ordered_thresholds,
        quadratic_weighted_kappa,
    )


VALIDATION_SUMMARY_PATH = REPORTS_DIR / "validation_summary.json"
OOF_PATH = OOF_DIR / "tfidf_ridge_oof.csv"
TEST_PREDICTIONS_PATH = OOF_DIR / "tfidf_ridge_test.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "tfidf_ridge_fixed.csv"
FOLD_METRICS_PATH = REPORTS_DIR / "tfidf_baseline_fold_metrics.csv"
AGGREGATE_METRICS_PATH = REPORTS_DIR / "tfidf_baseline_metrics.csv"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "tfidf_baseline_confusion_matrix.csv"
PREDICTION_DISTRIBUTION_PATH = (
    REPORTS_DIR / "tfidf_baseline_prediction_distribution.csv"
)
BASELINE_SUMMARY_PATH = REPORTS_DIR / "tfidf_baseline_summary.json"
BASELINE_REPORT_PATH = REPORTS_DIR / "tfidf_baseline_report.md"


@dataclass(frozen=True)
class BaselineConfig:
    """Resource-bounded feature and model settings."""

    lowercase: bool = True
    strip_accents: str | None = None
    norm: str = "l2"
    word_analyzer: str = "word"
    word_ngram_min: int = 1
    word_ngram_max: int = 2
    word_min_df: int = 2
    word_max_features: int = 100_000
    char_analyzer: str = "char_wb"
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    char_min_df: int = 3
    char_max_features: int = 120_000
    max_df: float = 0.995
    sublinear_tf: bool = True
    ridge_alphas: tuple[float, ...] = (0.3, 1.0, 3.0)
    ridge_solver: str = "lsqr"
    ridge_tolerance: float = 1e-4
    ridge_max_iterations: int = 5_000


def file_sha256(path: Path) -> str:
    """Return a SHA-256 checksum for data and prediction provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpha_slug(alpha: float) -> str:
    """Create a stable column-name suffix from a Ridge alpha."""

    return f"{alpha:g}".replace("-", "neg").replace(".", "p")


def build_vectorizer(config: BaselineConfig) -> FeatureUnion:
    """Build separate word and raw-character TF-IDF feature spaces."""

    shared: dict[str, Any] = {
        "lowercase": config.lowercase,
        "strip_accents": config.strip_accents,
        "norm": config.norm,
        "max_df": config.max_df,
        "sublinear_tf": config.sublinear_tf,
        "dtype": np.float32,
    }
    word_vectorizer = TfidfVectorizer(
        analyzer=config.word_analyzer,
        ngram_range=(config.word_ngram_min, config.word_ngram_max),
        min_df=config.word_min_df,
        max_features=config.word_max_features,
        token_pattern=r"(?u)\b\w+\b",
        **shared,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer=config.char_analyzer,
        ngram_range=(config.char_ngram_min, config.char_ngram_max),
        min_df=config.char_min_df,
        max_features=config.char_max_features,
        **shared,
    )
    return FeatureUnion(
        [("word", word_vectorizer), ("char", char_vectorizer)],
        n_jobs=1,
    )


def sparse_megabytes(matrix: Any) -> float:
    """Estimate CSR/CSC storage used by one sparse matrix."""

    total_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return float(total_bytes / (1024**2))


def regression_metrics(y_true: np.ndarray, raw_predictions: np.ndarray) -> dict[str, Any]:
    """Calculate continuous errors and fixed-threshold competition score."""

    fixed_predictions = apply_ordered_thresholds(raw_predictions)
    errors = raw_predictions - y_true
    integer_errors = np.abs(fixed_predictions.astype(int) - y_true.astype(int))
    return {
        "qwk_fixed": quadratic_weighted_kappa(y_true, fixed_predictions),
        "mae_raw": float(np.mean(np.abs(errors))),
        "rmse_raw": float(np.sqrt(np.mean(np.square(errors)))),
        "severe_error_count": int((integer_errors >= 2).sum()),
        "prediction_min": float(raw_predictions.min()),
        "prediction_max": float(raw_predictions.max()),
        "prediction_mean": float(raw_predictions.mean()),
        "prediction_std": float(raw_predictions.std(ddof=1)),
    }


def validate_and_load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data and prove that the persisted fold mapping still matches it."""

    required_paths = [
        TRAIN_PATH,
        TEST_PATH,
        SAMPLE_SUBMISSION_PATH,
        FOLD_ASSIGNMENTS_PATH,
        VALIDATION_SUMMARY_PATH,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("missing required files: " + ", ".join(missing_paths))

    validation_summary = json.loads(
        VALIDATION_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    if file_sha256(TRAIN_PATH) != validation_summary.get("train_sha256"):
        raise ValueError("train.csv no longer matches the validation audit")
    if file_sha256(FOLD_ASSIGNMENTS_PATH) != validation_summary.get(
        "assignment_sha256"
    ):
        raise ValueError("fold assignment no longer matches the validation audit")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    if train.columns.tolist() != [ID_COLUMN, TEXT_COLUMN, TARGET_COLUMN]:
        raise ValueError("train.csv has an unexpected schema")
    if test.columns.tolist() != [ID_COLUMN, TEXT_COLUMN]:
        raise ValueError("test.csv has an unexpected schema")
    if sample_submission.columns.tolist() != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError("sample_submission.csv has an unexpected schema")
    if not sample_submission[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError("sample submission IDs do not match test IDs and order")

    folds = pd.read_csv(
        FOLD_ASSIGNMENTS_PATH,
        dtype={ID_COLUMN: str, "group_id": str},
    )
    required_fold_columns = {ID_COLUMN, TARGET_COLUMN, "fold", "group_id"}
    if not required_fold_columns.issubset(folds.columns):
        raise ValueError("fold assignment has an unexpected schema")
    if folds[ID_COLUMN].duplicated().any():
        raise ValueError("fold assignment contains duplicate essay IDs")
    if set(folds["fold"]) != set(range(N_SPLITS)):
        raise ValueError("fold assignment does not contain exactly folds 0..4")

    train = train.assign(_source_order=np.arange(len(train), dtype=int)).merge(
        folds[[ID_COLUMN, TARGET_COLUMN, "fold", "group_id"]].rename(
            columns={TARGET_COLUMN: "fold_file_score"}
        ),
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    train = train.sort_values("_source_order", kind="stable").reset_index(drop=True)
    if train["fold"].isna().any() or len(train) != len(folds):
        raise ValueError("fold assignment IDs do not exactly match train.csv")
    if not train[TARGET_COLUMN].equals(train["fold_file_score"]):
        raise ValueError("fold assignment scores do not match train.csv labels")
    if train.groupby("group_id")["fold"].nunique().max() != 1:
        raise ValueError("a duplicate group crosses folds")
    return train, test, sample_submission


def dataframe_to_markdown(frame: pd.DataFrame, digits: int = 5) -> str:
    """Render a small table without an optional Markdown dependency."""

    def format_value(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    """Fit five fold-local vectorizers and write OOF/test predictions."""

    start_time = time.perf_counter()
    config = BaselineConfig()
    train, test, sample_submission = validate_and_load_data()
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    fold_values = train["fold"].to_numpy(dtype=np.int8)

    oof_predictions = {
        alpha: np.full(len(train), np.nan, dtype=np.float64)
        for alpha in config.ridge_alphas
    }
    test_predictions_by_fold = {
        alpha: np.full((N_SPLITS, len(test)), np.nan, dtype=np.float64)
        for alpha in config.ridge_alphas
    }
    fold_rows: list[dict[str, Any]] = []

    for fold in range(N_SPLITS):
        train_indices = np.flatnonzero(fold_values != fold)
        validation_indices = np.flatnonzero(fold_values == fold)
        training_groups = set(train.loc[train_indices, "group_id"])
        validation_groups = set(train.loc[validation_indices, "group_id"])
        if training_groups & validation_groups:
            raise RuntimeError(f"fold {fold} has duplicate-group leakage")

        print(
            f"[fold {fold + 1}/{N_SPLITS}] fitting TF-IDF on "
            f"{len(train_indices):,} essays...",
            flush=True,
        )
        vectorizer_start = time.perf_counter()
        vectorizer = build_vectorizer(config)
        x_train = vectorizer.fit_transform(
            train.loc[train_indices, TEXT_COLUMN]
        )
        x_validation = vectorizer.transform(
            train.loc[validation_indices, TEXT_COLUMN]
        )
        x_test = vectorizer.transform(test[TEXT_COLUMN])
        vectorizer_seconds = time.perf_counter() - vectorizer_start
        if not all(
            sparse.issparse(matrix)
            for matrix in (x_train, x_validation, x_test)
        ):
            raise RuntimeError("TF-IDF unexpectedly produced a dense matrix")

        fitted_transformers = dict(vectorizer.transformer_list)
        word_features = len(fitted_transformers["word"].vocabulary_)
        char_features = len(fitted_transformers["char"].vocabulary_)
        matrix_megabytes = (
            sparse_megabytes(x_train)
            + sparse_megabytes(x_validation)
            + sparse_megabytes(x_test)
        )
        print(
            f"[fold {fold + 1}/{N_SPLITS}] {x_train.shape[1]:,} features, "
            f"{matrix_megabytes:.1f} MiB sparse matrices, "
            f"vectorization {vectorizer_seconds:.1f}s",
            flush=True,
        )

        for alpha in config.ridge_alphas:
            model_start = time.perf_counter()
            model = Ridge(
                alpha=alpha,
                fit_intercept=True,
                solver=config.ridge_solver,
                tol=config.ridge_tolerance,
                max_iter=config.ridge_max_iterations,
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_train, y[train_indices])
            if any(
                issubclass(warning.category, ConvergenceWarning)
                for warning in caught_warnings
            ):
                raise RuntimeError(
                    f"Ridge failed to converge for fold={fold}, alpha={alpha:g}"
                )
            if not np.isfinite(model.coef_).all():
                raise RuntimeError("Ridge coefficients contain NaN or Inf")
            validation_predictions = np.asarray(
                model.predict(x_validation), dtype=np.float64
            )
            test_predictions = np.asarray(model.predict(x_test), dtype=np.float64)
            model_seconds = time.perf_counter() - model_start
            if not np.isfinite(validation_predictions).all():
                raise RuntimeError("validation predictions contain NaN or Inf")
            if not np.isfinite(test_predictions).all():
                raise RuntimeError("test predictions contain NaN or Inf")
            if np.isfinite(oof_predictions[alpha][validation_indices]).any():
                raise RuntimeError("OOF rows were predicted more than once")
            oof_predictions[alpha][validation_indices] = validation_predictions
            test_predictions_by_fold[alpha][fold] = test_predictions

            metrics = regression_metrics(y[validation_indices], validation_predictions)
            fold_rows.append(
                {
                    "fold": fold,
                    "alpha": alpha,
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "word_features": word_features,
                    "char_features": char_features,
                    "total_features": x_train.shape[1],
                    "train_nonzeros": x_train.nnz,
                    "sparse_matrix_megabytes": matrix_megabytes,
                    "vectorizer_seconds": vectorizer_seconds,
                    "model_seconds": model_seconds,
                    "ridge_iterations": (
                        int(np.asarray(model.n_iter_).max())
                        if model.n_iter_ is not None
                        else None
                    ),
                    **metrics,
                }
            )
            print(
                f"[fold {fold + 1}/{N_SPLITS}] alpha={alpha:g}: "
                f"QWK={metrics['qwk_fixed']:.5f}, "
                f"RMSE={metrics['rmse_raw']:.5f}, fit {model_seconds:.1f}s",
                flush=True,
            )
            del model, validation_predictions, test_predictions

        del vectorizer, x_train, x_validation, x_test
        gc.collect()

    fold_metrics = pd.DataFrame(fold_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for alpha in config.ridge_alphas:
        raw_predictions = oof_predictions[alpha]
        if not np.isfinite(raw_predictions).all():
            raise RuntimeError(f"alpha={alpha:g} has incomplete OOF predictions")
        overall_metrics = regression_metrics(y, raw_predictions)
        alpha_fold_metrics = fold_metrics.loc[fold_metrics["alpha"].eq(alpha)]
        aggregate_rows.append(
            {
                "alpha": alpha,
                **overall_metrics,
                "fold_qwk_mean": float(alpha_fold_metrics["qwk_fixed"].mean()),
                "fold_qwk_std": float(
                    alpha_fold_metrics["qwk_fixed"].std(ddof=1)
                ),
                "fold_rmse_mean": float(alpha_fold_metrics["rmse_raw"].mean()),
                "fold_rmse_std": float(alpha_fold_metrics["rmse_raw"].std(ddof=1)),
            }
        )
    aggregate_metrics = pd.DataFrame(aggregate_rows)
    selected_row = aggregate_metrics.sort_values(
        ["qwk_fixed", "rmse_raw"], ascending=[False, True]
    ).iloc[0]
    selected_alpha = float(selected_row["alpha"])
    aggregate_metrics["selected_by_fixed_qwk"] = aggregate_metrics["alpha"].eq(
        selected_alpha
    )

    selected_oof_raw = oof_predictions[selected_alpha]
    selected_oof_fixed = apply_ordered_thresholds(selected_oof_raw)
    averaged_test_predictions = {
        alpha: test_predictions_by_fold[alpha].mean(axis=0)
        for alpha in config.ridge_alphas
    }
    selected_test_raw = averaged_test_predictions[selected_alpha]
    selected_test_fixed = apply_ordered_thresholds(selected_test_raw)

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    oof_frame = train[[ID_COLUMN, TARGET_COLUMN, "fold", "group_id"]].copy()
    for alpha in config.ridge_alphas:
        oof_frame[f"pred_raw_alpha_{alpha_slug(alpha)}"] = oof_predictions[alpha]
    oof_frame["pred_raw"] = selected_oof_raw
    oof_frame["pred_fixed"] = selected_oof_fixed.astype(int)
    oof_frame.to_csv(OOF_PATH, index=False, float_format="%.10f")

    test_prediction_frame = test[[ID_COLUMN]].copy()
    for fold in range(N_SPLITS):
        test_prediction_frame[f"pred_fold_{fold}"] = test_predictions_by_fold[
            selected_alpha
        ][fold]
    for alpha in config.ridge_alphas:
        test_prediction_frame[f"pred_raw_alpha_{alpha_slug(alpha)}"] = (
            averaged_test_predictions[alpha]
        )
    test_prediction_frame["pred_raw"] = selected_test_raw
    test_prediction_frame["pred_fixed"] = selected_test_fixed.astype(int)
    test_prediction_frame.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
        float_format="%.10f",
    )

    submission = sample_submission[[ID_COLUMN]].copy()
    submission[TARGET_COLUMN] = selected_test_fixed.astype(int)
    submission.to_csv(SUBMISSION_PATH, index=False)

    saved_oof = pd.read_csv(OOF_PATH, dtype={ID_COLUMN: str})
    saved_test = pd.read_csv(TEST_PREDICTIONS_PATH, dtype={ID_COLUMN: str})
    saved_submission = pd.read_csv(SUBMISSION_PATH, dtype={ID_COLUMN: str})
    if len(saved_oof) != len(train) or not saved_oof[ID_COLUMN].is_unique:
        raise RuntimeError("saved OOF file failed row or ID validation")
    if not saved_oof[ID_COLUMN].equals(train[ID_COLUMN]):
        raise RuntimeError("saved OOF IDs do not match train order")
    if not saved_test[ID_COLUMN].equals(test[ID_COLUMN]):
        raise RuntimeError("saved test predictions do not match test order")
    if saved_submission.columns.tolist() != [ID_COLUMN, TARGET_COLUMN]:
        raise RuntimeError("saved submission has incorrect columns")
    if not saved_submission[ID_COLUMN].equals(sample_submission[ID_COLUMN]):
        raise RuntimeError("saved submission IDs do not match sample order")
    if not saved_submission[TARGET_COLUMN].between(SCORE_MIN, SCORE_MAX).all():
        raise RuntimeError("saved submission contains scores outside 1..6")
    if not np.array_equal(
        saved_submission[TARGET_COLUMN],
        saved_submission[TARGET_COLUMN].astype(int),
    ):
        raise RuntimeError("saved submission scores are not integers")

    fold_metrics.to_csv(FOLD_METRICS_PATH, index=False, float_format="%.10f")
    aggregate_metrics.to_csv(
        AGGREGATE_METRICS_PATH,
        index=False,
        float_format="%.10f",
    )
    confusion = confusion_matrix(
        y,
        selected_oof_fixed,
        labels=list(range(SCORE_MIN, SCORE_MAX + 1)),
    )
    confusion_frame = pd.DataFrame(
        confusion,
        index=[f"true_{score}" for score in range(SCORE_MIN, SCORE_MAX + 1)],
        columns=[f"pred_{score}" for score in range(SCORE_MIN, SCORE_MAX + 1)],
    )
    confusion_frame.to_csv(CONFUSION_MATRIX_PATH, index_label="true_score")

    true_counts = pd.Series(y).value_counts().reindex(range(1, 7), fill_value=0)
    oof_counts = (
        pd.Series(selected_oof_fixed)
        .value_counts()
        .reindex(range(1, 7), fill_value=0)
    )
    test_counts = (
        pd.Series(selected_test_fixed)
        .value_counts()
        .reindex(range(1, 7), fill_value=0)
    )
    prediction_distribution = pd.DataFrame(
        {
            "score": range(SCORE_MIN, SCORE_MAX + 1),
            "train_true_count": true_counts.to_numpy(dtype=int),
            "oof_fixed_count": oof_counts.to_numpy(dtype=int),
            "test_fixed_count": test_counts.to_numpy(dtype=int),
        }
    )
    prediction_distribution["train_true_percent"] = (
        100.0 * prediction_distribution["train_true_count"] / len(train)
    )
    prediction_distribution["oof_fixed_percent"] = (
        100.0 * prediction_distribution["oof_fixed_count"] / len(train)
    )
    prediction_distribution["test_fixed_percent"] = (
        100.0 * prediction_distribution["test_fixed_count"] / len(test)
    )
    prediction_distribution.to_csv(
        PREDICTION_DISTRIBUTION_PATH,
        index=False,
        float_format="%.10f",
    )

    elapsed_seconds = time.perf_counter() - start_time
    selected_metrics = regression_metrics(y, selected_oof_raw)
    selected_fold_metrics = fold_metrics.loc[
        fold_metrics["alpha"].eq(selected_alpha)
    ].copy()
    summary = {
        "model": "word_char_tfidf_ridge",
        "selection_rule": "highest concatenated OOF fixed-threshold QWK; RMSE breaks ties",
        "selected_alpha": selected_alpha,
        "fixed_thresholds": list(DEFAULT_THRESHOLDS),
        "config": asdict(config),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "runtime_seconds": elapsed_seconds,
        "selected_oof_metrics": {
            **selected_metrics,
            "fold_qwk_mean": float(selected_fold_metrics["qwk_fixed"].mean()),
            "fold_qwk_std": float(selected_fold_metrics["qwk_fixed"].std(ddof=1)),
        },
        "artifacts": {
            "fold_assignment_sha256": file_sha256(FOLD_ASSIGNMENTS_PATH),
            "oof_sha256": file_sha256(OOF_PATH),
            "test_predictions_sha256": file_sha256(TEST_PREDICTIONS_PATH),
            "submission_sha256": file_sha256(SUBMISSION_PATH),
        },
        "submission_score_counts": {
            str(int(score)): int(count)
            for score, count in test_counts.items()
        },
    }
    BASELINE_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = f"""# TF-IDF Ridge Baseline

## Result

- Features: word `(1,2)` TF-IDF plus character-within-word `(3,5)` TF-IDF.
- Validation: the persisted `{N_SPLITS}` grouped folds; each vectorizer is fitted
  only on that fold's training essays.
- Alpha selection: highest concatenated OOF fixed-threshold QWK among
  `{list(config.ridge_alphas)}`; selected **{selected_alpha:g}**.
- Concatenated OOF QWK with fixed thresholds `{list(DEFAULT_THRESHOLDS)}`:
  **{selected_metrics['qwk_fixed']:.5f}**.
- Fold QWK: **{selected_fold_metrics['qwk_fixed'].mean():.5f} +/-
  {selected_fold_metrics['qwk_fixed'].std(ddof=1):.5f}**.
- OOF RMSE: **{selected_metrics['rmse_raw']:.5f}**; MAE:
  **{selected_metrics['mae_raw']:.5f}**.
- Severe OOF errors (absolute integer error at least 2):
  **{selected_metrics['severe_error_count']:,}**.
- Runtime: **{elapsed_seconds:.1f} seconds**.

## Alpha comparison

{dataframe_to_markdown(aggregate_metrics)}

## Selected model by fold

{dataframe_to_markdown(selected_fold_metrics[['fold', 'qwk_fixed', 'mae_raw', 'rmse_raw', 'severe_error_count', 'total_features', 'vectorizer_seconds', 'model_seconds']])}

## Prediction distribution

{dataframe_to_markdown(prediction_distribution)}

## Leakage controls

The fold file checksum is checked before fitting. Training and validation groups
must be disjoint. Word and character vocabularies are fitted separately inside
each fold; validation and test texts are transform-only. The score thresholds
remain the fixed half-integers in this baseline and have not been optimized.

The ready-to-submit file is `submissions/tfidf_ridge_fixed.csv`.
"""
    BASELINE_REPORT_PATH.write_text(report, encoding="utf-8")

    print("\nBaseline complete", flush=True)
    print(
        f"Selected alpha={selected_alpha:g}; OOF QWK={selected_metrics['qwk_fixed']:.5f}; "
        f"RMSE={selected_metrics['rmse_raw']:.5f}",
        flush=True,
    )
    print(f"OOF predictions: {OOF_PATH}", flush=True)
    print(f"Test predictions: {TEST_PREDICTIONS_PATH}", flush=True)
    print(f"Submission: {SUBMISSION_PATH}", flush=True)
    print(f"Report: {BASELINE_REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
