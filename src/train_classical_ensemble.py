"""Train the reproducible three-way classical essay-scoring ensemble.

The pipeline fits every text vectorizer and estimator from the raw competition
data using the persisted grouped folds.  It deliberately does not consume any
file under ``artifacts/experiments``.  The three production signals are:

1. the established word + within-word-character TF-IDF / style blend;
2. a raw-character TF-IDF Ridge model; and
3. five cumulative word TF-IDF logistic heads for ordinal scoring.

The continuous signals are combined with fixed, predeclared weights.  Ordered
thresholds are fitted on the four complementary folds and their coordinate-wise
median is used for the test submission.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import resource
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Keep simultaneous sparse solvers predictable and memory-bounded.  These are
# set before NumPy/SciPy imports so their native runtimes see the limits.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import differential_evolution
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDClassifier

try:
    from src.config import (
        ID_COLUMN,
        N_SPLITS,
        OOF_DIR,
        PROJECT_ROOT,
        RANDOM_SEED,
        REPORTS_DIR,
        SCORE_MAX,
        SCORE_MIN,
        SUBMISSIONS_DIR,
        TARGET_COLUMN,
        TEXT_COLUMN,
    )
    from src.eda import TEXT_FEATURES, add_text_features
    from src.metrics import apply_ordered_thresholds, quadratic_weighted_kappa
    from src.train_tfidf import (
        BaselineConfig,
        build_vectorizer as build_base_vectorizer,
        dataframe_to_markdown,
        file_sha256,
        sparse_megabytes,
        validate_and_load_data,
    )
    from src.train_tfidf_style_blend import BlendConfig, build_style_model
except ModuleNotFoundError:  # Support ``python src/train_classical_ensemble.py``.
    from config import (  # type: ignore[no-redef]
        ID_COLUMN,
        N_SPLITS,
        OOF_DIR,
        PROJECT_ROOT,
        RANDOM_SEED,
        REPORTS_DIR,
        SCORE_MAX,
        SCORE_MIN,
        SUBMISSIONS_DIR,
        TARGET_COLUMN,
        TEXT_COLUMN,
    )
    from eda import TEXT_FEATURES, add_text_features  # type: ignore[no-redef]
    from metrics import (  # type: ignore[no-redef]
        apply_ordered_thresholds,
        quadratic_weighted_kappa,
    )
    from train_tfidf import (  # type: ignore[no-redef]
        BaselineConfig,
        build_vectorizer as build_base_vectorizer,
        dataframe_to_markdown,
        file_sha256,
        sparse_megabytes,
        validate_and_load_data,
    )
    from train_tfidf_style_blend import (  # type: ignore[no-redef]
        BlendConfig,
        build_style_model,
    )


OOF_PATH = OOF_DIR / "classical_ensemble_oof.csv"
TEST_PREDICTIONS_PATH = OOF_DIR / "classical_ensemble_test.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "classical_three_way_calibrated.csv"
SUMMARY_PATH = REPORTS_DIR / "classical_ensemble_summary.json"
REPORT_PATH = REPORTS_DIR / "classical_ensemble_report.md"
FOLD_METRICS_PATH = REPORTS_DIR / "classical_ensemble_fold_metrics.csv"
COMPONENT_METRICS_PATH = REPORTS_DIR / "classical_ensemble_component_metrics.csv"
COMPONENT_FOLD_METRICS_PATH = (
    REPORTS_DIR / "classical_ensemble_component_fold_metrics.csv"
)
RESOURCE_METRICS_PATH = REPORTS_DIR / "classical_ensemble_resource_metrics.csv"
PREDICTION_DISTRIBUTION_PATH = (
    REPORTS_DIR / "classical_ensemble_prediction_distribution.csv"
)


@dataclass(frozen=True)
class ClassicalEnsembleConfig:
    """Locked settings selected by the preceding classical experiments."""

    base_tfidf_alpha: float = 3.0
    base_tfidf_weight_in_style: float = 0.65
    numeric_style_weight_in_style: float = 0.35

    raw_char_ngram_min: int = 3
    raw_char_ngram_max: int = 6
    raw_char_min_df: int = 3
    raw_char_max_df: float = 0.995
    raw_char_max_features: int = 180_000
    raw_char_alpha: float = 1.0

    ordinal_word_ngram_min: int = 1
    ordinal_word_ngram_max: int = 3
    ordinal_word_min_df: int = 2
    ordinal_word_max_df: float = 0.995
    ordinal_word_max_features: int = 180_000
    ordinal_sgd_alpha: float = 1e-5
    ordinal_sgd_max_iterations: int = 100
    ordinal_sgd_tolerance: float = 1e-4

    style_blend_weight: float = 0.80
    raw_char_weight: float = 0.10
    word_ordinal_weight: float = 0.10

    ridge_solver: str = "lsqr"
    ridge_tolerance: float = 1e-4
    ridge_max_iterations: int = 5_000

    threshold_min_gap: float = 0.025
    threshold_population_size: int = 10
    threshold_max_iterations: int = 100
    threshold_seed: int = 1_100


THRESHOLD_BOUNDS = (
    (0.7, 2.5),
    (1.4, 3.3),
    (2.2, 4.3),
    (3.1, 5.3),
    (4.0, 6.4),
)


def _validate_config(config: ClassicalEnsembleConfig) -> None:
    if not np.isclose(
        config.base_tfidf_weight_in_style
        + config.numeric_style_weight_in_style,
        1.0,
    ):
        raise ValueError("the two internal style-blend weights must sum to one")
    if not np.isclose(
        config.style_blend_weight
        + config.raw_char_weight
        + config.word_ordinal_weight,
        1.0,
    ):
        raise ValueError("the three ensemble weights must sum to one")
    if min(
        config.style_blend_weight,
        config.raw_char_weight,
        config.word_ordinal_weight,
    ) < 0.0:
        raise ValueError("ensemble weights must be non-negative")


def _ensure_fold_integrity(train: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    training_indices = np.flatnonzero(train["fold"].to_numpy() != fold)
    validation_indices = np.flatnonzero(train["fold"].to_numpy() == fold)
    training_groups = set(train.loc[training_indices, "group_id"])
    validation_groups = set(train.loc[validation_indices, "group_id"])
    if training_groups & validation_groups:
        raise RuntimeError(f"fold {fold} has duplicate-group leakage")
    return training_indices, validation_indices


def _finite(name: str, *arrays: np.ndarray) -> None:
    if any(not np.isfinite(array).all() for array in arrays):
        raise RuntimeError(f"{name} predictions contain NaN or Inf")


def _ridge(config: ClassicalEnsembleConfig, alpha: float) -> Ridge:
    return Ridge(
        alpha=alpha,
        fit_intercept=True,
        solver=config.ridge_solver,
        tol=config.ridge_tolerance,
        max_iter=config.ridge_max_iterations,
    )


def _fit_ridge_checked(model: Ridge, matrix: Any, target: np.ndarray) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(matrix, target)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Ridge failed to converge")
    if not np.isfinite(model.coef_).all():
        raise RuntimeError("Ridge coefficients contain NaN or Inf")


def _build_raw_char_vectorizer(config: ClassicalEnsembleConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(config.raw_char_ngram_min, config.raw_char_ngram_max),
        min_df=config.raw_char_min_df,
        max_df=config.raw_char_max_df,
        max_features=config.raw_char_max_features,
        lowercase=True,
        strip_accents=None,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )


def _build_ordinal_word_vectorizer(
    config: ClassicalEnsembleConfig,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(
            config.ordinal_word_ngram_min,
            config.ordinal_word_ngram_max,
        ),
        min_df=config.ordinal_word_min_df,
        max_df=config.ordinal_word_max_df,
        max_features=config.ordinal_word_max_features,
        lowercase=True,
        strip_accents=None,
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )


def _ordinal_score(head_predictions: np.ndarray) -> np.ndarray:
    heads = np.clip(np.asarray(head_predictions, dtype=np.float64), 0.0, 1.0)
    # Independently fitted cumulative heads can cross.  Projecting each row onto
    # a non-increasing sequence preserves the ordinal interpretation cheaply.
    heads = np.minimum.accumulate(heads, axis=1)
    return 1.0 + heads.sum(axis=1)


def _fit_predict_ordinal_heads(
    config: ClassicalEnsembleConfig,
    x_train: Any,
    y_train: np.ndarray,
    x_validation: Any,
    x_test: Any,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    validation_heads = np.zeros((x_validation.shape[0], 5), dtype=np.float64)
    test_heads = np.zeros((x_test.shape[0], 5), dtype=np.float64)
    iterations: list[int] = []
    for head_index, boundary in enumerate(range(SCORE_MIN, SCORE_MAX)):
        binary_target = (y_train > boundary).astype(np.int8)
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=config.ordinal_sgd_alpha,
            class_weight="balanced",
            average=True,
            max_iter=config.ordinal_sgd_max_iterations,
            tol=config.ordinal_sgd_tolerance,
            random_state=RANDOM_SEED + fold * 10 + head_index,
            n_jobs=1,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, binary_target)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError(
                f"ordinal SGD failed to converge for fold={fold}, head={head_index}"
            )
        positive_columns = np.flatnonzero(model.classes_ == 1)
        if len(positive_columns) != 1:
            raise RuntimeError("ordinal head did not expose the positive class")
        positive_column = int(positive_columns[0])
        validation_heads[:, head_index] = model.predict_proba(x_validation)[
            :, positive_column
        ]
        test_heads[:, head_index] = model.predict_proba(x_test)[:, positive_column]
        iterations.append(int(model.n_iter_))
    return _ordinal_score(validation_heads), _ordinal_score(test_heads), iterations


def _component_arrays(train_rows: int, test_rows: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full(train_rows, np.nan, dtype=np.float64),
        np.full((N_SPLITS, test_rows), np.nan, dtype=np.float64),
    )


def train_style_blend(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    config: ClassicalEnsembleConfig,
    base_config: BaselineConfig,
    style_config: BlendConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    """Fit the formal TF-IDF and numeric-style components fold by fold."""

    tfidf_oof, tfidf_test_by_fold = _component_arrays(len(train), len(test))
    style_oof, style_test_by_fold = _component_arrays(len(train), len(test))
    train_features = add_text_features(train[[ID_COLUMN, TEXT_COLUMN]].copy())
    test_features = add_text_features(test[[ID_COLUMN, TEXT_COLUMN]].copy())
    x_style = train_features[TEXT_FEATURES].to_numpy(dtype=np.float64)
    x_test_style = test_features[TEXT_FEATURES].to_numpy(dtype=np.float64)
    _finite("numeric style features", x_style, x_test_style)
    rows: list[dict[str, Any]] = []

    for fold in range(N_SPLITS):
        fold_started = time.perf_counter()
        training, validation = _ensure_fold_integrity(train, fold)
        vectorizer_started = time.perf_counter()
        vectorizer = build_base_vectorizer(base_config)
        x_train = vectorizer.fit_transform(train.loc[training, TEXT_COLUMN])
        x_validation = vectorizer.transform(train.loc[validation, TEXT_COLUMN])
        x_test = vectorizer.transform(test[TEXT_COLUMN])
        vectorizer_seconds = time.perf_counter() - vectorizer_started

        ridge_started = time.perf_counter()
        ridge = _ridge(config, config.base_tfidf_alpha)
        _fit_ridge_checked(ridge, x_train, y[training])
        tfidf_oof[validation] = ridge.predict(x_validation)
        tfidf_test_by_fold[fold] = ridge.predict(x_test)
        ridge_seconds = time.perf_counter() - ridge_started

        style_started = time.perf_counter()
        style_model = build_style_model(style_config, fold)
        style_model.fit(x_style[training], y[training])
        style_oof[validation] = style_model.predict(x_style[validation])
        style_test_by_fold[fold] = style_model.predict(x_test_style)
        style_seconds = time.perf_counter() - style_started

        fitted = dict(vectorizer.transformer_list)
        rows.append(
            {
                "component": "style_blend",
                "fold": fold,
                "train_rows": len(training),
                "validation_rows": len(validation),
                "features": int(x_train.shape[1]),
                "word_features": len(fitted["word"].vocabulary_),
                "char_wb_features": len(fitted["char"].vocabulary_),
                "train_nonzeros": int(x_train.nnz),
                "sparse_matrix_megabytes": (
                    sparse_megabytes(x_train)
                    + sparse_megabytes(x_validation)
                    + sparse_megabytes(x_test)
                ),
                "vectorizer_seconds": vectorizer_seconds,
                "model_seconds": ridge_seconds + style_seconds,
                "ridge_seconds": ridge_seconds,
                "style_seconds": style_seconds,
                "style_iterations": int(style_model.n_iter_),
                "fold_seconds": time.perf_counter() - fold_started,
            }
        )
        print(
            f"[style fold {fold + 1}/{N_SPLITS}] "
            f"features={x_train.shape[1]:,}, "
            f"vectorize={vectorizer_seconds:.1f}s, "
            f"models={ridge_seconds + style_seconds:.1f}s",
            flush=True,
        )
        del (
            vectorizer,
            ridge,
            style_model,
            x_train,
            x_validation,
            x_test,
        )
        gc.collect()

    _finite("base TF-IDF", tfidf_oof, tfidf_test_by_fold)
    _finite("numeric style", style_oof, style_test_by_fold)
    tfidf_test = tfidf_test_by_fold.mean(axis=0)
    style_test = style_test_by_fold.mean(axis=0)
    style_blend_oof = (
        config.base_tfidf_weight_in_style * tfidf_oof
        + config.numeric_style_weight_in_style * style_oof
    )
    style_blend_test = (
        config.base_tfidf_weight_in_style * tfidf_test
        + config.numeric_style_weight_in_style * style_test
    )
    return (
        tfidf_oof,
        tfidf_test_by_fold,
        style_oof,
        style_test_by_fold,
        style_blend_oof,
        style_blend_test,
        rows,
    )


def train_raw_char(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    config: ClassicalEnsembleConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    oof, test_by_fold = _component_arrays(len(train), len(test))
    rows: list[dict[str, Any]] = []
    for fold in range(N_SPLITS):
        fold_started = time.perf_counter()
        training, validation = _ensure_fold_integrity(train, fold)
        vectorizer_started = time.perf_counter()
        vectorizer = _build_raw_char_vectorizer(config)
        x_train = vectorizer.fit_transform(train.loc[training, TEXT_COLUMN])
        x_validation = vectorizer.transform(train.loc[validation, TEXT_COLUMN])
        x_test = vectorizer.transform(test[TEXT_COLUMN])
        vectorizer_seconds = time.perf_counter() - vectorizer_started
        model_started = time.perf_counter()
        model = _ridge(config, config.raw_char_alpha)
        _fit_ridge_checked(model, x_train, y[training])
        oof[validation] = model.predict(x_validation)
        test_by_fold[fold] = model.predict(x_test)
        model_seconds = time.perf_counter() - model_started
        rows.append(
            {
                "component": "raw_char",
                "fold": fold,
                "train_rows": len(training),
                "validation_rows": len(validation),
                "features": int(x_train.shape[1]),
                "word_features": 0,
                "char_wb_features": 0,
                "train_nonzeros": int(x_train.nnz),
                "sparse_matrix_megabytes": (
                    sparse_megabytes(x_train)
                    + sparse_megabytes(x_validation)
                    + sparse_megabytes(x_test)
                ),
                "vectorizer_seconds": vectorizer_seconds,
                "model_seconds": model_seconds,
                "ridge_seconds": model_seconds,
                "style_seconds": 0.0,
                "style_iterations": 0,
                "fold_seconds": time.perf_counter() - fold_started,
            }
        )
        print(
            f"[raw-char fold {fold + 1}/{N_SPLITS}] "
            f"features={x_train.shape[1]:,}, "
            f"vectorize={vectorizer_seconds:.1f}s, model={model_seconds:.1f}s",
            flush=True,
        )
        del vectorizer, model, x_train, x_validation, x_test
        gc.collect()
    _finite("raw-character", oof, test_by_fold)
    return oof, test_by_fold, rows


def train_word_ordinal(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    config: ClassicalEnsembleConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    oof, test_by_fold = _component_arrays(len(train), len(test))
    rows: list[dict[str, Any]] = []
    for fold in range(N_SPLITS):
        fold_started = time.perf_counter()
        training, validation = _ensure_fold_integrity(train, fold)
        vectorizer_started = time.perf_counter()
        vectorizer = _build_ordinal_word_vectorizer(config)
        x_train = vectorizer.fit_transform(train.loc[training, TEXT_COLUMN])
        x_validation = vectorizer.transform(train.loc[validation, TEXT_COLUMN])
        x_test = vectorizer.transform(test[TEXT_COLUMN])
        vectorizer_seconds = time.perf_counter() - vectorizer_started
        model_started = time.perf_counter()
        validation_predictions, test_predictions, iterations = (
            _fit_predict_ordinal_heads(
                config,
                x_train,
                y[training],
                x_validation,
                x_test,
                fold,
            )
        )
        model_seconds = time.perf_counter() - model_started
        oof[validation] = validation_predictions
        test_by_fold[fold] = test_predictions
        rows.append(
            {
                "component": "word_ordinal",
                "fold": fold,
                "train_rows": len(training),
                "validation_rows": len(validation),
                "features": int(x_train.shape[1]),
                "word_features": int(x_train.shape[1]),
                "char_wb_features": 0,
                "train_nonzeros": int(x_train.nnz),
                "sparse_matrix_megabytes": (
                    sparse_megabytes(x_train)
                    + sparse_megabytes(x_validation)
                    + sparse_megabytes(x_test)
                ),
                "vectorizer_seconds": vectorizer_seconds,
                "model_seconds": model_seconds,
                "ridge_seconds": 0.0,
                "style_seconds": 0.0,
                "style_iterations": 0,
                "sgd_max_iterations_used": max(iterations),
                "fold_seconds": time.perf_counter() - fold_started,
            }
        )
        print(
            f"[ordinal fold {fold + 1}/{N_SPLITS}] "
            f"features={x_train.shape[1]:,}, "
            f"vectorize={vectorizer_seconds:.1f}s, heads={model_seconds:.1f}s",
            flush=True,
        )
        del (
            vectorizer,
            x_train,
            x_validation,
            x_test,
            validation_predictions,
            test_predictions,
        )
        gc.collect()
    _finite("word ordinal", oof, test_by_fold)
    return oof, test_by_fold, rows


def _fast_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_index = np.asarray(y_true, dtype=np.int16) - SCORE_MIN
    pred_index = np.asarray(y_pred, dtype=np.int16) - SCORE_MIN
    size = SCORE_MAX - SCORE_MIN + 1
    observed = np.zeros((size, size), dtype=np.float64)
    np.add.at(observed, (true_index, pred_index), 1.0)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / len(y_true)
    indices = np.arange(size)
    weights = ((indices[:, None] - indices[None, :]) / (size - 1)) ** 2
    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        return 1.0 if np.array_equal(true_index, pred_index) else 0.0
    return float(1.0 - (weights * observed).sum() / denominator)


def _threshold_objective(
    thresholds: np.ndarray,
    y_true: np.ndarray,
    raw_predictions: np.ndarray,
    min_gap: float,
) -> float:
    if np.any(np.diff(thresholds) <= min_gap):
        return 2.0
    return -_fast_qwk(
        y_true,
        apply_ordered_thresholds(raw_predictions, thresholds),
    )


def fit_thresholds(
    y_true: np.ndarray,
    raw_predictions: np.ndarray,
    config: ClassicalEnsembleConfig,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    result = differential_evolution(
        _threshold_objective,
        bounds=THRESHOLD_BOUNDS,
        args=(y_true, raw_predictions, config.threshold_min_gap),
        seed=seed,
        popsize=config.threshold_population_size,
        maxiter=config.threshold_max_iterations,
        tol=1e-7,
        polish=False,
        workers=1,
        updating="immediate",
    )
    thresholds = np.asarray(result.x, dtype=np.float64)
    if np.any(np.diff(thresholds) <= config.threshold_min_gap):
        raise RuntimeError("threshold optimizer returned unordered boundaries")
    return thresholds, float(-result.fun), int(result.nfev)


def crossfit_thresholds(
    name: str,
    raw_predictions: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    config: ClassicalEnsembleConfig,
) -> tuple[np.ndarray, np.ndarray, float, list[dict[str, Any]]]:
    cross_fitted = np.zeros(len(y), dtype=np.int8)
    threshold_matrix = np.zeros((N_SPLITS, SCORE_MAX - SCORE_MIN), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for fold in range(N_SPLITS):
        calibration = folds != fold
        validation = ~calibration
        thresholds, calibration_qwk, evaluations = fit_thresholds(
            y[calibration],
            raw_predictions[calibration],
            config,
            seed=config.threshold_seed + fold,
        )
        fold_prediction = apply_ordered_thresholds(
            raw_predictions[validation], thresholds
        )
        cross_fitted[validation] = fold_prediction
        threshold_matrix[fold] = thresholds
        row: dict[str, Any] = {
            "component": name,
            "fold": fold,
            "calibration_rows": int(calibration.sum()),
            "validation_rows": int(validation.sum()),
            "calibration_qwk": calibration_qwk,
            "validation_qwk": quadratic_weighted_kappa(
                y[validation], fold_prediction
            ),
            "optimizer_evaluations": evaluations,
        }
        row.update(
            {
                f"threshold_{index + 1}": float(value)
                for index, value in enumerate(thresholds)
            }
        )
        rows.append(row)
    return (
        cross_fitted,
        threshold_matrix,
        quadratic_weighted_kappa(y, cross_fitted),
        rows,
    )


def evaluate_components(
    components: dict[str, np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    config: ClassicalEnsembleConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    crossfit_predictions: dict[str, np.ndarray] = {}
    production_thresholds: dict[str, np.ndarray] = {}
    for name, predictions in components.items():
        crossfit, threshold_matrix, crossfit_qwk, rows = crossfit_thresholds(
            name, predictions, y, folds, config
        )
        thresholds = np.median(threshold_matrix, axis=0)
        production_prediction = apply_ordered_thresholds(predictions, thresholds)
        heldout_qwks = np.asarray([row["validation_qwk"] for row in rows])
        threshold_std = np.std(threshold_matrix, axis=0, ddof=1)
        metric_rows.append(
            {
                "component": name,
                "fixed_threshold_qwk": quadratic_weighted_kappa(
                    y, apply_ordered_thresholds(predictions)
                ),
                "cross_fitted_threshold_qwk": crossfit_qwk,
                "heldout_qwk_mean": float(heldout_qwks.mean()),
                "heldout_qwk_std": float(heldout_qwks.std(ddof=1)),
                "production_median_threshold_qwk": quadratic_weighted_kappa(
                    y, production_prediction
                ),
                "raw_rmse": float(np.sqrt(np.mean(np.square(predictions - y)))),
                "raw_mae": float(np.mean(np.abs(predictions - y))),
                "threshold_mean_std": float(threshold_std.mean()),
                "threshold_max_std": float(threshold_std.max()),
                **{
                    f"production_threshold_{index + 1}": float(value)
                    for index, value in enumerate(thresholds)
                },
            }
        )
        fold_rows.extend(rows)
        crossfit_predictions[name] = crossfit
        production_thresholds[name] = thresholds
        print(
            f"[calibration] {name}: cross-fit QWK={crossfit_qwk:.6f}, "
            f"production-median QWK={metric_rows[-1]['production_median_threshold_qwk']:.6f}",
            flush=True,
        )
    metrics = pd.DataFrame(metric_rows).sort_values(
        "cross_fitted_threshold_qwk", ascending=False, kind="stable"
    )
    return metrics, pd.DataFrame(fold_rows), crossfit_predictions, production_thresholds


def _validate_saved_outputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    saved_oof = pd.read_csv(OOF_PATH, dtype={ID_COLUMN: str})
    saved_test = pd.read_csv(TEST_PREDICTIONS_PATH, dtype={ID_COLUMN: str})
    saved_submission = pd.read_csv(SUBMISSION_PATH, dtype={ID_COLUMN: str})
    if len(saved_oof) != len(train) or not saved_oof[ID_COLUMN].is_unique:
        raise RuntimeError("saved OOF row count or IDs are invalid")
    if not saved_oof[ID_COLUMN].equals(train[ID_COLUMN]):
        raise RuntimeError("saved OOF order differs from train.csv")
    if not saved_test[ID_COLUMN].equals(test[ID_COLUMN]):
        raise RuntimeError("saved test-prediction order differs from test.csv")
    if saved_submission.columns.tolist() != [ID_COLUMN, TARGET_COLUMN]:
        raise RuntimeError("saved submission schema is invalid")
    if not saved_submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise RuntimeError("saved submission order differs from sample submission")
    if not saved_submission[TARGET_COLUMN].between(SCORE_MIN, SCORE_MAX).all():
        raise RuntimeError("saved submission scores are outside 1..6")
    if not np.array_equal(
        saved_submission[TARGET_COLUMN].to_numpy(),
        saved_submission[TARGET_COLUMN].astype(int).to_numpy(),
    ):
        raise RuntimeError("saved submission scores are not integers")
    numeric_oof = saved_oof.select_dtypes(include=[np.number]).to_numpy()
    numeric_test = saved_test.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric_oof).all() or not np.isfinite(numeric_test).all():
        raise RuntimeError("saved prediction artifacts contain NaN or Inf")


def _peak_rss_megabytes() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return float(peak / (1024**2))
    return float(peak / 1024)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def main() -> None:
    started = time.perf_counter()
    config = ClassicalEnsembleConfig()
    base_config = BaselineConfig()
    style_config = BlendConfig()
    _validate_config(config)
    train, test, sample = validate_and_load_data()
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    folds = train["fold"].to_numpy(dtype=np.int8)

    training_started = time.perf_counter()
    (
        tfidf_oof,
        tfidf_test_by_fold,
        style_oof,
        style_test_by_fold,
        style_blend_oof,
        style_blend_test,
        style_resource_rows,
    ) = train_style_blend(
        train, test, y, config, base_config, style_config
    )
    raw_char_oof, raw_char_test_by_fold, raw_char_resource_rows = train_raw_char(
        train, test, y, config
    )
    ordinal_oof, ordinal_test_by_fold, ordinal_resource_rows = train_word_ordinal(
        train, test, y, config
    )
    training_seconds = time.perf_counter() - training_started

    raw_char_test = raw_char_test_by_fold.mean(axis=0)
    ordinal_test = ordinal_test_by_fold.mean(axis=0)
    ensemble_oof = (
        config.style_blend_weight * style_blend_oof
        + config.raw_char_weight * raw_char_oof
        + config.word_ordinal_weight * ordinal_oof
    )
    ensemble_test = (
        config.style_blend_weight * style_blend_test
        + config.raw_char_weight * raw_char_test
        + config.word_ordinal_weight * ordinal_test
    )
    _finite(
        "complete ensemble",
        style_blend_oof,
        style_blend_test,
        raw_char_oof,
        raw_char_test,
        ordinal_oof,
        ordinal_test,
        ensemble_oof,
        ensemble_test,
    )

    components = {
        "base_tfidf": tfidf_oof,
        "numeric_style": style_oof,
        "style_blend": style_blend_oof,
        "raw_char": raw_char_oof,
        "word_ordinal": ordinal_oof,
        "classical_ensemble": ensemble_oof,
    }
    calibration_started = time.perf_counter()
    (
        component_metrics,
        component_fold_metrics,
        crossfit_predictions,
        production_thresholds,
    ) = evaluate_components(components, y, folds, config)
    calibration_seconds = time.perf_counter() - calibration_started
    ensemble_thresholds = production_thresholds["classical_ensemble"]
    ensemble_production_oof = apply_ordered_thresholds(
        ensemble_oof, ensemble_thresholds
    )
    ensemble_test_class = apply_ordered_thresholds(
        ensemble_test, ensemble_thresholds
    )

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    oof_frame = train[[ID_COLUMN, TARGET_COLUMN, "fold", "group_id"]].copy()
    oof_frame["base_tfidf_raw"] = tfidf_oof
    oof_frame["numeric_style_raw"] = style_oof
    oof_frame["style_blend_raw"] = style_blend_oof
    oof_frame["raw_char_raw"] = raw_char_oof
    oof_frame["word_ordinal_raw"] = ordinal_oof
    oof_frame["ensemble_raw"] = ensemble_oof
    oof_frame["ensemble_cross_fitted_prediction"] = crossfit_predictions[
        "classical_ensemble"
    ].astype(int)
    oof_frame["ensemble_production_prediction"] = ensemble_production_oof.astype(int)
    oof_frame.to_csv(OOF_PATH, index=False, float_format="%.10f")

    test_frame = test[[ID_COLUMN]].copy()
    for fold in range(N_SPLITS):
        test_frame[f"base_tfidf_fold_{fold}"] = tfidf_test_by_fold[fold]
        test_frame[f"numeric_style_fold_{fold}"] = style_test_by_fold[fold]
        test_frame[f"raw_char_fold_{fold}"] = raw_char_test_by_fold[fold]
        test_frame[f"word_ordinal_fold_{fold}"] = ordinal_test_by_fold[fold]
    test_frame["base_tfidf_raw"] = tfidf_test_by_fold.mean(axis=0)
    test_frame["numeric_style_raw"] = style_test_by_fold.mean(axis=0)
    test_frame["style_blend_raw"] = style_blend_test
    test_frame["raw_char_raw"] = raw_char_test
    test_frame["word_ordinal_raw"] = ordinal_test
    test_frame["ensemble_raw"] = ensemble_test
    test_frame["ensemble_prediction"] = ensemble_test_class.astype(int)
    test_frame.to_csv(
        TEST_PREDICTIONS_PATH, index=False, float_format="%.10f"
    )

    submission = sample[[ID_COLUMN]].copy()
    submission[TARGET_COLUMN] = ensemble_test_class.astype(int)
    submission.to_csv(SUBMISSION_PATH, index=False)

    main_fold_metrics = component_fold_metrics.loc[
        component_fold_metrics["component"].eq("classical_ensemble")
    ].reset_index(drop=True)
    main_fold_metrics.to_csv(
        FOLD_METRICS_PATH, index=False, float_format="%.10f"
    )
    component_metrics.to_csv(
        COMPONENT_METRICS_PATH, index=False, float_format="%.10f"
    )
    component_fold_metrics.to_csv(
        COMPONENT_FOLD_METRICS_PATH, index=False, float_format="%.10f"
    )
    resource_metrics = pd.DataFrame(
        style_resource_rows + raw_char_resource_rows + ordinal_resource_rows
    )
    resource_metrics.to_csv(
        RESOURCE_METRICS_PATH, index=False, float_format="%.10f"
    )

    true_counts = pd.Series(y).value_counts().reindex(range(1, 7), fill_value=0)
    oof_counts = (
        pd.Series(ensemble_production_oof)
        .value_counts()
        .reindex(range(1, 7), fill_value=0)
    )
    test_counts = (
        pd.Series(ensemble_test_class)
        .value_counts()
        .reindex(range(1, 7), fill_value=0)
    )
    distribution = pd.DataFrame(
        {
            "score": range(1, 7),
            "train_true_count": true_counts.to_numpy(dtype=int),
            "oof_production_count": oof_counts.to_numpy(dtype=int),
            "test_prediction_count": test_counts.to_numpy(dtype=int),
        }
    )
    distribution["train_true_percent"] = (
        100.0 * distribution["train_true_count"] / len(train)
    )
    distribution["oof_production_percent"] = (
        100.0 * distribution["oof_production_count"] / len(train)
    )
    distribution["test_prediction_percent"] = (
        100.0 * distribution["test_prediction_count"] / len(test)
    )
    distribution.to_csv(
        PREDICTION_DISTRIBUTION_PATH, index=False, float_format="%.10f"
    )

    _validate_saved_outputs(train, test, sample)
    elapsed = time.perf_counter() - started
    ensemble_metrics = component_metrics.loc[
        component_metrics["component"].eq("classical_ensemble")
    ].iloc[0]
    fold_qwks = main_fold_metrics["validation_qwk"]
    summary = {
        "model": "classical_three_way_ensemble",
        "description": (
            "80% formal TF-IDF/style blend + 10% raw-char Ridge + "
            "10% word cumulative-logistic ordinal model"
        ),
        "experiment_artifact_dependency": False,
        "config": asdict(config),
        "base_tfidf_config": asdict(base_config),
        "numeric_style_config": asdict(style_config),
        "style_features": list(TEXT_FEATURES),
        "validation": {
            "cross_fitted_threshold_qwk": float(
                ensemble_metrics["cross_fitted_threshold_qwk"]
            ),
            "heldout_qwk_mean": float(fold_qwks.mean()),
            "heldout_qwk_std": float(fold_qwks.std(ddof=1)),
            "production_median_threshold_qwk": float(
                ensemble_metrics["production_median_threshold_qwk"]
            ),
            "raw_rmse": float(ensemble_metrics["raw_rmse"]),
            "raw_mae": float(ensemble_metrics["raw_mae"]),
            "production_thresholds": ensemble_thresholds.tolist(),
            "threshold_max_std": float(ensemble_metrics["threshold_max_std"]),
        },
        "component_metrics": component_metrics.to_dict(orient="records"),
        "submission_score_counts": {
            str(score): int(count) for score, count in test_counts.items()
        },
        "runtime_seconds": {
            "training": training_seconds,
            "threshold_calibration": calibration_seconds,
            "total": elapsed,
        },
        "resources": {
            "peak_process_rss_megabytes": _peak_rss_megabytes(),
            "max_sparse_matrix_megabytes": float(
                resource_metrics["sparse_matrix_megabytes"].max()
            ),
        },
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input_hashes": {
            "train": file_sha256(PROJECT_ROOT / "data/raw/train.csv"),
            "test": file_sha256(PROJECT_ROOT / "data/raw/test.csv"),
            "fold_assignments": file_sha256(
                PROJECT_ROOT / "artifacts/folds/fold_assignments.csv"
            ),
        },
        "output_hashes": {
            "oof": file_sha256(OOF_PATH),
            "test_predictions": file_sha256(TEST_PREDICTIONS_PATH),
            "submission": file_sha256(SUBMISSION_PATH),
        },
        "validation_note": (
            "Each text model and style model is fitted separately inside every "
            "training fold. Thresholds for each held-out fold use only the other "
            "four folds. Fixed ensemble weights were selected by preceding OOF "
            "experiments, so the reported score remains a model-selection estimate."
        ),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

    report = f"""# Classical three-way ensemble

## Result

- Components: `{config.style_blend_weight:.2f}` formal TF-IDF/style blend +
  `{config.raw_char_weight:.2f}` raw-character Ridge +
  `{config.word_ordinal_weight:.2f}` word cumulative-logistic ordinal model.
- Validation: persisted `{N_SPLITS}` duplicate-safe grouped folds; all vectorizers
  and estimators are fitted inside each training fold.
- Complementary-fold threshold QWK: **{ensemble_metrics['cross_fitted_threshold_qwk']:.6f}**.
- Held-out fold QWK: **{fold_qwks.mean():.6f} +/- {fold_qwks.std(ddof=1):.6f}**.
- Production-median-threshold OOF QWK: **{ensemble_metrics['production_median_threshold_qwk']:.6f}**.
- Production thresholds: `{ensemble_thresholds.tolist()}`.
- OOF RMSE: **{ensemble_metrics['raw_rmse']:.6f}**.
- Runtime: **{elapsed:.1f}s** total ({training_seconds:.1f}s training,
  {calibration_seconds:.1f}s threshold calibration).
- Peak process RSS: **{summary['resources']['peak_process_rss_megabytes']:.1f} MiB**.

## Component diagnostics

{dataframe_to_markdown(component_metrics)}

## Ensemble fold diagnostics

{dataframe_to_markdown(main_fold_metrics)}

## Prediction distribution

{dataframe_to_markdown(distribution)}

## Reproducibility and leakage controls

The script reads only raw train/test data and the persisted fold assignment; it
does not read any `artifacts/experiments` output.  IDs, labels, groups, finite
predictions, submission schema, score range, and saved row order are validated.
The threshold for each validation fold is fitted on the other four OOF folds,
then the coordinate-wise median threshold vector is used for test predictions.

The ensemble weights were selected during the preceding OOF experiments, so
the validation score is a model-selection estimate rather than a fully nested,
unbiased estimate.  Reproduce with:

```bash
python -m src.train_classical_ensemble
```

The ready-to-submit file is `submissions/classical_three_way_calibrated.csv`.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(
        "Classical ensemble complete\n"
        f"  Cross-fitted threshold QWK: "
        f"{ensemble_metrics['cross_fitted_threshold_qwk']:.6f}\n"
        f"  Production-median OOF QWK: "
        f"{ensemble_metrics['production_median_threshold_qwk']:.6f}\n"
        f"  Thresholds: {np.array2string(ensemble_thresholds, precision=6)}\n"
        f"  Submission: {SUBMISSION_PATH}\n"
        f"  Runtime: {elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
