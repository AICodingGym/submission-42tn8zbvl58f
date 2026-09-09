"""Blend TF-IDF Ridge predictions with numeric essay-style features.

This stage reuses the leakage-safe TF-IDF out-of-fold predictions produced by
``src.train_tfidf``.  A small gradient-boosted regressor restores explicit
length and style signals that L2-normalized TF-IDF tends to suppress.  Five
ordered score thresholds are fitted on complementary OOF folds and their
column-wise median is used for the test submission.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import differential_evolution
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from src.config import (
        ID_COLUMN,
        N_SPLITS,
        OOF_DIR,
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
    from src.train_tfidf import TEST_PREDICTIONS_PATH, OOF_PATH, validate_and_load_data
except ModuleNotFoundError:  # Support ``python src/train_tfidf_style_blend.py``.
    from config import (  # type: ignore[no-redef]
        ID_COLUMN,
        N_SPLITS,
        OOF_DIR,
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
        OOF_PATH,
        TEST_PREDICTIONS_PATH,
        validate_and_load_data,
    )


STYLE_OOF_PATH = OOF_DIR / "tfidf_style_blend_oof.csv"
STYLE_TEST_PATH = OOF_DIR / "tfidf_style_blend_test.csv"
STYLE_SUBMISSION_PATH = SUBMISSIONS_DIR / "tfidf_style_blend_calibrated.csv"
STYLE_FOLD_METRICS_PATH = REPORTS_DIR / "tfidf_style_blend_fold_metrics.csv"
STYLE_SUMMARY_PATH = REPORTS_DIR / "tfidf_style_blend_summary.json"
STYLE_REPORT_PATH = REPORTS_DIR / "tfidf_style_blend_report.md"


@dataclass(frozen=True)
class BlendConfig:
    """Deterministic style model, blend, and threshold settings."""

    tfidf_alpha_column: str = "pred_raw_alpha_3"
    tfidf_weight: float = 0.65
    style_weight: float = 0.35
    learning_rate: float = 0.05
    max_iterations: int = 200
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 30
    l2_regularization: float = 2.0
    threshold_min_gap: float = 0.025
    threshold_population_size: int = 10
    threshold_max_iterations: int = 100


THRESHOLD_BOUNDS = (
    (0.7, 2.5),
    (1.4, 3.3),
    (2.2, 4.3),
    (3.1, 5.3),
    (4.0, 6.4),
)


def file_sha256(path: Path) -> str:
    """Return a SHA-256 checksum for an output artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_style_model(config: BlendConfig, fold: int) -> HistGradientBoostingRegressor:
    """Create one fold-local model for inexpensive numeric style features."""

    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=config.learning_rate,
        max_iter=config.max_iterations,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping="auto",
        random_state=RANDOM_SEED + fold,
    )


def load_aligned_tfidf_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: BlendConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load TF-IDF raw predictions and align them by ID, never by row accident."""

    if not OOF_PATH.exists() or not TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "TF-IDF artifacts are missing; run `python -m src.train_tfidf` first"
        )
    oof = pd.read_csv(OOF_PATH, dtype={ID_COLUMN: str})
    test_predictions = pd.read_csv(
        TEST_PREDICTIONS_PATH,
        dtype={ID_COLUMN: str},
    )
    required_oof = {ID_COLUMN, TARGET_COLUMN, "fold", config.tfidf_alpha_column}
    required_test = {ID_COLUMN, config.tfidf_alpha_column}
    if not required_oof.issubset(oof.columns):
        raise ValueError("TF-IDF OOF artifact lacks required columns")
    if not required_test.issubset(test_predictions.columns):
        raise ValueError("TF-IDF test artifact lacks required columns")
    if oof[ID_COLUMN].duplicated().any() or test_predictions[ID_COLUMN].duplicated().any():
        raise ValueError("TF-IDF prediction artifacts contain duplicate IDs")

    train_alignment = train[[ID_COLUMN, TARGET_COLUMN, "fold"]].merge(
        oof[[ID_COLUMN, TARGET_COLUMN, "fold", config.tfidf_alpha_column]].rename(
            columns={
                TARGET_COLUMN: "tfidf_score",
                "fold": "tfidf_fold",
            }
        ),
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if train_alignment[config.tfidf_alpha_column].isna().any():
        raise ValueError("TF-IDF OOF IDs do not exactly cover training IDs")
    if not train_alignment[TARGET_COLUMN].equals(train_alignment["tfidf_score"]):
        raise ValueError("TF-IDF OOF labels do not match the current training data")
    if not train_alignment["fold"].equals(train_alignment["tfidf_fold"]):
        raise ValueError("TF-IDF OOF folds do not match the persisted folds")

    test_alignment = test[[ID_COLUMN]].merge(
        test_predictions[[ID_COLUMN, config.tfidf_alpha_column]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if test_alignment[config.tfidf_alpha_column].isna().any():
        raise ValueError("TF-IDF test predictions do not exactly cover test IDs")
    return (
        train_alignment[config.tfidf_alpha_column].to_numpy(dtype=np.float64),
        test_alignment[config.tfidf_alpha_column].to_numpy(dtype=np.float64),
    )


def threshold_objective(
    thresholds: np.ndarray,
    y_true: np.ndarray,
    raw_predictions: np.ndarray,
    min_gap: float,
) -> float:
    """Return negative QWK, rejecting unordered or nearly equal boundaries."""

    if np.any(np.diff(thresholds) <= min_gap):
        return 2.0
    discrete = apply_ordered_thresholds(raw_predictions, thresholds)
    return -quadratic_weighted_kappa(y_true, discrete)


def fit_thresholds(
    y_true: np.ndarray,
    raw_predictions: np.ndarray,
    config: BlendConfig,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Fit five ordered thresholds with deterministic global optimization."""

    result = differential_evolution(
        threshold_objective,
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
    if not result.success and result.nit < config.threshold_max_iterations:
        raise RuntimeError(f"threshold optimization failed: {result.message}")
    if np.any(np.diff(thresholds) <= config.threshold_min_gap):
        raise RuntimeError("threshold optimization returned unordered boundaries")
    return thresholds, float(-result.fun)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without requiring an optional Markdown package."""

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        formatted = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def main() -> None:
    """Train the style component, calibrate the blend, and write a submission."""

    started = time.perf_counter()
    config = BlendConfig()
    if not np.isclose(config.tfidf_weight + config.style_weight, 1.0):
        raise ValueError("blend weights must sum to one")

    train, test, sample_submission = validate_and_load_data()
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    fold_values = train["fold"].to_numpy(dtype=np.int8)
    tfidf_oof, tfidf_test = load_aligned_tfidf_predictions(train, test, config)

    train_with_features = add_text_features(train[[ID_COLUMN, TEXT_COLUMN]])
    test_with_features = add_text_features(test[[ID_COLUMN, TEXT_COLUMN]])
    x_style = train_with_features[TEXT_FEATURES].to_numpy(dtype=np.float64)
    x_test_style = test_with_features[TEXT_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(x_style).all() or not np.isfinite(x_test_style).all():
        raise ValueError("style features contain NaN or Inf")

    style_oof = np.full(len(train), np.nan, dtype=np.float64)
    style_test_by_fold = np.full((N_SPLITS, len(test)), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(N_SPLITS):
        training = fold_values != fold
        validation = ~training
        model = build_style_model(config, fold)
        model.fit(x_style[training], y[training])
        style_oof[validation] = model.predict(x_style[validation])
        style_test_by_fold[fold] = model.predict(x_test_style)
        raw_fold_blend = (
            config.tfidf_weight * tfidf_oof[validation]
            + config.style_weight * style_oof[validation]
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(training.sum()),
                "validation_rows": int(validation.sum()),
                "style_iterations": int(model.n_iter_),
                "style_rmse": float(
                    np.sqrt(np.mean(np.square(style_oof[validation] - y[validation])))
                ),
                "blend_fixed_qwk": quadratic_weighted_kappa(
                    y[validation],
                    apply_ordered_thresholds(raw_fold_blend),
                ),
            }
        )

    if not np.isfinite(style_oof).all() or not np.isfinite(style_test_by_fold).all():
        raise RuntimeError("style model predictions are incomplete")
    style_test = style_test_by_fold.mean(axis=0)
    blend_oof = config.tfidf_weight * tfidf_oof + config.style_weight * style_oof
    blend_test = config.tfidf_weight * tfidf_test + config.style_weight * style_test

    cross_fitted_discrete = np.empty(len(train), dtype=np.int8)
    thresholds_by_fold = np.empty((N_SPLITS, SCORE_MAX - SCORE_MIN), dtype=np.float64)
    for fold in range(N_SPLITS):
        calibration = fold_values != fold
        validation = ~calibration
        thresholds, calibration_qwk = fit_thresholds(
            y[calibration],
            blend_oof[calibration],
            config,
            seed=RANDOM_SEED + 100 + fold,
        )
        thresholds_by_fold[fold] = thresholds
        cross_fitted_discrete[validation] = apply_ordered_thresholds(
            blend_oof[validation], thresholds
        )
        fold_rows[fold].update(
            {
                "calibration_qwk": calibration_qwk,
                "validation_calibrated_qwk": quadratic_weighted_kappa(
                    y[validation], cross_fitted_discrete[validation]
                ),
                **{
                    f"threshold_{index + 1}": float(value)
                    for index, value in enumerate(thresholds)
                },
            }
        )

    production_thresholds = np.median(thresholds_by_fold, axis=0)
    if np.any(np.diff(production_thresholds) <= config.threshold_min_gap):
        raise RuntimeError("median production thresholds are not ordered")
    calibrated_oof = apply_ordered_thresholds(blend_oof, production_thresholds)
    calibrated_test = apply_ordered_thresholds(blend_test, production_thresholds)
    fixed_oof = apply_ordered_thresholds(blend_oof)

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    oof_frame = train[[ID_COLUMN, TARGET_COLUMN, "fold", "group_id"]].copy()
    oof_frame["tfidf_raw"] = tfidf_oof
    oof_frame["style_raw"] = style_oof
    oof_frame["blend_raw"] = blend_oof
    oof_frame["pred_fixed"] = fixed_oof.astype(int)
    oof_frame["pred_cross_fitted_thresholds"] = cross_fitted_discrete.astype(int)
    oof_frame["pred_production_thresholds"] = calibrated_oof.astype(int)
    oof_frame.to_csv(STYLE_OOF_PATH, index=False, float_format="%.10f")

    test_frame = test[[ID_COLUMN]].copy()
    for fold in range(N_SPLITS):
        test_frame[f"style_pred_fold_{fold}"] = style_test_by_fold[fold]
    test_frame["tfidf_raw"] = tfidf_test
    test_frame["style_raw"] = style_test
    test_frame["blend_raw"] = blend_test
    test_frame["pred_calibrated"] = calibrated_test.astype(int)
    test_frame.to_csv(STYLE_TEST_PATH, index=False, float_format="%.10f")

    submission = sample_submission[[ID_COLUMN]].copy()
    submission[TARGET_COLUMN] = calibrated_test.astype(int)
    submission.to_csv(STYLE_SUBMISSION_PATH, index=False)

    saved_submission = pd.read_csv(STYLE_SUBMISSION_PATH, dtype={ID_COLUMN: str})
    if saved_submission.columns.tolist() != [ID_COLUMN, TARGET_COLUMN]:
        raise RuntimeError("saved submission has incorrect columns")
    if not saved_submission[ID_COLUMN].equals(sample_submission[ID_COLUMN]):
        raise RuntimeError("saved submission IDs do not match sample order")
    if len(saved_submission) != len(test) or not saved_submission[ID_COLUMN].is_unique:
        raise RuntimeError("saved submission has incorrect rows or duplicate IDs")
    if not saved_submission[TARGET_COLUMN].between(SCORE_MIN, SCORE_MAX).all():
        raise RuntimeError("saved submission contains scores outside 1..6")
    if not np.array_equal(
        saved_submission[TARGET_COLUMN], saved_submission[TARGET_COLUMN].astype(int)
    ):
        raise RuntimeError("saved submission scores are not integers")

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(
        STYLE_FOLD_METRICS_PATH,
        index=False,
        float_format="%.10f",
    )
    elapsed = time.perf_counter() - started
    summary = {
        "model": "tfidf_ridge_style_hist_gradient_boosting_blend",
        "config": asdict(config),
        "style_features": list(TEXT_FEATURES),
        "production_threshold_rule": "column-wise median of five complementary-fold fits",
        "production_thresholds": production_thresholds.tolist(),
        "validation": {
            "baseline_tfidf_alpha3_fixed_qwk": quadratic_weighted_kappa(
                y, apply_ordered_thresholds(tfidf_oof)
            ),
            "style_only_fixed_qwk": quadratic_weighted_kappa(
                y, apply_ordered_thresholds(style_oof)
            ),
            "blend_fixed_qwk": quadratic_weighted_kappa(y, fixed_oof),
            "blend_cross_fitted_threshold_qwk": quadratic_weighted_kappa(
                y, cross_fitted_discrete
            ),
            "blend_production_threshold_qwk": quadratic_weighted_kappa(
                y, calibrated_oof
            ),
            "blend_raw_rmse": float(
                np.sqrt(np.mean(np.square(blend_oof - y)))
            ),
        },
        "submission_score_counts": {
            str(score): int((calibrated_test == score).sum())
            for score in range(SCORE_MIN, SCORE_MAX + 1)
        },
        "runtime_seconds": elapsed,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": {
            "source_tfidf_oof_sha256": file_sha256(OOF_PATH),
            "source_tfidf_test_sha256": file_sha256(TEST_PREDICTIONS_PATH),
            "oof_sha256": file_sha256(STYLE_OOF_PATH),
            "test_predictions_sha256": file_sha256(STYLE_TEST_PATH),
            "submission_sha256": file_sha256(STYLE_SUBMISSION_PATH),
        },
    }
    STYLE_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = summary["validation"]
    report = f"""# TF-IDF + Style Blend

## Result

- TF-IDF component: Ridge alpha 3 raw prediction, weight `{config.tfidf_weight:.2f}`.
- Style component: HistGradientBoosting over {len(TEXT_FEATURES)} deterministic
  length/style features, weight `{config.style_weight:.2f}`.
- Baseline alpha-3 fixed-threshold OOF QWK:
  **{validation['baseline_tfidf_alpha3_fixed_qwk']:.5f}**.
- Blend fixed-threshold OOF QWK: **{validation['blend_fixed_qwk']:.5f}**.
- Complementary-fold threshold stability QWK:
  **{validation['blend_cross_fitted_threshold_qwk']:.5f}**.
- Production-median threshold OOF QWK:
  **{validation['blend_production_threshold_qwk']:.5f}**.
- Production thresholds: `{production_thresholds.tolist()}`.
- Runtime: **{elapsed:.1f} seconds**.

## Fold diagnostics

{frame_to_markdown(fold_metrics)}

## Validation note

Each base and style OOF prediction excludes that row's label.  Blend weight and
threshold-family selection were nevertheless made using the shared OOF set, so
the calibrated scores are model-selection estimates rather than a fully nested,
unbiased performance estimate.  The complementary-fold score and threshold
spread are reported to check that the gain is not confined to one fold.

The ready-to-submit file is `{STYLE_SUBMISSION_PATH.relative_to(STYLE_SUBMISSION_PATH.parents[1])}`.
"""
    STYLE_REPORT_PATH.write_text(report, encoding="utf-8")

    print(
        "TF-IDF + style blend complete\n"
        f"  Cross-fitted threshold QWK: {validation['blend_cross_fitted_threshold_qwk']:.5f}\n"
        f"  Production-threshold QWK:   {validation['blend_production_threshold_qwk']:.5f}\n"
        f"  Thresholds: {np.array2string(production_thresholds, precision=6)}\n"
        f"  Submission: {STYLE_SUBMISSION_PATH}\n"
        f"  Runtime: {elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
