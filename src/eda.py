"""Run reproducible exploratory analysis for the essay-scoring data."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    from src.config import (
        FIGURES_DIR,
        ID_COLUMN,
        N_SPLITS,
        PROJECT_ROOT,
        RANDOM_SEED,
        REPORTS_DIR,
        SAMPLE_SUBMISSION_PATH,
        SCORE_MAX,
        SCORE_MIN,
        TARGET_COLUMN,
        TEST_PATH,
        TEXT_COLUMN,
        TRAIN_PATH,
    )
except ModuleNotFoundError:  # Support `python src/eda.py` as well as `-m`.
    from config import (  # type: ignore[no-redef]
        FIGURES_DIR,
        ID_COLUMN,
        N_SPLITS,
        PROJECT_ROOT,
        RANDOM_SEED,
        REPORTS_DIR,
        SAMPLE_SUBMISSION_PATH,
        SCORE_MAX,
        SCORE_MIN,
        TARGET_COLUMN,
        TEST_PATH,
        TEXT_COLUMN,
        TRAIN_PATH,
    )

# Keep Matplotlib caches inside the writable project directory.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_RE = re.compile(r"[.!?]+")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")
ARGUMENT_CONNECTORS = (
    "although",
    "because",
    "consequently",
    "for example",
    "furthermore",
    "however",
    "in conclusion",
    "in contrast",
    "moreover",
    "nevertheless",
    "on the other hand",
    "since",
    "therefore",
    "thus",
)
CONNECTOR_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in ARGUMENT_CONNECTORS) + r")\b",
    flags=re.IGNORECASE,
)

TEXT_FEATURES = [
    "char_count",
    "word_count",
    "sentence_count",
    "paragraph_count",
    "avg_word_length",
    "avg_sentence_words",
    "unique_word_ratio",
    "long_word_ratio",
    "punctuation_per_1k_chars",
    "uppercase_ratio",
    "digit_ratio",
    "connector_per_1k_words",
]

QUANTILES = {
    "min": 0.00,
    "p01": 0.01,
    "p05": 0.05,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p95": 0.95,
    "p99": 0.99,
    "max": 1.00,
}

NEAR_DUPLICATE_SHINGLE_SIZE = 5
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.80
NEAR_DUPLICATE_SEQUENCE_THRESHOLD = 0.95


def file_sha256(path: Path) -> str:
    """Return a stable checksum without loading the complete file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Normalize only for duplicate detection; model text remains untouched."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def boundary_fingerprint(normalized_text: str) -> str:
    """Hash essay boundaries to flag, but not declare, near-duplicate candidates."""

    words = normalized_text.split()
    boundary_words = words if len(words) <= 80 else words[:40] + words[-40:]
    return hashlib.sha256(" ".join(boundary_words).encode("utf-8")).hexdigest()


def word_shingles(words: list[str], size: int) -> set[tuple[str, ...]]:
    """Return contiguous word shingles for inexpensive pair similarity."""

    if len(words) < size:
        return {tuple(words)} if words else set()
    return {
        tuple(words[index : index + size])
        for index in range(len(words) - size + 1)
    }


def near_duplicate_similarity(left: str, right: str) -> tuple[float, float]:
    """Measure full-text similarity after boundary-hash candidate generation."""

    left_words = left.split()
    right_words = right.split()
    left_shingles = word_shingles(left_words, NEAR_DUPLICATE_SHINGLE_SIZE)
    right_shingles = word_shingles(right_words, NEAR_DUPLICATE_SHINGLE_SIZE)
    union = left_shingles | right_shingles
    shingle_jaccard = (
        len(left_shingles & right_shingles) / len(union) if union else 1.0
    )
    sequence_ratio = SequenceMatcher(
        None,
        left_words,
        right_words,
        autojunk=False,
    ).ratio()
    return float(shingle_jaccard), float(sequence_ratio)


def build_near_duplicate_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    normalized_train: pd.Series,
    normalized_test: pd.Series,
    train_fingerprints: pd.Series,
    test_fingerprints: pd.Series,
) -> pd.DataFrame:
    """Generate boundary-hash pairs and confirm them with full-text similarity."""

    columns = [
        "relationship",
        "left_split",
        "left_essay_id",
        "left_score",
        "right_split",
        "right_essay_id",
        "right_score",
        "left_word_count",
        "right_word_count",
        "shingle_jaccard",
        "word_sequence_ratio",
        "confirmed_near_duplicate",
        "conflicting_train_scores",
    ]
    rows: list[dict[str, Any]] = []

    def add_pair(
        relationship: str,
        left_split: str,
        left_frame: pd.DataFrame,
        left_normalized: pd.Series,
        left_index: Any,
        right_split: str,
        right_frame: pd.DataFrame,
        right_normalized: pd.Series,
        right_index: Any,
    ) -> None:
        left_text = left_normalized.loc[left_index]
        right_text = right_normalized.loc[right_index]
        shingle_jaccard, sequence_ratio = near_duplicate_similarity(
            left_text, right_text
        )
        confirmed = bool(
            shingle_jaccard >= NEAR_DUPLICATE_JACCARD_THRESHOLD
            and sequence_ratio >= NEAR_DUPLICATE_SEQUENCE_THRESHOLD
        )
        left_score = (
            int(left_frame.loc[left_index, TARGET_COLUMN])
            if TARGET_COLUMN in left_frame
            else None
        )
        right_score = (
            int(right_frame.loc[right_index, TARGET_COLUMN])
            if TARGET_COLUMN in right_frame
            else None
        )
        conflicting_scores = bool(
            confirmed
            and relationship == "within_train"
            and left_score != right_score
        )
        rows.append(
            {
                "relationship": relationship,
                "left_split": left_split,
                "left_essay_id": str(left_frame.loc[left_index, ID_COLUMN]),
                "left_score": left_score,
                "right_split": right_split,
                "right_essay_id": str(right_frame.loc[right_index, ID_COLUMN]),
                "right_score": right_score,
                "left_word_count": len(left_text.split()),
                "right_word_count": len(right_text.split()),
                "shingle_jaccard": shingle_jaccard,
                "word_sequence_ratio": sequence_ratio,
                "confirmed_near_duplicate": confirmed,
                "conflicting_train_scores": conflicting_scores,
            }
        )

    train_groups = {
        fingerprint: list(indices)
        for fingerprint, indices in train_fingerprints.groupby(
            train_fingerprints
        ).groups.items()
    }
    test_groups = {
        fingerprint: list(indices)
        for fingerprint, indices in test_fingerprints.groupby(
            test_fingerprints
        ).groups.items()
    }

    for indices in train_groups.values():
        for left_index, right_index in combinations(indices, 2):
            add_pair(
                "within_train",
                "train",
                train,
                normalized_train,
                left_index,
                "train",
                train,
                normalized_train,
                right_index,
            )
    for indices in test_groups.values():
        for left_index, right_index in combinations(indices, 2):
            add_pair(
                "within_test",
                "test",
                test,
                normalized_test,
                left_index,
                "test",
                test,
                normalized_test,
                right_index,
            )
    for fingerprint in sorted(set(train_groups) & set(test_groups)):
        for left_index in train_groups[fingerprint]:
            for right_index in test_groups[fingerprint]:
                add_pair(
                    "cross_split",
                    "train",
                    train,
                    normalized_train,
                    left_index,
                    "test",
                    test,
                    normalized_test,
                    right_index,
                )

    return pd.DataFrame(rows, columns=columns)


def extract_text_features(text: str) -> dict[str, float | int]:
    """Extract inexpensive style and length features from one essay."""

    words = WORD_RE.findall(text)
    lower_words = [word.casefold() for word in words]
    word_count = len(words)
    char_count = len(text)
    sentence_count = len(SENTENCE_RE.findall(text))
    paragraph_count = max(
        1,
        len([paragraph for paragraph in PARAGRAPH_RE.split(text) if paragraph.strip()]),
    )
    letter_count = sum(char.isalpha() for char in text)
    punctuation_count = sum(char in ".,;:!?\"'()-" for char in text)
    connector_count = len(CONNECTOR_RE.findall(text))

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_word_length": (
            sum(len(word) for word in words) / word_count if word_count else 0.0
        ),
        "avg_sentence_words": word_count / max(1, sentence_count),
        "unique_word_ratio": (
            len(set(lower_words)) / word_count if word_count else 0.0
        ),
        "long_word_ratio": (
            sum(len(word) >= 7 for word in words) / word_count if word_count else 0.0
        ),
        "punctuation_per_1k_chars": (
            1000.0 * punctuation_count / char_count if char_count else 0.0
        ),
        "uppercase_ratio": (
            sum(char.isupper() for char in text) / letter_count if letter_count else 0.0
        ),
        "digit_ratio": (
            sum(char.isdigit() for char in text) / char_count if char_count else 0.0
        ),
        "connector_per_1k_words": (
            1000.0 * connector_count / word_count if word_count else 0.0
        ),
    }


def add_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a dataset with deterministic numeric text features."""

    features = pd.DataFrame(
        [extract_text_features(text) for text in frame[TEXT_COLUMN]],
        index=frame.index,
    )
    return pd.concat([frame.copy(), features], axis=1)


def require_columns(frame: pd.DataFrame, expected: list[str], name: str) -> None:
    """Fail early when downloaded files do not match the expected schema."""

    if frame.columns.tolist() != expected:
        raise ValueError(
            f"{name} columns are {frame.columns.tolist()}, expected {expected}"
        )


def standardized_mean_difference(train: pd.Series, test: pd.Series) -> float:
    """Compute a scale-free train/test mean difference for one feature."""

    pooled_variance = (train.var(ddof=1) + test.var(ddof=1)) / 2.0
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return 0.0
    return float((train.mean() - test.mean()) / np.sqrt(pooled_variance))


def build_split_summary(
    train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Summarize numeric feature distributions and univariate drift."""

    rows: list[dict[str, float | str]] = []
    for feature in TEXT_FEATURES:
        ks_result = ks_2samp(train[feature], test[feature])
        rows.append(
            {
                "feature": feature,
                "train_mean": float(train[feature].mean()),
                "test_mean": float(test[feature].mean()),
                "train_median": float(train[feature].median()),
                "test_median": float(test[feature].median()),
                "train_p95": float(train[feature].quantile(0.95)),
                "test_p95": float(test[feature].quantile(0.95)),
                "standardized_mean_difference": standardized_mean_difference(
                    train[feature], test[feature]
                ),
                "ks_statistic": float(ks_result.statistic),
                "ks_pvalue": float(ks_result.pvalue),
            }
        )
    return pd.DataFrame(rows)


def build_score_summary(train: pd.DataFrame) -> pd.DataFrame:
    """Summarize class balance and major style features by target score."""

    rows: list[dict[str, float | int]] = []
    for score, group in train.groupby(TARGET_COLUMN, sort=True):
        rows.append(
            {
                "score": int(score),
                "count": int(len(group)),
                "percent": float(100.0 * len(group) / len(train)),
                "mean_words": float(group["word_count"].mean()),
                "median_words": float(group["word_count"].median()),
                "mean_chars": float(group["char_count"].mean()),
                "median_sentences": float(group["sentence_count"].median()),
                "median_paragraphs": float(group["paragraph_count"].median()),
                "mean_unique_word_ratio": float(group["unique_word_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_score_correlations(train: pd.DataFrame) -> pd.DataFrame:
    """Measure monotonic associations between text features and essay score."""

    rows: list[dict[str, float | str]] = []
    for feature in TEXT_FEATURES:
        result = spearmanr(train[TARGET_COLUMN], train[feature])
        rows.append(
            {
                "feature": feature,
                "spearman_correlation": float(result.statistic),
                "pvalue": float(result.pvalue),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "spearman_correlation", key=lambda values: values.abs(), ascending=False
    )


def build_feature_quantiles(
    train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Return detailed train/test quantiles used to diagnose long tails."""

    rows: list[dict[str, float | str]] = []
    for split_name, frame in (("train", train), ("test", test)):
        for feature in TEXT_FEATURES:
            row: dict[str, float | str] = {"split": split_name, "feature": feature}
            for label, quantile in QUANTILES.items():
                row[label] = float(frame[feature].quantile(quantile))
            rows.append(row)
    return pd.DataFrame(rows)


def build_fold_score_counts(train: pd.DataFrame) -> pd.DataFrame:
    """Check score-only stratification feasibility before grouped folds exist."""

    folds = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    rows: list[dict[str, float | int]] = []
    for fold, (_, validation_indices) in enumerate(
        folds.split(train, train[TARGET_COLUMN])
    ):
        validation_scores = train.iloc[validation_indices][TARGET_COLUMN]
        counts = validation_scores.value_counts().sort_index()
        for score in range(SCORE_MIN, SCORE_MAX + 1):
            count = int(counts.get(score, 0))
            rows.append(
                {
                    "fold": fold,
                    "score": score,
                    "count": count,
                    "percent": float(100.0 * count / len(validation_scores)),
                }
            )
    result = pd.DataFrame(rows)
    if (result["count"] == 0).any():
        raise ValueError("at least one validation fold is missing a score level")
    return result


def adversarial_validation_auc(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[float, float, list[float]]:
    """Estimate train/test drift from style features using domain classification."""

    combined = pd.concat(
        [train[TEXT_FEATURES], test[TEXT_FEATURES]], ignore_index=True
    )
    domain = np.concatenate(
        [np.zeros(len(train), dtype=np.int8), np.ones(len(test), dtype=np.int8)]
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_SEED,
        ),
    )
    folds = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    scores = cross_val_score(
        model,
        combined,
        domain,
        scoring="roc_auc",
        cv=folds,
        n_jobs=1,
    )
    return float(scores.mean()), float(scores.std(ddof=1)), scores.tolist()


def dataframe_to_markdown(
    frame: pd.DataFrame, digits: int = 4
) -> str:
    """Render a small DataFrame as Markdown without an optional dependency."""

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


def create_overview_plot(
    train: pd.DataFrame,
    test: pd.DataFrame,
    score_correlations: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create one compact visual summary for manual inspection."""

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(19, 10))

    score_counts = train[TARGET_COLUMN].value_counts().sort_index()
    axes[0, 0].bar(score_counts.index.astype(str), score_counts.values)
    axes[0, 0].set_title("Training score distribution")
    axes[0, 0].set_xlabel("Score")
    axes[0, 0].set_ylabel("Essays")
    for index, value in enumerate(score_counts.values):
        axes[0, 0].text(index, value, f"{value:,}", ha="center", va="bottom")

    word_limit = float(
        pd.concat([train["word_count"], test["word_count"]]).quantile(0.99)
    )
    combined_word_counts = pd.concat(
        [train["word_count"], test["word_count"]], ignore_index=True
    )
    word_bins = np.linspace(float(combined_word_counts.min()), word_limit, 46)
    axes[0, 1].hist(
        train.loc[train["word_count"] <= word_limit, "word_count"],
        bins=word_bins,
        density=True,
        alpha=0.55,
        label="train",
    )
    axes[0, 1].hist(
        test.loc[test["word_count"] <= word_limit, "word_count"],
        bins=word_bins,
        density=True,
        alpha=0.55,
        label="test",
    )
    axes[0, 1].set_title("Word-count distribution (clipped at p99)")
    axes[0, 1].set_xlabel("Words")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()

    char_limit = float(
        pd.concat([train["char_count"], test["char_count"]]).quantile(0.99)
    )
    combined_char_counts = pd.concat(
        [train["char_count"], test["char_count"]], ignore_index=True
    )
    char_bins = np.linspace(float(combined_char_counts.min()), char_limit, 46)
    axes[0, 2].hist(
        train.loc[train["char_count"] <= char_limit, "char_count"],
        bins=char_bins,
        density=True,
        alpha=0.55,
        label="train",
    )
    axes[0, 2].hist(
        test.loc[test["char_count"] <= char_limit, "char_count"],
        bins=char_bins,
        density=True,
        alpha=0.55,
        label="test",
    )
    axes[0, 2].set_title("Character-count distribution (clipped at p99)")
    axes[0, 2].set_xlabel("Characters")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].legend()

    sns.boxplot(
        data=train,
        x=TARGET_COLUMN,
        y="word_count",
        showfliers=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Word count by essay score")
    axes[1, 0].set_xlabel("Score")
    axes[1, 0].set_ylabel("Words")

    sns.boxplot(
        data=train,
        x=TARGET_COLUMN,
        y="char_count",
        showfliers=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Character count by essay score")
    axes[1, 1].set_xlabel("Score")
    axes[1, 1].set_ylabel("Characters")

    correlation_plot = score_correlations.sort_values("spearman_correlation")
    colors = [
        "#c44e52" if value < 0 else "#4c72b0"
        for value in correlation_plot["spearman_correlation"]
    ]
    axes[1, 2].barh(
        correlation_plot["feature"],
        correlation_plot["spearman_correlation"],
        color=colors,
    )
    axes[1, 2].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 2].set_title("Feature association with score (Spearman)")
    axes[1, 2].set_xlabel("Correlation")

    figure.suptitle("Automated Essay Scoring — EDA Overview", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def interpret_domain_auc(auc: float) -> str:
    """Provide a conservative, plain-language interpretation of domain AUC."""

    if auc < 0.55:
        return "little detectable drift in the measured style features"
    if auc < 0.65:
        return "mild detectable drift in the measured style features"
    return "material detectable drift that should be investigated before modeling"


def main() -> None:
    """Load the challenge data, validate it, and write EDA artifacts."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    require_columns(
        train_raw,
        [ID_COLUMN, TEXT_COLUMN, TARGET_COLUMN],
        "train.csv",
    )
    require_columns(test_raw, [ID_COLUMN, TEXT_COLUMN], "test.csv")
    require_columns(
        sample_submission,
        [ID_COLUMN, TARGET_COLUMN],
        "sample_submission.csv",
    )

    if train_raw[ID_COLUMN].duplicated().any():
        raise ValueError("train.csv contains duplicate essay IDs")
    if test_raw[ID_COLUMN].duplicated().any():
        raise ValueError("test.csv contains duplicate essay IDs")
    if set(train_raw[ID_COLUMN]) & set(test_raw[ID_COLUMN]):
        raise ValueError("train and test essay IDs overlap")
    if train_raw[ID_COLUMN].isna().any() or test_raw[ID_COLUMN].isna().any():
        raise ValueError("essay_id contains missing values")
    if train_raw[ID_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("train.csv contains blank essay IDs")
    if test_raw[ID_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("test.csv contains blank essay IDs")
    if not sample_submission[ID_COLUMN].equals(test_raw[ID_COLUMN]):
        raise ValueError("sample submission IDs do not match test IDs and order")
    if train_raw[TEXT_COLUMN].isna().any() or test_raw[TEXT_COLUMN].isna().any():
        raise ValueError("full_text contains missing values")
    if train_raw[TEXT_COLUMN].str.strip().eq("").any():
        raise ValueError("train.csv contains blank essays")
    if test_raw[TEXT_COLUMN].str.strip().eq("").any():
        raise ValueError("test.csv contains blank essays")
    if not train_raw[TARGET_COLUMN].between(SCORE_MIN, SCORE_MAX).all():
        raise ValueError("training scores fall outside the expected 1..6 range")
    if not np.allclose(
        train_raw[TARGET_COLUMN], train_raw[TARGET_COLUMN].astype(int)
    ):
        raise ValueError("training scores must be integers")
    expected_scores = set(range(SCORE_MIN, SCORE_MAX + 1))
    if set(train_raw[TARGET_COLUMN].astype(int)) != expected_scores:
        raise ValueError("training data does not contain every score from 1 through 6")

    train = add_text_features(train_raw)
    test = add_text_features(test_raw)

    normalized_train = train[TEXT_COLUMN].map(normalize_text)
    normalized_test = test[TEXT_COLUMN].map(normalize_text)
    train_fingerprints = normalized_train.map(boundary_fingerprint)
    test_fingerprints = normalized_test.map(boundary_fingerprint)
    near_duplicate_candidates = build_near_duplicate_candidates(
        train,
        test,
        normalized_train,
        normalized_test,
        train_fingerprints,
        test_fingerprints,
    )
    confirmed_near_duplicates = near_duplicate_candidates.loc[
        near_duplicate_candidates["confirmed_near_duplicate"]
    ]
    normalized_score_counts = pd.DataFrame(
        {
            "normalized_text": normalized_train,
            TARGET_COLUMN: train[TARGET_COLUMN],
        }
    ).groupby("normalized_text")[TARGET_COLUMN].nunique()
    duplicate_summary = {
        "raw_duplicate_train_rows_all": int(
            train[TEXT_COLUMN].duplicated(keep=False).sum()
        ),
        "raw_duplicate_train_rows_beyond_first": int(
            train[TEXT_COLUMN].duplicated().sum()
        ),
        "raw_duplicate_train_groups": int(
            train[TEXT_COLUMN].value_counts().gt(1).sum()
        ),
        "raw_duplicate_test_rows_all": int(
            test[TEXT_COLUMN].duplicated(keep=False).sum()
        ),
        "raw_duplicate_test_rows_beyond_first": int(
            test[TEXT_COLUMN].duplicated().sum()
        ),
        "raw_duplicate_test_groups": int(
            test[TEXT_COLUMN].value_counts().gt(1).sum()
        ),
        "raw_cross_split_matching_test_rows": int(
            test[TEXT_COLUMN].isin(set(train[TEXT_COLUMN])).sum()
        ),
        "raw_cross_split_unique_texts": int(
            len(set(train[TEXT_COLUMN]) & set(test[TEXT_COLUMN]))
        ),
        "normalized_duplicate_train_rows_all": int(
            normalized_train.duplicated(keep=False).sum()
        ),
        "normalized_duplicate_train_rows_beyond_first": int(
            normalized_train.duplicated().sum()
        ),
        "normalized_duplicate_train_groups": int(
            normalized_train.value_counts().gt(1).sum()
        ),
        "normalized_duplicate_test_rows_all": int(
            normalized_test.duplicated(keep=False).sum()
        ),
        "normalized_duplicate_test_rows_beyond_first": int(
            normalized_test.duplicated().sum()
        ),
        "normalized_duplicate_test_groups": int(
            normalized_test.value_counts().gt(1).sum()
        ),
        "normalized_cross_split_matching_test_rows": int(
            normalized_test.isin(set(normalized_train)).sum()
        ),
        "normalized_cross_split_unique_texts": int(
            len(set(normalized_train) & set(normalized_test))
        ),
        "normalized_duplicate_groups_with_conflicting_scores": int(
            normalized_score_counts.gt(1).sum()
        ),
        "boundary_fingerprint_train_candidate_rows_all": int(
            train_fingerprints.duplicated(keep=False).sum()
        ),
        "boundary_fingerprint_train_candidate_groups": int(
            train_fingerprints.value_counts().gt(1).sum()
        ),
        "boundary_fingerprint_test_candidate_rows_all": int(
            test_fingerprints.duplicated(keep=False).sum()
        ),
        "boundary_fingerprint_test_candidate_groups": int(
            test_fingerprints.value_counts().gt(1).sum()
        ),
        "boundary_fingerprint_cross_split_matching_test_rows": int(
            test_fingerprints.isin(set(train_fingerprints)).sum()
        ),
        "boundary_fingerprint_cross_split_unique_fingerprints": int(
            len(set(train_fingerprints) & set(test_fingerprints))
        ),
    }
    near_duplicate_summary = {
        "boundary_candidate_pairs_total": int(len(near_duplicate_candidates)),
        "boundary_candidate_pairs_within_train": int(
            near_duplicate_candidates["relationship"].eq("within_train").sum()
        ),
        "boundary_candidate_pairs_within_test": int(
            near_duplicate_candidates["relationship"].eq("within_test").sum()
        ),
        "boundary_candidate_pairs_cross_split": int(
            near_duplicate_candidates["relationship"].eq("cross_split").sum()
        ),
        "confirmed_pairs_total": int(len(confirmed_near_duplicates)),
        "confirmed_pairs_within_train": int(
            confirmed_near_duplicates["relationship"].eq("within_train").sum()
        ),
        "confirmed_pairs_within_test": int(
            confirmed_near_duplicates["relationship"].eq("within_test").sum()
        ),
        "confirmed_pairs_cross_split": int(
            confirmed_near_duplicates["relationship"].eq("cross_split").sum()
        ),
        "confirmed_train_pairs_with_conflicting_scores": int(
            confirmed_near_duplicates["conflicting_train_scores"].sum()
        ),
        "confirmation_thresholds": {
            "word_shingle_size": NEAR_DUPLICATE_SHINGLE_SIZE,
            "minimum_shingle_jaccard": NEAR_DUPLICATE_JACCARD_THRESHOLD,
            "minimum_word_sequence_ratio": NEAR_DUPLICATE_SEQUENCE_THRESHOLD,
        },
    }

    split_summary = build_split_summary(train, test)
    score_summary = build_score_summary(train)
    score_correlations = build_score_correlations(train)
    feature_quantiles = build_feature_quantiles(train, test)
    fold_score_counts = build_fold_score_counts(train)
    domain_auc, domain_auc_std, domain_fold_scores = adversarial_validation_auc(
        train, test
    )

    split_summary.to_csv(REPORTS_DIR / "split_feature_summary.csv", index=False)
    score_summary.to_csv(REPORTS_DIR / "score_feature_summary.csv", index=False)
    score_correlations.to_csv(
        REPORTS_DIR / "score_feature_correlations.csv", index=False
    )
    feature_quantiles.to_csv(REPORTS_DIR / "feature_quantiles.csv", index=False)
    fold_score_counts.to_csv(REPORTS_DIR / "fold_score_counts.csv", index=False)
    near_duplicate_candidates.to_csv(
        REPORTS_DIR / "near_duplicate_candidates.csv", index=False
    )
    create_overview_plot(
        train,
        test,
        score_correlations,
        FIGURES_DIR / "eda_overview.png",
    )

    class_counts = train[TARGET_COLUMN].value_counts().sort_index()
    max_smd_row = split_summary.iloc[
        split_summary["standardized_mean_difference"].abs().argmax()
    ]
    top_correlation = score_correlations.iloc[0]
    summary = {
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH)
        },
        "shapes": {
            "train": [int(value) for value in train_raw.shape],
            "test": [int(value) for value in test_raw.shape],
            "sample_submission": [int(value) for value in sample_submission.shape],
        },
        "data_quality": {
            "missing_train_texts": int(train_raw[TEXT_COLUMN].isna().sum()),
            "missing_test_texts": int(test_raw[TEXT_COLUMN].isna().sum()),
            "duplicate_train_ids": int(train_raw[ID_COLUMN].duplicated().sum()),
            "duplicate_test_ids": int(test_raw[ID_COLUMN].duplicated().sum()),
            "train_test_id_overlap": int(
                len(set(train_raw[ID_COLUMN]) & set(test_raw[ID_COLUMN]))
            ),
            "sample_matches_test_order": bool(
                sample_submission[ID_COLUMN].equals(test_raw[ID_COLUMN])
            ),
            **duplicate_summary,
        },
        "near_duplicate_audit": near_duplicate_summary,
        "score_counts": {
            str(int(score)): int(count) for score, count in class_counts.items()
        },
        "score_percentages": {
            str(int(score)): float(100.0 * count / len(train))
            for score, count in class_counts.items()
        },
        "score_statistics": {
            "mean": float(train[TARGET_COLUMN].mean()),
            "median": float(train[TARGET_COLUMN].median()),
            "std": float(train[TARGET_COLUMN].std(ddof=1)),
        },
        "extreme_text_counts": {
            "train_under_100_words": int((train["word_count"] < 100).sum()),
            "test_under_100_words": int((test["word_count"] < 100).sum()),
            "train_without_sentence_end_punctuation": int(
                train[TEXT_COLUMN].map(lambda text: not SENTENCE_RE.search(text)).sum()
            ),
            "test_without_sentence_end_punctuation": int(
                test[TEXT_COLUMN].map(lambda text: not SENTENCE_RE.search(text)).sum()
            ),
            "train_with_nul_character": int(
                train[TEXT_COLUMN].str.contains("\x00", regex=False).sum()
            ),
            "test_with_nul_character": int(
                test[TEXT_COLUMN].str.contains("\x00", regex=False).sum()
            ),
        },
        "domain_validation": {
            "numeric_style_feature_auc_mean": domain_auc,
            "numeric_style_feature_auc_std": domain_auc_std,
            "fold_scores": domain_fold_scores,
            "interpretation": interpret_domain_auc(domain_auc),
        },
        "largest_absolute_standardized_mean_difference": {
            "feature": str(max_smd_row["feature"]),
            "value": float(max_smd_row["standardized_mean_difference"]),
        },
        "strongest_absolute_score_correlation": {
            "feature": str(top_correlation["feature"]),
            "spearman": float(top_correlation["spearman_correlation"]),
        },
    }
    (REPORTS_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rarest_score = int(class_counts.idxmin())
    rarest_count = int(class_counts.min())
    report = f"""# Exploratory Data Analysis

## Executive summary

- The downloaded split contains **{len(train):,} training essays** and
  **{len(test):,} test essays**.
- All required IDs and texts pass integrity checks. Normalized duplicate checks
  found **{duplicate_summary['normalized_duplicate_train_rows_all']} involved
  train rows**, **{duplicate_summary['normalized_duplicate_test_rows_all']}
  involved test rows**, and
  **{duplicate_summary['normalized_cross_split_unique_texts']} unique cross-split
  matches**.
- Boundary fingerprints generated
  **{near_duplicate_summary['boundary_candidate_pairs_total']} candidate pairs**.
  Full-text similarity confirmed
  **{near_duplicate_summary['confirmed_pairs_within_train']} within-train** and
  **{near_duplicate_summary['confirmed_pairs_cross_split']} cross-split** pair;
  **{near_duplicate_summary['confirmed_train_pairs_with_conflicting_scores']}**
  confirmed training pair has conflicting labels. Candidate generation is
  heuristic and does not rule out every possible near duplicate.
- The rarest class is score **{rarest_score}**, with **{rarest_count:,} essays
  ({100.0 * rarest_count / len(train):.2f}%)**. Stratified validation is required.
- The strongest univariate association with score is
  **{top_correlation['feature']}** (Spearman
  **{top_correlation['spearman_correlation']:.3f}**).
- Five-fold adversarial validation using numeric style features produced ROC AUC
  **{domain_auc:.3f} +/- {domain_auc_std:.3f}**, indicating
  **{interpret_domain_auc(domain_auc)}**.
- The largest train/test standardized mean difference is
  **{max_smd_row['feature']}={max_smd_row['standardized_mean_difference']:.3f}**.

## Score and feature summary

{dataframe_to_markdown(score_summary)}

## Train/test feature comparison

{dataframe_to_markdown(split_summary.drop(columns=['ks_pvalue']))}

## Feature association with score

{dataframe_to_markdown(score_correlations)}

## Score-only stratification feasibility audit

This table only checks whether all classes can be balanced across five folds.
The persisted, near-duplicate-safe split is generated separately by
`python -m src.make_folds` and audited in `validation_report.md`.

{dataframe_to_markdown(fold_score_counts)}

## Modeling implications

1. Use five-fold stratification by score and retain complete out-of-fold
   predictions for QWK evaluation and threshold calibration.
2. Treat this as ordinal regression: train on continuous scores, then map to
   integers 1 through 6 with ordered thresholds.
3. Preserve punctuation, capitalization, spelling, and paragraph breaks because
   they carry scoring information.
4. Include word and character TF-IDF together with basic length/style signals.
5. Report both fixed-threshold and calibrated-threshold QWK so calibration gains
   are not confused with model gains.
6. Fit every vectorizer on the training portion of each fold only; transform the
   validation and test partitions without refitting.
7. Keep confirmed near-duplicate training essays in the same validation fold.
   Retain their supplied labels even when they conflict, and do not copy a
   training label onto a similar test essay.
8. Interpret type-token ratio cautiously because it mechanically decreases with
   essay length and is strongly confounded by length here.

![EDA overview](figures/eda_overview.png)
"""
    (REPORTS_DIR / "eda_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote {REPORTS_DIR / 'eda_report.md'}")
    print(f"Wrote {REPORTS_DIR / 'eda_summary.json'}")
    print(f"Wrote {FIGURES_DIR / 'eda_overview.png'}")
    print(f"Domain-classification AUC: {domain_auc:.4f} +/- {domain_auc_std:.4f}")
    print(
        "Strongest score correlation: "
        f"{top_correlation['feature']}="
        f"{top_correlation['spearman_correlation']:.4f}"
    )


if __name__ == "__main__":
    main()
