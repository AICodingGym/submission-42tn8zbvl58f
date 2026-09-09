"""Create and audit one deterministic grouped five-fold assignment."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

try:
    from src.config import (
        EDA_SUMMARY_PATH,
        FOLD_ASSIGNMENTS_PATH,
        FOLDS_DIR,
        ID_COLUMN,
        NEAR_DUPLICATE_CANDIDATES_PATH,
        N_SPLITS,
        RANDOM_SEED,
        REPORTS_DIR,
        SCORE_MAX,
        SCORE_MIN,
        TARGET_COLUMN,
        TEXT_COLUMN,
        TRAIN_PATH,
    )
    from src.metrics import apply_ordered_thresholds, quadratic_weighted_kappa
except ModuleNotFoundError:  # Support `python src/make_folds.py`.
    from config import (  # type: ignore[no-redef]
        EDA_SUMMARY_PATH,
        FOLD_ASSIGNMENTS_PATH,
        FOLDS_DIR,
        ID_COLUMN,
        NEAR_DUPLICATE_CANDIDATES_PATH,
        N_SPLITS,
        RANDOM_SEED,
        REPORTS_DIR,
        SCORE_MAX,
        SCORE_MIN,
        TARGET_COLUMN,
        TEXT_COLUMN,
        TRAIN_PATH,
    )
    from metrics import (  # type: ignore[no-redef]
        apply_ordered_thresholds,
        quadratic_weighted_kappa,
    )


FOLD_SUMMARY_PATH = REPORTS_DIR / "fold_assignment_summary.csv"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.md"
VALIDATION_SUMMARY_PATH = REPORTS_DIR / "validation_summary.json"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a local artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_boolean(series: pd.Series, name: str) -> pd.Series:
    """Parse strict CSV booleans without treating non-empty strings as true."""

    parsed = series.astype(str).str.strip().str.casefold().map(
        {"true": True, "false": False}
    )
    if parsed.isna().any():
        raise ValueError(f"{name} contains values other than true/false")
    return parsed.astype(bool)


def validate_eda_inputs() -> None:
    """Ensure duplicate candidates were produced from the current train file."""

    if not EDA_SUMMARY_PATH.exists() or not NEAR_DUPLICATE_CANDIDATES_PATH.exists():
        raise FileNotFoundError(
            "EDA artifacts are missing; run `python -m src.eda` first"
        )
    eda_summary = json.loads(EDA_SUMMARY_PATH.read_text(encoding="utf-8"))
    recorded_hash = eda_summary.get("files", {}).get(TRAIN_PATH.name, {}).get("sha256")
    current_hash = file_sha256(TRAIN_PATH)
    if recorded_hash != current_hash:
        raise ValueError(
            "EDA artifacts do not match the current train.csv; rerun src.eda"
        )


def normalized_text_hashes(train: pd.DataFrame) -> pd.Series:
    """Hash normalized text so exact duplicate grouping is reproducible."""

    def normalized_hash(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        normalized = " ".join(normalized.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    return train[TEXT_COLUMN].map(normalized_hash).rename("normalized_text_sha256")


def build_group_ids(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    text_hashes: pd.Series,
) -> pd.Series:
    """Create connected components from confirmed within-train duplicate pairs."""

    required_columns = {
        "relationship",
        "left_essay_id",
        "right_essay_id",
        "confirmed_near_duplicate",
    }
    missing_columns = required_columns - set(candidates.columns)
    if missing_columns:
        raise ValueError(
            "near-duplicate audit is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    essay_ids = train[ID_COLUMN].astype(str)
    known_ids = set(essay_ids)
    parent = {essay_id: essay_id for essay_id in essay_ids}

    def find(essay_id: str) -> str:
        while parent[essay_id] != essay_id:
            parent[essay_id] = parent[parent[essay_id]]
            essay_id = parent[essay_id]
        return essay_id

    def union(left_id: str, right_id: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root == right_root:
            return
        canonical_root = min(left_root, right_root)
        other_root = max(left_root, right_root)
        parent[other_root] = canonical_root

    exact_duplicate_groups = pd.DataFrame(
        {ID_COLUMN: essay_ids, "normalized_text_sha256": text_hashes}
    ).groupby("normalized_text_sha256")[ID_COLUMN]
    for _, member_ids in exact_duplicate_groups:
        members = member_ids.astype(str).tolist()
        for right_id in members[1:]:
            union(members[0], right_id)

    confirmed = parse_boolean(
        candidates["confirmed_near_duplicate"],
        "confirmed_near_duplicate",
    )
    within_train = candidates.loc[
        confirmed & candidates["relationship"].eq("within_train")
    ]
    for row in within_train.itertuples(index=False):
        left_id = str(row.left_essay_id)
        right_id = str(row.right_essay_id)
        unknown_ids = {left_id, right_id} - known_ids
        if unknown_ids:
            raise ValueError(
                "near-duplicate audit references unknown train IDs: "
                + ", ".join(sorted(unknown_ids))
            )
        union(left_id, right_id)

    return essay_ids.map(find).rename("group_id")


def make_assignments(
    train: pd.DataFrame,
    group_ids: pd.Series,
    text_hashes: pd.Series,
) -> pd.DataFrame:
    """Assign every training essay to one deterministic validation fold."""

    working = pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN].astype(str),
            TARGET_COLUMN: train[TARGET_COLUMN].astype(int),
            "group_id": group_ids.astype(str),
            "normalized_text_sha256": text_hashes.astype(str),
        }
    ).sort_values(ID_COLUMN, kind="stable", ignore_index=True)
    distinct_groups_per_score = (
        working[[TARGET_COLUMN, "group_id"]]
        .drop_duplicates()
        .groupby(TARGET_COLUMN)["group_id"]
        .nunique()
    )
    if (distinct_groups_per_score < N_SPLITS).any():
        raise ValueError(
            "each score must occur in at least one distinct group per fold"
        )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    fold_values = np.full(len(working), -1, dtype=np.int8)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(
            working,
            working[TARGET_COLUMN],
            groups=working["group_id"],
        )
    ):
        if (fold_values[validation_indices] != -1).any():
            raise RuntimeError("an essay was assigned to more than one fold")
        fold_values[validation_indices] = fold

    if (fold_values < 0).any():
        raise RuntimeError("at least one essay was not assigned to a fold")
    working["fold"] = fold_values.astype(int)
    group_sizes = working["group_id"].value_counts()
    working["group_size"] = working["group_id"].map(group_sizes).astype(int)
    assignments = working[
        [
            ID_COLUMN,
            TARGET_COLUMN,
            "fold",
            "group_id",
            "group_size",
            "normalized_text_sha256",
        ]
    ]
    if assignments.groupby("group_id")["fold"].nunique().max() != 1:
        raise RuntimeError("a near-duplicate group crosses validation folds")
    return assignments


def persist_immutable_assignments(assignments: pd.DataFrame) -> str:
    """Write once, then reject any silently changed fold mapping."""

    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    if FOLD_ASSIGNMENTS_PATH.exists():
        existing = pd.read_csv(
            FOLD_ASSIGNMENTS_PATH,
            dtype={
                ID_COLUMN: str,
                "group_id": str,
                "normalized_text_sha256": str,
            },
        )
        expected_columns = assignments.columns.tolist()
        if existing.columns.tolist() != expected_columns:
            raise RuntimeError(
                "existing fold assignment has an unexpected schema; "
                "remove it explicitly before regenerating"
            )
        comparable = existing.astype(
            {TARGET_COLUMN: int, "fold": int, "group_size": int}
        ).reset_index(drop=True)
        if not comparable.equals(assignments.reset_index(drop=True)):
            raise RuntimeError(
                "new fold mapping differs from the persisted mapping; "
                "refusing to overwrite it"
            )
        return "verified_existing"

    assignments.to_csv(FOLD_ASSIGNMENTS_PATH, index=False)
    return "created"


def build_fold_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    """Build long-form fold and score counts for easy auditing."""

    counts = (
        assignments.groupby(["fold", TARGET_COLUMN], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    fold_sizes = assignments.groupby("fold").size().rename("fold_size")
    counts = counts.merge(fold_sizes, on="fold", validate="many_to_one")
    counts["percent"] = 100.0 * counts["count"] / counts["fold_size"]
    return counts


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact DataFrame without requiring tabulate."""

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{value:.3f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    """Generate the fixed folds and machine-readable audit artifacts."""

    validate_eda_inputs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    if train.columns.tolist() != [ID_COLUMN, TEXT_COLUMN, TARGET_COLUMN]:
        raise ValueError("train.csv has an unexpected schema")
    if train[ID_COLUMN].duplicated().any():
        raise ValueError("train.csv contains duplicate essay IDs")
    if not train[TARGET_COLUMN].between(SCORE_MIN, SCORE_MAX).all():
        raise ValueError("train scores fall outside the expected range")

    candidates = pd.read_csv(
        NEAR_DUPLICATE_CANDIDATES_PATH,
        dtype={"left_essay_id": str, "right_essay_id": str},
    )
    text_hashes = normalized_text_hashes(train)
    group_ids = build_group_ids(train, candidates, text_hashes)
    assignments = make_assignments(train, group_ids, text_hashes)
    repeated_assignments = make_assignments(train, group_ids, text_hashes)
    if not repeated_assignments.equals(assignments):
        raise RuntimeError("fold generation is not deterministic")
    persistence_status = persist_immutable_assignments(assignments)
    fold_summary = build_fold_summary(assignments)
    fold_summary.to_csv(FOLD_SUMMARY_PATH, index=False)

    expected_folds = set(range(N_SPLITS))
    observed_folds = set(assignments["fold"])
    fold_score_combinations = fold_summary[["fold", TARGET_COLUMN]].shape[0]
    expected_combinations = N_SPLITS * (SCORE_MAX - SCORE_MIN + 1)
    group_sizes = assignments.groupby("group_id").size()
    per_score_spread = fold_summary.groupby(TARGET_COLUMN)["count"].agg(
        lambda values: int(values.max() - values.min())
    )

    metric_truth = list(range(SCORE_MIN, SCORE_MAX + 1))
    metric_cases = {
        "perfect_qwk": quadratic_weighted_kappa(metric_truth, metric_truth),
        "reverse_qwk": quadratic_weighted_kappa(metric_truth, metric_truth[::-1]),
        "constant_qwk": quadratic_weighted_kappa(metric_truth, [3] * 6),
        "near_qwk": quadratic_weighted_kappa(
            metric_truth,
            [2, 2, 3, 4, 5, 5],
        ),
    }
    threshold_probe = apply_ordered_thresholds(
        [1.5, 1.500001, 2.5, 5.5, 5.500001]
    ).tolist()
    expected_metric_cases = {
        "perfect_qwk": 1.0,
        "reverse_qwk": -1.0,
        "constant_qwk": 0.0,
        "near_qwk": 0.9259259259259259,
    }
    if any(
        not np.isclose(metric_cases[name], expected)
        for name, expected in expected_metric_cases.items()
    ):
        raise RuntimeError("QWK smoke tests failed")
    if threshold_probe != [1, 2, 2, 5, 6]:
        raise RuntimeError("ordered-threshold smoke test failed")
    if observed_folds != expected_folds:
        raise RuntimeError("fold IDs are incomplete")
    if fold_score_combinations != expected_combinations:
        raise RuntimeError("at least one fold is missing a score class")
    training_score_coverage = {
        fold: set(assignments.loc[assignments["fold"] != fold, TARGET_COLUMN])
        for fold in expected_folds
    }
    expected_scores = set(range(SCORE_MIN, SCORE_MAX + 1))
    if any(scores != expected_scores for scores in training_score_coverage.values()):
        raise RuntimeError("at least one fold's training partition is missing a score")

    group_label_counts = assignments.groupby("group_id")[TARGET_COLUMN].nunique()
    overall_score_rates = assignments[TARGET_COLUMN].value_counts(normalize=True)
    fold_score_rates = (
        fold_summary.pivot(index="fold", columns=TARGET_COLUMN, values="percent")
        / 100.0
    )
    maximum_rate_deviation = float(
        fold_score_rates.sub(overall_score_rates, axis="columns")
        .abs()
        .to_numpy()
        .max()
    )

    summary: dict[str, Any] = {
        "splitter": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_seed": RANDOM_SEED,
        "scikit_learn_version": sklearn.__version__,
        "assignment_rows": int(len(assignments)),
        "assignment_sha256": file_sha256(FOLD_ASSIGNMENTS_PATH),
        "train_sha256": file_sha256(TRAIN_PATH),
        "immutable_mapping_check": "passed",
        "near_duplicate_candidates_sha256": file_sha256(
            NEAR_DUPLICATE_CANDIDATES_PATH
        ),
        "eda_summary_sha256": file_sha256(EDA_SUMMARY_PATH),
        "normalization": "Unicode NFKC + casefold + whitespace collapse",
        "group_audit": {
            "group_count": int(group_sizes.size),
            "multi_essay_group_count": int(group_sizes.gt(1).sum()),
            "essays_in_multi_essay_groups": int(group_sizes[group_sizes.gt(1)].sum()),
            "largest_group_size": int(group_sizes.max()),
            "groups_with_conflicting_scores": int(group_label_counts.gt(1).sum()),
            "groups_crossing_folds": int(
                assignments.groupby("group_id")["fold"].nunique().gt(1).sum()
            ),
        },
        "balance_audit": {
            "fold_sizes": {
                str(int(fold)): int(count)
                for fold, count in assignments["fold"]
                .value_counts()
                .sort_index()
                .items()
            },
            "maximum_fold_count_spread_by_score": int(per_score_spread.max()),
            "maximum_absolute_score_rate_deviation": maximum_rate_deviation,
            "fold_count_spread_by_score": {
                str(int(score)): int(spread)
                for score, spread in per_score_spread.items()
            },
            "every_fold_contains_every_score": True,
        },
        "metric_smoke_tests": {
            **metric_cases,
            "strict_threshold_probe": threshold_probe,
        },
    }
    VALIDATION_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = f"""# Validation Setup

- Splitter: **StratifiedGroupKFold**, `{N_SPLITS}` folds, shuffled with seed
  `{RANDOM_SEED}`.
- Assignment rows: **{len(assignments):,}**; every essay appears exactly once as
  validation data.
- Duplicate grouping: **{int(group_sizes.gt(1).sum())}** multi-essay group,
  **{int(assignments.groupby('group_id')['fold'].nunique().gt(1).sum())}** groups
  crossing folds.
- All six scores appear in every fold; the largest count difference for a score
  between folds is **{int(per_score_spread.max())}** essay(s).
- Metric smoke test: perfect predictions give QWK
  **{metric_cases['perfect_qwk']:.1f}**.
- The immutable mapping check passed. A different future mapping is rejected
  instead of silently overwriting the saved file.

## Fold distribution

{dataframe_to_markdown(fold_summary)}

## Usage contract

Join `artifacts/folds/fold_assignments.csv` to training data by `essay_id`.
Fit text vectorizers, feature transforms, models, and calibration only on rows
whose `fold` differs from the current validation fold. The confirmed conflicting
near-duplicate pair is kept together and therefore cannot leak across a fold.
"""
    VALIDATION_REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Fold mapping: {FOLD_ASSIGNMENTS_PATH} ({persistence_status})")
    print(f"Assignment SHA-256: {summary['assignment_sha256']}")
    print(f"Wrote {VALIDATION_REPORT_PATH}")
    print(f"Wrote {VALIDATION_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
