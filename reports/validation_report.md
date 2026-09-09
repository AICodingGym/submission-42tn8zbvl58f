# Validation Setup

- Splitter: **StratifiedGroupKFold**, `5` folds, shuffled with seed
  `42`.
- Assignment rows: **15,576**; every essay appears exactly once as
  validation data.
- Duplicate grouping: **1** multi-essay group,
  **0** groups
  crossing folds.
- All six scores appear in every fold; the largest count difference for a score
  between folds is **1** essay(s).
- Metric smoke test: perfect predictions give QWK
  **1.0**.
- The immutable mapping check passed. A different future mapping is rejected
  instead of silently overwriting the saved file.

## Fold distribution

| fold | score | count | fold_size | percent |
| --- | --- | --- | --- | --- |
| 0 | 1 | 225 | 3116 | 7.221 |
| 0 | 2 | 850 | 3116 | 27.279 |
| 0 | 3 | 1126 | 3116 | 36.136 |
| 0 | 4 | 712 | 3116 | 22.850 |
| 0 | 5 | 176 | 3116 | 5.648 |
| 0 | 6 | 27 | 3116 | 0.866 |
| 1 | 1 | 225 | 3115 | 7.223 |
| 1 | 2 | 850 | 3115 | 27.287 |
| 1 | 3 | 1126 | 3115 | 36.148 |
| 1 | 4 | 712 | 3115 | 22.857 |
| 1 | 5 | 175 | 3115 | 5.618 |
| 1 | 6 | 27 | 3115 | 0.867 |
| 2 | 1 | 225 | 3115 | 7.223 |
| 2 | 2 | 849 | 3115 | 27.255 |
| 2 | 3 | 1126 | 3115 | 36.148 |
| 2 | 4 | 713 | 3115 | 22.889 |
| 2 | 5 | 175 | 3115 | 5.618 |
| 2 | 6 | 27 | 3115 | 0.867 |
| 3 | 1 | 225 | 3115 | 7.223 |
| 3 | 2 | 850 | 3115 | 27.287 |
| 3 | 3 | 1125 | 3115 | 36.116 |
| 3 | 4 | 713 | 3115 | 22.889 |
| 3 | 5 | 175 | 3115 | 5.618 |
| 3 | 6 | 27 | 3115 | 0.867 |
| 4 | 1 | 224 | 3115 | 7.191 |
| 4 | 2 | 850 | 3115 | 27.287 |
| 4 | 3 | 1126 | 3115 | 36.148 |
| 4 | 4 | 713 | 3115 | 22.889 |
| 4 | 5 | 175 | 3115 | 5.618 |
| 4 | 6 | 27 | 3115 | 0.867 |

## Usage contract

Join `artifacts/folds/fold_assignments.csv` to training data by `essay_id`.
Fit text vectorizers, feature transforms, models, and calibration only on rows
whose `fold` differs from the current validation fold. The confirmed conflicting
near-duplicate pair is kept together and therefore cannot leak across a fold.
