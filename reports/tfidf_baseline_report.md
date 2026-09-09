# TF-IDF Ridge Baseline

## Result

- Features: word `(1,2)` TF-IDF plus character-within-word `(3,5)` TF-IDF.
- Validation: the persisted `5` grouped folds; each vectorizer is fitted
  only on that fold's training essays.
- Alpha selection: highest concatenated OOF fixed-threshold QWK among
  `[0.3, 1.0, 3.0]`; selected **1**.
- Concatenated OOF QWK with fixed thresholds `[1.5, 2.5, 3.5, 4.5, 5.5]`:
  **0.75845**.
- Fold QWK: **0.75840 +/-
  0.00833**.
- OOF RMSE: **0.61244**; MAE:
  **0.48386**.
- Severe OOF errors (absolute integer error at least 2):
  **246**.
- Runtime: **96.5 seconds**.

## Alpha comparison

| alpha | qwk_fixed | mae_raw | rmse_raw | severe_error_count | prediction_min | prediction_max | prediction_mean | prediction_std | fold_qwk_mean | fold_qwk_std | fold_rmse_mean | fold_rmse_std | selected_by_fixed_qwk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.30000 | 0.75375 | 0.50060 | 0.63290 | 297 | 0.26850 | 6.27159 | 2.97287 | 0.88251 | 0.75368 | 0.00884 | 0.63284 | 0.00935 | False |
| 1.00000 | 0.75845 | 0.48386 | 0.61244 | 246 | 0.44342 | 6.05423 | 2.97295 | 0.84361 | 0.75840 | 0.00833 | 0.61238 | 0.00986 | True |
| 3.00000 | 0.74895 | 0.48221 | 0.60973 | 253 | 0.59449 | 5.73239 | 2.97182 | 0.79021 | 0.74888 | 0.01008 | 0.60967 | 0.00994 | False |

## Selected model by fold

| fold | qwk_fixed | mae_raw | rmse_raw | severe_error_count | total_features | vectorizer_seconds | model_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.74906 | 0.49177 | 0.62508 | 62 | 202841 | 13.93940 | 1.63022 |
| 1 | 0.75710 | 0.48376 | 0.61262 | 49 | 202683 | 14.22450 | 1.59848 |
| 2 | 0.77197 | 0.46985 | 0.59737 | 47 | 202531 | 14.21282 | 1.59098 |
| 3 | 0.75677 | 0.48935 | 0.61387 | 40 | 202466 | 14.22354 | 1.59341 |
| 4 | 0.75708 | 0.48456 | 0.61294 | 48 | 202647 | 14.18098 | 1.63062 |

## Prediction distribution

| score | train_true_count | oof_fixed_count | test_fixed_count | train_true_percent | oof_fixed_percent | test_fixed_percent |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1124 | 511 | 53 | 7.21623 | 3.28069 | 3.06181 |
| 2 | 4249 | 4214 | 476 | 27.27915 | 27.05444 | 27.49856 |
| 3 | 5629 | 6666 | 749 | 36.13893 | 42.79661 | 43.26979 |
| 4 | 3563 | 3560 | 386 | 22.87494 | 22.85568 | 22.29925 |
| 5 | 876 | 603 | 63 | 5.62404 | 3.87134 | 3.63951 |
| 6 | 135 | 22 | 4 | 0.86672 | 0.14124 | 0.23108 |

## Leakage controls

The fold file checksum is checked before fitting. Training and validation groups
must be disjoint. Word and character vocabularies are fitted separately inside
each fold; validation and test texts are transform-only. The score thresholds
remain the fixed half-integers in this baseline and have not been optimized.

The ready-to-submit file is `submissions/tfidf_ridge_fixed.csv`.
