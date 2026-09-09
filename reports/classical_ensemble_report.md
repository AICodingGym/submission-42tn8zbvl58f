# Classical three-way ensemble

## Result

- Components: `0.80` formal TF-IDF/style blend +
  `0.10` raw-character Ridge +
  `0.10` word cumulative-logistic ordinal model.
- Validation: persisted `5` duplicate-safe grouped folds; all vectorizers
  and estimators are fitted inside each training fold.
- Complementary-fold threshold QWK: **0.805996**.
- Held-out fold QWK: **0.805994 +/- 0.004711**.
- Production-median-threshold OOF QWK: **0.807890**.
- Production thresholds: `[1.8661961716515134, 2.6407604664431434, 3.340777411460179, 4.121773554605743, 4.708563874702927]`.
- OOF RMSE: **0.582567**.
- Runtime: **344.0s** total (278.8s training,
  64.9s threshold calibration).
- Peak process RSS: **3060.4 MiB**.

## Component diagnostics

| component | fixed_threshold_qwk | cross_fitted_threshold_qwk | heldout_qwk_mean | heldout_qwk_std | production_median_threshold_qwk | raw_rmse | raw_mae | threshold_mean_std | threshold_max_std | production_threshold_1 | production_threshold_2 | production_threshold_3 | production_threshold_4 | production_threshold_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classical_ensemble | 0.76926 | 0.80600 | 0.80599 | 0.00471 | 0.80789 | 0.58257 | 0.45715 | 0.01702 | 0.03414 | 1.86620 | 2.64076 | 3.34078 | 4.12177 | 4.70856 |
| style_blend | 0.75911 | 0.80226 | 0.80222 | 0.00594 | 0.80465 | 0.58969 | 0.46118 | 0.01959 | 0.04744 | 1.94309 | 2.64297 | 3.32512 | 4.07448 | 4.67497 |
| raw_char | 0.76327 | 0.79033 | 0.79032 | 0.00468 | 0.79371 | 0.59888 | 0.47254 | 0.02296 | 0.03208 | 1.82065 | 2.61290 | 3.38121 | 4.14888 | 4.84170 |
| base_tfidf | 0.74895 | 0.78719 | 0.78715 | 0.00487 | 0.78937 | 0.60973 | 0.48221 | 0.00964 | 0.02094 | 1.89732 | 2.69730 | 3.38563 | 4.10313 | 4.81035 |
| word_ordinal | 0.76827 | 0.76755 | 0.76750 | 0.00371 | 0.77118 | 0.65503 | 0.50912 | 0.06658 | 0.09355 | 1.49492 | 2.63559 | 3.43889 | 4.52112 | 5.75507 |
| numeric_style | 0.71307 | 0.74476 | 0.74474 | 0.00428 | 0.74900 | 0.65931 | 0.50542 | 0.03522 | 0.09491 | 1.87047 | 2.59248 | 3.32989 | 4.11333 | 4.76084 |

## Ensemble fold diagnostics

| component | fold | calibration_rows | validation_rows | calibration_qwk | validation_qwk | optimizer_evaluations | threshold_1 | threshold_2 | threshold_3 | threshold_4 | threshold_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classical_ensemble | 0 | 12460 | 3116 | 0.80952 | 0.80037 | 4050 | 1.85578 | 2.63994 | 3.34359 | 4.12145 | 4.70837 |
| classical_ensemble | 1 | 12461 | 3115 | 0.80812 | 0.80658 | 4200 | 1.86620 | 2.64076 | 3.36651 | 4.12177 | 4.70856 |
| classical_ensemble | 2 | 12461 | 3115 | 0.80635 | 0.81309 | 4550 | 1.84528 | 2.64375 | 3.34078 | 4.17070 | 4.71629 |
| classical_ensemble | 3 | 12461 | 3115 | 0.80783 | 0.80643 | 4500 | 1.86786 | 2.63992 | 3.33780 | 4.12121 | 4.78638 |
| classical_ensemble | 4 | 12461 | 3115 | 0.80899 | 0.80351 | 3400 | 1.86744 | 2.64404 | 3.34073 | 4.17204 | 4.70849 |

## Prediction distribution

| score | train_true_count | oof_production_count | test_prediction_count | train_true_percent | oof_production_percent | test_prediction_percent |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1124 | 1133 | 124 | 7.21623 | 7.27401 | 7.16349 |
| 2 | 4249 | 4590 | 539 | 27.27915 | 29.46841 | 31.13807 |
| 3 | 5629 | 4868 | 543 | 36.13893 | 31.25321 | 31.36915 |
| 4 | 3563 | 3729 | 388 | 22.87494 | 23.94068 | 22.41479 |
| 5 | 876 | 997 | 112 | 5.62404 | 6.40087 | 6.47025 |
| 6 | 135 | 259 | 25 | 0.86672 | 1.66281 | 1.44425 |

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
