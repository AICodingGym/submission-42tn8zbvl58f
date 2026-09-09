# TF-IDF + Style Blend

## Result

- TF-IDF component: Ridge alpha 3 raw prediction, weight `0.65`.
- Style component: HistGradientBoosting over 12 deterministic
  length/style features, weight `0.35`.
- Baseline alpha-3 fixed-threshold OOF QWK:
  **0.74895**.
- Blend fixed-threshold OOF QWK: **0.75911**.
- Complementary-fold threshold stability QWK:
  **0.80189**.
- Production-median threshold OOF QWK:
  **0.80449**.
- Production thresholds: `[1.9050094145878114, 2.642963302795047, 3.325066299375084, 4.092550412058884, 4.675041210440747]`.
- Runtime: **73.6 seconds**.

## Fold diagnostics

| fold | train_rows | validation_rows | style_iterations | style_rmse | blend_fixed_qwk | calibration_qwk | validation_calibrated_qwk | threshold_1 | threshold_2 | threshold_3 | threshold_4 | threshold_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 12460 | 3116 | 153 | 0.665567 | 0.751257 | 0.806642 | 0.795038 | 1.839799 | 2.632228 | 3.345033 | 4.074104 | 4.673526 |
| 1 | 12461 | 3115 | 187 | 0.668064 | 0.756746 | 0.804974 | 0.803192 | 1.943495 | 2.642981 | 3.325066 | 4.074045 | 4.676332 |
| 2 | 12461 | 3115 | 136 | 0.652719 | 0.764991 | 0.802686 | 0.810855 | 1.849238 | 2.642963 | 3.316604 | 4.139379 | 4.740139 |
| 3 | 12461 | 3115 | 152 | 0.649965 | 0.763810 | 0.805255 | 0.801795 | 1.943369 | 2.642971 | 3.317010 | 4.092550 | 4.675041 |
| 4 | 12461 | 3115 | 188 | 0.660030 | 0.758708 | 0.805628 | 0.798526 | 1.905009 | 2.629653 | 3.326523 | 4.134994 | 4.664145 |

## Validation note

Each base and style OOF prediction excludes that row's label.  Blend weight and
threshold-family selection were nevertheless made using the shared OOF set, so
the calibrated scores are model-selection estimates rather than a fully nested,
unbiased performance estimate.  The complementary-fold score and threshold
spread are reported to check that the gain is not confined to one fold.

The ready-to-submit file is `submissions/tfidf_style_blend_calibrated.csv`.
