# Q1 Demo Model Effect Report

## Current Demo Decision

The current frozen Demo model is **Ridge regression** with fold-fitted one-hot categorical encoding and standardized numeric features. It is selected solely because it has the lowest mean original-scale temporal-validation MSE among the Demo candidates and is better in every validation period.

## Common Evaluation Contract

- Training labels: 560 rows, converted from people to millions of people.
- Hidden official test: 140 rows, never used for model selection.
- Validation: three expanding time windows grouped by complete match date.
- Metrics: MSE is primary; RMSE and MAE are auxiliary, all in millions of people.
- Feature boundary: only shared pre-match information. Current-match results, xG, shots, possession, attendance, IDs, and future-only composite commercial indices are excluded.

## Leaderboard

| model | mse | mse_std | rmse | mae | worst_fold_mse |
| --- | --- | --- | --- | --- | --- |
| ridge | 225.0601 | 48.8524 | 14.9446 | 11.8346 | 278.9919 |
| hist_gradient_boosting | 272.9405 | 76.5655 | 16.4180 | 13.1340 | 360.2006 |
| mean | 581.8325 | 64.2237 | 24.0963 | 19.3522 | 640.6581 |

## Fold Evidence

| model | fold | mse | rmse | mae | n_validation |
| --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting | fold_1 | 360.2006 | 18.9790 | 14.7501 | 122.0000 |
| hist_gradient_boosting | fold_2 | 241.6204 | 15.5441 | 12.4656 | 108.0000 |
| hist_gradient_boosting | fold_3 | 217.0006 | 14.7309 | 12.1863 | 132.0000 |
| mean | fold_1 | 640.6581 | 25.3112 | 20.3965 | 122.0000 |
| mean | fold_2 | 513.3129 | 22.6564 | 18.3901 | 108.0000 |
| mean | fold_3 | 591.5264 | 24.3213 | 19.2699 | 132.0000 |
| ridge | fold_1 | 278.9919 | 16.7031 | 13.0983 | 122.0000 |
| ridge | fold_2 | 212.4149 | 14.5745 | 11.5962 | 108.0000 |
| ridge | fold_3 | 183.7736 | 13.5563 | 10.8093 | 132.0000 |

Ridge improves mean MSE by 61.3% versus the mean baseline. Its newest-fold MSE is 183.77, below both other Demo candidates.

## Outputs

- `result_1_test_prediction.csv`: 140 rows, values in millions of viewers.
- `result_1_match_prediction.csv`: 72 rows, values in people.
- Independent reload validation: PASS.
- SHA-256 values: see `artifact_hashes.json`.

## Figures

- `figures/model_comparison.png`: candidate leaderboard and fold-level MSE.
- `figures/ridge_diagnostics.png`: actual-predicted relation and temporal residuals.

## Interpretation And Limitations

This is a valid end-to-end Demo, not a final optimality claim. Ridge outperforms the initial constrained histogram gradient boosting model, indicating that the current small, common feature set has a largely stable linear signal or that the tree model is under-tuned. The model has not yet compared CatBoost, LightGBM, XGBoost, Extra Trees, SVR, target transformation variants, or a tightly controlled two-model blend. The future 72-match setting is a World Cup group-stage domain shift while the historical data contains only a limited number of World Cup matches; therefore later iterations must report subgroup performance and cannot infer hidden-test superiority from this Demo alone.

## Next Upgrade Question

The next high-stakes reasoning step is whether to extend the feature contract with strictly lagged historical-form variables and how to compare boosted-tree models under the World Cup domain shift. Per the project rule, switching to `sol/ultra` for that route decision requires explicit user approval first.
