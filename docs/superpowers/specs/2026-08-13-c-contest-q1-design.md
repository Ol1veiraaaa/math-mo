# C Contest Q1 Design Specification

## Scope And Gate

This specification implements only Question 1: historical analysis, pre-match feature construction, television audience prediction, and the two required CSV files. Question 2 is out of scope. After Q1 verification, work stops at a result-review gate until the user explicitly accepts the model or requests another Q1 modeling iteration.

## Authoritative Inputs

- The fixed `historical_matches.dataset_split` supplies 560 training matches and 140 hidden-label test matches. It must not be changed.
- `teams` supplies static team attributes.
- `groups_matches` supplies the 72 future group matches.
- `base_predictions` supplies transparent pre-match win/draw/loss probabilities for those 72 matches.
- The official output templates define column names, column order, and units.
- Original workbooks, PDFs, and templates remain read-only.

## Target And Units

The workbook stores historical `tv_viewers` in persons even though the hidden-test scoring scale is millions of persons. The training target is therefore

`target_millions = tv_viewers / 1_000_000`.

MSE, RMSE, and MAE are calculated on this millions scale. The hidden-test CSV writes millions directly. The 72-match CSV explicitly converts predictions back to persons by multiplying by `1_000_000`.

## Prediction-Time Boundary

Only information known before kickoff may enter the production feature matrix. The forbidden field set is:

- `goals_a`, `goals_b`
- `xg_a`, `xg_b`
- `shots_a`, `shots_b`
- `possession_a`, `possession_b`
- `attendance`
- `tv_viewers`, except as the training label
- identifiers used as numeric predictors

The pipeline exposes a final feature manifest and fails if a forbidden field is present.

## Shared Historical/Future Feature Bridge

Historical decimal odds are converted to normalized implied probabilities:

`p_k = (1 / odds_k) / sum_j(1 / odds_j)`.

The 72 future matches use `base_predictions.p_a_win`, `p_draw`, and `p_b_win` directly. Only transparent probability fields are used. Future-only composite fields such as `attractiveness_index`, `commercial_value_index`, and `expected_attendance_base` are excluded because no identical historical construction is available.

## Feature Manifest And Rationale

All A/B team aggregates are symmetric because A/B does not denote home/away.

### Direct contextual features

- `competition` and `stage`: broadcast audience differs systematically by tournament and elimination importance.
- `neutral`: venue neutrality can alter supporter access and match salience.
- calendar year and month: capture broad temporal and seasonal demand without identifying individual rows.

### Team-strength features

- `elo_mean`: the overall quality of both teams is expected to raise audience interest.
- `elo_abs_diff`: the strength gap describes competitive balance.
- `rank_mean`: a second, interpretable summary of the overall strength tier. Rank direction is retained explicitly.

### Match-uncertainty features

- normalized `p_a_win`, `p_draw`, `p_b_win`, sorted to make the representation exchange-invariant.
- probability entropy: measures uncertainty across all three outcomes.
- maximum probability: measures favorite dominance.

### Audience-pull features

- `fan_sum` and `fan_max`: total reachable supporter base and the largest single draw.
- `star_sum`: recognizable-player appeal.
- `log_market_sum`: diminishing marginal effect of combined market value.

### Structural features

- `same_confederation`: regional familiarity and rivalry.
- `same_timezone_region`: broadcast convenience for overlapping audiences.
- `host_involved`: host participation generally raises local and international attention.

### Limited interactions

- `fan_sum_world_cup`: World Cup exposure amplifies the supporter-base effect.
- `elo_mean_entropy`: high-quality, uncertain matches combine quality and suspense.

No target-derived rolling feature is in the first Q1 run. It may be proposed at the review gate only if diagnostics show a clear missing temporal signal, and then it must use strict `source_date < current_date` logic with same-date grouping.

## Validation Protocol

Rows are sorted by date and same-date matches remain in one block. The fixed development folds are:

- Fold 1: train through 2020-04-29; validate 2020-05-07 through 2021-09-28.
- Fold 2: train through 2021-09-28; validate 2021-10-03 through 2023-02-23.
- Lockbox: train through 2023-02-23; validate 2023-03-02 through 2024-07-27.

Folds 1 and 2 support broad screening and tuning. The lockbox is evaluated once after candidates and ensemble rules are frozen. Preprocessing, imputation, category encoding, scaling, and parameter selection are fitted within each training fold only. The hidden 140-row test set never participates in model choice.

Primary selection metric: original-scale MSE in millions. Supporting metrics: RMSE, MAE, fold standard deviation, worst-fold MSE, and subgroup residuals. A model must improve the primary metric without an unacceptable deterioration on the newest fold.

## Candidate Models

The common protocol compares:

- mean and grouped shrinkage baselines;
- Ridge and Elastic Net;
- RBF SVR;
- Extra Trees and Random Forest;
- HistGradientBoosting;
- CatBoost, LightGBM, and XGBoost installed into a local Python 3.12 environment;
- a shallow MLP as a recorded challenger, not a favored default.

TabPFN or AutoGluon may be evaluated only after the core pool is stable and only if installation/runtime remains reproducible. They do not displace the common split, feature, or metric contract.

Broad screening uses small, regularized parameter sets. Only stable leading families receive constrained tuning. Randomized families use multiple seeds. A non-negative two-model blend is allowed only if OOF residual complementarity is present and the blend improves development MSE by at least 1 percent, improves at least two evaluated periods, and does not underperform the best constituent on the lockbox.

## Diagnostics And Explanation

The accepted candidate must provide:

- fold and aggregate MSE, RMSE, and MAE;
- baseline and candidate comparison with mean and dispersion;
- actual-versus-predicted, residual, temporal error, and feature-importance plots;
- residual slices for tournament/stage and high-audience matches;
- permutation importance or SHAP, chosen according to the final model;
- A/B swap-invariance check;
- five-seed stability for stochastic finalists;
- target-shuffle negative control;
- prediction-range and distribution-shift review;
- a documented limitation statement.

## Artifacts And Contracts

The Q1 run produces versioned metrics, OOF predictions, lockbox predictions, feature manifests, model artifacts, plots, environment metadata, and hashes. One registry selects the frozen run.

Required output files:

- `result_1_test_prediction.csv`: exactly 140 rows; columns `match_id_test,predicted_test_tv_viewers`; finite nonnegative values in millions.
- `result_1_match_prediction.csv`: exactly 72 rows; columns `match_id,team_a,team_b,predicted_tv_viewers`; finite nonnegative values in persons.

Both are generated from the official templates, joined by identifiers, and independently reloaded for validation.

## Q1 Stop Gate

Q1 is ready for user review only when the full local reproduction command succeeds and the result package contains the candidate leaderboard, lockbox evidence, diagnostic plots, feature rationales, CSV validation report, and known limitations. At that point work stops. The next permitted action is either another Q1 experiment requested after reviewing evidence or explicit acceptance of Q1. Q2 must not begin automatically.

