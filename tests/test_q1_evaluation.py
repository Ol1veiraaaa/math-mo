from __future__ import annotations

import numpy as np

from c_contest_q1.data import load_q1_tables
from c_contest_q1.evaluation import (
    DEFAULT_TEMPORAL_FOLDS,
    evaluate_model_specs,
    make_temporal_folds,
)
from c_contest_q1.features import build_historical_features, target_in_millions
from c_contest_q1.models import demo_model_specs, sklearn_challenger_specs


def test_temporal_folds_preserve_complete_dates_and_order() -> None:
    frame = build_historical_features(load_q1_tables())
    folds = make_temporal_folds(frame.metadata)

    assert len(folds) == len(DEFAULT_TEMPORAL_FOLDS)
    for fold in folds:
        train_dates = frame.metadata.iloc[fold.train_indices]["date"]
        valid_dates = frame.metadata.iloc[fold.valid_indices]["date"]
        assert train_dates.max() < valid_dates.min()
        assert set(train_dates).isdisjoint(set(valid_dates))
        assert len(fold.train_indices) > 100
        assert len(fold.valid_indices) > 80


def test_demo_evaluation_returns_finite_metrics_and_oof_predictions() -> None:
    frame = build_historical_features(load_q1_tables())
    result = evaluate_model_specs(
        frame.features,
        frame.metadata,
        target_in_millions(frame),
        demo_model_specs(),
    )

    assert {"mean", "ridge", "hist_gradient_boosting"}.issubset(
        set(result.leaderboard["model"])
    )
    assert np.isfinite(result.leaderboard[["mse", "rmse", "mae"]].to_numpy()).all()
    assert result.oof_predictions["prediction"].notna().all()
    assert result.oof_predictions["actual"].notna().all()
    assert result.oof_predictions["fold"].nunique() == len(DEFAULT_TEMPORAL_FOLDS)


def test_sklearn_challengers_share_the_common_temporal_contract() -> None:
    frame = build_historical_features(load_q1_tables())
    result = evaluate_model_specs(
        frame.features,
        frame.metadata,
        target_in_millions(frame),
        sklearn_challenger_specs(),
    )

    assert set(result.leaderboard["model"]) == {
        "elastic_net",
        "svr_rbf",
        "extra_trees",
        "random_forest",
    }
    assert np.isfinite(result.leaderboard[["mse", "rmse", "mae"]].to_numpy()).all()
