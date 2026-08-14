from __future__ import annotations

import numpy as np

from c_contest_q1.data import load_q1_tables
from c_contest_q1.features import (
    FORBIDDEN_SOURCE_COLUMNS,
    build_future_features,
    build_historical_features,
    target_in_millions,
)


def test_historical_and_future_share_exact_feature_contract() -> None:
    tables = load_q1_tables()
    historical = build_historical_features(tables)
    future = build_future_features(tables)

    assert list(historical.features.columns) == list(future.features.columns)
    assert len(historical.features) == 700
    assert len(future.features) == 72
    assert historical.features.notna().all().all()
    assert future.features.notna().all().all()


def test_probability_bridge_is_normalized_and_symmetric() -> None:
    tables = load_q1_tables()
    historical = build_historical_features(tables)
    future = build_future_features(tables)

    probability_columns = ["prob_low", "prob_mid", "prob_high"]
    assert np.allclose(historical.features[probability_columns].sum(axis=1), 1.0)
    assert np.allclose(future.features[probability_columns].sum(axis=1), 1.0)
    assert np.allclose(
        future.features["prob_max"],
        future.features["prob_high"],
    )


def test_final_features_exclude_all_post_match_source_columns() -> None:
    tables = load_q1_tables()
    frame = build_historical_features(tables)

    assert not set(FORBIDDEN_SOURCE_COLUMNS).intersection(frame.features.columns)
    assert "match_id" not in frame.features.columns
    assert "team_a" not in frame.features.columns
    assert "team_b" not in frame.features.columns
    assert "tv_viewers" not in frame.features.columns


def test_target_is_explicitly_converted_from_people_to_millions() -> None:
    tables = load_q1_tables()
    historical = build_historical_features(tables)
    y = target_in_millions(historical)

    assert y.shape == (560,)
    assert 140 < y.mean() < 150
    assert 90 < y.min() < 100
    assert 220 < y.max() < 230

