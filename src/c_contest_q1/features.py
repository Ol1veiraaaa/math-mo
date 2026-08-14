"""Leakage-safe, symmetric pre-match feature construction for Q1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Q1Tables


FORBIDDEN_SOURCE_COLUMNS = frozenset(
    {
        "goals_a",
        "goals_b",
        "xg_a",
        "xg_b",
        "shots_a",
        "shots_b",
        "possession_a",
        "possession_b",
        "attendance",
        "tv_viewers",
        "match_id",
        "team_a",
        "team_b",
    }
)


@dataclass(frozen=True)
class FeatureFrame:
    features: pd.DataFrame
    metadata: pd.DataFrame
    target_people: pd.Series | None


def _sorted_pair(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.DataFrame({"left": left.astype("string"), "right": right.astype("string")}).apply(
        lambda row: "|".join(sorted((str(row["left"]), str(row["right"])))), axis=1
    )


def _team_attributes(tables: Q1Tables, matches: pd.DataFrame, *, future: bool) -> pd.DataFrame:
    team_columns = [
        "team_name",
        "confederation",
        "market_value_musd",
        "star_index",
        "fan_base_index",
        "home_timezone_region",
    ]
    teams = tables.teams.loc[:, team_columns]
    left = teams.rename(columns={column: f"{column}_a" for column in team_columns if column != "team_name"})
    left = left.rename(columns={"team_name": "team_a"})
    right = teams.rename(columns={column: f"{column}_b" for column in team_columns if column != "team_name"})
    right = right.rename(columns={"team_name": "team_b"})
    enriched = matches.merge(left, on="team_a", how="left", validate="many_to_one")
    enriched = enriched.merge(right, on="team_b", how="left", validate="many_to_one")
    if enriched.filter(regex=r"^(confederation|market_value|star_index|fan_base|home_timezone)").isna().any().any():
        source = "future" if future else "historical"
        raise ValueError(f"{source} match has an unmapped team attribute")
    return enriched


def _probability_features(probabilities: pd.DataFrame) -> pd.DataFrame:
    values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("pre-match probabilities must be positive and finite")
    normalized = values / values.sum(axis=1, keepdims=True)
    ordered = np.sort(normalized, axis=1)
    entropy = -(normalized * np.log(normalized)).sum(axis=1) / np.log(3)
    return pd.DataFrame(
        {
            "prob_low": ordered[:, 0],
            "prob_mid": ordered[:, 1],
            "prob_high": ordered[:, 2],
            "prob_max": ordered[:, 2],
            "prob_entropy": entropy,
        },
        index=probabilities.index,
    )


def _assemble_features(enriched: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
    features = _probability_features(probabilities)
    elo_a = enriched["elo_a"].to_numpy(dtype=float)
    elo_b = enriched["elo_b"].to_numpy(dtype=float)
    rank_a = enriched["strength_rank_a"].to_numpy(dtype=float)
    rank_b = enriched["strength_rank_b"].to_numpy(dtype=float)
    fan_a = enriched["fan_base_index_a"].to_numpy(dtype=float)
    fan_b = enriched["fan_base_index_b"].to_numpy(dtype=float)
    star_a = enriched["star_index_a"].to_numpy(dtype=float)
    star_b = enriched["star_index_b"].to_numpy(dtype=float)
    value_a = enriched["market_value_musd_a"].to_numpy(dtype=float)
    value_b = enriched["market_value_musd_b"].to_numpy(dtype=float)
    features["competition"] = enriched["competition"].astype("string")
    features["stage"] = enriched["stage"].astype("string")
    features["elo_mean"] = (elo_a + elo_b) / 2.0
    features["elo_abs_diff"] = np.abs(elo_a - elo_b)
    features["rank_mean"] = (rank_a + rank_b) / 2.0
    features["fan_sum"] = fan_a + fan_b
    features["fan_max"] = np.maximum(fan_a, fan_b)
    features["star_sum"] = star_a + star_b
    features["star_max"] = np.maximum(star_a, star_b)
    features["log_market_sum"] = np.log1p(value_a + value_b)
    features["same_confederation"] = (
        enriched["confederation_a"].eq(enriched["confederation_b"]).astype(int)
    )
    features["confederation_pair"] = _sorted_pair(
        enriched["confederation_a"], enriched["confederation_b"]
    ).astype("string")
    features["timezone_region_pair"] = _sorted_pair(
        enriched["home_timezone_region_a"], enriched["home_timezone_region_b"]
    ).astype("string")
    if features.isna().any().any():
        raise ValueError("feature construction produced missing values")
    return features


def build_historical_features(tables: Q1Tables) -> FeatureFrame:
    matches = _team_attributes(tables, tables.historical.copy(), future=False)
    probabilities = pd.DataFrame(
        {
            "a": 1.0 / matches["odds_a"].astype(float),
            "draw": 1.0 / matches["odds_draw"].astype(float),
            "b": 1.0 / matches["odds_b"].astype(float),
        },
        index=matches.index,
    )
    metadata = matches.loc[:, ["match_id", "date", "dataset_split", "team_a", "team_b"]].copy()
    return FeatureFrame(
        features=_assemble_features(matches, probabilities),
        metadata=metadata,
        target_people=matches["tv_viewers"].copy(),
    )


def build_future_features(tables: Q1Tables) -> FeatureFrame:
    matches = tables.group_matches.merge(
        tables.base_predictions.loc[:, ["match_id", "p_a_win", "p_draw", "p_b_win"]],
        on="match_id",
        how="inner",
        validate="one_to_one",
    )
    matches["competition"] = "World Cup"
    matches["stage"] = "Group"
    matches["elo_a"] = matches["team_a"].map(tables.teams.set_index("team_name")["elo_rating"])
    matches["elo_b"] = matches["team_b"].map(tables.teams.set_index("team_name")["elo_rating"])
    matches["strength_rank_a"] = matches["team_a"].map(
        tables.teams.set_index("team_name")["strength_rank"]
    )
    matches["strength_rank_b"] = matches["team_b"].map(
        tables.teams.set_index("team_name")["strength_rank"]
    )
    matches = _team_attributes(tables, matches, future=True)
    probabilities = matches.loc[:, ["p_a_win", "p_draw", "p_b_win"]].rename(
        columns={"p_a_win": "a", "p_draw": "draw", "p_b_win": "b"}
    )
    metadata = matches.loc[:, ["match_id", "team_a", "team_b", "group_id", "round_in_group"]].copy()
    return FeatureFrame(
        features=_assemble_features(matches, probabilities),
        metadata=metadata,
        target_people=None,
    )


def target_in_millions(frame: FeatureFrame) -> pd.Series:
    if frame.target_people is None:
        raise ValueError("future feature frames have no target")
    train_mask = frame.metadata["dataset_split"].eq("train")
    target = frame.target_people.loc[train_mask]
    if target.isna().any():
        raise ValueError("training target contains missing values")
    return target.astype(float) / 1_000_000

