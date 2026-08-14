"""Q4 sourced actual-schedule and scale-neutral structural comparison."""

from __future__ import annotations

import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import Q1Tables


ACTUAL_REQUIRED = [
    "match_id", "competition", "stage", "group_id", "round_in_group", "team_a", "team_b",
    "venue", "city", "country", "date", "local_kickoff_time", "utc_datetime", "goals_a",
    "goals_b", "source_url", "retrieval_date",
]


ACTUAL_STADIUMS_COLUMNS = ["city", "venue", "country", "utc_offset", "capacity", "latitude", "longitude"]


def validate_actual_schedule(actual: pd.DataFrame) -> tuple[str, ...]:
    errors: list[str] = []
    if list(actual.columns) != ACTUAL_REQUIRED:
        return ("actual schedule columns do not follow the official template",)
    if actual.empty:
        errors.append("actual schedule has no rows")
    if not actual.match_id.is_unique:
        errors.append("actual schedule match_id must be unique")
    if actual[["source_url", "retrieval_date"]].isna().any().any() or actual[["source_url", "retrieval_date"]].astype(str).apply(lambda column: column.str.strip().eq("")).any().any():
        errors.append("every actual schedule row requires source_url and retrieval_date")
    if pd.to_datetime(actual.utc_datetime, utc=True, errors="coerce").isna().any():
        errors.append("actual schedule has unparsable UTC datetimes")
    return tuple(errors)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _team_transitions(frame: pd.DataFrame, venue_column: str, coordinates: dict[str, tuple[float, float]], timezone_offsets: dict[str, float]) -> pd.DataFrame:
    utc = pd.to_datetime(frame.utc_datetime, utc=True)
    rows: list[dict[str, float | str]] = []
    for team in sorted(set(frame.team_a).union(frame.team_b)):
        games = frame.loc[frame.team_a.eq(team) | frame.team_b.eq(team)].copy()
        games["utc_value"] = utc.loc[games.index]
        games = games.sort_values("utc_value")
        previous = None
        for game in games.itertuples(index=False):
            if previous is not None:
                previous_venue, previous_time = getattr(previous, venue_column), previous.utc_value
                venue = getattr(game, venue_column)
                lat1, lon1 = coordinates[previous_venue]
                lat2, lon2 = coordinates[venue]
                rows.append({
                    "team": team,
                    "from_venue": previous_venue,
                    "to_venue": venue,
                    "distance_km": _haversine_km(lat1, lon1, lat2, lon2),
                    "rest_hours": (game.utc_value - previous_time).total_seconds() / 3600,
                    "timezone_crossing": float(not np.isclose(timezone_offsets[previous_venue], timezone_offsets[venue])),
                })
            previous = game
    return pd.DataFrame(rows)


def _venue_metrics(frame: pd.DataFrame, venue_column: str) -> dict[str, float]:
    shares = frame[venue_column].value_counts(normalize=True)
    counts = frame[venue_column].value_counts()
    return {
        "unique_venues": float(len(counts)),
        "matches_per_venue_mean": float(counts.mean()),
        "matches_per_venue_cv": float(counts.std(ddof=0) / counts.mean()),
        "venue_match_hhi": float((shares ** 2).sum()),
    }


def _metrics_actual(actual: pd.DataFrame, stadiums: pd.DataFrame) -> dict[str, float]:
    coordinates = {row.venue: (row.latitude, row.longitude) for row in stadiums.itertuples(index=False)}
    offsets = {row.venue: row.utc_offset for row in stadiums.itertuples(index=False)}
    capacities = {row.venue: row.capacity for row in stadiums.itertuples(index=False)}
    transitions = _team_transitions(actual, "venue", coordinates, offsets)
    metrics = _venue_metrics(actual, "venue")
    metrics.update({
        "matches": float(len(actual)),
        "teams": float(pd.concat([actual.team_a, actual.team_b]).nunique()),
        "travel_km_per_team": float(transitions.distance_km.sum() / pd.concat([actual.team_a, actual.team_b]).nunique()),
        "mean_transition_km": float(transitions.distance_km.mean()),
        "mean_rest_hours": float(transitions.rest_hours.mean()),
        "minimum_rest_hours": float(transitions.rest_hours.min()),
        "rest_under_60h_rate": float(transitions.rest_hours.lt(60).mean()),
        "timezone_crossings_per_team": float(transitions.timezone_crossing.sum() / pd.concat([actual.team_a, actual.team_b]).nunique()),
        "mean_stadium_capacity": float(actual.venue.map(capacities).mean()),
    })
    return metrics


def _metrics_optimized(tables: Q1Tables, optimized: pd.DataFrame) -> dict[str, float]:
    venues = pd.read_excel(tables.paths.workbook, sheet_name="venues").set_index("venue_id")
    coordinates = {venue: (float(row.latitude), float(row.longitude)) for venue, row in venues.iterrows()}
    offsets: dict[str, float] = {}
    reference_time = pd.Timestamp("2026-06-20T12:00:00Z")
    for venue, row in venues.iterrows():
        offsets[venue] = reference_time.tz_convert(ZoneInfo(row.timezone)).utcoffset().total_seconds() / 3600
    transitions = _team_transitions(optimized, "venue_id", coordinates, offsets)
    metrics = _venue_metrics(optimized, "venue_id")
    team_count = pd.concat([optimized.team_a, optimized.team_b]).nunique()
    metrics.update({
        "matches": float(len(optimized)),
        "teams": float(team_count),
        "travel_km_per_team": float(transitions.distance_km.sum() / team_count),
        "mean_transition_km": float(transitions.distance_km.mean()),
        "mean_rest_hours": float(transitions.rest_hours.mean()),
        "minimum_rest_hours": float(transitions.rest_hours.min()),
        "rest_under_60h_rate": float(transitions.rest_hours.lt(60).mean()),
        "timezone_crossings_per_team": float(transitions.timezone_crossing.sum() / team_count),
        "mean_stadium_capacity": float(optimized.venue_id.map(venues.capacity).mean()),
    })
    return metrics


def compare_structural_schedule(tables: Q1Tables, actual_csv: Path, optimized_csv: Path) -> pd.DataFrame:
    actual = pd.read_csv(actual_csv)
    errors = validate_actual_schedule(actual)
    if errors:
        raise ValueError("; ".join(errors))
    stadiums_path = Path(actual_csv).parent / "actual_stadiums.csv"
    if not stadiums_path.exists():
        raise FileNotFoundError(f"actual stadiums table missing: {stadiums_path}")
    stadiums = pd.read_csv(stadiums_path)
    if list(stadiums.columns) != ACTUAL_STADIUMS_COLUMNS:
        raise ValueError("actual stadiums columns do not follow the expected template")
    optimized = pd.read_csv(optimized_csv)
    actual_metrics, optimized_metrics = _metrics_actual(actual, stadiums), _metrics_optimized(tables, optimized)
    definitions = [
        ("matches", "scale", "target"),
        ("teams", "scale", "target"),
        ("travel_km_per_team", "travel", "lower"),
        ("mean_transition_km", "travel", "lower"),
        ("mean_rest_hours", "rest", "higher"),
        ("minimum_rest_hours", "rest", "higher"),
        ("rest_under_60h_rate", "rest", "lower"),
        ("timezone_crossings_per_team", "travel", "lower"),
        ("unique_venues", "venue_utilization", "target"),
        ("matches_per_venue_mean", "venue_utilization", "target"),
        ("matches_per_venue_cv", "venue_utilization", "lower"),
        ("venue_match_hhi", "venue_utilization", "lower"),
        ("mean_stadium_capacity", "capacity", "higher"),
    ]
    output: list[dict[str, object]] = []
    for name, category, direction in definitions:
        observed, proposed = actual_metrics[name], optimized_metrics[name]
        difference = proposed - observed
        if direction == "higher":
            relative = difference / abs(observed) if observed else np.nan
            result = "improved" if difference > 0 else "not_improved"
        elif direction == "lower":
            relative = -difference / abs(observed) if observed else np.nan
            result = "improved" if difference < 0 else "not_improved"
        else:
            relative = np.nan
            result = "context_only"
        output.append({
            "indicator_name": name,
            "indicator_category": category,
            "actual_schedule_value": observed,
            "optimized_schedule_value": proposed,
            "absolute_difference": difference,
            "relative_improvement": relative,
            "preferred_direction": direction,
            "evaluation_result": result,
        })
    return pd.DataFrame(output)
