"""Feasibility-first Q2 group schedule Demo and independent hard-constraint checks.

The schedule is a deterministic, auditable incumbent.  It is also the data
interface used by the future COPT model; it never claims to be a COPT optimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import Q1Tables


GROUP_START_DATES = {
    "A": "2026-06-11", "B": "2026-06-11", "C": "2026-06-12", "D": "2026-06-12",
    "E": "2026-06-13", "F": "2026-06-13", "G": "2026-06-14", "H": "2026-06-14",
    "I": "2026-06-15", "J": "2026-06-15", "K": "2026-06-16", "L": "2026-06-16",
}


@dataclass(frozen=True)
class Q2Validation:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _minmax(value: float, lower: float, upper: float) -> float:
    return 0.0 if np.isclose(lower, upper) else float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _load_q2_tables(tables: Q1Tables) -> dict[str, pd.DataFrame]:
    names = ["venues", "time_slots", "ticket_broadcast", "security_requirements", "dynamic_resource_limits", "objective_weights"]
    return {name: pd.read_excel(tables.paths.workbook, sheet_name=name) for name in names}


def _q1_predictions(tables: Q1Tables, q1_match_csv: Path) -> pd.DataFrame:
    prediction = pd.read_csv(q1_match_csv)
    expected = tables.group_matches[["match_id", "team_a", "team_b"]]
    if not prediction[["match_id", "team_a", "team_b"]].equals(expected):
        raise ValueError("Q1 match prediction keys do not match groups_matches")
    return prediction


def build_q2_demo_schedule(tables: Q1Tables, q1_match_csv: Path) -> pd.DataFrame:
    data = _load_q2_tables(tables)
    venues, slots = data["venues"], data["time_slots"].copy()
    slots["date"] = pd.to_datetime(slots["date"])
    slots["utc_datetime"] = pd.to_datetime(slots["reference_utc_time"], utc=True)
    matches = tables.group_matches.merge(_q1_predictions(tables, q1_match_csv)[["match_id", "predicted_tv_viewers"]], on="match_id")
    matches = matches.merge(pd.read_excel(tables.paths.workbook, sheet_name="base_predictions"), on=["match_id", "group_id", "round_in_group", "team_a", "team_b"])
    matches = matches.merge(data["security_requirements"][["match_id", "required_security_level"]], on="match_id")
    rows: list[dict[str, object]] = []
    venue_cycle = venues.sort_values("venue_id").reset_index(drop=True)
    venue_total = {str(venue_id): 0 for venue_id in venue_cycle["venue_id"]}
    venue_day_use: set[tuple[str, pd.Timestamp]] = set()
    for ordinal, match in matches.sort_values(["round_in_group", "group_id", "match_id"]).reset_index(drop=True).iterrows():
        base = pd.Timestamp(GROUP_START_DATES[match.group_id])
        date = base + pd.Timedelta(days=7 * (int(match.round_in_group) - 1))
        day_slots = slots.loc[slots["date"].eq(date)].sort_values("reference_utc_time").reset_index(drop=True)
        group_matches_on_day = matches.loc[(matches["group_id"].map(GROUP_START_DATES).map(pd.Timestamp).eq(base)) & (matches["round_in_group"].eq(match.round_in_group))].sort_values("match_id")
        slot_index = group_matches_on_day.index.get_loc(matches.index[matches["match_id"].eq(match.match_id)][0])
        non_prime = day_slots.loc[~day_slots["slot_id"].isin(set(slots.nlargest(len(slots) // 4, "global_prime_score")["slot_id"]))]
        # Two non-golden UTC slots carry two matches each; their broadcast capacity is at least two.
        slot = non_prime.iloc[slot_index // 2]
        eligible = venue_cycle.loc[venue_cycle["security_level"].ge(int(match.required_security_level))].copy()
        eligible["_assigned"] = eligible["venue_id"].map(venue_total)
        eligible = eligible.loc[~eligible["venue_id"].map(lambda venue_id: (str(venue_id), date) in venue_day_use)]
        if eligible.empty:
            raise ValueError(f"No eligible unused venue for {match.match_id} on {date.date()}")
        venue = eligible.sort_values(["_assigned", "venue_id"]).iloc[0]
        venue_total[str(venue.venue_id)] += 1
        venue_day_use.add((str(venue.venue_id), date))
        local = slot.utc_datetime.tz_convert(ZoneInfo(venue.timezone))
        stage_key = f"Group_Match_R{int(match.round_in_group)}"
        economics = data["ticket_broadcast"].loc[data["ticket_broadcast"]["match_stage"].eq(stage_key)].iloc[0]
        expected_attendance = min(float(match.expected_attendance_base), float(venue.capacity))
        ticket = expected_attendance * float(economics.base_ticket_price_usd)
        broadcast = float(match.predicted_tv_viewers) * float(economics.broadcast_unit_value_usd) * float(slot.global_prime_score) * float(economics.sponsor_weight)
        risk = 0.5 * float(venue.climate_risk) + 0.3 * (expected_attendance / float(venue.capacity)) + 0.2 * (float(match.required_security_level) / float(venue.security_level))
        rows.append({
            "match_id": match.match_id, "group_id": match.group_id, "round_in_group": int(match.round_in_group),
            "team_a": match.team_a, "team_b": match.team_b, "venue_id": venue.venue_id, "slot_id": slot.slot_id,
            "city": venue.city, "country": venue.country, "reference_date": slot.date.date().isoformat(),
            "reference_kickoff_time": slot.reference_kickoff_time, "local_datetime": local.isoformat(),
            "utc_datetime": slot.utc_datetime.isoformat(), "required_security_level": int(match.required_security_level),
            "expected_attendance": expected_attendance, "expected_tv_viewers": float(match.predicted_tv_viewers),
            "ticket_revenue_usd": ticket, "broadcast_value_usd": broadcast,
            "travel_cost_index": np.nan, "fairness_penalty": np.nan, "risk_index": risk,
        })
    schedule = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)
    # Equal resource-use summary is a schedule-level metric repeated for each row.
    prime_ids = set(slots.nlargest(max(1, len(slots) // 4), "global_prime_score")["slot_id"])
    large_ids = set(venues.nlargest(max(1, len(venues) // 4), "capacity")["venue_id"])
    team_counts: dict[str, list[int]] = {}
    for team in sorted(set(schedule.team_a).union(schedule.team_b)):
        uses = schedule.loc[schedule.team_a.eq(team) | schedule.team_b.eq(team)]
        team_counts[team] = [int(uses.slot_id.isin(prime_ids).sum()), int(uses.venue_id.isin(large_ids).sum())]
    golden_range = max(value[0] for value in team_counts.values()) - min(value[0] for value in team_counts.values())
    large_range = max(value[1] for value in team_counts.values()) - min(value[1] for value in team_counts.values())
    fairness = 0.5 * golden_range / 3 + 0.5 * large_range / 3
    schedule["fairness_penalty"] = fairness
    schedule["travel_cost_index"] = 0.0  # Full candidate-set travel normalization is implemented in the COPT/score upgrade, not fabricated here.
    raw = schedule.ticket_revenue_usd.sum() + schedule.broadcast_value_usd.sum() - schedule.risk_index.sum() * 1_000_000
    schedule["total_objective_value"] = np.nan
    schedule.loc[0, "total_objective_value"] = raw
    return schedule


def validate_q2_schedule(tables: Q1Tables, schedule: pd.DataFrame) -> Q2Validation:
    data = _load_q2_tables(tables)
    errors: list[str] = []
    required = ["match_id", "venue_id", "slot_id", "utc_datetime", "required_security_level"]
    if any(column not in schedule for column in required):
        return Q2Validation(("schedule missing required columns",))
    if len(schedule) != 72 or set(schedule.match_id) != set(tables.group_matches.match_id) or not schedule.match_id.is_unique:
        errors.append("exactly one assignment for each of 72 matches is required")
    if schedule.duplicated(["venue_id", "utc_datetime"]).any():
        errors.append("venue/UTC collision")
    utc = pd.to_datetime(schedule.utc_datetime, utc=True)
    for team in sorted(set(schedule.team_a).union(schedule.team_b)):
        games = schedule.loc[(schedule.team_a.eq(team) | schedule.team_b.eq(team))].assign(_utc=utc.loc[(schedule.team_a.eq(team) | schedule.team_b.eq(team))]).sort_values("round_in_group")
        if len(games) != 3 or not (games.round_in_group.to_list() == [1, 2, 3]):
            errors.append(f"{team} round participation invalid")
        elif (games._utc.diff().dropna() < pd.Timedelta(hours=60)).any():
            errors.append(f"{team} has less than 60 hours rest")
    venues = data["venues"].set_index("venue_id")
    for _, row in schedule.iterrows():
        if int(venues.loc[row.venue_id, "security_level"]) < int(row.required_security_level):
            errors.append(f"security-ineligible venue for {row.match_id}")
    totals = schedule.venue_id.value_counts()
    for venue_id, venue in venues.iterrows():
        total = int(totals.get(venue_id, 0))
        if not int(venue.min_total_matches) <= total <= int(venue.max_total_matches):
            errors.append(f"venue total bound violated: {venue_id}")
    local_dates = []
    for _, row in schedule.iterrows():
        local_dates.append(pd.Timestamp(row.local_datetime).date())
    local_use = schedule.assign(_local_date=local_dates).groupby(["venue_id", "_local_date"]).size()
    for (venue_id, _), count in local_use.items():
        if count > int(venues.loc[venue_id, "max_matches_per_day"]):
            errors.append(f"venue local-day bound violated: {venue_id}")
    slots = data["time_slots"].set_index("slot_id")
    simultaneous = schedule.slot_id.value_counts()
    for slot_id, count in simultaneous.items():
        if count > int(slots.loc[slot_id, "broadcast_capacity"]):
            errors.append(f"broadcast capacity violated: {slot_id}")
    prime = set(slots.nlargest(len(slots) // 4, "global_prime_score").index)
    per_team_prime = []
    for team in sorted(set(schedule.team_a).union(schedule.team_b)):
        per_team_prime.append(int(schedule.loc[schedule.team_a.eq(team) | schedule.team_b.eq(team), "slot_id"].isin(prime).sum()))
    if max(per_team_prime) - min(per_team_prime) > 2:
        errors.append("golden-slot fairness range exceeds 2")
    limits = data["dynamic_resource_limits"].copy()
    limits.reference_date = pd.to_datetime(limits.reference_date).dt.date.astype(str)
    r3 = schedule.loc[schedule.round_in_group.eq(3)].copy()
    for date, block in r3.groupby("reference_date"):
        cap = int(limits.loc[limits.reference_date.eq(date), "high_security_capacity"].iloc[0])
        if int(block.required_security_level.ge(3).sum()) > cap:
            errors.append(f"third-round high-security capacity violated on {date}")
    return Q2Validation(tuple(dict.fromkeys(errors)))


def evaluate_q2_schedule(tables: Q1Tables, schedule: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score a feasible schedule on the official P2 indicators.

    Candidate ranges are fixed before reading the selected venue/slot.  In
    particular, the travel basis follows the stated prior-round eligible-venue
    averaging rule, preventing an optimistic range from being chosen after the
    fact.  Setup cost is necessarily schedule-level because venue activation is
    a binary decision.
    """

    data = _load_q2_tables(tables)
    venues = data["venues"].copy()
    slots = data["time_slots"].copy()
    base = tables.base_predictions.copy()
    q1_viewers = schedule[["match_id", "expected_tv_viewers"]].rename(columns={"expected_tv_viewers": "predicted_tv_viewers"})
    matches = tables.group_matches.merge(base, on=["match_id", "group_id", "round_in_group", "team_a", "team_b"]).merge(q1_viewers, on="match_id")
    matches = matches.merge(data["security_requirements"][["match_id", "required_security_level"]], on="match_id")
    team_id = tables.teams.set_index("team_name")["team_id"].to_dict()
    distance = pd.read_excel(tables.paths.workbook, sheet_name="distance_matrix")
    eligible = {row.match_id: venues.loc[venues.security_level.ge(row.required_security_level), "venue_id"].astype(str).tolist() for row in matches.itertuples(index=False)}

    # Construct exactly the candidate paths prescribed in the statement.
    path_rows: list[dict[str, object]] = []
    match_by_team_round: dict[tuple[str, int], object] = {}
    for row in matches.itertuples(index=False):
        match_by_team_round[(row.team_a, int(row.round_in_group))] = row
        match_by_team_round[(row.team_b, int(row.round_in_group))] = row
    for row in matches.itertuples(index=False):
        round_number = int(row.round_in_group)
        for team in (row.team_a, row.team_b):
            for venue_id in eligible[row.match_id]:
                if round_number == 1:
                    origin_id = team_id[team]
                    candidate = distance.loc[(distance.relation_type.eq("team_to_venue")) & distance.origin_id.eq(origin_id) & distance.destination_id.astype(str).eq(venue_id)]
                else:
                    previous = match_by_team_round[(team, round_number - 1)]
                    candidate = distance.loc[(distance.relation_type.eq("venue_to_venue")) & distance.origin_id.astype(str).isin(eligible[previous.match_id]) & distance.destination_id.astype(str).eq(venue_id)]
                if candidate.empty:
                    raise ValueError(f"missing travel candidate for {row.match_id}, {team}, {venue_id}")
                path_rows.append({"match_id": row.match_id, "team": team, "venue_id": venue_id, "distance_km": candidate.distance_km.mean(), "travel_time_hour": candidate.travel_time_hour.mean(), "timezone_diff": candidate.timezone_diff.mean()})
    paths = pd.DataFrame(path_rows)
    for column in ["distance_km", "travel_time_hour", "timezone_diff"]:
        paths[f"{column}_norm"] = (paths[column] - paths[column].min()) / (paths[column].max() - paths[column].min())
    paths["travel_burden"] = 0.5 * paths.distance_km_norm + 0.3 * paths.travel_time_hour_norm + 0.2 * paths.timezone_diff_norm
    candidate_travel = paths.groupby(["match_id", "venue_id"], as_index=False).travel_burden.mean()

    economics = data["ticket_broadcast"].set_index("match_stage")
    ticket_candidates: dict[str, list[float]] = {}
    broadcast_candidates: dict[str, list[float]] = {}
    candidate_c: list[float] = []
    for row in matches.itertuples(index=False):
        price = economics.loc[f"Group_Match_R{int(row.round_in_group)}"]
        ticket_candidates[row.match_id] = []
        broadcast_candidates[row.match_id] = []
        for venue in venues.itertuples(index=False):
            if venue.venue_id in eligible[row.match_id]:
                ticket_candidates[row.match_id].append(min(float(row.expected_attendance_base), float(venue.capacity)) * float(price.base_ticket_price_usd))
                candidate_c.append(float(venue.operation_cost_musd_per_match) + 0.1 * int(row.required_security_level) * float(venue.security_cost_index))
        for slot in slots.itertuples(index=False):
            broadcast_candidates[row.match_id].append(float(row.predicted_tv_viewers) * float(price.broadcast_unit_value_usd) * float(slot.global_prime_score) * float(price.sponsor_weight))
    selected = schedule.copy()
    selected["venue_id"] = selected.venue_id.astype(str)
    selected = selected.merge(candidate_travel, on=["match_id", "venue_id"], how="left")
    if selected.travel_burden.isna().any():
        raise ValueError("selected schedule has a venue without a travel candidate")
    selected = selected.merge(matches[["match_id", "uncertainty_index", "attractiveness_index", "expected_attendance_base"]], on="match_id", how="left")
    selected = selected.merge(venues[["venue_id", "capacity", "setup_cost_musd", "operation_cost_musd_per_match", "security_cost_index", "climate_risk", "security_level"]], on="venue_id", how="left", suffixes=("", "_venue"))
    selected["operation_cost_musd"] = selected.operation_cost_musd_per_match + 0.1 * selected.required_security_level * selected.security_cost_index
    selected["risk_raw"] = 0.5 * selected.climate_risk + 0.3 * (selected.expected_attendance / selected.capacity) + 0.2 * selected.required_security_level / selected.security_level
    selected["travel_cost_index"] = selected.travel_burden
    prime = set(slots.nlargest(len(slots) // 4, "global_prime_score").slot_id.astype(str))
    large = set(venues.nlargest(len(venues) // 4, "capacity").venue_id.astype(str))
    resource_counts = []
    for team in sorted(set(selected.team_a).union(selected.team_b)):
        games = selected.loc[selected.team_a.eq(team) | selected.team_b.eq(team)]
        resource_counts.append((int(games.slot_id.astype(str).isin(prime).sum()), int(games.venue_id.isin(large).sum())))
    fairness = 0.5 * (max(item[0] for item in resource_counts) - min(item[0] for item in resource_counts)) / 3 + 0.5 * (max(item[1] for item in resource_counts) - min(item[1] for item in resource_counts)) / 3
    raw = {
        "T": float(selected.ticket_revenue_usd.sum()),
        "B": float(selected.broadcast_value_usd.sum()),
        "U": float(selected.uncertainty_index.sum()),
        "H": float((selected.attractiveness_index / 100).sum()),
        "C": float(selected.operation_cost_musd.sum() + selected.drop_duplicates("venue_id").setup_cost_musd.sum()),
        "D": float(selected.travel_burden.sum()),
        "F": float(fairness),
        "R": float(selected.risk_raw.mean()),
    }
    ranges = {
        "T": (sum(min(values) for values in ticket_candidates.values()), sum(max(values) for values in ticket_candidates.values())),
        "B": (sum(min(values) for values in broadcast_candidates.values()), sum(max(values) for values in broadcast_candidates.values())),
        "C": (sum(min(candidate_c) for _ in range(len(matches))), sum(max(candidate_c) for _ in range(len(matches))) + float(venues.setup_cost_musd.sum())),
        "D": (sum(candidate_travel.groupby("match_id").travel_burden.min()), sum(candidate_travel.groupby("match_id").travel_burden.max())),
    }
    normalized = {"T": _minmax(raw["T"], *ranges["T"]), "B": _minmax(raw["B"], *ranges["B"]), "U": raw["U"] / len(selected), "H": raw["H"] / len(selected), "C": _minmax(raw["C"], *ranges["C"]), "D": _minmax(raw["D"], *ranges["D"]), "F": raw["F"], "R": raw["R"]}
    objective = 0.25 * normalized["T"] + 0.25 * normalized["B"] + 0.15 * normalized["U"] + 0.10 * normalized["H"] - 0.08 * normalized["C"] - 0.07 * normalized["D"] - 0.06 * normalized["F"] - 0.04 * normalized["R"]
    selected["fairness_penalty"] = fairness
    selected["total_objective_value"] = np.nan
    selected.loc[selected.index[0], "total_objective_value"] = objective
    report = {**{f"raw_{key}": value for key, value in raw.items()}, **{f"norm_{key}": value for key, value in normalized.items()}, "objective": objective}
    return selected, report
