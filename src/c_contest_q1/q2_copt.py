"""Restricted-candidate COPT MIP for the Q2 venue/time joint schedule.

The evaluation COPT license limits MIP models to 2,000 variables.  We therefore
retain each incumbent venue plus the two largest security-eligible alternatives
and every candidate time slot on that match's incumbent reference date.  The
restriction is explicit, reproducible, and preserves the official hard
constraints.  The full P2 evaluator still scores the selected schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import Q1Tables
from .q2_demo import _load_q2_tables, evaluate_q2_schedule


@dataclass(frozen=True)
class CoptQ2Artifacts:
    schedule: pd.DataFrame
    candidate_count: int
    status: int
    mip_objective: float
    score_report: dict[str, float]
    solve_evidence: dict[str, float | int]


def _candidate_travel(tables: Q1Tables, incumbent: pd.DataFrame, matches: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    """Construct official travel burdens, normalizing before candidate restriction."""

    distance = pd.read_excel(tables.paths.workbook, sheet_name="distance_matrix")
    team_id = tables.teams.set_index("team_name")["team_id"].to_dict()
    eligible = {row.match_id: venues.loc[venues.security_level.ge(row.required_security_level), "venue_id"].astype(str).tolist() for row in matches.itertuples(index=False)}
    by_team_round: dict[tuple[str, int], object] = {}
    for row in matches.itertuples(index=False):
        by_team_round[(row.team_a, int(row.round_in_group))] = row
        by_team_round[(row.team_b, int(row.round_in_group))] = row
    allowed = incumbent.groupby("match_id").candidate_venue_id.unique().to_dict()
    rows: list[dict[str, object]] = []
    for row in matches.itertuples(index=False):
        for team in (row.team_a, row.team_b):
            for venue_id in eligible[row.match_id]:
                if int(row.round_in_group) == 1:
                    candidate = distance.loc[(distance.relation_type.eq("team_to_venue")) & distance.origin_id.eq(team_id[team]) & distance.destination_id.astype(str).eq(venue_id)]
                else:
                    previous = by_team_round[(team, int(row.round_in_group) - 1)]
                    candidate = distance.loc[(distance.relation_type.eq("venue_to_venue")) & distance.origin_id.astype(str).isin(eligible[previous.match_id]) & distance.destination_id.astype(str).eq(venue_id)]
                if candidate.empty:
                    raise ValueError(f"missing travel record for {row.match_id}/{team}/{venue_id}")
                rows.append({"match_id": row.match_id, "candidate_venue_id": venue_id, "distance_km": candidate.distance_km.mean(), "travel_time_hour": candidate.travel_time_hour.mean(), "timezone_diff": candidate.timezone_diff.mean()})
    travel = pd.DataFrame(rows)
    for column in ("distance_km", "travel_time_hour", "timezone_diff"):
        spread = travel[column].max() - travel[column].min()
        travel[f"n_{column}"] = 0.0 if np.isclose(spread, 0) else (travel[column] - travel[column].min()) / spread
    travel["travel_burden"] = 0.5 * travel.n_distance_km + 0.3 * travel.n_travel_time_hour + 0.2 * travel.n_timezone_diff
    travel = travel.groupby(["match_id", "candidate_venue_id"], as_index=False).travel_burden.mean()
    keep = pd.DataFrame(
        [(match_id, venue_id) for match_id, venue_ids in allowed.items() for venue_id in venue_ids],
        columns=["match_id", "candidate_venue_id"],
    )
    return keep.merge(travel, on=["match_id", "candidate_venue_id"], validate="one_to_one")


def build_q2_copt_candidates(tables: Q1Tables, incumbent: pd.DataFrame) -> pd.DataFrame:
    data = _load_q2_tables(tables)
    venues, slots = data["venues"].copy(), data["time_slots"].copy()
    slots["date"] = pd.to_datetime(slots.date).dt.date.astype(str)
    slots["utc_datetime"] = pd.to_datetime(slots.reference_utc_time, utc=True)
    q1_viewers = incumbent[["match_id", "expected_tv_viewers"]].rename(columns={"expected_tv_viewers": "predicted_tv_viewers"})
    matches = tables.group_matches.merge(tables.base_predictions, on=["match_id", "group_id", "round_in_group", "team_a", "team_b"]).merge(q1_viewers, on="match_id").merge(data["security_requirements"][["match_id", "required_security_level"]], on="match_id")
    incumbent = incumbent[["match_id", "venue_id", "reference_date"]].copy()
    incumbent["venue_id"] = incumbent.venue_id.astype(str)
    rows: list[dict[str, object]] = []
    for match in matches.itertuples(index=False):
        incumbent_venue = incumbent.loc[incumbent.match_id.eq(match.match_id), "venue_id"].iloc[0]
        alternatives = venues.loc[venues.security_level.ge(match.required_security_level)].sort_values(["capacity", "venue_id"], ascending=[False, True]).venue_id.astype(str).tolist()
        candidate_venues = list(dict.fromkeys([incumbent_venue, *alternatives[:2]]))
        for venue_id in candidate_venues:
            rows.append({"match_id": match.match_id, "candidate_venue_id": venue_id})
    restricted = pd.DataFrame(rows).drop_duplicates()
    travel = _candidate_travel(tables, restricted, matches, venues)
    base = incumbent[["match_id", "reference_date"]].merge(matches, on="match_id").merge(restricted, on="match_id").merge(travel, on=["match_id", "candidate_venue_id"])
    base = base.merge(venues.add_prefix("venue_").rename(columns={"venue_venue_id": "candidate_venue_id"}), on="candidate_venue_id")
    economics = data["ticket_broadcast"].set_index("match_stage")
    for index, row in base.iterrows():
        economics_row = economics.loc[f"Group_Match_R{int(row.round_in_group)}"]
        base.loc[index, "ticket_revenue_usd"] = min(float(row.expected_attendance_base), float(row.venue_capacity)) * float(economics_row.base_ticket_price_usd)
        base.loc[index, "operation_cost_musd"] = float(row.venue_operation_cost_musd_per_match) + 0.1 * int(row.required_security_level) * float(row.venue_security_cost_index)
        base.loc[index, "risk_index"] = 0.5 * float(row.venue_climate_risk) + 0.3 * min(float(row.expected_attendance_base), float(row.venue_capacity)) / float(row.venue_capacity) + 0.2 * int(row.required_security_level) / int(row.venue_security_level)
    candidates = base.merge(slots[["slot_id", "date", "reference_kickoff_time", "utc_datetime", "global_prime_score", "broadcast_capacity"]], left_on="reference_date", right_on="date")
    candidates["slot_id"] = candidates.slot_id.astype(str)
    candidates["broadcast_value_usd"] = candidates.predicted_tv_viewers * candidates.apply(lambda row: float(economics.loc[f"Group_Match_R{int(row.round_in_group)}"].broadcast_unit_value_usd) * float(row.global_prime_score) * float(economics.loc[f"Group_Match_R{int(row.round_in_group)}"].sponsor_weight), axis=1)
    # Use the full official candidate universe for every Min-Max bound, even
    # though the evaluation-license MIP retains only three venues per match.
    eligible = {
        row.match_id: venues.loc[
            venues.security_level.ge(row.required_security_level), "venue_id"
        ].astype(str).tolist()
        for row in matches.itertuples(index=False)
    }
    ticket_bounds: dict[str, tuple[float, float]] = {}
    broadcast_bounds: dict[str, tuple[float, float]] = {}
    operation_candidates: list[float] = []
    for row in matches.itertuples(index=False):
        price = economics.loc[f"Group_Match_R{int(row.round_in_group)}"]
        ticket_values = [
            min(float(row.expected_attendance_base), float(venue.capacity))
            * float(price.base_ticket_price_usd)
            for venue in venues.loc[venues.venue_id.astype(str).isin(eligible[row.match_id])].itertuples(index=False)
        ]
        broadcast_values = [
            float(row.predicted_tv_viewers)
            * float(price.broadcast_unit_value_usd)
            * float(slot.global_prime_score)
            * float(price.sponsor_weight)
            for slot in slots.itertuples(index=False)
        ]
        operation_candidates.extend(
            float(venue.operation_cost_musd_per_match)
            + 0.1 * int(row.required_security_level) * float(venue.security_cost_index)
            for venue in venues.loc[venues.venue_id.astype(str).isin(eligible[row.match_id])].itertuples(index=False)
        )
        ticket_bounds[row.match_id] = (min(ticket_values), max(ticket_values))
        broadcast_bounds[row.match_id] = (min(broadcast_values), max(broadcast_values))

    travel_all = _candidate_travel(
        tables,
        pd.DataFrame(
            [(match_id, venue_id) for match_id, venue_ids in eligible.items() for venue_id in venue_ids],
            columns=["match_id", "candidate_venue_id"],
        ),
        matches,
        venues,
    )
    ranges = {
        "T": (
            sum(value[0] for value in ticket_bounds.values()),
            sum(value[1] for value in ticket_bounds.values()),
        ),
        "B": (
            sum(value[0] for value in broadcast_bounds.values()),
            sum(value[1] for value in broadcast_bounds.values()),
        ),
        "C": (
            len(matches) * min(operation_candidates),
            len(matches) * max(operation_candidates) + float(venues.setup_cost_musd.sum()),
        ),
        "D": (
            float(travel_all.groupby("match_id").travel_burden.min().sum()),
            float(travel_all.groupby("match_id").travel_burden.max().sum()),
        ),
    }
    candidates.attrs["z2_ranges"] = ranges
    candidates.attrs["z2_fixed"] = {
        "U": float(matches.uncertainty_index.sum() / len(matches)),
        "H": float((matches.attractiveness_index / 100).sum() / len(matches)),
    }
    return candidates


def _scaled_coefficient(weight: float, bounds: tuple[float, float]) -> tuple[float, float]:
    lower, upper = bounds
    if np.isclose(lower, upper):
        return 0.0, 0.0
    return weight / (upper - lower), -weight * lower / (upper - lower)


def _copt_log_name(path: Path) -> str:
    """Use an ASCII relative name because COPT 8.0.6 cannot open Chinese Windows paths."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def solve_copt_q2(
    tables: Q1Tables,
    incumbent_csv: Path,
    log_path: Path | None = None,
) -> CoptQ2Artifacts:
    try:
        import coptpy as cp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("coptpy is required for the COPT Q2 run") from exc
    incumbent = pd.read_csv(incumbent_csv)
    candidates = build_q2_copt_candidates(tables, incumbent)
    ranges = candidates.attrs["z2_ranges"]
    fixed = candidates.attrs["z2_fixed"]
    data = _load_q2_tables(tables)
    venues, slots = data["venues"].copy(), data["time_slots"].copy()
    slots["slot_id"] = slots.slot_id.astype(str)
    prime = set(slots.nlargest(len(slots) // 4, "global_prime_score").slot_id)
    limits = data["dynamic_resource_limits"].copy()
    limits["reference_date"] = limits.reference_date.astype(str)
    env = cp.Envr()
    model = env.createModel("q2_restricted_venue_slot_mip")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.unlink(missing_ok=True)
        model.setLogFile(_copt_log_name(log_path))
    x = {index: model.addVar(vtype=cp.COPT.BINARY, name=f"x_{index}") for index in candidates.index}
    y = {str(venue): model.addVar(vtype=cp.COPT.BINARY, name=f"use_{venue}") for venue in venues.venue_id}
    for match_id, block in candidates.groupby("match_id"):
        model.addConstr(cp.quicksum(x[index] for index in block.index) == 1, name=f"assignment_{match_id}")
    for venue in venues.itertuples(index=False):
        block = candidates.loc[candidates.candidate_venue_id.eq(str(venue.venue_id))]
        # The input table imposes each venue's stage-wide lower bound directly;
        # it is not an optional activation threshold.
        model.addConstr(cp.quicksum(x[index] for index in block.index) >= int(venue.min_total_matches), name=f"min_use_{venue.venue_id}")
        model.addConstr(cp.quicksum(x[index] for index in block.index) <= int(venue.max_total_matches) * y[str(venue.venue_id)], name=f"max_use_{venue.venue_id}")
    for (venue, date), block in candidates.groupby(["candidate_venue_id", "reference_date"]):
        max_daily = int(venues.loc[venues.venue_id.astype(str).eq(venue), "max_matches_per_day"].iloc[0])
        model.addConstr(cp.quicksum(x[index] for index in block.index) <= max_daily, name=f"daily_{venue}_{date}")
    for slot_id, block in candidates.groupby("slot_id"):
        capacity = int(slots.loc[slots.slot_id.eq(slot_id), "broadcast_capacity"].iloc[0])
        model.addConstr(cp.quicksum(x[index] for index in block.index) <= capacity, name=f"slot_{slot_id}")
    for (venue, utc), block in candidates.groupby(["candidate_venue_id", "utc_datetime"]):
        model.addConstr(cp.quicksum(x[index] for index in block.index) <= 1, name=f"collision_{venue}_{str(utc)[:10]}")
    g_max, g_min = model.addVar(lb=0.0, ub=3.0, name="golden_max"), model.addVar(lb=0.0, ub=3.0, name="golden_min")
    l_max, l_min = model.addVar(lb=0.0, ub=3.0, name="large_venue_max"), model.addVar(lb=0.0, ub=3.0, name="large_venue_min")
    large = set(venues.nlargest(len(venues) // 4, "capacity").venue_id.astype(str))
    for team in sorted(set(incumbent.team_a).union(incumbent.team_b)):
        team_block = candidates.loc[(candidates.team_a.eq(team) | candidates.team_b.eq(team)) & candidates.slot_id.isin(prime)]
        total = cp.quicksum(x[index] for index in team_block.index)
        model.addConstr(total <= g_max, name=f"golden_max_{team}")
        model.addConstr(total >= g_min, name=f"golden_min_{team}")
        large_block = candidates.loc[(candidates.team_a.eq(team) | candidates.team_b.eq(team)) & candidates.candidate_venue_id.isin(large)]
        large_total = cp.quicksum(x[index] for index in large_block.index)
        model.addConstr(large_total <= l_max, name=f"large_max_{team}")
        model.addConstr(large_total >= l_min, name=f"large_min_{team}")
    model.addConstr(g_max - g_min <= 2, name="golden_fairness")
    r3 = candidates.loc[candidates.round_in_group.eq(3)]
    for date, block in r3.groupby("reference_date"):
        capacity = int(limits.loc[limits.reference_date.eq(date), "high_security_capacity"].iloc[0])
        high = block.loc[block.required_security_level.ge(3)]
        model.addConstr(cp.quicksum(x[index] for index in high.index) <= capacity, name=f"third_round_security_{date}")
    t_coef, t_constant = _scaled_coefficient(0.25, ranges["T"])
    b_coef, b_constant = _scaled_coefficient(0.25, ranges["B"])
    c_coef, c_constant = _scaled_coefficient(-0.08, ranges["C"])
    d_coef, d_constant = _scaled_coefficient(-0.07, ranges["D"])
    fixed_constant = t_constant + b_constant + c_constant + d_constant + 0.15 * fixed["U"] + 0.10 * fixed["H"]
    additive = cp.quicksum(
        (
            t_coef * float(candidates.loc[index, "ticket_revenue_usd"])
            + b_coef * float(candidates.loc[index, "broadcast_value_usd"])
            + c_coef * float(candidates.loc[index, "operation_cost_musd"])
            + d_coef * float(candidates.loc[index, "travel_burden"])
            - 0.04 / len(incumbent) * float(candidates.loc[index, "risk_index"])
        ) * x[index]
        for index in candidates.index
    )
    setup = cp.quicksum(
        c_coef * float(venues.loc[venues.venue_id.astype(str).eq(venue), "setup_cost_musd"].iloc[0]) * y[venue]
        for venue in y
    )
    fairness = -0.01 * (g_max - g_min) - 0.01 * (l_max - l_min)
    model.setObjective(fixed_constant + additive + setup + fairness, cp.COPT.MAXIMIZE)
    model.solve()
    if not getattr(model, "hasmipsol", False) and not getattr(model, "hasmipSol", False):
        raise RuntimeError(f"COPT returned no MIP solution, status={model.status}")
    chosen = candidates.loc[[index for index, variable in x.items() if variable.x > 0.5]].copy()
    if len(chosen) != 72:
        raise RuntimeError("COPT did not choose exactly 72 match assignments")
    output = chosen.rename(columns={"candidate_venue_id": "venue_id", "venue_city": "city", "venue_country": "country", "venue_timezone": "timezone"})
    output["local_datetime"] = [pd.Timestamp(value).tz_convert(ZoneInfo(timezone)).isoformat() for value, timezone in zip(output.utc_datetime, output.timezone, strict=True)]
    output["utc_datetime"] = output.utc_datetime.map(lambda value: pd.Timestamp(value).isoformat())
    output["expected_attendance"] = np.minimum(output.expected_attendance_base, output.venue_capacity)
    output["expected_tv_viewers"] = output.predicted_tv_viewers
    output["fairness_penalty"] = np.nan
    output["travel_cost_index"] = output.travel_burden
    output["total_objective_value"] = np.nan
    columns = ["match_id", "group_id", "round_in_group", "team_a", "team_b", "venue_id", "slot_id", "city", "country", "reference_date", "reference_kickoff_time", "local_datetime", "utc_datetime", "required_security_level", "expected_attendance", "expected_tv_viewers", "ticket_revenue_usd", "broadcast_value_usd", "travel_cost_index", "fairness_penalty", "risk_index", "total_objective_value"]
    scored, report = evaluate_q2_schedule(tables, output[columns])
    if not np.isclose(float(model.objval), report["objective"], atol=1e-8):
        raise RuntimeError(
            f"COPT Z2 {float(model.objval)} disagrees with independent evaluator {report['objective']}"
        )
    evidence = {
        "rows": int(model.getAttr(cp.COPT.Attr.Rows)),
        "columns": int(model.getAttr(cp.COPT.Attr.Cols)),
        "binary_variables": int(model.getAttr(cp.COPT.Attr.Bins)),
        "best_bound": float(model.getAttr(cp.COPT.Attr.BestObj)),
        "relative_gap": float(model.getAttr(cp.COPT.Attr.BestGap)),
        "solve_time_seconds": float(model.getAttr(cp.COPT.Attr.SolvingTime)),
    }
    return CoptQ2Artifacts(
        schedule=scored[columns].sort_values("match_id").reset_index(drop=True),
        candidate_count=len(candidates),
        status=int(model.status),
        mip_objective=float(model.objval),
        score_report=report,
        solve_evidence=evidence,
    )
