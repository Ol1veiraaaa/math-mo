"""Official Q3 state update and synchronous qualification simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Q1Tables


OUTPUT_COLUMNS = [
    "match_id", "group_id", "team_a", "team_b",
    "updated_p_team_a_advance", "updated_p_team_b_advance",
    "updated_expected_attendance", "updated_expected_tv_viewers",
    "stakeless_risk", "collusion_risk", "updated_attractiveness",
    "recommended_broadcast_priority", "recommended_security_level",
    "recommended_transport_level", "recommended_ticket_adjustment",
    "updated_ticket_revenue_usd", "updated_broadcast_value_usd",
    "resource_cost_index", "risk_exposure_index", "static_net_value",
    "dynamic_net_value", "improvement_rate",
]


@dataclass(frozen=True)
class Q3StateArtifacts:
    matches: pd.DataFrame
    simulation_summary: pd.DataFrame


@dataclass(frozen=True)
class Q3Validation:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _table(tables: Q1Tables, name: str) -> pd.DataFrame:
    return pd.read_excel(tables.paths.workbook, sheet_name=name)


def _team_state(live: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for row in live.itertuples(index=False):
        for side, other in (("a", "b"), ("b", "a")):
            gf, ga = getattr(row, f"goals_{side}"), getattr(row, f"goals_{other}")
            records.append({
                "team": getattr(row, f"team_{side}"),
                "points": 3 if gf > ga else 1 if gf == ga else 0,
                "gf": gf, "ga": ga,
                "xgf": getattr(row, f"xg_{side}"), "xga": getattr(row, f"xg_{other}"),
                "red": getattr(row, f"red_cards_{side}"), "games": 1,
            })
    state = pd.DataFrame(records).groupby("team", as_index=False).sum(numeric_only=True)
    state["s"] = (
        0.45 * state.points / (3 * state.games)
        + 0.35 * (0.5 + 0.5 * np.tanh(((state.xgf - state.xga) / state.games) / 1.25))
        + 0.20 * np.exp(-0.55 * state.red / state.games)
    ).clip(0.0, 1.0)
    return state


def _simulate_qualification(
    r3: pd.DataFrame,
    state: pd.DataFrame,
    n_simulations: int = 20_000,
    seed: int = 20260814,
) -> pd.DataFrame:
    """Simulate all 24 R3 matches together and retain conditional qualification risks."""

    rng = np.random.default_rng(seed)
    r3 = r3.copy().reset_index(drop=True)
    state_idx = state.set_index("team")
    r3["state_a"] = r3.team_a.map(state_idx.s)
    r3["state_b"] = r3.team_b.map(state_idx.s)
    delta = 0.32 * (r3.state_a - r3.state_b) - 0.055 * (r3.injury_a - r3.injury_b)
    r3["lambda_a"] = np.clip(r3.expected_goals_a * np.exp(delta - 0.04 * r3.injury_a), 0.15, 4.50)
    r3["lambda_b"] = np.clip(r3.expected_goals_b * np.exp(-delta - 0.04 * r3.injury_b), 0.15, 4.50)
    if not np.isfinite(r3[["lambda_a", "lambda_b"]].to_numpy()).all():
        raise ValueError("third-round Poisson means must be finite")

    groups = sorted(r3.group_id.unique())
    team_sets = [
        sorted(set(r3.loc[r3.group_id.eq(group), "team_a"]).union(r3.loc[r3.group_id.eq(group), "team_b"]))
        for group in groups
    ]
    group_lookup = {group: index for index, group in enumerate(groups)}
    team_lookup = {
        group: {team: index for index, team in enumerate(teams)}
        for group, teams in zip(groups, team_sets, strict=True)
    }
    points = np.array([[state_idx.loc[team, "points"] for team in teams] for teams in team_sets], dtype=float)
    gf = np.array([[state_idx.loc[team, "gf"] for team in teams] for teams in team_sets], dtype=float)
    ga = np.array([[state_idx.loc[team, "ga"] for team in teams] for teams in team_sets], dtype=float)
    match_group = r3.group_id.map(group_lookup).to_numpy()
    match_a = np.array([team_lookup[group][team] for group, team in zip(r3.group_id, r3.team_a, strict=True)])
    match_b = np.array([team_lookup[group][team] for group, team in zip(r3.group_id, r3.team_b, strict=True)])

    sim_points = np.broadcast_to(points, (n_simulations, *points.shape)).copy()
    sim_gf = np.broadcast_to(gf, (n_simulations, *gf.shape)).copy()
    sim_ga = np.broadcast_to(ga, (n_simulations, *ga.shape)).copy()
    scores_a = rng.poisson(r3.lambda_a.to_numpy(), size=(n_simulations, len(r3)))
    scores_b = rng.poisson(r3.lambda_b.to_numpy(), size=(n_simulations, len(r3)))
    draws = np.arange(n_simulations)
    for index in range(len(r3)):
        group, a, b = match_group[index], match_a[index], match_b[index]
        goals_a, goals_b = scores_a[:, index], scores_b[:, index]
        sim_gf[:, group, a] += goals_a
        sim_ga[:, group, a] += goals_b
        sim_gf[:, group, b] += goals_b
        sim_ga[:, group, b] += goals_a
        sim_points[:, group, a] += np.where(goals_a > goals_b, 3, np.where(goals_a == goals_b, 1, 0))
        sim_points[:, group, b] += np.where(goals_b > goals_a, 3, np.where(goals_a == goals_b, 1, 0))

    sim_gd = sim_gf - sim_ga
    qualified = np.zeros((n_simulations, len(groups), 4), dtype=bool)
    third_team = np.empty((n_simulations, len(groups)), dtype=int)
    for group_index in range(len(groups)):
        order = np.lexsort((rng.random((n_simulations, 4)), sim_gf[:, group_index], sim_gd[:, group_index], sim_points[:, group_index]), axis=1)[:, ::-1]
        qualified[draws[:, None], group_index, order[:, :2]] = True
        third_team[:, group_index] = order[:, 2]
    third_points = np.take_along_axis(sim_points, third_team[:, :, None], axis=2).squeeze(2)
    third_gd = np.take_along_axis(sim_gd, third_team[:, :, None], axis=2).squeeze(2)
    third_gf = np.take_along_axis(sim_gf, third_team[:, :, None], axis=2).squeeze(2)
    third_order = np.lexsort((rng.random((n_simulations, len(groups))), third_gf, third_gd, third_points), axis=1)[:, ::-1]
    for rank in range(8):
        selected_group = third_order[:, rank]
        qualified[draws, selected_group, third_team[draws, selected_group]] = True

    probabilities: dict[str, float] = {}
    for group_index, teams in enumerate(team_sets):
        for team_index, team in enumerate(teams):
            probabilities[team] = float(qualified[:, group_index, team_index].mean())
    r3["updated_p_team_a_advance"] = r3.team_a.map(probabilities)
    r3["updated_p_team_b_advance"] = r3.team_b.map(probabilities)
    r3["stakeless_risk"] = (
        0.5 * (2 * r3.updated_p_team_a_advance - 1) ** 2
        + 0.5 * (2 * r3.updated_p_team_b_advance - 1) ** 2
    )

    collusion: list[float] = []
    conditional_rows: list[dict[str, float | str]] = []
    for index, match in enumerate(r3.itertuples(index=False)):
        group, a, b = match_group[index], match_a[index], match_b[index]
        q_a, q_b = qualified[:, group, a], qualified[:, group, b]
        draw_mask = scores_a[:, index] == scores_b[:, index]
        a_win_mask = scores_a[:, index] > scores_b[:, index]
        b_win_mask = scores_b[:, index] > scores_a[:, index]
        p_a_draw = float(q_a[draw_mask].mean())
        p_b_draw = float(q_b[draw_mask].mean())
        p_a_win = float(q_a[a_win_mask].mean())
        p_b_win = float(q_b[b_win_mask].mean())
        common_draw = min(p_a_draw, p_b_draw)
        incentive_gain = 0.5 * max(p_a_win - p_a_draw, 0.0) + 0.5 * max(p_b_win - p_b_draw, 0.0)
        risk = float(np.clip(common_draw * (1 - np.clip(incentive_gain, 0, 1)) * (1 - 0.35 * r3.loc[index, "stakeless_risk"]), 0, 1))
        collusion.append(risk)
        conditional_rows.append({
            "match_id": match.match_id,
            "p_a_advance_given_draw": p_a_draw,
            "p_b_advance_given_draw": p_b_draw,
            "p_a_advance_given_a_win": p_a_win,
            "p_b_advance_given_b_win": p_b_win,
            "common_draw_benefit": common_draw,
            "win_incentive_gain": incentive_gain,
        })
    r3["collusion_risk"] = collusion
    conditional = pd.DataFrame(conditional_rows)
    r3 = r3.merge(conditional, on="match_id", how="left")
    return r3


def build_q3_state(
    tables: Q1Tables,
    q2_csv: Path,
    q1_csv: Path,
    n_simulations: int = 20_000,
    seed: int = 20260814,
) -> Q3StateArtifacts:
    """Build every pre-resource static and feedback-updated dynamic Q3 quantity."""

    live = _table(tables, "live_group_results")
    base = _table(tables, "base_predictions")
    venues = _table(tables, "venues")
    ticket = _table(tables, "ticket_broadcast")
    security = _table(tables, "security_requirements")
    schedule = pd.read_csv(q2_csv)
    q1 = pd.read_csv(q1_csv)
    r3 = schedule.loc[schedule.round_in_group.eq(3)].copy()
    r3 = r3.drop(columns=[column for column in ("expected_tv_viewers",) if column in r3]).merge(q1[["match_id", "predicted_tv_viewers"]], on="match_id")
    r3 = r3.merge(base[["match_id", "expected_goals_a", "expected_goals_b", "uncertainty_index", "attractiveness_index"]], on="match_id")
    r3 = r3.merge(venues[["venue_id", "capacity", "security_level"]].rename(columns={"capacity": "venue_capacity", "security_level": "venue_security_level"}), on="venue_id")
    r3 = r3.merge(security[["match_id", "security_demand_score"]], on="match_id")
    r3["match_stage"] = "Group_Match_R3"
    r3 = r3.merge(ticket, on="match_stage")

    injury_records: list[dict[str, float | str]] = []
    for row in live.itertuples(index=False):
        injury_records.extend((
            {"team": row.team_a, "injury": float(row.injury_impact_level)},
            {"team": row.team_b, "injury": float(row.injury_impact_level)},
        ))
    injury = pd.DataFrame(injury_records).groupby("team").injury.mean().clip(0.0, 3.0)
    r3["injury_a"] = r3.team_a.map(injury).fillna(0.0)
    r3["injury_b"] = r3.team_b.map(injury).fillna(0.0)
    r3 = _simulate_qualification(r3, _team_state(live), n_simulations=n_simulations, seed=seed)

    prior = live.merge(base[["match_id", "expected_attendance_base"]], on="match_id").merge(q1[["match_id", "predicted_tv_viewers"]], on="match_id")
    feedback_rows: list[dict[str, float | str]] = []
    for row in prior.itertuples(index=False):
        attendance_ratio = float(np.clip(row.attendance / row.expected_attendance_base, 0.60, 1.50))
        tv_ratio = float(np.clip(row.tv_viewers / row.predicted_tv_viewers, 0.60, 1.50))
        for team in (row.team_a, row.team_b):
            feedback_rows.append({"team": team, "attendance_ratio": attendance_ratio, "tv_ratio": tv_ratio})
    feedback = pd.DataFrame(feedback_rows).groupby("team").mean(numeric_only=True).clip(0.75, 1.25)
    r3["feedback_attendance"] = np.sqrt(r3.team_a.map(feedback.attendance_ratio) * r3.team_b.map(feedback.attendance_ratio)).fillna(1.0)
    r3["feedback_tv"] = np.sqrt(r3.team_a.map(feedback.tv_ratio) * r3.team_b.map(feedback.tv_ratio)).fillna(1.0)
    r3["feedback_index"] = np.clip((0.5 * r3.feedback_attendance + 0.5 * r3.feedback_tv - 0.75) / 0.50, 0.0, 1.0)
    r3["importance"] = (
        4 * r3.updated_p_team_a_advance * (1 - r3.updated_p_team_a_advance)
        + 4 * r3.updated_p_team_b_advance * (1 - r3.updated_p_team_b_advance)
    ) / 2
    r3["state_mean"] = (r3.state_a + r3.state_b) / 2
    r3["injury_mean"] = (r3.injury_a + r3.injury_b) / 6
    r3["updated_attractiveness"] = np.clip(
        0.50 * r3.attractiveness_index / 100
        + 0.25 * r3.importance
        + 0.12 * r3.feedback_index
        + 0.08 * r3.state_mean
        + 0.05 * (1 - r3.injury_mean),
        0.0, 1.0,
    )
    r3["dynamic_attendance_demand"] = (
        r3.expected_attendance * r3.feedback_attendance
        * (0.88 + 0.27 * r3.importance)
        * (0.94 + 0.12 * r3.state_mean)
        * (1 - 0.10 * r3.injury_mean)
    )
    r3["dynamic_tv_demand"] = (
        r3.predicted_tv_viewers * r3.feedback_tv
        * (0.87 + 0.30 * r3.importance)
        * (0.90 + 0.20 * r3.updated_attractiveness)
        * (1 - 0.06 * r3.injury_mean)
    )
    occupancy = np.clip(r3.dynamic_attendance_demand / r3.venue_capacity, 0.0, 1.0)
    r3["dynamic_security_demand"] = np.clip(
        0.45 * r3.security_demand_score + 0.25 * occupancy
        + 0.15 * r3.updated_attractiveness
        + 0.15 * (r3.stakeless_risk + r3.collusion_risk) / 2,
        0.0, 1.0,
    )
    r3["dynamic_attractiveness"] = r3.updated_attractiveness
    r3["dynamic_stakeless_risk"] = r3.stakeless_risk
    r3["dynamic_collusion_risk"] = r3.collusion_risk
    r3["static_attractiveness"] = r3.attractiveness_index / 100
    r3["static_attendance_demand"] = r3.expected_attendance
    r3["static_tv_demand"] = r3.predicted_tv_viewers
    r3["static_stakeless_risk"] = 1 - r3.uncertainty_index
    r3["static_collusion_risk"] = 0.0
    static_occupancy = np.clip(r3.static_attendance_demand / r3.venue_capacity, 0.0, 1.0)
    r3["static_security_demand"] = np.clip(
        0.45 * r3.security_demand_score + 0.25 * static_occupancy
        + 0.15 * r3.static_attractiveness + 0.15 * r3.static_stakeless_risk / 2,
        0.0, 1.0,
    )
    summary_columns = [
        "match_id", "lambda_a", "lambda_b", "updated_p_team_a_advance",
        "updated_p_team_b_advance", "p_a_advance_given_draw",
        "p_b_advance_given_draw", "p_a_advance_given_a_win",
        "p_b_advance_given_b_win", "common_draw_benefit",
        "win_incentive_gain", "importance", "updated_attractiveness",
        "stakeless_risk", "collusion_risk",
    ]
    return Q3StateArtifacts(matches=r3.sort_values("match_id").reset_index(drop=True), simulation_summary=r3[summary_columns].sort_values("match_id").reset_index(drop=True))


def validate_q3_result(tables: Q1Tables, q2_csv: Path, result: pd.DataFrame) -> Q3Validation:
    errors: list[str] = []
    if list(result.columns) != OUTPUT_COLUMNS:
        return Q3Validation(("result_3 columns do not follow the official template order",))
    if len(result) != 24 or not result.match_id.is_unique:
        errors.append("result_3 must contain each of the 24 third-round matches exactly once")
    numeric = result.drop(columns=["match_id", "group_id", "team_a", "team_b"])
    allowed_nan = np.asarray(numeric.columns == "improvement_rate")
    finite_or_allowed = np.isfinite(numeric.to_numpy(dtype=float)) | np.broadcast_to(allowed_nan, numeric.shape)
    if not finite_or_allowed.all():
        errors.append("result_3 contains non-finite required values")
    probabilities = result[["updated_p_team_a_advance", "updated_p_team_b_advance"]].to_numpy(dtype=float)
    if not np.logical_and(probabilities >= 0, probabilities <= 1).all():
        errors.append("advance probabilities must lie in [0, 1]")
    if not np.isclose(probabilities.sum(), 32.0, atol=1e-8):
        errors.append("total expected qualification places must equal 32")
    for column in ("stakeless_risk", "collusion_risk", "updated_attractiveness", "risk_exposure_index"):
        if not result[column].between(0, 1).all():
            errors.append(f"{column} must lie in [0, 1]")

    schedule = pd.read_csv(q2_csv).loc[lambda frame: frame.round_in_group.eq(3), ["match_id", "venue_id", "reference_date", "required_security_level"]]
    venues = _table(tables, "venues").set_index("venue_id")
    limits = _table(tables, "dynamic_resource_limits").copy()
    limits["reference_date"] = limits.reference_date.astype(str)
    checked = result.merge(schedule, on="match_id", how="left")
    if checked[["venue_id", "reference_date"]].isna().any().any():
        return Q3Validation(tuple(dict.fromkeys([*errors, "result_3 does not match the fixed third-round schedule"])))
    for row in checked.itertuples(index=False):
        if not int(row.required_security_level) <= int(row.recommended_security_level) <= int(venues.loc[row.venue_id, "security_level"]):
            errors.append(f"security level outside permitted range for {row.match_id}")
    for date, day in checked.groupby("reference_date"):
        limit = limits.loc[limits.reference_date.eq(str(date))].iloc[0]
        if day.recommended_broadcast_priority.eq(3).sum() > int(limit.high_broadcast_capacity):
            errors.append(f"high broadcast capacity exceeded on {date}")
        if day.recommended_security_level.ge(3).sum() > int(limit.high_security_capacity):
            errors.append(f"high security capacity exceeded on {date}")
        if day.recommended_transport_level.eq(3).sum() > int(limit.enhanced_transport_capacity):
            errors.append(f"enhanced transport capacity exceeded on {date}")
        if day.resource_cost_index.sum() > float(limit.daily_resource_budget_index) + 1e-9:
            errors.append(f"daily resource budget exceeded on {date}")
        if not day.recommended_ticket_adjustment.between(-float(limit.max_ticket_discount_rate), float(limit.max_ticket_increase_rate)).all():
            errors.append(f"ticket adjustment outside daily bounds on {date}")
    return Q3Validation(tuple(dict.fromkeys(errors)))
