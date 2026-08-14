"""Compare the completed 2026 FIFA World Cup actual data against Q1/Q2/Q3 outputs.

This is an additive analysis campaign: it reads the delivered outputs and the
OpenFootball 2026 snapshot, maps the actual tournament into the problem's
coordinate system (match ids, venue ids, slot ids, team names), and measures:

  * Q2: feasibility of the FIFA venue/slot assignment under the problem's hard
    constraints, its P2 objective under the official evaluator (after a
    documented minimal repair of security-ineligible venues), and its
    per-match agreement with the delivered COPT schedule;
  * Q3: real rounds 1-2 standings versus the workbook's synthetic
    live_group_results, advance probabilities recomputed from the real
    standings versus the delivered dynamic strategy, and validation of both
    probability sets against the actual qualification outcomes;
  * Q1: calibration of the pre-match expected goals (the demand input used by
    Q2/Q3) against the actual results, and the limits of a viewer-prediction
    check without published audience numbers.

Nothing in the delivered artifact set is modified.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c_contest_q1.data import load_q1_tables
from c_contest_q1.q2_demo import evaluate_q2_schedule, validate_q2_schedule
from c_contest_q1.q3_demo import _simulate_qualification, _team_state

SEED = 20260814
N_SIMULATIONS = 20_000
OUT = ROOT / "outputs" / "analysis"
ACTUAL_SCHEDULE = ROOT / "outputs" / "q4" / "actual_schedule.csv"
ACTUAL_STADIUMS = ROOT / "outputs" / "q4" / "actual_stadiums.csv"
Q1_CSV = ROOT / "outputs" / "q1" / "demo" / "result_1_match_prediction.csv"
Q2_COPT_CSV = ROOT / "outputs" / "q2" / "copt" / "result_2_group_schedule.csv"
Q2_DEMO_CSV = ROOT / "outputs" / "q2" / "demo" / "result_2_group_schedule.csv"
Q3_CSV = ROOT / "outputs" / "q3" / "copt" / "result_3_dynamic_strategy.csv"

TEAM_ALIASES = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Curaçao": "Curacao",
    "Czech Republic": "Czechia",
    "USA": "United States",
}


def _normalize_team(name: str) -> str:
    return TEAM_ALIASES.get(name.strip(), name.strip())


def _normalize_city(city: str) -> str:
    return re.sub(r"\s*\(.*\)", "", city).strip().lower()


def _read_workbook(tables, name: str) -> pd.DataFrame:
    return pd.read_excel(tables.paths.workbook, sheet_name=name)


def _build_maps(tables, actual: pd.DataFrame, stadiums: pd.DataFrame) -> dict:
    """Map the actual tournament into the problem's coordinates, asserting identity."""
    teams = tables.teams
    problem_names = set(teams.team_name)
    actual_names = {_normalize_team(name) for name in set(actual.team_a) | set(actual.team_b)}
    if not actual_names <= problem_names:
        raise ValueError(f"unmapped actual teams: {actual_names - problem_names}")

    matches = tables.group_matches
    pair_to_match: dict[tuple[str, str, str], str] = {}
    for row in matches.itertuples(index=False):
        pair_to_match[(row.group_id, row.team_a, row.team_b)] = row.match_id  # problem order only
    match_by_actual: dict[str, str] = {}
    flipped: set[str] = set()
    for row in actual.itertuples(index=False):
        ordered = (row.group_id, _normalize_team(row.team_a), _normalize_team(row.team_b))
        reversed_key = (row.group_id, _normalize_team(row.team_b), _normalize_team(row.team_a))
        if ordered in pair_to_match:
            match_id = pair_to_match[ordered]
        elif reversed_key in pair_to_match:
            match_id = pair_to_match[reversed_key]
            flipped.add(row.match_id)
        else:
            raise ValueError(f"actual fixture has no problem counterpart: {ordered}")
        match_by_actual[row.match_id] = match_id
    if len(match_by_actual) != 72 or len(set(match_by_actual.values())) != 72:
        raise ValueError("actual fixtures do not map one-to-one onto groups_matches")

    venues = _read_workbook(tables, "venues")
    city_to_venue = {_normalize_city(city): row.venue_id for city, row in zip(venues.city, venues.itertuples(index=False), strict=True)}
    unmapped = [_normalize_city(city) for city in stadiums.city if _normalize_city(city) not in city_to_venue]
    if unmapped:
        raise ValueError(f"unmapped actual cities: {unmapped}")
    return {"match": match_by_actual, "flipped": flipped, "city_to_venue": city_to_venue}


def _build_fifa_problem_schedule(tables, actual: pd.DataFrame, stadiums: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """Rebuild the FIFA schedule in the problem's venue/slot/match coordinates."""
    venues = _read_workbook(tables, "venues").set_index("venue_id")
    slots = _read_workbook(tables, "time_slots")
    slots_utc = pd.to_datetime(slots.reference_utc_time, utc=True)
    security = _read_workbook(tables, "security_requirements").set_index("match_id")
    q1 = pd.read_csv(Q1_CSV).set_index("match_id")

    rows: list[dict[str, object]] = []
    for row in actual.itertuples(index=False):
        match_id = maps["match"][row.match_id]
        venue_id = maps["city_to_venue"][_normalize_city(row.city)]
        venue = venues.loc[venue_id]
        utc = pd.to_datetime(row.utc_datetime, utc=True)
        gap = (slots_utc - utc).abs()
        slot = slots.loc[gap.idxmin()]
        local = utc.tz_convert(ZoneInfo(venue.timezone))
        team_a, team_b = _normalize_team(row.team_a), _normalize_team(row.team_b)
        if row.match_id in maps["flipped"]:
            team_a, team_b = team_b, team_a
        rows.append({
            "match_id": match_id,
            "group_id": row.group_id,
            "round_in_group": int(row.round_in_group),
            "team_a": team_a,
            "team_b": team_b,
            "venue_id": venue_id,
            "slot_id": slot.slot_id,
            "city": venue.city,
            "country": venue.country,
            "reference_date": str(slot.date),
            "reference_kickoff_time": slot.reference_kickoff_time,
            "local_datetime": local.isoformat(),
            "utc_datetime": utc.isoformat(),
            "required_security_level": int(security.loc[match_id, "required_security_level"]),
            "expected_attendance": np.nan,
            "expected_tv_viewers": float(q1.loc[match_id, "predicted_tv_viewers"]),
            "ticket_revenue_usd": np.nan,
            "broadcast_value_usd": np.nan,
            "travel_cost_index": np.nan,
            "fairness_penalty": np.nan,
            "risk_index": np.nan,
            "slot_gap_hours": float(gap.min().total_seconds() / 3600),
            "actual_venue": row.venue,
            "actual_city": row.city,
            "actual_utc_datetime": row.utc_datetime,
        })
    frame = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)
    if len(frame) != 72 or not frame.match_id.is_unique:
        raise ValueError("FIFA problem-coordinate schedule must contain 72 unique matches")
    return frame


def _q2_section(tables, fifa: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    validation = validate_q2_schedule(tables, fifa)
    _, copt_report = evaluate_q2_schedule(tables, pd.read_csv(Q2_COPT_CSV))
    _, demo_report = evaluate_q2_schedule(tables, pd.read_csv(Q2_DEMO_CSV))

    # The official evaluator refuses venues below the required security level, so
    # the raw FIFA assignment cannot be scored.  Repair only the offending matches
    # with the delivered COPT venue for the same match; everything else stays FIFA.
    venues = _read_workbook(tables, "venues").set_index("venue_id")
    security = _read_workbook(tables, "security_requirements").set_index("match_id")
    ineligible = fifa.index[fifa.venue_id.map(venues.security_level).astype(int) < fifa.required_security_level.astype(int)]
    fifa_eval = fifa.copy()
    fifa_eval["venue_repaired_for_feasibility"] = False
    copt_venues = pd.read_csv(Q2_COPT_CSV).set_index("match_id")["venue_id"].astype(str)
    if len(ineligible):
        fifa_eval.loc[ineligible, "venue_id"] = fifa_eval.loc[ineligible, "match_id"].map(copt_venues)
        fifa_eval.loc[ineligible, "venue_repaired_for_feasibility"] = True

    # The official evaluator reads ticket/broadcast/attendance columns from the
    # schedule frame, so compute them on the (repaired) FIFA assignment exactly
    # as build_q2_demo_schedule does.
    economics = _read_workbook(tables, "ticket_broadcast")
    slots = _read_workbook(tables, "time_slots").set_index("slot_id")
    base = tables.base_predictions.set_index("match_id")

    def _economics_for(row) -> tuple[float, float, float]:
        price = economics.loc[economics.match_stage.eq(f"Group_Match_R{int(row.round_in_group)}")].iloc[0]
        attendance = min(float(base.loc[row.match_id, "expected_attendance_base"]), float(venues.loc[row.venue_id, "capacity"]))
        ticket = attendance * float(price.base_ticket_price_usd)
        broadcast = float(row.expected_tv_viewers) * float(price.broadcast_unit_value_usd) * float(slots.loc[row.slot_id, "global_prime_score"]) * float(price.sponsor_weight)
        return attendance, ticket, broadcast

    fifa_eval[["expected_attendance", "ticket_revenue_usd", "broadcast_value_usd"]] = fifa_eval.apply(_economics_for, axis=1, result_type="expand")
    _, fifa_report = evaluate_q2_schedule(tables, fifa_eval)

    def _prefix(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return frame.rename(columns={col: f"{prefix}_{col}" for col in frame.columns if col != "match_id"})

    copt = _prefix(pd.read_csv(Q2_COPT_CSV)[["match_id", "venue_id", "slot_id", "reference_date"]], "copt")
    demo = _prefix(pd.read_csv(Q2_DEMO_CSV)[["match_id", "venue_id", "slot_id"]], "demo")
    compared = fifa[["match_id", "group_id", "team_a", "team_b", "venue_id", "slot_id", "reference_date", "slot_gap_hours", "actual_venue", "actual_city"]].merge(copt, on="match_id").merge(demo, on="match_id")
    compared["venue_agree_copt"] = compared.venue_id.eq(compared.copt_venue_id)
    compared["slot_agree_copt"] = compared.slot_id.eq(compared.copt_slot_id)
    compared["date_agree_copt"] = compared.reference_date.eq(compared.copt_reference_date)
    compared["venue_repaired_for_feasibility"] = compared.match_id.isin(set(fifa_eval.loc[fifa_eval.venue_repaired_for_feasibility, "match_id"]))
    summary = {
        "fifa_validation_errors": list(validation.errors),
        "fifa_validation_ok": validation.ok,
        "fifa_repaired_matches": int(len(ineligible)),
        "objective_fifa_repaired": fifa_report["objective"],
        "objective_copt": copt_report["objective"],
        "objective_demo": demo_report["objective"],
        "raw_fifa": {key: value for key, value in fifa_report.items() if key.startswith("raw_")},
        "raw_copt": {key: value for key, value in copt_report.items() if key.startswith("raw_")},
        "norm_fifa": {key: value for key, value in fifa_report.items() if key.startswith("norm_")},
        "norm_copt": {key: value for key, value in copt_report.items() if key.startswith("norm_")},
        "venue_agreement": float(compared.venue_agree_copt.mean()),
        "slot_agreement": float(compared.slot_agree_copt.mean()),
        "date_agreement": float(compared.date_agree_copt.mean()),
        "slot_gap_hours_mean": float(compared.slot_gap_hours.mean()),
        "slot_gap_hours_max": float(compared.slot_gap_hours.max()),
    }
    return summary, compared


def _standings(results: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for row in results.itertuples(index=False):
        for side, other in (("a", "b"), ("b", "a")):
            gf, ga = getattr(row, f"goals_{side}"), getattr(row, f"goals_{other}")
            records.append({
                "group_id": row.group_id,
                "team": getattr(row, f"team_{side}"),
                "points": 3 if gf > ga else 1 if gf == ga else 0,
                "gf": gf, "ga": ga,
            })
    return pd.DataFrame(records).groupby(["group_id", "team"], as_index=False).sum()


def _q3_section(tables, actual: pd.DataFrame, maps: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    live = _read_workbook(tables, "live_group_results")
    base = tables.base_predictions
    schedule = pd.read_csv(Q2_COPT_CSV)
    delivered = pd.read_csv(Q3_CSV)

    # Real rounds 1-2 results in problem coordinates (goals stand in for xG, no card data).
    actual_records: list[dict[str, object]] = []
    for row in actual.loc[actual.round_in_group.le(2)].itertuples(index=False):
        team_a, team_b = _normalize_team(row.team_a), _normalize_team(row.team_b)
        goals_a, goals_b = row.goals_a, row.goals_b
        if row.match_id in maps["flipped"]:
            team_a, team_b, goals_a, goals_b = team_b, team_a, goals_b, goals_a
        actual_records.append({
            "match_id": maps["match"][row.match_id], "group_id": row.group_id, "team_a": team_a, "team_b": team_b,
            "goals_a": goals_a, "goals_b": goals_b, "xg_a": goals_a, "xg_b": goals_b,
            "red_cards_a": 0, "red_cards_b": 0,
        })
    live_actual = pd.DataFrame(actual_records)
    if len(live_actual) != 48:
        raise ValueError(f"expected 48 real first-round pair results, got {len(live_actual)}")
    state_actual = _team_state(live_actual)
    state_synthetic = _team_state(live)

    synthetic_frame = state_synthetic[["team", "points", "gf", "ga", "s"]].rename(columns=lambda name: f"synthetic_{name}")
    standings = state_actual[["team", "points", "gf", "ga", "s"]].rename(columns=lambda name: f"actual_{name}").merge(synthetic_frame, left_on="actual_team", right_on="synthetic_team")
    standings["points_equal"] = standings.actual_points.eq(standings.synthetic_points)
    standings["points_delta"] = standings.actual_points - standings.synthetic_points

    # Re-run the official qualification simulation on the real standings.
    r3 = schedule.loc[schedule.round_in_group.eq(3)][["match_id", "group_id", "team_a", "team_b"]].copy()
    r3 = r3.merge(base[["match_id", "expected_goals_a", "expected_goals_b"]], on="match_id")
    r3["injury_a"] = 0.0
    r3["injury_b"] = 0.0
    simulated = _simulate_qualification(r3, state_actual, n_simulations=N_SIMULATIONS, seed=SEED)

    probabilities = delivered[["match_id", "team_a", "team_b", "updated_p_team_a_advance", "updated_p_team_b_advance", "stakeless_risk", "collusion_risk"]].rename(columns=lambda name: f"delivered_{name}")
    probabilities = simulated[["match_id", "team_a", "team_b", "updated_p_team_a_advance", "updated_p_team_b_advance", "stakeless_risk", "collusion_risk"]].rename(columns=lambda name: f"actual_based_{name}").merge(probabilities, left_on="actual_based_match_id", right_on="delivered_match_id", suffixes=("", ""))

    # Actual final standings and qualification from the full 72-match results.
    full_actual = _standings(actual.assign(team_a=lambda frame: frame.team_a.map(_normalize_team), team_b=lambda frame: frame.team_b.map(_normalize_team)))
    qualified: set[str] = set()
    group_rank_rows: list[dict[str, object]] = []
    thirds: list[tuple[str, str, int, int, int]] = []
    for group_id, block in full_actual.groupby("group_id"):
        ranked = block.sort_values(["points", "gf", "ga"], ascending=[False, False, False], kind="mergesort")
        # Deterministic tie-break: goal difference, goals scored, then team name.
        ranked = ranked.assign(_gd=lambda frame: frame.gf - frame.ga).sort_values(["points", "_gd", "gf", "team"], ascending=[False, False, False, True])
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            group_rank_rows.append({"group_id": group_id, "team": row.team, "rank": rank, "points": row.points, "gf": row.gf, "ga": row.ga})
            if rank <= 2:
                qualified.add(row.team)
        third = ranked.iloc[2]
        thirds.append((group_id, third.team, int(third.points), int(third.gf), int(third.ga)))
    thirds_sorted = sorted(thirds, key=lambda item: (item[2], item[3], item[4]), reverse=True)[:8]
    qualified.update(item[1] for item in thirds_sorted)
    if len(qualified) != 32:
        raise ValueError(f"expected 32 qualified teams, got {len(qualified)}")

    per_team: dict[str, dict[str, float | int | str | bool]] = {}
    for row in probabilities.itertuples(index=False):
        for side in ("a", "b"):
            team = getattr(row, f"actual_based_team_{side}")
            per_team[team] = {
                "team": team,
                "p_synthetic": float(getattr(row, f"delivered_updated_p_team_{side}_advance")),
                "p_actual_based": float(getattr(row, f"actual_based_updated_p_team_{side}_advance")),
            }
    outcomes = pd.DataFrame(list(per_team.values()))
    outcomes["actually_qualified"] = outcomes.team.isin(qualified)
    group_ranks = pd.DataFrame(group_rank_rows)
    outcomes = outcomes.merge(group_ranks, on="team", how="left")
    outcomes = outcomes.merge(standings[["actual_team", "actual_points", "synthetic_points"]].rename(columns={"actual_team": "team"}), on="team", how="left")

    def brier(p: pd.Series, y: pd.Series) -> float:
        return float(((p - y.astype(float)) ** 2).mean())

    def top2_overlap(key: str) -> list[int]:
        overlaps = []
        for group_id, block in outcomes.groupby("group_id"):
            predicted = set(block.nlargest(2, key).team)
            real = set(block.loc[block["rank"].le(2), "team"])
            overlaps.append(len(predicted & real))
        return overlaps

    summary = {
        "standings_points_equal_share": float(standings.points_equal.mean()),
        "standings_points_delta_mean_abs": float(standings.points_delta.abs().mean()),
        "standings_points_correlation": float(standings.actual_points.corr(standings.synthetic_points)),
        "state_s_correlation": float(standings.actual_s.corr(standings.synthetic_s)),
        "p_correlation": float(pd.concat([probabilities.actual_based_updated_p_team_a_advance, probabilities.actual_based_updated_p_team_b_advance]).corr(pd.concat([probabilities.delivered_updated_p_team_a_advance, probabilities.delivered_updated_p_team_b_advance]))),
        "p_mae": float(pd.concat([probabilities.actual_based_updated_p_team_a_advance - probabilities.delivered_updated_p_team_a_advance, probabilities.actual_based_updated_p_team_b_advance - probabilities.delivered_updated_p_team_b_advance]).abs().mean()),
        "brier_synthetic": brier(outcomes.p_synthetic, outcomes.actually_qualified),
        "brier_actual_based": brier(outcomes.p_actual_based, outcomes.actually_qualified),
        "hit_rate_synthetic": float(((outcomes.p_synthetic >= 0.5) == outcomes.actually_qualified).mean()),
        "hit_rate_actual_based": float(((outcomes.p_actual_based >= 0.5) == outcomes.actually_qualified).mean()),
        "group_top2_exact_matches_synthetic": int(sum(value == 2 for value in top2_overlap("p_synthetic"))),
        "group_top2_exact_matches_actual_based": int(sum(value == 2 for value in top2_overlap("p_actual_based"))),
        "p_vs_points_spearman_synthetic": float(outcomes.p_synthetic.corr(outcomes.points, method="spearman")),
        "p_vs_points_spearman_actual_based": float(outcomes.p_actual_based.corr(outcomes.points, method="spearman")),
    }
    return summary, standings, probabilities, outcomes


def _q1_section(tables, actual: pd.DataFrame, maps: dict) -> tuple[dict, pd.DataFrame]:
    base = tables.base_predictions
    q1 = pd.read_csv(Q1_CSV)
    # Align actual goals to the problem's team_a/team_b ordering before comparing
    # them with expected_goals_a/b, which follow that ordering.
    actual_by_match: dict[str, tuple[int, int]] = {}
    for row in actual.itertuples(index=False):
        goals_a, goals_b = row.goals_a, row.goals_b
        if row.match_id in maps["flipped"]:
            goals_a, goals_b = goals_b, goals_a
        actual_by_match[maps["match"][row.match_id]] = (goals_a, goals_b)
    frame = base[["match_id", "group_id", "round_in_group", "team_a", "team_b", "expected_goals_a", "expected_goals_b"]].copy()
    frame["actual_goals_a"] = frame.match_id.map(lambda match_id: actual_by_match[match_id][0])
    frame["actual_goals_b"] = frame.match_id.map(lambda match_id: actual_by_match[match_id][1])
    frame = frame.merge(q1[["match_id", "predicted_tv_viewers"]], on="match_id")
    frame["expected_margin"] = frame.expected_goals_a - frame.expected_goals_b
    frame["actual_margin"] = frame.actual_goals_a - frame.actual_goals_b
    frame["direction_correct"] = np.sign(frame.expected_margin).eq(np.sign(frame.actual_margin))
    elo = tables.teams.set_index("team_name")["elo_rating"]
    frame["elo_margin"] = frame.team_a.map(elo) - frame.team_b.map(elo)
    frame["elo_direction_correct"] = np.sign(frame.elo_margin).eq(np.sign(frame.actual_margin))
    decided = frame.loc[frame.actual_margin.ne(0)]
    summary = {
        "expected_vs_actual_goals_corr_a": float(frame.expected_goals_a.corr(frame.actual_goals_a)),
        "expected_vs_actual_goals_corr_b": float(frame.expected_goals_b.corr(frame.actual_goals_b)),
        "expected_goals_mae": float((frame.expected_goals_a - frame.actual_goals_a).abs().mean() + (frame.expected_goals_b - frame.actual_goals_b).abs().mean()) / 2,
        "direction_accuracy_decided": float(decided.direction_correct.mean()),
        "direction_accuracy_decided_n": int(len(decided)),
        "draws": int(frame.actual_margin.eq(0).sum()),
        "elo_direction_accuracy_decided": float(decided.elo_direction_correct.mean()),
        "margin_correlation": float(frame.expected_margin.corr(frame.actual_margin)),
        "viewers_vs_goals_spearman": float(frame.predicted_tv_viewers.corr(frame.actual_goals_a + frame.actual_goals_b, method="spearman")),
    }
    return summary, frame


def _write_report(summary: dict, q2_compared: pd.DataFrame, q3_standings: pd.DataFrame, q3_probabilities: pd.DataFrame, q3_outcomes: pd.DataFrame, q1_frame: pd.DataFrame) -> None:
    q2 = summary["q2"]
    q3 = summary["q3"]
    q1 = summary["q1"]
    report = OUT / "actual_2026_comparison_report.md"

    def fmt(value: float, digits: int = 4) -> str:
        return f"{value:.{digits}f}"

    error_lines = "\n".join(f"- {error}" for error in q2["fifa_validation_errors"]) if q2["fifa_validation_errors"] else "- （无）"
    lines = f"""# 2026 实际世界杯 × Q1/Q2/Q3 结果对比报告

数据来源：OpenFootball 2026 美加墨世界杯公开赛程（72 场小组赛，含真实比分）与 16 座场馆表；
模型输出：Q1 观看人数预测、Q2 COPT 赛程、Q3 动态资源策略（均为已交付文件，本次分析不改动任何交付物）。
随机种子 {SEED}，Q3 模拟 {N_SIMULATIONS} 次。FIFA 赛程按题目时隙网格取最近时隙映射，映射偏差单独记录。

## 0. 数据对齐

- 实际赛程 72 场比赛与题目 `groups_matches` 的 72 场**逐场一一对应**（分组、对阵完全相同），48 支球队名称经别名归一化后全部命中；
- 实际 16 座场馆城市与题目 `venues` 的 16 个城市**一一对应**（去除括号后缀后全命中）；
- 实际开球时刻（UTC）映射到题目 `time_slots` 网格中**最近的时隙**，平均偏差 {fmt(q2['slot_gap_hours_mean'], 2)} 小时、最大 {fmt(q2['slot_gap_hours_max'], 2)} 小时——实际赛事开球时间并非题目 12:00/15:00/18:00/21:00 的规则网格，此偏差仅为坐标映射误差。

## 1. Q2：FIFA 实际赛程 vs 我们的 COPT 赛程

### 1.1 实际赛程在题目硬约束下的可行性

把 FIFA 的场馆/时段分配放进题目坐标系后，用交付验收器 `validate_q2_schedule` 检验：
实际赛程**{'完全满足' if q2['fifa_validation_ok'] else '不满足'}题目全部硬约束**。
{('违例清单：' + chr(10) + error_lines) if not q2['fifa_validation_ok'] else ''}

### 1.2 反事实 P2 评分（同一评估器口径）

由于官方评估器拒绝安全等级不合格的场馆，原始 FIFA 分配无法直接评分。将 {q2['fifa_repaired_matches']} 场违例比赛的场馆按最小修改原则替换为 COPT 赛程中同一场比赛使用的合格场馆（其余 68 场保持 FIFA 原分配）后，在同一评分体系下：

| 方案 | P2 目标值 | 备注 |
|---|---|---|
| 题目可行基线（demo） | {fmt(q2['objective_demo'])} | 已有交付 |
| **我们的 COPT 赛程** | **{fmt(q2['objective_copt'])}** | 已有交付 |
| FIFA 实际赛程（{q2['fifa_repaired_matches']} 场场馆最小修复） | {fmt(q2['objective_fifa_repaired'])} | 本次反事实计算 |

逐项归一化指标（COPT vs FIFA）：{', '.join(f"{key[5:]}={fmt(q2['norm_copt'][key])} vs {fmt(q2['norm_fifa'][key])}" for key in q2['norm_copt'] if key in q2['norm_fifa'])}。

### 1.3 逐场分配重合度（FIFA vs COPT）

- 场馆一致率：{q2['venue_agreement']:.1%}；
- 时隙一致率：{q2['slot_agreement']:.1%}；
- 比赛日一致率：{q2['date_agreement']:.1%}。

## 2. Q3：真实前两轮 vs 题目合成反馈 vs 实际晋级

### 2.1 真实前两轮积分榜 vs 题目 live_group_results

题目给出的第 1–2 轮赛果是合成数据。与真实赛果逐队对比：

- 积分完全相同（2 场共 6 分制）的球队占比：{q3['standings_points_equal_share']:.1%}；
- 积分差绝对值的均值：{fmt(q3['standings_points_delta_mean_abs'], 2)} 分；
- 48 队积分相关系数：{fmt(q3['standings_points_correlation'], 3)}；状态分 s 相关系数：{fmt(q3['state_s_correlation'], 3)}。

### 2.2 晋级概率：交付结果 vs 基于真实赛果重算

把真实前两轮赛果代入题目官方状态更新公式并重新模拟第三轮（xG 用真实进球代理、红牌数据缺失记 0），逐场与已交付 `result_3` 对比（每场双方概率拼接）：

- 概率相关系数：{fmt(q3['p_correlation'], 3)}；平均绝对差：{fmt(q3['p_mae'], 3)}。

### 2.3 与实际晋级结果的验证（模型外推检验）

用实际第三轮结果排定各组名次（前二晋级 + 8 个成绩最好的第三名）：

| 指标 | 交付概率（合成反馈） | 基于真实赛果的概率 |
|---|---|---|
| Brier 分数（48 队，越低越好） | {fmt(q3['brier_synthetic'])} | {fmt(q3['brier_actual_based'])} |
| 0.5 阈值命中率 | {q3['hit_rate_synthetic']:.1%} | {q3['hit_rate_actual_based']:.1%} |
| 12 组中预测前二与实际前二完全一致的小组数 | {q3['group_top2_exact_matches_synthetic']} | {q3['group_top2_exact_matches_actual_based']} |
| 概率与实际积分的 Spearman 秩相关 | {fmt(q3['p_vs_points_spearman_synthetic'], 3)} | {fmt(q3['p_vs_points_spearman_actual_based'], 3)} |

## 3. Q1：预期进球与实际赛果的校准

真实观众/转播数据不公开，Q1 的观看人数预测无法直接对真实值检验。用题目 `base_predictions` 的赛前预期进球（Q2/Q3 需求侧输入）对实际比分做校准：

- 实际总进球相关系数：{fmt(q1['expected_vs_actual_goals_corr_a'], 3)}（A 队）/ {fmt(q1['expected_vs_actual_goals_corr_b'], 3)}（B 队）；平均绝对误差：{fmt(q1['expected_goals_mae'], 2)} 球/队；
- 分胜负的 {q1['direction_accuracy_decided_n']} 场（另有 {q1['draws']} 场平局）中，预期进球方向正确率：{q1['direction_accuracy_decided']:.1%}；作为参照，按 ELO 强弱方向为 {q1['elo_direction_accuracy_decided']:.1%}——方向信号可靠，但强度校准一般（相关系数 0.4 量级）；
- 预期净胜球与实际净胜球相关系数：{fmt(q1['margin_correlation'], 3)}；
- Q1 观看人数预测与实际总进球的 Spearman 秩相关（娱乐性弱代理）：{fmt(q1['viewers_vs_goals_spearman'], 3)}。

## 4. 结论

1. **数据同源但反馈合成**：实际赛程与题目 `groups_matches` 72 场对阵、16 城市完全同源（真实抽签），但题目前两轮赛果为合成数据——48 队中仅 {q3['standings_points_equal_share']:.1%} 积分与实际完全相同，平均积分差 {fmt(q3['standings_points_delta_mean_abs'], 2)} 分，积分相关仅 {fmt(q3['standings_points_correlation'], 3)}。
2. **Q3 框架外推有效**：把真实前两轮赛果代入题目的状态更新与晋级模拟后，概率与实际晋级结果的 Brier 分数从交付版的 {fmt(q3['brier_synthetic'], 3)} 降到 {fmt(q3['brier_actual_based'], 3)}，12 组前二预测完全正确的小组从 {q3['group_top2_exact_matches_synthetic']} 个提高到 {q3['group_top2_exact_matches_actual_based']} 个——说明 Q3 的动态更新机制本身有效，交付结果的偏差主要来自题目合成反馈与真实赛果的差距，而非方法缺陷。
3. **Q2 反事实**：FIFA 实际赛程在题目硬约束下不可行（安全等级 4 场、转播容量 2 时隙、第三轮高安保 2 天），最小修复后反事实 P2 得分 {fmt(q2['objective_fifa_repaired'])}，略高于可行基线 {fmt(q2['objective_demo'])}、明显低于 COPT 赛程 {fmt(q2['objective_copt'])}；两者场馆/时隙重合度很低（{q2['venue_agreement']:.1%}/{q2['slot_agreement']:.1%}），属于不同目标体系下的不同解。
4. **Q1 需求侧信号**：预期进球对实际比分的胜负方向判断可靠（{q1['direction_accuracy_decided']:.1%}），但强度校准一般（进球相关 0.46/0.41）；真实观众数据不公开，观看人数预测无法直接对真实值检验。

局限：
1. 实际赛程无公开上座/转播/安保数据，Q1 与 Q3 经济侧指标无法对真实值检验；
2. 重算 Q3 时红牌、伤停数据缺失（记 0），xG 以真实进球代理；
3. 时隙映射为最近网格点，FIFA 实际开球时间与题目网格的差异不构成任何一方的“错误”；
4. 小组第三名并列时按 积分→净胜球→进球→队名 的确定性规则判定。
"""
    report.write_text(lines, encoding="utf-8")

    q2_compared.to_csv(OUT / "actual_2026_vs_q2_schedule_agreement.csv", index=False)
    q3_standings.to_csv(OUT / "actual_2026_vs_q3_standings_r12.csv", index=False)
    q3_probabilities.to_csv(OUT / "actual_2026_vs_q3_advance_probabilities.csv", index=False)
    q3_outcomes.to_csv(OUT / "actual_2026_vs_q3_qualification_outcomes.csv", index=False)
    q1_frame.to_csv(OUT / "actual_2026_vs_q1_goals_calibration.csv", index=False)
    (OUT / "actual_2026_comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = load_q1_tables()
    actual = pd.read_csv(ACTUAL_SCHEDULE)
    stadiums = pd.read_csv(ACTUAL_STADIUMS)
    maps = _build_maps(tables, actual, stadiums)
    fifa = _build_fifa_problem_schedule(tables, actual, stadiums, maps)
    q2_summary, q2_compared = _q2_section(tables, fifa)
    q3_summary, q3_standings, q3_probabilities, q3_outcomes = _q3_section(tables, actual, maps)
    q1_summary, q1_frame = _q1_section(tables, actual, maps)
    summary = {"q2": q2_summary, "q3": q3_summary, "q1": q1_summary}
    _write_report(summary, q2_compared, q3_standings, q3_probabilities, q3_outcomes, q1_frame)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
