"""Formal COPT static/dynamic resource optimization for Question 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Q1Tables
from .q3_demo import OUTPUT_COLUMNS, build_q3_state


@dataclass(frozen=True)
class CoptQ3Artifacts:
    result: pd.DataFrame
    simulation_summary: pd.DataFrame
    static_selection: pd.DataFrame
    dynamic_selection: pd.DataFrame
    lower_bound_selection: pd.DataFrame
    candidate_count: int
    static_status: int
    dynamic_status: int
    lower_bound_status: int
    static_optimization_objective: float
    static_updated_evaluation: float
    dynamic_optimization_objective: float
    feasible_objective_lower_bound: float
    solve_evidence: dict[str, dict[str, float | int]]


def _table(tables: Q1Tables, name: str) -> pd.DataFrame:
    return pd.read_excel(tables.paths.workbook, sheet_name=name)


def _normalization_bounds(candidates: pd.DataFrame, prefix: str) -> dict[str, tuple[float, float]]:
    return {
        metric: (float(candidates[f"{prefix}_{metric}"].min()), float(candidates[f"{prefix}_{metric}"].max()))
        for metric in ("ticket_value", "broadcast_value", "attractiveness", "resource_cost", "risk")
    }


def _normalize(values: pd.Series, bounds: tuple[float, float]) -> pd.Series:
    lower, upper = bounds
    if np.isclose(lower, upper):
        return pd.Series(0.0, index=values.index)
    return (values - lower) / (upper - lower)


def _add_scores(candidates: pd.DataFrame, prefix: str, bounds: dict[str, tuple[float, float]], score_name: str) -> None:
    normalized: dict[str, pd.Series] = {}
    for metric in bounds:
        normalized[metric] = _normalize(candidates[f"{prefix}_{metric}"], bounds[metric])
    candidates[score_name] = (
        0.35 * normalized["ticket_value"]
        + 0.35 * normalized["broadcast_value"]
        + 0.10 * normalized["attractiveness"]
        - 0.10 * normalized["resource_cost"]
        - 0.10 * normalized["risk"]
    )


def _copt_log_name(path: Path) -> str:
    """Use an ASCII relative name because COPT 8.0.6 cannot open Chinese Windows paths."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def build_q3_candidates(tables: Q1Tables, matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Enumerate a common configuration set and evaluate it in both information environments."""

    costs = _table(tables, "dynamic_resource_costs").set_index(["resource_type", "resource_level"])
    limits = _table(tables, "dynamic_resource_limits").copy()
    limits["reference_date"] = limits.reference_date.astype(str)
    rows: list[dict[str, float | int | str]] = []
    for match in matches.itertuples(index=False):
        limit = limits.loc[limits.reference_date.eq(str(match.reference_date))].iloc[0]
        price_grid = np.linspace(-float(limit.max_ticket_discount_rate), float(limit.max_ticket_increase_rate), 5)
        for broadcast in (1, 2, 3):
            for security in range(int(match.required_security_level), int(match.venue_security_level) + 1):
                for transport in (1, 2, 3):
                    for delta in price_grid:
                        multipliers = {
                            "broadcast": float(costs.loc[("broadcast", broadcast), "demand_multiplier"]),
                            "transport": float(costs.loc[("transport", transport), "demand_multiplier"]),
                            "risk": float(costs.loc[("security", security), "risk_multiplier"]),
                        }
                        resource_cost = sum(
                            float(costs.loc[(kind, level), "unit_cost_index"])
                            for kind, level in (("broadcast", broadcast), ("security", security), ("transport", transport))
                        )
                        row: dict[str, float | int | str] = {
                            "match_id": match.match_id,
                            "reference_date": str(match.reference_date),
                            "broadcast": broadcast,
                            "security": security,
                            "transport": transport,
                            "ticket_adjustment": float(delta),
                        }
                        valid = True
                        for prefix in ("static", "dynamic"):
                            attendance_demand = float(getattr(match, f"{prefix}_attendance_demand"))
                            no_price_attendance = min(float(match.venue_capacity), attendance_demand * multipliers["transport"])
                            attendance = min(
                                float(match.venue_capacity),
                                attendance_demand * multipliers["transport"] * (1 + float(delta)) ** float(match.price_elasticity),
                            )
                            if attendance + 1e-9 < 0.88 * no_price_attendance:
                                valid = False
                                break
                            viewers = float(getattr(match, f"{prefix}_tv_demand")) * multipliers["broadcast"]
                            stake = float(getattr(match, f"{prefix}_stakeless_risk"))
                            collusion = float(getattr(match, f"{prefix}_collusion_risk"))
                            security_demand = float(getattr(match, f"{prefix}_security_demand"))
                            row.update({
                                f"{prefix}_attendance": attendance,
                                f"{prefix}_tv_viewers": viewers,
                                f"{prefix}_ticket_value": float(match.base_ticket_price_usd) * (1 + float(delta)) * attendance,
                                f"{prefix}_broadcast_value": float(match.broadcast_unit_value_usd) * viewers,
                                f"{prefix}_attractiveness": float(getattr(match, f"{prefix}_attractiveness")),
                                f"{prefix}_resource_cost": resource_cost,
                                f"{prefix}_risk": 0.40 * stake + 0.40 * collusion + 0.20 * security_demand * multipliers["risk"],
                            })
                        if valid:
                            rows.append(row)
    candidates = pd.DataFrame(rows)
    if candidates.empty or candidates.groupby("match_id").size().size != 24:
        raise ValueError("common Q3 candidate set is incomplete")
    dynamic_bounds = _normalization_bounds(candidates, "dynamic")
    static_bounds = _normalization_bounds(candidates, "static")
    _add_scores(candidates, "static", static_bounds, "static_optimization_score")
    _add_scores(candidates, "dynamic", dynamic_bounds, "updated_evaluation_score")
    return candidates, dynamic_bounds


def _solve_selection(
    candidates: pd.DataFrame,
    limits: pd.DataFrame,
    score_column: str,
    model_name: str,
    log_path: Path | None = None,
    maximize: bool = True,
) -> tuple[pd.DataFrame, int, float, dict[str, float | int]]:
    try:
        import coptpy as cp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("coptpy is required for Q3 optimization") from exc

    environment = cp.Envr()
    model = environment.createModel(model_name)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.unlink(missing_ok=True)
        model.setLogFile(_copt_log_name(log_path))
    decision = {index: model.addVar(vtype=cp.COPT.BINARY, name=f"x_{index}") for index in candidates.index}
    for match_id, block in candidates.groupby("match_id"):
        model.addConstr(cp.quicksum(decision[index] for index in block.index) == 1, name=f"assign_{match_id}")
    for date, block in candidates.groupby("reference_date"):
        limit = limits.loc[limits.reference_date.eq(str(date))].iloc[0]
        model.addConstr(cp.quicksum(decision[index] for index in block.index if candidates.loc[index, "broadcast"] == 3) <= int(limit.high_broadcast_capacity), name=f"broadcast_{date}")
        model.addConstr(cp.quicksum(decision[index] for index in block.index if candidates.loc[index, "security"] >= 3) <= int(limit.high_security_capacity), name=f"security_{date}")
        model.addConstr(cp.quicksum(decision[index] for index in block.index if candidates.loc[index, "transport"] == 3) <= int(limit.enhanced_transport_capacity), name=f"transport_{date}")
        model.addConstr(cp.quicksum(float(candidates.loc[index, "dynamic_resource_cost"]) * decision[index] for index in block.index) <= float(limit.daily_resource_budget_index), name=f"budget_{date}")
    objective = cp.quicksum(
        float(candidates.loc[index, score_column]) * decision[index]
        for index in candidates.index
    )
    model.setObjective(objective, cp.COPT.MAXIMIZE if maximize else cp.COPT.MINIMIZE)
    model.solve()
    if not getattr(model, "hasmipsol", False) and not getattr(model, "hasmipSol", False):
        raise RuntimeError(f"COPT returned no solution for {model_name}; status={model.status}")
    chosen = candidates.loc[[index for index, variable in decision.items() if variable.x > 0.5]].copy()
    if len(chosen) != 24:
        raise RuntimeError(f"{model_name} selected {len(chosen)} rather than 24 configurations")
    evidence = {
        "rows": int(model.getAttr(cp.COPT.Attr.Rows)),
        "columns": int(model.getAttr(cp.COPT.Attr.Cols)),
        "binary_variables": int(model.getAttr(cp.COPT.Attr.Bins)),
        "best_bound": float(model.getAttr(cp.COPT.Attr.BestObj)),
        "relative_gap": float(model.getAttr(cp.COPT.Attr.BestGap)),
        "solve_time_seconds": float(model.getAttr(cp.COPT.Attr.SolvingTime)),
    }
    return chosen.sort_values("match_id").reset_index(drop=True), int(model.status), float(model.objval), evidence


def solve_copt_q3(
    tables: Q1Tables,
    q2_csv: Path,
    q1_csv: Path,
    n_simulations: int = 20_000,
    seed: int = 20260814,
    log_dir: Path | None = None,
) -> CoptQ3Artifacts:
    """Optimize static and dynamic plans, then evaluate both under the updated environment."""

    state = build_q3_state(tables, q2_csv, q1_csv, n_simulations=n_simulations, seed=seed)
    candidates, _ = build_q3_candidates(tables, state.matches)
    limits = _table(tables, "dynamic_resource_limits").copy()
    limits["reference_date"] = limits.reference_date.astype(str)
    static_log = log_dir / "static_solver.log" if log_dir is not None else None
    dynamic_log = log_dir / "dynamic_solver.log" if log_dir is not None else None
    lower_bound_log = log_dir / "lower_bound_solver.log" if log_dir is not None else None
    static, static_status, static_objective, static_evidence = _solve_selection(
        candidates, limits, "static_optimization_score", "q3_static_resource_mip", static_log
    )
    dynamic, dynamic_status, dynamic_objective, dynamic_evidence = _solve_selection(
        candidates, limits, "updated_evaluation_score", "q3_dynamic_resource_mip", dynamic_log
    )
    lower_bound, lower_bound_status, lower_bound_objective, lower_bound_evidence = _solve_selection(
        candidates,
        limits,
        "updated_evaluation_score",
        "q3_feasible_objective_lower_bound_mip",
        lower_bound_log,
        maximize=False,
    )
    if log_dir is not None:
        combined = (
            "=== Q3 STATIC COPT MODEL ===\n"
            + static_log.read_text(encoding="utf-8", errors="replace")
            + "\n=== Q3 DYNAMIC COPT MODEL ===\n"
            + dynamic_log.read_text(encoding="utf-8", errors="replace")
            + "\n=== Q3 FEASIBLE OBJECTIVE LOWER-BOUND COPT MODEL ===\n"
            + lower_bound_log.read_text(encoding="utf-8", errors="replace")
        )
        (log_dir / "solver.log").write_text(combined, encoding="utf-8")

    static_eval = static[["match_id", "updated_evaluation_score"]].rename(columns={"updated_evaluation_score": "static_net_value"})
    dynamic_output = dynamic.rename(columns={
        "broadcast": "recommended_broadcast_priority",
        "security": "recommended_security_level",
        "transport": "recommended_transport_level",
        "ticket_adjustment": "recommended_ticket_adjustment",
        "dynamic_attendance": "updated_expected_attendance",
        "dynamic_tv_viewers": "updated_expected_tv_viewers",
        "dynamic_ticket_value": "updated_ticket_revenue_usd",
        "dynamic_broadcast_value": "updated_broadcast_value_usd",
        "dynamic_resource_cost": "resource_cost_index",
        "dynamic_risk": "risk_exposure_index",
        "updated_evaluation_score": "dynamic_net_value",
    })
    state_columns = [
        "match_id", "group_id", "team_a", "team_b",
        "updated_p_team_a_advance", "updated_p_team_b_advance",
        "stakeless_risk", "collusion_risk", "updated_attractiveness",
    ]
    result = state.matches[state_columns].merge(dynamic_output[[
        "match_id", "recommended_broadcast_priority", "recommended_security_level",
        "recommended_transport_level", "recommended_ticket_adjustment",
        "updated_expected_attendance", "updated_expected_tv_viewers",
        "updated_ticket_revenue_usd", "updated_broadcast_value_usd",
        "resource_cost_index", "risk_exposure_index", "dynamic_net_value",
    ]], on="match_id").merge(static_eval, on="match_id")
    result["improvement_rate"] = (result.dynamic_net_value - result.static_net_value) / result.static_net_value.abs().replace(0, np.nan)
    result = result[OUTPUT_COLUMNS].sort_values("match_id").reset_index(drop=True)
    return CoptQ3Artifacts(
        result=result,
        simulation_summary=state.simulation_summary,
        static_selection=static,
        dynamic_selection=dynamic,
        lower_bound_selection=lower_bound,
        candidate_count=len(candidates),
        static_status=static_status,
        dynamic_status=dynamic_status,
        lower_bound_status=lower_bound_status,
        static_optimization_objective=static_objective,
        static_updated_evaluation=float(static.updated_evaluation_score.sum()),
        dynamic_optimization_objective=dynamic_objective,
        feasible_objective_lower_bound=lower_bound_objective,
        solve_evidence={
            "static": static_evidence,
            "dynamic": dynamic_evidence,
            "feasible_lower_bound": lower_bound_evidence,
        },
    )
