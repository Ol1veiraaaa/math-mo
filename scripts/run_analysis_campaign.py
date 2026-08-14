from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c_contest_q1.data import load_q1_tables
from c_contest_q1.q3_demo import build_q3_state


P2_KEYS = ["T", "B", "U", "H", "C", "D", "F", "R"]
P2_DIRECTIONS = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
P2_WEIGHTS = np.array([0.25, 0.25, 0.15, 0.10, 0.08, 0.07, 0.06, 0.04], dtype=float)


def q2_weight_robustness(out: Path, n_samples: int = 10_000, seed: int = 20260814) -> dict[str, float]:
    baseline = json.loads((ROOT / "outputs" / "q2" / "demo" / "objective_score.json").read_text(encoding="utf-8"))
    copt = json.loads((ROOT / "outputs" / "q2" / "copt" / "solve_metadata.json").read_text(encoding="utf-8"))["full_p2_score"]
    baseline_values = np.array([baseline[f"norm_{key}"] for key in P2_KEYS], dtype=float)
    copt_values = np.array([copt[f"norm_{key}"] for key in P2_KEYS], dtype=float)
    rng = np.random.default_rng(seed)
    factors = rng.uniform(0.8, 1.2, size=(n_samples, len(P2_KEYS)))
    weights = factors * P2_WEIGHTS
    weights /= weights.sum(axis=1, keepdims=True)
    baseline_scores = (weights * P2_DIRECTIONS * baseline_values).sum(axis=1)
    copt_scores = (weights * P2_DIRECTIONS * copt_values).sum(axis=1)
    differences = copt_scores - baseline_scores
    samples = pd.DataFrame(weights, columns=[f"weight_{key}" for key in P2_KEYS])
    samples["baseline_score"] = baseline_scores
    samples["copt_score"] = copt_scores
    samples["copt_minus_baseline"] = differences
    samples.to_csv(out / "q2_weight_robustness_samples.csv", index=False)
    return {
        "samples": n_samples,
        "copt_win_rate": float((differences > 0).mean()),
        "difference_min": float(differences.min()),
        "difference_p05": float(np.quantile(differences, 0.05)),
        "difference_median": float(np.median(differences)),
        "difference_p95": float(np.quantile(differences, 0.95)),
        "difference_max": float(differences.max()),
    }


def q3_simulation_convergence(out: Path) -> dict[str, float]:
    tables = load_q1_tables()
    q2 = ROOT / "outputs" / "q2" / "copt" / "result_2_group_schedule.csv"
    q1 = ROOT / "outputs" / "q1" / "demo" / "result_1_match_prediction.csv"
    counts = [5_000, 10_000, 20_000, 50_000]
    probability_runs: dict[int, pd.DataFrame] = {}
    summaries: list[dict[str, float | int]] = []
    for count in counts:
        state = build_q3_state(tables, q2, q1, n_simulations=count, seed=20260814)
        frame = state.matches[["match_id", "team_a", "team_b", "updated_p_team_a_advance", "updated_p_team_b_advance", "stakeless_risk", "collusion_risk"]].copy()
        probability_runs[count] = frame
        frame.to_csv(out / f"q3_simulation_{count}.csv", index=False)
    reference = probability_runs[50_000]
    for count in counts[:-1]:
        current = probability_runs[count]
        differences = np.concatenate([
            current.updated_p_team_a_advance.to_numpy() - reference.updated_p_team_a_advance.to_numpy(),
            current.updated_p_team_b_advance.to_numpy() - reference.updated_p_team_b_advance.to_numpy(),
        ])
        risk_difference = current.collusion_risk.to_numpy() - reference.collusion_risk.to_numpy()
        summaries.append({
            "simulations": count,
            "probability_mae_vs_50000": float(np.abs(differences).mean()),
            "probability_max_abs_vs_50000": float(np.abs(differences).max()),
            "collusion_risk_mae_vs_50000": float(np.abs(risk_difference).mean()),
            "qualification_total": float(current[["updated_p_team_a_advance", "updated_p_team_b_advance"]].sum().sum()),
        })
    convergence = pd.DataFrame(summaries)
    convergence.to_csv(out / "q3_simulation_convergence.csv", index=False)
    row_20k = convergence.loc[convergence.simulations.eq(20_000)].iloc[0]
    return {
        "reference_simulations": 50_000,
        "probability_mae_20000_vs_50000": float(row_20k.probability_mae_vs_50000),
        "probability_max_abs_20000_vs_50000": float(row_20k.probability_max_abs_vs_50000),
        "collusion_risk_mae_20000_vs_50000": float(row_20k.collusion_risk_mae_vs_50000),
    }


def q3_resource_comparison(out: Path) -> dict[str, float]:
    static = pd.read_csv(ROOT / "outputs" / "q3" / "copt" / "static_selection.csv")
    dynamic = pd.read_csv(ROOT / "outputs" / "q3" / "copt" / "dynamic_selection.csv")
    result = pd.read_csv(ROOT / "outputs" / "q3" / "copt" / "result_3_dynamic_strategy.csv")
    keys = ["match_id", "broadcast", "security", "transport", "ticket_adjustment"]
    decisions = static[keys].merge(dynamic[keys], on="match_id", suffixes=("_static", "_dynamic"))
    decisions["decision_changed"] = (
        decisions.broadcast_static.ne(decisions.broadcast_dynamic)
        | decisions.security_static.ne(decisions.security_dynamic)
        | decisions.transport_static.ne(decisions.transport_dynamic)
        | ~np.isclose(decisions.ticket_adjustment_static, decisions.ticket_adjustment_dynamic)
    )
    decisions.to_csv(out / "q3_static_dynamic_decisions.csv", index=False)
    limits = _load_daily_limits()
    utilization_rows: list[dict[str, float | str]] = []
    for plan_name, selection in (("Static", static), ("Dynamic", dynamic)):
        daily = selection.groupby("reference_date").agg(
            broadcast_used=("broadcast", lambda values: int(values.eq(3).sum())),
            security_used=("security", lambda values: int(values.ge(3).sum())),
            transport_used=("transport", lambda values: int(values.eq(3).sum())),
            budget_used=("dynamic_resource_cost", "sum"),
        ).reset_index()
        daily = daily.merge(limits, on="reference_date", validate="one_to_one")
        for row in daily.itertuples(index=False):
            utilization_rows.append({
                "reference_date": row.reference_date,
                "plan": plan_name,
                "broadcast_used": row.broadcast_used,
                "broadcast_capacity": row.high_broadcast_capacity,
                "broadcast_utilization": row.broadcast_used / row.high_broadcast_capacity,
                "security_used": row.security_used,
                "security_capacity": row.high_security_capacity,
                "security_utilization": row.security_used / row.high_security_capacity,
                "transport_used": row.transport_used,
                "transport_capacity": row.enhanced_transport_capacity,
                "transport_utilization": row.transport_used / row.enhanced_transport_capacity,
                "budget_used": row.budget_used,
                "budget_capacity": row.daily_resource_budget_index,
                "budget_utilization": row.budget_used / row.daily_resource_budget_index,
            })
    pd.DataFrame(utilization_rows).to_csv(out / "q3_daily_resource_utilization.csv", index=False)

    total_metrics = {
        "attendance": "dynamic_attendance",
        "tv_viewers": "dynamic_tv_viewers",
        "ticket_value_usd": "dynamic_ticket_value",
        "broadcast_value_usd": "dynamic_broadcast_value",
        "resource_cost_index": "dynamic_resource_cost",
        "risk_exposure_index": "dynamic_risk",
        "net_value": "updated_evaluation_score",
    }
    total_rows = []
    for metric, column in total_metrics.items():
        static_value = float(static[column].sum())
        dynamic_value = float(dynamic[column].sum())
        total_rows.append({
            "metric": metric,
            "static_value": static_value,
            "dynamic_value": dynamic_value,
            "absolute_change": dynamic_value - static_value,
            "relative_change": (dynamic_value - static_value) / abs(static_value) if not np.isclose(static_value, 0) else np.nan,
        })
    pd.DataFrame(total_rows).to_csv(out / "q3_static_dynamic_totals.csv", index=False)
    max_row = result.loc[result.improvement_rate.idxmax()]
    min_row = result.loc[result.improvement_rate.idxmin()]
    return {
        "matches_with_changed_decision": int(decisions.decision_changed.sum()),
        "changed_decision_rate": float(decisions.decision_changed.mean()),
        "matches_with_positive_contribution_change": int(result.improvement_rate.gt(0).sum()),
        "matches_with_negative_contribution_change": int(result.improvement_rate.lt(0).sum()),
        "mean_match_improvement_rate": float(result.improvement_rate.mean()),
        "total_static_updated_value": float(result.static_net_value.sum()),
        "total_dynamic_value": float(result.dynamic_net_value.sum()),
        "total_improvement_rate": float((result.dynamic_net_value.sum() - result.static_net_value.sum()) / abs(result.static_net_value.sum())),
        "maximum_improvement_match": str(max_row.match_id),
        "maximum_improvement_rate": float(max_row.improvement_rate),
        "minimum_improvement_match": str(min_row.match_id),
        "minimum_improvement_rate": float(min_row.improvement_rate),
    }


def _load_daily_limits() -> pd.DataFrame:
    tables = load_q1_tables()
    limits = pd.read_excel(tables.paths.workbook, sheet_name="dynamic_resource_limits")
    limits["reference_date"] = limits.reference_date.astype(str)
    return limits[[
        "reference_date",
        "high_broadcast_capacity",
        "high_security_capacity",
        "enhanced_transport_capacity",
        "daily_resource_budget_index",
    ]]


def main() -> None:
    out = ROOT / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "execution_envelope": {
            "processor": "Intel Core i9-14900HX",
            "q3_solver": "COPT 8.0.6 evaluation mode, 2000-variable MIP limit",
            "random_seed": 20260814,
        },
        "q2_weight_robustness": q2_weight_robustness(out),
        "q3_simulation_convergence": q3_simulation_convergence(out),
        "q3_resource_comparison": q3_resource_comparison(out),
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Modeling Analysis Campaign",
        "",
        "Parent claims: the restricted-candidate Q2 COPT schedule improves the fixed feasibility baseline, and Q3 feedback re-optimization improves the static resource plan under the same updated evaluation bounds.",
        "",
        "## Q2 Weight Robustness",
        "",
        f"Across {summary['q2_weight_robustness']['samples']:,} independent +/-20% weight perturbations normalized to sum to one, the COPT schedule win rate was {summary['q2_weight_robustness']['copt_win_rate']:.2%}. The 5th percentile objective advantage was {summary['q2_weight_robustness']['difference_p05']:.6f}.",
        "",
        "## Q3 Monte Carlo Convergence",
        "",
        f"Against 50,000 simulations, the official 20,000-draw run had probability MAE {summary['q3_simulation_convergence']['probability_mae_20000_vs_50000']:.6f} and maximum absolute difference {summary['q3_simulation_convergence']['probability_max_abs_20000_vs_50000']:.6f}.",
        "",
        "## Q3 Static vs Dynamic Resources",
        "",
        f"Feedback changed at least one resource or price decision for {summary['q3_resource_comparison']['matches_with_changed_decision']} of 24 matches. The globally constrained dynamic objective improved by {summary['q3_resource_comparison']['total_improvement_rate']:.3%} over the fixed static decisions evaluated in the same updated environment.",
        "",
        "Comparability: Q2 changes weights only; Q3 convergence changes simulation count only; static/dynamic Q3 uses one candidate set, identical capacities and budgets, and common updated-environment normalization bounds.",
        "",
        "Next route: use these results in the paper and proceed to Q4 external schedule comparison.",
    ]
    (out / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
