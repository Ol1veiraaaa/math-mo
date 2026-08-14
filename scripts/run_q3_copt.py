from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c_contest_q1.data import load_q1_tables
from c_contest_q1.q3_copt import solve_copt_q3
from c_contest_q1.q3_demo import validate_q3_result


def main() -> None:
    out = ROOT / "outputs" / "q3" / "copt"
    out.mkdir(parents=True, exist_ok=True)
    artifacts = solve_copt_q3(
        load_q1_tables(),
        ROOT / "outputs" / "q2" / "copt" / "result_2_group_schedule.csv",
        ROOT / "outputs" / "q1" / "demo" / "result_1_match_prediction.csv",
        log_dir=out,
    )
    validation = validate_q3_result(load_q1_tables(), ROOT / "outputs" / "q2" / "copt" / "result_2_group_schedule.csv", artifacts.result)
    if not validation.ok:
        raise RuntimeError("Q3 validation failed: " + "; ".join(validation.errors))
    artifacts.result.to_csv(out / "result_3_dynamic_strategy.csv", index=False)
    artifacts.simulation_summary.to_csv(out / "simulation_summary.csv", index=False)
    artifacts.static_selection.to_csv(out / "static_selection.csv", index=False)
    artifacts.dynamic_selection.to_csv(out / "dynamic_selection.csv", index=False)
    artifacts.lower_bound_selection.to_csv(out / "feasible_lower_bound_selection.csv", index=False)
    metadata = {
        "candidate_count": artifacts.candidate_count,
        "static_status": artifacts.static_status,
        "dynamic_status": artifacts.dynamic_status,
        "lower_bound_status": artifacts.lower_bound_status,
        "static_optimization_objective": artifacts.static_optimization_objective,
        "static_updated_evaluation": artifacts.static_updated_evaluation,
        "dynamic_optimization_objective": artifacts.dynamic_optimization_objective,
        "feasible_objective_lower_bound": artifacts.feasible_objective_lower_bound,
        "feasible_objective_upper_bound": artifacts.dynamic_optimization_objective,
        "solver": "COPT 8.0.6",
        "license_mode": "non-commercial evaluation, 2000-variable/constraint limit",
        "solve_evidence": artifacts.solve_evidence,
        "total_improvement_rate": (
            artifacts.dynamic_optimization_objective - artifacts.static_updated_evaluation
        ) / abs(artifacts.static_updated_evaluation),
    }
    (out / "solve_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(artifacts.result.head().to_string(index=False))
    print(metadata)


if __name__ == "__main__":
    main()
