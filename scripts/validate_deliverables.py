from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c_contest_q1.data import load_q1_tables
from c_contest_q1.outputs import validate_demo_outputs
from c_contest_q1.paths import discover_template_path
from c_contest_q1.q2_demo import validate_q2_schedule
from c_contest_q1.q3_demo import validate_q3_result
from c_contest_q1.q4_demo import validate_actual_schedule


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _official_templates() -> dict[str, Path]:
    expected = {
        "actual_schedule_template.csv",
        "result_1_match_prediction_template.csv",
        "result_1_test_prediction_template.csv",
        "result_2_template.csv",
        "result_3_template.csv",
        "result_4_template.csv",
    }
    return {name: discover_template_path(name) for name in expected}


def _schema_check(name: str, output: Path, template: Path) -> Check:
    actual_columns = list(pd.read_csv(output, nrows=0).columns)
    expected_columns = list(pd.read_csv(template, nrows=0).columns)
    return Check(
        name=f"{name}_schema",
        ok=actual_columns == expected_columns,
        detail=f"columns={len(actual_columns)}, exact_order={actual_columns == expected_columns}",
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_all() -> tuple[list[Check], dict[str, str]]:
    tables = load_q1_tables()
    templates = _official_templates()
    paths = {
        "q1_test": ROOT / "outputs/q1/demo/result_1_test_prediction.csv",
        "q1_match": ROOT / "outputs/q1/demo/result_1_match_prediction.csv",
        "q2": ROOT / "outputs/q2/copt/result_2_group_schedule.csv",
        "q3": ROOT / "outputs/q3/copt/result_3_dynamic_strategy.csv",
        "actual": ROOT / "outputs/q4/actual_schedule.csv",
        "q4": ROOT / "outputs/q4/demo/result_4_schedule_comparison.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required deliverables not found: {missing}")

    checks: list[Check] = []
    schema_pairs = {
        "q1_test": "result_1_test_prediction_template.csv",
        "q1_match": "result_1_match_prediction_template.csv",
        "q2": "result_2_template.csv",
        "q3": "result_3_template.csv",
        "actual": "actual_schedule_template.csv",
        "q4": "result_4_template.csv",
    }
    checks.extend(
        _schema_check(name, paths[name], templates[template])
        for name, template in schema_pairs.items()
    )

    q1_report = validate_demo_outputs(tables, paths["q1_test"], paths["q1_match"])
    checks.append(Check("q1_content_and_units", q1_report.ok, "; ".join(q1_report.errors) or "140 test rows and 72 match rows validated"))

    q2 = pd.read_csv(paths["q2"])
    q2_metadata = json.loads((ROOT / "outputs/q2/copt/solve_metadata.json").read_text(encoding="utf-8"))
    q2_report = validate_q2_schedule(tables, q2)
    objective_locations = np.flatnonzero(q2.total_objective_value.notna().to_numpy()).tolist()
    q2_objective_ok = objective_locations == [0] and np.isfinite(q2.loc[0, "total_objective_value"])
    checks.append(Check("q2_six_hard_constraint_groups", q2_report.ok, "; ".join(q2_report.errors) or "all independent assertions passed"))
    checks.append(Check("q2_objective_first_row_only", bool(q2_objective_ok), f"nonempty_rows={objective_locations}"))
    q2_reconciled = np.isclose(
        float(q2_metadata["mip_objective"]),
        float(q2_metadata["full_p2_score"]["objective"]),
        atol=1e-8,
    )
    q2_evidence = q2_metadata["solve_evidence"]
    q2_log = (ROOT / "outputs/q2/copt/solver.log").read_text(encoding="utf-8", errors="replace")
    q2_solver_ok = (
        q2_evidence["rows"] == 865
        and q2_evidence["columns"] == 844
        and q2_evidence["binary_variables"] == 840
        and np.isclose(q2_evidence["best_bound"], q2_metadata["mip_objective"], atol=1e-10)
        and np.isclose(q2_evidence["relative_gap"], 0.0, atol=1e-12)
        and "865 rows, 844 columns" in q2_log
        and "840 binaries" in q2_log
        and "Best solution   : 0.458163650" in q2_log
        and "Best gap        : 0.0000%" in q2_log
        and "Solution status : integer optimal" in q2_log
        and "integrality :            0" in q2_log
    )
    checks.append(Check(
        "q2_copt_objective_reconciliation",
        bool(q2_reconciled and q2_solver_ok),
        f"copt={q2_metadata['mip_objective']:.12f}, evaluator={q2_metadata['full_p2_score']['objective']:.12f}, rows/cols/bins=865/844/840, gap={q2_evidence['relative_gap']:.1f}",
    ))

    q3 = pd.read_csv(paths["q3"])
    q3_report = validate_q3_result(tables, paths["q2"], q3)
    q3_metadata = json.loads((ROOT / "outputs/q3/copt/solve_metadata.json").read_text(encoding="utf-8"))
    q3_log = (ROOT / "outputs/q3/copt/solver.log").read_text(encoding="utf-8", errors="replace")
    q3_solver_ok = all(
        evidence["rows"] == 48
        and evidence["columns"] == 1623
        and evidence["binary_variables"] == 1623
        and np.isclose(evidence["relative_gap"], 0.0, atol=1e-12)
        for evidence in q3_metadata["solve_evidence"].values()
    ) and (
        np.isclose(q3_metadata["solve_evidence"]["static"]["best_bound"], q3_metadata["static_optimization_objective"], atol=1e-10)
        and np.isclose(q3_metadata["solve_evidence"]["dynamic"]["best_bound"], q3_metadata["dynamic_optimization_objective"], atol=1e-10)
        and np.isclose(q3_metadata["solve_evidence"]["feasible_lower_bound"]["best_bound"], q3_metadata["feasible_objective_lower_bound"], atol=1e-10)
        and q3_metadata["feasible_objective_lower_bound"] <= q3_metadata["static_updated_evaluation"] <= q3_metadata["feasible_objective_upper_bound"]
        and "=== Q3 STATIC COPT MODEL ===" in q3_log
        and "=== Q3 DYNAMIC COPT MODEL ===" in q3_log
        and "=== Q3 FEASIBLE OBJECTIVE LOWER-BOUND COPT MODEL ===" in q3_log
        and q3_log.count("48 rows, 1623 columns") == 3
        and q3_log.count("1623 binaries") >= 3
        and q3_log.count("Best gap        : 0.0000%") == 3
        and q3_log.count("Solution status : integer optimal") == 3
        and q3_log.count("integrality :            0") == 3
    )
    q3_detail = "; ".join(q3_report.errors) or "24 matches, probability total 32, all capacity checks passed"
    checks.append(Check("q3_schedule_and_resource_constraints", bool(q3_report.ok and q3_solver_ok), q3_detail + "; three 48x1623 COPT models at zero gap, including the feasible Z3 lower bound"))

    actual = pd.read_csv(paths["actual"])
    actual_errors = list(validate_actual_schedule(actual))
    teams = pd.concat([actual.team_a, actual.team_b])
    appearances = pd.concat([
        actual[["team_a", "round_in_group"]].rename(columns={"team_a": "team"}),
        actual[["team_b", "round_in_group"]].rename(columns={"team_b": "team"}),
    ])
    actual_structure_ok = (
        len(actual) == 72
        and actual.group_id.nunique() == 12
        and teams.nunique() == 48
        and appearances.groupby("team").size().eq(3).all()
        and actual.groupby("group_id").size().eq(6).all()
    )
    checks.append(Check("q4_actual_source_contract", not actual_errors, "; ".join(actual_errors) or "row-level URL and retrieval date present"))
    checks.append(Check("q4_actual_tournament_structure", bool(actual_structure_ok), f"matches={len(actual)}, groups={actual.group_id.nunique()}, teams={teams.nunique()}"))

    q4 = pd.read_csv(paths["q4"])
    q4_numeric = q4[["actual_schedule_value", "optimized_schedule_value", "absolute_difference"]]
    q4_ok = len(q4) >= 10 and np.isfinite(q4_numeric.to_numpy()).all()
    checks.append(Check("q4_comparison_content", bool(q4_ok), f"indicators={len(q4)}, finite_core_values={bool(np.isfinite(q4_numeric.to_numpy()).all())}"))

    hashes = {name: _hash(path) for name, path in paths.items()}
    return checks, hashes


def main() -> None:
    checks, hashes = validate_all()
    report = {
        "ok": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
        "sha256": hashes,
    }
    out = ROOT / "outputs/validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "deliverable_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for check in checks:
        print(f"[{'PASS' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    if not report["ok"]:
        raise SystemExit(1)
    print(f"Validated {len(checks)} deliverable contracts; report={out / 'deliverable_validation.json'}")


if __name__ == "__main__":
    main()
