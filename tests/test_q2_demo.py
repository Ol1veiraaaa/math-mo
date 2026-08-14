from __future__ import annotations

import numpy as np

from c_contest_q1.data import load_q1_tables
from c_contest_q1.q2_demo import build_q2_demo_schedule, evaluate_q2_schedule, validate_q2_schedule


def test_q2_schedule_is_feasible_and_has_a_finite_official_score() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    tables = load_q1_tables()
    schedule = build_q2_demo_schedule(tables, root / "outputs" / "q1" / "demo" / "result_1_match_prediction.csv")
    validation = validate_q2_schedule(tables, schedule)
    scored, report = evaluate_q2_schedule(tables, schedule)

    assert validation.ok, validation.errors
    assert len(scored) == 72
    assert scored.travel_cost_index.notna().all()
    assert np.isfinite(report["objective"])
    assert scored.total_objective_value.notna().sum() == 1
