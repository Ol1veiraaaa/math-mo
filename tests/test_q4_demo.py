from __future__ import annotations

import pandas as pd

from c_contest_q1.data import load_q1_tables
from c_contest_q1.q4_demo import compare_structural_schedule, validate_actual_schedule


def test_q4_actual_schedule_and_comparison_contract() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    actual_path = root / "outputs" / "q4" / "actual_schedule.csv"
    actual = pd.read_csv(actual_path)
    assert not validate_actual_schedule(actual)
    assert len(actual) == 72
    assert pd.concat([actual.team_a, actual.team_b]).value_counts().eq(3).all()
    comparison = compare_structural_schedule(
        load_q1_tables(),
        actual_path,
        root / "outputs" / "q2" / "copt" / "result_2_group_schedule.csv",
    )
    assert len(comparison) == 13
    assert comparison.indicator_name.is_unique
    assert comparison[["actual_schedule_value", "optimized_schedule_value"]].notna().all().all()
