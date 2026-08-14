from __future__ import annotations

import numpy as np

from c_contest_q1.data import load_q1_tables
from c_contest_q1.q3_copt import build_q3_candidates
from c_contest_q1.q3_demo import build_q3_state


def test_q3_simulation_and_candidate_contract() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    tables = load_q1_tables()
    artifacts = build_q3_state(
        tables,
        root / "outputs" / "q2" / "copt" / "result_2_group_schedule.csv",
        root / "outputs" / "q1" / "demo" / "result_1_match_prediction.csv",
        n_simulations=100,
    )
    candidates, bounds = build_q3_candidates(tables, artifacts.matches)

    assert len(artifacts.matches) == 24
    assert np.isclose(artifacts.matches[["updated_p_team_a_advance", "updated_p_team_b_advance"]].sum().sum(), 32.0)
    assert artifacts.simulation_summary[["lambda_a", "lambda_b"]].ge(0.15).all().all()
    assert artifacts.simulation_summary[["lambda_a", "lambda_b"]].le(4.50).all().all()
    q3_scores = artifacts.simulation_summary[["importance", "updated_attractiveness"]]
    assert q3_scores.ge(0).all().all()
    assert q3_scores.le(1).all().all()
    assert artifacts.matches.collusion_risk.between(0, 1).all()
    assert candidates.groupby("match_id").size().size == 24
    assert set(bounds) == {"ticket_value", "broadcast_value", "attractiveness", "resource_cost", "risk"}
