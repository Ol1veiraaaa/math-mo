from __future__ import annotations

import numpy as np
import pandas as pd

from c_contest_q1.data import load_q1_tables
from c_contest_q1.outputs import build_demo_outputs, validate_demo_outputs


def test_demo_outputs_follow_official_templates_and_units(tmp_path) -> None:
    tables = load_q1_tables()
    artifacts = build_demo_outputs(tables, tmp_path)
    report = validate_demo_outputs(tables, artifacts.test_csv, artifacts.match_csv)
    test_prediction = pd.read_csv(artifacts.test_csv)
    match_prediction = pd.read_csv(artifacts.match_csv)

    assert report.ok, report.errors
    assert list(test_prediction.columns) == list(tables.test_template.columns)
    assert list(match_prediction.columns) == list(tables.match_template.columns)
    assert len(test_prediction) == 140
    assert len(match_prediction) == 72
    assert np.isfinite(test_prediction["predicted_test_tv_viewers"]).all()
    assert np.isfinite(match_prediction["predicted_tv_viewers"]).all()
    assert test_prediction["predicted_test_tv_viewers"].between(50, 300).all()
    assert match_prediction["predicted_tv_viewers"].between(50_000_000, 300_000_000).all()

