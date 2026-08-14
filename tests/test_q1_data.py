from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c_contest_q1.data import load_q1_tables, validate_q1_tables
from c_contest_q1.paths import discover_source_paths


def test_discovers_official_workbook_and_templates() -> None:
    paths = discover_source_paths()

    assert paths.workbook.is_file()
    assert not paths.workbook.name.startswith("~$")
    assert paths.test_template.name == "result_1_test_prediction_template.csv"
    assert paths.match_template.name == "result_1_match_prediction_template.csv"


def test_loads_fixed_q1_row_contract() -> None:
    tables = load_q1_tables()

    assert tables.historical.shape == (700, 25)
    assert tables.teams.shape == (48, 16)
    assert tables.group_matches.shape == (72, 7)
    assert tables.base_predictions.shape == (72, 14)
    assert tables.historical["match_id"].is_unique
    assert tables.teams["team_id"].is_unique
    assert tables.group_matches["match_id"].is_unique


def test_preserves_official_split_and_target_units() -> None:
    tables = load_q1_tables()
    history = tables.historical
    train = history.loc[history["dataset_split"].eq("train")]
    test = history.loc[history["dataset_split"].eq("test")]

    assert len(train) == 560
    assert len(test) == 140
    assert train["tv_viewers"].notna().all()
    assert test["tv_viewers"].isna().all()
    assert pd.Timestamp(train["date"].min()) == pd.Timestamp("2018-01-12")
    assert pd.Timestamp(train["date"].max()) == pd.Timestamp("2024-07-27")
    assert pd.Timestamp(test["date"].min()) == pd.Timestamp("2024-08-05")
    assert pd.Timestamp(test["date"].max()) == pd.Timestamp("2025-12-28")
    assert 90_000_000 < train["tv_viewers"].min() < 100_000_000
    assert 220_000_000 < train["tv_viewers"].max() < 230_000_000
    target_millions = train["tv_viewers"].to_numpy() / 1_000_000
    assert np.isclose(target_millions.mean(), 145.64310036428572)


def test_validates_cross_table_keys_and_template_columns() -> None:
    tables = load_q1_tables()
    report = validate_q1_tables(tables)

    assert report.ok, report.errors
    assert set(tables.group_matches["match_id"]) == set(tables.base_predictions["match_id"])
    assert list(tables.test_template.columns) == [
        "match_id_test",
        "predicted_test_tv_viewers",
    ]
    assert list(tables.match_template.columns) == [
        "match_id",
        "team_a",
        "team_b",
        "predicted_tv_viewers",
    ]


def test_source_files_are_external_or_in_the_exact_bundle_directory() -> None:
    paths = discover_source_paths()
    project_root = Path(__file__).resolve().parents[1]
    bundled_root = (project_root / "data" / "official").resolve()

    for source in (paths.workbook, paths.test_template, paths.match_template):
        resolved = source.resolve()
        assert resolved.parent == bundled_root or project_root not in resolved.parents
