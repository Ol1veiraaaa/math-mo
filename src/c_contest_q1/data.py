"""Read-only Q1 data loading and source-contract validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .paths import SourcePaths, discover_source_paths


@dataclass(frozen=True)
class Q1Tables:
    paths: SourcePaths
    historical: pd.DataFrame
    teams: pd.DataFrame
    group_matches: pd.DataFrame
    base_predictions: pd.DataFrame
    test_template: pd.DataFrame
    match_template: pd.DataFrame


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _read_workbook(paths: SourcePaths, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(paths.workbook, sheet_name=sheet_name, engine="openpyxl")


def load_q1_tables(paths: SourcePaths | None = None) -> Q1Tables:
    """Load the immutable Q1 inputs and normalize only parsing types."""

    paths = paths or discover_source_paths()
    historical = _read_workbook(paths, "historical_matches")
    historical["date"] = pd.to_datetime(historical["date"], errors="raise")
    historical["dataset_split"] = historical["dataset_split"].astype("string").str.strip()
    return Q1Tables(
        paths=paths,
        historical=historical,
        teams=_read_workbook(paths, "teams"),
        group_matches=_read_workbook(paths, "groups_matches"),
        base_predictions=_read_workbook(paths, "base_predictions"),
        test_template=pd.read_csv(paths.test_template),
        match_template=pd.read_csv(paths.match_template),
    )


def validate_q1_tables(tables: Q1Tables) -> ValidationReport:
    """Validate contest invariants before any model can consume the inputs."""

    errors: list[str] = []
    history = tables.historical
    train = history.loc[history["dataset_split"].eq("train")]
    test = history.loc[history["dataset_split"].eq("test")]

    expected_shapes = {
        "historical": (700, 25),
        "teams": (48, 16),
        "group_matches": (72, 7),
        "base_predictions": (72, 14),
    }
    actual_shapes = {
        "historical": history.shape,
        "teams": tables.teams.shape,
        "group_matches": tables.group_matches.shape,
        "base_predictions": tables.base_predictions.shape,
    }
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            errors.append(f"{name} shape {actual_shapes[name]} != {expected}")

    for name, frame, column in (
        ("historical", history, "match_id"),
        ("teams", tables.teams, "team_id"),
        ("group_matches", tables.group_matches, "match_id"),
    ):
        if not frame[column].is_unique:
            errors.append(f"{name}.{column} is not unique")

    if len(train) != 560 or len(test) != 140:
        errors.append(f"split rows train={len(train)}, test={len(test)}")
    if not train["tv_viewers"].notna().all():
        errors.append("train tv_viewers contains missing labels")
    if not test["tv_viewers"].isna().all():
        errors.append("test tv_viewers must be hidden")
    if train["date"].min() != pd.Timestamp("2018-01-12"):
        errors.append("unexpected train start date")
    if train["date"].max() != pd.Timestamp("2024-07-27"):
        errors.append("unexpected train end date")
    if test["date"].min() != pd.Timestamp("2024-08-05"):
        errors.append("unexpected test start date")
    if test["date"].max() != pd.Timestamp("2025-12-28"):
        errors.append("unexpected test end date")
    if set(tables.group_matches["match_id"]) != set(tables.base_predictions["match_id"]):
        errors.append("group_matches and base_predictions match IDs differ")
    if list(tables.test_template.columns) != ["match_id_test", "predicted_test_tv_viewers"]:
        errors.append("unexpected test template columns")
    if list(tables.match_template.columns) != [
        "match_id",
        "team_a",
        "team_b",
        "predicted_tv_viewers",
    ]:
        errors.append("unexpected match template columns")
    return ValidationReport(tuple(errors))

