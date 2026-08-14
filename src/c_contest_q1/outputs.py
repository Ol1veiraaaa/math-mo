"""Full-train Q1 Demo prediction files and independent output validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Q1Tables
from .evaluation import _column_groups
from .features import build_future_features, build_historical_features, target_in_millions
from .models import demo_model_specs


@dataclass(frozen=True)
class OutputArtifacts:
    test_csv: Path
    match_csv: Path


@dataclass(frozen=True)
class OutputValidationReport:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _fit_demo_ridge(tables: Q1Tables):
    historical = build_historical_features(tables)
    future = build_future_features(tables)
    train_mask = historical.metadata["dataset_split"].eq("train")
    numeric, categorical = _column_groups(historical.features)
    ridge_spec = next(spec for spec in demo_model_specs() if spec.name == "ridge")
    model = ridge_spec.factory(numeric, categorical)
    assert model is not None
    model.fit(historical.features.loc[train_mask], target_in_millions(historical))
    return model, historical, future


def build_demo_outputs(tables: Q1Tables, output_dir: Path) -> OutputArtifacts:
    """Train the selected Demo Ridge model and write official Q1-format CSVs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    model, historical, future = _fit_demo_ridge(tables)
    test_mask = historical.metadata["dataset_split"].eq("test")
    test_predictions = np.clip(model.predict(historical.features.loc[test_mask]), 0.0, None)
    future_predictions_people = np.clip(model.predict(future.features) * 1_000_000, 0.0, None)

    test_output = pd.DataFrame(
        {
            "match_id_test": historical.metadata.loc[test_mask, "match_id"].to_numpy(),
            "predicted_test_tv_viewers": test_predictions,
        }
    )
    match_output = future.metadata.loc[:, ["match_id", "team_a", "team_b"]].copy()
    match_output["predicted_tv_viewers"] = future_predictions_people
    test_csv = output_dir / "result_1_test_prediction.csv"
    match_csv = output_dir / "result_1_match_prediction.csv"
    test_output.to_csv(test_csv, index=False)
    match_output.to_csv(match_csv, index=False)
    return OutputArtifacts(test_csv=test_csv, match_csv=match_csv)


def validate_demo_outputs(
    tables: Q1Tables, test_csv: Path, match_csv: Path
) -> OutputValidationReport:
    """Reload generated files and verify exact identifiers, schemas, and units."""

    errors: list[str] = []
    test_output = pd.read_csv(test_csv)
    match_output = pd.read_csv(match_csv)
    expected_test_ids = tables.historical.loc[
        tables.historical["dataset_split"].eq("test"), "match_id"
    ]
    expected_match_keys = tables.group_matches.loc[:, ["match_id", "team_a", "team_b"]]
    if list(test_output.columns) != list(tables.test_template.columns):
        errors.append("test output columns differ from template")
    if list(match_output.columns) != list(tables.match_template.columns):
        errors.append("match output columns differ from template")
    if len(test_output) != 140 or set(test_output["match_id_test"]) != set(expected_test_ids):
        errors.append("test output does not contain exactly the official 140 IDs")
    if len(match_output) != 72:
        errors.append("match output does not contain 72 rows")
    elif not match_output.loc[:, ["match_id", "team_a", "team_b"]].equals(expected_match_keys):
        errors.append("match output keys differ from groups_matches")
    for name, values in (
        ("test predictions", test_output.get("predicted_test_tv_viewers", pd.Series(dtype=float))),
        ("match predictions", match_output.get("predicted_tv_viewers", pd.Series(dtype=float))),
    ):
        if len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
            errors.append(f"{name} must be finite and nonnegative")
    if len(test_output) and not test_output["predicted_test_tv_viewers"].between(1.0, 500.0).all():
        errors.append("test predictions are not expressed in millions")
    if len(match_output) and not match_output["predicted_tv_viewers"].between(1_000_000, 500_000_000).all():
        errors.append("match predictions are not expressed in people")
    return OutputValidationReport(tuple(errors))

