"""Date-grouped expanding-window model evaluation for Q1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .models import ModelSpec


DEFAULT_TEMPORAL_FOLDS = (
    ("fold_1", "2020-04-29", "2020-05-07", "2021-09-28"),
    ("fold_2", "2021-09-28", "2021-10-03", "2023-02-23"),
    ("fold_3", "2023-02-23", "2023-03-02", "2024-07-27"),
)


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_indices: np.ndarray
    valid_indices: np.ndarray


@dataclass(frozen=True)
class EvaluationResult:
    leaderboard: pd.DataFrame
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame


def make_temporal_folds(metadata: pd.DataFrame) -> tuple[TemporalFold, ...]:
    train_metadata = metadata.loc[metadata["dataset_split"].eq("train")].copy()
    dates = pd.to_datetime(train_metadata["date"], errors="raise")
    folds: list[TemporalFold] = []
    for name, train_end, valid_start, valid_end in DEFAULT_TEMPORAL_FOLDS:
        train_mask = dates.le(pd.Timestamp(train_end))
        valid_mask = dates.between(pd.Timestamp(valid_start), pd.Timestamp(valid_end), inclusive="both")
        train_indices = train_metadata.index[train_mask].to_numpy()
        valid_indices = train_metadata.index[valid_mask].to_numpy()
        if not len(train_indices) or not len(valid_indices):
            raise ValueError(f"{name} is empty")
        if dates.loc[train_indices].max() >= dates.loc[valid_indices].min():
            raise ValueError(f"{name} violates temporal order")
        folds.append(TemporalFold(name, train_indices, valid_indices))
    return tuple(folds)


def _column_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = features.select_dtypes(include=["string", "object", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    return numeric, categorical


def evaluate_model_specs(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    target: pd.Series,
    specs: tuple[ModelSpec, ...],
) -> EvaluationResult:
    train_mask = metadata["dataset_split"].eq("train")
    train_features = features.loc[train_mask]
    train_metadata = metadata.loc[train_mask]
    if not target.index.equals(train_features.index):
        raise ValueError("target and training features must share index")
    numeric_columns, categorical_columns = _column_groups(train_features)
    folds = make_temporal_folds(metadata)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for spec in specs:
        for fold in folds:
            x_train = features.loc[fold.train_indices]
            y_train = target.loc[fold.train_indices]
            x_valid = features.loc[fold.valid_indices]
            y_valid = target.loc[fold.valid_indices]
            estimator = spec.factory(numeric_columns, categorical_columns)
            if estimator is None:
                prediction = np.full(len(x_valid), y_train.mean(), dtype=float)
            else:
                estimator.fit(x_train, y_train)
                prediction = np.asarray(estimator.predict(x_valid), dtype=float)
            mse = float(mean_squared_error(y_valid, prediction))
            metric_rows.append(
                {
                    "model": spec.name,
                    "fold": fold.name,
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "mae": float(mean_absolute_error(y_valid, prediction)),
                    "n_validation": len(y_valid),
                }
            )
            for index, actual, predicted in zip(fold.valid_indices, y_valid, prediction, strict=True):
                prediction_rows.append(
                    {
                        "model": spec.name,
                        "fold": fold.name,
                        "row_index": index,
                        "match_id": train_metadata.loc[index, "match_id"],
                        "date": train_metadata.loc[index, "date"],
                        "actual": actual,
                        "prediction": predicted,
                        "residual": predicted - actual,
                    }
                )
    fold_metrics = pd.DataFrame(metric_rows).sort_values(["model", "fold"]).reset_index(drop=True)
    leaderboard = (
        fold_metrics.groupby("model", as_index=False)
        .agg(
            mse=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            worst_fold_mse=("mse", "max"),
        )
        .fillna({"mse_std": 0.0})
        .sort_values(["mse", "worst_fold_mse"])
        .reset_index(drop=True)
    )
    return EvaluationResult(
        leaderboard=leaderboard,
        fold_metrics=fold_metrics,
        oof_predictions=pd.DataFrame(prediction_rows).sort_values(["model", "row_index"]).reset_index(drop=True),
    )

