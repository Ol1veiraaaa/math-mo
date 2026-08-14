"""Small-sample baseline model specifications for the first Q1 Demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[list[str], list[str]], object | None]


def _preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _ridge(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    return Pipeline(
        [("preprocess", _preprocessor(numeric_columns, categorical_columns)), ("model", Ridge(alpha=10.0))]
    )


def _hist_gradient_boosting(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(numeric_columns, categorical_columns)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_leaf_nodes=12,
                    min_samples_leaf=20,
                    l2_regularization=2.0,
                    early_stopping=False,
                    random_state=20260814,
                ),
            ),
        ]
    )


def _elastic_net(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(numeric_columns, categorical_columns)),
            ("model", ElasticNet(alpha=0.15, l1_ratio=0.15, max_iter=20_000, random_state=20260814)),
        ]
    )


def _svr(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(numeric_columns, categorical_columns)),
            ("model", SVR(C=3.0, epsilon=0.1, gamma="scale")),
        ]
    )


def _extra_trees(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(numeric_columns, categorical_columns)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=500,
                    max_features=0.8,
                    min_samples_leaf=4,
                    random_state=20260814,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _random_forest(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(numeric_columns, categorical_columns)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_features=0.8,
                    min_samples_leaf=5,
                    random_state=20260814,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def demo_model_specs() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec("mean", lambda numeric, categorical: None),
        ModelSpec("ridge", _ridge),
        ModelSpec("hist_gradient_boosting", _hist_gradient_boosting),
    )


def sklearn_challenger_specs() -> tuple[ModelSpec, ...]:
    """Second-pass challengers using the unchanged Demo feature and split contract."""

    return (
        ModelSpec("elastic_net", _elastic_net),
        ModelSpec("svr_rbf", _svr),
        ModelSpec("extra_trees", _extra_trees),
        ModelSpec("random_forest", _random_forest),
    )
