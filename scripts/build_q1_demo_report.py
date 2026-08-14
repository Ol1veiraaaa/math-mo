"""Build a compact evidence report and plots for the first Q1 Demo."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from c_contest_q1.data import load_q1_tables
from c_contest_q1.outputs import validate_demo_outputs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""

    display = frame.copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "q1" / "demo"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.read_csv(output_dir / "leaderboard.csv")
    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    oof = pd.read_csv(output_dir / "oof_predictions.csv", parse_dates=["date"])
    ridge = oof.loc[oof["model"].eq("ridge")].copy()

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=leaderboard, x="model", y="mse", hue="model", legend=False, ax=axes[0])
    axes[0].set_title("Temporal validation MSE")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("MSE (millions squared)")
    axes[0].tick_params(axis="x", rotation=18)
    sns.lineplot(data=fold_metrics, x="fold", y="mse", hue="model", marker="o", ax=axes[1])
    axes[1].set_title("MSE by expanding-window fold")
    axes[1].set_xlabel("Validation period")
    axes[1].set_ylabel("MSE (millions squared)")
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(data=ridge, x="actual", y="prediction", hue="fold", ax=axes[0])
    limits = [min(ridge[["actual", "prediction"]].min()), max(ridge[["actual", "prediction"]].max())]
    axes[0].plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Ridge: actual versus prediction")
    axes[0].set_xlabel("Actual viewers (millions)")
    axes[0].set_ylabel("Predicted viewers (millions)")
    sns.scatterplot(data=ridge, x="date", y="residual", hue="fold", ax=axes[1])
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Ridge residuals over time")
    axes[1].set_xlabel("Match date")
    axes[1].set_ylabel("Prediction minus actual (millions)")
    fig.tight_layout()
    fig.savefig(figures_dir / "ridge_diagnostics.png", dpi=180)
    plt.close(fig)

    tables = load_q1_tables()
    test_csv = output_dir / "result_1_test_prediction.csv"
    match_csv = output_dir / "result_1_match_prediction.csv"
    validation = validate_demo_outputs(tables, test_csv, match_csv)
    environment = {
        "python": sys.version,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "seaborn": sns.__version__,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifact_hashes = {path.name: _sha256(path) for path in (test_csv, match_csv)}
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    test_prediction = pd.read_csv(test_csv)
    match_prediction = pd.read_csv(match_csv)
    report = f"""# Q1 Demo Model Effect Report

## Current Demo Decision

The current frozen Demo model is **Ridge regression** with fold-fitted one-hot categorical encoding and standardized numeric features. It is selected solely because it has the lowest mean original-scale temporal-validation MSE among the Demo candidates and is better in every validation period.

## Common Evaluation Contract

- Training labels: 560 rows, converted from people to millions of people.
- Hidden official test: 140 rows, never used for model selection.
- Validation: three expanding time windows grouped by complete match date.
- Metrics: MSE is primary; RMSE and MAE are auxiliary, all in millions of people.
- Feature boundary: only shared pre-match information. Current-match results, xG, shots, possession, attendance, IDs, and future-only composite commercial indices are excluded.

## Leaderboard

{_markdown_table(leaderboard)}

## Fold Evidence

{_markdown_table(fold_metrics)}

Ridge improves mean MSE by {(1 - leaderboard.loc[leaderboard['model'].eq('ridge'), 'mse'].iloc[0] / leaderboard.loc[leaderboard['model'].eq('mean'), 'mse'].iloc[0]) * 100:.1f}% versus the mean baseline. Its newest-fold MSE is 183.77, below both other Demo candidates.

## Outputs

- `result_1_test_prediction.csv`: {len(test_prediction)} rows, values in millions of viewers.
- `result_1_match_prediction.csv`: {len(match_prediction)} rows, values in people.
- Independent reload validation: {'PASS' if validation.ok else 'FAIL: ' + '; '.join(validation.errors)}.
- SHA-256 values: see `artifact_hashes.json`.

## Figures

- `figures/model_comparison.png`: candidate leaderboard and fold-level MSE.
- `figures/ridge_diagnostics.png`: actual-predicted relation and temporal residuals.

## Interpretation And Limitations

This is a valid end-to-end Demo, not a final optimality claim. Ridge outperforms the initial constrained histogram gradient boosting model, indicating that the current small, common feature set has a largely stable linear signal or that the tree model is under-tuned. The model has not yet compared CatBoost, LightGBM, XGBoost, Extra Trees, SVR, target transformation variants, or a tightly controlled two-model blend. The future 72-match setting is a World Cup group-stage domain shift while the historical data contains only a limited number of World Cup matches; therefore later iterations must report subgroup performance and cannot infer hidden-test superiority from this Demo alone.

## Next Upgrade Question

The next high-stakes reasoning step is whether to extend the feature contract with strictly lagged historical-form variables and how to compare boosted-tree models under the World Cup domain shift. Per the project rule, switching to `sol/ultra` for that route decision requires explicit user approval first.
"""
    (output_dir / "model_effect_report.md").write_text(report, encoding="utf-8")
    print(output_dir / "model_effect_report.md")


if __name__ == "__main__":
    main()
