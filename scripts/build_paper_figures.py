from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
COLORS = {"actual": "#4D6A86", "optimized": "#C97A40", "accent": "#4F8A5B", "muted": "#8A8A8A"}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.22,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def q1_model_comparison() -> None:
    leaderboard = pd.read_csv(ROOT / "outputs/q1/sklearn_challengers/leaderboard.csv").sort_values("mse")
    folds = pd.read_csv(ROOT / "outputs/q1/sklearn_challengers/fold_metrics.csv")
    labels = leaderboard.model.str.replace("_", " ").str.title()
    colors = [COLORS["accent"] if model == "ridge" else COLORS["actual"] for model in leaderboard.model]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1.05, 1.2]})
    axes[0].barh(labels[::-1], leaderboard.mse[::-1], color=colors[::-1])
    axes[0].set_xlabel("Mean temporal-validation MSE")
    axes[0].set_title("Seven-model leaderboard")
    for model, block in folds.groupby("model"):
        if model in {"ridge", "extra_trees", "mean"}:
            axes[1].plot(block.fold, block.mse, marker="o", label=model.replace("_", " ").title())
    axes[1].set_ylabel("MSE (million viewers squared)")
    axes[1].set_title("Representative expanding-window folds")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "q1_model_comparison")


def q2_indicator_decomposition() -> None:
    baseline = json.loads((ROOT / "outputs/q2/demo/objective_score.json").read_text(encoding="utf-8"))
    copt = json.loads((ROOT / "outputs/q2/copt/solve_metadata.json").read_text(encoding="utf-8"))["full_p2_score"]
    keys = ["T", "B", "U", "H", "C", "D", "F", "R"]
    labels = ["Ticket", "Broadcast", "Uncertainty", "Attractiveness", "Cost", "Travel", "Fairness", "Risk"]
    x = np.arange(len(keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar(x - width / 2, [baseline[f"norm_{k}"] for k in keys], width, label="Feasible baseline", color=COLORS["actual"])
    ax.bar(x + width / 2, [copt[f"norm_{k}"] for k in keys], width, label="Restricted COPT", color=COLORS["optimized"])
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylabel("Normalized indicator")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Q2 schedule indicator decomposition")
    _save(fig, "q2_indicator_decomposition")


def q2_weight_robustness() -> None:
    data = pd.read_csv(ROOT / "outputs/analysis/q2_weight_robustness_samples.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.hist(data.copt_minus_baseline, bins=36, color=COLORS["accent"], alpha=0.88)
    ax.axvline(data.copt_minus_baseline.quantile(0.05), color=COLORS["optimized"], linestyle="--", label="5th percentile")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("COPT score minus baseline score")
    ax.set_ylabel("Weight perturbations")
    ax.set_title("Q2 robustness under 10,000 weight perturbations")
    ax.legend(frameon=False)
    _save(fig, "q2_weight_robustness")


def q3_probabilities() -> None:
    data = pd.read_csv(ROOT / "outputs/q3/copt/result_3_dynamic_strategy.csv")
    long = pd.concat([
        data[["team_a", "updated_p_team_a_advance"]].rename(columns={"team_a": "team", "updated_p_team_a_advance": "probability"}),
        data[["team_b", "updated_p_team_b_advance"]].rename(columns={"team_b": "team", "updated_p_team_b_advance": "probability"}),
    ]).sort_values("probability")
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    colors = np.where(long.probability.between(0.25, 0.75), COLORS["optimized"], COLORS["actual"])
    ax.barh(long.team, long.probability, color=colors)
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability of advancing")
    ax.set_title("Q3 advancement probabilities after two rounds")
    _save(fig, "q3_advancement_probabilities")


def q3_importance_attractiveness() -> None:
    data = pd.read_csv(ROOT / "outputs/q3/copt/simulation_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    axes[0].hist(data.importance, bins=10, color=COLORS["actual"], alpha=0.9)
    axes[0].set_xlabel("Qualification importance Q")
    axes[0].set_ylabel("Third-round matches")
    axes[0].set_title("Qualification importance distribution")
    axes[1].scatter(
        data.importance,
        data.updated_attractiveness,
        c=data.stakeless_risk,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.5,
        s=48,
    )
    axes[1].set_xlabel("Qualification importance Q")
    axes[1].set_ylabel("Updated attractiveness A")
    axes[1].set_title("Importance and updated attractiveness")
    fig.tight_layout()
    _save(fig, "q3_importance_attractiveness")


def q3_static_dynamic() -> None:
    data = pd.read_csv(ROOT / "outputs/q3/copt/result_3_dynamic_strategy.csv")
    data["change"] = data.dynamic_net_value - data.static_net_value
    data = data.sort_values("change")
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    colors = np.where(data.change >= 0, COLORS["accent"], COLORS["optimized"])
    ax.barh(data.match_id, data.change, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Dynamic minus static contribution")
    ax.set_title("Q3 match-level reallocation effects")
    _save(fig, "q3_static_dynamic_contributions")


def q3_convergence() -> None:
    data = pd.read_csv(ROOT / "outputs/analysis/q3_simulation_convergence.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.plot(data.simulations, data.probability_mae_vs_50000, marker="o", color=COLORS["actual"], label="Probability MAE")
    ax.plot(data.simulations, data.collusion_risk_mae_vs_50000, marker="s", color=COLORS["optimized"], label="Collusion-risk MAE")
    ax.set_xlabel("Monte Carlo draws")
    ax.set_ylabel("Absolute error vs 50,000 draws")
    ax.set_title("Q3 simulation convergence")
    ax.legend(frameon=False)
    _save(fig, "q3_simulation_convergence")


def q3_daily_resources() -> None:
    data = pd.read_csv(ROOT / "outputs/analysis/q3_daily_resource_utilization.csv")
    metrics = [
        ("broadcast_utilization", "Priority-3 broadcast"),
        ("security_utilization", "High security"),
        ("transport_utilization", "Level-3 transport"),
        ("budget_utilization", "Resource budget"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.2), sharex=True, sharey=True)
    for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
        for plan, color, marker in (("Static", COLORS["actual"], "o"), ("Dynamic", COLORS["optimized"], "s")):
            block = data.loc[data.plan.eq(plan)].sort_values("reference_date")
            ax.plot(
                block.reference_date.str.slice(5),
                block[metric],
                marker=marker,
                color=color,
                label=plan,
                markersize=7 if plan == "Static" else 4.5,
                markerfacecolor="white" if plan == "Static" else color,
                markeredgewidth=1.4,
                linewidth=1.8 if plan == "Static" else 1.4,
            )
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_ylim(0, 1.08)
        ax.tick_params(axis="x", rotation=28)
    axes[0, 0].set_ylabel("Used / capacity")
    axes[1, 0].set_ylabel("Used / capacity")
    axes[0, 1].legend(frameon=False, ncol=2, loc="lower left")
    fig.suptitle("Q3 daily resource utilization under supplied limits", y=1.01)
    fig.tight_layout()
    _save(fig, "q3_daily_resource_utilization")


def q4_tradeoffs() -> None:
    data = pd.read_csv(ROOT / "outputs/q4/demo/result_4_schedule_comparison.csv").set_index("indicator_name")
    names = ["travel_km_per_team", "mean_rest_hours", "minimum_rest_hours", "matches_per_venue_cv", "venue_match_hhi", "mean_stadium_capacity"]
    labels = ["Travel / team", "Mean rest", "Minimum rest", "Venue-load CV", "Venue HHI", "Mean capacity"]
    actual = data.loc[names, "actual_schedule_value"].to_numpy(dtype=float)
    optimized = data.loc[names, "optimized_schedule_value"].to_numpy(dtype=float)
    directions = data.loc[names, "preferred_direction"].to_numpy()
    actual_score = np.empty_like(actual)
    optimized_score = np.empty_like(optimized)
    for index, direction in enumerate(directions):
        low, high = min(actual[index], optimized[index]), max(actual[index], optimized[index])
        if np.isclose(low, high):
            actual_score[index] = optimized_score[index] = 1.0
        elif direction == "lower":
            actual_score[index] = (high - actual[index]) / (high - low)
            optimized_score[index] = (high - optimized[index]) / (high - low)
        else:
            actual_score[index] = (actual[index] - low) / (high - low)
            optimized_score[index] = (optimized[index] - low) / (high - low)
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.bar(x - width / 2, actual_score, width, color=COLORS["actual"], label="Actual 2026")
    ax.bar(x + width / 2, optimized_score, width, color=COLORS["optimized"], label="Q2 optimized")
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylabel("Pairwise direction-aligned score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Q4 structural trade-offs (higher is preferable)")
    ax.legend(frameon=False, ncol=2)
    _save(fig, "q4_structural_tradeoffs")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    source_figures = ROOT / "outputs/q1/demo/figures"
    (OUT / "q1_ridge_diagnostics.png").write_bytes((source_figures / "ridge_diagnostics.png").read_bytes())
    q1_model_comparison()
    q2_indicator_decomposition()
    q2_weight_robustness()
    q3_probabilities()
    q3_importance_attractiveness()
    q3_static_dynamic()
    q3_convergence()
    q3_daily_resources()
    q4_tradeoffs()
    print(f"Paper figures written to {OUT}")


if __name__ == "__main__":
    main()
