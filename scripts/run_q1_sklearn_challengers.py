"""Compare sklearn Q1 challengers under the immutable Demo evaluation contract."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from c_contest_q1.data import load_q1_tables
from c_contest_q1.evaluation import evaluate_model_specs
from c_contest_q1.features import build_historical_features, target_in_millions
from c_contest_q1.models import demo_model_specs, sklearn_challenger_specs


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "q1" / "sklearn_challengers"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_historical_features(load_q1_tables())
    result = evaluate_model_specs(
        frame.features,
        frame.metadata,
        target_in_millions(frame),
        demo_model_specs() + sklearn_challenger_specs(),
    )
    result.leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    result.fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    result.oof_predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    print(result.leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()

