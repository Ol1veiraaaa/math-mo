"""Run the first complete Q1 Demo evaluation and persist reproducible evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from c_contest_q1.data import load_q1_tables, validate_q1_tables
from c_contest_q1.evaluation import evaluate_model_specs
from c_contest_q1.features import build_historical_features, target_in_millions
from c_contest_q1.models import demo_model_specs
from c_contest_q1.outputs import build_demo_outputs, validate_demo_outputs


def main() -> None:
    output_dir = Path("outputs/q1/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_q1_tables()
    report = validate_q1_tables(tables)
    if not report.ok:
        raise RuntimeError("Input validation failed: " + "; ".join(report.errors))
    frame = build_historical_features(tables)
    result = evaluate_model_specs(
        frame.features,
        frame.metadata,
        target_in_millions(frame),
        demo_model_specs(),
    )
    result.leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    result.fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    result.oof_predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    artifacts = build_demo_outputs(tables, output_dir)
    output_report = validate_demo_outputs(tables, artifacts.test_csv, artifacts.match_csv)
    if not output_report.ok:
        raise RuntimeError("Output validation failed: " + "; ".join(output_report.errors))
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(
            {
                "features": list(frame.features.columns),
                "target_unit": "millions_of_people",
                "n_training_rows": int(target_in_millions(frame).shape[0]),
                "n_hidden_test_rows": int(frame.metadata["dataset_split"].eq("test").sum()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result.leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
