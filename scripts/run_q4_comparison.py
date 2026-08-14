from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c_contest_q1.data import load_q1_tables
from c_contest_q1.q4_demo import compare_structural_schedule


def main() -> None:
    actual = ROOT / "outputs" / "q4" / "actual_schedule.csv"
    optimized = ROOT / "outputs" / "q2" / "copt" / "result_2_group_schedule.csv"
    if not actual.exists():
        raise FileNotFoundError(
            "Q4 actual_schedule.csv is intentionally absent. Acquire a publicly sourced completed World Cup schedule "
            "using the official template and source_url/retrieval_date fields, then rerun this script."
        )
    output = compare_structural_schedule(load_q1_tables(), actual, optimized)
    out_dir = ROOT / "outputs" / "q4" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_dir / "result_4_schedule_comparison.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
