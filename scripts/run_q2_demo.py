from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c_contest_q1.data import load_q1_tables
from c_contest_q1.paths import discover_template_path
from c_contest_q1.q2_demo import build_q2_demo_schedule, evaluate_q2_schedule, validate_q2_schedule


def main() -> None:
    tables = load_q1_tables()
    q1 = ROOT / "outputs" / "q1" / "demo" / "result_1_match_prediction.csv"
    out_dir = ROOT / "outputs" / "q2" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule = build_q2_demo_schedule(tables, q1)
    validation = validate_q2_schedule(tables, schedule)
    if not validation.ok:
        raise RuntimeError("Q2 feasibility failed: " + "; ".join(validation.errors))
    schedule, score = evaluate_q2_schedule(tables, schedule)
    template = discover_template_path("result_2_template.csv")
    columns = __import__('pandas').read_csv(template).columns.tolist()
    schedule.loc[:, columns].to_csv(out_dir / "result_2_group_schedule.csv", index=False)
    (out_dir / "feasibility_report.txt").write_text(
        "PASS: all six hard-constraint groups recomputed from CSV.\n"
        "P2 score uses fixed eligible-candidate min-max ranges for ticket, broadcast, operation cost, and travel.\n",
        encoding="utf-8",
    )
    __import__('json').dump(score, (out_dir / "objective_score.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(schedule.loc[:, columns].head().to_string(index=False))


if __name__ == "__main__":
    main()
