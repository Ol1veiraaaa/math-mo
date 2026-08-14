from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c_contest_q1.data import load_q1_tables
from c_contest_q1.q2_copt import solve_copt_q2
from c_contest_q1.q2_demo import validate_q2_schedule


def main() -> None:
    out = ROOT / "outputs" / "q2" / "copt"
    out.mkdir(parents=True, exist_ok=True)
    artifacts = solve_copt_q2(
        load_q1_tables(),
        ROOT / "outputs" / "q2" / "demo" / "result_2_group_schedule.csv",
        log_path=out / "solver.log",
    )
    report = validate_q2_schedule(load_q1_tables(), artifacts.schedule)
    if not report.ok:
        raise RuntimeError("COPT Q2 feasibility failure: " + "; ".join(report.errors))
    artifacts.schedule.to_csv(out / "result_2_group_schedule.csv", index=False)
    (out / "solve_metadata.json").write_text(json.dumps({"solver": "COPT 8.0.6", "license_mode": "non-commercial evaluation, 2000-variable/constraint limit", "candidate_count": artifacts.candidate_count, "status": artifacts.status, "mip_objective": artifacts.mip_objective, "solve_evidence": artifacts.solve_evidence, "full_p2_score": artifacts.score_report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(artifacts.schedule.head().to_string(index=False))
    print({"candidate_count": artifacts.candidate_count, "status": artifacts.status, "mip_objective": artifacts.mip_objective, "full_p2_objective": artifacts.score_report["objective"]})


if __name__ == "__main__":
    main()
