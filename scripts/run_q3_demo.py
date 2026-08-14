from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from c_contest_q1.data import load_q1_tables
from c_contest_q1.q3_demo import build_q3_state

def main() -> None:
    out = ROOT / "outputs" / "q3" / "demo"; out.mkdir(parents=True, exist_ok=True)
    artifacts = build_q3_state(load_q1_tables(), ROOT / "outputs" / "q2" / "copt" / "result_2_group_schedule.csv", ROOT / "outputs" / "q1" / "demo" / "result_1_match_prediction.csv")
    artifacts.matches.to_csv(out / "q3_state_inputs.csv", index=False)
    artifacts.simulation_summary.to_csv(out / "simulation_summary.csv", index=False)
    print(artifacts.simulation_summary.head().to_string(index=False))

if __name__ == "__main__": main()
