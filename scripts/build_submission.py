"""Build the anonymous paper and a whitelist-only support archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
SIZE_LIMIT = 20 * 1024 * 1024

SUPPORT_FILES = (
    "README.md",
    "pyproject.toml",
    "requirements-demo.txt",
    "requirements-copt.txt",
    "docs/project/COPT_EVIDENCE.md",
    "docs/project/Q4_DATA_CONTRACT.md",
    "outputs/q1/demo/result_1_test_prediction.csv",
    "outputs/q1/demo/result_1_match_prediction.csv",
    "outputs/q1/demo/leaderboard.csv",
    "outputs/q1/demo/fold_metrics.csv",
    "outputs/q1/demo/feature_manifest.json",
    "outputs/q2/demo/result_2_group_schedule.csv",
    "outputs/q2/demo/objective_score.json",
    "outputs/q2/demo/feasibility_report.txt",
    "outputs/q2/copt/result_2_group_schedule.csv",
    "outputs/q2/copt/solve_metadata.json",
    "outputs/q2/copt/solver.log",
    "outputs/q3/copt/result_3_dynamic_strategy.csv",
    "outputs/q3/copt/simulation_summary.csv",
    "outputs/q3/copt/static_selection.csv",
    "outputs/q3/copt/dynamic_selection.csv",
    "outputs/q3/copt/feasible_lower_bound_selection.csv",
    "outputs/q3/copt/solve_metadata.json",
    "outputs/q3/copt/solver.log",
    "outputs/q4/actual_schedule.csv",
    "outputs/q4/source_hashes.json",
    "outputs/q4/demo/result_4_schedule_comparison.csv",
    "outputs/analysis/analysis_summary.json",
    "outputs/analysis/analysis_report.md",
    "outputs/analysis/q2_weight_robustness_samples.csv",
    "outputs/analysis/q3_simulation_convergence.csv",
    "outputs/analysis/q3_static_dynamic_decisions.csv",
    "outputs/analysis/q3_daily_resource_utilization.csv",
    "outputs/analysis/q3_static_dynamic_totals.csv",
    "outputs/validation/deliverable_validation.json",
    "research/q4/openfootball_2026_cup.txt",
    "research/q4/openfootball_2026_stadiums.csv",
    "research/q4/SOURCE.md",
)

SUPPORT_TREES = ("src", "scripts", "tests")
TREE_SUFFIXES = {".py", ".ps1"}
PAPER_SOURCES = (
    "paper/main.tex",
    "paper/ai_usage.tex",
    "paper/modelingpaper.sty",
    "paper/references.bib",
    "paper/source_appendix.tex",
)
OFFICIAL_TEMPLATES = (
    "actual_schedule_template.csv",
    "result_1_match_prediction_template.csv",
    "result_1_test_prediction_template.csv",
    "result_2_template.csv",
    "result_3_template.csv",
    "result_4_template.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--team-id",
        required=True,
        help="Three digits for the official submission, or XXX for a placeholder build.",
    )
    return parser.parse_args()


def validate_team_id(value: str) -> str:
    value = value.strip().upper()
    if not re.fullmatch(r"(?:\d{3}|XXX)", value):
        raise ValueError("team id must be exactly three digits, or XXX for a placeholder build")
    return value


def required_sources() -> list[Path]:
    paths = [ROOT / relative for relative in SUPPORT_FILES]
    paths.append(ROOT / "paper/build/main.pdf")
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required submission artifacts are missing: {missing}")
    return paths


def add_support_tree(stage: Path) -> None:
    for tree_name in SUPPORT_TREES:
        tree = ROOT / tree_name
        for source in sorted(tree.rglob("*")):
            if source.is_file() and source.suffix.lower() in TREE_SUFFIXES:
                destination = stage / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)


def add_paper_sources(stage: Path) -> None:
    for relative in PAPER_SOURCES:
        source = ROOT / relative
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def add_official_inputs(stage: Path) -> None:
    sys_path = str(ROOT / "src")
    if sys_path not in __import__("sys").path:
        __import__("sys").path.insert(0, sys_path)
    from c_contest_q1.paths import discover_source_paths, discover_template_path

    destination_root = stage / "data" / "official"
    destination_root.mkdir(parents=True, exist_ok=True)
    source_paths = discover_source_paths()
    shutil.copy2(source_paths.workbook, destination_root / "c_contest_inputs.xlsx")
    for name in OFFICIAL_TEMPLATES:
        shutil.copy2(discover_template_path(name), destination_root / name)


def build(team_id: str) -> dict[str, object]:
    required_sources()
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    paper_output = SUBMISSION / f"{team_id}_参赛论文.pdf"
    zip_output = SUBMISSION / f"{team_id}_支撑材料.zip"
    shutil.copy2(ROOT / "paper/build/main.pdf", paper_output)

    with tempfile.TemporaryDirectory(prefix="c_contest_submission_") as tmp:
        stage = Path(tmp) / f"{team_id}_支撑材料"
        for relative in SUPPORT_FILES:
            source = ROOT / relative
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        shutil.copy2(ROOT / "paper/build/ai_usage.pdf", stage / "AI工具使用详情.pdf")
        add_support_tree(stage)
        add_paper_sources(stage)
        add_official_inputs(stage)

        entries = []
        for source in sorted(path for path in stage.rglob("*") if path.is_file()):
            relative = source.relative_to(stage).as_posix()
            entries.append({"path": relative, "bytes": source.stat().st_size, "sha256": sha256(source)})
        (stage / "support_manifest.json").write_text(
            json.dumps({"team_id": team_id, "files": entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for source in sorted(path for path in stage.rglob("*") if path.is_file()):
                archive.write(source, source.relative_to(stage).as_posix())

    for path in (paper_output, zip_output):
        if path.stat().st_size >= SIZE_LIMIT:
            raise ValueError(f"{path.name} is {path.stat().st_size} bytes, exceeding the strict <20 MB limit")

    with zipfile.ZipFile(zip_output) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC validation failed at {bad_member}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")

    manifest = {
        "team_id": team_id,
        "placeholder_name": team_id == "XXX",
        "size_limit_bytes": SIZE_LIMIT,
        "files": {
            paper_output.name: {"bytes": paper_output.stat().st_size, "sha256": sha256(paper_output)},
            zip_output.name: {"bytes": zip_output.stat().st_size, "sha256": sha256(zip_output)},
        },
        "support_archive_members": len(names),
    }
    (SUBMISSION / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    team_id = validate_team_id(parse_args().team_id)
    print(json.dumps(build(team_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
