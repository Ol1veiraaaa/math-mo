"""Discovery of immutable contest inputs outside the generated project tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_SHEETS = frozenset(
    {"historical_matches", "teams", "groups_matches", "base_predictions"}
)


@dataclass(frozen=True)
class SourcePaths:
    """Resolved locations for the official Q1 workbook and its two CSV templates."""

    workbook: Path
    test_template: Path
    match_template: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def contest_root() -> Path:
    return project_root().parent


def bundled_data_root() -> Path:
    return project_root() / "data" / "official"


def _find_unique_file(name: str) -> Path:
    bundled = bundled_data_root() / name
    if bundled.is_file():
        return bundled.resolve()
    candidates = [
        path.resolve()
        for path in contest_root().rglob(name)
        if not path.name.startswith("~$") and project_root() not in path.resolve().parents
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one official {name}; found {candidates}")
    return candidates[0]


def _find_workbook() -> Path:
    bundled_candidates = _workbooks_with_required_sheets(bundled_data_root().glob("*.xlsx"))
    if len(bundled_candidates) == 1:
        return bundled_candidates[0]
    if len(bundled_candidates) > 1:
        raise FileNotFoundError(f"Expected exactly one bundled C-contest workbook; found {bundled_candidates}")
    candidates = _workbooks_with_required_sheets(contest_root().rglob("*.xlsx"), exclude_project=True)
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one C-contest workbook; found {candidates}")
    return candidates[0]


def _workbooks_with_required_sheets(
    workbooks: object,
    exclude_project: bool = False,
) -> list[Path]:
    candidates: list[Path] = []
    for workbook in workbooks:
        if workbook.name.startswith("~$"):
            continue
        if exclude_project and project_root() in workbook.resolve().parents:
            continue
        try:
            sheets = set(pd.ExcelFile(workbook).sheet_names)
        except (OSError, ValueError):
            continue
        if REQUIRED_SHEETS.issubset(sheets):
            candidates.append(workbook.resolve())
    return candidates


def discover_template_path(name: str) -> Path:
    """Locate one official CSV template, preferring a bundled submission copy."""

    return _find_unique_file(name)


def discover_source_paths() -> SourcePaths:
    """Locate contest inputs without relying on locale-sensitive directory names."""

    return SourcePaths(
        workbook=_find_workbook(),
        test_template=_find_unique_file("result_1_test_prediction_template.csv"),
        match_template=_find_unique_file("result_1_match_prediction_template.csv"),
    )
