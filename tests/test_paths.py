from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from c_contest_q1 import paths


OFFICIAL_TEMPLATES = (
    "actual_schedule_template.csv",
    "result_1_match_prediction_template.csv",
    "result_1_test_prediction_template.csv",
    "result_2_template.csv",
    "result_3_template.csv",
    "result_4_template.csv",
)


def _write_workbook(path: Path, sheet_names: set[str] | frozenset[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        for sheet_name in sorted(sheet_names):
            pd.DataFrame({"value": [1]}).to_excel(
                writer, sheet_name=sheet_name, index=False
            )


def _patch_roots(monkeypatch: pytest.MonkeyPatch, project: Path, contest: Path) -> None:
    monkeypatch.setattr(paths, "project_root", lambda: project)
    monkeypatch.setattr(paths, "contest_root", lambda: contest)


def test_bundled_inputs_take_priority_over_external_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "support"
    bundled = project / "data" / "official"
    external = tmp_path / "official_attachments"
    _patch_roots(monkeypatch, project, tmp_path)

    _write_workbook(bundled / "c_contest_inputs.xlsx", paths.REQUIRED_SHEETS)
    _write_workbook(external / "external_inputs.xlsx", paths.REQUIRED_SHEETS)
    for name in OFFICIAL_TEMPLATES:
        bundled.mkdir(parents=True, exist_ok=True)
        (bundled / name).write_text("bundled\n", encoding="utf-8")
        external.mkdir(parents=True, exist_ok=True)
        (external / name).write_text("external\n", encoding="utf-8")

    sources = paths.discover_source_paths()

    assert sources.workbook == (bundled / "c_contest_inputs.xlsx").resolve()
    assert sources.test_template == (
        bundled / "result_1_test_prediction_template.csv"
    ).resolve()
    assert sources.match_template == (
        bundled / "result_1_match_prediction_template.csv"
    ).resolve()
    for name in OFFICIAL_TEMPLATES:
        assert paths.discover_template_path(name) == (bundled / name).resolve()


def test_external_inputs_are_discovered_when_bundle_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "support"
    external = tmp_path / "official_attachments"
    _patch_roots(monkeypatch, project, tmp_path)

    workbook = external / "inputs.xlsx"
    _write_workbook(workbook, paths.REQUIRED_SHEETS)
    external.mkdir(parents=True, exist_ok=True)
    for name in OFFICIAL_TEMPLATES:
        (external / name).write_text("official\n", encoding="utf-8")

    sources = paths.discover_source_paths()

    assert sources.workbook == workbook.resolve()
    for name in OFFICIAL_TEMPLATES:
        assert paths.discover_template_path(name) == (external / name).resolve()


def test_multiple_external_workbooks_fail_instead_of_guessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "support"
    _patch_roots(monkeypatch, project, tmp_path)
    _write_workbook(tmp_path / "attachments_a" / "inputs.xlsx", paths.REQUIRED_SHEETS)
    _write_workbook(tmp_path / "attachments_b" / "inputs.xlsx", paths.REQUIRED_SHEETS)

    with pytest.raises(FileNotFoundError, match="Expected exactly one C-contest workbook"):
        paths._find_workbook()

