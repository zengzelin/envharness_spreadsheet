from __future__ import annotations

from pathlib import Path

import openpyxl

from envharness.bridges.spreadsheetbench.bridge import (
    SpreadsheetBenchEnv,
    SpreadsheetBenchEnvState,
)
from envharness.core.types import Action


def _execution_env(tmp_path: Path) -> SpreadsheetBenchEnv:
    input_path = tmp_path / "input.xlsx"
    output_path = tmp_path / "output.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.title = "Sheet1"
    workbook.save(input_path)
    workbook.save(output_path)

    env = SpreadsheetBenchEnv()
    env._workdir = str(tmp_path)
    env.state = SpreadsheetBenchEnvState(
        task_id="task-1",
        instruction="Write A1",
        answer_position="'Sheet1'!A1",
        answer_sheet="Sheet1",
        input_path=str(input_path),
        output_path=str(output_path),
    )
    return env


def test_run_python_predefines_paths_and_workbook_helpers(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)

    response = env.step(Action(name="run_python", kwargs={
        "code": (
            "wb = load_workbook_for_edit()\n"
            "wb['Sheet1']['A1'] = input_path\n"
            "save_workbook(wb)"
        )
    }))

    workbook = openpyxl.load_workbook(env.state.output_path)
    assert workbook["Sheet1"]["A1"].value == env.state.input_path
    assert response.reward == 0.0
    assert response.info["returncode"] == 0
    assert response.info["python_error"] is False
    assert response.info["python_error_type"] == ""


def test_run_python_returns_structured_syntax_error_and_penalty(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._python_error_penalty = -0.05
    env._syntax_error_penalty = -0.1

    response = env.step(Action(name="run_python", kwargs={
        "code": "for value in:\n    pass"
    }))

    assert response.reward == -0.1
    assert response.info["returncode"] != 0
    assert response.info["python_error"] is True
    assert response.info["python_error_type"] == "SyntaxError"
    assert response.info["syntax_error"] is True


def test_run_python_classifies_exception_before_output_truncation(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._obs_truncate = 100

    response = env.step(Action(name="run_python", kwargs={
        "code": "print('x' * 500)\nraise ValueError('bad data')"
    }))

    assert response.info["python_error_type"] == "ValueError"
    assert response.observation.text.endswith("...[output truncated]")


def test_validate_workbook_checks_output_without_grading(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)

    response = env.step(Action(name="validate_workbook", kwargs={}))

    assert response.reward == 0.0
    assert response.terminated is False
    assert response.info["validation_ok"] is True
    assert "Sheet1" in response.observation.text
