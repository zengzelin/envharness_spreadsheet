from __future__ import annotations

from pathlib import Path
import json

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
    workbook["Sheet1"]["A1"] = "Name"
    workbook["Sheet1"]["A2"] = "Alpha"
    workbook["Sheet1"]["B2"] = "=1+1"
    second = workbook.create_sheet("Hidden Data")
    second.sheet_state = "hidden"
    second["C3"] = "Needle"
    workbook.save(input_path)
    workbook.save(output_path)

    env = SpreadsheetBenchEnv()
    env._tool_set = "native_read"
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


def _tool_payload(response) -> dict:
    marker = "last tool output:\n"
    assert marker in response.observation.text
    return json.loads(response.observation.text.split(marker, 1)[1])


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


def test_list_sheets_reports_order_visibility_and_dimensions(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)

    response = env.step(Action(name="list_sheets", kwargs={}))

    payload = _tool_payload(response)
    assert response.reward == 0.0
    assert response.info["tool_ok"] is True
    assert payload["status"] == "success"
    assert payload["sheet_count"] == 2
    assert payload["sheets"][0] == {
        "name": "Sheet1",
        "index": 0,
        "state": "visible",
        "active": True,
        "max_row": 2,
        "max_column": 2,
    }
    assert payload["sheets"][1]["state"] == "hidden"


def test_inspect_range_returns_formulas_and_summary(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)

    cells_response = env.step(Action(name="inspect_range", kwargs={
        "range": "A1:B2",
        "sheet_name": "Sheet1",
    }))
    summary_response = env.step(Action(name="inspect_range", kwargs={
        "range": "'Hidden Data'!A1:C3",
        "mode": "summary",
    }))

    cells = _tool_payload(cells_response)
    summary = _tool_payload(summary_response)
    assert cells["cells"][3]["address"] == "B2"
    assert cells["cells"][3]["formula"] == "=1+1"
    assert summary["summary"]["cells"] == 9
    assert summary["summary"]["non_empty_cells"] == 1
    assert summary["summary"]["formula_cells"] == 0


def test_find_cells_searches_values_and_formulas(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)

    value_response = env.step(Action(name="find_cells", kwargs={
        "query": "alpha",
        "match": "equals",
        "case_sensitive": False,
        "include_values": True,
    }))
    formula_response = env.step(Action(name="find_cells", kwargs={
        "query": "1+1",
        "search_in": "formulas",
        "return_mode": "all",
    }))

    values = _tool_payload(value_response)
    formulas = _tool_payload(formula_response)
    assert values["matches"] == [{
        "sheet": "Sheet1", "address": "A2", "value": "Alpha"
    }]
    assert formulas["matches"] == [{
        "sheet": "Sheet1", "address": "B2", "formula": "=1+1"
    }]


def test_native_read_tools_are_disabled_in_python_tool_set(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "python"

    response = env.step(Action(name="list_sheets", kwargs={}))

    assert response.info["error"] == "tool_disabled"
    assert "SPREADSHEETBENCH_TOOL_SET=native_read" in response.observation.text


def test_tool_schemas_follow_selected_tool_set(monkeypatch) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "python")
    python_names = {
        item["function"]["name"] for item in SpreadsheetBenchEnv.tool_schemas()
    }
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_read")
    native_names = {
        item["function"]["name"] for item in SpreadsheetBenchEnv.tool_schemas()
    }

    assert python_names == {"run_python", "validate_workbook", "submit"}
    assert native_names == python_names | {
        "list_sheets", "inspect_range", "find_cells"
    }


def test_native_read_tool_parameter_error_is_nonfatal(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)

    response = env.step(Action(name="inspect_range", kwargs={
        "range": "A:A",
    }))

    payload = _tool_payload(response)
    assert response.terminated is False
    assert response.reward == 0.0
    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "invalid_range"
    assert payload["status"] == "error"
