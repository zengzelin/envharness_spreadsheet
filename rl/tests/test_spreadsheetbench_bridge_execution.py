from __future__ import annotations

import json
from pathlib import Path
import time

import openpyxl
import pytest

from envharness.bridges.spreadsheetbench import recalc_tools, write_tools
from envharness.bridges.spreadsheetbench.bridge import (
    SpreadsheetBenchEnv,
    SpreadsheetBenchEnvState,
    parse_badcase_diagnostics_options,
)
from envharness.bridges.spreadsheetbench.dataset import SBTask
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


def _grading_task(tmp_path: Path, expected: object) -> SBTask:
    golden_path = tmp_path / "golden.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.title = "Sheet1"
    workbook["Sheet1"]["A1"] = expected
    workbook.save(golden_path)
    workbook.close()
    return SBTask(
        id="task-1",
        instruction="Write A1",
        instruction_type="Cell-Level Manipulation",
        answer_position="Sheet1!A1",
        answer_sheet="Sheet1",
        init_path=str(tmp_path / "input.xlsx"),
        golden_path=str(golden_path),
    )


def test_badcase_diagnostic_options_validate_modes_and_positive_limits(
    monkeypatch,
) -> None:
    for name in (
        "SPREADSHEETBENCH_BADCASE_DIAGNOSTICS",
        "SPREADSHEETBENCH_BADCASE_MAX_SCAN_CELLS",
        "SPREADSHEETBENCH_BADCASE_MAX_EXAMPLES",
    ):
        monkeypatch.delenv(name, raising=False)

    mode, limits = parse_badcase_diagnostics_options({})
    assert mode == "light"
    assert limits.max_scan_cells == 200_000
    assert limits.max_examples == 20

    mode, limits = parse_badcase_diagnostics_options({
        "badcase_diagnostics_mode": "FULL",
        "badcase_max_scan_cells": "17",
        "badcase_max_examples": 3,
    })
    assert mode == "full"
    assert limits.max_scan_cells == 17
    assert limits.max_examples == 3

    with pytest.raises(ValueError, match="off, light, or full"):
        parse_badcase_diagnostics_options({"badcase_diagnostics_mode": "verbose"})
    with pytest.raises(ValueError, match="max_scan_cells must be positive"):
        parse_badcase_diagnostics_options({"badcase_max_scan_cells": 0})


def test_full_badcase_diagnostics_run_after_official_failure(
    monkeypatch, tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._recalc_golden = False
    env._badcase_diagnostics_mode = "full"
    task = _grading_task(tmp_path, "expected")
    monkeypatch.setattr(
        "envharness.bridges.spreadsheetbench.bridge."
        "online_judge_eval.recalc_with_libreoffice",
        lambda *args, **kwargs: True,
    )

    result = env._grade(task)

    assert result.success is False
    assert result.score == 0.0
    assert result.metrics["diff"].startswith("value diff at A1")
    diagnostics = result.metrics["badcase_diagnostics"]
    assert diagnostics["mode"] == "full"
    assert diagnostics["eligible"] is True
    assert diagnostics["status"] == "complete"
    assert diagnostics["answer_cells_mismatched"] == 1


def test_full_badcase_diagnostics_do_not_run_on_success(
    monkeypatch, tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._recalc_golden = False
    env._badcase_diagnostics_mode = "full"
    workbook = openpyxl.load_workbook(env.state.output_path)
    workbook["Sheet1"]["A1"] = "expected"
    workbook.save(env.state.output_path)
    workbook.close()
    task = _grading_task(tmp_path, "expected")
    monkeypatch.setattr(
        "envharness.bridges.spreadsheetbench.bridge."
        "online_judge_eval.recalc_with_libreoffice",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "envharness.bridges.spreadsheetbench.bridge.diagnose_failed_workbook",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("diagnostics must not scan successful workbooks")
        ),
    )

    result = env._grade(task)

    assert result.success is True
    assert result.score == 1.0
    assert result.metrics["badcase_diagnostics"] == {
        "version": 1,
        "mode": "full",
        "eligible": False,
        "status": "skipped_success",
    }


def test_badcase_diagnostic_errors_do_not_change_official_failure(
    monkeypatch, tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._recalc_golden = False
    env._badcase_diagnostics_mode = "full"
    task = _grading_task(tmp_path, "expected")
    monkeypatch.setattr(
        "envharness.bridges.spreadsheetbench.bridge."
        "online_judge_eval.recalc_with_libreoffice",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "envharness.bridges.spreadsheetbench.bridge.diagnose_failed_workbook",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private path")),
    )

    result = env._grade(task)

    assert result.success is False
    assert result.score == 0.0
    assert result.metrics["diff"].startswith("value diff at A1")
    assert result.metrics["badcase_diagnostics"] == {
        "version": 1,
        "mode": "full",
        "eligible": True,
        "status": "diagnostic_error",
        "diagnostic_error_type": "RuntimeError",
    }


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


def test_run_python_rolls_back_workbook_after_runtime_failure(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    before = Path(env.state.output_path).read_bytes()

    response = env.step(Action(name="run_python", kwargs={
        "code": (
            "wb = load_workbook_for_edit()\n"
            "wb['Sheet1']['A1'] = 'must not persist'\n"
            "save_workbook(wb)\n"
            "raise RuntimeError('after save')"
        )
    }))

    assert response.info["python_error"] is True
    assert response.info["transaction_committed"] is False
    assert response.info["transaction_rolled_back"] is True
    assert Path(env.state.output_path).read_bytes() == before


def test_run_python_rejects_corrupt_workbook_before_commit(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    before = Path(env.state.output_path).read_bytes()

    response = env.step(Action(name="run_python", kwargs={
        "code": "open(output_path, 'wb').write(b'not an xlsx')",
    }))

    assert response.info["python_error"] is True
    assert response.info["python_error_type"] == "WorkbookValidationError"
    assert response.info["transaction_committed"] is False
    assert response.info["transaction_rolled_back"] is True
    assert Path(env.state.output_path).read_bytes() == before


def test_run_python_restores_committed_path_after_direct_path_corruption(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    before = Path(env.state.output_path).read_bytes()

    response = env.step(Action(name="run_python", kwargs={
        "code": (
            "import os\n"
            "committed = os.path.join(working_directory, 'output.xlsx')\n"
            "open(committed, 'wb').write(b'corrupt committed path')\n"
            "raise RuntimeError('after direct corruption')"
        ),
    }))

    assert response.info["transaction_rolled_back"] is True
    assert Path(env.state.output_path).read_bytes() == before


def test_run_python_rollback_replaces_agent_created_output_symlink(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    output = Path(env.state.output_path)
    before = output.read_bytes()

    response = env.step(Action(name="run_python", kwargs={
        "code": (
            "import os\n"
            "committed = os.path.join(working_directory, 'output.xlsx')\n"
            "os.unlink(committed)\n"
            "os.symlink(output_path, committed)\n"
            "raise RuntimeError('leave a symlink')"
        ),
    }))

    assert response.info["transaction_rolled_back"] is True
    assert output.is_symlink() is False
    assert output.read_bytes() == before


def test_run_python_commits_valid_workbook_and_reports_transaction(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)

    response = env.step(Action(name="run_python", kwargs={
        "code": (
            "wb = load_workbook_for_edit()\n"
            "wb['Sheet1']['A1'] = 'committed'\n"
            "save_workbook(wb)"
        ),
    }))

    workbook = openpyxl.load_workbook(env.state.output_path)
    assert workbook["Sheet1"]["A1"].value == "committed"
    assert response.info["transaction_committed"] is True
    assert response.info["transaction_rolled_back"] is False
    assert env.state.extras["workbook_revision"] == 1


def test_run_python_timeout_reaps_child_process_output_pipes(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._step_timeout = 0.2
    started = time.monotonic()

    response = env.step(Action(name="run_python", kwargs={
        "code": (
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'], "
            "stdout=sys.stdout, stderr=sys.stderr)\n"
            "time.sleep(5)\n"
        )
    }))

    assert time.monotonic() - started < 1.5
    assert response.info["python_error_type"] == "TimeoutExpired"


def test_run_python_logs_task_and_stage(capsys, tmp_path: Path) -> None:
    env = _execution_env(tmp_path)

    env.step(Action(name="run_python", kwargs={"code": "print('ok')"}))

    output = capsys.readouterr().out
    assert "task_id=task-1 step=1 action=run_python stage=run_python START" in output
    assert "task_id=task-1 step=1 action=run_python stage=run_python END" in output


def test_validate_workbook_checks_output_without_grading(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)

    response = env.step(Action(name="validate_workbook", kwargs={}))

    assert response.reward == 0.0
    assert response.terminated is False
    assert response.info["validation_ok"] is True
    assert "Sheet1" in response.observation.text


def test_validate_workbook_rejects_malformed_formula(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    workbook = openpyxl.load_workbook(env.state.output_path)
    workbook["Sheet1"]["A1"] = "=SUM(A2:B2"
    workbook.save(env.state.output_path)

    response = env.step(Action(name="validate_workbook", kwargs={}))

    assert response.info["validation_ok"] is False
    assert response.info["formula_validation_errors"] == 1
    assert response.reward == 0.0
    assert "malformed formula Sheet1!A1" in response.observation.text


def test_validate_workbook_rejects_broken_formula_reference(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    workbook = openpyxl.load_workbook(env.state.output_path)
    workbook["Sheet1"]["A1"] = "=#REF!+1"
    workbook.save(env.state.output_path)

    response = env.step(Action(name="validate_workbook", kwargs={}))

    assert response.info["validation_ok"] is False
    assert response.info["formula_validation_errors"] == 1
    assert "broken #REF! reference" in response.observation.text


def test_validate_workbook_rejects_formula_text_wrapper(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    workbook = openpyxl.load_workbook(env.state.output_path)
    workbook["Sheet1"]["A1"] = '="=SUM(A1:A2)"'
    workbook.save(env.state.output_path)

    response = env.step(Action(name="validate_workbook", kwargs={}))

    assert response.info["validation_ok"] is False
    assert response.info["formula_validation_errors"] == 1
    assert "wrapped as formula text" in response.observation.text


def test_validate_workbook_rejects_formula_looking_text_cell(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    workbook = openpyxl.load_workbook(env.state.output_path)
    cell = workbook["Sheet1"]["A1"]
    cell.value = "=SUM(A1:A2)"
    cell.data_type = "s"
    workbook.save(env.state.output_path)

    response = env.step(Action(name="validate_workbook", kwargs={}))

    assert response.info["validation_ok"] is False
    assert response.info["formula_validation_errors"] == 1
    assert "stored as text" in response.observation.text


def test_validate_workbook_accepts_quoted_delimiters_in_formula(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    workbook = openpyxl.load_workbook(env.state.output_path)
    workbook["Sheet1"]["A1"] = '=IF(A2="(","{ok}","quoted ""text""")'
    workbook.save(env.state.output_path)

    response = env.step(Action(name="validate_workbook", kwargs={}))

    assert response.info["validation_ok"] is True
    assert response.info["formula_validation_errors"] == 0


def test_validate_workbook_ignores_formula_errors_outside_answer_ranges(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    workbook = openpyxl.load_workbook(env.state.output_path)
    workbook["Sheet1"]["C3"] = "=#REF!"
    workbook.save(env.state.output_path)

    response = env.step(Action(name="validate_workbook", kwargs={}))

    assert response.info["validation_ok"] is True
    assert response.info["formula_validation_errors"] == 0


def test_submit_gate_requires_current_validation(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._require_validation_before_submit = True

    rejected = env.step(Action(name="submit", kwargs={}))

    assert rejected.terminated is False
    assert rejected.reward == 0.0
    assert rejected.info["submit_rejected"] is True
    assert rejected.info["error"] == "submit_validation_required"

    validated = env.step(Action(name="validate_workbook", kwargs={}))
    assert validated.info["validation_ok"] is True
    submitted = env.step(Action(name="submit", kwargs={}))
    assert submitted.terminated is True
    assert submitted.info["submit_rejected"] is False


def test_submit_gate_rejects_stale_validation_after_edit(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    env._require_validation_before_submit = True
    assert env.step(Action(name="validate_workbook", kwargs={})).info[
        "validation_ok"
    ] is True
    assert env.step(Action(name="write_range", kwargs={
        "range": "A1", "data": "changed",
    })).info["tool_ok"] is True

    response = env.step(Action(name="submit", kwargs={}))

    assert response.terminated is False
    assert response.info["submit_rejected"] is True
    assert response.info["validation_stale"] is True


def test_recalc_with_formula_errors_does_not_satisfy_submit_gate(
    tmp_path: Path, monkeypatch,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    env._require_validation_before_submit = True
    monkeypatch.setattr(
        "envharness.bridges.spreadsheetbench.bridge."
        "execute_recalculate_and_read",
        lambda *args, **kwargs: ({
            "formula_error_cells": 1,
            "elapsed_ms": 1.0,
        }, "recalculated values include #REF!"),
    )

    recalculated = env.step(Action(name="recalculate_and_read", kwargs={
        "cell_ranges": ["Sheet1!A1"],
    }))
    submitted = env.step(Action(name="submit", kwargs={}))

    assert recalculated.info["tool_ok"] is False
    assert recalculated.info["tool_error"] == "formula_errors_found"
    assert env.state.extras.get("last_recalc_revision", -1) == -1
    assert submitted.info["submit_rejected"] is True


def test_validate_workbook_scans_large_range_sequentially(
    tmp_path: Path, monkeypatch,
) -> None:
    env = _execution_env(tmp_path)
    workbook = openpyxl.load_workbook(env.state.output_path)
    sheet = workbook["Sheet1"]
    sheet["J10411"] = "last"
    workbook.save(env.state.output_path)
    env.state.answer_position = "'Sheet1'!A1:J10411"

    from openpyxl.worksheet._read_only import ReadOnlyWorksheet

    def fail_random_access(self, key):
        raise AssertionError(f"read-only random cell access: {key}")

    monkeypatch.setattr(ReadOnlyWorksheet, "__getitem__", fail_random_access)
    response = env.step(Action(name="validate_workbook", kwargs={}))

    assert response.info["validation_ok"] is True
    assert "cells=104110, nonempty=4, formulas=1" in response.observation.text


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
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_basic")
    basic_names = {
        item["function"]["name"] for item in SpreadsheetBenchEnv.tool_schemas()
    }

    assert python_names == {"run_python", "validate_workbook", "submit"}
    assert native_names == python_names | {
        "list_sheets", "inspect_range", "find_cells"
    }
    assert basic_names == native_names | {
        "write_range", "clear_range", "fill_formula",
        "recalculate_and_read", "format_range", "delete_rows",
        "delete_columns", "manage_sheet",
    }
    write_schema = next(
        item for item in SpreadsheetBenchEnv.tool_schemas()
        if item["function"]["name"] == "write_range"
    )
    assert write_schema["function"]["parameters"]["required"] == ["range", "data"]
    assert write_schema["function"]["parameters"]["properties"]["data"]["type"] == [
        "array", "string", "number", "boolean"
    ]
    fill_schema = next(
        item for item in SpreadsheetBenchEnv.tool_schemas()
        if item["function"]["name"] == "fill_formula"
    )
    assert fill_schema["function"]["parameters"]["required"] == [
        "start_cell", "formula_template"
    ]


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


def test_native_basic_write_range_updates_current_workbook(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"

    response = env.step(Action(name="write_range", kwargs={
        "range": "C2:D3",
        "sheet_name": "Sheet1",
        "data": [[1, 2], [3, 4]],
    }))

    workbook = openpyxl.load_workbook(env.state.output_path, data_only=False)
    assert response.info["tool_ok"] is True
    assert response.info["tool_category"] == "write"
    assert workbook["Sheet1"]["C2"].value == 1
    assert workbook["Sheet1"]["D3"].value == 4


def test_native_basic_write_range_accepts_column_data(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"

    response = env.step(Action(name="write_range", kwargs={
        "range": "C2:C4",
        "sheet_name": "Sheet1",
        "data": [1, 2, 3],
    }))

    workbook = openpyxl.load_workbook(env.state.output_path, data_only=False)
    assert response.info["tool_ok"] is True
    assert [workbook["Sheet1"][f"C{row}"].value for row in range(2, 5)] == [
        1, 2, 3
    ]


def test_native_basic_fill_formula_translates_relative_and_absolute_refs(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"

    response = env.step(Action(name="fill_formula", kwargs={
        "sheet_name": "Sheet1",
        "start_cell": "C2",
        "end_row": 4,
        "end_col": "D",
        "formula_template": "=A2+$B$1+C$1+$A2",
    }))

    workbook = openpyxl.load_workbook(env.state.output_path, data_only=False)
    assert response.info["tool_ok"] is True
    assert response.info["tool_category"] == "write"
    assert workbook["Sheet1"]["C2"].value == "=A2+$B$1+C$1+$A2"
    assert workbook["Sheet1"]["D2"].value == "=B2+$B$1+D$1+$A2"
    assert workbook["Sheet1"]["C4"].value == "=A4+$B$1+C$1+$A4"


def test_native_basic_fill_formula_rejects_non_formula_without_mutation(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    before = Path(env.state.output_path).read_bytes()

    response = env.step(Action(name="fill_formula", kwargs={
        "start_cell": "C2",
        "end_row": 4,
        "formula_template": "SUM(A1:A2)",
    }))

    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "invalid_formula"
    assert Path(env.state.output_path).read_bytes() == before


def test_native_basic_fill_formula_rejects_malformed_formula_without_mutation(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    before = Path(env.state.output_path).read_bytes()

    response = env.step(Action(name="fill_formula", kwargs={
        "start_cell": "C2",
        "end_row": 4,
        "formula_template": "=SUM(A2:B2",
    }))

    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "invalid_formula"
    assert "unclosed '('" in response.observation.text
    assert Path(env.state.output_path).read_bytes() == before


def test_native_basic_fill_formula_save_failure_preserves_workbook(
    tmp_path: Path, monkeypatch,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    before = Path(env.state.output_path).read_bytes()

    def fail_save(workbook, path):
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(write_tools, "_atomic_save", fail_save)
    response = env.step(Action(name="fill_formula", kwargs={
        "start_cell": "C2",
        "end_row": 4,
        "formula_template": "=A2+B2",
    }))

    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "tool_execution_failed"
    assert Path(env.state.output_path).read_bytes() == before


def test_native_basic_fill_formula_rejects_oversized_range(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"

    response = env.step(Action(name="fill_formula", kwargs={
        "start_cell": "A1",
        "end_row": 300001,
        "formula_template": "=B1",
    }))

    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "range_too_large"


def test_fill_formula_rejects_formula_wrapped_as_string(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"

    response = env.step(Action(name="fill_formula", kwargs={
        "start_cell": "C2",
        "formula_template": '="=TEXT(B2, ""dddd"")"',
    }))

    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "invalid_formula"


def test_recalculate_and_read_uses_temporary_copy(
    monkeypatch, tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    before = Path(env.state.output_path).read_bytes()

    def fake_recalc(path, soffice_path=None, timeout=120):
        workbook = openpyxl.load_workbook(path)
        workbook["Sheet1"]["B2"] = 2
        workbook.save(path)
        return True

    monkeypatch.setattr(
        recalc_tools.online_judge_eval, "recalc_with_libreoffice", fake_recalc
    )
    response = env.step(Action(name="recalculate_and_read", kwargs={
        "cell_ranges": ["Sheet1!A1:B2"],
    }))

    payload = _tool_payload(response)
    assert response.info["tool_ok"] is True
    assert payload["ranges"][0]["values"][-1][-1] == 2
    assert Path(env.state.output_path).read_bytes() == before


def test_recalculate_and_read_is_bounded_to_one_call(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    env._recalc_calls = 1

    response = env.step(Action(name="recalculate_and_read", kwargs={
        "cell_ranges": ["A1"],
    }))

    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "recalc_limit_reached"


def test_run_python_preflight_rejects_cell_formula_assignment(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)

    response = env.step(Action(name="run_python", kwargs={
        "code": "wb = load_workbook_for_edit()\nwb.active['A1'].formula = '=1+1'",
    }))

    assert response.info["python_error"] is True
    assert response.info["python_error_type"] == "PreflightError"
    assert "cell.value" in response.observation.text


def test_format_range_updates_style_and_dimensions(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    workbook = openpyxl.load_workbook(env.state.output_path)
    workbook["Sheet1"]["A1"].font = openpyxl.styles.Font(
        name="Arial", italic=True
    )
    workbook.save(env.state.output_path)

    response = env.step(Action(name="format_range", kwargs={
        "sheet_name": "Sheet1",
        "range": "A1:B2",
        "font": {"bold": True, "color": "FF0000"},
        "fill": {"color": "FFFF00"},
        "alignment": {"horizontal": "center"},
        "number_format": "0.00",
        "column_width": 18,
        "row_height": 24,
    }))

    assert response.info["tool_ok"] is True
    workbook = openpyxl.load_workbook(env.state.output_path)
    assert workbook["Sheet1"]["A1"].font.bold is True
    assert workbook["Sheet1"]["A1"].font.italic is True
    assert workbook["Sheet1"]["A1"].font.name == "Arial"
    assert workbook["Sheet1"]["A1"].number_format == "0.00"
    assert workbook["Sheet1"].column_dimensions["A"].width == 18
    assert workbook["Sheet1"].row_dimensions[1].height == 24


def test_format_range_rejects_oversized_range_without_mutation(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    before = Path(env.state.output_path).read_bytes()

    response = env.step(Action(name="format_range", kwargs={
        "sheet_name": "Sheet1",
        "range": "A1:J10000",
        "font": {"name": "Times New Roman", "size": 12},
    }))

    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "range_too_large"
    assert "100000 cells" in response.observation.text
    assert Path(env.state.output_path).read_bytes() == before


def test_fill_formula_rejects_more_than_write_cell_limit_without_mutation(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    before = Path(env.state.output_path).read_bytes()

    response = env.step(Action(name="fill_formula", kwargs={
        "sheet_name": "Sheet1",
        "start_cell": "A1",
        "end_row": 50001,
        "formula_template": "=1+1",
    }))

    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "range_too_large"
    assert "maximum is 50000" in response.observation.text
    assert Path(env.state.output_path).read_bytes() == before


def test_delete_rows_rolls_back_on_invalid_range(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    before = Path(env.state.output_path).read_bytes()

    response = env.step(Action(name="delete_rows", kwargs={
        "sheet_name": "Sheet1", "rows": ["0:2"],
    }))

    assert response.info["tool_ok"] is False
    assert Path(env.state.output_path).read_bytes() == before


def test_delete_rows_rejects_out_of_bounds_without_mutation(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    before = Path(env.state.output_path).read_bytes()

    response = env.step(Action(name="delete_rows", kwargs={
        "sheet_name": "Sheet1", "rows": ["3:4"],
    }))

    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "range_out_of_bounds"
    assert Path(env.state.output_path).read_bytes() == before


def test_delete_rows_merges_overlapping_ranges_and_warns_about_formulas(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"

    response = env.step(Action(name="delete_rows", kwargs={
        "sheet_name": "Sheet1", "rows": ["2", "2:2"],
    }))

    payload = _tool_payload(response)
    assert response.info["tool_ok"] is True
    assert payload["deleted_ranges"] == ["2:2"]
    assert payload["formula_reference_update"] == "not_guaranteed_by_openpyxl"
    workbook = openpyxl.load_workbook(env.state.output_path)
    assert workbook["Sheet1"].max_row == 1


def test_manage_sheet_refuses_to_hide_only_visible_sheet(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"

    response = env.step(Action(name="manage_sheet", kwargs={
        "operation": "hide", "sheet_name": "Sheet1",
    }))

    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "cannot_hide_only_visible_sheet"


def test_manage_sheet_create_rename_copy_move_and_visibility(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"

    for arguments in (
        {"operation": "create", "sheet_name": "Work"},
        {"operation": "rename", "sheet_name": "Work", "new_name": "Result"},
        {"operation": "copy", "sheet_name": "Result", "new_name": "Result Copy"},
        {"operation": "move", "sheet_name": "Result Copy", "index": 0},
        {"operation": "hide", "sheet_name": "Result"},
        {"operation": "unhide", "sheet_name": "Hidden Data"},
    ):
        response = env.step(Action(name="manage_sheet", kwargs=arguments))
        assert response.info["tool_ok"] is True

    workbook = openpyxl.load_workbook(env.state.output_path)
    assert workbook.sheetnames[0] == "Result Copy"
    assert workbook["Result"].sheet_state == "hidden"
    assert workbook["Hidden Data"].sheet_state == "visible"


def test_submit_reports_recalc_staleness_after_later_edit(
    monkeypatch, tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"

    monkeypatch.setattr(
        recalc_tools.online_judge_eval,
        "recalc_with_libreoffice",
        lambda path, soffice_path=None, timeout=120: True,
    )
    recalc = env.step(Action(name="recalculate_and_read", kwargs={
        "cell_ranges": ["Sheet1!B2"],
    }))
    assert recalc.info["tool_ok"] is True
    write = env.step(Action(name="write_range", kwargs={
        "sheet_name": "Sheet1", "range": "A2", "data": "changed",
    }))
    assert write.info["tool_ok"] is True

    submitted = env.step(Action(name="submit", kwargs={}))

    assert submitted.info["submitted_after_recalc"] == 1
    assert submitted.info["recalc_stale_at_submit"] == 1


def test_native_basic_write_range_rejects_formulas_without_mutation(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    before = Path(env.state.output_path).read_bytes()

    response = env.step(Action(name="write_range", kwargs={
        "range": "A1",
        "data": "  =SUM(A2:A3)",
    }))

    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "formula_not_allowed"
    assert Path(env.state.output_path).read_bytes() == before


def test_native_basic_write_range_rejects_null_top_level_data(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"

    response = env.step(Action(name="write_range", kwargs={
        "range": "A1",
        "data": None,
    }))

    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "invalid_data"


def test_native_basic_write_range_runtime_failure_is_nonfatal(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"
    workbook = openpyxl.load_workbook(env.state.output_path)
    workbook["Sheet1"].merge_cells("C2:D2")
    workbook.save(env.state.output_path)

    response = env.step(Action(name="write_range", kwargs={
        "range": "D2",
        "data": "blocked",
    }))

    assert response.terminated is False
    assert response.info["tool_ok"] is False
    assert response.info["tool_error"] == "tool_execution_failed"


def test_native_basic_clear_range_clears_values_atomically(tmp_path: Path) -> None:
    env = _execution_env(tmp_path)
    env._tool_set = "native_basic"

    response = env.step(Action(name="clear_range", kwargs={
        "range": "A2:B2",
        "sheet_name": "Sheet1",
    }))

    workbook = openpyxl.load_workbook(env.state.output_path, data_only=False)
    assert response.info["tool_ok"] is True
    assert response.info["tool_category"] == "write"
    assert workbook["Sheet1"]["A2"].value is None
    assert workbook["Sheet1"]["B2"].value is None


def test_native_write_tools_are_disabled_in_native_read_mode(
    tmp_path: Path,
) -> None:
    env = _execution_env(tmp_path)

    response = env.step(Action(name="write_range", kwargs={
        "range": "A1",
        "data": "changed",
    }))

    assert response.info["error"] == "tool_disabled"
    workbook = openpyxl.load_workbook(env.state.output_path)
    assert workbook["Sheet1"]["A1"].value == "Name"
