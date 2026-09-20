from __future__ import annotations

import json

import pytest

from envharness_rl.spreadsheetbench.projection import (
    envharness_spreadsheetbench_projection_diagnostics,
    envharness_spreadsheetbench_projection,
    project_action_batch,
)


def _tool_call(name: str, arguments) -> str:
    payload = json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False
    )
    return f"reasoning\n<tool_call>\n{payload}\n</tool_call>"


def test_projection_preserves_multiline_python_source() -> None:
    code = "from openpyxl import load_workbook\nName = 'Q3_TOTAL'\nprint(Name)"

    actions, valids = envharness_spreadsheetbench_projection(
        [_tool_call("run_python", {"code": code})]
    )

    assert valids == [1]
    assert actions[0].name == "run_python"
    assert actions[0].kwargs == {"code": code}


def test_projection_accepts_json_encoded_arguments() -> None:
    code = "print('CaseSensitive')"

    actions, valids = envharness_spreadsheetbench_projection(
        [_tool_call("run_python", json.dumps({"code": code}))]
    )

    assert valids == [1]
    assert actions[0].kwargs["code"] == code


def test_projection_accepts_validate_workbook() -> None:
    actions, valids = envharness_spreadsheetbench_projection(
        [_tool_call("validate_workbook", {})]
    )

    assert actions[0].name == "validate_workbook"
    assert actions[0].kwargs == {}
    assert valids == [1]


def test_projection_accepts_submit_without_arguments() -> None:
    actions, valids = envharness_spreadsheetbench_projection(
        ["<tool_call>{\"name\":\"submit\"}</tool_call>"]
    )

    assert valids == [1]
    assert actions[0].name == "submit"
    assert actions[0].kwargs == {}


def test_project_action_batch_parses_multiple_native_calls(monkeypatch) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_basic")
    model_output = "\n".join([
        _tool_call("list_sheets", {}),
        _tool_call("inspect_range", {"range": "A1:B3"}),
        _tool_call("fill_formula", {
            "start_cell": "C2", "end_row": 4, "formula_template": "=A2+B2",
        }),
        _tool_call("validate_workbook", {}),
    ])

    actions, diagnostics = project_action_batch(model_output)

    assert [action.name for action in actions] == [
        "list_sheets", "inspect_range", "fill_formula", "validate_workbook"
    ]
    assert [item["call_index"] for item in diagnostics] == [0, 1, 2, 3]
    assert all(item["valid"] == 1 for item in diagnostics)


def test_project_action_batch_rejects_more_than_four_calls() -> None:
    model_output = "\n".join(_tool_call("submit", {}) for _ in range(5))

    actions, diagnostics = project_action_batch(model_output)

    assert [action.name for action in actions] == ["invalid"]
    assert diagnostics[0]["status"] == "too_many_tool_calls"


def test_project_action_batch_preserves_invalid_call_index(monkeypatch) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_read")
    model_output = (
        _tool_call("list_sheets", {})
        + '\n<tool_call>{"name": bad json}</tool_call>\n'
        + _tool_call("inspect_range", {"range": "A1"})
    )

    actions, diagnostics = project_action_batch(model_output)

    assert [action.name for action in actions] == [
        "list_sheets", "invalid", "inspect_range"
    ]
    assert diagnostics[1]["call_index"] == 1
    assert diagnostics[1]["status"] == "invalid_json"


def test_project_action_batch_requires_submit_to_be_last() -> None:
    model_output = _tool_call("submit", {}) + "\n" + _tool_call(
        "validate_workbook", {}
    )

    actions, diagnostics = project_action_batch(model_output)

    assert actions[0].name == "invalid"
    assert diagnostics[0]["status"] == "submit_not_last"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("list_sheets", {}),
        ("inspect_range", {"range": "A1:B3", "mode": "cells"}),
        ("find_cells", {"query": "Total", "return_mode": "all"}),
    ],
)
def test_projection_accepts_native_read_tools(
    monkeypatch, name: str, arguments: dict,
) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_read")

    actions, valids = envharness_spreadsheetbench_projection([
        _tool_call(name, arguments)
    ])

    assert valids == [1]
    assert actions[0].name == name
    assert actions[0].kwargs == arguments


def test_projection_rejects_native_read_tools_in_python_tool_set(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "python")

    actions, valids = envharness_spreadsheetbench_projection([
        _tool_call("list_sheets", {})
    ])

    assert valids == [0]
    assert actions[0].name == "invalid"
    assert "native_read" in actions[0].kwargs["error"]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("write_range", {"range": "A1:B1", "data": [1, 2]}),
        ("clear_range", {"range": "A1:B2"}),
        ("fill_formula", {
            "start_cell": "J2",
            "end_row": 41,
            "formula_template": "=MOD(H2-G2,1)*24-I2/60",
        }),
    ],
)
def test_projection_accepts_native_basic_write_tools(
    monkeypatch, name: str, arguments: dict,
) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_basic")

    actions, valids = envharness_spreadsheetbench_projection([
        _tool_call(name, arguments)
    ])

    assert valids == [1]
    assert actions[0].name == name
    assert actions[0].kwargs == arguments


@pytest.mark.parametrize(
    "arguments",
    [
        {"start_cell": "A1"},
        {"start_cell": "A1", "formula_template": "SUM(B1:B2)"},
        {"start_cell": "A3", "end_row": 2, "formula_template": "=B3"},
    ],
)
def test_projection_rejects_invalid_fill_formula_arguments(
    monkeypatch, arguments: dict,
) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_basic")

    actions, valids = envharness_spreadsheetbench_projection([
        _tool_call("fill_formula", arguments)
    ])

    assert valids == [0]
    assert actions[0].name == "invalid"


def test_projection_rejects_native_write_tools_in_native_read_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_read")

    actions, valids = envharness_spreadsheetbench_projection([
        _tool_call("clear_range", {"range": "A1"})
    ])

    assert valids == [0]
    assert actions[0].name == "invalid"
    assert "native_basic" in actions[0].kwargs["error"]


def test_projection_parses_tool_call_after_final_think_close() -> None:
    code = "print('after thinking')"
    model_output = (
        "<think>draft plan</think>\n"
        + _tool_call("run_python", {"code": code})
    )

    actions, valids = envharness_spreadsheetbench_projection([model_output])

    assert valids == [1]
    assert actions[0].name == "run_python"
    assert actions[0].kwargs == {"code": code}


def test_projection_accepts_bare_json_after_final_think_close() -> None:
    code = "print('bare json')"
    model_output = (
        "<think>draft plan with <tool_call>example</tool_call></think>\n"
        + json.dumps({"name": "run_python", "arguments": {"code": code}})
    )

    actions, valids = envharness_spreadsheetbench_projection([model_output])

    assert valids == [1]
    assert actions[0].name == "run_python"
    assert actions[0].kwargs == {"code": code}


def test_projection_accepts_fenced_json_after_final_think_close() -> None:
    code = "print('fenced json')"
    model_output = (
        "<think>draft plan</think>\n"
        "```json\n"
        + json.dumps({"name": "run_python", "arguments": {"code": code}})
        + "\n```"
    )

    actions, valids = envharness_spreadsheetbench_projection([model_output])

    assert valids == [1]
    assert actions[0].name == "run_python"
    assert actions[0].kwargs == {"code": code}


def test_projection_diagnostics_classifies_recovered_fenced_json() -> None:
    code = "print('fenced json')"
    model_output = (
        "<think>draft plan</think>\n"
        "```json\n"
        + json.dumps({"name": "run_python", "arguments": {"code": code}})
        + "\n```"
    )

    diagnostics = envharness_spreadsheetbench_projection_diagnostics(
        [model_output]
    )

    assert diagnostics[0]["status"] == "fenced_json_recovered"
    assert diagnostics[0]["valid"] == 1
    assert diagnostics[0]["recovered"] == 1
    assert diagnostics[0]["native_valid"] == 0
    assert diagnostics[0]["has_markdown_fence"] == 1
    assert diagnostics[0]["has_think"] == 1


def test_projection_diagnostics_detects_thinking_from_closing_tag() -> None:
    diagnostics = envharness_spreadsheetbench_projection_diagnostics([
        "reasoning without visible opening tag</think>\n"
        '<tool_call>{"name":"submit"}</tool_call>'
    ])

    assert diagnostics[0]["has_think"] == 1


def test_projection_diagnostics_classifies_wrong_tool_name() -> None:
    diagnostics = envharness_spreadsheetbench_projection_diagnostics(
        [_tool_call("submit_output", {})]
    )

    assert diagnostics[0]["status"] == "wrong_tool_name"
    assert diagnostics[0]["valid"] == 0
    assert diagnostics[0]["tool_name"] == "submit_output"


def test_projection_accepts_legacy_tool_code_payload() -> None:
    code = "print('legacy payload')"
    model_output = (
        "<think>draft plan</think>\n"
        + json.dumps({"tool": "run_python", "code": code})
    )

    actions, valids = envharness_spreadsheetbench_projection([model_output])

    assert valids == [1]
    assert actions[0].name == "run_python"
    assert actions[0].kwargs == {"code": code}


def test_projection_uses_final_action_after_reasoning_examples() -> None:
    code = "print('final action')"
    model_output = (
        "<think>Example: <tool_call>{\"name\":\"submit\"}</tool_call>. "
        "Now solve.</think>\n"
        + json.dumps({"name": "run_python", "arguments": {"code": code}})
    )

    actions, valids = envharness_spreadsheetbench_projection([model_output])

    assert valids == [1]
    assert actions[0].name == "run_python"
    assert actions[0].kwargs == {"code": code}


def test_projection_uses_final_tool_call_after_reasoning_examples() -> None:
    actions, valids = envharness_spreadsheetbench_projection(
        [
            "<think>draft <tool_call>{\"name\":\"submit\"}</tool_call></think>\n"
            '<tool_call>{"name":"submit"}</tool_call>'
        ]
    )

    assert valids == [1]
    assert actions[0].name == "submit"
    assert actions[0].kwargs == {}


def test_projection_rejects_tool_call_inside_unclosed_reasoning() -> None:
    actions, valids = envharness_spreadsheetbench_projection(
        ['<think>draft <tool_call>{"name":"submit"}</tool_call>']
    )

    assert valids == [0]
    assert actions[0].name == "invalid"
    assert "reasoning" in actions[0].kwargs["error"]


def test_projection_reports_malformed_tool_call_error() -> None:
    actions, valids = envharness_spreadsheetbench_projection(
        ["</think>\n<tool_call>{bad json}</tool_call>"]
    )

    assert valids == [0]
    assert actions[0].name == "invalid"
    assert "invalid JSON" in actions[0].kwargs["error"]


@pytest.mark.parametrize(
    "model_output",
    [
        "no tool call",
        "<tool_call>{bad json}</tool_call>",
        _tool_call("delete_file", {}),
        _tool_call("run_python", {}),
        _tool_call("run_python", {"code": "   "}),
    ],
)
def test_projection_marks_malformed_or_unknown_calls_invalid(
    model_output: str,
) -> None:
    actions, valids = envharness_spreadsheetbench_projection([model_output])

    assert valids == [0]
    assert actions[0].name == "invalid"
    assert actions[0].kwargs["error"].startswith("Tool call parse error:")
