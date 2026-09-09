from __future__ import annotations

import json

import pytest

from envharness_rl.spreadsheetbench.projection import (
    envharness_spreadsheetbench_projection_diagnostics,
    envharness_spreadsheetbench_projection,
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
