"""Project verl-agent text generations into SpreadsheetBench actions."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from envharness.core.types import Action


_START_TAG = "<tool_call>"
_END_TAG = "</tool_call>"
_THINK_START_TAG = "<think>"
_THINK_END_TAG = "</think>"
_FENCED_JSON_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_NATIVE_READ_TOOLS = frozenset({"list_sheets", "inspect_range", "find_cells"})
_NATIVE_WRITE_TOOLS = frozenset({"write_range", "clear_range", "fill_formula"})
_NATIVE_RECALC_TOOLS = frozenset({"recalculate_and_read"})
_NATIVE_STRUCTURE_TOOLS = frozenset({
    "format_range", "delete_rows", "delete_columns", "manage_sheet"
})
_CELL_RE = re.compile(
    r"^(?:(?:'[^']+'|[^!]+)!)?\$?([A-Za-z]{1,3})\$?([1-9]\d*)$"
)


def _tool_set() -> str:
    value = os.environ.get("SPREADSHEETBENCH_TOOL_SET", "python")
    return value.strip().lower().replace("-", "_")


def _native_read_enabled() -> bool:
    return _tool_set() in {"native_read", "native_basic"}


def _native_write_enabled() -> bool:
    return _tool_set() == "native_basic"


def _column_index(label: str) -> int:
    value = 0
    for character in label.upper():
        value = value * 26 + ord(character) - ord("A") + 1
    return value


def _fill_formula_error(arguments: dict[str, Any]) -> str | None:
    start_cell = arguments.get("start_cell")
    formula = arguments.get("formula_template")
    if not isinstance(start_cell, str) or not start_cell.strip():
        return "fill_formula requires a non-empty string 'start_cell' argument."
    match = _CELL_RE.fullmatch(start_cell.strip())
    if match is None:
        return "fill_formula start_cell must be one A1 cell."
    if not isinstance(formula, str) or not formula.startswith("="):
        return "fill_formula formula_template must start with '='."
    if formula.startswith(('="=', "='=")):
        return (
            "fill_formula formula_template looks like a quoted formula; use the "
            "Excel formula directly, for example '=A2+B2'."
        )
    start_col, start_row_text = match.groups()
    start_row = int(start_row_text)
    end_row = arguments.get("end_row", start_row)
    if isinstance(end_row, bool) or not isinstance(end_row, int) or end_row < start_row:
        return "fill_formula end_row must be at or below start_cell."
    end_col = arguments.get("end_col", start_col)
    if not isinstance(end_col, str) or not end_col.strip().replace("$", "").isalpha():
        return "fill_formula end_col must be a column label."
    if _column_index(end_col.strip().replace("$", "")) < _column_index(start_col):
        return "fill_formula end_col must be at or to the right of start_cell."
    return None


def _invalid_action(error: str = "Tool call parse error: invalid action") -> Action:
    return Action(name="invalid", kwargs={"error": error})


def _parse_region(model_output: str) -> tuple[str, str | None]:
    think_end = model_output.rfind(_THINK_END_TAG)
    if think_end >= 0:
        return model_output[think_end + len(_THINK_END_TAG) :], None
    if _THINK_START_TAG in model_output:
        return "", (
            "Tool call parse error: response is still in reasoning content "
            "and has no final '</think>'. Put one complete tool call after "
            "'</think>'."
        )
    return model_output, None


def _json_decode_object(candidate: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(candidate.strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _find_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _extract_payload(
    model_output: str,
) -> tuple[dict[str, Any] | None, str | None, str]:
    parse_region, region_error = _parse_region(model_output)
    if region_error is not None:
        return None, region_error, "unclosed_reasoning"

    start = parse_region.rfind(_START_TAG)
    if start >= 0:
        start += len(_START_TAG)
        end = parse_region.find(_END_TAG, start)
        if end < 0:
            return None, (
                "Tool call parse error: found '<tool_call>' but missing "
                "'</tool_call>'. Close the tool call block."
            ), "missing_tool_call_end"
        payload = _json_decode_object(parse_region[start:end])
        if payload is None:
            return None, (
                "Tool call parse error: invalid JSON inside "
                "'<tool_call>...</tool_call>'. Use JSON with keys 'name' and "
                "'arguments'."
            ), "invalid_json"
        return payload, None, "native_tool_call"

    for match in reversed(list(_FENCED_JSON_RE.finditer(parse_region))):
        payload = _find_json_object(match.group(1))
        if payload is not None:
            return payload, None, "fenced_json_recovered"

    payload = _find_json_object(parse_region)
    if payload is None:
        return None, "Tool call parse error: no JSON tool call found.", (
            "missing_tool_call"
        )
    return payload, None, "bare_json_recovered"


def _decode_arguments(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return raw if isinstance(raw, dict) else None


def _base_diagnostics(model_output: str) -> dict[str, Any]:
    text = str(model_output or "")
    return {
        "status": "unknown",
        "valid": 0,
        "native_valid": 0,
        "recovered": 0,
        "invalid": 1,
        "error": "",
        "tool_name": "",
        "output_char_len": len(text),
        "has_think": int(
            _THINK_START_TAG in text or _THINK_END_TAG in text
        ),
        "has_markdown_fence": int("```" in text),
        "tool_call_pos": text.rfind(_START_TAG),
    }


def _project_one_with_diagnostics(
    model_output: str,
) -> tuple[Action, int, dict[str, Any]]:
    text = str(model_output or "")
    diagnostics = _base_diagnostics(text)
    payload, error, status = _extract_payload(text)
    diagnostics["status"] = status
    if payload is None:
        diagnostics["error"] = error or "Tool call parse error: invalid action"
        return _invalid_action(diagnostics["error"]), 0, diagnostics

    name = payload.get("name", payload.get("tool"))
    diagnostics["tool_name"] = str(name or "")
    arguments = _decode_arguments(
        payload.get("arguments", payload.get("kwargs"))
    )
    if arguments is None:
        diagnostics["status"] = "invalid_arguments"
        diagnostics["error"] = (
            "Tool call parse error: 'arguments' must be a JSON object."
        )
        return _invalid_action(diagnostics["error"]), 0, diagnostics
    if not arguments and "code" in payload:
        arguments = {"code": payload.get("code")}

    if name in ("submit", "validate_workbook"):
        diagnostics["valid"] = 1
        diagnostics["invalid"] = 0
        diagnostics["native_valid"] = int(status == "native_tool_call")
        diagnostics["recovered"] = int(status.endswith("_recovered"))
        return Action(name=name, kwargs={}), 1, diagnostics
    if name in _NATIVE_READ_TOOLS:
        if not _native_read_enabled():
            diagnostics["status"] = "tool_disabled"
            diagnostics["error"] = (
                "Tool call parse error: structured read tools require "
                "SPREADSHEETBENCH_TOOL_SET=native_read."
            )
            return _invalid_action(diagnostics["error"]), 0, diagnostics
        if name == "list_sheets" and arguments:
            diagnostics["status"] = "invalid_arguments"
            diagnostics["error"] = (
                "Tool call parse error: list_sheets takes no arguments."
            )
            return _invalid_action(diagnostics["error"]), 0, diagnostics
        required_key = "range" if name == "inspect_range" else "query"
        if name != "list_sheets":
            required_value = arguments.get(required_key)
            if not isinstance(required_value, str) or not required_value.strip():
                diagnostics["status"] = f"missing_{required_key}"
                diagnostics["error"] = (
                    f"Tool call parse error: {name} requires a non-empty "
                    f"string '{required_key}' argument."
                )
                return _invalid_action(diagnostics["error"]), 0, diagnostics
        diagnostics["valid"] = 1
        diagnostics["invalid"] = 0
        diagnostics["native_valid"] = int(status == "native_tool_call")
        diagnostics["recovered"] = int(status.endswith("_recovered"))
        return Action(name=name, kwargs=arguments), 1, diagnostics
    if name in _NATIVE_WRITE_TOOLS | _NATIVE_RECALC_TOOLS | _NATIVE_STRUCTURE_TOOLS:
        if not _native_write_enabled():
            diagnostics["status"] = "tool_disabled"
            diagnostics["error"] = (
                "Tool call parse error: structured write tools require "
                "SPREADSHEETBENCH_TOOL_SET=native_basic."
            )
            return _invalid_action(diagnostics["error"]), 0, diagnostics
        required = {
            "write_range": {"range", "data"},
            "clear_range": {"range"},
            "fill_formula": {"start_cell", "formula_template"},
            "recalculate_and_read": {"cell_ranges"},
            "format_range": {"range"},
            "delete_rows": {"rows"},
            "delete_columns": {"columns"},
            "manage_sheet": {"operation", "sheet_name"},
        }
        missing = [key for key in required[name] if key not in arguments]
        if missing:
            diagnostics["status"] = "missing_arguments"
            diagnostics["error"] = (
                f"Tool call parse error: {name} requires "
                + ", ".join(sorted(missing))
                + "."
            )
            return _invalid_action(diagnostics["error"]), 0, diagnostics
        if name == "fill_formula":
            fill_error = _fill_formula_error(arguments)
            if fill_error is not None:
                diagnostics["status"] = "invalid_arguments"
                diagnostics["error"] = f"Tool call parse error: {fill_error}"
                return _invalid_action(diagnostics["error"]), 0, diagnostics
        range_value = arguments.get("range")
        if name in {"write_range", "clear_range", "format_range"} and (
            not isinstance(range_value, str) or not range_value.strip()
        ):
            diagnostics["status"] = "missing_range"
            diagnostics["error"] = (
                f"Tool call parse error: {name} requires a non-empty string "
                "'range' argument."
            )
            return _invalid_action(diagnostics["error"]), 0, diagnostics
        diagnostics["valid"] = 1
        diagnostics["invalid"] = 0
        diagnostics["native_valid"] = int(status == "native_tool_call")
        diagnostics["recovered"] = int(status.endswith("_recovered"))
        return Action(name=name, kwargs=arguments), 1, diagnostics
    if name != "run_python":
        diagnostics["status"] = "wrong_tool_name"
        diagnostics["error"] = (
            "Tool call parse error: unknown tool name. Use an enabled "
            "SpreadsheetBench tool."
        )
        return _invalid_action(diagnostics["error"]), 0, diagnostics

    code = arguments.get("code")
    if not isinstance(code, str) or not code.strip():
        diagnostics["status"] = "missing_code"
        diagnostics["error"] = (
            "Tool call parse error: run_python requires a non-empty string "
            "'code' argument."
        )
        return _invalid_action(diagnostics["error"]), 0, diagnostics
    diagnostics["valid"] = 1
    diagnostics["invalid"] = 0
    diagnostics["native_valid"] = int(status == "native_tool_call")
    diagnostics["recovered"] = int(status.endswith("_recovered"))
    return Action(name="run_python", kwargs={"code": code}), 1, diagnostics


def _project_one(model_output: str) -> tuple[Action, int]:
    action, valid, _ = _project_one_with_diagnostics(model_output)
    return action, valid


def project_action_batch(
    model_output: str, max_calls: int = 4,
) -> tuple[list[Action], list[dict[str, Any]]]:
    """Parse an ordered, bounded list of native tool calls from one turn."""
    text = str(model_output or "")
    parse_region, region_error = _parse_region(text)
    if region_error is not None:
        action, _, diagnostic = _project_one_with_diagnostics(text)
        diagnostic["call_index"] = 0
        return [action], [diagnostic]

    blocks: list[str] = []
    cursor = 0
    while True:
        start = parse_region.find(_START_TAG, cursor)
        if start < 0:
            break
        end = parse_region.find(_END_TAG, start + len(_START_TAG))
        if end < 0:
            blocks.append(parse_region[start:])
            break
        end += len(_END_TAG)
        blocks.append(parse_region[start:end])
        cursor = end

    if not blocks:
        action, _, diagnostic = _project_one_with_diagnostics(text)
        diagnostic["call_index"] = 0
        return [action], [diagnostic]
    if len(blocks) > max_calls:
        message = (
            f"Tool call parse error: {len(blocks)} calls exceed the "
            f"per-turn maximum of {max_calls}."
        )
        diagnostic = _base_diagnostics(text)
        diagnostic.update({
            "status": "too_many_tool_calls",
            "error": message,
            "call_index": 0,
        })
        return [_invalid_action(message)], [diagnostic]

    actions: list[Action] = []
    diagnostics: list[dict[str, Any]] = []
    for call_index, block in enumerate(blocks):
        action, _, diagnostic = _project_one_with_diagnostics(block)
        diagnostic["call_index"] = call_index
        diagnostic["output_char_len"] = len(text)
        diagnostic["has_think"] = int(
            _THINK_START_TAG in text or _THINK_END_TAG in text
        )
        diagnostic["has_markdown_fence"] = int("```" in text)
        actions.append(action)
        diagnostics.append(diagnostic)

    for call_index, action in enumerate(actions[:-1]):
        if action.name != "submit":
            continue
        message = "Tool call parse error: submit must be the final call in a batch."
        actions[call_index] = _invalid_action(message)
        diagnostics[call_index].update({
            "status": "submit_not_last",
            "valid": 0,
            "invalid": 1,
            "error": message,
        })
    for call_index, action in enumerate(actions[:-1]):
        if action.name != "recalculate_and_read":
            continue
        message = (
            "Tool call parse error: recalculate_and_read must be the final "
            "call in a batch so its values can be inspected before submit."
        )
        actions[call_index] = _invalid_action(message)
        diagnostics[call_index].update({
            "status": "recalc_not_last", "valid": 0, "invalid": 1,
            "error": message,
        })
    return actions, diagnostics


def envharness_spreadsheetbench_projection(
    actions: list[str],
) -> tuple[list[Action], list[int]]:
    """Parse one `<tool_call>` JSON object from each model generation."""
    projected: list[Action] = []
    valids: list[int] = []
    for model_output in actions:
        action, valid = _project_one(model_output)
        projected.append(action)
        valids.append(valid)
    return projected, valids


def envharness_spreadsheetbench_projection_diagnostics(
    actions: list[str],
) -> list[dict[str, Any]]:
    """Return parser/output diagnostics for SpreadsheetBench generations."""
    diagnostics: list[dict[str, Any]] = []
    for model_output in actions:
        _, _, diagnostic = _project_one_with_diagnostics(model_output)
        diagnostics.append(diagnostic)
    return diagnostics
