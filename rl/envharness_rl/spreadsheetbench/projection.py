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


def _native_read_enabled() -> bool:
    value = os.environ.get("SPREADSHEETBENCH_TOOL_SET", "python")
    return value.strip().lower().replace("-", "_") == "native_read"


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
