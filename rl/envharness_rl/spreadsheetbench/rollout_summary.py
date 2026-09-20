"""Aggregate SpreadsheetBench environment trajectory diagnostics."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_trajectories(paths: Iterable[Path]) -> dict:
    tool_counts: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    episodes = successes = submitted = time_limit = 0
    steps = valid_actions = python_errors = syntax_errors = 0
    read_calls = read_successes = read_errors = 0
    write_calls = write_successes = write_errors = 0

    for path in paths:
        try:
            trajectory = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        episodes += 1
        final_info = trajectory.get("final_info") or {}
        successes += int(bool(final_info.get("won")))
        submitted += int(bool(final_info.get("submitted")))
        time_limit += int(bool(final_info.get("time_limit_reached")))
        for step in trajectory.get("steps") or []:
            steps += 1
            valid_actions += int(bool(step.get("action_valid")))
            action = step.get("projected_action") or {}
            tool_counts[str(action.get("name") or "unknown")] += 1
            diagnostics = step.get("diagnostics") or {}
            python_errors += int(bool(diagnostics.get("env/python_error")))
            syntax_errors += int(bool(diagnostics.get("env/syntax_error")))
            read_calls += int(bool(diagnostics.get("env/read_tool_call")))
            read_successes += int(bool(diagnostics.get("env/read_tool_success")))
            read_errors += int(bool(diagnostics.get("env/read_tool_error")))
            write_calls += int(bool(diagnostics.get("env/write_tool_call")))
            write_successes += int(bool(diagnostics.get("env/write_tool_success")))
            write_errors += int(bool(diagnostics.get("env/write_tool_error")))
            error_type = str(
                diagnostics.get("env/python_error_type")
                or diagnostics.get("env/write_tool_error_type")
                or diagnostics.get("env/read_tool_error_type")
                or ""
            )
            if error_type:
                error_types[error_type] += 1

    return {
        "episodes": episodes,
        "steps": steps,
        "success_rate": _ratio(successes, episodes),
        "submitted_rate": _ratio(submitted, episodes),
        "time_limit_rate": _ratio(time_limit, episodes),
        "valid_action_ratio": _ratio(valid_actions, steps),
        "python_error_ratio": _ratio(python_errors, steps),
        "syntax_error_ratio": _ratio(syntax_errors, steps),
        "read_tool_call_ratio": _ratio(read_calls, steps),
        "read_tool_success_rate": _ratio(read_successes, read_calls),
        "read_tool_error_rate": _ratio(read_errors, read_calls),
        "write_tool_call_ratio": _ratio(write_calls, steps),
        "write_tool_success_rate": _ratio(write_successes, write_calls),
        "write_tool_error_rate": _ratio(write_errors, write_calls),
        "tool_counts": dict(tool_counts.most_common()),
        "error_types": dict(error_types.most_common()),
    }
