"""Aggregate SpreadsheetBench environment trajectory diagnostics."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _projected_actions(step: dict) -> list[dict]:
    actions = step.get("projected_actions")
    if not isinstance(actions, list):
        action = step.get("projected_action") or {}
        actions = action if isinstance(action, list) else [action]
    return [action for action in actions if isinstance(action, dict)]


def collect_multi_call_events(paths: Iterable[Path]) -> list[dict]:
    """Return task-level diagnostics for every projected multi-call turn."""
    events: list[dict] = []
    for path in paths:
        try:
            trajectory = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        task_id = str(trajectory.get("task_id") or "unknown")
        for step_index, step in enumerate(trajectory.get("steps") or [], 1):
            projected_actions = _projected_actions(step)
            if len(projected_actions) <= 1:
                continue
            info = step.get("info") or {}
            diagnostics = step.get("diagnostics") or {}
            tool_results = info.get("tool_results")
            if not isinstance(tool_results, list):
                tool_results = []
            tool_results = [
                result for result in tool_results if isinstance(result, dict)
            ]
            executed_count = len(tool_results)
            if not tool_results:
                executed_count = int(info.get(
                    "tool_executed_count", len(projected_actions)
                ))
            skipped_count = int(diagnostics.get(
                "env/multi_call_skipped_calls",
                info.get(
                    "tool_skipped_count",
                    max(len(projected_actions) - executed_count, 0),
                ),
            ))
            failure_result = next((
                result for result in tool_results if not result.get("ok")
            ), None)
            failure_index = int(diagnostics.get(
                "env/multi_call_failure_index",
                info.get(
                    "multi_call_failure_index",
                    (failure_result or {}).get("action_index", -1),
                ),
            ))
            failure_info = (failure_result or {}).get("info") or {}
            events.append({
                "task_id": task_id,
                "trajectory": str(path),
                "step": int(step.get("step") or step_index),
                "projected_tools": [
                    str(action.get("name") or "unknown")
                    for action in projected_actions
                ],
                "executed_tools": [
                    str(result.get("action_name") or "unknown")
                    for result in tool_results
                ],
                "projected_count": len(projected_actions),
                "executed_count": executed_count,
                "skipped_count": skipped_count,
                "completed": bool(diagnostics.get(
                    "env/multi_call_completed",
                    info.get(
                        "multi_call_completed",
                        executed_count == len(projected_actions),
                    ),
                )),
                "all_success": bool(diagnostics.get(
                    "env/multi_call_all_success",
                    info.get(
                        "multi_call_all_success",
                        bool(tool_results)
                        and executed_count == len(projected_actions)
                        and all(result.get("ok") for result in tool_results),
                    ),
                )),
                "short_circuit": bool(diagnostics.get(
                    "env/multi_call_short_circuit",
                    info.get("multi_call_short_circuit", skipped_count > 0),
                )),
                "partial_failure": bool(diagnostics.get(
                    "env/multi_call_partial_failure",
                    info.get("multi_call_partial_failure", failure_result is not None),
                )),
                "stop_reason": str(
                    info.get("multi_call_stop_reason")
                    or diagnostics.get("env/multi_call_stop_reason")
                    or "unknown"
                ),
                "failure_index": failure_index,
                "failure_tool": str(
                    (failure_result or {}).get("action_name") or ""
                ),
                "failure_error_type": str(
                    failure_info.get("tool_error")
                    or failure_info.get("python_error_type")
                    or ""
                ),
            })
    return events


def summarize_trajectories(paths: Iterable[Path]) -> dict:
    tool_counts: Counter[str] = Counter()
    projected_tool_counts: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    episodes = successes = submitted = time_limit = 0
    steps = valid_actions = python_errors = syntax_errors = 0
    read_calls = read_successes = read_errors = 0
    write_calls = write_successes = write_errors = 0
    write_call_count = write_call_successes = 0
    episodes_with_write_error = 0
    tool_calls = projected_tool_calls = 0
    multi_call_steps = projected_multi_call_steps = 0
    multi_call_partial_failures = 0
    multi_call_completed = multi_call_all_success = 0
    multi_call_short_circuits = multi_call_skipped_calls = 0
    multi_call_executed_fraction_sum = 0.0
    multi_call_stop_reasons: Counter[str] = Counter()
    multi_call_failure_indexes: Counter[str] = Counter()

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
        episode_has_write_error = False
        for step in trajectory.get("steps") or []:
            steps += 1
            valid_actions += int(bool(step.get("action_valid")))
            projected_actions = _projected_actions(step)
            for action in projected_actions:
                if isinstance(action, dict):
                    projected_tool_counts[str(action.get("name") or "unknown")] += 1

            info = step.get("info") or {}
            tool_results = info.get("tool_results")
            if isinstance(tool_results, list):
                executed_actions = [
                    {"name": result.get("action_name")}
                    for result in tool_results if isinstance(result, dict)
                ]
            else:
                # Trajectories written before multi-call execution diagnostics
                # did not record tool_results, so projected calls are the best
                # available approximation for those files.
                executed_actions = projected_actions
            write_names = {"write_range", "clear_range", "fill_formula"}
            if isinstance(tool_results, list):
                write_results = [
                    result for result in tool_results
                    if isinstance(result, dict) and (
                        result.get("action_name") in write_names
                        or (result.get("info") or {}).get("tool_category") == "write"
                    )
                ]
                write_call_count += len(write_results)
                write_call_successes += sum(
                    int(bool(result.get("ok"))) for result in write_results
                )
                episode_has_write_error |= any(
                    not bool(result.get("ok")) for result in write_results
                )
            for action in executed_actions:
                if isinstance(action, dict):
                    tool_counts[str(action.get("name") or "unknown")] += 1

            diagnostics = step.get("diagnostics") or {}
            calls_this_turn = len(executed_actions)
            projected_calls_this_turn = len(projected_actions)
            tool_calls += calls_this_turn
            projected_tool_calls += projected_calls_this_turn
            multi_call_steps += int(calls_this_turn > 1)
            is_projected_multi_call = projected_calls_this_turn > 1
            projected_multi_call_steps += int(is_projected_multi_call)
            multi_call_partial_failures += int(bool(
                diagnostics.get("env/multi_call_partial_failure")
            ))
            if is_projected_multi_call:
                completed = bool(diagnostics.get(
                    "env/multi_call_completed",
                    calls_this_turn == projected_calls_this_turn,
                ))
                all_success = bool(diagnostics.get(
                    "env/multi_call_all_success",
                    completed and isinstance(tool_results, list)
                    and all(bool(result.get("ok")) for result in tool_results),
                ))
                skipped_calls = int(diagnostics.get(
                    "env/multi_call_skipped_calls",
                    max(projected_calls_this_turn - calls_this_turn, 0),
                ))
                short_circuit = bool(diagnostics.get(
                    "env/multi_call_short_circuit", skipped_calls > 0
                ))
                executed_fraction = float(diagnostics.get(
                    "env/multi_call_executed_fraction",
                    calls_this_turn / projected_calls_this_turn,
                ))
                failure_index = int(diagnostics.get(
                    "env/multi_call_failure_index",
                    next((
                        int(result.get("action_index", index))
                        for index, result in enumerate(tool_results or [])
                        if isinstance(result, dict) and not result.get("ok")
                    ), -1),
                ))
                stop_reason = str(
                    info.get("multi_call_stop_reason")
                    or diagnostics.get("env/multi_call_stop_reason")
                    or "unknown"
                )
                multi_call_completed += int(completed)
                multi_call_all_success += int(all_success)
                multi_call_short_circuits += int(short_circuit)
                multi_call_skipped_calls += skipped_calls
                multi_call_executed_fraction_sum += executed_fraction
                multi_call_stop_reasons[stop_reason] += 1
                if failure_index >= 0:
                    multi_call_failure_indexes[str(failure_index)] += 1
            python_errors += int(bool(diagnostics.get("env/python_error")))
            syntax_errors += int(bool(diagnostics.get("env/syntax_error")))
            read_calls += int(bool(diagnostics.get("env/read_tool_call")))
            read_successes += int(bool(diagnostics.get("env/read_tool_success")))
            read_errors += int(bool(diagnostics.get("env/read_tool_error")))
            write_calls += int(bool(diagnostics.get("env/write_tool_call")))
            write_successes += int(bool(diagnostics.get("env/write_tool_success")))
            write_errors += int(bool(diagnostics.get("env/write_tool_error")))
            if not isinstance(tool_results, list) and diagnostics.get(
                "env/write_tool_call"
            ):
                write_call_count += 1
                write_call_successes += int(bool(
                    diagnostics.get("env/write_tool_success")
                ))
            episode_has_write_error |= bool(
                diagnostics.get("env/write_tool_error")
            )
            error_type = str(
                diagnostics.get("env/python_error_type")
                or diagnostics.get("env/write_tool_error_type")
                or diagnostics.get("env/read_tool_error_type")
                or ""
            )
            if error_type:
                error_types[error_type] += 1
        episodes_with_write_error += int(episode_has_write_error)

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
        "write_turn_all_success_rate": _ratio(write_successes, write_calls),
        "write_call_count": write_call_count,
        "write_call_success_rate": _ratio(
            write_call_successes, write_call_count
        ),
        "episodes_with_write_error_rate": _ratio(
            episodes_with_write_error, episodes
        ),
        "write_tool_error_rate": _ratio(write_errors, write_calls),
        "tool_calls_per_turn": _ratio(tool_calls, steps),
        "projected_tool_calls_per_turn": _ratio(projected_tool_calls, steps),
        "multi_call_ratio": _ratio(multi_call_steps, steps),
        "projected_multi_call_ratio": _ratio(projected_multi_call_steps, steps),
        "multi_call_projected_turns": projected_multi_call_steps,
        "multi_call_executed_turns": multi_call_steps,
        "multi_call_completed_rate": _ratio(
            multi_call_completed, projected_multi_call_steps
        ),
        "multi_call_all_success_rate": _ratio(
            multi_call_all_success, projected_multi_call_steps
        ),
        "multi_call_short_circuit_rate": _ratio(
            multi_call_short_circuits, projected_multi_call_steps
        ),
        "multi_call_mean_executed_fraction": _ratio(
            multi_call_executed_fraction_sum, projected_multi_call_steps
        ),
        "multi_call_skipped_call_count": multi_call_skipped_calls,
        "multi_call_partial_failure_rate": _ratio(
            multi_call_partial_failures, projected_multi_call_steps
        ),
        "multi_call_stop_reasons": dict(multi_call_stop_reasons.most_common()),
        "multi_call_failure_indexes": dict(
            multi_call_failure_indexes.most_common()
        ),
        "tool_counts": dict(tool_counts.most_common()),
        "projected_tool_counts": dict(projected_tool_counts.most_common()),
        "error_types": dict(error_types.most_common()),
    }
