from __future__ import annotations

import json
from pathlib import Path

from envharness_rl.spreadsheetbench.rollout_summary import summarize_trajectories


def test_summary_counts_tools_errors_submission_and_success(tmp_path: Path) -> None:
    trajectory = {
        "task_id": "task-1",
        "steps": [
            {
                "projected_action": {"name": "list_sheets", "kwargs": {}},
                "action_valid": True,
                "diagnostics": {
                    "env/read_tool_call": 1,
                    "env/read_tool_success": 1,
                    "env/python_error": 0,
                },
            },
            {
                "projected_action": {"name": "submit", "kwargs": {}},
                "action_valid": True,
                "diagnostics": {"env/python_error": 0},
            },
        ],
        "final_info": {"won": True, "submitted": True},
    }
    path = tmp_path / "task-1.json"
    path.write_text(json.dumps(trajectory), encoding="utf-8")

    summary = summarize_trajectories([path])

    assert summary["episodes"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["submitted_rate"] == 1.0
    assert summary["tool_counts"] == {"list_sheets": 1, "submit": 1}
    assert summary["read_tool_success_rate"] == 1.0


def test_summary_counts_each_action_in_multi_call_turn(tmp_path: Path) -> None:
    trajectory = {
        "task_id": "task-2",
        "steps": [{
            "projected_action": [
                {"name": "inspect_range", "kwargs": {"range": "A1:B2"}},
                {"name": "fill_formula", "kwargs": {"start_cell": "C2"}},
            ],
            "projected_actions": [
                {"name": "inspect_range", "kwargs": {"range": "A1:B2"}},
                {"name": "fill_formula", "kwargs": {"start_cell": "C2"}},
            ],
            "action_valid": True,
            "diagnostics": {
                "episode/tool_calls_per_turn": 2,
                "episode/multi_call": 1,
                "env/multi_call_partial_failure": 0,
            },
        }],
        "final_info": {"won": False},
    }
    path = tmp_path / "task-2.json"
    path.write_text(json.dumps(trajectory), encoding="utf-8")

    summary = summarize_trajectories([path])

    assert summary["tool_counts"] == {"inspect_range": 1, "fill_formula": 1}
    assert summary["tool_calls_per_turn"] == 2.0
    assert summary["multi_call_ratio"] == 1.0
    assert summary["multi_call_partial_failure_rate"] == 0.0


def test_summary_separates_projected_and_executed_tool_calls(tmp_path: Path) -> None:
    trajectory = {
        "task_id": "task-3",
        "steps": [{
            "projected_actions": [
                {"name": "run_python", "kwargs": {"code": "broken"}},
                {"name": "submit", "kwargs": {}},
            ],
            "action_valid": True,
            "diagnostics": {
                "episode/tool_calls_per_turn": 2,
                "episode/multi_call": 1,
                "env/multi_call_partial_failure": 1,
            },
            "info": {
                "tool_results": [
                    {"action_name": "run_python", "ok": False},
                ],
                "tool_call_count": 2,
                "tool_executed_count": 1,
            },
        }],
        "final_info": {"won": False},
    }
    path = tmp_path / "task-3.json"
    path.write_text(json.dumps(trajectory), encoding="utf-8")

    summary = summarize_trajectories([path])

    assert summary["tool_counts"] == {"run_python": 1}
    assert summary["projected_tool_counts"] == {
        "run_python": 1,
        "submit": 1,
    }
    assert summary["tool_calls_per_turn"] == 1.0
    assert summary["projected_tool_calls_per_turn"] == 2.0
    assert summary["multi_call_ratio"] == 0.0
    assert summary["projected_multi_call_ratio"] == 1.0
    assert summary["multi_call_partial_failure_rate"] == 1.0
