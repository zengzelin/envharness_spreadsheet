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
