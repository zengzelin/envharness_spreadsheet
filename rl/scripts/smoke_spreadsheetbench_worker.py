#!/usr/bin/env python
"""No-GPU smoke for the SpreadsheetBench verl-agent worker.

Run from the EnvHarness repository root:

  PYTHONPATH=.:rl python rl/scripts/smoke_spreadsheetbench_worker.py
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from envharness.core.types import Action
from envharness_rl.spreadsheetbench.envs import EnvharnessSpreadsheetWorker
from envharness_rl.spreadsheetbench.rollout_summary import (
    classify_badcase,
    collect_badcase_records,
    summarize_trajectories,
)


def main() -> None:
    data_path = os.environ.get("SPREADSHEETBENCH_DATA", "")
    if not data_path:
        raise SystemExit("set SPREADSHEETBENCH_DATA to a SpreadsheetBench dataset")

    worker = EnvharnessSpreadsheetWorker(
        seed=0,
        data_path=data_path,
        max_steps=2,
        reset_options={
            "badcase_diagnostics_mode": "full",
            "badcase_max_scan_cells": 200_000,
            "badcase_max_examples": 20,
        },
    )
    try:
        text, reset_info = worker.reset()
        assert text
        assert isinstance(reset_info["task_id"], str)
        assert reset_info["won"] is False

        _, reward, done, final_info = worker.step(
            Action(name="submit", kwargs={})
        )
        assert done is True
        assert isinstance(reward, float)
        assert isinstance(final_info["won"], bool)
        assert final_info["won"] is False
        assert final_info["task_id"] == reset_info["task_id"]
        diagnosis = classify_badcase(
            final_info=final_info,
            steps=[{
                "step": 1,
                "projected_action": {"name": "submit", "kwargs": {}},
                "action_valid": True,
                "done": True,
                "info": final_info,
            }],
        )
        assert diagnosis["eligible"] is True
        assert diagnosis["status"] == "complete"
        assert isinstance(diagnosis["failure_tags"], list)
        assert diagnosis["failure_tags"]

        with TemporaryDirectory(prefix="spreadsheet-badcase-smoke-") as temp:
            artifact_dir = Path(temp)
            trajectory_path = artifact_dir / "controlled_failure.json"
            payload = {
                "schema_version": 2,
                "split": "val",
                "task_id": final_info["task_id"],
                "steps": [{"step": 1, "done": True}],
                "final_info": {
                    "task_id": final_info["task_id"],
                    "won": False,
                    "score": float(final_info.get("score", 0.0)),
                    "badcase_diagnostics": diagnosis,
                },
            }
            trajectory_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            records = collect_badcase_records([trajectory_path])
            summary = summarize_trajectories([trajectory_path])
            assert len(records) == 1
            assert summary["badcase_complete_failure_count"] == 1
            assert not list(artifact_dir.rglob("*.xlsx"))
        print(
            "SPREADSHEETBENCH WORKER SMOKE OK: "
            f"task_id={final_info['task_id']} reward={reward} "
            f"won={final_info['won']} tags={diagnosis['failure_tags']}"
        )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
