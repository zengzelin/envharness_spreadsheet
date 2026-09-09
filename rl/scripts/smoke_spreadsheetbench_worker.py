#!/usr/bin/env python
"""No-GPU smoke for the SpreadsheetBench verl-agent worker.

Run from the EnvHarness repository root:

  PYTHONPATH=.:rl python rl/scripts/smoke_spreadsheetbench_worker.py
"""
from __future__ import annotations

import os

from envharness.core.types import Action
from envharness_rl.spreadsheetbench.envs import EnvharnessSpreadsheetWorker


def main() -> None:
    data_path = os.environ.get("SPREADSHEETBENCH_DATA", "")
    if not data_path:
        raise SystemExit("set SPREADSHEETBENCH_DATA to a SpreadsheetBench dataset")

    worker = EnvharnessSpreadsheetWorker(
        seed=0,
        data_path=data_path,
        max_steps=2,
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
        assert final_info["task_id"] == reset_info["task_id"]
        print(
            "SPREADSHEETBENCH WORKER SMOKE OK: "
            f"task_id={final_info['task_id']} reward={reward} "
            f"won={final_info['won']}"
        )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
