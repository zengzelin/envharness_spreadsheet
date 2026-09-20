#!/usr/bin/env python3
"""Print aggregate metrics for saved SpreadsheetBench env trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from envharness_rl.spreadsheetbench.rollout_summary import summarize_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    args = parser.parse_args()
    trajectory_dir = args.run_dir / "rollouts" / "env" / args.split
    summary = summarize_trajectories(sorted(trajectory_dir.glob("*.json")))
    summary["split"] = args.split
    summary["trajectory_dir"] = str(trajectory_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
