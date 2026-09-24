#!/usr/bin/env python3
"""Print aggregate metrics for saved SpreadsheetBench env trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from envharness_rl.spreadsheetbench.rollout_summary import (
    collect_multi_call_events,
    summarize_trajectories,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--multi-call-events-output", type=Path)
    args = parser.parse_args()
    trajectory_dir = args.run_dir / "rollouts" / "env" / args.split
    paths = sorted(trajectory_dir.glob("*.json"))
    summary = summarize_trajectories(paths)
    summary["split"] = args.split
    summary["trajectory_dir"] = str(trajectory_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.multi_call_events_output:
        args.multi_call_events_output.parent.mkdir(parents=True, exist_ok=True)
        events = collect_multi_call_events(paths)
        with args.multi_call_events_output.open("w", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
