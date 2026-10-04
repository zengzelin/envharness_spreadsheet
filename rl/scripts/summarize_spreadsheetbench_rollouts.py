#!/usr/bin/env python3
"""Print aggregate metrics for saved SpreadsheetBench env trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from envharness_rl.spreadsheetbench.rollout_summary import (
    collect_badcase_records,
    collect_multi_call_events,
    summarize_trajectories,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--multi-call-events-output", type=Path)
    parser.add_argument("--badcase-jsonl-output", type=Path)
    parser.add_argument("--badcase-summary-output", type=Path)
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
    if args.badcase_jsonl_output:
        args.badcase_jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        records = collect_badcase_records(paths)
        with args.badcase_jsonl_output.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.badcase_summary_output:
        args.badcase_summary_output.parent.mkdir(parents=True, exist_ok=True)
        badcase_summary = {
            key: value for key, value in summary.items()
            if key.startswith("badcase_")
            or key.endswith("_rate")
            or key == "answer_match_ratio_failed_mean"
        }
        args.badcase_summary_output.write_text(
            json.dumps(badcase_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
