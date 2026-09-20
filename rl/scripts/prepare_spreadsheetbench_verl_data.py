#!/usr/bin/env python
"""Create the small text parquet files required by verl-agent, offline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def build_rows(split: str, size: int) -> list[dict[str, Any]]:
    if size < 1:
        raise ValueError("size must be positive")
    return [
        {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": split, "index": index},
            "env_kwargs": {"split": split, "task_index": index},
        }
        for index in range(size)
    ]


def write_parquet(output_dir: Path, train_size: int, val_size: int) -> None:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            "The verl-agent environment needs the 'datasets' package"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(build_rows("train", train_size)).to_parquet(
        str(output_dir / "train.parquet")
    )
    Dataset.from_list(build_rows("test", val_size)).to_parquet(
        str(output_dir / "test.parquet")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--val-size", type=int, required=True)
    args = parser.parse_args()
    write_parquet(args.output_dir, args.train_size, args.val_size)
    print(f"[spreadsheet-data] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
