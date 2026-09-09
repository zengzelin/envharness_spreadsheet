from __future__ import annotations

import json
from pathlib import Path

from envharness.bridges.spreadsheetbench import dataset
from envharness.bridges.spreadsheetbench.bridge import SpreadsheetBenchEnv


def _write_task(root: Path, thread_dir: str, instruction: dict) -> None:
    task_dir = root / thread_dir
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.json").write_text(json.dumps(instruction), encoding="utf-8")
    (task_dir / "input.xlsx").write_bytes(b"input")
    (task_dir / "output.xlsx").write_bytes(b"output")
    (task_dir / "target.xlsx").write_bytes(b"target")


def test_spreadsheet_rl_loader_maps_parquet_split_to_sb_tasks(
    tmp_path: Path, monkeypatch
) -> None:
    _write_task(
        tmp_path,
        "excelforum/formulas/task-a",
        {
            "instruction": "Fill K2 with the tiered fee.",
            "type": "Formulas/Functions",
            "answer_position": "'Sheet1'!K2",
        },
    )
    parquet_path = tmp_path / "train_hermes.parquet"
    parquet_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        dataset,
        "_read_spreadsheet_rl_parquet_rows",
        lambda path: [
            {
                "reward_model": {"ground_truth": "excelforum/formulas/task-a"},
                "extra_info": {"id": "task-a", "split": "train"},
            }
        ],
        raising=False,
    )

    tasks = dataset.load_spreadsheet_rl_dataset(str(tmp_path), "train_hermes.parquet")

    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "task-a"
    assert task.instruction == "Fill K2 with the tiered fee."
    assert task.instruction_type == "Formulas/Functions"
    assert task.answer_position == "'Sheet1'!K2"
    assert task.folder == str(tmp_path / "excelforum/formulas/task-a")
    assert task.init_path == str(tmp_path / "excelforum/formulas/task-a/output.xlsx")
    assert task.golden_path == str(tmp_path / "excelforum/formulas/task-a/target.xlsx")


def test_select_task_uses_spreadsheet_rl_split_file(tmp_path: Path, monkeypatch) -> None:
    for name in ("task-a", "task-b"):
        _write_task(
            tmp_path,
            f"excelforum/formulas/{name}",
            {
                "instruction": name,
                "type": "Formulas/Functions",
                "answer_position": "A1",
            },
        )
    (tmp_path / "train_hermes.parquet").write_bytes(b"placeholder")
    monkeypatch.setattr(
        dataset,
        "_read_spreadsheet_rl_parquet_rows",
        lambda path: [
            {
                "reward_model": {"ground_truth": "excelforum/formulas/task-a"},
                "extra_info": {"id": "task-a"},
            },
            {
                "reward_model": {"ground_truth": "excelforum/formulas/task-b"},
                "extra_info": {"id": "task-b"},
            },
        ],
        raising=False,
    )

    task, index = dataset.select_task(
        str(tmp_path), seed=3, split_file="train_hermes.parquet"
    )

    assert index == 1
    assert task.id == "task-b"
    assert task.init_path.endswith("/task-b/output.xlsx")
    assert task.golden_path.endswith("/task-b/target.xlsx")


def test_spreadsheet_rl_task_ids_are_sanitized_for_sandbox_paths(
    tmp_path: Path, monkeypatch
) -> None:
    _write_task(
        tmp_path,
        "excelforum/excel-general/task-a",
        {
            "instruction": "Fill the sheet.",
            "type": "Formulas/Functions",
            "answer_position": "A1",
        },
    )
    (tmp_path / "train_hermes.parquet").write_bytes(b"placeholder")
    monkeypatch.setattr(
        dataset,
        "_read_spreadsheet_rl_parquet_rows",
        lambda path: [
            {
                "reward_model": {
                    "ground_truth": "excelforum/excel-general/task-a"
                },
                "extra_info": {"id": "excelforum/excel-general/task-a"},
            }
        ],
        raising=False,
    )
    monkeypatch.setattr(
        SpreadsheetBenchEnv,
        "_build_preview",
        lambda self, path: "preview",
    )

    env = SpreadsheetBenchEnv()
    try:
        response = env.reset(
            seed=0,
            options={
                "data_path": str(tmp_path),
                "split_file": "train_hermes.parquet",
                "sandbox_root": str(tmp_path),
            },
        )
    finally:
        env.close()

    assert response.info["task_id"] == "excelforum/excel-general/task-a"
    assert response.observation.data["input_path"].endswith(
        "excelforum_excel-general_task-a_input.xlsx"
    )
    assert response.observation.data["output_path"].endswith(
        "excelforum_excel-general_task-a_output.xlsx"
    )
