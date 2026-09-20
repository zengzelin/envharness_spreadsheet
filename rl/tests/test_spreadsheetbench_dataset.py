from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_select_task_only_materializes_selected_spreadsheet_rl_row(
    tmp_path: Path, monkeypatch
) -> None:
    _write_task(
        tmp_path,
        "excelforum/formulas/task-a",
        {
            "instruction": "selected task",
            "type": "Formulas/Functions",
            "answer_position": "A1",
        },
    )
    broken_dir = tmp_path / "excelforum/formulas/task-b"
    broken_dir.mkdir(parents=True)
    (broken_dir / "instruction.json").write_text("{broken", encoding="utf-8")
    (broken_dir / "output.xlsx").write_bytes(b"output")
    (broken_dir / "target.xlsx").write_bytes(b"target")
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
    )

    task, index = dataset.select_task(
        str(tmp_path), seed=0, split_file="train_hermes.parquet"
    )

    assert index == 0
    assert task.id == "task-a"


def test_spreadsheet_rl_rows_are_cached_across_seed_selection(
    tmp_path: Path, monkeypatch
) -> None:
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
    calls = 0

    def read_rows(path: str):
        nonlocal calls
        calls += 1
        return [
            {
                "reward_model": {"ground_truth": "excelforum/formulas/task-a"},
                "extra_info": {"id": "task-a"},
            },
            {
                "reward_model": {"ground_truth": "excelforum/formulas/task-b"},
                "extra_info": {"id": "task-b"},
            },
        ]

    monkeypatch.setattr(dataset, "_read_spreadsheet_rl_parquet_rows", read_rows)

    first, first_index = dataset.select_task(
        str(tmp_path), seed=0, split_file="train_hermes.parquet"
    )
    second, second_index = dataset.select_task(
        str(tmp_path), seed=1, split_file="train_hermes.parquet"
    )

    assert (first.id, first_index) == ("task-a", 0)
    assert (second.id, second_index) == ("task-b", 1)
    assert calls == 1


def test_select_spreadsheet_rl_task_matches_ground_truth_path_without_scanning_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    _write_task(
        tmp_path,
        "excelforum/formulas/task-a",
        {
            "instruction": "selected by path",
            "type": "Formulas/Functions",
            "answer_position": "A1",
        },
    )
    broken_dir = tmp_path / "excelforum/formulas/task-b"
    broken_dir.mkdir(parents=True)
    (broken_dir / "instruction.json").write_text("{broken", encoding="utf-8")
    (broken_dir / "output.xlsx").write_bytes(b"output")
    (broken_dir / "target.xlsx").write_bytes(b"target")
    (tmp_path / "test_hermes.parquet").write_bytes(b"placeholder")
    monkeypatch.setattr(
        dataset,
        "_read_spreadsheet_rl_parquet_rows",
        lambda path: [
            {
                "reward_model": {"ground_truth": "excelforum/formulas/task-a"},
                "extra_info": {},
            },
            {
                "reward_model": {"ground_truth": "excelforum/formulas/task-b"},
                "extra_info": {},
            },
        ],
    )

    task, index = dataset.select_task(
        str(tmp_path),
        seed=None,
        instance_id="excelforum/formulas/task-a",
        split_file="test_hermes.parquet",
    )

    assert index == 0
    assert task.instruction == "selected by path"


def test_select_spreadsheet_rl_task_rejects_ambiguous_extra_info_id(
    tmp_path: Path, monkeypatch
) -> None:
    for name in ("task-a", "task-b"):
        _write_task(
            tmp_path,
            f"excelforum/formulas/{name}",
            {"instruction": name, "type": "type", "answer_position": "A1"},
        )
    (tmp_path / "test_hermes.parquet").write_bytes(b"placeholder")
    monkeypatch.setattr(
        dataset,
        "_read_spreadsheet_rl_parquet_rows",
        lambda path: [
            {
                "reward_model": {"ground_truth": "excelforum/formulas/task-a"},
                "extra_info": {"id": "duplicate"},
            },
            {
                "reward_model": {"ground_truth": "excelforum/formulas/task-b"},
                "extra_info": {"id": "duplicate"},
            },
        ],
    )

    with pytest.raises(ValueError, match="ambiguous"):
        dataset.select_task(
            str(tmp_path),
            seed=None,
            instance_id="duplicate",
            split_file="test_hermes.parquet",
        )


def test_spreadsheet_rl_task_ids_are_sanitized_for_sandbox_paths(
    tmp_path: Path, monkeypatch, capsys
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
    output = capsys.readouterr().out
    assert "stage=select_task START" in output
    assert "stage=select_task END" in output
    assert "stage=prepare_sandbox START" in output
    assert "stage=prepare_sandbox END" in output
    assert "stage=build_preview START" in output
    assert "stage=build_preview END" in output
