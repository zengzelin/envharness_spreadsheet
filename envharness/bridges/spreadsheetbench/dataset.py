# Copyright 2026 The EnvHarness Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SpreadsheetBench dataset loader (verified_400 layout).

Self-contained: reads `dataset.json` and resolves, per task, the input
spreadsheet (`*_init.xlsx`), the ground-truth spreadsheet (`*_golden.xlsx`),
and the answer-position metadata. The loader is deliberately tolerant of the
quirks in the released data:

  - `id` is sometimes an int (e.g. 36236) and sometimes a string ("13-1");
    always coerced to str.
  - File naming is usually `{N}_{id}_init.xlsx` / `{N}_{id}_golden.xlsx` but a
    handful use `initial.xlsx` / `golden.xlsx`, and one task has a typo'd
    golden filename. We glob `*init*.xlsx` / `*golden*.xlsx` as a last resort.
  - Two tasks carry an `exclude` flag (known-bad goldens). They are KEPT in
    the ordered list by default so absolute indexing (train 0:100 /
    held-out 100:400) stays stable; the flag is surfaced on the task for any
    caller that wants to drop them.

The verified_400 set has exactly ONE test case per task, so OJ grading
reduces to a single output-vs-golden comparison at `answer_position`.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any

# Env var the Bridge falls back to when reset_options omits `data_path`.
DATA_PATH_ENV = "SPREADSHEETBENCH_DATA"


@dataclass(frozen=True)
class SBTask:
    """One SpreadsheetBench instance. Pure paths + metadata; no file handles."""
    id: str
    instruction: str
    instruction_type: str          # "Cell-Level Manipulation" | "Sheet-Level Manipulation"
    answer_position: str           # "B3" | "A3:D32" | "Sheet!A1:B2,Sheet!C1" ...
    answer_sheet: str = ""         # only populated for Sheet-Level tasks
    data_position: str = ""        # only populated for Sheet-Level tasks
    excluded: bool = False
    missing: bool = False          # spreadsheet folder absent on disk; placeholder row
    folder: str = ""
    init_path: str = ""
    golden_path: str = ""


def _norm_pos(s: str) -> str:
    """Normalize full-width punctuation in answer_position. A few (912) tasks
    use U+FF1A '：' / U+FF0C '，' / U+FF01 '！' instead of ASCII ':' ',' '!';
    openpyxl's range parser then raises 'not a valid coordinate or range',
    silently failing the task. Map them to ASCII."""
    return (str(s).replace("：", ":").replace("，", ",")
            .replace("！", "!").replace("；", ";"))


def _resolve_one(folder: str, id_: str, keyword: str,
                 extra_names: tuple[str, ...]) -> str:
    """Find the single xlsx matching `keyword` ("init" | "golden") in `folder`.

    Priority: `{N}_{id}_{keyword}.xlsx` -> explicit `extra_names` -> any
    `*{keyword}*.xlsx`. Returns "" if nothing matches.
    """
    names = sorted(os.listdir(folder))
    pat = re.compile(rf"^\d+_{re.escape(id_)}_{keyword}\.xlsx$")
    hits = [n for n in names if pat.match(n)]
    if hits:
        return os.path.join(folder, hits[0])
    for n in extra_names:
        if n in names:
            return os.path.join(folder, n)
    fuzzy = sorted(glob.glob(os.path.join(folder, f"*{keyword}*.xlsx")))
    return fuzzy[0] if fuzzy else ""


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_thread_dir(value: Any) -> str:
    """Normalize Spreadsheet-RL reward_model.ground_truth to a safe relative dir."""
    if not isinstance(value, (str, list, tuple)) and hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                "Spreadsheet-RL ground_truth must be a string or one-item list, "
                f"got {value!r}"
            )
        value = value[0]
    thread_dir = str(value or "").strip()
    if not thread_dir:
        raise ValueError("Spreadsheet-RL ground_truth is empty")
    posix_path = PurePosixPath(thread_dir)
    if posix_path.is_absolute() or any(part in ("", ".", "..") for part in posix_path.parts):
        raise ValueError(f"Unsafe Spreadsheet-RL ground_truth path: {thread_dir!r}")
    if "\x00" in thread_dir or "_workspaces" in posix_path.parts:
        raise ValueError(f"Unsafe Spreadsheet-RL ground_truth path: {thread_dir!r}")
    return "/".join(posix_path.parts)


def _read_spreadsheet_rl_parquet_rows(parquet_path: str) -> list[dict[str, Any]]:
    """Read Spreadsheet-RL parquet rows.

    Kept as a small helper so tests can monkeypatch it without requiring a
    local parquet engine. The cluster image normally has pandas/pyarrow.
    """
    try:
        import pandas as pd

        return pd.read_parquet(parquet_path).to_dict(orient="records")
    except ImportError:
        from datasets import Dataset

        return list(Dataset.from_parquet(parquet_path))


def _resolve_split_path(data_path: str, split_file: str) -> str:
    path = os.path.expanduser(str(split_file))
    if not os.path.isabs(path):
        path = os.path.join(data_path, path)
    return os.path.abspath(path)


@lru_cache(maxsize=8)
def load_dataset(data_path: str) -> tuple[SBTask, ...]:
    """Load and resolve every task under `data_path`.

    `data_path` is the directory holding `dataset.json` and `spreadsheet/`
    (i.e. `.../spreadsheetbench_verified_400`). Cached per path. Returns a
    tuple (immutable -> safe to cache and share across episodes).
    """
    ds_json = os.path.join(data_path, "dataset.json")
    if not os.path.isfile(ds_json):
        raise FileNotFoundError(
            f"SpreadsheetBench dataset.json not found at {ds_json!r}. Set "
            f"reset_options['data_path'] or ${DATA_PATH_ENV} to the directory "
            "that contains dataset.json + spreadsheet/."
        )
    with open(ds_json) as f:
        rows = json.load(f)

    sheet_root = os.path.join(data_path, "spreadsheet")
    tasks: list[SBTask] = []
    for row in rows:
        id_ = str(row["id"])
        folder = os.path.join(sheet_root, id_)
        missing = not os.path.isdir(folder)
        if missing:
            # Do NOT drop the entry -- dropping would shift every subsequent
            # index (reasoning_bank_eval's train/held-out split relies on absolute
            # indices, same reason exclude-flagged tasks are KEPT). Keep a
            # placeholder row flagged `missing`; select_task raises a clear
            # error if a missing task is actually selected.
            init_path = ""
            golden_path = ""
        else:
            init_path = _resolve_one(folder, id_, "init", ("initial.xlsx", "init.xlsx"))
            golden_path = _resolve_one(folder, id_, "golden", ("golden.xlsx",))
        tasks.append(SBTask(
            id=id_,
            instruction=str(row.get("instruction", "")),
            instruction_type=str(row.get("instruction_type", "")),
            answer_position=_norm_pos(row.get("answer_position", "")),
            answer_sheet=str(row.get("answer_sheet", "") or ""),
            data_position=str(row.get("data_position", "") or ""),
            excluded=bool(row.get("exclude", False)),
            missing=missing,
            folder=folder,
            init_path=init_path,
            golden_path=golden_path,
        ))
    return tuple(tasks)


@lru_cache(maxsize=8)
def load_dataset_multi(data_path: str) -> tuple[SBTask, ...]:
    """912-style loader: each task has N test cases (`{n}_{id}_input.xlsx` +
    `{n}_{id}_answer.xlsx`). Expanded into ONE SBTask per (task, test_case),
    id = `{base}#{n}`. The harder OJ metric groups these back by base id and
    requires ALL test cases to pass (avg_hard_score). Tasks/test-cases with a
    missing folder or answer file are skipped (grouping is by base id, not by
    absolute index, so skipping is safe here -- unlike the verified_400 loader)."""
    ds_json = os.path.join(data_path, "dataset.json")
    if not os.path.isfile(ds_json):
        raise FileNotFoundError(f"dataset.json not found at {ds_json!r}.")
    rows = json.load(open(ds_json))
    sheet_root = os.path.join(data_path, "spreadsheet")
    tasks: list[SBTask] = []
    for row in rows:
        id_ = str(row["id"])
        folder = os.path.join(sheet_root, id_)
        if not os.path.isdir(folder):
            continue
        inputs = sorted(glob.glob(os.path.join(folder, f"*_{id_}_input.xlsx")),
                        key=lambda p: int(os.path.basename(p).split("_", 1)[0]))
        for ip in inputs:
            n = os.path.basename(ip).split("_", 1)[0]
            ap = os.path.join(folder, f"{n}_{id_}_answer.xlsx")
            if not os.path.isfile(ap):
                continue
            tasks.append(SBTask(
                id=f"{id_}#{n}",
                instruction=str(row.get("instruction", "")),
                instruction_type=str(row.get("instruction_type", "")),
                answer_position=_norm_pos(row.get("answer_position", "")),
                answer_sheet=str(row.get("answer_sheet", "") or ""),
                data_position=str(row.get("data_position", "") or ""),
                excluded=bool(row.get("exclude", False)),
                missing=False, folder=folder,
                init_path=ip, golden_path=ap,
            ))
    return tuple(tasks)


@lru_cache(maxsize=8)
def load_spreadsheet_rl_dataset(data_path: str, split_file: str) -> tuple[SBTask, ...]:
    """Load a Spreadsheet-RL parquet split as SBTask rows.

    Spreadsheet-RL rows point at task directories through
    `reward_model.ground_truth`. In that layout, `output.xlsx` is the initial
    workbook copied into the agent workspace and `target.xlsx` is the oracle
    workbook used for reward.
    """
    data_path = os.path.abspath(os.path.expanduser(data_path))
    parquet_path = _resolve_split_path(data_path, split_file)
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(
            f"Spreadsheet-RL split file not found at {parquet_path!r}."
        )

    tasks: list[SBTask] = []
    for row_idx, row in enumerate(_read_spreadsheet_rl_parquet_rows(parquet_path)):
        reward_model = _coerce_mapping(row.get("reward_model"))
        extra_info = _coerce_mapping(row.get("extra_info"))
        thread_dir = _normalize_thread_dir(reward_model.get("ground_truth"))
        folder = os.path.join(data_path, thread_dir)
        instruction_path = os.path.join(folder, "instruction.json")
        metadata: dict[str, Any] = {}
        if os.path.isfile(instruction_path):
            with open(instruction_path, encoding="utf-8") as handle:
                metadata = json.load(handle)

        task_id = (
            extra_info.get("id")
            or metadata.get("id")
            or PurePosixPath(thread_dir).name
            or f"row-{row_idx}"
        )
        init_path = os.path.join(folder, "output.xlsx")
        golden_path = os.path.join(folder, "target.xlsx")
        missing = (
            not os.path.isdir(folder)
            or not os.path.isfile(instruction_path)
            or not os.path.isfile(init_path)
            or not os.path.isfile(golden_path)
        )
        tasks.append(SBTask(
            id=str(task_id),
            instruction=str(metadata.get("instruction", "")),
            instruction_type=str(
                metadata.get("instruction_type")
                or metadata.get("type")
                or extra_info.get("type")
                or ""
            ),
            answer_position=_norm_pos(
                metadata.get("answer_position")
                or extra_info.get("answer_position")
                or ""
            ),
            answer_sheet=str(
                metadata.get("answer_sheet")
                or extra_info.get("answer_sheet")
                or ""
            ),
            data_position=str(
                metadata.get("data_position")
                or extra_info.get("data_position")
                or ""
            ),
            excluded=False,
            missing=missing,
            folder=folder,
            init_path=init_path if os.path.isfile(init_path) else "",
            golden_path=golden_path if os.path.isfile(golden_path) else "",
        ))
    return tuple(tasks)


def base_id(instance_id: str) -> str:
    """'13-1#2' -> '13-1' (group key for the all-test-cases-pass hard metric)."""
    return str(instance_id).split("#", 1)[0]


def select_task(data_path: str, seed: int | None,
                instance_id: str | None = None,
                multi: bool = False,
                split_file: str | None = None) -> tuple[SBTask, int]:
    """Resolve (task, index) for an episode.

    Priority:
      1. `instance_id` (the SpreadsheetBench id string) -> exact match.
      2. `seed % len(tasks)` -> deterministic, alfworld/swebench-style.

    NOTE: the orchestrator injects `options["task_id"]` = the project-level
    OrchestratorConfig.task_id label (a string like "spreadsheetbench-corpus"),
    NOT an instance id. That value must NOT be used for selection; the per-task
    instance comes from `seed` (= ctx.task_id int). `instance_id` is the
    explicit hook used by the eval driver, which drives the env directly.
    """
    if split_file:
        tasks = load_spreadsheet_rl_dataset(data_path, split_file)
    else:
        tasks = load_dataset_multi(data_path) if multi else load_dataset(data_path)
    if not tasks:
        raise RuntimeError(f"No SpreadsheetBench tasks loaded from {data_path!r}.")
    if instance_id:
        for i, t in enumerate(tasks):
            if t.id == str(instance_id):
                _check_not_missing(t, i)
                return t, i
        raise ValueError(
            f"instance_id={instance_id!r} not found among {len(tasks)} tasks "
            f"in {data_path!r}."
        )
    idx = (int(seed) if seed is not None else 0) % len(tasks)
    _check_not_missing(tasks[idx], idx)
    return tasks[idx], idx


def _check_not_missing(task: SBTask, idx: int) -> None:
    if task.missing:
        raise RuntimeError(
            f"SpreadsheetBench task id={task.id!r} (index {idx}) is a "
            f"placeholder: its spreadsheet folder {task.folder!r} is missing "
            "on disk. The row is kept only to preserve absolute indexing; it "
            "cannot be run."
        )
