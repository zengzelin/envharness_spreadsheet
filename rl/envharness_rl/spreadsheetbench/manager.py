"""verl-agent environment manager for SpreadsheetBench."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any
import uuid

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from envharness_rl.spreadsheetbench.projection import project_action_batch


_PYTHON_TOOL_INSTRUCTIONS = """You are solving a SpreadsheetBench task by editing the workbook with Python.
The Python runtime predefines input_path, output_path, working_directory,
load_workbook_for_edit(), and save_workbook(wb). load_workbook_for_edit() opens
the current output workbook, which is initially a copy of the input and retains
edits from earlier turns. Prefer openpyxl for workbook-preserving edits. Before
assuming a sheet or table layout, inspect workbook.sheetnames and dimensions.
Return one to four ordered JSON tool calls per turn, each wrapped in
<tool_call> tags. Calls execute sequentially in one episode turn. Put submit
last. If you
write a <think>...</think> reasoning block, put the tool call only after the
final </think>. Do not wrap the tool call in Markdown fences.

Run Python:
<tool_call>{\"name\":\"run_python\",\"arguments\":{\"code\":\"print('inspect')\"}}</tool_call>

Safe edit pattern:
<tool_call>{\"name\":\"run_python\",\"arguments\":{\"code\":\"wb = load_workbook_for_edit()\\nprint(wb.sheetnames)\\nws = wb[wb.sheetnames[0]]\\nws['A1'] = 'done'\\nsave_workbook(wb)\"}}</tool_call>

Check that the output workbook opens and contains the required sheets/ranges:
<tool_call>{\"name\":\"validate_workbook\",\"arguments\":{}}</tool_call>

Submit the finished output workbook:
<tool_call>{\"name\":\"submit\",\"arguments\":{}}</tool_call>

Python imports and local variables are stateless between calls; workbook edits
persist through output_path. Do not submit until output_path contains the final
answer."""

_NATIVE_READ_TOOL_INSTRUCTIONS = """

Structured read tools are enabled. They read the current output workbook, so
they include edits from earlier turns. Do not pass a file path.

Use list_sheets when the initial preview does not establish names or dimensions.
Do not repeat an identical successful read call:
<tool_call>{"name":"list_sheets","arguments":{}}</tool_call>

Inspect one finite A1 range. mode is cells or summary:
<tool_call>{"name":"inspect_range","arguments":{"sheet_name":"Sheet1","range":"A1:D20","mode":"cells"}}</tool_call>

Find text in values or formulas. match is contains, equals, or prefix; search_in
is values, formulas, or both; return_mode is first or all:
<tool_call>{"name":"find_cells","arguments":{"query":"Total","search_in":"values","return_mode":"all","max_results":20}}</tool_call>

Prefer these tools over run_python for workbook discovery. They are read-only;
use run_python for edits, then validate_workbook and submit as usual. Move from
inspection to editing as soon as the required layout is known."""

_NATIVE_BASIC_TOOL_INSTRUCTIONS = """

Structured basic write tools are enabled. They update the current output
workbook atomically and return a short result.

Write static values. Formula strings are rejected; null entries skip cells:
<tool_call>{"name":"write_range","arguments":{"sheet_name":"Sheet1","range":"A1:B2","data":[[1,2],[3,4]]}}</tool_call>

Clear values or formulas without shifting cells:
<tool_call>{"name":"clear_range","arguments":{"sheet_name":"Sheet1","range":"A1:B20"}}</tool_call>

Fill formulas from one template. Relative and mixed references are translated:
<tool_call>{"name":"fill_formula","arguments":{"sheet_name":"Sheet1","start_cell":"C2","end_row":100,"formula_template":"=A2+B2"}}</tool_call>

Use run_python for formatting, sorting, structural edits, or other
operations not covered by these tools."""


def _tool_instructions() -> str:
    tool_set = os.environ.get("SPREADSHEETBENCH_TOOL_SET", "python")
    normalized = tool_set.strip().lower().replace("-", "_")
    if normalized in {"native_read", "native_basic"}:
        instructions = _PYTHON_TOOL_INSTRUCTIONS + _NATIVE_READ_TOOL_INSTRUCTIONS
        if normalized == "native_basic":
            instructions += _NATIVE_BASIC_TOOL_INSTRUCTIONS
        return instructions
    return _PYTHON_TOOL_INSTRUCTIONS


class SpreadsheetBenchEnvironmentManager(EnvironmentManagerBase):
    """Format SpreadsheetBench observations and track short tool history."""

    def __init__(
        self,
        envs,
        projection_f,
        config,
        split: str = "train",
        trajectory_dir: str | Path | None = None,
    ) -> None:
        super().__init__(envs, projection_f, config)
        raw_trajectory_dir = trajectory_dir
        if raw_trajectory_dir is None:
            raw_trajectory_dir = os.environ.get(
                "SPREADSHEETBENCH_TRAJECTORY_DIR", ""
            )
        self._trajectory_dir = (
            Path(raw_trajectory_dir).expanduser() / split
            if raw_trajectory_dir
            else None
        )
        self._split = split
        self._history: list[list[tuple[str, str]]] = []
        self._last_observations: list[str] = []
        self._initial_observations: list[str] = []
        self._task_ids: list[str] = []
        self._trajectory_ids: list[str] = []
        self._trajectory_steps: list[list[dict[str, Any]]] = []
        self._trajectory_dumped: list[bool] = []
        self._history_mode = os.environ.get(
            "SPREADSHEETBENCH_HISTORY_MODE", "full"
        ).strip().lower()
        self._history_action_chars = int(
            os.environ.get("SPREADSHEETBENCH_HISTORY_ACTION_CHARS", "1200")
        )
        self._history_obs_chars = int(
            os.environ.get("SPREADSHEETBENCH_HISTORY_OBS_CHARS", "2000")
        )
        self._tool_instructions = _tool_instructions()
        self._max_steps = int(getattr(config.env, "max_steps", 10))

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {
                str(key): SpreadsheetBenchEnvironmentManager._jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                SpreadsheetBenchEnvironmentManager._jsonable(item)
                for item in value
            ]
        if hasattr(value, "item"):
            try:
                return SpreadsheetBenchEnvironmentManager._jsonable(value.item())
            except (TypeError, ValueError):
                pass
        if hasattr(value, "tolist"):
            try:
                return SpreadsheetBenchEnvironmentManager._jsonable(
                    value.tolist()
                )
            except (TypeError, ValueError):
                pass
        return str(value)

    def _dump_trajectory(self, index: int, final_info: dict[str, Any]) -> None:
        if self._trajectory_dir is None or self._trajectory_dumped[index]:
            return
        self._trajectory_dir.mkdir(parents=True, exist_ok=True)
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._task_ids[index])
        destination = self._trajectory_dir / (
            f"{safe_task_id}_{self._trajectory_ids[index]}.json"
        )
        payload = {
            "schema_version": 1,
            "split": self._split,
            "trajectory_id": self._trajectory_ids[index],
            "task_id": self._task_ids[index],
            "initial_observation": self._initial_observations[index],
            "steps": self._trajectory_steps[index],
            "final_info": self._jsonable(final_info),
        }
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(destination)
        self._trajectory_dumped[index] = True

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[truncated]"

    @classmethod
    def _step_diagnostics(
        cls,
        parser_diagnostic: dict[str, Any],
        env_info: dict[str, Any],
    ) -> dict[str, Any]:
        tool_results = env_info.get("tool_results") or []
        sub_infos = [
            item.get("info") or {}
            for item in tool_results if isinstance(item, dict)
        ]
        python_error = int(bool(
            env_info.get("python_error", False)
            or any(info.get("python_error", False) for info in sub_infos)
        ))
        syntax_error = int(bool(
            env_info.get("syntax_error", False)
            or any(info.get("syntax_error", False) for info in sub_infos)
        ))
        error_type = str(env_info.get("python_error_type") or "")
        if not error_type:
            error_type = next((
                str(info.get("python_error_type") or "")
                for info in sub_infos if info.get("python_error_type")
            ), "")
        tool_names = [
            str(item.get("action_name") or "")
            for item in tool_results if isinstance(item, dict)
        ]
        tool_name = str(env_info.get("tool_name") or "")
        parsed_tool_name = str(parser_diagnostic.get("tool_name") or "")
        action_name = tool_name or parsed_tool_name
        tool_category = str(env_info.get("tool_category") or "")
        tool_error = str(env_info.get("tool_error") or "")
        read_names = {"list_sheets", "inspect_range", "find_cells"}
        write_names = {"write_range", "clear_range", "fill_formula"}
        read_results = [
            item for item in tool_results
            if isinstance(item, dict) and (
                item.get("action_name") in read_names
                or (item.get("info") or {}).get("tool_category") == "read"
            )
        ]
        write_results = [
            item for item in tool_results
            if isinstance(item, dict) and (
                item.get("action_name") in write_names
                or (item.get("info") or {}).get("tool_category") == "write"
            )
        ]
        read_call = bool(
            read_results or tool_name and (
                tool_category == "read"
                or tool_name in read_names
            )
        )
        write_call = bool(
            write_results or tool_name and (
                tool_category == "write"
                or tool_name in write_names
            )
        )
        read_success = (
            all(bool(item.get("ok")) for item in read_results)
            if read_results else bool(env_info.get("tool_ok", False))
        )
        write_success = (
            all(bool(item.get("ok")) for item in write_results)
            if write_results else bool(env_info.get("tool_ok", False))
        )
        read_error = any(not bool(item.get("ok")) for item in read_results)
        write_error = any(not bool(item.get("ok")) for item in write_results)
        read_error_type = next((
            str((item.get("info") or {}).get("tool_error") or "")
            for item in read_results
            if (item.get("info") or {}).get("tool_error")
        ), tool_error if read_call else "")
        write_error_type = next((
            str((item.get("info") or {}).get("tool_error") or "")
            for item in write_results
            if (item.get("info") or {}).get("tool_error")
        ), tool_error if write_call else "")
        projected_call_count = int(
            env_info.get("tool_call_count", 1 if action_name else 0)
        )
        executed_call_count = int(
            env_info.get("tool_executed_count", projected_call_count)
        )
        result = {
            "parser/status": parser_diagnostic.get("status", "unknown"),
            "parser/native_valid": int(
                parser_diagnostic.get("native_valid", 0)
            ),
            "parser/recovered": int(parser_diagnostic.get("recovered", 0)),
            "parser/invalid": int(parser_diagnostic.get("invalid", 1)),
            "parser/error": parser_diagnostic.get("error", ""),
            "parser/tool_name": parser_diagnostic.get("tool_name", ""),
            "output/char_len": int(
                parser_diagnostic.get("output_char_len", 0)
            ),
            "output/has_think": int(parser_diagnostic.get("has_think", 0)),
            "output/has_markdown_fence": int(
                parser_diagnostic.get("has_markdown_fence", 0)
            ),
            "output/tool_call_pos": int(
                parser_diagnostic.get("tool_call_pos", -1)
            ),
            "env/python_error": python_error,
            "env/syntax_error": syntax_error,
            "env/python_runtime_error": int(
                bool(python_error and not syntax_error)
                and error_type != "TimeoutExpired"
            ),
            "env/python_timeout": int(error_type == "TimeoutExpired"),
            "env/python_error_type": error_type,
            "env/read_tool_call": int(read_call),
            "env/read_tool_success": int(
                read_call and read_success
            ),
            "env/read_tool_error": int(
                read_call and (read_error or bool(tool_error))
            ),
            "env/read_tool_error_type": read_error_type,
            "env/write_tool_call": int(write_call),
            "env/write_tool_success": int(
                write_call and write_success
            ),
            "env/write_tool_error": int(
                write_call and (write_error or bool(tool_error))
            ),
            "env/write_tool_error_type": write_error_type,
            "env/tool_error_type": write_error_type or read_error_type or tool_error,
            "episode/tool_calls_per_turn": executed_call_count,
            "episode/projected_tool_calls_per_turn": projected_call_count,
            "episode/multi_call": int(executed_call_count > 1),
            "episode/projected_multi_call": int(
                bool(env_info.get("multi_call", projected_call_count > 1))
            ),
            "env/multi_call_partial_failure": int(bool(
                env_info.get("multi_call_partial_failure", False)
            )),
        }
        for name in (
            "run_python", "list_sheets", "inspect_range", "find_cells",
            "write_range", "clear_range", "fill_formula",
            "validate_workbook", "submit",
        ):
            result[f"tool/{name}"] = int(
                action_name == name or name in tool_names
            )
        return result

    def _format_history_action(
        self,
        raw_action: str,
        projected_action: dict[str, Any] | list[dict[str, Any]] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> str:
        if self._history_mode != "compact":
            return raw_action
        projected_action = projected_action or {}
        diagnostics = diagnostics or {}
        if isinstance(projected_action, list):
            name = ",".join(
                str(item.get("name", "unknown")) for item in projected_action
            )
        else:
            name = projected_action.get("name", "unknown")
        status = diagnostics.get("parser/status", "unknown")
        output_chars = diagnostics.get("output/char_len", len(raw_action))
        parser_error = diagnostics.get("parser/error") or ""
        parts = [
            f"projected_action: {name}",
            f"parser_status: {status}",
            f"output_char_len: {output_chars}",
        ]
        if parser_error:
            parts.append(
                "parser_error: "
                + self._truncate_text(parser_error, self._history_action_chars)
            )
        return "\n".join(parts)

    @staticmethod
    def _task_seeds_from_kwargs(kwargs: Any) -> list[int] | None:
        if kwargs is None:
            return None
        if hasattr(kwargs, "tolist"):
            kwargs = kwargs.tolist()
        if isinstance(kwargs, dict):
            indexes = kwargs.get("task_index")
            if indexes is None:
                return None
            if hasattr(indexes, "tolist"):
                indexes = indexes.tolist()
            if not isinstance(indexes, (list, tuple)):
                indexes = [indexes]
            return [int(index) for index in indexes]
        if not isinstance(kwargs, (list, tuple)):
            raise ValueError(
                "validation env_kwargs must be a sequence of mappings"
            )
        task_seeds: list[int] = []
        for index, item in enumerate(kwargs):
            if hasattr(item, "item") and not isinstance(item, dict):
                try:
                    item = item.item()
                except (TypeError, ValueError):
                    pass
            if not isinstance(item, dict) or "task_index" not in item:
                raise ValueError(
                    f"validation env_kwargs[{index}] lacks task_index"
                )
            task_seeds.append(int(item["task_index"]))
        return task_seeds

    def reset(self, kwargs):
        task_seeds = (
            self._task_seeds_from_kwargs(kwargs)
            if self._split == "val"
            else None
        )
        text_obs, image_obs, infos = self.envs.reset(task_seeds=task_seeds)
        self._history = [[] for _ in text_obs]
        self._last_observations = list(text_obs)
        self._initial_observations = list(text_obs)
        self._task_ids = [str(info.get("task_id") or "unknown") for info in infos]
        self._trajectory_ids = [uuid.uuid4().hex for _ in text_obs]
        self._trajectory_steps = [[] for _ in text_obs]
        self._trajectory_dumped = [False for _ in text_obs]
        observations = {
            "text": self.build_text_obs(text_obs),
            "image": image_obs,
            "anchor": list(text_obs),
        }
        return observations, infos

    def step(self, text_actions: list[str]):
        action_batches = []
        parser_diagnostic_batches = []
        for model_output in text_actions:
            actions, diagnostics = project_action_batch(model_output)
            action_batches.append(actions)
            parser_diagnostic_batches.append(diagnostics)
        valids = [
            int(all(item.get("valid", 0) for item in diagnostics))
            for diagnostics in parser_diagnostic_batches
        ]
        parser_diagnostics = []
        for diagnostics in parser_diagnostic_batches:
            aggregate = dict(diagnostics[0])
            aggregate.update({
                "status": (
                    diagnostics[0].get("status", "unknown")
                    if len(diagnostics) == 1
                    else "native_tool_batch"
                    if all(item.get("valid", 0) for item in diagnostics)
                    else "invalid_tool_batch"
                ),
                "valid": int(all(item.get("valid", 0) for item in diagnostics)),
                "invalid": int(any(item.get("invalid", 1) for item in diagnostics)),
                "native_valid": int(all(
                    item.get("native_valid", 0) for item in diagnostics
                )),
                "recovered": int(any(item.get("recovered", 0) for item in diagnostics)),
                "tool_name": ",".join(
                    str(item.get("tool_name") or "") for item in diagnostics
                ),
                "error": "; ".join(
                    str(item.get("error"))
                    for item in diagnostics if item.get("error")
                ),
                "calls": diagnostics,
            })
            parser_diagnostics.append(aggregate)
        if hasattr(self.envs, "step_many"):
            text_obs, image_obs, rewards, dones, infos = self.envs.step_many(
                action_batches
            )
        else:
            if any(len(batch) != 1 for batch in action_batches):
                raise RuntimeError("environment does not support multi-call actions")
            text_obs, image_obs, rewards, dones, infos = self.envs.step(
                [batch[0] for batch in action_batches]
            )

        for index, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[index])
            info["tool_calling"] = sum(
                int(item.get("valid", 0))
                for item in parser_diagnostic_batches[index]
            )
            diagnostics = self._step_diagnostics(
                parser_diagnostics[index], info
            )
            info.update(diagnostics)
            if not bool(valids[index]):
                parser_error = parser_diagnostics[index].get(
                    "error", "Tool call parse error: invalid action."
                )
                info["parser_error"] = parser_error
                text_obs[index] = f"{parser_error}\n\n{str(text_obs[index] or '')}"

        for index, raw_action in enumerate(text_actions):
            action_valid = bool(valids[index])
            projected_payloads = [{
                "name": action.name,
                "kwargs": self._jsonable(action.kwargs),
            } for action in action_batches[index]]
            projected_payload: dict[str, Any] | list[dict[str, Any]] = (
                projected_payloads[0]
                if len(projected_payloads) == 1 else projected_payloads
            )
            diagnostics = self._step_diagnostics(
                parser_diagnostics[index], infos[index]
            )
            self._trajectory_steps[index].append({
                "step": len(self._trajectory_steps[index]) + 1,
                "observation": self._last_observations[index],
                "model_output": raw_action,
                "projected_action": projected_payload,
                "projected_actions": projected_payloads,
                "action_valid": action_valid,
                "diagnostics": diagnostics,
                "next_observation": text_obs[index],
                "reward": self._jsonable(rewards[index]),
                "done": bool(dones[index]),
                "info": self._jsonable(infos[index]),
            })
            self._history[index].append(
                (
                    self._last_observations[index],
                    self._format_history_action(
                        raw_action, projected_payload, diagnostics
                    ),
                )
            )
            if bool(dones[index]):
                self._dump_trajectory(index, infos[index])
        self._last_observations = list(text_obs)

        observations = {
            "text": self.build_text_obs(text_obs),
            "image": image_obs,
            "anchor": list(text_obs),
        }
        return observations, to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, text_obs: list[str]) -> list[str]:
        history_length = max(int(self.config.env.history_length), 0)
        prompts: list[str] = []
        for index, current_observation in enumerate(text_obs):
            history = self._history[index][-history_length:] if history_length else []
            history_text = ""
            if history:
                turns = []
                for turn, (observation, raw_action) in enumerate(history, start=1):
                    if self._history_mode == "compact":
                        observation = self._truncate_text(
                            observation, self._history_obs_chars
                        )
                    turns.append(
                        f"Turn {turn} observation:\n{observation}\n"
                        f"Turn {turn} tool call:\n{raw_action}"
                    )
                history_text = "\n\nRecent interaction history:\n" + "\n\n".join(turns)
            prompts.append(
                f"{self._tool_instructions}{history_text}\n\n"
                f"episode_step: {len(self._trajectory_steps[index])}\n"
                f"steps_remaining: {max(self._max_steps - len(self._trajectory_steps[index]), 0)}\n\n"
                f"Current environment observation:\n{current_observation}"
            )
        return prompts

    def success_evaluator(self, *args, **kwargs):
        metrics = super().success_evaluator(*args, **kwargs)
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        components = {
            "reward_workbook_score": [],
            "reward_execution_penalty": [],
            "reward_env_total": [],
        }
        info_keys = {
            "reward_workbook_score": "reward/workbook_score",
            "reward_execution_penalty": "reward/execution_penalty",
            "reward_env_total": "reward/env_total",
        }
        for batch_items, infos in zip(total_batch_list, total_infos):
            final_info = next(
                info
                for item, info in reversed(list(zip(batch_items, infos)))
                if item["active_masks"]
            )
            for metric_key, info_key in info_keys.items():
                fallback = final_info.get("score", 0.0) if metric_key == "reward_workbook_score" else 0.0
                components[metric_key].append(float(final_info.get(info_key, fallback)))
        metrics.update({
            key: np.asarray(values, dtype=np.float32)
            for key, values in components.items()
        })
        return metrics
