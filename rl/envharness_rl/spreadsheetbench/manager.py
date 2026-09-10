"""verl-agent environment manager for SpreadsheetBench."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any
import uuid

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from envharness_rl.spreadsheetbench.projection import (
    envharness_spreadsheetbench_projection_diagnostics,
)


_PYTHON_TOOL_INSTRUCTIONS = """You are solving a SpreadsheetBench task by editing the workbook with Python.
The Python runtime predefines input_path, output_path, working_directory,
load_workbook_for_edit(), and save_workbook(wb). load_workbook_for_edit() opens
the current output workbook, which is initially a copy of the input and retains
edits from earlier turns. Prefer openpyxl for workbook-preserving edits. Before
assuming a sheet or table layout, inspect workbook.sheetnames and dimensions.
Return exactly one JSON tool call per turn, wrapped in <tool_call> tags. If you
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

List worksheets before assuming their names or dimensions:
<tool_call>{"name":"list_sheets","arguments":{}}</tool_call>

Inspect one finite A1 range. mode is cells or summary:
<tool_call>{"name":"inspect_range","arguments":{"sheet_name":"Sheet1","range":"A1:D20","mode":"cells"}}</tool_call>

Find text in values or formulas. match is contains, equals, or prefix; search_in
is values, formulas, or both; return_mode is first or all:
<tool_call>{"name":"find_cells","arguments":{"query":"Total","search_in":"values","return_mode":"all","max_results":20}}</tool_call>

Prefer these tools over run_python for workbook discovery. They are read-only;
use run_python for edits, then validate_workbook and submit as usual."""


def _tool_instructions() -> str:
    tool_set = os.environ.get("SPREADSHEETBENCH_TOOL_SET", "python")
    normalized = tool_set.strip().lower().replace("-", "_")
    if normalized == "native_read":
        return _PYTHON_TOOL_INSTRUCTIONS + _NATIVE_READ_TOOL_INSTRUCTIONS
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
        python_error = int(bool(env_info.get("python_error", False)))
        syntax_error = int(bool(env_info.get("syntax_error", False)))
        error_type = str(env_info.get("python_error_type") or "")
        tool_name = str(env_info.get("tool_name") or "")
        tool_error = str(env_info.get("tool_error") or "")
        return {
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
            "env/read_tool_call": int(bool(tool_name)),
            "env/read_tool_success": int(
                bool(tool_name) and bool(env_info.get("tool_ok", False))
            ),
            "env/read_tool_error": int(bool(tool_error)),
            "env/read_tool_error_type": tool_error,
        }

    def _format_history_action(
        self,
        raw_action: str,
        projected_action: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> str:
        if self._history_mode != "compact":
            return raw_action
        projected_action = projected_action or {}
        diagnostics = diagnostics or {}
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

    def reset(self, kwargs):
        del kwargs
        text_obs, image_obs, infos = self.envs.reset()
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
        actions, valids = self.projection_f(text_actions)
        parser_diagnostics = envharness_spreadsheetbench_projection_diagnostics(
            text_actions
        )
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)

        for index, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[index])
            info["tool_calling"] = int(bool(valids[index]))
            diagnostics = self._step_diagnostics(
                parser_diagnostics[index], info
            )
            info.update(diagnostics)
            if not bool(valids[index]):
                parser_error = actions[index].kwargs.get(
                    "error", "Tool call parse error: invalid action."
                )
                info["parser_error"] = parser_error
                text_obs[index] = f"{parser_error}\n\n{str(text_obs[index] or '')}"

        for index, raw_action in enumerate(text_actions):
            action_valid = bool(valids[index])
            projected_action = actions[index]
            projected_payload = {
                "name": projected_action.name,
                "kwargs": self._jsonable(projected_action.kwargs),
            }
            diagnostics = self._step_diagnostics(
                parser_diagnostics[index], infos[index]
            )
            self._trajectory_steps[index].append({
                "step": len(self._trajectory_steps[index]) + 1,
                "observation": self._last_observations[index],
                "model_output": raw_action,
                "projected_action": projected_payload,
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
                f"Current environment observation:\n{current_observation}"
            )
        return prompts
