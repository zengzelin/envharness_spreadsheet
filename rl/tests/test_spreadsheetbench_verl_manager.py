from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from envharness_rl.spreadsheetbench.manager import (
    SpreadsheetBenchEnvironmentManager,
)
from envharness_rl.spreadsheetbench.projection import (
    envharness_spreadsheetbench_projection,
)


class FakeVectorEnvs:
    def __init__(self) -> None:
        self.actions = None
        self.action_batches = None
        self.reset_task_seeds = None

    def reset(self, task_seeds=None):
        self.reset_task_seeds = task_seeds
        return ["instruction: Fill Summary!A1\noutput_path: /tmp/out.xlsx"], None, [
            {"task_id": "task-1", "won": False}
        ]

    def step(self, actions):
        self.actions = actions
        return ["python completed"], None, [0.0], [False], [
            {"task_id": "task-1", "won": False}
        ]

    def step_many(self, action_batches):
        self.action_batches = action_batches
        result = self.step([batch[0] for batch in action_batches])
        for batch, info in zip(action_batches, result[4]):
            info.update({
                "tool_call_count": len(batch),
                "tool_success_count": len(batch),
                "tool_failure_count": 0,
                "multi_call": len(batch) > 1,
                "multi_call_partial_failure": False,
                "tool_results": [],
            })
        return result

    def close(self):
        pass


def test_manager_builds_tool_prompt_and_preserves_history() -> None:
    envs = FakeVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config
    )

    observations, infos = manager.reset(kwargs={})
    next_observations, rewards, dones, next_infos = manager.step(
        [
            '<tool_call>{"name":"run_python","arguments":'
            '{"code":"print(\\"CaseSensitive\\")"}}</tool_call>'
        ]
    )

    initial_prompt = observations["text"][0]
    assert "<tool_call>" in initial_prompt
    assert '"name":"run_python"' in initial_prompt
    assert "instruction: Fill Summary!A1" in initial_prompt
    assert infos[0]["task_id"] == "task-1"
    assert envs.actions[0].kwargs["code"] == 'print("CaseSensitive")'
    assert rewards.tolist() == [0.0]
    assert dones.tolist() == [False]
    assert next_infos[0]["is_action_valid"].item() == 1
    assert next_infos[0]["tool_calling"] == 1
    assert "CaseSensitive" in next_observations["text"][0]
    assert "python completed" in next_observations["text"][0]
    assert next_observations["anchor"] == ["python completed"]
    assert next_infos[0]["parser/status"] == "native_tool_call"
    assert next_infos[0]["parser/native_valid"] == 1
    assert next_infos[0]["output/has_think"] == 0


def test_manager_executes_and_records_multiple_calls_from_one_model_turn(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_read")
    envs = FakeVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config
    )
    manager.reset(kwargs={})
    model_output = (
        '<tool_call>{"name":"list_sheets","arguments":{}}</tool_call>\n'
        '<tool_call>{"name":"inspect_range","arguments":'
        '{"range":"A1:B2"}}</tool_call>'
    )

    _, _, _, infos = manager.step([model_output])

    assert [[action.name for action in batch] for batch in envs.action_batches] == [
        ["list_sheets", "inspect_range"]
    ]
    assert infos[0]["tool_calling"] == 2
    assert infos[0]["episode/tool_calls_per_turn"] == 2
    assert infos[0]["episode/multi_call"] == 1
    assert infos[0]["parser/invalid"] == 0


class ShortCircuitVectorEnvs(FakeVectorEnvs):
    def step_many(self, action_batches):
        result = self.step([batch[0] for batch in action_batches])
        for batch, info in zip(action_batches, result[4]):
            info.update({
                "tool_call_count": len(batch),
                "tool_executed_count": 1,
                "tool_skipped_count": len(batch) - 1,
                "tool_success_count": 0,
                "tool_failure_count": 1,
                "multi_call": True,
                "multi_call_partial_failure": True,
                "multi_call_completed": False,
                "multi_call_all_success": False,
                "multi_call_short_circuit": True,
                "multi_call_stop_reason": "run_python_failure",
                "multi_call_failure_index": 0,
                "multi_call_executed_fraction": 0.5,
                "tool_results": [{
                    "action_index": 0,
                    "action_name": "run_python",
                    "ok": False,
                    "info": {"python_error": True},
                }],
            })
        return result


def test_manager_exports_multi_call_short_circuit_diagnostics() -> None:
    envs = ShortCircuitVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config
    )
    manager.reset(kwargs={})

    _, _, _, infos = manager.step([
        '<tool_call>{"name":"run_python","arguments":{"code":"broken"}}</tool_call>'
        '<tool_call>{"name":"submit","arguments":{}}</tool_call>'
    ])

    assert infos[0]["episode/projected_tool_calls_per_turn"] == 2
    assert infos[0]["episode/tool_calls_per_turn"] == 1
    assert infos[0]["env/multi_call_skipped_calls"] == 1
    assert infos[0]["env/multi_call_completed"] == 0
    assert infos[0]["env/multi_call_all_success"] == 0
    assert infos[0]["env/multi_call_short_circuit"] == 1
    assert infos[0]["env/multi_call_failure_index"] == 0
    assert infos[0]["env/multi_call_executed_fraction"] == 0.5
    assert infos[0]["env/multi_call_stop_reason"] == "run_python_failure"


def test_validation_manager_resets_exact_parquet_task_indexes() -> None:
    envs = FakeVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config, split="val"
    )

    manager.reset(kwargs=[
        {"split": "test", "task_index": 384},
    ])

    assert envs.reset_task_seeds == [384]


def test_train_manager_ignores_placeholder_indexes_and_keeps_seed_stride() -> None:
    envs = FakeVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config, split="train"
    )

    manager.reset(kwargs=[{"split": "train", "task_index": 0}])

    assert envs.reset_task_seeds is None


def test_manager_native_read_prompt_describes_structured_tools(monkeypatch) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_read")
    envs = FakeVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config
    )

    observations, _ = manager.reset(kwargs={})

    prompt = observations["text"][0]
    assert '"name":"list_sheets"' in prompt
    assert '"name":"inspect_range"' in prompt
    assert '"name":"find_cells"' in prompt
    assert "read the current output workbook" in prompt


def test_manager_python_prompt_omits_native_read_tools(monkeypatch) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "python")
    envs = FakeVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config
    )

    observations, _ = manager.reset(kwargs={})

    prompt = observations["text"][0]
    assert '"name":"list_sheets"' not in prompt
    assert '"name":"inspect_range"' not in prompt
    assert '"name":"find_cells"' not in prompt


def test_manager_native_basic_prompt_describes_formula_and_multi_call(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_basic")
    envs = FakeVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config
    )

    observations, _ = manager.reset(kwargs={})

    prompt = observations["text"][0]
    assert '"name":"fill_formula"' in prompt
    assert "one to four ordered JSON tool calls" in prompt
    assert "Put submit\nlast" in prompt


def test_manager_returns_parser_error_observation_after_invalid_action() -> None:
    envs = FakeVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config
    )
    manager.reset(kwargs={})

    next_observations, _, _, next_infos = manager.step(
        ['<think>draft <tool_call>{"name":"submit"}</tool_call>']
    )

    assert next_infos[0]["is_action_valid"].item() == 0
    assert next_infos[0]["tool_calling"] == 0
    assert "Tool call parse error:" in next_observations["text"][0]
    assert "reasoning" in next_observations["text"][0]
    assert next_infos[0]["parser/status"] == "unclosed_reasoning"
    assert next_infos[0]["parser/invalid"] == 1


class TerminalVectorEnvs(FakeVectorEnvs):
    def step(self, actions):
        self.actions = actions
        return ["graded"], None, [0.75], [True], [
            {"task_id": "task-1", "won": True, "score": 0.75}
        ]


def test_manager_dumps_terminal_trajectory_with_raw_and_projected_actions(
    tmp_path: Path,
) -> None:
    envs = TerminalVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs,
        envharness_spreadsheetbench_projection,
        config,
        split="train",
        trajectory_dir=tmp_path,
    )
    manager.reset(kwargs={})

    _, _, _, infos = manager.step([
        '<tool_call>{"name":"submit","arguments":{}}</tool_call>'
    ])

    files = list((tmp_path / "train").glob("*.json"))
    assert len(files) == 1
    trajectory = json.loads(files[0].read_text())
    assert trajectory["schema_version"] == 1
    assert trajectory["split"] == "train"
    assert trajectory["task_id"] == "task-1"
    assert trajectory["steps"][0]["model_output"].startswith("<tool_call>")
    assert trajectory["steps"][0]["projected_action"] == {
        "name": "submit",
        "kwargs": {},
    }
    assert trajectory["steps"][0]["action_valid"] is True
    assert trajectory["steps"][0]["diagnostics"]["parser/status"] == (
        "native_tool_call"
    )
    assert trajectory["steps"][0]["reward"] == 0.75
    assert trajectory["final_info"]["won"] is True
    assert infos[0]["tool_calling"] == 1


class ErrorVectorEnvs(FakeVectorEnvs):
    def step(self, actions):
        self.actions = actions
        return ["Traceback\nSyntaxError: bad quote"], None, [0.0], [False], [
            {
                "task_id": "task-1",
                "won": False,
                "returncode": 1,
                "python_error": True,
                "python_error_type": "SyntaxError",
                "syntax_error": True,
            }
        ]


def test_manager_compact_history_omits_full_raw_action(monkeypatch) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_HISTORY_MODE", "compact")
    monkeypatch.setenv("SPREADSHEETBENCH_HISTORY_ACTION_CHARS", "20")
    monkeypatch.setenv("SPREADSHEETBENCH_HISTORY_OBS_CHARS", "40")
    envs = ErrorVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config
    )
    manager.reset(kwargs={})
    long_marker = "UNIQUE_LONG_REASONING_MARKER"
    raw_action = (
        long_marker
        + "x" * 200
        + '<tool_call>{"name":"run_python","arguments":{"code":"print(1)"}}</tool_call>'
    )

    next_observations, _, _, next_infos = manager.step([raw_action])

    prompt = next_observations["text"][0]
    assert long_marker not in prompt
    assert "projected_action: run_python" in prompt
    assert "SyntaxError" in prompt
    assert next_infos[0]["env/python_error"] == 1
    assert next_infos[0]["env/syntax_error"] == 1
    assert next_infos[0]["env/python_error_type"] == "SyntaxError"


class WarningVectorEnvs(FakeVectorEnvs):
    def step(self, actions):
        self.actions = actions
        return ["FutureWarning: an error: example in documentation"], None, [
            0.0
        ], [False], [{
            "task_id": "task-1",
            "won": False,
            "returncode": 0,
            "python_error": False,
            "python_error_type": "",
            "syntax_error": False,
        }]


def test_manager_does_not_classify_warning_text_as_python_error() -> None:
    envs = WarningVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config
    )
    manager.reset(kwargs={})

    _, _, _, infos = manager.step([
        '<tool_call>{"name":"run_python","arguments":'
        '{"code":"print(1)"}}</tool_call>'
    ])

    assert infos[0]["env/python_error"] == 0
    assert infos[0]["env/syntax_error"] == 0
    assert infos[0]["env/python_runtime_error"] == 0
    assert infos[0]["env/python_timeout"] == 0


def test_manager_does_not_classify_parser_failure_as_python_error() -> None:
    envs = FakeVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config
    )
    manager.reset(kwargs={})

    _, _, _, infos = manager.step(["not a tool call"])

    assert infos[0]["parser/invalid"] == 1
    assert infos[0]["env/python_error"] == 0
    assert infos[0]["env/syntax_error"] == 0


def test_step_diagnostics_preserves_earlier_python_error_in_multi_call() -> None:
    diagnostics = SpreadsheetBenchEnvironmentManager._step_diagnostics(
        {"status": "native_tool_batch", "valid": 1, "invalid": 0},
        {
            "tool_call_count": 2,
            "multi_call": True,
            "tool_results": [
                {
                    "action_name": "run_python",
                    "ok": False,
                    "info": {
                        "python_error": True,
                        "syntax_error": True,
                        "python_error_type": "SyntaxError",
                    },
                },
                {
                    "action_name": "inspect_range",
                    "ok": True,
                    "info": {"tool_category": "read", "tool_ok": True},
                },
            ],
        },
    )

    assert diagnostics["env/python_error"] == 1
    assert diagnostics["env/syntax_error"] == 1
    assert diagnostics["env/python_error_type"] == "SyntaxError"


def test_step_diagnostics_separates_projected_and_executed_call_counts() -> None:
    diagnostics = SpreadsheetBenchEnvironmentManager._step_diagnostics(
        {"status": "native_tool_batch", "valid": 1, "invalid": 0},
        {
            "tool_call_count": 2,
            "tool_executed_count": 1,
            "multi_call": True,
            "tool_results": [
                {
                    "action_name": "run_python",
                    "ok": False,
                    "info": {
                        "python_error": True,
                        "python_error_type": "SyntaxError",
                    },
                },
            ],
        },
    )

    assert diagnostics["episode/tool_calls_per_turn"] == 1
    assert diagnostics["episode/projected_tool_calls_per_turn"] == 2
    assert diagnostics["episode/multi_call"] == 0
    assert diagnostics["episode/projected_multi_call"] == 1


class ReadToolVectorEnvs(FakeVectorEnvs):
    def step(self, actions):
        self.actions = actions
        return ["read result"], None, [0.0], [False], [{
            "task_id": "task-1",
            "won": False,
            "tool_name": "inspect_range",
            "tool_ok": True,
            "tool_error": "",
        }]


def test_manager_records_read_tool_diagnostics(monkeypatch) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_read")
    envs = ReadToolVectorEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SpreadsheetBenchEnvironmentManager(
        envs, envharness_spreadsheetbench_projection, config
    )
    manager.reset(kwargs={})

    _, _, _, infos = manager.step([
        '<tool_call>{"name":"inspect_range","arguments":'
        '{"range":"A1:B2"}}</tool_call>'
    ])

    assert infos[0]["env/read_tool_call"] == 1
    assert infos[0]["env/read_tool_success"] == 1
    assert infos[0]["env/read_tool_error"] == 0
    assert infos[0]["tool/inspect_range"] == 1
    assert infos[0]["tool/list_sheets"] == 0


def test_manager_records_per_tool_and_write_diagnostics(monkeypatch) -> None:
    monkeypatch.setenv("SPREADSHEETBENCH_TOOL_SET", "native_basic")

    class WriteToolVectorEnvs(FakeVectorEnvs):
        def step(self, actions):
            self.actions = actions
            return ["write result"], None, [0.0], [False], [{
                "task_id": "task-1",
                "won": False,
                "tool_name": "write_range",
                "tool_category": "write",
                "tool_ok": True,
                "tool_error": "",
            }]

    config = SimpleNamespace(env=SimpleNamespace(history_length=2, max_steps=10))
    manager = SpreadsheetBenchEnvironmentManager(
        WriteToolVectorEnvs(), envharness_spreadsheetbench_projection, config
    )
    manager.reset(kwargs={})

    observations, _, _, infos = manager.step([
        '<tool_call>{"name":"write_range","arguments":'
        '{"range":"A1","data":"done"}}</tool_call>'
    ])

    assert infos[0]["env/write_tool_call"] == 1
    assert infos[0]["env/write_tool_success"] == 1
    assert infos[0]["env/write_tool_error"] == 0
    assert infos[0]["tool/write_range"] == 1
    assert infos[0]["tool/run_python"] == 0
    assert "episode_step: 1" in observations["text"][0]
    assert "steps_remaining: 9" in observations["text"][0]


def test_manager_exposes_terminal_reward_components() -> None:
    config = SimpleNamespace(env=SimpleNamespace(history_length=2, max_steps=2))
    manager = SpreadsheetBenchEnvironmentManager(
        FakeVectorEnvs(), envharness_spreadsheetbench_projection, config
    )
    success = manager.success_evaluator(
        total_batch_list=[[
            {"active_masks": True},
            {"active_masks": True},
        ]],
        total_infos=[[
            {"won": False},
            {
                "won": True,
                "reward/workbook_score": 0.75,
                "reward/execution_penalty": -0.2,
                "reward/env_total": 0.55,
            },
        ]],
    )

    assert success["success_rate"].tolist() == [1.0]
    assert success["reward_workbook_score"].tolist() == [0.75]
    assert success["reward_execution_penalty"].tolist() == pytest.approx([-0.2])
    assert success["reward_env_total"].tolist() == pytest.approx([0.55])
