from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from envharness_rl.spreadsheetbench.manager import (
    SpreadsheetBenchEnvironmentManager,
)
from envharness_rl.spreadsheetbench.projection import (
    envharness_spreadsheetbench_projection,
)


class FakeVectorEnvs:
    def __init__(self) -> None:
        self.actions = None

    def reset(self):
        return ["instruction: Fill Summary!A1\noutput_path: /tmp/out.xlsx"], None, [
            {"task_id": "task-1", "won": False}
        ]

    def step(self, actions):
        self.actions = actions
        return ["python completed"], None, [0.0], [False], [
            {"task_id": "task-1", "won": False}
        ]

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
