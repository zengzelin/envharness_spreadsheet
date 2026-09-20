from __future__ import annotations

from types import SimpleNamespace

import pytest

from envharness.core.types import (
    Action,
    EnvResetResponse,
    EnvResponse,
    EvaluationResult,
    Observation,
)
from envharness_rl.spreadsheetbench.envs import (
    EnvharnessSpreadsheetEnvs,
    EnvharnessSpreadsheetWorker,
    _ray_get_with_diagnostics,
    _require_initialized_ray,
    _worker_seeds,
)


class FakeSpreadsheetEnv:
    def __init__(self) -> None:
        self.reset_seed = None
        self.reset_options = None
        self.actions = []
        self.evaluate_calls = 0
        self.closed = False

    def reset(self, seed=None, options=None):
        self.reset_seed = seed
        self.reset_options = options
        return EnvResetResponse(
            observation=Observation(
                text="instruction: Keep CaseSensitive values",
                data={"input_path": "/tmp/in.xlsx", "output_path": "/tmp/out.xlsx"},
            ),
            info={"task_id": "13-1", "instruction_type": "Sheet-Level"},
        )

    def step(self, action):
        self.actions.append(action)
        submitted = action.name == "submit"
        return EnvResponse(
            observation=Observation(text="submitted" if submitted else "ran"),
            reward=0.0,
            terminated=submitted,
            truncated=False,
            info={"submitted": submitted},
        )

    def evaluate(self):
        self.evaluate_calls += 1
        return EvaluationResult(
            success=True,
            score=0.75,
            metrics={"answer_position": "A1:B2", "diff": ""},
        )

    def close(self):
        self.closed = True


class PenalizedSpreadsheetEnv(FakeSpreadsheetEnv):
    def step(self, action):
        self.actions.append(action)
        return EnvResponse(
            observation=Observation(text="SyntaxError: bad quote"),
            reward=-0.1,
            terminated=False,
            truncated=False,
            info={
                "returncode": 1,
                "python_error": True,
                "python_error_type": "SyntaxError",
                "syntax_error": True,
            },
        )


def test_worker_grades_submit_and_returns_verifier_score() -> None:
    fake = FakeSpreadsheetEnv()
    worker = EnvharnessSpreadsheetWorker(
        seed=7,
        data_path="/dataset",
        max_steps=5,
        reset_options={"step_timeout": 9},
        env_factory=lambda: fake,
    )

    text, info = worker.reset()
    next_text, reward, done, final_info = worker.step(
        Action(name="submit", kwargs={})
    )

    assert text == "instruction: Keep CaseSensitive values"
    assert info["won"] is False
    assert fake.reset_seed == 7
    assert fake.reset_options == {"data_path": "/dataset", "step_timeout": 9}
    assert next_text == "submitted"
    assert reward == 0.75
    assert done is True
    assert final_info["won"] is True
    assert final_info["task_id"] == "13-1"
    assert final_info["answer_position"] == "A1:B2"
    assert fake.evaluate_calls == 1


def test_worker_logs_action_and_grade_with_task_context(capsys) -> None:
    worker = EnvharnessSpreadsheetWorker(
        seed=0,
        data_path="/dataset",
        max_steps=1,
        worker_index=88,
        env_factory=FakeSpreadsheetEnv,
    )
    worker.reset()

    worker.step(Action(name="submit", kwargs={}))

    output = capsys.readouterr().out
    assert "worker_index=88 task_id=13-1 action=submit episode_step=1 stage=env_step START" in output
    assert "worker_index=88 task_id=13-1 action=submit episode_step=1 stage=grade START" in output
    assert "worker_index=88 task_id=13-1 action=submit episode_step=1 stage=grade END" in output


def test_worker_logs_reset_seed_and_boundaries(capsys) -> None:
    worker = EnvharnessSpreadsheetWorker(
        seed=17,
        data_path="/dataset",
        max_steps=1,
        worker_index=9,
        env_factory=FakeSpreadsheetEnv,
    )

    worker.reset()

    output = capsys.readouterr().out
    assert "worker_index=9 task_id=- action=reset episode_step=0 stage=reset seed=17 START" in output
    assert "worker_index=9 task_id=13-1 action=reset episode_step=0 stage=reset seed=17 END" in output


def test_worker_grades_current_workbook_at_max_steps() -> None:
    fake = FakeSpreadsheetEnv()
    worker = EnvharnessSpreadsheetWorker(
        seed=0,
        data_path="/dataset",
        max_steps=1,
        env_factory=lambda: fake,
    )
    worker.reset()

    _, reward, done, info = worker.step(
        {"name": "run_python", "kwargs": {"code": "print('A')"}}
    )

    assert reward == 0.75
    assert done is True
    assert info["won"] is True
    assert info["time_limit_reached"] is True
    assert fake.evaluate_calls == 1
    assert fake.actions[0].kwargs["code"] == "print('A')"


def test_worker_preserves_execution_penalty_when_max_step_is_graded() -> None:
    fake = PenalizedSpreadsheetEnv()
    worker = EnvharnessSpreadsheetWorker(
        seed=0,
        data_path="/dataset",
        max_steps=1,
        env_factory=lambda: fake,
    )
    worker.reset()

    _, reward, done, info = worker.step(
        Action(name="run_python", kwargs={"code": "broken"})
    )

    assert reward == 0.65
    assert done is True
    assert info["won"] is True
    assert info["python_error_type"] == "SyntaxError"


def test_worker_converts_transient_eval_error_to_failed_episode() -> None:
    class RecalcFailingEnv(FakeSpreadsheetEnv):
        def evaluate(self):
            self.evaluate_calls += 1
            raise RuntimeError(
                "eval_error: LibreOffice recalc of agent output failed "
                "(transient); not a policy failure"
            )

    fake = RecalcFailingEnv()
    worker = EnvharnessSpreadsheetWorker(
        seed=0,
        data_path="/dataset",
        max_steps=1,
        env_factory=lambda: fake,
    )
    worker.reset()

    _, reward, done, info = worker.step(Action(name="submit", kwargs={}))

    assert reward == 0.0
    assert done is True
    assert info["won"] is False
    assert info["score"] == 0.0
    assert info["task_id"] == "13-1"
    assert info["error"].startswith("eval_error: LibreOffice recalc")
    assert fake.evaluate_calls == 1


def test_worker_still_raises_unexpected_eval_errors() -> None:
    class BrokenEnv(FakeSpreadsheetEnv):
        def evaluate(self):
            self.evaluate_calls += 1
            raise ValueError("parser bug")

    fake = BrokenEnv()
    worker = EnvharnessSpreadsheetWorker(
        seed=0,
        data_path="/dataset",
        max_steps=1,
        env_factory=lambda: fake,
    )
    worker.reset()

    try:
        worker.step(Action(name="submit", kwargs={}))
    except ValueError as error:
        assert str(error) == "parser bug"
    else:
        raise AssertionError("unexpected eval errors must still surface")


def test_worker_close_releases_bridge_resources() -> None:
    fake = FakeSpreadsheetEnv()
    worker = EnvharnessSpreadsheetWorker(
        seed=0, data_path="/dataset", max_steps=1, env_factory=lambda: fake
    )

    worker.close()

    assert fake.closed is True


def test_worker_advances_to_the_next_nonoverlapping_task_batch() -> None:
    fake = FakeSpreadsheetEnv()
    worker = EnvharnessSpreadsheetWorker(
        seed=20,
        seed_stride=3,
        data_path="/dataset",
        max_steps=1,
        env_factory=lambda: fake,
    )

    worker.reset()
    assert fake.reset_seed == 20
    worker.reset()
    assert fake.reset_seed == 23


def test_validation_worker_reuses_the_same_task_seed() -> None:
    fake = FakeSpreadsheetEnv()
    worker = EnvharnessSpreadsheetWorker(
        seed=1020,
        seed_stride=32,
        advance_seed=False,
        data_path="/dataset",
        max_steps=1,
        env_factory=lambda: fake,
    )

    worker.reset()
    assert fake.reset_seed == 1020


def test_worker_explicit_task_seed_overrides_but_does_not_advance_sequence() -> None:
    fake = FakeSpreadsheetEnv()
    worker = EnvharnessSpreadsheetWorker(
        seed=1020,
        seed_stride=32,
        advance_seed=False,
        data_path="/dataset",
        max_steps=1,
        env_factory=lambda: fake,
    )

    worker.reset(task_seed=384)
    assert fake.reset_seed == 384
    worker.reset()
    assert fake.reset_seed == 1020
    worker.reset()
    assert fake.reset_seed == 1020


def test_worker_reports_reward_components_at_episode_end() -> None:
    fake = PenalizedSpreadsheetEnv()
    worker = EnvharnessSpreadsheetWorker(
        seed=0,
        data_path="/dataset",
        max_steps=2,
        env_factory=lambda: fake,
    )
    worker.reset()

    _, first_reward, first_done, _ = worker.step(
        Action(name="run_python", kwargs={"code": "broken"})
    )
    _, final_reward, final_done, info = worker.step(
        Action(name="run_python", kwargs={"code": "broken again"})
    )

    assert first_reward == -0.1
    assert first_done is False
    assert final_reward == 0.65
    assert final_done is True
    assert info["reward/workbook_score"] == 0.75
    assert info["reward/execution_penalty"] == pytest.approx(-0.2)
    assert info["reward/env_total"] == pytest.approx(0.55)


def test_worker_executes_multiple_actions_in_one_episode_turn() -> None:
    fake = FakeSpreadsheetEnv()
    worker = EnvharnessSpreadsheetWorker(
        seed=0,
        data_path="/dataset",
        max_steps=2,
        env_factory=lambda: fake,
    )
    worker.reset()

    text, reward, done, info = worker.step_many([
        Action(name="list_sheets", kwargs={}),
        Action(name="inspect_range", kwargs={"range": "A1"}),
    ])

    assert [action.name for action in fake.actions] == [
        "list_sheets", "inspect_range"
    ]
    assert worker._episode_steps == 1
    assert reward == 0.0
    assert done is False
    assert info["tool_call_count"] == 2
    assert info["tool_success_count"] == 2
    assert info["tool_failure_count"] == 0
    assert info["multi_call"] is True
    assert "call 1 list_sheets" in text
    assert "call 2 inspect_range" in text


def test_worker_stops_batch_after_terminated_action() -> None:
    fake = FakeSpreadsheetEnv()
    worker = EnvharnessSpreadsheetWorker(
        seed=0,
        data_path="/dataset",
        max_steps=3,
        env_factory=lambda: fake,
    )
    worker.reset()

    _, _, done, info = worker.step_many([
        Action(name="submit", kwargs={}),
        Action(name="list_sheets", kwargs={}),
    ])

    assert done is True
    assert [action.name for action in fake.actions] == ["submit"]
    assert info["tool_call_count"] == 2
    assert info["tool_executed_count"] == 1


def test_worker_seeds_keep_each_grpo_group_on_the_same_task() -> None:
    assert _worker_seeds(seed=20, env_num=3, group_n=2) == [20, 20, 21, 21, 22, 22]


class FakeRayLifecycle:
    def __init__(self, initialized: bool) -> None:
        self.initialized = initialized
        self.init_calls = 0

    def is_initialized(self) -> bool:
        return self.initialized

    def init(self, *args, **kwargs) -> None:
        self.init_calls += 1


def test_vector_env_requires_caller_to_connect_ray_first() -> None:
    ray_module = FakeRayLifecycle(initialized=False)

    try:
        _require_initialized_ray(ray_module)
    except RuntimeError as error:
        assert "connect to Ray before building SpreadsheetBench envs" in str(error)
    else:
        raise AssertionError("missing external-Ray lifecycle error")

    assert ray_module.init_calls == 0


def test_vector_env_accepts_an_existing_ray_connection() -> None:
    ray_module = FakeRayLifecycle(initialized=True)

    _require_initialized_ray(ray_module)

    assert ray_module.init_calls == 0


class ImmediateRemoteCall:
    def __init__(self, function) -> None:
        self.function = function

    def remote(self, *args):
        return self.function(*args)


class ImmediateWorker:
    def __init__(self, index: int) -> None:
        self.index = index
        self.reset_seeds = []
        self.batches = []
        self.reset = ImmediateRemoteCall(self._reset)
        self.step_many = ImmediateRemoteCall(self._step_many)

    def _reset(self, seed=None):
        self.reset_seeds.append(seed)
        return f"task-{seed}", {"task_id": f"task-{seed}"}

    def _step_many(self, actions):
        self.batches.append(actions)
        return "done", 0.0, False, {"task_id": f"worker-{self.index}"}


class ImmediateRay:
    exceptions = SimpleNamespace(GetTimeoutError=TimeoutError)

    def get(self, refs, timeout=None):
        return refs


def test_vector_env_activates_only_workers_in_last_validation_batch() -> None:
    envs = object.__new__(EnvharnessSpreadsheetEnvs)
    envs._ray = ImmediateRay()
    envs._actor_timeout_s = 30.0
    envs.workers = [ImmediateWorker(index) for index in range(64)]
    envs._active_workers = list(envs.workers)
    envs._task_ids = []

    text_obs, _, infos = envs.reset(task_seeds=list(range(384, 399)))
    result = envs.step_many([
        [Action(name="list_sheets", kwargs={})] for _ in range(15)
    ])

    assert len(envs._active_workers) == 15
    assert len(text_obs) == 15
    assert [info["task_id"] for info in infos] == [
        f"task-{index}" for index in range(384, 399)
    ]
    assert all(len(worker.batches) == 1 for worker in envs.workers[:15])
    assert all(not worker.batches for worker in envs.workers[15:])
    assert len(result[0]) == 15


class FakeActorId:
    def __init__(self, value: str) -> None:
        self.value = value

    def hex(self) -> str:
        return self.value


class FakeActorHandle:
    def __init__(self, actor_id: str) -> None:
        self._actor_id = FakeActorId(actor_id)


class FakeRayGetTimeoutError(Exception):
    pass


class TimingOutRay:
    exceptions = SimpleNamespace(GetTimeoutError=FakeRayGetTimeoutError)

    def get(self, refs, timeout=None):
        assert timeout == 12.5
        raise FakeRayGetTimeoutError("deadline exceeded")

    def wait(self, refs, num_returns, timeout):
        assert num_returns == len(refs)
        assert timeout == 0
        return refs[:1], refs[1:]


def test_ray_get_timeout_reports_pending_actor_indexes_and_ids(capsys) -> None:
    refs = [object(), object(), object()]
    workers = [
        FakeActorHandle("actor-0"),
        FakeActorHandle("actor-1"),
        FakeActorHandle("actor-2"),
    ]

    with pytest.raises(TimeoutError) as captured:
        _ray_get_with_diagnostics(
            TimingOutRay(), refs, workers,
            operation="step", timeout_s=12.5,
            task_ids=["task-0", "task-1", "task-2"],
            actions=["submit", "run_python", "write_range"],
        )

    message = str(captured.value)
    assert "operation=step" in message
    assert "ready=1/3" in message
    assert "1:actor-1" in message
    assert "2:actor-2" in message
    assert "task_id=task-1 action=run_python" in message
    assert "task_id=task-2 action=write_range" in message
    output = capsys.readouterr().out
    assert "[spreadsheet-ray] START operation=step" in output
    assert "[spreadsheet-ray] TIMEOUT operation=step" in output


class SuccessfulRay:
    exceptions = SimpleNamespace(GetTimeoutError=FakeRayGetTimeoutError)

    def get(self, refs, timeout=None):
        assert timeout == 30.0
        return ["done"]


def test_ray_get_logs_successful_operation_boundaries(capsys) -> None:
    result = _ray_get_with_diagnostics(
        SuccessfulRay(), [object()], [FakeActorHandle("actor-0")],
        operation="reset", timeout_s=30.0,
    )

    assert result == ["done"]
    output = capsys.readouterr().out
    assert "[spreadsheet-ray] START operation=reset actors=1 timeout_s=30.0" in output
    assert "[spreadsheet-ray] END operation=reset" in output
