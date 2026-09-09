from __future__ import annotations

from envharness.core.types import (
    Action,
    EnvResetResponse,
    EnvResponse,
    EvaluationResult,
    Observation,
)
from envharness_rl.spreadsheetbench.envs import (
    EnvharnessSpreadsheetWorker,
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
