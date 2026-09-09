"""Ray-parallel SpreadsheetBench environments for verl-agent."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from envharness.bridges.spreadsheetbench.bridge import SpreadsheetBenchEnv
from envharness.core.types import Action, EvaluationResult


def _worker_seeds(seed: int, env_num: int, group_n: int) -> list[int]:
    """Return one deterministic task seed per rollout worker."""
    if env_num < 0:
        raise ValueError(f"env_num must be non-negative, got {env_num}")
    if group_n <= 0:
        raise ValueError(f"group_n must be positive, got {group_n}")
    return [int(seed) + i // group_n for i in range(env_num * group_n)]


def _require_initialized_ray(ray_module: Any) -> None:
    """Keep Ray cluster lifecycle outside the environment adapter."""
    if not ray_module.is_initialized():
        raise RuntimeError(
            "connect to Ray before building SpreadsheetBench envs; start the "
            "cluster with rl/scripts/mpi_ray_up.sh, then use ray.init(address="
            "'auto') or submit the driver with +ray_init.address=auto"
        )


class EnvharnessSpreadsheetWorker:
    """Own one SpreadsheetBench bridge and expose verl-agent's worker API."""

    def __init__(
        self,
        seed: int,
        data_path: str,
        max_steps: int,
        seed_stride: int = 1,
        reset_options: dict[str, Any] | None = None,
        env_factory: Callable[[], SpreadsheetBenchEnv] = SpreadsheetBenchEnv,
    ) -> None:
        if not data_path:
            raise ValueError(
                "SpreadsheetBench data_path is empty; set SPREADSHEETBENCH_DATA"
            )
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if seed_stride <= 0:
            raise ValueError(f"seed_stride must be positive, got {seed_stride}")
        self._next_seed = int(seed)
        self._seed_stride = int(seed_stride)
        self._data_path = str(data_path)
        self._max_steps = int(max_steps)
        self._reset_options = dict(reset_options or {})
        self._env = env_factory()
        self._episode_steps = 0
        self._task_id = ""
        self._done = False

    def reset(self) -> tuple[str, dict[str, Any]]:
        options = dict(self._reset_options)
        options["data_path"] = self._data_path
        response = self._env.reset(seed=self._next_seed, options=options)
        self._next_seed += self._seed_stride
        self._episode_steps = 0
        self._done = False
        info = dict(response.info or {})
        self._task_id = str(info.get("task_id") or "")
        info["task_id"] = self._task_id
        info["won"] = False
        return response.observation.text or "", info

    @staticmethod
    def _coerce_action(action: Action | dict[str, Any]) -> Action:
        if isinstance(action, Action):
            return action
        if isinstance(action, dict):
            return Action.model_validate(action)
        return Action(name="invalid", kwargs={})

    def _grade(self, info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        try:
            result: EvaluationResult = self._env.evaluate()
        except RuntimeError as exc:
            message = str(exc)
            if not message.startswith("eval_error:"):
                raise
            info["error"] = message
            info["won"] = False
            info["score"] = 0.0
            info["task_id"] = self._task_id
            return 0.0, info
        info.update(result.metrics)
        info["won"] = bool(result.success)
        info["score"] = float(result.score)
        info["task_id"] = self._task_id
        return float(result.score), info

    def step(
        self, action: Action | dict[str, Any]
    ) -> tuple[str, float, bool, dict[str, Any]]:
        projected = self._coerce_action(action)
        response = self._env.step(projected)
        self._episode_steps += 1

        info = dict(response.info or {})
        info["task_id"] = self._task_id
        reached_limit = self._episode_steps >= self._max_steps
        done = bool(response.terminated or response.truncated or reached_limit)
        reward = float(response.reward or 0.0)

        if done:
            if reached_limit and not response.terminated and not response.truncated:
                info["time_limit_reached"] = True
            terminal_reward, info = self._grade(info)
            reward += terminal_reward
            self._done = True
        else:
            info["won"] = False

        return response.observation.text or "", reward, done, info

    def close(self) -> None:
        self._env.close()


class EnvharnessSpreadsheetEnvs:
    """Parallel wrapper matching verl-agent's text-environment API."""

    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict[str, Any] | None,
        is_train: bool = True,
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        del is_train  # Dataset/split selection is controlled by data_path.
        import ray

        kwargs = dict(env_kwargs or {})
        raw_data_path = str(kwargs.pop("data_path", "") or "")
        data_path = (
            str(Path(raw_data_path).expanduser().resolve())
            if raw_data_path
            else ""
        )
        max_steps = int(kwargs.pop("max_steps", 10))
        _require_initialized_ray(ray)

        seeds = _worker_seeds(seed, env_num, group_n)
        worker_cls = ray.remote(**(resources_per_worker or {}))(
            EnvharnessSpreadsheetWorker
        )
        self._ray = ray
        self.num_processes = len(seeds)
        self.group_n = group_n
        self.workers = [
            worker_cls.remote(
                seed=worker_seed,
                seed_stride=env_num,
                data_path=data_path,
                max_steps=max_steps,
                reset_options=kwargs,
            )
            for worker_seed in seeds
        ]

    def reset(self):
        results = self._ray.get([worker.reset.remote() for worker in self.workers])
        text_obs = [result[0] for result in results]
        infos = [result[1] for result in results]
        return text_obs, None, infos

    def step(self, actions):
        if len(actions) != self.num_processes:
            raise ValueError(
                f"actions={len(actions)} != num_processes={self.num_processes}"
            )
        results = self._ray.get(
            [
                worker.step.remote(action)
                for worker, action in zip(self.workers, actions)
            ]
        )
        text_obs = [result[0] for result in results]
        rewards = [result[1] for result in results]
        dones = [result[2] for result in results]
        infos = [result[3] for result in results]
        return text_obs, None, rewards, dones, infos

    def close(self) -> None:
        if not self.workers:
            return
        try:
            self._ray.get([worker.close.remote() for worker in self.workers])
        finally:
            for worker in self.workers:
                self._ray.kill(worker)
            self.workers = []


def build_envharness_spreadsheetbench_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict[str, Any] | None,
    is_train: bool = True,
    env_kwargs: dict[str, Any] | None = None,
) -> EnvharnessSpreadsheetEnvs:
    return EnvharnessSpreadsheetEnvs(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
