"""Ray-parallel SpreadsheetBench environments for verl-agent."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import os
from pathlib import Path
import time
from typing import Any, Iterator

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


def _actor_id(worker: Any, index: int) -> str:
    actor_id = getattr(worker, "_actor_id", None)
    try:
        value = actor_id.hex()
    except (AttributeError, TypeError):
        value = str(actor_id or "unknown")
    return f"{index}:{value}"


def _actor_label(
    worker: Any, index: int, task_ids: list[str] | None,
    actions: list[str] | None,
) -> str:
    parts = [_actor_id(worker, index)]
    if task_ids is not None and index < len(task_ids):
        parts.append(f"task_id={task_ids[index]}")
    if actions is not None and index < len(actions):
        parts.append(f"action={actions[index]}")
    return " ".join(parts)


def _ray_get_with_diagnostics(
    ray_module: Any,
    refs: list[Any],
    workers: list[Any],
    *,
    operation: str,
    timeout_s: float,
    task_ids: list[str] | None = None,
    actions: list[str] | None = None,
) -> list[Any]:
    """Wait for actor calls with bounded, actor-specific diagnostics."""
    started = time.monotonic()
    print(
        f"[spreadsheet-ray] START operation={operation} actors={len(refs)} "
        f"timeout_s={timeout_s}",
        flush=True,
    )
    try:
        results = ray_module.get(refs, timeout=timeout_s)
    except ray_module.exceptions.GetTimeoutError as exc:
        try:
            ready, pending = ray_module.wait(
                refs, num_returns=len(refs), timeout=0
            )
            pending_refs = set(pending)
            pending_actors = [
                _actor_label(worker, index, task_ids, actions)
                for index, (worker, ref) in enumerate(zip(workers, refs))
                if ref in pending_refs
            ]
            ready_count = len(ready)
        except Exception as wait_exc:  # noqa: BLE001
            pending_actors = [
                _actor_label(worker, index, task_ids, actions)
                for index, worker in enumerate(workers)
            ]
            ready_count = 0
            pending_actors.append(f"ray.wait_error={wait_exc!r}")
        elapsed = time.monotonic() - started
        detail = (
            f"operation={operation} "
            f"elapsed_s={elapsed:.1f} ready={ready_count}/{len(refs)} "
            f"pending_actors={pending_actors}"
        )
        message = f"SpreadsheetBench Ray actor timeout: {detail}"
        print(f"[spreadsheet-ray] TIMEOUT {detail}", flush=True)
        raise TimeoutError(message) from exc
    except BaseException as exc:
        elapsed = time.monotonic() - started
        print(
            f"[spreadsheet-ray] ERROR operation={operation} "
            f"elapsed_s={elapsed:.1f} error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    elapsed = time.monotonic() - started
    print(
        f"[spreadsheet-ray] END operation={operation} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return results


class EnvharnessSpreadsheetWorker:
    """Own one SpreadsheetBench bridge and expose verl-agent's worker API."""

    def __init__(
        self,
        seed: int,
        data_path: str,
        max_steps: int,
        seed_stride: int = 1,
        advance_seed: bool = True,
        reset_options: dict[str, Any] | None = None,
        env_factory: Callable[[], SpreadsheetBenchEnv] = SpreadsheetBenchEnv,
        worker_index: int = -1,
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
        self._advance_seed = bool(advance_seed)
        self._data_path = str(data_path)
        self._max_steps = int(max_steps)
        self._reset_options = dict(reset_options or {})
        self._env = env_factory()
        self._episode_steps = 0
        self._task_id = ""
        self._done = False
        self._worker_index = worker_index
        self._execution_penalty = 0.0

    @contextmanager
    def _stage(
        self, name: str, action: str, episode_step: int, detail: str = ""
    ) -> Iterator[None]:
        def prefix() -> str:
            task_id = self._task_id or "-"
            suffix = f" {detail}" if detail else ""
            return (
                f"[spreadsheet-worker] worker_index={self._worker_index} "
                f"task_id={task_id} action={action} "
                f"episode_step={episode_step} stage={name}{suffix}"
            )

        started = time.monotonic()
        print(f"{prefix()} START", flush=True)
        try:
            yield
        except BaseException as exc:
            print(
                f"{prefix()} ERROR elapsed_s={time.monotonic() - started:.1f} "
                f"error={type(exc).__name__}: {exc}", flush=True,
            )
            raise
        print(
            f"{prefix()} END elapsed_s={time.monotonic() - started:.1f}",
            flush=True,
        )

    def reset(self) -> tuple[str, dict[str, Any]]:
        options = dict(self._reset_options)
        options["data_path"] = self._data_path
        reset_seed = self._next_seed
        self._task_id = ""
        with self._stage("reset", "reset", 0, detail=f"seed={reset_seed}"):
            response = self._env.reset(seed=reset_seed, options=options)
            if self._advance_seed:
                self._next_seed += self._seed_stride
            self._episode_steps = 0
            self._done = False
            self._execution_penalty = 0.0
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
        episode_step = self._episode_steps + 1
        with self._stage("env_step", projected.name, episode_step):
            response = self._env.step(projected)
        self._episode_steps += 1

        info = dict(response.info or {})
        info["task_id"] = self._task_id
        reached_limit = self._episode_steps >= self._max_steps
        done = bool(response.terminated or response.truncated or reached_limit)
        reward = float(response.reward or 0.0)
        self._execution_penalty += reward

        if done:
            if reached_limit and not response.terminated and not response.truncated:
                info["time_limit_reached"] = True
            with self._stage("grade", projected.name, episode_step):
                terminal_reward, info = self._grade(info)
            reward += terminal_reward
            info["reward/workbook_score"] = terminal_reward
            info["reward/execution_penalty"] = self._execution_penalty
            info["reward/env_total"] = self._execution_penalty + terminal_reward
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
        self._actor_timeout_s = float(os.environ.get(
            "SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS", "600"
        ))
        if self._actor_timeout_s <= 0:
            raise ValueError(
                "SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS must be positive"
            )
        self.num_processes = len(seeds)
        self.group_n = group_n
        self._task_ids: list[str] = []
        self.workers = [
            worker_cls.remote(
                seed=worker_seed,
                worker_index=index,
                seed_stride=env_num,
                advance_seed=is_train,
                data_path=data_path,
                max_steps=max_steps,
                reset_options=kwargs,
            )
            for index, worker_seed in enumerate(seeds)
        ]

    def reset(self):
        refs = [worker.reset.remote() for worker in self.workers]
        results = _ray_get_with_diagnostics(
            self._ray, refs, self.workers,
            operation="reset", timeout_s=self._actor_timeout_s,
        )
        text_obs = [result[0] for result in results]
        infos = [result[1] for result in results]
        self._task_ids = [str(info.get("task_id") or "unknown") for info in infos]
        return text_obs, None, infos

    def step(self, actions):
        if len(actions) != self.num_processes:
            raise ValueError(
                f"actions={len(actions)} != num_processes={self.num_processes}"
            )
        refs = [
            worker.step.remote(action)
            for worker, action in zip(self.workers, actions)
        ]
        results = _ray_get_with_diagnostics(
            self._ray, refs, self.workers,
            operation="step", timeout_s=self._actor_timeout_s,
            task_ids=self._task_ids,
            actions=[
                action.name if isinstance(action, Action)
                else str(action.get("name", "invalid"))
                if isinstance(action, dict) else "invalid"
                for action in actions
            ],
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
            refs = [worker.close.remote() for worker in self.workers]
            _ray_get_with_diagnostics(
                self._ray, refs, self.workers,
                operation="close", timeout_s=self._actor_timeout_s,
            )
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
