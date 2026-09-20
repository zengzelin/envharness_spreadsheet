#!/usr/bin/env python
"""Reset Spreadsheet-RL environments at training scale without loading a model."""
from __future__ import annotations

import os
from pathlib import Path
import time

import ray

from envharness_rl.spreadsheetbench.envs import EnvharnessSpreadsheetEnvs


def _positive_int(name: str, default: str) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise SystemExit(f"{name} must be positive, got {value}")
    return value


def _worker_pythonpath() -> str:
    """Build absolute shared-filesystem imports for every Ray worker node."""
    repo_root = Path(__file__).resolve().parents[2]
    entries = [
        str(repo_root),
        str(repo_root / "rl"),
        str(repo_root / "third_party/verl-agent"),
    ]
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def main() -> None:
    data_path = os.environ.get("SPREADSHEET_RL_DATA_ROOT") or os.environ.get(
        "SPREADSHEETBENCH_DATA", ""
    )
    if not data_path:
        raise SystemExit("SPREADSHEET_RL_DATA_ROOT or SPREADSHEETBENCH_DATA is required")
    split_file = os.environ.get(
        "SPREADSHEET_RL_TRAIN_FILE", "train_hermes.parquet"
    )
    env_num = _positive_int("SPREADSHEETBENCH_SCALE_ENV_NUM", "16")
    group_n = _positive_int("SPREADSHEETBENCH_SCALE_GROUP_N", "8")
    seed = int(os.environ.get("SPREADSHEETBENCH_SCALE_SEED", "0"))
    os.environ.setdefault("SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS", "180")

    address = os.environ.get("RAY_ADDRESS", "auto")
    worker_pythonpath = _worker_pythonpath()
    print(
        f"[loader-scale] ray={address} data={data_path} split={split_file} "
        f"env_num={env_num} group_n={group_n}",
        flush=True,
    )
    print(f"[loader-scale] worker_pythonpath={worker_pythonpath}", flush=True)
    ray.init(
        address=address,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": worker_pythonpath}},
    )
    envs = EnvharnessSpreadsheetEnvs(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker={"num_cpus": 0.2, "num_gpus": 0},
        env_kwargs={
            "data_path": data_path,
            "split_file": split_file,
            "max_steps": 1,
        },
    )
    started = time.monotonic()
    primary_error: BaseException | None = None
    try:
        observations, _, infos = envs.reset()
        elapsed = time.monotonic() - started
        task_ids = [str(info["task_id"]) for info in infos]
        expected = env_num * group_n
        if len(observations) != expected or len(task_ids) != expected:
            raise AssertionError(
                f"reset size mismatch: observations={len(observations)} "
                f"task_ids={len(task_ids)} expected={expected}"
            )
        for start in range(0, expected, group_n):
            group = task_ids[start:start + group_n]
            if len(set(group)) != 1:
                raise AssertionError(
                    f"group task mismatch at workers {start}:{start + group_n}: {group}"
                )
        if len(set(task_ids)) != env_num:
            raise AssertionError(
                f"unique task mismatch: got={len(set(task_ids))} expected={env_num}"
            )
        print(
            "SPREADSHEETBENCH LOADER SCALE SMOKE OK: "
            f"actors={expected} unique_tasks={len(set(task_ids))} "
            f"reset_elapsed_s={elapsed:.1f}",
            flush=True,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            envs.close()
        except BaseException as close_exc:
            if primary_error is None:
                raise
            print(
                "[loader-scale] cleanup failed after primary error: "
                f"{type(close_exc).__name__}: {close_exc}",
                flush=True,
            )
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
