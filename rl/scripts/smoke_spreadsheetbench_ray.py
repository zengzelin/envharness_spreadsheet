#!/usr/bin/env python
"""Full SpreadsheetBench adapter smoke on an already-running Ray cluster."""
from __future__ import annotations

import json
import os

import ray
from omegaconf import OmegaConf


def _tool_call(name: str, arguments: dict | None = None) -> str:
    return "<tool_call>" + json.dumps({
        "name": name,
        "arguments": arguments or {},
    }) + "</tool_call>"


def main() -> None:
    data_path = os.environ.get("SPREADSHEETBENCH_DATA", "")
    if not data_path:
        raise SystemExit("SPREADSHEETBENCH_DATA is required")

    address = os.environ.get("RAY_ADDRESS", "auto")
    print(f"[ray-smoke] connecting to Ray at {address}", flush=True)
    ray.init(address=address, ignore_reinit_error=True)
    print(f"[ray-smoke] cluster_resources={ray.cluster_resources()}", flush=True)

    config = OmegaConf.create({
        "env": {
            "env_name": "envharness_rl/spreadsheetbench",
            "seed": 0,
            "max_steps": 3,
            "history_length": 2,
            "resources_per_worker": {"num_cpus": 0.2, "num_gpus": 0},
            "rollout": {"n": 2},
        },
        "data": {"train_batch_size": 2, "val_batch_size": 1},
    })

    from agent_system.environments.env_manager import make_envs

    print("[ray-smoke] creating train and validation env actors", flush=True)
    train_envs, val_envs = make_envs(config)
    try:
        print("[ray-smoke] resetting train envs", flush=True)
        observations, infos = train_envs.reset(kwargs={})
        task_ids = [info["task_id"] for info in infos]
        assert len(observations["text"]) == 4
        assert task_ids[0] == task_ids[1]
        assert task_ids[2] == task_ids[3]
        print(f"[ray-smoke] task_ids={task_ids}", flush=True)

        code = """from pathlib import Path
import shutil
input_path = next(Path('.').glob('*_input.xlsx'))
output_path = next(Path('.').glob('*_output.xlsx'))
shutil.copyfile(input_path, output_path)
print(f'copied {input_path.name} -> {output_path.name}')"""
        run_action = _tool_call("run_python", {"code": code})
        _, rewards, dones, step_infos = train_envs.step([run_action] * 4)
        assert rewards.tolist() == [0.0] * 4
        assert dones.tolist() == [False] * 4
        assert all(info["is_action_valid"].item() == 1 for info in step_infos)
        print("[ray-smoke] run_python completed", flush=True)

        submit_action = _tool_call("submit")
        _, rewards, dones, final_infos = train_envs.step([submit_action] * 4)
        assert dones.tolist() == [True] * 4
        assert rewards[0] == rewards[1]
        assert rewards[2] == rewards[3]
        assert all(info["is_action_valid"].item() == 1 for info in final_infos)
        print(
            "[ray-smoke] submit completed "
            f"rewards={rewards.tolist()} "
            f"won={[info['won'] for info in final_infos]}",
            flush=True,
        )
        print("SPREADSHEETBENCH EXTERNAL-RAY SMOKE OK", flush=True)
    finally:
        train_envs.close()
        val_envs.close()
        ray.shutdown()


if __name__ == "__main__":
    main()
