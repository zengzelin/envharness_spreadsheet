from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_ray_up_falls_back_to_pod_ip_when_launcher_name_does_not_resolve(
    tmp_path: Path,
) -> None:
    source_hostfile = tmp_path / "source_hostfile"
    source_hostfile.write_text("does-not-resolve-launcher slots=8\n")
    run_dir = tmp_path / "ray"
    env = dict(os.environ)
    env.update({
        "SOURCE_HOSTFILE": str(source_hostfile),
        "RAY_RUN_DIR": str(run_dir),
        "RAY_LAUNCH_DRY_RUN": "1",
        "__POD_IP__": "10.20.30.40",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/mpi_ray_up.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "master=10.20.30.40" in completed.stdout
    assert "dry run complete" in completed.stdout


def test_ray_up_single_node_dry_run_can_skip_hostfile(tmp_path: Path) -> None:
    run_dir = tmp_path / "ray"
    env = dict(os.environ)
    env.update({
        "NNODES": "1",
        "SOURCE_HOSTFILE": str(tmp_path / "missing-hostfile"),
        "RAY_RUN_DIR": str(run_dir),
        "RAY_LAUNCH_DRY_RUN": "1",
        "__POD_IP__": "10.20.30.41",
        "GPUS_PER_NODE": "8",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/mpi_ray_up.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "launch_mode=single-node" in completed.stdout
    assert "nnodes=1" in completed.stdout
    assert "master=10.20.30.41" in completed.stdout


def test_ray_up_multinode_dry_run_uses_requested_host_count(
    tmp_path: Path,
) -> None:
    source_hostfile = tmp_path / "source_hostfile"
    source_hostfile.write_text(
        "host-a slots=8\n"
        "host-b slots=8\n"
        "host-c slots=8\n"
    )
    run_dir = tmp_path / "ray"
    env = dict(os.environ)
    env.update({
        "NNODES": "2",
        "SOURCE_HOSTFILE": str(source_hostfile),
        "RAY_RUN_DIR": str(run_dir),
        "RAY_LAUNCH_DRY_RUN": "1",
        "MASTER_ADDR": "10.20.30.42",
        "GPUS_PER_NODE": "8",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/mpi_ray_up.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "launch_mode=multi-node" in completed.stdout
    assert "nnodes=2" in completed.stdout
    assert "host-a slots=1" in completed.stdout
    assert "host-b slots=1" in completed.stdout
    assert "host-c" not in completed.stdout
