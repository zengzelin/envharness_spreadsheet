from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_prepare_module():
    path = ROOT / "rl/scripts/prepare_spreadsheetbench_verl_data.py"
    spec = importlib.util.spec_from_file_location("prepare_spreadsheetbench_verl_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_eval_compare_module():
    path = ROOT / "rl/scripts/compare_spreadsheetbench_evals.py"
    spec = importlib.util.spec_from_file_location(
        "compare_spreadsheetbench_evals", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_pipeline_exports_native_tool_metrics() -> None:
    rollout_source = (
        ROOT / "third_party/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py"
    ).read_text()
    trainer_source = (
        ROOT / "third_party/verl-agent/verl/trainer/ppo/ray_trainer.py"
    ).read_text()

    for batch_key, metric_name in {
        "env_read_tool_call": "env/read_tool_call_ratio",
        "tool_list_sheets": "tool/list_sheets_ratio",
        "tool_write_range": "tool/write_range_ratio",
        "tool_fill_formula": "tool/fill_formula_ratio",
        "tool_recalculate_and_read": "tool/recalculate_and_read_ratio",
        "tool_format_range": "tool/format_range_ratio",
        "env_recalc_tool_call": "env/recalc_tool_call_ratio",
        "env_recalc_formula_error_count": "env/recalc_formula_error_count_mean",
        "env_python_preflight_reject": "env/python_preflight_reject_ratio",
        "tool_submit": "tool/submit_ratio",
        "episode_tool_calls_per_turn": "episode/tool_calls_per_turn",
        "episode_projected_tool_calls_per_turn": "episode/projected_tool_calls_per_turn",
        "episode_multi_call": "episode/multi_call_ratio",
        "env_multi_call_partial_failure": "env/multi_call_partial_failure_rate",
        "env_multi_call_completed": "env/multi_call_completed_rate",
        "env_multi_call_all_success": "env/multi_call_all_success_rate",
        "env_multi_call_short_circuit": "env/multi_call_short_circuit_rate",
        "env_multi_call_skipped_calls": "env/multi_call_skipped_calls_mean",
        "env_multi_call_executed_fraction": "env/multi_call_executed_fraction_mean",
        "env_multi_call_failure_index": "env/multi_call_failure_index_mean",
    }.items():
        assert batch_key in rollout_source
        assert batch_key in trainer_source
        assert metric_name in trainer_source

    assert "success_rate_weights" in trainer_source
    assert "np.average(v, weights=success_rate_weights[k])" in trainer_source

    # Success/error rates are aggregated generically for both read and write
    # tools, so their concrete batch keys are intentionally not duplicated.
    assert "env_write_tool_success" in rollout_source
    assert "outcome_key = f'env_{category}_tool_{outcome}'" in trainer_source
    assert "metrics[f'env/{category}_tool_{outcome}_rate']" in trainer_source
    assert "metric_dict[f'val/env/{category}_tool_{outcome}_rate']" in trainer_source

    for batch_key, metric_name in {
        "reward_workbook_score": "reward/workbook_score_mean",
        "reward_execution_penalty": "reward/execution_penalty_mean",
        "reward_env_total": "reward/env_total_mean",
        "reward/invalid_action_penalty_mean": "reward/invalid_action_penalty_mean",
    }.items():
        assert batch_key in trainer_source
        assert metric_name in trainer_source


def test_training_scripts_forward_recalc_limits() -> None:
    for script_name in (
        "run_spreadsheetbench_grpo.sh",
        "submit_spreadsheetbench_grpo.sh",
    ):
        source = (ROOT / "rl/scripts" / script_name).read_text()
        assert "SPREADSHEETBENCH_MAX_RECALC_CALLS" in source
        assert "SPREADSHEETBENCH_RECALC_TIMEOUT_SECONDS" in source


def test_training_pipeline_logs_spreadsheet_rollout_phase_boundaries() -> None:
    rollout_source = (
        ROOT / "third_party/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py"
    ).read_text()

    for phase in ("env_reset", "generate_sequences", "env_step"):
        assert f'"{phase}"' in rollout_source
    assert "SPREADSHEETBENCH_PHASE_HEARTBEAT_SECONDS" in rollout_source
    assert "rollout_call=self._rollout_invocation" in rollout_source


def test_rollout_summary_cli_can_write_multi_call_events() -> None:
    source = (
        ROOT / "rl/scripts/summarize_spreadsheetbench_rollouts.py"
    ).read_text()

    assert '"--multi-call-events-output"' in source
    assert "collect_multi_call_events" in source


def test_fetch_verl_agent_reproduces_spreadsheet_metrics_patch() -> None:
    source = (ROOT / "rl/scripts/fetch_verl_agent.sh").read_text()
    patch = (ROOT / "rl/integration/verl_agent_spreadsheetbench_metrics.patch").read_text()

    assert "verl_agent_spreadsheetbench_metrics.patch" in source
    assert "git -C \"$DEST\" apply \"$SPREADSHEET_METRICS_PATCH_FILE\"" in source
    assert "reward_workbook_score" in patch
    assert "success_rate_weights" in patch
    assert "episode_tool_calls_per_turn" in patch
    assert "env_multi_call_short_circuit" in patch
    tool_patch = (
        ROOT / "rl/integration/verl_agent_spreadsheetbench_tool_metrics.patch"
    ).read_text()
    assert "SPREADSHEET_TOOL_METRICS_PATCH_FILE" in source
    assert "env_recalc_tool_call" in tool_patch
    assert "tool_format_range" in tool_patch
    eval_patch = (
        ROOT / "rl/integration/verl_agent_spreadsheetbench_eval.patch"
    ).read_text()
    assert "SPREADSHEET_EVAL_PATCH_FILE" in source
    assert 'config.trainer.get("val_only", False)' in eval_patch


def test_loader_scale_smoke_exercises_128_actor_reset_without_model() -> None:
    source = (
        ROOT / "rl/scripts/smoke_spreadsheetbench_loader_scale.py"
    ).read_text()

    assert 'SPREADSHEETBENCH_SCALE_ENV_NUM", "16"' in source
    assert 'SPREADSHEETBENCH_SCALE_GROUP_N", "8"' in source
    assert "EnvharnessSpreadsheetEnvs(" in source
    assert "envs.reset()" in source
    assert 'runtime_env={"env_vars": {"PYTHONPATH": worker_pythonpath}}' in source
    assert 'repo_root / "rl"' in source
    assert 'repo_root / "third_party/verl-agent"' in source
    assert '"data_path": data_path' in source
    # Cleanup errors must only replace the result when reset itself succeeded.
    # If reset already failed, the script logs cleanup failure and preserves
    # the original exception.
    assert "if primary_error is None:" in source
    assert "cleanup failed after primary error" in source
    assert "group task mismatch" in source
    assert "LOADER SCALE SMOKE OK" in source


def test_placeholder_rows_are_offline_text_agent_examples() -> None:
    module = _load_prepare_module()

    rows = module.build_rows("train", 2)

    assert rows == [
        {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": "train", "index": 0},
            "env_kwargs": {"split": "train", "task_index": 0},
        },
        {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": "train", "index": 1},
            "env_kwargs": {"split": "train", "task_index": 1},
        },
    ]


def test_training_dry_run_targets_external_ray_and_spreadsheetbench(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]\n")
    data_dir = tmp_path / "verl-data"
    run_root = tmp_path / "runs"
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "RAY_ADDRESS": "auto",
        "SPREADSHEETBENCH_DATA": str(dataset),
        "VERL_DATA_DIR": str(data_dir),
        "RUN_ROOT": str(run_root),
        "PY": "/usr/bin/python",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/run_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "env.env_name=envharness_rl/spreadsheetbench" in completed.stdout
    assert "+ray_init.address=auto" in completed.stdout
    assert "data.train_files=" + str(data_dir / "train.parquet") in completed.stdout
    assert "trainer.nnodes=1" in completed.stdout
    assert "trainer.n_gpus_per_node=2" in completed.stdout
    assert "logger=['console','wandb','tensorboard']" in completed.stdout
    assert "trainer.logger=" in completed.stdout
    assert "trainer.rollout_data_dir=" in completed.stdout
    assert "rollouts/verl" in completed.stdout
    assert "actor_timeout_s=600" in completed.stdout
    assert "phase_heartbeat_s=60" in completed.stdout
    assert "dry run complete" in completed.stdout


def test_training_dry_run_exposes_optimization_and_penalty_knobs(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]\n")
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "RAY_ADDRESS": "auto",
        "SPREADSHEETBENCH_DATA": str(dataset),
        "RUN_ROOT": str(tmp_path / "runs"),
        "PY": "/usr/bin/python",
        "ACTOR_LR": "3e-7",
        "USE_INVALID_ACTION_PENALTY": "False",
        "INVALID_ACTION_PENALTY_COEF": "0.025",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/run_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "actor_rollout_ref.actor.optim.lr=3e-7" in completed.stdout
    assert "actor_rollout_ref.actor.use_invalid_action_penalty=False" in completed.stdout
    assert "actor_rollout_ref.actor.invalid_action_penalty_coef=0.025" in completed.stdout
    assert "actor_lr=3e-7" in completed.stdout
    assert "invalid_action_penalty=False coef=0.025" in completed.stdout


def test_training_dry_run_decouples_validation_size_and_concurrency(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]\n")
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "RAY_ADDRESS": "auto",
        "SPREADSHEETBENCH_DATA": str(dataset),
        "RUN_ROOT": str(tmp_path / "runs"),
        "PY": "/usr/bin/python",
        "VAL_SIZE": "399",
        "VAL_CONCURRENCY": "64",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/run_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "data.val_batch_size=64" in completed.stdout
    assert "actor_rollout_ref.rollout.val_kwargs.n=1" in completed.stdout
    assert "val_size=399 val_concurrency=64" in completed.stdout


def test_training_rejects_epoch_budget_shorter_than_requested_steps(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]\n")
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "MODE": "diagnostic",
        "RAY_ADDRESS": "auto",
        "SPREADSHEETBENCH_DATA": str(dataset),
        "RUN_ROOT": str(tmp_path / "runs"),
        "PY": "/usr/bin/python",
        "EPOCHS": "20",
        "TOTAL_TRAINING_STEPS": "80",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/run_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "EPOCHS=20 cannot cover TOTAL_TRAINING_STEPS=80" in completed.stderr


@pytest.mark.parametrize("tool_set", ["native_read", "native_basic"])
def test_training_dry_run_tracks_native_tool_set(
    tmp_path: Path, tool_set: str,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]\n")
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "RAY_ADDRESS": "auto",
        "SPREADSHEETBENCH_DATA": str(dataset),
        "SPREADSHEETBENCH_TOOL_SET": tool_set,
        "RUN_ROOT": str(tmp_path / "runs"),
        "PY": "/usr/bin/python",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/run_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"tool_set={tool_set}" in completed.stdout


def test_training_dry_run_accepts_spreadsheet_rl_parquet_splits(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "Spreadsheet-RL"
    dataset.mkdir()
    (dataset / "train_hermes.parquet").write_bytes(b"placeholder")
    (dataset / "test_verified_hermes.parquet").write_bytes(b"placeholder")
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "RAY_ADDRESS": "auto",
        "SPREADSHEETBENCH_DATA_FORMAT": "spreadsheet_rl",
        "SPREADSHEET_RL_DATA_ROOT": str(dataset),
        "RUN_ROOT": str(tmp_path / "runs"),
        "PY": "/usr/bin/python",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/run_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "data_format=spreadsheet_rl data=" + str(dataset) in completed.stdout
    assert "train_split=train_hermes.parquet" in completed.stdout
    assert "val_split=test_verified_hermes.parquet" in completed.stdout
    assert "dry run complete" in completed.stdout


def test_eval_submit_dry_run_uses_full_validation_with_bounded_concurrency(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "Spreadsheet-RL"
    dataset.mkdir()
    (dataset / "train_hermes.parquet").write_bytes(b"placeholder")
    (dataset / "test_verified_hermes.parquet").write_bytes(b"placeholder")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    state_file = tmp_path / "ray_address.env"
    state_file.write_text(
        "export RAY_ADDRESS=10.0.0.1:6379\n"
        "export RAY_DASHBOARD_ADDRESS=http://10.0.0.1:8265\n"
        "export NNODES=1\n"
        "export GPUS_PER_NODE=8\n"
    )
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "RAY_STATE_FILE": str(state_file),
        "SPREADSHEET_RL_DATA_ROOT": str(dataset),
        "RUN_ROOT": str(tmp_path / "runs"),
        "PY": "/usr/bin/python",
        "VAL_SIZE": "399",
        "VAL_CONCURRENCY": "64",
    })

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "rl/scripts/submit_spreadsheetbench_eval.sh"),
            f"step50={model}",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert "[spreadsheet-eval] label=step50" in completed.stdout
    assert '"VAL_SIZE": "399"' in completed.stdout
    assert '"VAL_CONCURRENCY": "64"' in completed.stdout
    assert '"TRAIN_BS": "8"' in completed.stdout
    assert '"PPO_MINI_BS": "8"' in completed.stdout
    assert '"GROUP_N": "1"' in completed.stdout
    assert '"EXTRA_HYDRA": "trainer.val_only=True"' in completed.stdout
    assert '"SPREADSHEETBENCH_TOOL_SET": "native_basic"' in completed.stdout
    assert '"MODEL": "' + str(model) + '"' in completed.stdout


def test_eval_submit_rejects_model_without_weights(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "rl/scripts/submit_spreadsheetbench_eval.sh"),
            f"broken={model}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "model weights not found" in completed.stderr


def _write_eval_run(
    run_dir: Path,
    outcomes: dict[str, bool],
    *,
    python_error_tasks: set[str] | None = None,
    write_success_tasks: set[str] | None = None,
    evaluator_error_tasks: set[str] | None = None,
    environment_overrides: dict[str, str] | None = None,
) -> None:
    trajectory_dir = run_dir / "rollouts/env/val"
    trajectory_dir.mkdir(parents=True)
    environment = {
        "MODEL": "test-model",
        "SPREADSHEETBENCH_DATA_FORMAT": "spreadsheet_rl",
        "SPREADSHEET_RL_VAL_FILE": "test_verified_hermes.parquet",
        "VAL_SIZE": str(len(outcomes)),
        "MAX_STEPS": "15",
        "MAX_PROMPT_LENGTH": "8192",
        "MAX_RESPONSE_LENGTH": "16384",
        "APPLY_CHAT_TEMPLATE_ENABLE_THINKING": "True",
        "ENVHARNESS_DISABLE_THINKING": "0",
        "VAL_TEMPERATURE": "0",
        "VAL_DO_SAMPLE": "False",
        "SPREADSHEETBENCH_TOOL_SET": "native_basic",
        "SPREADSHEETBENCH_HISTORY_MODE": "compact",
        "SPREADSHEETBENCH_HISTORY_ACTION_CHARS": "1200",
        "SPREADSHEETBENCH_HISTORY_OBS_CHARS": "2000",
        "ROLLOUT_MAX_MODEL_LEN": "24576",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS": "32768",
    }
    environment.update(environment_overrides or {})
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "manifest_version": 2,
        "git_commit": "abc123",
        "environment": environment,
    }))
    python_error_tasks = python_error_tasks or set()
    write_success_tasks = write_success_tasks or set()
    evaluator_error_tasks = evaluator_error_tasks or set()
    for index, (task_id, won) in enumerate(outcomes.items()):
        diagnostics = {
            "env/python_error": int(task_id in python_error_tasks),
            "env/write_tool_call": 1,
            "env/write_tool_success": int(task_id in write_success_tasks),
            "env/write_tool_error": int(task_id not in write_success_tasks),
        }
        payload = {
            "task_id": task_id,
            "steps": [{
                "action_valid": True,
                "projected_action": {"name": "write_range"},
                "diagnostics": diagnostics,
            }],
            "final_info": (
                {
                    "won": False,
                    "submitted": True,
                    "error": "eval_error: LibreOffice recalc failed",
                }
                if task_id in evaluator_error_tasks
                else {"won": won, "submitted": True}
            ),
        }
        (trajectory_dir / f"{index}.json").write_text(json.dumps(payload))


def test_compare_evals_reports_paired_outcomes_and_diagnostics(
    tmp_path: Path,
) -> None:
    module = _load_eval_compare_module()
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_eval_run(
        base,
        {"a": True, "b": True, "c": False, "d": False},
        python_error_tasks={"d"},
        write_success_tasks={"a", "b"},
    )
    _write_eval_run(
        candidate,
        {"a": True, "b": False, "c": True, "d": True},
        write_success_tasks={"a", "c", "d"},
    )

    report = module.compare_runs(base, [candidate], expected_tasks=4)

    assert report["base"]["success_count"] == 2
    assert report["base"]["valid_task_count"] == 4
    assert len(report["base"]["success_rate_ci95"]) == 2
    compared = report["candidates"][0]
    assert compared["success_count"] == 3
    assert {
        key: compared["paired"][key]
        for key in (
            "both_success", "base_only", "candidate_only", "both_failed",
            "net_success_gain", "paired_task_count",
            "excluded_evaluator_error",
        )
    } == {
        "both_success": 1,
        "base_only": 1,
        "candidate_only": 2,
        "both_failed": 0,
        "net_success_gain": 1,
        "paired_task_count": 4,
        "excluded_evaluator_error": 0,
    }
    assert compared["task_results"] == [
        {
            "task_id": "a",
            "base_won": True,
            "candidate_won": True,
            "transition": "both_success",
        },
        {
            "task_id": "b",
            "base_won": True,
            "candidate_won": False,
            "transition": "base_only",
        },
        {
            "task_id": "c",
            "base_won": False,
            "candidate_won": True,
            "transition": "candidate_only",
        },
        {
            "task_id": "d",
            "base_won": False,
            "candidate_won": True,
            "transition": "candidate_only",
        },
    ]
    assert report["base"]["python_error_ratio"] == pytest.approx(0.25)
    assert compared["write_call_success_rate"] == pytest.approx(0.75)
    assert 0.0 <= compared["paired"]["mcnemar_exact_pvalue"] <= 1.0


def test_compare_evals_rejects_mismatched_task_sets(tmp_path: Path) -> None:
    module = _load_eval_compare_module()
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_eval_run(base, {"a": True, "b": False})
    _write_eval_run(candidate, {"a": True, "c": False})

    with pytest.raises(ValueError, match="task set mismatch"):
        module.compare_runs(base, [candidate], expected_tasks=2)


def test_compare_evals_rejects_behavioral_config_mismatch(tmp_path: Path) -> None:
    module = _load_eval_compare_module()
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_eval_run(base, {"a": True})
    _write_eval_run(
        candidate,
        {"a": True},
        environment_overrides={"MAX_STEPS": "10"},
    )

    with pytest.raises(ValueError, match="evaluation config mismatch.*MAX_STEPS"):
        module.compare_runs(base, [candidate], expected_tasks=1)

    report = module.compare_runs(
        base,
        [candidate],
        expected_tasks=1,
        allow_config_mismatch=True,
    )
    assert report["candidates"][0]["config_mismatches"]["MAX_STEPS"] == {
        "base": "15",
        "candidate": "10",
    }


def test_compare_evals_rejects_evaluator_errors_by_default(tmp_path: Path) -> None:
    module = _load_eval_compare_module()
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_eval_run(base, {"a": True, "b": False})
    _write_eval_run(
        candidate,
        {"a": True, "b": False},
        evaluator_error_tasks={"b"},
    )

    with pytest.raises(ValueError, match="evaluator error"):
        module.compare_runs(base, [candidate], expected_tasks=2)

    report = module.compare_runs(
        base,
        [candidate],
        expected_tasks=2,
        allow_evaluator_errors=True,
    )
    compared = report["candidates"][0]
    assert compared["evaluator_error_count"] == 1
    assert compared["valid_task_count"] == 1
    assert compared["paired"]["paired_task_count"] == 1
    assert compared["paired"]["excluded_evaluator_error"] == 1
    assert compared["task_results"][1]["transition"] == "evaluator_error"


def test_eval_manifest_records_behavioral_comparison_fields() -> None:
    source = (ROOT / "rl/scripts/run_spreadsheetbench_grpo.sh").read_text()
    manifest_source = source[source.index("keys = ["):]
    exported_names = {
        token
        for line in source.splitlines()
        if line.startswith("export ")
        for token in line[len("export "):].split()
    }

    for name in (
        "APPLY_CHAT_TEMPLATE_ENABLE_THINKING",
        "ENVHARNESS_DISABLE_THINKING",
        "VAL_TEMPERATURE",
        "VAL_DO_SAMPLE",
        "ROLLOUT_MAX_MODEL_LEN",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS",
    ):
        assert f'"{name}"' in manifest_source
        assert name in exported_names


def test_val_only_spreadsheet_env_skips_train_actor_pool() -> None:
    source = (
        ROOT / "third_party/verl-agent/agent_system/environments/env_manager.py"
    ).read_text()
    spreadsheet_route = source[
        source.index('startswith("envharness_rl/spreadsheetbench")'):
        source.index('    if "search" in config.env.env_name.lower():')
    ]

    assert '_val_only = bool(config.trainer.get("val_only", False))' in spreadsheet_route
    assert "trainer.val_only=True requires trainer.val_before_train=True" in spreadsheet_route
    assert "if not _val_only:" in spreadsheet_route
    assert "envs = None" in spreadsheet_route


def test_full_training_uses_qwen3_4b_for_150_steps_and_periodic_checkpoints(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]\n")
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "MODE": "full",
        "RAY_ADDRESS": "auto",
        "SPREADSHEETBENCH_DATA": str(dataset),
        "RUN_ROOT": str(tmp_path / "runs"),
        "PY": "/usr/bin/python",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/run_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        "actor_rollout_ref.model.path=/mnt/geminisgceph1/geminicephfs/"
        "mmsearch-luban-universal/luban/common/models/Qwen3-4B-Thinking-2507"
    ) in completed.stdout
    assert "env.max_steps=15" in completed.stdout
    assert "data.max_prompt_length=8192" in completed.stdout
    assert "data.max_response_length=24576" in completed.stdout
    assert "+data.apply_chat_template_kwargs.enable_thinking=True" in completed.stdout
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in completed.stdout
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.85" in completed.stdout
    assert "actor_rollout_ref.rollout.max_model_len=32768" in completed.stdout
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=32768" in completed.stdout
    assert "actor_rollout_ref.rollout.top_k=20" in completed.stdout
    assert "actor_rollout_ref.rollout.top_p=0.95" in completed.stdout
    assert "actor_rollout_ref.rollout.temperature=0.6" in completed.stdout
    assert "actor_rollout_ref.rollout.val_kwargs.temperature=0.0" in completed.stdout
    assert "actor_rollout_ref.rollout.val_kwargs.do_sample=False" in completed.stdout
    assert "actor_rollout_ref.actor.kl_loss_coef=0.001" in completed.stdout
    assert "actor_rollout_ref.actor.entropy_coeff=0" in completed.stdout
    assert "trainer.total_training_steps=150" in completed.stdout
    assert "trainer.total_epochs=150" in completed.stdout
    assert "trainer.test_freq=10" in completed.stdout
    assert "trainer.save_freq=10" in completed.stdout
    assert "trainer.val_before_train=True" in completed.stdout
    assert "checkpoint.contents=" in completed.stdout


def test_training_refuses_to_start_without_external_ray(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]\n")
    env = dict(os.environ)
    env.pop("RAY_ADDRESS", None)
    env.update({
        "DRY_RUN": "1",
        "SPREADSHEETBENCH_DATA": str(dataset),
        "RUN_ROOT": str(tmp_path / "runs"),
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/run_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode != 0
    assert "external Ray" in completed.stderr


def test_submit_dry_run_uses_saved_ray_addresses(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]\n")
    state_file = tmp_path / "ray_address.env"
    state_file.write_text(
        "export RAY_ADDRESS=10.0.0.1:6379\n"
        "export RAY_DASHBOARD_ADDRESS=http://10.0.0.1:8265\n"
        "export NNODES=1\n"
        "export GPUS_PER_NODE=8\n"
    )
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "RAY_STATE_FILE": str(state_file),
        "SPREADSHEETBENCH_DATA": str(dataset),
        "WANDB_BASE_URL": "https://wandb.example.test",
        "WANDB_API_KEY": "must-not-appear-in-output",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/submit_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "verifying 10.0.0.1:6379" in completed.stdout
    assert "submitting through http://10.0.0.1:8265" in completed.stdout
    assert "[spreadsheet-train-submit] job_log=" in completed.stdout
    assert '"RAY_ADDRESS": "auto"' in completed.stdout
    assert '"WANDB_BASE_URL": "https://wandb.example.test"' in completed.stdout
    assert '"WANDB_API_KEY": "***"' in completed.stdout
    assert '"WANDB_DIR":' in completed.stdout
    assert '"TENSORBOARD_DIR":' in completed.stdout
    assert '"SPREADSHEETBENCH_TRAJECTORY_DIR":' in completed.stdout
    assert '"SPREADSHEETBENCH_TOOL_SET": "python"' in completed.stdout
    assert '"SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS": "600"' in completed.stdout
    assert '"SPREADSHEETBENCH_PHASE_HEARTBEAT_SECONDS": "60"' in completed.stdout
    assert "must-not-appear-in-output" not in completed.stdout
    assert "dry run complete" in completed.stdout


def test_submit_dry_run_does_not_inject_wandb_credentials(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]\n")
    state_file = tmp_path / "ray_address.env"
    state_file.write_text(
        "export RAY_ADDRESS=10.0.0.1:6379\n"
        "export RAY_DASHBOARD_ADDRESS=http://10.0.0.1:8265\n"
        "export NNODES=1\n"
        "export GPUS_PER_NODE=8\n"
    )
    env = dict(os.environ)
    env.pop("WANDB_BASE_URL", None)
    env.pop("WANDB_API_KEY", None)
    env.update({
        "DRY_RUN": "1",
        "RAY_STATE_FILE": str(state_file),
        "SPREADSHEETBENCH_DATA": str(dataset),
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/submit_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"WANDB_BASE_URL"' not in completed.stdout
    assert '"WANDB_API_KEY"' not in completed.stdout


def test_submit_dry_run_forwards_spreadsheet_rl_splits_to_ray(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "Spreadsheet-RL"
    dataset.mkdir()
    (dataset / "train_hermes.parquet").write_bytes(b"placeholder")
    (dataset / "test_verified_hermes.parquet").write_bytes(b"placeholder")
    state_file = tmp_path / "ray_address.env"
    state_file.write_text(
        "export RAY_ADDRESS=10.0.0.1:6379\n"
        "export RAY_DASHBOARD_ADDRESS=http://10.0.0.1:8265\n"
        "export NNODES=1\n"
        "export GPUS_PER_NODE=8\n"
    )
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "RAY_STATE_FILE": str(state_file),
        "SPREADSHEETBENCH_DATA_FORMAT": "spreadsheet_rl",
        "SPREADSHEET_RL_DATA_ROOT": str(dataset),
        "WANDB_API_KEY": "must-not-appear-in-output",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/submit_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "data_format=spreadsheet_rl data=" + str(dataset) in completed.stdout
    assert '"SPREADSHEETBENCH_DATA_FORMAT": "spreadsheet_rl"' in completed.stdout
    assert '"SPREADSHEET_RL_DATA_ROOT": "' + str(dataset) + '"' in completed.stdout
    assert '"SPREADSHEET_RL_TRAIN_FILE": "train_hermes.parquet"' in completed.stdout
    assert '"SPREADSHEET_RL_VAL_FILE": "test_verified_hermes.parquet"' in completed.stdout
    assert "must-not-appear-in-output" not in completed.stdout


def test_submit_dry_run_accepts_explicit_ray_addresses_without_state_file(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]\n")
    missing_state_file = tmp_path / "missing-ray-address.env"
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "RAY_STATE_FILE": str(missing_state_file),
        "RAY_ADDRESS": "10.0.0.2:6379",
        "RAY_DASHBOARD_ADDRESS": "http://10.0.0.2:8265",
        "NNODES": "2",
        "GPUS_PER_NODE": "8",
        "SPREADSHEETBENCH_DATA": str(dataset),
        "WANDB_API_KEY": "must-not-appear-in-output",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/submit_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "using explicit Ray addresses" in completed.stdout
    assert "verifying 10.0.0.2:6379" in completed.stdout
    assert "submitting through http://10.0.0.2:8265" in completed.stdout
    assert "working_dir=" + str(ROOT) in completed.stdout
    assert '"NNODES": "2"' in completed.stdout
    assert '"GPUS_PER_NODE": "8"' in completed.stdout
    assert '"N_GPUS_PER_NODE": "8"' in completed.stdout
    assert '"excludes":' in completed.stdout
    assert '".git"' in completed.stdout
    assert "must-not-appear-in-output" not in completed.stdout


def test_full_submit_forwards_qwen3_4b_model_to_ray(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]\n")
    state_file = tmp_path / "ray_address.env"
    state_file.write_text(
        "export RAY_ADDRESS=10.0.0.1:6379\n"
        "export RAY_DASHBOARD_ADDRESS=http://10.0.0.1:8265\n"
        "export NNODES=1\n"
        "export GPUS_PER_NODE=8\n"
    )
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "MODE": "full",
        "RAY_STATE_FILE": str(state_file),
        "SPREADSHEETBENCH_DATA": str(dataset),
        "ROLLOUT_MAX_MODEL_LEN": "28672",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS": "24576",
        "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU": "32768",
        "LOG_PROB_MAX_TOKEN_LEN_PER_GPU": "65536",
        "ROLLOUT_ENABLE_CHUNKED_PREFILL": "True",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/submit_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"MODE": "full"' in completed.stdout
    assert (
        '"MODEL": "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/'
        'luban/common/models/Qwen3-4B-Thinking-2507"'
    ) in completed.stdout
    assert '"ROLLOUT_MAX_MODEL_LEN": "28672"' in completed.stdout
    assert '"ROLLOUT_MAX_NUM_BATCHED_TOKENS": "24576"' in completed.stdout
    assert '"ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU": "32768"' in completed.stdout
    assert '"LOG_PROB_MAX_TOKEN_LEN_PER_GPU": "65536"' in completed.stdout
    assert '"ROLLOUT_ENABLE_CHUNKED_PREFILL": "True"' in completed.stdout


def test_training_rejects_batch_size_not_divisible_by_total_gpus(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]\n")
    env = dict(os.environ)
    env.update({
        "DRY_RUN": "1",
        "RAY_ADDRESS": "auto",
        "SPREADSHEETBENCH_DATA": str(dataset),
        "RUN_ROOT": str(tmp_path / "runs"),
        "TRAIN_BS": "3",
        "N_GPUS_PER_NODE": "2",
        "NNODES": "1",
    })

    completed = subprocess.run(
        ["bash", str(ROOT / "rl/scripts/run_spreadsheetbench_grpo.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode != 0
    assert "TRAIN_BS must be divisible by total GPUs" in completed.stderr
