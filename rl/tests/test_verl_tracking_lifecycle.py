from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[2]


def _load_tracking_module():
    path = ROOT / "third_party/verl-agent/verl/utils/tracking.py"
    spec = importlib.util.spec_from_file_location("verl_tracking_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracking_finish_is_explicit_and_idempotent(monkeypatch) -> None:
    finish_calls = []
    wandb = types.ModuleType("wandb")
    wandb.init = lambda **kwargs: None
    wandb.finish = lambda exit_code=0: finish_calls.append(exit_code)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    module = _load_tracking_module()
    tracking = module.Tracking("project", "run", ["wandb"], config={})

    tracking.finish()
    tracking.finish()
    tracking.__del__()

    assert finish_calls == [0]


def test_ppo_trainer_explicitly_finishes_tracking_before_return() -> None:
    source = (
        ROOT / "third_party/verl-agent/verl/trainer/ppo/ray_trainer.py"
    ).read_text()

    assert source.count("logger.finish()") >= 2
