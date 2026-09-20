from __future__ import annotations

import subprocess
import sys
import time

import pytest

from envharness.bridges.spreadsheetbench.process_utils import run_process_group


def test_run_process_group_returns_completed_output() -> None:
    result = run_process_group(
        [sys.executable, "-c", "print('ready')"],
        timeout=2,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == "ready\n"


def test_run_process_group_kills_descendants_holding_stdout() -> None:
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'], "
        "stdout=sys.stdout, stderr=sys.stderr)\n"
        "time.sleep(5)\n"
    )
    started = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired):
        run_process_group(
            [sys.executable, "-c", code],
            timeout=0.2,
            capture_output=True,
            text=True,
        )

    assert time.monotonic() - started < 1.5
