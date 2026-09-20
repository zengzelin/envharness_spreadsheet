"""Run SpreadsheetBench subprocesses without leaving descendants on timeout."""
from __future__ import annotations

import os
import signal
import subprocess
from typing import Any


def run_process_group(
    args: list[str], *, timeout: float, **kwargs: Any
) -> subprocess.CompletedProcess:
    if kwargs.pop("capture_output", False):
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError(
                "stdout and stderr arguments may not be used with capture_output"
            )
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    process = subprocess.Popen(args, start_new_session=True, **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # A descendant that escaped the process group may still hold a pipe.
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
            process.wait(timeout=5)
        raise
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
