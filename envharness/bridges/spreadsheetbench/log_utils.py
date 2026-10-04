"""Small logging gate shared by SpreadsheetBench bridge components."""
from __future__ import annotations

import os


LOG_LEVEL_ENV = "SPREADSHEETBENCH_LOG_LEVEL"
VALID_LOG_LEVELS = frozenset({"verbose", "error"})


def spreadsheet_log_level(raw_value: str | None = None) -> str:
    """Return the configured SpreadsheetBench console log level."""
    value = (
        os.environ.get(LOG_LEVEL_ENV, "verbose")
        if raw_value is None
        else raw_value
    )
    normalized = str(value).strip().lower()
    if normalized not in VALID_LOG_LEVELS:
        expected = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ValueError(
            f"{LOG_LEVEL_ENV} must be one of {expected}, got {value!r}"
        )
    return normalized


def spreadsheet_log(message: str, *, error: bool = False) -> None:
    """Print normal events in verbose mode and always print error events."""
    if error or spreadsheet_log_level() == "verbose":
        print(message, flush=True)
