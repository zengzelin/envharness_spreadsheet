# Copyright 2026 The EnvHarness Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SpreadsheetBench tool registry.

Tools are dispatched directly by `SpreadsheetBenchEnv.step` (the working
directory + subprocess execution live on the Bridge, not in env_state, so we
bypass `Tool.invoke` exactly like the alfworld / swebench bridges). These stubs
exist only so `tool_schemas()` can emit the function-call schema the Policy and
the Rules-driving HarnessAgent see in their prompts.
"""
from __future__ import annotations

from typing import Any

from envharness.core.tool import Tool


class RunPython(Tool):
    name = "run_python"
    description = (
        "Execute a Python 3 snippet inside the task's working directory "
        "(openpyxl and pandas are available). Use it to inspect the input "
        "spreadsheet and to produce the answer. You MUST save your final "
        "result to output_path. The runtime predefines input_path, output_path, "
        "working_directory, load_workbook_for_edit(), and save_workbook(). The "
        "snippet's combined stdout+stderr is returned as the next observation. "
        "Commands are stateless between calls (each runs in a fresh process), "
        "so write a self-contained script each time."
    )

    @classmethod
    def invoke(cls, env_state: Any, code: str) -> Any:
        raise NotImplementedError(
            "RunPython.invoke is unused; SpreadsheetBenchEnv.step executes the "
            "code in the episode working directory directly."
        )


class ListSheets(Tool):
    name = "list_sheets"
    description = (
        "Read the current output workbook and list worksheet names, order, "
        "visibility, active state, and dimensions."
    )

    @classmethod
    def invoke(cls, env_state: Any) -> Any:
        raise NotImplementedError("SpreadsheetBenchEnv.step executes list_sheets")


class InspectRange(Tool):
    name = "inspect_range"
    description = (
        "Read a finite A1 range from the current output workbook. cells mode "
        "returns values/formulas; summary mode returns content counts."
    )

    @classmethod
    def invoke(
        cls,
        env_state: Any,
        range: str,
        sheet_name: str = "",
        include_details: bool = False,
        mode: str = "cells",
    ) -> Any:
        raise NotImplementedError("SpreadsheetBenchEnv.step executes inspect_range")


class FindCells(Tool):
    name = "find_cells"
    description = (
        "Find text in values or formulas in the current output workbook. "
        "Optionally restrict the search to a worksheet or finite A1 range."
    )

    @classmethod
    def invoke(
        cls,
        env_state: Any,
        query: str,
        sheet_name: str = "",
        range: str = "",
        match: str = "contains",
        search_in: str = "values",
        case_sensitive: bool = False,
        include_values: bool = False,
        return_mode: str = "first",
        max_results: int = 20,
    ) -> Any:
        raise NotImplementedError("SpreadsheetBenchEnv.step executes find_cells")


class WriteRange(Tool):
    name = "write_range"
    description = (
        "Write static values to a finite range in the current output workbook. "
        "A single-cell range anchors row or matrix data, and scalar data fills "
        "the target range. Formula strings are rejected; null entries skip cells."
    )

    @classmethod
    def get_info(cls) -> dict:
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "range": {"type": "string"},
                        "data": {
                            "type": ["array", "string", "number", "boolean"]
                        },
                        "sheet_name": {"type": "string"},
                    },
                    "required": ["range", "data"],
                },
            },
        }

    @classmethod
    def invoke(
        cls,
        env_state: Any,
        range: str,
        data: list | str | int | float | bool | None,
        sheet_name: str = "",
    ) -> Any:
        raise NotImplementedError("SpreadsheetBenchEnv.step executes write_range")


class ClearRange(Tool):
    name = "clear_range"
    description = (
        "Clear values and formulas from a finite range in the current output "
        "workbook without deleting or shifting cells."
    )

    @classmethod
    def invoke(
        cls,
        env_state: Any,
        range: str,
        sheet_name: str = "",
    ) -> Any:
        raise NotImplementedError("SpreadsheetBenchEnv.step executes clear_range")


class ValidateWorkbook(Tool):
    name = "validate_workbook"
    description = (
        "Check that output_path exists, opens as an Excel workbook, and "
        "contains every sheet and range named by answer_position. This does "
        "not compare against the hidden target and does not end the episode."
    )

    @classmethod
    def invoke(cls, env_state: Any) -> Any:
        raise NotImplementedError(
            "ValidateWorkbook.invoke is unused; SpreadsheetBenchEnv.step "
            "validates the episode output directly."
        )


class Submit(Tool):
    name = "submit"
    description = (
        "Call this with no arguments once the spreadsheet at output_path is "
        "final and correct. It ends the episode and triggers Online-Judge "
        "grading of output_path against the ground truth at answer_position."
    )

    @classmethod
    def invoke(cls, env_state: Any) -> Any:
        raise NotImplementedError(
            "Submit.invoke is unused; SpreadsheetBenchEnv.step handles "
            "submission termination directly."
        )
