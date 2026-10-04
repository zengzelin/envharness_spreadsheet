from __future__ import annotations

import json
from pathlib import Path

from envharness_rl.spreadsheetbench.rollout_summary import (
    classify_badcase,
    collect_badcase_records,
    collect_multi_call_events,
    summarize_trajectories,
)


def test_collect_badcase_records_is_deterministic_and_compact(
    tmp_path: Path,
) -> None:
    failed = tmp_path / "z-failed.json"
    failed.write_text(json.dumps({
        "task_id": "task-z",
        "steps": [{"action_valid": True}],
        "final_info": {
            "won": False,
            "score": 0.0,
            "submitted": True,
            "badcase_diagnostics": {
                "status": "complete",
                "eligible": True,
                "answer_match_ratio": 0.5,
                "failure_tags": ["answer_partially_correct"],
                "first_mismatches": [{"coordinate": "A1"}],
            },
        },
    }), encoding="utf-8")
    success = tmp_path / "a-success.json"
    success.write_text(json.dumps({
        "task_id": "task-a",
        "steps": [],
        "final_info": {"won": True, "score": 1.0},
    }), encoding="utf-8")

    records = collect_badcase_records([failed, success])

    assert records == [{
        "task_id": "task-z",
        "trajectory": str(failed),
        "score": 0.0,
        "submitted": True,
        "time_limit_reached": False,
        "diagnostic_mode": "legacy",
        "diagnostic_status": "complete",
        "diagnostic_eligible": True,
        "failure_tags": ["answer_partially_correct", "execution_clean_score_zero"],
        "evidence": {
            "answer_match_ratio": 0.5,
            "first_mismatches": [{"coordinate": "A1"}],
        },
    }]


def test_classify_badcase_combines_exact_and_workbook_tags() -> None:
    result = classify_badcase(
        final_info={
            "won": False,
            "score": 0.0,
            "submitted": False,
            "time_limit_reached": True,
            "badcase_diagnostics": {
                "version": 1,
                "mode": "full",
                "eligible": True,
                "status": "complete",
                "answer_cells_scanned": 4,
                "answer_cells_matched": 3,
                "answer_cells_mismatched": 1,
                "answer_match_ratio": 0.75,
                "answer_range_untouched": True,
                "outside_answer_cells_changed": 2,
                "formula_to_static_count": 1,
                "formula_result_mismatch_count": 0,
                "format_mismatch_count": 1,
                "diagnostic_truncated": False,
            },
        },
        steps=[{
            "action_valid": True,
            "diagnostics": {
                "env/submit_gate_reject": 1,
                "parser/tool_calls_truncated": 2,
            },
            "info": {"tool_results": [{"action_name": "submit", "ok": True}]},
        }],
    )

    assert result["tool_all_success"] is True
    assert result["failure_tags"] == [
        "answer_partially_correct",
        "answer_range_untouched",
        "execution_clean_score_zero",
        "format_mismatch_info",
        "formula_to_static",
        "likely_wrong_sheet_or_range",
        "no_submit",
        "submit_gate_unrecovered",
        "time_limit",
        "tool_call_truncated",
    ]


def test_classify_badcase_excludes_evaluator_errors() -> None:
    result = classify_badcase(
        final_info={
            "won": False,
            "score": 0.0,
            "error": "eval_error: LibreOffice recalc failed",
        },
        steps=[],
    )

    assert result["eligible"] is False
    assert result["status"] == "excluded_eval_error"
    assert result["failure_tags"] == ["eval_error"]


def test_summary_reports_badcase_rates_with_explicit_denominators(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "task_id": "complete-failure",
            "steps": [],
            "final_info": {
                "won": False,
                "submitted": True,
                "badcase_diagnostics": {
                    "status": "complete",
                    "eligible": True,
                    "answer_match_ratio": 0.5,
                    "failure_tags": [
                        "answer_partially_correct",
                        "execution_clean_score_zero",
                    ],
                },
            },
        },
        {
            "task_id": "legacy-failure",
            "steps": [],
            "final_info": {"won": False, "submitted": False},
        },
        {
            "task_id": "eval-error",
            "steps": [],
            "final_info": {
                "won": False,
                "error": "eval_error: recalc failed",
            },
        },
    ]
    paths = []
    for row in rows:
        path = tmp_path / f"{row['task_id']}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        paths.append(path)

    summary = summarize_trajectories(paths)

    assert summary["badcase_policy_episode_count"] == 2
    assert summary["badcase_complete_failure_count"] == 1
    assert summary["badcase_unavailable_failure_count"] == 1
    assert summary["execution_clean_score_zero_rate"] == 0.5
    assert summary["answer_partially_correct_rate"] == 1.0
    assert summary["answer_match_ratio_failed_mean"] == 0.5
    assert summary["badcase_tag_counts"] == {
        "answer_partially_correct": 1,
        "execution_clean_score_zero": 1,
        "no_submit": 1,
    }


def test_summary_counts_tools_errors_submission_and_success(tmp_path: Path) -> None:
    trajectory = {
        "task_id": "task-1",
        "steps": [
            {
                "projected_action": {"name": "list_sheets", "kwargs": {}},
                "action_valid": True,
                "diagnostics": {
                    "env/read_tool_call": 1,
                    "env/read_tool_success": 1,
                    "env/python_error": 0,
                },
            },
            {
                "projected_action": {"name": "submit", "kwargs": {}},
                "action_valid": True,
                "diagnostics": {"env/python_error": 0},
            },
        ],
        "final_info": {"won": True, "submitted": True},
    }
    path = tmp_path / "task-1.json"
    path.write_text(json.dumps(trajectory), encoding="utf-8")

    summary = summarize_trajectories([path])

    assert summary["episodes"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["submitted_rate"] == 1.0
    assert summary["tool_counts"] == {"list_sheets": 1, "submit": 1}
    assert summary["read_tool_success_rate"] == 1.0


def test_summary_counts_each_action_in_multi_call_turn(tmp_path: Path) -> None:
    trajectory = {
        "task_id": "task-2",
        "steps": [{
            "projected_action": [
                {"name": "inspect_range", "kwargs": {"range": "A1:B2"}},
                {"name": "fill_formula", "kwargs": {"start_cell": "C2"}},
            ],
            "projected_actions": [
                {"name": "inspect_range", "kwargs": {"range": "A1:B2"}},
                {"name": "fill_formula", "kwargs": {"start_cell": "C2"}},
            ],
            "action_valid": True,
            "diagnostics": {
                "episode/tool_calls_per_turn": 2,
                "episode/multi_call": 1,
                "env/multi_call_partial_failure": 0,
            },
        }],
        "final_info": {"won": False},
    }
    path = tmp_path / "task-2.json"
    path.write_text(json.dumps(trajectory), encoding="utf-8")

    summary = summarize_trajectories([path])

    assert summary["tool_counts"] == {"inspect_range": 1, "fill_formula": 1}
    assert summary["tool_calls_per_turn"] == 2.0
    assert summary["multi_call_ratio"] == 1.0
    assert summary["multi_call_partial_failure_rate"] == 0.0


def test_summary_reports_recalc_and_preflight_metrics(tmp_path: Path) -> None:
    path = tmp_path / "recalc.json"
    path.write_text(json.dumps({
        "task_id": "recalc",
        "steps": [{
            "projected_action": {"name": "recalculate_and_read"},
            "action_valid": True,
            "diagnostics": {
                "env/recalc_tool_call": 1,
                "env/recalc_tool_success": 1,
                "env/recalc_elapsed_ms": 250.0,
                "env/recalc_formula_error_count": 2,
                "env/python_preflight_reject": 1,
                "env/python_preflight_warning_count": 3,
            },
        }, {
            "projected_action": {"name": "submit"},
            "action_valid": True,
            "diagnostics": {
                "env/submitted_after_recalc": 1,
                "env/recalc_stale_at_submit": 0,
            },
        }],
        "final_info": {"won": False, "submitted": True},
    }), encoding="utf-8")

    summary = summarize_trajectories([path])

    assert summary["recalc_tool_call_ratio"] == 0.5
    assert summary["recalc_tool_success_rate"] == 1.0
    assert summary["recalc_formula_error_count_mean"] == 2.0
    assert summary["recalc_elapsed_ms_mean"] == 250.0
    assert summary["submitted_after_recalc_rate"] == 1.0
    assert summary["recalc_stale_at_submit_rate"] == 0.0
    assert summary["python_preflight_reject_ratio"] == 0.5
    assert summary["python_preflight_warning_count_mean"] == 1.5


def test_summary_separates_projected_and_executed_tool_calls(tmp_path: Path) -> None:
    trajectory = {
        "task_id": "task-3",
        "steps": [{
            "projected_actions": [
                {"name": "run_python", "kwargs": {"code": "broken"}},
                {"name": "submit", "kwargs": {}},
            ],
            "action_valid": True,
            "diagnostics": {
                "episode/tool_calls_per_turn": 2,
                "episode/multi_call": 1,
                "env/multi_call_partial_failure": 1,
            },
            "info": {
                "tool_results": [
                    {"action_name": "run_python", "ok": False},
                ],
                "tool_call_count": 2,
                "tool_executed_count": 1,
            },
        }],
        "final_info": {"won": False},
    }
    path = tmp_path / "task-3.json"
    path.write_text(json.dumps(trajectory), encoding="utf-8")

    summary = summarize_trajectories([path])

    assert summary["tool_counts"] == {"run_python": 1}
    assert summary["projected_tool_counts"] == {
        "run_python": 1,
        "submit": 1,
    }
    assert summary["tool_calls_per_turn"] == 1.0
    assert summary["projected_tool_calls_per_turn"] == 2.0
    assert summary["multi_call_ratio"] == 0.0
    assert summary["projected_multi_call_ratio"] == 1.0
    assert summary["multi_call_partial_failure_rate"] == 1.0


def test_summary_reports_per_call_and_per_turn_write_success(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({
        "task_id": "task-1",
        "final_info": {"won": False},
        "steps": [{
            "action_valid": True,
            "projected_actions": [
                {"name": "write_range"},
                {"name": "fill_formula"},
            ],
            "info": {"tool_results": [
                {"action_name": "write_range", "ok": True, "info": {}},
                {"action_name": "fill_formula", "ok": False, "info": {
                    "tool_error": "ValueError",
                }},
            ]},
            "diagnostics": {
                "env/write_tool_call": 1,
                "env/write_tool_success": 0,
                "env/write_tool_error": 1,
            },
        }],
    }))

    summary = summarize_trajectories([path])

    assert summary["write_call_count"] == 2
    assert summary["write_call_success_rate"] == 0.5
    assert summary["write_turn_all_success_rate"] == 0.0
    assert summary["episodes_with_write_error_rate"] == 1.0


def test_summary_reports_multi_call_completion_and_stop_reasons(tmp_path: Path) -> None:
    completed = tmp_path / "completed.json"
    completed.write_text(json.dumps({
        "task_id": "completed",
        "final_info": {"won": True},
        "steps": [{
            "projected_actions": [
                {"name": "list_sheets"},
                {"name": "inspect_range"},
            ],
            "info": {
                "tool_results": [
                    {"action_index": 0, "action_name": "list_sheets", "ok": True},
                    {"action_index": 1, "action_name": "inspect_range", "ok": True},
                ],
                "multi_call_stop_reason": "completed",
            },
            "diagnostics": {
                "env/multi_call_completed": 1,
                "env/multi_call_all_success": 1,
                "env/multi_call_short_circuit": 0,
                "env/multi_call_skipped_calls": 0,
                "env/multi_call_executed_fraction": 1.0,
                "env/multi_call_failure_index": -1,
            },
        }],
    }))
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({
        "task_id": "failed",
        "final_info": {"won": False},
        "steps": [{
            "projected_actions": [
                {"name": "run_python"},
                {"name": "submit"},
            ],
            "info": {
                "tool_results": [
                    {"action_index": 0, "action_name": "run_python", "ok": False},
                ],
                "multi_call_stop_reason": "run_python_failure",
            },
            "diagnostics": {
                "env/multi_call_partial_failure": 1,
                "env/multi_call_completed": 0,
                "env/multi_call_all_success": 0,
                "env/multi_call_short_circuit": 1,
                "env/multi_call_skipped_calls": 1,
                "env/multi_call_executed_fraction": 0.5,
                "env/multi_call_failure_index": 0,
            },
        }],
    }))

    summary = summarize_trajectories([completed, failed])

    assert summary["multi_call_projected_turns"] == 2
    assert summary["multi_call_executed_turns"] == 1
    assert summary["multi_call_completed_rate"] == 0.5
    assert summary["multi_call_all_success_rate"] == 0.5
    assert summary["multi_call_short_circuit_rate"] == 0.5
    assert summary["multi_call_mean_executed_fraction"] == 0.75
    assert summary["multi_call_skipped_call_count"] == 1
    assert summary["multi_call_stop_reasons"] == {
        "completed": 1,
        "run_python_failure": 1,
    }
    assert summary["multi_call_failure_indexes"] == {"0": 1}

    events = collect_multi_call_events([completed, failed])
    assert events[1] == {
        "task_id": "failed",
        "trajectory": str(failed),
        "step": 1,
        "projected_tools": ["run_python", "submit"],
        "executed_tools": ["run_python"],
        "projected_count": 2,
        "executed_count": 1,
        "skipped_count": 1,
        "completed": False,
        "all_success": False,
        "short_circuit": True,
        "partial_failure": True,
        "stop_reason": "run_python_failure",
        "failure_index": 0,
        "failure_tool": "run_python",
        "failure_error_type": "",
    }
