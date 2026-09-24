#!/usr/bin/env python3
"""Compare paired SpreadsheetBench validation runs by task id."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from envharness_rl.spreadsheetbench.rollout_summary import summarize_trajectories


COMPARISON_ENV_FIELDS = (
    "SPREADSHEETBENCH_DATA_FORMAT", "SPREADSHEET_RL_VAL_FILE",
    "VAL_SIZE", "MAX_STEPS",
    "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH",
    "APPLY_CHAT_TEMPLATE_ENABLE_THINKING", "ENVHARNESS_DISABLE_THINKING",
    "VAL_TEMPERATURE", "VAL_DO_SAMPLE", "SPREADSHEETBENCH_TOOL_SET",
    "SPREADSHEETBENCH_HISTORY_MODE", "SPREADSHEETBENCH_HISTORY_ACTION_CHARS",
    "SPREADSHEETBENCH_HISTORY_OBS_CHARS", "ROLLOUT_MAX_MODEL_LEN",
    "ROLLOUT_MAX_NUM_BATCHED_TOKENS",
)


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def _paired_delta_interval(deltas: list[int]) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    mean = sum(deltas) / len(deltas)
    if len(deltas) == 1:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
    margin = 1.959963984540054 * math.sqrt(variance / len(deltas))
    return [max(-1.0, mean - margin), min(1.0, mean + margin)]


def _mcnemar_exact_pvalue(base_only: int, candidate_only: int) -> float:
    discordant = base_only + candidate_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(
        math.comb(discordant, value)
        for value in range(min(base_only, candidate_only) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2 * lower_tail)


def _comparison_config(manifest: dict[str, Any]) -> dict[str, Any]:
    environment = manifest.get("environment") or {}
    return {
        "git_commit": manifest.get("git_commit"),
        **{name: environment.get(name) for name in COMPARISON_ENV_FIELDS},
    }


def _config_mismatches(
    base: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    base_config = base["comparison_config"]
    candidate_config = candidate["comparison_config"]
    unavailable = {None, "", "unknown"}
    return {
        name: {"base": base_config.get(name), "candidate": candidate_config.get(name)}
        for name in base_config
        if base_config.get(name) != candidate_config.get(name)
        or base_config.get(name) in unavailable
        or candidate_config.get(name) in unavailable
    }


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run(run_dir: Path, expected_tasks: int | None) -> dict[str, Any]:
    trajectory_dir = run_dir / "rollouts" / "env" / "val"
    paths = sorted(trajectory_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no validation trajectories found: {trajectory_dir}")

    outcomes: dict[str, bool | None] = {}
    evaluator_error_tasks: list[str] = []
    duplicates: list[str] = []
    for path in paths:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(trajectory.get("task_id") or "")
        if not task_id:
            raise ValueError(f"trajectory has no task_id: {path}")
        if task_id in outcomes:
            duplicates.append(task_id)
        final_info = trajectory.get("final_info") or {}
        error = str(final_info.get("error") or "")
        if error.startswith("eval_error:"):
            outcomes[task_id] = None
            evaluator_error_tasks.append(task_id)
        elif not isinstance(final_info.get("won"), bool):
            raise ValueError(f"trajectory has no boolean final_info.won: {path}")
        else:
            outcomes[task_id] = final_info["won"]
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(
            f"duplicate task ids in {run_dir}: {preview}; "
            "compare evaluation-only runs with one trajectory per task"
        )
    if expected_tasks is not None and len(outcomes) != expected_tasks:
        raise ValueError(
            f"expected {expected_tasks} tasks in {run_dir}, got {len(outcomes)}"
        )

    manifest = _load_manifest(run_dir)
    environment = manifest.get("environment") or {}
    summary = summarize_trajectories(paths)
    valid_outcomes = [won for won in outcomes.values() if won is not None]
    success_count = sum(valid_outcomes)
    result = {
        "run_dir": str(run_dir),
        "experiment": environment.get("EXP_NAME") or run_dir.name,
        "model": environment.get("MODEL"),
        "git_commit": manifest.get("git_commit"),
        "task_count": len(outcomes),
        "valid_task_count": len(valid_outcomes),
        "evaluator_error_count": len(evaluator_error_tasks),
        "evaluator_error_tasks": sorted(evaluator_error_tasks),
        "success_count": success_count,
        "success_rate": success_count / len(valid_outcomes) if valid_outcomes else 0.0,
        "success_rate_ci95": _wilson_interval(success_count, len(valid_outcomes)),
        "python_error_ratio": summary["python_error_ratio"],
        "write_call_success_rate": summary["write_call_success_rate"],
        "write_turn_all_success_rate": summary["write_turn_all_success_rate"],
        "episodes_with_write_error_rate": summary["episodes_with_write_error_rate"],
        "multi_call_partial_failure_rate": summary[
            "multi_call_partial_failure_rate"
        ],
        "multi_call_completed_rate": summary["multi_call_completed_rate"],
        "multi_call_all_success_rate": summary["multi_call_all_success_rate"],
        "multi_call_short_circuit_rate": summary[
            "multi_call_short_circuit_rate"
        ],
        "multi_call_mean_executed_fraction": summary[
            "multi_call_mean_executed_fraction"
        ],
        "multi_call_stop_reasons": summary["multi_call_stop_reasons"],
        "multi_call_failure_indexes": summary["multi_call_failure_indexes"],
        "tool_calls_per_turn": summary["tool_calls_per_turn"],
        "comparison_config": _comparison_config(manifest),
        "outcomes": outcomes,
    }
    return result


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in {"outcomes", "comparison_config"}
    }


def compare_runs(
    base_run: Path,
    candidate_runs: Iterable[Path],
    *,
    expected_tasks: int | None = 399,
    allow_config_mismatch: bool = False,
    allow_evaluator_errors: bool = False,
) -> dict[str, Any]:
    base = _load_run(Path(base_run), expected_tasks)
    if base["evaluator_error_count"] and not allow_evaluator_errors:
        raise ValueError(
            f"evaluator error in {base_run}: {base['evaluator_error_count']} task(s); "
            "rerun them before comparison"
        )
    candidates = []
    base_tasks = set(base["outcomes"])
    for candidate_dir in candidate_runs:
        candidate = _load_run(Path(candidate_dir), expected_tasks)
        if candidate["evaluator_error_count"] and not allow_evaluator_errors:
            raise ValueError(
                f"evaluator error in {candidate_dir}: "
                f"{candidate['evaluator_error_count']} task(s); rerun them before comparison"
            )
        mismatches = _config_mismatches(base, candidate)
        if mismatches and not allow_config_mismatch:
            names = ", ".join(sorted(mismatches))
            raise ValueError(
                f"evaluation config mismatch for {candidate_dir}: {names}; "
                "use --allow-config-mismatch only for diagnostic comparisons"
            )
        candidate_tasks = set(candidate["outcomes"])
        if candidate_tasks != base_tasks:
            missing = sorted(base_tasks - candidate_tasks)[:5]
            extra = sorted(candidate_tasks - base_tasks)[:5]
            raise ValueError(
                f"task set mismatch for {candidate_dir}: "
                f"missing={missing} extra={extra}"
            )
        both_success = base_only = candidate_only = both_failed = 0
        excluded_evaluator_error = 0
        paired_deltas: list[int] = []
        task_results = []
        for task_id in sorted(base_tasks):
            base_won = base["outcomes"][task_id]
            candidate_won = candidate["outcomes"][task_id]
            if base_won is None or candidate_won is None:
                excluded_evaluator_error += 1
                transition = "evaluator_error"
            elif base_won and candidate_won:
                both_success += 1
                transition = "both_success"
                paired_deltas.append(0)
            elif base_won:
                base_only += 1
                transition = "base_only"
                paired_deltas.append(-1)
            elif candidate_won:
                candidate_only += 1
                transition = "candidate_only"
                paired_deltas.append(1)
            else:
                both_failed += 1
                transition = "both_failed"
                paired_deltas.append(0)
            task_results.append({
                "task_id": task_id,
                "base_won": base_won,
                "candidate_won": candidate_won,
                "transition": transition,
            })
        public_candidate = _public_result(candidate)
        public_candidate["config_mismatches"] = mismatches
        public_candidate["success_rate_delta"] = (
            sum(paired_deltas) / len(paired_deltas) if paired_deltas else 0.0
        )
        public_candidate["paired"] = {
            "both_success": both_success,
            "base_only": base_only,
            "candidate_only": candidate_only,
            "both_failed": both_failed,
            "net_success_gain": candidate_only - base_only,
            "paired_task_count": len(paired_deltas),
            "excluded_evaluator_error": excluded_evaluator_error,
            "success_rate_delta_ci95": _paired_delta_interval(paired_deltas),
            "mcnemar_exact_pvalue": _mcnemar_exact_pvalue(
                base_only, candidate_only
            ),
        }
        public_candidate["task_results"] = task_results
        candidates.append(public_candidate)
    return {"base": _public_result(base), "candidates": candidates}


def render_markdown(report: dict[str, Any]) -> str:
    base = report["base"]
    lines = [
        "| Run | Success/valid | Tasks | Eval errors | Rate (95% CI) | "
        "Paired delta (95% CI) | "
        "McNemar p | Candidate-only | Base-only | Python error | Write-call success | "
        "Multi-call complete | Multi-call short circuit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
        (
            f"| {base['experiment']} | {base['success_count']}/{base['valid_task_count']} | "
            f"{base['task_count']} | {base['evaluator_error_count']} | "
            f"{base['success_rate']:.3f} "
            f"[{base['success_rate_ci95'][0]:.3f}, {base['success_rate_ci95'][1]:.3f}] | "
            f"- | - | - | - | "
            f"{base['python_error_ratio']:.3f} | "
            f"{base['write_call_success_rate']:.3f} | "
            f"{base['multi_call_completed_rate']:.3f} | "
            f"{base['multi_call_short_circuit_rate']:.3f} |"
        ),
    ]
    for candidate in report["candidates"]:
        paired = candidate["paired"]
        lines.append(
            f"| {candidate['experiment']} | "
            f"{candidate['success_count']}/{candidate['valid_task_count']} | "
            f"{candidate['task_count']} | {candidate['evaluator_error_count']} | "
            f"{candidate['success_rate']:.3f} "
            f"[{candidate['success_rate_ci95'][0]:.3f}, "
            f"{candidate['success_rate_ci95'][1]:.3f}] | "
            f"{candidate['success_rate_delta']:+.3f} "
            f"[{paired['success_rate_delta_ci95'][0]:+.3f}, "
            f"{paired['success_rate_delta_ci95'][1]:+.3f}] | "
            f"{paired['mcnemar_exact_pvalue']:.4f} | "
            f"{paired['candidate_only']} | {paired['base_only']} | "
            f"{candidate['python_error_ratio']:.3f} | "
            f"{candidate['write_call_success_rate']:.3f} | "
            f"{candidate['multi_call_completed_rate']:.3f} | "
            f"{candidate['multi_call_short_circuit_rate']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_run", type=Path)
    parser.add_argument("candidate_runs", type=Path, nargs="+")
    parser.add_argument("--expected-tasks", type=int, default=399)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--task-jsonl-output", type=Path)
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--allow-evaluator-errors", action="store_true")
    args = parser.parse_args()

    try:
        report = compare_runs(
            args.base_run,
            args.candidate_runs,
            expected_tasks=args.expected_tasks,
            allow_config_mismatch=args.allow_config_mismatch,
            allow_evaluator_errors=args.allow_evaluator_errors,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    markdown = render_markdown(report)
    print(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown + "\n", encoding="utf-8")
    if args.task_jsonl_output:
        args.task_jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        with args.task_jsonl_output.open("w", encoding="utf-8") as stream:
            for candidate in report["candidates"]:
                for task_result in candidate["task_results"]:
                    row = {
                        "candidate_experiment": candidate["experiment"],
                        **task_result,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
