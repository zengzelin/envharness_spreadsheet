# SpreadsheetBench Badcase Observability Implementation Plan

> **历史计划说明（实施期：2026-10-01 至 2026-10-09）：** 本文的 task checklist 保留
> 原设计与审计过程；核心能力已经实现，少数运行验收仍待补充。判断当前完成情况请使用
> [`spreadsheet_work_agent_status.md`](spreadsheet_work_agent_status.md)，不要按下方未回填的
> granular checkbox 重新实现功能。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status（2026-10-09）：** 已实现并进入 `main`。核心实现提交为 `18f9142`，训练 shuffle
与日志级别配套修改在 `45426b7`。下方原始 task checklist 保留作为设计/审计记录；当前
验收状态以“Implementation status”一节为准。

**Goal:** Preserve the official SpreadsheetBench score and reward semantics while adding bounded, structured observability that explains failed episodes and produces per-task badcase records during experiments.

**Architecture:** Keep official grading as the source of truth. Derive cheap execution tags from the existing trajectory for every episode, and optionally run a bounded workbook diagnostic only after a formally graded failure. Store the compact result in `final_info`, let the existing trajectory writer persist it, and aggregate the same schema into JSON/JSONL, Markdown, and validation metrics.

**Tech Stack:** Python 3.11, openpyxl, existing EnvHarness SpreadsheetBench bridge, verl-agent trajectory/metrics integration, pytest.

**Spec:** This document, Sections 1-9.

## Global Constraints

- Do not change `won`, `score`, `reward/workbook_score`, execution penalties, or the official `compare_workbooks()` pass/fail result.
- Do not add a new reward component from answer match ratio or failure tags in this change.
- Diagnostic failures must never turn an otherwise valid evaluation into an evaluator error or policy failure.
- Training uses `light` diagnostics by default; formal evaluation uses `full`; `off` remains available.
- Full diagnostics run only after `score == 0` and only after the official LibreOffice recalc and comparison have completed.
- A full diagnostic inspects at most 200,000 unique cell coordinates across all answer and outside-answer passes combined, and retains at most 20 mismatch examples.
- Persist compact JSON only. Do not retain complete failed `.xlsx` files in this phase.
- Truncate serialized cell/formula examples to 200 characters and never print `run_python` source code to console logs.
- Failure classification is multi-label. Heuristic labels must use a `likely_` prefix.
- Formatting diagnostics are informational because the current official scorer compares values only.
- Existing schema-v1 trajectories must remain readable by all summary and comparison tools.
- Existing uncommitted files under `md/` are user-owned and must not be overwritten or reformatted as part of implementation.

## Review Focus

- Missing/corrupt output workbooks: keep the official failure result and emit a bounded diagnostic status instead of raising a second error; covered in Task 1 tests.
- Very large or whole-column answer ranges: stop at 200,000 scanned cells and set `diagnostic_truncated=true`; covered in Task 1 tests.
- Formula cells with equivalent values but different formula text: do not call them semantic errors when the recalculated values match; covered in Task 1 tests.
- Old trajectories without `badcase_diagnostics`: summarize them without crashing and mark diagnosis unavailable; covered in Task 3 tests.
- Evaluator errors (`eval_error:`): exclude them from policy badcase rates and paired failure transitions; covered in Tasks 3 and 4 tests.

---

## 1. Current baseline

Implementation baseline is SG1 `main` at commit `d92e6a2`.

Already available:

- step-level raw model output, projected calls, tool results, observations, diagnostics, and `final_info` in `rollouts/env/{train,val}/*.json`;
- parser, read/write/recalc, rollback, submit-gate, formula-validation, multi-call, and truncation signals;
- official value comparison over `answer_position`;
- aggregate trajectory summary and paired Base/candidate comparison.

Current gaps:

- the official evaluator returns only binary success plus the first value mismatch;
- there is no matched/total answer-cell count;
- there is no compact comparison of initial, output, and golden workbook states;
- formula-to-static and formula-result failures are not distinguished;
- no common multi-label badcase schema exists;
- paired comparison reports task transitions but not failure-tag transitions;
- some hardening fields exist at step level but are not summarized as episode recovery outcomes.

## 2. Selected approach

### 2.1 Diagnostic modes

`SPREADSHEETBENCH_BADCASE_DIAGNOSTICS` accepts:

- `off`: no new classification or workbook diagnostics;
- `light`: derive execution-only episode tags from trajectory/final state;
- `full`: `light` plus bounded workbook comparison for officially failed tasks.

Defaults:

- regular GRPO training/fast validation: `light`;
- `submit_spreadsheetbench_eval.sh`: `full`;
- invalid values fail at startup with a clear configuration error.

Supporting limits:

- `SPREADSHEETBENCH_BADCASE_MAX_SCAN_CELLS=200000`;
- `SPREADSHEETBENCH_BADCASE_MAX_EXAMPLES=20`.

All three variables must be written to `run_manifest.json` so different runs are not silently compared under different diagnostic protocols.

### 2.2 Data flow

```text
official grade
    |
    +-- success --------------------------> existing final_info
    |
    +-- evaluator error ------------------> diagnosis excluded
    |
    +-- policy failure
           |
           +-- light execution evidence
           +-- full bounded workbook diff (formal eval)
                        |
                        v
                 badcase_diagnostics
                        |
              existing trajectory JSON
                        |
          summary / JSONL / paired comparison / W&B
```

The bridge computes workbook evidence while the episode workdir still exists. The manager enriches it with episode-level execution evidence before writing the trajectory.

## 3. Badcase schema

Each terminal trajectory gains:

```json
{
  "schema_version": 2,
  "final_info": {
    "badcase_diagnostics": {
      "version": 1,
      "mode": "full",
      "eligible": true,
      "status": "complete",
      "tool_all_success": true,
      "answer_cells_total": 120,
      "answer_cells_scanned": 120,
      "answer_cells_matched": 87,
      "answer_cells_mismatched": 33,
      "answer_match_ratio": 0.725,
      "answer_cells_changed_from_input": 87,
      "outside_answer_cells_changed": 12,
      "mutated_sheets": ["Sheet1"],
      "formula_cells_expected": 24,
      "formula_cells_output": 12,
      "formula_to_static_count": 12,
      "formula_result_mismatch_count": 8,
      "format_mismatch_count": 3,
      "diagnostic_truncated": false,
      "first_mismatches": [],
      "failure_tags": ["answer_partially_correct", "formula_to_static"]
    }
  }
}
```

`status` is one of `complete`, `light_only`, `skipped_success`, `excluded_eval_error`, `output_missing`, `load_error`, or `diagnostic_error`.

Each mismatch example contains only sheet, coordinate, value/formula type, and bounded initial/output/golden representations. Full workbook paths and Python source are not included.

## 4. Classification rules

Tags are deterministic and multi-label:

| Tag | Rule | Confidence |
| --- | --- | --- |
| `eval_error` | terminal error begins with `eval_error:` | exact; excluded from policy badcase denominator |
| `no_submit` | terminal `submitted` is false | exact |
| `time_limit` | terminal `time_limit_reached` is true | exact |
| `submit_gate_unrecovered` | at least one submit-gate rejection and no final successful submit | exact |
| `tool_call_truncated` | any step has `parser/tool_calls_truncated > 0` | exact |
| `tool_error` | any executed `tool_results[].ok` is false | exact |
| `execution_clean_score_zero` | score is zero, no evaluator error, all parsed/executed calls succeeded, and terminal workbook validation did not fail | exact for execution, not semantic correctness |
| `answer_range_untouched` | all scanned answer cells equal input while golden differs from input in at least one scanned answer cell | exact within scan limit |
| `likely_wrong_sheet_or_range` | `answer_range_untouched` and at least one changed cell exists outside the answer range | heuristic |
| `answer_partially_correct` | `0 < answer_cells_matched < answer_cells_scanned` | exact within scan limit |
| `formula_to_static` | golden raw cell is a formula and output raw cell is not | exact structural observation |
| `formula_result_mismatch` | output raw cell is a formula and its recalculated value differs from golden | exact value result; formula semantics are inferred |
| `formula_validation_error` | existing static/recalc formula validation reports an error | exact |
| `format_mismatch_info` | normalized style differs in scanned answer cells | informational; never presented as official score cause |
| `diagnostic_truncated` | scan or example limit was reached | exact |

Do not label different-but-equivalent formulas as failures when their recalculated values match.

## 5. Metrics to add

### 5.1 Episode/validation metrics

- `val/env/execution_clean_score_zero_rate`
- `val/env/no_submit_rate`
- `val/env/time_limit_rate`
- `val/env/submit_gate_unrecovered_rate`
- `val/env/tool_call_truncated_episode_rate`
- `val/env/mutation_budget_exceeded_episode_rate`
- `val/env/answer_match_ratio_failed_mean`
- `val/env/answer_range_untouched_rate`
- `val/env/likely_wrong_sheet_or_range_rate`
- `val/env/answer_partially_correct_rate`
- `val/env/formula_to_static_rate`
- `val/env/formula_result_mismatch_rate`
- `val/env/diagnostic_truncated_rate`
- `val/env/diagnostic_error_rate`

Ratios must document their denominator. Workbook-derived rates use only eligible failed episodes with `status=complete`; operational rates use non-evaluator-error episodes. Do not silently treat unavailable diagnostics as zero.

### 5.2 Offline outputs

Extend the rollout summary CLI with:

```text
--badcase-jsonl-output PATH
--badcase-summary-output PATH
```

The JSONL contains one row per failed task with task ID, trajectory path, score/submission state, tags, compact evidence, and diagnostic availability. The summary JSON contains counts, rates with denominators, and tag co-occurrence counts.

Extend paired evaluation output so each task row includes Base/candidate tags and tag transition. Aggregate at least:

- failure tag counts per run;
- `base_only` and `candidate_only` tags;
- `both_failed` tag transitions;
- unavailable/full/truncated diagnostic counts.

## 6. Performance and failure isolation

- `light` performs no additional workbook load or cell scan.
- `full` runs only after an official policy failure, never before grading.
- Iterate answer cells lazily and stop at the configured cap.
- For outside-answer mutation evidence, scan the union of populated dimensions from input and output. Answer and outside-answer passes consume one shared coordinate budget; revisiting a coordinate for value, formula, and style evidence does not consume the budget again.
- Treat formula and style comparison as evidence collection, not pass/fail logic.
- Catch every diagnostic exception at the bridge boundary and return `status=diagnostic_error`, `diagnostic_error_type`, and a bounded message.
- Do not print mismatch values, formula text, or workbook paths to training console logs.

## 7. Documentation outcomes

After implementation and verification:

- append a dated section to `md/rollout_badcase_report.md` describing the taxonomy and first observed distributions;
- update `md/independent_evaluation_plan.md` with the exact formal-eval flags and generated artifact paths;
- update `md/spreadsheet_rl_vs_envharness_comparison.md` to distinguish official value scoring from informational formula/style diagnostics;
- keep run-specific results in the existing experiment-results ledger rather than embedding checkpoint numbers in code documentation.

Because those three files are currently modified in SG1, implementation must preserve their existing user changes and append narrowly scoped sections only.

## 8. Acceptance criteria

1. Official success/failure results are byte-for-byte unchanged for the same workbook pair.
2. A failed task with a valid output receives either complete bounded diagnostics or an explicit non-complete status.
3. Evaluator errors are never counted as policy badcases.
4. Old schema-v1 trajectories remain summarizable and comparable.
5. Whole-column/large-range diagnostics terminate at 200,000 cells and mark truncation.
6. Formal evaluation produces per-task badcase JSONL and an aggregate summary without retaining full failed workbooks.
7. W&B and offline rates expose denominators or unavailable counts.
8. Console logs contain neither full Python source nor full cell/formula payloads.
9. All `rl/tests` pass in the training image.

## 9. Alternatives considered

### A. Save every failed workbook and analyze later

Rejected for the first iteration because it creates uncontrolled storage, retention, and privacy costs and still lacks a common schema.

### B. Compute full diagnostics on every training rollout

Rejected because most training episodes fail early in training and repeated openpyxl scans would distort throughput.

### C. Convert partial correctness into reward immediately

Rejected because diagnostic match ratios have not yet been validated as stable reward signals. This plan first collects evidence without changing optimization behavior.

---

## Implementation status（2026-10-09）

- [x] Task 1：有界 workbook failure diagnostics 已实现并有专项测试；
- [x] Task 2：bridge 集成、`off|light|full` 配置和诊断故障隔离已实现；
- [x] Task 3：schema-v2、多标签分类、旧 schema 兼容和 denominator-aware 汇总已实现；
- [x] Task 4：`badcases.jsonl`、`badcase_summary.json` 和 paired transition 已实现；
- [x] Task 5：trainer/W&B 指标、launcher 默认值、manifest 字段和评测后自动报告已实现；
- [x] Task 6 的代码、文档和真实 full-399 formal-eval 已完成；NJ5
  `runs/parallel_eval/shufflefix_20261004_202150` 已生成 12 个评测点的报告；
- [ ] 在当前 `main` 上重新跑完整 `rl/tests`：2026-10-09 文档编辑节点没有 `python`
  命令，仍需在训练镜像完成最终回归；
- [ ] 独立 CPU worker smoke 没有可核对的运行产物，暂不标记完成。

实际 full-399 结果显示观测链路没有改变官方评分，并成功揭示“tool error 下降但
execution-clean score-zero 上升”的分离现象。具体分布见 `md/rollout_badcase_report.md`，
checkpoint 结果见 `md/experiment_results_20260930.md`。

## Historical implementation task checklist

以下 checkbox 是实施前按 commit 粒度编写的原始步骤，不再作为当前完成状态来源。实际改动
集中提交于 `18f9142`，没有强行拆成计划中的六个 commit。

### Task 1: Bounded workbook failure diagnostics

**Files:**
- Create: `envharness/bridges/spreadsheetbench/badcase_diagnostics.py`
- Test: `rl/tests/test_spreadsheetbench_badcase_diagnostics.py`

**Interfaces:**
- Produces: `DiagnosticLimits(max_scan_cells: int, max_examples: int)`.
- Produces: `diagnose_failed_workbook(input_path: str, golden_path: str, output_path: str, answer_position: str, *, limits: DiagnosticLimits) -> dict[str, Any]`.
- Produces: compact workbook evidence only; it does not assign execution-history tags or official scores.

- [ ] **Step 1: Add failing tests for answer-cell match counts and bounded mismatch examples.**

  Cover exact match counts, first-20 cap, and scalar serialization capped at 200 characters.

- [ ] **Step 2: Add failing tests for input/output mutation evidence.**

  Cover untouched answer range, changes outside the answer range, multiple sheets, and scan truncation at exactly 200,000 cells.

- [ ] **Step 3: Add failing tests for formula and style evidence.**

  Cover formula-to-static, formula result mismatch, equivalent result with different formula text, and informational normalized style mismatch.

- [ ] **Step 4: Implement the pure bounded diagnostic module.**

  Reuse `_split_answer_position` and existing cell normalization where possible; do not modify `compare_workbooks()`.

- [ ] **Step 5: Run focused tests.**

  Run: `PYTHONPATH=.:rl:third_party/verl-agent python -m pytest -q rl/tests/test_spreadsheetbench_badcase_diagnostics.py`

  Expected: all tests pass.

- [ ] **Step 6: Commit Task 1.**

  Commit message: `feat: add bounded spreadsheet badcase diagnostics`

### Task 2: Bridge integration and fault isolation

**Files:**
- Modify: `envharness/bridges/spreadsheetbench/bridge.py`
- Modify: `rl/envharness_rl/spreadsheetbench/envs.py`
- Modify: `rl/integration/verl_agent_spreadsheetbench_hardening.patch`
- Test: `rl/tests/test_spreadsheetbench_bridge_execution.py`
- Test: `rl/tests/test_spreadsheetbench_envs.py`

**Interfaces:**
- Consumes: `DiagnosticLimits` and `diagnose_failed_workbook()` from Task 1.
- Produces: terminal `final_info.badcase_diagnostics` workbook evidence and explicit diagnostic status.
- Produces: reset options `badcase_diagnostics_mode`, `badcase_max_scan_cells`, and `badcase_max_examples`.

- [ ] **Step 1: Add failing configuration tests.**

  Assert `off|light|full` parsing, exact default limits, and startup failure for invalid modes/non-positive limits.

- [ ] **Step 2: Add failing bridge tests for full-mode failure diagnostics.**

  Assert diagnostics run only after official failure, do not run on success, and do not alter `success`, `score`, or official `diff`.

- [ ] **Step 3: Add failing fault-isolation tests.**

  Force diagnostic load/scan errors and assert official failure remains intact with `status=diagnostic_error`.

- [ ] **Step 4: Implement bridge integration after official grading.**

  Use the already recalculated output and golden copy. Keep official comparison order and result unchanged.

- [ ] **Step 5: Wire configuration through verl-agent integration.**

  Update both the checked-out `third_party/verl-agent` target and the reproducible hardening patch; ensure a fresh `fetch_verl_agent.sh` checkout receives identical options.

- [ ] **Step 6: Run focused tests.**

  Run: `PYTHONPATH=.:rl:third_party/verl-agent python -m pytest -q rl/tests/test_spreadsheetbench_bridge_execution.py rl/tests/test_spreadsheetbench_envs.py rl/tests/test_spreadsheetbench_training_scripts.py`

  Expected: all tests pass.

- [ ] **Step 7: Commit Task 2.**

  Commit message: `feat: record failed workbook diagnostics`

### Task 3: Episode classification and trajectory schema

**Files:**
- Modify: `rl/envharness_rl/spreadsheetbench/manager.py`
- Modify: `rl/envharness_rl/spreadsheetbench/rollout_summary.py`
- Test: `rl/tests/test_spreadsheetbench_verl_manager.py`
- Test: `rl/tests/test_spreadsheetbench_rollout_summary.py`

**Interfaces:**
- Produces: `classify_badcase(*, final_info: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]` with execution evidence and sorted unique `failure_tags`.
- Produces: schema-v2 terminal trajectories while accepting schema v1 in readers.
- Produces: aggregate counts and denominator-aware rates from `summarize_trajectories()`.

- [ ] **Step 1: Add failing tests for exact execution tags.**

  Cover no-submit, time-limit, submit-gate recovery/unrecovered, truncation, mutation-budget rejection, tool error, and execution-clean score-zero.

- [ ] **Step 2: Add failing tests for multi-label merge.**

  Assert bridge workbook tags merge with execution tags without duplicates and heuristic tags keep the `likely_` prefix.

- [ ] **Step 3: Add backward-compatibility and denominator tests.**

  Assert schema-v1 trajectories are accepted, unavailable full diagnostics are counted separately, and evaluator errors are excluded.

- [ ] **Step 4: Implement classification before `_dump_trajectory()`.**

  Append the terminal step first, classify from the complete step list plus `final_info`, then dump. Add only compact fields to `final_info`; keep raw `model_output` behavior unchanged.

- [ ] **Step 5: Extend `summarize_trajectories()`.**

  Return tag counts, tag co-occurrences, eligible/unavailable/truncated counts, and all metrics from Section 5 with explicit denominator fields.

- [ ] **Step 6: Run focused tests.**

  Run: `PYTHONPATH=.:rl:third_party/verl-agent python -m pytest -q rl/tests/test_spreadsheetbench_verl_manager.py rl/tests/test_spreadsheetbench_rollout_summary.py`

  Expected: all tests pass.

- [ ] **Step 7: Commit Task 3.**

  Commit message: `feat: classify spreadsheet rollout badcases`

### Task 4: Offline badcase and paired reports

**Files:**
- Modify: `rl/scripts/summarize_spreadsheetbench_rollouts.py`
- Modify: `rl/scripts/compare_spreadsheetbench_evals.py`
- Modify: `rl/tests/test_spreadsheetbench_rollout_summary.py`
- Modify: `rl/tests/test_spreadsheetbench_training_scripts.py`

**Interfaces:**
- Consumes: Task 3 classification and summary schema.
- Produces: per-task badcase JSONL and aggregate badcase JSON.
- Produces: Base/candidate failure tags and tag transitions in paired task JSONL.

- [ ] **Step 1: Add failing CLI/output tests for badcase JSONL and summary JSON.**

  Assert deterministic task ordering, trajectory provenance, compact evidence, counts, rates, denominators, and tag co-occurrences.

- [ ] **Step 2: Add failing paired-transition tests.**

  Cover `candidate_only`, `base_only`, and `both_failed` with changed tags; evaluator errors remain excluded.

- [ ] **Step 3: Implement the two summary CLI flags.**

  Preserve current stdout JSON and `--multi-call-events-output` behavior.

- [ ] **Step 4: Enrich paired comparison outputs.**

  Preserve existing success statistics and McNemar calculation; add diagnosis availability and tag transitions without changing task outcome logic.

- [ ] **Step 5: Run focused tests.**

  Run: `PYTHONPATH=.:rl:third_party/verl-agent python -m pytest -q rl/tests/test_spreadsheetbench_rollout_summary.py rl/tests/test_spreadsheetbench_training_scripts.py`

  Expected: all tests pass.

- [ ] **Step 6: Commit Task 4.**

  Commit message: `feat: report spreadsheet badcase transitions`

### Task 5: Validation metrics, launch defaults, and manifests

**Files:**
- Modify: `rl/envharness_rl/spreadsheetbench/manager.py`
- Modify: `third_party/verl-agent/verl/trainer/ppo/ray_trainer.py`
- Modify: `rl/integration/verl_agent_spreadsheetbench_hardening.patch`
- Modify: `rl/scripts/run_spreadsheetbench_grpo.sh`
- Modify: `rl/scripts/submit_spreadsheetbench_grpo.sh`
- Modify: `rl/scripts/submit_spreadsheetbench_eval.sh`
- Modify: `rl/tests/test_spreadsheetbench_training_scripts.py`
- Modify: `rl/tests/test_spreadsheetbench_verl_manager.py`

**Interfaces:**
- Consumes: Task 3 episode metrics.
- Produces: Section 5 validation metrics in trainer logs/W&B.
- Produces: manifest-recorded diagnostic mode and limits.

- [ ] **Step 1: Add failing tests for validation metric propagation and denominators.**

  Assert unavailable full diagnostics are not averaged as zeros and evaluator errors do not enter policy rates.

- [ ] **Step 2: Add failing launch/manifest tests.**

  Assert training defaults to `light`, formal evaluation defaults to `full`, overrides work, and all variables appear in `run_manifest.json`.

- [ ] **Step 3: Implement manager/trainer metric propagation.**

  Follow existing SpreadsheetBench diagnostic-key mappings and update the reproducible integration patch in the same change.

- [ ] **Step 4: Implement launch defaults and automatic report generation.**

  After a successful formal evaluation, generate `badcases.jsonl` and `badcase_summary.json` under the run directory. Report generation failure must be visible and must not rewrite evaluation outcomes.

- [ ] **Step 5: Run focused tests.**

  Run: `PYTHONPATH=.:rl:third_party/verl-agent python -m pytest -q rl/tests/test_spreadsheetbench_verl_manager.py rl/tests/test_spreadsheetbench_training_scripts.py`

  Expected: all tests pass.

- [ ] **Step 6: Commit Task 5.**

  Commit message: `feat: expose spreadsheet badcase metrics`

### Task 6: Documentation and end-to-end verification

**Files:**
- Modify: `md/rollout_badcase_report.md`
- Modify: `md/independent_evaluation_plan.md`
- Modify: `md/spreadsheet_rl_vs_envharness_comparison.md`
- Modify: `rl/scripts/smoke_spreadsheetbench_worker.py`
- Test: `rl/tests`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: documented experiment workflow and a smoke artifact proving the trajectory/report path.

- [ ] **Step 1: Extend the worker smoke to create one controlled failed trajectory.**

  Assert it contains the expected compact badcase schema and no persisted workbook artifact.

- [ ] **Step 2: Append the documentation sections from Section 7.**

  Preserve all existing uncommitted user text. Document that format evidence is informational and that failure tags do not affect reward.

- [ ] **Step 3: Run the complete RL test suite.**

  Run: `PYTHONPATH=.:rl:third_party/verl-agent python -m pytest -q rl/tests`

  Expected: all tests pass.

- [ ] **Step 4: Run the CPU worker smoke in the training image.**

  Run: `PYTHONPATH=.:rl:third_party/verl-agent python rl/scripts/smoke_spreadsheetbench_worker.py`

  Expected: controlled success/failure episodes complete and generated diagnostics satisfy the schema.

- [ ] **Step 5: Run one small formal-eval smoke.**

  Use `VAL_SIZE=4`, deterministic decoding, and `SPREADSHEETBENCH_BADCASE_DIAGNOSTICS=full`; verify trajectory, `badcases.jsonl`, `badcase_summary.json`, and W&B/log metrics agree on denominators.

- [ ] **Step 6: Review the diff for scorer/reward neutrality.**

  Confirm no modifications to official pass/fail comparison, reward calculation, or invalid-action penalty coefficients.

- [ ] **Step 7: Commit Task 6.**

  Commit message: `docs: document spreadsheet badcase observability`

## Execution order and decision gate

Execute Tasks 1-6 sequentially because later schemas depend on earlier interfaces. Do not begin Task 1 until this plan is approved. Before full Verified-399 evaluation, first inspect the 4-task smoke for runtime overhead and schema correctness.
