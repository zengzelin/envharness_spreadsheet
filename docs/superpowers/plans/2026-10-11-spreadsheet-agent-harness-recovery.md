# Spreadsheet Agent Harness Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent edit ledger, temporary-copy recalculation during validation, and recoverable soft tool-call constraints to the SpreadsheetBench RL harness.

**Architecture:** The bridge remains authoritative for workbook revisions and validation state; a focused recalculation helper validates cached answer-range values on a temporary copy. The worker owns invalid-streak termination, while the manager owns bounded ledger rendering and metric aggregation. Projection performs only unambiguous syntax recovery and early schema checks; runtime tools remain authoritative.

**Tech Stack:** Python 3, pytest, openpyxl, LibreOffice headless recalculation, Ray-compatible environment adapters, Bash/Hydra launch scripts.

**Spec:** `docs/superpowers/specs/2026-10-11-spreadsheet-agent-harness-recovery-design.md`

## Global Constraints

- Preserve transactional Python execution, static formula validation, workbook revisions, the submit gate, four-call truncation, and existing metric semantics.
- Keep official SpreadsheetBench evaluation binary and unchanged; never consult the golden workbook during validation.
- Edit ledger keeps 12 whole entries and renders at most 4,000 characters; Python source is never retained.
- Missing-end-tag recovery is allowed only for exactly one complete JSON object followed by whitespace.
- Invalid-streak termination defaults to 3 consecutive invalid batches and resets on a valid projected batch.
- Recalc-aware validation operates on a temporary copy and never mutates the committed output workbook.
- Direct bridge callers remain static-only by default; training and evaluation explicitly enable recalculation validation.
- Do not add hard grammar-constrained decoding or reward shaping.
- Preserve all unrelated user changes in the dirty worktree; each commit stages only files named by its task.

## Review Focus

- A whole-column answer range must use actual worksheet bounds, scan for every formula error, and still emit only a bounded preview; Task 2 adds this test.
- A multi-call batch that short-circuits must ledger only calls that actually executed, not merely projected calls; Task 4 adds this test.
- A valid projected tool that fails at runtime must reset/not increment the parser-invalid streak; Task 6 adds this test.
- Missing `</tool_call>` recovery must reject a valid JSON object followed by prose or a second object; Task 5 adds this test.
- A validation infrastructure failure must leave both revisions stale and must not be misclassified as a policy formula error; Task 3 adds this test.

---

### Task 1: Uniform workbook revision state reporting

**Files:**
- Modify: `envharness/bridges/spreadsheetbench/bridge.py`
- Test: `rl/tests/test_spreadsheetbench_bridge_execution.py`

**Interfaces:**
- Produces: `SpreadsheetBenchEnv._revision_state_info() -> dict[str, int | bool]` returning `workbook_revision`, `last_validation_revision`, `last_recalc_revision`, `validation_current`, and `recalc_current`.
- Produces: every `SpreadsheetBenchEnv.step()` response merges those five fields into `info` after the action's state transition.

- [ ] **Step 1: Write failing state-contract tests**

Add `test_bridge_reports_revision_state_after_read_write_validate_and_rejected_submit`. Assert reset starts at revision `0`, a successful mutation advances it, validation/recalc flags become stale, a successful validation marks the current revision, and a later mutation makes the flags stale again. Add `test_failed_mutation_does_not_advance_reported_revision`.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest -q rl/tests/test_spreadsheetbench_bridge_execution.py -k 'revision_state'`

Expected: FAIL because the uniform revision fields are absent.

- [ ] **Step 3: Implement `_revision_state_info()` and merge it on every step return**

Use revision equality for the two booleans. Ensure merge happens after state changes and cannot be overridden by older fields in a tool payload.

- [ ] **Step 4: Run bridge tests**

Run: `pytest -q rl/tests/test_spreadsheetbench_bridge_execution.py`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add envharness/bridges/spreadsheetbench/bridge.py rl/tests/test_spreadsheetbench_bridge_execution.py
git commit -m "feat: report spreadsheet workbook revision state"
```

### Task 2: Temporary-copy answer-range recalculation validator

**Files:**
- Modify: `envharness/bridges/spreadsheetbench/recalc_tools.py`
- Test: `rl/tests/test_spreadsheetbench_bridge_execution.py`

**Interfaces:**
- Produces: `validate_recalculated_answer_ranges(path: str, answer_position: str, answer_sheet: str, *, soffice_path: str | None = None, timeout: int = DEFAULT_RECALC_TIMEOUT, max_preview_cells: int = 40) -> tuple[dict[str, Any], str]`.
- Payload fields: `status`, `updated=False`, `values_source`, `formula_cells`, `formula_error_cells`, `elapsed_ms`, `ranges`, and `preview_truncated`.
- Errors: raises `ReadToolError` with infrastructure codes `recalc_unavailable`, `recalc_failed`, or `workbook_open_failed`; formula cells with Excel errors are returned in the payload rather than raised.

- [ ] **Step 1: Write failing helper tests**

Add tests that monkeypatch `online_judge_eval.recalc_with_libreoffice` and assert: correct formulas return cached values; the source file bytes are unchanged; all supported Excel errors are counted; whole-column ranges use sheet bounds; more than 40 cells truncates only the preview, not error scanning; quoted sheet names and multiple answer segments resolve correctly.

- [ ] **Step 2: Run helper tests and verify failure**

Run: `pytest -q rl/tests/test_spreadsheetbench_bridge_execution.py -k 'recalculated_answer_ranges'`

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement `validate_recalculated_answer_ranges(...)`**

Reuse `_FORMULA_ERRORS`, `_json_value`, `_render_payload`, `_workbook_lock`, and the official answer-position parsing helpers. Copy under the workbook lock, recalculate only the copy, scan cached and formula workbooks sequentially, and bound serialized preview records without bounding validation coverage.

- [ ] **Step 4: Run helper and existing explicit-recalc tests**

Run: `pytest -q rl/tests/test_spreadsheetbench_bridge_execution.py -k 'recalc or recalculated_answer_ranges'`

Expected: PASS, including existing `recalculate_and_read` limits and temporary-copy behavior.

- [ ] **Step 5: Commit Task 2**

```bash
git add envharness/bridges/spreadsheetbench/recalc_tools.py rl/tests/test_spreadsheetbench_bridge_execution.py
git commit -m "feat: validate recalculated spreadsheet answer ranges"
```

### Task 3: Recalc-aware `validate_workbook` with revision cache

**Files:**
- Modify: `envharness/bridges/spreadsheetbench/bridge.py`
- Test: `rl/tests/test_spreadsheetbench_bridge_execution.py`

**Interfaces:**
- Consumes: `validate_recalculated_answer_ranges(...)` from Task 2 and `_revision_state_info()` from Task 1.
- Produces: reset option `validate_with_recalc: bool`, default `False`, with environment fallback `SPREADSHEETBENCH_VALIDATE_WITH_RECALC`.
- Produces validation info: `validation_recalc_attempted`, `validation_recalc_cache_hit`, `validation_recalc_elapsed_ms`, `validation_recalc_formula_error_count`, and `validation_recalc_infrastructure_error`.
- Cache key: current `workbook_revision`; cache value: bounded validation observation plus structured recalc metrics.

- [ ] **Step 1: Write failing integration tests**

Add tests for: static-first rejection without recalculation; successful recalculation marking both revisions current; formula-error cells leaving both revisions stale; infrastructure failure using `recalc_validation_error` and not formula error; unchanged second validation cache hit; mutation cache invalidation; automatic validation not consuming `_recalc_calls`; static-only default preserving current behavior.

- [ ] **Step 2: Run integration tests and verify failure**

Run: `pytest -q rl/tests/test_spreadsheetbench_bridge_execution.py -k 'validate_with_recalc or validation_recalc'`

Expected: FAIL because the option and metrics are absent.

- [ ] **Step 3: Implement the two-stage validation and revision-keyed cache**

Call `_validate_output()` first. Invoke Task 2 only when enabled and static validation succeeds. On clean recalc, update both validation revisions; on any failure update neither. Clear or bypass the cache whenever `workbook_revision` changes.

- [ ] **Step 4: Run submit-gate, validation, and recalc regression tests**

Run: `pytest -q rl/tests/test_spreadsheetbench_bridge_execution.py -k 'validate or submit_gate or recalc'`

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add envharness/bridges/spreadsheetbench/bridge.py rl/tests/test_spreadsheetbench_bridge_execution.py
git commit -m "feat: recalculate formulas during workbook validation"
```

### Task 4: Persistent bounded edit ledger

**Files:**
- Modify: `rl/envharness_rl/spreadsheetbench/manager.py`
- Test: `rl/tests/test_spreadsheetbench_verl_manager.py`

**Interfaces:**
- Consumes: bridge revision fields from Task 1 and existing `tool_results` execution records.
- Produces: manager members `_edit_ledgers: list[list[dict[str, Any]]]` and `_workbook_states: list[dict[str, Any]]` initialized by `reset()`.
- Produces: `_ledger_entry(action: Action, result: dict[str, Any], episode_step: int) -> dict[str, Any]`, `_update_edit_ledger(index: int, actions: list[Action], info: dict[str, Any]) -> None`, and `_render_workbook_state(index: int) -> str`.
- Fixed bounds: 12 entries, 4,000 rendered characters, 500 characters for formula/query string fields; `run_python` records only `code_chars`.

- [ ] **Step 1: Write failing ledger tests**

Add tests asserting: ledger appears with `history_length=0`; native write records sheet/range/formula; Python source sentinel is absent while `code_chars` remains; failed calls include error code; short-circuited projected calls are absent; 13 calls evict the oldest whole record; rendered state stays at or below 4,000 characters; validation/recalc current flags update from bridge info.

- [ ] **Step 2: Run ledger tests and verify failure**

Run: `pytest -q rl/tests/test_spreadsheetbench_verl_manager.py -k 'ledger or persistent_workbook_state'`

Expected: FAIL because no persistent state section exists.

- [ ] **Step 3: Implement ledger normalization, state updates, and rendering**

Match actions to `tool_results` by executed `action_index`; never infer success from projection alone. Insert `Persistent workbook state` between tool instructions and recent interaction history in `build_text_obs()`.

- [ ] **Step 4: Run manager tests**

Run: `pytest -q rl/tests/test_spreadsheetbench_verl_manager.py`

Expected: PASS, including compact-history and raw trajectory tests.

- [ ] **Step 5: Commit Task 4**

```bash
git add rl/envharness_rl/spreadsheetbench/manager.py rl/tests/test_spreadsheetbench_verl_manager.py
git commit -m "feat: add persistent spreadsheet edit ledger"
```

### Task 5: Safe parser recovery and explicit tool schemas

**Files:**
- Modify: `rl/envharness_rl/spreadsheetbench/projection.py`
- Test: `rl/tests/test_spreadsheetbench_projection.py`

**Interfaces:**
- Produces: module constants `TOOL_ARGUMENT_SCHEMAS`, `TOOL_CALL_EXAMPLES`, and `DEFAULT_INVALID_OUTPUT_CHAR_BUDGET = 16_000`.
- Produces: diagnostics `missing_end_recovered`, `schema_error`, and `invalid_output_over_budget` as integer fields on every projected call.
- Preserves: existing status names for valid native/fenced/bare calls and existing four-call ordering/truncation rules.

- [ ] **Step 1: Write failing parser tests**

Add parameterized tests for: unambiguous missing-end recovery; rejection with trailing prose, partial JSON, or second object; unknown tool feedback listing only enabled tools; missing, extra, wrong-type, and zero-argument schema errors; stable corrective example; oversized invalid classification without malformed payload echo; all current native/fenced/bare/multi-call cases unchanged.

- [ ] **Step 2: Run parser tests and verify failure**

Run: `pytest -q rl/tests/test_spreadsheetbench_projection.py -k 'missing_end or schema or over_budget'`

Expected: FAIL because new recovery and diagnostics are absent.

- [ ] **Step 3: Implement schema tables and unambiguous suffix decoding**

Use `json.JSONDecoder.raw_decode` and require a whitespace-only remainder. Validate allowed/required keys and top-level types before tool-specific semantic checks. Keep bridge/runtime validation authoritative.

- [ ] **Step 4: Run the full projection suite**

Run: `pytest -q rl/tests/test_spreadsheetbench_projection.py`

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add rl/envharness_rl/spreadsheetbench/projection.py rl/tests/test_spreadsheetbench_projection.py
git commit -m "feat: harden spreadsheet tool call projection"
```

### Task 6: Consecutive invalid-action loop guard

**Files:**
- Modify: `rl/envharness_rl/spreadsheetbench/envs.py`
- Modify: `rl/envharness_rl/spreadsheetbench/manager.py`
- Test: `rl/tests/test_spreadsheetbench_envs.py`
- Test: `rl/tests/test_spreadsheetbench_verl_manager.py`

**Interfaces:**
- Produces: worker constructor parameter `max_consecutive_invalid_actions: int = 3`, validated as positive.
- Produces: worker state `_consecutive_invalid_actions` and `_max_consecutive_invalid_actions_seen`, reset per episode.
- Produces terminal info: `invalid_streak_terminated`, `consecutive_invalid_actions`, and `consecutive_invalid_actions_max`.
- Manager maps these to `episode/invalid_streak_terminated`, `episode/consecutive_invalid_actions`, and `episode/consecutive_invalid_actions_max`.

- [ ] **Step 1: Write failing worker tests**

Add tests that three consecutive `Action(name="invalid")` turns terminate and grade; two invalid turns do not terminate; a valid projected action resets the streak; a valid tool with runtime failure does not increment it; a multi-call batch containing an invalid projected action increments once for the turn.

- [ ] **Step 2: Run worker guard tests and verify failure**

Run: `pytest -q rl/tests/test_spreadsheetbench_envs.py -k 'invalid_streak or consecutive_invalid'`

Expected: FAIL because worker streak state is absent.

- [ ] **Step 3: Implement streak tracking and terminal grading in the worker**

Determine parser invalidity only from projected action name `invalid`, before runtime execution. At threshold, finish the current invalid response, mark the episode done, and call `_grade()` once using the unchanged workbook.

- [ ] **Step 4: Add manager feedback and metric tests**

Assert the next observation gives a concise streak count before termination, terminal trajectory contains the new fields, and `_step_diagnostics()` exports the three episode metrics.

- [ ] **Step 5: Run worker and manager suites**

Run: `pytest -q rl/tests/test_spreadsheetbench_envs.py rl/tests/test_spreadsheetbench_verl_manager.py`

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

```bash
git add rl/envharness_rl/spreadsheetbench/envs.py rl/envharness_rl/spreadsheetbench/manager.py rl/tests/test_spreadsheetbench_envs.py rl/tests/test_spreadsheetbench_verl_manager.py
git commit -m "feat: terminate repeated invalid spreadsheet actions"
```

### Task 7: Configuration, manifests, and aggregate metrics

**Files:**
- Modify: `rl/scripts/run_spreadsheetbench_grpo.sh`
- Modify: `rl/scripts/submit_spreadsheetbench_grpo.sh`
- Modify: `rl/scripts/submit_spreadsheetbench_eval.sh`
- Modify: `rl/envharness_rl/spreadsheetbench/envs.py`
- Modify: `rl/envharness_rl/spreadsheetbench/manager.py`
- Modify: `rl/envharness_rl/spreadsheetbench/rollout_summary.py`
- Modify if required: `rl/integration/verl_agent_spreadsheetbench_tool_metrics.patch`
- Test: `rl/tests/test_spreadsheetbench_training_scripts.py`
- Test: `rl/tests/test_spreadsheetbench_verl_manager.py`
- Test: `rl/tests/test_spreadsheetbench_rollout_summary.py`

**Interfaces:**
- Produces environment settings `SPREADSHEETBENCH_VALIDATE_WITH_RECALC=true`, `SPREADSHEETBENCH_MAX_CONSECUTIVE_INVALID_ACTIONS=3`, and `SPREADSHEETBENCH_INVALID_OUTPUT_CHAR_BUDGET=16000` in training/evaluation target flows.
- Produces all metric names listed in the spec without renaming existing metrics.

- [ ] **Step 1: Write failing configuration and aggregation tests**

Assert run/submit/eval scripts validate, export, log, and record the three settings in manifests. Assert manager validation aggregation and rollout summaries include recalc, ledger, parser recovery/schema/over-budget, and invalid-streak metrics with explicit denominators.

- [ ] **Step 2: Run focused integration tests and verify failure**

Run: `pytest -q rl/tests/test_spreadsheetbench_training_scripts.py rl/tests/test_spreadsheetbench_rollout_summary.py -k 'validation_recalc or invalid_streak or tool_schema or edit_ledger'`

Expected: FAIL because settings and aggregate metrics are absent.

- [ ] **Step 3: Wire settings from scripts through reset options and manifests**

Parse booleans and positive integers at script boundaries. Pass the invalid threshold into workers and the validation flag into bridge reset options. Keep direct bridge defaults unchanged.

- [ ] **Step 4: Add manager and rollout-summary aggregation**

Reuse existing scalar metric mappings. Update the vendored patch only if trainer propagation is otherwise missing, and preserve its idempotent application tests.

- [ ] **Step 5: Run integration suites**

Run: `pytest -q rl/tests/test_spreadsheetbench_training_scripts.py rl/tests/test_spreadsheetbench_rollout_summary.py rl/tests/test_spreadsheetbench_verl_manager.py`

Expected: PASS.

- [ ] **Step 6: Commit Task 7**

```bash
git add rl/scripts/run_spreadsheetbench_grpo.sh rl/scripts/submit_spreadsheetbench_grpo.sh rl/scripts/submit_spreadsheetbench_eval.sh rl/envharness_rl/spreadsheetbench/envs.py rl/envharness_rl/spreadsheetbench/manager.py rl/envharness_rl/spreadsheetbench/rollout_summary.py rl/tests/test_spreadsheetbench_training_scripts.py rl/tests/test_spreadsheetbench_verl_manager.py rl/tests/test_spreadsheetbench_rollout_summary.py rl/integration/verl_agent_spreadsheetbench_tool_metrics.patch
git commit -m "feat: configure spreadsheet harness recovery controls"
```

### Task 8: Full regression and smoke verification

**Files:**
- Modify only if failures reveal a defect in Task 1-7 files.

**Interfaces:**
- Consumes: all interfaces from Tasks 1-7.
- Produces: verified harness ready for frozen-checkpoint evaluation; this task does not launch a long GRPO run.

- [ ] **Step 1: Run focused harness suites**

Run: `pytest -q rl/tests/test_spreadsheetbench_projection.py rl/tests/test_spreadsheetbench_envs.py rl/tests/test_spreadsheetbench_verl_manager.py rl/tests/test_spreadsheetbench_bridge_execution.py rl/tests/test_spreadsheetbench_rollout_summary.py rl/tests/test_spreadsheetbench_training_scripts.py`

Expected: PASS.

- [ ] **Step 2: Run remaining RL regression suite**

Run: `pytest -q rl/tests`

Expected: PASS.

- [ ] **Step 3: Run shell syntax and dry-run checks**

Run: `bash -n rl/scripts/run_spreadsheetbench_grpo.sh rl/scripts/submit_spreadsheetbench_grpo.sh rl/scripts/submit_spreadsheetbench_eval.sh`

Run: the existing native-basic diagnostic dry-run test command documented in `rl/tests/test_spreadsheetbench_training_scripts.py`.

Expected: syntax succeeds and dry-run manifests contain the three new settings.

- [ ] **Step 4: Run fixed-task worker smoke test**

Run: `python rl/scripts/smoke_spreadsheetbench_worker.py` with the repository's documented local test dataset and `SPREADSHEETBENCH_VALIDATE_WITH_RECALC=true`.

Expected: ledger state is present, validation recalculates a temporary copy, submit accepts the current revision, and no evaluator error occurs.

- [ ] **Step 5: Inspect final diff and repository status**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; only task files and pre-existing user changes are present.

- [ ] **Step 6: Commit any verification-only corrections separately**

```bash
git add <only corrected task files>
git commit -m "fix: close spreadsheet harness recovery regressions"
```

Skip this commit when no correction was needed.
