# Spreadsheet Harness Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add transactional Python execution, current-revision validation, formula hardening, and safe four-call truncation to the NJ5 SpreadsheetBench harness while preserving its existing 50,000/100,000-cell protections and console-only source redaction.

**Architecture:** The bridge owns workbook transactions, formula validation, revision state, and submit admission. Projection owns the fixed four-call prefix, the manager turns parser/environment results into model feedback and metrics, and the existing worker remains the final per-turn resource-budget enforcement point. Training launchers and a focused verl-agent patch enable the gate and carry the new diagnostics without importing SG1's category-specific reward policy.

**Tech Stack:** Python 3.10+, openpyxl, LibreOffice/soffice, pytest, Bash, Ray/verl-agent integration patches.

**Spec:** `docs/superpowers/specs/2026-09-30-spreadsheet-harness-hardening-design.md`

## Global Constraints

- Preserve every pre-existing uncommitted NJ5 change; never replace a dirty file wholesale with the SG1 version.
- Keep `MAX_WRITE_CELLS`, `MAX_FILL_CELLS`, and `MAX_FORMAT_CELLS` at exactly `50_000`.
- Keep `MAX_MULTI_CALL_MUTATION_CELLS` at exactly `100_000` and count admitted mutation attempts, including failed attempts.
- Admit at most four tool calls per model turn; this is a fixed hard cap with no configuration path above four.
- Redact Python source only from console action logs; retain original model output and projected code in trajectory JSON and prompt history.
- Preserve the current reward policy; do not add SG1 read/write/recalc/validation/submit-gate penalty defaults.
- Limit formula validation to task answer ranges and never consult the golden workbook.
- Do not stage or commit overlapping pre-existing dirty implementation files during execution. Use verified diff checkpoints; request explicit user approval before a final code commit.

## Review Focus

- A child process that writes the committed filename through `working_directory` and then fails must still be restored from the recovery backup; Task 1 tests this direct-path corruption case.
- A transaction that exits zero after deleting or corrupting its transaction file must fail without replacing the committed workbook; Task 1 tests both cases.
- Formula validation must not reject legitimate formulas containing quoted text, escaped quotes, or braces; Task 2 adds positive regression cases alongside malformed cases.
- A `submit` or `recalculate_and_read` in the retained four-call prefix must remain invalid when an original fifth call followed it; Task 3 pins original-sequence finality.
- Truncation, validation rejection, or mutation-budget rejection must leave later calls unexecuted while reporting accurate executed/skipped counts; Tasks 3 and 4 add integration assertions.

---

### Task 1: Transactional `run_python`

**Files:**
- Modify: `envharness/bridges/spreadsheetbench/bridge.py:543-596,743-796`
- Test: `rl/tests/test_spreadsheetbench_bridge_execution.py:51-129`

**Interfaces:**
- Consumes: existing `_python_bootstrap()`, `run_process_group()`, `state.output_path`, and `_python_error_type()`.
- Produces: `_run_python(code: str) -> tuple[str, int, str, bool]` where the final value is `transaction_committed`; response info keys `transaction_committed` and `transaction_rolled_back`.

- [ ] **Step 1: Add failing transaction tests**

Add tests named:

- `test_run_python_rolls_back_workbook_after_runtime_failure`
- `test_run_python_restores_committed_path_after_direct_path_corruption`
- `test_run_python_rolls_back_workbook_after_timeout`
- `test_run_python_rejects_corrupt_workbook_before_commit`
- `test_run_python_rejects_missing_transaction_before_commit`
- `test_run_python_commits_valid_workbook_and_reports_transaction`

Each failure test saves `Path(env.state.output_path).read_bytes()` before the call and asserts exact byte equality afterward, `transaction_committed is False`, and `transaction_rolled_back is True`. The success test asserts the intended cell value persisted and the flags are the inverse.

- [ ] **Step 2: Run the new tests and verify the current direct-write implementation fails**

Run:

```bash
PYTHONPATH=.:rl python3 -m pytest -q \
  rl/tests/test_spreadsheetbench_bridge_execution.py \
  -k 'transaction or direct_path_corruption or corrupt_workbook or missing_transaction'
```

Expected: the new rollback/validation assertions fail before implementation.

- [ ] **Step 3: Implement transaction creation, commit, restoration, and cleanup**

In `bridge.py`, change `_run_python` to the produced signature. Create the transaction in `self._workdir`, keep a recovery backup outside the exposed transaction path, pass the transaction as the bootstrap `output_path`, validate it with `openpyxl`, and commit with `os.replace`. Restore the backup on child failure, timeout, exception, direct committed-path corruption, missing transaction, or invalid xlsx. Preserve existing output truncation and error-type classification; classify invalid xlsx as `WorkbookValidationError`.

- [ ] **Step 4: Propagate transaction fields through `step()`**

Unpack the fourth return value, increment `workbook_revision` only on a committed success, and add `transaction_committed`, `transaction_rolled_back`, and a stable `failure_class` for Python failures to response info.

- [ ] **Step 5: Run bridge transaction tests**

Run the Step 2 command.

Expected: all selected tests pass.

- [ ] **Step 6: Checkpoint the Task 1 diff**

```bash
git diff --check -- envharness/bridges/spreadsheetbench/bridge.py rl/tests/test_spreadsheetbench_bridge_execution.py
```

Expected: exit 0; do not stage the overlapping dirty test file.

### Task 2: Formula validation and current-revision submit gate

**Files:**
- Modify: `envharness/bridges/spreadsheetbench/write_tools.py:142-215,400-405`
- Modify: `envharness/bridges/spreadsheetbench/bridge.py:84-92,203-225,252-402,439-507,798-877`
- Test: `rl/tests/test_spreadsheetbench_bridge_execution.py:130-161,639-663`

**Interfaces:**
- Consumes: Task 1 `transaction_committed`; existing `workbook_revision` and `last_recalc_revision`; `execute_recalculate_and_read()` payload keys `formula_error_cells` and `ranges`.
- Produces: `validate_formula_syntax(formula: str) -> None`; `_validate_output() -> tuple[str, bool, dict[str, int]]`; reset option `require_validation_before_submit`; response keys `formula_validation_errors`, `submit_rejected`, `validation_stale`, and `validation_revision`.

- [ ] **Step 1: Add failing formula-validation tests**

Add tests named:

- `test_validate_workbook_rejects_malformed_formula`
- `test_validate_workbook_rejects_broken_formula_reference`
- `test_validate_workbook_rejects_formula_text_wrapper`
- `test_validate_workbook_accepts_valid_formula_with_quoted_delimiters`
- `test_validate_workbook_ignores_formula_errors_outside_answer_ranges`

Assert invalid formulas return `validation_ok=False`, a positive `formula_validation_errors`, and a cell-specific message; valid or out-of-scope formulas must not be rejected.

- [ ] **Step 2: Add failing submit-gate and recalc tests**

Add tests named:

- `test_submit_gate_requires_current_validation`
- `test_submit_gate_rejects_stale_validation_after_native_edit`
- `test_submit_gate_rejects_stale_validation_after_python_commit`
- `test_failed_python_does_not_stale_current_validation`
- `test_clean_recalc_satisfies_submit_gate`
- `test_recalc_with_formula_errors_does_not_satisfy_submit_gate`

Assert rejection is non-terminal, does not mark submitted, and reports whether validation is missing or stale. Assert only successful mutations increment the revision.

- [ ] **Step 3: Run the new formula/gate tests and verify failure**

```bash
PYTHONPATH=.:rl python3 -m pytest -q \
  rl/tests/test_spreadsheetbench_bridge_execution.py \
  -k 'formula or submit_gate or stale_validation or clean_recalc'
```

Expected: new validation and gate assertions fail.

- [ ] **Step 4: Expose and reuse the formula validator**

Rename `_validate_formula_syntax` to `validate_formula_syntax`, update `fill_formula`, import it into `bridge.py`, and extend `_validate_output` to return metrics while checking answer-range formulas, `#REF!`, and formula-text wrappers. Report at most 20 detailed formula errors while retaining the total count.

- [ ] **Step 5: Implement revision-aware validation and submit**

Initialize `last_validation_revision=-1`, parse `require_validation_before_submit` consistently for booleans and common boolean strings, record successful validate/recalc revisions, and reject missing/stale submit without terminating. A recalc updates its revision only when both runtime and static formula checks are clean.

- [ ] **Step 6: Update bridge instructions and response metadata**

Document that validation is revision-specific, that edits require revalidation, and that `run_python` is transactional. Preserve the existing execution reward values.

- [ ] **Step 7: Run Task 2 tests and bridge regression tests**

```bash
PYTHONPATH=.:rl python3 -m pytest -q rl/tests/test_spreadsheetbench_bridge_execution.py
```

Expected: all bridge execution tests pass.

- [ ] **Step 8: Checkpoint the Task 2 diff**

```bash
git diff --check -- envharness/bridges/spreadsheetbench/write_tools.py envharness/bridges/spreadsheetbench/bridge.py rl/tests/test_spreadsheetbench_bridge_execution.py
```

Expected: exit 0; leave implementation changes unstaged.

### Task 3: Fixed four-call prefix truncation

**Files:**
- Modify: `rl/envharness_rl/spreadsheetbench/projection.py:319-397`
- Modify: `rl/envharness_rl/spreadsheetbench/manager.py:547-612`
- Test: `rl/tests/test_spreadsheetbench_projection.py:64-143`
- Test: `rl/tests/test_spreadsheetbench_verl_manager.py:85-140`

**Interfaces:**
- Consumes: existing `_project_one_with_diagnostics()` and manager batch execution.
- Produces: fixed `MAX_TOOL_CALLS_PER_TURN = 4`; diagnostic keys `batch_truncated`, `total_call_count`, and `truncated_call_count`; a next-observation truncation message.

- [ ] **Step 1: Replace the old rejection test with failing truncation tests**

Add tests named:

- `test_project_action_batch_safely_truncates_more_than_four_calls`
- `test_project_action_batch_does_not_execute_truncated_submit`
- `test_project_action_batch_rejects_retained_submit_before_truncated_calls`
- `test_project_action_batch_rejects_retained_recalc_before_truncated_calls`
- `test_project_action_batch_never_admits_more_than_four_calls`

Assert five valid calls return exactly four actions and complete truncation diagnostics. Assert original-sequence finality for submit/recalc.

- [ ] **Step 2: Add a failing manager feedback test**

Add `test_manager_reports_safely_truncated_tool_batch`, asserting only four actions reach vector envs, `parser/tool_calls_truncated == 1`, and the anchor observation states that four of five calls were accepted.

- [ ] **Step 3: Run projection and manager tests and verify failure**

```bash
PYTHONPATH=.:rl python3 -m pytest -q \
  rl/tests/test_spreadsheetbench_projection.py \
  rl/tests/test_spreadsheetbench_verl_manager.py \
  -k 'truncat or more_than_four or retained_submit or retained_recalc'
```

Expected: old whole-batch rejection behavior fails the new assertions.

- [ ] **Step 4: Implement fixed prefix truncation**

Extract all blocks, compute original counts, slice to the first four, attach truncation diagnostics to admitted actions, and evaluate submit/recalc finality against the original call count. Keep four as a non-configurable hard cap.

- [ ] **Step 5: Aggregate and report truncation in the manager**

Carry batch counts into aggregate diagnostics, expose `parser/tool_calls_truncated`, and prepend a recovery instruction to the observation without marking an otherwise valid retained prefix invalid.

- [ ] **Step 6: Run Task 3 tests**

Run the Step 3 command without `-k`.

Expected: both test files pass.

- [ ] **Step 7: Checkpoint the Task 3 diff**

```bash
git diff --check -- rl/envharness_rl/spreadsheetbench/projection.py rl/envharness_rl/spreadsheetbench/manager.py rl/tests/test_spreadsheetbench_projection.py rl/tests/test_spreadsheetbench_verl_manager.py
```

Expected: exit 0; leave implementation changes unstaged.

### Task 4: Worker recovery behavior, metrics, and logging contract

**Files:**
- Modify: `rl/envharness_rl/spreadsheetbench/envs.py:320-463`
- Modify: `rl/envharness_rl/spreadsheetbench/manager.py:27-100,210-450,613-648`
- Test: `rl/tests/test_spreadsheetbench_envs.py:110-166,365-515`
- Test: `rl/tests/test_spreadsheetbench_verl_manager.py`

**Interfaces:**
- Consumes: Tasks 1-3 response and parser keys; existing mutation-budget helpers.
- Produces: manager metrics `env/python_transaction_rollback`, `env/submit_gate_reject`, `env/formula_validation_error_count`, and `parser/tool_calls_truncated`; explicit stop behavior for failed validate/submit actions.

- [ ] **Step 1: Add failing worker and manager metric tests**

Add tests asserting:

- a failed `validate_workbook` or rejected `submit` short-circuits remaining calls;
- transaction rollback, submit rejection, formula validation count, and truncation count reach `_step_diagnostics`;
- a truncation followed by mutation-budget rejection reports correct projected, executed, skipped, and failure counts;
- the existing console sentinel source remains absent;
- an in-memory trajectory step retains the sentinel in `model_output` and projected `code`.

- [ ] **Step 2: Run selected env/manager tests and verify failure**

```bash
PYTHONPATH=.:rl python3 -m pytest -q \
  rl/tests/test_spreadsheetbench_envs.py \
  rl/tests/test_spreadsheetbench_verl_manager.py \
  -k 'rollback or submit_reject or formula_validation or truncat or log_run_python or trajectory'
```

Expected: missing metrics and validate/submit short-circuit assertions fail; existing console redaction test continues to pass.

- [ ] **Step 3: Extend worker failure classification without changing rewards**

Treat failed `validate_workbook` and rejected `submit` as short-circuiting actions. Preserve the existing 100,000 attempted-cell counter and zero mutation-budget reward.

- [ ] **Step 4: Extend manager instructions and diagnostics**

Tell the model that Python edits are transactional and validation is revision-specific. Aggregate the new response fields into the produced metric names and preserve raw trajectory/model history content under Scheme A.

- [ ] **Step 5: Run Task 4 tests**

Run the Step 2 command without `-k`.

Expected: both test files pass.

- [ ] **Step 6: Checkpoint the Task 4 diff**

```bash
git diff --check -- rl/envharness_rl/spreadsheetbench/envs.py rl/envharness_rl/spreadsheetbench/manager.py rl/tests/test_spreadsheetbench_envs.py rl/tests/test_spreadsheetbench_verl_manager.py
```

Expected: exit 0; leave implementation changes unstaged.

### Task 5: Training launchers and verl-agent integration

**Files:**
- Modify: `rl/scripts/run_spreadsheetbench_grpo.sh`
- Modify: `rl/scripts/submit_spreadsheetbench_grpo.sh`
- Modify: `rl/scripts/fetch_verl_agent.sh`
- Create: `rl/integration/verl_agent_spreadsheetbench_hardening.patch`
- Modify: `rl/integration/README.md`
- Modify: `rl/README.md`
- Test: `rl/tests/test_spreadsheetbench_training_scripts.py`

**Interfaces:**
- Consumes: Task 4 metric names and bridge reset option `require_validation_before_submit`.
- Produces: `SPREADSHEETBENCH_REQUIRE_VALIDATION_BEFORE_SUBMIT=true` in target train/eval paths; idempotent external reset-option and metric propagation.

- [ ] **Step 1: Add failing launcher and patch-content tests**

Add `test_training_scripts_forward_hardening_controls` and patch/fetch assertions requiring:

- both launchers default, validate, export, forward, and print `SPREADSHEETBENCH_REQUIRE_VALIDATION_BEFORE_SUBMIT`;
- the hardening patch maps the four new metrics and reset option;
- `fetch_verl_agent.sh` applies the new patch and recognizes an already-patched checkout;
- no category-specific error-penalty variables are introduced by this task.

- [ ] **Step 2: Run training-script tests and verify failure**

```bash
PYTHONPATH=.:rl python3 -m pytest -q \
  rl/tests/test_spreadsheetbench_training_scripts.py \
  -k 'hardening or validation_before_submit or patch'
```

Expected: missing launcher variables and patch references fail.

- [ ] **Step 3: Update launchers and documentation**

Add strict boolean validation, export/forward the setting, include it in startup diagnostics, and document that target training/eval defaults to a required current validation.

- [ ] **Step 4: Add and wire the focused integration patch**

Patch the pinned external env manager to pass the reset option and patch rollout/trainer metric mappings for the four new diagnostics. Update fetch ordering and idempotency checks without modifying unrelated existing patches.

- [ ] **Step 5: Validate patch application and shell syntax**

```bash
bash -n rl/scripts/run_spreadsheetbench_grpo.sh
bash -n rl/scripts/submit_spreadsheetbench_grpo.sh
bash -n rl/scripts/fetch_verl_agent.sh
```

Expected: all commands exit 0.

Apply the full integration patch sequence to a fresh copy of the pinned verl-agent checkout using the repository's fetch/apply test fixture; expected result is every patch applies once and a second pass reports already applied.

- [ ] **Step 6: Run Task 5 tests**

```bash
PYTHONPATH=.:rl python3 -m pytest -q rl/tests/test_spreadsheetbench_training_scripts.py
```

Expected: all training-script tests pass.

- [ ] **Step 7: Checkpoint the Task 5 diff**

```bash
git diff --check -- rl/scripts/run_spreadsheetbench_grpo.sh rl/scripts/submit_spreadsheetbench_grpo.sh rl/scripts/fetch_verl_agent.sh rl/integration/verl_agent_spreadsheetbench_hardening.patch rl/integration/README.md rl/README.md rl/tests/test_spreadsheetbench_training_scripts.py
```

Expected: exit 0; leave implementation changes unstaged.

### Task 6: Combined regression and acceptance checks

**Files:**
- Modify only if a regression exposes a requirement violation; do not perform opportunistic refactors.
- Test: all SpreadsheetBench bridge/RL tests and repository diff hygiene.

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: a verified combined NJ5 hardening change set ready for a fixed-task smoke and diagnostic training run.

- [ ] **Step 1: Run focused combined tests**

```bash
PYTHONPATH=.:rl python3 -m pytest -q \
  rl/tests/test_spreadsheetbench_bridge_execution.py \
  rl/tests/test_spreadsheetbench_envs.py \
  rl/tests/test_spreadsheetbench_projection.py \
  rl/tests/test_spreadsheetbench_verl_manager.py \
  rl/tests/test_spreadsheetbench_training_scripts.py
```

Expected: all tests pass.

- [ ] **Step 2: Run the complete RL test suite**

```bash
PYTHONPATH=.:rl python3 -m pytest -q rl/tests
```

Expected: all tests pass with no new skips or warnings attributable to the change.

- [ ] **Step 3: Run static checks**

```bash
git diff --check
bash -n rl/scripts/run_spreadsheetbench_grpo.sh
bash -n rl/scripts/submit_spreadsheetbench_grpo.sh
bash -n rl/scripts/fetch_verl_agent.sh
```

Expected: every command exits 0.

- [ ] **Step 4: Run the fixed-task worker smoke test**

Use the repository's existing SpreadsheetBench worker smoke command with the same data path and pinned dependency environment used by NJ5 training. Expected: one transactional Python success, one recoverable rollback case, a current validation, and a successful submit with no evaluator error.

- [ ] **Step 5: Review the final diff against all eleven requirements**

Confirm the final diff preserves the original seven dirty NJ5 files' intent, contains no category-specific reward additions, retains raw trajectory code, and has no tool-call configuration path above four.

- [ ] **Step 6: Report final unstaged files and request commit approval**

Run `git status --short` and list the implementation files separately from the
two committed planning documents. Do not stage or commit code until the user
reviews the final diff and explicitly authorizes the commit.
