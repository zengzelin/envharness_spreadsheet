# Spreadsheet Harness Hardening Design

**Date:** 2026-09-30

## Goal

Harden the NJ5 SpreadsheetBench training and evaluation path so failed Python
actions cannot corrupt the workbook, formula defects are caught before submit,
every submitted workbook has a current validation, and native multi-call work is
bounded by fixed per-call, per-turn, and call-count limits.

## Existing State

The NJ5 worktree already contains uncommitted implementations for:

- a 50,000-cell limit on `write_range`, `clear_range`, `fill_formula`, and
  `format_range`;
- a 100,000-cell cumulative mutation-attempt budget per model turn;
- bounded worker action logging that replaces `run_python.code` with
  `code_chars`;
- tests and a training metric for the cumulative mutation budget.

These changes must be preserved. The SG1 worktree is a behavioral reference,
not a source to copy wholesale, because its shared files would overwrite the
NJ5 changes and it also contains reward-shaping changes outside this scope.

## Requirements

1. A failed, timed-out, or invalid `run_python` call leaves the committed
   workbook byte-for-byte unchanged.
2. A `run_python` call may commit only an xlsx that `openpyxl` can reopen.
3. `validate_workbook` rejects malformed formulas, `#REF!` references, and
   formulas represented as formula text in the task answer ranges.
4. Submit is accepted only when the current workbook revision has a successful
   `validate_workbook` or clean `recalculate_and_read` result.
5. Any successful workbook mutation makes earlier validation stale.
6. Cell-addressed mutation tools have a fixed 50,000-cell per-call limit.
7. A model turn has a fixed 100,000-cell cumulative mutation-attempt budget.
8. Worker console logs never print full `run_python` source. Rollout trajectory
   JSON intentionally retains the full model output and projected action for
   training diagnosis.
9. At most four tool calls from one model turn are admitted. Calls after the
   fourth are not executed, and the next observation reports the truncation.
10. The existing execution reward policy is preserved. This change does not
    add SG1's category-specific read, write, recalc, validation, or submit-gate
    penalties.

## Non-goals

- Redacting Python source from rollout trajectory JSON.
- Raising the per-turn tool-call limit above four.
- Adding new SpreadsheetBench business tools.
- Changing the official final workbook comparison or its score definition.
- Importing unrelated SG1 documentation or reward-shaping defaults.
- Applying a cell-count estimate to row deletion, column deletion, or sheet
  management. Those structure tools remain bounded by their existing argument
  and worksheet-boundary validation.

## Design

### Transactional `run_python`

Each call creates a transaction copy of the current output workbook in the
sandbox and a recovery backup outside the path exposed as `output_path`. The
Python bootstrap receives the transaction path, so normal helpers read and
write the transaction rather than the committed workbook.

The transaction commits with `os.replace()` only when the child exits with code
zero and `openpyxl.load_workbook(..., read_only=True, data_only=False)` can open
the resulting file. Syntax errors, runtime errors, timeouts, process-launch
errors, missing transaction files, and invalid xlsx files restore the original
from the backup and report `transaction_committed=False` and
`transaction_rolled_back=True`. All script, transaction, and backup files are
cleaned up on every path.

`_run_python` returns `(output, returncode, error_type,
transaction_committed)`. A successful commit increments `workbook_revision`;
a rollback does not.

### Formula validation

The existing formula validator in `write_tools.py` becomes the public internal
helper `validate_formula_syntax(formula: str) -> None`, and existing callers are
updated. `validate_workbook` reuses it while scanning formulas in the task's
answer ranges. The scan also rejects:

- formulas containing `#REF!`, case-insensitively;
- formula-text wrappers such as `="=..."` and `='=...'`;
- any formula the existing structural/tokenizer checks reject.

Validation remains limited to answer ranges. Pre-existing defects in unrelated
input cells must not block submission. The validation response reports
`formula_validation_errors`, includes up to 20 concrete cell errors in the
observation, and never consults the golden workbook.

After LibreOffice recalculation, the bridge also performs this static answer-
range validation. A recalculation counts as clean only when LibreOffice reports
no formula error values and static validation succeeds.

### Workbook revisions and submit gate

Environment state tracks:

- `workbook_revision`, initially `0`;
- `last_validation_revision`, initially `-1`;
- `last_recalc_revision`, initially `-1`.

Successful native write/structure operations and committed `run_python` calls
increment `workbook_revision`. Successful `validate_workbook` records the
current revision in `last_validation_revision`. A clean recalculation records
it in `last_recalc_revision`. Failed operations do not change these fields.

When `require_validation_before_submit` is enabled, `submit` is accepted only
if either validation revision equals `workbook_revision`. Otherwise it returns
`submit_validation_required`, sets `submit_rejected=True`, explains whether the
previous validation is stale, and leaves the episode running so the model can
recover.

The bridge option remains disabled by default for direct external callers and
legacy unit fixtures. NJ5 GRPO/Ray training and evaluation launch paths set
`SPREADSHEETBENCH_REQUIRE_VALIDATION_BEFORE_SUBMIT=true` and pass the reset
option explicitly, making the gate mandatory in the target workflows.

### Mutation limits

The fixed limits are:

- `MAX_WRITE_CELLS = 50_000` for `write_range` and `clear_range`;
- `MAX_FILL_CELLS = 50_000` for `fill_formula`;
- `MAX_FORMAT_CELLS = 50_000` for `format_range`;
- `MAX_MULTI_CALL_MUTATION_CELLS = 100_000` per model turn.

The cumulative budget counts the declared cells of each admitted mutation
attempt before the next action is allowed. An attempted action continues to
consume budget even if the environment rejects it; this is intentionally a
resource-attempt budget rather than a successful-write counter. It covers
`write_range`, `clear_range`, `fill_formula`, and `format_range`.

### Four-call prefix truncation

`project_action_batch` extracts the complete ordered call list, records its
original length, and admits only the first four blocks. It adds
`batch_truncated`, `total_call_count`, and `truncated_call_count` to every
admitted diagnostic.

The final-position rule is evaluated against the original sequence, not only
the retained prefix. Therefore a `submit` or `recalculate_and_read` within the
first four calls is invalid if any original call followed it, including a
truncated call.

The manager executes the admitted prefix and prepends a message to the next
observation stating how many calls were accepted and how many were not
executed. The limit is a fixed hard cap of four; no environment variable may
raise it.

### Logging and trajectories

Worker console output uses the existing bounded action renderer. Native
arguments may be printed up to 1,000 characters; a `code` argument is removed
and replaced by its character count before serialization.

Trajectory JSON and compact/full interaction history retain the original model
output. This is deliberate because those artifacts are required to diagnose
generated Python and reproduce failed rollouts. They must be treated as
restricted training artifacts rather than ordinary console logs.

### Metrics and integration

The manager exposes and the verl-agent integration aggregates:

- `env/python_transaction_rollback`;
- `env/submit_gate_reject`;
- `env/formula_validation_error_count`;
- `parser/tool_calls_truncated`;
- the existing `env/multi_call_mutation_budget_exceeded`.

Training and submission scripts export the submit-gate setting. A focused,
idempotent verl-agent integration patch carries the reset option and metric
mappings, and `fetch_verl_agent.sh` applies and verifies that patch without
absorbing unrelated SG1 reward changes.

## Error and Recovery Semantics

- A transaction rollback is a normal non-terminal Python failure; the model
  receives the existing Python error penalty and can try again.
- A formula validation failure is non-terminal and does not update a validation
  revision.
- A rejected submit is non-terminal and does not set `state.submitted`.
- A formula-error recalculation is a failed recalc and does not update
  `last_recalc_revision`.
- A mutation-budget rejection stops the remaining actions in that model turn.
- A truncated batch is not itself an invalid action; valid admitted calls run
  until normal short-circuit rules stop them.

## Test Strategy

Tests must cover:

1. rollback after a script saves and then raises;
2. rollback after timeout and process-launch failure;
3. rejection of a corrupt xlsx before commit;
4. successful transaction commit and revision increment;
5. malformed formula, `#REF!`, and formula-text rejection;
6. current validation permits submit, while an intervening mutation makes it
   stale and rejects submit;
7. a clean recalc satisfies the gate and a formula-error recalc does not;
8. five emitted calls execute at most the first four and report one truncation;
9. truncated `submit` is not executed and non-final retained `submit`/recalc is
   invalid;
10. per-call 50,000-cell limits and the 100,000-cell attempted-mutation budget
    continue to pass together with four-call truncation;
11. console output omits a sentinel Python source string while trajectory data
    retains the original action;
12. new metrics reach manager info, training batch mappings, and validation
    aggregation.

The targeted bridge, worker, projection, manager, training-script, integration-
patch, and shell syntax tests run before the complete `rl/tests` suite.

## Rollout Acceptance

Before long training:

1. run the targeted and complete test suites;
2. run a fixed-task worker smoke test;
3. run a five-step diagnostic training job;
4. confirm evaluator errors remain zero;
5. confirm rollback, submit rejection, formula validation, batch truncation, and
   mutation-budget metrics are observable and recoverable;
6. evaluate the same Base checkpoint under the combined harness to establish a
   new, non-cross-version baseline.
