# Spreadsheet Agent Harness Recovery Design

**Date:** 2026-10-11

## Goal

Improve SpreadsheetBench agent reliability before the next GRPO run by adding
three cooperating harness features:

1. a compact, persistent edit ledger that survives short interaction-history
   windows;
2. validation that recalculates formula outputs and exposes actionable answer-
   range results before submit; and
3. soft tool-call constraints that recover safe formatting mistakes, reject
   invalid schemas precisely, and stop repeated invalid-action loops.

The target outcome is lower wrong-sheet/range, formula-result-mismatch,
invalid-action, and no-submit rates without changing the official final
SpreadsheetBench score.

## Context and Constraints

The current harness already provides transactional Python execution, static
formula validation, workbook revisions, a submit-validation gate, a bounded
four-call tool batch, parser diagnostics, and an explicit
`recalculate_and_read` tool. Those behaviors and their existing tests must be
preserved.

The diagnostic GRPO run used compact history with only two turns. In compact
mode, prior actions are reduced to tool names and parser status, so the next
prompt does not reliably retain the sheet, range, formula, or validation state.
Static `validate_workbook` can approve a structurally valid formula without
showing its recalculated value. Repeated invalid actions consume the remaining
episode turns and are only penalized at the reward layer.

This design intentionally avoids grammar-constrained decoding. Hard decoding
masks require matching masks in rollout, old-policy, current-policy, and
reference-policy log-probabilities before they are safe for PPO.

## Non-goals

- Changing the official binary workbook evaluator or golden comparison.
- Adding dense reward shaping in this change.
- Adding new spreadsheet business operations such as formula copying.
- Replacing the existing compact/full history modes.
- Enabling hard grammar-constrained or guided decoding.
- Automatically repairing workbook contents or formulas.
- Exposing golden workbook values during validation.

## Design Overview

The bridge remains the source of truth for workbook revision and validation
state. The environment manager owns a bounded per-episode edit ledger derived
from actions that actually executed. The projection layer owns syntactic tool-
call recovery and schema feedback. The worker owns the invalid-action streak
because it can terminate and grade the episode, while the manager renders the
reported streak and other structured state into a stable prompt section.

## Edit Ledger

### Stored entries

The manager maintains one ledger per active episode. After every environment
step, it appends one entry for each executed tool call, in execution order.
Entries contain only bounded structured data:

- monotonic episode step and action index;
- tool name;
- normalized target summary, such as sheet/range, start/end cell, queried
  text, or sheet operation;
- for `fill_formula`, a bounded formula template;
- for `run_python`, code length only, never source;
- outcome: success, failure, rollback, rejection, or termination;
- concise error code when present;
- workbook revision, current validation revision, and current recalc revision
  reported by the bridge after the action.

Projected but unexecuted calls are not recorded as edits. They remain visible
through existing parser and truncation diagnostics.

### Bounds and rendering

The ledger retains the most recent 12 entries and renders at most 4,000
characters. Old entries are dropped as whole records. Individual string fields
are bounded before serialization so a formula, query, or sheet name cannot
consume the ledger budget.

Every prompt receives a `Persistent workbook state` section independent of
`HISTORY_LENGTH`. It includes:

- current workbook revision;
- whether the current revision is validated and/or recalculated;
- consecutive invalid-action count;
- recent ledger entries.

The existing recent interaction history remains responsible for natural-
language observations. The ledger is a deterministic state summary, not a
second transcript.

### Bridge state contract

Every bridge response, including reads, writes, validation, recalc, Python,
submit rejection, and unknown tools, reports:

- `workbook_revision`;
- `last_validation_revision`;
- `last_recalc_revision`;
- `validation_current`;
- `recalc_current`.

Failed or rolled-back mutations do not advance the workbook revision. A
successful mutation makes both current flags false through revision mismatch.

## Recalc-aware Validation

### Validation flow

`validate_workbook` becomes a two-stage operation:

1. Run the existing static validation against the committed output workbook.
2. If static validation passes, copy the workbook to a temporary validation
   path, recalculate that copy with LibreOffice, and inspect the task answer
   ranges using cached values.

The committed output workbook is never modified by validation. Temporary
files and the isolated LibreOffice profile are removed on all success and
failure paths.

### Result checks

Recalc-aware validation fails when:

- LibreOffice recalculation times out or fails;
- the recalculated workbook cannot be opened;
- an answer-range cached value contains an Excel error such as `#REF!`,
  `#VALUE!`, `#NAME?`, `#DIV/0!`, `#NUM!`, or `#N/A`; or
- the existing static formula checks fail.

It does not compare against the golden workbook and therefore cannot assert
task correctness.

On success, the response includes a bounded preview of answer-range cached
values and records both `last_validation_revision` and
`last_recalc_revision` as the current workbook revision. On failure, neither
revision advances.

### Caching and explicit recalc

Repeated validation of an unchanged workbook returns the cached validation
report without invoking LibreOffice again. A mutation invalidates the cache.

The explicit `recalculate_and_read` tool remains available for agent-selected
ranges and keeps its existing per-episode call limit. Automatic recalculation
inside `validate_workbook` does not consume that explicit-tool budget.

Recalc infrastructure failure is distinguished from workbook formula failure:

- infrastructure failure uses `recalc_validation_error` and is observable as
  an environment error rather than evidence that the policy's formula is
  wrong;
- formula/error-cell failure uses `formula_validation_error` and gives the
  model bounded cell-level feedback.

### Compatibility switch

Direct legacy bridge callers retain static-only validation by default.
Training and evaluation launch paths set
`SPREADSHEETBENCH_VALIDATE_WITH_RECALC=true`. The setting is exported, logged,
and captured in the run manifest.

## Soft Tool-call Constraints

### Safe parser recovery

The projection layer continues accepting native `<tool_call>` blocks and the
existing fenced/bare JSON recovery. It additionally recovers a missing closing
`</tool_call>` only when the suffix after `<tool_call>` contains exactly one
complete JSON object and only whitespace after that object. Ambiguous or
partially decoded JSON remains invalid.

Recovery never changes a tool name, invents an argument, coerces an arbitrary
scalar into an object, or admits more than the existing four-call limit.

### Schema validation and feedback

Each enabled tool has an explicit allowed-key schema in the projection layer.
The parser rejects:

- unknown tool names;
- missing required keys;
- unexpected keys;
- wrong top-level argument types; and
- arguments for zero-argument tools.

Errors use a compact machine-readable diagnostic code plus one corrective
example for the attempted tool. Unknown-tool feedback includes the enabled
tool-name whitelist. The observation avoids replaying the model's malformed
payload.

Schema checks in the bridge remain authoritative. Projection checks are early
feedback and must not weaken runtime validation.

### Invalid-loop guard

The worker tracks consecutive turns whose aggregate projected action batch is
invalid. A valid projected batch resets the count. When the count reaches a
configurable threshold, default `3`, the episode terminates and is graded in
its current workbook state.

The terminating response reports:

- `invalid_streak_terminated=true`;
- `consecutive_invalid_actions`;
- the final parser diagnostic code.

The guard does not terminate on a syntactically valid tool whose execution
fails, because execution feedback may allow recovery. It also does not change
the invalid-action penalty; reward changes are outside this design.

### Long invalid outputs

Projection records `invalid_output_over_budget` when an invalid response
exceeds a configurable character threshold, default 16,000. This is a
diagnostic classification and concise-feedback path, not retrospective token
savings. Actual generation-length changes will be evaluated separately in the
training configuration.

## Metrics

Add and aggregate:

- `env/edit_ledger_entries`;
- `env/validation_recalc_attempted`;
- `env/validation_recalc_cache_hit`;
- `env/validation_recalc_elapsed_ms`;
- `env/validation_recalc_formula_error_count`;
- `env/validation_recalc_infrastructure_error`;
- `parser/missing_end_recovered`;
- `parser/schema_error`;
- `parser/invalid_output_over_budget`;
- `episode/consecutive_invalid_actions_max`;
- `episode/invalid_streak_terminated`.

Existing metric names and semantics remain unchanged.

## Error and Recovery Semantics

- Ledger serialization failure must not fail an environment action; the
  manager emits an empty bounded entry and a diagnostic metric.
- Validation infrastructure failure is non-terminal and leaves validation
  stale so the model can retry or use explicit recalc.
- Workbook formula failure is non-terminal and provides bounded cell feedback.
- The invalid-loop guard is terminal only at the configured threshold.
- Submit still requires validation of the current workbook revision.
- No recovery path may execute a tool call whose JSON boundary or schema is
  ambiguous.

## Implementation Boundaries

Expected product changes are limited to:

- `envharness/bridges/spreadsheetbench/bridge.py` for state reporting and
  recalc-aware validation;
- the existing recalc/read helper module for reusable temporary-copy
  recalculation and cached-value inspection;
- `rl/envharness_rl/spreadsheetbench/envs.py` for the invalid-streak guard and
  terminal grading;
- `rl/envharness_rl/spreadsheetbench/manager.py` for ledger storage,
  rendering, feedback, and metrics;
- `rl/envharness_rl/spreadsheetbench/projection.py` for safe recovery and
  explicit schemas;
- training/submission scripts and manifests for the new settings;
- the focused bridge, projection, manager, environment, rollout-summary, and
  training-script tests;
- the vendored verl-agent integration patch only if new metric propagation
  cannot be expressed through the existing manager aggregation.

No unrelated refactor is included.

## Test Strategy

### Edit ledger

1. Successful native writes retain bounded sheet/range/formula summaries.
2. Python entries retain code length but never source.
3. Failed and rolled-back mutations do not claim a new revision.
4. Multi-call short-circuit records only executed calls.
5. More than 12 entries evicts whole oldest records and stays below the prompt
   character budget.
6. Ledger state remains visible with `HISTORY_LENGTH=0` and compact history.

### Recalc-aware validation

1. A correct formula recalculates on a temporary copy and validates.
2. The committed workbook remains byte-for-byte unchanged.
3. Recalculated Excel errors reject validation with bounded cell feedback.
4. Static formula errors prevent a LibreOffice invocation.
5. An unchanged second validation is a cache hit.
6. Mutation invalidates the cached report and submit gate.
7. Infrastructure failure does not advance either validation revision.
8. Automatic validation does not consume the explicit recalc-tool limit.
9. Static-only compatibility mode preserves existing direct-caller behavior.

### Soft tool-call constraints

1. A complete single JSON object with a missing closing tag is recovered.
2. Partial, trailing, or multiple ambiguous objects are rejected.
3. Unknown, missing, extra, and wrong-type arguments receive stable codes and
   bounded corrective feedback.
4. Existing native, fenced, bare, and four-call truncation cases remain valid.
5. Three consecutive invalid batches terminate; a valid batch resets the
   streak.
6. Valid tool execution failures do not increment the invalid streak.
7. Oversized invalid output is classified without echoing its content.

### Verification sequence

Run focused tests first, followed by the complete `rl/tests` and relevant
bridge tests. Then run a fixed-task worker smoke test and frozen Base/step-100
evaluations before any new GRPO training. Acceptance requires no evaluator
errors, no official-score regression attributable to infrastructure, lower
invalid/no-submit rates, and observable ledger/recalc/invalid-streak metrics.

## Rollout and Compatibility Risks

- LibreOffice validation adds latency. Revision caching and static-first
  validation bound repeated cost.
- More prompt state consumes tokens. Fixed entry and character budgets prevent
  unbounded growth.
- Strict unexpected-key rejection may expose previously ignored model output.
  Frozen-checkpoint evaluation quantifies this before training.
- Invalid-streak termination may end recoverable episodes. The default of
  three and reset-on-valid behavior preserve two repair opportunities.
- Recalc-aware validation can reveal formula errors that static validation
  previously accepted; this is intentional, and the episode remains open for
  repair.
