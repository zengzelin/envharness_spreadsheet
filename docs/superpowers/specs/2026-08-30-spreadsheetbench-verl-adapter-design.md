# SpreadsheetBench verl-agent Adapter Design

## Goal

Connect `envharness.bridges.spreadsheetbench.SpreadsheetBenchEnv` to
verl-agent's multi-turn GRPO environment API under
`env.env_name=envharness_rl/spreadsheetbench`.

## Architecture

`envharness_rl.spreadsheetbench` owns the stable adapter. A Ray worker owns one
`SpreadsheetBenchEnv`, translates projected actions into EnvHarness `Action`
objects, and grades the workbook on submit or the final allowed step. A vector
wrapper groups workers using verl-agent's `(env_num, group_n)` convention.

verl-agent gets a dedicated `SpreadsheetBenchEnvironmentManager`. It preserves
multiline Python exactly and supplies a SpreadsheetBench-specific prompt and
history instead of reusing ALFWorld's admissible-command prompt.

## Action Protocol

The model emits one JSON tool call per turn:

```text
<tool_call>
{"name":"run_python","arguments":{"code":"print('inspect')"}}
</tool_call>
```

or:

```text
<tool_call>{"name":"submit","arguments":{}}</tool_call>
```

The projection accepts `arguments` as either an object or a JSON-encoded
object. Malformed or unknown calls become an invalid action and receive
`is_action_valid=0`; they do not crash or end the episode.

## Episode Semantics

- Reset selects tasks deterministically from the dataset by seed. Subsequent
  resets advance by one full environment batch so training covers new tasks.
- All members of one GRPO group use the same task seed.
- Intermediate reward is `0.0`.
- Submit grades immediately and returns `EvaluationResult.score`.
- Reaching `max_steps` grades the current output workbook and ends the episode.
- Final info always contains boolean `won`, `task_id`, and grading metrics.
- `close()` tears down work directories and Ray actors.

## Configuration

The dataset path comes from `SPREADSHEETBENCH_DATA`; no cluster mount path is
hardcoded. Optional reset controls use existing bridge environment variables or
adapter builder arguments. The repo-local fetch patch is the source of truth
for verl-agent registration; the untracked `third_party/verl-agent` checkout is
updated only so the current environment is immediately runnable.

Ray cluster lifecycle is owned by launch scripts, not the adapter. The cluster
must pass a readiness check before a smoke or training job is submitted; the
driver then connects with `ray.init(address="auto")` before calling `make_envs`.

## Verification

Tests cover projection validity and code preservation, worker reward/termination
behavior with a real lightweight fake EnvHarness environment, vector seed
grouping, manager prompt/history behavior, and clean application of the
registration patch to the pinned verl-agent commit.
