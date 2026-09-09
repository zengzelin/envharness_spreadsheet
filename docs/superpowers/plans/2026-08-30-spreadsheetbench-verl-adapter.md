# SpreadsheetBench verl-agent Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested `envharness_rl/spreadsheetbench` environment and register it with verl-agent.

**Architecture:** A dedicated projection parses JSON tool calls without changing Python source. Ray workers wrap `SpreadsheetBenchEnv`, while a dedicated verl-agent manager supplies prompts, history, validity metadata, and NumPy outputs.

**Tech Stack:** Python 3.11+, EnvHarness, Ray, pytest, verl-agent, LibreOffice

**Spec:** `docs/superpowers/specs/2026-08-30-spreadsheetbench-verl-adapter-design.md`

## Global Constraints

- Do not hardcode either host or cluster mount paths.
- Treat `rl/integration/verl_agent_env_manager.patch` as the tracked source of truth for registration.
- Do not add the untracked `third_party/verl-agent` tree to the EnvHarness repository.
- Preserve multiline Python source byte-for-byte after JSON decoding.
- Use `EvaluationResult.score` as final reward.

---

### Task 1: Action Projection

**Files:**
- Create: `rl/tests/test_spreadsheetbench_projection.py`
- Create: `rl/envharness_rl/spreadsheetbench/projection.py`
- Create: `rl/envharness_rl/spreadsheetbench/__init__.py`

**Interfaces:**
- Produces: `envharness_spreadsheetbench_projection(actions: list[str]) -> tuple[list[dict], list[int]]`

- [ ] Write tests for valid `run_python`, valid `submit`, string arguments, malformed JSON, and unknown tools.
- [ ] Run `pytest -q rl/tests/test_spreadsheetbench_projection.py` and confirm imports fail because the module is absent.
- [ ] Implement the minimal parser and action validation.
- [ ] Re-run the projection tests and confirm they pass.

### Task 2: Worker and Vector Environment

**Files:**
- Create: `rl/tests/test_spreadsheetbench_envs.py`
- Create: `rl/envharness_rl/spreadsheetbench/envs.py`
- Modify: `rl/envharness_rl/spreadsheetbench/__init__.py`
- Modify: `rl/pyproject.toml`

**Interfaces:**
- Produces: `EnvharnessSpreadsheetWorker`, `EnvharnessSpreadsheetEnvs`, and `build_envharness_spreadsheetbench_envs(seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None)`.

- [ ] Write worker tests using an injected environment factory and literal reset/step/evaluation results.
- [ ] Write a grouping test proving seeds are `[seed, seed, seed+1, seed+1]` for `env_num=2, group_n=2`.
- [ ] Run `pytest -q rl/tests/test_spreadsheetbench_envs.py` and confirm failures identify missing behavior.
- [ ] Implement worker reset, action conversion, terminal grading, max-step grading, cleanup, and vector dispatch.
- [ ] Re-run worker tests and projection tests.

### Task 3: verl-agent Manager and Registration

**Files:**
- Create: `rl/tests/test_spreadsheetbench_verl_manager.py`
- Modify: `third_party/verl-agent/agent_system/environments/env_manager.py`
- Modify: `rl/integration/verl_agent_env_manager.patch`

**Interfaces:**
- Produces: `SpreadsheetBenchEnvironmentManager` and the `envharness_rl/spreadsheetbench` route in `make_envs(config)`.

- [ ] Write a manager test that exercises reset, prompt construction, projected step, history, rewards, dones, and action validity.
- [ ] Run the test against the current verl-agent checkout and confirm it fails because the manager is absent.
- [ ] Implement the manager and route in the current checkout.
- [ ] Regenerate the tracked patch from pinned upstream state plus all maintained EnvHarness routes.
- [ ] Re-run manager and adapter tests.

### Task 4: Smoke and Repository Verification

**Files:**
- Create: `rl/scripts/smoke_spreadsheetbench_worker.py`
- Modify: `rl/integration/README.md`

**Interfaces:**
- Produces: a no-GPU worker smoke using `SPREADSHEETBENCH_DATA`.

- [ ] Add a smoke that resets one real task, submits the preseeded output, checks typed reward/info, and closes the worker.
- [ ] Run all `rl/tests` plus the real worker smoke in the available cluster environment.
- [ ] Verify the tracked patch applies cleanly to pinned commit `796ed310287fa605c9292a0fce07a86d79fde05e` in a temporary checkout.
- [ ] Inspect `git diff --check`, `git status --short`, and the final diff; do not stage unrelated files.
