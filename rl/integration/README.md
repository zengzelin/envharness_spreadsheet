# verl-agent integration

`envharness_rl` plugs into verl-agent through additive routes in:

    third_party/verl-agent/agent_system/environments/env_manager.py

The SpreadsheetBench route is checked before verl-agent's stock environment
branches:

```python
if config.env.env_name.lower().startswith("envharness_rl/spreadsheetbench"):
    from envharness_rl.spreadsheetbench.envs import (
        build_envharness_spreadsheetbench_envs)
    from envharness_rl.spreadsheetbench.manager import (
        SpreadsheetBenchEnvironmentManager)
    ...
    return envs, val_envs
```

The ALFWorld route remains inside the stock ALFWorld branch and fires first:

```python
if config.env.env_name.lower().startswith("envharness_rl/alfworld"):
    # put release_version/ AND release_version_rl/ on sys.path
    from envharness_rl.alfworld import (
        build_envharness_alfworld_envs, envharness_alfworld_projection)
    ...
    return envs, val_envs
```

It is intentionally separate from the legacy `envharness/alfworld` route that
follows it -- active training runs depend on the legacy path, so it is left
untouched. Selecting between them is purely `env.env_name`:

| `env.env_name`            | env layer |
|---------------------------|-----------|
| `envharness_rl/spreadsheetbench` | `SpreadsheetBenchEnv` + OJ verifier |
| `envharness_rl/alfworld`  | release_version `AlfworldEnv` + `Rules` (this package) |
| `envharness/alfworld`     | legacy `AlfworldBridge` + `MutationLayer` |
| `alfworld/AlfredTWEnv`    | stock verl-agent alfworld |

## Env vars the route reads

| var | effect |
|---|---|
| `SPREADSHEETBENCH_DATA` | Required for SpreadsheetBench; directory containing `dataset.json` and `spreadsheet/`. |
| `SPREADSHEETBENCH_STEP_TIMEOUT` | Optional per-`run_python` timeout in seconds; default `60`. |
| `SPREADSHEETBENCH_OBS_TRUNCATE` | Optional maximum tool observation length; default `6000`. |
| `SPREADSHEETBENCH_PYTHON_ERROR_PENALTY` | Failed non-syntax `run_python` reward; default `-0.05`. |
| `SPREADSHEETBENCH_SYNTAX_ERROR_PENALTY` | Syntax/indentation error reward; default `-0.1`. |
| `ALFWORLD_TRAIN_SUBSET_PATH` | JSONL of `{"game_file": ...}`; restricts TRAIN. Train-only. |
| `ENVHARNESS_MUTATION_CORPUS` | JSONL of `{game_file, rules_code, in_env_actions}`; applies the matching `Rules` per train episode. Train-only. |
| `ENVHARNESS_SUBSET_AUTHORITATIVE` | `1/true/yes` -> treat the subset as authoritative (load game COPIES outside alfworld's scanned dir). Train-only. |

`scripts/run_grpo.sh` sets these from the bundled example data by default.

## Ray lifecycle

`envharness_rl.spreadsheetbench` requires the caller to initialize Ray. It
never calls a bare `ray.init()` and therefore cannot accidentally create a
local cluster while train and validation actors are being constructed. Use:

```bash
bash rl/scripts/mpi_ray_up.sh
bash rl/scripts/submit_spreadsheetbench_ray_smoke.sh
```

The second command first runs `ray status`, then submits through the dashboard
address saved by the first command. verl-agent training jobs should likewise be
submitted through Ray Jobs with `+ray_init.address=auto`.

> Attribution: the `.patch` files in this directory are diffs against
> [verl-agent](https://github.com/langfengQ/verl-agent) (Apache-2.0) and
> therefore contain verbatim upstream context lines alongside our additions.

## Fetching verl-agent

verl-agent is not checked into this repository. `scripts/fetch_verl_agent.sh`
clones upstream at the pinned commit (`796ed310`) into
`third_party/verl-agent/` (gitignored) and applies
`verl_agent_env_manager.patch` (the maintained environment routes above) and
`verl_agent_tracking_lifecycle.patch` (explicit, idempotent tracker shutdown).
Both modes also apply `verl_agent_spreadsheetbench_runtime.patch`, which adds
Spreadsheet-RL split routing and structured rollout diagnostics.
They also apply `verl_agent_spreadsheetbench_stall_diagnostics.patch` for
phase heartbeats and `verl_agent_spreadsheetbench_metrics.patch` for native-tool
metrics, weighted validation, and reward-component logging.
`PATCH=all` applies `verl_agent_all_changes.patch`
instead of the route patch, followed by the same tracking lifecycle patch (a
superset: DAPO / Qwen3-8B / webshop / SWE-Gym support; see
`ENVHARNESS_CHANGES.md` for the per-file breakdown).
