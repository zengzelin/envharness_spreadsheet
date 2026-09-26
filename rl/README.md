# EnvHarness RL — Agentic GRPO Adapters

RL-training adapters that drive [verl-agent](https://github.com/langfengQ/verl-agent)
GRPO on EnvHarness environments. ALFWorld is selected with
`env.env_name=envharness_rl/alfworld`; SpreadsheetBench is selected with
`env.env_name=envharness_rl/spreadsheetbench`. verl-agent is NOT
checked in — a fetch script reproduces the exact tree (upstream at a pinned
commit + additive EnvHarness routes).

## Environment

- conda env `verl-agent` (Python 3.12; TextWorld's PDDL grammar is not 3.13-safe),
  with verl / vLLM 0.11 / flash-attn installed.
- ALFWorld game data in `~/.cache/alfworld/`.
- GPUs (2 for the smoke, 8 for the full run).

SpreadsheetBench uses an externally started Ray cluster. The adapter does not
start a local Ray instance; this keeps cluster bootstrap separate from
environment reset, `run_python`, LibreOffice grading, and training submission.

```bash
# env layer loads (no GPU)
PYTHONPATH=..:. \
  ~/miniconda3/envs/verl-agent/bin/python scripts/smoke_worker.py

# SpreadsheetBench worker + real OJ grading (run from repository root)
export SPREADSHEETBENCH_DATA="$PWD/experiments/spreadsheetbench/data/spreadsheetbench_verified_400"
PYTHONPATH=.:rl python rl/scripts/smoke_spreadsheetbench_worker.py
```

## External Ray workflow

Start Ray first and wait for a successful readiness check:

```bash
bash rl/scripts/mpi_ray_up.sh
```

The command writes `runs/ray/ray_address.env`. Submit the full Ray adapter
smoke only after that file exists:

```bash
bash rl/scripts/submit_spreadsheetbench_ray_smoke.sh
```

The submitted job connects with `ray.init(address="auto")`, creates train and
validation environment actors, executes `run_python`, submits all train
workbooks, and checks grouped rewards. Training launchers must follow the same
pattern and pass `+ray_init.address=auto` to verl-agent.

Before a full Spreadsheet-RL run, validate the loader at the same 128-actor
shape used by `TRAIN_BS=16 GROUP_N=8`. This smoke connects to the existing Ray
cluster and performs reset/close only; it does not load the policy model:

```bash
source runs/ray/ray_address.env
export SPREADSHEET_RL_DATA_ROOT="$PWD/experiments/spreadsheetbench/data/Spreadsheet-RL"
export SPREADSHEET_RL_TRAIN_FILE=train_hermes.parquet
export SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS=180
PYTHONPATH=.:rl:third_party/verl-agent \
  python rl/scripts/smoke_spreadsheetbench_loader_scale.py
```

Success requires 128 completed resets, 16 unique tasks, and eight identical
task IDs in each rollout group. Reduce `SPREADSHEETBENCH_SCALE_ENV_NUM` or
`SPREADSHEETBENCH_SCALE_GROUP_N` only for local diagnosis, not for the final
scale acceptance check. The script sends absolute repository import paths to
Ray workers through `runtime_env`, so a directly connected multi-node run does
not depend on raylet inheriting the driver's temporary `PYTHONPATH`.

After the external-Ray smoke succeeds, launch the one-epoch GRPO smoke:

```bash
bash rl/scripts/submit_spreadsheetbench_grpo.sh
```

This defaults to the shared local model at
`../llm_model/Qwen2.5-1.5B-Instruct`, two GPUs, two prompts, two rollouts per
prompt, three environment steps, one epoch, and a checkpoint after the smoke
step. Override settings through
environment variables, for example:

```bash
MODEL=/shared/models/Qwen2.5-7B-Instruct \
N_GPUS_PER_NODE=8 TP=4 TRAIN_BS=8 GROUP_N=4 EPOCHS=10 MAX_STEPS=8 \
bash rl/scripts/submit_spreadsheetbench_grpo.sh
```

The formal configuration uses the shared `Qwen2.5-7B-Instruct` model, eight
GPUs, and exactly 150 optimizer steps. It validates before training, then
evaluates and saves a checkpoint every 10 steps; the trainer also evaluates
and saves at the final step:

```bash
MODE=full bash rl/scripts/submit_spreadsheetbench_grpo.sh
```

`VAL_SIZE` controls how many validation rows are generated and
`VAL_CONCURRENCY` controls the reusable Ray actor pool. For example, the full
399-task verified split can be evaluated in batches of at most 64 actors:

```bash
SPREADSHEETBENCH_TOOL_SET=native_basic \
VAL_SIZE=399 VAL_CONCURRENCY=64 \
MODE=full bash rl/scripts/submit_spreadsheetbench_grpo.sh
```

With `native_basic`, the model can call `write_range`, `clear_range`,
`fill_formula`, `format_range`, `delete_rows`, `delete_columns`,
`manage_sheet`, and `recalculate_and_read` in addition to the structured read
tools. One model turn may contain up to four ordered `<tool_call>` blocks; they
execute sequentially and consume one episode step. `submit` must be the final
call in its batch. `recalculate_and_read` recalculates a temporary copy with
LibreOffice, must be last in its turn, and should be followed by `submit` in a
new turn after the returned values have been checked.

The launcher writes `launch.log`, `train.log`, and checkpoints under
`runs/grpo_spreadsheetbench_<mode>_<timestamp>/`; the corresponding
`*_latest` symlink points at the newest run. The parquet inputs are generated
offline under `runs/data/spreadsheetbench_agent/text/`.

Training enables console, W&B, and TensorBoard logging by default. Set the
credential in the shell rather than storing it in the repository:

```bash
export WANDB_API_KEY='...'
export WANDB_BASE_URL='https://your-wandb-server.example'
bash rl/scripts/submit_spreadsheetbench_grpo.sh
```

Leave `WANDB_BASE_URL` unset to use the W&B SDK default endpoint, or set it in
the shell when using a self-hosted service. Use `WANDB_MODE=offline` when the
service is unavailable. Each run stores W&B and TensorBoard files under
`wandb/` and `tensorboard/`, decoded verl generations under `rollouts/verl/`,
and complete train/validation environment trajectories under
`rollouts/env/{train,val}/`. `run_manifest.json` records the source commit,
resolved settings, and Hydra command without credentials.

SpreadsheetBench rollout phases are logged as
`[spreadsheet-rollout] START|HEARTBEAT|END|ERROR`, including whether the phase
is train or validation, the turn number, active trajectory count, and elapsed
time. `rollout_call` is a process-local sequence number, so consecutive
validation and training collections can be distinguished. Ray environment
calls additionally log `[spreadsheet-ray]` boundaries.
This separates stalls in `env_reset`, model `generate_sequences`, and
`env_step`; an actor timeout reports the indexes, Ray actor IDs, task IDs, and
actions that did not return. Each actor also logs `[spreadsheet-worker]`
boundaries around its environment step and grading. The bridge logs
`[spreadsheet-bridge]` boundaries for `run_python`, native read/write tools,
LibreOffice output/golden recalculation, and workbook comparison.

`run_python` and LibreOffice execute in isolated process groups. On timeout the
whole process group is killed and reaped, preventing a child process from
holding an output pipe, workbook, or LibreOffice profile after its parent has
timed out.

```bash
# Defaults shown; override before submitting when a workload needs more time.
export SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS=600
export SPREADSHEETBENCH_PHASE_HEARTBEAT_SECONDS=60
export SPREADSHEETBENCH_MAX_RECALC_CALLS=1
export SPREADSHEETBENCH_RECALC_TIMEOUT_SECONDS=120
```

The actor timeout bounds each parallel environment `reset`, `step`, and
`close`. It does not terminate a stalled distributed model generation; the
periodic `generate_sequences` heartbeat identifies that case for inspection in
the Ray/NCCL worker logs. Both values are recorded in `launch.log` and
`run_manifest.json`.

## Getting verl-agent (clone + patch)

verl-agent is not part of this repository — you reproduce the exact tree the
experiments ran against by cloning upstream at a pinned commit and applying
our patch. The one-command way:

```bash
bash rl/scripts/fetch_verl_agent.sh
```

That clones [verl-agent](https://github.com/langfengQ/verl-agent) at commit
`796ed310287fa605c9292a0fce07a86d79fde05e` into `third_party/verl-agent/`
(gitignored) and applies the ordered patches under `rl/integration/`, including
the EnvHarness routes, tracking lifecycle, SpreadsheetBench runtime,
diagnostics, metrics, val-only actor allocation, and native-tool metrics. It is
idempotent: re-running on an already-patched tree is a no-op.

Equivalent manual steps, if you'd rather drive git yourself or place the
tree elsewhere (then point `$VERL_AGENT` at it):

```bash
git clone https://github.com/langfengQ/verl-agent third_party/verl-agent
cd third_party/verl-agent
git checkout 796ed310287fa605c9292a0fce07a86d79fde05e
git apply ../../rl/integration/verl_agent_env_manager.patch
```

Two patches are available (see `rl/integration/ENVHARNESS_CHANGES.md` for
the per-file breakdown):

| Patch | What it applies | When |
|---|---|---|
| `verl_agent_env_manager.patch` | maintained EnvHarness environment routes | default; enough for ALFWorld and SpreadsheetBench |
| `verl_agent_all_changes.patch` | superset: env route + DAPO / Qwen3-8B / webshop / SWE-Gym adaptations | `PATCH=all bash rl/scripts/fetch_verl_agent.sh`, or apply manually INSTEAD of the default |

Apply exactly one of the two — they overlap, so applying both fails.

## Run

```bash
# 2-GPU GRPO smoke: Qwen2.5-1.5B, 2 epochs, on the 6 bundled mutated games.
bash scripts/run_grpo.sh

# unmutated control (no corpus, full TRAIN)
MUTATION_CORPUS= TRAIN_SUBSET_PATH= bash scripts/run_grpo.sh

# full 8-GPU run (Qwen3-8B)
MODE=full bash scripts/run_grpo.sh
```

Output lands under `runs/grpo_envrl_alfworld_<mode>_<ts>/`. `val_before_train`
reports per-task-type success rates; each step logs `actor/pg_loss`,
`episode/reward/mean`, `episode/success_rate`.

## Knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `MODE` | `smoke` | `smoke` (Qwen2.5-1.5B, 2 GPU) or `full` (Qwen3-8B, 8 GPU) |
| `MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | policy model |
| `MUTATION_CORPUS` | bundled `example_corpus.jsonl` | `{game_file, rules_code, in_env_actions}` per line; empty = unmutated |
| `ALFWORLD_DATA` | -- | ALFWorld data root; the bundled JSONLs list `game_file` relative to it |
| `TRAIN_SUBSET_PATH` | bundled `train_subset.jsonl` | restrict TRAIN to these games; empty = full TRAIN |
| `VERL_AGENT` | `third_party/verl-agent` | which verl-agent to run (populate with `scripts/fetch_verl_agent.sh`) |
| `VLLM_ATTENTION_BACKEND` | `FLASH_ATTN` | keep FLASH_ATTN on vLLM 0.11 (XFORMERS V1 needs block_size % 256) |
| `N_GPUS` / `TP` | `2 / 2` (smoke) | GPUs / tensor-parallel |

## Files

| File | Purpose |
|---|---|
| `scripts/run_grpo.sh` | GRPO launcher (smoke / full) |
| `scripts/fetch_verl_agent.sh` | fetch verl-agent @ pinned commit + apply the env route patch |
| `scripts/smoke_worker.py` | no-GPU env sanity check |
| `scripts/smoke_spreadsheetbench_worker.py` | no-GPU SpreadsheetBench reset/grade sanity check |
| `scripts/mpi_ray_up.sh` / `scripts/mpi_ray_node.sh` | start and readiness-check the external Ray cluster |
| `scripts/submit_spreadsheetbench_ray_smoke.sh` | submit the full adapter smoke through Ray Jobs |
| `scripts/smoke_spreadsheetbench_ray.py` | external-Ray reset/run/submit/OJ integration driver |
| `scripts/smoke_spreadsheetbench_loader_scale.py` | 128-actor Spreadsheet-RL reset/close scale check without a model |
| `scripts/submit_spreadsheetbench_grpo.sh` | submit SpreadsheetBench GRPO to the ready Ray cluster |
| `scripts/run_spreadsheetbench_grpo.sh` | build and execute the verl-agent GRPO command |
| `scripts/prepare_spreadsheetbench_verl_data.py` | generate offline text parquet placeholders for verl-agent |
| `scripts/build_corpus.py` | build the example corpus + subset from a legacy corpus |
| `envharness_rl/alfworld/envs.py` | Ray-actor parallel `AlfworldEnv` + `Rules` workers |
| `envharness_rl/alfworld/projection.py` | `<action>`/`<think>` extraction + admissible-command normalize |
| `envharness_rl/spreadsheetbench/` | SpreadsheetBench projection, workers, and verl-agent manager |
| `experiments/alfworld/data/` | `example_corpus.jsonl` (6 mutated games) + `train_subset.jsonl` |
| `../third_party/verl-agent/` | fetched verl-agent (gitignored; upstream + maintained EnvHarness routes) |
| `integration/ENVHARNESS_CHANGES.md` | what differs from upstream |
| `integration/verl_agent_env_manager.patch` | maintained environment routes applied by the fetch script |
| `integration/verl_agent_all_changes.patch` | full patch (re-enables DAPO / Qwen3-8B / webshop / SWE-Gym) |

## Acknowledgements

The RL experiments run on the third-party
[**verl-agent**](https://github.com/langfengQ/verl-agent) repository (GiGPO;
Apache-2.0), at upstream commit `796ed31` — fetched by
`scripts/fetch_verl_agent.sh`, not redistributed here. All RL training /
rollout infrastructure is theirs — we only add additive EnvHarness environment
routes (see `integration/ENVHARNESS_CHANGES.md`).
verl-agent in turn builds on [verl](https://github.com/volcengine/verl).
Please cite/credit them when using this.
