# Spreadsheet Agent Harness：GRPO 操作手册

本目录包含 [Spreadsheet Work Agent](../SPREADSHEET_WORK_AGENT.md) 的训练与评测适配器。
Spreadsheet Agent Harness 将 [verl-agent](https://github.com/langfengQ/verl-agent)
GRPO 与有状态工作簿环境、事务式工具、校验门禁、可信 SpreadsheetBench 评测器和结构化失败
诊断连接起来。

当前 Spreadsheet 运行链路为：

```text
verl-agent
  -> SpreadsheetBenchEnvironmentManager
  -> EnvharnessSpreadsheetEnvs
  -> Ray EnvharnessSpreadsheetWorker actors
  -> SpreadsheetBenchEnv
  -> 工作簿工具 + 可信评测器
```

每个 Worker 直接实例化 `SpreadsheetBenchEnv`。本文描述的行为由 Spreadsheet Bridge 和
RL Adapter 自身实现；当前 Spreadsheet GRPO 训练没有启用其他环境包装层或 Skill
Injection Pipeline。

仓库不直接提交 verl-agent；拉取脚本通过固定的上游 Commit 和维护中的 Adapter Patch 重建
经过测试的代码树。

## 文档结构

- 运行环境与 Smoke 检查
- 外部 Ray 工作流
- Spreadsheet 训练
- Full-399 评测
- 安全与可观测配置
- 诊断与故障排查
- 获取 verl-agent
- 其他 RL Adapter
- 文件索引

## 运行环境与 Smoke 检查

- Conda 环境 `verl-agent`：Python 3.12，并安装 verl、vLLM 0.11 和 flash-attn；
  TextWorld 的 PDDL Grammar 与 Python 3.13 不兼容。
- ALFWorld 游戏数据位于 `~/.cache/alfworld/`。
- Smoke 默认使用 2 张 GPU，完整训练通常使用 8 张或更多 GPU。

SpreadsheetBench 使用外部启动的 Ray 集群。Adapter 不会自行启动本地 Ray，从而将集群
Bootstrap 与环境 Reset、`run_python`、LibreOffice 评分和训练提交分离。

```bash
# 检查环境层能否加载（不使用 GPU）
PYTHONPATH=..:. \
  ~/miniconda3/envs/verl-agent/bin/python scripts/smoke_worker.py

# SpreadsheetBench Worker + 真实 OJ 评分（从仓库根目录执行）
export SPREADSHEETBENCH_DATA="$PWD/experiments/spreadsheetbench/data/spreadsheetbench_verified_400"
PYTHONPATH=.:rl python rl/scripts/smoke_spreadsheetbench_worker.py
```

## 外部 Ray 工作流

首先启动 Ray，并等待 Readiness Check 成功：

```bash
bash rl/scripts/mpi_ray_up.sh
```

命令会写入 `runs/ray/ray_address.env`。只有该文件生成后，才能提交完整 Ray Adapter
Smoke：

```bash
bash rl/scripts/submit_spreadsheetbench_ray_smoke.sh
```

提交的 Job 使用 `ray.init(address="auto")` 连接集群，创建训练和验证环境 Actor，执行
`run_python`，提交全部训练工作簿，并检查分组 Reward。训练脚本必须遵循同一模式，并向
verl-agent 传入 `+ray_init.address=auto`。

开始完整 Spreadsheet-RL 训练前，应按 `TRAIN_BS=16 GROUP_N=8` 对应的 128 Actor 规模
验收 Loader。该 Smoke 连接现有 Ray 集群，只执行 Reset/Close，不加载 Policy Model：

```bash
source runs/ray/ray_address.env
export SPREADSHEET_RL_DATA_ROOT="$PWD/experiments/spreadsheetbench/data/Spreadsheet-RL"
export SPREADSHEET_RL_TRAIN_FILE=train_hermes.parquet
export SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS=180
PYTHONPATH=.:rl:third_party/verl-agent \
  python rl/scripts/smoke_spreadsheetbench_loader_scale.py
```

成功标准是完成 128 次 Reset、得到 16 个唯一任务，并保证每个 Rollout Group 中的 8 个
Task ID 相同。`SPREADSHEETBENCH_SCALE_ENV_NUM` 和
`SPREADSHEETBENCH_SCALE_GROUP_N` 只能在本地诊断时缩小，不能用于最终规模验收。脚本通过
`runtime_env` 向 Ray Worker 传递绝对仓库 Import Path，因此多节点运行不依赖 Raylet 继承
Driver 的临时 `PYTHONPATH`。

## Spreadsheet 训练

启动脚本的默认值仅用于兼容性冒烟测试，并依赖宿主机上实际存在的共享模型路径。
正式训练 Work Agent 时，不要直接使用不带参数的启动命令。外部 Ray 冒烟测试通过后，
应明确指定 Spreadsheet-RL 数据、工具、校验和评测配置：

```bash
MODEL=/path/to/Qwen3-4B-Thinking-2507 \
MODE=diagnostic \
NNODES=2 N_GPUS_PER_NODE=8 GPUS_PER_NODE=8 TP=2 \
TRAIN_BS=16 PPO_MINI_BS=16 GROUP_N=8 \
VAL_SIZE=399 VAL_CONCURRENCY=64 VAL_BEFORE=True \
TOTAL_TRAINING_STEPS=100 EPOCHS=100 TEST_FREQ=20 SAVE_FREQ=20 \
MAX_STEPS=10 MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=16384 \
SPREADSHEETBENCH_DATA_FORMAT=spreadsheet_rl \
SPREADSHEET_RL_DATA_ROOT=/path/to/Spreadsheet-RL \
SPREADSHEET_RL_TRAIN_FILE=train_hermes.parquet \
SPREADSHEET_RL_VAL_FILE=test_verified_hermes.parquet \
SPREADSHEETBENCH_TOOL_SET=native_basic \
SPREADSHEETBENCH_HISTORY_MODE=compact \
SPREADSHEETBENCH_REQUIRE_VALIDATION_BEFORE_SUBMIT=true \
SPREADSHEETBENCH_BADCASE_DIAGNOSTICS=full \
SPREADSHEETBENCH_MAX_RECALC_CALLS=1 \
SPREADSHEETBENCH_RECALC_TIMEOUT_SECONDS=120 \
SPREADSHEETBENCH_TRAIN_SHUFFLE_SEED=0 \
SPREADSHEETBENCH_LOG_LEVEL=error \
ACTOR_LR=3e-7 KL_LOSS_COEF=0.005 \
USE_INVALID_ACTION_PENALTY=True INVALID_ACTION_PENALTY_COEF=0.1 \
WANDB_MODE=offline \
bash rl/scripts/submit_spreadsheetbench_grpo.sh
```

进行单步集成冒烟测试时，应保留相同的数据、工具和校验配置，只缩小 `TRAIN_BS`、
`GROUP_N`、`VAL_SIZE`、`TOTAL_TRAINING_STEPS` 和 GPU 数量等负载参数。
`VAL_SIZE` 表示验证样本数；`VAL_CONCURRENCY` 表示可复用 Ray Actor 池的最大规模，
因此正式的 399 条全量验证可以按 64 并发分批完成，无须同时创建 399 个 Actor。

使用 `native_basic` 时，除结构化读取工具外，模型还可调用 `write_range`、
`clear_range`、`fill_formula`、`format_range`、`delete_rows`、
`delete_columns`、`manage_sheet` 和 `recalculate_and_read`。模型单轮最多可输出
4 个有序的 `<tool_call>` 块，这些调用依次执行，并共同消耗一个 episode step。
`submit` 必须是该批次的最后一个调用。`recalculate_and_read` 会用 LibreOffice
重新计算工作簿临时副本，因此必须是当前轮最后一个调用；检查返回值后，应在新一轮调用
`submit`。

正式配置要求：提交前必须成功校验当前工作簿版本。`run_python` 采用事务式写入，
Python 调用失败或输出无效 xlsx 时会自动回滚。纳入单元格计数的原生操作
（`write_range`、`clear_range`、`fill_formula` 和 `format_range`）单次最多处理
50,000 个单元格，单轮累计最多尝试处理 100,000 个单元格。Python 兜底和结构性操作
不计入该预算。模型若输出超过 4 个调用，只接纳前 4 个；下一条观察会要求模型在后续轮次继续。

启动脚本会将 `launch.log`、`train.log` 和 checkpoint 写入
`runs/grpo_spreadsheetbench_<mode>_<timestamp>/`；对应的 `*_latest` 符号链接
指向最近一次运行。parquet 输入会离线生成到
`runs/data/spreadsheetbench_agent/text/`。

训练默认启用控制台、W&B 和 TensorBoard 日志。凭据应在 shell 中设置，不要写入仓库：

```bash
export WANDB_API_KEY=...
export WANDB_BASE_URL='https://your-wandb-server.example'
```

然后将上述完整训练命令中的 `WANDB_MODE` 设为 `online`。

不设置 `WANDB_BASE_URL` 时使用 W&B SDK 默认端点；使用自建服务时，在 shell 中设置该变量。
服务不可用时使用 `WANDB_MODE=offline`。每次运行会把 W&B 和 TensorBoard 文件分别保存到
`wandb/` 和 `tensorboard/`，把解码后的 verl 生成结果保存到 `rollouts/verl/`，
并把完整的训练/验证环境轨迹保存到 `rollouts/env/{train,val}/`。
`run_manifest.json` 会记录源码提交、解析后的配置和不含凭据的 Hydra 命令。

## 399 条全量评测

正式 checkpoint 对比使用确定性解码和全部 399 条已验证任务。同一个 Ray 集群内的评测任务
按顺序运行。若要并行运行多组评测，每个物理集群都必须使用不同的 `CLUSTER_ID`、
`RUN_ROOT`、`RAY_STATE_FILE`、`HOSTFILE` 和 `LOG_FILE`；两个活动集群不能共用同一个
Ray 状态文件。

```bash
export CLUSTER_ID=eval_a
export RUN_ROOT="$PWD/runs/parallel_eval/$CLUSTER_ID"
export RAY_STATE_FILE="$PWD/runs/ray/ray_address_${CLUSTER_ID}.env"
export HOSTFILE="$PWD/runs/ray/hostfile_${CLUSTER_ID}"
export LOG_FILE="$PWD/runs/ray/ray_up_${CLUSTER_ID}.log"

NNODES=2 N_GPUS_PER_NODE=8 GPUS_PER_NODE=8 \
  RAY_STATE_FILE="$RAY_STATE_FILE" HOSTFILE="$HOSTFILE" LOG_FILE="$LOG_FILE" \
  bash rl/scripts/mpi_ray_up.sh
source "$RAY_STATE_FILE"
```

向该集群提交 Base 模型和一个或多个 checkpoint：

```bash
TRAIN_RUN=/path/to/training-run

RAY_STATE_FILE="$RAY_STATE_FILE" RUN_ROOT="$RUN_ROOT" \
NNODES=2 N_GPUS_PER_NODE=8 GPUS_PER_NODE=8 TP=2 \
VAL_SIZE=399 VAL_CONCURRENCY=64 MAX_STEPS=10 \
MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=16384 \
SPREADSHEET_RL_DATA_ROOT=/path/to/Spreadsheet-RL \
SPREADSHEET_RL_TRAIN_FILE=train_hermes.parquet \
SPREADSHEET_RL_VAL_FILE=test_verified_hermes.parquet \
SPREADSHEETBENCH_TOOL_SET=native_basic \
SPREADSHEETBENCH_HISTORY_MODE=compact \
SPREADSHEETBENCH_REQUIRE_VALIDATION_BEFORE_SUBMIT=true \
SPREADSHEETBENCH_BADCASE_DIAGNOSTICS=full \
SPREADSHEETBENCH_BADCASE_MAX_SCAN_CELLS=200000 \
SPREADSHEETBENCH_BADCASE_MAX_EXAMPLES=20 \
SPREADSHEETBENCH_MAX_RECALC_CALLS=1 \
SPREADSHEETBENCH_RECALC_TIMEOUT_SECONDS=120 \
SPREADSHEETBENCH_LOG_LEVEL=error \
WANDB_MODE=offline \
bash rl/scripts/submit_spreadsheetbench_eval.sh \
  base=/path/to/base-model \
  step100="$TRAIN_RUN/ckpts/global_step_100/actor/huggingface"
```

评测器会为每个标签创建一个运行目录。以下命令按任务 ID 对比已完成的运行目录；默认拒绝
配置不一致或存在评测器错误的结果：

```bash
PYTHONPATH=.:rl:third_party/verl-agent \
python rl/scripts/compare_spreadsheetbench_evals.py \
  /path/to/base-run /path/to/step100-run \
  --expected-tasks 399 \
  --json-output /path/to/comparison.json \
  --markdown-output /path/to/comparison.md \
  --task-jsonl-output /path/to/comparison_tasks.jsonl
```

## 安全与可观测性配置

| 配置项 | 正式运行推荐值 | 作用 |
| --- | --- | --- |
| `SPREADSHEETBENCH_TOOL_SET` | `native_basic` | 启用结构化读取、原生修改、公式填充、格式设置、结构编辑、重算、Python 兜底、校验和提交。 |
| `SPREADSHEETBENCH_REQUIRE_VALIDATION_BEFORE_SUBMIT` | `true` | 提交前必须校验当前工作簿版本。 |
| `SPREADSHEETBENCH_MAX_RECALC_CALLS` | `1` | 限制每个 episode 的中间 LibreOffice 重算次数。 |
| `SPREADSHEETBENCH_RECALC_TIMEOUT_SECONDS` | `120` | 限制每个重算子进程的运行时间。 |
| `SPREADSHEETBENCH_BADCASE_DIAGNOSTICS` | 正式评测用 `full`；训练用 `light` | 选择有边界的评测后证据采集模式。 |
| `SPREADSHEETBENCH_BADCASE_MAX_SCAN_CELLS` | `200000` | 限制完整诊断扫描的单元格数量。 |
| `SPREADSHEETBENCH_BADCASE_MAX_EXAMPLES` | `20` | 限制每类诊断结果序列化的样例数。 |
| `SPREADSHEETBENCH_LOG_LEVEL` | `error` | 隐藏常规 bridge/worker 日志，同时保留错误日志。 |
| `SPREADSHEETBENCH_TRAIN_SHUFFLE_SEED` | 如 `0` 的整数 | 仅对训练任务执行确定性乱序；设为 `off` 可关闭。 |
| `SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS` | `600` | 限制并行环境 reset、step 和 close 的等待时间。 |
| `SPREADSHEETBENCH_PHASE_HEARTBEAT_SECONDS` | `60` | 控制 rollout 阶段心跳的报告间隔。 |

另有 3 项安全限制是固定的实现常量，而非环境变量：每次计数型原生单元格修改最多 50,000
个单元格；模型单轮内这些操作累计最多尝试处理 100,000 个单元格；模型单轮最多接纳 4 个
工具调用。`run_python`、`delete_rows`、`delete_columns` 和 `manage_sheet` 不计入
单元格预算。

## 诊断与故障排查

每个正式评测目录都包含环境轨迹、`badcases.jsonl` 和 `badcase_summary.json`。
成对比较还会记录仅 Base 成功、仅候选模型成功的任务，以及每个任务的失败标签变化。
诊断标签不能替代官方工作簿结果。

SpreadsheetBench rollout 阶段以
`[spreadsheet-rollout] START|HEARTBEAT|END|ERROR` 形式记录，内容包括当前是训练还是
验证阶段、轮次、活动轨迹数和已用时间。`rollout_call` 是进程内的序列号，可据此区分连续的
验证和训练采样。Ray 环境调用还会记录 `[spreadsheet-ray]` 边界。
这些信息可以区分卡在 `env_reset`、模型 `generate_sequences` 还是 `env_step`。
Actor 超时时，会报告未返回调用的索引、Ray Actor ID、任务 ID 和动作。每个 Actor 还会在
环境 step 和评分前后记录 `[spreadsheet-worker]` 边界。bridge 会为 `run_python`、
原生读写工具、LibreOffice 输出/标准答案重算以及工作簿比较记录
`[spreadsheet-bridge]` 边界。

`run_python` 和 LibreOffice 在隔离的进程组中执行。发生超时时，整个进程组都会被终止并回收，
避免父进程超时后子进程继续占用输出管道、工作簿或 LibreOffice profile。

```bash
# 以下为默认值；工作负载需要更长时间时，请在提交前覆盖。
export SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS=600
export SPREADSHEETBENCH_PHASE_HEARTBEAT_SECONDS=60
export SPREADSHEETBENCH_MAX_RECALC_CALLS=1
export SPREADSHEETBENCH_RECALC_TIMEOUT_SECONDS=120
export SPREADSHEETBENCH_REQUIRE_VALIDATION_BEFORE_SUBMIT=true
```

Actor 超时会限制每次并行环境 `reset`、`step` 和 `close` 的等待时间。它不会终止卡住的
分布式模型生成；周期性的 `generate_sequences` 心跳可识别这种情况，随后应检查 Ray/NCCL
Worker 日志。这两个值都会记录在 `launch.log` 和 `run_manifest.json` 中。

## 获取 verl-agent（克隆并应用补丁）

verl-agent 不属于本仓库。若要复现实验使用的准确代码树，需要克隆固定提交的上游仓库，
再应用本项目补丁。可用以下单条命令完成：

```bash
bash rl/scripts/fetch_verl_agent.sh
```

该脚本会把 [verl-agent](https://github.com/langfengQ/verl-agent) 的
`796ed310287fa605c9292a0fce07a86d79fde05e` 提交克隆到被 git 忽略的
`third_party/verl-agent/`，并依次应用 `rl/integration/` 下的补丁，包括 EnvHarness
路由、追踪生命周期、SpreadsheetBench 运行时、诊断指标、仅验证 Actor 分配、原生工具指标和
SpreadsheetBench 加固指标。脚本具备幂等性：已完成补丁的代码树再次执行时不会重复修改。

如果希望手动操作 git，或将代码树放在其他位置（随后让 `$VERL_AGENT` 指向该目录），
等价步骤如下：

```bash
git clone https://github.com/langfengQ/verl-agent third_party/verl-agent
cd third_party/verl-agent
git checkout 796ed310287fa605c9292a0fce07a86d79fde05e
git apply ../../rl/integration/verl_agent_env_manager.patch
```

提供两种补丁，逐文件说明见 `rl/integration/ENVHARNESS_CHANGES.md`：

| 补丁 | 应用内容 | 使用场景 |
|---|---|---|
| `verl_agent_env_manager.patch` | 当前维护的 EnvHarness 环境路由 | 默认选择；足以支持 ALFWorld 和 SpreadsheetBench |
| `verl_agent_all_changes.patch` | 超集：环境路由及 DAPO / Qwen3-8B / webshop / SWE-Gym 适配 | 执行 `PATCH=all bash rl/scripts/fetch_verl_agent.sh`，或手动应用它来取代默认补丁 |

两种补丁只能选择一种；二者内容重叠，同时应用会失败。

## 其他 RL 适配器

仓库还包含一个 ALFWorld 适配器，它与上文介绍的 Spreadsheet Work Agent 路径相互独立。

```bash
# 双 GPU GRPO 冒烟：Qwen2.5-1.5B、2 个 epoch、6 个随附的变异游戏。
bash scripts/run_grpo.sh

# 未变异对照组（无语料，使用完整 TRAIN）
MUTATION_CORPUS= TRAIN_SUBSET_PATH= bash scripts/run_grpo.sh

# 8 GPU 完整运行（Qwen3-8B）
MODE=full bash scripts/run_grpo.sh
```

输出保存在 `runs/grpo_envrl_alfworld_<mode>_<ts>/`。`val_before_train` 会报告
各任务类型的成功率；每一步都会记录 `actor/pg_loss`、`episode/reward/mean` 和
`episode/success_rate`。

### ALFWorld 参数（环境变量）

| 变量 | 默认值 | 含义 |
|---|---|---|
| `MODE` | `smoke` | `smoke`（Qwen2.5-1.5B，2 GPU）或 `full`（Qwen3-8B，8 GPU） |
| `MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | 策略模型 |
| `MUTATION_CORPUS` | 随附的 `example_corpus.jsonl` | 每行包含 `{game_file, rules_code, in_env_actions}`；空值表示不做变异 |
| `ALFWORLD_DATA` | -- | ALFWorld 数据根目录；随附 JSONL 中的 `game_file` 使用相对该目录的路径 |
| `TRAIN_SUBSET_PATH` | 随附的 `train_subset.jsonl` | 将 TRAIN 限制为这些游戏；空值表示完整 TRAIN |
| `VERL_AGENT` | `third_party/verl-agent` | 要运行的 verl-agent（通过 `scripts/fetch_verl_agent.sh` 获取） |
| `VLLM_ATTENTION_BACKEND` | `FLASH_ATTN` | vLLM 0.11 保持使用 FLASH_ATTN（XFORMERS V1 要求 block_size 能被 256 整除） |
| `N_GPUS` / `TP` | `2 / 2`（smoke） | GPU 数量 / 张量并行度 |

## 文件索引

| 文件 | 用途 |
|---|---|
| `scripts/run_grpo.sh` | GRPO 启动脚本（smoke / full） |
| `scripts/fetch_verl_agent.sh` | 获取固定提交的 verl-agent，并应用环境路由补丁 |
| `scripts/smoke_worker.py` | 无 GPU 环境健全性检查 |
| `scripts/smoke_spreadsheetbench_worker.py` | 无 GPU 的 SpreadsheetBench reset/grade 健全性检查 |
| `scripts/mpi_ray_up.sh` / `scripts/mpi_ray_node.sh` | 启动外部 Ray 集群并检查就绪状态 |
| `scripts/submit_spreadsheetbench_ray_smoke.sh` | 通过 Ray Jobs 提交完整适配器冒烟测试 |
| `scripts/smoke_spreadsheetbench_ray.py` | 外部 Ray reset/run/submit/OJ 集成驱动程序 |
| `scripts/smoke_spreadsheetbench_loader_scale.py` | 不加载模型的 128 Actor Spreadsheet-RL reset/close 规模检查 |
| `scripts/submit_spreadsheetbench_grpo.sh` | 向已就绪的 Ray 集群提交 SpreadsheetBench GRPO |
| `scripts/run_spreadsheetbench_grpo.sh` | 构造并执行 verl-agent GRPO 命令 |
| `scripts/prepare_spreadsheetbench_verl_data.py` | 为 verl-agent 生成离线文本 parquet 占位数据 |
| `scripts/build_corpus.py` | 从旧语料构建示例语料和子集 |
| `envharness_rl/alfworld/envs.py` | 基于 Ray Actor 并行执行的 `AlfworldEnv` 和 `Rules` Worker |
| `envharness_rl/alfworld/projection.py` | 提取 `<action>`/`<think>` 并规范化为可执行命令 |
| `envharness_rl/spreadsheetbench/` | SpreadsheetBench 投影、Worker 和 verl-agent Manager |
| `experiments/alfworld/data/` | `example_corpus.jsonl`（6 个变异游戏）及 `train_subset.jsonl` |
| `../third_party/verl-agent/` | 获取的 verl-agent（git 忽略；上游代码及当前维护的 EnvHarness 路由） |
| `integration/ENVHARNESS_CHANGES.md` | 相对上游代码的差异说明 |
| `integration/verl_agent_env_manager.patch` | 获取脚本应用的当前环境路由补丁 |
| `integration/verl_agent_all_changes.patch` | 完整补丁（重新启用 DAPO / Qwen3-8B / webshop / SWE-Gym） |

## 致谢

RL 实验基于第三方 [**verl-agent**](https://github.com/langfengQ/verl-agent)
仓库（GiGPO，Apache-2.0）上游提交 `796ed31` 运行。该依赖由
`scripts/fetch_verl_agent.sh` 获取，本仓库不重新分发。RL 训练与 rollout 基础设施归属于
verl-agent；本项目仅添加 EnvHarness 环境路由，详见
`integration/ENVHARNESS_CHANGES.md`。verl-agent 又构建于
[verl](https://github.com/volcengine/verl) 之上，使用时请按要求引用并致谢。
