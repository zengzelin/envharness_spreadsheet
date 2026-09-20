# Spreadsheet-RL Loader 优化实施计划

> **执行要求：** 实施时按任务顺序推进，每个任务先补失败测试，再写最小实现并验证。本文只定义修改方案，当前尚未修改 loader 代码。

**目标：** 消除 Spreadsheet-RL 训练启动阶段 128 个 Ray actor 并发全量扫描 5,925 个任务目录导致的 reset 超时，同时保持任务选择、group rollout、显式 `instance_id` 和现有全量枚举接口的行为兼容。

**总体方案：** 将“读取 parquet 索引”和“把一行物化为 `SBTask`”拆开。训练 reset 先用 seed/instance id 选中一行，再只访问该任务的 `instruction.json`、`output.xlsx` 和 `target.xlsx`；全量 `list_tasks` 仍保留显式物化全部任务的兼容接口。第一阶段不引入新服务或数据库，只有在 128 actor 压测仍不达标时才升级为 driver 预加载并通过 Ray object store 广播 manifest。

**涉及技术：** Python、pandas/pyarrow parquet、`functools.lru_cache`、Ray actors、pytest、SpreadsheetBench bridge。

## 1. 现象与证据

失败实验：`runs/grpo_spreadsheetbench_full_20260918_004429`。

- 初始 validation 使用 32 个环境，reset 约 13.4 秒完成，success rate 为 0.28125；
- 进入训练后创建 128 个环境，即 `TRAIN_BS=16 * rollout.n=8`；
- reset 在 600 秒后超时，`ready=0/128`，尚未产生训练 rollout；
- 因此故障发生在环境 reset/数据加载阶段，不是模型生成、PPO 更新、GPU OOM 或 evaluator 打分阶段。

当前调用链为：

```text
verl rollout
  -> SpreadsheetBenchEnvironmentManager.reset()
  -> EnvharnessSpreadsheetEnvs.reset()
  -> 128 个 Ray SpreadsheetBenchWorker.reset()
  -> SpreadsheetBenchEnv.reset()
  -> select_task(data_path, split_file, seed, instance_id)
  -> load_spreadsheet_rl_dataset()
```

当前 `load_spreadsheet_rl_dataset()` 虽然使用了 `lru_cache`，但缓存只在单个 actor 进程内有效。每个新 actor 第一次 reset 都会读取 parquet 的 5,925 行，并逐行访问任务目录中的 `instruction.json`、`output.xlsx`、`target.xlsx` 后构造完整 `SBTask` 列表。128 actor 最坏会触发约 `128 * 5925 = 758,400` 次任务元数据访问，形成 Ceph 小文件并发风暴。

Spreadsheet-RL 原生训练路径不同：parquet 样本由 verl 数据层流入，每条 rollout 已携带该样本的 `extra_info` 和 `reward_model.ground_truth`，环境不会在每个 rollout actor 内重新扫描整份数据集。这是当前 envharness loader 需要补齐的关键差异。

## 2. 修改边界

本轮只优化 loader 和 reset 可观测性，不改变以下语义：

- seed 到任务下标的映射；
- `rollout.n` 同组任务重复方式；
- `task_stride` / `task_offset` 行为；
- `extra_info.id`、`instruction.json.id`、ground truth basename、`row-N` 的 task id 优先级；
- `output.xlsx` 作为初始工作簿、`target.xlsx` 作为 reward target；
- evaluator、工具集合、prompt、projection 和 reward 计算；
- `load_spreadsheet_rl_dataset()` 的全量枚举能力。

## 3. Loader 接口设计

### 3.1 缓存 parquet 原始行

在 `envharness/bridges/spreadsheetbench/dataset.py` 新增：

```python
@lru_cache(maxsize=8)
def _load_spreadsheet_rl_rows(
    data_path: str,
    split_file: str,
) -> tuple[dict[str, object], ...]:
    ...
```

职责：

- 将 `data_path` 和 `split_file` 规范化为绝对路径后作为 cache key；
- 只读取 parquet 并转换为不可变行序列；
- 不打开任务目录，不检查 xlsx，不读取 `instruction.json`；
- 对 parquet 缺列、空 split 和无法解析的结构给出包含 split 路径的异常。

缓存仍是进程级，但每个 actor 只做一次 parquet 顺序读，不再做 5,925 次小文件访问。这个阶段先解决主要瓶颈，不提前引入跨进程缓存复杂度。

### 3.2 单行物化任务

新增：

```python
def _spreadsheet_rl_task_from_row(
    data_path: str,
    row: Mapping[str, object],
    row_index: int,
) -> SBTask:
    ...
```

职责：

- 从当前行提取 `extra_info` 和 `reward_model.ground_truth`；
- 定位唯一任务目录；
- 仅读取该目录的 `instruction.json`；
- 校验当前任务的 `output.xlsx` 和 `target.xlsx`；
- 按现有优先级构造 task id；
- 返回字段与当前 `SBTask` 完全一致。

错误必须包含 `split row index`、候选 task id 和实际路径，避免只得到笼统的 `FileNotFoundError`。

### 3.3 懒选择入口

新增：

```python
def select_spreadsheet_rl_task(
    data_path: str,
    split_file: str,
    seed: int,
    instance_id: str | None = None,
) -> tuple[SBTask, int]:
    ...
```

无 `instance_id` 时：

1. 调用 `_load_spreadsheet_rl_rows()`；
2. 用当前 `select_task()` 完全相同的规则计算 row index；
3. 只调用一次 `_spreadsheet_rl_task_from_row()`；
4. 返回选中的 task 和 index。

`select_task()` 检测到 `split_file` 后直接路由到该入口，不再先调用全量 loader。

### 3.4 保留全量 loader

`load_spreadsheet_rl_dataset()` 保留原签名，改为：

```python
rows = _load_spreadsheet_rl_rows(...)
return tuple(
    _spreadsheet_rl_task_from_row(data_path, row, index)
    for index, row in enumerate(rows)
)
```

它只供离线检查、`list_tasks` 和兼容调用使用。训练 reset 不得再经过该函数。

## 4. `instance_id` 处理

显式任务评测通常使用 `instance_id`，应兼顾正确性和速度：

1. 先匹配 parquet `extra_info.id`；
2. 再匹配 `reward_model.ground_truth` 的目录 basename/规范化相对路径；
3. 只有前两种无法命中时，才允许逐行读取 `instruction.json` 做兼容回退；
4. 多条匹配时直接报歧义错误，不静默选择第一条；
5. 未匹配错误中打印 split、instance id 和总行数。

训练 seed 选择不会进入慢回退。慢回退主要服务旧评测调用，后续可根据日志决定是否删除。

## 5. Reset 阶段诊断

在以下位置增加结构化阶段日志，不改变返回值：

- `rl/envharness_rl/spreadsheetbench/envs.py`：worker reset 的 START/END/ERROR，记录 actor id、seed、instance id、elapsed；
- `envharness/bridges/spreadsheetbench/bridge.py`：`select_task`、`prepare_sandbox`、`build_preview` 分阶段耗时；
- loader：仅记录 split 首次读取耗时、行数和选中 row index，禁止每行打印。

日志目标是区分 parquet 慢、任务目录慢、复制工作簿慢和 preview 慢，避免下一次只能看到 Ray 全局 timeout。

## 6. 文件修改清单

### Task 1：拆分 loader 并保持兼容

**文件：**

- 修改：`envharness/bridges/spreadsheetbench/dataset.py`
- 测试：`rl/tests/test_spreadsheetbench_dataset.py`

**产出接口：**

- `_load_spreadsheet_rl_rows(data_path, split_file)`
- `_spreadsheet_rl_task_from_row(data_path, row, row_index)`
- `select_spreadsheet_rl_task(data_path, split_file, seed, instance_id=None)`
- 保持 `load_spreadsheet_rl_dataset()` 和 `select_task()` 向后兼容

实施步骤：

- [ ] 增加“未选中行的 `instruction.json` 损坏时，选择其他行仍成功”的失败测试；旧实现应失败。
- [ ] 增加 parquet 原始行缓存测试，重复选择不同 seed 时 parquet reader 只调用一次。
- [ ] 增加 seed modulo、group repeat、stride/offset 回归测试。
- [ ] 增加“未选中任务缺文件不失败，选中任务缺文件明确失败”的测试。
- [ ] 实现原始行读取、单行物化和懒选择。
- [ ] 将 `select_task()` 的 split 分支切到懒选择。
- [ ] 用新原语重写全量 loader，保证现有调用结果不变。
- [ ] 在测试 teardown 中同时清理 raw-row cache 和 full-dataset cache。

验证命令：

```bash
PYTHONPATH=.:rl:third_party/verl-agent \
python -m pytest -q rl/tests/test_spreadsheetbench_dataset.py
```

### Task 2：完善显式任务选择

**文件：**

- 修改：`envharness/bridges/spreadsheetbench/dataset.py`
- 测试：`rl/tests/test_spreadsheetbench_dataset.py`

实施步骤：

- [ ] 为 `extra_info.id` 精确匹配写失败测试。
- [ ] 为 ground-truth basename/相对路径匹配写失败测试。
- [ ] 为 metadata id 慢回退写兼容测试。
- [ ] 为重复 id 和未知 id 写错误信息测试。
- [ ] 实现先快后慢的匹配顺序。
- [ ] 运行 dataset 测试，确认 task id 优先级没有变化。

### Task 3：补 reset 分阶段日志

**文件：**

- 修改：`envharness/bridges/spreadsheetbench/bridge.py`
- 修改：`rl/envharness_rl/spreadsheetbench/envs.py`
- 测试：`rl/tests/test_spreadsheetbench_envs.py`
- 测试：`rl/tests/test_spreadsheetbench_bridge_execution.py`

实施步骤：

- [ ] 写日志字段和异常上下文测试。
- [ ] 在 worker reset 增加 START/END/ERROR。
- [ ] 在 bridge reset 增加 `select_task`、`prepare_sandbox`、`build_preview` 阶段计时。
- [ ] 确保异常保留原 traceback，日志不能吞异常。
- [ ] 运行 envs 和 bridge execution 测试。

### Task 4：增加 128 actor loader 压测脚本

**文件：**

- 新增：`rl/scripts/smoke_spreadsheetbench_loader_scale.py`
- 修改：`rl/tests/test_spreadsheetbench_training_scripts.py`
- 修改：`rl/README.md`

脚本行为：

- 连接已启动的外部 Ray 集群；
- 使用 Spreadsheet-RL `train_hermes.parquet`；
- 创建 `env_num=16`、`group_n=8`，共 128 个 actor；
- 只执行 reset、校验 task grouping、close，不加载模型、不训练；
- 输出总耗时、成功数、唯一任务数和最慢阶段；
- 任一 actor 超时或 task grouping 错误时以非零状态退出。

验收标准：

- 128/128 actor 在 `SPREADSHEETBENCH_RAY_GET_TIMEOUT=180` 内完成；
- 产生 16 个唯一任务，每个任务连续重复 8 次；
- 无 actor crash、无 reset timeout；
- 相同 seed 重跑得到相同 task id 顺序。

## 7. 分层验证流程

先运行目标测试：

```bash
PYTHONPATH=.:rl:third_party/verl-agent \
python -m pytest -q \
  rl/tests/test_spreadsheetbench_dataset.py \
  rl/tests/test_spreadsheetbench_envs.py \
  rl/tests/test_spreadsheetbench_bridge_execution.py
```

再运行完整 RL 测试：

```bash
PYTHONPATH=.:rl:third_party/verl-agent \
python -m pytest -q rl/tests
```

然后执行三层运行验证：

1. 单进程分别 reset 两个 seed，确认只物化两个任务；
2. 外部 Ray 上跑 128 actor loader scale smoke；
3. 用正式 Spreadsheet-RL 配置跑 5 个训练 step，确认能生成 train rollout、reward 和 checkpoint。

只有三层都通过后，才恢复 371-step 正式训练。不能直接用完整训练来验证 loader 修复，因为模型启动成本高且会掩盖 reset 问题。

## 8. 升级条件与备选方案

第一阶段完成后，如果 128 actor reset 仍超过 180 秒，再实施第二阶段：

- driver 进程读取一次 parquet 并生成轻量 manifest；
- 使用 Ray object store 广播 manifest；
- actor reset 只接收所选行或 manifest 引用；
- 不广播 workbook bytes，xlsx 仍从共享存储按任务读取。

只有在第二阶段仍受 Ceph 元数据限制时，才考虑离线生成 SQLite/JSONL manifest。当前不建议直接使用数据库方案，因为会增加数据构建、版本一致性和部署维护成本，且尚无证据表明单行懒物化不够。

## 9. 当前进展

- [x] 定位 `20260918_004429` 的失败阶段和调用链。
- [x] 确认 32-env validation 正常、128-env train reset 超时。
- [x] 确认主要瓶颈是 actor 内全量物化，而非 evaluator 或 GPU。
- [x] 完成 lazy row selection 的接口和兼容性设计。
- [x] 完成测试矩阵、压测标准和升级条件设计。
- [x] 编写 loader 懒物化、缓存、显式 ID 和分阶段日志回归测试。
- [x] 实现 loader 拆分与单行懒物化。
- [x] 增加 reset 分阶段诊断。
- [x] 新增 128 actor scale smoke 脚本和运行文档。
- [ ] 在训练镜像中运行完整 `rl/tests`（当前编辑节点没有项目 Python/pytest）。
- [ ] 实际执行 128 actor scale smoke 并达到 180 秒验收线。
- [ ] 完成 5-step Spreadsheet-RL 训练验证。

### 2026-09-18 实施记录

已修改：

- `envharness/bridges/spreadsheetbench/dataset.py`：新增 parquet row cache、单行
  `SBTask` 物化、懒选择和 `instance_id` 快速匹配；训练 reset 不再调用全量 loader；
- `envharness/bridges/spreadsheetbench/bridge.py`：reset 增加 `select_task`、
  `prepare_sandbox`、`build_preview` 三段计时；
- `rl/envharness_rl/spreadsheetbench/envs.py`：worker reset 增加 seed、worker index、
  task id 和异常边界日志；
- `rl/scripts/smoke_spreadsheetbench_loader_scale.py`：新增默认 16 prompts x 8 rollouts
  的 128 actor reset/close 压测；
- `rl/tests/` 和 `rl/README.md`：补充回归覆盖及压测命令。

当前编辑节点只有 Python 3.6，且未安装 pytest，无法执行本项目要求的 Python 3.11+
测试。代码验证必须在训练镜像中完成；未通过测试和 scale smoke 前，实施状态仍视为
“代码完成、运行验收未完成”。

### 2026-09-18 Ray worker 导入修复

第一次直接运行 scale smoke 时，driver 可以导入 `envharness_rl`，但远端 Ray actor
创建失败并报告 `ModuleNotFoundError: No module named 'envharness_rl'`。原因是命令行前缀
中的 `PYTHONPATH=.:rl:third_party/verl-agent` 只进入当前 driver；连接到已经运行的外部
Ray 集群后，raylet 创建的 worker 不会自动继承 driver 的临时 shell 路径。

scale smoke 现已与正式 Ray Jobs 提交脚本对齐：根据脚本位置生成仓库根目录、`rl` 和
`third_party/verl-agent` 的绝对路径，并通过 `ray.init(runtime_env.env_vars.PYTHONPATH)`
传播给所有节点。重跑时不需要重启 Ray 集群。

第二次运行已证明远端 actor 可以导入代码，但 actor 构造收到空 `data_path`。这是压测
脚本参数装配遗漏：`EnvharnessSpreadsheetEnvs` 从 `env_kwargs["data_path"]` 取数据根目录，
而脚本只传了 `split_file` 和 `max_steps`。现已补传 `data_path`；同时调整 cleanup，确保
reset 已失败时，actor close 的二次异常只作为附加日志，不能覆盖首个根因 traceback。
