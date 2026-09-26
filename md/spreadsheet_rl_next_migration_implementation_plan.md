# Spreadsheet-RL 下一阶段迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不替换现有 EnvHarness + verl-agent 训练主线的前提下，依次实现完整验评分批、`fill_formula`、单轮多工具调度和剩余高价值表格工具。

**Architecture:** 保留现有 Ray actor-per-environment 和 external environment manager。验评层将“数据集大小”与“actor 并发数”解耦；工具层先增加可原子提交的公式填充，再将 projection 从单 `Action` 扩展为有界 `ActionBatch`。每个里程碑单独 A/B 验证，不同时改 reward 数值、学习率或基座模型。

**Tech Stack:** Python 3.11、Pydantic、openpyxl、LibreOffice headless、Ray、verl-agent、pytest、Bash、Parquet/datasets。

**Spec:** `md/spreadsheet_rl_vs_envharness_comparison.md`

## Global Constraints

- 固定基座模型为 `Qwen3-4B-Thinking-2507`，不在本计划中切换模型。
- 当前运行中的 Ray job 使用提交时代码快照；新代码只用于新 job。
- 保留 `run_python`、LibreOffice evaluator、checkpoint、W&B、TensorBoard 和 env/verl 双轨迹。
- 默认训练为 `train_hermes.parquet`，主验评为本地可用的 399 条 `test_verified_hermes.parquet`。
- 写工具必须共用 workbook lock，经过临时文件校验后用 `os.replace()` 原子提交。
- 任何新工具失败都不得覆盖上一个有效 workbook。
- 完整验评以 `success_rate` 为主指标；parser、penalty 和 shaped reward 只作诊断。
- 每个里程碑先跑 `rl/tests`，再跑 worker smoke，最后启动小步数 GRPO。

## Review Focus

- 399 不能被 64 整除：最后 15 条必须只激活 15 个 worker，且无重复无遗漏。
- train 仍需按 seed stride 轮换：val 必须按 parquet `task_index` 固定，两种 reset 语义不能混用。
- 公式平移必须保留 `$A$1`、`A$1`、`$A1` 和带空格的跨 sheet 引用。
- multi-call 中写后读、写失败、`submit` 非末位和超过最大调用数必须有确定行为。
- Ray actor 超时、LibreOffice 失败或 workbook 损坏时，轨迹必须保留 task ID、action index 和原始错误。

---

## Milestone 0: 固化当前对照实验

### Task 0: 封存当前训练基线

**Files:**
- Modify: `md/spreadsheet_rl_vs_envharness_comparison.md`
- Modify: `md/grpo_stabilization_plan.md`

**Interfaces:**
- Consumes: 当前 run 的 `run_manifest.json`、`train.log`、`ckpts/global_step_*`。
- Produces: 新迁移实验的固定对照参数和 checkpoint 选择记录。

- [ ] **Step 1: 等待当前 run 生成 `global_step_20` checkpoint**

Run:

```bash
rg "step:20 -" runs/grpo_spreadsheetbench_diagnostic_20260919_215017/train.log
test -d runs/grpo_spreadsheetbench_diagnostic_20260919_215017/ckpts/global_step_20
```

Expected: 日志存在 step 20 指标，checkpoint 目录存在。

- [ ] **Step 2: 对 step 12 和 step 20 运行相同 399 条 val-only 评测**

Expected: 两次评测 task ID 集合完全相同，均有 399 条非重复终局轨迹。

- [ ] **Step 3: 在两份文档写入 Base/step12/step20 的成功数、parser 错误率、Python 错误率和每任务 wall time**

- [ ] **Step 4: 检查文档差异**

Run: `git diff --check -- md/`

Expected: exit code 0。

- [ ] **Step 5: 提交基线记录**

```bash
git add md/spreadsheet_rl_vs_envharness_comparison.md md/grpo_stabilization_plan.md
git commit -m "docs: freeze spreadsheet migration baseline"
```

---

## Milestone 1: 完整验证分批

### Task 1: 在 placeholder parquet 中传递任务索引

**Files:**
- Modify: `rl/scripts/prepare_spreadsheetbench_verl_data.py`
- Modify: `rl/tests/test_spreadsheetbench_training_scripts.py`

**Interfaces:**
- Produces: `build_rows(split: str, size: int) -> list[dict[str, Any]]`，每行包含 `env_kwargs={"split": split, "task_index": index}`。
- Consumes: verl-agent 已有 `gen_batch.non_tensor_batch["env_kwargs"]` 通道。

- [ ] **Step 1: 先修改 placeholder row 单测，要求顶层 `env_kwargs` 存在**

```python
assert rows[1]["env_kwargs"] == {"split": "test", "task_index": 1}
```

- [ ] **Step 2: 运行单测并确认失败**

Run: `PYTHONPATH=.:rl:third_party/verl-agent python -m pytest -q rl/tests/test_spreadsheetbench_training_scripts.py::test_placeholder_rows_are_offline_text_agent_examples`

Expected: FAIL，差异为缺少 `env_kwargs`。

- [ ] **Step 3: 在 `build_rows()` 加入结构化索引**

```python
"env_kwargs": {"split": split, "task_index": index},
```

- [ ] **Step 4: 运行该测试和数据脚本测试集**

Run: `PYTHONPATH=.:rl:third_party/verl-agent python -m pytest -q rl/tests/test_spreadsheetbench_training_scripts.py`

Expected: PASS。

- [ ] **Step 5: 提交 parquet metadata 改动**

```bash
git add rl/scripts/prepare_spreadsheetbench_verl_data.py rl/tests/test_spreadsheetbench_training_scripts.py
git commit -m "feat: carry spreadsheet task indexes in verl data"
```

### Task 2: 使 validation worker pool 支持变长 batch

**Files:**
- Modify: `rl/envharness_rl/spreadsheetbench/envs.py`
- Modify: `rl/envharness_rl/spreadsheetbench/manager.py`
- Test: `rl/tests/test_spreadsheetbench_envs.py`
- Test: `rl/tests/test_spreadsheetbench_verl_manager.py`

**Interfaces:**
- Produces: `EnvharnessSpreadsheetWorker.reset(task_seed: int | None = None)`。
- Produces: `EnvharnessSpreadsheetEnvs.reset(task_seeds: list[int] | None = None)`。
- Produces: `SpreadsheetBenchEnvironmentManager._task_seeds_from_kwargs(kwargs, batch_size) -> list[int] | None`。
- Invariant: `step()` 只向上一次 reset 激活的 worker 发送 action。

- [ ] **Step 1: 增加 64/15 变长 reset 失败测试**

```python
envs.reset(task_seeds=list(range(64)))
assert len(envs._active_workers) == 64
envs.reset(task_seeds=list(range(384, 399)))
assert len(envs._active_workers) == 15
```

- [ ] **Step 2: 增加 manager kwargs 解码测试**

```python
kwargs = np.asarray([
    {"split": "test", "task_index": 384},
    {"split": "test", "task_index": 385},
], dtype=object)
assert manager._task_seeds_from_kwargs(kwargs, 2) == [384, 385]
```

- [ ] **Step 3: 运行两组测试并确认现有 API 无法通过**

Run: `PYTHONPATH=.:rl:third_party/verl-agent python -m pytest -q rl/tests/test_spreadsheetbench_envs.py rl/tests/test_spreadsheetbench_verl_manager.py`

Expected: FAIL，失败点为 reset 不接受 task seeds 和 manager 丢弃 kwargs。

- [ ] **Step 4: 实现 worker 的显式 seed reset**

```python
def reset(self, task_seed: int | None = None):
    reset_seed = self._next_seed if task_seed is None else int(task_seed)
    ...
    if task_seed is None and self._advance_seed:
        self._next_seed += self._seed_stride
```

- [ ] **Step 5: 实现可复用 worker pool 和 `_active_workers`**

`reset(task_seeds=None)` 激活全部 worker；显式 seeds 时激活前 `len(task_seeds)` 个。`step()`、task ID 诊断和 `close()` 均使用正确的 active/pool 集合。

- [ ] **Step 6: 让 manager reset 转发 val task index**

train split 忽略 parquet index，保持 seed stride；val split 必须传入当前 dataloader batch 的 task indexes。

- [ ] **Step 7: 运行两组测试**

Expected: PASS，且现有 train seed grouping 测试仍通过。

- [ ] **Step 8: 提交 worker pool 改动**

```bash
git add rl/envharness_rl/spreadsheetbench/envs.py rl/envharness_rl/spreadsheetbench/manager.py rl/tests/test_spreadsheetbench_envs.py rl/tests/test_spreadsheetbench_verl_manager.py
git commit -m "feat: batch full spreadsheet validation through actor pool"
```

### Task 3: 解耦验证规模与并发数

**Files:**
- Modify: `rl/scripts/run_spreadsheetbench_grpo.sh`
- Modify: `rl/scripts/submit_spreadsheetbench_grpo.sh`
- Modify: `rl/tests/test_spreadsheetbench_training_scripts.py`
- Modify: `rl/README.md`

**Interfaces:**
- Produces: `VAL_SIZE`，默认等于旧 `VAL_BS`。
- Produces: `VAL_CONCURRENCY`，默认等于旧 `VAL_BS`。
- Backward compatibility: 未设新参数时，旧启动命令行为不变。

- [ ] **Step 1: 增加 dry-run 测试**

```python
env.update({"VAL_SIZE": "399", "VAL_CONCURRENCY": "64"})
assert "data.val_batch_size=64" in completed.stdout
assert "val_size=399 val_concurrency=64" in completed.stdout
```

- [ ] **Step 2: 修改两个启动脚本的验评参数、export、runtime env 和 manifest 字段**

`prepare_spreadsheetbench_verl_data.py --val-size` 使用 `VAL_SIZE`；Hydra `data.val_batch_size` 使用 `VAL_CONCURRENCY`。

- [ ] **Step 3: 运行 dry-run 测试和全部 `rl/tests`**

Run: `PYTHONPATH=.:rl:third_party/verl-agent python -m pytest -q rl/tests`

Expected: PASS。

- [ ] **Step 4: 运行 130 条三 batch smoke**

Use: `VAL_SIZE=130 VAL_CONCURRENCY=64`。

Expected: batch sizes 为 64、64、2；130 个 task index 每个出现一次。

- [ ] **Step 5: 运行 399 条 val-only 验收**

Use: `VAL_SIZE=399 VAL_CONCURRENCY=64 VAL_BEFORE=True EXTRA_HYDRA="trainer.val_only=True"`。

Expected: `64 * 6 + 15 = 399`，峰值 actor 数不超过 64，指标与旧独立 399 评测在同一 checkpoint 上一致。

- [ ] **Step 6: 提交验评解耦**

```bash
git add rl/scripts/run_spreadsheetbench_grpo.sh rl/scripts/submit_spreadsheetbench_grpo.sh rl/tests/test_spreadsheetbench_training_scripts.py rl/README.md
git commit -m "feat: decouple spreadsheet validation size and concurrency"
```

---

## Milestone 2: `fill_formula`

### Task 4: 定义公式工具协议和 projection

**Files:**
- Modify: `envharness/bridges/spreadsheetbench/tools.py`
- Modify: `envharness/bridges/spreadsheetbench/read_tools.py`
- Modify: `rl/envharness_rl/spreadsheetbench/projection.py`
- Modify: `rl/envharness_rl/spreadsheetbench/manager.py`
- Test: `rl/tests/test_spreadsheetbench_projection.py`
- Test: `rl/tests/test_spreadsheetbench_verl_manager.py`

**Interfaces:**
- Produces: `Action(name="fill_formula", kwargs={start_cell, formula_template, end_row?, end_col?, sheet_name?})`。
- Constraint: `formula_template` 必须以 `=` 开头，`end_row >= start_row`，`end_col >= start_col`。

- [ ] **Step 1: 写 projection 失败测试**

覆盖有效调用、缺公式、非 `=` 公式、反向范围和 tool-set disabled。

- [ ] **Step 2: 运行 projection/manager 测试并确认失败**

- [ ] **Step 3: 增加 tool schema、允许集合、prompt 说明和 `tool/fill_formula` 指标键**

- [ ] **Step 4: 运行 projection/manager 测试**

Expected: PASS。

- [ ] **Step 5: 提交协议层**

```bash
git add envharness/bridges/spreadsheetbench/tools.py envharness/bridges/spreadsheetbench/read_tools.py rl/envharness_rl/spreadsheetbench/projection.py rl/envharness_rl/spreadsheetbench/manager.py rl/tests/test_spreadsheetbench_projection.py rl/tests/test_spreadsheetbench_verl_manager.py
git commit -m "feat: define spreadsheet formula fill tool"
```

### Task 5: 实现公式平移和原子提交

**Files:**
- Modify: `envharness/bridges/spreadsheetbench/write_tools.py`
- Modify: `envharness/bridges/spreadsheetbench/bridge.py`
- Test: `rl/tests/test_spreadsheetbench_bridge_execution.py`

**Interfaces:**
- Produces: `execute_fill_formula(output_path: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], str]`。
- Uses: `openpyxl.formula.translate.Translator(formula, origin=start.coordinate).translate_formula(target.coordinate)`。

- [ ] **Step 1: 写公式平移测试**

覆盖相对引用、混合/绝对引用、跨 sheet 引用、单格、矩形、合并单元格占位和 300,000-cell 上限。

- [ ] **Step 2: 写回滚测试**

模拟保存失败和重新打开校验失败，断言原 workbook hash 不变。

- [ ] **Step 3: 实现有界公式平移**

使用现有 lock/tempfile/atomic replace 路径；不在每次工具调用后启动 LibreOffice，终局评分时统一重算。

- [ ] **Step 4: 在 bridge dispatch 注册 `fill_formula` 并返回 changed count/sample**

- [ ] **Step 5: 运行 bridge、projection 和全量单测**

Run: `PYTHONPATH=.:rl:third_party/verl-agent python -m pytest -q rl/tests`

Expected: PASS。

- [ ] **Step 6: 运行 5-step 固定 seed A/B**

A 使用 `native_basic`；B 使用包含 `fill_formula` 的新 tool set。其余参数和 task ID 必须一致。

Gate: `write_tool_success_rate` 不低于 A，公式被 `write_range` 拒绝的次数下降，无 workbook 损坏。

- [ ] **Step 7: 提交公式工具实现**

```bash
git add envharness/bridges/spreadsheetbench/write_tools.py envharness/bridges/spreadsheetbench/bridge.py rl/tests/test_spreadsheetbench_bridge_execution.py
git commit -m "feat: fill spreadsheet formulas atomically"
```

---

## Milestone 3: 单轮 multi-call

### Task 6: 将 projection 扩展为有界 action batch

**Files:**
- Modify: `rl/envharness_rl/spreadsheetbench/projection.py`
- Test: `rl/tests/test_spreadsheetbench_projection.py`

**Interfaces:**
- Produces: `project_action_batch(model_output: str, max_calls: int = 4) -> tuple[list[Action], list[dict[str, Any]]]`。
- Compatibility: 原 `envharness_spreadsheetbench_projection()` 保留为单 action wrapper，直到 manager 迁移完成。

- [ ] **Step 1: 写 parser 测试**

覆盖 1/2/4 个 native tool calls、fenced JSON 单调用恢复、5 个调用超限、中间 invalid JSON、`submit` 非末位。

- [ ] **Step 2: 确认新测试失败**

- [ ] **Step 3: 提取所有完整 `<tool_call>` block，逐个复用现有 payload/action 校验**

Rules: 最多 4 个；`submit` 必须最后；批次中任何 parse error 保留其 call index。

- [ ] **Step 4: 运行 projection 测试**

Expected: PASS，现有宽松单调用用例不回归。

- [ ] **Step 5: 提交 batch parser**

```bash
git add rl/envharness_rl/spreadsheetbench/projection.py rl/tests/test_spreadsheetbench_projection.py
git commit -m "feat: parse bounded spreadsheet tool batches"
```

### Task 7: 在一个 episode turn 内执行 action batch

**Files:**
- Modify: `envharness/bridges/spreadsheetbench/bridge.py`
- Modify: `rl/envharness_rl/spreadsheetbench/envs.py`
- Modify: `rl/envharness_rl/spreadsheetbench/manager.py`
- Test: `rl/tests/test_spreadsheetbench_bridge_execution.py`
- Test: `rl/tests/test_spreadsheetbench_envs.py`
- Test: `rl/tests/test_spreadsheetbench_verl_manager.py`

**Interfaces:**
- Produces: `EnvharnessSpreadsheetWorker.step_many(actions: list[Action])` 。
- Produces: batch info 中的 `tool_results`、`tool_call_count`、`tool_success_count`、`tool_failure_count`。
- Turn semantics: 整个 batch 只增加一次 `_episode_steps`。

- [ ] **Step 1: 写执行顺序测试**

覆盖 read+read、read+write+read、write 失败后停止后续 mutation、`submit` 终止、`run_python` 与 native write 串行。

- [ ] **Step 2: 确认新测试失败**

- [ ] **Step 3: 先实现全部按输出顺序串行的 `step_many()`**

首个版本不引入线程并行，先固定语义。对每个子调用记录 START/END/ERROR 和 `action_index`。

- [ ] **Step 4: 聚合 observation 和 reward**

工具结果按 call index 生成有界 JSON/text；reward 求和；任意 action 终止后不再执行后续 action。

- [ ] **Step 5: manager 记录一个 model output 对应的 action list 和逐调用 diagnostics**

- [ ] **Step 6: 运行 bridge/env/manager 测试和全部 `rl/tests`**

Expected: PASS。

- [ ] **Step 7: 在串行语义稳定后，仅对连续只读 action 组使用有界线程池**

Gate: 并行与串行版的 tool result 完全一致，同一 batch 中不允许读取越过前序 write。

- [ ] **Step 8: 提交 multi-call 执行层**

```bash
git add envharness/bridges/spreadsheetbench/bridge.py rl/envharness_rl/spreadsheetbench/envs.py rl/envharness_rl/spreadsheetbench/manager.py rl/tests/test_spreadsheetbench_bridge_execution.py rl/tests/test_spreadsheetbench_envs.py rl/tests/test_spreadsheetbench_verl_manager.py
git commit -m "feat: execute spreadsheet multi-call turns"
```

### Task 8: 增加 multi-call 训练指标和 A/B 验收

**Files:**
- Modify: `third_party/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py`
- Modify: `third_party/verl-agent/verl/trainer/ppo/ray_trainer.py`
- Modify: `rl/envharness_rl/spreadsheetbench/rollout_summary.py`
- Modify: `rl/scripts/summarize_spreadsheetbench_rollouts.py`
- Modify: `rl/tests/test_spreadsheetbench_training_scripts.py`
- Modify: `rl/tests/test_spreadsheetbench_rollout_summary.py`

**Interfaces:**
- Produces metrics: `episode/tool_calls_per_turn`、`episode/multi_call_ratio`、`env/multi_call_partial_failure_rate`。

- [x] **Step 1: 写 trainer/source 导出和离线 summary 测试**
- [x] **Step 2: 实现 batch field 传递、train/val aggregate 和 trajectory summary**
- [ ] **Step 3: 运行全部 `rl/tests`**
- [ ] **Step 4: 运行单 worker multi-call smoke，验证一轮两个 read 只消耗一个 turn**
- [ ] **Step 5: 运行 5-step 固定 task A/B**

Gate: 平均 episode length 相对单 call 版下降，目标不高于 7.5；固定 64 成功率不下降；单步 wall time 下降或持平。

- [ ] **Step 6: 提交指标与验收记录**

2026-09-24 补充：代码侧已进一步区分 projected/executed/skipped、completed、all-success、
short-circuit、首个失败位置和停止原因，并支持逐 turn JSONL badcase 导出。Step 3/4/5
仍必须在 Python 3.11 训练镜像和真实 Ray 环境完成，不能以静态检查替代。

```bash
git add third_party/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py third_party/verl-agent/verl/trainer/ppo/ray_trainer.py rl/envharness_rl/spreadsheetbench/rollout_summary.py rl/scripts/summarize_spreadsheetbench_rollouts.py rl/tests/test_spreadsheetbench_training_scripts.py rl/tests/test_spreadsheetbench_rollout_summary.py
git commit -m "feat: report spreadsheet multi-call metrics"
```

---

## Milestone 4: 剩余高价值工具

### Task 9: 补齐格式和结构工具

**Files:**
- Create: `envharness/bridges/spreadsheetbench/format_tools.py`
- Create: `envharness/bridges/spreadsheetbench/structure_tools.py`
- Modify: `envharness/bridges/spreadsheetbench/tools.py`
- Modify: `envharness/bridges/spreadsheetbench/bridge.py`
- Modify: `rl/envharness_rl/spreadsheetbench/projection.py`
- Modify: `rl/envharness_rl/spreadsheetbench/manager.py`
- Test: `rl/tests/test_spreadsheetbench_bridge_execution.py`
- Test: `rl/tests/test_spreadsheetbench_projection.py`

**Interfaces:**
- Produces: `format_range`、`delete_rows`、`delete_columns`、`manage_sheet`。
- Constraint: 所有 mutation 继续使用相同 lock/atomic commit helper。

- [ ] **Step 1: 从最新 badcase 统计四类操作频次，按频次确定实现顺序**
- [ ] **Step 2: 为 `format_range` 写值、数字格式、字体、填充、对齐、边框和行列尺寸测试**
- [ ] **Step 3: 实现 `format_range` 并通过回滚测试**
- [ ] **Step 4: 为行列删除写范围边界、公式移动和 merged cell 测试**
- [ ] **Step 5: 实现 `delete_rows` / `delete_columns`**
- [ ] **Step 6: 为 sheet create/rename/copy/move/hide 写重名、唯一可见 sheet 和非法索引测试**
- [ ] **Step 7: 实现 `manage_sheet`**
- [ ] **Step 8: 补 schema、projection、prompt、metrics 和 summary**
- [ ] **Step 9: 运行全部单测与 5-step 固定任务 A/B**

Gate: 相应 badcase 的 `run_python` 使用数下降，workbook 损坏数为 0，完整 399 成功率不低于上一里程碑。

- [ ] **Step 10: 提交格式和结构工具**

```bash
git add envharness/bridges/spreadsheetbench/format_tools.py envharness/bridges/spreadsheetbench/structure_tools.py envharness/bridges/spreadsheetbench/tools.py envharness/bridges/spreadsheetbench/bridge.py rl/envharness_rl/spreadsheetbench/projection.py rl/envharness_rl/spreadsheetbench/manager.py rl/tests
git commit -m "feat: add spreadsheet format and structure tools"
```

### Task 10: 评估中间重算工具

**Files:**
- Create: `envharness/bridges/spreadsheetbench/recalc_tools.py`
- Modify: `envharness/bridges/spreadsheetbench/bridge.py`
- Modify: `envharness/bridges/spreadsheetbench/tools.py`
- Test: `rl/tests/test_spreadsheetbench_bridge_execution.py`

**Interfaces:**
- Produces: `recalculate_and_read(range: str) -> bounded result`。
- Constraint: 每个 episode 最多调用一次，复用现有 LibreOffice 超时和进程组回收。

- [ ] **Step 1: 用 32 条公式任务测量单次 LibreOffice 重算 p50/p95 wall time**
- [ ] **Step 2: 只在 p95 不导致 rollout 预算失控时，写入中间重算失败、超时、有效读回和原 workbook 不被破坏测试**
- [ ] **Step 3: 实现单次有界重算和 range 读回**
- [ ] **Step 4: 运行 32 条 A/B，比较成功率与 wall time**

Gate: 只在成功率改善大于重算失败引入的回归，且单步 wall time 增幅可接受时合入默认 tool set；否则保持实验开关关闭。

---

## Final Verification

- [ ] 运行 `PYTHONPATH=.:rl:third_party/verl-agent python -m pytest -q rl/tests`。
- [ ] 运行 loader scale smoke，确认 128 train actor reset 通过。
- [ ] 运行 399/64 validation smoke，确认 399 个唯一 task。
- [ ] 运行 native tool worker smoke，覆盖 formula 和 multi-call。
- [ ] 运行 5-step GRPO，确认 rollout、reward、PPO update、validation 和 checkpoint。
- [ ] 对 Base 和候选 checkpoint 各跑完整 399 条 paired evaluation。
- [ ] 验评报告必须同时列出 success count/rate、Python error、write success、episode length、multi-call ratio 和 wall time。
- [ ] 更新 `md/spreadsheet_rl_vs_envharness_comparison.md`、`md/rollout_badcase_report.md` 和 `rl/README.md`。
- [ ] 运行 `git diff --check`，确认 exit code 0。

## Stop/Go Gates

1. Milestone 1 必须证明 399 条无重复无遗漏，才开始公式工具。
2. Milestone 2 必须证明无 workbook 损坏且公式类 badcase 下降，才开始 multi-call。
3. Milestone 3 必须降低 episode length 且不使固定验评成功率下降，才扩展剩余工具。
4. 任何里程碑在完整 399 上显著低于 Base `122/399`，都不进入长训练，先回滚并分析 paired badcase。
5. 本计划完成后仍不默认迁移 native verl loop；只在 `run_python` 占比、episode length 和多轮 wall time 仍是主瓶颈时，另立 PoC 设计与计划。

## 2026-09-24 五阶段扩展实施状态

在既有 validation batching、`fill_formula` 和 multi-call 基础上，本轮按确认顺序完成：

- [x] Linux LibreOffice `recalculate_and_read` 临时副本实现与每 episode 调用上限；
- [x] `fill_formula` malformed/quoted-formula 双层校验；
- [x] submit 前重算协议、workbook revision 和 stale-recalc 指标；
- [x] `run_python` AST preflight 与 reject/warning 指标；
- [x] `format_range`、`delete_rows`、`delete_columns`、`manage_sheet` 的有界原子实现；
- [x] manager、rollout、trainer、offline summary 和启动脚本指标透传；
- [x] 新增可复现 verl-agent tool metrics integration patch；
- [ ] Python 3.11 集群完整单测；
- [ ] 真实 LibreOffice recalc smoke；
- [ ] 5-step 固定任务 A/B；
- [ ] Base/候选 checkpoint 完整 399 条 paired evaluation。

代码验收前不修改现有长期训练结论。正在运行的 Ray job 使用提交时 runtime package，
不会自动加载本轮修改；必须新提交 job 才能观察这些工具和指标。
