# Spreadsheet-RL 与当前 EnvHarness 方案对比及适配建议

> 首次整理：2026-09-10；最后更新：2026-09-20
> 对比代码：Spreadsheet-RL `389be8c`；EnvHarness `180e258`  
> 当前固定基座模型：`Qwen3-4B-Thinking-2507`

## 1. 结论先行

截至 2026-09-20，已完成**数据层适配、部分原生工具迁移、大规模 loader
优化、评分/进程稳定性修复和训练可观测性建设**。尚未迁移 Spreadsheet-RL 的
native verl agent loop、单轮多工具调度、完整表格工具、SandboxFusion 和 Windows
Excel reward service。

当前方案不是“不能使用 Spreadsheet-RL 数据”。相反，EnvHarness 已经能够读取它的 parquet split，并正确映射：

- `reward_model.ground_truth`：任务目录相对路径；
- `extra_info`：任务元数据；
- `output.xlsx`：rollout 初始工作簿；
- `target.xlsx`：隐藏目标工作簿；
- `train_hermes.parquet`：训练集；
- `test_verified_hermes.parquet`：严格独立的验证集。

当前主要差距仍位于**环境和工具层**，但 EnvHarness 已不再是只有
`run_python`：已迁移三个只读工具和两个基础写工具，并保留 Python
兜底。实验表明这些工具、宽松 parser 和历史压缩已改善动作可解析性和
Python 错误率，但尚未证明能提高最终 workbook success rate。当前瓶颈已从
“环境不能闭环”转为“写操作覆盖不足、轨迹过长、reward 与最终成功不完全对齐”。

因此，建议继续维护 EnvHarness 这一条主线，但分阶段移植 Spreadsheet-RL 的工具设计。优先移植读取工具和常用写入工具，保留 `run_python` 作为兜底。现阶段不建议直接替换成 Spreadsheet-RL 全套训练栈，也不建议先部署 Windows Excel reward 服务。

## 2. 两套系统解决的问题

### 2.1 Spreadsheet-RL

Spreadsheet-RL 是论文对应的完整训练栈，包含：

- Spreadsheet Data Agent：构造训练任务和 oracle workbook；
- Spreadsheet Gym：多轮表格环境、工作区、原生工具和代码执行；
- fork 后的 native verl：异步多轮 tool agent rollout；
- SandboxFusion：隔离运行模型生成的 Python；
- Windows Excel reward service：用真实 Excel 重算公式并评分；
- GRPO 配置、Slurm 多机启动和 rollout/validation 数据落盘；
- audit UI：人工检查数据质量。

入口文档和配置：

- `../Spreadsheet-RL/README.md`
- `../Spreadsheet-RL/configs/qwen3-4b.sh`
- `../Spreadsheet-RL/configs/spreadsheet_rl_multiturn_grpo.yaml`
- `../Spreadsheet-RL/configs/tool/spreadsheet_tools.yaml`

### 2.2 当前 EnvHarness

当前项目是在通用 EnvHarness 和 `third_party/verl-agent` 上增加 SpreadsheetBench adapter，目标是：

- 用统一 `ActionableEnv` 接口承载 SpreadsheetBench；
- 将每条环境放在独立 Ray actor 中；
- 通过外部 environment manager 驱动多轮 rollout；
- 用本机 Python/openpyxl 修改工作簿；
- 用 LibreOffice 重算，再使用 SpreadsheetBench Online Judge 评分；
- 支持单机/多机外部 Ray、W&B、TensorBoard、checkpoint 和 badcase 轨迹落盘。

主要实现：

- `envharness/bridges/spreadsheetbench/bridge.py`
- `envharness/bridges/spreadsheetbench/dataset.py`
- `envharness/bridges/spreadsheetbench/online_judge_eval.py`
- `envharness/bridges/spreadsheetbench/tools.py`
- `rl/envharness_rl/spreadsheetbench/envs.py`
- `rl/envharness_rl/spreadsheetbench/manager.py`
- `rl/envharness_rl/spreadsheetbench/projection.py`
- `rl/scripts/run_spreadsheetbench_grpo.sh`

## 3. 架构对比

| 维度 | Spreadsheet-RL | 当前 EnvHarness | 影响 |
| --- | --- | --- | --- |
| 训练框架 | 仓库内 fork 的 native verl | `third_party/verl-agent` 加自定义 env 注册 | 两者的 rollout 接口和版本依赖不同 |
| 多轮控制 | verl 内部 `ToolAgentLoop` 状态机 | 外部 `SpreadsheetBenchEnvironmentManager` | native loop 更容易保留 token mask、logprob 和工具调用边界 |
| 工具协议 | OpenAI/Hermes schema 由 chat template 注入 | prompt 手写协议，generation 后做 projection | 当前链路更容易出现格式偏差和宽松恢复 |
| 单轮工具数 | 支持多个 tool call | 每轮只投影为一个 `Action` | 原版可并行读操作，并串行执行写操作 |
| 工作区 | 数据根目录下 `_workspaces/<request_id>` | 每个 episode 的 `/tmp/sb_*` | 都隔离任务，但原版有 manifest、文件锁和原子复制 |
| 代码执行 | SandboxFusion HTTP 服务 | 本地 Python subprocess | 当前部署简单，但隔离和资源治理较弱 |
| 表格重算 | Windows Microsoft Excel | Linux LibreOffice headless | Excel 兼容性更高；LibreOffice 更易部署 |
| reward | 异步上传、队列、轮询 | episode 结束时本地同步评分 | 原版适合慢 reward 和高并发；当前实现更直接 |
| 失败隔离 | 服务端持久队列、重试、超时 | evaluator 异常转为 `eval_error` | 当前已避免单条坏样本杀死整批，但缺少持久任务恢复 |
| rollout 记录 | native rollout/validation data | verl rollout + env trajectory 两套记录 | 当前 badcase 可追踪，但 token/tool 原生统计不如 native loop 完整 |

## 4. Harness 和 rollout 的关键差异

### 4.1 Spreadsheet-RL 的 native agent loop

核心代码是：

`../Spreadsheet-RL/verl/verl/experimental/agent_loop/tool_agent_loop.py`

它在一次 rollout 内部直接完成以下状态转换：

1. 用 tokenizer chat template 注入工具 schema；
2. 调用 vLLM 生成 assistant token；
3. 从 token 序列解析 Hermes tool call；
4. 根据工具的 `parallel_safe` 属性划分并行读和串行写；
5. 执行工具并把结果追加为 tool message；
6. 继续下一轮生成，直到没有工具调用或达到 turn/token 上限；
7. 返回 `response_ids`、`response_mask`、`response_logprobs`、turn scores 和 tool rewards。

这类“native”不是指工具本身一定由 Excel 实现，而是指**多轮工具调用属于 verl rollout 的内部流程**。训练框架知道哪些 token 是模型输出、哪些是工具返回，因此 loss mask 和概率统计更自然。

### 4.2 EnvHarness 的外部环境管理

当前流程是：

1. verl-agent 生成一段文本；
2. `projection.py` 从文本中抽取一个 `Action`；
3. environment manager 调用远程 EnvHarness Ray actor；
4. 将观察、历史和工具反馈重新拼进下一轮 prompt；
5. 在 `submit` 或 `max_steps` 时评分。

当前 projection 已支持：

- 标准 `<tool_call>...</tool_call>`；
- 去掉 `<think>` 前缀后解析；
- fenced JSON 恢复；
- bare JSON 恢复；
- 对格式错误、未知工具和缺少参数返回诊断。

这提高了对 Qwen3 Thinking 输出的兼容性，但它仍然是 generation 后的文本恢复，而且每轮只执行一个动作。它不能完全等价于 native verl 的多 tool-call token-aware parser。

### 4.3 哪一个更适合当前项目

- 如果目标是**快速验证环境、reward、数据和 GRPO 是否闭环**，当前 EnvHarness 更容易维护和调试。
- 如果目标是**复现论文工具协议、吞吐和最终效果**，native verl loop 更接近原实验。
- 当前成功率瓶颈主要在工具可用性和模型生成代码的可靠性，尚不足以证明必须先整体迁移 native loop。

## 5. 工具能力差异

### 5.1 Spreadsheet-RL 原生工具

`configs/tool/spreadsheet_tools.yaml` 注册了以下工具：

| 工具 | 作用 | 并发属性 |
| --- | --- | --- |
| `inspect_range` | 按 A1 range 读取值、公式、样式或统计摘要 | 只读，可并行 |
| `list_sheets` | 列 sheet、可见性、顺序和维度 | 只读，可并行 |
| `find_cells` | 在值/公式中查找文本 | 只读，可并行 |
| `write_range` | 批量写静态值 | 写操作，串行 |
| `format_range` | 设置样式、数字格式、行列尺寸等 | 写操作，串行 |
| `fill_formula` | 按模板填充公式并触发重算检查 | 写操作，串行 |
| `clear_range` | 清除值、公式或样式 | 写操作，串行 |
| `delete_rows` | 删除行 | 写操作，串行 |
| `delete_columns` | 删除列 | 写操作，串行 |
| `manage_sheet` | 新建、重命名、复制、移动、隐藏 sheet | 写操作，串行 |
| `recalculate_and_read` | 用 Excel 重算并读取指定区域 | 远程重算 |
| `code_interpreter` | 通过 SandboxFusion 执行 Python | 隔离代码执行 |

这些工具的重要价值不是“能实现 Python 做不到的事”，而是：

- 模型只需要生成短小的结构化参数，不需要反复生成 openpyxl 样板代码；
- schema 能约束输入并给出明确错误；
- 读取结果可以限制范围和长度，降低 prompt 膨胀；
- 写操作可加锁、限制最大单元格数并原子提交；
- 常见操作失败时更容易重试，不需要重新生成整段 Python。

### 5.2 EnvHarness 当前工具

当前工具由 `SPREADSHEETBENCH_TOOL_SET` 分三级开启：

- `python`：`run_python`、`validate_workbook`、`submit`；
- `native_read`：在 `python` 基础上增加 `list_sheets`、`inspect_range`、
  `find_cells`；
- `native_basic`：再增加 `write_range`、`clear_range` 和 `fill_formula`。

只读工具已支持范围、返回字符数和 workbook 大小限制。写工具已使用统一文件锁、
临时文件校验和原子替换，避免失败写入损坏工作簿。`run_python` 仍是公式、格式、
排序和结构操作的兜底路径。

优点：能力上限高，开发量小，复杂任务可直接写任意 openpyxl/pandas 逻辑。

缺点：

- 模型经常生成 syntax error、NameError、API 使用错误和未保存文件；
- 检查、修改、验证都要重新生成代码，输出长度明显增加；
- fresh process 不保留 Python 变量，模型必须靠工作簿和文本历史恢复状态；
- 一个小参数错误会使整段操作失败；
- 难以把“工具选择错误”和“Python 实现错误”分开分析。

已完成的 compact history、错误分类、Python error penalty、`validate_workbook`
和基础结构化工具能改善可观测性与部分动作稳定性，但尚未覆盖公式、格式、
行列和 sheet 管理，也未迁移单轮多工具调度。

## 6. 工作区、执行环境和安全性

### 6.1 Spreadsheet-RL

- 每条 rollout 使用 request ID 创建独立共享工作区；
- 从任务目录的 `output.xlsx` 原子复制为工作文件；
- 使用 manifest 防止同一个 workspace ID 指向不同任务；
- 写工具使用文件锁，读工具可并行；
- 对相对路径、软链接、压缩包和文件大小做限制；
- `code_interpreter` 通过 SandboxFusion 执行，并挂载同一个工作区；
- Slurm 脚本负责在各节点启动 Ray 和 SandboxFusion。

### 6.2 EnvHarness

- `reset()` 在本机 `/tmp` 创建独立 episode 目录；
- 将初始文件复制成 `*_output.xlsx`；
- `run_python` 的 cwd 指向该目录；
- subprocess 有 step timeout，但没有容器级内存、系统调用和网络隔离；
- 工作簿在 episode 内跨 turn 持久化，Python 进程状态不持久化；
- Ray actor 持有环境状态，关闭时清理临时目录。

当前研究集群是可信模型和受控代码场景，因此本地 subprocess 足以继续实验；如果后续开放给不可信模型、并发规模显著扩大或出现节点稳定性问题，再接入 SandboxFusion 更合理。

## 7. Reward 与公式重算差异

### 7.1 Spreadsheet-RL 的 Excel reward service

原版服务位于：

- `../Spreadsheet-RL/reward/async_reward_api/`
- `../Spreadsheet-RL/verl/verl/utils/reward_score/sheet_arena.py`
- `../Spreadsheet-RL/verl/verl/experimental/reward_loop/reward_manager/limited.py`

训练端上传 agent workbook 和 `thread_dir`，Windows 服务端使用 Excel COM 重算，再与 `target.xlsx` 的答案区域比较。服务使用 FastAPI、SQLite 持久任务、worker pool、submit/poll、超时、重试和并发限制。它还提供 `/recalculate`，供公式工具在 episode 中间重算和读回 cached value。

其优势是对 Excel 特有公式、动态数组、日期/数字格式和计算缓存的兼容性最好。代价是必须长期维护 Windows + Microsoft Excel 服务，处理网络、队列、Excel COM 卡死和数据同步。

### 7.2 EnvHarness 的本地 evaluator

当前 evaluator 在 Linux 上调用 LibreOffice headless 重算 agent workbook 和缓存后的 golden workbook，然后调用 SpreadsheetBench Online Judge 比较 answer range。

已经增加 evaluator 异常隔离：单个样本的重算或评分失败会转成 `eval_error` 和失败 reward，不再让 Ray worker 异常退出并杀死整个训练批次。

需要保留的风险说明：LibreOffice 与 Microsoft Excel 并非完全等价。对于新 Excel 函数、外部链接、宏、复杂图表和部分格式，reward 可能有偏差。因此最终报告应区分：

- Linux/LibreOffice validation；
- 官方或 Windows/Excel validation。

## 8. 数据层：已经适配到什么程度

### 8.1 Spreadsheet-RL 官方结构

parquet schema 为：

```text
data_source, agent_name, prompt, ability, reward_model, extra_info
```

workbook archive 中每个任务目录包含：

```text
instruction.json
input.xlsx
output.xlsx
target.xlsx
```

这里需要特别明确：原训练代码实际将 `output.xlsx` 作为可编辑初始文件，将 `target.xlsx` 作为 oracle。`input.xlsx` 存在于归档中，但不是当前 reward workspace 的默认 seed。

官方 README 给出的 split 为：

| split | 用途 | README 行数 |
| --- | --- | ---: |
| `train_hermes.parquet` | ExcelForum 训练集 | 5,925 |
| `test_hermes.parquet` | SpreadsheetBench | 2,722 |
| `test_2_hermes.parquet` | SpreadsheetBench-2 非视觉子集 | 297 |
| `test_verified_hermes.parquet` | SpreadsheetBench-Verified | 400 |
| `test_domain_hermes.parquet` | Domain-Spreadsheet | 1,662 |

此前本地 loader 曾读到 `test_verified` 为 399 条，而当前 Spreadsheet-RL README 标为 400 条。这可能来自本地数据版本或坏样本过滤差异。后续实验必须把 parquet 实际行数和有效任务数写入 run manifest，不能只依赖 README 数字。

### 8.2 当前已完成的适配

`envharness/bridges/spreadsheetbench/dataset.py` 已实现：

- 读取 Spreadsheet-RL parquet；
- 解析 `reward_model.ground_truth`；
- 合并 `extra_info` 中的 instruction、answer position/type/sheet 等字段；
- 校验相对任务路径，拒绝越界路径；
- 将 `output.xlsx` 映射为 `SBTask.init_path`；
- 将 `target.xlsx` 映射为 `SBTask.golden_path`；
- 将包含 `/` 的 task ID 安全转换为临时目录和输出文件名；
- 允许 train 和 val 指定不同 split。

训练脚本支持：

```bash
SPREADSHEETBENCH_DATA_FORMAT=spreadsheet_rl
SPREADSHEET_RL_TRAIN_FILE=train_hermes.parquet
SPREADSHEET_RL_VAL_FILE=test_verified_hermes.parquet
```

所以，**使用 Spreadsheet-RL 数据训练和独立评测已经属于可用能力，不是尚未开始的未来阶段**。尚未适配的是原版环境工具和训练 loop，而不是数据。

## 9. 原始训练配置与当前配置

Spreadsheet-RL 的 Qwen3-4B 默认配置：

| 参数 | 原版设置 |
| --- | ---: |
| Base model | `Qwen3-4B-Thinking-2507` |
| train batch size | 128 |
| val batch size | 1400 |
| rollout `n` | 8 |
| max prompt length | 4096 |
| max response length | 24576 |
| max model length | 28672 |
| max user / assistant turns | 15 / 15 |
| temperature / top-p / top-k | 0.6 / 0.95 / 20 |
| learning rate | `2e-6` |
| PPO mini batch | 64 |
| actor/ref strategy | FSDP2 |
| rollout mode | async |
| reward concurrency | 2 |
| val before train | true |
| save / test frequency | 5 / 15 |
| total epochs | 2 |

当前 EnvHarness diagnostic 默认配置：

| 参数 | 当前设置 |
| --- | ---: |
| Base model | `Qwen3-4B-Thinking-2507` |
| train batch size | `16 * NNODES` |
| val batch size | `32 * NNODES` |
| rollout `n` | 8 |
| max prompt length | 8192 |
| max response length | 16384 |
| max env steps | 10 |
| learning rate | `1e-6` |
| tensor parallel | 2 |
| history | compact，保留最近 2 轮并截断 action/observation |
| val before train | true |
| total training steps | 20 |

不能只把原版 batch/length 参数原样复制过来。原版有 native async loop、结构化工具和另一套 verl/FSDP2 实现；当前每一步需要 Ray actor 往返并可能执行 Python/LibreOffice。相同 batch size 不代表相同吞吐、显存和 CPU 压力。

## 10. 已经借鉴的内容与尚未借鉴的内容

### 10.1 已经完成或部分完成

- Spreadsheet-RL parquet 和 workbook 语义适配；
- 真正分离的 train/val split；
- Hermes-compatible tool-call 格式；
- Qwen3 Thinking 输出的宽松解析；
- 多轮表格操作；
- 每 episode 独立工作区；
- outcome-based workbook reward；
- rollout group 内相同 task seed；
- 环境轨迹和 verl rollout 双份记录；
- evaluator crash 隔离；
- 单机/多机外部 Ray 启动；
- checkpoint、W&B、TensorBoard 和 run manifest；
- Python syntax/runtime 分类及诊断 penalty。
- `list_sheets` / `inspect_range` / `find_cells` 只读工具；
- `write_range` / `clear_range` 基础写工具，包含文件锁和原子提交；
- native tool 调用率、成功率、错误率和分工具训练/验证指标；
- Spreadsheet-RL loader 按 seed 先选 parquet row、再只物化单个任务；
- train worker 轮换任务、validation worker 固定任务的可比较采样；
- workbook score、execution penalty、env total 和 invalid-action penalty 分解指标；
- Ray actor/reset/step/close 超时、心跳、actor ID/task ID 和阶段耗时诊断；
- `run_python` / LibreOffice 独立进程组超时回收；
- Base 与 checkpoint 在同一 399 条 Verified 任务上的 paired evaluation。

### 10.2 尚未完成

- native verl `ToolAgentLoop`；
- token-aware Hermes parser 和原生 response mask；
- 单轮多个 tool call；
- 读工具并行、写工具串行；
- Spreadsheet-RL 剩余原生工具：公式、格式、行列、sheet 管理和中间重算；
- SandboxFusion；
- Windows Excel reward/recalculate 服务；
- async rate-limited reward queue；
- 原版 FSDP2、dynamic batching 和 fused kernel 性能栈；
- audit UI；
- Qwen3-Coder parser。当前固定使用 Qwen3 Thinking，因此最后一项不是近期需求。
- 验证集总规模与 actor 并发数解耦；当前 `VAL_BS=64` 同时限制在线验证任务数。

## 11. 可借鉴项、收益和适配难度

| 借鉴项 | 预期收益 | 难度 | 风险/依赖 | 建议 |
| --- | --- | --- | --- | --- |
| `list_sheets`、`inspect_range`、`find_cells` | 降低检查代码错误和 observation 长度 | 中 | 需要稳定的 A1 parser 和输出限长 | 第一优先级 |
| `write_range`、`clear_range` | 降低简单写值任务的 Python 错误 | 中 | 需要锁、shape 校验和原子保存 | 第一优先级 |
| `fill_formula` | 直接覆盖大量公式任务 | 中高 | 公式平移、重算、失败回滚复杂 | 第二优先级 |
| `format_range`、行列/sheet 管理 | 覆盖格式和结构任务 | 中高 | openpyxl 与 Excel 行为差异 | 第二优先级 |
| 保留 `run_python` 兜底 | 保持复杂任务能力上限 | 低 | 仍需错误治理 | 必须保留 |
| workspace lock/atomic commit | 避免写并发和损坏文件 | 中 | 需统一所有 mutation 路径 | 随写工具一起做 |
| parser error feedback | 降低格式错误后的无效循环 | 低中 | 需保持训练 prompt 一致 | 当前已有，继续加强 |
| 单轮多 tool call | 降低 turn 数，提高吞吐 | 中高 | 当前 manager/action 接口只接一个动作 | 工具稳定后再做 |
| native verl loop | 协议、mask、吞吐更接近论文 | 高 | 需要替换/大改 verl-agent 接口 | 独立分支验证 |
| SandboxFusion | 更强隔离和资源限制 | 中高 | 服务部署、镜像和多节点路由 | 非当前成功率首要瓶颈 |
| Windows Excel reward | 最高 Excel 语义一致性 | 很高 | Windows、Office、COM、网络和队列 | 最终严格评测再接入 |
| FSDP2/dynamic batch/fused kernel | 提高训练吞吐 | 高 | 对 torch/transformers/vLLM/verl 版本敏感 | 与环境适配分开推进 |

## 12. 推荐适配路线

### 阶段 A：保持当前架构，补齐高价值原生工具

目标：先验证“减少 Python 生成”能否明显提高 valid action ratio 和 success rate。

建议实现顺序：

1. `list_sheets`；
2. `inspect_range`；
3. `find_cells`；
4. `write_range`；
5. `clear_range`；
6. `fill_formula`；
7. `format_range`、`manage_sheet`、`delete_rows`、`delete_columns`。

实现原则：

- 工具直接操作当前 episode 的 `output_path`；
- schema、参数校验和错误码尽量与 Spreadsheet-RL 对齐；
- 所有 mutation 使用同一把 workbook lock；
- 保存采用临时文件 + `os.replace`，失败不覆盖旧结果；
- 每个工具限制最大 range、文件大小和返回字符数；
- `run_python` 继续存在，用于 native tools 无法覆盖的复杂任务；
- 工具结果进入现有 compact history 和 trajectory 日志。

验证方法：固定模型、数据、seed、batch 和采样参数，只对比工具集合：

- A0：当前 `run_python + validate + submit`；
- A1：A0 + 三个只读工具；
- A2：A1 + 常用写工具。

主要指标：

- `episode/success_rate`；
- `episode/valid_action_ratio`；
- syntax/runtime error rate；
- response/prompt clip ratio；
- episode length 和 tool call count；
- 每 rollout、每 train step wall time；
- 不同任务类型的成功率。

### 阶段 B：增强 manager/projection，但不替换 verl

目标：支持一轮多个 tool call，并让读操作并行、写操作有序串行。

需要修改：

- projection 从 `Action` 扩展为 action list；
- manager 执行 partition 和结果回填；
- 明确定义 mutation 顺序和后续动作失败语义；
- 更新 response mask/无效动作统计；
- 增加多工具、部分失败、超限和冲突测试。

这是中高难度修改，但比整体换 trainer 风险低。只有 A1/A2 证明原生工具有效后才值得实施。

### 阶段 C：独立验证 native verl loop

不要直接在现有主训练脚本上替换。建议建立独立 compatibility branch，做最小闭环：

1. 用 Spreadsheet-RL fork 的 verl 加载当前固定模型；
2. 只注册 EnvHarness 适配后的 3 个只读工具和 `submit`；
3. 在 8 到 16 条任务上验证 prompt、tool mask、reward 和 checkpoint；
4. 与当前 external manager 对比初始 validation 和 rollout 文本；
5. 确认版本、显存和多机稳定性后，再决定是否迁移正式训练。

### 阶段 D：严格 Excel reward 和性能工程

只有出现以下情况时再部署 Windows reward：

- LibreOffice 与 Excel 对同一批任务评分差异不可接受；
- 目标是严格复现论文分数；
- 公式重算成为 native tool 的必要能力。

FSDP2、dynamic batching、fused kernel、SandboxFusion 和多机吞吐应作为独立性能路线推进，避免与工具语义改动混在同一组实验中。

## 13. 预计改动范围

### 13.1 采用推荐路线 A

主要改动集中在：

- `envharness/bridges/spreadsheetbench/tools.py`：增加工具 schema；
- `envharness/bridges/spreadsheetbench/bridge.py`：dispatch、锁、原子保存和 observation；
- `rl/envharness_rl/spreadsheetbench/projection.py`：允许新工具并校验参数；
- `rl/envharness_rl/spreadsheetbench/manager.py`：更新工具 prompt、指标和 trajectory；
- `rl/tests/`：工具单测、bridge 集成测试、projection 和 manager 测试；
- `rl/scripts/run_spreadsheetbench_grpo.sh`：增加 tool-set 开关，便于 A/B 实验。

读取工具属于中等改动；完整写工具和公式工具属于中高改动。难点不在 openpyxl API 本身，而在参数边界、公式语义、工作簿不损坏、错误可恢复以及与现有 rollout 指标兼容。

### 13.2 整体迁移 Spreadsheet-RL native stack

将涉及：

- 替换或并存两套 verl；
- 重写训练启动、Hydra config 和 Ray/Slurm 编排；
- 迁移 `BaseTool` 生命周期；
- 接入 async agent loop 和 reward manager；
- 重新验证 FSDP、vLLM、torch、transformers 和模型架构版本；
- 部署 SandboxFusion；
- 如果追求完全一致，还要部署 Windows Excel 服务。

这不是小改动，且容易与当前已经稳定的 Ray、checkpoint、日志和 evaluator 路径发生耦合。它适合作为论文复现分支，不适合作为下一次诊断实验之前的阻塞项。

## 14. 下一步建议

### 14.1 2026-09-10 实施状态

路线 A1 的第一批代码已经加入当前分支，使用
`SPREADSHEETBENCH_TOOL_SET=native_read` 开启：

- `list_sheets`；
- `inspect_range` 的单个有限 A1 range、`cells/summary` 模式；
- `find_cells` 的 values/formulas、contains/equals/prefix 查询；
- 10,240 字符返回上限、400-cell inspect 上限和 100 MB 文件上限；
- projection、prompt、环境指标、trajectory 和启动配置追踪；
- `SPREADSHEETBENCH_TOOL_SET=python` 保留原 baseline 行为。

路线 A2 的第一批受控写工具也已实现，使用
`SPREADSHEETBENCH_TOOL_SET=native_basic` 开启 A1 的全部工具以及：

- `write_range`：写入静态 scalar、row、column 或二维矩阵；单元格数组中的
  `null` 表示跳过该单元格；拒绝公式字符串；
- `clear_range`：只清除有限范围的值和公式，不移动行列；
- 写操作共用 workbook lock，并通过同目录临时文件校验后原子替换；
- 单次 mutation 最多 50,000 个单元格，单个字符串最多 8,192 字符，
  workbook 最大 100 MB；
- `run_python` 保留，负责公式、格式、排序、插入删除行列等复杂操作。

同时补充了训练和验证观测指标：

- `env/read_tool_call_ratio`、`env/read_tool_success_rate`、
  `env/read_tool_error_rate`；
- `env/write_tool_call_ratio`、`env/write_tool_success_rate`、
  `env/write_tool_error_rate`；
- `tool/{tool_name}_ratio`，覆盖 `run_python`、三个读工具、两个写工具、
  `validate_workbook` 和 `submit`；
- validation 使用对应的 `val/env/...` 和 `val/tool/...` 名称；
- 离线可运行 `rl/scripts/summarize_spreadsheetbench_rollouts.py` 汇总 env
  trajectory 中的调用、成功、错误和最终 success rate。

当前仍未支持多 range、regex、公式写入或单轮多个 tool call。A2 应先在
集群跑单测和 worker smoke，再用与 A1 完全一致的模型、数据、seed、batch、
长度和训练步数做对照；已有运行中的 Ray job 不会自动加载提交后的新代码。

下一步不应继续只调 learning rate、response length 或 max steps。当前 badcase 已经说明，模型经常在“生成和修复 Python”上浪费 rollout 预算。建议：

1. 以 Spreadsheet-RL split 作为正式数据线：`train_hermes` 训练、`test_verified_hermes` 验证；
2. 实现第一批只读 native tools：`list_sheets`、`inspect_range`、`find_cells`；
3. 增加 tool-set 配置开关，保证同一代码可以跑 A0/A1 对比；
4. 用 10 到 20 个 training steps 做固定 seed 诊断，不先追求长训练；
5. 若 valid action ratio、clip ratio 和 wall time 明显改善，再实现写工具；
6. 若结构化工具有效但 external manager 成为明显性能/协议瓶颈，再进入 native verl loop PoC；
7. 最终再做 Windows Excel reward 对齐测试，而不是立即把训练依赖到 Windows 服务。

## 15. 阶段判断

- **数据能否用**：能，且 train/val split 和 `output.xlsx -> target.xlsx` 映射已经适配。
- **当前 codebase 是否需要放弃**：不需要。它仍然适合环境闭环、badcase 诊断和增量工具实验。
- **Spreadsheet-RL 最值得借鉴什么**：结构化表格工具、单轮多工具调度、工作簿锁与原子提交、工具输出限长。其中只读工具、基础写工具、文件锁和原子提交已迁移。
- **是否立即迁移 native verl**：不建议。先用当前架构证明工具层收益，再做独立 PoC。
- **是否立即部署 SandboxFusion/Windows Excel**：不建议作为当前 success rate 优化的第一步；前者主要解决隔离，后者主要解决评分语义。
- **适配难度**：数据层低且已完成；只读工具中等；完整写工具中高；native verl loop 高；完整复现原版基础设施很高。

## 16. 2026-09-14 状态补充

### 数据部署状态

`gemininjceph5` 当前仓库的
`experiments/spreadsheetbench/data/Spreadsheet-RL` 已于 2026-09-14 从
`geminisgceph1` 的完整副本复制并解压。loader 和启动脚本现在可以直接使用该数据线。
相邻的 `../Spreadsheet-RL` 仍只是代码仓库；正式数据由 Hugging Face dataset 发布，
不在 GitHub 仓库中。

目标数据包含 5,925 条 ExcelForum、2,722 条 SpreadsheetBench、297 条
SpreadsheetBench-2、399 条 SpreadsheetBench-Verified 和 1,662 条 Domain 任务。
归档完整性以及关键 parquet/zip 文件大小已经校验。具体路径和评测命令见
`md/independent_evaluation_plan.md`。

### 最新工程修复

在已有 native read/write、compact history 和 evaluator 异常隔离基础上，又完成：

- `run_python` 和 LibreOffice 使用独立进程组，超时后杀死并回收整个进程组；
- Ray reset/step/close 增加全局超时、actor ID、task ID 和 action 诊断；
- rollout 的 reset、generate、env step 增加 START/HEARTBEAT/END/ERROR；
- bridge 为 Python、native tool、recalc、compare 和 validate 增加阶段耗时；
- `validate_workbook` 对大 answer range 从 read-only 随机单元格访问改为
  `iter_rows()` 单次顺序扫描；任务 `455-35` 的 `A1:J10411` 范围由超过 600 秒降到约
  1.3 秒的本地扫描时间；
- range parser 同时支持单格 `A1`、矩形 `A1:J10411` 和整列 `A:J`。

### 最新实验

`grpo_spreadsheetbench_diagnostic_20260912_194504` 使用两节点、16 GPU、
Qwen3-4B-Thinking-2507、`native_basic`、`MAX_STEPS=15` 和 50 个训练 step。截至 step 34
没有再次发生 evaluator/actor timeout，说明上述稳定性修复起作用。

但训练效果尚未形成稳定提升：初始和 step 30 validation success rate 都为 `0.281`，
step 10 达到过 `0.344`；训练 success rate 在 step 33 为 `0.453`，step 34 又降到
`0.078`，同时 `actor/kl_loss=7.532`、response clip ratio 为 `0.111`。当前应先建立
严格独立、全量、paired 的 checkpoint 评测，再决定是否继续调学习率或扩大训练。

注意：当前 checkpoint 在 Verified-400 上训练，所以
`test_verified_hermes.parquet` 不能未经重叠审计就称为 held-out。当前 checkpoint 应优先
用 `test_domain_hermes.parquet` 或 `test_2_hermes.parquet` 评测；未来使用
`train_hermes.parquet` 训练时，再以 `test_verified_hermes.parquet` 作为主 held-out。

## 17. 2026-09-18：Spreadsheet-RL 全量训练的 loader 阻塞

全量训练实验 `grpo_spreadsheetbench_full_20260918_004429` 暴露了新的规模问题：
32 个 validation actor 可以完成 reset，但训练阶段的 128 个 actor 在 600 秒内
`ready=0/128`。故障发生在首批 train rollout 的 reset 阶段，尚未进入模型生成、
reward 计算或 PPO 更新。

根因是当前 envharness 的 Spreadsheet-RL loader 在每个 Ray actor 首次 reset 时都会
完整读取 5,925 行 parquet，并逐任务访问 `instruction.json`、`output.xlsx` 和
`target.xlsx`。进程内 `lru_cache` 无法跨 actor 复用，128 actor 会把全量任务目录扫描
放大为约 75.8 万次小文件访问。Spreadsheet-RL 原生实现由 verl 数据层逐样本传递
`extra_info` 和 `ground_truth`，不存在 actor 内反复构建全量任务表的问题。

已完成修改方案设计，见
`md/spreadsheet_rl_loader_optimization_plan.md`。第一阶段采用最小兼容方案：缓存 parquet
原始行、先根据 seed 选 row、再只物化被选中的一个 `SBTask`；全量 loader 仅保留给
离线枚举和兼容调用。同时补充 reset 分阶段日志与 128 actor 独立 loader 压测。若该方案
仍无法在 180 秒内完成 128 actor reset，再升级为 driver 预加载并通过 Ray object store
广播 manifest。

当前状态：该阻塞已解除。loader 已按 row cache + lazy materialization 修改，
128 actor scale smoke、5-step GRPO 和后续 20-step 训练均已能越过 reset 阶段。
`grpo_spreadsheetbench_diagnostic_20260919_215017` 进一步证明该 loader 能支撑单机
8 卡、16 个 task group / 128 条 rollout 的持续训练。

### Loader 实施更新

2026-09-18 已按计划完成第一阶段代码修改：训练 reset 现在先从 parquet row cache 中
选择目标行，再只物化一个任务；全量 loader 保留给离线枚举。worker/bridge reset 的
阶段日志和 128 actor 无模型压测脚本也已加入。训练集群后续完成了单测、
scale smoke、5-step 和 20-step GRPO 验收；原先 `ready=0/128` 的 reset 卡死未再复现。

## 18. 2026-09-20：迁移台账与有效性结论

### 18.1 已完成的适配

| 模块 | 完成内容 | 当前状态 |
| --- | --- | --- |
| 数据 | Spreadsheet-RL parquet schema、`ground_truth` / `extra_info`、`output.xlsx` / `target.xlsx`、train/val split | 已完成 |
| 大规模 loader | parquet row cache，先选 row 再 lazy materialize 单任务，安全 task ID | 已完成 |
| verl-agent 注册 | `env.env_name=envharness_rl/spreadsheetbench`，GRPO、checkpoint、W&B、TensorBoard、run manifest | 已完成 |
| Hermes 协议 | 标准 tool call，`<think>` / fenced JSON / bare JSON 宽松恢复，错误反馈 | 已完成 |
| 历史管理 | compact history，分别限制 action 和 observation，保留 projected action | 已完成 |
| 只读工具 | `list_sheets`、`inspect_range`、`find_cells` | 已完成 |
| 基础写工具 | `write_range`、`clear_range`，文件锁、范围校验、原子替换 | 已完成 |
| Python 兜底 | subprocess、超时、进程组回收、syntax/runtime/timeout 分类 | 已完成 |
| evaluator | LibreOffice 重算、OJ compare、golden cache、单样本异常隔离 | 已完成 |
| Ray 稳定性 | reset/step/close 超时，actor/task/action 定位，rollout 阶段心跳 | 已完成 |
| 轨迹与指标 | env/verl 双轨迹，parser、tool、Python error、reward 分解指标 | 已完成 |
| 验证可比性 | train task 轮换，fast-val task 固定，确定性 validation | 已完成 |
| 独立评测 | Base/checkpoint 使用相同 399 条 Verified 任务做 paired evaluation | 已完成 |
| 完整验证分批 | `VAL_SIZE` / `VAL_CONCURRENCY` 解耦，按 parquet task index 激活变长 worker batch | 代码完成，待集群验收 |
| 公式工具 | `fill_formula` 公式平移、锁、临时文件校验和原子替换 | 代码完成，待集群验收 |
| 单轮 multi-call | 最多四个调用、顺序执行、单 turn 计数、逐调用诊断与指标 | 代码完成，待集群验收 |

### 18.2 哪些修改已证明起作用

**工程闭环和稳定性：已证明。**

- Spreadsheet-RL 5,925 条训练 split 已能启动 128 rollout actor；旧 loader 的
  `ready=0/128` reset 超时不再复现。
- 5-step、20-step 和当前 50-step 配置均已越过 reset、rollout、reward、PPO
  update 和 checkpoint 环节。
- `grpo_spreadsheetbench_diagnostic_20260919_215017` 已完成 step 13，并进入后续
  rollout；截至记录时没有再出现 evaluator/actor 卡死。
- 该 run 产生了 448 条 validation env trajectory：64 个固定 task 各出现 7 次，
  对应 step 0/2/4/6/8/10/12。这证明“固定 fast-val”实际生效。
- reward 分解和 native tool 指标已出现在 train/validation 日志中，能区分
  workbook score、execution penalty、invalid-action penalty 和分工具成功率。

**动作可解析性和执行质量：部分改善。**

- 完整 399 条 paired evaluation 中，step 5 相比 Base 的 native parser 有效率从
  `0.7457` 升到 `0.9440`，Python 错误率从 `0.1960` 降到 `0.1143`。
- 当前低学习率 run 的 fast-val 中，parser invalid ratio 从 step 0 的 `0.094`
  降到 step 12 的 `0.073`；只读工具成功率基本为 `1.0`。
- compact history 后，该 run 已完成 step 的 prompt clip ratio 大多为 `0`至 `0.031`，
  response clip ratio 大多为 `0.002`至 `0.026`，明显低于早期 4096 response 配置。

### 18.3 哪些修改还没有证明提升效果

**最终任务成功率：尚未提升。**

- 完整 399 条评测中，Base 为 `122/399 = 0.3058`，step 5 为
  `113/399 = 0.2832`。parser 和 Python 错误改善没有转化为 workbook success 提升。
- 当前低学习率 run 的固定 64 条 fast-val 从 step 0 的 `0.312` 波动到
  step 12 的 `0.359`，中间值为 `0.312`至 `0.344`。尚未对 step 12 跑完整
  399 条，因此不能宣称训练效果已改善。
- `write_range` 只接受静态值，模型经常将公式传给它并收到
  `write_range accepts static values only`。当前 run 的 train write success rate 在不同
  step 间约为 `0.06`至 `0.86`，波动很大。
- 模型仍然主要使用 `run_python`；当前 run 中该工具占比约 `0.66`至 `0.85`。
  这说明基础 native tools 还没有替代高风险的大段 Python 代码。
- episode length 仍长期接近 `MAX_STEPS=10`，很多轨迹没有提前 submit。细粒度工具
  在当前“一轮一动作”调度下也会占用 turn 预算。

### 18.4 与 Spreadsheet-RL 原版 validation 的剩余差异

Spreadsheet-RL 原版默认使用 `test_hermes.parquet` 2,722 条验证任务，
`val_batch_size=1400` 只是 dataloader 的分批大小；trainer 会遍历完整验证集。若改用
`test_verified_hermes.parquet`，则会在一个 batch 中评测完整 400/本地可用
399 条。原版配置是 step 0 validation，之后每 15 step 评测。

修改前 EnvHarness 中 `VAL_BS` 同时决定验证任务数和 Ray env actor 数，因此
`VAL_BS=64` 只是固定 64 条 fast-val。当前代码已解耦为：

```text
validation_size = 399
validation_concurrency = 32 或 64
```

新 job 可在不同时创建 399 个 actor 的情况下分批遍历完整验证集；399/64 的集群
验收尚未执行。

### 18.5 当前优先级

1. 保留固定 64 条 fast-val 做频繁健康检查，对候选 checkpoint 跑完整 399 条。
2. 对已实现的 validation 分批、`fill_formula` 和串行 multi-call 跑完整验收。
3. 验收通过后再补格式、行列和 sheet 管理工具。
4. 串行 multi-call 稳定后再评估连续只读调用并行，写工具始终按序执行。
5. 候选 checkpoint 的选择以完整验证集 `success_rate` 为主，parser 和 shaping
   reward 只用于诊断，不再单独作为效果结论。

## 19. 已确认的下一阶段实施顺序

2026-09-20 已确认按以下顺序推进：

1. 完整验证集大小与 Ray actor 并发数解耦；
2. 迁移 `fill_formula`；
3. 迁移单轮 multi-call；
4. 按 badcase 频次迁移格式、行列和 sheet 管理等剩余工具。

详细文件、接口、TDD 步骤、smoke 命令、A/B 实验和 Stop/Go 门槛见
`md/spreadsheet_rl_next_migration_implementation_plan.md`。当前运行中的实验保持不变，
新代码只用于后续新建 Ray job。

## 20. 2026-09-20 实施进展

本轮已在代码层完成前三项迁移：

1. placeholder parquet 逐行携带 `env_kwargs.task_index`；validation manager
   按 batch 中的索引 reset，Ray worker pool 支持最后一个不足并发上限的 batch。
2. 启动参数新增 `VAL_SIZE` 和 `VAL_CONCURRENCY`。前者决定验证集行数，后者
   决定 validation actor pool 大小；未设置时都继承旧 `VAL_BS`，保持旧命令兼容。
3. `native_basic` 新增 `fill_formula`，通过 openpyxl `Translator` 平移相对、混合
   和绝对引用，并复用 workbook lock、临时文件重开校验和原子替换。
4. projection 支持每轮最多四个有序 `<tool_call>`；`submit` 只能位于末尾。
   worker 在一个 episode turn 内顺序执行 batch，记录逐调用结果，终止或写失败后
   停止后续调用。
5. 新增 `episode/tool_calls_per_turn`、`episode/multi_call_ratio`、
   `env/multi_call_partial_failure_rate` 和 `tool/fill_formula_ratio`，并同步到
   train/val trainer 指标和离线 rollout summary。

当前编辑节点只有 Python 3.6，无法导入项目使用的 Python 3.10+ 类型语法，也没有
pytest/openpyxl。因此这里只完成了 shell 语法、diff 和静态代码检查；必须在训练镜像
运行完整 `rl/tests`、formula worker smoke、399/64 validation smoke 和 5-step GRPO
后，才能把本节状态从“代码完成”升级为“运行验收完成”。

## 21. 2026-09-24：专用评测工具实施进展

为避免继续手工拼接 `trainer.val_only=True` 命令和人工读取聚合日志，本轮新增：

1. `rl/scripts/submit_spreadsheetbench_eval.sh`
   - 输入一个或多个 `LABEL=MODEL`；
   - 只接受带 `config.json` 和模型权重的 Hugging Face 目录，不接受 optimizer resume
     checkpoint；
   - 默认完整评测 399 条，最多并发 64 个 validation actor；
   - 固定确定性 validation、Spreadsheet-RL verified split、`native_basic` 和 compact
     history；
   - 每个模型单独建立 run 目录并顺序提交，前一个失败时停止后续评测。
2. `rl/scripts/compare_spreadsheetbench_evals.py`
   - 要求 evaluation-only run 中每个 task ID 恰好出现一次；
   - 检查 Base/Candidate task ID 集合完全一致；
   - 输出 success count/rate、paired gain/loss、Python error、write success 和 multi-call
     partial failure；
   - 支持聚合 JSON/Markdown 和逐任务 JSONL 结果落盘。

当前 2-node Ray state 上的 dry-run 已确认 launcher 将 `VAL_SIZE=399`、
`VAL_CONCURRENCY=64` 和 `trainer.val_only=True` 传入 Ray runtime env，且 checkpoint
预检查可以识别 `global_step_50/actor/huggingface`。完整 pytest 和真实 399-task job 仍需
在训练镜像执行后，才能关闭评测工具的运行验收项。

下一步顺序保持不变：先用该工具完成新代码下 Base/step 50 的 399 条 paired evaluation；
若 success rate 不低于 Base，再按 badcase 频次进入 `format_range`、行列操作和 sheet 管理
工具迁移。数据 split ID/hash 重叠审计和 manifest 数据 hash 仍是评测基础设施的剩余项。

### 21.1 评测工具代码审查后的修正

后续审查发现第一版 launcher/comparison 仍有五个风险：配置不同仍可比较、LibreOffice
`eval_error` 被算作策略失败、val-only 创建 128 个无用 train actors、缺少置信区间，以及
multi-call 写工具指标实际按 turn 而非 call 统计。现已完成对应修正：

1. comparison 对 manifest 中的关键行为配置执行严格一致性检查；缺字段同样视为不可比。
2. evaluator error 默认中止正式比较；诊断模式可显式放行，但从有效任务和 paired 分母排除。
3. 输出 success-rate Wilson 95% CI、paired delta 95% CI、exact McNemar p-value、
   evaluator error 数和任务 ID。
4. 新增逐写调用、逐写 turn 和 episode 级错误指标，原
   `write_tool_success_rate` 仅作为向后兼容字段保留。
5. `trainer.val_only=True` 时 SpreadsheetBench route 只创建 validation actor pool；对应变更
   保存于 `rl/integration/verl_agent_spreadsheetbench_eval.patch`，新 checkout 可通过
   `fetch_verl_agent.sh` 复现。
6. manifest 升级为 version 2，并补齐 thinking、validation sampling、vLLM rollout 长度等
   比较字段。

当前宿主环境已完成 shell 语法、Python 语法、integration patch 可应用性以及 comparison/
rollout-summary 的无 pytest 行为测试。完整 `PYTHONPATH=.:rl:third_party/verl-agent python -m
pytest -q rl/tests` 仍需在 Python 3.11 训练镜像运行。

### 21.2 2026-09-24：multi-call 诊断强化

原有指标只能看到“模型输出了几个调用”和“该 turn 是否部分失败”，无法区分完整执行、
因失败提前停止、因 `submit` 终止以及究竟跳过了几个调用。本轮补齐以下三级诊断：

1. worker 在每个 turn 记录 projected/executed/skipped call 数、首个失败位置、执行比例、
   是否完整执行、是否全部成功、是否短路，以及 `completed`、`terminated`、`truncated`、
   `<tool>_failure`、`completed_with_failure` 等停止原因。
2. manager 和 verl rollout 将数值字段传入 train/val batch。WandB 新增
   `episode/projected_tool_calls_per_turn`、`episode/projected_multi_call_ratio`、
   `env/multi_call_completed_rate`、`env/multi_call_all_success_rate`、
   `env/multi_call_short_circuit_rate`、`env/multi_call_skipped_calls_mean` 和
   `env/multi_call_executed_fraction_mean`；发生失败时另记
   `env/multi_call_failure_index_mean`。这些 outcome 只在 projected multi-call turn
   上计算，避免单调用 turn 稀释结果。
3. 离线 summary 新增停止原因和失败位置分布；
   `summarize_spreadsheetbench_rollouts.py --multi-call-events-output FILE.jsonl`
   可以导出逐任务、逐 turn 的调用序列、短路位置和失败工具，直接用于 badcase 排查。

评测 comparison 同步展示 multi-call 完整执行率和短路率，并在 JSON 中保留全部新增聚合。
`verl_agent_spreadsheetbench_metrics.patch` 和 tracking patch 已按实际应用顺序重建，并已在
干净的 pinned verl-agent checkout 上验证整套 patch 可顺序应用。当前宿主完成 Python 3.8
语法检查和离线聚合行为检查；Python 3.11 训练镜像中的完整 `rl/tests` 及真实 rollout
验收仍待执行。
