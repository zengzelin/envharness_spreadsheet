# Spreadsheet-RL 与当前 EnvHarness 方案对比及适配建议

> 整理日期：2026-09-10  
> 对比代码：Spreadsheet-RL `389be8c`；EnvHarness `180e258`  
> 当前固定基座模型：`Qwen3-4B-Thinking-2507`

## 1. 结论先行

我们之前确实讨论过 Spreadsheet-RL 与当前实现的差异，但当时主要完成了**数据层适配**，没有迁移 Spreadsheet-RL 的 native verl agent loop、原生表格工具、SandboxFusion 和 Windows Excel reward service。

当前方案不是“不能使用 Spreadsheet-RL 数据”。相反，EnvHarness 已经能够读取它的 parquet split，并正确映射：

- `reward_model.ground_truth`：任务目录相对路径；
- `extra_info`：任务元数据；
- `output.xlsx`：rollout 初始工作簿；
- `target.xlsx`：隐藏目标工作簿；
- `train_hermes.parquet`：训练集；
- `test_verified_hermes.parquet`：严格独立的验证集。

当前主要差距位于**环境和工具层**：Spreadsheet-RL 给模型提供结构化、低风险、低 token 成本的表格原生工具；EnvHarness 主要让模型生成完整 Python 代码。后者直接导致目前观察到的 Python syntax/runtime error、长输出、重复检查和低成功率。

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

当前只有三种动作：

- `run_python(code)`：用 fresh subprocess 运行自包含 Python；
- `validate_workbook()`：检查结果文件和目标 range 是否存在；
- `submit()`：结束 episode 并评分。

优点：能力上限高，开发量小，复杂任务可直接写任意 openpyxl/pandas 逻辑。

缺点：

- 模型经常生成 syntax error、NameError、API 使用错误和未保存文件；
- 检查、修改、验证都要重新生成代码，输出长度明显增加；
- fresh process 不保留 Python 变量，模型必须靠工作簿和文本历史恢复状态；
- 一个小参数错误会使整段操作失败；
- 难以把“工具选择错误”和“Python 实现错误”分开分析。

当前增加的 compact history、错误分类、Python error penalty 和 `validate_workbook` 能改善可观测性，但不能从根本上替代结构化工具。

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

### 10.2 尚未完成

- native verl `ToolAgentLoop`；
- token-aware Hermes parser 和原生 response mask；
- 单轮多个 tool call；
- 读工具并行、写工具串行；
- Spreadsheet-RL 全套表格原生工具；
- SandboxFusion；
- Windows Excel reward/recalculate 服务；
- async rate-limited reward queue；
- 原版 FSDP2、dynamic batching 和 fused kernel 性能栈；
- audit UI；
- Qwen3-Coder parser。当前固定使用 Qwen3 Thinking，因此最后一项不是近期需求。

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

这一版是原工具语义的受控子集，尚未支持多 range、regex、单轮多个
tool call 或任何写工具。集群测试和 smoke 通过后，再开始 A0/A1 对比。

下一步不应继续只调 learning rate、response length 或 max steps。当前 badcase 已经说明，模型经常在“生成和修复 Python”上浪费 rollout 预算。建议：

1. 以 Spreadsheet-RL split 作为正式数据线：`train_hermes` 训练、`test_verified_hermes` 验证；
2. 实现第一批只读 native tools：`list_sheets`、`inspect_range`、`find_cells`；
3. 增加 tool-set 配置开关，保证同一代码可以跑 A0/A1 对比；
4. 用 10 到 20 个 training steps 做固定 seed 诊断，不先追求长训练；
5. 若 valid action ratio、clip ratio 和 wall time 明显改善，再实现写工具；
6. 若结构化工具有效但 external manager 成为明显性能/协议瓶颈，再进入 native verl loop PoC；
7. 最终再做 Windows Excel reward 对齐测试，而不是立即把训练依赖到 Windows 服务。

## 15. 最终判断

- **数据能否用**：能，且 train/val split 和 `output.xlsx -> target.xlsx` 映射已经适配。
- **当前 codebase 是否需要放弃**：不需要。它仍然适合环境闭环、badcase 诊断和增量工具实验。
- **Spreadsheet-RL 最值得借鉴什么**：结构化表格工具、读写并发规则、工作簿锁与原子提交、工具输出限长。
- **是否立即迁移 native verl**：不建议。先用当前架构证明工具层收益，再做独立 PoC。
- **是否立即部署 SandboxFusion/Windows Excel**：不建议作为当前 success rate 优化的第一步；前者主要解决隔离，后者主要解决评分语义。
- **适配难度**：数据层低且已完成；只读工具中等；完整写工具中高；native verl loop 高；完整复现原版基础设施很高。
