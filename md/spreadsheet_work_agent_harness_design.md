# Spreadsheet Work Agent / Agent Harness 设计说明

## 1. 项目定位

项目对外名称为 **Spreadsheet Work Agent**。它是一个能够理解自然语言表格任务、检查
工作簿、执行多步修改、重新计算并验证最终结果的长程工作 Agent。

支撑 Agent 可靠运行和训练的系统称为 **Spreadsheet Agent Harness**。Harness 不代表
另一套 Agent，也不代表环境改写；它是 Agent 与真实工作簿之间的运行时和验证层。

项目主叙事只包含这两个概念：

```text
Spreadsheet Work Agent（表格工作智能体）
        |
        v
Spreadsheet Agent Harness（表格智能体运行框架）
        |
        v
工作簿沙箱 + 可信评测器
```

本文及后续项目首页不把 Setup、Rules、Link、EnvRigger 或技能归纳描述为当前训练系统的
组成部分。它们属于仓库的其他研究方向，不是本项目当前结果的来源。

## 2. 核心问题

电子表格工作不是一次文本生成。Agent 必须在一个有状态的文件上完成长程交互，并同时
处理以下风险：

- 找错 sheet、range 或表头；
- 把公式写成静态值，或生成语义错误的公式；
- 一次修改过多单元格；
- Python 或原生工具执行失败后留下半成品；
- 生成损坏的 xlsx；
- 修改后没有重新计算和检查；
- 校验之后继续修改，却直接提交旧校验对应之外的版本；
- 工具调用过长，耗尽 episode 但没有提交；
- 工具执行成功，但最终工作簿语义仍然错误。

Spreadsheet Agent Harness 的目标不是替 Agent 解题，而是让这些风险可控制、可验证、
可观测，并为强化学习提供可信的任务级奖励。

## 3. 系统架构

```text
自然语言任务 + 输入工作簿
                    |
                    v
              观察与检查
      预览 / list_sheets / inspect_range / find_cells
                    |
                    v
             Work Agent 策略
                    |
                    v
             有界动作路由
       解析 / 投影 / 最多 4 次调用 / 每轮 100K 单元格
                    |
                    v
           事务式工作簿工具
 run_python / 写入 / fill_formula / 格式 / 结构
                    |
                    v
              重算与校验门禁
       LibreOffice / 公式检查 / 版本追踪
                    |
                    v
             提交与可信任务评测器
                    |
          +---------+----------+
          |                    |
          v                    v
     GRPO 任务奖励          轨迹 / badcase
```

### 3.1 观察（Observation）

Harness 向 Agent 提供任务要求、输入和输出路径、答案位置、sheet 预览以及工具结果。Agent
可以使用结构化读取工具继续定位真实数据，避免把整个工作簿塞入上下文。

当前 observation 使用固定 sheet 预览和字符上限，属于可用基线；按任务语义选择 sheet、
range、公式和表头仍是后续优化方向，不能描述为已经完成。

### 3.2 动作（Actions）

当前 `native_basic` 工具集覆盖：

- `list_sheets`、`inspect_range`、`find_cells`；
- `write_range`、`clear_range`、`fill_formula`；
- `format_range`；
- `delete_rows`、`delete_columns`、`manage_sheet`；
- `recalculate_and_read`；
- `run_python`、`validate_workbook`、`submit`。

`run_python` 是处理未被结构化工具覆盖的复杂操作的兜底能力，不是推荐的默认路径。

### 3.3 安全约束

- `run_python` 和原生 mutation 使用事务式写入；失败或输出损坏时回滚。
- `write_range`、`clear_range`、`fill_formula` 和 `format_range` 这四类计数型
  cell-addressed native operation 单次最多影响 50,000 个单元格。
- 同一模型 turn 中，上述计数型操作累计最多尝试修改 100,000 个单元格；`run_python`
  和结构操作不在该 cell counter 内。
- 每个 turn 最多接纳 4 个工具调用，超出的调用被截断并反馈给 Agent。
- 公式写入、`#REF!`、损坏 workbook 和非法 xlsx 在提交前被检查。
- 控制台日志不输出完整 Python 源码，trajectory 保留动作以便训练诊断。

### 3.4 校验协议

Harness 跟踪 workbook revision。修改后必须重新调用 `validate_workbook`；校验成功后如果
继续修改，旧校验立即失效。只有当前 revision 已通过校验时才允许 `submit`。

公式任务可通过 `recalculate_and_read` 在临时副本上调用 LibreOffice，读取公式重算后的
值。最终任务奖励仍由官方 workbook evaluator 决定，而不是由工具成功、格式检查或
badcase 标签替代。

### 3.5 训练与可观测性

训练使用 verl-agent GRPO，同一个任务可以产生多条 rollout。Harness 记录：

- 投影和执行的工具序列；
- 工具错误、短路、截断和 mutation 预算；
- 是否完成校验和提交；
- workbook success、test score 和任务级 reward；
- 公式、值、格式、sheet/range 和 execution-clean-zero 等 badcase 证据。

Badcase 标签用于解释失败，不直接改变官方 reward。

## 4. 当前结果及声明边界

在 399 条固定验证集上的 NJ5 shufflefix 实验中：

| Checkpoint | 成功任务 | 成功率 | 测试得分 |
| --- | ---: | ---: | ---: |
| Base | 94/399 | 23.56% | 0.1573 |
| Step 100 | 119/399 | 29.82% | 0.1930 |

Step 100 相对 Base 净增加 25 个成功任务，paired exact McNemar
`p=0.005228`。与此同时，`tool_error` 从 178 降到 100，疑似写错 sheet/range 从 44
降到 29。

该结果只能描述为当前最强的单次完整评测结果，不能描述为已跨 seed 复现的稳定提升：

- Step 100 是多个 checkpoint 中事后选出的最佳点；
- 第二个训练 seed 尚未完成；
- Step 110 回落到 101/399；
- 公式结果错误从 173 仅降到 169；
- execution-clean score-zero 从 127 增加到 180。

因此，当前证据支持“Agent 更熟练地遵守工具协议，并在一次完整实验中提高成功率”，但
不支持“工具执行正确即可解决表格任务”或“RL 提升已经稳定复现”。

## 5. 项目展示原则

项目首页重点展示：

1. 长程 spreadsheet work 的困难；
2. Agent 的 inspect-edit-recalculate-validate-submit 工作流；
3. 事务安全、预算和验证协议；
4. 可信 evaluator 与任务级 GRPO；
5. badcase 可观测性；
6. 完整 399 条评测中有边界的实验结论。

项目首页不展示：

- 已经被淘汰的启动参数尝试；
- fast-eval 波动和无结论的中间实验；
- 与当前训练无关的环境改写或技能归纳路线；
- 没有完成实现或独立验证的能力；
- 把 parser/tool 指标改善等同于 workbook success 的结论。

## 6. 后续路线

### P0：项目化和可复现入口

- 建立独立项目首页和架构说明；
- 给出训练、完整评测和 badcase 分析的最短可复现命令；
- 将历史研究文档与当前操作手册分开；
- 清理已实现但仍显示为未完成的历史 checkbox。

### P1：语义 Observation Projection

- 根据 instruction 和 `answer_position` 选择相关 sheet/range；
- 优先展示表头、公式、非空边界和异常单元格；
- 显式说明被省略内容以及取回方式；
- 用固定任务做 projection-on/off A/B，而不是直接进入长训练。

### P2：实验可追溯性

- manifest 增加 dirty flag 和 diff checksum；
- 记录数据文件 hash、task ID fingerprint 和 tool/prompt version；
- 完成第二个训练 seed；
- 使用预先定义的 checkpoint selection 或 early stopping。

### P3：语义正确性

- 优先分析 execution-clean score-zero 和公式结果错误；
- 改进 inspect 与 recalc 的使用提示；
- 评估 verification-aware curriculum；
- 任何新增信号先做离线相关性和 A/B，再考虑加入 reward。
