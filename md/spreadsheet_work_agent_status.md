# Spreadsheet Work Agent 当前实现状态

**更新时间：2026-10-10**

本文是 Spreadsheet Work Agent 和 Spreadsheet Agent Harness 的当前状态索引。早期实施计划
中的细粒度复选框保留当时的执行过程，可能没有随着后续提交逐项回填；判断功能是否
已经落地时，以本文、当前代码和可核对的实验产物为准。

状态分为四类：

- **已实现**：当前 `main` 中存在代码、测试或正式实验产物；
- **已实现，待独立验证**：代码和单元测试
  已存在，但缺少计划要求的真实集群验收或当前 checkout 的完整测试记录；
- **待完成**：当前没有满足目标的实现或实验；
- **历史或已替代**：保留用于解释演进，不再作为当前基线或 TODO。

## 已实现

| 能力 | 当前事实 | 代码或证据 |
| --- | --- | --- |
| 有状态工作簿环境 | 每个 Episode 创建独立工作目录，支持 `reset/step/observe/evaluate` 和官方工作簿比较。 | `envharness/bridges/spreadsheetbench/bridge.py`、`online_judge_eval.py` |
| 结构化检查 | 已提供 Sheet 枚举、Range 读取和内容查找。 | `read_tools.py`、`rl/tests/test_spreadsheetbench_bridge_execution.py` |
| 原生值与公式修改 | 已提供 `write_range`、`clear_range`、`fill_formula`，包含范围检查、公式平移和原子保存。 | `write_tools.py`、Bridge Execution Tests |
| 格式与结构修改 | 已提供 `format_range`、行列删除，以及 Sheet 创建、重命名、复制、移动和隐藏。 | `structure_tools.py`、Bridge Execution Tests |
| 事务式 Python 兜底 | `run_python` 执行失败、超时或输出 XLSX 损坏时回滚；控制台不打印完整源码。 | `bridge.py`、`process_utils.py`、`test_spreadsheetbench_bridge_execution.py` |
| 公式与工作簿校验 | 检查损坏 XLSX、非法公式和 `#REF!`；修改后要求重新校验当前版本。 | `tools.py`、`python_preflight.py`、`bridge.py` |
| 重算并读取 | 支持对临时副本运行 LibreOffice，并有每个 Episode 的调用上限和超时。 | `recalc_tools.py`、Bridge Execution Tests |
| 有界多工具调用 | 每轮最多接纳 4 个有序调用；超量截断并反馈；失败写入会短路后续修改。 | `rl/envharness_rl/spreadsheetbench/projection.py`、`envs.py`、Projection/Env Tests |
| 修改预算 | `write_range`、`clear_range`、`fill_formula`、`format_range` 单次最多 50,000 Cells，单轮累计最多 100,000 Attempted Cells；Python 兜底和结构操作不计入 Cell Counter。 | `write_tools.py`、`structure_tools.py`、`envs.py` |
| Badcase 可观测性 | 已实现 light/full 诊断、逐任务 `badcases.jsonl`、聚合 Summary、成对失败转移和 Validation/W&B 指标。 | `badcase_diagnostics.py`、`rollout_summary.py`、`summarize_spreadsheetbench_rollouts.py`、提交 `18f9142` |
| 训练任务 Shuffle | 训练使用确定性任务排列；Validation 顺序不变；可用 Seed 或 `off` 控制。 | `dataset.py`、`envs.py`、提交 `45426b7` |
| Full-399 评测 | `VAL_SIZE` 与 `VAL_CONCURRENCY` 解耦，支持顺序 Checkpoint 评测、独立 Ray 集群和任务级 Paired Comparison。 | `submit_spreadsheetbench_eval.sh`、`compare_spreadsheetbench_evals.py`、`runs/parallel_eval/shufflefix_20261004_202150` |
| 当前最佳结果 | Base `94/399`，Step100 `119/399`，Paired Exact McNemar `p=0.005228`。 | `md/experiment_results_20260930.md`、`md/rollout_badcase_report.md` |

## 已实现，待独立验证

| 项目 | 已有内容 | 仍缺少的验收 |
| --- | --- | --- |
| Spreadsheet-RL 懒加载 | Parquet 原始行缓存、单行物化、Seed/Instance ID 选择及 Reset 阶段日志已经实现并有单元测试。 | 在目标训练镜像中执行独立 128 Actor、180 秒门槛的 Scale Smoke，并保存可核对报告。 |
| 当前代码完整测试集 | 相关 Bridge、Dataset、Projection、Manager 和 Script Tests 已存在，用户侧最近一次记录为修复前 `212 passed, 1 failed`。 | 在当前 `main`、Python 3.11 训练镜像重新运行完整 `rl/tests` 并保存最终通过记录。 |
| 真实重算行为 | 正式评测已使用 LibreOffice Evaluator，`recalculate_and_read` 也有实现和回归测试。 | 单独保存真实工作簿的 Recalc Smoke、超时和公式读回验收记录。 |

这一类表示“不应重新设计实现”，但在对外宣称规模或平台可靠性前仍需要补充运行证据。

## 待完成

| 优先级 | 项目 | 完成标准 |
| ---: | --- | --- |
| P0 | 第二训练 Seed 复现 | 在相同 Commit、数据、Harness 配置和 full-399 协议下运行新的 Shuffle Seed；预先定义 Checkpoint 选择规则。 |
| P0 | Run Manifest 溯源 | 记录 Dirty Worktree Flag、Diff Checksum、Dataset Hash、Task ID Fingerprint 以及 Prompt/Tool Schema Version。 |
| P1 | 语义 Observation Projection | 根据 Instruction、`answer_position`、表头、公式和非空边界选择相关上下文，并完成固定任务 Projection On/Off A/B。 |
| P1 | 公式语义正确性 | 对 `execution_clean_score_zero + formula_result_mismatch` 任务建立人工核对样本和可重复评测；不能只降低 Tool Error。 |
| P1 | Checkpoint 选择 | 使用预先声明的 Early Stopping 或更密集的 80-110 保存/评测规则，避免事后只挑最佳点。 |
| P2 | 提高重算使用率 | 在不增加无效 Wall Time 的前提下，提高真正需要公式重算任务的“修改—重算—检查—提交”闭环率。 |

## 历史或已替代

| 项目 | 当前处理 |
| --- | --- |
| `Base=122/399` Gate | 来自旧协议，不再作为当前 Stop/Go 基线；当前统一 full-399 对照为同组重新评测的 Base。 |
| 使用 Fast-eval 64 排名模型 | 仅用于训练健康检查，不用于最终 Checkpoint 选择；正式结论使用固定 full-399。 |
| Python-only 工具协议 | 保留兼容和消融实验；当前正式 Work Agent 配置使用 `native_basic`。 |
| 无 Shuffle 的 SG1 Run | 保留为历史对照，不能与不同 Base/Config 的 Run 直接做因果归因。 |
| 逐个迁移 `fill_formula`、Multi-call、格式/结构/重算的旧 TODO | 相应功能已通过后续五阶段迁移和 Hardening 落地；旧 Checkbox 只记录原实施顺序。 |
| 把 Tool Success 当作最终能力 | 已被 full-399 Badcase 证据否定；官方 Workbook Success 始终是主指标。 |

## 当前实验结论

1. Spreadsheet Agent Harness 已经把工具错误、损坏 workbook、无限调用和旧 revision 提交
   变成可约束、可诊断的失败。
2. 当前最佳单次实验从 `94/399` 提高到 `119/399`，但第二 seed 尚未复现，且 Step110
   回落到 `101/399`，不能声称稳定单调提升。
3. Step100 的 `tool_error` 从 178 降到 100，但 `formula_result_mismatch` 仅从 173 降到
   169，`execution_clean_score_zero` 从 127 增加到 180。下一瓶颈是任务和公式语义，而
   不是继续只优化 parser 或工具执行率。
4. Badcase 标签只用于观测；在完成离线相关性验证和独立 A/B 前，不直接加入 reward。

## 文档入口

- 项目首页：`SPREADSHEET_WORK_AGENT.md`
- 设计边界：`md/spreadsheet_work_agent_harness_design.md`
- 训练与评测操作：`rl/README.md`
- 完整实验记录：`md/experiment_results_20260930.md`
- Badcase 证据：`md/rollout_badcase_report.md`
