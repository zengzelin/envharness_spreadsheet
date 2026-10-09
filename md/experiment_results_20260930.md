# Spreadsheet Agent 实验结果汇总（截至 2026-10-09）

> 文件名保留 `20260930` 以避免破坏既有引用；本文内容已持续更新到 2026-10-09。

本文统一记录 njceph5 与 sgceph1 上已经完成的 Spreadsheet-RL 训练和
Verified-399 独立评测。checkpoint 选择以 399 条任务的 workbook success 为主，
`test_score` 只作为部分正确和执行惩罚的辅助指标。

## 1. 可比性边界

早期三组结果分别使用了不同的代码 commit 或 submit 协议，只能在各自组内比较：

| 组别 | 训练 run | 代码/协议 | 说明 |
| --- | --- | --- | --- |
| NJ-A | `20260926_155341` | commit `f206a2b`；历史评测命令显式关闭 submit gate | 旧 harness 对照 |
| SG-A | `20260922_105252` | commit `c3c03ee`；旧 manifest 未记录 gate | 五阶段迁移前后的早期 sg1 对照 |
| SG-B | `20260926_232503` | commit `a543f3d`；`REQUIRE_VALIDATION_BEFORE_SUBMIT=true` | hardened harness |

三组都使用 Qwen3-4B-Thinking-2507、`test_verified_hermes.parquet`、确定性 validation、
`MAX_STEPS=10`、8192 prompt、16384 response 和 399 条任务，但 Base success 分别是
17.04%、24.31% 和 22.56%。这说明 harness/commit 差异足以改变绝对分数，禁止跨组直接
计算 checkpoint 增益。

## 2. njceph5：`20260926_155341`

完整评测目录位于 njceph5 `runs/grpo_spreadsheetbench_eval_*_20260929_*`。每个成功 run
均产生 399 条 validation trajectory。

| 模型 | 成功数 | Success rate | Test score | 相对本组 Base |
| --- | ---: | ---: | ---: | ---: |
| Base | 68/399 | 17.04% | 0.1179 | - |
| Step 20 | 104/399 | 26.07% | 0.1640 | +9.02 pp |
| **Step 40** | **117/399** | **29.32%** | **0.2059** | **+12.28 pp** |
| Step 60 | 108/399 | 27.07% | 0.1734 | +10.03 pp |
| Step 80 | 105/399 | 26.32% | 0.1660 | +9.27 pp |
| Step 100 | 107/399 | 26.82% | 0.1786 | +9.77 pp |

结论：训练明确有效，Step 40 是该组最佳 checkpoint；Step 40 后进入平台期并回落。
Step 100 的 parser/multi-call 指标继续改善，但 workbook success 没有继续提高，说明后半程
主要瓶颈已经从工具协议转向范围、公式和任务语义正确性。

### 2.1 大范围 mutation 证据

从该训练日志可见：

- `format_range` 记录 794 次，最大合理范围 41,435 cells，没有超过 50,000；
- `fill_formula` 记录约 21,750 次，只有 6 次超过 50,000，均为 93,196 或 99,999 cells；
- 这些超大 fill 分别来自把横向公式错误向下填到第 100,000 行，以及远超评分范围的
  邮编列填充，属于范围判断错误；
- multi-call 100,000-cell budget 只拒绝了 4 个明显异常调用：两次 702,000-cell clear 和
  两次 1,048,568-cell fill，且累计值均从 0 开始，并非正常多工具累计误伤。

因此当前 `format_range=50,000`、`fill_formula=50,000`、单轮累计 100,000 的保护对现有
benchmark 合理。若未来需要支持真实几十万行任务，应优先增加分块写入，而不是恢复无界
单次 mutation。

## 3. sgceph1 早期对照：`20260922_105252`

完整评测目录为 `grpo_spreadsheetbench_eval_{base,step50,step100}_20260926_*`。

| 模型 | 成功数 | Success rate | Test score | 相对本组 Base |
| --- | ---: | ---: | ---: | ---: |
| Base | 97/399 | 24.31% | 0.1383 | - |
| Step 50 | 110/399 | 27.57% | 0.1353 | +3.26 pp |
| Step 100 | 110/399 | 27.57% | 0.1799 | +3.26 pp |

结论：训练使 success 增加 13 个任务，但 Step 50 到 Step 100 没有继续增加成功数。
Step 100 的 test score 上升主要反映部分正确度改善，不能解释为更多任务完全成功。

## 4. sgceph1 hardened 训练：`20260926_232503`

训练完整跑到 Step 100，每 5 步均保存 checkpoint。64 条在线 fast-eval 从 Step 0 的
11/64 上升到 Step 10 和 Step 100 的 18/64，但曲线波动明显，因此以下 399 条结果才是
checkpoint 选择依据。

| 模型 | 评测状态 | 成功数 | Success rate | Test score | 相对本组 Base |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | 完成 | 90/399 | 22.56% | 0.1476 | - |
| Step 10 | 完成 | 102/399 | 25.56% | 0.1183 | +3.01 pp |
| Step 20 | 完成 | 110/399 | 27.57% | 0.1566 | +5.01 pp |
| Step 40 | 完成 | 109/399 | 27.32% | 0.1473 | +4.76 pp |
| **Step 60** | **完成** | **114/399** | **28.57%** | **0.1747** | **+6.02 pp** |
| Step 80 | 完成 | 104/399 | 26.07% | 0.1305 | +3.51 pp |
| Step 100 | **失败** | - | - | - | - |

当前本组最佳 checkpoint 是 Step 60。Step 80 已出现回落，Step 100 不能用不完整结果参与
比较。

### 4.1 Step 100 评测失败

Step 100 评测在 `rollout_call=4`、turn 3 等待 600 秒后失败。64 个 actor 中 63 个返回，
唯一 pending actor 为：

```text
task_id=spreadsheetbench_verified/spreadsheet/1_399-14
action=run_python,format_range,format_range,validate_workbook
```

该任务要求把“等线”字体应用到整个 sheet。sgceph1 当时的代码仍是：

- `MAX_FILL_CELLS=300_000`；
- 没有 `MAX_FORMAT_CELLS`；
- 没有 `MAX_MULTI_CALL_MUTATION_CELLS`。

`format_range` 会逐单元格创建/复制样式并原子保存。模型给出过大的“整个 sheet”范围后，
单 actor 卡住并使全批 evaluation 失败。该 run 最终没有完整 399 条指标。普通公式或
`write_range` 参数错误是非致命工具反馈，不是 Job 失败原因。

## 5. 汇总结论

1. 三组实验都证明 GRPO 可以提高 Verified-399 success，但收益在 40-60 步附近达到峰值。
2. 当前旧协议最高结果是 NJ-A Step 40：117/399（29.32%）。
3. 当前 hardened 协议最高结果是 SG-B Step 60：114/399（28.57%）。
4. 训练后 parser、Python error 和 multi-call 执行通常改善，但这些指标继续改善不保证
   workbook success 同步增长。
5. 64 条 fast-eval 适合监控，不适合单独选择 checkpoint；正式选择必须跑固定 399 条。
6. submit gate、工具实现和 mutation 上限会改变 agent 行为，Base 与 checkpoint 必须在
   同一 checkout、同一 manifest 配置下重评。

## 6. 2026-10-01：SG1 无 shuffle 长训练

训练目录：

```text
sgceph1/runs/grpo_spreadsheetbench_diagnostic_20261001_221238
```

该 run 使用 hardened harness、固定 399 条确定性验证，但训练数据仍按 parquet 原顺序取样，
没有在训练开始时打乱 5,925 条任务。100 个训练 step 只覆盖前部约 1,600 个 task position，
因此任务顺序可能成为训练偏差来源。

| 模型 | 成功数 | Success rate | Test score | 相对本组 Base |
| --- | ---: | ---: | ---: | ---: |
| Base | 102/399 | 25.56% | 0.1952 | - |
| Step 10 | 96/399 | 24.06% | 0.1476 | -1.50 pp |
| Step 20 | 101/399 | 25.31% | 0.1684 | -0.25 pp |
| Step 30 | 104/399 | 26.07% | 0.1792 | +0.50 pp |
| Step 40 | 101/399 | 25.31% | 0.1531 | -0.25 pp |
| Step 50 | 95/399 | 23.81% | 0.1612 | -1.75 pp |
| Step 60 | 106/399 | 26.57% | 0.1855 | +1.00 pp |
| **Step 70** | **107/399** | **26.82%** | **0.1964** | **+1.25 pp** |
| Step 80 | 98/399 | 24.56% | 0.1610 | -1.00 pp |
| Step 90 | 103/399 | 25.81% | 0.1709 | +0.25 pp |
| Step 100 | 104/399 | 26.07% | 0.1610 | +0.50 pp |

Step 70 相对 Base 只净增 5 个成功任务；逐任务 exact McNemar `p=0.625`，没有统计显著
提升。工具错误减少，但 execution-clean score-zero 增加，说明协议执行变好没有转化为
工作簿语义正确性。该 run 不能作为“继续增加训练步数就会提升”的证据。

## 7. 2026-10-04：NJ5 shufflefix 长训练与全量评测

训练与评测目录：

```text
runs/grpo_spreadsheetbench_diagnostic_20261004_202150
runs/parallel_eval/shufflefix_20261004_202150
```

该 run 启动日志确认 `data.shuffle=true`、`train_shuffle_seed=0`。启动时 shuffle 修改已在
工作区生效，但尚未形成后来的提交 `45426b7`，因此 manifest 中记录的 commit 仍可能是
`18f9142`。这是实验溯源限制，不代表实际训练没有启用 shuffle。

| 模型 | 成功数 | Success rate | Test score | 相对本组 Base |
| --- | ---: | ---: | ---: | ---: |
| Base | 94/399 | 23.56% | 0.1573 | - |
| Step 10 | 107/399 | 26.82% | 0.1762 | +3.26 pp |
| Step 20 | 101/399 | 25.31% | 0.1588 | +1.75 pp |
| Step 30 | 109/399 | 27.32% | 0.1734 | +3.76 pp |
| Step 40 | 100/399 | 25.06% | 0.1484 | +1.50 pp |
| Step 50 | 110/399 | 27.57% | 0.1744 | +4.01 pp |
| Step 60 | 97/399 | 24.31% | 0.1523 | +0.75 pp |
| Step 70 | 98/399 | 24.56% | 0.1783 | +1.00 pp |
| Step 80 | 107/399 | 26.82% | **0.1966** | +3.26 pp |
| Step 90 | 101/399 | 25.31% | 0.1448 | +1.75 pp |
| **Step 100** | **119/399** | **29.82%** | 0.1930 | **+6.27 pp** |
| Step 110 | 101/399 | 25.31% | 0.1601 | +1.75 pp |

### 7.1 Step 100 的统计结论

Step 100 相对 Base：

- Base 失败、Step 100 成功：50 条；
- Base 成功、Step 100 失败：25 条；
- 净增加 25 条，success rate 提高 6.27 pp；
- exact McNemar `p=0.005228`。

这是当前统一 hardened/full-399 协议下最强的 checkpoint 结果。但 Step 100 是从多个
checkpoint 中事后选出的最佳点，存在 multiple-checkpoint selection 偏差；必须用第二训练
seed 或独立重复实验确认，不能仅凭一次 run 宣称 shuffle 已建立因果提升。

Step 110 回落到 101/399，相对 Step 100 净少 18 条，exact McNemar `p=0.02734`。因此
Step 100 后继续训练已出现显著退化，当前不支持盲目延长训练。

### 7.2 Badcase 转移

| 失败标签 | Base | Step 100 | Step 110 |
| --- | ---: | ---: | ---: |
| `tool_error` | 178 | 100 | 84 |
| `no_submit` | 24 | 16 | 19 |
| `formula_to_static` | 92 | 63 | 63 |
| `likely_wrong_sheet_or_range` | 44 | 29 | 27 |
| `formula_result_mismatch` | 173 | 169 | 182 |
| `execution_clean_score_zero` | 127 | 180 | 214 |

Step 100 明显减少工具错误、未提交、公式静态化和疑似错误范围，但公式结果错误几乎没有
下降；`execution_clean_score_zero_rate` 反而从 31.83% 升到 45.11%。Step 110 的
`tool_error` 继续降到 84，成功数却降到 101，且公式结果错误升到 182。这证明工具执行
指标只能解释协议熟练度，不能替代 workbook success；当前主要瓶颈已经是公式和任务语义。

### 7.3 训练动态

对比 Step 1-10 与 Step 102-111 的训练窗口：

- train success 均值约从 0.2477 升到 0.2814，但 81-90 窗口达到约 0.3241 后回落；
- valid action 从约 0.8245 升到 0.9431，parser invalid 从 0.1755 降到 0.0569；
- Python error 从约 0.0514 降到 0.0332，preflight reject 从 0.0300 降到 0.0056；
- multi-call all-success 从约 0.6760 升到 0.8275，short-circuit 从 0.2805 降到 0.1489；
- inspect-range 使用下降，recalc 和 submitted-after-recalc 仍很低；
- response length 与 prompt clipping 上升，entropy 从约 0.2248 降到 0.1036，KL loss 上升。

没有发现数值崩溃或梯度异常；模型主要学会了动作协议和工具执行，后期语义正确性进入平台
并出现策略收缩。训练内 64 条 fast-val 没有稳定超过 Base，而 full-399 的 Step 100 明显
更好，进一步确认 fast-val 只能做健康检查，不能用于最终 checkpoint 排名。

## 8. 当前状态与下一步

### 已完成

- [x] 50K 单次 mutation、100K 单轮累计预算和 limit-reject 指标已进入两边 `main`；
- [x] badcase schema、离线 JSONL/summary、validation/W&B 指标已实现；
- [x] Base 与候选 checkpoint 的 full-399 顺序/并行评测流程已跑通；
- [x] 训练集 shuffle 已实现并提交为 `45426b7`；
- [x] SG1 无 shuffle run 与 NJ5 shufflefix run 已完成全量 checkpoint 对比。

### 仍需完成

1. 用相同 commit、配置和 full-399 协议复现一个新的 shuffle seed，确认 Step100 提升；
2. 将 Step80、Step100 作为重点候选，同时增加更密集的 80-110 区间保存/全量评测或采用
   early stopping，避免错过峰值；
3. 优先改善公式语义、主动 inspect、重算后提交和执行成功但零分的问题，不再只优化
   parser/tool-error；
4. manifest 增加 dirty-worktree 标志、diff checksum、dataset fingerprint 和 task ID 列表；
5. 并行 Ray 评测继续使用独立 `CLUSTER_ID`、`RAY_STATE_FILE`、`HOSTFILE`、`LOG_FILE`
   和 `RUN_ROOT`；不得在活跃训练节点运行会调用 `ray stop --force` 的启动脚本。
