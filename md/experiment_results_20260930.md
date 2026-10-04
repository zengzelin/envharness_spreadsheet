# Spreadsheet Agent 实验结果汇总（截至 2026-09-30）

本文统一记录 njceph5 与 sgceph1 上已经完成的 Spreadsheet-RL 训练和
Verified-399 独立评测。checkpoint 选择以 399 条任务的 workbook success 为主，
`test_score` 只作为部分正确和执行惩罚的辅助指标。

## 1. 可比性边界

以下三组结果分别使用了不同的代码 commit 或 submit 协议，只能在各自组内比较：

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

## 6. 下一步

1. 把 njceph5 的 50K/100K mutation 防护同步到 sgceph1，并补充 limit-reject 指标。
2. 在同一 hardened harness 下重评 Base、Step 60 和 Step 100；不要只提高 actor timeout。
3. 使用 `compare_spreadsheetbench_evals.py` 输出逐任务 paired gain/loss、McNemar 检验和
   badcase JSONL。
4. 对比 Step 60 新做对、后续 checkpoint 又做错、以及所有 checkpoint 始终失败的任务。
5. 下一轮训练不应只延长步数；优先修正公式/范围语义、重算协议使用率和大范围 mutation
   分块策略。
