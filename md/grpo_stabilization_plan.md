# Spreadsheet GRPO 稳定化结论与 TODO

更新日期：2026-09-19

## 1. 全量 399 条验证集结论

Base model 与 `global_step_5` 使用完全相同的 399 个
`test_verified_hermes.parquet` 任务进行确定性评测，两个 Ray job 均正常结束。

| 指标 | Base model | global_step_5 | 变化 |
| --- | ---: | ---: | ---: |
| 成功任务数 | 122/399 | 113/399 | -9 |
| `val/success_rate` | 0.3058 | 0.2832 | -0.0226 |
| `val/text/test_score` | 0.1940 | 0.2137 | +0.0197 |
| native parser 有效率 | 0.7457 | 0.9440 | +0.1983 |
| parser 无效率 | 0.1035 | 0.0502 | -0.0533 |
| Python 错误率 | 0.1960 | 0.1143 | -0.0817 |
| 写工具成功率 | 0.5824 | 0.4602 | -0.1222 |
| 平均工具调用数 | 8.401 | 9.098 | +0.697 |
| 撞到 episode 上限 | 347/399 | 362/399 | +15 |

逐任务比较中，两者共同成功 92 条，仅 Base 成功 30 条，仅 step 5 成功 21 条。
McNemar 精确检验约为 `p=0.262`。因此不能证明 step 5 显著劣于 Base，但也没有证据表明
它提高了任务完成率。此前固定 32 条评测中 step 5 的 `9/32` 对 Base 的 `7/32` 属于
小样本波动，不能用于选择 checkpoint。

`val/text/test_score` 与 `val/success_rate` 方向相反，是因为前者包含 Python 错误和无效
动作等 shaping penalty。step 5 改善了动作格式和代码可执行性，因而 shaped reward 上升；
但 workbook 写入准确性、提交率和最终成功率没有同步提高。后续 checkpoint 选择以固定
验证集上的 `val/success_rate` 和 workbook score 为主，shaped reward 只作训练诊断。

## 2. 本轮代码修改

### 固定在线 validation

- train worker 每次 reset 继续按 `seed_stride` 前进，覆盖不同训练任务；
- validation worker 不再推进 seed，同一次训练中的每个评测点复用完全相同的任务集合；
- 保留 `env.seed + 1000` 作为 validation 起始 seed，训练集和验证集仍由各自 parquet
  split 决定。

这项修改解决旧实验中 step 5/10/15/20 在线 validation 任务发生轮换、曲线不能直接
比较的问题。修改之后仍需在日志或 rollout 中核对各评测点 task ID 集合。

### 显式优化和惩罚参数

启动脚本新增并记录以下参数：

```text
ACTOR_LR                       默认 1e-6
USE_INVALID_ACTION_PENALTY     默认 True
INVALID_ACTION_PENALTY_COEF    默认 0.1
```

默认值保持旧实验行为，避免已有命令静默改变。新实验必须在 `run_manifest.json` 中检查
实际值。提交脚本会将这些变量转发到 Ray runtime environment。

### 训练预算检查

当前数据生成方式每个 epoch 只有一个 train batch。启动脚本会拒绝
`EPOCHS < TOTAL_TRAINING_STEPS`，避免目标 80 step 但因 `EPOCHS=20` 在 20 多步提前退出。

### Reward 分解指标

环境终局新增以下字段，并传入 train/validation 指标：

```text
reward/workbook_score_mean
reward/execution_penalty_mean
reward/env_total_mean
reward/invalid_action_penalty_mean
```

其中 `workbook_score` 是 evaluator 的最终工作簿得分，`execution_penalty` 是 episode 内
Python 等执行惩罚累计值，`env_total` 是两者之和；invalid-action penalty 在 trainer
阶段单独施加。当前只增加可观测性，不修改 reward 数值和优化目标。

## 3. 下一组诊断实验

不要从 step 5 继续训练。以原始 Base model 为相同起点，先跑 10 step 的低学习率对照：

```bash
RUN_TS=$(date +%Y%m%d_%H%M%S)
EXP_NAME="spreadsheetrl_grpo_stable_lr3e7_${RUN_TS}"

MODE=diagnostic \
MODEL=/mnt/gemininjceph5/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_zelinnzeng/llm_model/Qwen3-4B-Thinking-2507 \
SPREADSHEETBENCH_DATA_FORMAT=spreadsheet_rl \
SPREADSHEET_RL_DATA_ROOT="$PWD/experiments/spreadsheetbench/data/Spreadsheet-RL" \
SPREADSHEET_RL_TRAIN_FILE=train_hermes.parquet \
SPREADSHEET_RL_VAL_FILE=test_verified_hermes.parquet \
SPREADSHEETBENCH_TOOL_SET=native_basic \
NNODES=1 N_GPUS_PER_NODE=8 TP=2 \
TRAIN_BS=16 VAL_BS=64 GROUP_N=8 PPO_MINI_BS=16 \
MAX_STEPS=10 MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=16384 \
ACTOR_LR=3e-7 KL_LOSS_COEF=0.005 \
USE_INVALID_ACTION_PENALTY=True INVALID_ACTION_PENALTY_COEF=0.1 \
TOTAL_TRAINING_STEPS=10 EPOCHS=10 TEST_FREQ=2 SAVE_FREQ=2 VAL_BEFORE=True \
RUN_TS="$RUN_TS" EXP_NAME="$EXP_NAME" WANDB_NAME="$EXP_NAME" \
bash rl/scripts/submit_spreadsheetbench_grpo.sh
```

中间 validation 的 64 条任务现在固定不变。Base、step 2/4/6/8/10 需要使用相同的
64 条任务比较；实验结束后，再用 Base 和候选 checkpoint 各跑完整 399 条验证集。

## 4. TODO

### P0：本轮完成和验证

- [x] train reset 轮换任务，validation reset 固定任务；
- [x] 学习率和 invalid-action penalty 参数化并写入 manifest；
- [x] 检查 epoch/step 预算冲突；
- [x] 导出 workbook score、执行惩罚、环境总 reward 和无效动作惩罚；
- [ ] 在训练镜像执行完整 `rl/tests`；
- [ ] 用两次连续 validation 核对 task ID 集合完全一致；
- [ ] 跑上述 10-step 低学习率诊断实验。

### P1：根据诊断实验决定 reward 修改

- [ ] 比较 `workbook_score`、`execution_penalty`、invalid-action penalty 与最终
  `success_rate` 的相关性；
- [ ] 统计 `returncode=0` 但 workbook score 为 0 的轨迹；
- [ ] 将写工具错误细分为公式被拒绝、shape 不匹配、sheet/range 不存在和 workbook
  损坏；
- [ ] 若 workbook score 没有改善但 penalty 持续下降，降低 shaping penalty 权重或改为
  只做监控，避免优化目标被格式正确性主导。

### P2：环境能力改进

- [ ] 评估 `write_range` 是否需要显式支持公式，而不是将公式一律拒绝为 static-value
  error；
- [ ] 增加提交前的轻量 workbook 自检，减少未提交和撞到 turn 上限；
- [ ] 研究单轮多工具调用，降低细粒度工具导致的 episode 膨胀；
- [ ] 建立固定 64 条 fast-val 清单和独立 399 条 paired-eval 汇总脚本。

## 5. 验收标准

本轮工程修改的验收标准不是 success rate 立即提高，而是：

1. 同一次训练所有 validation 使用同一组 task ID；
2. W&B 同时出现四项 reward 分解指标；
3. 启动日志和 manifest 能还原学习率、KL 和惩罚配置；
4. `EPOCHS < TOTAL_TRAINING_STEPS` 时启动立即失败并给出明确错误；
5. 完整 399 条评测继续以 success count/rate 为 checkpoint 主指标。
