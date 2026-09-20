# Spreadsheet Agent 独立评测方案与当前状态

更新日期：2026-09-19

## 0. 2026-09-19 全量 Verified-399 评测结果

已经完成 Base model 与 `global_step_5` 在同一组 399 条
`test_verified_hermes.parquet` 任务上的确定性评测。两个评测的 task ID 集合完全一致，
各 399 条且无重复，Ray job 均成功退出。

| 模型 | 成功数 | success rate | shaped test score |
| --- | ---: | ---: | ---: |
| Base Qwen3-4B-Thinking-2507 | 122/399 | 0.3058 | 0.1940 |
| global_step_5 | 113/399 | 0.2832 | 0.2137 |

step 5 的 parser 和 Python 错误指标明显改善，但最终成功任务减少 9 条。`test_score`
上升而 success rate 下降，说明 shaping penalty 与真实 workbook 完成率存在偏离。完整分析
与后续稳定化 TODO 见 `md/grpo_stabilization_plan.md`。

## 1. 当前结论

当前代码已经具备独立评测所需的底层能力，但还没有独立的评测启动脚本：

- `dataset.py` 能读取 Spreadsheet-RL parquet 中的
  `reward_model.ground_truth` 和 `extra_info`；
- Spreadsheet-RL 任务目录中的 `output.xlsx` 被当作初始工作簿，
  `target.xlsx` 被当作 reward target；
- verl trainer 支持 `trainer.val_only=True`，会先加载模型或 checkpoint，执行一次
  validation 后退出；
- 当前 `run_spreadsheetbench_grpo.sh` 可以通过 `EXTRA_HYDRA` 传入
  `trainer.val_only=True`；
- 当前缺少专用 `submit_spreadsheetbench_eval.sh`、checkpoint 批量评测和任务重叠审计。

因此现在可以进行一次性独立评测，但正式实验前应补一个专用 launcher，避免训练参数和
评测参数混在一起。

## 2. 数据现状

数据已于 2026-09-14 复制并解压到本实验仓库：

```text
experiments/spreadsheetbench/data/Spreadsheet-RL
```

相邻的 `../Spreadsheet-RL` 是代码仓库，不包含 parquet 和 workbook 数据。官方数据不在
GitHub 代码仓库中，而在 Hugging Face dataset `Spreadsheet-RL/Spreadsheet-RL`。官方
数据包含 parquet split、约 1.2 GB 的 `spreadsheets.zip`，解压后包含任务目录及
`instruction.json`、`output.xlsx`、`target.xlsx`。

数据来源是以下已有的完整下载和解压副本：

```text
/apdcephfs/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/
group_semantic_video/group_semantic_video/user_zelinnzeng/envharness/
experiments/spreadsheetbench/data/Spreadsheet-RL
```

目标目录已经包含 `train_hermes.parquet`、`test_hermes.parquet`、
`test_verified_hermes.parquet`、`test_2_hermes.parquet`、
`test_domain_hermes.parquet`、`spreadsheets.zip` 以及已解压的 workbook 目录。复制时先
传输归档和 parquet，再在目标文件系统解压，避免跨 Ceph 逐个复制大量小文件。

已完成的任务目录数量为：ExcelForum 5,925、SpreadsheetBench 2,722、
SpreadsheetBench-2 297、SpreadsheetBench-Verified 399、Domain 1,662。归档通过
`unzip -tq` 完整性检查，关键 parquet 和归档文件大小与源目录一致。

复现复制过程可使用：

```bash
SRC=/apdcephfs/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_video/group_semantic_video/user_zelinnzeng/envharness/experiments/spreadsheetbench/data/Spreadsheet-RL
DST=/apdcephfs/mnt/gemininjceph5/geminicephfs/mmsearch-luban-universal/group_semantic_video/group_semantic_video/user_zelinnzeng/envharness/experiments/spreadsheetbench/data/Spreadsheet-RL

mkdir -p "$(dirname "$DST")"
rsync -a --info=progress2 "$SRC/" "$DST/"
```

如果必须重新下载，官方方式是：

```bash
DST=/apdcephfs/mnt/gemininjceph5/geminicephfs/mmsearch-luban-universal/group_semantic_video/group_semantic_video/user_zelinnzeng/envharness/experiments/spreadsheetbench/data/Spreadsheet-RL
hf download Spreadsheet-RL/Spreadsheet-RL --repo-type dataset --local-dir "$DST"
unzip -q "$DST/spreadsheets.zip" -d "$DST"
```

集群镜像没有外网时，第二种方式会失败，应使用已有副本或内部镜像。

## 3. 什么才算独立评测

`grpo_spreadsheetbench_diagnostic_20260912_194504` 使用
`spreadsheetbench_verified_400` 同时产生训练和 validation 任务。因此，对该 checkpoint
再次运行 `test_verified_hermes.parquet` 不能直接视为严格 held-out，必须先审计任务 ID
和 workbook 是否与训练数据重叠。

建议分两种情况：

1. 评测当前 Verified-400 checkpoint：优先使用 `test_domain_hermes.parquet`，其次使用
   `test_2_hermes.parquet`；同时计算与 Verified-400 的 ID/hash 重叠率。
2. 后续使用 `train_hermes.parquet` 的 5,925 个 ExcelForum 任务训练：使用
   `test_verified_hermes.parquet` 作为主 held-out，并可增加 `test_2` 和 `test_domain`。

独立评测必须固定：

- base model 与 checkpoint 使用同一任务列表；
- `temperature=0`、`do_sample=False`；
- `Qwen3-4B-Thinking-2507`、Hermes parser、`native_basic`；
- `MAX_STEPS=15`、prompt 8192、response 16384，与当前 checkpoint 训练设置一致；
- 同一 EnvHarness commit、LibreOffice evaluator 和工具 schema；
- 报告成功数/总数、置信区间、evaluator error 数以及逐任务 paired delta。

## 4. 当前可用的临时评测命令

下面命令先完整评测 `test_2_hermes.parquet` 的 297 条任务。需要先启动 Ray，并将数据
复制到目标目录。当前 adapter 会按 `VAL_BS` 创建同等数量的环境 actor，因此不建议在
没有分片支持时直接设置 `VAL_BS=1662` 评测 Domain 全集。Domain 全量评测应在加入分片
与结果合并后进行；不能只把 `VAL_BS` 改小后把一个子集当作全量结果。

### Base model

```bash
RUN_TS=$(date +%Y%m%d_%H%M%S)
EXP_NAME="eval_base_qwen3_4b_test2_${RUN_TS}"

MODE=diagnostic \
MODEL=/mnt/gemininjceph5/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_zelinnzeng/llm_model/Qwen3-4B-Thinking-2507 \
NNODES=2 N_GPUS_PER_NODE=8 TP=2 \
TRAIN_BS=16 VAL_BS=297 GROUP_N=8 PPO_MINI_BS=16 \
MAX_STEPS=15 MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=16384 \
SPREADSHEETBENCH_TOOL_SET=native_basic \
SPREADSHEETBENCH_DATA_FORMAT=spreadsheet_rl \
SPREADSHEET_RL_DATA_ROOT="$PWD/experiments/spreadsheetbench/data/Spreadsheet-RL" \
SPREADSHEET_RL_TRAIN_FILE=train_hermes.parquet \
SPREADSHEET_RL_VAL_FILE=test_2_hermes.parquet \
VAL_BEFORE=True EXTRA_HYDRA="trainer.val_only=True" \
RUN_TS="$RUN_TS" EXP_NAME="$EXP_NAME" WANDB_NAME="$EXP_NAME" \
bash rl/scripts/submit_spreadsheetbench_grpo.sh
```

### Step 30 checkpoint

不需要恢复 optimizer；直接加载 checkpoint 保存的 Hugging Face 权重：

```bash
CKPT="$PWD/runs/grpo_spreadsheetbench_diagnostic_20260912_194504/ckpts/global_step_30/actor/huggingface"
RUN_TS=$(date +%Y%m%d_%H%M%S)
EXP_NAME="eval_step30_qwen3_4b_test2_${RUN_TS}"

MODE=diagnostic MODEL="$CKPT" \
NNODES=2 N_GPUS_PER_NODE=8 TP=2 \
TRAIN_BS=16 VAL_BS=297 GROUP_N=8 PPO_MINI_BS=16 \
MAX_STEPS=15 MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=16384 \
SPREADSHEETBENCH_TOOL_SET=native_basic \
SPREADSHEETBENCH_DATA_FORMAT=spreadsheet_rl \
SPREADSHEET_RL_DATA_ROOT="$PWD/experiments/spreadsheetbench/data/Spreadsheet-RL" \
SPREADSHEET_RL_TRAIN_FILE=train_hermes.parquet \
SPREADSHEET_RL_VAL_FILE=test_2_hermes.parquet \
VAL_BEFORE=True EXTRA_HYDRA="trainer.val_only=True" \
RUN_TS="$RUN_TS" EXP_NAME="$EXP_NAME" WANDB_NAME="$EXP_NAME" \
bash rl/scripts/submit_spreadsheetbench_grpo.sh
```

## 5. 下一步代码工作

1. 增加 `submit_spreadsheetbench_eval.sh`，明确区分 base/HF checkpoint 与训练 resume。
2. 增加数据审计脚本，输出 split 行数、有效任务数、缺失文件以及跨 split ID/hash 重叠。
3. 支持全量评测分片与合并，避免一次创建 1,662 个环境 actor。
4. 输出逐任务 JSONL 和 paired summary，而不只记录聚合 success rate。
5. 在 run manifest 中记录 dataset 文件 hash、任务 ID 列表、模型路径和 Git commit。

完成上述工作后，才将“独立评测”作为 checkpoint 选择依据。当前训练内的 32-task val
只用于健康检查和趋势观测。
