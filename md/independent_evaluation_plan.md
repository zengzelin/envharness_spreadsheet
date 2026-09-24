# Spreadsheet Agent 独立评测方案与当前状态

更新日期：2026-09-24

## 0.1 2026-09-24 专用评测工具

当前已增加两个专用入口，正式评测不再手工拼接训练命令：

- `rl/scripts/submit_spreadsheetbench_eval.sh`：接受一个或多个
  `LABEL=MODEL`，检查 Hugging Face 配置和权重，顺序提交 val-only Ray job；默认
  `VAL_SIZE=399`、`VAL_CONCURRENCY=64`、确定性解码和 `native_basic`。
- `rl/scripts/compare_spreadsheetbench_evals.py`：按 `task_id` 检查任务集合、重复和遗漏，
  输出 Base-only、Candidate-only、success delta、Python error、写工具成功率和
  multi-call partial failure，并可同时落盘 JSON 和 Markdown。

典型使用方式：

```bash
bash rl/scripts/submit_spreadsheetbench_eval.sh \
  base=/path/to/Qwen3-4B-Thinking-2507 \
  step50=/path/to/global_step_50/actor/huggingface

PYTHONPATH=.:rl:third_party/verl-agent \
python rl/scripts/compare_spreadsheetbench_evals.py \
  runs/grpo_spreadsheetbench_eval_base_<timestamp> \
  runs/grpo_spreadsheetbench_eval_step50_<timestamp> \
  --expected-tasks 399 \
  --json-output runs/eval_step50_comparison.json \
  --markdown-output runs/eval_step50_comparison.md \
  --task-jsonl-output runs/eval_step50_tasks.jsonl
```

评测脚本默认 `MAX_STEPS=15`。对比历史 checkpoint 时必须显式覆盖成其训练/基线采用的
值；不同 `MAX_STEPS`、tool set、数据 split 或 EnvHarness commit 的结果不能当作严格 paired
实验。

## 0.2 2026-09-24 评测工具严格性修正

代码审查后已补齐以下保护：

- paired comparison 默认校验 Git commit、validation split、任务数、turn 上限、上下文长度、
  thinking 开关、确定性采样参数、tool set、history 配置和 vLLM 长度配置；任一字段缺失或
  不一致都会停止比较。仅诊断历史结果时可显式传入 `--allow-config-mismatch`，报告会保留
  `config_mismatches`，不能把这类结果作为正式 paired 结论。
- `eval_error:` 不再静默计作模型失败。默认只要 Base 或 Candidate 存在 evaluator error，
  comparison 就失败并要求补跑；临时诊断可使用 `--allow-evaluator-errors`，这时相关任务从
  有效分母和 paired delta 中排除，并报告错误数及任务 ID。
- success rate 增加 Wilson 95% 置信区间；paired delta 增加基于逐任务差值标准误的 95%
  区间，并输出 exact McNemar p-value。JSON 和 Markdown 均保留这些结果。
- 写工具指标拆分为逐调用 `write_call_success_rate`、逐 turn 全部成功
  `write_turn_all_success_rate` 和 episode 级 `episodes_with_write_error_rate`，避免 multi-call
  下把多个写调用折算成一个调用。
- val-only SpreadsheetBench job 不再创建未使用的 train actor pool；launcher 的兜底
  `TRAIN_BS/PPO_MINI_BS` 会按 Ray 总 GPU 数设置，`GROUP_N=1`，避免无关的整除失败。
- `run_manifest.json` 升级为 version 2，补记 thinking、validation sampling 和 rollout 长度
  参数；verl-agent 的 val-only 修改也已落入可复现 integration patch。

正式比较仍应保持默认严格模式：

```bash
PYTHONPATH=.:rl:third_party/verl-agent \
python rl/scripts/compare_spreadsheetbench_evals.py \
  runs/grpo_spreadsheetbench_eval_base_<timestamp> \
  runs/grpo_spreadsheetbench_eval_step50_<timestamp> \
  --expected-tasks 399 \
  --json-output runs/eval_step50_comparison.json \
  --markdown-output runs/eval_step50_comparison.md \
  --task-jsonl-output runs/eval_step50_tasks.jsonl
```

不要在正式 checkpoint 选择中使用两个 `--allow-*` 开关。历史 manifest 缺少严格字段时，
应重新评测，而不是绕过检查。

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

当前代码已经具备独立评测所需的底层能力和专用评测入口：

- `dataset.py` 能读取 Spreadsheet-RL parquet 中的
  `reward_model.ground_truth` 和 `extra_info`；
- Spreadsheet-RL 任务目录中的 `output.xlsx` 被当作初始工作簿，
  `target.xlsx` 被当作 reward target；
- verl trainer 支持 `trainer.val_only=True`，会先加载模型或 checkpoint，执行一次
  validation 后退出；
- 当前 `run_spreadsheetbench_grpo.sh` 可以通过 `EXTRA_HYDRA` 传入
  `trainer.val_only=True`；
- 专用 launcher 支持顺序评测多个 HF model/checkpoint；paired summary 支持验证任务集合
  并比较逐任务结果；
- 数据 split 之间的 ID/hash 重叠审计仍未实现。

因此 Base/checkpoint 的 Verified-399 paired evaluation 已形成可复用流程；严格跨数据集
独立性仍需补 split 重叠审计。

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

## 4. 历史临时评测命令

以下命令保留用于复现实验，新增实验应优先使用 0.1 节的专用 launcher。下面命令先完整
评测 `test_2_hermes.parquet` 的 297 条任务。需要先启动 Ray，并将数据复制到目标目录。
这些命令产生于 `VAL_SIZE` / `VAL_CONCURRENCY` 解耦之前，当时 `VAL_BS` 会同时扩大
actor 数；当前新入口可以用较小 `VAL_CONCURRENCY` 分批遍历完整 split。不能只把
`VAL_SIZE` 改小后把一个子集当作全量结果。

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

1. [x] 增加 `submit_spreadsheetbench_eval.sh`，明确区分 base/HF checkpoint 与训练 resume。
2. 增加数据审计脚本，输出 split 行数、有效任务数、缺失文件以及跨 split ID/hash 重叠。
3. [x] 通过 `VAL_SIZE` / `VAL_CONCURRENCY` 支持完整验证分批，避免验证规模等于 actor 数。
4. [x] 基于 env trajectory 输出逐任务 paired summary，而不只记录聚合 success rate。
5. [部分完成] run manifest 已记录模型路径、Git commit 和关键行为参数；dataset 文件 hash
   和任务 ID 列表仍待补充。
6. [x] multi-call 评测诊断区分完整执行、全部成功和短路，并输出停止原因、失败位置及
   逐 turn JSONL badcase；正式对比表展示 completed rate 和 short-circuit rate。

完成数据重叠审计和 manifest 数据指纹后，才能把跨 split 结果称为严格 held-out。
训练中的固定 64-task fast-val 只用于健康检查和趋势观测，checkpoint 选择仍以完整
399-task paired evaluation 为准。
