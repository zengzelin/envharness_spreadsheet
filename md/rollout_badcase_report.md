# SpreadsheetBench Rollout 错误样例报告

更新日期：2026-09-10

分析实验：

- `grpo_spreadsheetbench_full_20260901_114006`
- `grpo_spreadsheetbench_full_20260901_210619`
- `grpo_spreadsheetbench_full_20260902_011616`
- `grpo_spreadsheetbench_diagnostic_20260906_111741`
- `grpo_spreadsheetbench_diagnostic_20260906_114633`
- `grpo_spreadsheetbench_diagnostic_20260909_144747`

数据来源：

- `rollouts/verl/*.jsonl`：模型原始输出、即时 reward 和 verl step 记录。
- `rollouts/env/{train,val}/*.json`：完整环境轨迹、投影动作、执行结果和最终评分。
- `train.log`、`submit.log`：训练指标、验证结果和 Ray 作业状态。

## 总结

目前失败样例可分为三个层次：

1. 交互协议错误：没有生成合法 tool call、JSON 不合法、工具名错误。
2. Python 执行错误：tool call 合法，但代码存在语法、API、索引或数据类型错误。
3. 任务求解错误：代码成功执行并提交了工作簿，但修改内容与目标答案不一致。

`20260906_114633` 已基本解决第一层问题：最终
`parser/invalid_ratio=0.002`、`episode/valid_action_ratio=0.998`，并且响应没有被
16384 token 截断。当前主要瓶颈已经转移到 Python 代码质量和 Spreadsheet 任务求解。

`20260909_144747` 使用修正后的 Python 错误统计、运行辅助函数和即时错误惩罚重新执行
了 20 步诊断训练。实验完整结束，Ray job 成功，并保存了 step 10 和 step 20 checkpoint。
该实验确认语法错误显著下降，但最终验证成功率没有稳定提高。因此目前可以确认工程
修复有效，尚不能确认 Vanilla GRPO 已经提高 Spreadsheet 任务能力。

但 `20260906_114633` 中的 `env/python_error_ratio=0.353` 不是准确的 Python 进程失败率。
旧实现通过扫描完整 observation 中是否包含 `traceback`、`exception`、`error:`
等文本来判断错误，因此会把以下情况误报为 Python 错误：

- `Tool call parse error`；
- 正常返回码为 0 的 warning；
- 任务描述或历史 observation 中包含 `error`；
- 上一轮错误仍出现在上下文中。

后续代码已改为只根据 `run_python` 的真实 `returncode` 和异常类型统计。新旧实验的
`env/python_error_ratio` 口径不同，不能直接画在同一条曲线上比较。

## 各实验现象

### `20260901_114006`

后期出现明显生成退化。`rollouts/verl/100.jsonl` 中：

- rollout 记录数：1168；
- `score=-0.1`：552；
- `score=0.0`：598；
- `score=1.0`：18；
- 1091 条输出没有 `<tool_call>`；
- 346 条输出存在明显重复或乱码。

代表输出：

```text
201688C".S".S".S".F".R".P".L".L".a".e".ed".ed"...
```

结论：这是策略生成退化，不是普通的表格推理失败，不应继续作为效果实验。

### `20260901_210619`

相比 `114006` 更稳定，但仍存在较多超长输出和不兼容工具调用。
`rollouts/verl/90.jsonl` 中：

- rollout 记录数：912；
- `score=-0.1`：168；
- `score=0.0`：493；
- `score=1.0`：247；
- 237 条输出没有 `<tool_call>`；
- 207 条输出使用 Markdown fenced block；
- 160 条输出很长。

代表错误工具名：

```json
{"name":"submit_output","arguments":{}}
```

环境只支持 `run_python`、`validate_workbook` 和 `submit`。

### `20260902_011616`

三组早期 full 实验中最健康，工具格式退化较少，但仍有较多代码执行错误。代表错误：

```text
SyntaxError: unterminated string literal
```

常见触发方式是 Excel 公式、Python f-string 和 JSON 字符串三层引号嵌套。

### `20260906_111741`

配置为 `MAX_STEPS=6`、`MAX_RESPONSE_LENGTH=8192`、compact history。

- 最终 val success rate：0.250；
- 最终 train success rate：0.305；
- parser invalid ratio：0.240；
- response clip ratio：0.294；
- episode length mean：5.938，接近 6 步上限。

结论：compact history 消除了 prompt 截断，但 8192 response 和 6 steps 对 thinking
模型仍偏紧，格式失败与截断继续污染训练。

### `20260906_114633`

配置为 `MAX_STEPS=10`、`MAX_RESPONSE_LENGTH=16384`、compact history。

- 初始 val success rate：0.188；
- step 5 val success rate：0.375；
- 最终 val success rate：0.219；
- 最终 train success rate：0.281；
- parser invalid ratio：0.002；
- response clip ratio：0；
- prompt clip ratio：0；
- episode length mean/min/max：10/10/10；
- 旧口径 `env/python_error_ratio`：0.353；
- `env/syntax_error_ratio`：0.073。

结论：16384 response、10 steps 和 compact history 已经解决主要协议与截断问题。
但轨迹几乎全部撞到 10 步上限，说明模型在执行错误后反复试错，尚未形成有效的
“检查结构—执行修改—验证—提交”闭环。

### `20260909_144747`

这是 Python 执行修复后的对照实验，使用：

```text
model = Qwen3-4B-Thinking-2507
NNODES = 2
GPU = 8 × 2
TRAIN_BS = 16
GROUP_N = 8
MAX_STEPS = 10
MAX_PROMPT_LENGTH = 8192
MAX_RESPONSE_LENGTH = 16384
history = compact
total_training_steps = 20
val_before_train = True
test_freq = 5
save_freq = 10
```

训练完整运行 20 步，总耗时约 12 小时 42 分钟。验证结果为：

| Step | Val success rate | Val score |
| ---: | ---: | ---: |
| 0 | 0.250 | 0.155 |
| 5 | 0.250 | 0.172 |
| 10 | **0.312** | **0.256** |
| 15 | 0.281 | 0.212 |
| 20 | 0.219 | 0.156 |

step 10 是该实验的最佳 checkpoint；step 20 已回落到初始水平以下。验证集只有 32 条，
每个任务对应 3.125 个百分点，因此 step 10 相比 step 0 只多成功约 2 个任务，不能据此
认定训练产生了稳定提升。

关键行为指标：

- `parser/invalid_ratio` 大多数 step 低于 0.02；
- `episode/valid_action_ratio` 大多数 step 高于 0.98；
- prompt 截断率为 0，response 截断率约为 0～0.011；
- Python timeout 比率为 0；
- Python runtime error 仍约为 0.07～0.12；
- episode length 后期基本固定为 10；
- 每条轨迹平均约 10 个 tool call；
- 平均 response 长度约 6100～7400 token，tool call 通常出现在输出末尾；
- `actor/ppo_kl` 约为 0.004～0.014，没有出现明显策略数值崩溃。

单个普通训练 step 约耗时 33～37 分钟，其中 rollout generation 约 20～25 分钟，actor
update 约 7.5～8.7 分钟。带 validation 和 checkpoint 的 step 可达到 42～50 分钟。
当前速度瓶颈是多轮长 thinking rollout，不是 evaluator；reward 计算通常只有约 2 秒。

日志中的 `Traceback` 主要是环境内某次 `run_python` 的失败输出，不代表 trainer crash。
最终 Ray job 状态为 succeeded。

## 错误类型与代表样例

### 1. 缺少工具调用

```text
Tool call parse error: no JSON tool call found.
```

模型只输出长篇解释，没有生成可执行动作。宽松 projection 可以恢复裸 JSON 和 fenced
JSON，但纯自然语言不能恢复。

### 2. Tool call 内 JSON 不合法

```text
Tool call parse error: invalid JSON inside '<tool_call>...</tool_call>'.
```

常见原因是代码中的换行、反斜杠和引号没有正确 JSON 转义。

### 3. 工具名错误

```text
Tool call parse error: unknown tool name.
```

代表错误是使用 `submit_output`，反映模型混用了其他工具协议。

### 4. 推理过长或输出截断

模型经常先输出数千 token 的重复分析，再把 tool call 放在末尾。其影响包括 tool call
被截断、下一轮历史膨胀、训练变慢，以及模型在长推理中反复改变方案。

`20260906_114633` 使用 16384 response 后截断率归零，但 tool call 位置仍然很靠后，
因此速度和代码稳定性问题仍然存在。

### 5. Python 语法错误

```text
SyntaxError: invalid syntax
IndentationError: expected an indented block after 'for' statement
SyntaxError: unexpected character after line continuation character
SyntaxError: unterminated string literal
```

实际错误包括把 compound statement 压缩到分号后、把 `\n` 作为字面字符写入 Python
文件，以及 Excel 公式、Python f-string、JSON 三层引号嵌套错误。

### 6. openpyxl range 返回 tuple

```text
AttributeError: 'tuple' object has no attribute 'row'
AttributeError: 'tuple' object has no attribute 'value'
AttributeError: 'tuple' object has no attribute 'font'
```

错误写法：

```python
for cell in ws["C3:C6"]:
    cell.value = "..."
```

单列 range 的元素仍是一行 tuple，正确写法是：

```python
for (cell,) in ws["C3:C6"]:
    cell.value = "..."
```

也可以直接按行号调用 `ws.cell(row=row, column=3)`。

### 7. 合并单元格 API 使用错误

```text
AttributeError: 'MergedCell' object attribute 'value' is read-only
AttributeError: 'Worksheet' object has no attribute 'unmerge'
TypeError: expected <class 'int'>
```

根因包括向合并区域的非左上角单元格写值、使用不存在的 `ws.unmerge(...)`，以及错误地
用 `MergedCellRange` 对象做坐标判断。正确 API 是：

```python
for merged_range in list(ws.merged_cells.ranges):
    ws.unmerge_cells(str(merged_range))
```

### 8. Sheet 名称假设错误

```text
ValueError: Worksheet named 'Agent Categories' not found
KeyError: 'WHS'
```

模型根据任务自然语言猜测 sheet 名，而没有先读取 `workbook.sheetnames`。实际名称可能
包含空格、大小写差异或尾随空格。

### 9. pandas 行列与 header 假设错误

```text
IndexError: positional indexers are out-of-bounds
ValueError: Length of values (12) does not match length of index (13)
KeyError: None of [...] are in the [columns]
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

根因包括错误假设第一行是 header、混淆 Excel 行号与 DataFrame 索引、未检查 shape
就使用固定位置，以及数值列混有字符串、空值或公式。格式与公式操作应优先使用
openpyxl；确实需要 pandas 时，先检查 sheet 名、shape、columns 和数据类型。

### 10. 缺少 import 或变量

```text
NameError: name 'openpyxl' is not defined
NameError: name 'datetime' is not defined
```

每次 `run_python` 都是新进程，普通 import 和局部变量不会跨 turn 保留。运行时只预置
路径变量和 workbook helper，其他依赖仍需在当前代码中显式 import。

### 11. Timeout 或外部进程错误

```text
[ERROR] run_python timed out after 60s
```

典型原因是遍历整张巨大工作表、低效逐单元格处理或错误循环。该类别应与普通 runtime
error 分开统计。

### 12. 执行成功但工作簿错误

症状是 `run_python returncode=0`、`submit` 合法，但最终 score 为 0。常见原因包括：

- 修改了错误 sheet 或范围；
- 公式引用错位；
- 从 `input_path` 重新加载并覆盖前一轮修改；
- pandas 重写导致样式、公式、合并区域或其他 sheet 丢失；
- 没有保存到 `output_path`；
- 只完成了任务的一部分。

这是当前最重要的任务能力问题。消除 exception 后，还需要单独分析这类 badcase。

## 已实施的工程修改

### SpreadsheetBench 与 verl-agent 接入

1. 实现 `envharness_rl/spreadsheetbench` adapter，并注册
   `env.env_name=envharness_rl/spreadsheetbench`。
2. 支持 Ray 批量环境、GRPO 同任务多 rollout、环境 reward 回传和多轮 episode。
3. 支持 `run_python`、`validate_workbook`、`submit` 三种动作。
4. 实现 Hermes-compatible tool-call projection，并宽松恢复 fenced JSON、裸 JSON 和带
   thinking 前缀的合法动作。
5. 分别保存 `rollouts/verl/*.jsonl` 和 `rollouts/env/{train,val}/*.json`，用于复现模型
   输出、投影动作、Python 执行结果和最终评分。
6. 增加单机/多机 Ray 启动与提交脚本，记录 Ray job id、训练日志、W&B、TensorBoard 和
   checkpoint。

### 数据与评测接入

1. 支持原始 SpreadsheetBench 数据格式。
2. 实现 Spreadsheet-RL parquet loader，读取 `reward_model.ground_truth` 和 `extra_info`。
3. 将 Spreadsheet-RL 的 `output.xlsx` 映射为初始 workbook，将 `target.xlsx` 映射为
   golden workbook。
4. 已验证 `train_hermes.parquet` 5925 条训练任务和
   `test_verified_hermes.parquet` 399 条验证任务能够加载和 reset。
5. 修正 evaluator 异常处理，单个 workbook 评测失败不会导致整批 validation crash。

### Python 执行与环境反馈

1. Python error 改为根据真实 `returncode` 判断，不再扫描完整 observation 文本。
2. 记录 `python_error_type`，并分别统计 syntax、runtime 和 timeout 比率。
3. `run_python` 预置 `input_path`、`output_path`、`working_directory`。
4. 提供 `load_workbook_for_edit()` 和 `save_workbook(wb)`，默认在已有 output 上继续修改。
5. 新增不访问 golden 的 `validate_workbook`，检查输出文件能否打开以及目标 sheet/range
   是否存在。
6. syntax error 即时 reward 为 `-0.1`，其他 Python 执行错误为 `-0.05`；最终一步的
   执行惩罚与 workbook grader 分数相加，不再被终局评分覆盖。
7. prompt 增加安全编辑模式，要求先检查实际 sheet 和维度，优先使用 openpyxl。

## Python 修复前后对比

`20260906_114633` 与 `20260909_144747` 使用相同的基础模型、20 个训练 step、
`MAX_STEPS=10`、`MAX_RESPONSE_LENGTH=16384` 和 compact history。后者加入了 Python
执行辅助与修正后的指标口径，并将 rollout `max_model_len` 从 16384 提高到 24576；
因此这里主要用于验证错误修复方向，不能视为只改变单一变量的严格消融。

| Step | 旧 syntax error | 新 syntax error | 新 runtime error |
| ---: | ---: | ---: | ---: |
| 5 | 0.075 | 0.030 | 0.096 |
| 10 | 0.076 | 0.016 | 0.120 |
| 15 | 0.081 | 0.003 | 0.078 |
| 20 | 0.073 | 0.020 | 0.072 |

四个评测点的 syntax error 平均值从约 0.076 降到 0.017，相对下降约 77%。这表明安全
编辑 prompt、workbook helper 和语法错误反馈确实起作用。旧版 `python_error_ratio` 会
混入 parser 文本和历史错误，不能与新版总错误率直接比较；新版 runtime error 是当前
更可信、也更需要解决的指标。

任务效果没有同步出现稳定提升：旧实验和新实验的最终 val success rate 都是 0.219；
新实验最佳点是 step 10 的 0.312，随后回落。因此“减少 Python 语法错误”是必要条件，
但不足以解决 Spreadsheet 任务语义、正确范围选择、公式构造和修改后验证问题。

## 当前实验结论

1. `MAX_RESPONSE_LENGTH=16384`、`MAX_STEPS=10`、compact history 是目前可用的诊断
   配置，协议错误和截断已不再是首要瓶颈。
2. Python 语法错误修复有效；下一层瓶颈是 runtime error 和“执行成功但 workbook
   错误”。
3. episode 仍长期撞到 10 步上限，说明模型重复试错较多，缺少稳定的
   “检查—编辑—验证—提交”闭环。
4. train success rate 受每 step 仅约 2 个不同任务、每任务 8 条 rollout 的影响，波动
   很大，不能单独用于判断训练趋势。
5. 当前 validation 只有 32 条，且 train/val 都来自 Verified 400，不是严格 held-out
   划分。这些实验只能作为工程和行为诊断，不能作为最终泛化 baseline。
6. 后续基础模型固定使用 `Qwen3-4B-Thinking-2507`，不再考虑 Qwen3.5-4B，避免同时
   改变模型和训练环境。

## 下一步实验

### 1. 建立严格 held-out baseline

切换到 Spreadsheet-RL 的明确划分：

```text
train = train_hermes.parquet                 # 5925 tasks
val   = test_verified_hermes.parquet         # 399 tasks
```

首先在相同的 399 条验证任务上确定性评测：

```text
A. Base Qwen3-4B-Thinking-2507
B. 20260909_144747/global_step_10
```

保持 `temperature=0`、`do_sample=False`、`MAX_STEPS=10`、
`MAX_RESPONSE_LENGTH=16384`、compact history 和相同 evaluator。399 条验证集上至少需要
提高约 3～5 个百分点，即多成功约 12～20 条任务，才认为 checkpoint 有明确价值。

### 2. 建立固定快速验证集

从 399 条验证任务中固定选取 64 条作为 fast-val。训练前和训练结束运行完整 399 条，
中间每 5 步只运行固定 fast-val。禁止每次随机抽样，否则任务难度变化会污染曲线。

### 3. 运行严格划分的 20 步 GRPO

在 base 与 step 10 的完整验证对比完成后，再使用 Spreadsheet-RL train split 运行 20
步训练。暂时保持现有 PPO 参数不变，不同时修改学习率、KL、reward、response 长度和
最大步数。

### 4. 深入分析任务语义 badcase

重点抽取以下轨迹：

```text
run_python returncode = 0
submit 合法
最终 workbook score = 0
```

继续分类为错误 sheet/range、公式引用错误、只完成部分任务、保存错误、格式破坏、缺少
验证和过早提交。若 Python error 下降而 success rate 仍不提高，下一阶段应研究
verification-aware environment 或 trajectory-level credit assignment，而不是继续修改
parser 或盲目延长训练。

## 后续关注指标

保持 `MAX_STEPS=10`、`MAX_RESPONSE_LENGTH=16384`、compact history，先跑 20 步，
不要同时调整 PPO 参数。重点观察：

- `env/python_error_ratio`：目标低于 0.20；
- `env/syntax_error_ratio`：目标低于 0.03；
- `env/python_runtime_error_ratio`；
- `env/python_timeout_ratio`；
- `parser/invalid_ratio`：应继续保持低于 0.02；
- `episode/length/mean`：不应长期固定在 10；
- train/val success rate：确认代码错误下降能否转化为任务成功率提升。

若 Python 错误明显下降但 success rate 没有提高，下一阶段应集中分析“执行成功但
工作簿错误”的任务语义 badcase，而不是继续修改 parser 或 response 长度。
