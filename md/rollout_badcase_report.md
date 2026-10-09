# SpreadsheetBench Rollout 错误样例报告

更新日期：2026-09-14

分析实验：

- `grpo_spreadsheetbench_full_20260901_114006`
- `grpo_spreadsheetbench_full_20260901_210619`
- `grpo_spreadsheetbench_full_20260902_011616`
- `grpo_spreadsheetbench_diagnostic_20260906_111741`
- `grpo_spreadsheetbench_diagnostic_20260906_114633`
- `grpo_spreadsheetbench_diagnostic_20260909_144747`
- `grpo_spreadsheetbench_diagnostic_20260912_194504`

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

### `20260912_194504` 补充结论

该实验加入 `native_basic`、两节点 16 GPU、`MAX_STEPS=15`、阶段 heartbeat 和 actor
超时诊断。此前 `validate_workbook` 在任务 `455-35` 的大范围校验中超过 600 秒；修复为
顺序扫描后，截至 step 34 未再次发生 evaluator 或 Ray actor timeout。

工程稳定性改善不等于训练效果收敛：初始 validation success rate 为 `0.281`，step 10
为 `0.344`，step 30 又回到 `0.281`。训练 success rate 在 step 33 为 `0.453`，step 34
降至 `0.078`；后者同时出现 `actor/kl_loss=7.532`、response clip ratio `0.111`。因此
下一步应比较 base 与 checkpoint 的全量独立评测，而不是根据单步训练 reward 选择模型。

当前训练使用 Verified-400，训练内 validation 也来自同一数据集合。即使 Spreadsheet-RL
提供 `test_verified_hermes.parquet`，也必须先做 ID/hash 重叠审计，不能直接把它当作该
checkpoint 的严格 held-out。独立评测设计见 `md/independent_evaluation_plan.md`。

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

## 2026-09-19 全量验证集补充结论

Base model 与 Spreadsheet-RL `global_step_5` 已在完全相同的 399 条
`test_verified_hermes.parquet` 任务上完成确定性评测：Base 成功 `122/399`，step 5 成功
`113/399`。两者共同成功 92 条，仅 Base 成功 30 条，仅 step 5 成功 21 条。此前 32 条
验证集上 step 5 的小幅领先没有在全量验证集复现。

step 5 将 native parser 有效率从 `0.746` 提高到 `0.944`，Python 错误率从 `0.196`
降低到 `0.114`，但写工具成功率从 `0.582` 降到 `0.460`，撞到 episode 上限的任务从
347 条增加到 362 条。这说明当前主要 badcase 已从“动作无法解析或 Python 无法执行”
转移为以下类型：

1. 工具调用合法，但写入范围 shape 不匹配；
2. 使用 `write_range` 写公式时被 static-value 限制拒绝；
3. 多轮检查和修改后仍未调用 `submit`；
4. Python 返回码为 0，但修改了错误 sheet/range 或 workbook 内容不完整；
5. 工具调用变多、episode 变长，最终撞到 `MAX_STEPS`。

后续 checkpoint 选择以固定验证集的 workbook success rate 为主，不再用包含执行惩罚的
`val/text/test_score` 代替任务准确率。代码稳定化方案和 TODO 见
`md/grpo_stabilization_plan.md`。

## 2026-09-30 大范围格式化导致整批评测失败

sgceph1 `20260926_232503` 的 Step 100 在完整 399 条评测中只完成部分任务。失败点不是
GPU OOM 或模型加载，而是 Ray `operation=step` 等待 600 秒后仍只有 63/64 actor 返回。
pending actor 对应：

```text
task_id=spreadsheetbench_verified/spreadsheet/1_399-14
action=run_python,format_range,format_range,validate_workbook
```

该任务要求对整个 sheet 使用指定字体。模型连续发出两个 `format_range`，而当时 sgceph1
代码没有 `MAX_FORMAT_CELLS` 和单轮 mutation budget，导致超大空白范围可能被逐单元格
实例化和保存。一个 actor 卡死最终中止整批评测。

该 badcase 对应两层修复：

1. 工具层：`format_range` 和 `fill_formula` 单次最多 50K cells，单轮 mutation 累计最多
   100K cells；超限返回可恢复工具错误，不进入实际 workbook mutation。
2. 评测层：保留 pending actor/task/action 诊断；后续应增加 per-tool 超时或将单任务工具
   超时计为该任务失败，避免一个 badcase 中止完整 399 条评测。

完整跨集群结果见 `md/experiment_results_20260930.md`。

## 2026-10-01 workbook 零分 badcase 观测

轨迹 schema v2 在 terminal `final_info.badcase_diagnostics` 中记录多标签失败证据，覆盖
未提交、达到步数上限、submit gate 未恢复、工具调用截断、mutation budget 超限，以及
答案区域未修改、疑似写错 sheet/range、部分完成、公式静态化和公式结果错误。正式评测使用
`full`，训练默认使用不扫描 workbook 的 `light`；诊断异常与扫描截断都有显式状态。

这些标签和 `answer_match_ratio` 只用于观测，不参与 reward、官方 workbook 比较或
checkpoint 成功判定。格式差异采用归一化 style 比较，仅作为 `format_mismatch_info`
信息，不应解释成官方零分原因。控制台不会打印完整 Python、单元格值或公式文本；轨迹仍按
既有方案保留原始动作供训练诊断。

离线分析可生成逐失败任务 `badcases.jsonl`、聚合 `badcase_summary.json`，paired comparison
还会给出 Base/candidate 标签及 added/removed/retained 转移。各比率必须结合
`badcase_policy_episode_count`、`badcase_complete_failure_count` 和 unavailable/truncated
计数阅读，不能把未执行 full 诊断的任务当作零发生率。

## 2026-10-09 shufflefix full-399 badcase 结论

NJ5 `runs/parallel_eval/shufflefix_20261004_202150` 已对 Base 和 Step10-110 完成正式
full-diagnostics 评测。最佳 Step 100 为 `119/399`，Base 为 `94/399`。以下对比来自各 run
的 `badcase_summary.json`：

| 指标/标签 | Base | Step 100 | Step 110 |
| --- | ---: | ---: | ---: |
| 成功任务 | 94 | 119 | 101 |
| `tool_error` | 178 | 100 | 84 |
| `no_submit` | 24 | 16 | 19 |
| `tool_call_truncated` | 13 | 4 | 9 |
| `formula_to_static` | 92 | 63 | 63 |
| `likely_wrong_sheet_or_range` | 44 | 29 | 27 |
| `formula_result_mismatch` | 173 | 169 | 182 |
| `execution_clean_score_zero` | 127 | 180 | 214 |
| `answer_match_ratio_failed_mean` | 0.2760 | 0.2707 | 0.2903 |

### 可以确认的改善

- 模型更少产生工具错误、未提交、调用截断和公式静态化；
- Step 100 的 multi-call completed rate 从 Base 的 65.74% 提高到 81.34%，short-circuit
  rate 从 34.26% 降到 18.66%；
- 疑似写错 sheet/range 的失败数从 44 降到 29；
- mutation budget exceeded 为 0，说明 50K/100K 上限在本次 full-399 中没有误伤正常任务。

### 仍然存在的主要问题

1. 公式结果错误几乎没有改善，Step 110 甚至恶化到 182 条；
2. `execution_clean_score_zero` 随训练显著增加，说明“工具全部执行成功”越来越不能代表任务
   正确；
3. Step 110 的 `tool_error` 比 Step 100 更低，但 workbook success 少 18 条，不能把降低
   tool error 直接当作训练目标；
4. `submitted_after_recalc_rate` 在 Step 100 只有约 0.25%，Step 110 为 0，模型仍很少形成
   “修改—重算—检查—提交”闭环；
5. failed-task 的平均 answer match ratio 没有随成功率稳定上升，partial correctness 暂时不适合
   未经校准直接并入 reward。

### 后续 badcase 分析优先级

按以下集合抽取具体任务，而不是只看聚合标签：

1. Base 失败、Step 100 成功的 50 条：识别真正学到的公式/range 模式；
2. Base 成功、Step 100 失败的 25 条：识别训练引入的回归；
3. Step 100 成功、Step 110 失败的任务：定位后期策略收缩；
4. Step 100 中 `execution_clean_score_zero + formula_result_mismatch` 的任务：优先分析公式语义；
5. 有 recalc 调用但未在 recalc 后 submit 的任务：检查工具提示、调用顺序和 turn 预算。

格式差异仍只作为信息证据；这些标签都不应在未经独立 A/B 验证前直接转换成 reward。
