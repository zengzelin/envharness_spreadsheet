# Spreadsheet Work Agent 技术深挖文档（面试准备版）

> 更新时间：2026-10-14
> 依据：当前仓库代码（`envharness/bridges/spreadsheetbench/`、`rl/envharness_rl/spreadsheetbench/`、`rl/scripts/`）
> 与实验记录（`md/rollout_badcase_report.md`、`md/experiment_results_20260930.md`、
> `md/grpo_stabilization_plan.md`、`md/spreadsheet_rl_vs_envharness_comparison.md` 等）。
>
> 写作原则：每个机制都按 **“观察到的现象 → 设计决策 → 代码实现 → 效果证据”** 四段组织。
> 面试时被追问“为什么这么设计”，答案都在“现象”一栏；被追问“怎么证明有用”，答案都在“证据”一栏。

---

## 0. 一页速览

**一句话**：以 Qwen3-4B-Thinking-2507 为基座，在 SpreadsheetBench Verified（本地 399 条）上构建
一个工作簿原生的 Agent Harness（检查—修改—重算—校验—提交闭环 + 事务安全 + 版本门禁 + 预算），
完成 Teacher 数据合成与 LoRA SFT 方案设计，并在 Harness 中开展 GRPO 后训练。
任务成功由官方评测器判定；当前训练 Reward 为终局成功奖励加 Python 和 Invalid Action 错误惩罚。

**核心设计原则（贯穿全文）**：

1. **任务结果与训练信号分开报告**：官方工作簿比较定义成功；训练可包含有限的错误惩罚。
   工具成功率和 Badcase 标签用于分析，不能直接代表任务成功。
2. **围绕实际失败设计约束**：观察错误、定位原因、增加约束，再检查适用范围和运行开销。
3. **执行有资源边界**：调用数、原生单元格修改、返回字符、诊断扫描、重算和进程等待设有上限；
   Python 和结构操作不计入原生单元格计数，另由事务和超时保护。
4. **区分“工具执行成功”和“任务完成正确”**：这是项目最重要的实验结论，也是后续瓶颈。

---

## 1. 展示口径与结果索引

正文围绕简历中的三项工作组织：**Agent Harness、数据合成与 SFT 设计、GRPO 与 Reward**。
主结果沿用 `23.56% → 29.82%`，展示问题、方法与效果；历史数值保留在本节供追问时查阅。

| 数值 | 实验阶段与来源 | 展示方式 |
| --- | --- | --- |
| 17.04%（68/399） | NJ-A `20260926_155341` 历史 Base，commit `f206a2b`，提交门禁关闭 | 历史记录，不与当前 Base 拼成 Harness 因果提升链 |
| 23.56%（94/399） | NJ5 shufflefix 组 Base，`native_basic`、提交门禁开启 | 固定 Harness 下 GRPO 的起点 |
| 27.32%（109/399） | 现有记录中对应 GRPO 中间 Checkpoint；暂无对应 SFT 评测记录 | 如保留此数字，注明现有来源，不标作已验证的 SFT 效果 |
| 29.82%（119/399） | NJ5 Step 100，从 Base 直接 GRPO；配对 McNemar `p=0.005228` | 同组 GRPO 最佳单次结果 |

SFT 目前用“完成数据合成与训练方案设计”描述，后续数据生产、SFT 训练及 SFT→GRPO 结果另行补充。
PPT 主体不需要展开所有历史数字，详细来源与比较边界保留在技术文档。

工具消融已经取得 Base Python 与 Native Read 的 full-399 结果，见 §4.15。
工具消融回答工具配置的作用；Base→Step100 回答固定 `native_basic` 下后训练的作用。

---

## 2. 问题定义

### 2.1 任务

- 输入：自然语言指令 + 一个真实 `.xlsx` 工作簿 + `answer_position`（如 `Sheet1!A3:D32`）。
- 当前协议向 Agent 展示目标位置；Golden 值、最终比较差异和隐藏评测文件不作为 Agent 输入。
- 输出：修改后的工作簿。
- 类型：Cell-Level Manipulation（写值/公式）与 Sheet-Level Manipulation（整表/结构操作）。
- 数据：训练用 Spreadsheet-RL `train_hermes.parquet`（5,925 条 ExcelForum 任务）；评测用
  `test_verified_hermes.parquet`（上游 README 标 400，实际发行包只有 399，缺 id=54675，本地下载无损）。

### 2.2 评测（官方 Online-Judge）

- 先用 LibreOffice headless 重算 agent 输出与 golden（否则 openpyxl `data_only=True` 读公式为 `None`）；
- 对 `answer_position` 的每个单元格做官方规则比较：数值 round 2 位、日期转 Excel 序列数、`"" ≡ None` 等；
- **全部单元格匹配才算成功，无部分分**。
- 当前评分主要比较目标单元格值，不保证格式要求、公式表达方式及全部结构要求均被覆盖；
  这些维度由工作簿诊断和拟实施的专项数据校验补充分析。

### 2.3 为什么难（这是 Harness 存在的理由）

电子表格任务不是文本生成，Agent 修改的是**有状态的二进制文件**：

- 工具返回成功 ≠ 任务正确：可能写错 sheet/range、公式语义错、公式被写成静态值、样式/合并区丢失；
- 文件会损坏：半成品写入、pandas 重写导致其他 sheet 丢失、生成无法打开的 xlsx；
- 公式值需要重算才可见；
- 多轮长程交互：thinking 模型输出长、上下文膨胀、10 步预算有限；
- 训练规模：128 个并发 rollout，任何一个 actor 卡死都会拖垮整批。

---

## 3. 系统架构

### 3.1 三层结构

```text
verl-agent（GRPO 训练框架，固定 commit 796ed31 + rl/integration/*.patch）
  └─ SpreadsheetBenchEnvironmentManager     rl/envharness_rl/spreadsheetbench/manager.py
       │  Prompt 组装 / 历史压缩 / 投影 / 诊断汇总 / 轨迹落盘 / badcase 分类
       └─ EnvharnessSpreadsheetEnvs + Ray Worker   rl/envharness_rl/spreadsheetbench/envs.py
            │  Ray actor 池 / 单轮多调用执行 / mutation 预算 / 失败短路 / 终局评分 / reward 分解
            └─ SpreadsheetBenchEnv（Bridge）   envharness/bridges/spreadsheetbench/bridge.py
                 │  沙箱 / 工具路由 / 事务 / 版本追踪 / 提交门禁 / 官方评分
                 ├─ read_tools / write_tools / structure_tools / recalc_tools
                 ├─ python_preflight / process_utils
                 ├─ online_judge_eval（官方评测器 vendored）
                 ├─ badcase_diagnostics（只解释，不评分）
                 └─ dataset（Spreadsheet-RL parquet 惰性加载）
```

### 3.2 一个 episode 的完整生命周期

```text
reset
  ├─ dataset：从 parquet 行缓存中选中一行（训练：确定性 shuffle；验证：固定 task_index）
  ├─ 沙箱：mkdtemp → 复制 input.xlsx 与 output.xlsx（output 预置为 input 副本，no-op 也可评分）
  ├─ 预览：每个 sheet 前 20 行 × 16 列
  └─ 版本计数器：workbook_revision=0，last_validation_revision=-1，last_recalc_revision=-1

每一轮（最多 MAX_STEPS=10 轮）
  ├─ Manager 组装 prompt：工具说明（按工具集）+ 最近 2 轮压缩历史 + 剩余步数 + 当前观察
  ├─ 模型生成（Qwen3 thinking，最长 16384 token）
  ├─ projection：截取 </think> 之后 → 抽取 ≤4 个 <tool_call> → 参数前置校验 → 顺序约束
  ├─ Worker.step_many：逐个执行
  │     ├─ mutation 预算检查（累计 ≤100K cells）
  │     ├─ Bridge.step：工具门禁 → 执行（事务/原子保存）→ 版本号更新
  │     └─ 失败短路：写/重算/Python/校验/提交失败后跳过本轮剩余调用
  └─ 观察合并（单轮 12,000 字符预算）+ 诊断指标 + 轨迹记录

终止（submit 被放行 / 达到 MAX_STEPS）
  ├─ 官方评分：LibreOffice 重算 output → golden（缓存）→ compare_workbooks
  ├─ reward = 官方得分(0/1) + 本 episode 执行惩罚累计
  ├─ 失败时 badcase 诊断（light/full）+ 执行证据合并 → failure_tags
  └─ 轨迹 JSON 落盘（schema v2）
```

---

## 4. Harness 机制详解（现象 → 设计 → 实现 → 证据）

### 4.1 动作协议与投影（Projection）

**现象**（`rollout_badcase_report.md`）

- `20260901_114006` step 100：1,168 条 rollout 中 **1,091 条没有 `<tool_call>`**，346 条重复或乱码；
- `20260901_210619`：237 条无 tool call、207 条使用 Markdown fenced block、出现 `submit_output` 等**不存在的工具名**；
- thinking 模型常先写数千 token 推理，再把 tool call 放在末尾；推理未闭合时整段输出没有动作。

**设计**

- 只解析最后一个 `</think>` 之后的内容；有 `<think>` 无 `</think>` 时返回明确的“仍在推理中”错误，而不是去推理文本里捞 JSON；
- 三级解析：标准 `<tool_call>` → fenced JSON 恢复 → 裸 JSON 恢复；恢复成功的动作照常执行，但记 `recovered=1`；
- 参数前置校验：工具是否在当前工具集启用、必填参数、`fill_formula` 坐标与方向、公式是否被包成文本；
- 每类错误都有独立 `status`（`unclosed_reasoning`、`missing_tool_call_end`、`invalid_json`、`wrong_tool_name`、`missing_code`……），
  错误文本前插到下一轮观察，让模型知道错在哪里。

**实现**：`projection.py` 的 `_parse_region`、`_extract_payload`、`_project_one_with_diagnostics`。

**证据**：历史 run `20260906_114633` 记录 `parser/invalid_ratio=0.002`、`valid_action_ratio=0.998`；
它同时涉及 Prompt、生成预算及训练状态变化，不作为投影模块的独立消融收益。
NJ5 同一训练 run 的前后窗口中，valid action 从 0.8245 升到 0.9431，parser invalid 从 0.1755 降到 0.0569，
说明训练中的协议遵循改善。

### 4.2 上下文与长度预算

**现象**

- `20260906_111741`（8192 response、6 steps）：parser invalid 0.240，**response clip 0.294**，episode 长度 5.938（几乎全撞上限）；
- 多轮历史中完整 Python 代码和完整观察导致 prompt 膨胀、被截断。

**设计**

- `MAX_RESPONSE_LENGTH=16384`、`MAX_STEPS=10`、`MAX_PROMPT_LENGTH=8192`、vLLM `max_model_len=24576`；
- **compact history**：只保留最近 `history_length=2` 轮；历史观察截断到 2,000 字符；历史动作替换为
  “投影动作名 + parser 状态 + 输出长度”，不再回放原始长输出；
- 观察本身有界：`run_python` 输出 6,000 字符、读工具响应 10,240 字符、单轮合并观察 12,000 字符（头 2/3 + 尾 1/3 截断）；
- 读工具返回结构化 JSON 且限量（inspect 最多 400 格），避免把整个工作簿塞进上下文。

**实现**：`manager.py` `_format_history_action` / `build_text_obs`；`envs.py` `step_many` 中的 `observation_budget`。

**证据**：`20260906_114633` 起 response clip 与 prompt clip 均为 0。

### 4.3 `run_python` 执行环境（Bootstrap Helper）

**现象**（badcase 第 10、12 类）

- 每次 `run_python` 是新进程，模型忘记 import → `NameError: name 'openpyxl' is not defined`；
- **从 `input_path` 重新加载并覆盖前一轮修改**；没有保存到 `output_path`；
- pandas 重写整个文件导致样式、公式、合并区、其他 sheet 丢失。

**设计**

- 运行时预置 `input_path`、`output_path`、`working_directory`、`load_workbook_for_edit()`（打开**当前 output**，保留前几轮修改）、`save_workbook(wb)`；
- Prompt 给出“安全编辑模式”范例，要求先看 `sheetnames` 和维度，优先 openpyxl；
- 代码写在沙箱外的临时文件中执行（cwd 仍是沙箱），避免模型 `os.listdir` 看到一堆 `_step_*.py` 浪费轮次去“找评分逻辑”。

**实现**：`bridge.py` `_python_bootstrap`、`_run_python`；`manager.py` `_PYTHON_TOOL_INSTRUCTIONS`。

**证据**：`20260906_114633` → `20260909_144747`，四个评测点 syntax error 均值 **0.076 → 0.017（−77%）**。
但两者最终 val success 都是 0.219——这是项目第一次观察到“执行质量改善 ≠ 任务成功提升”。

### 4.4 AST Preflight（执行前静态检查）

**现象**

- 语法错误高发：f-string / Excel 公式 / JSON 三层引号嵌套、复合语句压成一行、`\n` 被写成字面量；
- 模型频繁使用不存在的 `cell.formula = ...` API（openpyxl 里公式通过 `cell.value = "=..."` 写入）。

**设计**

- 启动子进程前先 `ast.parse`：语法错误直接拒绝，不浪费进程启动和超时；
- 硬拒绝：对 `.formula` 属性赋值；
- 警告（不拒绝，记指标）：裸 `except`、从 `input_path` 加载做编辑、疑似修改单元格却没调用 `save_workbook`。

**实现**：`python_preflight.py`；`bridge.step` 中 preflight 失败直接返回惩罚（语法 −0.1，其余 −0.05），记 `python_preflight_rejected`。

**证据**：NJ5 训练中 preflight reject 从 0.0300 降到 0.0056。

### 4.5 事务式执行与回滚

**现象**：Python 执行到一半失败或超时，output 留下半成品；写出无法打开的 xlsx 会让后续所有轮次和最终评分都失效。

**设计（`run_python`）**

```text
output.xlsx ──复制──> .sbtxn_xxx.xlsx（事务副本，代码实际写这个文件）
           └─复制──> sbbackup_xxx.xlsx（备份，放在沙箱外）
执行代码
  ├─ returncode==0：检查事务文件是普通文件（拒绝符号链接）→ openpyxl 试打开 → os.replace 原子提交
  ├─ returncode!=0 / 超时 / 试打开失败：从备份原子恢复（先复制到临时文件再 os.replace，不跟随 agent 创建的符号链接）
  └─ finally：清理脚本、事务副本、备份
```

**设计（原生写工具，共用）**：`fcntl` 文件锁（30 秒超时）→ openpyxl 打开 → 修改 → 保存到同目录临时文件
→ 重新打开校验 → `os.replace`。任何一步失败都不碰原文件。

**为什么这样做而不是“失败了让模型自己修”**：半成品状态对模型不可见（它只看到报错），下一轮会在错误基础上叠加修改；
回滚保证每一步要么完整生效、要么完全没发生，模型的心智模型与文件状态一致。

**实现**：`bridge.py` `_run_python`；`write_tools.py` `_workbook_lock`、`_atomic_save`；info 中记 `transaction_committed/rolled_back`。

**证据**：既有 hardening 后 full-399 记录未报告因工作簿损坏导致的评分失败，事务机制的行为以对应回滚测试验证。
SFT 方案设计中，第一版主数据排除含工具失败或回滚的示范轨迹；该过滤链路尚待实现。

### 4.6 进程组超时回收

**现象**：父进程超时后子进程（或 LibreOffice 的派生进程）仍存活，占着输出管道、工作簿或 LibreOffice profile，导致后续调用挂起。
并发 headless LibreOffice 共享同一 profile 会互相加锁、随机转换失败。

**设计**：`start_new_session=True` 启动新进程组，超时 `os.killpg(SIGKILL)` 整组回收，再兜底关闭管道；
LibreOffice 使用按 PID 隔离的 profile（`-env:UserInstallation=file:///tmp/lo_profile_<pid>`）。

**实现**：`process_utils.py` `run_process_group`；`online_judge_eval.py` `recalc_with_libreoffice`。

### 4.7 原生工具层（`native_basic`）

**现象**

- 只有 `run_python` 时，模型在“检查结构”这种简单操作上也要写 openpyxl 样板代码，错误高发：
  `ws["C3:C6"]` 迭代得到 tuple 而非 cell、写合并单元格的非左上角、`ws.unmerge` 不存在、根据自然语言**猜 sheet 名**（`KeyError: 'WHS'`）；
- 早期基础写工具上线后，`run_python` 仍占 66%~85% 的调用；`write_range` 只接受静态值，模型经常把公式传给它被拒绝，
  写工具成功率在 0.06~0.86 间剧烈波动；
- 公式类任务较多，早期没有原生公式填充工具，模型需要通过 Python 写公式；部分轨迹直接写静态值，
  后续通过 `formula_to_static` 标签观测这一行为。

**设计**：参考 Spreadsheet-RL 原生工具设计，分三档渐进开启（便于消融）：

| 档位 | 工具 |
| --- | --- |
| `python` | `run_python`、`validate_workbook`、`submit` |
| `native_read` | + `list_sheets`、`inspect_range`、`find_cells` |
| `native_basic` | + `write_range`、`clear_range`、`fill_formula`、`format_range`、`delete_rows`、`delete_columns`、`manage_sheet`、`recalculate_and_read` |

结构化工具的价值不是“能做 Python 做不到的事”，而是：模型只需生成短参数；schema 能约束输入并给出明确错误；
读结果有界；写操作可加锁、限额、原子提交；失败后重试成本低。`run_python` 保留为兜底（排序等未覆盖操作）。

**每个工具的关键约束**

| 工具 | 关键约束 | 针对的问题 |
| --- | --- | --- |
| `inspect_range` | 有限 A1 矩形；cells 模式最多返回 400 格；summary 模式最多扫 10 万格 | 防止整表读取撑爆上下文 |
| `find_cells` | values/formulas/both；contains/equals/prefix；最多 1,000 条 | 定位表头和关键字，替代猜测 |
| `list_sheets` | 返回名称、顺序、可见性、维度 | 替代猜 sheet 名 |
| `write_range` | 只写静态值，**拒绝以 `=` 开头的字符串**；单格锚点可展开矩阵；shape 不匹配报错；单值最长 8,192 字符；≤50K cells | 值与公式走不同通道，避免公式被写成文本 |
| `fill_formula` | 从 `start_cell` 按 openpyxl `Translator` 平移相对/混合/绝对引用；拒绝 `="=..."` 包裹、括号/引号不闭合、缺操作数；≤50K cells | 公式任务的正路 |
| `clear_range` | 不移位；正确处理合并区锚点 | 合并单元格误操作 |
| `format_range` | 白名单字体/填充/对齐/边框/数字格式；颜色正则；行高 0–409、列宽 0–255；≤50K cells | 格式任务；大范围格式化卡死（见 4.10） |
| `delete_rows/columns` | 区间合并、越界拒绝、与合并区相交拒绝；显式返回 `formula_reference_update=not_guaranteed_by_openpyxl` | openpyxl 不重写依赖公式的诚实声明 |
| `manage_sheet` | create/rename/copy/move/hide/unhide；非法名/重名/越界/隐藏最后一个可见 sheet 均拒绝 | 结构操作安全 |
| `recalculate_and_read` | 见 4.9 | 公式值可见性 |

**实现**：`read_tools.py`、`write_tools.py`、`structure_tools.py`、`recalc_tools.py`；Bridge 中统一门禁（未启用的工具返回 `tool_disabled`）。

**证据**：NJ5 Base→Step 100 的 `formula_to_static` 从 92 降到 63；`tool_error` 从 178 降到 100。
这是固定工具集下的训练前后变化。工具配置对 Base 的对照结果另见 §4.15。

### 4.8 公式与工作簿校验（`validate_workbook`）

**现象**：公式被保存成文本（`'=SUM(...)` 或 `="=SUM(...)"`）、括号不闭合、引用被删成 `#REF!`——openpyxl 都会**静默保存**，直到评分才发现。

**设计**：不接触 golden 的结构自检——output 能打开、`answer_position` 中的 sheet 与范围存在；
答案区内逐格检查：文本形式的伪公式、公式语法（括号/引号配对 + openpyxl Tokenizer 检查缺操作数）、`#REF!`、`="=` 包裹。
统计每个范围的格数、非空数、公式数返回给模型。

**检查边界**：这些检查覆盖指定目标区域中的文件结构、部分公式语法和引用错误，不证明公式语义正确，
也不是对整个工作簿每个公式的完整计算验证。

**性能细节**：任务 `455-35` 的答案区 `A1:J10411`，read-only 模式下随机单元格访问每次都从 XML 流开头扫描，
校验**超过 600 秒**导致 Ray actor 超时；改为 `iter_rows()` 单次顺序扫描后降到约 **1.3 秒**。

**实现**：`bridge.py` `_validate_output`；`write_tools.py` `validate_formula_syntax`。

### 4.9 中间重算：`recalculate_and_read`

**现象**：公式写入后，模型看不到公式算出的值，无法自检公式语义；`formula_result_mismatch` 是数量最多的失败标签（Base 173 条）。

**设计**

- 在文件锁下复制当前 output 到临时目录，只对副本做 LibreOffice 重算，**从不修改 output**；
- 一次最多 10 个范围、400 个单元格；返回缓存值、公式格数、Excel 错误（`#DIV/0!`、`#N/A` 等 14 种）、耗时；
- **每 episode 默认只允许 1 次**、超时 120 秒（LibreOffice 冷启动贵，训练吞吐敏感）；
- projection 强制它是本轮最后一个调用：必须先看到重算结果，下一轮才能 submit；
- 重算成功的同时会执行一次静态校验；只有无公式错误且静态校验通过才算成功，并记 `last_recalc_revision`。

**实现**：`recalc_tools.py`；`bridge.step` 的 recalc 分支；`projection.py` 的 `recalc_not_last` 规则。

**证据与局限**：机制有效但**使用率极低**——Step 100 的 `submitted_after_recalc_rate` 只有约 0.25%，Step 110 为 0。
模型没有自发学会“修改—重算—检查—提交”闭环，这是下一阶段的重点之一。

### 4.10 版本绑定的提交门禁（Submit Gate）

**现象**：模型在没有任何检查的情况下直接 submit；或者先 validate 一次，又继续修改，然后提交了一个从未被检查过的版本。

**设计**：维护三个计数器：

```text
workbook_revision         每次成功的写/结构工具、每次 returncode=0 的 run_python 后 +1
last_validation_revision  validate_workbook 成功时记录当前 revision
last_recalc_revision      recalculate_and_read 成功时记录当前 revision

submit 放行条件：max(last_validation_revision, last_recalc_revision) == workbook_revision
```

不满足时拒绝提交、episode 不结束，返回两种不同提示：“从未校验过” vs “校验后又修改了，请重新校验”，记 `submit_rejected`、`validation_stale`。
对 `run_python` 采取保守策略：只要成功执行就视为可能修改了工作簿。

**协议边界**：该门禁约束显式 `submit` 的放行条件。达到 `MAX_STEPS` 时，Worker 仍会对当前文件
执行终局评分，即使没有成功提交；因此不能声称所有被评分文件都通过了提交门禁。

**为什么要版本号而不是布尔标记**：布尔“已校验”无法识别“校验后又改”的情况；版本号让“校验的版本 = 提交的版本”成为协议保证。

**实现**：`bridge.py` `step()` 的 submit 分支；`SPREADSHEETBENCH_REQUIRE_VALIDATION_BEFORE_SUBMIT=true`。

**证据**：badcase 标签 `submit_gate_unrecovered`（被拒后再也没提交成功）可单独统计；NJ5 Base→Step 100 的 `no_submit` 从 24 降到 16。
注意门禁会改变 Agent 行为和绝对分数，所以 Base 与 checkpoint 必须在相同门禁配置下评测。

### 4.11 单轮多工具调用

**现象**

- “一轮一动作”下，细粒度工具反而**消耗 turn 预算**：Base 有 347/399 条撞到 10 步上限，step 5 增加到 362/399；
- 很多任务需要“读→写→校验”多步，每步占一轮，10 轮不够用。

**设计**

- 每轮最多接纳 4 个有序 `<tool_call>`，在同一个 episode step 内顺序执行；超过 4 个只执行前 4 个，并提示“剩余调用请下一轮继续”；
- `submit` 必须是最后一个；`recalculate_and_read` 必须是最后一个；
- **失败短路**：写/结构/重算/Python/校验/提交/非法动作失败后，跳过本轮剩余调用（避免在失败状态上继续修改）；读工具失败不短路；
- 记录 projected/executed/skipped、首个失败位置、停止原因（`completed`、`terminated`、`<tool>_failure`、`tool_call_limit`……）。

**为什么是 4**：覆盖“读+写+校验”或“写+写+校验”等常见组合，同时限制单轮爆炸半径和观察长度（12,000 字符按调用数均分）。

**实现**：`projection.py` `project_action_batch`；`envs.py` `step_many`。

**证据**：NJ5 Base→Step 100：multi-call completed rate **65.74% → 81.34%**，short-circuit rate **34.26% → 18.66%**。
两者均已启用 multi-call，说明后训练改善了调用组合的执行质量；该数字不是单调用与多调用的机制消融。

### 4.12 Mutation 预算（50K / 100K）

**现象（真实事故）**：sgceph1 hardened run 的 Step 100 全量评测失败。64 个 actor 中 63 个返回，唯一卡住的是：

```text
task_id = spreadsheetbench_verified/spreadsheet/1_399-14
action  = run_python, format_range, format_range, validate_workbook
```

任务要求“整个 sheet 使用等线字体”，模型对超大空白范围连续两次 `format_range`；当时代码没有 `MAX_FORMAT_CELLS` 和单轮预算，
openpyxl 逐单元格实例化样式并保存，单 actor 超过 600 秒，**一个 badcase 中止了整批 399 条评测**。

**设计**

- 单次上限：`write_range`/`clear_range`/`fill_formula`/`format_range` 各 ≤50,000 cells；
- 单轮累计：上述四类操作在同一轮内累计尝试 ≤100,000 cells（执行前按参数估算，超限直接返回可恢复错误，不进入真实修改）；
- `run_python` 与行列/sheet 结构操作不计入（无法静态估算，由事务和超时兜底）。

**阈值怎么定的（数据驱动，不是拍脑袋）**：统计 njceph5 训练日志——

- `format_range` 794 次，**最大正常范围 41,435 cells**，没有超过 50K；
- `fill_formula` 约 21,750 次，只有 6 次超过 50K（93,196 或 99,999 cells），全部是范围判断错误（横向公式误填到第 10 万行、远超评分区的邮编列）；
- 100K 单轮预算只拒绝过 4 个调用：两次 702,000-cell clear、两次 1,048,568-cell fill（整列），全是异常调用。

**证据**：NJ5 该组 full-399 中 `mutation_budget_exceeded` 为 0，表示本次评测未触发累计修改预算限制。
阈值覆盖已分析日志中的正常操作范围，尚不能推广为对所有大表任务零误伤。未来更大范围的任务应增加分块写入。

### 4.13 训练规模与评测稳定性

| 现象 | 根因 | 修复 | 结果 |
| --- | --- | --- | --- |
| 全量训练首批 reset：128 个 actor 600 秒内 `ready=0/128` | 每个 actor 首次 reset 都完整读 5,925 行 parquet 并逐任务访问 3 个文件；进程内缓存不能跨 actor 共享，放大为约 **75.8 万次**小文件访问 | parquet 原始行缓存 → 先按 seed 选行 → 只物化一个任务（惰性加载）；128 actor 无模型压测脚本 | 128 actor reset 不再卡死，支撑后续全部训练 |
| 一个 actor 卡住导致整批失败，且不知道是谁 | Ray `get` 无超时、无定位信息 | reset/step/close 全局超时（默认 600s），超时时列出 pending actor 索引、actor ID、task ID、action；rollout 阶段 START/HEARTBEAT/END 心跳 | 可以直接定位到 `1_399-14` 这样的具体任务 |
| LibreOffice 偶发重算失败被记成模型失败 | 重算失败时 `data_only` 读公式为 `None`，正确答案被判错 | 标记 `eval_error`；正式配对比较默认拒绝含此错误的运行，诊断比较可排除对应配对任务；重跑需由调度流程执行 | 评测噪声与策略失败分离；训练 Worker 当前仍返回 score=0，尚需专门处理训练中的基础设施错误样本 |
| golden 缓存了未重算的版本，导致某任务在之后**每次运行都误判** | 旧代码 `recalc` 失败返回 False 不抛异常，异常保护没触发 | 重算失败拒绝写入缓存并显式报错 | 消除持久性误判 |
| Bridge/Worker 常规日志刷屏、打印代码 | 执行日志记录完整 Python 源码 | 改为参数摘要和代码长度，轨迹保留完整动作 | 常规执行日志更简洁；Trainer 评测样例和生成展示仍可能包含源码，此项不是所有输出渠道的脱敏保证 |

### 4.14 Harness 全部常量速查

| 类别 | 常量 | 值 | 位置 |
| --- | --- | --- | --- |
| 回合 | `MAX_STEPS` | 10 | 启动参数 |
| 回合 | 每轮最大调用 | 4 | `projection.py` |
| 长度 | prompt / response / model_len | 8192 / 16384 / 24576 | 启动参数 |
| 历史 | history_length / obs 截断 | 2 轮 / 2,000 字符 | `manager.py` |
| 观察 | 单轮合并 / run_python / 读工具 | 12,000 / 6,000 / 10,240 字符 | `envs.py` / `bridge.py` / `read_tools.py` |
| 读 | inspect cells / summary / find | 400 / 100,000 / 1,000 | `read_tools.py` |
| 写 | 单次 write/clear/fill/format | 50,000 cells | `write_tools.py` / `structure_tools.py` |
| 写 | 单轮累计 | 100,000 cells | `envs.py` |
| 写 | 单值字符 / 文件锁 / 工作簿大小 | 8,192 / 30s / 100MB | `write_tools.py` / `read_tools.py` |
| 重算 | 次数 / 超时 / 范围 / 格数 | 1 次 / 120s / 10 / 400 | `recalc_tools.py` / 环境变量 |
| 执行 | run_python 超时 | 60s | `bridge.py` |
| 稳定性 | Ray actor 超时 / 心跳 | 600s / 60s | 环境变量 |
| 诊断 | 扫描预算 / 样例数 / 单值序列化 | 200,000 / 20 / 200 字符 | `badcase_diagnostics.py` |

### 4.15 Base 工具配置消融（2026-10-11 已完成结果）

固定 Qwen3-4B-Thinking-2507、相同 399 条任务、10 轮预算、确定性解码、Compact History 和提交门禁，比较三档工具。

| Base 配置 | 成功数 | Success Rate | Shaped Test Score | 平均工具调用数 |
| --- | ---: | ---: | ---: | ---: |
| A0：Python | 104/399 | 26.07% | 0.1703 | 5.10 |
| A1：Native Read | 117/399 | 29.32% | 0.2246 | 5.84 |
| A2：Native Basic（历史运行） | 94/399 | 23.56% | 0.1573 | 6.48 |

来源目录均位于 `runs/parallel_eval/`：

- A0：`eval_a/grpo_spreadsheetbench_eval_base_python_20261011_153229`；
- A1：`eval_b/grpo_spreadsheetbench_eval_base_native_read_20261011_161336`；
- A2：`shufflefix_20261004_202150/eval_a/grpo_spreadsheetbench_eval_base_20261005_153016`。

A0/A1 的 Commit 为 `0744af5`；A2 为 `45426b7`。已检查两个 Commit 间受控运行时代码无变化，
变动为文档及忽略规则，但 Manifest 未记录 Dirty Diff，正式同 Commit A2 重评仍有必要。

| 配对比较 | 仅前者成功 | 仅后者成功 | 后者净增 | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: |
| Python → Native Read | 28 | 41 | +13（+3.26 pp） | 0.1480 |
| Native Basic → Native Read | 27 | 50 | +23（+5.76 pp） | 0.0117 |

**可支持的结论**：Native Read 是当前三档 Base 配置中数值最好的一档；相对 Python 的差异尚未达到
常用 0.05 显著性阈值，相对历史 Native Basic 的配对差异达到该阈值。这些 p 值为单次两两比较，
尚未做多重比较校正或重复运行验证。

**解释假设**：结构化读取帮助定位，Python 能集中完成修改；完整写工具可能增加选择负担和步骤消耗。
该解释需要核对 gain/loss 任务验证。Native Read 的读取成功率为 99.74%，但不能由此推断所有定位错误已解决。

Native Read Base 的 117/399 接近 Native Basic GRPO 最佳点的 119/399。展示时分别讨论工具配置与后训练，
不把这两个数解释成仅由模型训练造成的差异。Teacher 更换工具集也需要 Pilot，不能直接套用 Base 的工具排名。

---

## 5. 奖励设计

### 5.1 构成

```text
R_episode = workbook_score                     # 官方 OJ：答案区全部匹配为 1，否则 0（终局一次）
          + Σ execution_penalty                # 每次 Python 失败：语法/缩进类 −0.1，其余（运行时、超时、preflight）−0.05
          - 0.1 × invalid_action_count          # Trainer 对投影无效的决策轮次扣分
```

- 原生工具失败、submit 被门禁拒绝：环境 Reward 为 0，通过错误反馈、短路和消耗轮次影响后续结果；
  当前没有额外数值惩罚。多调用投影有效性按决策轮次汇总，Invalid Action Count 不等于逐子调用失败数；
- 分解指标全部落到 W&B：`reward/workbook_score`、`reward/execution_penalty`、`reward/env_total`、`reward/invalid_action_penalty`。
- 工具成功率、校验状态与 Badcase 标签主要用于观测；KL 系数 0.005 用于 Actor Loss 正则，当前不进入环境 Reward。

对应简历中的“工作簿最终得分（官方评测）+ 工具调用合法性（invalid action penalty）+ 执行结果（Python 执行惩罚）”。

### 5.2 终局正确性为主，过程惩罚为辅

当前选择二值主奖励，是因为它直接对应官方任务成功；部分完成度尚未进入训练 Reward。
`test_score` 是含错误惩罚的轨迹得分，不能称为单元格部分正确率。

历史 Base vs Step5 的 `test_score` 从 0.1940 升到 0.2137，但成功数从 122 降到 113，说明训练得分与
任务成功可能方向不同。Step110 工具错误比 Step100 更少（84 vs 100），成功却少18条，进一步说明
单个过程指标不足以评价任务能力。这些观察不证明部分完成奖励或过程塑形必然无效。

失败任务平均 `answer_match_ratio` 还受失败任务集合变化影响，不能直接用跨 Checkpoint 的均值相关性
判断是否适合作 Reward。拟采用渐进实验：先加入相对原始工作簿的完成度，再评估是否扩展过程惩罚。

```text
R_v1 = 0.75 × official_success + 0.25 × progress + 现有错误惩罚
progress = clip((output_match - input_match) / (1 - input_match), 0, 1)
```

这是待验证方案，尚未生效。`input_match=1`、空目标、扫描截断及基础设施错误需单独定义处理规则；
完整成功时，进度分按完成处理。匹配只在隐藏离线评分侧计算，不向 Agent 展示 Golden。
第一轮仅改完成度，不同时添加效率、格式、范围等大量系数。比较 Success Rate、组内 Reward 方差
以及无操作获得奖励的比例，再决定是否继续扩展。

### 5.3 惩罚为什么区分语法错误与运行时错误

语法错误可在执行前静态检测，采用更重惩罚（−0.1）；运行时错误可能来自结构假设或 API 使用错误，
采用较轻惩罚（−0.05）。这是工程启发式，独立效果尚未消融。单次惩罚小于成功奖励1.0，但多次累计后
仍可能影响轨迹排序，不能仅凭单次幅度断言不会抑制探索。

---

## 6. GRPO 训练

### 6.1 配置（NJ5 shufflefix run `20261004_202150`，manifest 已核对）

| 项 | 值 | 说明 |
| --- | --- | --- |
| 基座 | Qwen3-4B-Thinking-2507 | 全程固定，不同时换模型 |
| 资源 | 2 节点 × 8 GPU，TP=2 | vLLM rollout |
| batch | 16 个任务 × group 8 = 128 条 rollout/step | 同一 group 8 条同任务 |
| 优势估计 | GRPO（组内归一化） | `algorithm.adv_estimator=grpo` |
| 学习率 | 3e-7 | 早期 1e-6 出现生成退化 |
| KL | `kl_loss_coef=0.005`，`low_var_kl`，不进 reward | 早期 0.001 |
| 采样 | 训练 temperature 0.6 / top-p 0.95 / top-k 20；验证 temperature 0、不采样 | 验证确定性 |
| 训练数据 | `train_hermes.parquet` 5,925 条，确定性 shuffle（seed 0） | 见 6.3 |
| 在线验证 | 固定 64 条 fast-val，每 10 步 | 只做健康检查 |
| 保存 | 每 10 步，含 HF 权重 | 便于 full-399 评测 |

### 6.2 为什么调低学习率、调高 KL

- `20260901_114006`：后期严重生成退化，输出 `201688C".S".S".S"...` 这类乱码，1,091/1,168 条没有动作——这不是表格推理失败，而是策略崩溃；
- `20260912_194504` step 34：`actor/kl_loss=7.532`、response clip 0.111，训练成功率从 0.453 骤降到 0.078；
- 因此改为 lr 3e-7、KL 0.005；NJ5 长训练中 `ppo_kl` 无数值崩溃、无梯度异常。

### 6.3 训练任务 shuffle（一个关键的数据 bug）

- **现象**：SG1 run `20261001_221238` 按 parquet 原顺序取任务；100 步 × 16 任务只覆盖了前约 1,600 个位置（共 5,925 条），
  任务分布严重偏斜。最佳 Step 70 只比 Base 多 5 条，McNemar **p=0.625**，无显著提升；
- **修复**：按 `blake2b(seed:epoch:index)` 生成每个 epoch 的确定性排列；验证集顺序不变；seed 写入 manifest（`dataset.py` `_spreadsheet_rl_permutation`）；
- **结果**：NJ5 shufflefix run Step 100 达到 119/399，McNemar **p=0.005228**。
- 注意：这是不同集群、不同 Base 的两组实验，不能将收益全部归因于 Shuffle。需要固定其余条件的
  Shuffle 开/关消融确认独立贡献，另用第二个训练 Seed 检查提升是否可复现。

### 6.4 其他训练基础设施修正

- 训练 reset 推进 seed、验证 reset 固定 seed：保证同一次训练的每个验证点评测同一组任务，曲线可比；
- `EPOCHS < TOTAL_TRAINING_STEPS` 时启动直接失败：当前数据生成方式每 epoch 只有 1 个 batch，避免“想训 80 步却 20 步就结束”；
- 记录窗口内每 Step 约33–37分钟，模型生成约20–25分钟；Trainer Reward 聚合约2秒。
  终局 LibreOffice 重算与工作簿比较已包含在 Rollout 环境执行阶段，2秒不代表整个评测成本。
  模型生成占主要耗时，环境执行、终局评测和 Reward 聚合需分别统计。

### 6.5 训练动态（NJ5，Step 1–10 vs Step 102–111）

| 指标 | 前期 | 后期 | 解读 |
| --- | ---: | ---: | --- |
| train success | 0.2477 | 0.2814（81–90 步峰值 0.3241 后回落） | 有提升但不单调 |
| valid action | 0.8245 | 0.9431 | 协议学会了 |
| parser invalid | 0.1755 | 0.0569 | |
| Python error | 0.0514 | 0.0332 | |
| preflight reject | 0.0300 | 0.0056 | |
| multi-call all-success | 0.6760 | 0.8275 | |
| entropy | 0.2248 | 0.1036 | 策略收缩 |

结论：动作协议与工具执行指标改善更明显；语义错误仍大量存在，后期同时观察到 Entropy 下降、
Inspect 使用减少和 Recalc 使用很少。策略收缩是解释退化的候选因素，尚未完成独立因果验证。

---

## 7. 评测体系

### 7.1 Full-399 配对评测

- 确定性解码、399 条全量、`VAL_SIZE=399` 与 `VAL_CONCURRENCY=64` 解耦（64 个 actor 分批跑完，不必同时开 399 个）；
- 对比脚本 `compare_spreadsheetbench_evals.py`：
  - 要求每个 task ID 恰好一次、两边任务集合完全一致；
  - 校验 16 个关键配置字段 + git commit 一致，否则拒绝比较；
  - `eval_error` 任务默认中止比较（诊断模式可放行但从分母剔除）；
  - 输出 Wilson 95% CI、配对差 95% CI、**exact McNemar p 值**、Base-only / Candidate-only 任务列表与逐任务失败标签转移。

### 7.2 为什么不用 fast-val 选 checkpoint

- 早期 32 条验证：每条 3.125 个百分点，step 10 比 step 0 多 2 条就显示 +6pp；
- step 5 在 32 条上 9/32 领先 Base 7/32，到 full-399 变成 113 vs 122（落后）；
- NJ5 的 64 条 fast-val **没有识别出** Step 100 的提升，而 full-399 显著。
- 所以：fast-val 只看服务健康和明显退化，正式结论只用 full-399 配对检验。

### 7.3 可比性边界

不同 commit / 门禁 / 工具实现 / mutation 上限都会改变 Agent 行为。项目里五组实验的 Base 分别为
17.04%（NJ-A）、24.31%（SG-A）、22.56%（SG-B）、25.56%（SG1 无 shuffle）、23.56%（NJ5）。
训练收益使用同组Base，历史记录作为演进背景；工具消融则明确指定工具集为变化因素，并核对其余条件。

---

## 8. Badcase 可观测性

### 8.1 分层设计

- **官方评分路径完全不变**；只有官方判定失败的任务，在 `full` 模式下额外加载 input / golden / output 三份工作簿（各取公式视图与值视图）做诊断；
- 训练默认 `light`（只记执行侧证据，不扫工作簿，保护吞吐）；正式评测用 `full`；
- 共享 20 万坐标扫描预算、最多 20 个紧凑样例、不保存失败 xlsx、诊断异常有显式状态。

### 8.2 标签体系（多标签，不做互斥单因归因）

| 层 | 标签 | 判定 |
| --- | --- | --- |
| 执行 | `no_submit` / `time_limit` | 未提交 / 撞步数上限 |
| 执行 | `submit_gate_unrecovered` | 被门禁拒绝后再没成功提交 |
| 执行 | `tool_call_truncated` / `mutation_budget_exceeded` | 超 4 调用 / 超预算 |
| 执行 | `tool_error` / `formula_validation_error` | 任一工具失败 / 公式校验失败 |
| 执行 | `execution_clean_score_zero` | 未记录到 Parser、工具或校验错误，但最终得分为0；不代表动作语义正确 |
| 工作簿 | `answer_partially_correct` | 答案区部分匹配 |
| 工作簿 | `answer_range_untouched` | 需要改的答案区一格没改 |
| 工作簿 | `likely_wrong_sheet_or_range` | 答案区没改 + 答案区外有改动 |
| 工作簿 | `formula_to_static` | golden 是公式，output 是静态值 |
| 工作簿 | `formula_result_mismatch` | 两边都是公式但值不同 |
| 工作簿 | `format_mismatch_info` | 样式不同（仅信息，官方不按格式扣分） |
| 排除 | `eval_error` | 评测器故障；正式比较拒绝，诊断统计可排除；训练Worker的零分处理另待修正 |

### 8.3 NJ5 结果与解读

| 标签 | Base | Step 100 | Step 110 |
| --- | ---: | ---: | ---: |
| 成功 | 94 | **119** | 101 |
| `tool_error` | 178 | 100 | 84 |
| `no_submit` | 24 | 16 | 19 |
| `tool_call_truncated` | 13 | 4 | 9 |
| `formula_to_static` | 92 | 63 | 63 |
| `likely_wrong_sheet_or_range` | 44 | 29 | 27 |
| `formula_result_mismatch` | 173 | 169 | 182 |
| `execution_clean_score_zero` | 127 | 180 | 214 |

- Base→Step 100：Base 失败 Step 100 成功 50 条，反向 25 条，净 +25，p=0.005228；
- Step 100→Step 110：净−18，单次未校正配对p=0.02734，观察到回落；跨Checkpoint选点偏差尚未校正；
- `execution_clean_score_zero` 比例从 31.83% 升到 45.11%：越训练，“工具全部成功”越不能代表任务正确；
- `formula_result_mismatch` 数量变化较小，提示公式与任务语义是优先排查方向；失败标签多标签共存，
  并非同一批任务的互斥原因，仍需逐任务 gain/loss 核对。

---

## 9. SFT 数据合成 Pipeline（设计阶段）

> 状态：`md/spreadsheet_sft_data_synthesis_pipeline.md` 前七部分已经整理完成，覆盖 Teacher API、轨迹转换、
> 三级过滤、专项校验和 LoRA SFT 配置、Checkpoint 选择、GRPO 衔接。数据生产、SFT 训练和对照实验待实施。

### 9.1 流程

```text
5,925 条训练任务 → 固定 split manifest（seed 固定，记录 hash）→ 分层抽样约 1,000 条 SFT 合成池（与 GRPO 任务池按来源隔离）
  → Teacher（OpenAI-compatible API）在同一个 Harness、同一套 native_basic 工具和安全约束下交互（每任务最多 4 次，有界采样）
  → Attempt 结束后官方评测（Golden答案值与逐单元格差异不进入Teacher上下文）
  → 一级：完整性与安全（文件可打开、无越界、无预算超限、无未记录修改）
  → 二级：正确性与独立重放（重放动作序列得到同样结果，排除未记录状态依赖）
  → 三级：示范质量（无工具错误/回滚/门禁拒绝；校验后若再改必须重校；修改有观察依据；无严重冗余）
  → 质量分级 A+/A 入主数据，B 留待审计，C 只做 badcase
  → Canonical Trajectory → 逐决策轮次导出（与学生推理 prompt 完全一致）
```

### 9.2 关键设计点（面试可讲）

- **在真实 Harness 中采样**：复用 Prompt、工具与约束，减少序列化和环境差异；Teacher 与 Student 的
  状态访问分布仍可能不同，不能认为分布完全一致；
- **官方评测只做事后筛选**：防止答案泄漏进示范；
- **独立重放**：检查结果可复现，过滤未记录状态依赖；语义质量仍需专项校验，Replay本身不证明泛化；
- **fixed-399 不参与任何生成、过滤规则开发或训练**：保证评测干净；
- **合成池与 RL 池隔离**：避免 SFT 记住 RL 阶段要探索的任务。

### 9.3 SFT 训练设计与 GRPO 衔接

SFT 训练每条轨迹中的下一动作：输入为任务、当前观察和历史，输出为规范化工具调用；Prompt 不计算 Loss。
使用 Student Tokenizer 检查长度和 Prompt Token 一致性，超长样本禁止静默截断。

初始方案为 LoRA Rank32、Alpha64、学习率2e-5、2个 Epoch，全局 Batch64；Combined Length 按
Token P99.5 和显存测试选择。这些是待 Pilot 验证的配置，不是已经运行的训练结果。

流程为小样本过拟合检查 → 10%数据 Pilot → 全量SFT → Dev交互评测选点 → 合并Adapter →
验证完整HF模型可加载 → 作为GRPO初始化。SFT Dev选择Checkpoint，fixed-399用于阶段性正式比较。

SFT 侧重示范工具选择、状态读取、修改和提交习惯；GRPO 通过环境结果进一步优化任务成功。
完整矩阵为 Base、Base→SFT、Base→GRPO、Base→SFT→GRPO，现有29.82%属于 Base→GRPO。

Teacher 工具配置当前设计为 Native Basic。Base 的 Native Read 消融结果只提示需要重新检查工具配置，
不能直接决定外部Teacher的最佳工具集；数据生产前需在训练来源任务上做小规模Teacher Pilot。

---

## 10. 实验演进时间线

| 时间 | 实验 / 事件 | 现象 | 采取的行动 |
| --- | --- | --- | --- |
| 09-01 | full 系列 3 个 run | 生成退化、无 tool call、错误工具名、fenced JSON | 宽松投影 + 错误反馈；后续降 lr、升 KL |
| 09-06 | 8192 response / 6 steps | clip 0.294、invalid 0.24 | 16384 / 10 steps / compact history |
| 09-06 | 114633 | 协议问题解决，但轨迹全撞 10 步；python_error 统计口径错误 | 改为按 returncode 统计 |
| 09-09 | 144747 | 语法错误 −77%，success 不变 | 确认“执行质量 ≠ 任务成功” |
| 09-12 | 194504 native_basic | `455-35` 校验 >600s；KL 尖峰 7.532 | 顺序扫描；Ray 超时与心跳 |
| 09-18 | full 20260918 | `ready=0/128` reset 卡死 | 惰性 loader |
| 09-19 | Base vs step5 full-399 | 122 vs 113；test_score 反向 | checkpoint 只看 success；fast-val 固定 |
| 09-20~24 | 迁移 | — | fill_formula、multi-call、format/结构、recalc、preflight、submit gate、评测/对比工具 |
| 09-26~30 | 三组跨集群 | 组内 +13~+49 条；`1_399-14` 卡死整批 | 50K/100K 预算（数据驱动定阈值） |
| 10-01 | SG1 无 shuffle | p=0.625 | 发现 shuffle 缺失 |
| 10-01 | badcase observability | — | light/full 诊断、标签体系 |
| 10-04~11 | NJ5 shufflefix | Step 100 119/399，p=0.005；Step 110 退化 | 结论：瓶颈转向公式语义 |
| 10-10 | SFT pipeline 设计 | — | Teacher Rollout + 三级过滤 |
| 10-11 | Base工具消融 | Python104/399；Native Read117/399 | 补充工具选择与调用预算分析；历史Native Basic94/399作为对照 |
| 10-14 | 展示文档更新 | 前七部分SFT设计已齐全，尚待实施 | 主线统一为Harness→数据合成与SFT设计→GRPO与Reward→结果 |

---

## 11. 结论、局限与下一步

**已证明**

1. Harness 把工具错误、损坏工作簿、无限调用、旧版本提交变成了可约束、可诊断的失败；
2. 在固定 Native Basic 和统一 Hardened 协议下，一次 GRPO 训练的最佳点为23.56%→29.82%
   （+25条，未校正的单次配对 `p=0.005228`）；
3. 工具执行指标改善与任务成功可以分离——这是项目最有价值的经验结论。

**设计成果**：Teacher API、数据转换、重放与过滤、专项校验及 LoRA SFT→GRPO 衔接方案已整理。
**消融观察**：Native Read Base 为29.32%，当前完整工具开放未直接带来更好的零样本效果；需进一步做任务级分析。

**局限（主动说出来比被问出来好）**

1. 单 seed；Step 100 是多个 checkpoint 中事后挑出的最佳点，存在选择偏差；Step 110 显著回落；
2. 训练集（ExcelForum）与 Verified 评测集之间尚未做 ID/hash 重叠审计，严格称为 held-out 之前需要补齐；
3. LibreOffice 与 Excel 不完全等价（新函数、动态数组、部分格式）；
4. `run_python` 只有进程级隔离，没有容器级资源/网络隔离（可信研究集群场景可接受）；
5. 删除行列后 openpyxl 不保证重写依赖公式；
6. 观察只有固定预览，没有按任务语义选择上下文。

**下一步**

1. 第二 seed 复现 + 预先声明的 checkpoint 选择规则 / early stopping；
2. 语义 Observation Projection（按 instruction 与 answer_position 选表头、公式、非空边界）；
3. 公式语义：针对 `execution_clean_score_zero + formula_result_mismatch` 做人工核对样本；提高 recalc 闭环率；
4. Teacher SFT 冷启动，再做 GRPO；
5. Manifest 增加 dirty flag、diff checksum、dataset hash、task fingerprint。

---

## 附录 A：关键文件索引

| 文件 | 职责 |
| --- | --- |
| `envharness/bridges/spreadsheetbench/bridge.py` | 环境主体：沙箱、工具路由、事务、版本门禁、校验、官方评分 |
| `envharness/bridges/spreadsheetbench/read_tools.py` | 读工具与统一错误协议 `ReadToolError` |
| `envharness/bridges/spreadsheetbench/write_tools.py` | 写工具、文件锁、原子保存、公式语法校验 |
| `envharness/bridges/spreadsheetbench/structure_tools.py` | 格式、行列删除、sheet 管理 |
| `envharness/bridges/spreadsheetbench/recalc_tools.py` | 临时副本重算读值 |
| `envharness/bridges/spreadsheetbench/python_preflight.py` | AST 预检 |
| `envharness/bridges/spreadsheetbench/process_utils.py` | 进程组超时回收 |
| `envharness/bridges/spreadsheetbench/online_judge_eval.py` | 官方评测器（vendored） |
| `envharness/bridges/spreadsheetbench/badcase_diagnostics.py` | 失败工作簿诊断 |
| `envharness/bridges/spreadsheetbench/dataset.py` | 数据加载、惰性物化、确定性 shuffle |
| `rl/envharness_rl/spreadsheetbench/projection.py` | 文本→动作投影、多调用约束 |
| `rl/envharness_rl/spreadsheetbench/envs.py` | Ray worker、多调用执行、预算、短路、reward 分解 |
| `rl/envharness_rl/spreadsheetbench/manager.py` | Prompt、历史压缩、诊断指标、轨迹落盘 |
| `rl/envharness_rl/spreadsheetbench/rollout_summary.py` | badcase 分类与聚合 |
| `rl/scripts/submit_spreadsheetbench_eval.sh` | full-399 评测入口 |
| `rl/scripts/compare_spreadsheetbench_evals.py` | 配对比较与 McNemar |

## 附录 B：面试高频数字

- 数据：训练 5,925 条；评测 399 条（上游名义 400，缺 id=54675）
- 结果：Base 94/399（23.56%）→ Step 100 119/399（29.82%），+50/−25，p=0.005228；Step 110 101/399（p=0.027 退化）
- 训练：16 任务 × 8 rollout = 128 条/步；lr 3e-7；KL 0.005；每步约 35 分钟
- 预算：4 调用/轮；50K cells/次；100K cells/轮；重算 1 次/episode；10 轮/episode
- 事故：`1_399-14` 整表格式化卡死整批；`455-35` 校验 600s→1.3s；`ready=0/128`（75.8 万次小文件访问）
- 阈值依据：已分析日志中正常 format 最大41,435 cells；超50K的fill仅6次且属于错误范围；该组full-399未触发累计预算限制
- 瓶颈：`execution_clean_score_zero` 127→180；`formula_result_mismatch` 173→169；recalc 后提交率约 0.25%
- Base工具消融：Python104/399（26.07%），Native Read117/399（29.32%），历史Native Basic94/399（23.56%）
- SFT状态：前七部分设计完成；LoRA Rank32、LR2e-5、2 Epochs为拟定配置，数据合成和训练待实施
