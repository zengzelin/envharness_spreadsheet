# Spreadsheet Work Agent 项目实施计划

> **面向 Agent 执行者：** 必须使用子技能 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐项实施本计划。步骤使用复选框（`- [ ]`）跟踪。

**目标：** 将已实现的 SpreadsheetBench 运行时、工具、校验机制、GRPO 集成和实验依据整理为一个聚焦 Spreadsheet Work Agent 的项目，并由 Spreadsheet Agent Harness 提供支撑。

**架构：** 新增一份自包含的项目首页，让现有实验文档和 RL 文档指向该首页，同时保留各自的操作细节。项目叙事与仓库原有的环境塑造研究相互独立：首页只描述当前 Spreadsheet GRPO 路径实际使用的代码，未完成能力统一标为后续规划。

**技术栈：** Markdown、Mermaid、Python/openpyxl、LibreOffice、Ray、verl-agent、GRPO、SpreadsheetBench/Spreadsheet-RL。

**设计说明：** `md/spreadsheet_work_agent_harness_design.md`

## 全局约束

- 产品名称统一使用 **Spreadsheet Work Agent**，运行时与训练基础设施统一使用 **Spreadsheet Agent Harness**。
- 不把 Setup、Rules、Link、EnvRigger 或技能归纳描述为当前 Spreadsheet 训练路径的组成部分。
- 项目首页不展示失败或已经被替代的实验。
- 首页声明的每项已实现能力都必须能追溯到当前代码或测试。
- 将 `94/399 -> 119/399` 表述为单次运行中最强的 399 条全量结果，并明确 checkpoint 事后选择及缺少第二个 seed 的局限。
- 不得将工具执行成功、解析器判定有效或 badcase 标签等同于官方工作簿任务成功。
- 不得写入凭据、内部 W&B 地址或机器相关 API Key。

## 审查重点

- 读者可能混淆仓库名称和项目运行时：首页必须明确定义两个项目名称，且不能宣称使用了环境塑造方法。
- 读者可能误以为规划能力已经实现：所有规划内容必须使用将来时，并与已实现能力分开。
- 读者可能复制过时的启动命令：所有正式命令必须使用 `native_basic`、完整校验配置和占位凭据。
- 读者可能过度解读 Step 100：结果表附近必须同时注明单次运行和事后选择 checkpoint 的局限。
- 历史计划存在未勾选的旧任务：状态索引必须区分已被替代的步骤与确实尚待验证的工作。

---

### 任务 1：创建项目首页

**文件：**

- 新建：`SPREADSHEET_WORK_AGENT.md`
- 参考：`md/spreadsheet_work_agent_harness_design.md`
- 参考：`envharness/bridges/spreadsheetbench/bridge.py`
- 参考：`rl/envharness_rl/spreadsheetbench/envs.py`

**接口：**

- 输入：设计说明中的命名、架构、已实现能力边界和结果局限。
- 输出：所有 Spreadsheet Work Agent 文档共同链接的规范项目入口。

- [ ] **步骤 1：撰写规范的项目叙事**

  补充问题定义、检查—编辑—重算—校验—提交流程、架构图、已实现的工具/安全/校验层，并简要区分 Work Agent 与 Harness。

- [ ] **步骤 2：补充经过核对的快速开始命令**

  说明数据预检、外部 Ray 启动、训练、399 条全量评测和 badcase 输出。模型路径和凭据使用占位符；完整变量说明链接到 `rl/README.md`。

- [ ] **步骤 3：补充有边界的结果章节**

  同时列出 Base `94/399`、Step 100 `119/399`、精确 McNemar `p=0.005228`，以及单次运行和事后选择 checkpoint 的局限。

- [ ] **步骤 4：补充当前局限与后续规划**

  包括语义化观察投影、公式语义、manifest 溯源、第二个 seed 复现和 checkpoint 选择；不包含环境塑造或技能归纳。

- [ ] **步骤 5：核对术语与无依据声明**

  ```bash
  rg -n "EnvRigger|Setup|Rules|Link|two.layer|两层" SPREADSHEET_WORK_AGENT.md
  rg -n "94/399|119/399|0.005228|single|单次|second.seed|第二.*seed" SPREADSHEET_WORK_AGENT.md
  ```

  预期：第一条命令不返回任何项目方法声明；第二条确认结果和局限出现在同一文档中。

### 任务 2：让项目易于发现，同时保留上游历史

**文件：**

- 修改：`README.md`
- 修改：`experiments/spreadsheetbench/README.md`

**接口：**

- 输入：任务 1 产出的规范 `SPREADSHEET_WORK_AGENT.md`。
- 输出：从仓库首页和 SpreadsheetBench 实验目录进入项目的稳定导航。

- [ ] **步骤 1：在根 README 中添加重点项目入口**

  新增简短章节并链接 `SPREADSHEET_WORK_AGENT.md`。说明项目提供原生表格运行时、受校验保护的动作、GRPO 训练和失败诊断；不改写上游 EnvHarness 论文介绍。

- [ ] **步骤 2：在 SpreadsheetBench 实验 README 中添加当前训练提示**

  在文档开头链接规范项目页，说明当前加固后的 GRPO 命令位于 `rl/`；保留原有复现实验说明，并标注为独立的历史/研究流程。

- [ ] **步骤 3：核对链接与命名**

  ```bash
  rg -n "SPREADSHEET_WORK_AGENT.md|Spreadsheet Work Agent|Spreadsheet Agent Harness" README.md experiments/spreadsheetbench/README.md
  ```

  预期：两个入口都链接到规范项目页，并使用统一名称。

### 任务 3：将 RL README 整理为 Harness 操作手册

**文件：**

- 修改：`rl/README.md`
- 参考：`rl/scripts/submit_spreadsheetbench_grpo.sh`
- 参考：`rl/scripts/submit_spreadsheetbench_eval.sh`
- 参考：`rl/scripts/compare_spreadsheetbench_evals.py`

**接口：**

- 输入：任务 1 的项目术语和脚本实际支持的变量。
- 输出：项目快速开始所链接的规范操作手册。

- [ ] **步骤 1：添加 Spreadsheet Agent Harness 概述与目录**

  将运行路径定义为 `verl-agent -> manager -> Ray workers -> SpreadsheetBenchEnv -> trusted evaluator`，并明确当前 Spreadsheet 训练会直接实例化 bridge。

- [ ] **步骤 2：分离环境准备、训练、评测、诊断和故障排查**

  重新组织现有内容，但不删除有效命令。为外部 Ray 集群状态、`CLUSTER_ID`、`RUN_ROOT`、399 条全量评测和 badcase 产物设置独立章节。

- [ ] **步骤 3：添加安全与可观测性配置表**

  说明工具集、提交前校验、重算次数/超时、日志级别、修改预算、最大工具调用数、badcase 模式和 shuffle seed。不是环境变量的常量应明确标为固定实现限制。

- [ ] **步骤 4：移除当前训练路径没有使用的能力声明**

  确保 Spreadsheet 章节不会暗示 Setup/Rules/Link、EnvRigger 或技能注入参与 GRPO。

- [ ] **步骤 5：对照脚本核验命令**

  ```bash
  rg -o "SPREADSHEETBENCH_[A-Z0-9_]+" rl/README.md | sort -u
  rg -n "SPREADSHEETBENCH_[A-Z0-9_]+" rl/scripts/submit_spreadsheetbench_grpo.sh rl/scripts/run_spreadsheetbench_grpo.sh rl/scripts/submit_spreadsheetbench_eval.sh
  ```

  预期：每个已记录的运行时变量都被代码使用，或被明确说明为固定常量。

### 任务 4：发布实现状态索引，废弃旧清单的当前状态含义

**文件：**

- 新建：`md/spreadsheet_work_agent_status.md`
- 修改：`md/spreadsheet_rl_next_migration_implementation_plan.md`
- 修改：`md/spreadsheet_badcase_observability_plan.md`
- 修改：`md/spreadsheet_rl_loader_optimization_plan.md`
- 参考：`md/experiment_results_20260930.md`
- 参考：`md/rollout_badcase_report.md`

**接口：**

- 输入：设计说明中的实现边界，以及代码、测试和当前实验报告中的依据。
- 输出：唯一权威的当前状态页面；旧文档保留为历史记录，不再作为互相冲突的状态来源。

- [ ] **步骤 1：创建状态矩阵**

  将事项分为“已实现”“已实现，待独立验证”“待完成”和“历史或已替代”。每项已实现声明都要附源码或实验产物依据。

- [ ] **步骤 2：为历史计划添加提示**

  在每份旧计划中补充简短说明，标明日期、细粒度复选框保留原始执行历史，以及当前状态由 `md/spreadsheet_work_agent_status.md` 统一维护。

- [ ] **步骤 3：不要机械勾选旧计划中的所有复选框**

  通过后续设计或提交实现的步骤仍按历史原貌保留。当前是否完成由状态页面决定，而不是回改旧任务细节。

- [ ] **步骤 4：核对当前待办**

  确认第二个 seed 复现、128 Actor 规模验收、manifest 溯源和语义化观察投影仍待完成；确认原生工具、校验门禁、预算、badcase 产物、shuffle 和 399 条全量对比已经实现。

- [ ] **步骤 5：校验文档改动**

  ```bash
  git diff --check
  rg -n "Implemented|awaiting|Pending|Historical|已实现|待验证|待完成|历史" md/spreadsheet_work_agent_status.md
  git diff -- README.md SPREADSHEET_WORK_AGENT.md experiments/spreadsheetbench/README.md rl/README.md md/
  ```

  预期：没有空白字符错误，4 种状态均存在，diff 只包含文档改动。

### 任务 5：最终事实核查

**文件：**

- 审查：`SPREADSHEET_WORK_AGENT.md`
- 审查：`README.md`
- 审查：`experiments/spreadsheetbench/README.md`
- 审查：`rl/README.md`
- 审查：`md/spreadsheet_work_agent_status.md`

**接口：**

- 输入：前述全部文档产物。
- 输出：一套内容一致、具备事实依据、可供评审的项目文档。

- [ ] **步骤 1：对照代码检查每条架构链路**

  使用引用的源文件核对 Manager、Worker、bridge、工具分发、校验、评测和轨迹记录等声明。

- [ ] **步骤 2：对照实验记录检查每个结果**

  对照 `md/experiment_results_20260930.md` 和 `md/rollout_badcase_report.md` 核对 `94/399`、`119/399`、`0.005228`、Step 110 退化和 badcase 数量。

- [ ] **步骤 3：检查凭据及机器相关秘密是否泄露**

  ```bash
  rg -n "WANDB_API_KEY=['\"]?[^.<{]|local-[0-9a-f]{20,}|lubanml\.woa\.com" README.md SPREADSHEET_WORK_AGENT.md experiments/spreadsheetbench/README.md rl/README.md md/spreadsheet_work_agent_*.md
  ```

  预期：项目文档没有引入凭据或内部服务地址。

- [ ] **步骤 4：执行最终文档校验**

  ```bash
  git diff --check
  git status --short
  ```

  预期：仅计划内文档发生修改，此外只保留原先就存在且无关的文件。
