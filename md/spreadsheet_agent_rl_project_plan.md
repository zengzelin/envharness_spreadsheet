# Spreadsheet Agent RL 项目整理：Codebase、数据与分阶段执行路线

## 0. 项目目标

目标不是完整复现 Spreadsheet-RL，而是基于现有开源组件，搭一个：

> **Linux-only、可多轮交互、可执行验证、可直接接入 GRPO/Agentic RL 的 Spreadsheet Agent 训练环境。**

推荐组合：

```text
SpreadsheetBench
    ↓
真实 Excel 任务 + Workbook + OJ Verifier
    ↓
EnvHarness SpreadsheetBench Bridge
    ↓
Linux-only 多轮 Spreadsheet Environment
    ↓
verl-agent / verl
    ↓
GRPO / 后续 RL 算法
```

核心原则：

1. **不依赖 Windows / Microsoft Excel**
2. **不重新造 Spreadsheet Gym**
3. **不重新实现 Verifier**
4. **先把现成组件分别跑通，再做最薄的一层 Adapter**
5. **先跑 Vanilla GRPO baseline，再做 trajectory diagnosis**
6. **最后再决定做 Credit Assignment / Verification / Inspection 等算法改进**

---

# 1. 总准备清单

## 1.1 需要准备的 Codebase

### A. EnvHarness

Repository：

```text
https://github.com/google-research/envharness
```

Paper：

```text
https://arxiv.org/abs/2608.19880
```

### 主要用途

EnvHarness 已经帮我们完成了：

- SpreadsheetBench 的 `ActionableEnv` 封装
- 每个 episode 独立 workbook workspace
- `reset()`
- `step()`
- `observe()`
- `evaluate()`
- `run_python(code)`
- `submit()`
- Python 执行
- LibreOffice 重计算
- SpreadsheetBench OJ verifier
- SpreadsheetBench Verified 400 的 corpus/eval pipeline
- 统一的 environment abstraction
- ALFWorld 上的 `verl-agent + GRPO` RL 接入样例

最重要的现成代码：

```text
envharness/
└── bridges/
    └── spreadsheetbench/
        ├── __init__.py
        ├── bridge.py
        ├── dataset.py
        ├── online_judge_eval.py
        └── tools.py
```

其中：

```text
bridge.py
```

已经实现了完整的 Spreadsheet Gym-like 环境。

Action space 非常简单：

```text
run_python(code)
submit()
```

RL 参考实现：

```text
rl/
└── envharness_rl/
    └── alfworld/
        ├── envs.py
        └── projection.py
```

后续我们主要是照这个结构新增：

```text
rl/
└── envharness_rl/
    └── spreadsheetbench/
        ├── __init__.py
        ├── envs.py
        └── projection.py
```

---

### B. SpreadsheetBench

Repository：

```text
https://github.com/RUCKBReasoning/SpreadsheetBench
```

Paper：

```text
https://arxiv.org/abs/2406.14991
```

### 主要用途

SpreadsheetBench 提供：

- 真实 Spreadsheet Manipulation tasks
- Excel input workbook
- Golden / answer workbook
- `answer_position`
- 多 test case OJ-style evaluation
- Linux / macOS / Windows evaluator
- LibreOffice formula recalculation
- SpreadsheetBench Verified 400
- 完整 912 benchmark

SpreadsheetBench 本身不需要作为主训练框架。

推荐把它理解成：

```text
Task source
+
Workbook source
+
Ground truth
+
Verifier specification
```

真正执行 agent episode 时，优先使用 EnvHarness 中已经包装好的：

```text
SpreadsheetBenchEnv
```

而不是重新调用 SpreadsheetBench 原始 inference pipeline。

---

### C. verl-agent

Repository：

```text
https://github.com/langfengQ/verl-agent
```

EnvHarness 自己的 RL 实验也是基于 `verl-agent`。

EnvHarness pin 的版本：

```text
commit:
796ed310287fa605c9292a0fce07a86d79fde05e
```

EnvHarness 已经提供自动拉取脚本：

```bash
cd envharness
bash rl/scripts/fetch_verl_agent.sh
```

会：

```text
clone verl-agent
↓
checkout 固定 commit
↓
apply EnvHarness patch
```

最终得到：

```text
envharness/
└── third_party/
    └── verl-agent/
```

### 主要用途

提供：

- GRPO
- verl
- vLLM rollout
- Ray environment workers
- multi-turn agent RL
- FSDP
- trajectory collection
- reward → policy update

第一阶段尽量不要自己重写 RL trainer。

---

## 1.2 数据准备

---

### 数据集 A：SpreadsheetBench Verified 400

这是第一阶段最推荐的数据。

下载来源：

```text
SpreadsheetBench repository
```

EnvHarness 已经明确支持。

下载：

```bash
mkdir -p experiments/spreadsheetbench/data

curl -L \
  -o /tmp/sb400.tar.gz \
  https://raw.githubusercontent.com/RUCKBReasoning/SpreadsheetBench/main/data/spreadsheetbench_verified_400.tar.gz

tar xzf /tmp/sb400.tar.gz \
  -C experiments/spreadsheetbench/data
```

最终目录：

```text
envharness/
└── experiments/
    └── spreadsheetbench/
        └── data/
            └── spreadsheetbench_verified_400/
```

作用：

```text
第一阶段 smoke
+
trajectory collection
+
小规模 RL
```

优点：

- 数据已经 verified
- 单 test case layout
- EnvHarness 已经直接兼容
- 配置成本最低

缺点：

- 只有约 400 tasks
- 对正式 RL paper 来说数据量偏小

因此它更适合：

> **先把训练 pipeline 打通。**

---

### 数据集 B：SpreadsheetBench 912

下载：

```bash
mkdir -p experiments/spreadsheetbench/data/_dl

curl -L \
  -o /tmp/sb912.tar.gz \
  https://raw.githubusercontent.com/RUCKBReasoning/SpreadsheetBench/main/data/spreadsheetbench_912_v0.1.tar.gz

tar xzf /tmp/sb912.tar.gz \
  -C experiments/spreadsheetbench/data/_dl
```

最终：

```text
experiments/spreadsheetbench/data/_dl/all_data_912_v0.1/
```

用途：

```text
正式 validation / held-out evaluation
```

特点：

- 912 个 instruction
- 2729 个 test cases
- 每个 instruction 平均约 3 个 test cases
- OJ style
- 更适合判断模型是否真正学会通用操作，而不是记住单一 workbook

EnvHarness 已经有：

```text
experiments/spreadsheetbench/data/held_out_idx.txt
```

可直接使用官方固定 held-out split。

---

### 数据集 C：Spreadsheet-RL 5925 train tasks

Repository：

```text
https://github.com/Spreadsheet-RL/Spreadsheet-RL
```

Dataset：

```text
https://huggingface.co/datasets/Spreadsheet-RL/Spreadsheet-RL
```

后续正式做训练时，可以把它只当：

> **训练数据来源**

不需要使用 Spreadsheet-RL 的 Windows reward infrastructure。

它包含：

```text
5925 ExcelForum train tasks
```

可以在项目第二阶段或第三阶段接入。

建议顺序：

```text
Verified 400
    ↓
先跑通
    ↓
Spreadsheet-RL 5925
    ↓
扩大 RL training scale
```

暂时不要第一天就下载、适配全部 5925。

---

## 1.3 系统依赖

推荐环境：

```text
Linux
Python 3.12
CUDA
GPU
LibreOffice
openpyxl
pandas
Ray
verl / verl-agent
vLLM
```

Spreadsheet Environment 最重要的额外系统依赖：

```bash
sudo apt install -y libreoffice-calc
```

Python：

```bash
pip install openpyxl pandas
```

EnvHarness：

```bash
git clone https://github.com/google-research/envharness
cd envharness

conda create -n envharness-sb python=3.12
conda activate envharness-sb

pip install -e .
pip install openpyxl pandas
```

---

# 2. 最终推荐的项目目录结构

建议尽量围绕 EnvHarness repo 做，不要同时维护三套主 repo。

```text
envharness/
│
├── envharness/
│   └── bridges/
│       └── spreadsheetbench/
│           ├── bridge.py
│           ├── dataset.py
│           ├── tools.py
│           └── online_judge_eval.py
│
├── experiments/
│   └── spreadsheetbench/
│       ├── data/
│       │   ├── spreadsheetbench_verified_400/
│       │   └── _dl/
│       │       └── all_data_912_v0.1/
│       ├── corpus.yaml
│       ├── reproduce_smoke.sh
│       └── reproduce.py
│
├── rl/
│   ├── envharness_rl/
│   │   ├── alfworld/
│   │   │   ├── envs.py
│   │   │   └── projection.py
│   │   │
│   │   └── spreadsheetbench/      # 后续新增
│   │       ├── __init__.py
│   │       ├── envs.py
│   │       └── projection.py
│   │
│   └── scripts/
│
└── third_party/
    └── verl-agent/
```

---

# 3. 三个现有项目分别已经解决什么

## 3.1 SpreadsheetBench 已经完成

```text
真实 Spreadsheet tasks
✓

输入 Workbook
✓

Golden Workbook
✓

answer_position
✓

多 test case
✓

OJ-style evaluation
✓

LibreOffice recalculation
✓

Linux evaluation
✓
```

不需要重新设计 benchmark。

---

## 3.2 EnvHarness 已经完成

```text
SpreadsheetBenchEnv
✓

Gym-like ActionableEnv
✓

reset()
✓

step()
✓

observe()
✓

evaluate()
✓

run_python()
✓

submit()
✓

per-episode workspace
✓

Python execution
✓

LibreOffice recalc
✓

OJ verifier
✓

trajectory running
✓

SpreadsheetBench experiment pipeline
✓
```

因此：

> **不要重新写 Spreadsheet Gym。**

---

## 3.3 verl-agent / EnvHarness RL 已经完成

在 ALFWorld 上已经有：

```text
EnvHarness
+
Ray Workers
+
verl-agent
+
GRPO
+
vLLM
+
FSDP
```

并提供：

```text
2-GPU smoke
+
8-GPU full training
```

我们不需要重写 GRPO trainer。

---

# 4. 当前真正缺什么

最终真正缺的核心只有：

```text
SpreadsheetBenchEnv
        ↕
verl-agent
```

也就是：

> **Spreadsheet RL Adapter**

主要新增：

```text
rl/envharness_rl/spreadsheetbench/envs.py
rl/envharness_rl/spreadsheetbench/projection.py
```

---

## 4.1 `envs.py`

作用：

把：

```text
SpreadsheetBenchEnv.reset()
SpreadsheetBenchEnv.step()
SpreadsheetBenchEnv.evaluate()
```

转换成 verl-agent 所需要的 batched / Ray env API。

大体：

```python
class EnvharnessSpreadsheetWorker:

    def __init__(...):
        self.env = SpreadsheetBenchEnv()

    def reset(...):
        response = self.env.reset(...)
        return observation, info

    def step(action):
        response = self.env.step(action)

        if response.terminated:
            result = self.env.evaluate()
            reward = result.score
        else:
            reward = 0

        return obs, reward, done, info
```

然后外层：

```text
Ray Actor
× N rollout envs
```

---

## 4.2 `projection.py`

作用：

把模型输出转成 EnvHarness 的：

```python
Action(...)
```

需要识别：

```text
run_python
submit
```

例如：

```text
<tool_call>
{
  "name": "run_python",
  "arguments": {
    "code": "..."
  }
}
</tool_call>
```

转换为：

```python
Action(
    name="run_python",
    kwargs={"code": "..."}
)
```

submit：

```python
Action(
    name="submit",
    kwargs={}
)
```

---

# 5. 分阶段执行路线

---

# Phase 0：只验证 Spreadsheet 环境

## 目标

完全不训练。

只确认：

```text
SpreadsheetBench
↓
EnvHarness
↓
run_python
↓
output.xlsx
↓
LibreOffice
↓
Verifier
```

在公司 Linux 集群能正常工作。

---

## Step 0.1 安装 EnvHarness

```bash
git clone https://github.com/google-research/envharness
cd envharness

conda create -n envharness-sb python=3.12
conda activate envharness-sb

pip install -e .
pip install openpyxl pandas

sudo apt install -y libreoffice-calc
```

---

## Step 0.2 下载 Verified 400 + 912 Eval

按照前面的命令下载。

---

## Step 0.3 运行环境检查

```bash
python scripts/check_env.py spreadsheetbench
```

需要确认：

```text
Python deps
✓

LibreOffice
✓

Verified 400
✓

912 benchmark
✓
```

---

## Step 0.4 运行 SpreadsheetBench smoke

```bash
bash experiments/spreadsheetbench/reproduce_smoke.sh
```

这一阶段主要确认：

```text
reset task
✓

run_python
✓

修改 workbook
✓

submit
✓

LibreOffice recalculation
✓

OJ evaluator
✓
```

如果这一步不能稳定跑：

> **先不要开始 RL。**

---

# Phase 1：只验证 RL Framework

## 目标

完全不碰 Spreadsheet RL。

先确认 EnvHarness 官方：

```text
ALFWorld + verl-agent + GRPO
```

在公司 GPU 集群能跑。

---

## Step 1.1 下载 verl-agent

```bash
bash rl/scripts/fetch_verl_agent.sh
```

---

## Step 1.2 运行 ALFWorld no-GPU env smoke

```bash
PYTHONPATH=..:. \
python rl/scripts/smoke_worker.py
```

具体根据 repo 当前路径调整。

---

## Step 1.3 跑官方 2-GPU GRPO smoke

EnvHarness 默认：

```text
Qwen2.5-1.5B
2 GPU
GRPO
```

目标只确认：

```text
vLLM
✓

Ray
✓

verl-agent
✓

rollout
✓

reward
✓

policy update
✓
```

这一阶段也不要改算法。

---

# Phase 2：接 SpreadsheetBench ↔ verl-agent

## 目标

实现整个项目最关键但最薄的一层：

```text
SpreadsheetBenchEnv
        ↕
verl-agent
```

---

## Step 2.1 新建目录

```text
rl/envharness_rl/spreadsheetbench/
```

新增：

```text
__init__.py
envs.py
projection.py
```

---

## Step 2.2 复制 ALFWorld adapter 作为模板

参考：

```text
rl/envharness_rl/alfworld/envs.py
rl/envharness_rl/alfworld/projection.py
```

但不要复制 ALFWorld-specific logic。

Spreadsheet 的核心 action：

```text
run_python(code)
submit()
```

环境：

```text
SpreadsheetBenchEnv
```

---

## Step 2.3 注册新的 verl-agent environment

增加类似：

```text
env.env_name=envharness_rl/spreadsheetbench
```

目标：

```text
verl-agent
↓
识别 spreadsheet environment
↓
创建 SpreadsheetBench Ray Workers
```

---

# Phase 3：最小 Spreadsheet GRPO Smoke

## 目标

只证明：

> **Spreadsheet Agent 可以在 Linux 环境内 online rollout + GRPO。**

不要追求结果。

---

## 数据

先取：

```text
20~50 tasks
```

来自：

```text
SpreadsheetBench Verified 400
```

---

## 模型

推荐：

```text
Qwen2.5-1.5B-Instruct
```

或者当前 verl-agent 最稳定支持的小模型。

不要第一版直接上 8B。

---

## 配置

大致：

```text
2 GPU
20~50 train tasks
group size = 4
max turns = 10~20
GRPO
binary final reward
few training steps
```

---

## 必须观察

### 环境

```text
reset success rate
LibreOffice failure rate
Python timeout rate
output missing rate
```

### rollout

```text
平均 turns
run_python 次数
submit 次数
Python error rate
```

### RL

```text
episode/reward/mean
episode/success_rate
actor/pg_loss
response length
KL
entropy
```

目标只有：

```text
reward 能被正确回传
+
policy 能正常 update
```

---

# Phase 4：建立 Vanilla GRPO Baseline

## 目标

得到第一个真正 baseline：

```text
Base Model
vs
Vanilla GRPO
```

---

## 模型

优先：

```text
Qwen3-4B
```

如果训练基础设施更适合：

```text
Qwen3-8B
```

也可以后面再补。

---

## Train

第一版：

```text
200~400 Verified tasks
```

后续：

```text
Spreadsheet-RL 5925 train tasks
```

---

## Eval

主 eval：

```text
SpreadsheetBench held-out 912
```

建议先用固定 subset 做快速验证。

---

## 指标

### Task

```text
Pass@1
Mean Score
```

### Agent Behavior

```text
Tool Calls
Tokens
Turns
Python Error Rate
Timeout Rate
Submission Rate
```

### Efficiency

```text
success / tool call
success / token
```

---

# Phase 5：Trajectory Diagnosis

这一阶段比马上改算法更重要。

目标：

> **先找到 Vanilla GRPO 的真实 failure mode。**

---

## 5.1 把每个 `run_python` action 分类

建议至少分：

```text
Inspection
Edit
Verification
Correction
Mixed
```

---

### Inspection

例如：

```python
print(wb.sheetnames)
print(ws.max_row)
print(ws["A1"].value)
```

只读取，不修改。

---

### Edit

例如：

```python
ws["B4"] = "=SUM(B1:B3)"
wb.save(output_path)
```

---

### Verification

修改后重新读取：

```python
print(ws["B4"].value)
```

或重新打开 output workbook 检查。

---

### Correction

在之前已经 edit 的基础上再次修改同一目标区域。

---

## 5.2 统计行为指标

每条 trajectory：

```text
num_inspection
num_edit
num_verify
num_correction

first_edit_step
first_verify_step

inspection_before_first_edit
verification_after_edit

edit_to_verify_ratio
inspection_to_edit_ratio

total_turns
total_tokens
```

---

## 5.3 Success vs Failure 对比

例如：

```text
Successful trajectories
vs
Failed trajectories
```

观察：

```text
是否成功轨迹 inspection 更多？
是否失败轨迹更早 edit？
是否成功轨迹更常 post-edit verify？
是否失败轨迹更容易 premature submit？
是否 correction 和成功显著相关？
```

---

## 5.4 Training Dynamics

不仅比较最终模型。

还应该比较：

```text
Base
Step 5
Step 10
Step 20
...
```

观察 GRPO 是否：

```text
inspection ↓
edit ↑
verification ↓
```

或者：

```text
inspection ↑
correction ↑
```

这决定后续真正要解决的问题。

---

# Phase 6：根据诊断决定研究方向

不要提前固定方法。

优先级如下。

---

## 方向 A：Inspection-Aware RL

### Hypothesis

```text
Outcome-only reward
↓
inspection action delayed credit
↓
agent 过早 edit
```

研究问题：

> 如何给 information acquisition action 更合理的 credit？

适合迁移：

```text
step-level credit
turn-level advantage
process reward
GiGPO-style grouping
```

---

## 方向 B：Verification-Aware RL

### Hypothesis

```text
Agent 做完 edit
↓
直接 submit
↓
没有 verification / self-correction
```

研究：

```text
什么时候 verify
如何奖励有价值的 verify
如何避免无意义重复检查
```

注意不要简单：

```text
每次 inspect +0.1
```

否则容易 reward hacking。

更合理：

```text
Verification
↓
发现错误
↓
Correction
↓
最终结果改善
```

才给 credit。

---

## 方向 C：Credit-Aware GRPO

原始：

```text
trajectory:
a1 a2 a3 ... aT
        ↓
final reward
```

问题：

```text
所有 step 共用 trajectory-level advantage
```

但真实作用不同：

```text
inspection
edit
verification
correction
```

可以研究：

```text
A_t = w_t × A_trajectory
```

其中：

```text
w_t
```

根据：

```text
action type
downstream success
correction outcome
state improvement
```

确定。

---

## 方向 D：EnvHarness Environment Shaping

EnvHarness 最大的额外能力是：

> 不改 reward，而改训练环境。

例如：

```text
没有 inspect
↓
直接 edit
↓
环境返回提醒 / 限制
```

或者：

```text
没做 verification
↓
submit 被 block
```

可以研究：

```text
Reward Shaping
vs
Environment Shaping
```

非常适合做一组对照：

```text
Vanilla GRPO
EnvHarness + Vanilla GRPO
Credit-aware GRPO
EnvHarness + Credit-aware GRPO
```

---

# Phase 7：扩大训练数据

如果 Verified 400 已经证明方法有效，再接：

```text
Spreadsheet-RL 5925 train tasks
```

推荐只取：

```text
数据 + workbook
```

不要使用它的：

```text
Windows Excel reward server
Spreadsheet-RL full infrastructure
```

目标变成：

```text
Spreadsheet-RL train data
+
EnvHarness Spreadsheet Environment
+
LibreOffice Verifier
+
verl-agent
```

---

# Phase 8：扩展 Evaluation

最终可以形成：

## ID

```text
SpreadsheetBench
```

## Verified

```text
SpreadsheetBench Verified 400
```

## Harder workflow

```text
SpreadsheetBench 2
```

## OOD

后续可以考虑：

```text
SheetCopilot
```

---

# 6. 推荐的第一个完整实验矩阵

## Stage 1：Baseline

| Method | Environment | RL |
|---|---|---|
| Base | Original | No |
| GRPO | Original | Vanilla GRPO |

先确认：

```text
GRPO 是否真的提升 SpreadsheetBench
```

---

## Stage 2：Behavior Diagnosis

比较：

```text
Base
vs
GRPO
```

统计：

```text
Inspection
Edit
Verification
Correction
Tool calls
Tokens
Errors
```

---

## Stage 3：最小算法

根据 diagnosis 选一个：

```text
Inspection-aware
或
Verification-aware
或
Credit-aware
```

不要同时上三个。

---

## Stage 4：EnvHarness 对照

增加：

| Method | Training Env |
|---|---|
| Vanilla GRPO | Original |
| Harness-GRPO | EnvHarness |
| Ours | Original |
| Ours + Harness | EnvHarness |

Eval 统一：

```text
Untouched SpreadsheetBench
```

---

# 7. 一个比较合理的最终论文故事

如果 diagnosis 支持：

```text
成功 agent 更会 inspection / verification
```

可以包装成：

# Learning to Inspect Before Acting:
## Credit Assignment for Spreadsheet Agents

### Observation

Spreadsheet Agent 失败并不只是因为不会写公式。

很多 failure 来源于：

```text
inspection 不足
↓
target grounding 错误
↓
premature edit
↓
lack of verification
↓
premature submit
```

### Mechanism

```text
Outcome-only reward
↓
information-gathering action delayed credit
↓
trajectory-level GRPO 无法区分各 step 的贡献
```

### Method

```text
Credit-aware / Verification-aware RL
```

### Evaluation

```text
SpreadsheetBench
SpreadsheetBench Verified
SpreadsheetBench 2
```

### Analysis

```text
Pass@1
+
trajectory behavior
+
tool efficiency
+
correction rate
```

---

# 8. 当前最优先做什么

暂时完全不要做算法改进。

严格按：

```text
1. EnvHarness SpreadsheetBench smoke

2. EnvHarness ALFWorld GRPO smoke

3. Spreadsheet ↔ verl-agent adapter

4. 20~50 task Spreadsheet GRPO smoke

5. 200~400 task Vanilla GRPO baseline

6. Trajectory diagnosis

7. 再决定方法
```

其中：

```text
1 和 2 都不需要写新代码。
```

只有前两步在公司集群稳定通过后，才值得开始实现 Spreadsheet RL Adapter。

---

# 9. 一句话项目定位

工程层面：

> **SpreadsheetBench + EnvHarness + verl-agent = Linux-Native Spreadsheet Agent RL Testbed**

研究层面：

> **利用这个 testbed 研究 Spreadsheet / Work Agent 中 inspection、verification 和 long-horizon credit assignment。**

这比直接复现 Spreadsheet-RL 更适合快速起项目，因为：

```text
没有 Windows
没有 Excel COM
没有远程 Reward Server
没有异步 HTTP Reward Queue
```

核心环境和 verifier 都已经现成，真正需要写的主要只是：

```text
SpreadsheetBenchEnv
↕
verl-agent
```

这一层 adapter。
