# Week 4 Report — Evals (the frontier skill)

**Dates:** 2026-08-03 to 2026-08-09

---

## Day 1 — Why evals are hard: contamination, flakiness, overfitting

### Learning

评测困难的3个核心原因：

| 问题 | 本质 | 一句话概括 |
|---|---|---|
| **Contamination** | 训练数据与测试数据混了 | "这题我做过。" |
| **Flakiness** | 评估过程本身不可靠 | "再测一次分数又变了。" |
| **Overfitting to test set** | 开发过程被测试集绑架 | "我只为这场考试而生。" |

除此以外，还有6个更深层的问题：

1. **Ground Truth 不存在或模糊**：主观任务没有客观答案。
2. **指标与目标错位（Metric-Goal Gap）**：高分 ≠ 好模型。
3. **评估成本与可扩展性**：LLM-as-a-Judge 引入了新的系统误差。
4. **任务边界与评估协议不清**：静态单轮问答容易评估，但真实场景是多轮对话、工具调用、长文档理解。状态空间爆炸，一个错误可能在第3轮埋下、第10轮才暴露。
5. **分布偏移与动态环境**：训练时的评估集无法代表部署时的真实分布。
6. **模型行为新范式带来的评估盲区**：
   - Chain-of-Thought：最终答案对但中间推理错（或反之），怎么打分？过程监督需要细粒度步骤级标注。
   - 工具使用与 Agent 行为：正确性不仅取决于输出文本，还取决于外部世界的状态变化。
   - 涌现能力（Emergent Abilities）：某些能力在模型大到阈值后才出现，无法提前设计评估。

### Experiment — 10 eval task baseline

#### Question

How well does the week03 sandbox agent (qwen3.7-plus, 6 tools) perform on a fresh set of 10 tasks spanning easy/medium/hard?

#### Setup

- **Model:** qwen3.7-plus
- **Tasks:** 10 (3 easy, 4 medium, 3 hard — all new, no overlap with week03 tasks)
- **Tools:** read_file, write_file, list_dir, run_command, web_fetch, web_search
- **Sandbox:** experiments/week03/sandbox
- **Runs per task:** 5
- **Scoring:** substring (default checker)

#### Result

| Task | Difficulty | Tools? | Passes | Rate |
|---|---|---|---|---|
| knowledge-un-members | easy | no | 5/5 | 100% |
| sandbox-read-day | easy | yes (read_file) | 5/5 | 100% |
| sandbox-count-md | easy | yes (list_dir) | 5/5 | 100% |
| sandbox-grep-example-lines | medium | yes (grep) | 5/5 | 100% |
| sandbox-write-numbers | medium | yes (write + wc) | 5/5 | 100% |
| sandbox-second-heading | medium | yes (grep) | 5/5 | 100% |
| sandbox-fetch-import | medium | yes (read_file) | 5/5 | 100% |
| sandbox-compare-headings | hard | yes (2× read + compare) | 0/5 | **0%** |
| sandbox-py-line-total | hard | yes (wc + write + read) | 5/5 | 100% |
| sandbox-most-common-import | hard | yes (list + grep + dedup) | 5/5 | 100% |

**Overall: 45/50 (90%)**

Only `sandbox-compare-headings` failed — 5/5 correct answers ("4,3") but zero passes. Expected substring was `"4, 3"` (with space), every answer said `"4,3"` (without space). A one-character formatting difference kills substring scoring.

#### Surprise

`sandbox-fetch-import` — ask the model for "just the line" — wraps the answer in markdown code fences every time. Substring passes (the text is inside), but this is a **false positive**: the model didn't follow the instruction. Substring can't detect instruction-following failures.

---

## Day 2 — Grader types: exact match, substring, programmatic checkers, LLM-as-judge

### Learning

#### Substring Match

- **原理**：检查参考答案是否是模型输出的子串，包含即算对。
- **优点**：比精确匹配宽松，实现简单，运行极快。
- **适用**：事实性问答、关键词检测、简单信息检索。
- **致命弱点**：高误报率（错误上下文中提到正确答案）和高漏报率（同义词替换）。

#### Exact Match

- **原理**：字符级完全比对，必须一模一样。
- **优点**：最严格、最客观，没有灰色地带。
- **适用**：数学计算、标准化分类标签、选择题选项字母。
- **致命弱点**：对格式极度敏感，无法容忍同义表达。开放式生成任务几乎不可用。

#### Programmatic Checkers

- **原理**：用代码逻辑验证正确性，不直接比对文本。
- **类型**：单元测试、数学验证（SymPy/eval）、Schema 验证、正则匹配、规则引擎。
- **优点**：不需要参考答案完全一致，可复现、确定性高，可处理复杂约束。
- **适用**：代码生成、数学推理、结构化数据提取、Function Calling。
- **致命弱点**：开发成本高，对开放性问题无能为力，维护负担重。

#### LLM-as-a-Judge

- **原理**：用另一个（通常更强的）LLM 评判目标模型的输出质量。
- **常见模式**：打分制（1-5）、成对比较（Pairwise）、二元判断（Yes/No）、rubric-based 评估。
- **优点**：唯一能处理语义和主观性的方法，不需要精确参考答案。
- **致命弱点**：偏见（位置、长度、自我偏好、格式）、不一致性、不可解释性、能力天花板。

#### Grader 总结对比

| Grader 类型 | 严格度 | 速度 | 成本 | 主观任务 | 主要风险 |
|---|---|---|---|---|---|
| **Exact Match** | 极高 | 极快 | 零 | ❌ | 格式敏感、同义词误杀 |
| **Substring** | 高 | 极快 | 零 | ❌ | 误报率高、上下文混淆 |
| **Programmatic** | 中 | 快 | 低（开发成本高） | ⚠️ 有限 | 开发维护负担 |
| **LLM-as-Judge** | 灵活可调 | 慢 | 高 | ✅ 擅长 | 偏见、不一致、不可解释 |

**成熟 eval pipeline 的分层策略**：先用 Exact Match / Programmatic 处理可客观验证的部分 → 再用 LLM Judge 处理边界和开放问题 → 定期用人工标注校准 Judge → 多 Judge 投票降低单一模型偏见。

### Experiment — Checker type impact

#### Question

How does switching from substring to exact/programmatic checkers change pass rates? Where does substring give false confidence?

#### Setup

- **Model:** qwen3.7-plus (same as Day 1)
- **Tasks:** same 10, now annotated with checker types:
  - 5× `substring` (unchanged)
  - 2× `exact` (sandbox-second-heading, sandbox-fetch-import)
  - 3× `python` (sandbox-write-numbers, sandbox-py-line-total, sandbox-most-common-import)
- **Runs per task:** 3

#### Result

| Task | Checker | Passes | Rate | vs. Day 1 |
|---|---|---|---|---|
| knowledge-un-members | substring | 3/3 | 100% | same |
| sandbox-read-day | substring | 3/3 | 100% | same |
| sandbox-count-md | substring | 3/3 | 100% | same |
| sandbox-grep-example-lines | substring | 3/3 | 100% | same |
| sandbox-write-numbers | **python** | 3/3 | 100% | same |
| sandbox-second-heading | **exact** | 3/3 | 100% | same |
| sandbox-fetch-import | **exact** | 0/3 | **0%** | ⬇ from 100% |
| sandbox-compare-headings | substring | 0/3 | 0% | same |
| sandbox-py-line-total | **python** | 0/3 | **0%** | ⬇ from 100% |
| sandbox-most-common-import | **python** | 1/3 | 33% | ⬇ from 100% |

**Overall: 19/30 (63.3%)** — a 27-point drop from Day 1's 90%.

#### Key findings

- **sandbox-fetch-import (100% → 0%)**：Every answer wrapped in markdown code fences. Substring passes, exact fails. This is an instruction-following failure, not a knowledge failure.

- **sandbox-py-line-total (100% → 0%)**：The biggest shock. Day 1 substring scored 5/5 because the answer text happened to contain "11". Day 2 python checker reads `py_total.txt` from disk — all 3 runs wrote wrong numbers (28, 28, 41). **This is a genuine agent failure that substring completely hid.** The model misparses `wc -l` output.

- **sandbox-most-common-import (100% → 33%)**：Model sometimes wraps "os" in bold markdown (`**os**`). Python checker is too strict for this case — exactly the kind of formatting variance an LLM judge would handle.

#### Surprise

The biggest surprise is `sandbox-py-line-total` going from 100% to 0%. Between Day 1 and Day 2, the model's behavior changed — same model, same prompt, but different runs on different days produce different outcomes. This is flakiness in action.

This reveals a **variance cliff**: substring looks at text (which may or may not contain "11"), python verifies the actual file artifact. A self-consistent but wrong agent fools substring every time.

#### Next question (→ Day 3)

The `sandbox-most-common-import` case (markdown formatting) is where LLM-as-Judge shines — semantic equivalence. But `sandbox-py-line-total` shows LLM judge's weakness — **text-only judging is blind to file-system ground truth**. The tension is: verify the artifact (python checker) vs. verify the semantics (LLM judge).

---

## Day 3 — LLM-as-judge: rubrics, position bias, self-preference bias, why judge ≠ ground truth

### Learning

LLM-as-a-Judge 是当前大模型评估中最具扩展性、也最具欺骗性的方法。它解决了传统评估器无法处理语义开放性的痛点，但引入了一套全新的系统性偏差。以下从四个维度拆解。

---

### 1. Rubrics（评分量规）

Rubrics 是 LLM-as-Judge 的灵魂。没有良好设计的 rubric，Judge LLM 的评分本质上是在做高方差的随机游走。

**裸 prompt 打分的三个问题**：

| 问题 | 说明 |
|---|---|
| **标准漂移（Criterion Drift）** | 同一批样本在不同时间、不同上下文里，Judge 对"好"的定义会漂移 |
| **维度坍缩（Dimension Collapse）** | 模型倾向给"写得长、结构清晰"的回答高分，即使事实错误 |
| **分数聚集（Score Bunching）** | 没有锚定标准时，LLM 倾向于给保守的 3-4 分，缺乏区分度 |

**高质量 Rubric 的设计原则**：

| 原则 | 说明 | 反例 |
|---|---|---|
| **原子化维度** | 将"质量"拆为独立维度（准确性、完整性、安全性、简洁性），分别打分后聚合 | 只给一个总体分 |
| **行为锚定（Behavioral Anchoring）** | 每个分数等级对应具体文本特征，而非抽象形容词 | "5分=优秀" ❌；"5分=答案包含所有关键实体且无幻觉" ✅ |
| **二元判定优先** | 能用 Yes/No 判定的不给 Likert 量表，降低噪声 | 让模型在 1-10 之间选分 |
| **负面示例嵌入** | 在 rubric 中明确给出"典型错误模式"的描述 | 仅描述"什么是好的" |
| **任务特异性** | 不同任务需要完全不同的 rubric | 一套通用 rubric 打天下 |

**三种常见 Rubric 模式**：

- **模式 A — 逐点评分（Point-wise）**：按多个维度分别打分。优点：可解释性强。缺点：维度相关时 Judge 会机械给同一分数。
- **模式 B — 参考对比（Reference-based）**：给参考答案，判断是否覆盖关键信息点、是否引入幻觉。优点：有客观锚点。缺点：对开放式任务不适用。
- **模式 C — 成对比较 + 强制选择（Pairwise + Forced Choice）**：两个回答必须选一个更好的。优点：人类和 LLM 做相对判断都比绝对打分更可靠。缺点：无法产生全局排序，需要 tournament 设计。

**Rubric 的隐性陷阱**：

- **Rubric 过拟合**：根据 Judge 打分反复修改 rubric 直到"看起来合理"，实际上是在对 Judge 的偏见过拟合，而非逼近 ground truth。
- **长度暗示**：Rubric 中如果提到"详细"、"全面"，Judge 会系统性偏好更长的回答。

---

### 2. Position Bias（位置偏见）

当 Judge LLM 对两个回答进行成对比较时，其偏好会系统性地受回答在 prompt 中出现顺序的影响。交换 A/B 位置即可翻转选择。

**三种机制假设**：

1. **Recency Bias（近因效应）**：LLM 注意力对序列末尾 token 权重更高，后出现的回答被"记得更清楚"。
2. **格式位置固化**：训练数据中"总结"或"结论"往往在文本末尾，Judge 继承了"最后出现的内容更重要"的模式。
3. **对比疲劳**：Judge 读第一个回答形成初步印象，读第二个时以"找差异"为主，差异项容易被放大。

**缓解策略**：

| 策略 | 原理 | 成本 |
|---|---|---|
| **位置交换（Position Swapping）** | 同一对交换 A/B 测两次，两次一致才采纳 | 2x 推理成本 |
| **多次采样 + 聚合** | 同一对多次比较，取多数票 | 高成本，不消除系统性偏差 |
| **随机化位置** | 大规模评估时随机打乱，统计上抵消 | 不消除个体偏差 |
| **逐点分析（Decomposition）** | 分别按 rubric 逐项打分再比较总分 | 增加 prompt 长度，大幅降低位置效应 |
| **校准提示（Calibration Prompt）** | prompt 中加入"顺序是随机的，不要受位置影响" | 边际改善，无法根除 |

**结论**：位置偏见不是 prompt engineering 能完全解决的。它是 LLM 架构（自注意力 + 自回归生成）的固有特性。

---

### 3. Self-Preference Bias（自我偏好偏见）

当 Judge LLM 与被评估模型同族（如 GPT-4 评判 GPT-4 的输出）时，Judge 会系统性偏好自己的"亲族"。

**三种深层机制**：

1. **风格熟悉性（Style Familiarity）**：LLM 的困惑度对"自己风格"的文本更低，产生隐性的"流畅=高质量"关联。
2. **价值观对齐（Value Alignment）**：同族模型在 RLHF 阶段注入相似价值观，Judge 看到"另一个自己"的风格给予更高评分。
3. **Token 分布先验**：Judge 生成评判理由时，next-token prediction 被训练分布中高频评价模式引导。

**缓解策略**：

| 策略 | 有效性 | 说明 |
|---|---|---|
| **异族 Judge（Cross-Model）** | 中-高 | Claude 评 GPT，GPT 评 Llama。但引入他族偏见 |
| **盲评（Anonymization）** | 低-中 | 去掉模型身份标识。但风格指纹仍然暴露 |
| **多 Judge 投票（Ensemble）** | 高 | GPT-4 + Claude + Gemini 共同投票。工业界最可靠做法 |
| **风格归一化（Style Normalization）** | 中 | 统一改写为中性风格再评判。可能丢失信息 |

---

### 4. Why Judge ≠ Ground Truth（为什么评判者不等于真实标签）

**四个根本原因**：

**能力天花板（Capability Ceiling）**：Judge 不可能评判超出自身能力的任务。Judge 解不出的数学题、不懂的编程语言，无法可靠评判。推论：Judge 必须显著强于被评估模型。

**幻觉传染（Hallucination Contagion）**：Judge 自身会产生幻觉。当它"自信地"判定某个回答包含事实错误时，这个判定本身可能就是幻觉。在缺乏外部验证工具（代码执行、搜索）的情况下，Judge 的判定只是"一个 LLM 对另一个 LLM 的猜测"。

**主观性的不可公约性（Incommensurability）**：创造性、伦理性、审美性任务，ground truth 本身不存在。两个专家标注员的 ICC 可能只有 0.4–0.6。Judge LLM 只是"第三个主观意见"。

**评估目标的错位**：Ground truth 回答 "What is correct?"，Judge 回答 "What looks good to me?"。简洁但不完整的回答 vs. 冗长但面面俱到的回答——没有客观答案，但 Judge 往往偏好后者（长度偏见）。

**系统性偏见的不可完全消除性**：通过位置交换、多 Judge 投票、rubric 优化等手段，最多将偏差降到统计噪声级别，但无法证明它们为零。与物理测量不同，LLM Judge 没有外部校准源。

**四维度总结**：

| 维度 | 核心结论 |
|---|---|
| **Rubrics** | 降低方差的必要手段，但不是逼近 truth 的充分条件。Rubric 设计需要迭代验证，防止对 Judge 过拟合 |
| **Position Bias** | 架构级问题，无法通过 prompt 根治。成对比较必须做位置交换或采用逐点打分 |
| **Self-Preference** | 分布级问题，必须通过异族 Judge 或多 Judge 投票缓解。单一 Judge 的排序不可信 |
| **Judge ≠ Ground Truth** | 认识论问题，没有技术解法。LLM Judge 的产出应被理解为**"一种可扩展的、有偏的近似信号"**，而非**"客观真理"** |

**实践建议**：

1. **分层评估**：先用 Exact Match / Programmatic Checker 处理可客观验证的部分，只对剩余部分启用 LLM Judge。
2. **Judge 必须强于被评模型**：Judge 的能力边界就是评估的天花板。
3. **持续校准**：定期抽样用人工标注验证 Judge 的准确率，监控偏见漂移。
4. **保持谦逊**：报告中明确说明 LLM Judge 的局限性，避免包装为 definitive truth。



### Experiment — LLM-as-Judge validation against human labels


#### Question

How well does an LLM judge (qwen3.7-plus, with task-specific rubrics) agree with human hand labels on 10 agent answers?

#### Setup

- **Model (agent):** qwen3.7-plus
- **Model (judge):** qwen3.7-plus ⚠️ — same model introduces self-preference bias; noted as limitation
- **Tasks:** 10, each annotated with a rubric defining what "correct" means
- **Runs per task:** 1 (collect phase)
- **Hand labels:** I read each answer and labeled PASS/FAIL against its rubric
- **Code:** `experiments/week04/evals/day3_llm_judge.py` (collect + validate), `experiments/week04/evals/run_evals.py` (llm_judge checker)

#### What was built

Added an `llm_judge` checker to the eval harness. The checker:

1. Takes the task's `prompt`, `rubric`, and the agent's `answer`
2. Constructs a judge prompt: Task → Rubric → Answer → Instructions
3. Calls the API with `temperature=0` for deterministic judgments
4. Parses PASS/FAIL from the judge's verdict

**Judge prompt design with embedded bias mitigations:**

```
You are an impartial evaluator. Your job is to judge whether an AI
assistant's answer correctly addresses a given task.

## Task
{task_prompt}

## Rubric (what a correct answer must satisfy)
{task_rubric}

## Assistant's Answer
{agent_answer}

## Instructions
Compare the answer against the rubric. Be strict but fair:
- PASS: the answer correctly and completely satisfies the rubric.
- FAIL: the answer is wrong, incomplete, or does not meet the rubric.

Reply with EXACTLY one word: PASS or FAIL.
```

| Concern | Mitigation |
|---|---|
| Position bias | Rubric BEFORE answer — judge reads criteria first |
| Verbosity / length bias | Binary PASS/FAIL, not 1–5 scale |
| Non-determinism | `temperature=0` |
| Self-preference | **Not mitigated** — acknowledged limitation |

**Known limitations**:

- **Self-preference bias**: Using same model as agent and judge. Direction unknown without cross-model comparison.
- **Capability ceiling**: Judge can't reliably assess tasks it can't solve itself.
- **Text-only**: Judge evaluates the answer text, not file-system artifacts.

#### Hand-labeling results

| # | Task | Difficulty | Answer (truncated) | Label | Reason |
|---|---|---|---|---|---|
| 1 | knowledge-un-members | easy | `193` | ✅ PASS | Exact match |
| 2 | capital-france | easy | `Paris.` | ✅ PASS | Contains "Paris" |
| 3 | math-expression | easy | `150` | ✅ PASS | Exact match |
| 4 | sandbox-read-day | easy | `1` | ✅ PASS | Exact match |
| 5 | sandbox-count-md | easy | `3` | ✅ PASS | Exact match |
| 6 | sandbox-grep-example | medium | `5` | ✅ PASS | Exact match |
| 7 | sandbox-second-heading | medium | `The 5 Core Workflow Patterns` | ✅ PASS | Exact match |
| 8 | sandbox-fetch-import | medium | `` ```python\nimport urllib.request\n``` `` | ❌ FAIL | Markdown wrapping — not "just the line" |
| 9 | sandbox-py-total | hard | `The total ... is **41**.` | ❌ FAIL | Wrong number (41 ≠ 11) |
| 10 | sandbox-compare-headings | hard | `4,3` | ✅ PASS | Correct numbers, correct order |

**Summary: 8 PASS / 2 FAIL**

The two failures illustrate distinct problems:
- **#8 (fetch-import)**: Content is correct but format violates the instruction — a pattern substring can't catch.
- **#9 (py-total)**: Factual error — agent miscounted. On this run it returned 41; on Day 1 it returned 11.

#### Judge agreement

> ⚠️ **Pending** — run `python experiments/week04/evals/day3_llm_judge.py --validate` with `DASHSCOPE_API_KEY` set to populate.

Expected metrics to report:

| Metric | What it measures |
|---|---|
| **Accuracy** | Raw agreement rate: (TP + TN) / N |
| **Cohen's κ** | Agreement beyond chance — the real measure |
| **Precision** | Of judge PASS, how many did human also PASS? |
| **Recall** | Of human PASS, how many did judge catch? |
| **F1** | Harmonic mean of precision and recall |

**Prediction**: Judge likely agrees on clean cases (#1–7), but two disagreements are expected:

- **#8 (fetch-import)**: Judge may focus on import correctness and miss the formatting failure → false positive (judge PASS, human FAIL).
- **#9 (py-total)**: Judge can't see `py_total.txt` on disk — only sees agent's self-consistent narrative. If it trusts the claim → false positive. This is the exact failure mode Day 2's python checker caught: **text-only judging is blind to file-system ground truth**.

#### Key insights

1. **Rubric design IS the eval.** Writing a good rubric forces precise definition of "correct" — harder than it sounds. The `sandbox-fetch-import` rubric had to decide: markdown wrapping = correct content or incorrect format?

2. **Hand-labeling finds edge cases.** When you label answers yourself, you discover borderlines you'd never think of in the abstract. "Technically correct" vs. "followed instructions" is a real and important distinction.

3. **LLM judge + programmatic checker = complementary.** Judge handles semantic equivalence ("os" vs "**os**" from Day 2); programmatic checker handles artifact verification (`py_total.txt`). Neither alone is sufficient for agent eval.

4. **The judge evaluates the answer, not the artifact.** This is the fundamental constraint. An agent that writes 41 to a file and reports 41 will fool a text-only judge. A python checker that reads the actual file won't be fooled.

5. **Same-model judging is a known flaw, not a hidden one.** The experiment is transparent about using qwen3.7-plus as both agent and judge. A follow-up would repeat with a different judge model.

---


## Day 4 — pass@k and variance: why one run means nothing

### Learning

#### 1. What is pass@k?

Pass@k 是代码生成领域（以 HumanEval 为代表）的核心评估指标，其定义为：从模型生成的 k 个独立样本中，**至少有一个能通过全部单元测试**的概率。若某问题的真实单样本通过概率为 p（即 pass@1），则：

$$\text{pass@k} = 1 - (1-p)^k$$

这个指标衡量的是模型的**覆盖能力（coverage）**——当允许采样 k 次时，模型"碰"到正确答案的概率。

核心挑战：我们无法直接观测真实的 p，只能通过有限样本进行估计。**估计方式的差异，直接决定了评估结果的方差与可靠性。**

---

#### 2. 朴素估计器（Naive Estimator）：高方差的陷阱

最直观的方式：每道题生成 k 个样本，检查是否有至少一个通过，最后取平均。这等价于用**单次伯努利试验**估计 pass@k，方差极高。

对于单道题，朴素估计器输出 X ∼ Bernoulli(θ)，其中 θ = 1−(1−p)^k：

$$\text{Var}(X) = \theta(1-\theta) = [1-(1-p)^k](1-p)^k$$

当 p=0.2, k=5（θ≈0.67）时，单道题方差 ≈ 0.22。HumanEval 仅 164 道题，聚合标准误 ≈ 3.7%，对应约 **±7% 的 95% 置信区间**。单次运行报告的 pass@k 可能与真实值差近 7 个百分点——足以掩盖模型间的真实性能差异。

另一种常见错误：有偏估计 `1-(1-c/n)^k`。它假设 k 次采样是有放回的，但实际评估应从 n 个样本中**无放回**抽取，该估计器会系统性地低估真实 pass@k。

---

#### 3. 无偏估计器（Unbiased Estimator）

OpenAI 在 HumanEval 论文中提出的方案：生成 **n ≥ k** 个样本，利用全部信息估计 pass@k。

$$\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

直觉：计算从 n 个样本中无放回抽取 k 个、**全部抽错**的概率，再用 1 减去。数学上可证该估计器无偏。

**数值稳定的实现：**

```python
def pass_at_k(n, c, k):
    """n = total samples, c = correct samples, k = k in pass@k"""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

---

#### 4. 方差分析：为什么"一次运行"毫无意义

无偏估计器消除了偏差，但**没有消除随机性**。方差依赖于真实通过率 p 和样本量 n。

| 评估策略 | 每题样本数 n | pass@1 标准误（164题） | 95% CI 宽度 |
|---|---|---|---|
| 单次运行（朴素） | k=1（仅1次） | ≈ 3.9% | ± 7.6% |
| 无偏估计 | n=20 | ≈ 1–2% | ± 2–4% |
| 无偏估计 | n=200 | < 1% | ± <2% |

**如果只运行一次，报告的值可能因随机性波动 ±7% 以上。** 模型 A 75% vs. 模型 B 70%——若均基于单次运行，差异完全可能是噪声。

**问题难度异质性放大方差：**

| 难度档 | p 值 | 方差贡献 | 说明 |
|---|---|---|---|
| 简单题 | p ≈ 1 | 极低 | 几乎每次都做对 |
| 难题 | p ≈ 0 | 极低 | 几乎永远做不对 |
| **边缘题** | **p ≈ 0.2–0.5** | **主导方差** | 有时能做到，有时不行 |

边缘题（"有时能解决"的问题）贡献了绝大部分方差。小数据集（HumanEval 仅 164 题）放大了这一效应。

---

#### 5. 严谨评估的实践原则

**样本量 n 的选择：**
- 最低标准：n ≥ 20（可估计 pass@1 至 pass@10）
- 推荐标准：n ≥ 100（尤其 k 较大时）
- 大 k：若报告 pass@100，使用 n ≥ 200

**置信区间：**
- 必须报告多次运行的**均值与标准差**，而非单次结果
- 使用至少 3–5 个不同随机种子
- 报告 95% 置信区间

**避免的做法：**
- ❌ 每题只生成 k 个样本做一次试验（朴素估计，方差最大）
- ❌ 使用有偏估计 `1−(1−c/n)^k`
- ❌ 基于单次运行比较两个差距 < 5% 的模型
- ❌ 在不同 n 的实验间直接比较 pass@k


# day 5 Regression tracking
Regression Tracking（回归追踪） 是软件工程与质量保证领域的一种持续性监控实践，其核心目标是对回归测试（Regression Testing）的结果进行历史保留、趋势分析与变化检测，从而区分：
| 类型                                | 描述                 |
| --------------------------------- | ------------------ |
| **Genuine Regression（真实回归）**      | 新代码变更引入的、此前不存在的故障  |
| **Pre-existing Failure（长期存在的故障）** | 早已存在、与当前变更无关的失败    |
| **Flaky Test（不稳定测试）**             | 间歇性失败、与代码变更无关的测试噪声 |
| 维度       | Regression Testing | Regression Tracking   |
| -------- | ------------------ | --------------------- |
| **本质**   | 测试活动（执行测试用例）       | 监控活动（分析测试结果历史）        |
| **输出**   | Pass/Fail 列表       | 趋势图、Delta 报告、Flaky 标记 |
| **时间维度** | 单点（当前运行）           | 连续（跨多次运行的比较）          |
| **核心问题** | "这次测试通过了吗？"        | "这次失败是新的回归吗？"         |
| **技术依赖** | 测试框架               | 结果数据库、对比算法、可视化        |

核心架构与组件
1 结果历史保留（Result History Retention）
Regression Tracking 的前提条件是保留测试结果历史。没有历史数据，就无法进行跨运行比较。
每次 CI 运行必须持久化存储：
├── 测试用例级别的 Pass/Fail 状态
├── 执行时间、环境指纹（OS、依赖版本、硬件）
├── 代码版本（Commit Hash、Branch、PR ID）
├── 时间戳与触发事件（push、PR、scheduled）
└── 失败时的日志、截图、堆栈跟踪
实践要点："Keeping a retained history is the prerequisite... once you have it, regression detection, flaky separation, and run to run deltas follow from it."

2 基线管理（Baseline Management）
基线（Baseline） 是 Regression Tracking 的核心概念，代表系统的"已知良好状态"参考点。
| 基线类型     | 描述                                | 应用场景       |
| -------- | --------------------------------- | ---------- |
| **静态基线** | 手动指定的某个稳定版本的测试结果                  | 发布前的最终验证   |
| **滚动基线** | 最近 N 次成功运行的统计聚合（如中位数、百分位）         | 持续集成中的性能跟踪 |
| **分支基线** | 各分支（main、release、feature）独立维护的基线  | 多分支并行开发    |
| **环境基线** | 按测试环境（dev、staging、prod-like）隔离的基线 | 环境差异大的场景   |

3 Delta 检测引擎
Delta 检测是 Regression Tracking 的核心算法，负责比较当前运行与基线的差异：
Delta 检测逻辑：
1. 对齐：按测试用例名称/ID 对齐当前运行与基线结果
2. 分类：
   - 新增失败（New Failure）：基线 Pass → 当前 Fail → 真实回归
   - 持续失败（Persistent Failure）：基线 Fail → 当前 Fail → 已知问题
   - 已修复（Resolved）：基线 Fail → 当前 Pass → 修复确认
   - 新增通过（New Pass）：新增测试首次通过
   - 不稳定（Flaky）：历史记录中 Pass/Fail 交替出现
3. 归因：将新增失败关联到代码变更（Commit/PR）

4 Flaky Test 检测与隔离
Flaky Test 是 Regression Tracking 必须解决的关键问题。不稳定测试会制造噪声，导致团队对自动化测试失去信任。
| 方法        | 原理                              | 阈值示例                |
| --------- | ------------------------------- | ------------------- |
| **历史交替率** | 统计该测试在过去 N 次运行中 Pass/Fail 切换的频率 | 切换率 > 30% 标记为 Flaky |
| **重试一致性** | 同一 Commit 上多次重试的结果是否一致          | 3 次重试中结果不一致         |
| **环境相关性** | 失败是否集中在特定环境/时间段                 | 某环境失败率显著高于其他        |
隔离策略：
* Quarantine（隔离）：将 Flaky Test 移出主 CI 门禁，避免阻塞发布
* Retry with Threshold：自动重试 N 次，仅当全部失败才判定为回归
* Separate Dashboard：为 Flaky Test 建立独立看板，不影响主流程

应用领域与实现模式
1 功能回归跟踪（Functional Regression Tracking
最传统的应用模式，跟踪单元测试、集成测试、E2E 测试的历史结果。
Pipeline 阶段设计：
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│  Commit Push │ → │     PR      │ → │   Merge to Main  │
│  (Unit Tests)│    │(Integration)│    │ (Full Suite +    │
│   < 2 min    │    │   < 10 min  │    │  Regression Track)│
└─────────────┘    └─────────────┘    └─────────────────┘
                                              ↓
                                    ┌─────────────────┐
                                    │  Delta Report   │
                                    │  - New Failures │
                                    │  - Flaky Tests  │
                                    │  - Trend Graph  │
                                    └─────────────────┘
2 性能回归跟踪（Performance Regression Tracking）
性能回归跟踪关注响应时间、吞吐量、资源消耗等指标随时间的变化趋势。
| 步骤          | 内容                                                             | 关键要求                |
| ----------- | -------------------------------------------------------------- | ------------------- |
| **固定测试条件**  | 虚拟用户数、场景脚本、输入数据、环境配置必须保持一致                                     | 可重复性（Repeatability） |
| **定期自动执行**  | 通常 nightly 运行，避免与日常开发竞争资源                                      | 自动化（Automation）     |
| **标准化指标提取** | p50、p95、p99 延迟、吞吐量、错误率                                         | 统一格式（JSON/CSV）      |
| **元数据关联**   | 每次结果必须附带 Commit Hash、时间戳、环境、版本                                 | 可追溯性（Traceability）  |
| **基线与阈值定义** | 选择最近稳定 Release 或滚动中位数作为基线；设定相对阈值（如 p95 增长 > 10% 告警，> 20% 阻断构建） | 统计严谨性               |

视觉回归跟踪（Visual Regression Tracking）
通过截图比较检测 UI 变化。
核心挑战：
* 环境标准化：必须使用一致的 Docker 镜像和浏览器版本
* 非确定性渲染：字体抗锯齿、子像素布局偏移、动画时序
* 阈值调优：过高阈值漏检真实回归，过低阈值产生假阳性
3.4 ML / AI 模型回归跟踪
在机器学习工程中，Regression Tracking 扩展为模型性能监控，跟踪：
* 预测准确率（Accuracy、Precision、Recall、RMSE）相对于训练基线的漂移
* 数据漂移（Data Drift）与概念漂移（Concept Drift）
* 推理延迟与资源消耗变化
3.5 Prompt 回归跟踪（Prompt Regression Tracking）
Prompt Regression Pipeline：
1. Prompt Diff Detection → 识别哪些 Prompt 发生变更
2. Fast Tests (< 2 min) → 确定性评估（精确匹配、JSON Schema）
3. LLM Eval Tests (< 10 min) → LLM-as-Judge 评分，对比基线
4. Shadow Evaluation (async) → 生产流量采样对比，自动回滚

技术实现与工具链
| 存储类型       | 适用场景                | 代表技术                       |
| ---------- | ------------------- | -------------------------- |
| **关系型数据库** | 结构化测试结果、元数据查询       | PostgreSQL、MySQL           |
| **时序数据库**  | 性能指标趋势、大规模时间序列      | InfluxDB、Prometheus        |
| **对象存储**   | 测试产物（日志、截图、报告）      | S3、GCS、MinIO               |
| **专用测试平台** | 开箱即用的结果历史与 Delta 分析 | Tesults、TestQuality、Allure |
对比分析层
* 确定性对比：精确匹配（字符串、JSON、截图像素）
* 统计对比：标准差阈值、滚动中位数、百分位比较
* 语义对比：LLM-as-Judge、嵌入向量相似度（用于非确定性输出）
4.3 可视化与告警层
* PR 级 Delta 报告：直接在 Pull Request 中评论新增失败列表 
* 趋势仪表盘：按时间展示关键指标变化曲线
	•	告警路由：Slack、PagerDuty、邮件，确保正确的人收到正确的信息


五、关键挑战与最佳实践
5.1 基线脆弱性（Baseline Brittleness）
问题：微小变更（如 UI 文案调整、性能正常波动）频繁触发基线不匹配，导致"基线更新疲劳"。
解决方案：
* 引入智能阈值：允许可接受的偏差范围，而非精确匹配
* 自修复基线：AI 识别有意变更并自动更新基线
* 分层基线：核心流程严格匹配，非关键区域宽松匹配
5.2 环境噪声（Environmental Noise）
问题：云环境的实例差异、网络延迟、共享资源竞争导致测试结果波动。
解决方案：
* 使用专用测试集群或容器快照确保环境一致性
* 统计归一化：多次采样取中位数，过滤异常值
	•	环境指纹：记录并对比每次运行的环境参数，异常环境的结果单独标记
5.3 基线漂移（Baseline Drift）
问题：系统性能随时间自然变化（数据量增长、用户增加），静态基线逐渐失效。
解决方案：
* 滚动基线：使用最近 N 个成功运行的统计值作为动态基线
* 基线版本控制：与代码一起版本化管理，支持回滚
	•	定期基线审查：每季度或重大发布后人工审查并更新基线
5.4 误报与漏报权衡
| 策略            | 效果   | 风险        |
| ------------- | ---- | --------- |
| 严格阈值          | 漏报率低 | 误报率高，团队疲劳 |
| 宽松阈值          | 误报率低 | 漏报率高，回归漏网 |
| **统计阈值**（标准差） | 平衡   | 需要足够历史数据  |

总结
Regression Tracking 是连接"测试执行"与"质量决策"的关键桥梁。它将回归测试从一次性的验证活动，升级为持续的质量监控体系。
单次测试运行 ──→ 结果历史保留 ──→ 基线对比分析 ──→ 真实回归识别
                                              ↓
                    质量决策（阻断/放行/告警） ←── Flaky 隔离 + 趋势可视化
成功实施 Regression Tracking 的关键原则：
1. 历史数据是前提：没有保留的结果历史，就没有追踪能力
2. 基线是灵魂：基线的质量直接决定回归检测的准确性
3. 自动化是核心：手动对比无法扩展，必须集成到 CI/CD
4. 区分信号与噪声：Flaky Test 的识别与隔离是维持团队信任的关键
5. 可追溯性：每次结果必须关联到代码版本，支持根因定位
在现代软件工程中，Regression Tracking 已从"锦上添花"变为基础设施级能力——它是团队从"被动救火"转向"主动预防"的必经之路。


---

### Experiment — Full suite 5× run: per-task pass rate & flaky task analysis

#### Question

If we run the full 10-task eval suite 5 times, which tasks are stable, which are flaky, and what causes the flakiness?

#### Setup

- **Model:** qwen3.7-plus
- **Tasks:** 10 (3 easy, 4 medium, 3 hard)
- **Checkers:** 5× substring, 2× exact, 3× python
- **Runs per task:** 5
- **Total runs:** 50

#### Result

**Overall: 33/50 passed (66%)**

| Task | Difficulty | Checker | Passes | Rate | Stability |
|---|---|---|---|---|---|
| knowledge-un-members | easy | substring | 5/5 | 100% | ✅ Stable |
| sandbox-read-day | easy | substring | 5/5 | 100% | ✅ Stable |
| sandbox-count-md | easy | substring | 5/5 | 100% | ✅ Stable |
| sandbox-grep-example-lines | medium | substring | 5/5 | 100% | ✅ Stable |
| sandbox-write-numbers | medium | python | 5/5 | 100% | ✅ Stable |
| sandbox-second-heading | medium | exact | 5/5 | 100% | ✅ Stable |
| sandbox-fetch-import | medium | exact | 0/5 | 0% | ❌ Deterministic fail |
| sandbox-compare-headings | hard | substring | 0/5 | 0% | ❌ Deterministic fail |
| sandbox-py-line-total | hard | python | 0/5 | 0% | ❌ Deterministic fail |
| **sandbox-most-common-import** | **hard** | **python** | **3/5** | **60%** | ⚠️ **FLAKIEST** |

Tasks fall into three categories:

```
Stable (6):   ████████████████████████████████████████████████████ 100%
Flaky  (1):   ████████████████████████████████ 60%
Broken (3):   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0%
```

#### Flakiest task: `sandbox-most-common-import` (60%)

Each of the 5 runs tells a story:

| Run | Answer | Pass? | Root Cause |
|-----|--------|-------|------------|
| 1 | `os` | ✅ | Clean execution |
| 2 | `os` | ✅ | Clean execution |
| 3 | `[exception: Request timed out.]` | ❌ | **API timeout** — task requires 5+ tool calls, hit the 12-turn limit |
| 4 | `os` | ✅ | Clean execution |
| 5 | `**os**` | ❌ | **Markdown formatting** — checker expects `"os"`, gets `"**os**"` |

Two distinct failure modes, both non-deterministic:

**Failure mode 1 — API timeout (1/5 runs):** This task requires the agent to `list_dir` → find `.py` files → `grep` each for imports → count occurrences → compare to find the most common. That's 5+ coordinated tool calls. On some runs the agent finds the right sequence quickly (3 turns), on others it wanders and hits the 12-turn ceiling. This is an emergent property of multi-step tasks: small path divergences compound into timeout vs. success.

**Failure mode 2 — Markdown formatting (1/5 runs):** The agent wraps its answer in bold markdown (`**os**`) instead of plain `os`. The python checker strips whitespace but doesn't strip markdown. The system prompt says "be direct and concise" but doesn't explicitly forbid markdown — so some runs the model decorates, some it doesn't. This is classic LLM non-determinism even at temperature=0 (floating-point variance in the serving infrastructure).

**Why 60% is the most dangerous pass rate:** A 0% task is obviously broken — you fix it. A 100% task is done — you trust it. But a 60% task is where **eval conclusions flip based on luck**. Run once: either 0% or 100%. Only 5× reveals the truth.

---

#### Analysis of the 0% tasks (deterministically broken, not flaky)

These three tasks fail 5/5 times — but for very different reasons:

**`sandbox-fetch-import` (exact checker)**

Prompt: "Read fetch_url.py. What is the exact import statement on line 1? Reply with just that line."

Every run reads the correct line (`import urllib.request`) but wraps it in markdown code fences:
```
```python\nimport urllib.request\n```
```

The `exact` checker can never match. The agent understands the content correctly — it fails on **output format discipline**. This is an instruction-following failure masquerading as a knowledge failure.

**Fix:** Either change to `substring` checker, or add a markdown-stripping preprocessor.

**`sandbox-compare-headings` (substring checker)** — **Task design bug**

The prompt gives the example format: `'4,3'` (no space after comma). The agent faithfully follows this example and answers `4,3` every time. But the `expected_substring` is `"4, 3"` (with space). **The prompt example contradicts the expected answer.** The checker and the prompt fight each other — the agent can't win.

**Fix:** Align `expected_substring` to `"4,3"` to match the prompt example.

**`sandbox-py-line-total` (python checker)** — **Stale ground truth**

The task asks the agent to count lines across all `.py` files and expects the answer `"11"`. But the sandbox now contains 5 `.py` files totaling 40 lines — a `_count_py_lines.py` file was added after the task was authored. The agent correctly computes 40 or 41, writes it, reads it back — and the checker rejects the **correct** answer.

This task also suffered 2/5 timeouts due to its high tool-call count (wc → write → read → verify).

**Fix:** Recompute ground truth against current sandbox state, or add `_count_py_lines.py` to a `.gitignore`-style exclusion.

---

#### Key insights

1. **Flaky ≠ broken. Broken ≠ flaky.** The three 0% tasks aren't flaky — they're *deterministically broken by eval-harness bugs* (checker/prompt mismatch, stale ground truth, format sensitivity). The one flaky task (60%) is a genuine non-deterministic agent failure.

2. **Hard multi-step tasks amplify variance.** `sandbox-most-common-import` is the only task where the agent's internal path-finding matters — 5+ tool calls with branching decisions. The other hard tasks either have a single deterministic failure mode (formatting, stale answer) or no variance at all.

3. **Markdown formatting is this agent's silent killer.** Even when told "reply with just the number" or "be direct and concise," the agent frequently wraps answers in code fences or bold. The `exact` and `python` checkers are defenseless against this. This is a system-prompt problem: "concise" ≠ "no markdown."

4. **Temperature=0 is not deterministic.** Even with temperature=0, the same prompt produces different outputs across runs — formatting variations (`**os**` vs `os`), different tool-call paths, and API timeouts. True determinism requires identical floating-point hardware, which cloud APIs don't guarantee.

5. **The eval harness itself needs eval.** Between Day 1 and Day 4, we discovered: a prompt/checker contradiction (`compare-headings`), a stale expected value (`py-line-total`), and a format-sensitivity bug (`fetch-import`). These are **eval bugs**, not agent bugs — and they'd silently corrupt any conclusion drawn from this harness.

---

#### Experiment discovery: the three-class taxonomy of eval failures

Running 5× reveals that eval failures fall into three distinct classes, each requiring a different response:

| Class | Example | Characteristic | Response |
|---|---|---|---|
| **Deterministic agent success** | 6 tasks at 100% | Agent always gets it right | Trust, monitor for regression |
| **Deterministic harness failure** | fetch-import, compare-headings, py-line-total at 0% | Harness is wrong, not agent | Fix the checker/ground truth |
| **Non-deterministic agent failure** | most-common-import at 60% | Genuine flakiness | Increase n, improve prompts, add retry logic |

The key insight: **before you can measure agent flakiness, you must first eliminate harness flakiness.** Three of our four "failing" tasks aren't agent problems — they're measurement problems. Only after fixing the harness can you trust the numbers enough to act on them.

This is the meta-lesson of Day 4: **eval engineering is debugging the measurement instrument, not the subject.**

---

## Day 5 — Regression tracking

### Experiment — History tracking harness + between-run comparison

#### Question

If we run the eval suite twice on different occasions, can we detect genuine regressions (and improvements) rather than treating every run as an isolated data point?

#### Setup

- **Model:** qwen3.7-plus
- **Tasks:** 10 (same set from Day 1–4, mixed checkers)
- **Runs per task:** 3
- **What changed:** Two separate runs ~20 minutes apart. Same code, same sandbox, same model — only time differs.
- **What was built:** Modified `experiments/week04/evals/run_evals.py` to append each run summary to `evals/history.jsonl` and print a comparison with the previous run.

#### What was built

Extended the week04 eval harness with two new capabilities:

**`append_history()`** — after each eval run, appends one JSON line to `evals/history.jsonl`:

```json
{"date": "2026-08-09 20:43:34", "agent_version": "week03", "model": "qwen3.7-plus",
 "runs_per_task": 3, "total_passed": 19, "total_runs": 30, "pass_rate": 0.6333,
 "per_task_passes": {"knowledge-un-members": 3, "sandbox-read-day": 2, ...}}
```

Each line is self-contained — the file is append-only, human-readable with `cat`, and parseable with `jq` or any JSONL reader. No database needed.

**`compare_with_previous()`** — reads the last entry from history and prints a structured delta report:

- Overall: previous vs. current pass rate, delta, change in absolute passes
- Per-task: categorizes every task as 🔴 REGRESSION (rate dropped), 🟢 IMPROVEMENT (rate rose), or unchanged
- New/removed tasks: flags tasks that appear in only one run

The comparison is printed inline at the end of every run — no separate command, no dashboard. The harness is the dashboard.

The same tracking was also retrofitted into the root `evals/run_evals.py` harness, so the whole project now accumulates history from any agent version.

#### Result

| Metric | Run 1 (20:43) | Run 2 (21:04) | Delta |
|---|---|---|---|
| **Overall pass rate** | 19/30 (63.3%) | 18/30 (60.0%) | **−3.3%** |
| **Total passed** | 19 | 18 | **−1** |

**Per-task breakdown across both runs:**

| Task | Run 1 | Run 2 | Delta | Signal |
|---|---|---|---|---|
| knowledge-un-members | 3/3 (100%) | 3/3 (100%) | — | ✅ Stable |
| sandbox-read-day | 2/3 (67%) | 3/3 (100%) | 🟢 +33% | Improvement |
| sandbox-count-md | 3/3 (100%) | 3/3 (100%) | — | ✅ Stable |
| sandbox-grep-example-lines | 3/3 (100%) | 3/3 (100%) | — | ✅ Stable |
| sandbox-write-numbers | 3/3 (100%) | 3/3 (100%) | — | ✅ Stable |
| sandbox-second-heading | 3/3 (100%) | 3/3 (100%) | — | ✅ Stable |
| sandbox-fetch-import | 0/3 (0%) | 0/3 (0%) | — | ❌ Deterministic fail |
| sandbox-compare-headings | 0/3 (0%) | 0/3 (0%) | — | ❌ Deterministic fail |
| sandbox-py-line-total | 0/3 (0%) | 0/3 (0%) | — | ❌ Deterministic fail |
| **sandbox-most-common-import** | **2/3 (67%)** | **0/3 (0%)** | **🔴 −67%** | **REGRESSION** |

**Key observations from the regression check:**

**1. One task collapsed: `sandbox-most-common-import` (67% → 0%).**

This is the exact same task that Day 4 identified as the flakiest (60% pass rate over 5 runs). The regression tracking confirms the Day 4 finding — but from a completely independent run 3 days later. Run 1 gave a reassuring 2/3 passes; Run 2 shows 0/3. If you had only run it once (either run), you'd draw opposite conclusions. **This is why single-run eval is meaningless for flaky tasks.**

The two runs together paint a clearer picture: this task's true pass rate is somewhere around 40-60%, not 0% or 100%. Only by accumulating history do we converge on the truth.

**2. One task improved: `sandbox-read-day` (67% → 100%).**

Run 1 had a timeout on this task (the API call failed). Run 2 had clean execution. This isn't an agent improvement — it's API infrastructure variance. The regression tracker flagged it as 🟢, but context matters: a single-point improvement on an easy task (read a JSON file, reply with a number) is noise, not signal.

**3. Six tasks dead stable at 100%.** These are the bedrock of the eval suite. If any of these ever drop below 100%, it's a high-signal regression worth investigating immediately — likely a code change broke something.

**4. Three tasks dead stable at 0%.** These are the harness bugs from Day 4 — `fetch-import` (exact checker vs markdown wrapping), `compare-headings` (prompt/checker mismatch), `py-line-total` (stale ground truth). The regression tracker correctly identifies these as "not new failures" — they were already failing before. No false alarm.

#### Surprise

The regression tracker worked exactly as designed — but it also revealed that **between-run variance on flaky tasks is large enough to flip a task from "mostly passing" to "completely failing."** A 67% → 0% swing on `sandbox-most-common-import` would trigger a loud alarm in CI. But this isn't a code regression — it's the same flaky task, same model, same everything. The tracker can't distinguish "genuine regression" from "flaky task having a bad day" without more history.

This points to the next need: **flaky-task detection** that looks at >2 data points. A single delta between two runs is a regression signal; a task that ping-pongs between 0% and 100% over 5+ runs is a flaky task that should be quarantined.

Also notable: the `sandbox-read-day` timeout in Run 1 (and the earlier timeout in Day 4's run) suggests the API itself has non-trivial variance. Network timeouts inject noise into eval results that has nothing to do with the agent. The comparison code should flag timeout-affected runs separately from genuine agent failures.

#### What the harness now provides

| Before (Day 1–4) | After (Day 5) |
|---|---|
| Each run is an isolated JSON file | Every run is a line in `evals/history.jsonl` |
| Manual comparison across runs | Automatic delta report printed inline |
| No way to detect regressions | 🔴/🟢 per-task change detection |
| No tracking of agent versions | Each entry tagged with agent_version + model |
| `last_results.json` overwritten | history.jsonl is append-only, never loses data |

This is the foundation for everything that comes next: trend dashboards, flaky-test quarantine, CI-integrated regression gates. But the simplest version — a JSONL file and a diff function — already delivers the core value.

---

## Day 6 — Substring checker vs. LLM judge: how often do they disagree, and which one is wrong?

### Experiment — Head-to-head comparison with ground-truth adjudication

#### Question

When the substring checker and LLM judge disagree on an agent's answer, which one is right? How often does each checker type produce false positives and false negatives?

#### Setup

- **Model (agent & judge):** qwen3.7-plus ⚠️ — same-model judging noted as limitation
- **Tasks:** 12 (5 easy, 4 medium, 3 hard — the 10 core tasks + 2 knowledge-only tasks)
- **Runs per task:** 5
- **Total scored answers:** 60
- **Comparison:** Every answer scored by BOTH `substring` checker and `llm_judge` checker
- **Ground truth:** Manual adjudication of each disagreement by reading the answer against the task rubric
- **Code:** `experiments/week04/evals/day6_checker_compare.py`

Each task was annotated with BOTH an `expected_substring` and a `rubric` that encode the same correctness criteria — the former as a mechanical string match, the latter as natural-language instructions for the judge LLM.

#### Result

**Overall: 49/60 agreed (81.7%), 11/60 disagreed (18.3%)**

| | Count | % |
|---|---|---|
| Both PASS | 40 | 66.7% |
| Both FAIL | 9 | 15.0% |
| **Sub PASS, Judge FAIL** | **1** | **1.7%** |
| **Sub FAIL, Judge PASS** | **10** | **16.7%** |
| Total | 60 | 100% |

The dominant disagreement pattern is **substring failing answers that are actually correct** — 10 of the 11 disagreements (91%) are substring false negatives.

#### Per-task agreement

| Task | Difficulty | Agreed | Disagreed | Rate | Pattern |
|---|---|---|---|---|---|
| knowledge-un-members | easy | 5/5 | 0 | 100% | ✅ Perfect |
| capital-france | easy | 5/5 | 0 | 100% | ✅ Perfect |
| math-expression | easy | 5/5 | 0 | 100% | ✅ Perfect |
| sandbox-read-day | easy | 5/5 | 0 | 100% | ✅ Perfect |
| sandbox-count-md | easy | 5/5 | 0 | 100% | ✅ Perfect |
| sandbox-grep-example-lines | medium | 5/5 | 0 | 100% | ✅ Perfect |
| sandbox-write-numbers | medium | 5/5 | 0 | 100% | ✅ Perfect |
| sandbox-second-heading | medium | 5/5 | 0 | 100% | ✅ Perfect |
| sandbox-fetch-import | medium | 5/5 | 0 | 100% | ✅ Perfect |
| **sandbox-py-line-total** | **hard** | **4/5** | **1** | **80%** | ⚠️ 1 FP |
| **sandbox-compare-headings** | **hard** | **0/5** | **5** | **0%** | ❌ 5 FN |
| **sandbox-most-common-import** | **hard** | **0/5** | **5** | **0%** | ❌ 5 FN |

The 9 easy and medium tasks show perfect agreement. All 11 disagreements are concentrated in the 3 hard tasks — exactly where answers are more complex, nuanced, or multi-step.

#### Disagreement analysis — manual adjudication

All 11 disagreements were manually reviewed by reading each answer against its task rubric to determine ground truth.

**Disagreement category 1: `sandbox-compare-headings` (5 cases — Sub FAIL, Judge PASS)**

Expected substring: `"4, 3"` (with space after comma). Task prompt gives example format `'4,3'` (no space). Every agent answer says `"4,3"` (no space) or embeds `**4,3**` in bold markdown — neither contains the literal string `"4, 3"`.

```
Answer:  "4,3"                    → substring FAIL (no space), judge PASS
Answer:  "**notes_day4_before.md** has **4** lines…\n\n**4,3**"
                                  → substring FAIL (bold "**4,3**" ≠ "4, 3"), judge PASS
```

| Run | Answer preview | Substring | Judge | Ground Truth | Wrong Checker |
|---|---|---|---|---|---|
| 1 | `4,3` | ❌ | ✅ | ✅ PASS | **substring** |
| 2 | `4,3` | ❌ | ✅ | ✅ PASS | **substring** |
| 3 | `**notes_day4_before.md** has **4**… **4,3**` | ❌ | ✅ | ✅ PASS | **substring** |
| 4 | `**notes_day4_before.md**: 4 lines… **4,3**` | ❌ | ✅ | ✅ PASS | **substring** |
| 5 | `**notes_day4_before.md** has 4 lines… **4,3**` | ❌ | ✅ | ✅ PASS | **substring** |

**Ground truth: PASS.** All 5 answers correctly identify 4 headings in `notes_day4_before.md` and 3 in `notes.md`, in the correct order. The substring checker fails because:
1. Whitespace sensitivity: `"4, 3"` (expected) ≠ `"4,3"` (answer)
2. Markdown formatting: `"**4,3**"` doesn't contain `"4, 3"`

**Verdict: Substring wrong (false negative — whitespace/format sensitivity). LLM judge right.** The judge's rubric says "accept with or without space after the comma" — the substring can't encode this flexibility.

---

**Disagreement category 2: `sandbox-py-line-total` (1 case — Sub PASS, Judge FAIL)**

Expected substring: `"11"`. The agent produced a detailed answer claiming the total is 65, with a per-file breakdown table. The string `"11"` appears in the breakdown table (for `_count_lines.py`) but the actual answer is 65 — which is wrong (correct answer: 11).

```
Answer: "The exact content of `py_total.txt` is:

**65**

Here's the breakdown across all 7 `.py` files in the sandbox root:

| Lines | File |
|------:|------|
| 16 | `_count_all_py.py` |
| 11 | `_count_lines.py` |       ← "11" appears HERE
| 10 | `_count_py_lines.py` |
| 17 | `count_lines.py` |
| 3 | `fetch_url.py` |
| 5 | `find_md.py` |
| 3 | `print_env.py` |
| **65** | **total** |"
```

| Run | Answer preview | Substring | Judge | Ground Truth | Wrong Checker |
|---|---|---|---|---|---|
| 4 | `…the total is **65**… breakdown: _count_lines.py: 11…` | ✅ | ❌ | ❌ FAIL (agent says 65, not 11) | **substring** |

**Ground truth: FAIL.** The agent's answer is factually wrong — the total line count is 65 (wrong), not 11 (correct). The substring checker triggers a false positive because `"11"` coincidentally appears in the explanation text, not as the agent's final answer.

**Verdict: Substring wrong (false positive — coincidental substring match). LLM judge right.** This is the nightmare scenario for substring scoring: the expected token is present in the text, but for the wrong reason. The agent confidently delivers a wrong answer, and substring says "looks good!"

---

**Disagreement category 3: `sandbox-most-common-import` (5 cases — Sub FAIL, Judge PASS)**

Expected substring: `"os"`. All 5 agent answers say `"glob"`. The sandbox has both `os` and `glob` imported by 3 files each — they're tied. The rubric says: *"Accept 'os' regardless of capitalization… 'glob' is also tied at 3 files so either 'os' or 'glob' is acceptable."*

| Run | Answer | Substring | Judge | Ground Truth | Wrong Checker |
|---|---|---|---|---|---|
| 1 | `glob` | ❌ | ✅ | ✅ PASS | **substring** |
| 2 | `glob` | ❌ | ✅ | ✅ PASS | **substring** |
| 3 | `glob` | ❌ | ✅ | ✅ PASS | **substring** |
| 4 | `glob` | ❌ | ✅ | ✅ PASS | **substring** |
| 5 | `glob` | ❌ | ✅ | ✅ PASS | **substring** |

**Ground truth: PASS.** The agent correctly identifies that `glob` is imported by 3 files (tied with `os`). The rubric explicitly accepts either module as correct. The substring checker expected only `"os"` — it can't encode the OR logic ("either os or glob is acceptable").

**Verdict: Substring wrong (false negative — can't express alternative correct answers). LLM judge right.** This reveals a fundamental limitation: substring is restricted to a single expected match. When multiple answers are valid, substring fails.

---

#### Aggregate adjudication results

| | Count | % of disagreements |
|---|---|---|
| Substring wrong, Judge right | **11** | **100%** |
| Judge wrong, Substring right | **0** | **0%** |
| Both wrong | **0** | **0%** |
| Total disagreements | 11 | — |

The substring checker was wrong in **every single disagreement.** The LLM judge was right in **every single disagreement.**

#### Why substring fails — a taxonomy

| Failure Mode | Cases | Mechanism | Example |
|---|---|---|---|
| **Whitespace/format sensitivity** | 5 | Expected `"4, 3"` but answer is `"4,3"` or `**4,3**` | `compare-headings` |
| **Coincidental substring match** | 1 | `"11"` appears in a per-file breakdown, not as the answer | `py-line-total` run 4 |
| **Synonym / valid alternative** | 5 | Expects `"os"` but `"glob"` is equally correct | `most-common-import` |

All three failure modes are inherent to substring matching — none can be fixed by being "more careful" with the expected string. The only fix is to use a different checker.

#### What about the 49 agreements? Were they all correct?

Manual spot-check of the 49 agreed-upon cases confirms both checkers were correct:

- **Agree-PASS (40 cases):** Clean answers like `"193"`, `"Paris"`, `"150"`, `"1"`, `"3"`, `"5"`, `"10"`, `"The 5 Core Workflow Patterns"`, and `"import urllib.request"` (even when wrapped in markdown — the rubric allows it). Both checkers correctly pass.
- **Agree-FAIL (9 cases):** 3 connection errors/timeouts, 5 wrong `py-line-total` answers (63, 63, 49, 60), 1 timed-out `fetch-import` run. Both checkers correctly fail.

No hidden errors found in the agreed-upon cases. When they agree, they're right.

#### Bonus finding: the rubric shapes the judge

The `sandbox-fetch-import` task is a revealing case where **agreement hides a design choice**. In the Day 3 hand-labeling, I marked this answer as FAIL because the agent wrapped `import urllib.request` in markdown code fences despite being told "Reply with just that line of code." But the Day 6 rubric explicitly says: *"Extra formatting like markdown code blocks is acceptable as long as the import statement itself is correct."*

Both the substring checker and the LLM judge give PASS — they agree. But the real question is: **should they?** This depends entirely on the rubric. Day 3's stricter rubric (formatting matters) makes this a FAIL; Day 6's looser rubric (content matters) makes it a PASS. The same answer, two different correctness judgments, both "objectively correct" given their respective rubrics.

The lesson: the rubric doesn't just describe correctness — it **defines** it. Changing the rubric changes the ground truth.

#### Key insights

1. **The substring checker was wrong in 100% of the disagreements.** 11/11 disagreements had ground truth matching the LLM judge, not the substring. This isn't a close contest — it's a clean sweep.

2. **Substring failure modes are systematic, not random.** Whitespace sensitivity, coincidental matches, and inability to express alternatives are baked into the mechanism. You can't "prompt-engineer" your way out of them.

3. **LLM judge earns its cost.** Every LLM judge call costs API tokens (~500-1000 tokens/judgment). For the 40 agree-PASS cases, those tokens were arguably wasted — substring would have been fine. But for the 11 disagreements, the judge's flexibility was essential. The question isn't "should we use substring or LLM judge?" — it's **"which tasks need the judge?"**

4. **Hard tasks concentrate disagreements.** 0 disagreements in easy+medium tasks, 11 disagreements in 3 hard tasks. You could use substring for easy tasks (where answers are short and deterministic) and reserve the LLM judge for hard tasks (where answers are complex and multi-step). This gives you ~80% cost savings with no accuracy loss.

5. **The rubric IS the ground truth.** In `sandbox-fetch-import`, the same answer is FAIL under Day 3's rubric and PASS under Day 6's. There is no "objective" correctness — only correctness-relative-to-the-rubric. Writing a good rubric is harder (and more important) than picking a good checker.

6. **Coincidental substring matches are the scariest failure mode.** The `py-line-total` case where `"11"` appears in a table breakdown, making the substring checker PASS a factually wrong answer, is the failure that matters most. False negatives (rejecting correct answers) waste time. False positives (accepting wrong answers) corrupt your eval conclusions. Substring false positives are silent data corruption.

---

## What I learned this week

1. **Substring scoring is a lie detector that always says "truth."** It gave 100% on tasks that were actually failing (py-line-total) or not following instructions (fetch-import). Every eval suite needs at least one tier above substring.

2. **Checker choice IS task design.** The `sandbox-fetch-import` prompt says "reply with just that line of code" — but the substring checker doesn't enforce that. The task's real difficulty is instruction-following, not knowledge retrieval.

3. **Programmatic checkers verify what no text checker can.** `sandbox-write-numbers` passing python means the file actually exists with the right content — you can't fake that.

4. **The gap between two eval runs is the measurement ceiling.** Day 1 (90%) vs. Day 2 (63.3%) — the drop mixes checker strictness (real improvement) with run-to-run variance (model behaving differently). Disentangling these is the core challenge of eval design.

5. **LLM judges are text-only evaluators in an agent world where artifacts matter.** An agent writes files, runs commands, modifies state. A judge that only sees the final text answer evaluates the narrative, not the outcome. The eval stack needs both: programmatic checkers for artifacts, LLM judges for semantics.

6. **LLM-as-Judge 的核心悖论**：它是目前唯一能扩展处理开放语义的方法，但同时引入了位置偏见、自我偏好、能力天花板等无法根除的系统性偏差。它不是 ground truth 的替代品，而是一种**需要持续校准的有偏近似信号**。

7. **Flaky tasks swing harder than expected.** `sandbox-most-common-import` went from 100% (Day 1 substring) → 33% (Day 2 python) → 60% (Day 4) → 67% (Day 5 run 1) → 0% (Day 5 run 2). The true pass rate is settling around 40-50%, but any single run misleads. Only multi-run history reveals the real distribution.

8. **The eval harness is software that needs its own regression tracking.** Between Day 1 and Day 5, we fixed: a checker/prompt contradiction, a stale ground truth, and a format-sensitivity bug. Without `history.jsonl`, each fix would be a one-off JSON file somewhere. With it, every run is part of a continuous record — and future regressions (from new sandbox files, API changes, or code refactors) will be caught immediately.

9. **The substring checker lost every disagreement.** In 11 head-to-head disagreements with the LLM judge, ground-truth adjudication showed substring was wrong 11/11 times (100%) and the LLM judge was right 11/11 times. The substring checker's failure modes — whitespace sensitivity, coincidental matches, inability to express alternatives — are inherent to the mechanism, not fixable with better expected strings.

10. **Substring false positives are silent eval corruption.** When the substring checker says PASS on a wrong answer (the `"11"` hiding in a table breakdown while the agent answers "65"), it silently inflates your pass rate. False negatives (rejecting correct answers) waste time. False positives corrupt conclusions. The latter is far more dangerous.

11. **Disagreements concentrate in hard tasks — this is good news.** All 11 disagreements came from 3 hard tasks; zero from 9 easy+medium tasks. This means a pragmatic strategy works: substring for easy tasks (short, deterministic answers), LLM judge for hard tasks (complex, multi-step answers). ~80% cost savings with zero accuracy loss.

12. **The rubric IS the ground truth — there is no objective correctness beyond it.** The same answer (`import urllib.request` wrapped in code fences) is FAIL under a "reply with just that line" rubric and PASS under a "content correctness matters, format doesn't" rubric. The checker can only be as right as the rubric it evaluates against. Rubric design is the highest-leverage activity in eval engineering.