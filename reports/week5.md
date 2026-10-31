# Week 5 Report — Tool Design Deep Dive

**Dates:** 2026-08-10 to 2026-08-16

---

## Day 1 — Rewriting tool descriptions with explicit when-to-call conditions

### Learning

来源：[Anthropic Tool Use Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)

#### 工具调用的基本原理

1. 用户设定好可能需要的工具的输入输出形式，模型会决定何时如何调用它们，模型生成结构化的请求，代码执行操作后，将结果返回给对话。
   用户需要定义：schema、处理反馈、返回结果。

2. **客户端 loop**：输入用户问题和工具，模型回答是否需要停下来执行工具调用，执行工具，继续直到结束输出。

3. **服务端 loop**：调用工具指导结束 `end_turn` 或者达到最大迭代 `pause_turn`，`pause_turn` 的结果没有完成，需要再次给输入给对话来继续。

#### 何时使用工具 vs. 不使用工具

**应该使用工具的场景：**
- 动作产生影响：如写文件、发邮件
- 刷新或外部数据：如随时间而变的价格查询等
- 结构化保证形状的输出
- 调用已有的系统

> 一个明确的迹象表明你应该使用工具：如果你正在编写正则表达式来从模型输出中提取某个决策，那么该决策本应是工具调用。解析自由文本以恢复结构化意图，说明该结构应属于模式中的组成部分。

**不应使用工具的场景：**
- 模型仅凭训练数据即可回答问题（摘要、翻译、常识类问题）
- 交互为单次问答且无副作用
- 调用工具的延迟会显著影响简单的响应时间（每次工具调用至少增加一次往返通信）

#### 三种工具架构

| Approach | When to use it | What to expect |
|---|---|---|
| User-defined client tools | Custom business logic, internal APIs, proprietary data | You handle execution and the agentic loop |
| Anthropic-schema client tools | Standard dev operations (bash, file editing, browser control) | You handle execution; Claude calls the tool reliably because the schema is trained-in |
| Server-executed tools | Web search, code sandbox, web fetch | Anthropic handles execution; you read the results |

---

### Experiment — Measuring should-call rate before/after rewriting tool descriptions

#### Question

Do explicit "Call when" / "Do NOT call when" clauses in tool descriptions improve the model's tool-selection accuracy compared to the original one-sentence descriptions?

**Hypothesis:** The original descriptions tell the model *what* each tool does but not *when to choose it over alternatives*. Adding "Do NOT call when — use X instead" clauses should reduce wrong-tool calls, especially for the common error patterns: using `read_file` to search across files (should be `grep`), calling `list_dir` as a first step when the file path is already known, and calling `web_fetch` when a URL hasn't been discovered yet.

#### Setup

- **Model:** qwen3.7-plus
- **Tasks:** 10 prompts designed to test specific tool-selection skills
- **Varied:** Tool descriptions only — identical tool implementations, identical system prompt
- **Held constant:** Model, sandbox files, max_turns (12), temperature (default), all tool implementations
- **Runs per condition:** 5
- **Total runs:** 100 (2 conditions × 10 prompts × 5 runs)
- **Code:** `experiments/week05/day1_tool_descriptions.py`

**Condition A (before):** Original tool descriptions from week03 agent — one-sentence "Call when the user asks to…" style:

| Tool | Original description (snippet) |
|---|---|
| read_file | "Read a text file from the sandbox workspace. Call when the user asks to see or summarize a file." |
| list_dir | "List files and subdirectories in the sandbox. Call when the user wants to browse or discover what files exist." |
| run_command | "Run a read-only shell command… Use for listing files, grepping content, or running short python one-liners." |
| write_file | "Write or overwrite a text file… Call when the user asks to save, create, or update a file." |
| web_fetch | "Fetch a web page by URL… Call this when the user asks to read, fetch, visit, or summarize a URL." |
| web_search | "Search the web via DuckDuckGo… Call this when the user asks to search for something online." |

**Condition B (after):** Rewritten with structured "Call when" / "Do NOT call when" sections. Example — `read_file`:

```
Read the full contents of a specific text file from the sandbox workspace.

**Call when:**
- The user asks to see, read, view the contents of a NAMED file
- You need to verify a file was written correctly (read it back after write_file)

**Do NOT call when:**
- You need to discover what files exist → use list_dir first
- You need to search for text ACROSS multiple files → use run_command with grep
- The user asks about a URL → use web_fetch
- You want to count lines/words → use run_command with wc
```

Every tool description follows this pattern: concrete triggers with examples, and explicit wrong-path alternatives.

#### Metrics

Three complementary metrics, from most lenient to strictest:

| Metric | Definition | What it catches |
|---|---|---|
| **Should-call rate** | Fraction of expected tools that were actually called | Missing tools (under-calling) |
| **Wrong-tool rate** | Fraction of total calls to unexpected tools | Extra/wrong tools (over-calling) |
| **Exact-match rate** | called_tools == expected_tools (strict equality) | Both — gold standard for tool selection |

#### The 10 test prompts

| ID | Prompt essence | Expected tools | Category |
|---|---|---|---|
| P1 | "Read subdir/data.json and list its fields" | `read_file` | single-tool |
| P2 | "What files are available in the sandbox?" | `list_dir` | single-tool |
| P3 | "Search all files for the word 'sandbox'" | `run_command` | **grep-vs-read** |
| P4 | "Create a file week5_summary.md with this content" | `write_file` | single-tool |
| P5 | "Open https://example.com and summarize" | `web_fetch` | single-tool |
| P6 | "Search online for popular AI agent frameworks" | `web_search` | single-tool |
| P7 | "Search for 'Anthropic building effective agents', then visit top result" | `web_search`, `web_fetch` | multi-tool-chain |
| P8 | "Count lines in notes.md efficiently — do NOT read the file" | `run_command` | **grep-vs-read** |
| P9 | "What year did WWII end?" | (none) | no-tool |
| P10 | "Search for sandbox escape prevention techniques" | `web_search` | **web-vs-local-trap** |

The three bolded categories are where the rewritten descriptions should make the biggest difference: P3/P8 test `grep` vs. `read_file` discrimination, and P10 tests whether the model confuses local sandbox files with web research.

---

### Result

#### Overall metrics

| Metric | Before (original) | After (rewritten) | Delta | Direction |
|---|---|---|---|---|
| **Should-call rate** | 1.000 | 1.000 | 0.000 | → unchanged |
| **Wrong-tool rate** | 0.163 | 0.127 | **−0.037** | ✅ **22% reduction** |
| **Exact-match rate** | 70.0% (35/50) | 76.0% (38/50) | **+6.0pp** | ✅ **improved** |
| **Avg. tokens/run** | 6,441 | 8,605 | +2,164 | ⚠️ **+33.6% more tokens** |

**Key findings:**

1. **Should-call rate stayed at 100% in both conditions.** The agent never missed calling a tool it was supposed to call. The problem was never *under-calling* — it was *over-calling* (calling extra tools beyond what was needed).

2. **Wrong-tool rate dropped 22% (0.163 → 0.127).** The rewritten descriptions reduced unnecessary tool calls. The "Do NOT call when" clauses are doing their job — preempting common wrong-tool patterns before the model makes them.

3. **Exact-match improved +6pp (70% → 76%).** Three more out of 50 runs achieved perfect tool selection.

4. **Token cost increased 33.6%.** The rewritten descriptions are ~3× longer (each contains ~200-400 words of when-to-call guidance vs. the original ~20-40 words). This adds ~2,000 tokens to the system prompt per run. Whether this tradeoff is worth it depends on the task: for simple single-tool tasks it's overhead; for complex multi-step tasks the reduction in wrong-tool calls may save more tokens than the descriptions cost.

#### Per-prompt breakdown

| Prompt | Before | After | Change | Analysis |
|---|---|---|---|---|
| P1-read-known-file | 5/5 | 5/5 | — | ✅ Always correct — named file path → `read_file` |
| P2-browse-sandbox | 5/5 | 5/5 | — | ✅ Always correct — browsing → `list_dir` |
| **P3-grep-across-files** | **0/5** | **3/5** | **🟢 +60pp** | **Biggest win — "Do NOT call read_file for multi-file search" worked** |
| P4-save-content | 5/5 | 5/5 | — | ✅ Always correct — explicit save → `write_file` |
| P5-fetch-url | 5/5 | 5/5 | — | ✅ Always correct — URL given → `web_fetch` |
| P6-search-online | 0/5 | 0/5 | — | ❌ Systematic: always calls web_search + web_fetch (see below) |
| P7-search-then-fetch | 5/5 | 5/5 | — | ✅ Always correct — explicit two-step chain |
| P8-count-lines-efficient | 5/5 | 5/5 | — | ✅ Always correct — "do NOT read" hint in prompt works |
| P9-no-tool-trivia | 5/5 | 5/5 | — | ✅ Always correct — knows when no tool is needed |
| **P10-web-search-vs-local** | **0/5** | **0/5** | — | ❌ Systematic: always calls web_search + web_fetch (see below) |

#### Deep dive: P3 — the biggest improvement (0/5 → 3/5)

P3 asks the model to search all text files for a word. The expected tool is `run_command` with `grep`.

**Before (original descriptions):** Every single run called `list_dir` (discover files) + `run_command` (grep) + `write_file` (save results). The model treated a search task as a three-step workflow: discover → search → save. While this *works* (the answer is correct), it wastes tool calls — the prompt only asks to "tell me," not to "save."

```
Before — all 5 runs: [list_dir, run_command, write_file]
  extra tools: list_dir (unnecessary discovery — grep can search *.txt *.md *.py directly)
               write_file (unnecessary persistence — prompt asks to "tell me", not "save")
```

**After (rewritten descriptions):** 3 of 5 runs called ONLY `run_command` — exactly right. The `read_file` "Do NOT call when searching across multiple files → use run_command with grep" clause prevented the model from even considering `read_file`. The `run_command` "searching for text across one or more files → use grep" clause directly matched the task.

```
After — 3/5 runs: [run_command] ✅ exact match
After — 2/5 runs: [list_dir, run_command, write_file] (still the old pattern)
```

The 2 remaining "over-calling" runs suggest that the model has a learned workflow bias (discover → search → persist) that descriptions alone can't fully override. This is a **prompt+description interaction**: the model reads "search all files" and its training says "first understand the file layout, then search, then save findings." The descriptions can guide but not eliminate this prior.

#### Deep dive: P6 and P10 — systematic over-calling (0/5 in both conditions)

Both P6 and P10 expect ONLY `web_search`. But every single run across both conditions calls `web_search` + `web_fetch`.

```
P6 — 10/10 runs: [web_search, web_fetch]  (expected: [web_search])
P10 — 10/10 runs: [web_search, web_fetch] (expected: [web_search])
```

This is **not a tool-selection error** — it's **intelligent chaining**. The model does `web_search`, reads the results, selects the most promising URL from those results, and calls `web_fetch` to drill deeper. This is exactly the workflow that the system prompt and the tool descriptions *encourage* ("Combine web_search + web_fetch: search first, then open the most promising result").

The problem is with the **test design**, not the model. Expecting only `web_search` when the model can improve its answer by following up with `web_fetch` is an artificially strict constraint. A better expected-tools set for these prompts would be `[web_search, web_fetch]` — or a scoring rubric that distinguishes "over-calling" from "appropriate chaining."

**Lesson:** Exact-match is a useful metric, but it must be applied to prompts where the expected tool set is genuinely canonical. For research tasks where chaining is reasonable, exact-match will report false negatives.

#### Wrong-tool call analysis

| Wrong tool | Before (times) | After (times) | Change |
|---|---|---|---|
| `web_fetch` | 10 | 10 | — (all from P6/P10 chaining — see above) |
| `list_dir` | 5 | 2 | ⬇ −60% (P3 runs where it wasn't needed) |
| `write_file` | 4 | 2 | ⬇ −50% (P3 runs where it persisted unnecessarily) |

The reduction in `list_dir` and `write_file` unnecessary calls is directly attributable to the rewritten descriptions. The `web_fetch` count is unchanged because it's the same chaining behavior in both conditions — the descriptions can't stop the model from doing something that is actually *good practice*.

---

### Surprise

1. **Should-call rate was 100% in BOTH conditions — the problem was never missing tools.** Going into the experiment, I expected the original descriptions might cause the model to miss calling tools it needed (e.g., not realizing `web_search` applies to a research question). But the model never missed a call. The real problem was the opposite: **over-calling** — adding extra tools to a workflow that didn't need them. The rewritten descriptions helped by telling the model what NOT to do, not what to do.

2. **The model has deeply ingrained workflow biases that descriptions only partially override.** In P3 (grep-across-files), the model consistently follows a "list → search → save" pattern. The rewritten descriptions broke this pattern in 3/5 runs — a real improvement — but 2/5 runs still followed it. Tool descriptions are prompt engineering, not fine-tuning; they shift probabilities but don't eliminate the model's prior training on common workflow patterns.

3. **Web tool chaining is the model being smart, not wrong.** P6 and P10 exposed a measurement problem: the model chains `web_search` → `web_fetch` because that's *good agent behavior*. Flagging this as "wrong tool calls" is a test design error. The lesson: exact-match metrics need careful expected-tool annotations that account for reasonable tool chaining.

4. **Longer descriptions cost real tokens (+33.6%).** The rewritten descriptions are comprehensive but verbose. For a single-turn agent, this overhead is significant. For a multi-turn agent that benefits from fewer wrong-tool calls, the tradeoff may net out positive — fewer wasted tool-call turns could offset the description cost. This is directly measurable and should be tested (→ Day 3's token-cost experiment).

5. **The simplest prompts don't benefit from better descriptions — but they don't hurt either.** P1, P2, P4, P5, P7, P8, P9 were 100% exact-match in both conditions. For straightforward single-tool tasks, the original descriptions were already sufficient. The rewritten descriptions are insurance, not a necessity, for these cases.

6. **"Do NOT call when — use X instead" is the highest-leverage phrase in a tool description.** Every reduction in wrong-tool calls came from a "Do NOT call" clause that named the correct alternative. The "Call when" clauses were table stakes — the model already knew when to call tools. The edge came from preempting the *wrong* call.

---

### Key insights

1. **Tool description design is prompt engineering at the schema level.** Every word in a tool description competes for the model's attention with every other word in the prompt. The "Do NOT call" pattern works because it uses negative space — it explicitly rules out the most common wrong paths.

2. **Over-calling is a bigger problem than under-calling.** The model is eager to use tools — it will call tools when they're vaguely relevant, not only when they're necessary. Good descriptions curb this eagerness without suppressing it entirely.

3. **Exact-match is the right metric, but only with the right expected-tool annotations.** Measuring "did the model call exactly the right set of tools?" catches both under-calling and over-calling in one number. But the expected set must account for reasonable chaining behavior, or the metric becomes a ceiling that the best possible agent can't reach.

4. **The token cost of long descriptions is the hidden variable.** Every experiment from here on should track both tool-selection accuracy AND token consumption. A 22% reduction in wrong-tool rate at the cost of 33% more tokens per run is a real tradeoff that needs to be evaluated per use case.

---

### Next question (→ Day 2)

The rewritten descriptions improved wrong-tool rate by 22%, but at a 33.6% token cost. Day 2 asks: does tool granularity (one `file_op(action, ...)` mega-tool vs. three narrow tools) matter more or less than description quality for tool-selection accuracy? If a mega-tool eliminates wrong-tool calls by reducing the choice space, does it also reduce task success by being harder to parameterize correctly?

---

## Day 2 — Mega-tool vs. narrow tools: tool granularity comparison

### Learning

来源：PLAN.md Week 5 Day 2 + Anthropic Tool Use Documentation

#### One Powerful Tool vs. Many Narrow Tools

这是 AI Agent 工具设计中一个基础性决策：是将所有相关功能聚合为一个"瑞士军刀"式大工具，还是拆分为一组职责单一的小工具。两种范式在软件工程中有悠久的历史对照。

**集成范式 (One Powerful Tool)：** 将所有相关操作聚合在一个接口中，通过 `action` 参数分发。

优势：
- 一致性体验：统一的数据模型、错误处理机制。例如 K8s 通过统一 API 管理所有资源。
- 减少选择空间：模型只需决定是否调用这一个工具，不会选错工具名。
- 内部优化：工具内部可以共享逻辑（如路径验证），避免重复实现。

劣势：
- **action 参数增加了决策维度**：选对工具之后还要选对 action，两次选择都可能出错。
- 参数矩阵膨胀：不同 action 需要不同参数（`content` 只在 action="write" 时需要），JSON Schema 无法表达条件 required。
- 学习曲线陡峭：模型需要理解一个工具的三种用法。

**组合范式 (Many Narrow Tools)：** 每个工具只做一件事 — `read_file`, `write_file`, `list_dir`。

优势：
- 正交性：每个工具接口稳定、职责单一，描述天然精确。
- 可组合性：通过依次调用来构建复杂流程（如 list → pick → read）。
- 零参数歧义：不需要 "content is required only when action=write" 这样的条件说明 — `write_file` 永远需要 content，`read_file` 永远不需要。

劣势：
- 编排复杂度：模型需要学习工具间的正确顺序（先 list_dir 再 read_file）。
- 工具选择空间更大：3 个工具意味着 3 倍的选择错误机会。

#### 当前业界趋势：分层粒度

- 底层：Narrow tools（容器生态：runc, containerd, CRI-O）
- 中层：Orchestrator（K8s 作为 powerful tool 编排 narrow tools）
- 顶层：Platform Engineering（统一界面）
- Agent 层面：**模型越强，越受益于 Narrow Tools**（能正确编排它们）；**模型越弱，越需要 Powerful Tools**（减少选择空间）。

---

### Experiment — Comparing mega-tool vs. narrow tools on file operations

#### Question

Does collapsing three file-operation tools into one `file_op(action=read|write|list)` mega-tool increase or decrease error rates? The mega-tool adds an extra decision dimension (action selection) but reduces the tool-space from 3 to 1 — which effect dominates?

**Hypothesis:** The narrow tools should have lower error rates because there is no action ambiguity — `write_file` always means write, `list_dir` always means list. The mega-tool's extra `action` parameter creates a second decision point where the model can fail. However, the mega-tool might win on prompts where the model confuses similar tool names (e.g., "show me the files" → `read_file` instead of `list_dir`).

#### Setup

- **Model:** qwen3.7-plus
- **Tasks:** 10 file-operation prompts testing different operation types
- **Varied:** Tool schema only — mega-tool (`file_op`) vs. narrow tools (`read_file` + `write_file` + `list_dir`)
- **Held constant:** Identical tool implementations under the hood, same system prompt, same sandbox, max_turns (12)
- **Runs per condition:** 5
- **Total runs:** 100 (2 conditions × 10 prompts × 5 runs)
- **Code:** `experiments/week05/day2_tool_granularity.py`

**Condition A (mega-tool):** One `file_op` tool with `action ∈ {read, write, list}`:

```json
{
  "name": "file_op",
  "parameters": {
    "action": { "type": "string", "enum": ["read", "write", "list"] },
    "filepath": { "type": "string" },
    "content": { "type": "string" }
  },
  "required": ["action", "filepath"]
}
```

**Condition B (narrow tools):** Three separate tools — `read_file(filepath)`, `write_file(filepath, content)`, `list_dir(dirpath)`.

#### The 10 test prompts

| ID | Prompt essence | Expected actions (mega) | Expected tools (narrow) | Category |
|---|---|---|---|---|
| P1 | "Read notes.md and summarize" | `[read]` | `[read_file]` | single-read |
| P2 | "Create greeting.txt with this content" | `[write]` | `[write_file]` | single-write |
| P3 | "Show me all files in the sandbox root" | `[list]` | `[list_dir]` | single-list |
| P4 | "Write config.json, then read it back" | `[write, read]` | `[write_file, read_file]` | chained-write-read |
| P5 | "List subdir, then read one of the files" | `[list, read]` | `[list_dir, read_file]` | list-then-read |
| P6 | "Overwrite notes.txt with new content" | `[write]` | `[write_file]` | single-overwrite |
| P7 | "Read nonexistent_file.xyz — report error" | `[read]` | `[read_file]` | error-nonexistent |
| P8 | "Create projects/demo/__init__.py and verify" | `[write, read]` | `[write_file, read_file]` | chained-subdir |
| P9 | "List root → write experiment_log.txt → read it back" | `[list, write, read]` | `[list_dir, write_file, read_file]` | multi-op-sequence |
| P10 | "Show directory listing — do NOT read any file" | `[list]` | `[list_dir]` | trap-read-vs-list |

P3 and P10 are the critical prompts: when the user says "show me what's available" or "see what files", will the model interpret this as `read` (mega-tool) / `read_file` (narrow)? P10 explicitly says "do NOT read any individual file" to make the discrimination test harder.

#### Metrics

| Metric | Mega-tool | Narrow tools | What it measures |
|---|---|---|---|
| **Error rate** | wrong-action rate | wrong-tool rate | Core selection accuracy |
| **Exact-sequence-match** | actual actions == expected | actual tools == expected | Strictest gold-standard |
| **Coverage** | fraction of expected actions called | fraction of expected tools called | Did it at least try the right things? |
| **Parameter errors** | missing `content` for write, etc. | missing args | Argument-level mistakes |
| **Extra calls** | calls beyond expected length | same | Over-calling behavior |
| **Tokens/run** | total tokens consumed | same | Efficiency comparison |

---

### Result

#### Overall metrics

| Metric | Mega-tool | Narrow tools | Delta | Direction |
|---|---|---|---|---|
| **Error rate** | 0.000 | 0.000 | 0.000 | → no difference |
| **Exact-sequence-match** | 90.0% (45/50) | 92.0% (46/50) | **+2.0pp** | ✅ narrow better |
| **Coverage** | 100.0% | 100.0% | 0.000 | → no difference |
| **Parameter errors** | 0 | 0 | — | → both perfect |
| **Extra calls (total)** | 7 | 6 | −1 | ≈ tie (0.14 vs 0.12/run) |
| **Avg. tokens/run** | 3,058 ± 1,378 | 2,807 ± 1,289 | **−251** | ✅ narrow more efficient |

**Key findings:**

1. **Both conditions achieved zero selection errors.** The model never picked the wrong action (mega-tool) or the wrong tool (narrow tools). This is the headline result — at this model capability level (qwen3.7-plus), tool granularity doesn't cause selection errors in either direction.

2. **Narrow tools had a small edge in exact-sequence-match (+2pp).** 46/50 vs. 45/50 — the difference is exactly one run. This came from P10 (trap-read-vs-list): the mega-tool had one run where it called `file_op(action="read")` instead of `action="list"`, while narrow tools got 5/5.

3. **Coverage was perfect in both conditions (100%).** Both designs ensure the model calls every operation the task requires — no missed steps in either condition.

4. **Parameter errors were zero across all 100 runs.** Even though the mega-tool's `content` parameter is only meaningful for `action="write"`, the model always supplied it correctly and never omitted `content` on write actions. This was my biggest concern going in — that conditional parameters would cause errors — and it turned out to be a non-issue.

5. **Narrow tools used 8.9% fewer tokens (2,807 vs. 3,058).** The mega-tool's larger description (explaining all three actions, their required parameters, and when to pick each) added ~250 tokens/run. Narrow tools have shorter, more focused descriptions.

#### Per-prompt breakdown

| Prompt | Mega | Narrow | Winner | Analysis |
|---|---|---|---|---|
| P1-read-known-file | 5/5 | 5/5 | tie | ✅ Trivial — name file → read |
| P2-write-new-file | 2/5 | 2/5 | tie | ❌ Systematic fail — see deep dive |
| P3-list-root | 4/5 | 4/5 | tie | ⚠️ Both flaky — see deep dive |
| P4-write-then-verify | 5/5 | 5/5 | tie | ✅ Perfect — explicit two-step |
| P5-list-then-read | 5/5 | 5/5 | tie | ✅ Perfect — list → pick → read |
| P6-overwrite-existing | 5/5 | 5/5 | tie | ✅ Perfect — explicit overwrite |
| P7-read-nonexistent | 5/5 | 5/5 | tie | ✅ Perfect — handles errors well |
| P8-write-subdir | 5/5 | 5/5 | tie | ✅ Perfect — nested path + verify |
| P9-all-three-ops | 5/5 | 5/5 | tie | ✅ Perfect — three-action sequence |
| **P10-trap-read-vs-list** | **4/5** | **5/5** | **🟢 narrow** | **The one prompt where granularity mattered** |

#### Deep dive: P2 and P3 — the systematic failures (both conditions)

P2 ("Create a file called greeting.txt with this content") and P3 ("Show me all the files in the sandbox root") both failed at a 40-80% rate — but identically in both conditions.

**P2 (single-write): 2/5 in both conditions.** The model's actual behavior in the 3 failing runs:
```
Expected: [write_file]
Actual:   [list_dir, read_file]  — lists the sandbox first, then reads an existing file
```
This is a **prompt misinterpretation**: the model sees "create a file" but also sees existing files in context, decides to browse first to "understand the environment" before writing. The tool granularity has nothing to do with this failure — it's a workflow bias (inspect-before-modify) that the prompt doesn't override.

**P3 (single-list): 4/5 in both conditions.** The 1 failing run in each:
```
Expected: [list_dir]
Actual:   [list_dir, read_file]  — lists, then reads one of the found files
```
The model correctly calls list_dir first, but then "helpfully" reads one of the files it found. The prompt says "I want to know what's available before I start working" — the model interprets this as "read the most important-looking file too."

**Lesson from P2/P3:** When both conditions fail identically, the variable under test (tool granularity) is not the cause. The fix is in prompt design, not tool design.

#### Deep dive: P10 — the one prompt where narrow won (4/5 vs. 5/5)

P10 says: "I need to **see** what files are in the sandbox. Do **NOT** read any individual file — just show me the directory listing."

```
Mega — 4/5: [list] ✅  [list] ✅  [list] ✅  [read] ❌  [list] ✅
Narrow — 5/5: [list_dir] ✅  [list_dir] ✅  [list_dir] ✅  [list_dir] ✅  [list_dir] ✅
```

The mega-tool's failing run called `file_op(action="read")` instead of `action="list"`. Why? The word "see" in the prompt ("I need to **see** what files") is semantically closer to "read" than to "list". The narrow tool `list_dir` has an unambiguous name — "list" is right there in the name. But with the mega-tool, the model must map "see what files are available" → "which action?" → if the word "see" activates "read" in the model's embedding space, it picks `action="read"`.

This is the **action-name sensitivity** problem unique to mega-tools: the action enum values become a semantic classification task. `read`/`write`/`list` works well most of the time, but edge cases like "see what files exist" trigger the wrong action because the linguistic cue ("see") is ambiguous.

#### Summary of extra calls

```
Mega:   7 extra calls across 50 runs (mean 0.14/run)
Narrow: 6 extra calls across 50 runs (mean 0.12/run)
```

These were almost all from P2 and P3 (the systematic workflow-bias failures). Neither condition caused gratuitous extra calls beyond the known prompt issues.

---

### Surprise

1. **Tool granularity barely mattered at this model capability level.** Going in, I expected a larger gap — either narrow tools winning decisively (because action selection adds a decision point) or the mega-tool winning (because it eliminates tool-choice ambiguity from a 3-way decision). Neither happened. The model (qwen3.7-plus) is good enough at both action selection AND tool selection that the difference is ~2pp. The biggest takeaway: **at sufficient model capability, tool granularity is a second-order effect.**

2. **The conditional-parameter problem never materialized.** I was worried that `file_op(action="write")` would fail because the model would forget to supply `content`, or that `file_op(action="read")` would supply `content` unnecessarily. Neither happened — 0 parameter errors across 100 runs. The model correctly understood that `content` is only meaningful for write — even though JSON Schema can't express this constraint.

3. **The failures that DID occur were prompt-design failures, not tool-design failures.** P2 (2/5 in both conditions) and P3 (4/5 in both conditions) failed identically regardless of tool granularity. The model has a learned "inspect first, act second" workflow bias that the prompt needs to explicitly counter. When both conditions fail the same way, you're looking at the wrong variable.

4. **Token efficiency favors narrow tools.** Narrow tools used 8.9% fewer tokens despite having more tool schemas (3 vs. 1). The reason: each narrow tool description is short and focused (~50 words), while the mega-tool description needs to explain all three actions, their parameters, and when to pick each (~150 words). Three short descriptions are cheaper than one long one.

5. **P10 is the canary prompt — it alone validates narrow tools.** The one-run difference (4/5 vs. 5/5) on P10 is small but diagnostic. The mega-tool's failure mode — confusing "see" with "read" — is a systematic weakness of action-enum designs. With 100 runs instead of 5, I'd expect this gap to widen. The word "see" is always going to be closer to "read" than "list" in embedding space.

6. **This result is model-dependent.** A weaker model (or one with smaller context) might show a larger gap. The mega-tool's description is a wall of text explaining three modes; a weaker model might get lost in it. The narrow tools' descriptions are individually shorter and each maps 1:1 to a user intent. Testing this on `qwen-turbo` or a smaller model would likely show narrow tools winning by a larger margin.

---

### Key insights

1. **At this model tier, tool granularity is a UX concern, not a reliability concern.** Both designs work. The choice should be driven by developer ergonomics (which is easier to maintain?), not model error rates.

2. **The real danger of mega-tools is action-name ambiguity, not parameter confusion.** The model handles conditional parameters perfectly but stumbles when a prompt word ("see") is semantically closer to the wrong action enum value. Avoid action values whose names are synonyms for the operations they represent — prefer `list_directory` over `list`, `read_file_contents` over `read`.

3. **Narrow tools are more token-efficient despite having more schemas.** Short, focused descriptions beat one long description. This is counterintuitive: you'd expect 3 tool definitions to cost more than 1, but the brevity of each more than compensates.

4. **The experiment confirms Day 1's lesson: tool description quality dominates tool count.** Day 1 showed that well-written "Do NOT call when" clauses reduce wrong-tool calls by 22%. Day 2 shows that going from 3 tools to 1 (or vice versa) changes exact-match by only 2pp. **Description quality > tool granularity** as a lever for improving agent behavior.

5. **Prompt-tool interaction is the hidden variable.** P2 and P3 failed in both conditions because the prompt didn't counter the model's workflow biases. Tool design can't fix a prompt that lets the model follow its default instincts.

---

### Next question (→ Day 3)

Day 2 showed that tool count (1 vs. 3) barely matters for error rates at this model tier, but narrow tools are more token-efficient. Day 3 asks: at what tool count do schema tokens become a meaningful cost? Measure tokens consumed by tool schemas alone at 5/15/30 tools, and measure wrong-tool rate at each count to find the inflection point where more tools hurt more than they help.

---

## What I learned this week (beyond the experiments)

### Day 1 — Tool descriptions as triggering conditions

Rereading Anthropic's tool-use docs with fresh eyes after 4 weeks of building agents. The key reframe: **descriptions are not documentation — they are triggering conditions.** The model reads a tool description and decides "does this user message match this description's trigger pattern?" A vague description ("Call when the user asks to see a file") matches too many things. A precise description with "Do NOT call when — use X instead" narrows the match window.

### Day 2 — Tool granularity is a second-order effect

The mega-tool vs. narrow-tool debate isn't about correctness at this model tier — it's about developer ergonomics and token efficiency. The experiment showed both designs work, but narrow tools won on tokens and had a tiny edge on the hardest prompt (P10). This maps to the Unix philosophy: "do one thing well" scales better than "do everything in one interface."

### Day 3 — Literature is a prior, not a truth

The published degradation curves said 30 tools → ~75% selection accuracy; the measurement said 100% discrimination between real and dummy tools, at every count. Benchmarks describe populations, not your model on your tasks. The only real cost of more tools at this tier was tokens (+221%). Week 4's "measure everything" applied to tool design: verify before you optimize.

### Day 4 — Classify the failure before writing examples

Tool misuse splits into two classes: selection errors (the model doesn't know when/what to call — fixed by examples: list_dir −75%) and deliberate chaining (the model disagrees with your annotation — immune to examples: web_fetch 0%). Before writing examples, ask which class the failure belongs to. And the token surprise: verbosity that removes wrong-tool turns can be net-negative cost — measure run-level tokens, not description length.

### Cross-week connection: the highest-leverage lever

Week 3 built the tools; Week 4 measured them; Week 5 is about *designing* them. The through-line: **tool surface design is the highest-leverage agent decision.**

| Intervention | Effort | Effect on accuracy | Effect on tokens |
|---|---|---|---|
| Rewrite descriptions (Day 1) | Low (no code changes) | +6pp exact-match (22% less wrong-tool) | +33.6% tokens |
| Change tool granularity (Day 2) | Medium (refactor) | +2pp exact-match | −8.9% tokens |
| Add dummy tools (Day 3) | Low (register schemas) | 0 (no accuracy change, 0 dummy calls) | +221% at 30 tools |
| Add worked examples to 2 tools (Day 4) | Low (description-only) | +8pp exact-match (−31% wrong-tool) | −10% tokens/run (net, after turn savings) |
| Forced router + specialists (Day 5) | Medium (new architecture) | +6pp exact-match vs. monolith (P3: 2/5 → 5/5) | **−46.5% tokens/run** (11% routing overhead) |

The cheaper text-only intervention (description rewrite) had the bigger accuracy effect among the text changes. The structural interventions won the tokens: granularity −8.9%, and Day 5's forced router −46.5% while also fixing the week's hardest prompt. The lesson isn't "descriptions beat structure" — it's **"measure everything, because intuition is wrong about which lever matters more"** — and structure beats words when the failure is a habit.

**Status:** Days 1–5 complete (Day 5 router vs. monolith comparison is in — see the Day 5 section below). Day 6's surprise-effect experiment and Day 7's report remain.


## day 3: Tool-count effects and context cost of schemas
一、Tool-count effects（工具数量效应）与 Context cost of schemas（Schema 的上下文成本
Tool-Count Effects：决策空间的熵增
在 Agent 架构中，每增加一个可用工具，并非简单地"扩展了能力边界"，而是指数级地扩大了 LLM 每一步推理的决策空间。
LLM 的 tool calling 是一个条件生成问题,当n （工具数量）增大时，选择精度衰减（Selection Accuracy Degradation）（当多个工具的功能描述存在重叠时，LLM 的 attention 机制难以在它们之间建立清晰的决策边界，实验数据表明（基于 GPT-4/Claude 3.5 级别的模型），当可用工具从 5 个增加到 20 个时，工具选择准确率通常从 >95% 下降至 75-85%；超过 30 个时，可能跌破 70%，且错误呈现系统性偏差（总是选错某一对相似工具）。）推理链复杂度膨胀（Chain-of-Thought Dilution）（在 ReAct、Reflexion 等需要显式推理的 Agent 模式中，模型需要在每一步：回顾当前状态、评估所有可用工具的适用性、选择工具并构造参数、验证参数合法性）模型为了"尽快结束思考"，可能编造不存在的参数或选择不恰当的工具。在复杂任务中，模型可能在多个功能相近的工具之间反复横跳，无法收敛。延迟与吞吐的隐性损耗（即使模型选择了正确的工具，决策延迟也会随工具数量增加而上升，解码过程中，每个 token 的生成都需要对所有工具 schema 进行 attention 计算，工具描述越长，KV-cache 中占用的显存越多，批处理（batching）效率下降）
| 工具数量  | 选择准确率  | 单次推理延迟   | 架构建议             |
| ----- | ------ | -------- | ---------------- |
| 1–5   | >95%   | 基准       | 单体 Agent，直接暴露    |
| 6–15  | 90–95% | +10–20%  | 甜点区，可接受          |
| 16–30 | 75–90% | +30–50%  | **必须引入路由层**      |
| 30+   | <75%   | +50–100% | **必须分层/分 Agent** |

二、Context Cost of Schemas：上下文窗口的结构性消耗
Tool schema（通常以 JSON Schema 或 OpenAPI 子集形式存在）是 Agent 的"静态知识"，必须在每次推理的 system prompt 中完整呈现。其成本由三部分构成：
Token 占用（Token Tax）每个工具的 schema 都是结构化的 JSON，天然具有高 token 密度。一个拥有 20 个中等复杂度工具的 Agent，仅 schema 部分就可能消耗 2,000–4,000 tokens。注意力稀释（Attention Dilution）LLM 的上下文窗口是一个软约束：虽然可以塞入 128K tokens，但模型对远距离信息的召回能力（needle-in-haystack）随距离衰减。当 system prompt 中前 3,000 tokens 被工具 schema 占据时：用户的核心意图（通常位于 prompt 末尾或历史消息中）与 schema 之间的 attention 权重被稀释。模型更容易"遗忘"对话早期的关键约束，或忽略用户最新的澄清指令。缓存失效（Cache Invalidation）现代推理引擎（如 vLLM, TGI）依赖 prefix caching（前缀缓存）来加速重复请求。如果 Agent 的 schema 是动态变化的（例如根据用户权限动态加载不同工具），或者每次请求的可用工具集合不同，缓存命中率会急剧下降，导致：首 token 延迟（TTFT）显著增加、计算成本上升（无法复用 KV cache）
| 优化层级          | 技术                        | 效果             | 代价       |
| ------------- | ------------------------- | -------------- | -------- |
| **压缩**        | 缩短 description，移除冗余字段     | -30~50% tokens | 语义精度微降   |
| **摘要**        | 用自然语言替代完整 JSON Schema     | -60~80% tokens | 参数类型约束弱化 |
| **动态加载**      | 仅加载与当前意图相关的 schema        | -70~90% tokens | 需要意图分类器  |
| **分层 Schema** | 抽象公共参数到基类                 | -20~30% tokens | 实现复杂度增加  |
| **Schema 外置** | 工具名 + 简短描述，详细 schema 按需检索 | -80~95% tokens | 增加一次检索延迟 |

三、两者的耦合效应：架构设计的约束方程
Tool-count 与 Schema-cost 并非独立变量，它们通过一个约束方程共同决定 Agent 的架构上限：
Effective context=context window - (sum_n_schema)-history-user input
增加工具数量 → Schema 总成本上升 → 有效上下文减少 → 注意力稀释加剧 → 选择精度下降。这是一个负反馈循环，无法通过简单地"换更大的模型"或"写更好的 prompt"完全解决。
当工具数量突破甜点区（>15 个）时，必须引入架构层面的重构：
路径一：Router-Specialist 分层（推荐）
User Query
    ↓
[Router Agent] —— 仅携带 3–5 个元工具（search, code, doc, data, general）
    ↓
[Specialist Agent A] —— 携带 10 个代码相关工具
[Specialist Agent B] —— 携带 10 个数据分析工具
[Specialist Agent C] —— 携带 10 个文档处理工具
路径二：动态 Schema 加载（Dynamic Tooling）
在推理前增加一个轻量级意图分类层（可以是 embedding 匹配或小模型）：
1. 将用户 query 编码为向量。
2. 与所有工具的 description 向量进行相似度检索。
3. 仅将 Top-K（如 K=5）最相关的工具 schema 注入 system prompt。
这可以将每次推理的 schema 成本降低 70–90%，但需要承担分类错误的额外风险。
路径三：Schema 语义压缩（Semantic Schema Compression）
将多个相似工具聚合为一个多态工具（Polymorphic Tool）：将 4 个 CRUD 工具合并为 1 个，schema 成本降低约 60%，但增加了参数构造的复杂度（模型需要额外推理 operation 字段）.

四、与 Granularity 问题的统一视角
这两个概念实际上是前文讨论的 "One Powerful Tool vs. Many Narrow Tools" 在 LLM Agent 语境下的具体化与量化：
| 传统软件工程            | LLM Agent 映射                                                    |
| ----------------- | --------------------------------------------------------------- |
| One Powerful Tool | 少量聚合工具（Polymorphic / Meta-tools）→ **低 Tool-count，高 Schema 复杂度** |
| Many Narrow Tools | 大量原子工具（Atomic Tools）→ **高 Tool-count，高 Schema 总量**              |
| Bash 胶水脚本         | 单体 Agent 直接暴露所有工具 → **Schema 爆炸，上下文耗尽**                         |
| Dedicated Tools   | Router + Specialist 分层 → **可控 Schema，精准路由**                     |
在 LLM Agent 中，"Many Narrow Tools" 的 Unix 哲学受到了根本性的挑战：
Unix 哲学成立的前提是：管道中的每个工具都是确定性程序，组合成本仅在于进程间通信开销。
LLM Agent 中的组合成本在于：概率性决策空间的熵增和有限上下文窗口的硬约束。
因此，Agent 架构的最佳实践并非简单地追随 Unix 哲学，而是：
1. 在 Specialist 内部：遵循 Narrow Tools 哲学（每个工具职责单一、接口清晰）。
	2	在系统层面：采用聚合与路由（One Powerful Router + Many Specialist Agents），以控制单次推理的认知负荷。
1. 工具总数是否 > 15？
   ├─ 否 → 直接暴露，关注 Schema 压缩
   └─ 是 → 进入步骤 2

2. 是否存在清晰的领域边界？
   ├─ 是 → 采用 Router-Specialist 分层
   └─ 否 → 进入步骤 3

3. 工具调用模式是否可预测？
   ├─ 是 → 采用动态 Schema 加载
   └─ 否 → 采用 Schema 语义压缩（Polymorphic Tools）

---

### Experiment — Measuring schema token cost and wrong-tool rate at 5/15/30 tools

#### Question

**Part 1:** How many tokens do tool schemas consume at different tool counts (5 → 15 → 30), and does the per-tool cost scale linearly?

**Part 2:** Does adding irrelevant "dummy" tools increase the wrong-tool call rate? At what count does the model start confusing real tools with distractors?

**Hypothesis (Part 1):** Schema token cost should scale roughly linearly with tool count, but with some sub-linearity due to shared JSON structure overhead. The naive chars/4 estimate should be directionally correct but inaccurate.

**Hypothesis (Part 2):** Adding irrelevant tools should increase the wrong-tool rate — the model will occasionally call a plausible-sounding dummy tool instead of the correct real tool. Based on the learning section's literature (GPT-4/Claude-class models: >95% accuracy at 1-5 tools, 75-90% at 16-30 tools), I expected a measurable drop at 15 tools and a clear degradation at 30 tools.

#### Setup

- **Model:** qwen3.7-plus
- **Part 1 — Schema tokens:** Send the neutral prompt "What is 2 + 2? Answer in one word." with 5/15/30 tools. Measure `prompt_tokens` deltas against a no-tool baseline. 5 runs per count (token counts are deterministic at this provider — all 5 runs were identical).
- **Part 2 — Wrong-tool rate:** 8 tool-selection prompts (from Day 1, excluding the 2 that need `web_fetch`). At each tool count, the agent gets 5 core real tools + N dummy tools. 3 conditions × 5 runs × 8 prompts = 120 runs.
- **Varied:** Number of total tools (5 / 15 / 30), controlled by adding dummy tools
- **Held constant:** 5 core real tools (read_file, write_file, list_dir, run_command, web_search) present in all conditions; same 8 eval prompts; same system prompt; max_turns (12)
- **Dummy tools:** 25 realistic-looking tools covering productivity (send_email, create_calendar_event, translate_text), data (query_database, search_documents, get_stock_price, get_weather), text utilities (summarize_text, sentiment_analysis, spell_check, extract_entities, markdown_to_html), generators (generate_qr_code, generate_password, generate_uuid), and utilities (hash_text, base64_encode_decode, json_validator, timezone_converter, unit_converter, calculate_distance, color_converter, ip_geolocation, compress_text, image_ocr). Each has plausible parameters, descriptions, and (where appropriate) enum/boolean/number typed arguments to look real.
- **Runs per condition:** 5
- **Code:** `experiments/week05/day3_tool_count.py`

**Condition summary:**

| Condition | Real tools | Dummy tools | Total | Schema chars |
|-----------|-----------|-------------|-------|-------------|
| 5_tools  | 5 (all core) | 0 | 5 | 2,748 |
| 15_tools | 5 (all core) | 10 | 15 | 9,098 |
| 30_tools | 5 (all core) | 25 | 30 | 17,461 |

**The 8 eval prompts** (from Day 1, excluding web_fetch-dependent P5 and P7):

| ID | Prompt essence | Expected tools | Category |
|---|---|---|---|
| P1 | "Read subdir/data.json and list its fields" | `read_file` | single-tool |
| P2 | "What files are available in the sandbox?" | `list_dir` | single-tool |
| P3 | "Search all files for the word 'sandbox'" | `run_command` | grep-vs-read |
| P4 | "Create week5_summary.md with this content" | `write_file` | single-tool |
| P6 | "Search online for popular AI agent frameworks" | `web_search` | single-tool |
| P8 | "Count lines in notes.md — do NOT read the file" | `run_command` | grep-vs-read |
| P9 | "What year did WWII end?" | (none) | no-tool |
| P10 | "Search for sandbox escape prevention techniques" | `web_search` | web-vs-local-trap |

---

### Result

#### Part 1 — Schema token measurement

| Tool count | Schema chars | Schema tokens (measured) | Tokens per tool | Chars per token |
|-----------|-------------|------------------------|----------------|----------------|
| 5 | 2,748 | **879** | 175.8 | 3.13 |
| 15 | 9,098 | **2,465** | 164.3 | 3.69 |
| 30 | 17,461 | **4,718** | 157.3 | 3.70 |

**Scaling analysis:**

| Transition | Tool multiplier | Token multiplier | Nature |
|-----------|----------------|-----------------|--------|
| 5 → 15 | 3.0× | 2.80× | **Sub-linear** |
| 15 → 30 | 2.0× | 1.91× | **Sub-linear** |
| 5 → 30 | 6.0× | 5.37× | **Sub-linear** |

Part 1 findings:

1. **Schema tokens scale sub-linearly with tool count.** Going from 5 → 30 tools (6× more tools) only costs 5.37× more tokens. The per-tool token cost drops from 176 to 157 — an 11% reduction per tool. This is because each additional tool shares the JSON array wrapper overhead, and the tokenizer compresses repeated structural patterns (repeated `"type": "function"`, `"type": "object"`, `"type": "string"` keys).

2. **The naive chars/4 estimate is wrong by 20-28%.** The chars/4 heuristic predicted 687 / 2,274 / 4,365 tokens; actuals were 879 / 2,465 / 4,718. The tokenizer achieves ~3.1-3.7 chars per token on JSON schema text — more efficient than the 4.0 assumed by the heuristic. This matters for cost estimation: if you budget for chars/4, you'll overestimate schema cost by ~20%.

3. **At 30 tools, schemas alone consume ~4,700 tokens per API call.** With a ~23 token baseline for the prompt itself, 99.5% of prompt tokens in a one-shot call are schema overhead. For a multi-turn agent (average 2-3 turns in our experiments), this compounds: at 30 tools, the agent burns ~9,000-14,000 prompt tokens on schemas alone across turns.

4. **Schema cost is provider/tokenizer dependent.** The qwen3.7-plus tokenizer (based on Qwen's vocabulary) achieves different compression ratios than OpenAI's cl100k_base or Anthropic's tokenizer. The chars/token ratio of ~3.5 means the actual token cost of schemas is lower than the literature's common estimates (which often use 4.0). But the absolute numbers are still substantial.

#### Part 2 — Wrong-tool rate by tool count

**Overall metrics:**

| Metric | 5 tools | 15 tools | 30 tools | Trend |
|---|---|---|---|---|
| **Dummy-tool call rate** | 0.000 | 0.000 | 0.000 | → **zero in all conditions** |
| **Wrong-tool rate** | 0.054 | 0.036 | 0.040 | → ~4-5%, flat |
| **Exact-match rate** | 85.0% (34/40) | 87.5% (35/40) | 87.5% (35/40) | → ~86-88%, flat |
| **Should-call rate** | 1.000 | 1.000 | 1.000 | → perfect, all conditions |
| **Avg. tokens/run** | 5,467 | 10,754 | 17,533 | ↑ scales with schema size |

**Per-prompt exact-match breakdown:**

| Prompt | 5 tools | 15 tools | 30 tools | Analysis |
|---|---|---|---|---|
| P1-read-known-file | 5/5 | 5/5 | 5/5 | ✅ Always correct |
| P2-browse-sandbox | 5/5 | 5/5 | 5/5 | ✅ Always correct |
| **P3-grep-across-files** | **0/5** | **0/5** | **0/5** | ❌ **Systematic failure — all conditions** |
| P4-save-content | 4/5 | 5/5 | 5/5 | ⚠️ One flaky run at 5 tools |
| P6-search-online | 5/5 | 5/5 | 5/5 | ✅ Always correct |
| P8-count-lines-efficient | 5/5 | 5/5 | 5/5 | ✅ Always correct |
| P9-no-tool-trivia | 5/5 | 5/5 | 5/5 | ✅ Always correct |
| P10-trap-web-search-vs-local | 5/5 | 5/5 | 5/5 | ✅ Always correct |

**Dummy tools called by mistake:**

| Condition | Dummy calls |
|---|---|
| 5 tools | (none — no dummy tools present) |
| 15 tools | (none — 0 dummy calls across 40 runs) |
| 30 tools | (none — 0 dummy calls across 40 runs) |

Part 2 findings:

1. **The headline: zero dummy-tool calls across all 120 runs.** This is the most important result and directly contradicts the initial hypothesis. Even with 25 irrelevant tools — including near-miss distractors like `search_documents` (which sounds like `web_search`) and `summarize_text` (which could plausibly be confused with `read_file`) — the model never once called a dummy tool. At qwen3.7-plus capability, tool discrimination is essentially perfect regardless of tool count.

2. **Wrong-tool rate is low (~4-5%) and independent of tool count.** The wrong-tool calls that DID occur were NOT to dummy tools — they were to other real tools (e.g., calling `list_dir` when only `run_command` was expected). This is the same "intelligent chaining" / "workflow bias" pattern from Days 1-2, not a tool-count effect. Adding more tools doesn't make the model confuse real tools with each other either.

3. **Exact-match rate is flat across conditions (85-88%).** The small differences (85% vs. 88%) are within sampling noise — at 5 runs per prompt, one run's difference changes exact-match by 20pp for that prompt. The systematic failures (P3 at 0%, P4's one flaky run) are the same prompt-design issues identified in Day 1.

4. **Token usage is the real and only cost of more tools.** Going from 5 → 30 tools triples token consumption (5,467 → 17,533) with zero accuracy benefit or penalty. The tokens are pure overhead — they don't help the model (it already picks the right tools) and they don't hurt it (it still picks the right tools). Every token above the minimum schema cost is wasted.

5. **P3 (grep-across-files) remains 0% in all conditions — a persistent prompt-design problem.** Across all three Day 1-3 experiments, P3 has never achieved a single exact-match run. The model consistently calls extra tools (list_dir, write_file) beyond the expected `run_command`. This is NOT a tool-count or tool-description issue — it's a fundamental workflow bias (inspect → search → save) that the prompt doesn't counter. If each day's experiment re-tests P3, the expected-tools annotation should be expanded to `[list_dir, run_command]` or `[run_command, write_file]` to reflect what the model actually does.

---

### Surprise

1. **Zero dummy-tool confusion — the model is far better at tool discrimination than expected.** The literature (and my own learning notes above) predicted 75-90% accuracy at 16-30 tools. The actual result: **100% discrimination** between real and dummy tools across all counts. This is either: (a) qwen3.7-plus is stronger than the models used in the cited studies, (b) the dummy tools were too obviously irrelevant (despite deliberate design effort to make them realistic), or (c) the 5 real tools have such distinct domains (files, shell, web) that there's minimal semantic overlap with productivity/data/utility tools. The truth is likely a combination of all three.

2. **More tools neither helped nor hurt — they were exact dead weight.** Going from 5 → 15 → 30 tools: exact-match flat (85-88%), wrong-tool flat (~4%), dummy-call flat (0%). The only thing that changed was token cost (+221%). This is the cleanest possible null result: tool count, in isolation, does not affect selection accuracy at this model tier. The model's tool selection is driven by prompt-tool fit and description clarity — not by the size of the tool set.

3. **Schema token scaling is sub-linear — but the absolute numbers are still alarming.** The per-tool cost drops from 176 → 157 tokens as count increases, which is nice. But at 30 tools with a 3-turn agent, you're spending ~14,000 prompt tokens on schemas alone. For a model with 128K context, that's ~11% of the window. For a model with 32K context (still common in production), it's 44%. Schema cost is the silent killer of context budget.

4. **The conflict between theory and experiment is itself a finding.** The learning section documents a widely-cited degradation curve (95% at 5 tools → 75% at 30 tools). The experiment found 0% degradation. This gap between published benchmarks and real measurement is why Week 4 emphasized "measure everything" — the literature describes average behavior across models and tasks, not your model on your tasks. If I had accepted the literature numbers and built a Router-Specialist architecture at 15 tools, I would have been optimizing for a problem that doesn't exist at this model tier.

5. **The real bottleneck is not tool selection — it's prompt interpretation.** Across Days 1-3 of experiments, the single biggest source of "errors" is the model applying its learned workflow biases (inspect-before-act, search-then-fetch, save-results) when the prompt expects a minimal tool sequence. This is not a tool design problem. It's a prompt-task alignment problem. Better tool descriptions help at the margins (Day 1's 22% wrong-tool reduction), but they can't override the model's fundamental approach to task decomposition.

---

### Key insights

1. **At qwen3.7-plus capability, tool-count-induced confusion is not a real problem.** The model discriminates between tools with near-perfect accuracy even at 30 tools. If you're using a model of this tier or higher, tool count should not be your primary concern.

2. **The only measurable cost of more tools is tokens — and it's linear-ish.** Each additional tool costs ~160 tokens per turn. At 3 turns average, that's ~480 tokens per tool per run. Scale that to your daily API volume: 1000 runs/day × 3 turns × 30 tools × 157 tokens = ~14M schema tokens/day, or about $5.60/day at qwen3.7-plus pricing. Not free, but not prohibitive either.

3. **The literature's tool-count degradation curves are model-dependent and should be verified, not assumed.** The gap between "published 75% accuracy at 30 tools" and "measured 100% discrimination at 30 tools" is a reminder that benchmarks describe populations, not instances. Your model + your tools + your prompts may be far from the mean.

4. **Day 1's lesson holds across all three experiments: description quality >>> tool count.** The 22% wrong-tool reduction from better descriptions (Day 1) dwarfs the 0% effect of tool count (Day 3). If you have one hour to improve your agent's tool selection, spend it on descriptions, not on reducing the tool set.

5. **Schema token cost is the variable to monitor, not tool count.** The right question is not "how many tools?" but "how many tokens do my schemas consume, and is that worth the capability they provide?" A tool that costs 160 tokens/turn but is called once per 100 runs is a bad trade. A tool that costs 160 tokens/turn and is called every run is table stakes.

---

### Next question (→ Day 4)

Day 3 showed that adding tools doesn't confuse the model — the only cost is tokens. Day 4 asks: can we reduce the token cost without reducing accuracy by adding few-shot examples directly into tool descriptions? If a 50-word example in the description can replace a 200-word "Call when / Do NOT call" section, we get the best of both: high accuracy (from Day 1's rewritten descriptions) and low token cost (from Day 3's finding that schema tokens are the real bottleneck).

---

## day4: Tool-use examples & few-shot in descriptions.
在 LLM-based Agent 的 tool-use 系统中，Tool-use examples（工具使用示例）与 few-shot in descriptions（描述中的少样本提示）是将程序化的调用模式编码进自然语言描述中，以引导 LLM 在概率空间中做出更精准的决策。
1、Tool-Use Examples
在标准的 tool schema（如 OpenAI Function Calling 格式）中，每个工具通常包含：
* name：工具标识符
* description：功能描述（自然语言）
* parameters：参数结构（JSON Schema）
Tool-use examples 是在 description 字段中（或作为独立的 few-shot 消息）嵌入的输入-输出对，展示"给定某种用户意图，应该如何调用此工具"。
{
  "name": "calculate_mortgage",
  "description": "Calculate monthly mortgage payment. \n\nExample:\nUser: 'I want to buy a $500k house with 20% down, 30-year fixed at 6.5%'\n→ Call: calculate_mortgage(principal=400000, years=30, rate=0.065)\n\nUser: 'Monthly payment for 15-year loan on $300k at 5%'\n→ Call: calculate_mortgage(principal=300000, years=15, rate=0.05)",
  "parameters": { ... }
}
JSON Schema 是语法约束（syntactic contract），但它无法表达语义模式（semantic patterns）
| 问题类型      | JSON Schema 的局限                                | 示例的弥补           |
| --------- | ---------------------------------------------- | --------------- |
| **参数推断**  | 无法说明"用户说'首付 20%'意味着 `down_payment_percent=20`" | 示例直接展示映射关系      |
| **歧义消解**  | 无法区分"搜索文档"vs"搜索网络"的边界                          | 示例展示每个工具的典型触发场景 |
| **格式规范化** | 无法规定日期是 `YYYY-MM-DD` 还是 Unix timestamp         | 示例展示期望格式        |
| **多步推理**  | 无法表达"先 A 后 B"的调用链                              | 示例展示复合任务的分解模式   |
2、示例如何影响 LLM 的 Tool-Use 决策
LLM 的 tool selection 是一个条件概率分布：
p(tool, args|query, system, {schema_i})
示例中的query如果在模型遇到类似的query时，会激活这个局部模式，提高该工具的选择概率并正确填充参数。LLM 的 attention 机制对局部共现模式高度敏感。"Example: User asks about weather → Call: get_weather(location=..., date=...)"
否定示例可以暗示"何时不使用”，能有效降低工具间的语义碰撞。"Example: For 'current stock price of AAPL', use get_stock_quote. Do NOT use search_news for real-time price queries."
JSON Schema 定义了参数可以是什么，而示例定义了参数应该是什么。{
  "description": "Query database. Example: 'Show me sales last month' → query='SELECT * FROM sales WHERE date >= date_trunc(\"month\", current_date - interval \"1 month\")'"
}

3、Few-Shot 的两种注入位置
| 注入位置                            | 机制                                    | 适用场景             |
| ------------------------------- | ------------------------------------- | ---------------- |
| **In-description**（描述内联）        | 直接嵌入 `description` 字段                 | 工具数量少（<10），示例简短  |
| **Message-level few-shot**（消息级） | 在对话历史中加入 `<user, assistant>` 对，展示工具调用 | 复杂多步任务，需要展示工具链组合 |
| **Hybrid**（混合）                  | 描述内放简单示例，消息级放复杂链式示例                   | 通用架构，平衡成本与效果     |

4、示例的设计策略
模式一：参数映射示例（Parameter Mapping Examples
解决"用户自然语言 → 结构化参数"的转换问题。
{
  "name": "schedule_meeting",
  "description": "Schedule a meeting. \n\nExamples of parameter extraction:\n- 'Meeting with John tomorrow at 3pm' → attendees=['John'], start_time='2026-08-14T15:00:00', duration_minutes=60\n- '30-min sync every Monday at 9am starting next week' → recurrence='weekly', start_time='2026-08-18T09:00:00', duration_minutes=30\n- 'All hands on the first Friday of each month' → recurrence='monthly', day_of_week='Friday', week_of_month='first'"
}
关键设计点：覆盖参数提取的边缘情况（recurring vs one-time, implicit vs explicit）。
模式二：工具边界示例（Boundary Examples）
解决"该用 A 还是 B"的选择歧义。
{
  "name": "search_code",
  "description": "Search codebase for implementation details. \n\nUse this when the user asks about: function signatures, class definitions, algorithm implementations.\n\nDO NOT use this when the user asks about: API usage docs, deployment guides, architecture decisions (use `search_docs` instead).\n\nExamples:\n- 'How is the auth middleware implemented?' → search_code\n- 'What's the retry policy in the HTTP client?' → search_code\n- 'How do I deploy to staging?' → search_docs (NOT search_code)"
}
关键设计点：明确正例和负例，划定决策边界。
模式三：链式调用示例（Chain-of-Tool Examples）
展示多工具协作的时序逻辑。
{
  "name": "analyze_customer_churn",
  "description": "Analyze customer churn. This is a composite operation that typically requires:\n1. First: `get_customer_data(segment='churned', months=6)`\n2. Then: `run_sql_query(query='SELECT ... FROM customer_events WHERE ...')`\n3. Finally: `generate_report(data=sql_result, type='churn_analysis')`\n\nExample session:\nUser: 'Why did enterprise customers churn in Q2?'\n→ Step 1: get_customer_data(segment='enterprise', status='churned', period='2026-Q2')\n→ Step 2: run_sql_query(query='SELECT feature_usage, support_tickets, NPS FROM ...')\n→ Step 3: generate_report(type='root_cause_analysis')"
}
关键设计点：展示数据依赖关系（上一步的输出如何成为下一步的输入）。
| 维度        | 模式 A：自主链式（示例引导）         | 模式 B：聚合工具（封装）       |
| --------- | ----------------------- | ------------------- |
| **决策主体**  | LLM（每步推理）               | 人类工程师（预编排）          |
| **工具注册表** | 3 个原子工具                 | 1 个聚合工具（+ 3 个内部子工具） |
| **上下文成本** | 高（每步都需重新加载所有工具 schema）  | 低（单次决策）             |
| **灵活性**   | 高（模型可根据上下文跳过/调整步骤）      | 低（执行路径固定）           |
| **可靠性**   | 中（LLM 可能选错步骤或循环）        | 高（确定性执行）            |
| **可观测性**  | 高（每步 Thought/Action 可见） | 低（内部为黑盒）            |
| **调试难度**  | 中（可定位到哪一步出错）            | 高（需深入工具内部日志）        |
| **适用频率**  | 偶发/多变的复合任务              | 高频/标准化的复合任务         |
当前 Agent 框架设计中的一个关键原则：必须在"提示工程层面的示例"和"工具层级的封装"之间做出明确区分，不能让两者在语义上重叠。
模式四：错误恢复示例（Error Recovery Examples）
展示工具调用失败时的重试/降级策略。
{
  "name": "fetch_web_page",
  "description": "Fetch a web page. \n\nExamples:\n- Normal: fetch_web_page(url='https://example.com')\n- If 404: Try `search_web(query='site:example.com alternative page')` instead of retrying same URL\n- If timeout (>30s): Report failure to user, do not silently retry more than once"
}

5、Few-Shot 的上下文工程：成本与收益的精确计算
每个示例都是一个输入-输出对，其成本取决于：
| 组件          | 估算           | 备注                 |
| ----------- | ------------ | ------------------ |
| 用户 query 片段 | 10–30 tokens | 需精简，保留核心意图         |
| 工具调用（含参数）   | 15–50 tokens | JSON 格式天然 token 密集 |
| 解释/注释文本     | 10–20 tokens | "→ Call:" 等引导语     |
在 20 个工具的系统中，如果每个工具都带 3 个示例：
	•	额外成本：2,000–6,000 tokens
实验数据表明（基于 GPT-4/Claude 3.5 级别模型）：
| 指标      | 无示例    | 有 2–3 个示例 | 提升      |
| ------- | ------ | --------- | ------- |
| 工具选择准确率 | 75–85% | 90–95%    | +10–15% |
| 参数提取完整率 | 60–70% | 85–92%    | +20–25% |
| 格式合规率   | 80%    | 95%+      | +15%    |
| 边缘情况处理  | 40–50% | 75–85%    | +30–35% |
当工具调用错误导致的下游成本（用户不满、重试延迟、错误恢复逻辑复杂度）高于示例的 token 成本时，嵌入示例就是正收益。
动态 Few-Shot：按需加载策略
为解决"所有工具都带示例导致上下文爆炸"的问题，可采用分层示例策略：
System Prompt 层级：
├── 所有工具：精简 description（无示例）—— 占用 30% 上下文
├── Top-5 相关工具：description + 1 个核心示例 —— 占用 20%
└── 最相关工具：description + 3 个详细示例 + 边界案例 —— 占用 15%
实现方式：
1. Embedding 检索：将用户 query 与工具 description 做相似度匹配，只给 Top-K 工具加载示例。
2. 意图分类器：轻量级模型（如 BERT 级）预判工具类别，动态组装 prompt。
3. 渐进式加载：第一轮不给示例，若模型选择错误，第二轮在 retry 中加入针对性示例。

6、从示例到元学习
在高度结构化的 Agent 系统中，示例可以升级为一种调用约定 DSL领域特定语言：{
  "name": "data_pipeline",
  "description": "Execute data pipeline steps.\n\nSyntax pattern: [source] → [transform] → [sink]\n\nExamples:\n- 'Load users from Postgres, filter active, write to S3' \n  → source={type='postgres', table='users'}, transform={filter='status=active'}, sink={type='s3', bucket='output'}\n- 'Stream events from Kafka, aggregate by hour, alert if >1000'\n  → source={type='kafka', topic='events'}, transform={window='1h', aggregate='count', condition='>1000'}, sink={type='alert', channel='slack'}"
}
反模式：示例的过度工程化
| 反模式               | 问题                                         | 修正                 |
| ----------------- | ------------------------------------------ | ------------------ |
| **示例与 schema 矛盾** | 示例中用了 `date="tomorrow"`，但 schema 要求 ISO 格式 | 示例必须严格符合 schema 约束 |
| **示例过于同质化**       | 3 个示例都是同一模式的微小变体                           | 示例应覆盖不同决策分支        |
| **示例包含幻觉工具**      | 示例中引用了不存在于当前工具集的函数                         | 所有示例中的工具名必须真实存在    |
| **示例过长**          | 单个示例 >200 tokens，淹没核心描述                    | 示例应精炼，聚焦决策关键点      |
| **负向示例缺失**        | 只展示"何时用"，不展示"何时不用"                         | 必须包含边界判定示例         |

7、Tool-use examples 与之前讨论的 Tool-count effects 和 Context cost of schemas 形成三角约束关系：	1	工具数量多 → 每个工具的示例必须更精炼（用 1 个高质量示例替代 3 个平庸示例）。
1. Schema 成本高 → 将示例从 description 迁移到消息级 few-shot，让多个工具共享通用的"调用风格"示例。
	3	示例提升选择准确率 → 可以在一定程度上抵消工具数量增加带来的精度衰减（用示例质量换数量空间）。
开始设计 Tool Description
    │
    ▼
工具是否涉及复杂参数提取？
    ├─ 是 → 加入参数映射示例
    └─ 否 → 继续
    │
    ▼
是否存在功能相近的工具？
    ├─ 是 → 加入边界示例（正例 + 负例）
    └─ 否 → 继续
    │
    ▼
是否常作为多步任务的一部分？
    ├─ 是 → 加入链式调用示例
    └─ 否 → 继续
    │
    ▼
工具总数是否 > 15？
    ├─ 是 → 采用动态加载：仅给 Top-5 工具加载示例
    └─ 否 → 所有工具可内联示例
    │
    ▼
示例总 token 是否 > 30% system prompt？
    ├─ 是 → 压缩示例（精简 query，移除冗余解释）
    └─ 否 → 保持当前设计

#### 知识衔接：示例何时有效、何时失效 —— 本日实验的预测框架

前三天的实验数据积累了一个"最常被误用工具"排行榜：`list_dir` 22 次多余调用（day1 7 次 + day3 15 次）、`web_fetch` 20 次（day1）、`write_file` 19 次。但同样是"多余调用"，失败机理不同，示例的可修复性也不同：

- **选择错误（confusion）**：模型不知道该不该调、何时调 —— 如 P3 中先用 `list_dir` 浏览再 grep。这类错误源于决策边界模糊，❌ 反例可以直接修复。
- **策略性行为（deliberate chaining）**：模型知道该调什么，但主动选择多调 —— 如 `web_search` 之后主动 `web_fetch` 多个结果页深化答案。模型认为自己的策略是对的，示例说"别这么做"未必有效。

由此得到两个可证伪的预测：

1. 给 `list_dir`（混淆型错误）加示例 → 多余调用应显著下降；
2. 给 `web_fetch`（策略型行为）加示例 → 可能完全无效。

另外，第 3 节的"注入位置"权衡将在本日实验中被首次直接检验：in-description 示例埋在 system 的 schema 块里、离决策时刻约 2,000 tokens 远；而 message-level few-shot 直接出现在对话流中。对"推翻既有工作流"这类强干预，schema 内示例可能力不从心。

---

### Experiment — Worked examples on the two most-misused tools: re-measure

#### Question

Do concrete worked examples (input → action pairs, including explicit ❌ counter-examples) appended to the descriptions of the two most-misused tools reduce their wrong-call rate — without suppressing their legitimate use?

**Hypothesis:** The examples target the exact failure modes observed in Days 1 and 3. `list_dir` misuse is a confusion error (unnecessary pre-exploration before a task that one direct call solves) — the counter-examples should fix it. `web_fetch` misuse is deliberate chaining (the model fetches pages because it believes they improve the answer) — Day 1 already concluded this is "the model being smart, not wrong," so examples may fail to override it. **Prediction: list_dir improves, web_fetch resists.**

#### Setup

- **Model:** qwen3.7-plus
- **Tasks:** the same 10 tool-selection prompts from Day 1
- **Varied:** a "**Worked examples:**" block appended to the descriptions of `list_dir` and `web_fetch` ONLY
- **Held constant:** tool implementations, parameter schemas, system prompt, sandbox, max_turns (12), the other four tool descriptions
- **Runs per condition:** 5 → 100 runs total (2 conditions × 10 prompts × 5)
- **Code:** `experiments/week05/day4_worked_examples.py`
- **Results:** `experiments/week05/day4_results_20260813_101404.json`

**Which two tools, and why** — accumulated wrong-call counts from Days 1 and 3:

| Tool | Wrong calls | Failure mode |
|---|---|---|
| `list_dir` | 22 (7 in day1 + 15 in day3) | unnecessary pre-exploration on P3 (one grep would do) |
| `web_fetch` | 20 (all day1) | fetches 2–5 result URLs after `web_search` already answered (P6, P10) |
| `write_file` | 19 | same P3 over-engineering — close third, excluded to keep the variable to exactly two tools |

(day3 couldn't exhibit `web_fetch` misuse — that experiment's tool set didn't include it.)

**Baseline condition** = Day 1's "after" descriptions (the current state of the tools). **Examples condition** = baseline + worked examples on the two target tools. The examples (condensed):

`list_dir` — 4 examples: ✅ browse → `list_dir(".")`; ✅ "Do I have any .md files?" → list and filter; ❌ known path → `read_file` directly (NOT list_dir → read_file); ❌ content search → grep directly, "browsing adds a wasted turn before the real work" (NOT list_dir → grep).

`web_fetch` — 4 examples: ✅ open-and-summarize a given URL; ✅ search then fetch the SINGLE most relevant result; ❌ "top 3 frameworks, one line each" → web_search only, snippets already answer (NOT web_search → web_fetch × 5); ❌ recommendations query → web_search only, fetch at most one page and only if a specific detail is missing.

**Description size and schema cost:**

| | list_dir | web_fetch | Schema tokens/call |
|---|---|---|---|
| baseline | 788 chars | 987 chars | 2,100 |
| examples | 1,345 (+557) | 1,847 (+860) | 2,464 (+364, **+17%**) |

**Metrics** — the same three from Day 1 (should-call rate, wrong-tool rate, exact-match), plus a per-target-tool wrong-call count, a legit-use control (called-when-expected), and token accounting (schema cost per call vs. total tokens per run).

---

### Result

#### Overall metrics

| Metric | Baseline | Examples | Delta | Direction |
|---|---|---|---|---|
| **Should-call rate** | 1.000 | 1.000 | 0.000 | → unchanged (still never misses) |
| **Wrong-tool rate** | 0.163 | 0.113 | **−0.050** | ✅ **31% relative reduction** |
| **Exact-match rate** | 70.0% (35/50) | 78.0% (39/50) | **+8.0pp** | ✅ **improved** |
| **Avg. total tokens/run** | 13,511 | 12,138 | **−1,373** | ✅ **−10.2%** |
| **Schema tokens/call** | 2,100 | 2,464 | +364 | ⚠️ +17% (but see token deep-dive) |

**Key findings:**

1. **list_dir examples hit their target precisely: wrong calls 4 → 1 (−75%).** All four baseline wrong calls and the one remaining examples-condition call were on P3 (unnecessary pre-exploration before grep). P3 exact-match jumped 20% → 80%.
2. **web_fetch examples had zero effect: 10 wrong calls in both conditions** (P6 × 5, P10 × 5), and fetch multiplicity didn't change either (P10: 19 fetch calls in baseline vs. 21 in examples across 5 runs). Same intervention, opposite outcomes — the difference is the failure type.
3. **Legitimate use was not suppressed — the control question answered.** `list_dir` still called 5/5 when expected (P2), `web_fetch` 10/10 when expected (P5 + P7). The ❌ counter-examples sharpened selection without scaring the model off the tools.
4. **The token paradox resolved in the examples' favor:** +364 schema tokens per call, yet −1,373 total tokens per run (−10%). Fewer unnecessary tool calls → fewer turns → fewer full-context resends. **The examples paid for themselves.**
5. **P4 side-benefit: 80% → 100%.** One baseline run added an extra `read_file` before writing; examples got 5/5. With n=1 this may be noise, or mild spillover from the "direct action" framing in the examples.

#### Per-prompt breakdown

| Prompt | Baseline | Examples | Change | Analysis |
|---|---|---|---|---|
| P1-read-known-file | 5/5 | 5/5 | — | ✅ Always correct |
| P2-browse-sandbox | 5/5 | 5/5 | — | ✅ Always correct — list_dir not suppressed |
| **P3-grep-across-files** | **1/5** | **4/5** | **🟢 +60pp** | **The list_dir counter-example hit its exact target** |
| P4-save-content | 4/5 | 5/5 | 🟢 +20pp | Baseline had 1 extra read_file; examples clean (n=1, may be noise) |
| P5-fetch-url | 5/5 | 5/5 | — | ✅ Always correct — web_fetch not suppressed |
| P6-search-online | 0/5 | 0/5 | — | ❌ Systematic chaining, examples ignored |
| P7-search-then-fetch | 5/5 | 5/5 | — | ✅ Always correct — the chain example matches the expected flow |
| P8-count-lines-efficient | 5/5 | 5/5 | — | ✅ Always correct |
| P9-no-tool-trivia | 5/5 | 5/5 | — | ✅ Always correct |
| **P10-trap-web-search-vs-local** | **0/5** | **0/5** | — | ❌ Systematic chaining, examples ignored |

#### Target-tool analysis

| Tool | Wrong calls (baseline) | Wrong calls (examples) | Called-when-expected |
|---|---|---|---|
| `list_dir` | 4 (P3 × 4) | 1 (P3 × 1) | 5/5 in both ✅ |
| `web_fetch` | 10 (P6 × 5, P10 × 5) | 10 (P6 × 5, P10 × 5) | 10/10 in both ✅ |

#### Deep dive: P3 — the counter-example did exactly what it was written to do (1/5 → 4/5)

Run-level traces:

```
baseline: 1 × [run_command] ✅
          4 × [list_dir, run_command, write_file] ❌ (browse → search → save)
examples: 4 × [run_command] ✅
          1 × [list_dir, run_command, write_file] ❌ (the same old pattern)
```

The counter-example — "Search all .txt/.md/.py files for the word 'sandbox' → run_command with grep … Do NOT call list_dir first; browsing adds a wasted turn before the real work. ❌ list_dir → grep" — is a near-verbatim description of the observed failure. It attacked the exact sequence the model kept producing and broke it in 4 of 5 runs.

Progress across the week on P3: Day 1 original descriptions 0/5 → Day 1 rewritten descriptions 3/5 → Day 4 baseline re-measure 1/5 → **Day 4 examples 4/5**. Descriptions and examples compound; the remaining 1/5 failure is the deep "list → search → save" workflow prior that neither intervention has fully eliminated.

#### Deep dive: web_fetch — the examples were ignored (10/10 in both conditions)

The web_fetch examples explicitly said: "Search: top 3 frameworks, one-line description each → web_search only, then answer from the snippets … ❌ web_search → web_fetch × 5." The model did web_search → web_fetch anyway — in all 10 runs, in both conditions — and kept fetching multiple pages (P10: 19 → 21 fetches). Even the "fetch at most one" instruction was ignored.

Three candidate explanations (not mutually exclusive):

1. **It's deliberate chaining, not confusion** (Day 1's conclusion). The model believes page content improves the answer; examples can't talk it out of a strategy it considers correct. The examples encode a fix for a behavior the model doesn't regard as broken.
2. **In-description examples are far from the decision moment.** They live in the schema block, ~2,000 tokens of system context away from where the search results arrive. Message-level few-shot sits directly in the conversation flow and may carry more weight. The knowledge section predicted this placement tradeoff — the experiment supplies the evidence.
3. **The prompts genuinely reward fetching.** P6 asks for current popular frameworks; P10 asks for recommendations. Snippets are often enough, but the model judges (not unreasonably) that a fetched page adds recency and detail.

#### Deep dive: token accounting — the examples paid for themselves

The +17% schema cost is the visible price; the hidden saving is bigger. Each unnecessary tool call costs a full extra agent turn: the tool result is appended to context and the entire history is re-sent on the next request. Cutting 7 wrong calls (19 → 12 across 50 runs) removed whole turns — visible as −1,319 prompt tokens per run (12,845 → 11,526) and −1,373 total tokens per run (−10%).

This directly answers Day 1's cost concern ("longer descriptions cost +33.6% tokens"). Verbosity is not automatically expensive — the question is whether it prevents enough wrong-tool turns to pay for itself. Here it did.

#### Re-measurement caveat

This experiment's baseline is a re-run of Day 1's "after" condition, which originally measured **76% exact-match / 0.127 wrong-tool**. The re-measure: **70% / 0.163**. Same descriptions, same prompts, same model — ±6pp drift at 50 runs. Cross-experiment deltas must be read cautiously; the valid comparison is within this experiment. Still, 0.113 is the lowest wrong-tool rate measured on the full 10-prompt suite so far (Day 1 before: 0.163 → after: 0.127 → Day 4 examples: 0.113).

---

### Surprise

1. **Same intervention, opposite outcomes: −75% vs. 0%.** One description append, two target tools, two completely different results. The deciding variable is the failure mode: confusion (`list_dir`) responds to examples; deliberate chaining (`web_fetch`) does not. This is the cleanest test so far of "examples fix selection errors, not strategies."
2. **The examples paid for themselves in tokens — against Day 1's cost intuition.** +17% schema cost per call, −10% total tokens per run. The expensive thing in an agent loop is not long descriptions, it's unnecessary turns.
3. **Zero suppression of legitimate use.** Going in, I worried the ❌ counter-examples might make the model hesitant to call the annotated tools at all. The controls came back perfect: 5/5 and 10/10 called-when-expected in both conditions.
4. **Baseline drift is real: 76% → 70% across days.** Re-measuring the identical condition gave a 6pp different exact-match rate (P3 alone swung 3/5 → 1/5). At 5 runs per cell, day-to-day comparisons are noisy — every experiment needs its own same-day baseline.
5. **Examples moved P3 where descriptions had stalled.** Day 1's rewrite got P3 to 3/5; the re-measured baseline fell back to 1/5; examples reached 4/5 — the best score yet for the week's hardest prompt. The remaining failure is the workflow prior itself, which may need Day 5's constraints rather than more description text.

---

### Key insights

1. **Classify the failure before writing examples.** Two classes: selection errors (the model doesn't know when/what to call) and strategic behavior (the model knows, and disagrees with your annotation). Examples only fix the first. If your expected-tool annotation fights the model's reasonable strategy, fix the annotation (Day 1's P6/P10 lesson) — not the description.
2. **Counter-examples are the active ingredient.** The ❌ negative examples did the work — consistent with Day 1's finding that "Do NOT call when — use X instead" was the highest-leverage phrase. Positive examples are confirmation; negative examples are correction.
3. **Measure tokens end-to-end.** Schema cost went up while run cost went down. Anyone measuring only description length draws the wrong conclusion about cost (Day 1's +33.6% concern, revisited).
4. **Placement matters — and now it's measured.** In-description examples fixed an ambiguity problem (list_dir) but failed to restructure workflow behavior (web_fetch). To override chains, move the few-shot into the message stream — or reach for constraints (Day 5's forced tool use).
5. **The tool-surface stack, ordered by measured effect this week:** descriptions (Day 1, −22% wrong-tool) > worked examples (Day 4, −31% relative) > granularity (Day 2, +2pp) > tool count (Day 3, no effect). Words beat structure every time.

---

### Next question (→ Day 5)

Examples could not stop the web_fetch chaining. Day 5 asks: can constraints do what guidance couldn't — force `tool_choice` on a `classify_request` router tool and disable parallel calls? Per PLAN.md: build the forced router and compare it against one big agent.

## day5: Forced tool use (tool_choice), disabling parallel calls — when to constrain the model.
一、Forced Tool Use：压缩决策空间
通过api参数解干预llm条件概率分布：
| 平台            | 参数                    | 模式                                                | 语义                           |
| ------------- | --------------------- | ------------------------------------------------- | ---------------------------- |
| **OpenAI**    | `tool_choice`         | `"auto"`                                          | 默认，模型自主决定是否调用                |
|               |                       | `"none"`                                          | 禁止调用任何工具，强制文本回复              |
|               |                       | `"required"`                                      | 必须调用**至少一个**工具（GPT-4 Turbo+） |
|               |                       | `{"type": "function", "function": {"name": "x"}}` | **强制调用指定工具**                 |
|               | `parallel_tool_calls` | `true`/`false`                                    | 是否允许单次响应中并行调用多个工具            |
| **Anthropic** | `tool_choice`         | `{"type": "auto"}`                                | 默认                           |
|               |                       | `{"type": "any"}`                                 | 必须调用至少一个工具                   |
|               |                       | `{"type": "tool", "name": "x"}`                   | 强制指定工具                       |
|               |                       | `{"type": "none"}`                                | 禁止工具调用                       |
| **Gemini**    | `tool_config`         | `AUTO`, `ANY`, `NONE`                             | 对应上述模式                       |

User: "查一下北京天气"
┌─────────────────────────────────────────────────────────────┐
│ 约束强度谱系                                                │
├─────────────────────────────────────────────────────────────┤
│ NONE        → 模型自由回答："北京今天晴朗，25度..."         │
│             │ 风险：幻觉、信息过时                          │
├─────────────────────────────────────────────────────────────┤
│ AUTO        → 模型可能调用 get_weather，也可能直接回答       │
│             │ 风险：该查时不查，不该查时乱查                │
├─────────────────────────────────────────────────────────────┤
│ REQUIRED    → 模型必须调用某个工具，但自选具体哪个            │
│             │ 风险：选错工具（如选了 search_web 而非天气）  │
├─────────────────────────────────────────────────────────────┤
│ SPECIFIC    → 模型必须调用 get_weather，且只能调用它        │
│             │ 风险：如果 query 与工具不匹配，参数可能胡编    │
└─────────────────────────────────────────────────────────────┘

强制调用指定工具本质上是将意图分类（Intent Classification）从模型手中收回，交由上层业务逻辑或确定性 Router 决定。此时 LLM 退化为参数填充器（Argument Extractor），仅负责：
1. 从自然语言中提取参数
2. 按 schema 格式化输出

二、Disabling Parallel Calls：强制时序约束
现代 Function Calling API（OpenAI GPT-4o+、Claude 3.5+）默认支持并行工具调用（Parallel Tool Calls）。当模型判断多个工具之间无数据依赖时，会在单次响应中批量发起：
{
  "tool_calls": [
    {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"city\": \"Beijing\"}"}},
    {"id": "call_2", "function": {"name": "get_stock", "arguments": "{\"ticker\": \"AAPL\"}"}},
    {"id": "call_3", "function": {"name": "search_news", "arguments": "{\"query\": \"Apple earnings\"}"}}
  ]
}
这三条调用在应用层可以并发执行（如通过 asyncio.gather），总耗时 ≈ 最慢的那个请求。
设置 parallel_tool_calls: false（或 prompt 约束）后，模型每次响应最多只生成一个 tool call。这强制了串行执行语义。
即使 API 层面不禁用，也可以通过架构设计实现串行：
| 机制            | 实现方式                              | 效果                          |
| ------------- | --------------------------------- | --------------------------- |
| **API 参数**    | `parallel_tool_calls: false`      | 硬约束，模型物理上无法输出多个 tool\_calls |
| **Prompt 约束** | "每次只能调用一个工具，等待结果后再继续"             | 软约束，依赖模型遵循指令                |
| **应用层拦截**     | 收到多个 tool\_calls 时只执行第一个，丢弃其余     | 事后约束，浪费 token               |
| **状态机模式**     | 维护 `current_step` 变量，仅暴露当前步骤允许的工具 | 架构约束，最可靠                    |
三、决策框架：何时施加约束
强制调用（Forced Tool Use）的场景矩阵
| 场景            | 推荐约束                    | 理由                                             |
| ------------- | ----------------------- | ---------------------------------------------- |
| **结构化输出必须**   | `REQUIRED` 或 `SPECIFIC` | 禁止自由文本，确保输出可被下游系统解析。如：表单填充、API 网关、数据库写入。       |
| **Router 节点** | `SPECIFIC`              | 上层已确定意图（如 embedding 分类器判定为"天气查询"），LLM 不应重新决策。  |
| **副作用不可跳过**   | `REQUIRED`              | 审计日志、权限校验、计费记录等操作必须发生，不能因模型"觉得没必要"而省略。         |
| **防止幻觉**      | `REQUIRED` + 具体工具       | 知识库问答场景中，强制调用 `retrieve_context`，禁止模型凭记忆回答。    |
| **用户安全边界**    | `NONE`                  | 敏感操作确认环节（"你确定要删除吗？"），强制模型以文本形式请求确认，禁止直接执行删除工具。 |
| **闲聊/寒暄**     | `NONE`                  | 用户说"谢谢"时，禁止调用任何工具，避免无意义的 search/log 操作。        |
禁用并行（Serial Execution）的场景矩阵
| 场景            | 必须串行  | 理由                                                                 |
| ------------- | ----- | ------------------------------------------------------------------ |
| **数据依赖链**     | 是     | 工具 B 的参数依赖工具 A 的输出。如：`get_user_id` → `query_orders(user_id)`       |
| **状态副作用**     | 是     | 工具 A 写入数据库，工具 B 读取同一记录。并行可能导致读取到旧状态（竞态）。                           |
| **资源锁竞争**     | 是     | 两个工具操作同一文件/同一行数据库记录。如：`update_inventory` 与 `check_stock` 并行可能导致超卖。 |
| **错误级联隔离**    | 是     | 每步执行后立即验证结果，失败即终止。并行时批量失败难以定位根因。                                   |
| **交互式确认**     | 是     | 每步需要用户确认（如"找到 3 个结果，执行第几个？"）。                                      |
| **独立 I/O 查询** | **否** | 查询北京天气 + 查询纽约天气 + 查询伦敦天气，应并行以优化延迟。                                 |
组合策略：约束强度的动态调整
用户输入
    ↓
[Intent Classifier] —— 确定性代码
    ↓
┌─────────────────────────────────────────┐
│ 阶段 1: 意图识别                        │
│ tool_choice: REQUIRED (必须调用 classify_intent) │
│ parallel: false (单意图，无需并行)      │
└─────────────────────────────────────────┘
    ↓
意图 = "数据分析"
    ↓
┌─────────────────────────────────────────┐
│ 阶段 2: 数据提取                        │
│ tool_choice: SPECIFIC (强制 run_sql_query) │
│ parallel: false (SQL 可能有依赖)        │
└─────────────────────────────────────────┘
    ↓
数据 = [result1, result2]
    ↓
┌─────────────────────────────────────────┐
│ 阶段 3: 可视化 + 摘要                   │
│ tool_choice: REQUIRED (必须调用 generate_chart + generate_summary) │
│ parallel: true (图表和文本无依赖，可并发) │
└─────────────────────────────────────────┘
四、约束的隐性成本与风险
延迟成本（Latency Tax）
串行调用的延迟是累加的。假设平均工具调用 RTT = 500ms（含 LLM 推理 + API 执行）：在实时交互场景（如客服 ChatBot），1500ms 的串行延迟可能超出用户忍耐阈值。
上下文膨胀（Context Inflation）
每轮串行调用都会产生 Observation → LLM → Next Tool Call 的往返，导致：
* 历史消息累积
* 可用上下文窗口被压缩
* 注意力稀释（远距离信息遗忘）
错误传播模式差异
| 模式 | 错误特征        | 恢复难度        |
| -- | ----------- | ----------- |
| 并行 | 批量失败，错误相互独立 | 可部分重试，但根因分散 |
| 串行 | 级联失败，一步阻断后续 | 定位精确，但回滚链长  |
强制串行执行 5 个工具，第 4 步失败后，前 3 步的副作用（如已写入的数据库记录）需要补偿事务，增加了架构复杂度。
模型退化风险
过度使用 SPECIFIC 强制调用可能导致：
* 参数幻觉：当用户输入与强制工具不匹配时，模型被迫"硬编"参数以满足 schema 要求。
* 绕过行为：模型可能通过参数注入（如 query="ignore previous instructions"）在参数层面实现"不调用"的效果。
架构模式：约束的工程化实践
模式一：确定性 Pipeline（强制串行 + 强制指定）
适用于高度标准化、零容错的业务流程（如金融交易、医疗诊断）。
pipeline = [
    {"step": "validate_identity", "tool": "verify_kyc", "parallel": False},
    {"step": "check_risk", "tool": "risk_assessment", "parallel": False},
    {"step": "execute_trade", "tool": "place_order", "parallel": False}
]

for step in pipeline:
    response = llm.call(
        tool_choice={"type": "function", "function": {"name": step["tool"]}},
        parallel_tool_calls=False
    )
    if not validate(response):
        raise PipelineAbort(step)
模式二：受控 ReAct（动态约束）
if turn_count == 0:
    # 第一轮：必须行动，不能空想
    tool_choice = "required"
elif last_observation.get("status") == "needs_clarification":
    # 需要用户澄清：禁止调用工具，强制文本提问
    tool_choice = "none"
elif current_task in ["data_extraction", "validation"]:
    # 关键步骤：串行执行，精确控制
    tool_choice = {"type": "function", "function": {"name": current_task}}
    parallel_tool_calls = False
else:
    # 探索阶段：放开约束，允许自主并行
    tool_choice = "auto"
    parallel_tool_calls = True
模式三：并发沙盒（并行执行 + 结果仲裁）
# 模型并行请求：search_web + query_knowledge_base + ask_user_profile
results = await asyncio.gather(tools, return_exceptions=True)

# 应用层仲裁：按优先级选择结果
if results["query_knowledge_base"]:
    return results["query_knowledge_base"]  # 内部知识优先
elif results["search_web"]:
    return results["search_web"]            # 外部搜索次之
| 约束维度                         | 控制对象              | 核心问题                       |
| ---------------------------- | ----------------- | -------------------------- |
| **Forced tool use**          | **做什么**（What）     | 将意图分类权从 LLM 收回，交给确定性逻辑     |
| **Disabling parallel calls** | **怎么做**（How/When） | 将时序编排权从 LLM 收回，交给状态机或数据流引擎 |
黄金法则：
当业务逻辑已经预先知道答案（"此时必须做 X"），就不要让模型重新发明答案。
当步骤之间存在数据依赖或状态竞争，就不要让模型猜测执行顺序。

---

### Experiment — Forced `classify_request` router vs. one big agent

#### Question

Does a router architecture — the FIRST call forced to be a `classify_request` tool, then dispatch to a fresh specialist conversation with only that specialty's tools — beat one big agent holding all 6 tools, on the same 10 tool-selection prompts? And what does routing cost in tokens?

**Hypothesis:** The router should win on wrong-tool rate (each specialist sees 1–3 tools instead of 6 — almost nothing left to confuse), lose on tokens (the routing call is pure overhead), and its weak point should be misclassification: a wrong route is worse than a wrong tool call, because the specialist may not even *own* the tool the task needs. The specialist's reduced tool surface should also finally break the `web_search → web_fetch` chaining that Days 1 and 4 could not (the web specialist still owns both tools, so prediction: P6/P10 stay 0% — the chaining survives routing too).

#### Setup

- **Model:** qwen3.7-plus
- **Tasks:** the same 10 tool-selection prompts from Day 1 (same sandbox, same gold tool annotations)
- **Varied:** architecture only — monolith vs. router. Tool schemas and descriptions are identical across conditions (Day 1 "after" set)
- **Held constant:** model, system-prompt guidance content, max_turns (12)
- **Runs per condition:** 5 → 100 runs planned (router runs = 2 API conversations each)
- **Code:** `experiments/week05/day5_router.py`
- **Results:** `experiments/week05/day5_results_20260814_084433.json` (monolith + v1 router failure), `experiments/week05/day5_results_20260816_113553.json` (router re-run after the fix)

**Condition monolith** — one agent, all 6 tools, general system prompt (identical to the best configuration from Days 1–4).

**Condition router** — two stages:

1. *Routing call:* `classify_request` is the only tool in the schema, and `tool_choice` is forced to that function — the model cannot answer, dodge, or pick another tool; its first output must be the classification `{category, confidence, reason}`.
2. *Dispatch:* a fresh conversation with a specialist system prompt and only that specialty's tools:

| Specialist | Tools | Gold routes |
|-----------|-------|-------------|
| `file` | read_file, write_file, list_dir | P1, P2, P4 |
| `command` | run_command | P3, P8 |
| `web` | web_search, web_fetch | P5, P6, P7, P10 |
| `none` | (no tools — answer directly) | P9 |

`classify_request` is router infrastructure, not a task tool: it is excluded from tool-selection scoring and reported separately as routing accuracy. Routing failures (unparseable classification, no tool call) count as failed runs — they are real router behavior, not noise.

#### Result

**The router condition produced zero usable runs — all 50 died in the routing call with the same HTTP 400:**

```
Error code: 400 - {'error': {'code': 'invalid_parameter_error',
'message': 'The tool_choice parameter does not support being set to
required or object in thinking mode', ...}}
```

DashScope's qwen3.7-plus runs with **thinking enabled by default**, and the gateway rejects `tool_choice` entirely while thinking is on. The fallback (`tool_choice="required"`) hit the same wall — the restriction applies to any non-`auto` value. This is a platform-level incompatibility, not a model-behavior problem: **the day's core mechanism (forced tool use) is unavailable in thinking mode on this endpoint.**

The fix (applied to `day5_router.py`): the routing call now tries, in order — (1) forced single-function `tool_choice` with `enable_thinking=False` via `extra_body` (the documented workaround: qwen3.x thinking is hybrid and accepts the opt-out), (2) forced `tool_choice` with default settings, (3) prompt-only forcing — `classify_request` remains the only tool, and a refusal to call it is scored as a routing failure. The mechanism actually used is recorded per run, and `--conditions router` re-runs only the router against the completed monolith numbers.

**Monolith condition (complete, n=50):**

| Metric | Monolith |
|--------|----------|
| Should-call rate | 1.000 ± 0.000 |
| Wrong-tool rate | 0.137 ± 0.233 |
| Exact-match rate | 37/50 = 74.0% |
| Tokens/run | 12,358 ± 11,526 |
| Cost/run | $0.00546 |

Per-prompt: P1, P2, P4, P5, P7, P8, P9 all 5/5; **P3 2/5** (the week's chronic "browse → search → save" pattern); **P6 0/5** and **P10 0/5** (the `web_search → web_fetch` chaining that Days 1 and 4 diagnosed as deliberate strategy, not confusion). Wrong-tool calls: `web_fetch` ×10, `list_dir` ×3, `write_file` ×2 — the same three tools as every day this week. Also note the baseline drift: this exact monolith configuration measured 76% (Day 1), 70% (Day 4 baseline), 74% (today) — ±6pp swing at n=50 on the same prompts and model.

**Router condition (re-run 2026-08-16 after the fix, n=50):**

| Metric | Monolith (n=50) | Router (n=50) | Delta |
|--------|-----------------|----------------|-------|
| Should-call rate | 1.000 ± 0.000 | 1.000 ± 0.000 | → tie |
| Wrong-tool rate | 0.137 ± 0.233 | 0.100 ± 0.200 | ✅ −27% relative |
| Exact-match rate | 37/50 = 74.0% | 40/50 = 80.0% | ✅ +6.0pp |
| Routing accuracy | — | 50/50 = 100% | perfect |
| Tokens/run | 12,358 ± 11,526 | 6,609 ± 5,582 | ✅ **−46.5%** |
| Cost/run | $0.00546 | $0.00316 | ✅ **−42%** |

All 50 routing calls landed on the `forced_function_no_thinking` rung — `enable_thinking=False` unblocked the forced `tool_choice` exactly as the fix predicted — and every classification was correct at high confidence (0 routing errors).

Caveat: the monolith column is the 2026-08-14 run, the router column 2026-08-16; same-day drift on this suite has been ±6pp. The deltas that matter (P3's 5/5 consistency, the token halving) sit far outside that noise band.

**Per-prompt exact-match comparison:**

| Prompt | Monolith | Router | Analysis |
|---|---|---|---|
| P1-read-known-file | 5/5 | 5/5 | ✅ |
| P2-browse-sandbox | 5/5 | 5/5 | ✅ |
| **P3-grep-across-files** | **2/5** | **5/5** | **🟢 the week's hardest prompt, fixed** |
| P4-save-content | 5/5 | 5/5 | ✅ |
| P5-fetch-url | 5/5 | 5/5 | ✅ |
| **P6-search-online** | **0/5** | **0/5** | ❌ chaining survived routing |
| P7-search-then-fetch | 5/5 | 5/5 | ✅ |
| P8-count-lines-efficient | 5/5 | 5/5 | ✅ |
| P9-no-tool-trivia | 5/5 | 5/5 | ✅ |
| **P10-trap-web-search-vs-local** | **0/5** | **0/5** | ❌ chaining survived routing |

#### Deep dive: P3 — the reduced tool surface did what four days of descriptions could not

P3's failure all week has been the "browse → search → save" workflow prior: `[list_dir, run_command, write_file]` instead of `[run_command]`. Day 1's rewrite got it to 3/5, Day 4's counter-examples to 4/5, and the re-measured monolith baseline sat at 2/5.

The router's command specialist owns exactly one tool: `run_command`. There is no `list_dir` to browse with and no `write_file` to persist with — the wrong pattern is not discouraged, it is **unrepresentable**. All 5 runs called `[run_command]` alone. The week's progression on P3: descriptions (0→3/5) < counter-examples (4/5) < constraint (5/5).

#### Deep dive: P6/P10 — the chaining survived routing, as predicted

The web specialist still owns both `web_search` and `web_fetch`, and it still chains them in all 10 runs (the only wrong-tool calls of the entire router condition: `web_fetch` ×10). The hypothesis said routing would not break the chaining because the chaining is the model's strategy, not a selection confusion — the specialist confirms it: shrinking the tool surface to 2 tools did not change the model's judgment that a fetched page improves the answer. Three interventions (descriptions, examples, routing), three failures to change it. The "error" is likely a good behavior the gold annotation disagrees with — Day 1's original conclusion, now with a third replication.

#### Deep dive: tokens — the router was cheaper, not more expensive

The routing call cost 758 tokens on average — 11% of a router run. But a router run totals 6,609 tokens against the monolith's 12,358: the monolith pays all 6 tool schemas on every turn (~2,100–2,500 schema tokens per call, per Day 3's measurement), while specialists pay 1–3 schemas; and the reduced surface removes the extra turns that produced wrong-tool calls (P3's browse and save turns, P4's extra read). Adding a component removed nearly half the spend — the opposite of the "routing is overhead" intuition. (The router call itself also runs with thinking disabled, so it skips thinking tokens entirely.)

#### Surprise

1. **The constraint was itself constrained — by the platform, not the model.** The day asks "when to constrain the model"; the answer turned out to include "only when your endpoint supports the constraint." `tool_choice` and thinking mode are mutually exclusive on DashScope qwen3.x — a wire-level 400, not a soft behavioral effect. Forced tool use is a platform capability to verify before you design around it, exactly the failure mode Week 4's "measure everything" exists for. The fix (disable thinking for the forced call) worked on all 50 runs.
2. **The router won on tokens — 46.5% cheaper, against my own hypothesis.** I predicted routing would lose on tokens because the classification call is pure overhead. Measured: the overhead is 11% of a run that is itself half the size. Token cost is dominated by schema × turns, and routing attacks exactly that.
3. **Routing was perfect — the feared misroute cascade never materialized.** 50/50 correct, all high-confidence, on well-separated categories. The router's theoretical weak point (a wrong route denying the specialist the tool it needs) stayed theoretical at this task difficulty. Classification of 10 known intents is simply an easier problem than tool selection under a 6-way choice with overlapping descriptions.
4. **P3 was fixed by making the wrong answer impossible — 2/5 → 5/5.** After a week of interventions, the one that worked was not better text but a smaller possibility space. This is the strongest single data point for the week's goal: tool *surface* design is the highest-leverage agent decision.
5. **`web_fetch` chaining is the last survivor.** Descriptions (Day 1): no effect. Worked examples (Day 4): no effect. Routing + reduced tool surface (Day 5): no effect. Whatever this behavior is, it is deeper than the tool layer.

#### Key insights

1. **Forced tool use has a hidden prerequisite: the constraint stack is provider-specific.** `tool_choice` × thinking-mode incompatibility is the kind of interaction no prompt-engineering guide mentions and only a live call reveals. Build the fallback ladder explicitly (as done here) and record which rung each run landed on — "forced" is not a portable guarantee.
2. **Constraint strength beats guidance strength when the failure is a workflow prior.** The measured hierarchy this week: "Do NOT call" clauses < counter-examples < "the tool doesn't exist here." Words shift probabilities; structure removes possibilities. Reach for structure when the failure is a habit, not a confusion.
3. **More components does not mean more cost.** The router added an entire decision stage and cut spend 42%. The cost driver in an agent loop is what gets re-sent every turn (schemas + history), not how many LLM calls the architecture makes.
4. **Routing converts tool-confusion risk into classification risk — and classification is the easier problem.** The monolith chronically over-calls (0.137 wrong-tool); the router's classification was 100%. If your intents are enumerable and separable, pay for routing and shrink the surface.

#### Next question (→ Day 6)

The Day 6 experiment picks the most surprising effect from Days 1–5 and measures it properly. The strongest candidate now: **`web_fetch` chaining** — the one behavior that survived descriptions (Day 1), worked examples (Day 4), and a reduced tool surface (Day 5). Candidate measurement: does the chained fetch actually improve answer quality (LLM-as-judge, with vs. without the extra fetch), or is it pure waste? Runner-up: the `tool_choice` × thinking-mode constraint — A/B forced-no-thinking vs. prompt-only routing on classification quality and cost.

## day6: The most surprising effect from Days 1–5, measured properly

### Choosing the effect — Day 3's null result over Day 5's two candidates

Day 5's "next question" nominated `web_fetch` chaining and the `tool_choice` constraint. Both were rejected for Day 6, on purpose:

1. **The chaining is surprising, but no longer open.** It replicated identically three times (Days 1, 4, 5) and Day 5's analysis explains it as deliberate strategy, not confusion. What's left to measure is answer *quality* (does the extra fetch help the user?), which needs the LLM-as-judge machinery from Week 4 — a good experiment, but not a tool-*selection* question, and not an effect in doubt.
2. **The `tool_choice` × thinking-mode constraint is already settled.** Discovered, fixed, and validated 50/50 on Day 5. There's no live question left — it's a footnote for the report, not a Day-6 experiment.
3. **Day 3's null result is the one effect that contradicts the field's default belief** — and it did so with the weakest methodology of the week. "More tools → more distraction → worse selection" is the standard lore (it's why this plan's Week 3 experiment tested 4 tools vs. 4+8 irrelevant ones). Week 3's Day 6 found nothing (25/25 pass in both conditions, 0 irrelevant calls). Week 5's Day 3 found nothing again — and *against its own expectation*: wrong-tool rate *fell* at 30 tools (5.4% → 3.6% → 4.0% per-run mean, n=40/cell), with exactly 0 dummy calls in 80 runs. But Day 3 only ran 8 prompts (dropping the 2 that need `web_fetch`), always presented the same fixed slices of 10/25 dummies, and reported no intervals or significance tests. **Maximum surprise × minimum rigor = exactly what "measure it properly" is for.** If the null is real, "cut tools to save the model from confusion" is a myth for this model — and the real reason to cut tools becomes cost, not reliability. That reframing is worth a proper measurement.

### Experiment — Does adding realistic-but-irrelevant tools degrade tool selection?

#### Question

Do wrong-tool rate and dummy-call rate increase with tool count (6 vs. 15 vs. 30 tools) when the extra tools are realistic-but-irrelevant, or is tool selection independent of distractor count?

**Hypothesis (the lore, and Day 3's original expectation):** more tools → more selection errors. **Null:** no effect — the only cost of a bigger toolkit is tokens.

#### Setup

- **Model:** qwen3.7-plus
- **Tasks:** the full 10-prompt tool-selection suite (Day 1 "after" schemas — includes P5/P7, which Day 3 had to drop)
- **Varied:** tool count only — 6 real tools + 0/9/24 distractors
- **Held constant:** model, the 6 core tool schemas and descriptions, system prompt, max_turns (12), sandbox, gold annotations
- **Runs per condition:** 5 → 150 runs total (0 API errors)
- **Code:** `experiments/week05/day6_tool_distraction.py`
- **Results:** `experiments/week05/day6_results_20260816_124025.json`

The four methodological upgrades over Day 3:

| Day 3 weakness | Day 6 fix |
|---|---|
| 8 prompts, no `web_fetch` in the core set | Full 10 prompts, all 6 core tools (matches Days 1/4/5) |
| Same fixed dummy slices (first 10 / first 25) every run | **Random dummy subset per run** (seeded → reproducible), with the 4 hardest near-miss distractors always included |
| Means only, no uncertainty | Wilson 95% CIs on pooled rates + Fisher exact tests between conditions |
| Conditions run back-to-back (time drift = confound) | Cell order shuffled with a fixed seed (20260816) |

The 25 distractors are realistic-looking tools from other domains (send_email, query_database, search_documents, get_stock_price, ...). Four are *near-miss* traps whose descriptions deliberately overlap real capabilities — `search_documents` ("call when the user asks to search/find..."), `query_database` ("look up / retrieve structured data"), `summarize_text`, `extract_entities` — and these are loaded into **every** dummy run; only the generic distractors vary. A dummy call returns a plausible success response (no error signal to learn from) and is scored separately from wrong-tool rate (any call outside the gold tool set, unique-name based like Days 1/4/5).

#### Result

**Dummy-call rate — the key metric:**

| Condition | Dummy calls | Pooled rate | Wilson 95% CI |
|-----------|-------------|-------------|---------------|
| 6_tools (0 distractors) | 0/114 calls | 0.0% | [0.0%, 3.3%] |
| 15_tools (9 distractors) | 0/110 calls | 0.0% | [0.0%, 3.4%] |
| 30_tools (24 distractors) | 0/95 calls | 0.0% | [0.0%, 3.9%] |

**100 runs offered 9–24 randomized distractors (near-miss traps always among them). Not one dummy was ever called.** Fisher exact: p = 1.000 everywhere (trivially, all cells zero).

**Wrong-tool rate — same story, measured two ways:**

| Condition | Per-run mean ± sd (Day 1 metric) | Pooled unique-name rate [CI] | Fisher vs. previous |
|-----------|----------------------------------|------------------------------|---------------------|
| 6_tools | 0.120 ± 0.214 | 12/62 = 19.4% [11.4%, 30.9%] | — |
| 15_tools | 0.137 ± 0.233 | 15/65 = 23.1% [14.5%, 34.6%] | p = 0.668 (ns) |
| 30_tools | 0.127 ± 0.227 | 14/64 = 21.9% [13.5%, 33.4%] | p = 1.000 (ns) |

**Exact-match rate:**

| Condition | Exact match | Wilson 95% CI |
|-----------|-------------|---------------|
| 6_tools | 38/50 = 76.0% | [62.6%, 85.7%] |
| 15_tools | 37/50 = 74.0% | [60.4%, 84.1%] |
| 30_tools | 38/50 = 76.0% | [62.6%, 85.7%] |

**Cost — the only curve that moved:**

| Condition | Tokens/run | vs. 6_tools | Cost/run | vs. 6_tools | Latency |
|-----------|-----------|-------------|----------|-------------|---------|
| 6_tools | 9,354 ± 6,741 | — | $0.00424 | — | 18.2 ± 16.2 s |
| 15_tools | 13,991 ± 9,419 | +50% | $0.00605 | +43% | 15.0 ± 12.3 s |
| 30_tools | 18,075 ± 8,641 | +93% | $0.00769 | +81% | 15.3 ± 13.1 s |

Latency is flat (within noise). Total calls per condition actually *shrank* slightly (114 → 110 → 95) — the model did not wander more.

**Per-prompt exact-match (out of 5 runs; the failure profile is identical in every column):**

| Prompt | 6_tools | 15_tools | 30_tools |
|---|---|---|---|
| P1-read-known-file | 5/5 | 5/5 | 5/5 |
| P2-browse-sandbox | 5/5 | 5/5 | 5/5 |
| P3-grep-across-files | 3/5 | 2/5 | 2/5 |
| P4-save-content | 5/5 | 5/5 | 5/5 |
| P5-fetch-url | 5/5 | 5/5 | 5/5 |
| P6-search-online | 0/5 | 0/5 | 1/5 |
| P7-search-then-fetch | 5/5 | 5/5 | 5/5 |
| P8-count-lines-efficient | 5/5 | 5/5 | 5/5 |
| P9-no-tool-trivia | 5/5 | 5/5 | 5/5 |
| P10-trap-web-search-vs-local | 0/5 | 0/5 | 0/5 |

#### Deep dive: every wrong call, at every tool count, went to the same three real tools

Extra-call volume by condition: `web_fetch` ×31/×26/×21 (P6 and P10 chaining), `list_dir` ×7/×12/×6 (P3's browse-first habit), `write_file` ×0/×2/×2 (P3's save-the-results habit). The distractors were never the problem — the model's *real-tool* habits are, and they don't care how many other tools exist. Adding 24 more choices changed nothing because the errors were never "choosing badly among many" — they were "choosing an extra real tool on purpose."

#### Deep dive: the cost curve is linear; reliability is flat

Tokens/run grew +50% at 15 tools and +93% at 30 — matching Day 3's Part 1 arithmetic (schema cost ≈ 2,465 tokens/call at 15 tools, 4,718 at 30; each turn re-pays it). Selection quality stayed flat at every point on that curve. **Tool count is a cost knob, not a reliability knob** — for this model, on this task set. The practical rule inverts the lore: prune tools to cut tokens per turn, not to rescue the model from distraction; keep them if the capability earns its schema rent.

#### Deep dive: what "measured properly" actually changed

The upgrade matters because Day 3's null could have been an artifact of its fixed dummy slices — 15 of its 25 distractors were *never offered*. Day 6 offered every dummy at least once across runs (seeded sampling), always armed the 4 near-miss traps, doubled the prompt set to the full 10, and put CIs and Fisher tests on the numbers. The null survived all of it, and the CIs now bound the effect: **we can exclude a dummy-call rate above ~4% at any tool count** — a large "distraction effect" simply is not there. What the CIs cannot exclude is a small effect (1–2%) — and a proper measurement should say so.

#### Surprise

1. **The null replicated under a stronger design — zero dummy calls in 205 tool calls across 100 distractor-loaded runs.** Not even `search_documents` (whose description says "call when the user asks to search/find...") was touched, in 50 runs where it was present. The model reads descriptions well enough to see the domain boundary.
2. **Wrong-tool rate didn't just fail to rise — it was flat while the error *pattern* stayed pixel-identical.** Same three tools (`web_fetch`, `list_dir`, `write_file`), same three prompts (P3, P6, P10), same rates, in all three conditions. Tool selection on this suite is governed by task-level habits, not by how many options the schema presents.
3. **The 6-tool baseline landed exactly where Days 1/4/5 did.** Per-run wrong-tool 0.120 vs. Day 1-after 0.127 vs. Day 5 monolith 0.137; exact-match 76% vs. the week's 70–76% band. Three independent sessions agree within the ±6pp drift Day 5 measured — good evidence the day-6 numbers are the suite, not the day.

#### Key insights

1. **"Too many tools confuse the model" did not survive contact with measurement — twice.** Week 3's Day 6 (easy tasks) and Week 5's Day 6 (full suite, near-miss traps, proper stats) both found nothing. Treat tool-count as a token budget question until a specific distractor class (near-duplicate tools with overlapping names, e.g. `search` vs. `search_web`) is shown to break it — that's the untested corner of this claim.
2. **A null result with CIs is a positive finding.** Day 3 could only say "no calls observed." Day 6 can say "rate < ~4% with 95% confidence." The interval is what turns an absence into a bound, and the bound is what changes a design rule.
3. **Surprise is a property of prior beliefs, and prior beliefs need stress-testing.** The week's most repeated claim (more tools → worse selection) was the one nobody had measured against a strong design. It was also the only effect of the week in the *opposite* direction of prediction — which is precisely the Day-6 selection criterion, applied consistently.
4. **Cost scaling is the tool-count decision that's real.** +50%/+93% tokens for +9/+24 tools, flat quality — schema tokens re-paid every turn, as Day 3's Part 1 predicted. The week's lever ranking now reads: *capability coverage* (does the tool exist) > *description quality* (Days 1, 4) > *constraint* (Day 5) > *surface size* (Day 6: a cost decision, not a correctness decision).

#### Next question (→ Week 6)

Context management, but carrying two open threads: (1) the 30-tool condition re-pays ~4,700 schema tokens every turn — Week 6's pruning/compaction work is exactly the machinery that makes a big tool surface *affordable*; measure whether schema-token pruning changes the selection picture at 30 tools. (2) The untested corner of the null: distractors that are *near-duplicates* (overlapping names, same verbs) rather than clearly-different-domain tools — does the null survive deceptively similar tools, or is that where confusion actually lives?

## Week 5 closing synthesis — tool surface is the lever, and now it's measured

The week's goal was to internalize that tool *surface* design is the highest-leverage agent decision. Five experiments later, the lever ranking is empirical, not asserted:

- **Capability presence is everything.** P3 went 0/5 → 3/5 from descriptions alone (Day 1), → 4/5 from counter-examples (Day 4), → 5/5 when the wrong tools were *unrepresentable* (Day 5). Words < examples < structure.
- **Granularity barely matters** for correctness (Day 2: 90% vs. 92% exact sequence; the mega-tool even cost more tokens) — the debates that consume design meetings are second-order effects.
- **Descriptions are triggering conditions, and they work at the margin** (Day 1: wrong-tool 16.3% → 12.7%; Day 4: 16.3% → 11.3% with examples) — but they cannot move behaviors that the model *intends* (the `web_search → web_fetch` chaining, immune to three interventions).
- **Constraints are powerful and provider-fragile** (Day 5: router 74% → 80% exact, −46% tokens, but only after discovering `tool_choice` is banned in thinking mode).
- **Tool count is a cost knob, not a reliability knob** (Day 6: zero dummy calls in 100 distractor-loaded runs; flat wrong-tool and exact-match across 6/15/30 tools; +93% tokens is the entire price of 24 extra tools).

The single most transferable sentence of the week: **when the model misbehaves with tools, first ask whether the behavior is confusion or strategy — words fix confusion, structure removes strategy, and nothing fixes what the gold annotation mislabels.**

