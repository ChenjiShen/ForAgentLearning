# Week 1 学习报告 — Day 1 ~ Day 7

> 目标：理解 Chat LLM API 的工作原理，并亲手写出最小 Agent 循环。
> 实验代码：
> - Day 1：`week01_minimal_agent/invoke_llm_wo_tool.py`（无工具的原始 API 调用 + 手动多轮对话）
> - Day 2：`week01_minimal_agent/agent.py`（带工具调用的最小 Agent 循环，含详细注释）
> - Day 3：`week01_minimal_agent/agent.py`（新增 `read_file` 工具：路径安全检查 + 模糊 vs 精确 description 对比）
> - Day 4：`week01_minimal_agent/agent.py`（calculator 随机 50% 报错，对比空洞 vs 可操作的错误信息）
> - Day 5：`week01_minimal_agent/agent_v2.py`（凭记忆重写 agent 循环，加 `MAX_ITERATIONS` / `MAX_TOOL_RETRIES` 双层保险丝）
> - Day 6：`experiments/week01_tool_descriptions/run.py`（terse vs. detailed 工具描述的对照实验，2×7×5=70 次运行）
> - Day 7：`evals/run_evals.py` + `evals/tasks.json`（可复利的 eval harness，跑出第一条基线）

---

## Day 1 — Chat LLM API 是怎么工作的

### 1. API 是应用和 LLM 推理之间的桥梁

一次完整的调用流程分四步：

1. **应用 → API**：应用程序以 HTTP 请求的形式向 API 发送请求。发送前先把请求转换成 API 要求的数据格式（通常是 JSON），其中包含模型名称、实际提示词（prompt）和其他参数（temperature、max_tokens 等）。
2. **API → 模型**：API 收到请求后，转发给 LLM 进行推理。
3. **模型 → API**：模型生成响应（内容生成、问答、情感分析、摘要等），返回给 API。
4. **API → 应用**：API 把响应封装成 JSON 回传给应用程序。

主流的 API 提供商：Anthropic、OpenAI、Mistral（国内如阿里云 DashScope、DeepSeek 等大多兼容 OpenAI 格式）。

**接入方式**主要有三种：
- API 远程调用（本周使用的方式）
- 开源模型本地部署后调用
- SDK / 官方客户端库调用（本质上是对 HTTP API 的封装，如 `openai` Python 包）

### 2. messages array：对话的核心数据结构

`messages` 是一个**有序列表**，存储完整对话历史，让模型能理解上下文并生成连贯回复。每个元素是一条消息：

| 角色（role） | 含义 |
|---|---|
| `system` | 系统指令，设定模型的行为和人设 |
| `user` | 用户输入 |
| `assistant` | 模型的回复（也可能包含 `tool_calls`，见 Day 2） |
| `tool` | 工具执行结果，回传给模型（Day 2 才用到） |

`content` 是消息的具体内容：文本，或多模态数据（图片、工具调用结果等）。

### 3. API 是无状态的（stateless）

服务器**不保存**客户端的会话状态，所以每个请求必须携带完整的对话历史。

**补充 — 无状态的实际影响（Day 1 实验里亲手验证了这一点）：**
- "多轮对话"是客户端自己**拼出来**的：把上一轮的 assistant 回复 append 进 `messages`，再 append 新的 user 消息，整个列表重新发一遍。`invoke_llm_wo_tool.py` 的第 2 轮就是这么做的：

```python
messages.append({'role': 'assistant', 'content': reply_1})  # 上一轮模型的回答
messages.append({'role': 'user', 'content': '新的追问'})
completion = client.chat.completions.create(model=..., messages=messages)
```

- 代价：对话越长，每次请求的输入 token 越多，**费用和延迟随轮数增长**。这是后面（Week 2 Day 4 / Week 5）要学"上下文窗口管理"和"记忆压缩"的根本原因。
- 好处：服务端可水平扩展，客户端可以随意"篡改历史"（删掉某轮、改写某轮）来控制模型行为。

### 4. 常用参数（补充）

Day 1 实验里用到了两个采样参数，记录一下含义：

| 参数 | 含义 |
|---|---|
| `temperature` | 采样随机性。越高越发散（实验中用了 0.9），越低越确定；0 也不保证完全确定（Week 2 Day 5 会专门做实验） |
| `n` | 让模型对同一个 prompt 生成 n 个候选回答，都在 `choices` 列表里（`choices[0]`、`choices[1]`…）。实验中 `n=1`，所以只取 `choices[0]` |
| `max_tokens` | 输出长度上限，超过会被截断（对应 `finish_reason == "length"`，见 Day 2） |

### 5. 响应对象里的 `usage` 字段（补充）

完整的响应对象除了 `choices` 之外还有 `usage`，报告本次请求的 token 消耗：

- `prompt_tokens`：输入（整个 messages 数组）消耗的 token
- `completion_tokens`：模型输出消耗的 token
- `total_tokens`：两者之和

这是计费依据，也是观察"无状态导致历史越滚越大"的直接证据——多轮对话中 `prompt_tokens` 会逐轮上涨。Week 2 Day 3 会在 Agent 循环里加上 token/成本统计。

### 6. OpenAI SDK 的主要接口一览

| 接口 | 功能 | 典型用途 |
| --- | --- | --- |
| `client.chat.completions.create()` | 对话补全（最常用） | 聊天、问答、文本生成 |
| `client.completions.create()` | 文本补全（Legacy） | 续写、代码补全（旧版模型） |
| `client.embeddings.create()` | 获取文本向量（Embedding） | 语义搜索、RAG 检索、文本相似度计算 |
| `client.images.generate()` | 文生图 | 根据描述生成图片 |
| `client.audio.speech.create()` | 文本转语音（TTS） | 语音播报、有声书 |
| `client.audio.transcriptions.create()` | 语音转文字（Whisper） | 会议录音转录 |
| `client.files.create()` / `list()` | 文件上传/管理 | 微调训练数据上传 |
| `client.fine_tuning.jobs.create()` | 创建微调任务 | 定制专属模型 |
| `client.moderations.create()` | 内容审核 | 检测文本是否违规 |

---

## Day 2 — 工具调用与最小 Agent 循环

### 1. 核心认知：模型并不会"执行"工具

模型"调用工具"其实是：**模型输出一段 JSON（想调什么工具、传什么参数），然后停下来**。真正执行工具的是我的代码。执行完把结果塞回 `messages`，再发起下一次请求——模型看到结果后继续推理。

所以一个 Agent 的本质就是这个循环（`agent.py` 全部逻辑约 100 行）：

```
while 模型还想调工具:
    执行工具
    把结果喂回去
返回最终文本答案
```

LangChain、OpenAI Agents SDK 这些框架做的就是替你跑这个循环。手写一遍是为了理解它。

### 2. ChatCompletionMessage：模型返回的消息结构

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `content` | `str` / `None` | 模型生成的**文本内容**。如果模型只调工具不说话，这里可能是 `None` 或空字符串 |
| `refusal` | `str` / `None` | 模型**拒绝回答**的原因（如安全过滤）。`None` 表示正常回答 |
| `role` | `str` | 固定为 `"assistant"`（这是模型返回的消息） |
| `annotations` | `list` / `None` | 附加注释（如引用来源），目前较少使用 |
| `audio` | `object` / `None` | 语音输出相关（GPT-4o 语音模式），普通文本对话为 `None` |
| `function_call` | `object` / `None` | ⚠️ **旧版**函数调用格式（已废弃） |
| `tool_calls` | `list` / `None` | ⭐ **新版**工具调用列表，包含模型请求调用的所有工具 |

### 3. 工具定义 = Prompt Engineering（补充）

工具通过 JSON Schema 声明给模型（`agent.py` 的 `TOOLS`）：

```python
{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression. Call this for any math ...",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "..."}},
            "required": ["expression"],
        },
    },
}
```

关键点：`description` 决定的不只是"这个工具做什么"，更是"模型**什么时候**会选它"。写得含糊模型就会选错工具或不调工具——这是 Day 3 的实验主题。

### 4. 一次完整的循环迭代，`messages` 里发生了什么（补充）

以 `agent.py` 的问题 "What is 17 * 23 + 4? Also, what time is it right now?" 为例：

```
第 1 次请求前:  [system, user]
模型返回:      assistant 消息，content 为空，tool_calls 里有 2 个调用
                （calculator 和 get_current_time —— 并行工具调用）
执行工具后:    [system, user, assistant(tool_calls), tool(结果1), tool(结果2)]
第 2 次请求:   模型看到两个结果，返回纯文本 assistant 消息，无 tool_calls
循环结束:      返回 msg.content
```

三个容易踩坑的细节：

1. **assistant 消息（含 tool_calls）必须原样 append 回历史**，否则下一次请求模型不知道自己刚才调了什么工具，`tool` 消息会变成"没头的回复"直接报错。
2. **`tool_call_id` 必须一一对应**：一条 assistant 消息可以包含多个 `tool_calls`（并行调用），每个都要用独立的 `{"role": "tool", "tool_call_id": call.id, "content": 结果}` 回应，缺一个都会报错。
3. **`arguments` 是 JSON 字符串，不是 dict**，必须 `json.loads()`，而且要防备模型输出非法 JSON（`agent.py` 里用 try/except 兜底）。

### 5. `finish_reason`：模型为什么停下来（补充）

每个 `choice` 都带 `finish_reason`，是循环的分支依据：

| 值 | 含义 | 循环里的处理 |
|---|---|---|
| `stop` | 正常说完了 | 返回 `msg.content`，循环结束 |
| `tool_calls` | 模型想调工具 | 执行工具、回填结果、继续循环 |
| `length` | 达到 `max_tokens` 被截断 | 内容不完整，需要加大 max_tokens |
| `content_filter` | 触发内容过滤 | 检查 `refusal` |

（`agent.py` 里实际用 `msg.tool_calls` 是否为空来判断是否结束，并单独处理了 `length`。）

### 6. 错误处理：把错误"还给"模型（补充）

工具执行失败时**不要让程序崩溃**，而是把错误信息作为字符串放进 `tool` 消息回传：

```python
output = f"Error: {e}"
messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
```

OpenAI 格式没有 `is_error` 标志，这是约定俗成的做法。模型看到错误信息后会自行调整（换参数重试、换工具、或向用户解释）。错误信息越具体，模型恢复得越好——这是 Day 4 实验要验证的。

---

## Day 3 — JSON Schema：工具定义的语言

### 1. JSON 只管"表示"，JSON Schema 才管"约束"

JSON 是一种数据表示格式，只负责表示数据，不负责校验字段是否完整、类型是否合法。**JSON Schema 是用 JSON 写的一份"JSON 规则文档"**：描述哪些字段必须有、每个字段的类型、数值范围、字符串长度、数组元素的结构等。

它在接口参数校验、配置文件校验中本来就是通用标准；对 Agent 来说更关键——**tool 定义本质上就是一份 JSON Schema**，模型靠它理解"什么时候调用什么工具、需要传什么参数"。没有这份 Schema，模型就无法把自然语言的意图变成程序可解析的结构化数据，也就无法精确、可验证地调用工具。

以 `agent.py` 里的 calculator 为例，逐字段标注：

```python
{
    "type": "function",            # 声明这是一个可调用函数
    "function": {
        "name": "calculator",      # 工具名称（模型用它来指名调用）
        "description": (           # 决定模型"何时"调用这个工具
            "Evaluate an arithmetic expression. Call this for any math instead "
            "of computing it yourself, e.g. '17 * 23 + 4' or '2 ** 10'."
        ),
        "parameters": {            # 参数的 JSON Schema
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python arithmetic expression, digits and + - * / ** ( ) only",
                }
            },
            "required": ["expression"],  # 必填参数列表
        },
    },
},
```

### 2. 常用 Schema 关键字一览

| 参数 | 类型 | 作用 | 示例 |
| --- | --- | --- | --- |
| `type` | `string` | 定义数据类型：`string`/`integer`/`number`/`boolean`/`array`/`object`/`null` | `"type": "string"` |
| `description` | `string` | 对字段的文字说明，**提示模型如何填写** | `"description": "城市名称"` |
| `enum` | `array` | 枚举值，参数**只能从列表中选择** | `"enum": ["celsius", "fahrenheit"]` |
| `required` | `array` | 必填字段列表 | `"required": ["city", "date"]` |
| `properties` | `object` | 定义对象内各字段的 Schema | 见上方示例 |
| `default` | 任意 | 默认值，模型未提供时使用 | `"default": "celsius"` |
| `minimum` / `maximum` | `number` | 数值最小/最大值 | `"minimum": 1, "maximum": 100` |
| `minLength` / `maxLength` | `integer` | 字符串最小/最大长度 | `"maxLength": 50` |
| `pattern` | `string` | 正则表达式约束字符串格式 | `"pattern": "^[0-9]{6}$"` |
| `items` | `object` | 定义数组元素的 Schema | `"items": {"type": "string"}` |
| `minItems` / `maxItems` | `integer` | 数组最小/最大元素个数 | `"maxItems": 5` |
| `format` | `string` | 预定义格式：`email`/`date`/`uri`/`uuid` 等 | `"format": "email"` |
| `example` | 任意 | 给出一个示例值（辅助理解，不强制） | `"example": "Beijing"` |

### 3. enum 和 description：两种不同性质的"控制"手段

- **`enum` 是硬约束**：用列表"强制"模型只能从枚举值中选，确保程序稳定运行、行为不越界，是保证安全性的手段。凡是取值可穷举的参数（单位、模式、类别），都应该用 enum 而不是靠 description 描述。
- **`description` 是软引导**：引导模型"要不要调用工具"的决策逻辑，并通过示例约束模型给出合理的参数。它承担三种作用：

| 作用 | 说明 | 示例 |
| --- | --- | --- |
| **触发判断** | 模型根据描述判断当前场景是否需要此工具 | `"Call this for any math instead of computing it yourself"` |
| **使用指导** | 告诉模型正确的使用方式 | `"City name in English, e.g. 'Beijing' not '北京'"` |
| **边界限定** | 明确工具的适用范围，防止误用 | `"Only for current weather, not forecast"` |

写好 description 和写好 System Prompt 一样重要，都是让模型"听话"的关键手段。

### 4. 哪些约束是"真的"，哪些只是"提示"（补充）

一个容易误解的点：**tool 的 parameters Schema 默认只是作为文本给模型"读"的，API 并不会替你校验模型的输出是否符合 Schema**。也就是说 `minimum`、`pattern`、`maxLength` 这些关键字，模型会尽量参考，但不保证遵守——模型完全可能传出一个超出 `maximum` 的数或不匹配 `pattern` 的字符串。

- OpenAI 提供了 `strict: true`（Structured Outputs）：用受限解码从生成层面**保证**输出符合 Schema，但只支持 Schema 的一个子集（如 `type`/`enum`/`required`，且要求所有字段 required、禁止额外字段），`minimum`、`pattern`、`default` 等仍不在保证范围内。
- 所以原则是：**Schema 用来引导，代码用来兜底**。所有涉及安全或正确性的校验（比如下面 `read_file` 的路径检查），必须在工具实现里用代码再做一遍，不能指望模型"自觉"。这和 Day 2 的"把错误还给模型"是一套组合拳：代码校验拦截非法输入 → 抛出具体的错误信息 → 模型看到后自行纠正。

### 5. 实验：给 `agent.py` 加 `read_file` 工具（已完成，详见 agent.py）

工具实现里做了**四层路径安全检查**（对应上一节"代码兜底"）：

1. `os.path.normpath()` 规范化路径；
2. 禁止路径穿越（`..`）和绝对路径（`/` 或 `\` 开头）；
3. 扩展名白名单（`.txt` `.md` `.py` `.json` `.csv`）；
4. 拼接后用 `os.path.abspath()` 确认最终路径仍在 workspace 目录内。

description 写了两个版本做对比：

- **版本 A（模糊）**：`"Read a file."`
- **版本 B（精确）**：说明何时调用、路径必须是相对路径、允许的扩展名、禁止二进制文件、出错会返回错误信息。

**实验发现（两个都和"路径"有关）：**

1. **大小写敏感**：如果 description 里不强调 "the path string is case-sensitive"，模型会凭猜测使用错误的大小写（如 `plan.md` vs `PLAN.md`），导致找不到文件。
2. **workspace 的设定和路径拼接方式有决定性影响**：模型并不知道我的目录结构，它给出的相对路径最终能不能拼接成正确的绝对路径，取决于代码里 workspace 根目录怎么设、`os.path.join()` 怎么拼。这两件事没对齐时，工具调用必然失败。

**补充 — 由此引出的工程改进方向：**

- **别让模型"猜"路径**：配一个 `list_files` / `glob` 类工具，让模型先探测目录结构再读文件。Claude Code、Cursor 这类编码 Agent 都是这么做的（LS/Glob/Grep 先行，Read 在后）。
- **错误信息要能帮模型自救**：`FileNotFoundError` 时附上目录下的相似文件名（"did you mean PLAN.md?"），模型下一轮往往能直接纠正——这正好衔接 Day 4 的实验主题。
- **当前实现的一个边界漏洞**：`full_path.startswith(workspace)` 会把 `/path/ForLearning-evil/...` 也误判为合法（前缀恰好匹配）。更稳妥的写法是 `os.path.commonpath([workspace, full_path]) == workspace`，且先用 `os.path.realpath()` 解析符号链接。

### 6. Day 3 一句话总结

Schema 定义了模型和代码之间的"合同"：`enum`/`required` 划定硬边界，`description` 引导软决策，而真正的强制执行永远在工具实现的代码里。

## Day 4 — 错误处理：错误是模型的反馈信号

### 1. 错误的四种来源

带工具的 Agent 里，错误可能发生在循环的不同环节，处理位置也不同：

| 类型 | 典型场景 | 在哪里捕获 |
|---|---|---|
| **参数解析错误** | 模型生成畸形 JSON，`json.loads()` 失败 | 循环代码里 try/except，把解析错误作为 tool 结果回传 |
| **工具执行错误** | 参数合法但运行失败：文件不存在、网络超时、除零 | 工具实现里 try/except，转成错误字符串回传 |
| **模型决策错误** | 选错工具、误解参数含义、猜错文件名 | 无法直接"捕获"——靠工具返回的错误信息让模型自己发现并纠正 |
| **系统级错误** | API 限流（429）、模型不可用（5xx）、token 超限 | 发生在"循环之外"，模型根本没机会看到，只能由代码处理（见第 6 节） |

前三类的共同点：**它们发生在循环之内，模型有能力参与恢复**；第四类必须由代码兜底。

### 2. 三种处理策略：为什么选"返回给模型"

工具执行失败时有三种选择：

| 策略 | 结果 | 问题 |
|---|---|---|
| 直接崩溃（异常向上抛） | 整个 Agent 挂掉，对话终止 | 一次瞬时故障毁掉全部已完成的工作 |
| 把错误直接展示给用户 | 用户看到一堆 traceback | 用户体验差，且问题没有被解决——把责任推给了用户 |
| **把错误返回给模型（⭐推荐）** | 模型分析错误、调整策略、重试或换方案 | 需要写好错误信息（见实验三） |

选第三种的根本原因：**错误信息对模型来说是反馈（feedback）**。模型是循环里的"决策者"，它具有推理能力——给它错误信息，就像给人类程序员看报错日志：它能读懂、分析、然后调整策略。一次典型的自我修复流程：

```
tool 结果: "Error: File not found: notes.txt"
    ↓ 模型看到错误，自动调整：
"文件不存在，让我先看看目录里有什么……"
    → 调用 list_files() 探测目录
    → 或换一个相近的文件名重试
    → 或向用户确认
结果：Agent 自我修复，对话继续
```

这就是 Day 2 第 6 节"把错误还给模型"的展开：工程上只需要一行约定——

```python
output = f"Error: {e}"
messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
```

### 3. 设计原则

| 原则 | 说明 |
|---|---|
| **永不崩溃** | 工具内所有异常都捕获，转为字符串返回，循环不中断 |
| **错误即反馈** | 把错误视为模型的"学习信号"，而不是程序的失败 |
| **保持对话连续性** | 错误消息以 `role="tool"` 加入历史，上下文不丢失 |
| **给模型修复机会** | 提供备用工具（如 `list_files`），让模型能"自救" |
| **限制重试次数** | 用 `max_turns` 兜底，防止模型陷入"报错→重试→报错"的死循环 |

**补充 — 关于 `is_error` 标志**：OpenAI 格式里没有专门的错误字段，`"Error: ..."` 前缀纯属约定俗成；Anthropic 的 API 则在 `tool_result` 上有显式的 `is_error: true` 字段。两者效果类似——本质都是让模型在上下文里"看见"失败。

### 4. 实验设置与三个场景

实验代码：`agent.py`（Day 3 版本基础上，让 calculator 以 50% 概率随机报错）。三个场景分别验证错误处理的三道防线。

#### 实验一：Schema 是第一道防线——错误在调用前就被拦截

> User: Read the file `/PLAN.mdd` and tell me what this script doing. Also, what time is it right now?

`/PLAN.mdd` 有两处非法：绝对路径开头的 `/`，和白名单外的 `.mdd` 扩展名。由于 `read_file` 的 description 里明确写了这两条约束，模型**根本没有发起工具调用**，直接向用户解释：

> I am unable to read `/PLAN.mdd` for two reasons: **Path format** — I can only read files using relative paths; **File extension** — `.mdd` is not supported. If you meant **`PLAN.md`**, please let me know!

（时间问题照常调用 `get_current_time` 回答了。）

**结论**：Day 3 的"精确 description"在这里发挥了第二层价值——它不只引导模型正确调用，还能让明显非法的调用**在发出之前就被过滤掉**，省下一轮注定失败的往返。这是最便宜的错误处理：错误没发生。

#### 实验二：文件名打错——模型靠错误信息"自救"

> User: Read the file `PLANN.md` and tell me what this script doing. Also, what time is it right now?

执行轨迹：

1. 模型调用 `read_file("PLANN.md")` → 工具返回 "Error: File not found"；
2. 模型的推理（reasoning_content）：*"The file 'PLANN.md' was not found. It's highly likely the user meant 'PLAN.md' (a common filename). Let me try reading 'PLAN.md' instead."*
3. 第二次调用 `read_file("PLAN.md")` → 成功；
4. 最终回答里模型还主动向用户说明了这个纠正：*"I couldn't find a file named exactly `PLANN.md`, but I did find **`PLAN.md`**."*

**结论**：只要错误信息进入上下文，模型确实会推理出替代方案并自动重试——这正是"错误即反馈"原则的直接验证。而且它对用户透明地汇报了自己的纠错过程，这是好的 Agent 行为。

#### 实验三：错误信息的质量决定模型能否恢复

> question = "What is 17 * 23 + 4? Also, what time is it right now?"（calculator 50% 概率报错）

**版本 A — 空洞的错误信息 `"Error: "`（没有任何细节）：**

- 第一次报错后，模型的推理明显**陷入困惑**：*"looking at the result, it seems like the tool returned the time instead... Actually, looking more carefully... Wait, actually looking at the results again..."*——因为错误信息没内容，模型只能反复猜测发生了什么。
- 重试第二次仍然报错后，模型嘴上说 "Let me retry"，实际行为却是**放弃工具、自己心算**：*"The calculator is returning an error. Let me just compute it myself: 17 * 23 = 391, 391 + 4 = 395"*，然后直接把 395 报给用户。

**版本 B — 可操作的错误信息 `"Error: other unknown error, just retry again"`：**

- 模型看到后立刻发起一次干净的重试：*"The calculator returned an error. Let me retry the calculation."*
- 重试成功返回 `395`，模型正常汇报结果。整个恢复过程只多了一轮，没有任何困惑推理。

**两个重要观察：**

1. **好的错误信息 = 更短的恢复路径**。版本 A 里模型浪费大量推理在"猜错误原因"上；版本 B 里错误信息直接告诉它"可以重试"，模型一步到位。错误信息越具体、越可操作，模型恢复得越快——这验证了 Day 2 留下的假设。
2. **"自己算"是一个隐蔽的危险信号**。版本 A 里模型放弃工具、心算出了正确答案——这次运气好，因为算术题模型本来就会。但如果换成 `read_file` 这类**结果无法凭空推导**的工具，同样的行为就是幻觉：模型可能"编造"一个文件内容来完成回答。所以工具的错误信息必须明确告诉模型"该怎么办"（重试/换参数/放弃并告知用户），**不能留下让模型自由发挥的空间**。

### 5. 补充 — 好的工具错误信息长什么样

从实验三反推，一条帮模型自救的错误信息应包含三要素，和好的编译器报错是同一套标准：

1. **什么失败了**：`File not found: 'PLANN.md'`（带上实际收到的参数，方便模型对照自己的调用）；
2. **为什么 / 属于哪类**：是参数问题（重试同样参数没用）还是瞬时故障（重试可能就好）——这决定模型下一步该"换方案"还是"再试一次"；
3. **建议的下一步**：`Did you mean 'PLAN.md'? Available files: PLAN.md, notes.txt` 或 `This is a transient error, retrying may succeed`。

第 2 点尤其关键：**可重试（transient）和不可重试（permanent）的错误要区分开**。对确定性工具用相同参数重试是纯浪费；而版本 A 的 `"Error: "` 连这个最基本的信号都没给，模型只能瞎猜。

### 6. 补充 — 哪些错误不该喂给模型

系统级错误（API 限流 429、服务端 5xx、网络断开）发生在**发起请求这一步**，此时根本没有"模型的回复"可言，所以"返回给模型"这条路不存在。它们只能在循环代码里处理：

- **限流 / 瞬时故障**：指数退避重试（exponential backoff），SDK 通常内置（OpenAI SDK 默认自动重试 2 次）；
- **token 超限**：说明历史太长，需要截断或压缩上下文——这正是 Day 1 "无状态导致历史越滚越大"的下游问题，Week 5 的主题；
- **持续不可用**：达到重试上限后，向用户如实报告并保存已有进度，而不是无限等待。

一句话区分：**循环之内的错误交给模型，循环本身的错误交给代码。**

### 7. Day 4 一句话总结

错误处理的三道防线：精确的 Schema 让错误不发生（实验一），进入上下文的错误信息让模型自我修复（实验二），而错误信息的质量——具体、可操作、区分是否可重试——决定修复是一步到位还是退化成幻觉（实验三）。


## Day 5 — ReAct：推理与行动的协同（论文精读 + 凭记忆重写 agent 循环）

> 论文：*ReAct: Synergizing Reasoning and Acting in Language Models*（Yao et al., ICLR 2023）
> 实验代码：`week01_minimal_agent/agent_v2.py`（凭记忆重写的 agent 循环，新增 `MAX_ITERATIONS` 和 `MAX_TOOL_RETRIES` 两道保险丝）

### 1. 论文核心思想：让"想"和"做"交错进行

ReAct 让 LLM 以**交错（interleaved）**的方式，同时生成推理轨迹（reasoning traces）和任务动作（actions），实现双向协同。动机来自人类解决问题的方式——思考和行动本来就是结合在一起的：

- **Reason to Act（为了行动而思考）**：推理帮助模型分解任务、制定和更新行动计划、处理异常情况——先在脑子里规划，再去执行。
- **Act to Reason（为了思考而行动）**：行动让模型与外部资源（知识库、环境）交互，获取新信息来支撑下一步推理——就像人会去查字典、看地图。

### 2. 为什么两者缺一不可（补充 — 论文的对照实验）

论文的说服力来自消融对比，这解释了"为什么 Agent 必须同时具备推理和工具"：

| 模式 | 缺陷 | 对应的失败形态 |
|---|---|---|
| **只推理（CoT）** | 与外部世界隔绝，全靠参数里的记忆答题 | **幻觉**：编造事实，且错误在推理链里传播、无法被纠正 |
| **只行动（Act-only）** | 缺少规划和反思，只会机械地连续执行动作 | 不会分解目标、不会从失败中调整策略，容易卡死 |
| **ReAct（推理 + 行动）** | — | 推理规划行动，行动的反馈修正推理 |

论文在知识密集型任务（HotpotQA 多跳问答、FEVER 事实核查，配一个 Wikipedia 搜索 API）和决策任务（ALFWorld、WebShop 模拟环境）上验证：ReAct 显著**减少了 CoT 的幻觉**（答案有外部证据可查），且推理轨迹人类可读、可诊断。论文还发现 ReAct 和 CoT **结合**（ReAct 失败时回退到 CoT，或反过来）效果最好——工具不可用时模型"自己想"仍是有价值的兜底，只是要警惕 Day 4 实验三发现的幻觉风险。

### 3. (Thought, Action, Observation)：循环的核心三元组

ReAct 的运行流程是一个循环，每轮迭代产生一个三元组，**整个序列全部留在上下文里**，成为下一轮推理的输入：

| ReAct 概念 | 谁生成的 | 在我的 agent 里对应什么 |
|---|---|---|
| **Thought（思考）** | 模型 | `reasoning_content`（推理模型的思考轨迹）或 assistant 消息里的分析文本 |
| **Action（行动）** | 模型 | `msg.tool_calls`（结构化的工具调用请求） |
| **Observation（观察）** | **环境（不是模型！）** | `{"role": "tool", "content": 结果}` 消息 |

Observation 这一步必须强调：它是环境（我的代码执行工具后）"喂"给模型的，模型只能被动接收——这正是打破模型封闭性、引入外部真实信息的入口。Day 4 的"错误即反馈"其实就是一种特殊的 Observation。

以 "What is 17 * 23 + 4? Also, what time is it right now?" 为例，一轮完整的三元组：

```
Thought:      （reasoning_content）"用户有两个请求：算 17*23+4，查当前时间。
               分别用 calculator 和 get_current_time。两个调用相互独立，
               可以在同一轮并行发出。"
Action:       calculator(expression="17 * 23 + 4")
              get_current_time()          ← 一条 assistant 消息里的 2 个 tool_calls
Observation:  tool: "395"
              tool: "2026-07-16 18:49:56"  ← 环境（我的代码）执行后回填
→ 下一轮:      模型看到两个 Observation，输出最终答案：
              "17 * 23 + 4 is 395. The current time is 18:49:56 on July 16, 2026."
```

### 4. 实验：依赖关系决定循环的轮数

对比了两类问题，验证模型如何组织 Thought-Action-Observation 循环：

**场景一（无依赖 → 并行）**：上面的例子，两个工具调用互不依赖，模型在**一轮**里同时发出 2 个 tool_calls，一次 Observation 后直接给出答案。

**场景二（有依赖 → 串行）**：

> User: what time is it? and extract the hour number, then plus 10 then add 4.

模型分成**两轮**调用，而不是同时发出：

```
第 1 轮 Action:       get_current_time()
第 1 轮 Observation:  "2026-07-16 22:xx:xx"
第 2 轮 Action:       calculator(expression="22 + 10 + 4")   ← 参数里的 22 来自上一轮的观察！
第 2 轮 Observation:  "36"
→ 最终答案
```

**结论**：calculator 的参数依赖 `get_current_time` 的结果，模型**必须等到 Observation 回来才能构造下一个 Action**——信息依赖强制循环退化为逐步的 Thought→Action→Observation 模式。模型能自己识别依赖关系并规划调用顺序，这正是 Thought 环节"制定行动方案"能力的体现。

**补充 — 并行工具调用是现代 API 对 ReAct 的扩展**：原始 ReAct 每轮只产生**一个** Action（严格串行）。现在 API 允许一条 assistant 消息带多个 tool_calls，本质是对"无依赖动作"的批量优化，省下多轮往返的 token 和延迟。两个场景合起来看：模型实际在做一个隐式的 DAG 调度——无依赖的并行，有依赖的串行。

### 5. 重写 `agent_v2.py`：给循环装上两道保险丝

凭记忆重写 agent 循环时，针对"工具持续不可用导致 Agent 空转烧 token"的问题加了两层防护：

| 机制 | 类型 | 作用 |
|---|---|---|
| `MAX_TOOL_RETRIES = 3` | **软信号** | 按工具维护"连续失败"计数（成功一次清零）。达到上限后在错误信息里**明确告诉模型**："这个工具已连续失败 N 次，不要再调了，用已有知识回答或告知用户工具不可用" |
| `MAX_ITERATIONS = 10` | **硬兜底** | for 循环硬上限。就算模型不听劝继续调，循环也会强制终止并返回放弃信息 |

两层设计的思路和 Day 3 的结论一脉相承：**软引导（把建议写进模型能看到的文本）+ 硬约束（代码强制执行），不能指望模型"自觉"**。软信号给模型体面退出的机会（它可以向用户解释），硬兜底保证最坏情况下也不会无限烧钱。

**补充 — 这恰好对应论文承认的局限**：ReAct 最常见的失败模式就是**推理轨迹陷入重复循环**——模型反复生成相同的 Thought 和 Action 出不来。论文是在提示层面缓解，工程上的标准答案就是 `max_iterations` 这类外部熔断。另外论文还指出 ReAct 的表现**高度依赖 Observation 的信息量**（检索到没用的信息会直接把推理带偏）——这和 Day 4 "错误信息质量决定恢复路径"是同一个道理的两面：喂给模型的观察越有信息量，循环收敛越快。

### 6. 补充 — 从提示词技巧到原生能力：ReAct 就是今天的 function calling

一个把整周内容串起来的认知：**2022 年的 ReAct 是纯提示词工程**——靠 few-shot 示例让模型输出 `Thought: ... / Action: search[...]` 格式的**纯文本**，再用代码正则解析出动作去执行。而今天：

- Action 变成了结构化的 `tool_calls` 字段（模型经过专门训练输出它，不再靠解析文本，也不怕格式跑偏）；
- Thought 变成了原生的 `reasoning_content` / 思考模式；
- Observation 变成了标准的 `role="tool"` 消息。

所以 **Day 2 手写的那个 while 循环，本质就是 ReAct 的现代实现**——论文的思想被 API 产品化了。理解了这个映射，LangChain 的 `create_react_agent`、各家框架的 AgentExecutor 就都不神秘了：它们跑的都是同一个循环。

### 7. Day 5 一句话总结

ReAct 用 (Thought, Action, Observation) 三元组的交错循环把"推理防幻觉、行动取真据"结合起来；这个 2022 年的提示词技巧如今已产品化为 function calling，而它的经典失败模式（循环重复、空转）要靠 `agent_v2.py` 里"软信号劝退 + 硬上限熔断"的双层保险丝来兜底。

---

## Day 6 — 对照实验：terse vs. detailed 工具描述

> 实验代码：`experiments/week01_tool_descriptions/run.py`
> 原始数据：`experiments/week01_tool_descriptions/results_20260717_152833.json`
> 规模：2 conditions × 7 tasks × 5 runs = 70 次运行，模型 qwen3.7-plus

### 1. 实验设计：控制变量 + 陷阱任务

这是本周第一个"有对照组、有量化指标"的实验，设计上有三个刻意的选择：

1. **单一变量**：两个 condition 的工具实现、参数 Schema 完全相同，唯一区别是 description 文本——terse 版是一句话敷衍（`"Do math."` / `"Get time."` / `"Read a file."`），detailed 版是 Day 3 打磨过的精确描述（何时调用、参数格式、边界限制）。
2. **任务集分两类**：5 个 should-call 任务（单工具×3、并行、串行依赖）+ 2 个 **should-NOT-call 陷阱任务**（常识问题 `trap_no_tool`、二进制文件 `trap_binary_file`）。陷阱任务是专门用来放大"description 缺边界"的后果的——只测该调的场景，看不出差距。
3. **每格跑 5 次**：Day 4 已经见过采样随机性，单次运行的结果不可信。

### 2. 三个指标（程序化打分，不靠人眼）

| 指标 | 定义 | 衡量 description 的哪种引导 |
|---|---|---|
| **suitable** | 实际调用的工具集合 == 任务期望的集合（不多、不少、不错） | "**何时**调用"——触发判断 |
| **success** | 发出的所有调用都执行成功；没发出调用记为成功（真空成立） | "**如何**调用"——参数质量 |
| **wrong-tool** | 是否调用了不该调的工具 | 过度触发 |

打分逻辑是代码（集合相等判断），不是人眼评判——**评分自动化是实验能重复跑的前提**，这个思路直接延续到 Day 7 的 harness 和 Week 4 的评测周。

### 3. 结果

```
task               | condition | suitable | success | wrong-tool
-----------------------------------------------------------------
math_basic         | terse     |   5/5    |   5/5   |   0/5
math_basic         | detailed  |   5/5    |   5/5   |   0/5
time_basic         | terse     |   5/5    |   5/5   |   0/5
time_basic         | detailed  |   5/5    |   5/5   |   0/5
file_basic         | terse     |   5/5    |   5/5   |   0/5
file_basic         | detailed  |   5/5    |   5/5   |   0/5
multi_parallel     | terse     |   5/5    |   5/5   |   0/5
multi_parallel     | detailed  |   5/5    |   5/5   |   0/5
multi_sequential   | terse     |   5/5    |   5/5   |   0/5
multi_sequential   | detailed  |   4/5    |   5/5   |   0/5
trap_no_tool       | terse     |   5/5    |   5/5   |   0/5
trap_no_tool       | detailed  |   5/5    |   5/5   |   0/5
trap_binary_file   | terse     |   1/5    |   1/5   |   4/5   ← 分水岭
trap_binary_file   | detailed  |   5/5    |   5/5   |   0/5
-----------------------------------------------------------------
TOTAL              | terse     |  31/35   |  31/35  |   4/35
TOTAL              | detailed  |  34/35   |  35/35  |   0/35
```

### 4. 发现

**发现一：简单任务上两者毫无差别——description 的价值不在中心，而在边界。**
6/7 的任务里 terse 和 detailed 表现完全一样。当任务和工具的映射显而易见时（算术→calculator、"现在几点"→get_current_time），`"Do math."` 就够了。写 description 的功夫应该花在**边界情况**上：什么时候不该调、什么参数是非法的。

**发现二（核心）：`trap_binary_file` 是分水岭——detailed 的边界声明起到了"调用前过滤"的作用。**

> 任务："Open the file photo.jpg and describe what is in the picture."

- **terse**：5 次里 4 次直接发起 `read_file("photo.jpg")`，几乎全军覆没；
- **detailed**：5 次全部**没有发起调用**，直接向用户解释无法读图片——description 里那句 `"DO NOT call this for binary files (.jpg, .pdf, .exe, etc.)"` 生效了。这和 Day 4 实验一（`/PLAN.mdd` 被模型在调用前过滤掉）是同一机制：**最便宜的错误处理是让错误不发生**。

**发现三（细看原始数据才发现的）：terse 那 4 次失败里，最终答案其实都是对的——但这恰恰是危险所在。**
翻 `results_*.json` 里的明细：4 次失败的轨迹都是"发起 `read_file("photo.jpg")` → 被扩展名白名单的代码校验拦下报错 → 模型看到错误后向用户正确解释了自己只能读文本文件"。也就是说：

- **最终答案没错，但危险的调用已经发出了**——拦住它的是 Day 3 写的代码校验（第三道防线），不是模型的判断；
- 这次的代价只是浪费一轮往返。但在生产环境里，如果这个工具有**副作用**（写文件、删数据、执行程序、发请求），"调了之后靠报错兜底"就可能是真实伤害——很多操作不存在"报错回滚"；
- 所以结论不是"detailed description 万事大吉"，而是**防御必须纵深**：description 挡第一层（让多数非法调用不发生），参数解析/代码校验挡最后一层（保证漏网的调用无法造成伤害）。两层各司其职，缺一不可。

**发现四：detailed 唯一的一次 suitable 失败，是 Day 4 的"自己算"信号在自然条件下复现。**
`multi_sequential` run 4：模型调了 `get_current_time` 拿到 15 点后，**心算** 15+10+4=29 直接回答，没调 calculator。注意 detailed 的 calculator description 明明写了 `"Call this for any math instead of computing it yourself"`——模型仍有小概率抄近路。这印证了 Day 3 的结论：**description 是软引导，永远做不到 100%**。这次心算恰好算对了，但机制上它和 Day 4 实验三里"放弃工具自己算"是同一种行为，换个模型不会的任务就是幻觉。

**发现五：指标之间会"说不同的话"——结果正确 ≠ 过程合规。**
上面那次 run 的 success 是通过的（发出的调用都成功）、最终答案 29 也是对的，但 suitable 不合格。三个指标刻意分开，就是为了区分"答案对不对"和"过程对不对"。Day 7 的 eval 只检查最终答案（结果），Day 6 的 harness 检查调用轨迹（过程）——两者互补，Week 4 会把过程断言也纳入正式评测。

### 5. 实验的局限（第一次写实验报告时漏掉的，补上）

1. **n=5 太小，没有统计意义**。`multi_sequential` 的 4/5 vs 5/5 完全可能是噪声，不能下结论；只有 `trap_binary_file` 的 1/5 vs 5/5 这种悬殊差距才值得相信。以后写实验结论前先问一句：这个差距重跑一遍还在吗？
2. **只测了一个模型**（qwen3.7-plus）。description 敏感度是模型相关的——更强的模型可能对 terse 更鲁棒，换模型必须重跑，不能把本实验的数字当通用规律。
3. **陷阱任务只有 2 个**。wrong-tool 率的真实水平需要更多边界任务（越权路径、危险参数、语义相近但不该调的场景）支撑，Week 3 做真实工具时扩充。

### 6. Day 6 一句话总结

description 在简单任务上是锦上添花，在边界任务上是第一道安全防线；但 4/5 vs 0/5 证明它有效的同时，multi_sequential 那 1 次心算也证明它不可靠——软引导（description）挡第一层、硬校验（代码）挡最后一层的纵深防御，才是完整答案。

---

## Day 7 — 基线评测：可复利的 eval harness

> 实验代码：`evals/run_evals.py`；任务集：`evals/tasks.json`；结果：`evals/last_results.json`

### 1. harness 的设计思想：evals 是复利资产

`run_evals.py` 的核心约定只有两条，但决定了它长期的价值：

1. **任务只增不删**。每周把新能力对应的任务加进 `tasks.json`，永不删除旧任务。到 Week 12 时，agent 会自动被 Week 3 的任务回归测试——**回归以"失败行"的形式自动浮现**，而不是靠记忆和运气。
2. **评分必须自动化**。目前用最简单的 `expected_substring in answer` 子串匹配。checker 可以换更强的，但绝不允许退回"人眼看看差不多"——人眼评测无法每周重复、无法比较，也就无法发现回归。

这两条合起来，就是把"本周的 agent 有没有变好"从主观感觉变成可查询事实的机制。

### 2. 扩充任务集：给每个工具和调用模式都配上 eval 任务

初版 `tasks.json` 只有 3 个任务（calculator×2 + 不调工具），覆盖太窄。为了让任务集撑得起"多工具"场景，先给 `agent.py` 加了两个更复杂的工具：

- **`search_files`**：按文件名 glob + 可选内容关键词搜索工作区（复用 read_file 的安全约束）。它和 read_file 形成天然工具链——这正是 Day 3 结论里说的"别让模型猜路径，给它探测工具"；
- **`query_orders`**：模拟订单数据库，带枚举（status）、数值范围、排序、limit 等参数。数据故意小而固定（8 条订单），所有聚合答案都可手算验证。它**故意不做求和**——逼模型走"查询 → calculator 计算"的串行链。

然后把任务集扩到 8 个，每个任务的期望子串都是确定性的、事先手算过的：

| 任务 | 覆盖的能力 | 期望子串 |
|---|---|---|
| `math-basic` / `math-nested` | calculator 单工具 | `395` / `101` |
| `no-tool-needed` | 不该调工具（防"逢题必调"退化） | `Paris` |
| `read-file-basic` | read_file 单工具 | `openai` |
| `search-then-locate` | search_files（glob + 内容关键词） | `agent.py` |
| `orders-revenue-sequential` | query_orders → calculator 串行链 | `730.65` |
| `orders-filter-sort` | query_orders 多条件过滤 + 排序 | `O-1005` |
| `orders-compare-multitool` | 两次 query_orders + calculator 对比 | `595.25` |

写任务时的两个技巧（都是在绕 substring checker 的弱点）：prompt 里明确要求"reply with the exact total amount / the order id"，让可校验的值近乎必然出现在答案里；`search-then-locate` 特意写明 "python file"，避免内容搜索命中 `tasks.json` 自己（它现在也包含 `MOCK_ORDERS` 这个词）。

### 3. 基线结果：40/40（2026-07-17，qwen3.7-plus，每任务 5 次）

```
math-basic                5/5  ✅
math-nested               5/5  ✅
no-tool-needed            5/5  ✅
read-file-basic           5/5  ✅
search-then-locate        5/5  ✅
orders-revenue-sequential 5/5  ✅
orders-filter-sort        5/5  ✅
orders-compare-multitool  5/5  ✅
--------------------------------
Total: 40/40
```

`last_results.json` 现在带元数据信封（`model` / `date` / `runs_per_task` / `total_passed`），未来的基线可以直接对比。

翻明细记录还有两个值得记的观察：

1. **零方差**。8×5 全过，连答案措辞都高度稳定（`orders-revenue-sequential` 5 次里 3 次一字不差）。这说明当前任务集对这个模型来说**太简单了，没有区分度**——一个从不失败的 eval 探测不到退化的边缘。Week 2 的 temperature 实验和 Week 4 的"找出最 flaky 的任务"需要更难的任务才有东西可测。
2. **模型把工具输出的结构信息用起来了**。`search-then-locate` 的 5 次回答全都带上了 "line 312"——这来自 search_files 返回的 `match_line` 字段。工具返回结构化 JSON（而不是裸文本）的价值在这里直接可见：模型能引用、能定位，回答质量更高。这是 Week 3 Day 5 "tool-result design" 的预告。

### 4. substring checker 的已知弱点（现在就记下来，Week 4 逐个解决）

1. **只查结果，不查过程**。`math-basic` 就算模型没调 calculator、纯心算出 395 也算 pass——而 Day 6 发现四和 Day 4 实验三都证明"心算/编造"是需要被检测的行为。Day 6 的 harness 恰好相反（记录并评分调用轨迹）。两者合流是 Week 4 的任务。
2. **子串会误命中**。期望 `"395"`，但答案 "I got 3950 at first..." 也包含它。子串匹配对短数字尤其脆弱。
3. **测不了非确定性输出的工具**。`get_current_time` 的正确答案每分钟都在变，固定子串无从校验——这就是 8 个任务里唯一没被覆盖的工具是它的原因。解法在 Week 4 Day 2：programmatic checker，用一段代码验证答案，而不是比对固定字符串。
4. **`last_results.json` 每次运行都被整体覆盖**，历史基线只活到下一次运行为止（第一次跑的 5× 基线就是这么丢的——之后又跑了一次单次，把它冲掉了）。Week 4 Day 5 的 `history.jsonl`（追加而非覆盖）是正解。

### 5. Day 7 一句话总结

eval harness 的价值不在今天的 40/40，而在"任务只增不删 + 评分自动化 + 元数据可追溯"让每一周的进步可验证、每一次回归无处遁形；40/40 的零方差同时也是个警告——任务集还没难到能暴露问题，而一个从不失败的 eval 什么也探测不到。

---

## Week 1 总结

一句话串起七天：**Agent = 一个把错误和观察都喂回模型的 while 循环**。

- **Day 1–2** 搭骨架：API 无状态 → 历史靠客户端拼 → 工具调用只是"模型输出 JSON 然后停下"，循环由我的代码驱动；
- **Day 3–4** 学控制：Schema/description 是软引导，代码校验是硬约束；错误是模型的反馈信号，错误信息的质量决定恢复路径；
- **Day 5** 找源头：手写的循环就是 ReAct 的现代实现，(Thought, Action, Observation) 已被产品化为 reasoning / tool_calls / role="tool"；
- **Day 6** 用数据验证：description 的价值集中在边界任务上（4/5 vs 0/5 的 wrong-tool 差距），但软引导不可靠（detailed 仍有 1 次心算绕过工具），必须配纵深防御；
- **Day 7** 建基础设施：可复利的 eval harness（8 任务 × 5 次 = 40/40 基线，带元数据）——从本周起，"agent 变好了吗"由跑分回答，不由感觉回答。

贯穿全周的一条主线：**永远不要指望模型"自觉"**。description、错误信息、软信号劝退都只是提高概率的手段；安全性和正确性的最后一道防线，必须是代码强制执行的（路径校验、扩展名白名单、MAX_ITERATIONS、自动化评分）。

## 本周安排（自查清单）

- [x] Day 3：给 `agent.py` 加 `read_file` 工具（带路径安全检查）；对比模糊 vs 精确的 description
- [x] Day 4：让 calculator 随机 50% 报错，观察模型重试；对比无用 vs 详细的错误信息
- [x] Day 5：读 ReAct 论文 §1–3；凭记忆重写 agent 循环（`agent_v2.py`，加 `MAX_ITERATIONS` / `MAX_TOOL_RETRIES`）；对比无依赖（并行）vs 有依赖（串行）的工具调用
- [x] Day 6：实验——terse vs detailed 工具描述对错误调用率的影响（2×7×5=70 runs，结果见上）
- [x] Day 7：写周报告，跑基线评测（8 任务 × 5 次 = **40/40** 通过）
- [x] 遗留：`tasks.json` 已补 5 个新任务（read_file、search_files、query_orders 单工具/串行/多查询对比），共 8 个任务
- [x] 遗留：`run_evals.py` 已补跑 5 次基线，并写入元数据（model / date / runs_per_task / total_passed）
- [ ] 带去 Week 2/4 的欠账：`last_results.json` 仍是覆盖式（Week 4 Day 5 用 `history.jsonl` 归档）；任务集零方差、缺区分度（40/40 太容易），需要更难的任务；`get_current_time` 尚无 eval 覆盖（等 Week 4 Day 2 的 programmatic checker）
