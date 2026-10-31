# Week 8 Report — MCP (Model Context Protocol)

**Dates:** 2026-08-31 (Day 1), 2026-09-02 (Day 2), 2026-09-03 (Day 3), 2026-09-08 (Day 4; Days 5–7 进行中)
**Week goal (PLAN):** build both sides of MCP — it's becoming the standard integration layer.

---

## Day 1 — Learning: MCP 架构

### MCP 解决什么问题

Agent 应该怎样以**标准化、可组合、可发现**的方式调用外部能力？

- MCP = 给 LLM/Agent 连接外部世界的一套标准协议，2024-11 由 Anthropic 推出，官方比喻为 **"AI 应用的 USB-C 接口"**。
- 它不关心工具具体做什么，只标准化三件事：**能力如何自我宣告（declare）、如何被调用（invoke）、结果如何返回（return）**。只关注上下文交换协议，不规定 AI 应用如何使用 LLM。
- 核心思想：把"**如何连接外部系统**"和"**模型如何使用外部能力**"解耦。Agent 不需要知道订单数据库是 MySQL 还是 PostgreSQL，只需要知道 `get_order(order_id)`。

**传统做法 vs MCP（以订单系统为例）：**

```
传统：每个 Agent 重写一套连接逻辑           MCP：外部能力包装成标准 Server
Agent                                      Agent
 ├── 自己写订单 API client                    │ (MCP Client)
 ├── 自己写数据库 client              ┌──────┼──────┐
 ├── 自己写 OCR API                   ↓      ↓      ↓
 ├── 自己写 ES client            Orders    Knowledge  OCR
 ├── 自己写文件系统接口           Server    Server    Server
 └── 自己写各种 prompt              │         │         │
                                    ↓         ↓         ↓
                                  订单DB    向量库    OCR模型
```

### 三个角色：Host / Client / Server（严格分离）

```
┌────────────────────────────────┐
│          MCP Host              │   LLM / Agent / UI
│   ┌────────────┐ ┌───────────┐ │
│   │ MCP Client │ │ MCP Client│ │   ← 每个 Client 与一个 Server 严格 1:1
│   └─────┬──────┘ └─────┬─────┘ │
└─────────┼──────────────┼───────┘
          ↓              ↓
      Server A        Server B
```

| 角色 | 定义 | 核心职责 |
|---|---|---|
| **Host** | 真正运行 Agent 的宿主应用（Claude Desktop、Claude Code、Cursor、ChatGPT…） | 拥有模型循环（model loop）和会话；管理多个 Client 生命周期；执行安全策略与用户授权；**是信任边界**，所有外部操作必须经 Host 审核 |
| **Client** | 嵌入 Host 内部的协议组件，与**单个** Server 保持 1:1 专用连接 | 建立传输连接（stdio/HTTP）；握手与能力协商；JSON-RPC 2.0 序列化；双向路由消息；管理会话生命周期 |
| **Server** | 独立的轻量级程序，纯能力提供者 | 声明自身能力（tools/resources/prompts）；执行工具调用/资源读取/模板渲染；**对调用方完全无感知**（不知道哪个 Host 或模型在调它） |

关键约束与不变量：

- **Host 不能把 LLM API 调用放进 Server 内部执行**（模型循环归 Host）。
- **Client 与 Server 严格 1:1**：Host 连接 3 个 Server 就创建 3 个 Client 实例，一个 Server 故障不影响其他连接。
- **Server 保持"纯能力提供者"角色**，不感知 agent 或模型。
- MCP Server 本质是**协议适配层 / capability layer**——Server 后面才是真正的业务系统（MySQL、Redis、ES、REST API、文件系统…）。

### 三种原语：Tools / Resources / Prompts

三者由**不同角色控制**——这是选择用哪个原语的关键依据：

| 原语 | 控制者 | 触发方式 | 典型用途 | 类比 |
|---|---|---|---|---|
| **Tools** | 模型 | 模型自主决定调用 | 执行动作、查询动态数据 | 动词（做什么） |
| **Resources** | 应用/Host | Host 主动附加到上下文 | 提供静态/半静态上下文数据 | 名词（有什么） |
| **Prompts** | 用户 | 用户/UI 主动选择（斜杠命令、菜单） | 标准化交互模板、少样本示例 | 工作流（怎么做） |

- **Tools**：有 `name`、`description`、`inputSchema`（JSON Schema），通过 `tools/list` 发现、`tools/call` 执行，返回结构化 content 块。执行可能需要用户同意（Host 层强制）。
- **Resources**：**只读**被动数据源，通过 URI 寻址（`file:///home/user/doc.txt`），用 `resources/list` / `resources/read` 访问。设计哲学：让 Host 精确附加用户正在查看的数据（by URI），而不是让模型猜测该拉什么。**对 RAG 很重要**——外部知识系统可以包装成 resource（如 `catalog://product/rule-123`）。很多读操作应该做成 Resource 而非 Tool：被动数据 vs 主动操作，语义不同。
- **Prompts**：Server 维护的版本化模板，用 `prompts/list` / `prompts/get` 访问，交付"经过审核的分析模板"而非每次让模型重新发明提示。

一个完整的企业 Server 可以同时暴露三类：`get_order`（Tool）+ `order://…`（Resource）+ `order_review`（Prompt）。

### 协议双层结构：Data Layer + Transport Layer

**数据层 = JSON-RPC 2.0 之上的消息语义：**

| 功能 | 说明 |
|---|---|
| Discovery | `server/discover`（2026 新规范）/ `initialize`（2025 规范）——协商协议版本与能力 |
| Server Primitives | Tools / Resources / Prompts 的发现与调用 |
| Client Primitives | Elicitation（Server 向用户请求额外输入） |
| Notifications | 实时变更通知（如 `notifications/tools/list_changed`） |
| Utility | 进度追踪、长运行任务 |

**JSON-RPC 2.0 基础（实验中直接用到）：**
- 请求（request）带 `id`，必须有响应；通知（notification）无 `id`，不期待响应（如 `notifications/initialized`）。
- 标准错误码：`-32601 Method not found`（方法不存在）、`-32700` 解析错误、`-32600` 无效请求。**我们在实验里亲眼看到了 `-32601`**（见下）。
- 模型永远看不到原始 JSON-RPC——协议层完全封装在 Client-Server 之间；Host 是策略执行点（可要求人工审批、脱敏参数、限流或直接拒绝）。

**传输层：stdio vs Streamable HTTP**

| 维度 | stdio | Streamable HTTP |
|---|---|---|
| 场景 | 本地、同机进程间 | 远程/托管、多租户 |
| Server 形态 | 子进程（child process） | 独立服务进程 |
| 通信 | stdin/stdout，换行分隔的 JSON-RPC | HTTP POST，响应可选 SSE 流式 |
| TLS / LB / K8s | 不需要 | 需要/支持/非常适合 |
| 客户端数 | 通常单客户端 | 多客户端 |

- stdio：Client 把 Server 作为子进程启动。约束：消息不能含内嵌换行；Server 的日志走 **stderr**（stdout 只能有 MCP 消息）。
- Streamable HTTP：Server 提供单一端点（如 `https://example.com/mcp`），同时支持 POST 和 GET。
- 官方建议：**按信任边界选传输**——进程内用 stdio，跨网络用 HTTP，把认证恰好放在网络边界处。

### 能力发现与协商（Capability Discovery & Negotiation）

MCP 相对普通 function calling 最重要的增量之一：

```
连接 Server → Server 宣告"我有哪些 tools/resources/prompts"
→ Client 拿到定义 → Host/Agent 转成 LLM 能理解的 tool definitions
→ LLM 决定"要不要用 / 用哪个"
```

- Server 在握手时广告（advertise）自己支持的能力；双方必须尊重对方声明，可独立演进、向后兼容。
- 工程现实中几十个 Server / 几百个 Tool 不会全部塞进上下文——**发现 → 筛选 → 延迟加载**，只把当前相关的暴露给模型。
- 一句话分工：**Client 负责"有什么、怎么连、怎么调用"；Application 决定"当前给模型看什么"；LLM 决定"用哪个"。**

### 规范版本：我们用的 2025-06-18 vs 新规范 2026-07-28

⚠️ 容易混淆的点（本次实验切身碰到）：

| 规范 | 2025-06-18（当前主流） | 2026-07-28（新方向） |
|---|---|---|
| 握手 | `initialize` → 响应含 protocolVersion / capabilities / serverInfo → 客户端发 `notifications/initialized` | 转向 **stateless protocol core**，`server/discover` 自描述请求 |
| 会话 | 协议层 session 存在 | 不再要求协议层 handshake/session；应用状态由应用显式管理 |
| HTTP 形态 | Streamable HTTP | legacy HTTP+SSE 进入弃用路线，优先 Streamable HTTP |

新规范动机：强 session state 的 Server 在 Load Balancer 后面（请求打到 S1、下次打到 S2）设计复杂度很高。新架构倾向自包含 request → 任何 Server instance 都能处理。**协议层无 session ≠ 应用不能有 state**，而是显式管理。

我们 Day 1 实际用到的 filesystem server 说的是 **2025-06-18**（当前 SDK 的主流版本），Day 2 起的 Python SDK 实验也会基于它。

### MCP vs 其他东西

- **vs Function Calling**：MCP 是标准化、可复用、可管理、解耦的连接层。Function calling 是"模型怎么调函数"，MCP 是"能力怎么被宣告、发现、连接、鉴权"——它还包括 resources、prompts、传输、鉴权。
- **vs REST API**：REST 给程序调用；MCP 给 Agent/AI 应用使用（tool 有 description + JSON Schema input，服务 AI-native 集成）。它不是 REST 的替代品，是 AI-native integration protocol。
- **vs 具体技术**：MCP 不是 API、不是 tool call、不是 HTTP（还有 stdio）、不是 HTTP+SSE（而是 Streamable HTTP）。

### 架构分层：RAG / Memory / MCP 各管一段

```
AI Agent
 ├── RAG     → Knowledge → Retrieval    "我怎样找到相关知识？"
 ├── Memory  → State     → Notes        "我怎样保留跨步骤/跨 session 的信息？"
 └── MCP     → External World → Tools/Resources  "我怎样标准化连接外部能力和数据？"
```

这是非常清晰的工程分层：reasoning layer（LLM + Agent Loop）与 external capability layer（MCP Server → DB/API/Files）解耦。LLM 不需要直接理解数据库、HTTP API、Python SDK。

---

## Day 1 — Build: 安装 filesystem server 并用 Inspector 戳它

### Question

一个现成的 MCP server 装上后，我能通过 Inspector 完成一个完整的 MCP 会话吗？——握手、发现工具、调用工具，并观察它的安全边界和能力边界。

### Setup

- **Server:** `@modelcontextprotocol/server-filesystem`（npx 拉取），启动时只允许访问 `experiments/week08/sandbox/` 一个目录。
- **Inspector:** `@modelcontextprotocol/inspector` v2.4.0，Web UI 在 `http://127.0.0.1:6274`。Inspector 本身就是一个 **MCP Host**——它把 Server 作为子进程 spawn（stdio transport），并把 JSON-RPC 通过自己的 REST + SSE proxy 暴露给 UI/API。
- **戳法:** `poke_inspector.py` 无头驱动 Inspector 的 proxy API（`POST /api/mcp/connect` → `GET /api/mcp/events`（SSE）→ `POST /api/mcp/send`）——和 Web UI 走同一条路，所以这是真正的 Inspector 中介会话，不是裸 stdio client。Auth: `x-mcp-remote-auth: Bearer <token>`。

### 核心过程（实际观察到的完整握手）

```
1. POST /api/mcp/connect        → Inspector spawn 子进程（stdio），返回 sessionId
2. send initialize              → Server: "secure-filesystem-server 0.2.0"
                                   protocolVersion: 2025-06-18
                                   capabilities: {tools: {listChanged: true}}
3. send notifications/initialized (无 id 的通知，握手完成)
4. send tools/list              → 14 个工具
5. send tools/call × 6          → 4 次成功，2 次被服务器拒绝
6. send resources/list, resources/templates/list, resources/read, prompts/list
                                → 全部 -32601 Method not found
```

### Result

| Poke | 结果 | 数据 |
|---|---|---|
| `initialize` | ✅ | serverInfo `secure-filesystem-server 0.2.0`；协商版本 **2025-06-18**；只广告了 `tools` 能力 |
| `tools/list` | ✅ | **14 个工具**：read_file, read_text_file, read_media_file, read_multiple_files, write_file, edit_file, create_directory, list_directory, list_directory_with_sizes, directory_tree, move_file, search_files, get_file_info, list_allowed_directories |
| `list_allowed_directories` | ✅ | 只允许 sandbox 一个目录 |
| `list_directory (sandbox)` | ✅ | `[FILE] hello.txt`, `[DIR] notes` |
| `read_file (hello.txt)` | ✅ | 返回文件内容 |
| `get_file_info (notes/mcp-notes.md)` | ✅ | size: 143, created/modified 时间戳 |
| `read_file (/etc/hosts)` | 🚫 **blocked** | `Access denied - path outside allowed directories`，`isError: true` |
| `read_file (../PLAN.md)`（穿越） | 🚫 **blocked** | 同上 |
| `resources/list` 等 4 个方法 | ❌ not-supported | 全部 `-32601 Method not found` |

### Discoveries（实验核心发现）

1. **路径安全由 Server 端强制执行。** 绝对路径逃逸（`/etc/hosts`）和相对路径穿越（`../PLAN.md`）都被拒——"allowed directories"不是摆设。而且这个 server 暴露了 `write_file` / `edit_file` / `move_file` 等**写工具**，所以这个沙箱限制是唯一挡在 LLM 和磁盘写之间的东西，其余只能靠 Host 的权限层。
2. **MCP 能力是可选的，Server 只实现它需要的。** 这个 filesystem server 是 **tools-only**：resources 和 prompts 全部 `-32601`（JSON-RPC 标准错误码"方法不存在"）。旧版 filesystem server 曾提供 `file://` resources，新版删掉了。这直接印证了"能力协商"的设计——Client 必须容忍 Server 缺能力。
3. **Server 的演进写在工具描述里。** `read_file` 的描述以 `DEPRECATED: Use read_text_file instead` 开头——MCP 的自我宣告机制让"升级"也变成协议里可见的东西。
4. **Tool 结果是 content 块而非裸返回值。** 每个 tools/call 返回 `content: [{type: "text", text: ...}]`，拒绝时用 `isError: true` 标记——错误也是协议的一部分，不是 HTTP 层的东西。
5. **Inspector v2 换了内部形态。** 老教程描述的 `GET /sse` + `POST /message?sessionId=` 已不存在，现在是 REST API（`/api/mcp/connect`、`/api/mcp/send`）+ SSE 事件流（`/api/mcp/events`）。Inspector 本身就是一个 Host 的实例——它就是"自己吃自己的狗粮"。

### Surprise

- 预期"filesystem server 应该有 resources"（旧教程都这么写），实际它是个 tools-only server——协议能力会随版本演进，旧资料会过时。
- 实验环境的小坑：macOS 的**系统级**代理配置（127.0.0.1:7890，已死）会被 Python `urllib` 读到，导致访问 localhost 的 Inspector 直接 502，而 `curl` 不受影响（不读系统代理）。修复：`urllib.request.build_opener(urllib.request.ProxyHandler({}))`。

### Next question（喂给 Day 2）

- 用 Python SDK 自建一个暴露 week-3 工具的 MCP server，Inspector 能正常连上吗？（Day 2 的 Build 任务）
- Python SDK 实现的是哪个协议版本？2026 stateless 规范落地后，现有 session 型代码要改多少？

---

## Day 2 — Learning: MCP Python SDK

MCP Python SDK 是 MCP 的**官方 Python 实现**，由 Anthropic 维护，与 TypeScript、C#、Go 并列为官方 SDK（Python 为 **Tier 1**）。它的价值一句话：**把 MCP 协议层的复杂性封装掉，让你主要关注业务逻辑**。

用 SDK 可以：编写 Server / Client；暴露 Tools / Resources / Prompts；使用 stdio / Streamable HTTP 等 transport；处理 JSON-RPC、类型、schema、capabilities 等底层细节。

**版本选择：**

| | v1.x（维护） | v2.x（当前主线，本次实验使用） |
|---|---|---|
| 命名 | FastMCP | **MCPServer**（改名，import 路径随之变化） |
| 状态 | 维护模式 | 主线；面向 2026-07-28 新规范，修复 v1 架构问题 |

### 核心能力一：构建 Server —— 三个 decorator

最核心的设计：**decorator 注册能力**——一个普通 Python 函数加个装饰器，就变成模型可调用的能力。

**MCPServer 类：**

```python
from mcp.server.mcpserver import MCPServer, Context

mcp = MCPServer(
    name="Demo",           # 服务器标识
    title="Demo Server",   # 人类可读标题（v2 新增）
    description="...",     # 描述（v2 新增）
    instructions="...",    # 给模型的系统指令
    version="1.0.0",       # 版本号（v2 默认空串，应显式设置）
)
```

#### Tools（控制权：模型）

只需要使用 `@mcp.tool()` 装饰器，就能告诉 SDK"这个 Python 函数是一个 MCP tool"——一个普通的 Python 函数就能变成模型可调用的工具，**无需手动编写协议处理代码**：Tool 的名称、description、argument schema 从函数名、docstring、type hints 自动得到（这直接回答了之前的疑问"MCP Server 怎么告诉 Agent 自己有哪些 Tool"：Client 通过 `tools/list` 拿到这些信息，根本不需要手写 JSON Schema）。

```
Python type hints
       ↓
JSON Schema
       ↓
LLM Tool Definition
```

```python
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

SDK 自动推导：

- **名称** = 函数名
- **描述** = docstring
- **输入 Schema** = 从类型提示生成 JSON Schema（通过 Pydantic）
- **输出 Schema** = 从返回类型注解生成

- **参数验证**：`Annotated[..., Field(...)]` 添加约束（ge、le、description 等）；非法参数在函数执行前被拒绝，返回模型可读取的错误。
- **同步/异步**：`async def` 在事件循环上执行；普通 `def` 在 v2 中自动放入 worker 线程（anyio.to_thread.run_sync()），避免阻塞事件循环。
- **双通道返回**：

| 通道 | 用途 | 来源 |
|---|---|---|
| `result.content` | 文本，供**模型**阅读 | 返回值自动序列化 |
| `result.structured_content` | 结构化数据，供**应用**消费 | 返回类型注解的 Schema |

```
result.content             # [TextContent(text='{"temperature": 16.2, "humidity": 0.83}')]
result.structured_content  # {"temperature": 16.2, "humidity": 0.83}
```

- **Schema 验证**：返回值会针对 output_schema 验证；不匹配时返回 `is_error=True`，而非静默传递损坏数据。

#### Resources（控制权：应用/Host）

使用 `@mcp.resource()` 装饰器定义 **URI 模板化的只读数据源**：

```python
@mcp.resource("order://{order_id}")
def get_order_resource(order_id: str) -> str:
    ...
```

- URI 模板使用 **RFC 6570** 语法：`{order_id}` 与函数参数名必须匹配，否则**在导入时**抛出 ValueError。
- 官方 SDK 文档特别强调：URI 中 `{name}` 这种参数会被识别成 resource template 的参数。

#### Prompts（控制权：用户）

使用 `@mcp.prompt()` 装饰器定义可复用的提示词模板，SDK 会把它注册成 MCP Prompt：

```python
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    # ... 根据 style 生成不同的提示词 ...
    return f"Please write a warm greeting for {name}."
```

- 返回 `str` → 单条用户消息
- 返回 `list[Message]` → 多轮对话种子（可包含 UserMessage 和 AssistantMessage）
- 支持嵌入 Image、Audio、EmbeddedResource

#### 三个 decorator 一览

```
MCP Server
│
├── @mcp.tool()      → Tools（模型控制）
│
├── @mcp.resource()  → Resources（应用/Host 控制）
│
└── @mcp.prompt()    → Prompts（用户控制）
```

官方教程强调：**你不需要自己写 JSON Schema、request handler、capability declaration 或 JSON-RPC framing**，这些都由 SDK 处理。SDK 内部一次工具调用的完整流程：

```
① 注册 Tool → ② 获取名字 → ③ 获取 description → ④ 读取 type hints
→ ⑤ 生成 input schema → ⑥ 暴露 tools/list → ⑦ 接收 tools/call
→ ⑧ 验证参数 → ⑨ 调用 Python function → ⑩ 把结果包装成 MCP result
```



### 核心能力二：构建 Client

如果 Server 是提供能力，Client 则是消费能力：

```
Client → list tools → call tool → read resource
```

SDK 同样提供完整的客户端功能，让你能连接到任何符合规范的 MCP 服务器：

```python
from mcp import Client

# 1. 内存内连接（测试用）
# 不启动 subprocess、不打开端口、不使用真实 transport，
# 直接让 Client 连接 Server object 做测试
async with Client(mcp_server_instance) as client:
    result = await client.call_tool("add", {"a": 1, "b": 2})

# 2. HTTP 远程连接
async with Client("http://localhost:8000/mcp") as client:
    tools = await client.list_tools()

# 3. stdio 子进程连接
async with Client("path/to/server.py") as client:
    ...
```

stdio 模式要点：

- Client 将 Server 作为子进程启动；通过 stdin 写入 JSON-RPC，从 stdout 读取响应
- Server 生命周期与 Client 绑定
- **stdout 专用于协议消息，日志必须写入 stderr**

Client 是 async context manager，自动处理连接生命周期。

#### Context 类

注入到工具/资源/提示处理函数中的运行时上下文：

```python
from mcp.server.mcpserver import Context

@mcp.tool()
async def my_tool(x: int, ctx: Context) -> str:
    await ctx.report_progress(1, 2)      # 报告进度
    await ctx.info("Processing...")      # 发送日志（已弃用，建议用标准 logging）
    return str(x)
```

### SDK 依赖栈

官方 SDK 并不是简单几个 Python class，下面是它的基础设施：

```
mcp → mcp-types, anyio, pydantic, httpx, starlette, uvicorn,
      sse-starlette, jsonschema, pyjwt, OpenTelemetry
```

| 依赖 | 职责 |
|---|---|
| pydantic | schema 生成与 validation |
| anyio | 异步运行时抽象（兼容 asyncio / Trio） |
| httpx | HTTP client transport |
| starlette + uvicorn | HTTP server |
| jsonschema | tool structured output 验证 |

### 传输层：stdio / Streamable HTTP / SSE

SDK 全面支持 MCP 规范中定义的标准传输层：

- **stdio**：通过标准输入/输出与本地子进程通信，是本地集成的默认方式。
- **Streamable HTTP**：基于 HTTP 的传输方式，适用于远程服务部署。
- **SSE (Server-Sent Events)**：服务器向客户端推送事件，适用于需要流式响应的场景。

### 开发工具链（CLI）

| 命令 | 用途 |
|---|---|
| `mcp dev server.py` | 启动开发服务器 + 自动打开 MCP Inspector（交互式测试/调试 UI） |
| `mcp run server.py` | 运行 MCP 服务器 |
| `mcp install server.py` | 安装到 Claude Desktop |

（老版本需 `pip install "mcp[cli]"` extra；本次安装的 2.1.1 基础包即自带 `mcp` CLI。）


### 架构分层：高阶 API vs 低阶 API

1. **高阶 API：MCPServer（原 FastMCP）** —— 设计哲学：你写 Python 函数，SDK 负责生成 JSON Schema、处理协议、管理传输。
2. **低阶 API：Server** —— 当需要完全控制协议行为时使用。v2 中低阶 Server 的处理器通过构造函数的 `on_*` 参数注册（而非 v1 的装饰器模式）。

### 错误处理

| 异常类型 | 用途 | 模型可见？ |
|---|---|---|
| **`ToolError`** | 执行失败（模型可修复） | ✅ 是，`is_error=True`，消息在 `content` 中 |
| **`MCPError`** | 协议级错误（请求本身无效） | ❌ 否，JSON-RPC 错误返回给 Host |
| **其他异常** | 未预期的崩溃 | ⚠️ `is_error=True`，但模型只收到通用消息，traceback 写入服务器日志 |

### 为什么 SDK 大量使用 async？

因为 MCP Server 很容易遇到数据库、HTTP API、文件、模型、网络这些 **I/O-heavy 操作**。SDK 底层采用 AnyIO，兼容 asyncio / Trio。**MCP Server 本质上是一个异步 I/O service，而不是一个单纯的 Python 函数集合。**

### MCP SDK vs Tool Calling

以前是：

```
LLM → Tool Schema → LLM chooses tool → Application → Python function
```

现在是：

```
LLM → Tool Schema → LLM chooses tool → MCP Client → MCP Protocol → MCP Server → Python function
```

MCP Python SDK 并没有取代 Tool Calling——它是在 Tool Calling 和外部能力之间**增加了一个标准化协议层**。SDK 的本质不是一个工具库，而是"一个让 Python 程序能够作为 MCP Server 或 MCP Client 参与 MCP 网络的完整协议实现"。

### 设计取舍 Q&A：什么能力该进 MCP？

**Q：** 实际 agent 开发中，有的工具/资源/提示词是某个流程专有的——整体 agent 设计上，是建议用 MCP 纳管，还是在 workflow / Host 中独立加载？如何权衡？

**A：** MCP 管**可以复用**的能力；Workflow 管"怎么完成这个业务流程"；MCP 管"这个流程可以使用哪些外部能力"。以后遇到一个 Tool / Resource / Prompt，问自己：**"如果明天把这个 Agent 换成另一个 Agent，这个能力还有没有独立存在的价值？"** MCP 不应该成为 Agent 内部函数调用的替代品。

> Agent-specific capability → Host/Workflow；Shared / External / Independently governed capability → MCP。

| 能力 | 更建议 | 原因 |
|---|---|---|
| 多个 Agent / Workflow 都会用 | **MCP** | 复用、统一管理 |
| 外部系统能力（DB/API/OCR/搜索） | **MCP** | 解耦 Agent 与系统 |
| 需要独立权限/审计/部署 | **MCP** | Server 边界清晰 |
| 业务流程专属 Tool | Workflow/Host | MCP 增加不必要的复杂度 |
| 只在某一步内部使用的函数 | Workflow 内部 | 直接调用最快 |
| Agent 私有 Prompt / 状态转换 | Host/Workflow | 属于 Agent 编排逻辑 |
| 高频、低延迟的内部函数 | Host/Workflow | 避免额外协议开销 |

---

## Day 2 — Build: 自建 MCP server（stdio）并用 Inspector 测试

### Question

用 MCP Python SDK 把 2 个 week-3 工具包成 MCP server（stdio transport），Inspector 能完成完整会话吗？week-3 的安全边界和错误契约能否原封不动地穿过 MCP 协议？

### Setup

- **SDK:** `mcp` 2.1.1（官方 Python SDK，v2 主线；FastMCP 已改名 `MCPServer`）
- **Server:** 自建 `experiments/week08/mcp_server.py`，暴露 week-3 的 **`read_file` + `write_file`**（Day-1 文件工具对）——选这对是因为 write→read 往返能证明**状态真的流过协议**，而不只是回话。
- **复用而非复制:** 直接 `import experiments/week03/agent.py`，把模块级 `SANDBOX_ROOT` 一行改指到 `week08/sandbox` —— realpath + commonpath 包含检查、扩展名白名单、分类错误 envelope 全部原样复用。同一道安全边界，两个入口（agent 直调 / MCP）。
- **错误契约:** week-3 工具从不 raise，失败返回 `{status:"error", error:{type, message, hint}}`；server 把 error envelope 标成 `is_error=true` 的 content 块——模型在 week-3 学会读的格式，在 MCP 后面还是同一个格式。
- **传输:** `mcp.run(transport="stdio")` —— Inspector 把 server 作为子进程 spawn。
- **测试:** 无头驱动 Inspector REST proxy（复用 Day 1 的 `InspectorClient`）→ `poke_day2_server.py`；另用裸 stdio pipe 做 smoke test。

### 核心过程（与 Day 1 同一条路）

```
1. npx @modelcontextprotocol/inspector python experiments/week08/mcp_server.py
2. connect → 子进程 spawn，返回 sessionId
3. initialize → serverInfo: week08-sandbox-files 0.1.0，协议 2025-06-18
   capabilities: tools + prompts + resources（SDK 默认全集）
4. tools/list → 恰好 2 个工具，inputSchema 完整（含参数 description）
5. tools/call × 10 → 6 ok + 4 blocked
6. prompts/list、resources/list → 空列表
```

### Result

| Poke | 结果 | 数据 |
|---|---|---|
| `initialize` | ✅ | `week08-sandbox-files 0.1.0`；协商版本 **2025-06-18** |
| `tools/list` | ✅ | **恰好 2 个工具**：read_file、write_file，JSON Schema 自动生成（description + 参数 description 来自 docstring / `Annotated[Field(...)]`） |
| `read_file hello.txt` / `notes/mcp-notes.md` | ✅ | 返回真实内容（Day 1 的沙箱文件） |
| `write_file mcp-via-inspector.md` → `read_file` | ✅ | **往返 v1**：写 54 字节，读回内容完全一致 |
| `write_file`（覆盖）→ `read_file` | ✅ | **往返 v2**：58 字节，读到的是新内容——状态真的流过 MCP |
| `read_file /etc/hosts` | 🚫 **blocked** | `sandbox_escape`: Absolute paths not allowed，`isError: true` + hint |
| `read_file ../PLAN.md` | 🚫 **blocked** | `sandbox_escape`: Path escapes sandbox，`isError: true` + hint |
| `write_file ../evil.txt` | 🚫 **blocked** | 同上——**写方向同样受限** |
| `write_file binary.bin` | 🚫 **blocked** | `extension_not_allowed`：只允许 .txt/.md/.py/.json/.csv |
| `prompts/list`、`resources/list` | ✅ | 空列表（SDK 注册了这两个 capability） |

**SUMMARY: 13 pokes，9 ok，4 blocked-as-designed**（结果存档 `day2_results_20260902_215006.json`）

### Discoveries

1. **复用 week-3 工具的正确姿势是"指过去"，不是"抄过来"。** import + 一行 `SANDBOX_ROOT` 就把 week-3 的全部安全逻辑原样搬进 MCP——好的工具设计在协议迁移时零成本。Day 1 对 reference filesystem server 验证过的路径安全（绝对路径 + 穿越 + 扩展名），对我们的 server 同样成立。
2. **错误契约可以原封不动地穿过协议。** week-3 envelope 作为 text content 返回，error 用 `is_error=true` 标记（而非 SDK 的 `ToolError` raise 路径）——这样模型拿到的还是它认识的 `{status, error, hint}`，**Day 4 桥接回 agent 时零适配**。
3. **SDK 自动推导是真的。** tools/list 里的 inputSchema 完全来自 docstring + type hints + `Annotated[Field(...)]`，没有手写一行 JSON Schema——Learn 部分的"①~⑩ 流程"在真实握手里走了一遍。
4. **返回类型注解影响能力宣告。** 工具返回标注 `mcp_types.CallToolResult` → tools/list 里**没有** outputSchema，但换来 per-call 的 `is_error` 控制权；不标注则 SDK 生成 outputSchema 并做结构化验证。两种模式各有用途：结构化数据给应用（`structured_content`），文本契约给模型（`content`）。
5. **"Server 只实现它需要的"在 SDK 里变成了"SDK 给默认全集，你选择不填充"。** 我们的 server advertise 了 prompts 和 resources（返回空列表），Day 1 的 filesystem server 是 tools-only（`-32601`）——两种哲学都能通过握手，Client 必须都能容忍。
6. **版本协商是双向的。** SDK 2.x 面向 2026-07-28 新规范，但 Inspector 提出 2025-06-18，握手就定在 2025-06-18——现实世界的主流 Host 还在 2025 规范上，新 SDK 向后兼容。
7. **stdio 的生命周期归 Host。** smoke test 中发现：stdin 一关（EOF）server 立即退出，来不及响应的请求会丢——这不是 bug，是 stdio 的语义：**Host owns the process**（Day 1 的不变量在自建 server 上得到印证）。Inspector 会保持管道打开到 session 结束。

### Surprise

- 预期"自己写 MCP server 要写不少协议胶水"，实际 SDK 把 JSON-RPC framing、schema 生成、capability 宣告全部接走，代码主体是 **20 行复用 + 10 行包装**——真正花时间的是"错误该怎么跨协议表达"这个设计问题，而不是协议实现。
- mcp 2.x 把 FastMCP 改名 MCPServer 且 import 路径变了（`from mcp.server.mcpserver import MCPServer`）——网上教程全是 v1 FastMCP 写法，直接照抄会 `ModuleNotFoundError`（SDK 自己的报错信息里带了迁移指引）。又一个"旧资料会过时"的实证。

### Next question（喂给 Day 3 / Day 4）

- Day 3 手写 MCP client（stdio）直连这个 server——不经过 Inspector 的 `list_tools` / `call_tool` 有什么不同？
- Day 4 桥接：tools/list 的 inputSchema 能直接映射成 OpenAI tool schema 吗？`is_error` envelope 转成 tool message 后，week-3 的 agent 还能不能"读 hint 自救"？

---

## Day 3 — Learning: MCP client side（session lifecycle / list_tools / call_tool）

Day 1 学了协议架构，Day 2 学了怎么写 Server。Day 3 换到另一侧——**MCP Client**：嵌在 Host 里、与单个 Server 保持 1:1 连接的协议组件。它的核心交互围绕 `ClientSession` 展开，就三件事：**管理连接生命周期 + 发现能力（list_*）+ 调用能力（call_tool 等）**，构成"发现能力 → 使用能力"的完整闭环。

| MCP Client API | 解决的问题 | 类比 |
|---|---|---|
| `Client(...)` | 我怎么连接/使用这个 MCP Server？ | 建立客户端 |
| `list_tools()` | Server 有哪些工具？ | 查看 API 文档 |
| `call_tool()` | 我要执行某个工具 | 调 API |
| `list_resources()` / `read_resource()` | Server 有哪些资源？怎么读？ | 资源目录 / GET |
| `list_prompts()` / `get_prompt()` | Server 有哪些 Prompt？怎么拿？ | 查看/获取模板 |

### 规范演进：从有状态握手到无状态协议

| 维度 | 2025-11-25 及之前（Legacy） | 2026-07-28（Current） |
|---|---|---|
| **握手** | 强制的 `initialize` / `initialized` 双向握手 | 移除；每个请求自包含 |
| **会话状态** | `Mcp-Session-Id` 绑定连接 | 移除；协议层无会话概念 |
| **版本协商** | 握手时一次性确定 | 每请求通过 `_meta` 字段携带 |
| **能力发现** | 握手时交换 capabilities | 按需通过 `server/discover` 查询 |
| **负载均衡** | 需要 sticky session | 支持标准轮询负载均衡 |

核心设计原则一句话：**协议事实从"连接状态"变成"可见、可验证的请求数据"**。2026-07-28 规范通过 SEP-2575（Make MCP Stateless）和 SEP-2567（Sessionless MCP）实现这一转变。

旧版 MCP 需要先建立通信、初始化会话；新版正式移除了 `initialize` / `initialized` / `Mcp-Session-Id`，协议核心是无状态请求-回复。巨大的好处：**一个请求可以去 A，下一个请求可以去 B**——不需要 Client A 永远绑定 Server A，服务器之间也不需要共享 session state，容易横向扩展（LB 后面任何实例都能处理）。

### Client lifecycle：协议 stateless ≠ 客户端对象没有 lifecycle

那为什么 Python SDK 还有 `async with Client(...)`？

因为 **"MCP 协议 stateless" ≠ "Python Client 对象没有 lifecycle"**。Client 是 Python 程序与 MCP Server 对话的方式，它有对象生命周期：构造 → `async with` → 调用方法 → 退出清理。`async with` 管理的是 **SDK / transport / client object 的生命周期**，而不是旧版 MCP Session ID 的生命周期。

```
Client object 创建
       │
       ▼
进入 async with
       │
       ├── 建立 transport（spawn 子进程 / 连接 HTTP 端点）
       ├── 准备 protocol communication
       ├── 获取/确定 server information
       └── 准备调用
       │
       ▼
   client.list_tools() → client.call_tool() → client.read_resource() …
       │
       ▼
退出 async with
       │
       ▼
清理 transport / connection
```

一句话：`async with Client(...)` 管"客户端对象和传输层的生命周期"；`list_tools()` / `call_tool()` 管"具体 MCP 操作"。

> **实验注脚（Build 部分实证）：** mcp 2.1.1 的低阶 `ClientSession` 仍然按 legacy 流程执行 `initialize` 握手（Day 3 实测协商版本 2025-11-25）——新规范已经发布，但 session 型 API 向后兼容。"新旧并存"不是纸面结论，是我们亲手跑出来的。

### 一次工具调用的完整链路

```
User Query
    │
    ▼
┌─────────────┐
│  LLM 决定   │──── "需要调用 get_order"
└─────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│ Host 路由到对应 Client                  │
│ （根据工具名找到拥有该工具的 Client）     │
└────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│ Client 构建 JSON-RPC 请求              │
│ method: "tools/call"                  │
│ params: { name, arguments, _meta }    │
└────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│ Transport 发送（stdio / HTTP）          │
└────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│ Server：验证 inputSchema → 执行处理器   │
│         → 验证 outputSchema（v2）      │
└────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│ Client 收到 CallToolResult             │
│ ├─ content → 注入对话上下文（模型阅读）  │
│ ├─ structured_content → 应用消费       │
│ └─ is_error → 决定下一步动作           │
└────────────────────────────────────────┘
    │
    ▼
┌─────────────┐
│  LLM 继续   │──── 基于工具结果生成最终回复
└─────────────┘
```

### list_tools()：发现能力

`list_tools()` 是 **MCP Server → Host/Agent 的 Tool Schema Discovery**：返回的 Tool 定义包含 `name` / `title` / `description` / `input_schema`，其中 `input_schema` 是 JSON Schema，描述模型应该如何构造参数。

它和普通 function calling 的 tool schema 的区别：**普通 schema 是 agent 自己维护的；MCP schema 是外部工具的、可以被多个 agent 复用**。

但注意，**`list_tools()` ≠ 把全部工具塞进 LLM**——把根本用不到的工具加载进上下文是浪费。正确姿势：

```
   MCP Client
       │  discover capabilities
       ▼
  Tool Registry
       │
       ▼
   Tool Router
    /        \
relevant  irrelevant
   │           X
   ▼
  LLM
```

例如订单 Agent 当前在"判断这个订单是否满足发货条件"，只暴露 `get_order` / `get_inventory` / `get_rules`，而不是 email / calendar…

所以 `list_tools()` 和 Tool Routing 是两个不同的问题：

| 角色 | 职责 |
|---|---|
| **MCP Client** | 发现——"Server 有什么？" |
| **Host / Application** | 选择与暴露——"当前应该给 Agent 哪些？怎么路由？" |
| **LLM** | 决策——"在这些候选工具里，我现在要调用哪个？" |

两个工程细节：

- **分页（Pagination）**：不要假设 `list_tools` 一次返回全部工具。MCP 支持 cursor 分页，循环直到 `next_cursor is None`（当前 SDK 通过 `PaginatedRequestParams(cursor=…)` 传递）：

  ```python
  cursor = None
  while True:
      result = await client.list_tools(params=PaginatedRequestParams(cursor=cursor))
      for tool in result.tools:
          ...
      if result.next_cursor is None:
          break
      cursor = result.next_cursor
  ```

  官方文档明确指出每个 `list_*` 都支持 cursor。SDK 自己的 `MCPServer` 某些场景一次返回全部，所以简单代码常见只调一次——但写通用 client 要做循环。
- **Tool List Cache**：`tools/list`、`prompts/list`、`resources/list` 的结果现在可带 `ttlMs` / `cacheScope`——告诉 Client 这份 capability catalog 可以缓存多久、缓存范围是什么。所以不必每一次 agent run 都重新 `tools/list`，首次获取后走缓存。

```
            MCP Client
                │
     ┌──────────┴──────────┐
     ↓                     ↓
Capability Discovery    Invocation
     │                     │
     ↓                     ↓
list_tools()          call_tool()
list_resources()      read_resource()
list_prompts()        get_prompt()
     │                     │
     └──────────┬──────────┘
                ↓
          Agent Host
```

### call_tool()：执行能力

假设 LLM 决定调用 `get_order`，生成了 `{"name": "get_order", "arguments": {"order_id": "O12345"}}`。Agent Host **不应该**直接 `get_order("O12345")`，而是通过 MCP：

```python
result = await client.call_tool("get_order", {"order_id": "O12345"})
```

```
LLM ─tool call─▶ Host ─call_tool─▶ MCP Client ─tools/call─▶ MCP Server
                                                             │ execute
LLM ◀─result─── Host ◀────────── MCP Client ◀──result───── Backend/API/DB
```

`call_tool()` 返回 `CallToolResult`，主要看三个字段：

| 属性 | 消费者 | 说明 |
|---|---|---|
| `result.content` | **模型** | 内容块列表（文本、图片、音频、资源等） |
| `result.structured_content` | **应用/代码** | 结构化数据，匹配 `output_schema` |
| `result.is_error` | **两者** | 调用是否失败 |

```json
// content —— 给模型读的自然语言
{"content": [{"type": "text", "text": "Order O12345 was processed under product P001."}]}

// structured_content —— 给应用直接消费的结构化数据
{"structuredContent": {"order_id": "O12345", "processed": true, "amount": 1200, "product_id": "P001"}}
```

Workflow 可以直接 `result.structured_content["processed"]`，而不是让 LLM 自己解析 "The order appears to be processed…" 这句话。

### 生命周期 + 发现 + 执行串起来

```
             MCP Client Lifecycle
                     │
                     ▼
             Client 创建
                     │
                     ▼
            async with Client
                     │
                     ▼
          ┌─────────────────────┐
          │ Capability Discovery│
          └──────────┬──────────┘
                     │
                list_tools()
                     │
                     ▼
               Tool Catalog
                     │
                     ▼
              Tool Selection（Host / Router）
                     │
                     ▼
                  LLM ── tool decision
                     │
                     ▼
               call_tool()
                     │
                     ▼
              MCP Server ── Tool Execute
                     │
                     ▼
                  Result
                     │
                     ▼
             Agent ── Next Action
                     │
                     ▼
              exit async with
```

对应代码（官方 SDK 支持的**内存内连接**——传 Server 对象而非脚本路径/URL，不 spawn 子进程、不开端口、不走真实 transport，非常适合测试；v2 里 `MCPServer` 的正确 import 路径见 Day 2）：

```python
from mcp import Client
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Orders Server")

@mcp.tool()
def get_order(order_id: str) -> str:
    """Get order information."""
    return f"Order {order_id}: processed"

async def main():
    async with Client(mcp) as client:          # in-memory connection（测试用）
        # 1. 发现工具
        result = await client.list_tools()
        for tool in result.tools:
            print(tool.name, tool.description, tool.input_schema)
        # 2. 调用工具
        result = await client.call_tool("get_order", {"order_id": "O12345"})
        print(result.content)            # 模型读
        print(result.structured_content) # 应用消费
        print(result.is_error)
```

最后把三个角色区分开：

```
MCP Client  → "Server 能提供什么？"（发现）
Agent Host  → "哪些工具相关？哪些应该暴露？怎么路由？"（选择/暴露）
LLM         → "我要用哪个？"（决策）
```

**MCP Client ≠ Agent**。它知道"怎么发现和调用 MCP 能力"，但不负责替 Agent 做业务上"哪个能力最合适"的决策。

```
              Agent
                │
       ┌────────┴────────┐
       │                 │
  Reasoning        Tool Selection
       │                 │
       └────────┬────────┘
                ▼
              Host
                │
       ┌────────┴────────┐
       ▼                 ▼
  Local Tools       MCP Client
                        │
                 ┌──────┴──────┐
                 ▼             ▼
               Tool        Resource
                 │
                 ▼
            MCP Server
```

### 旧教程的坑：session + SSE + initialize 已不是核心模型

旧教程经常把 "session + SSE + initialize" 写成 MCP 的固定流程；对当前 2026-07-28 规范，这已不是正确的协议核心模型——当前规范转向 **stateless**，并使用 Streamable HTTP、可缓存的 capability lists，以及 **MRTR** 处理需要多轮交互的场景。但正如上面注脚所说：主流 SDK 的 session 型 API 仍然在跑 legacy 握手（Day 3 实验实测 2025-11-25），新旧并存——读资料、写代码都要先标注版本（Day 1 的结论再次成立）。

---

## Day 3 — Build: 手写 MCP client 直连自建 server（无 LLM，纯协议）

### Question

不经过 Inspector 的 REST/SSE 代理、直接用 SDK 的 client 侧组件连 Day 2 的 server，`initialize → list_tools → call_tool` 能完整走通吗？往返（write→read）与拒绝行为是否和 Day 2 的 Inspector 中介路径一致？

### Setup

- **SDK client 侧：** `mcp.client.stdio.stdio_client` + `ClientSession`（mcp 2.1.1 的低阶组合——显式展示握手，而不是高层 `Client(...)` 一步封装）
- **目标 server：** 自己的 `mcp_server.py`（Day 2 的 read_file + write_file，stdio）
- **无 LLM：** 无 API key、无模型、无 prompt——工具选择是硬编码的调用清单（Day 4 才换成 agent 的 tool-call 循环）
- **客户端是泛型的：** 工具集从 `list_tools` 动态发现而非硬编码，调用清单只是数据——换任何 stdio server，发现部分照常工作
- **每调用计时**（perf_counter）→ Day 6 native-vs-MCP 对比的基线数据
- 结果落盘 `day3_results_<timestamp>.json`

### 核心过程

```
1. stdio_client(StdioServerParameters(command=sys.executable,
                args=[mcp_server.py], cwd=项目根))
   → spawn server 子进程（与 Inspector 的 spawn 等价，但中间没有 REST/SSE 代理层）
2. async with ClientSession(read, write) → initialize
   → serverInfo: week08-sandbox-files 0.1.0
   → 协商协议 2025-11-25（SDK client 自己的默认版本）
   → capabilities: tools + prompts + resources
3. list_tools → 恰好 2 个工具，inputSchema 完整（与 Day 2 Inspector 看到的是同一份）
4. call_tool × 6：write → read（往返）→ overwrite → read hello.txt → 2 个拒绝用例
5. 每次调用 perf_counter 计时；结果落 day3_results_*.json
```

### Result

| Call | 结果 | 数据 |
|---|---|---|
| `write_file mcp-via-client.md` | ✅ | 4.3 ms，写入成功 |
| `read_file` 读回 | ✅ | 2.7 ms，**内容与写入完全一致（往返）** |
| `write_file` 覆盖 → `read_file` | ✅ | 1.6 ms，读到的是新内容——状态真的流过 MCP |
| `read_file hello.txt` | ✅ | 1.2 ms，返回 Day 1 沙箱文件真实内容 |
| `read_file /etc/hosts` | 🚫 **blocked** | 0.7 ms，`is_error=true` + `sandbox_escape` envelope + hint |
| `write_file binary.bin` | 🚫 **blocked** | 1.6 ms，`is_error=true` + `extension_not_allowed` envelope + hint |

**SUMMARY: 2 tools listed，6 calls，4 ok，2 blocked-as-designed**（存档 `day3_results_20260903_202508.json`）

### Discoveries

1. **client 侧与 server 侧是 SDK 的镜像。** 手写 client 不需要碰 JSON-RPC framing、request-id 匹配、通知处理——两个 context manager（`stdio_client` + `ClientSession`）+ 三个方法调用（`initialize` / `list_tools` / `call_tool`）就是全部协议代码。Day 2 说"SDK 把 server 侧的胶水接走了"，Day 3 证明 client 侧同样成立。
2. **协议版本是"每对连接"协商出来的，不是 server 的属性。** 同一个 server：Day 2 的 Inspector 提出 2025-06-18，Day 3 的 SDK client 提出自己的默认 2025-11-25——server 都接受。而且 mcp 2.1.1 的 `ClientSession` 仍然按 legacy 流程跑 `initialize` 握手（2025-11-25 是带握手的最后一版协议）——**Learning 部分"新旧并存"的结论在这里被实验证实**：新规范已发布，session 型 API 向后兼容。
3. **"被拒绝"在协议层是一次完全成功的往返。** `/etc/hosts` 和 `binary.bin` 的 JSON-RPC 层都有正常 result；业务拒绝写在 `result.is_error` + week-3 envelope 里，而不是 JSON-RPC error 里。Client 还显式区分了第三种情况：transport error（进程崩溃/超时）走异常捕获，不混进业务拒绝。**success / is_error / transport error 三态**——Day 4 的 agent 桥接直接依赖这个区分。
4. **直连 stdio 完全没有 Day 1 的代理坑。** 全程零 HTTP，死掉的 macOS 系统代理（127.0.0.1:7890）碰不到这条路径——Day 1 的 `ProxyHandler({})` 修法在这里根本不需要。"按信任边界选传输"的一个小实证。
5. **延迟数字（Day 6 的种子）。** per-call 0.7–4.3 ms；端到端 736 ms 由 server 子进程 spawn + client 解释器启动主导。Day 6 对比 native 直调时应该比 **per-call 差**，而不是端到端差——spawn 是一次性成本。
6. **泛型 client 是 Day 4 的桥接点。** 今天的 6 个调用只是数据清单；把这张清单换成 agent 的 tool-call 循环，再把 `list_tools` 的结果映射成 OpenAI tool schema，就是 Day 4。

### Surprise

- 预期"手写 MCP client 要写不少协议胶水"，实际 SDK client 侧封装和 server 侧一样完整——真正花时间的是"验证什么、怎么报告结果"，而不是协议实现。
- 预期直连 client 与 Inspector 中介路径行为一致，但两者协商出**不同的协议版本**（2025-11-25 vs 2025-06-18）——版本是协商出来的，Inspector 只是转发，不带协议语义。
- 从协议视角看，被沙箱拒绝的调用反而是"成功的调用"：`is_error` 在 result 里，不在 JSON-RPC error 里——协议层 200，业务层拒绝。

### Next question（喂给 Day 4 / Day 6）

- Day 4：`tools/list` 的 inputSchema 能直接映射成 OpenAI tool schema 吗？`is_error` envelope 转成 tool message 后，week-3 的 agent 还能不能"读 hint 自救"？
- Day 6：同一个 write→read 往返，native 直调 vs 经 MCP 的 per-call 延迟差多少？（已有 MCP 基线 0.7–4.3 ms）

---

## Day 4 — Learning: Bridging MCP to LLM tool calling
MCP 是"如何暴露和调用外部能力"的协议；LLM Tool Calling 是"模型如何表达它想调用某能力"的接口(每个模型厂商使用的schema语言会略有差异）。两者之间的鸿沟需要由 MCP Host 来填补。MCP 本身不是 LLM Tool Calling。MCP 提供的是标准化的“能力发现 + 能力调用协议”；Agent Host 负责把 MCP Tool 转换成 LLM 能理解的 tool schema，再把 LLM 产生的 tool call 转换回 MCP tools/call。bridging对于实现schema transform和call routing很重要。
所以，今天学习的核心是MCP Host。

┌─────────────────────────────────────────────────────────────────────┐
│                         MCP Host (AI Application)                    │
│                                                                      │
│  ┌──────────────┐      ┌─────────────────────┐      ┌────────────┐ │
│  │   LLM API    │◄────►│   Bridge / Adapter  │◄────►│ MCP Client │ │
│  │ (Claude/GPT) │      │  (Schema Transform   │      │  Manager   │ │
│  │              │      │   + Call Routing     │      │            │ │
│  └──────────────┘      └─────────────────────┘      └─────┬──────┘ │
│                                                           │        │
└───────────────────────────────────────────────────────────┼────────┘
                                                            │
                              ┌─────────────┐   ┌─────────┴────────┐
                              │  MCP Client │   │   MCP Client B   │
                              │   (Server A)│   │   (Server B)     │
                              └──────┬──────┘   └────────┬─────────┘
                                     │                    │
                              ┌──────▼──────┐   ┌────────▼────────┐
                              │ MCP Server A│   │  MCP Server B   │
                              │ (Filesystem)│   │   (Database)    │
                              └─────────────┘   └─────────────────┘
MCP Server
    │
    │ tools/list
    ▼
MCP Client
    │
    │ MCP Tool definitions
    ▼
Agent Host
    │
    │ 转换成 LLM Tool Schema
    ▼
LLM
    │
    │ tool_call(name, arguments)
    ▼
Agent Host
    │
    │ client.call_tool(...)
    ▼
MCP Client
    │
    │ tools/call
    ▼
MCP Server
    │
    ▼
真实系统 / API / DB / RAG / OCR

首先区分两个东西：MCP Tool ≠ LLM Tool
MCP 是“工具基础设施协议”，LLM Tool Calling 是“模型与 Agent 之间的工具决策接口”。
层	MCP	LLM Tool Calling
解决什么	Agent ↔ 外部能力的标准协议	LLM ↔ Agent 的工具决策
Tool Schema	MCP Tool	LLM provider 的 tool schema
发现	tools/list	通常由 Host 提供
调用	tools/call	LLM 返回 tool call
执行者	MCP Server	Host / Tool Executor
标准	MCP	OpenAI/Anthropic/Gemini 等各自 API 形式

为什么需要一个bridge？
因为mcp定义的工具和LLM知道的工具之间有差异。
Bridge主要做了三件事：第一件事是Schema Translation，将MCP工具定义转换为LLM工具定义；第二件事是Invocation Translation，当LLM 调 Tool 后，Host 如何把这个 Tool Call 转换成 MCP tools/call。第三件事是将MCP Result转化成LLM Tool Result，以供LLM继续推理。

Host 内部桥接层的三大职责：
1. 向上：将 MCP 能力注册到 LLM
流程：
1. Host 通过各 MCP Client 调用 tools/list
2. 收集所有 Server 的工具定义
3. Schema 转换：将 MCP 的 JSON Schema 映射为 LLM 厂商要求的工具描述格式
4. 在每次对话请求时，将转换后的工具列表作为 tools 参数传入 LLM API
例如claude：
# MCP 侧获取的工具
mcp_tools = await client.list_tools()
# tool.input_schema = {"type": "object", "properties": {"location": {"type": "string"}}, ...}

# 桥接转换：MCP → Anthropic Tool 格式
anthropic_tools = [
    {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,  # MCP 使用 JSON Schema，Claude 兼容
    }
    for tool in mcp_tools
]

# 传入 LLM
response = anthropic.messages.create(
    model="claude-opus-5",
    messages=messages,
    tools=anthropic_tools,  # ← 桥接点
    max_tokens=4096,
)
openai：
# OpenAI 的函数格式与 MCP 略有不同
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,  # 同样是 JSON Schema，但结构要求更严格
        }
    }
    for tool in mcp_tools
]
Schema 兼容性注意事项：
* MCP 使用 JSON Schema 2020-12（Pydantic 默认生成）
* OpenAI 要求 parameters 必须为 type: "object" 且包含 properties
* 某些 LLM 不支持 anyOf、allOf 或复杂嵌套；桥接层可能需要降级简化 Schema
* MCP 的 title 和 annotations 通常不传递给 LLM，只用于 UI 展示
 2. 向下：将 LLM 的 Tool Use 路由到 MCP Server
流程：
1. LLM 在响应中输出 tool_use / function_call 块
2. Host 拦截这些块，提取 name 和 arguments
3. 命名空间路由：根据工具名找到对应的 MCP Client（因为不同 Server 可能有同名工具，需要联邦策略）
4. 通过对应 Client 发起 tools/call
5. 将 MCP 的 CallToolResult 转换回 LLM 的 tool_result 格式
3. 双向：结果回注与对话连续性
工具调用不是一次性的，而是对话循环的一部分：
桥接层必须保证：
1. ID 一致性：LLM 的 tool_use ID 必须与回传的 tool_result ID 严格匹配
2. 顺序保持：多个 tool results 应按调用顺序回注（某些 LLM 对顺序敏感）
3. 错误传递：is_error=True 时，必须让 LLM 看到错误文本，使其有机会自我纠正
4. 内容截断：MCP 结果可能很长；Host 可能需要截断或摘要后再注入 LLM 上下文

MCP Client 在这里到底处于什么位置？它不是agent也不是tool router，而是适配器或者运行代理，Client 是 Python 程序与 MCP Server 对话的接口；每个协议 verb 都对应一个 async 方法。

为什么不让LLM直接调MCP：因为 LLM 根本不应该关心 MCP transport，它不需要关心怎么建立连接怎么传输，只需要知道外部资源有哪些即可。这是一个非常干净的分层设计。它实现了解耦。不管外部资源服务的存储状态设计等如何变化，都不影响LLM获取信息，它看到的仍然是get_order(order_id)，这也是MCP标准化的另一种意义的解释。

因为MCP不决定让LLM调用什么，所以tool router就变得非常重要。
生产上不会把所有的mcp tools都给LLM，而是：
                  MCP Servers
                       │
                       ▼
                  MCP Client
                       │
                 list_tools()
                       │
                       ▼
                 Tool Registry
                       │
                       ▼
                  Tool Router
                 /     |      \
                /      |       \
           relevant   maybe    irrelevant
              │        │           X
              ▼        ▼
             LLM      dynamic
这个tool router可以是：Rule-based（设定规则决定）、Embedding-based（根据用户任务和工具描述的相似度决定）、LLM-based（由模型决定），而实际生产中很可能是Hybrid混合的。
Permission Filter
       ↓
Workflow Filter
       ↓
Semantic Retrieval
       ↓
LLM

这也类似RAG，这其实是现代 Agent Architecture 的一个非常重要的思想：不要把所有东西都放进 LLM context；根据当前任务动态提供相关能力。
Context Management

Knowledge
    ↓
Retriever
    ↓
Relevant Context

Tools
    ↓
Tool Router
    ↓
Relevant Tools

Memory
    ↓
Memory Retriever
    ↓
Relevant Memories


当 Host 连接多个 MCP Server 时，可能出现工具名冲突：Server A (Filesystem): read_file, write_file  Server B (GitHub):     read_file, create_issue。需要
命名空间前缀（Prefixing）
def qualify_tool_name(server_name: str, tool_name: str) -> str:
    return f"{server_name}__{tool_name}"

# LLM 看到：filesystem__read_file, github__read_file
# 调用时 Host 解析前缀路由到对应 Client
能力描述消歧（Description Disambiguation）
Host 级去重与委托
# Host 维护工具注册表，冲突时选择优先级更高的 Server
tool_registry: dict[str, MCPClient] = {}

for client in clients:
    for tool in await client.list_tools():
        if tool.name in tool_registry:
            logger.warning(f"Tool name collision: {tool.name}")
            continue  # 跳过或按优先级覆盖
        tool_registry[tool.name] = client

MCP Server 通常用 Pydantic 生成 Schema，不同厂商支持的不同。
Streaming 与增量桥接，1增量 JSON 解析：需要处理 partial_json，在流结束时组装完整参数2 提前执行：某些框架在参数完整前就开始验证或预加载3	UI 反馈：流式显示 "正在调用工具..." 提升用户体验。

错误回复与对话鲁棒性：
桥接层必须处理多种故障模式：
| 故障            | 桥接层行为                                         | LLM 感知           |
| ------------- | --------------------------------------------- | ---------------- |
| MCP Server 断开 | 返回 `is_error=True`，"Server unavailable"       | 模型收到错误文本，可建议用户重试 |
| 工具参数验证失败      | 返回 `is_error=True`，包含具体验证错误                   | 模型可自我纠正参数        |
| 工具执行超时        | 返回 `is_error=True`，"Tool execution timed out" | 模型可建议替代方案        |
| LLM 生成非法工具名   | Host 返回 `Unknown tool: xxx` 作为 tool result    | 模型收到错误，可重新选择     |
| LLM 生成非法参数类型  | 被 MCP Server 的 Schema 验证拦截                    | 模型收到验证错误，可修正     |
关键设计：所有 MCP 层的失败都转化为对话内容（通过 is_error=True 的 tool result），而非抛出异常终止对话。这使 LLM 成为错误恢复的主体。

总结：桥接层的核心设计原则
| 原则                                     | 含义                                                   |
| -------------------------------------- | ---------------------------------------------------- |
| **Protocol Agnostic to LLM**           | MCP 不规定 LLM 如何交互；桥接层负责所有转换                           |
| **Host Owns the Loop**                 | Host 拥有模型循环，负责工具注册、路由、执行、回注的完整闭环                     |
| **Schema is the Contract**             | `input_schema` 是模型生成参数的依据，`output_schema` 是应用消费结果的依据 |
| **Errors are Content**                 | MCP 层错误通过 `is_error` + 文本内容传递给 LLM，使其能自我纠正           |
| **1:1 Client-Server, Many:1 Host-LLM** | 一个 Host 聚合多个 MCP Client 的工具，统一呈现给单个 LLM 对话           |
| **Transport Transparency**             | 无论底层是 STDIO 还是 HTTP，桥接层代码完全一致                        |

一个生产级 Orders Agent 架构：
                         User
                           │
                           ▼
                  Orders Agent Host
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
           Workflow     Memory       Tool Router
              │            │            │
              │            │            ▼
              │            │      Relevant MCP Tools
              │            │            │
              │            │            ▼
              │            │       MCP Client
              │            │            │
              │            │     ┌──────┼──────┐
              │            │     │      │      │
              │            │     ▼      ▼      ▼
              │            │    OCR   Catalog Orders
              │            │    MCP    MCP     MCP
              │            │
              ▼            ▼
             LLM ◄──── Context
              │
              ▼
          Tool Calling
              │
              ▼
          MCP Client
              │
              ▼
          MCP Server


六层模型：
Layer 1
User
│
│ task
▼

Layer 2
LLM
│
│ tool decision
▼

Layer 3
Agent Host
│
│ routing / policy / orchestration
▼

Layer 4
MCP Client
│
│ protocol translation
▼

Layer 5
MCP Server
│
│ capability implementation
▼

Layer 6
External Systems
DB / API / OCR / KB / SaaS

Bridging MCP to LLM Tool Calling 理解为：
Agent Host 将 MCP Server 通过 tools/list 暴露出来的标准化 Tool Definition，转换成 LLM Provider 所要求的 Tool Schema；当 LLM 产生 Tool Call 后，Host 再把这个调用转换成 MCP Client 的 call_tool() / tools/call 请求，并把 MCP 返回的结果重新转换成 LLM 可消费的 Tool Result。
而这正是前面学习的 MCP Client → list_tools → call_tool → Agent Tool Calling 的完整闭环。当前 MCP 规范强调 stateless protocol core，而 Python SDK v2 仍提供清晰的 Client lifecycle、list_tools() 和 call_tool() 抽象来完成这套客户端交互。

---

## Day 4 — Build: 把 agent 接上 MCP server——schema 转换 + 调用路由

### Question

`tools/list` 的 inputSchema 能直接映射成 OpenAI tool schema 吗？tool call 改走 `session.call_tool` 之后，week-3 的 agent 还能正常完成任务、并"读 hint 自救"吗？

### Setup

- **桥接文件:** `experiments/week08/day4_mcp_agent.py`。复用 week-3/week-7 的 agent 循环，只换两处（这正是"透明"的含义——agent 代码一行没动）：
  1. **`mcp_tools_to_openai()`**：`tools/list` → OpenAI tool schema（`{name, description, inputSchema}` → `{"type":"function","function":{…}}`）。参数 JSON Schema **透传**——MCP 和 OpenAI 两边都是 JSON Schema；转换只是套 envelope + 防御性归一化（缺 schema、缺 `type: object` 的 server 也能活）。
  2. **派发器替换**：`TOOL_FUNCTIONS` 字典换成 `await session.call_tool(name, args)`——模型产生的每个 tool call 都经 stdio JSON-RPC 路由到 server 子进程，返回文本原样回注为 tool message。
- **模型:** qwen3.7-plus（temperature 0.0），DashScope OpenAI 兼容端点（`trust_env=False` 绕过死代理）。
- **server:** 自建 `mcp_server.py`（Day 2 的 read_file + write_file），本次协商协议 2025-11-25。
- **任务 ×2（文件系统 = ground truth，week-4 风格打分 + trace 检查）：**
  - `t1_roundtrip`：写 `mcp-agent-notes.md` → 读回验证 → 引用 `hello.txt` 第一行（该文件**只存在于 server 侧的沙箱**，引用成功即证明数据流经 MCP）。
  - `t2_escape_denied`：读 `/etc/hosts`，要求最终回答报告 error.type / error.message / error.hint 三项，再读 `hello.txt` 证明理解了规则。
- 存档 `day4_results_*.json`（另有 `--selftest`：离线验证 listing→schema 转换，零 API 消耗）。

### 核心过程

```
1. stdio_client spawn Day-2 server 子进程 → ClientSession.initialize（2025-11-25）
2. list_tools → 2 个 MCP Tool（read_file, write_file）
3. mcp_tools_to_openai() → 2 个 OpenAI tool schema（参数 JSON Schema 透传）
4. agent 循环（week-3 同构）：
   LLM → tool_call(name, arguments)
        → session.call_tool(name, arguments)   ← 唯一的桥接点
        → CallToolResult.content 文本回注为 tool message
5. 打分：文件系统检查 + 最终回答检查 + trace 检查（is_error 是否穿过协议）
```

### Result

**t1_roundtrip（写 → 读回 → 引用沙箱既有文件）**

| 工具调用 | 结果 | 延迟 |
| --- | --- | --- |
| `write_file mcp-agent-notes.md` | ✅ 367 字节 | 6.3 ms |
| `read_file mcp-agent-notes.md` | ✅ 内容与写入一致（往返） | 4.1 ms |
| `read_file hello.txt` | ✅ 返回 server 沙箱真实内容 | 3.4 ms |

**检查 5/5 全过**；3 turns，2922 prompt / 592 completion tokens，11.8 s wall。

**t2_escape_denied（拒绝跨协议 + 读 hint 自救）**

| 运行 | 结果 |
| --- | --- |
| 第 1 版 prompt | 🚫 `/etc/hosts` → `is_error=true` + `sandbox_escape` + hint（1.4 ms）；agent 应用了 hint（随后成功读 hello.txt）。**机制检查全过**，但最终回答没有按指令报告 type/message/hint——只说 "The file was read successfully…" → **3/5** |
| prompt 收紧版（要求三条带标签输出） | 同样的拒绝穿过协议，agent **逐字报告三项**，再读 hello.txt 证明理解 → **5/5**（11.0 s，2672/360 tokens） |

**SUMMARY: 机制 10/10 全对；报告环节 3/5 → 收紧 prompt 后 5/5**（存档 `day4_results_20260908_214939.json`、`day4_results_20260908_215055.json`）

### Discoveries

1. **桥接就是两小步，agent 循环一行没动。** schema 转换 = 套 OpenAI envelope（参数 JSON Schema 透传）；路由 = 派发器换成 `session.call_tool`。week-3 的循环、week-7 的消息格式原样复用，agent 模块对 week-3 代码**零 import**——"透明"在代码上是真的。
2. **透明性是行为层面的，不是装饰性的。** agent 拿到的是与 week-3 相同的工具名、描述、结果 envelope，来自一个它看不见的子进程。t1 零新增指令即 5/5——write→read 往返改走 MCP 后，agent 毫无察觉。
3. **MCP 错误模型对 agent 生效——Day 3 的三态要求在真实 agent 里成立。** `/etc/hosts` 的拒绝以 `is_error=true` + 分类 envelope 穿过协议，agent 把它当"工具跑了、请求被拒"（不是 crash），后续调用证明它读了 hint。trace 检查确认 envelope 原样过线。
4. **发现：adaptation ≠ reporting（适应了 ≠ 报告了）。** 第 1 版 t2 机制全对，但最终回答含糊带过拒绝，没有按指令报告错误三要素；收紧 prompt 后 5/5。**桥接从来不是问题，prompt 才是**——两次运行都保留在结果文件里（审计轨迹）。
5. **SDK 命名坑：** 线上协议字段是 `inputSchema`，SDK 的 Python 对象暴露的是 `input_schema`——任何直接遍历 `tools/list` 载荷的代码都要注意。
6. **Day 6 的延迟种子：** agent 循环内的 MCP `call_tool` 延迟 1.4–6.3 ms/次（Day 3 裸 client 测得 0.7–4.3 ms）；任务 wall time（8–12 s）由 LLM turns 主导，不是 MCP——Day 6 对比 native 直调时应比 per-call 差。

### Surprise

- 预期"schema 转换要做不少适配"，实际 MCP inputSchema 与 OpenAI parameters 同是 JSON Schema，转换几乎是套壳（连 pydantic 生成的 `title` 字段都原样带过去，DashScope 照单全收）——真正花时间的是防御性归一化，而不是映射本身。
- 预期 agent 会被拒绝用例绊住，实际它读 hint 自救得非常顺；真正翻车的是**报告环节**——模型做了正确的事，却没说清做了什么事。又一次"工具没问题、prompt 有问题"的实证（与 week-3 的 hint 设计一脉相承：模型会行动，但显式报告需要显式指令）。

### Next question（喂给 Day 5 / Day 6）

- Day 5：连一个**不是我写的**第三方 MCP server，桥接还能这么顺吗？摩擦点会出现在哪里（schema 怪癖？工具名冲突？能力缺失？）？
- Day 6：同一个 write→read 往返，native 直调 vs 经 MCP 的延迟与可靠性差多少？（MCP 侧已有 agent 循环内 1.4–6.3 ms 基线）

---

## What I learned this week

- MCP 的价值主张不是"又一个调用方式"，而是**把能力当作可发现、可协商、可复用的一等公民**：Server 自我宣告、Client 发现、Host 决策暴露什么、LLM 决定用哪个——四层各司其职。
- 架构上的三个不变量最值得记：Host owns the loop、Client owns the connection（1:1）、Server owns the capability（无感知）。
- 纸上协议和线上协议有版本差：文档已经描述 2026 stateless 新规范，但真实世界的主流实现（包括刚装的 filesystem server 和我们自己的 server）还在 2025-06-18——学 MCP 必须**标注版本**，否则新旧混读。
- 动手戳一遍远胜读十遍：一次真实的握手、一次真实的 `-32601`、一次真实的 "Access denied"，把"能力协商""路径安全"这些抽象概念变成了亲眼所见的行为。
- **SDK 把"暴露能力"的成本降到一个 decorator**：type hints → JSON Schema → LLM tool definition 全自动，JSON-RPC framing、capability 宣告被完全封装——开发者只剩一件事：设计好函数签名和描述。
- **好的工具设计在协议迁移时零成本**：week-3 的工具 import 即用，安全边界和错误契约原封不动穿过 MCP——这反过来说明"工具即契约"（description、envelope、hint）是跨协议都成立的资产。
- **MCP 的边界感**：它管"可复用的外部能力"，不该变成 Agent 内部函数调用的替代品——判断标准是"换个 Agent，这个能力还有没有独立存在的价值"。
- **协议的三态结果**：success / `is_error`（业务拒绝）/ transport error（连接层失败）——client 和 agent 都必须能区分三者，这是 Day 4 桥接的基础。
- **版本协商是逐对连接的**：同一个 server，Inspector 连是 2025-06-18，SDK client 连是 2025-11-25——"学 MCP 必须标注版本"在 client 侧再次验证。
- **MCP Client 只做"发现 + 调用"**：Client 发现"有什么"、Host 筛选"给哪些"、LLM 决定"用哪个"——三层职责分开，Client 不替 Agent 做业务决策。
- **桥接就是两个小转换，agent 循环一行没动**：MCP listing → OpenAI schema 只是套 envelope（两边参数都是 JSON Schema，直接透传），tool call 路由只是把派发器换成 `session.call_tool`——"透明"在代码上是真的：agent 对 week-3 零 import，t1 零新增指令即 5/5。
- **MCP 的错误模型在真实 agent 里成立**：`is_error` + 分类 envelope 穿过协议后，agent 能区分"工具跑了、请求被拒"与"工具崩了"，并读 hint 自救——Day 3 的三态要求在行为层面被验证。
- **adaptation ≠ reporting（适应了 ≠ 报告了）**：agent 读 hint 自救很顺，但第一次 t2 的最终回答含糊带过拒绝（3/5）；收紧 prompt 后 5/5——**工具没问题、prompt 有问题**，显式报告需要显式指令。
- **Host 的职责在桥接层被实证**：schema 转换、调用路由、结果回注全部在 Host 内闭环，LLM 完全不接触 MCP transport——"Host owns the loop"从 Day 1 的概念变成 Day 4 跑通的代码。




## day5: MCP vs. plain function calling: when the indirection pays (sharing tools across agents/apps) and when it's overhead
既然 LLM 原生就支持 Function Calling，为什么还需要 MCP？MCP 到底创造了什么价值？什么时候值得引入，什么时候只是增加了一层复杂度？
最核心的答案不是“MCP 比 Function Calling 更先进”，而是：
Function Calling 解决的是“模型如何调用一个工具”；MCP 解决的是“工具如何被标准化地发现、描述、连接、复用和治理”。
因此 MCP 的价值主要来自 解耦和复用，而不是来自一次 Tool Call 本身。
Plain Function Calling:

LLM → Function

MCP:

LLM
 ↓
Host
 ↓
MCP Client
 ↓
MCP Protocol
 ↓
MCP Server
 ↓
Function
当工具需要跨 Agent、跨 Application、跨进程、跨机器复用和独立治理时，这个“额外间接层”开始产生价值。MCP 用“间接性”换取“解耦”，它不需要知道工具真实的实现在哪里。

MCP 与“纯函数调用”（Plain Function Calling）的对比，本质上是在追问一个问题：为工具调用增加一层标准化协议（即 MCP 的间接层）的收益，是否超过了其所带来的额外复杂度与性能成本？
Plain Function Calling 是所有 LLM 工具调用的基础形态——你将工具函数暴露给模型，通过 API（如 OpenAI tools 参数）让模型“选择”并“调用”它们。关键特征是：工具函数与应用进程绑定，生命周期一致，调用开销几乎为零。
MCP 则将工具服务外部化为独立进程或远程服务，应用通过标准化协议访问。这种间接层使得工具成为可独立部署、版本管理、权限控制的微服务。

Plain Function Calling（原生函数调用）
指应用内部直接调用函数，或通过框架（如 LangChain @tool、OpenAI Assistants API、Vercel AI SDK）将函数注册到模型上下文中的模式。其特征是：
* 工具定义与执行逻辑内嵌于应用代码
* 模型通过厂商特定的 tools / functions 参数接收 Schema
* 调用是进程内（in-process）的 Python/JavaScript 函数跳转
* 无运行时发现机制：工具在 prompt 中静态声明
最大的优势：简单，对于工具数量少的情况，不会过度工程化。

MCP Indirection（协议间接层）
MCP 在函数调用之上引入了三重间接：
1. 协议间接：JSON-RPC 2.0 消息格式替代直接函数调用
2. 进程/网络间接：工具运行在独立的 MCP Server 进程中（STDIO 子进程或远程 HTTP 服务）
3. 发现间接：运行时通过 tools/list 动态查询，而非编译时硬编码
核心洞察：Function calling 是模型如何表达意图；MCP 是意图如何被可移植地执行。两者不是竞争关系，而是互补的 Phase 1 与 Phase 2。
MCP最大的优势在于tool可以独立于agent，多个agent/application不需要反复代码实现反复部署。

Indirection 何时值得（The Payoff）
MCP Value≈Reuse+Decoupling+Governance+Interoperability-Complexity-Latency-Operational Cost
当：Reuse + Decoupling + Governance很高时：MCP 值得
当：Complexity + Latency更高时：Plain Function Calling 更合理
1 跨 Agent / 跨应用共享工具
实践案例：PagerDuty 告警查询应作为 MCP Server（跨团队共享），而特定 Slack 消息格式化应保留为函数调用（仅该 Agent 使用）。
2 多语言与多模型生态
MCP Server 可以是 Python、TypeScript、Java、Rust、C# 等任意语言实现，而 Host 无需关心。一个用 Python 写的数据库 MCP Server 可以同时服务于：
* Claude Desktop（Electron/TS Host）
* Cursor（Node.js Host）
* 内部 Java Agent（Spring AI MCP Client）
* 自定义 Python Agent
Plain function calling 要求工具与 Host 同语言、同进程、同运行时。
3 安全、审计与治理边界
MCP Server 作为独立的信任边界（trust boundary），提供了函数调用无法原生实现的治理能力。示例：在 MCP Server 中内置审计日志和角色校验，而函数调用需要每个应用重复实现。
能力	Plain Function Calling	MCP
凭证隔离	应用代码直接持有 API Key / DB 密码	Server 独立管理凭证，Client 无感知
审计日志	需在应用内手动实现	Server 层统一记录所有调用
权限控制	应用级（难以细化到工具级）	Server 内可实现 `user_role` 校验
速率限制	应用各自实现	Server 层统一限流
4 动态发现与运行时适配
MCP 的 tools/list + notifications/tools/list_changed 机制允许：
* 大型工具面：暴露数十到数百个工具，无需在每次对话请求中全部注入 prompt（客户端可选择性加载）
* 热更新：Server 新增/修改工具后，已连接的 Client 自动收到通知并刷新注册表
* 渐进式发现：Client 可按需分页加载工具，而非一次性全量传输
Plain function calling 中，工具 Schema 是静态注入 prompt 的，任何变更都需要重新部署 Host 应用。
5 超越 Tools 的原语：Resources 与 Prompts
MCP 不仅标准化函数调用，还定义了两种函数调用不具备的原语：
* Resources：只读数据通过 URI 寻址（file:///doc.txt、db://users/42），由 Host 决定何时附加到上下文，而非由模型猜测调用哪个工具去拉取
* Prompts：可复用的消息模板，由用户主动选择（如 /code-review 命令），实现版本化的交互工作流
函数调用仅限"模型决定做什么"（Tools）；MCP 额外支持"应用决定给模型看什么"（Resources）和"用户决定如何开始"（Prompts）。
6 MCP 还解决一个更深的问题：Ownership
比如：订单服务由订单核心系统团队维护，订单 agent 由 AI 团队维护，如果不用 mcp 直接集成，AI team 需要知道 API endpoint、Authentication、Schema、Version、Error codes、Rate limits，甚至自己维护 integration。而mcp下，capability ownership separation，能力的拥有者可以独立维护它。
7 MCP 还有一个非常重要的价值：Interoperability
MCP 还有一个非常重要的价值：Interoperability，如果每个 Agent 都直接连接业务 API，integration 会越来越碎片化。MCP 则提供一个相对统一的：每个 Host 可以把 MCP Tool 转换成自己 LLM 所需的 schema。MCP 把“工具实现”和“模型厂商”解耦了。
                    MCP Server
                        │
        ┌───────────────┼────────────────┐
        │               │                │
      Host A          Host B           Host C
        │               │                │
      LLM A           LLM B            LLM C



Indirection 何时是开销（The Cost）
MCP多了一层网络协议，比plain方案多了：serialization、transport、protocol handling、server process、client runtime、network latency。
1 性能：延迟与吞吐量的实证
MCP 的间接层引入进程边界（STDIO）或网络边界（HTTP），其性能代价是显著的：
指标	Native Function	MCP SSE (LAN)	差距
**吞吐量**	8,844 calls/sec	41 calls/sec	**215×**
**并发模式**	Thread（CPU 优化）	Process（I/O 隔离）	架构差异
延迟构成分析：
环节	Plain Function	MCP STDIO	MCP HTTP
函数调用本身	~1-2 CPU cycles	~1-2 ms	~5-50 ms
JSON 序列化/反序列化	框架内部（共享内存）	进程间管道	网络 + 协议栈
Schema 验证	可选（应用内）	强制（Server 端）	强制（Server 端）
连接管理	无	子进程生命周期	HTTP 连接池
对于延迟敏感、高频调用（如每轮对话调用 10+ 次工具、实时交互场景），MCP 的进程/网络开销可能成为瓶颈。

2 复杂度：开发、部署与调试成本
MCP需要考虑Server availability、Process lifecycle、Connection failures、Timeout、Retries、Versioning、Authentication、Observability。
并且debugging更复杂。它的错误不仅来源于工具函数内部，还来源于LLM tool call、Host mapping、MCP schema、MCP Client、HTTP、authentication、MCP Server、business API。MCP 增加了系统可扩展性，也增加了 debugging surface。
涉及schema转换，以匹配不同的LLM。
Tool数量很多的时候discovery也有成本，一旦工具很多时，必须配合Tool Registry、Tool Router、Capability filtering、Semantic retrieval、Permission filtering。
于是，架构变成：
MCP
 ↓
Capability Discovery
 ↓
Tool Registry
 ↓
Tool Retrieval / Routing
 ↓
Relevant Tools
 ↓
LLM

维度	Plain Function Calling	MCP
**首次实现**	~20-30 行代码，30 分钟	4-8 小时（Server + Client + 协议理解）
**调试体验**	标准栈追踪、IDE 断点	跨进程日志、JSON-RPC 消息检查、Inspector 工具
**部署拓扑**	单体应用	Server 进程/容器 + Host 应用 + 传输配置
**监控**	应用内 APM	需额外关注 Server 健康、连接状态、协议版本兼容
3 应用状态紧耦合场景
某些工具必须直接访问应用内存状态（如用户会话对象、UI 组件引用、前端路由状态）。MCP Server 作为独立进程/服务，天然无法访问 Host 的内部状态。
示例：
* "get_current_user_cart" 需要读取应用内存中的 session.cart
* "highlight_editor_line" 需要直接操作 VS Code 的编辑器实例
这类工具必须保留为函数调用，因为跨进程传递完整应用状态既不安全也不高效。

1. 性能开销（最显著）
* Plain Function Calling：函数调用是纳秒级。
* MCP：涉及进程间通信（IPC）或网络 RPC（HTTP），序列化/反序列化 JSON，延迟通常在毫秒到数十毫秒级。对于高频调用（如每个 User 消息触发 5 次工具调用），累积延迟显著。
2. 部署与运维复杂度
MCP 引入了独立的 Server 进程或容器，需要额外的部署、监控、故障恢复、负载均衡。Plain Function Calling 则与应用一体，简单直接。
3. 调试难度
MCP 的分布式调用链（Client → MCP Server → 可能的下游服务）使得调试变得复杂，需要分布式追踪。而 Plain Function Calling 可直接在本地 IDE 断点调试。
4. 错误处理与超时
MCP 调用可能因网络、进程崩溃、版本不兼容等原因失败，Client 必须实现重试、超时、降级策略。Plain Function Calling 异常处理更直接。
5. 协议僵化风险
虽然 MCP 有版本协商，但过度依赖特定协议特性（如特定原语）可能导致未来迁移成本。Plain Function Calling 完全由你控制接口。

你付出的是	你得到的是
进程/网络边界延迟	跨应用、跨语言、跨模型的可移植性
额外的部署与运维复杂度	统一的安全、审计、治理边界
初始开发时间（小时级 vs 分钟级）	动态发现、热更新、资源与提示原语
单点性能峰值（~40 vs ~8,800 TPS）	线性扩展的集成复杂度（O(n) vs O(n²)）

判断矩阵（现实中往往采用混合的架构）
特征	Plain Function Calling	MCP
单 Agent	⭐⭐⭐⭐⭐	⭐⭐
单进程	⭐⭐⭐⭐⭐	⭐⭐
Tool 很少	⭐⭐⭐⭐⭐	⭐⭐
高频低延迟 Tool	⭐⭐⭐⭐⭐	⭐⭐
Agent 私有逻辑	⭐⭐⭐⭐⭐	⭐
多 Agent 共享	⭐⭐	⭐⭐⭐⭐⭐
多 Application 共享	⭐	⭐⭐⭐⭐⭐
跨进程	⭐⭐	⭐⭐⭐⭐⭐
跨机器	⭐	⭐⭐⭐⭐⭐
独立团队维护	⭐⭐	⭐⭐⭐⭐⭐
独立部署	⭐	⭐⭐⭐⭐⭐
权限/治理要求高	⭐⭐	⭐⭐⭐⭐⭐
Tool 生命周期独立	⭐⭐	⭐⭐⭐⭐⭐
快速 Prototype	⭐⭐⭐⭐⭐	⭐⭐
系统长期扩展	⭐⭐⭐	⭐⭐⭐⭐⭐
                                     LLM
                     │
              Tool Calling
                     │
                     ▼
                Agent Host
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       Local Tool  MCP Tool   Another Tool
          │          │          │
          │          ▼          │
          │      MCP Client      │
          │          │          │
          │          ▼          │
          │      MCP Server      │
          │          │          │
          ▼          ▼          ▼
       Internal    API/DB     External SaaS
还有一个很重要的判断：Ownership > Reuse。并不是：“只要 Tool 可能被复用，就 MCP 化。”。更准确的是：如果 Tool 有独立的 capability ownership / lifecycle / security boundary，那么 MCP 的价值会更高。
“这个 capability 是否已经大到值得拥有一个独立的 protocol boundary？”
如果没有——直接 function call。
如果它已经成为一个可以被多个 Agent / App 独立消费、独立维护、独立治理的 capability——MCP 的 indirection 就开始值得。
这也与前面学的“Workflow/Host 专有能力是否 MCP 化”完全对应：Agent-specific → 留在 Host；Shared/External/Independently governed → MCP。

---

## Day 5 — Build: 连接一个不是我写的第三方 MCP server——摩擦点全记录

### Question

Day 4 的桥接对**自建** server 丝滑。换成**不是我写的**第三方 server（只通过协议认识它），桥接还能这么顺吗？摩擦点会出现在哪里？

### Setup

- **server:** `@modelcontextprotocol/server-filesystem`（官方 reference server，Day 1 装过，`npx` 启动，node 实现）——标识为 `secure-filesystem-server 0.2.0`，**14 个工具**，本次协商协议 2025-11-25。我们没写过它，client 只知道 `tools/list` 返回什么。
- **代码:** `experiments/week08/day5_third_party.py`。桥接代码（`mcp_tools_to_openai` + agent 循环）从 Day 4 **原样复用**，只改三处：server 启动参数（`npx`）、系统 prompt、任务。改动范围本身就是实验——测“桥接的复用边界”在哪里。
- **模型:** qwen3.7-plus（temperature 0.0），DashScope（`trust_env=False`）。
- **任务 ×3（文件系统 = ground truth）：**
  - `t1_discover_and_read`：列目录 → 读 `hello.txt` → 引用第一行 + 报告条目数。
  - `t2_write_roundtrip`：经第三方 server 写 `thirdparty-agent-notes.md` → 读回 → 引用标题。
  - `t3_escape_denied`：读 `/etc/hosts`，要求报告拒绝原文 + 教训，再读 `hello.txt` 证明适应。
- 存档 `day5_results_*.json`；**`friction_points` 列表是当天的交付物**——每条带 severity + live evidence + workaround，证据在运行现场采集（stderr 捕获、schema diff、拒绝文本检查），不是事后回忆。

### 核心过程

```
1. stdio_client spawn npx 子进程（npm → node 两级）→ initialize（2025-11-25）
2. list_tools → 14 个第三方工具（read_file/write_file 与自建 server 同名！）
3. mcp_tools_to_openai() → 14 个 OpenAI schema（8,747 字节，随每次 LLM 请求携带）
4. agent 循环（Day 4 同构，系统 prompt 重写为第三方契约）跑 3 个任务
5. 每个环节的摩擦证据实时采集进 friction_points
```

### Result

**11/11 checks 全过**（三个任务 4 turns、3–5 次调用、~8 s wall/任务——LLM turns 主导，与 Day 4 一致）。第一轮 9/11：2 个失败不是 agent 的错，而是我自己的打分器太严（agent 合法地使用绝对允许路径，打分器要求精确相对路径字符串）——修正打分器后 11/11，两轮结果都在审计轨迹里。

**摩擦点日志（10 条，按 severity）**

| # | Severity | 摩擦点 | 一句话 |
| --- | --- | --- | --- |
| 1 | 🔴 blocker | `prompt_contract_coupling` | Day 4 的 prompt 教了 agent 我们的 JSON envelope（读 error.hint）；第三方 server 返回纯文本 + 裸拒绝串。**prompt 是 server 专属配置**，换 server 必须重写，否则 agent 去找不存在的 hint |
| 2 | ⚠️ warning | `same_names_different_schemas` | 同名工具、不同参数名：我们的 `read_file(filepath)` vs 他们的 `read_file(path)`。为某个 server 写死的参数/包装代码换 server 就断 |
| 3 | ⚠️ warning | `error_contract_differs` | 拒绝 = `isError` + 裸字符串，没有 error.type/hint。线上证据：`had_json_envelope: false`。`isError` 是唯一跨 server 的错误信号 |
| 4 | ⚠️ warning | `tool_volume_schema_bloat` | 14 个工具 vs 我们的 2 个：8,747 字节 schema 随每次请求携带，且近义工具（read_file vs read_text_file）增加选错面 |
| 5 | ⚠️ warning | `absolute_path_convention` | 该 server 的输出/报错/`list_allowed_directories` 全是绝对路径——agent 放弃 prompt 里的相对路径指令（环境反馈压过指令），并**幻觉出漏了 `MyPycharm` 的路径**→ 2 次拒绝 → 重读 allowed dirs → 自愈 |
| 6 | ⚠️ warning | `orphan_child_tree_after_session_close` | npx 的 npm→node 孙进程链在 session close 后可能存活为孤儿，握着继承的 stderr pipe。我们的第一版 stderr 捕获（阻塞读 EOF）**永久挂起**——当天最痛的问题，工具链间接层比协议间接层更危险 |
| 7 | 🙂 annoyance | `spawn_and_call_latency` | npx spawn+init 1.4–2.0 s vs 自建 python server ~0.5 s；probe 调用 ~21 ms vs 2–5 ms |
| 8 | 🙂 annoyance | `roots_capability_not_sent` | Python SDK client 不声明 `roots` 能力，server 退回 CLI args 指定允许目录并打印到 stderr。完整 Host（Claude Desktop）会动态协商 roots |
| 9 | 🙂 annoyance | `stderr_log_leakage` | stdio 只有一个协议通道（stdout），server 日志走 stderr 漏进 Host 终端——需要自定义 errlog 捕获（我们做了） |
| 10 | 📝 note | `protocol_version_drift` | Day 1 Inspector 协商 2025-06-18，今天 SDK client 协商 2025-11-25，server 都接受。版本一致是**逐连接**属性 |

### Discoveries

1. **桥接代码零修改是当天最大的正面发现。** 换个 server 不用动 `mcp_tools_to_openai`、不用动 agent 循环——所有摩擦都发生在桥接**之外**：prompt、schema 假设、进程生命周期、错误解析。这反过来划出了桥接的复用边界：协议层通用，契约层不通用。
2. **agent 的自愈能力在裸文本契约下依然成立。** 路径幻觉 → 2 次拒绝 → 重读 `list_allowed_directories` → 修正，全程靠 bare-text 拒绝信息。但代价是 2 次额外调用——绝对路径约定（长路径转录）天然诱发这类错误，相对路径不会。
3. **第三方的错误契约让 Day 4 的“读 hint 自救”prompt 变成空话。** 没有 hint 可读——说明“工具即契约”里最容易断的是错误契约：我们自己设计了 envelope，第三方没有义务遵守。
4. **进程管理是 stdio MCP 的真实软肋。** 协议层好好的，`npx` 的两级进程链把 session 生命周期搞挂——孤儿进程 + 阻塞 EOF 的捕获器 = 无限挂起。修法是 stderr 非阻塞按需读（脚本最终实现）。
5. **同名工具的陷阱。** 两个 server 都有 `read_file`/`write_file`，schema 却不同——对“凭名字记忆工具”的任何代码（或人类）都是陷阱；对只看 schema 的 LLM 不是。

### Surprise

- 预期桥接要改不少代码，实际**桥接代码零修改**——摩擦全部集中在“协议之外的世界”：prompt 契约、schema 假设、进程生命周期。
- 预期第三方 server 的拒绝会难住 agent，实际 agent 靠 bare-text 拒绝 + 工具自省**自我修复了路径幻觉**——但这次“环境反馈压过 prompt 指令”（prompt 说相对路径，server 到处返回绝对路径，agent 就跟着用绝对路径）值得记住。
- 最痛的摩擦点不在协议本身，而在 **npx 的工具链间接层**：MCP 好好的，npm→node 的孙进程把整个进程管理搞挂了。学到的教训：**间接层每加一层（npx、container、代理），生命周期的责任就多一层**。

### Next question（喂给 Day 6）

- 第三方 server 单次调用 ~21 ms（node + npx），自建 python server 2–5 ms——**Day 6 的 native vs MCP 对比必须用自建 server**，否则 transport 开销与实现差异混在一起，测不出“MCP 本身”的成本。

---

## Day 6 — Experiment: 同一批工具 native 直调 vs 经 MCP——延迟与可靠性开销

### Question

同一个 write→read 往返，native 直调与经 MCP 相比，每调用多付多少延迟？可靠性有差异吗？任务级别（LLM 在环）这个开销还可见吗？

### Setup

- **两条路径，同一个实现：** native = `w3.read_file` / `w3.write_file` 直接调用；via MCP = 同一个函数经 Day-2 server（stdio JSON-RPC 往返）。同一个 week-3 函数、同一个沙箱——**唯一的变量是访问路径**。第三方 server 刻意不用（Day 5 的教训：node + 14 工具 + 不同契约会把 transport 开销与实现差异混在一起）。
- **代码:** `experiments/week08/day6_native_vs_mcp.py`。三层测量：
  1. **微基准**：25 轮 × 8 操作（读小文件 / 写 0.5 KB / 跨路径读对方写入 / 逃逸拒绝），native 与 MCP **交错执行**（时间漂移对两侧公平），3 轮 warmup 不计入统计。跨路径读还证明两条路径共享同一个沙箱状态。
  2. **固定成本**：native import vs MCP spawn+init。
  3. **agent 级**：同一任务（写 note → 读回 → 引用 `hello.txt`），每条件 5 次**交错**运行，同一 agent 循环、同一 schema，只有工具派发器不同。
- **模型:** qwen3.7-plus（temperature 0.0）。
- 存档 `day6_results_*.json`（另有 `--agent-runs 0` 只跑微基准，零 API 消耗）。

### Result

**微基准（均值延迟，25 轮/操作/条件）**

| 操作 | native | via MCP | 开销 |
| --- | --- | --- | --- |
| read_hello（读小文件） | 0.082 ms | 0.623 ms | **+0.541 ms（7.6x）** |
| write_bench（写 0.5 KB） | 0.119 ms | 0.652 ms | **+0.533 ms（5.5x）** |
| cross_read（跨路径读） | 0.086 ms | 0.621 ms | **+0.535 ms（7.2x）** |
| deny_escape（逃逸拒绝） | 0.007 ms | 0.558 ms | **+0.551 ms（79.7x）** |

- **固定成本：** native import ~60 ms vs MCP spawn+init ~465 ms（+~400 ms）——比单次调用开销大 700 倍，但每进程只付一次。
- **可靠性（两条件完全一致）：** 0 传输错误；25/25 拒绝在两侧都被正确分类为 `sandbox_escape`；50/50 跨路径读看到对方路径的写入（同一个沙箱状态，两条访问路径）；拒绝 envelope 两侧**逐字节一致**。

**agent 级（每条件 5 次，交错）**

| 条件 | checks | wall 均值（范围） | tool-call 延迟均值（max） | 传输错误 |
| --- | --- | --- | --- | --- |
| native | 15/15 | 6.94 s（5.8–8.3） | 0.677 ms（1.785） | 0 |
| via MCP | 15/15 | 6.16 s（5.1–7.4） | 4.153 ms（6.872） | 0 |

（token 用量两侧几乎一致，~3.1k/run——协议选择不影响 LLM 消耗。）

### Surprise

- **开销是常量 ~0.53 ms/调用，与操作无关。** 读、写、拒绝的开销一样——它是 JSON-RPC 编解码 + stdio 往返，不是文件工作量（逃逸拒绝零 I/O，开销照样 ~0.53 ms）。作为**比率**它是 79.7x，因为 native 拒绝在路径解析阶段就失败、几乎免费——比率会误导，绝对值才是真的。
- **在 agent 负载下两条路径都变慢 6–8x**（事件循环与活跃 LLM 会话争抢）：native 0.08→0.68 ms，MCP 0.62→4.15 ms——但 MCP 的**绝对差**稳定在 ~3.5 ms/调用，说明“MCP 成本”在负载下也是一个稳定常数。
- **任务级别开销完全不可见：** MCP 的 ~10 ms 开销（3–5 调用 × ~3.5 ms）是任务 wall time 的 **0.2%**——native 6.94 s vs MCP 6.16 s，统计上无差别，LLM 噪声（5.1–8.3 s）彻底淹没它。
- 微基准的 MCP 0.62 ms 比 Day 3/4 裸测的 0.7–4.3 ms 更稳更小——warmup + 交错轮次排除了首次调用噪声后，真实开销比“感觉”的小。

### 结论（回答 Day 5 的“何时值得”）+ Next question

- **判决：** MCP 的间接层成本 = **~0.5 ms/调用（独立）、~3.5 ms/调用（agent 负载下）、~0.5 s 启动**，可靠性零损失（0 传输错误、envelope 逐字节一致）。对跨 agent / 跨 app 共享工具的场景这是零钱；对单进程自调自己的函数是纯开销。**付费边界就是 Day 5 的 “Ownership > Reuse”：当 capability 需要独立 ownership / lifecycle / governance 时，这 ~0.5 ms 值得付。**
- **Next question（喂给 Week 9）：** 这个“微秒级工具开销 vs 秒级 LLM 开销”的量级差会在 Phase 3 反复出现——plan-then-execute 的额外规划调用、reflexion 的重试、multi-agent 的通信开销，全部是 LLM 秒级成本，工具层的毫秒开销永远不是瓶颈。**Phase 3 的性能账本应该记在 LLM turns 上，不是工具 transport 上。**
