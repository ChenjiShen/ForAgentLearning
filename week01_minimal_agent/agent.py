"""Week 1: a minimal agent loop from scratch (OpenAI-compatible API).

This is the whole idea of an "agent" in ~100 lines:

    while the model wants to use tools:
        run the tools
        feed the results back

Frameworks (LangChain, the OpenAI Agents SDK, etc.) do this loop for you —
in production you'd often use one. Here we write the loop by hand because
understanding it IS the lesson.

Works with any OpenAI-compatible endpoint (OpenAI, DeepSeek, Qwen, GLM,
one-api proxies, ...). Configure via environment variables:

    export OPENAI_API_KEY=sk-...
    export OPENAI_BASE_URL=https://your-provider.com/v1   # omit for openai.com
    export MODEL_NAME=your-model-name                     # e.g. deepseek-chat

Run:
    python week01_minimal_agent/agent.py

Week 1 exercises:
    1. Run it and read the printed trace until every step makes sense.
    2. Add a new tool (e.g. read_file). Notice how much the description matters.
    3. Break a tool on purpose (raise an exception) and watch how the model
       recovers when it receives an error string as the tool result.
    4. Ask a question that needs TWO tools and watch parallel tool calls arrive
       in a single assistant message (if your provider supports them).
"""

import json
import os
from datetime import datetime

from openai import OpenAI

MODEL = os.getenv("MODEL_NAME", "qwen3.7-plus")

# ---------------------------------------------------------------------------
# 1. Tool definitions — the schema the model sees.
#    The description is prompt engineering: it decides WHEN the model calls
#    the tool, not just what it does.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function", # 可调用函数
        "function": {
            "name": "calculator",  # 工具名称
            "description": (  # 何时调用工具
                "Evaluate an arithmetic expression. Call this for any math instead "
                "of computing it yourself, e.g. '17 * 23 + 4' or '2 ** 10'."
            ),
            "parameters": { # 所需参数
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
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time. Call this whenever the user asks about 'now', 'today', or the current time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # ==================== 新增：read_file 工具 ====================
    {
        "type": "function",
        "function": {
            "name": "read_file",
            # ========== 版本 A：模糊版描述（❌ 不推荐）==========
            # "description": "Read a file.",
            # ========== 版本 B：精确版描述（✅ 推荐）==========
            "description": (
                "Read the contents of a text file from the allowed workspace directory. "
                "Call this when the user asks to see, read, or summarize the contents of a file. "
                "The path must be a relative path (no '..' or leading '/'). "
                "Allowed extensions: .txt, .md, .py, .json, .csv. "
                "DO NOT call this for binary files (.jpg, .pdf, .exe, etc.). "
                "If the file does not exist or access is denied, an error will be returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": (
                            "Relative path to the file within the workspace, e.g. 'notes.txt' or 'data/report.csv'. the path string is case-sensitive"
                            "Must NOT contain '..' or start with '/'."
                        ),
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    # ==================== 新增：search_files 工具 ====================
    # 复杂点：多个参数、可选参数、与 read_file 形成天然的工具链
    # （先 search 找到文件 → 再 read_file 读内容 → 再总结）
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search for files in the workspace by filename pattern, and optionally "
                "filter to files whose CONTENT contains a keyword. "
                "Call this when the user wants to FIND or LOCATE files (e.g. 'which files "
                "mention X', 'find all python files'). Returns a JSON list of relative "
                "paths with a matching line snippet when a keyword is given. "
                "To actually read a file's full contents afterwards, use read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern for filenames, e.g. '*.py', '*.md', 'week01*/*.py'. "
                            "Use '*' to match all allowed text files."
                        ),
                    },
                    "keyword": {
                        "type": "string",
                        "description": (
                            "Optional. Case-insensitive keyword; only files whose content "
                            "contains it are returned, together with the first matching line."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Optional. Maximum number of files to return (default 10, max 50).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    # ==================== 新增：query_orders 工具 ====================
    # 复杂点：枚举参数、数值范围过滤、排序、limit —— 模型需要把用户的自然语言
    # 问题翻译成结构化查询，聚合类问题（总收入/最大客户）还要配合 calculator。
    {
        "type": "function",
        "function": {
            "name": "query_orders",
            "description": (
                "Query a (mock) e-commerce order database. Call this for ANY question about "
                "orders, customers, revenue or sales. Supports filtering by status, customer "
                "name and amount range, plus sorting and limiting. "
                "It only RETURNS raw matching orders as JSON — it does NOT compute sums or "
                "averages, so for totals/averages fetch the orders first and then use the "
                "calculator tool on the amounts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "shipped", "delivered", "cancelled"],
                        "description": "Optional. Only return orders with this status.",
                    },
                    "customer": {
                        "type": "string",
                        "description": "Optional. Case-insensitive exact customer name, e.g. 'Alice'.",
                    },
                    "min_amount": {
                        "type": "number",
                        "description": "Optional. Only orders with amount >= this value.",
                    },
                    "max_amount": {
                        "type": "number",
                        "description": "Optional. Only orders with amount <= this value.",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["amount", "date"],
                        "description": "Optional. Sort results by this field, descending.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional. Max number of orders to return (default 20).",
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 2. Tool implementations — plain Python functions.
# ---------------------------------------------------------------------------
def calculator(expression: str) -> str:  # 定义函数
    import random
    # prob = random.random()  # just for Exercise3
    # if prob < 0.5:
    #     raise ValueError("other unkown error, just retry again")
    allowed = set("0123456789+-*/.() ")
    if not set(expression) <= allowed:
        # Returning an error message (not crashing) lets the model see what
        # went wrong and try again — exercise 3 explores this.
        raise ValueError(f"Expression contains disallowed characters: {expression!r}")
    return str(eval(expression))  # fine for a learning sandbox; never do this with untrusted input in production


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ========== 新增：带路径安全检查的 read_file ==========
def read_file(filepath: str) -> str:
    """
    读取文件内容，并进行路径安全检查：
    1. 禁止路径穿越（..）
    2. 禁止绝对路径（/ 开头）
    3. 限制文件扩展名
    4. 限制在指定工作目录内
    """
    # 1. 规范化路径
    filepath = os.path.normpath(filepath)

    # 2. 安全检查
    if ".." in filepath:
        raise ValueError(f"Path traversal not allowed: {filepath!r}")
    if filepath.startswith("/") or filepath.startswith("\\"):
        raise ValueError(f"Absolute paths not allowed: {filepath!r}")

    # 3. 扩展名白名单
    allowed_extensions = {".txt", ".md", ".py", ".json", ".csv"}
    _, ext = os.path.splitext(filepath)
    if ext.lower() not in allowed_extensions:
        raise ValueError(f"File extension '{ext}' not allowed. Allowed: {allowed_extensions}")

    # 4. 限制在工作目录内（防止符号链接等绕过手段）
    # workspace = os.path.abspath("ForLearning")  # 假设工作目录是 ./workspace
    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_path = os.path.abspath(os.path.join(workspace, filepath))

    if not full_path.startswith(workspace):
        raise ValueError(f"File must be within workspace: {workspace}")

    # 5. 读取文件
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(full_path, "r", encoding="utf-8") as f:
        return json.dumps(f.read())

# ========== 新增：search_files（工作区文件搜索）==========
def search_files(pattern: str, keyword: str | None = None, max_results: int = 10) -> str:
    """
    在工作目录内按文件名 glob 搜索，可选按内容关键词过滤：
    1. 复用 read_file 的安全约束（工作目录内、扩展名白名单）
    2. keyword 命中时附带第一条匹配行，方便模型决定接下来 read 哪个文件
    3. 返回 JSON，模型好解析
    """
    import fnmatch

    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    allowed_extensions = {".txt", ".md", ".py", ".json", ".csv"}
    max_results = max(1, min(int(max_results), 50))

    if ".." in pattern or pattern.startswith(("/", "\\")):
        raise ValueError(f"Pattern must be relative, no '..': {pattern!r}")

    results = []
    for root, dirs, files in os.walk(workspace):
        # 跳过隐藏目录和虚拟环境等噪音目录
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"venv", "__pycache__", "node_modules"}]
        for name in sorted(files):
            rel_path = os.path.relpath(os.path.join(root, name), workspace)
            _, ext = os.path.splitext(name)
            if ext.lower() not in allowed_extensions:
                continue
            # 同时支持匹配纯文件名和相对路径两种写法
            if not (fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern)):
                continue

            entry: dict[str, str | int] = {"path": rel_path}
            if keyword:
                try:
                    with open(os.path.join(root, name), "r", encoding="utf-8") as f:
                        for lineno, line in enumerate(f, 1):
                            if keyword.lower() in line.lower():
                                entry["match_line"] = lineno
                                entry["match_text"] = line.strip()[:120]
                                break
                        else:
                            continue  # 内容里没有关键词，跳过该文件
                except (UnicodeDecodeError, OSError):
                    continue

            results.append(entry)
            if len(results) >= max_results:
                return json.dumps({"count": len(results), "truncated": True, "files": results})

    if not results:
        return json.dumps({"count": 0, "files": [], "hint": "No files matched. Try a broader pattern like '*'."})
    return json.dumps({"count": len(results), "truncated": False, "files": results})


# ========== 新增：query_orders（模拟订单数据库）==========
# 故意让数据小而可手算，方便验证模型的过滤/聚合是否正确。
MOCK_ORDERS = [
    {"id": "O-1001", "customer": "Alice",   "amount": 120.50, "status": "delivered", "date": "2026-07-01"},
    {"id": "O-1002", "customer": "Bob",     "amount": 75.00,  "status": "pending",   "date": "2026-07-03"},
    {"id": "O-1003", "customer": "Alice",   "amount": 300.00, "status": "shipped",   "date": "2026-07-05"},
    {"id": "O-1004", "customer": "Charlie", "amount": 45.99,  "status": "cancelled", "date": "2026-07-06"},
    {"id": "O-1005", "customer": "Bob",     "amount": 520.25, "status": "delivered", "date": "2026-07-08"},
    {"id": "O-1006", "customer": "Diana",   "amount": 89.90,  "status": "delivered", "date": "2026-07-10"},
    {"id": "O-1007", "customer": "Charlie", "amount": 210.00, "status": "shipped",   "date": "2026-07-12"},
    {"id": "O-1008", "customer": "Alice",   "amount": 15.75,  "status": "pending",   "date": "2026-07-14"},
]


def query_orders(
    status: str | None = None,
    customer: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    sort_by: str | None = None,
    limit: int = 20,
) -> str:
    """按条件过滤/排序模拟订单，返回 JSON。故意不做求和 —— 逼模型去调 calculator。"""
    valid_statuses = {"pending", "shipped", "delivered", "cancelled"}
    if status is not None and status not in valid_statuses:
        raise ValueError(f"Invalid status {status!r}. Must be one of {sorted(valid_statuses)}")
    if sort_by is not None and sort_by not in {"amount", "date"}:
        raise ValueError(f"Invalid sort_by {sort_by!r}. Must be 'amount' or 'date'")

    rows = MOCK_ORDERS
    if status is not None:
        rows = [r for r in rows if r["status"] == status]
    if customer is not None:
        rows = [r for r in rows if r["customer"].lower() == customer.lower()]
    if min_amount is not None:
        rows = [r for r in rows if r["amount"] >= float(min_amount)]
    if max_amount is not None:
        rows = [r for r in rows if r["amount"] <= float(max_amount)]
    if sort_by is not None:
        rows = sorted(rows, key=lambda r: r[sort_by], reverse=True)
    rows = rows[: max(1, int(limit))]

    return json.dumps({"count": len(rows), "orders": rows})


TOOL_FUNCTIONS = {  # 工具名称--实际函数映射表
    "calculator": lambda inp: calculator(inp["expression"]),
    "get_current_time": lambda inp: get_current_time(),
    "read_file": lambda inp: read_file(inp["filepath"]),
    "search_files": lambda inp: search_files(
        inp["pattern"], inp.get("keyword"), inp.get("max_results", 10)
    ),
    "query_orders": lambda inp: query_orders(
        status=inp.get("status"),
        customer=inp.get("customer"),
        min_amount=inp.get("min_amount"),
        max_amount=inp.get("max_amount"),
        sort_by=inp.get("sort_by"),
        limit=inp.get("limit", 20),
    ),
}


# ---------------------------------------------------------------------------
# 3. The agent loop.
# ---------------------------------------------------------------------------
def run_agent(user_input: str) -> str:
    # The client reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
    client = OpenAI(  # 初始化客户端和对话
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the tools when they apply."},
        {"role": "user", "content": user_input},
    ]

    while True:  # 当不需要调用工具时，输出
        response = client.chat.completions.create(
            model=MODEL,
            tools=TOOLS,
            messages=messages,
        )
        choice = response.choices[0]
        msg = choice.message

        # The model's reply (including its tool_calls) must go back into the
        # history, or the next request loses the thread.
        messages.append(msg)

        if choice.finish_reason == "length":  # 如果模型因为长度被截断，返回错误信息
            return "[Response was truncated — raise max_tokens.]"

        if not msg.tool_calls:  # 如果没有请求任何工具调用，说明它已经有最终答案了，返回文本内容。
            # The model is done — no more tool calls. Return the final text.
            return msg.content or "[no text in response]"

        # The model asked for one or more tools. A single assistant message can
        # contain MULTIPLE tool_calls (parallel tool calling) — answer every
        # one, each as its own role="tool" message with the matching id.
        for call in msg.tool_calls:
            # Arguments arrive as a JSON STRING — always json.loads them,
            # and be ready for the model to produce malformed JSON.
            try:
                args = json.loads(call.function.arguments)  # 解析工具调用的参数
            except json.JSONDecodeError as e:
                args, output, failed = None, f"Error: invalid JSON arguments ({e})", True
            else:
                print(f"  → tool call: {call.function.name}({call.function.arguments})")
                try:
                    output = TOOL_FUNCTIONS[call.function.name](args)
                    failed = False
                    print(f"  ← result:    {output}")
                except Exception as e:
                    # The OpenAI format has no is_error flag — the convention
                    # is to return an error string so the model can adjust its
                    # approach instead of trusting a bad result.
                    output, failed = f"Error: {e}", True
                    print(f"  ← error:     {e}")

            messages.append(  # 将工具调用结果输入回历史
                {"role": "tool", "tool_call_id": call.id, "content": output}
            )
        # ...and loop: the next request lets the model see the results.


if __name__ == "__main__":
    # ---------------- 老的测试用例 ----------------
    # question = "What is 17 * 23 + 4? Also, what time is it right now?"
    # question = "Read the file PLANN.md and tell me what this script doing. Also, what time is it right now?"
    # question = "what time is it? and extract the hour number, then plus 10 the add 4."

    # ---------------- search_files 测试用例 ----------------
    # 测试 1（工具链）：search_files → read_file → 总结。
    # 期待：先用 pattern='*.md' + keyword='agent' 搜索，再对结果调 read_file。
    # question = "Find markdown files in the workspace that mention 'agent', then read one of them and summarize it in two sentences."

    # 测试 2（可选参数）：只给 pattern，不给 keyword。
    # 期待：单次 search_files 调用，keyword 缺省。
    # question = "List all the python files in this workspace."

    # 测试 3（空结果恢复）：故意搜一个不存在的词。
    # 期待：第一次返回 count=0，看模型是换关键词重试还是老实承认没找到。
    # question = "Which files mention the word 'blockchain'? If none, just say so."

    # ---------------- query_orders 测试用例 ----------------
    # 测试 4（枚举 + 聚合，需要两个工具）：
    # 期待：query_orders(status='delivered') → 拿到 3 单 → calculator('120.50+520.25+89.90')。
    # 正确答案：总收入 730.65。
    question = "What is the total revenue from all delivered orders? Show me which orders you counted."

    # 测试 5（多条件过滤 + 排序）：
    # 期待：query_orders(min_amount=100, sort_by='amount')，答案是 Bob 的 O-1005（520.25）。
    # question = "Among orders of at least $100, which is the largest and who placed it?"

    # 测试 6（并行/多次调用 + 比较）：
    # 期待：分别按 customer='Alice' 和 customer='Bob' 查询（可能并行），再用 calculator 求和比较。
    # Alice: 120.50+300.00+15.75=436.25, Bob: 75.00+520.25=595.25 → Bob 更多。
    # question = "Who has spent more in total, Alice or Bob? By how much?"

    # 测试 7（无效参数恢复，练习 3 的变体）：
    # 期待：如果模型传了 status='completed'（不在枚举里），工具报错，模型应改用 'delivered' 重试。
    # question = "Show me all the completed orders."

    print(f"User: {question}\n")
    answer = run_agent(question)
    print(f"\nAgent: {answer}")