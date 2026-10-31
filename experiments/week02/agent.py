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
# 0. Token & cost tracking (Week 2, Day 3).
#    Every LLM call returns a `usage` block (prompt/completion/total tokens).
#    An agent makes MANY calls per run (one per loop iteration), so we
#    accumulate usage across all of them and estimate the dollar cost at the
#    end. Keep this pattern in every future agent — it's how you notice a
#    runaway loop or an expensive prompt before the bill does.
#
#    Prices are USD per 1,000,000 tokens (input, output). Providers publish
#    these; override for your model via env vars, or extend the table below.
# ---------------------------------------------------------------------------
PRICING_PER_1M = {
    # model_name: (input_price, output_price)  — USD per 1M tokens
    "qwen3.7-plus": (0.40, 1.20),
    "qwen-plus": (0.40, 1.20),
    "qwen-max": (1.60, 6.40),
    "qwen-turbo": (0.05, 0.20),
    "deepseek-chat": (0.27, 1.10),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def _model_prices() -> tuple[float, float]:
    """Return (input, output) price per 1M tokens for the active model.

    Env vars INPUT_PRICE_PER_1M / OUTPUT_PRICE_PER_1M win over the table so you
    can price a new model without editing code. Unknown model -> (0, 0), which
    reports tokens but a $0.00 estimate (better than a wrong number).
    """
    table_in, table_out = PRICING_PER_1M.get(MODEL, (0.0, 0.0))
    price_in = float(os.getenv("INPUT_PRICE_PER_1M", table_in))
    price_out = float(os.getenv("OUTPUT_PRICE_PER_1M", table_out))
    return price_in, price_out


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD cost for a given token split, using the active model's prices."""
    price_in, price_out = _model_prices()
    return prompt_tokens / 1_000_000 * price_in + completion_tokens / 1_000_000 * price_out


def print_usage_summary(usage_totals: dict[str, int], n_api_calls: int) -> None:
    """Print total tokens and estimated cost for the whole run."""
    price_in, price_out = _model_prices()
    cost = estimate_cost(usage_totals["prompt_tokens"], usage_totals["completion_tokens"])
    priced = price_in > 0 or price_out > 0
    print("\n" + "─" * 60)
    print(f"Usage ({MODEL}, {n_api_calls} API call{'s' if n_api_calls != 1 else ''}):")
    print(f"  prompt (input) tokens : {usage_totals['prompt_tokens']:>8,}")
    print(f"  completion (output)   : {usage_totals['completion_tokens']:>8,}")
    print(f"  total tokens          : {usage_totals['total_tokens']:>8,}")
    if priced:
        print(f"  estimated cost        : ${cost:.6f} USD")
    else:
        print("  estimated cost        : n/a (no price set for this model; "
              "set INPUT_PRICE_PER_1M / OUTPUT_PRICE_PER_1M)")
    print("─" * 60)


# ---------------------------------------------------------------------------
# 0.5 Context-window protection (Week 2, Day 4).
#     Long tool outputs are dangerous: EVERY loop iteration re-sends the whole
#     history, so one 50K-char tool result is paid for again on every
#     subsequent API call, drowns the instructions in noise, and — if big
#     enough — hits the hard context limit and the request just errors out.
#
#     The experiment:
#       1. Run with TRUNCATE_TOOL_RESULTS=0 (default) and watch the Day-3
#          usage summary: prompt tokens explode. Crank MOCK_LOG_CHARS up
#          (e.g. 2000000) to hit the provider's real context-length error.
#       2. Run with TRUNCATE_TOOL_RESULTS=1 and compare tokens/cost/answer.
#
#     This is the NAIVE fix — blind head-truncation. It can cut off exactly
#     the part the answer needed (here: errors late in the log). Day 5
#     explores smarter tool-result design.
# ---------------------------------------------------------------------------
TRUNCATE_TOOL_RESULTS = os.getenv("TRUNCATE_TOOL_RESULTS", "0") == "1"
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "4000"))


def truncate_tool_result(output: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Naively cap a tool result at `limit` chars, with an explicit marker.

    The marker matters: it tells the MODEL the data is incomplete, so it can
    say "based on a partial log..." instead of confidently over-claiming.
    """
    if len(output) <= limit:
        return output
    marker = f"\n...[TRUNCATED: showing first {limit:,} of {len(output):,} chars]"
    return output[:limit] + marker


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
    # ==================== 新增：get_server_logs 工具（Week 2, Day 4）====================
    # 故意设计的"上下文炸弹"：一次调用返回 ~50K 字符的原始日志。
    # 真实世界的对应物：抓取整个网页 HTML、cat 一个大文件、dump 整张数据库表。
    {
        "type": "function",
        "function": {
            "name": "get_server_logs",
            "description": (
                "Fetch the RAW debug log of the (mock) application server for the last "
                "few hours. Call this when the user asks about server errors, crashes, "
                "or what happened in the logs. WARNING: returns the entire raw log, "
                "which can be very large."
            ),
            "parameters": {"type": "object", "properties": {}},
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
    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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

    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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


# ========== 新增：get_server_logs（上下文炸弹，Week 2, Day 4）==========
# 生成 ~MOCK_LOG_CHARS 字符的模拟服务器日志。要点：
# 1. 用固定种子 → 每次运行内容一致，实验可复现。
# 2. ERROR 行故意集中在日志【后半段】→ 朴素的"砍头式"截断刚好把关键信息
#    砍掉，让你亲眼看到 naive truncation 的代价（Day 5 的伏笔）。
# 3. 生成后在控制台打印 ground truth（真实 ERROR 统计），方便对照模型答案。
MOCK_LOG_CHARS = int(os.getenv("MOCK_LOG_CHARS", "50000"))


def get_server_logs() -> str:
    import random

    rng = random.Random(42)  # 固定种子：可复现
    services = ["api-gateway", "auth-svc", "order-svc", "worker-3", "cache", "db-pool"]
    info_msgs = [
        "request handled in {ms}ms",
        "health check OK",
        "cache hit ratio {pct}%",
        "scheduled job completed",
        "connection acquired from pool",
        "GET /api/v1/orders 200",
        "POST /api/v1/login 200",
    ]
    error_msgs = [  # 加权：第一条出现得最多 → “最常见错误”有唯一正确答案
        "Connection pool exhausted (max=50)",
        "Connection pool exhausted (max=50)",
        "Connection pool exhausted (max=50)",
        "OutOfMemoryError in worker-3",
        "Upstream timeout after 30s",
    ]

    lines: list[str] = []
    error_counts: dict[str, int] = {}
    total, i = 0, 0
    while total < MOCK_LOG_CHARS:
        i += 1
        ts = f"2026-07-22 {3 + i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}"
        svc = rng.choice(services)
        # 前 70% 的日志几乎全是噪音；ERROR 集中在后 30%。
        in_tail = total > MOCK_LOG_CHARS * 0.7
        if in_tail and rng.random() < 0.15:
            msg = rng.choice(error_msgs)
            error_counts[msg] = error_counts.get(msg, 0) + 1
            line = f"{ts} ERROR [{svc}] {msg}"
        else:
            level = rng.choice(["INFO", "INFO", "INFO", "DEBUG", "WARN"])
            msg = rng.choice(info_msgs).format(ms=rng.randint(3, 800), pct=rng.randint(60, 99))
            line = f"{ts} {level} [{svc}] {msg}"
        lines.append(line)
        total += len(line) + 1

    # Ground truth 只打到控制台，不进模型上下文 —— 用来人工核对模型的答案。
    n_errors = sum(error_counts.values())
    top = max(error_counts, key=error_counts.get) if error_counts else "n/a"
    print(f"  [ground truth] {len(lines)} lines, {total:,} chars, "
          f"{n_errors} ERROR lines, most common: {top!r} x{error_counts.get(top, 0)}")

    return "\n".join(lines)


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
    "get_server_logs": lambda inp: get_server_logs(),
}


# ---------------------------------------------------------------------------
# 3. The agent loop.
# ---------------------------------------------------------------------------
def run_agent(
    user_input: str,
    temperature: float | None = None,
    capture: bool = False,
) -> "str | dict":
    """Run the agent loop and return the final answer.

    temperature: forwarded to the API on every call. None → provider default.
                 This is the knob the Week 2 Day 5/6 experiments sweep.
    capture:     when True, run SILENTLY (no streaming to stdout, no usage
                 summary) and return a structured trace dict instead of just
                 the answer string:

                     {
                       "answer":       <final text>,
                       "tool_calls":   [{"name": ..., "arguments": {...}}, ...],
                       "tool_signature": "calculator|query_orders",  # names, in order
                       "usage":        {prompt/completion/total tokens},
                       "n_api_calls":  <int>,
                       "temperature":  <float | None>,
                     }

                 The experiments need the exact SEQUENCE of tool calls (not
                 just the text) to answer "did temperature change the plan?".
    """
    quiet = capture

    # The client reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
    client = OpenAI(  # 初始化客户端和对话
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the tools when they apply."},
        {"role": "user", "content": user_input},
    ]

    # Run-level usage accumulator. Every iteration of the loop is a separate
    # API call with its own usage block; we sum them here so the total reflects
    # the WHOLE run, not just the last turn.
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    n_api_calls = 0
    # Every tool the model asked for, in order — the "plan" we compare across
    # temperatures. Filled in as we execute the tool-call loop below.
    trace_tool_calls: list[dict] = []

    def _finish(answer: str) -> "str | dict":
        """Package the return value once, honouring `capture`."""
        if not quiet:
            print_usage_summary(usage_totals, n_api_calls)
        if not capture:
            return answer
        return {
            "answer": answer,
            "tool_calls": trace_tool_calls,
            "tool_signature": "|".join(tc["name"] for tc in trace_tool_calls),
            "usage": dict(usage_totals),
            "n_api_calls": n_api_calls,
            "temperature": temperature,
        }

    while True:  # 当不需要调用工具时，输出
        # stream=True: the response now arrives as a series of chunks instead
        # of one object. We don't know in advance whether THIS turn will be a
        # tool call or the final answer, so we handle both while consuming
        # the stream:
        #   - text deltas  -> print immediately (that's the ask: show tokens
        #     as they arrive for the final answer)
        #   - tool_call deltas -> accumulate quietly by index and reconstruct
        #     the full call afterwards (streaming these token-by-token is
        #     fiddly — arguments arrive as fragmented JSON — so we just
        #     rebuild the whole thing once the stream ends)
        create_kwargs: dict[str, object] = dict(
            model=MODEL,
            tools=TOOLS,
            messages=messages,
            stream=True,
            # Ask the provider to append a final chunk carrying token usage.
            # Without this, a streamed response reports no usage at all.
            stream_options={"include_usage": True},
        )
        # Only send temperature when explicitly set, so the default CLI run
        # keeps the provider's default sampling behaviour.
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        stream = client.chat.completions.create(**create_kwargs)

        content_chunks = []
        tool_call_chunks = {}  # index -> accumulating dict
        finish_reason = None
        started_printing = False
        request_usage = None  # last usage seen for THIS request (cumulative)

        for chunk in stream:
            # Usage can ride on any chunk but typically arrives on the final,
            # choices-empty one. It's cumulative for the request, so keep the
            # latest rather than adding per chunk.
            if getattr(chunk, "usage", None):
                request_usage = chunk.usage

            if not chunk.choices:
                # Some providers send a trailing chunk with an empty
                # choices list (e.g. one that only carries usage stats).
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if delta.content:
                if not started_printing:
                    if not quiet:
                        print("Agent: ", end="", flush=True)
                    started_printing = True
                if not quiet:
                    print(delta.content, end="", flush=True)  # token as it arrives
                content_chunks.append(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    entry = tool_call_chunks.setdefault(
                        tc.index,
                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function and tc.function.name:
                        entry["function"]["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        entry["function"]["arguments"] += tc.function.arguments

        if started_printing and not quiet:
            print()  # newline after the streamed final answer

        # Fold this request's usage into the run total.
        n_api_calls += 1
        if request_usage is not None:
            usage_totals["prompt_tokens"] += request_usage.prompt_tokens or 0
            usage_totals["completion_tokens"] += request_usage.completion_tokens or 0
            usage_totals["total_tokens"] += request_usage.total_tokens or 0

        full_content = "".join(content_chunks) or None
        tool_calls = [tool_call_chunks[i] for i in sorted(tool_call_chunks)] or None

        # Reassemble the assistant message dict (the stream never hands us
        # one) and push it into the history, or the next request loses the
        # thread.
        assistant_msg: dict[str, object] = {"role": "assistant", "content": full_content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if finish_reason == "length":  # 如果模型因为长度被截断，返回错误信息
            return _finish("[Response was truncated — raise max_tokens.]")

        if not tool_calls:  # 如果没有请求任何工具调用，说明它已经有最终答案了，返回文本内容。
            # The model is done — no more tool calls. Return the final text
            # (already streamed to stdout above).
            return _finish(full_content or "[no text in response]")

        # The model asked for one or more tools. A single assistant message can
        # contain MULTIPLE tool_calls (parallel tool calling) — answer every
        # one, each as its own role="tool" message with the matching id.
        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args_raw = call["function"]["arguments"]
            # Arguments arrive as a JSON STRING — always json.loads them,
            # and be ready for the model to produce malformed JSON.
            try:
                args = json.loads(fn_args_raw)  # 解析工具调用的参数
            except json.JSONDecodeError as e:
                args, output, failed = None, f"Error: invalid JSON arguments ({e})", True
            else:
                # Record the call in the trace BEFORE running it — the
                # experiments compare the plan (which tools, with which args),
                # regardless of whether the tool then succeeded.
                trace_tool_calls.append({"name": fn_name, "arguments": args})
                if not quiet:
                    print(f"  → tool call: {fn_name}({fn_args_raw})")
                try:
                    output = TOOL_FUNCTIONS[fn_name](args)
                    failed = False
                    # 控制台只打印预览 —— 但注意：模型收到的是【完整】结果，
                    # 除非下面的 truncate 开关打开。终端好看 ≠ 上下文没爆。
                    preview = output if len(output) <= 300 else f"{output[:300]}... ({len(output):,} chars total)"
                    if not quiet:
                        print(f"  ← result:    {preview}")
                except Exception as e:
                    # The OpenAI format has no is_error flag — the convention
                    # is to return an error string so the model can adjust its
                    # approach instead of trusting a bad result.
                    output, failed = f"Error: {e}", True
                    if not quiet:
                        print(f"  ← error:     {e}")
            _ = failed  # kept for readability of the success/error branches above

            # ---- Week 2, Day 4: naive protection ----
            # 在结果进入 messages 之前拦截。这是唯一的咽喉要道：进了
            # history 的每个字符，后面每一轮 API 调用都要重复付费。
            if TRUNCATE_TOOL_RESULTS and len(output) > MAX_TOOL_RESULT_CHARS:
                if not quiet:
                    print(f"  ✂ truncated: {len(output):,} -> {MAX_TOOL_RESULT_CHARS:,} chars")
                output = truncate_tool_result(output)

            messages.append(  # 将工具调用结果输入回历史
                {"role": "tool", "tool_call_id": call["id"], "content": output}
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
    # question = "What is the total revenue from all delivered orders? Show me which orders you counted."

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

    # ---------------- Week 2, Day 4：上下文爆炸实验 ----------------
    # 实验步骤（对比三次运行的 usage 汇总）：
    #   A) 观察爆炸：python experiments/week02/agent.py
    #      → 50K 字符日志原样进入 history，prompt tokens 飙升（对照 Day 3 的
    #        usage 汇总），每多一轮循环就再付一次全款。
    #   B) 观察真正的失败：MOCK_LOG_CHARS=2000000 python experiments/week02/agent.py
    #      → 超过模型上下文上限，API 直接报 context_length_exceeded 类错误。
    #   C) 朴素防护：TRUNCATE_TOOL_RESULTS=1 python experiments/week02/agent.py
    #      → 结果被截到 MAX_TOOL_RESULT_CHARS（默认 4000）。tokens 正常了，
    #        但 ERROR 集中在日志尾部，被砍头式截断切掉了 —— 看模型是诚实地说
    #        "日志被截断，我看不到错误"，还是胡编一个答案。这就是 Day 5 要
    #        解决的问题：naive truncation 省了钱，却可能丢掉唯一重要的信息。
    # 工具会在控制台打印 [ground truth]（真实 ERROR 统计），用来核对模型答案。
    question = ("The server is acting up. Check the server logs: how many ERROR lines "
                "are there, and what is the most common error message?")

    print(f"User: {question}\n")
    answer = run_agent(question)  # final answer is streamed live inside run_agent
    _ = answer  # already printed token-by-token above; kept for programmatic use