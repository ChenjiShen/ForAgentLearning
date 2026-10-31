"""Week 1 Day 6 experiment: terse vs. detailed tool descriptions.

对比两种工具描述（condition）下，模型的工具调用质量：

    - terse    : 一句话敷衍描述（"Do math." / "Read a file."）
    - detailed : 精确描述（何时调用、参数格式、边界限制）

每个 condition × 每个 task 跑 N 次（默认 5），统计两个指标：

    - suitable rate（选择正确率）: 本次运行实际调用的工具集合 == 任务期望的工具集合。
      衡量 description 的 "何时调用" 引导是否有效（选错工具 / 漏调 / 多调都算不合格）。
    - success rate（执行成功率）  : 本次运行发出的所有工具调用都执行成功（参数合法、无异常）。
      没发出任何调用时记为成功（真空成立）。衡量 description 的 "如何调用" 引导。

运行：
    python experiments/week01_tool_descriptions/run.py            # 5 runs/task
    python experiments/week01_tool_descriptions/run.py --runs 3   # 快速试跑

结果打印汇总表，并保存明细到本目录 results_YYYYMMDD_HHMMSS.json。
"""

import argparse
import json
import os
from datetime import datetime

from openai import OpenAI

MODEL = "qwen3.7-plus"
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAX_ITERATIONS = 8      # Day 5 的教训：给循环装硬上限
MAX_FILE_CHARS = 1500   # 截断文件内容，避免长文件撑爆上下文（Week 2 Day 4 主题的预防针）

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# ---------------------------------------------------------------------------
# 1. 两个 condition 的工具描述。工具实现完全相同，唯一变量是 description 文本。
# ---------------------------------------------------------------------------
DESCRIPTIONS = {
    "terse": {
        "calculator": {
            "tool": "Do math.",
            "expression": "the expression",
        },
        "get_current_time": {
            "tool": "Get time.",
        },
        "read_file": {
            "tool": "Read a file.",
            "filepath": "the file path",
        },
    },
    "detailed": {
        "calculator": {
            "tool": (
                "Evaluate an arithmetic expression. Call this for any math instead "
                "of computing it yourself, e.g. '17 * 23 + 4' or '2 ** 10'."
            ),
            "expression": "A Python arithmetic expression, digits and + - * / ** ( ) only",
        },
        "get_current_time": {
            "tool": (
                "Get the current local date and time. Call this whenever the user "
                "asks about 'now', 'today', or the current time."
            ),
        },
        "read_file": {
            "tool": (
                "Read the contents of a text file from the allowed workspace directory. "
                "Call this when the user asks to see, read, or summarize the contents of a file. "
                "The path must be a relative path (no '..' or leading '/'). "
                "Allowed extensions: .txt, .md, .py, .json, .csv. "
                "DO NOT call this for binary files (.jpg, .pdf, .exe, etc.). "
                "If the file does not exist or access is denied, an error will be returned."
            ),
            "filepath": (
                "Relative path to the file within the workspace, e.g. 'notes.txt' or "
                "'data/report.csv'. The path string is case-sensitive. "
                "Must NOT contain '..' or start with '/'."
            ),
        },
    },
}


def build_tools(condition: str) -> list:
    """按 condition 生成 TOOLS schema——除 description 外两组完全一致。"""
    d = DESCRIPTIONS[condition]
    return [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": d["calculator"]["tool"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": d["calculator"]["expression"]},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": d["get_current_time"]["tool"],
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": d["read_file"]["tool"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": d["read_file"]["filepath"]},
                    },
                    "required": ["filepath"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# 2. 工具实现（与 agent.py 相同，两个 condition 共用）。
# ---------------------------------------------------------------------------
def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/.() ")
    if not set(expression) <= allowed:
        raise ValueError(f"Expression contains disallowed characters: {expression!r}")
    return str(eval(expression))  # learning sandbox only


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_file(filepath: str) -> str:
    filepath = os.path.normpath(filepath)
    if ".." in filepath:
        raise ValueError(f"Path traversal not allowed: {filepath!r}")
    if filepath.startswith("/") or filepath.startswith("\\"):
        raise ValueError(f"Absolute paths not allowed: {filepath!r}")
    allowed_extensions = {".txt", ".md", ".py", ".json", ".csv"}
    _, ext = os.path.splitext(filepath)
    if ext.lower() not in allowed_extensions:
        raise ValueError(f"File extension '{ext}' not allowed. Allowed: {allowed_extensions}")
    full_path = os.path.abspath(os.path.join(WORKSPACE, filepath))
    if os.path.commonpath([WORKSPACE, full_path]) != WORKSPACE:  # Day 3 发现的前缀漏洞的修正写法
        raise ValueError(f"File must be within workspace: {WORKSPACE}")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
    if len(content) > MAX_FILE_CHARS:
        content = content[:MAX_FILE_CHARS] + f"\n...[truncated at {MAX_FILE_CHARS} chars]"
    return content


TOOL_FUNCTIONS = {
    "calculator": lambda inp: calculator(inp["expression"]),
    "get_current_time": lambda inp: get_current_time(),
    "read_file": lambda inp: read_file(inp["filepath"]),
}


# ---------------------------------------------------------------------------
# 3. 测试任务集。expected_tools 是"本任务应该调用哪些工具"的黄金答案：
#    - 前 4 个是 should-call 任务（单工具 / 多工具并行 / 串行依赖）
#    - 后 3 个是 should-NOT-call 陷阱任务，专门放大 terse 描述的边界缺失
# ---------------------------------------------------------------------------
TASKS = [
    {
        "id": "math_basic",
        "question": "What is 847 * 362 - 519?",
        "expected_tools": ["calculator"],
        "why": "最基础的单工具任务：该调 calculator 且参数是合法算式",
    },
    {
        "id": "time_basic",
        "question": "What time is it right now?",
        "expected_tools": ["get_current_time"],
        "why": "最基础的单工具任务：该调 get_current_time",
    },
    {
        "id": "file_basic",
        "question": "Read the file PLAN.md and tell me in one sentence what it is about.",
        "expected_tools": ["read_file"],
        "why": "该调 read_file，且路径参数要用对（相对路径、大小写）",
    },
    {
        "id": "multi_parallel",
        "question": "What is 17 * 23 + 4? Also, what time is it right now?",
        "expected_tools": ["calculator", "get_current_time"],
        "why": "两个无依赖请求：漏掉任何一个工具都算选择不合格",
    },
    {
        "id": "multi_sequential",
        "question": "What time is it? Extract the hour number, then add 10, then add 4.",
        "expected_tools": ["get_current_time", "calculator"],
        "why": "串行依赖任务：terse 的 'Do math.' 是否还能让模型想到用 calculator 做加法",
    },
    {
        "id": "trap_no_tool",
        "question": "What is the capital of France?",
        "expected_tools": [],
        "why": "陷阱：常识问题，任何工具调用都是多余的（过度触发检测）",
    },
    {
        "id": "trap_binary_file",
        "question": "Open the file photo.jpg and describe what is in the picture.",
        "expected_tools": [],
        "why": "陷阱：detailed 版明确写了不许读二进制文件，模型应拒绝而不是调用后报错；terse 版没有这条边界",
    },
]


# ---------------------------------------------------------------------------
# 4. Agent 循环（返回结构化记录而不是只返回文本，供打分用）。
# ---------------------------------------------------------------------------
def run_agent(question: str, tools: list) -> dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the tools when they apply."},
        {"role": "user", "content": question},
    ]
    calls = []  # 每个元素: {"name", "arguments", "ok", "output"}

    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(model=MODEL, tools=tools, messages=messages)
        choice = response.choices[0]
        msg = choice.message
        messages.append(msg)

        if choice.finish_reason == "length":
            return {"answer": "[truncated]", "calls": calls, "finished": False}
        if not msg.tool_calls:
            return {"answer": msg.content or "", "calls": calls, "finished": True}

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments)
                output = TOOL_FUNCTIONS[name](args)
                ok = True
            except Exception as e:
                output, ok = f"Error: {e}", False
            calls.append({"name": name, "arguments": call.function.arguments, "ok": ok, "output": str(output)[:200]})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    return {"answer": "[gave up: max iterations]", "calls": calls, "finished": False}


def score_run(record: dict, expected_tools: list) -> dict:
    """从一次运行记录里算出两个指标。"""
    called_names = {c["name"] for c in record["calls"]}
    expected = set(expected_tools)
    return {
        "called": sorted(called_names),
        "suitable": called_names == expected,                                # 选择正确：不多、不少、不错
        "success": all(c["ok"] for c in record["calls"]),                    # 执行成功：发出的调用全部无错（没调用则真空成立）
        "wrong_tool": bool(called_names - expected),                         # 是否调了不该调的工具（PLAN.md 里的 wrong-tool-call）
        "num_calls": len(record["calls"]),
    }


# ---------------------------------------------------------------------------
# 5. 实验主流程：2 conditions × len(TASKS) tasks × N runs。
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="runs per task per condition")
    args = parser.parse_args()

    results = []
    for condition in ("terse", "detailed"):
        tools = build_tools(condition)
        for task in TASKS:
            for run_idx in range(1, args.runs + 1):
                try:
                    record = run_agent(task["question"], tools)
                except Exception as e:  # 系统级错误（限流/网络）不该算模型的账，单独记下
                    print(f"[{condition}] {task['id']} run {run_idx}: API ERROR {e}")
                    results.append({"condition": condition, "task": task["id"], "run": run_idx, "api_error": str(e)})
                    continue
                score = score_run(record, task["expected_tools"])
                results.append({
                    "condition": condition, "task": task["id"], "run": run_idx,
                    **score, "answer": record["answer"][:300], "calls": record["calls"],
                })
                flag = "✓" if score["suitable"] and score["success"] else "✗"
                print(f"[{condition}] {task['id']} run {run_idx}: {flag} "
                      f"called={score['called'] or ['(none)']} suitable={score['suitable']} success={score['success']}")

    # ---- 汇总表 ----
    print("\n" + "=" * 78)
    print(f"{'task':<18} | {'condition':<9} | {'suitable':>8} | {'success':>8} | {'wrong-tool':>10}")
    print("-" * 78)
    for task in TASKS:
        for condition in ("terse", "detailed"):
            rows = [r for r in results if r.get("task") == task["id"] and r.get("condition") == condition and "api_error" not in r]
            if not rows:
                continue
            n = len(rows)
            print(f"{task['id']:<18} | {condition:<9} | "
                  f"{sum(r['suitable'] for r in rows)}/{n:>5} | "
                  f"{sum(r['success'] for r in rows)}/{n:>5} | "
                  f"{sum(r['wrong_tool'] for r in rows)}/{n:>7}")
    print("-" * 78)
    for condition in ("terse", "detailed"):
        rows = [r for r in results if r.get("condition") == condition and "api_error" not in r]
        if rows:
            n = len(rows)
            print(f"{'TOTAL':<18} | {condition:<9} | "
                  f"{sum(r['suitable'] for r in rows)}/{n:>5} | "
                  f"{sum(r['success'] for r in rows)}/{n:>5} | "
                  f"{sum(r['wrong_tool'] for r in rows)}/{n:>7}")

    # ---- 保存明细 ----
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "runs_per_task": args.runs, "tasks": TASKS, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved raw results to {out_path}")


if __name__ == "__main__":
    main()
