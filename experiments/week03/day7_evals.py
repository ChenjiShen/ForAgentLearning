"""Week 3 Day 7: Week 3 eval suite — run the week03 agent against the week03 eval tasks.

This is the week-ending eval run. It measures how well the sandboxed agent
(6 tools: read_file, write_file, list_dir, run_command, web_fetch, web_search)
performs on tasks designed for week 3's capabilities.

The results compound into ``evals/last_results_week03.json`` so future weeks
can detect regressions on week-3-level tasks.

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week03/day7_evals.py
    python experiments/week03/day7_evals.py --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Import tool implementations from the week03 agent
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from agent import (  # noqa: E402
    MODEL,
    SANDBOX_ROOT,
    ALLOWED_EXECUTABLES,
    WEB_FETCH_DEFAULT_MAX_CHARS,
    WEB_SEARCH_DEFAULT_MAX_RESULTS,
    TOOLS,
    TOOL_FUNCTIONS,
    read_file,
    write_file,
    list_dir,
    run_command,
    web_fetch,
    web_search,
    _make_error,
    ErrorType,
)

from openai import OpenAI  # noqa: E402

# ============================================================================
# Eval tasks — exercises the sandboxed file/command/web tools
# ============================================================================

# Tasks 1–6 are week03 tasks that also live in evals/tasks.json.
# The 'prompt' and 'expected_substring' match tasks.json so the harness can
# cross-reference results later.

WEEK03_EVAL_TASKS = [
    # ---- Already in evals/tasks.json (cross-agent compatible) ----------------
    {
        "id": "no-tool-needed",
        "prompt": "What is the capital of France? Answer in one word.",
        "expected_substring": "Paris",
        "needed_tools": [],
    },
    # ---- Week 3 specific tasks (sandboxed file + command tools) ---------------
    {
        "id": "sandbox-list-py",
        "prompt": (
            "Use list_dir to list the sandbox root directory. How many .py "
            "files are in the sandbox root? Reply with just the number."
        ),
        "expected_substring": "3",
        "needed_tools": ["list_dir"],
    },
    {
        "id": "sandbox-read-json",
        "prompt": (
            "Read the file 'subdir/data.json' from the sandbox and tell me: "
            "what is the value of the 'project' field? Reply with just the "
            "project name."
        ),
        "expected_substring": "ForLearning",
        "needed_tools": ["read_file"],
    },
    {
        "id": "sandbox-write-verify",
        "prompt": (
            "Write a file named 'day7_test.txt' to the sandbox containing "
            "exactly 'week03 day7 evals pass'. Then read it back to verify. "
            "Reply with the exact content you read back."
        ),
        "expected_substring": "week03 day7 evals pass",
        "needed_tools": ["write_file", "read_file"],
    },
    {
        "id": "sandbox-count-lines",
        "prompt": (
            "Use run_command to count how many lines are in the file "
            "'notes.txt' in the sandbox. Reply with just the number."
        ),
        "expected_substring": "4",
        "needed_tools": ["run_command"],
    },
    {
        "id": "sandbox-grep-urllib",
        "prompt": (
            "Use run_command with grep to find which Python file in the "
            "sandbox contains the string 'urllib'. Reply with just the filename."
        ),
        "expected_substring": "fetch_url.py",
        "needed_tools": ["run_command"],
    },
    {
        "id": "sandbox-multi-step",
        "prompt": (
            "First use list_dir to see what's in the sandbox. Then use "
            "read_file to read 'notes.txt'. Tell me: what is the very first "
            "line of notes.txt? Reply with just that line."
        ),
        "expected_substring": "Week 3 sandbox notes.",
        "needed_tools": ["list_dir", "read_file"],
    },
]

# ============================================================================
# Pricing (same as week02 / day6)
# ============================================================================

_PRICING_PER_1M = {
    "qwen3.7-plus": (0.40, 1.20),
    "qwen-plus": (0.40, 1.20),
    "qwen-max": (1.60, 6.40),
    "qwen-turbo": (0.05, 0.20),
    "deepseek-chat": (0.27, 1.10),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def _model_prices() -> tuple[float, float]:
    table_in, table_out = _PRICING_PER_1M.get(MODEL, (0.0, 0.0))
    price_in = float(os.getenv("INPUT_PRICE_PER_1M", table_in))
    price_out = float(os.getenv("OUTPUT_PRICE_PER_1M", table_out))
    return price_in, price_out


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = _model_prices()
    return prompt_tokens / 1_000_000 * price_in + completion_tokens / 1_000_000 * price_out


# ============================================================================
# Non-interactive agent loop for evals
# ============================================================================

def run_agent_noninteractive(
    user_input: str,
    *,
    max_turns: int = 12,
) -> dict:
    """Run the week03 agent silently — no prints, no permission prompts.

    Returns a dict with: answer, tool_calls, n_turns, n_tool_calls, usage,
    estimated_cost, error.
    """
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    system_prompt = (
        "You are a helpful assistant with sandboxed file, command, and web tools. "
        f"All file paths and shell commands run inside {SANDBOX_ROOT}. "
        "File tools cannot access paths outside the sandbox. "
        f"run_command only permits: {', '.join(sorted(ALLOWED_EXECUTABLES))} "
        "(no shell — pipes/semicolons unsupported). "
        "Complete the user's task using the available tools. Be direct and "
        "concise — when you have the answer, state it clearly and stop.\n\n"
        "IMPORTANT — All tool results use a consistent JSON envelope:\n"
        '  {"status":"success"|"error", "data":{…}, "error":null|{…}, '
        '"truncated":bool, "truncated_at":int|null}\n'
        "- On error: read .error.hint for an actionable suggestion on what to "
        "try next. ALWAYS read the hint before retrying."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    tool_calls_trace: list[dict] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for turn in range(1, max_turns + 1):
        response = client.chat.completions.create(
            model=MODEL,
            tools=TOOLS,
            messages=messages,
        )

        if response.usage:
            usage_totals["prompt_tokens"] += response.usage.prompt_tokens or 0
            usage_totals["completion_tokens"] += response.usage.completion_tokens or 0
            usage_totals["total_tokens"] += response.usage.total_tokens or 0

        choice = response.choices[0]
        msg = choice.message

        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            return {
                "answer": msg.content or "[no text]",
                "tool_calls": tool_calls_trace,
                "n_turns": turn,
                "n_tool_calls": len(tool_calls_trace),
                "usage": dict(usage_totals),
                "estimated_cost": _estimate_cost(
                    usage_totals["prompt_tokens"], usage_totals["completion_tokens"]
                ),
                "error": None,
            }

        if choice.finish_reason == "length":
            return {
                "answer": "[truncated — max_tokens limit]",
                "tool_calls": tool_calls_trace,
                "n_turns": turn,
                "n_tool_calls": len(tool_calls_trace),
                "usage": dict(usage_totals),
                "estimated_cost": _estimate_cost(
                    usage_totals["prompt_tokens"], usage_totals["completion_tokens"]
                ),
                "error": "finish_reason=length",
            }

        # Execute tool calls — without confirm_callback (auto-approve)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                output = _make_error(
                    ErrorType.INVALID_INPUT,
                    f"Invalid JSON arguments: {e}",
                )
                tool_calls_trace.append({
                    "name": fn_name,
                    "arguments": tc.function.arguments,
                    "error": f"JSON parse: {e}",
                })
            else:
                try:
                    fn = TOOL_FUNCTIONS[fn_name]
                    # run_command without confirm_callback → no permission prompts
                    output = fn(args)
                except Exception as e:
                    output = _make_error(ErrorType.UNKNOWN, str(e))

                tool_calls_trace.append({
                    "name": fn_name,
                    "arguments": args,
                    "error": None,
                })

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

    return {
        "answer": f"[Agent stopped after {max_turns} turns — possible loop.]",
        "tool_calls": tool_calls_trace,
        "n_turns": max_turns,
        "n_tool_calls": len(tool_calls_trace),
        "usage": dict(usage_totals),
        "estimated_cost": _estimate_cost(
            usage_totals["prompt_tokens"], usage_totals["completion_tokens"]
        ),
        "error": "max_turns_exceeded",
    }


# ============================================================================
# Scoring
# ============================================================================

def score_run(task: dict) -> dict:
    """Run one task once and return a scored result record."""
    try:
        trace = run_agent_noninteractive(task["prompt"])
        answer = trace["answer"]
        expected = task["expected_substring"]
        passed = expected in answer

        tool_names = [tc["name"] for tc in trace["tool_calls"]]

    except Exception as e:
        answer = f"[exception: {e}]"
        passed = False
        tool_names = []
        trace = {
            "answer": answer,
            "tool_calls": [],
            "n_turns": 0,
            "n_tool_calls": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "estimated_cost": 0.0,
            "error": str(e),
        }

    return {
        "task_id": task["id"],
        "pass": passed,
        "answer": answer,
        "tool_names": tool_names,
        "n_turns": trace["n_turns"],
        "n_tool_calls": trace["n_tool_calls"],
        "prompt_tokens": trace["usage"]["prompt_tokens"],
        "completion_tokens": trace["usage"]["completion_tokens"],
        "total_tokens": trace["usage"]["total_tokens"],
        "estimated_cost": trace["estimated_cost"],
        "error": trace.get("error"),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Runs per task (default 5).",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="*",
        help="Task IDs to run (default: all).",
    )
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    tasks = WEEK03_EVAL_TASKS
    if args.tasks:
        tasks = [t for t in WEEK03_EVAL_TASKS if t["id"] in args.tasks]
        if not tasks:
            raise SystemExit(f"No tasks matched: {args.tasks}")

    print(f"Model:    {MODEL}")
    print(f"Tasks:    {len(tasks)}")
    print(f"Runs/task: {args.runs}")
    print(f"Sandbox:  {SANDBOX_ROOT.resolve()}")
    print()

    all_results: list[dict] = []
    per_task_passes: dict[str, int] = {}

    for task in tasks:
        passes = 0
        for i in range(args.runs):
            r = score_run(task)
            passes += r["pass"]
            all_results.append(r)

            icon = "✅" if r["pass"] else "❌"
            err_flag = f"  ⚠️ {r['error']}" if r.get("error") else ""
            tools_str = " → ".join(r["tool_names"]) if r["tool_names"] else "(no tools)"
            print(
                f"  [{icon}] {task['id']:<28} run {i + 1}/{args.runs}"
                f"  turns={r['n_turns']}  tools=[{tools_str}]"
                f"  tokens={r['total_tokens']}{err_flag}"
            )

        per_task_passes[task["id"]] = passes
        rate = passes / args.runs
        mark = "✅" if rate == 1.0 else "❌" if rate == 0.0 else "⚠️"
        print(f"  {mark} {task['id']}: {passes}/{args.runs} passed ({rate:.0%})")
        print()

    # ---- Summary ----
    total_passed = sum(r["pass"] for r in all_results)
    total_runs = len(all_results)
    overall_rate = total_passed / total_runs if total_runs else 0.0

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total: {total_passed}/{total_runs} passed ({overall_rate:.0%})")
    print()

    # Token stats
    tokens_all = [r["total_tokens"] for r in all_results if r["total_tokens"] > 0]
    if tokens_all:
        mean_t = sum(tokens_all) / len(tokens_all)
        cost_total = sum(r["estimated_cost"] for r in all_results)
        print(f"  Avg tokens/run: {mean_t:,.0f}")
        print(f"  Total est. cost: ${cost_total:.4f}")
        print()

    # Per-task pass count
    print(f"{'task':<28} {'passed':>8}  {'rate':>6}")
    print("-" * 46)
    for task in tasks:
        p = per_task_passes[task["id"]]
        print(f"{task['id']:<28} {p:>3}/{args.runs:<3}  {p/args.runs:>5.0%}")

    # ---- Save results ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _THIS_DIR / f"day7_results_{timestamp}.json"
    report = {
        "experiment": "week03_day7_evals",
        "model": MODEL,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs_per_task": args.runs,
        "tracked_tasks": [t["id"] for t in tasks],
        "total_passed": total_passed,
        "total_runs": total_runs,
        "pass_rate": overall_rate,
        "per_task_passes": per_task_passes,
        "avg_tokens_per_run": sum(tokens_all) / len(tokens_all) if tokens_all else 0,
        "total_estimated_cost": sum(r["estimated_cost"] for r in all_results),
        "results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()