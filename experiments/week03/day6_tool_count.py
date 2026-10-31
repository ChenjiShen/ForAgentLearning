"""Week 3 Day 6: Task success vs. number of available tools.

Experiment question:
    Does adding 8 irrelevant tools to an agent that already has the 4 tools it
    needs *reduce* task success? And if so, how — wrong-tool calls, confusion,
    or just wasted tokens?

The hypothesis worth forming before you look:
    More tools = more choices = more chances to pick the wrong one. The model
    sees plausible-sounding alternatives (``query_database`` when ``read_file``
    would work, ``send_email`` when ``write_file`` is the goal) and sometimes
    picks them. The cost isn't just a wasted tool call — it's the stray result
    injected into context, potentially derailing the entire plan.

Conditions:
    A (4 tools)  — read_file, write_file, list_dir, run_command
    B (12 tools) — those 4 + web_search, web_fetch, send_email,
                   create_calendar_event, translate_text, generate_image,
                   query_database, calculate_hash

Metrics:
    - Pass rate (expected_substring in final answer)
    - Wrong-tool-call rate (calls to irrelevant tools in Condition B)
    - Total tokens consumed
    - Number of tool-calling turns

    2 conditions × 5 runs × 5 tasks = 50 agent runs by default.

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week03/day6_tool_count.py
    python experiments/week03/day6_tool_count.py --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Import tool implementations from the week03 agent.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from agent import (  # noqa: E402
    MODEL,
    SANDBOX_ROOT,
    ALLOWED_EXECUTABLES,
    WEB_FETCH_DEFAULT_MAX_CHARS,
    WEB_SEARCH_DEFAULT_MAX_RESULTS,
    read_file,
    write_file,
    list_dir,
    run_command,
    web_fetch,
    web_search,
    _make_success,
    _make_error,
    ErrorType,
)

from openai import OpenAI  # noqa: E402

# ============================================================================
# Tool schemas — Condition A (4 core tools) vs Condition B (+8 irrelevant)
# ============================================================================

CORE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file from the sandbox workspace. Call when the user "
                "asks to see or summarize a file. Path must be relative to the "
                "sandbox root (no leading '/' or '..')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Relative path, e.g. 'notes.txt' or 'subdir/data.json'.",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write or overwrite a text file in the sandbox workspace. Call when "
                "the user asks to save, create, or update a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Relative path inside sandbox."},
                    "content": {"type": "string", "description": "Full file contents to write."},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List files and subdirectories in the sandbox. Call when the user "
                "wants to browse or discover what files exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dirpath": {
                        "type": "string",
                        "description": "Relative directory path; default '.' for sandbox root.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command inside the sandbox workspace. "
                "Only allowlisted executables are permitted ("
                + ", ".join(sorted(ALLOWED_EXECUTABLES))
                + "). No shell — pipes and semicolons "
                "are not supported. Use for listing files, grepping content, or "
                "running short python scripts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command argv string, e.g. 'ls -la' or 'grep pattern file'.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# 8 irrelevant tools — they exist, they work (return plausible results), but
# they are NEVER needed for the eval tasks.  Their presence is the *only*
# difference between Condition A and Condition B.
#
# Design note: each description is written to sound like a real tool — the
# model should perceive them as legitimate options, not obvious decoys.
# ---------------------------------------------------------------------------

IRRELEVANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web via DuckDuckGo and return a list of results. "
                "Call this when the user asks to search for something online, "
                "find information, or look up facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1–10, default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a web page by URL and return extracted plain text. "
                "Call this when the user asks to read, fetch, or summarize a URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full http/https URL."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Max chars of text (default 5000).",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Send an email with a subject and body. Call this when the user "
                "asks to notify someone, share results, or send a report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address."},
                    "subject": {"type": "string", "description": "Email subject line."},
                    "body": {"type": "string", "description": "Email body text."},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": (
                "Create a calendar event. Call this when the user asks to "
                "schedule a meeting, set a reminder, or plan a date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title."},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "description": {"type": "string", "description": "Optional event details."},
                },
                "required": ["title", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": (
                "Translate text from one language to another. Call this when "
                "the user asks to translate, or when working with multi-language "
                "content. Supports 50+ languages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to translate."},
                    "target_language": {
                        "type": "string",
                        "description": "Target language, e.g. 'Chinese', 'French', 'Spanish'.",
                    },
                },
                "required": ["text", "target_language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate an image from a text description. Call this when the "
                "user asks to create a picture, diagram, chart, or visual content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Image description."},
                    "style": {
                        "type": "string",
                        "enum": ["realistic", "cartoon", "sketch", "diagram"],
                        "description": "Visual style of the generated image.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Run a read-only SQL query against the project database. Call "
                "this when the user asks to look up structured data, find records, "
                "or filter/sort information stored in tables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SELECT query (read-only).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_hash",
            "description": (
                "Compute a cryptographic hash (SHA-256 or MD5) of a file in the "
                "sandbox. Call this when the user asks to verify file integrity, "
                "detect duplicates, or compute checksums."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Relative path to the file in the sandbox.",
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": ["sha256", "md5"],
                        "description": "Hash algorithm (default sha256).",
                    },
                },
                "required": ["filepath"],
            },
        },
    },
]

# ============================================================================
# Tool-function registry — combined for the experiment agent loop
# ============================================================================

CORE_TOOL_FUNCTIONS = {
    "read_file": lambda inp: read_file(inp["filepath"]),
    "write_file": lambda inp: write_file(inp["filepath"], inp["content"]),
    "list_dir": lambda inp: list_dir(inp.get("dirpath", ".")),
    "run_command": lambda inp: run_command(inp["command"]),
}

IRRELEVANT_TOOL_FUNCTIONS = {
    "web_search": lambda inp: web_search(
        inp["query"], inp.get("max_results", WEB_SEARCH_DEFAULT_MAX_RESULTS)
    ),
    "web_fetch": lambda inp: web_fetch(
        inp["url"], inp.get("max_chars", WEB_FETCH_DEFAULT_MAX_CHARS)
    ),
    "send_email": lambda inp: _make_success(
        {
            "to": inp["to"],
            "subject": inp["subject"],
            "sent": True,
            "note": "Email sent (simulated — this is a learning experiment).",
        }
    ),
    "create_calendar_event": lambda inp: _make_success(
        {
            "title": inp["title"],
            "date": inp["date"],
            "created": True,
            "note": "Calendar event created (simulated).",
        }
    ),
    "translate_text": lambda inp: _make_success(
        {
            "original": inp["text"][:200],
            "target": inp["target_language"],
            "translated": f"[Simulated translation to {inp['target_language']}]: {inp['text'][:200]}",
            "note": "Translation simulated — no external API called.",
        }
    ),
    "generate_image": lambda inp: _make_success(
        {
            "prompt": inp["prompt"],
            "style": inp.get("style", "realistic"),
            "url": "https://example.com/generated-image.png",
            "note": "Image generated (simulated — placeholder URL).",
        }
    ),
    "query_database": lambda inp: _make_success(
        {
            "query": inp["query"],
            "rows": [],
            "note": "Database query executed (simulated). The project database is empty — "
            "this tool does not contain the data you need. Consider reading files from "
            "the sandbox instead.",
        }
    ),
    "calculate_hash": lambda inp: _make_success(
        {
            "filepath": inp["filepath"],
            "algorithm": inp.get("algorithm", "sha256"),
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "note": "Hash computed (simulated — this is a learning experiment).",
        }
    ),
}

# Combined registries for convenience
ALL_TOOL_FUNCTIONS = {**CORE_TOOL_FUNCTIONS, **IRRELEVANT_TOOL_FUNCTIONS}

# Name sets for classification during scoring
CORE_TOOL_NAMES = frozenset(CORE_TOOL_FUNCTIONS)
IRRELEVANT_TOOL_NAMES = frozenset(IRRELEVANT_TOOL_FUNCTIONS)

# ============================================================================
# Eval tasks — designed to need ONLY the 4 core tools
# ============================================================================

# Each task:
#   id            — short slug
#   prompt        — the user message sent to the agent
#   expected_substrings — list of strings, ALL of which must appear in the
#                         final answer for the task to count as a pass
#   needed_tools  — (documentation only) which core tools the task exercises

EVAL_TASKS = [
    {
        "id": "count-lines",
        "prompt": (
            "Use run_command to count how many lines are in the file 'notes.txt' "
            "in the sandbox. Reply with the exact number and nothing else."
        ),
        "expected_substrings": ["4"],
        "needed_tools": ["run_command"],
    },
    {
        "id": "read-json",
        "prompt": (
            "Read the file 'subdir/data.json' in the sandbox and tell me: "
            "what is the value of the 'project' field? Reply with the project "
            "name and nothing else."
        ),
        "expected_substrings": ["ForLearning"],
        "needed_tools": ["read_file"],
    },
    {
        "id": "list-dir",
        "prompt": (
            "Use list_dir to list the contents of the sandbox root directory. "
            "Then tell me: which files with a '.md' extension are present? "
            "List them by name."
        ),
        "expected_substrings": ["notes.md", "python_website_notes.md"],
        "needed_tools": ["list_dir"],
    },
    {
        "id": "write-then-verify",
        "prompt": (
            "Write a file named 'day6_test.txt' to the sandbox containing exactly "
            "this text: 'experiment complete'. Then use read_file to read it back "
            "and confirm the content. Reply with the exact file content you read back."
        ),
        "expected_substrings": ["experiment complete"],
        "needed_tools": ["write_file", "read_file"],
    },
    {
        "id": "grep-md-files",
        "prompt": (
            "Use run_command to find all files ending in '.md' inside the entire "
            "sandbox (including subdirectories). Reply with the exact list of "
            "filenames you found. Do NOT use web_search — the files are in the "
            "local sandbox, not online."
        ),
        "expected_substrings": ["notes.md", "python_website_notes.md"],
        "needed_tools": ["run_command"],
    },
]

# ============================================================================
# Experiment agent loop
# ============================================================================

# Pricing table (USD per 1M tokens) — same as week02 agent
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


def run_experiment_agent(
    user_input: str,
    tools: list[dict],
    tool_functions: dict,
    *,
    max_turns: int = 12,
) -> dict:
    """Run the agent with a custom tool list and return structured results.

    Unlike the interactive ``run_agent`` in agent.py, this variant:
      - Accepts an arbitrary tool-schema list (the experiment variable).
      - Runs silently — no prints, no streaming, no permission prompts.
      - Returns a dict with everything the experiment needs to score and
        analyze: answer text, every tool call, token totals, and cost.

    Parameters
    ----------
    user_input : str
        The task prompt.
    tools : list[dict]
        OpenAI tool schemas visible to the model (4 or 12).
    tool_functions : dict
        name -> callable that takes the parsed arguments dict.
    max_turns : int
        Safety valve — stop after this many LLM turns.

    Returns
    -------
    dict with keys:
        answer, tool_calls, n_turns, n_tool_calls, usage, estimated_cost,
        error (or None)
    """
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    system_prompt = (
        "You are a helpful assistant with access to sandboxed tools. "
        f"All file paths and shell commands run inside {SANDBOX_ROOT}. "
        "Complete the user's task using the available tools. Be direct and "
        "concise — when you have the answer, state it clearly and stop.\n\n"
        "IMPORTANT: Only use tools that are actually needed for the task. "
        "Read the tool descriptions carefully before calling them. If a tool "
        "sounds vaguely related but isn't the right one for the job, skip it."
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
            tools=tools,
            messages=messages,
        )

        # Accumulate usage from this API call
        if response.usage:
            usage_totals["prompt_tokens"] += response.usage.prompt_tokens or 0
            usage_totals["completion_tokens"] += response.usage.completion_tokens or 0
            usage_totals["total_tokens"] += response.usage.total_tokens or 0

        choice = response.choices[0]
        msg = choice.message

        # Push the assistant message into history
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

        # If the model is done (no tool calls), return the answer
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

        # Execute each tool call
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
                    fn = tool_functions[fn_name]
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

    # Hit max_turns — the agent never finished
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

def score_run(
    task: dict,
    tools: list[dict],
    tool_functions: dict,
    condition_label: str,
) -> dict:
    """Run one task once and return a scored result record."""
    try:
        trace = run_experiment_agent(task["prompt"], tools, tool_functions)
        answer = trace["answer"]
        # ALL expected substrings must be present
        expected = task["expected_substrings"]
        passed = all(sub in answer for sub in expected)

        # Classify tool calls
        tool_names = [tc["name"] for tc in trace["tool_calls"]]
        core_calls = sum(1 for n in tool_names if n in CORE_TOOL_NAMES)
        irrelevant_calls = sum(1 for n in tool_names if n in IRRELEVANT_TOOL_NAMES)
        unknown_calls = len(tool_names) - core_calls - irrelevant_calls

        err = trace.get("error")

    except Exception as e:
        answer = f"[exception: {e}]"
        passed = False
        tool_names = []
        core_calls = 0
        irrelevant_calls = 0
        unknown_calls = 0
        trace = {
            "answer": answer,
            "tool_calls": [],
            "n_turns": 0,
            "n_tool_calls": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "estimated_cost": 0.0,
            "error": str(e),
        }
        err = str(e)

    return {
        "condition": condition_label,
        "task_id": task["id"],
        "pass": passed,
        "answer": answer,
        "tool_names": tool_names,
        "n_core_calls": core_calls,
        "n_irrelevant_calls": irrelevant_calls,
        "n_unknown_calls": unknown_calls,
        "n_turns": trace["n_turns"],
        "n_tool_calls": trace["n_tool_calls"],
        "prompt_tokens": trace["usage"]["prompt_tokens"],
        "completion_tokens": trace["usage"]["completion_tokens"],
        "total_tokens": trace["usage"]["total_tokens"],
        "estimated_cost": trace["estimated_cost"],
        "error": err,
        "tool_call_details": trace["tool_calls"],
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
        help="Runs per (task, condition) pair (default 5).",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="*",
        help="Task IDs to run (default: all).",
    )
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    # Filter tasks if requested
    tasks = EVAL_TASKS
    if args.tasks:
        tasks = [t for t in EVAL_TASKS if t["id"] in args.tasks]
        if not tasks:
            raise SystemExit(f"No tasks matched: {args.tasks}")

    conditions = {
        "4-tools": (CORE_TOOLS, CORE_TOOL_FUNCTIONS),
        "12-tools": (CORE_TOOLS + IRRELEVANT_TOOLS, ALL_TOOL_FUNCTIONS),
    }

    print(f"Model: {MODEL}")
    print(f"Tasks: {len(tasks)}  |  Runs per cell: {args.runs}")
    print(f"Conditions: {list(conditions)}")
    print(f"Total runs: {len(tasks) * len(conditions) * args.runs}")
    print(f"Sandbox:  {SANDBOX_ROOT.resolve()}")
    print()

    # ---- Run all cells -------------------------------------------------------
    all_results: list[dict] = []
    # per-condition per-task pass counts
    per_cond_task: dict[str, dict[str, int]] = {c: {} for c in conditions}
    per_cond_pass: dict[str, int] = {c: 0 for c in conditions}
    per_cond_total: dict[str, int] = {c: 0 for c in conditions}
    # Token and tool-call accumulators
    per_cond_tokens: dict[str, list[int]] = {c: [] for c in conditions}
    per_cond_irrelevant: dict[str, list[int]] = {c: [] for c in conditions}

    for cond_label, (tools, tool_fns) in conditions.items():
        n_tools = len(tools)
        print(f"{'=' * 70}")
        print(f"CONDITION: {cond_label} ({n_tools} tools)")
        print(f"{'=' * 70}")
        for task in tasks:
            passes = 0
            for i in range(args.runs):
                r = score_run(task, tools, tool_fns, cond_label)
                passes += r["pass"]
                all_results.append(r)
                per_cond_tokens[cond_label].append(r["total_tokens"])
                per_cond_irrelevant[cond_label].append(r["n_irrelevant_calls"])

                # Print per-run detail
                icon = "✅" if r["pass"] else "❌"
                irrel = f"  irrelevant: {r['n_irrelevant_calls']}" if r["n_irrelevant_calls"] else ""
                err = f"  ⚠️ {r['error']}" if r["error"] else ""
                print(f"  [{icon}] {task['id']:<22} run {i + 1}/{args.runs}"
                      f"  turns={r['n_turns']}  tools={r['n_tool_calls']}{irrel}"
                      f"  tokens={r['total_tokens']}{err}")

            per_cond_task[cond_label][task["id"]] = passes
            per_cond_pass[cond_label] += passes
            per_cond_total[cond_label] += args.runs

            rate = passes / args.runs
            mark = "✅" if rate == 1.0 else "❌" if rate == 0.0 else "⚠️"
            print(f"  {mark} {task['id']}: {passes}/{args.runs} passed ({rate:.0%})")
        print()

    # ---- Summary tables ------------------------------------------------------

    # Table 1 — Pass rate by condition
    print(f"{'=' * 70}")
    print("PASS RATE BY CONDITION")
    print(f"{'=' * 70}")
    header = f"{'condition':<14}{'passed':>10}{'total':>8}{'rate':>9}"
    print(header)
    print("-" * len(header))
    summary_by_cond = {}
    for cond_label in conditions:
        p = per_cond_pass[cond_label]
        t = per_cond_total[cond_label]
        rate = p / t if t else 0.0
        summary_by_cond[cond_label] = {"passed": p, "total": t, "rate": rate}
        print(f"{cond_label:<14}{p:>10}{t:>8}{rate:>8.0%}")

    # Table 2 — Per-task pass counts across conditions
    print(f"\n{'=' * 70}")
    print(f"PASS COUNT PER TASK  (out of {args.runs} runs each)")
    print(f"{'=' * 70}")
    ccols = "".join(f"{c:>14}" for c in conditions)
    print(f"{'task':<22}{ccols}")
    print("-" * (22 + 14 * len(conditions)))
    for task in tasks:
        row = "".join(f"{per_cond_task[c][task['id']]:>14}" for c in conditions)
        print(f"{task['id']:<22}{row}")

    # Table 3 — Token usage comparison
    print(f"\n{'=' * 70}")
    print("TOKEN USAGE BY CONDITION  (mean ± sd)")
    print(f"{'=' * 70}")
    for cond_label in conditions:
        tokens = per_cond_tokens[cond_label]
        if tokens:
            mean_t = sum(tokens) / len(tokens)
            var_t = sum((t - mean_t) ** 2 for t in tokens) / len(tokens)
            sd_t = var_t ** 0.5
            print(f"  {cond_label:<14} {mean_t:>8,.0f} ± {sd_t:>8,.0f} tokens/run")

    # Table 4 — Irrelevant-tool-call rate (Condition B only)
    print(f"\n{'=' * 70}")
    print("IRRELEVANT-TOOL-CALL RATE (Condition B only)")
    print(f"{'=' * 70}")
    irrel_b = per_cond_irrelevant.get("12-tools", [])
    if irrel_b:
        total_irrel = sum(irrel_b)
        runs_with_irrel = sum(1 for c in irrel_b if c > 0)
        print(f"  Total irrelevant calls: {total_irrel} across {len(irrel_b)} runs")
        print(f"  Runs with ≥1 irrelevant call: {runs_with_irrel}/{len(irrel_b)} "
              f"({runs_with_irrel / len(irrel_b):.0%})")
        # Which irrelevant tools were called?
        from collections import Counter
        irrel_counter: Counter = Counter()
        for r in all_results:
            if r["condition"] == "12-tools":
                for detail in r.get("tool_call_details", []):
                    name = detail["name"]
                    if name in IRRELEVANT_TOOL_NAMES:
                        irrel_counter[name] += 1
        if irrel_counter:
            print(f"\n  Most-called irrelevant tools:")
            for name, count in irrel_counter.most_common():
                print(f"    {name:<28} {count:>3} calls")

    # ---- Delta calculation ---------------------------------------------------
    print(f"\n{'=' * 70}")
    print("DELTA: 4-tools → 12-tools")
    print(f"{'=' * 70}")
    rate_4 = summary_by_cond["4-tools"]["rate"]
    rate_12 = summary_by_cond["12-tools"]["rate"]
    delta = rate_12 - rate_4
    direction = "↓ decreased" if delta < 0 else ("↑ increased" if delta > 0 else "→ unchanged")
    print(f"  Pass rate:  {rate_4:.0%} → {rate_12:.0%}  ({delta:+.0%}, {direction})")

    tokens_4 = per_cond_tokens["4-tools"]
    tokens_12 = per_cond_tokens["12-tools"]
    if tokens_4 and tokens_12:
        mean_4 = sum(tokens_4) / len(tokens_4)
        mean_12 = sum(tokens_12) / len(tokens_12)
        delta_t = mean_12 - mean_4
        dir_t = "more" if delta_t > 0 else "fewer"
        print(f"  Avg tokens: {mean_4:,.0f} → {mean_12:,.0f}  "
              f"({delta_t:+,.0f}, {abs(delta_t / mean_4):.0%} {dir_t} per run)")

    # ---- Save results --------------------------------------------------------
    out_dir = _THIS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"day6_results_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "week03_day6_tool_count",
                "model": MODEL,
                "runs_per_cell": args.runs,
                "conditions": list(conditions),
                "tasks": [t["id"] for t in tasks],
                "summary_by_condition": summary_by_cond,
                "pass_per_task": per_cond_task,
                "token_stats": {
                    c: {
                        "mean": sum(v) / len(v) if v else 0,
                        "total": sum(v),
                        "n": len(v),
                    }
                    for c, v in per_cond_tokens.items()
                },
                "irrelevant_calls": {
                    "12-tools": {
                        "total": sum(irrel_b),
                        "runs_with_any": sum(1 for c in irrel_b if c > 0),
                        "total_runs": len(irrel_b),
                    }
                } if irrel_b else {},
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()