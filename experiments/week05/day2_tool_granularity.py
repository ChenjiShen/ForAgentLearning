"""Week 5 Day 2: Mega-tool vs. narrow tools — tool granularity comparison.

Experiment question:
    Does collapsing three file-operation tools (read_file, write_file, list_dir)
    into one mega-tool ``file_op(action, ...)`` increase error rates?  The mega-tool
    requires the model to select the correct action value AND the right parameters
    for that action — an extra decision dimension that narrow tools don't have.

Method:
    Condition A (mega-tool)   — One `file_op` tool with action ∈ {read, write, list}
    Condition B (narrow tools) — Three separate tools: read_file, write_file, list_dir

    Both conditions share identical implementations under the hood; only the tool
    schema presented to the model differs.

    Each condition is tested on 10 file-operation prompts designed to stress
    different aspects of action/tool selection.

Metrics:
    - wrong-action rate  (A-only): did the model pick the right `action`?
    - wrong-tool rate    (B-only): did the model call the right narrow tool?
    - parameter-error rate: missing required args, JSON parse failures
    - task-success rate: did the agent take the correct sequence of actions?
    - token usage: does the mega-tool's larger schema cost more?
    - turns / tool calls: how many LLM turns are needed?

    Default: 2 conditions × 5 runs × 10 prompts = 100 agent runs

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week05/day2_tool_granularity.py
    python experiments/week05/day2_tool_granularity.py --runs 3   # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Import tool implementations from the week03 agent (shared across conditions).
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_WEEK03_DIR = _THIS_DIR.parent / "week03"
sys.path.insert(0, str(_WEEK03_DIR))

from agent import (  # noqa: E402
    MODEL,
    SANDBOX_ROOT,
    read_file,
    write_file,
    list_dir,
    _make_success,
    _make_error,
    ErrorType,
)

from openai import OpenAI  # noqa: E402

# ============================================================================
# Pricing (USD per 1M tokens)
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
# Condition A — Mega-tool: one file_op(action, ...) covers everything
# ============================================================================
# Design note for the mega-tool:
#   The `action` parameter is an enum with three values.  Each action has its
#   own required parameters, but JSON Schema doesn't support conditional
#   required fields — so we mark `filepath` as always required and `content`
#   only meaningful for action="write".  The description must compensate.

FILE_OP_MEGA_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_op",
        "description": (
            "Perform a file operation inside the sandbox workspace.  "
            "Select the action you need, then supply the parameters that "
            "action requires.\n\n"
            "**action values:**\n"
            "- `read`   — Read the contents of a file.  Requires: filepath.\n"
            "- `write`  — Create or overwrite a file.  Requires: filepath, content.\n"
            "- `list`   — List files and subdirectories in a directory.  "
            "Requires: filepath (the directory path; use '.' for sandbox root).\n\n"
            "**Call when:**\n"
            "- You need to inspect, save, or discover files in the sandbox.\n"
            "- Choose the action carefully: if you pick `read` when you need "
            "to list a directory, you will get an error and waste a turn.\n\n"
            "**Do NOT call when:**\n"
            "- You need to search file CONTENTS across multiple files → use "
            "run_command with grep instead.\n"
            "- The task can be answered without touching the filesystem → "
            "answer directly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "list"],
                    "description": (
                        "The file operation to perform: "
                        "'read' to get file contents, "
                        "'write' to create/overwrite a file, "
                        "'list' to browse directory contents."
                    ),
                },
                "filepath": {
                    "type": "string",
                    "description": (
                        "Relative path inside the sandbox. "
                        "For 'read'/'write': path to the file. "
                        "For 'list': directory path (use '.' for root). "
                        "Example: 'notes.txt' or 'subdir/data.json'."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "The full text content to write.  "
                        "REQUIRED when action='write'.  "
                        "IGNORED for action='read' and action='list'."
                    ),
                },
            },
            "required": ["action", "filepath"],
        },
    },
}


def _file_op_mega_impl(action: str, filepath: str, content: str | None = None) -> str:
    """Route to the correct implementation based on `action`."""
    if action == "read":
        return read_file(filepath)
    elif action == "write":
        if content is None:
            return _make_error(
                ErrorType.INVALID_INPUT,
                "The 'content' parameter is required when action='write'. "
                "Please provide the text content to write to the file.",
            )
        return write_file(filepath, content)
    elif action == "list":
        return list_dir(filepath)
    else:
        return _make_error(
            ErrorType.INVALID_INPUT,
            f"Unknown action {action!r}.  Must be one of: read, write, list.",
        )


# ============================================================================
# Condition B — Narrow tools: read_file, write_file, list_dir
# ============================================================================

NARROW_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file from the sandbox workspace. "
                "Call when the user asks to see, view, inspect, or read "
                "the contents of a specific named file. "
                "Path must be relative to the sandbox root (no leading '/' or '..')."
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
                "Write or overwrite a text file in the sandbox workspace. "
                "Call when the user asks to save, create, write, or store "
                "content to a file. "
                "Path must stay inside the sandbox."
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
                "List files and subdirectories in the sandbox. "
                "Call when the user wants to browse or discover what files "
                "exist, or needs to find a file but doesn't know its exact path."
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
]

# ============================================================================
# Eval prompts — 10 file-operation tasks
# ============================================================================

EVAL_PROMPTS = [
    {
        "id": "P1-read-known-file",
        "prompt": (
            "Read the file 'notes.md' from the sandbox. "
            "Tell me what it says — summarize it in one sentence."
        ),
        "expected_actions": ["read"],
        "expected_tool": "read_file",
        "category": "single-read",
        "note": "Simple read — tests whether the model picks the right action/tool "
                "for the most basic file operation.",
    },
    {
        "id": "P2-write-new-file",
        "prompt": (
            "Create a file called 'greeting.txt' in the sandbox with this content:\n"
            "Hello, world!\n"
            "Welcome to the sandbox.\n"
            "Do not add anything extra — write exactly these two lines."
        ),
        "expected_actions": ["write"],
        "expected_tool": "write_file",
        "category": "single-write",
        "note": "Simple write — tests basic file creation.",
    },
    {
        "id": "P3-list-root",
        "prompt": (
            "Show me all the files and directories in the sandbox root. "
            "I want to know what's available before I start working."
        ),
        "expected_actions": ["list"],
        "expected_tool": "list_dir",
        "category": "single-list",
        "note": "Simple directory listing — tests discovery tool selection. "
                "This is where mega-tool users might accidentally pick 'read' "
                "because they equate 'seeing files' with 'read'.",
    },
    {
        "id": "P4-write-then-verify",
        "prompt": (
            "Save the following content to a file called 'config.json' in the sandbox:\n"
            '{"version": "1.0", "debug": false, "max_retries": 3}\n'
            "After writing it, read it back to confirm the file was saved correctly."
        ),
        "expected_actions": ["write", "read"],
        "expected_tool": "write_file+read_file",
        "category": "chained-write-read",
        "note": "Chained write→read — tests whether the model handles sequential "
                "file operations correctly.  In the mega-tool condition, it must "
                "switch action from 'write' to 'read' between calls.",
    },
    {
        "id": "P5-list-then-read",
        "prompt": (
            "Look at what files are in the sandbox's 'subdir' directory. "
            "Then pick one of those files and read it — tell me what it says."
        ),
        "expected_actions": ["list", "read"],
        "expected_tool": "list_dir+read_file",
        "category": "list-then-read",
        "note": "Discovery → read — the model must list first to discover files, "
                "then read one.  Tests whether the model correctly sequences "
                "discovery-then-access.",
    },
    {
        "id": "P6-overwrite-existing",
        "prompt": (
            "Overwrite the file 'notes.txt' with this new content:\n"
            "This file has been updated.\n"
            "The previous contents are gone."
        ),
        "expected_actions": ["write"],
        "expected_tool": "write_file",
        "category": "single-overwrite",
        "note": "Overwrite existing file — tests whether the model hesitates "
                "or adds unnecessary read-before-write when told to overwrite.",
    },
    {
        "id": "P7-read-nonexistent",
        "prompt": (
            "Read the file 'nonexistent_file.xyz' from the sandbox. "
            "If the file doesn't exist, tell me what error you got."
        ),
        "expected_actions": ["read"],
        "expected_tool": "read_file",
        "category": "error-nonexistent",
        "note": "Read non-existent file — tests whether the model handles "
                "tool errors gracefully and reports them to the user.  "
                "Should NOT retry with wrong action.",
    },
    {
        "id": "P8-write-subdir",
        "prompt": (
            "Create a directory structure for a Python project. "
            "Write a file called 'projects/demo/__init__.py' with this content:\n"
            "# Demo package\n"
            "__version__ = '0.1.0'\n"
            "Then read back that file to confirm it was created."
        ),
        "expected_actions": ["write", "read"],
        "expected_tool": "write_file+read_file",
        "category": "chained-subdir",
        "note": "Write to nested path — the write_file implementation handles "
                "parent directory creation automatically.  Tests whether the "
                "model handles multi-segment paths and chained verification.",
    },
    {
        "id": "P9-all-three-ops",
        "prompt": (
            "I need to do three things in the sandbox:\n"
            "1. First, browse the root directory to see what's there.\n"
            "2. Then write a file called 'experiment_log.txt' with the line "
            "'Experiment started at 09:00'.\n"
            "3. Finally, read back 'experiment_log.txt' to verify it.\n"
            "Do all three steps in order."
        ),
        "expected_actions": ["list", "write", "read"],
        "expected_tool": "list_dir+write_file+read_file",
        "category": "multi-op-sequence",
        "note": "All three operations in sequence — the hardest test.  "
                "For the mega-tool, the model must pick action='list', then "
                "action='write', then action='read' — three correct action "
                "selections in a row.",
    },
    {
        "id": "P10-trap-read-vs-list",
        "prompt": (
            "I need to see what files are in the sandbox.  Do NOT read any "
            "individual file — just show me the directory listing."
        ),
        "expected_actions": ["list"],
        "expected_tool": "list_dir",
        "category": "trap-read-vs-list",
        "note": "Explicitly says 'do NOT read any individual file' — tests "
                "whether the model selects 'list' (or list_dir) when the user "
                "clearly means browsing, not reading.  The word 'see' is a "
                "known failure trigger for narrow-tool models (they pick "
                "read_file because 'see a file' ≈ 'read it').",
    },
]


# ============================================================================
# Tool-function registries
# ============================================================================

# Condition A: mega-tool
MEGA_TOOL_FUNCTIONS = {
    "file_op": lambda inp: _file_op_mega_impl(
        inp["action"],
        inp["filepath"],
        inp.get("content"),
    ),
}

# Condition B: narrow tools
NARROW_TOOL_FUNCTIONS = {
    "read_file": lambda inp: read_file(inp["filepath"]),
    "write_file": lambda inp: write_file(inp["filepath"], inp["content"]),
    "list_dir": lambda inp: list_dir(inp.get("dirpath", ".")),
}


# ============================================================================
# Agent loop (non-interactive — no prints, no permission prompts)
# ============================================================================


def run_experiment_agent(
    user_input: str,
    tools: list[dict],
    tool_functions: dict,
    *,
    max_turns: int = 12,
) -> dict:
    """Run the agent and return structured results.

    Returns a dict with: answer, tool_calls (list of {name, args, error}),
    n_turns, n_tool_calls, usage, estimated_cost, error.
    """
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    system_prompt = (
        "You are a helpful assistant with access to sandboxed file-operation "
        "tools. All file paths operate inside the sandbox workspace. "
        "Complete the user's task using the available tools.\n\n"
        "IMPORTANT:\n"
        "- Read each tool's description carefully before calling it.\n"
        "- Select the correct tool (or action) for what the user wants.\n"
        "- If a tool returns an error, read the error message and adjust.\n"
        "- When you have the answer, state it clearly and stop calling tools."
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

        if response.usage:
            usage_totals["prompt_tokens"] += response.usage.prompt_tokens or 0
            usage_totals["completion_tokens"] += response.usage.completion_tokens or 0
            usage_totals["total_tokens"] += response.usage.total_tokens or 0

        choice = response.choices[0]
        msg = choice.message

        # Push assistant message into history
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

        # Model is done — return the answer
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

    # Hit max_turns
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
# Scoring — metric definitions
# ============================================================================


def score_run_for_mega(trace: dict, expected_actions: list[str]) -> dict:
    """Score a mega-tool (Condition A) run.

    The mega-tool always calls `file_op`, so the key question is whether the
    `action` parameter was correct.  We also check for parameter errors.
    """
    calls = trace.get("tool_calls", [])
    actual_actions: list[str] = []
    param_errors = 0
    json_errors = 0

    for tc in calls:
        if tc.get("error") and "JSON parse" in str(tc["error"]):
            json_errors += 1
            continue
        args = tc.get("arguments", {})
        action = args.get("action", "")
        actual_actions.append(action)

        # Check: did they provide `content` when action is `write`?
        if action == "write" and not args.get("content"):
            param_errors += 1
        # Check: did they provide a non-empty filepath?
        if not args.get("filepath"):
            param_errors += 1

    # Wrong-action rate: how many calls had the wrong action?
    n_actions = len(actual_actions)
    if n_actions == 0:
        wrong_action_rate = 0.0
    else:
        # Compare actual vs expected actions, padding expected if needed
        wrong_count = 0
        for i, actual in enumerate(actual_actions):
            expected = expected_actions[i] if i < len(expected_actions) else None
            if expected is not None and actual != expected:
                wrong_count += 1
        wrong_action_rate = wrong_count / n_actions

    # Action-sequence match (strict)
    action_sequence_correct = actual_actions == expected_actions

    # Coverage: how many expected actions were performed at least once?
    if len(expected_actions) == 0:
        coverage = 1.0 if n_actions == 0 else 0.0
    else:
        covered = sum(1 for a in expected_actions if a in actual_actions)
        coverage = covered / len(expected_actions)

    # Extra calls beyond expected
    extra_calls = max(0, n_actions - len(expected_actions))

    return {
        "actual_actions": actual_actions,
        "expected_actions": expected_actions,
        "wrong_action_rate": wrong_action_rate,
        "action_sequence_correct": action_sequence_correct,
        "coverage": coverage,
        "param_errors": param_errors,
        "json_errors": json_errors,
        "extra_calls": extra_calls,
        "n_calls": n_actions,
        "n_turns": trace.get("n_turns", 0),
    }


def score_run_for_narrow(trace: dict, expected_tool: str) -> dict:
    """Score a narrow-tool (Condition B) run.

    The model chooses between read_file, write_file, list_dir.  We check
    whether the right tool was called for each step.
    """
    calls = trace.get("tool_calls", [])
    actual_tools: list[str] = []
    param_errors = 0
    json_errors = 0

    # Parse expected_tool (may be "list_dir+write_file+read_file" or "read_file")
    expected_sequence = expected_tool.split("+") if "+" in expected_tool else [expected_tool]

    for tc in calls:
        if tc.get("error") and "JSON parse" in str(tc["error"]):
            json_errors += 1
            continue
        name = tc.get("name", "")
        actual_tools.append(name)

        args = tc.get("arguments", {})
        # For write_file, content is required
        if name == "write_file" and not args.get("content"):
            param_errors += 1
        if not (args.get("filepath") or args.get("dirpath")):
            param_errors += 1

    # Wrong-tool rate
    n_calls = len(actual_tools)
    if n_calls == 0:
        wrong_tool_rate = 0.0
    else:
        wrong_count = 0
        for i, actual in enumerate(actual_tools):
            expected = expected_sequence[i] if i < len(expected_sequence) else None
            if expected is not None and actual != expected:
                wrong_count += 1
        wrong_tool_rate = wrong_count / n_calls

    # Tool-sequence match (strict)
    tool_sequence_correct = actual_tools == expected_sequence

    # Coverage
    if len(expected_sequence) == 0:
        coverage = 1.0 if n_calls == 0 else 0.0
    else:
        covered = sum(1 for t in expected_sequence if t in actual_tools)
        coverage = covered / len(expected_sequence)

    extra_calls = max(0, n_calls - len(expected_sequence))

    return {
        "actual_tools": actual_tools,
        "expected_tools": expected_sequence,
        "wrong_tool_rate": wrong_tool_rate,
        "tool_sequence_correct": tool_sequence_correct,
        "coverage": coverage,
        "param_errors": param_errors,
        "json_errors": json_errors,
        "extra_calls": extra_calls,
        "n_calls": n_calls,
        "n_turns": trace.get("n_turns", 0),
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
        help="Runs per (prompt, condition) pair (default 5).",
    )
    parser.add_argument(
        "--prompts", type=str, nargs="*",
        help="Prompt IDs to run (default: all 10).",
    )
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    # Filter prompts if requested
    prompts = EVAL_PROMPTS
    if args.prompts:
        prompts = [p for p in EVAL_PROMPTS if p["id"] in args.prompts]
        if not prompts:
            raise SystemExit(f"No prompts matched: {args.prompts}")

    conditions = ["mega", "narrow"]

    print(f"Model: {MODEL}")
    print(f"Prompts: {len(prompts)}  |  Runs per cell: {args.runs}")
    print(f"Conditions: mega-tool (file_op) vs. narrow tools (read_file+write_file+list_dir)")
    print(f"Total runs: {len(prompts) * len(conditions) * args.runs}")
    print(f"Sandbox:  {SANDBOX_ROOT.resolve()}")
    print()

    # ---- Run all cells ------------------------------------------------------
    all_results: list[dict] = []

    # Accumulators
    per_cond_sequence_correct: dict[str, list[int]] = {c: [] for c in conditions}
    per_cond_coverage: dict[str, list[float]] = {c: [] for c in conditions}
    per_cond_error_rate: dict[str, list[float]] = {c: [] for c in conditions}
    per_cond_tokens: dict[str, list[int]] = {c: [] for c in conditions}
    per_cond_param_errors: dict[str, list[int]] = {c: [] for c in conditions}
    per_cond_extra_calls: dict[str, list[int]] = {c: [] for c in conditions}
    per_cond_n_calls: dict[str, list[int]] = {c: [] for c in conditions}
    per_cond_prompt_ok: dict[str, dict[str, int]] = {c: {} for c in conditions}
    per_cond_prompt_total: dict[str, dict[str, int]] = {c: {} for c in conditions}

    for cond_label in conditions:
        if cond_label == "mega":
            tools = [FILE_OP_MEGA_TOOL_SCHEMA]
            tool_functions = MEGA_TOOL_FUNCTIONS
            desc_label = "MEGA-TOOL (file_op)"
        else:
            tools = NARROW_TOOL_SCHEMAS
            tool_functions = NARROW_TOOL_FUNCTIONS
            desc_label = "NARROW TOOLS (read_file + write_file + list_dir)"

        print(f"{'=' * 70}")
        print(f"CONDITION: {desc_label}")
        print(f"{'=' * 70}")

        # Print schema sizes for comparison
        schema_json = json.dumps(tools, ensure_ascii=False)
        schema_chars = len(schema_json)
        schema_tokens_est = schema_chars // 4  # rough estimate
        print(f"  Tool schemas: {len(tools)} tool(s), ~{schema_chars} chars, ~{schema_tokens_est} est. tokens")
        if cond_label == "mega":
            print(f"  file_op actions: read | write | list")
        else:
            print(f"  Tools: read_file, write_file, list_dir")
        print()

        for prompt in prompts:
            prompt_id = prompt["id"]
            per_cond_prompt_ok[cond_label][prompt_id] = 0
            per_cond_prompt_total[cond_label][prompt_id] = 0

            for run_idx in range(1, args.runs + 1):
                try:
                    trace = run_experiment_agent(prompt["prompt"], tools, tool_functions)
                except Exception as e:
                    print(f"  [{cond_label}] {prompt_id} run {run_idx}: API ERROR — {e}")
                    all_results.append({
                        "condition": cond_label,
                        "prompt_id": prompt_id,
                        "run": run_idx,
                        "api_error": str(e),
                    })
                    continue

                if cond_label == "mega":
                    score = score_run_for_mega(trace, prompt["expected_actions"])
                    error_rate = score["wrong_action_rate"]
                    ok = score["action_sequence_correct"]
                    actual_str = ", ".join(score["actual_actions"]) if score["actual_actions"] else "(none)"
                    expected_str = ", ".join(score["expected_actions"]) if score["expected_actions"] else "(none)"
                else:
                    score = score_run_for_narrow(trace, prompt["expected_tool"])
                    error_rate = score["wrong_tool_rate"]
                    ok = score["tool_sequence_correct"]
                    actual_str = ", ".join(score["actual_tools"]) if score["actual_tools"] else "(none)"
                    expected_str = ", ".join(score["expected_tools"]) if score["expected_tools"] else "(none)"

                all_results.append({
                    "condition": cond_label,
                    "prompt_id": prompt_id,
                    "run": run_idx,
                    **score,
                    "answer": trace["answer"][:300],
                    "tool_call_details": trace["tool_calls"],
                    "usage": trace["usage"],
                    "estimated_cost": trace["estimated_cost"],
                    "error": trace.get("error"),
                })

                per_cond_sequence_correct[cond_label].append(1 if ok else 0)
                per_cond_coverage[cond_label].append(score["coverage"])
                per_cond_error_rate[cond_label].append(error_rate)
                per_cond_tokens[cond_label].append(trace["usage"]["total_tokens"])
                per_cond_param_errors[cond_label].append(score.get("param_errors", 0))
                per_cond_extra_calls[cond_label].append(score.get("extra_calls", 0))
                per_cond_n_calls[cond_label].append(score["n_calls"])
                per_cond_prompt_ok[cond_label][prompt_id] += 1 if ok else 0
                per_cond_prompt_total[cond_label][prompt_id] += 1

                # Per-run detail line
                icon = "✅" if ok else "❌"
                extra_info = ""
                if score.get("param_errors"):
                    extra_info += f"  param-errs: {score['param_errors']}"
                if score.get("json_errors"):
                    extra_info += f"  json-errs: {score['json_errors']}"
                if score.get("extra_calls"):
                    extra_info += f"  extra-calls: {score['extra_calls']}"
                print(
                    f"  [{icon}] {prompt_id:<28} run {run_idx}/{args.runs}"
                    f"  actual=[{actual_str}]  expected=[{expected_str}]{extra_info}"
                )

            # Per-prompt summary
            n_ok = per_cond_prompt_ok[cond_label][prompt_id]
            n_total = per_cond_prompt_total[cond_label][prompt_id]
            rate = n_ok / n_total if n_total else 0
            mark = "✅" if rate == 1.0 else ("❌" if rate == 0.0 else "⚠️")
            print(f"  {mark} {prompt_id}: {n_ok}/{n_total} exact-sequence-match ({rate:.0%})")
        print()

    # =========================================================================
    # Summary tables
    # =========================================================================

    # Table 1 — Error rate by condition (wrong-action for mega, wrong-tool for narrow)
    print(f"{'=' * 70}")
    print("ERROR RATE BY CONDITION  (lower = better)")
    print(f"  mega:   wrong-action rate — fraction of file_op calls with wrong 'action'")
    print(f"  narrow: wrong-tool rate   — fraction of calls to the wrong tool")
    print(f"{'=' * 70}")
    header = f"{'condition':<14}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}"
    print(header)
    print("-" * len(header))
    for cond_label in conditions:
        rates = per_cond_error_rate[cond_label]
        if rates:
            mean_r = sum(rates) / len(rates)
            var_r = sum((r - mean_r) ** 2 for r in rates) / len(rates)
            sd_r = var_r ** 0.5
            min_r = min(rates)
            max_r = max(rates)
            print(f"{cond_label:<14}{mean_r:>8.3f}{sd_r:>8.3f}{min_r:>8.3f}{max_r:>8.3f}")

    # Table 2 — Exact-sequence-match rate (the strictest metric)
    print(f"\n{'=' * 70}")
    print("EXACT-SEQUENCE-MATCH RATE BY CONDITION  (higher = better)")
    print(f"  Did the agent pick exactly the right sequence of actions/tools?")
    print(f"{'=' * 70}")
    print(f"{'condition':<14}{'exact':>8}{'total':>8}{'rate':>9}")
    print("-" * 42)
    for cond_label in conditions:
        exacts = per_cond_sequence_correct[cond_label]
        if exacts:
            n_exact = sum(exacts)
            n_total = len(exacts)
            rate = n_exact / n_total if n_total else 0
            print(f"{cond_label:<14}{n_exact:>8}{n_total:>8}{rate:>8.0%}")

    # Table 3 — Coverage (lenient — at least called expected actions once)
    print(f"\n{'=' * 70}")
    print("COVERAGE BY CONDITION  (higher = better)")
    print(f"  Fraction of expected actions/tools that were called at least once.")
    print(f"{'=' * 70}")
    print(header.replace("error", "coverage"))
    print("-" * len(header))
    for cond_label in conditions:
        covs = per_cond_coverage[cond_label]
        if covs:
            mean_c = sum(covs) / len(covs)
            var_c = sum((c - mean_c) ** 2 for c in covs) / len(covs)
            sd_c = var_c ** 0.5
            min_c = min(covs)
            max_c = max(covs)
            print(f"{cond_label:<14}{mean_c:>8.3f}{sd_c:>8.3f}{min_c:>8.3f}{max_c:>8.3f}")

    # Table 4 — Per-prompt exact-match comparison
    print(f"\n{'=' * 70}")
    print(f"PER-PROMPT EXACT-MATCH COMPARISON  (out of {args.runs} runs each)")
    print(f"{'=' * 70}")
    print(f"{'prompt':<28}{'mega':>10}{'narrow':>10}{'delta':>11}{'category':<22}")
    print("-" * 83)
    for prompt in prompts:
        pid = prompt["id"]
        m_ok = per_cond_prompt_ok["mega"].get(pid, 0)
        n_ok = per_cond_prompt_ok["narrow"].get(pid, 0)
        m_total = per_cond_prompt_total["mega"].get(pid, 1)
        n_total = per_cond_prompt_total["narrow"].get(pid, 1)
        m_rate = m_ok / m_total if m_total else 0
        n_rate = n_ok / n_total if n_total else 0
        delta = n_rate - m_rate  # positive = narrow better
        delta_str = f"{delta:+.0%}"
        winner = "narrow" if delta > 0 else ("mega" if delta < 0 else "tie")
        print(f"{pid:<28}{m_rate:>9.0%}{n_rate:>9.0%}{delta_str:>11}{prompt['category']:<22}")

    # Table 5 — Parameter errors
    print(f"\n{'=' * 70}")
    print("PARAMETER ERRORS BY CONDITION  (lower = better)")
    print(f"{'=' * 70}")
    for cond_label in conditions:
        errs = per_cond_param_errors[cond_label]
        if errs:
            total = sum(errs)
            print(f"  {cond_label:<14} {total:>4} total param errors across {len(errs)} runs")

    # Table 6 — Extra calls
    print(f"\n{'=' * 70}")
    print("EXTRA CALLS BY CONDITION  (lower = better)")
    print(f"  Calls beyond the expected sequence length.")
    print(f"{'=' * 70}")
    for cond_label in conditions:
        extras = per_cond_extra_calls[cond_label]
        if extras:
            total = sum(extras)
            mean_e = total / len(extras)
            print(f"  {cond_label:<14} {total:>4} total extra calls, mean={mean_e:.2f}/run")

    # Table 7 — Token usage comparison
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

    # ---- Delta summary -------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("DELTA: mega → narrow  (positive delta = narrow better)")
    print(f"{'=' * 70}")

    for metric, data, better in [
        ("Error rate", per_cond_error_rate, "lower"),
        ("Exact-sequence-match", per_cond_sequence_correct, "higher"),
        ("Coverage", per_cond_coverage, "higher"),
    ]:
        m_vals = data["mega"]
        n_vals = data["narrow"]
        if m_vals and n_vals:
            if metric == "Exact-sequence-match":
                m_mean = sum(m_vals) / len(m_vals)
                n_mean = sum(n_vals) / len(n_vals)
            else:
                m_mean = sum(m_vals) / len(m_vals)
                n_mean = sum(n_vals) / len(n_vals)
            delta = n_mean - m_mean
            direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            improved = (
                (better == "higher" and delta > 0)
                or (better == "lower" and delta < 0)
            )
            verdict = (
                "✅ narrow better" if improved
                else ("⚠️  mega better" if delta != 0 else "→ no difference")
            )
            print(f"  {metric:<24} {m_mean:.3f} → {n_mean:.3f}  ({delta:+.3f} {direction})  {verdict}")

    # ---- Token delta ---------------------------------------------------------
    m_tokens = per_cond_tokens["mega"]
    n_tokens = per_cond_tokens["narrow"]
    if m_tokens and n_tokens:
        m_mean_t = sum(m_tokens) / len(m_tokens)
        n_mean_t = sum(n_tokens) / len(n_tokens)
        delta_t = n_mean_t - m_mean_t
        print(f"  {'Tokens/run':<24} {m_mean_t:,.0f} → {n_mean_t:,.0f}  ({delta_t:+,.0f})  "
              f"{'narrow used more' if delta_t > 0 else 'mega used more'}")

    # ---- Most common error types by prompt -----------------------------------
    print(f"\n{'=' * 70}")
    print("PER-CATEGORY BREAKDOWN  (mega vs narrow exact-match rate)")
    print(f"{'=' * 70}")

    categories = {}
    for prompt in prompts:
        cat = prompt["category"]
        if cat not in categories:
            categories[cat] = {"mega": {"ok": 0, "total": 0}, "narrow": {"ok": 0, "total": 0}}
        for cond in conditions:
            categories[cat][cond]["ok"] += per_cond_prompt_ok[cond].get(prompt["id"], 0)
            categories[cat][cond]["total"] += per_cond_prompt_total[cond].get(prompt["id"], 0)

    print(f"{'category':<24}{'mega':>10}{'narrow':>10}{'winner':>10}")
    print("-" * 56)
    for cat, data in sorted(categories.items()):
        m_rate = data["mega"]["ok"] / data["mega"]["total"] if data["mega"]["total"] else 0
        n_rate = data["narrow"]["ok"] / data["narrow"]["total"] if data["narrow"]["total"] else 0
        if n_rate > m_rate:
            winner = "narrow"
        elif m_rate > n_rate:
            winner = "mega"
        else:
            winner = "tie"
        print(f"{cat:<24}{m_rate:>9.0%}{n_rate:>9.0%}{winner:>10}")

    # ---- Save results --------------------------------------------------------
    out_dir = _THIS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"day2_results_{timestamp}.json"

    # Build clean summary
    summary = {
        "experiment": "week05_day2_tool_granularity",
        "model": MODEL,
        "runs_per_cell": args.runs,
        "conditions": conditions,
        "condition_labels": {
            "mega": "One file_op(action, ...) mega-tool",
            "narrow": "Three narrow tools: read_file, write_file, list_dir",
        },
        "prompts": [p["id"] for p in prompts],
        "error_rate": {
            c: {
                "mean": sum(v) / len(v) if v else 0,
                "n": len(v),
            }
            for c, v in per_cond_error_rate.items()
        },
        "exact_sequence_match": {
            c: {
                "rate": sum(v) / len(v) if v else 0,
                "n": len(v),
            }
            for c, v in per_cond_sequence_correct.items()
        },
        "coverage": {
            c: {
                "mean": sum(v) / len(v) if v else 0,
                "n": len(v),
            }
            for c, v in per_cond_coverage.items()
        },
        "param_errors": {
            c: {
                "total": sum(v),
                "n": len(v),
            }
            for c, v in per_cond_param_errors.items()
        },
        "extra_calls": {
            c: {
                "total": sum(v),
                "mean": sum(v) / len(v) if v else 0,
                "n": len(v),
            }
            for c, v in per_cond_extra_calls.items()
        },
        "token_usage": {
            c: {
                "mean": sum(v) / len(v) if v else 0,
                "total": sum(v),
                "n": len(v),
            }
            for c, v in per_cond_tokens.items()
        },
        "per_prompt_exact": {
            pid: {
                "mega": per_cond_prompt_ok["mega"].get(pid, 0),
                "narrow": per_cond_prompt_ok["narrow"].get(pid, 0),
                "total": args.runs,
            }
            for pid in [p["id"] for p in prompts]
        },
        "per_category": {
            cat: {
                "mega_rate": data["mega"]["ok"] / data["mega"]["total"] if data["mega"]["total"] else 0,
                "narrow_rate": data["narrow"]["ok"] / data["narrow"]["total"] if data["narrow"]["total"] else 0,
            }
            for cat, data in categories.items()
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "prompt_details": [
                    {"id": p["id"], "prompt": p["prompt"],
                     "expected_actions": p["expected_actions"],
                     "expected_tool": p["expected_tool"],
                     "category": p["category"], "note": p["note"]}
                    for p in prompts
                ],
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
