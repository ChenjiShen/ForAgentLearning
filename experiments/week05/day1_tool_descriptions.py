"""Week 5 Day 1: Rewrite tool descriptions with explicit when-to-call conditions.

Experiment question:
    Do explicit "when to call this / when NOT to call this" clauses in tool
    descriptions improve the model's tool-selection accuracy?

Method:
    1. Take the week03 agent's 6 tools with their original descriptions.
    2. Rewrite every description with structured "Call when" / "Do NOT call when"
       sections that name concrete triggers, concrete counter-examples, and the
       correct alternative tool for each wrong-path scenario.
    3. Run both description sets on 10 prompts designed to test tool selection.
    4. Compare should-call rate, wrong-tool rate, and exact-match rate.

The key insight being tested:
    A description that says "Call when the user asks to see a file" is weaker
    than one that says "Call when the user names a specific file → read_file.
    Do NOT call when searching across many files → use run_command with grep
    instead."  The "do not call" clause is the high-leverage addition — it
    preempts the most common class of tool-selection errors.

Conditions:
    A (before) — Original tool descriptions from week03 agent.py
    B (after)  — Rewritten with explicit when-to-call / when-NOT-to-call

Metrics:
    - should-call rate: fraction of expected tools actually called (recall)
    - wrong-tool rate: fraction of total calls that were to unexpected tools
    - exact-match rate: called_tools == expected_tools (strict)

    Default: 2 conditions × 5 runs × 10 prompts = 100 agent runs

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week05/day1_tool_descriptions.py
    python experiments/week05/day1_tool_descriptions.py --runs 3   # quick test
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
# Tool descriptions — Condition A (original) vs Condition B (rewritten)
# ============================================================================

# ---- Condition A: Original descriptions from week03 agent.py ---------------
# These are verbatim copies of the TOOLS list in experiments/week03/agent.py.
# The *only* difference from what the agent shipped with is that web_search and
# web_fetch descriptions are slightly trimmed for consistency (the originals
# already had reasonable call guidance, but we keep them as-is for "before").

CONDITION_A_DESCRIPTIONS = {
    "read_file": (
        "Read a text file from the sandbox workspace. Call when the user "
        "asks to see or summarize a file. Path must be relative to the "
        "sandbox root (no leading '/' or '..')."
    ),
    "write_file": (
        "Write or overwrite a text file in the sandbox workspace. Call when "
        "the user asks to save, create, or update a file. Path must stay "
        "inside the sandbox."
    ),
    "list_dir": (
        "List files and subdirectories in the sandbox. Call when the user "
        "wants to browse or discover what files exist."
    ),
    "run_command": (
        "Run a read-only shell command inside the sandbox workspace. "
        "Only allowlisted executables are permitted (ls, cat, echo, grep, "
        "wc, head, tail, python/python3). No shell — pipes and semicolons "
        "are not supported. Use for listing files, grepping content, or "
        "running short python one-liners."
    ),
    "web_fetch": (
        "Fetch a web page by URL and return the page title plus extracted "
        "plain text. Only http/https URLs are allowed — internal/private "
        "addresses are blocked. The response is capped at 5000 chars by "
        "default. Call this when the user asks to read, fetch, visit, or "
        "summarize a URL. Also call this instead of curl or wget (which "
        "are blocked by the command allowlist)."
    ),
    "web_search": (
        "Search the web via DuckDuckGo and return a list of results with "
        "titles, URLs, and short text snippets. Call this when the user "
        "asks to search for something online, find information, look up "
        "facts, or research current events. Combine with web_fetch to "
        "open the most promising result for more detail."
    ),
}

# ---- Condition B: Rewritten with explicit when-to-call / when-NOT-to-call --
# Each description now follows a consistent pattern:
#   1. What the tool does (one line)
#   2. **Call when:** concrete trigger examples
#   3. **Do NOT call when:** common mistakes + correct alternative

CONDITION_B_DESCRIPTIONS = {
    "read_file": (
        "Read the full contents of a specific text file from the sandbox "
        "workspace. Returns the file content with a byte count.\n\n"
        "**Call when:**\n"
        "- The user asks to see, read, view, inspect, or show what is inside "
        "a NAMED file (e.g. \"what's in notes.md?\", \"read subdir/data.json\").\n"
        "- You need to verify a file was written correctly (read it back after "
        "write_file).\n"
        "- The user asks about the content of a specific, known file path.\n\n"
        "**Do NOT call when:**\n"
        "- You need to discover what files exist → use list_dir first, then "
        "read_file once you know the path.\n"
        "- You need to search for text ACROSS multiple files → use run_command "
        "with grep. Reading every file individually wastes tokens and turns.\n"
        "- The user asks about a URL (http://…) → use web_fetch. read_file "
        "only accesses local sandbox files.\n"
        "- You want to count lines/words in a file → use run_command with wc "
        "(faster, and avoids loading the full file content)."
    ),
    "write_file": (
        "Write or overwrite a text file in the sandbox workspace.\n\n"
        "**Call when:**\n"
        "- The user explicitly asks to save, create, write, record, or store "
        "content to a file (e.g. \"save this to notes.txt\", \"create a file "
        "called summary.md\").\n"
        "- The user says \"put this in a file\" or \"write it down\".\n\n"
        "**Do NOT call when:**\n"
        "- The user just wants to SEE or read content → no tool needed, just "
        "answer directly.\n"
        "- The user says \"note\" or \"remember\" without specifying a filename "
        "→ ask which file to write to; don't invent a filename.\n"
        "- The user wants to append to or edit part of a file → not directly "
        "supported: read_file first, modify the content, then write_file the "
        "entire new version."
    ),
    "list_dir": (
        "List files and subdirectories inside the sandbox workspace. Returns "
        "structured metadata (name, path, type, accessibility) for each entry.\n\n"
        "**Call when:**\n"
        "- The user wants to browse, explore, or discover what files exist "
        "(e.g. \"what's in the sandbox?\", \"show me the directory structure\", "
        "\"what files are available?\").\n"
        "- You need to find a file but don't know its exact path → call "
        "list_dir BEFORE read_file.\n"
        "- The user asks \"do I have any X files?\" or \"list all .md files\".\n\n"
        "**Do NOT call when:**\n"
        "- You already know the exact file path and just need its contents → "
        "use read_file directly.\n"
        "- You need to search file CONTENTS for a pattern → use run_command "
        "with grep (list_dir only shows names, not what's inside files).\n"
        "- The user asks about web resources or online files → use web_search."
    ),
    "run_command": (
        "Run a shell command inside the sandbox workspace using allowlisted "
        "executables only: " + ", ".join(sorted(ALLOWED_EXECUTABLES)) + ". "
        "No shell interpretation — pipes | and semicolons ; are NOT supported. "
        "Timeout: 5 seconds. Output capped at 4000 chars.\n\n"
        "**Call when:**\n"
        "- Searching for text across one or more files → use grep (much faster "
        "than reading each file with read_file).\n"
        "- Counting lines, words, or characters → use wc.\n"
        "- Listing files with details or filters → use ls (more powerful than "
        "list_dir for pattern matching).\n"
        "- Running a Python script for computation or data processing → use "
        "python script.py (write the script first with write_file).\n"
        "- Any batch operation: finding, filtering, counting, or processing "
        "across multiple files.\n\n"
        "**Do NOT call when:**\n"
        "- Reading a single known file's content → use read_file (simpler, "
        "returns structured data).\n"
        "- Fetching a web page → use web_fetch (curl and wget are blocked by "
        "the allowlist).\n"
        "- Searching the internet → use web_search.\n"
        "- Saving content → use write_file.\n"
        "- Browsing available files → use list_dir (returns structured JSON, "
        "easier to parse than raw ls output).\n"
        "- Running destructive commands (rm, mv, chmod) → all blocked."
    ),
    "web_fetch": (
        "Fetch a web page by URL and return the page title plus extracted "
        "plain text. HTML is automatically stripped. Only http/https URLs; "
        "private/internal addresses are blocked.\n\n"
        "**Call when:**\n"
        "- The user provides a specific URL and asks you to fetch, visit, open, "
        "read, or summarize that web page (e.g. \"go to https://example.com\").\n"
        "- You have web_search results and want to drill into a specific "
        "result URL for more detail → call web_fetch AFTER web_search.\n"
        "- Use this instead of run_command with curl or wget — those "
        "executables are blocked by the allowlist.\n\n"
        "**Do NOT call when:**\n"
        "- You do NOT have a specific URL → use web_search first to find one. "
        "web_fetch without a URL is useless.\n"
        "- The user asks about a local sandbox file → use read_file (web_fetch "
        "only accesses the internet).\n"
        "- The user wants general web research without a target URL → use "
        "web_search.\n"
        "- The URL uses a non-http/https scheme (file://, ftp://) → these are "
        "blocked for security; explain why and suggest alternatives."
    ),
    "web_search": (
        "Search the web via DuckDuckGo and return a list of results with "
        "titles, URLs, and text snippets.\n\n"
        "**Call when:**\n"
        "- The user asks to search, find, look up, or research something "
        "online without providing a specific URL (e.g. \"search for Python "
        "3.13 release notes\", \"what's the latest on AI?\").\n"
        "- The user asks about current events, recent news, or facts you're "
        "unsure about — verify with a search.\n"
        "- The user wants to find a specific website but doesn't know the URL "
        "→ search first, then use web_fetch on the top result.\n\n"
        "**Do NOT call when:**\n"
        "- You already have a specific URL → use web_fetch directly.\n"
        "- The user asks about local sandbox files → use list_dir, read_file, "
        "or run_command (web_search searches the internet, not local files).\n"
        "- The user asks a general-knowledge question you're confident about "
        "→ just answer directly. Unnecessary searches waste time.\n"
        "- The question is about math, coding, or logic → use the appropriate "
        "local tool (calculator-like operations, file operations, etc.).\n\n"
        "**Standard research workflow:** web_search → web_fetch.\n"
        "First search to discover relevant URLs, then fetch the most promising "
        "ones for detailed reading."
    ),
}

# Tool parameter descriptions (shared across conditions — the experiment
# variable is the top-level tool description, not the parameter docs).
_PARAM_DESCRIPTIONS = {
    "read_file": {
        "filepath": "Relative path to the file inside the sandbox, e.g. 'notes.txt' or 'subdir/data.json'.",
    },
    "write_file": {
        "filepath": "Relative path inside the sandbox, e.g. 'output.txt'.",
        "content": "The full text content to write to the file.",
    },
    "list_dir": {
        "dirpath": "Relative directory path; defaults to '.' for sandbox root.",
    },
    "run_command": {
        "command": (
            "Command string with an allowlisted executable and its arguments, "
            "e.g. 'grep pattern file.txt' or 'wc -l *.py'."
        ),
    },
    "web_fetch": {
        "url": "Full URL starting with http:// or https://.",
        "max_chars": "Max chars of extracted text (default 5000). Use a smaller value for quick summaries.",
    },
    "web_search": {
        "query": "Search query — keep it concise and keyword-focused.",
        "max_results": "Number of results to return (1–10, default 5).",
    },
}

_PARAM_REQUIRED = {
    "read_file": ["filepath"],
    "write_file": ["filepath", "content"],
    "list_dir": [],
    "run_command": ["command"],
    "web_fetch": ["url"],
    "web_search": ["query"],
}


def build_tools(condition: str) -> list[dict]:
    """Build the OpenAI tool-schema list for a given description condition.

    The tool *implementations* are identical — only the top-level
    ``function.description`` field differs between conditions.
    """
    descriptions = (
        CONDITION_A_DESCRIPTIONS if condition == "before" else CONDITION_B_DESCRIPTIONS
    )

    # Per-tool parameter type overrides (default is "string")
    _PARAM_TYPES: dict[str, dict[str, str]] = {
        "web_fetch": {"max_chars": "integer"},
        "web_search": {"max_results": "integer"},
    }

    tools = []
    for name in ["read_file", "write_file", "list_dir", "run_command", "web_fetch", "web_search"]:
        props = {}
        type_overrides = _PARAM_TYPES.get(name, {})
        for pname, pdesc in _PARAM_DESCRIPTIONS.get(name, {}).items():
            ptype = type_overrides.get(pname, "string")
            props[pname] = {"type": ptype, "description": pdesc}

        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": descriptions[name],
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": _PARAM_REQUIRED.get(name, []),
                },
            },
        })
    return tools


# ============================================================================
# Tool-function registry (shared across conditions)
# ============================================================================

TOOL_FUNCTIONS = {
    "read_file": lambda inp: read_file(inp["filepath"]),
    "write_file": lambda inp: write_file(inp["filepath"], inp["content"]),
    "list_dir": lambda inp: list_dir(inp.get("dirpath", ".")),
    "run_command": lambda inp: run_command(inp["command"]),
    "web_fetch": lambda inp: web_fetch(inp["url"], inp.get("max_chars", WEB_FETCH_DEFAULT_MAX_CHARS)),
    "web_search": lambda inp: web_search(inp["query"], inp.get("max_results", WEB_SEARCH_DEFAULT_MAX_RESULTS)),
}

ALL_TOOL_NAMES = frozenset(TOOL_FUNCTIONS)

# ============================================================================
# Eval prompts — 10 prompts designed to test tool-selection judgment
# ============================================================================
# Each prompt:
#   id              — short slug
#   prompt          — the user message sent to the agent
#   expected_tools  — the set of tools the agent SHOULD call (gold answer)
#   category        — what kind of selection skill it tests
#   note            — why this prompt matters for the experiment

EVAL_PROMPTS = [
    {
        "id": "P1-read-known-file",
        "prompt": (
            "Read the file 'subdir/data.json' from the sandbox. "
            "What fields does it contain? List all the field names."
        ),
        "expected_tools": ["read_file"],
        "category": "single-tool",
        "note": "Named file → read_file directly. Tests whether the model "
                "recognises 'here is a specific file path → read it' without "
                "first calling list_dir unnecessarily.",
    },
    {
        "id": "P2-browse-sandbox",
        "prompt": (
            "I just started working in this sandbox. What files and directories "
            "are available here? Show me everything."
        ),
        "expected_tools": ["list_dir"],
        "category": "single-tool",
        "note": "Discovery/browsing → list_dir. Tests whether the model picks "
                "list_dir over run_command 'ls' for structured file browsing.",
    },
    {
        "id": "P3-grep-across-files",
        "prompt": (
            "Search all text files (.txt, .md, .py) in the sandbox for any "
            "mention of the word 'sandbox'. For each file that contains it, "
            "tell me which file it is and show the matching line."
        ),
        "expected_tools": ["run_command"],
        "category": "grep-vs-read",
        "note": "Multi-file text search → run_command with grep, NOT read_file "
                "on each file. This is the #1 tool-selection error that better "
                "descriptions should prevent: the before description says "
                "'call when you want to see a file', which is vague enough "
                "that a model might read every file instead of grepping.",
    },
    {
        "id": "P4-save-content",
        "prompt": (
            "Create a new file called 'week5_summary.md' in the sandbox with "
            "this exact content:\n\n"
            "# Week 5 Summary\n\n"
            "## Day 1\n"
            "- Experiment: tool description rewrite\n"
            "- Topic: when-to-call conditions\n"
            "- Status: running\n\n"
            "Save it now."
        ),
        "expected_tools": ["write_file"],
        "category": "single-tool",
        "note": "Explicit file creation → write_file. Tests simple single-tool "
                "recognition.",
    },
    {
        "id": "P5-fetch-url",
        "prompt": (
            "Open https://example.com and tell me: (1) what is the page title, "
            "and (2) what does the page say in its first paragraph?"
        ),
        "expected_tools": ["web_fetch"],
        "category": "single-tool",
        "note": "Specific URL → web_fetch directly. Tests whether the model "
                "goes straight to web_fetch or wastes a turn on web_search "
                "first when a URL is already provided.",
    },
    {
        "id": "P6-search-online",
        "prompt": (
            "Search online: what are the most popular open-source AI agent "
            "frameworks in 2026? Give me the top 3 with a one-line description "
            "of each."
        ),
        "expected_tools": ["web_search"],
        "category": "single-tool",
        "note": "No URL given → web_search. Tests whether the model recognises "
                "that web research starts with search, not fetch.",
    },
    {
        "id": "P7-search-then-fetch",
        "prompt": (
            "I want to learn about how Anthropic recommends building AI agents. "
            "First search for 'Anthropic building effective agents', then visit "
            "the most relevant result and give me a 3-sentence summary of what "
            "you find there."
        ),
        "expected_tools": ["web_search", "web_fetch"],
        "category": "multi-tool-chain",
        "note": "Explicit two-step: search first, then fetch the top result. "
                "Tests whether the model chains tools correctly when both are "
                "needed in sequence.",
    },
    {
        "id": "P8-count-lines-efficient",
        "prompt": (
            "How many lines are in the sandbox file 'notes.md'? Use the most "
            "efficient method — do NOT read the entire file just to count lines."
        ),
        "expected_tools": ["run_command"],
        "category": "grep-vs-read",
        "note": "Counting task → run_command with wc, NOT read_file. The prompt "
                "explicitly warns against reading the file. Tests whether the "
                "model defaults to read_file anyway (common with vague "
                "descriptions) or picks the efficient tool.",
    },
    {
        "id": "P9-no-tool-trivia",
        "prompt": (
            "What year did World War II end? Just give me the year."
        ),
        "expected_tools": [],
        "category": "no-tool",
        "note": "Pure knowledge question → no tools needed. Tests whether the "
                "model avoids unnecessary tool calls. Better descriptions should "
                "not increase the false-positive rate on no-tool prompts — the "
                "'Do NOT call when' clauses should help the model recognise "
                "when NO tool is the right answer.",
    },
    {
        "id": "P10-trap-web-search-vs-local",
        "prompt": (
            "I need the latest information about 'sandbox escape prevention "
            "techniques in AI agent systems'. Search for this topic and tell "
            "me what the top recommendations are."
        ),
        "expected_tools": ["web_search"],
        "category": "web-vs-local-trap",
        "note": "The word 'sandbox' appears in local file tools AND in the "
                "search topic — but the user is clearly asking for web research. "
                "The before description for web_fetch says 'call when the user "
                "asks to read/fetch/visit/summarize a URL' which might confuse "
                "models that try to read local sandbox files first. The after "
                "descriptions should help: read_file says 'Do NOT call for URLs' "
                "and web_search says 'Call when the user asks to search/find "
                "/research something online without a URL'.",
    },
]

# ============================================================================
# Agent loop (non-interactive — no permission prompts, no prints)
# ============================================================================


def run_experiment_agent(
    user_input: str,
    tools: list[dict],
    *,
    max_turns: int = 12,
) -> dict:
    """Run the agent with a given tool list and return structured results.

    Returns a dict with: answer, tool_calls (list of {name, args, error}),
    n_turns, n_tool_calls, usage, estimated_cost, error.
    """
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    system_prompt = (
        "You are a helpful assistant with access to sandboxed tools. "
        f"All file paths and shell commands run inside {SANDBOX_ROOT}. "
        "Complete the user's task using the available tools.\n\n"
        "IMPORTANT — Tool selection rules:\n"
        "- Read each tool's description carefully before calling it.\n"
        "- Only call tools that are actually needed for the task.\n"
        "- If a task can be done without any tool, answer directly.\n"
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
                    fn = TOOL_FUNCTIONS[fn_name]
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
# Scoring
# ============================================================================


def score_run(trace: dict, expected_tools: list[str]) -> dict:
    """Compute tool-selection metrics from a single run.

    Three metrics, from most lenient to most strict:

    1. **should-call rate (recall)** — of the expected tools, how many were
       actually called at least once?  1.0 = all expected tools called.
       This measures "did the model remember to call what it should?"

    2. **wrong-tool rate** — of the total tool calls made, how many were to
       tools NOT in the expected set?  0.0 = no wrong calls.
       This measures "did the model avoid calling what it shouldn't?"

    3. **exact-match rate** — called_tools == expected_tools exactly.
       This is the strictest metric: any missing or extra call fails.

    All three matter for different reasons:
    - High should-call + high wrong-tool → the model is over-calling (using
      extra tools beyond what's needed — wasteful but correct).
    - Low should-call + low wrong-tool → the model is under-calling (missing
      tools it needs — incomplete task execution).
    - Exact match is the gold standard: the model called exactly what was
      needed and nothing else.
    """
    called_names = {tc["name"] for tc in trace.get("tool_calls", [])}
    expected = set(expected_tools)

    # Should-call rate (recall per expected tool)
    if len(expected) == 0:
        should_call_rate = 1.0 if len(called_names) == 0 else 0.0
    else:
        n_called_correctly = len(called_names & expected)
        should_call_rate = n_called_correctly / len(expected)

    # Wrong-tool rate
    total_calls = len(called_names)
    if total_calls == 0:
        wrong_tool_rate = 0.0
    else:
        n_wrong = len(called_names - expected)
        wrong_tool_rate = n_wrong / total_calls

    # Exact match
    exact_match = called_names == expected

    # Also track which expected tools were missed and which wrong tools were called
    missed = sorted(expected - called_names)
    extra = sorted(called_names - expected)

    return {
        "called": sorted(called_names),
        "expected": sorted(expected),
        "should_call_rate": should_call_rate,
        "wrong_tool_rate": wrong_tool_rate,
        "exact_match": exact_match,
        "missed": missed,
        "extra": extra,
        "n_calls": len(trace.get("tool_calls", [])),
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

    conditions = ["before", "after"]

    print(f"Model: {MODEL}")
    print(f"Prompts: {len(prompts)}  |  Runs per cell: {args.runs}")
    print(f"Conditions: {conditions}")
    print(f"Total runs: {len(prompts) * len(conditions) * args.runs}")
    print(f"Sandbox:  {SANDBOX_ROOT.resolve()}")
    print()

    # ---- Run all cells ------------------------------------------------------
    all_results: list[dict] = []

    # Accumulators for summary stats
    per_cond_should_call: dict[str, list[float]] = {c: [] for c in conditions}
    per_cond_wrong_tool: dict[str, list[float]] = {c: [] for c in conditions}
    per_cond_exact: dict[str, list[int]] = {c: [] for c in conditions}
    per_cond_tokens: dict[str, list[int]] = {c: [] for c in conditions}
    per_cond_prompt_exact: dict[str, dict[str, int]] = {c: {} for c in conditions}
    per_cond_prompt_total: dict[str, dict[str, int]] = {c: {} for c in conditions}

    for cond_label in conditions:
        tools = build_tools(cond_label)
        desc_label = "BEFORE (original)" if cond_label == "before" else "AFTER (rewritten)"

        print(f"{'=' * 70}")
        print(f"CONDITION: {desc_label}")
        print(f"{'=' * 70}")
        # Print a snippet of the first tool description so the user can see the diff
        sample_name = "read_file"
        sample_desc = (
            CONDITION_A_DESCRIPTIONS[sample_name]
            if cond_label == "before"
            else CONDITION_B_DESCRIPTIONS[sample_name]
        )
        print(f"  Sample ({sample_name}): {sample_desc[:120]}…")
        print()

        for prompt in prompts:
            prompt_id = prompt["id"]
            expected = prompt["expected_tools"]
            per_cond_prompt_exact[cond_label][prompt_id] = 0
            per_cond_prompt_total[cond_label][prompt_id] = 0

            for run_idx in range(1, args.runs + 1):
                try:
                    trace = run_experiment_agent(prompt["prompt"], tools)
                except Exception as e:
                    print(f"  [{cond_label}] {prompt_id} run {run_idx}: API ERROR — {e}")
                    all_results.append({
                        "condition": cond_label,
                        "prompt_id": prompt_id,
                        "run": run_idx,
                        "api_error": str(e),
                    })
                    continue

                score = score_run(trace, expected)
                all_results.append({
                    "condition": cond_label,
                    "prompt_id": prompt_id,
                    "run": run_idx,
                    "expected_tools": expected,
                    **score,
                    "answer": trace["answer"][:300],
                    "tool_call_details": trace["tool_calls"],
                    "usage": trace["usage"],
                    "estimated_cost": trace["estimated_cost"],
                    "error": trace.get("error"),
                })

                per_cond_should_call[cond_label].append(score["should_call_rate"])
                per_cond_wrong_tool[cond_label].append(score["wrong_tool_rate"])
                per_cond_exact[cond_label].append(1 if score["exact_match"] else 0)
                per_cond_tokens[cond_label].append(trace["usage"]["total_tokens"])
                per_cond_prompt_exact[cond_label][prompt_id] += 1 if score["exact_match"] else 0
                per_cond_prompt_total[cond_label][prompt_id] += 1

                # Per-run detail line
                icon = "✅" if score["exact_match"] else "❌"
                call_str = ", ".join(score["called"]) if score["called"] else "(none)"
                exp_str = ", ".join(score["expected"]) if score["expected"] else "(none)"
                extra_info = ""
                if score["missed"]:
                    extra_info += f"  missed: {score['missed']}"
                if score["extra"]:
                    extra_info += f"  extra: {score['extra']}"
                print(
                    f"  [{icon}] {prompt_id:<28} run {run_idx}/{args.runs}"
                    f"  called=[{call_str}]  expected=[{exp_str}]{extra_info}"
                )

            # Per-prompt summary
            n_exact = per_cond_prompt_exact[cond_label][prompt_id]
            n_total = per_cond_prompt_total[cond_label][prompt_id]
            rate = n_exact / n_total if n_total else 0
            mark = "✅" if rate == 1.0 else ("❌" if rate == 0.0 else "⚠️")
            print(f"  {mark} {prompt_id}: {n_exact}/{n_total} exact-match ({rate:.0%})")
        print()

    # =========================================================================
    # Summary tables
    # =========================================================================

    # Table 1 — Overall should-call rate by condition
    print(f"{'=' * 70}")
    print("SHOULD-CALL RATE BY CONDITION  (higher = better)")
    print(f"{'=' * 70}")
    header = f"{'condition':<14}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}"
    print(header)
    print("-" * len(header))
    cond_summary = {}
    for cond_label in conditions:
        rates = per_cond_should_call[cond_label]
        if rates:
            mean_r = sum(rates) / len(rates)
            var_r = sum((r - mean_r) ** 2 for r in rates) / len(rates)
            sd_r = var_r ** 0.5
            min_r = min(rates)
            max_r = max(rates)
            cond_summary[cond_label] = {"mean": mean_r, "sd": sd_r}
            print(f"{cond_label:<14}{mean_r:>8.3f}{sd_r:>8.3f}{min_r:>8.3f}{max_r:>8.3f}")

    # Table 2 — Wrong-tool rate by condition
    print(f"\n{'=' * 70}")
    print("WRONG-TOOL RATE BY CONDITION  (lower = better)")
    print(f"{'=' * 70}")
    print(header.replace("should", "wrong "))
    print("-" * len(header))
    for cond_label in conditions:
        rates = per_cond_wrong_tool[cond_label]
        if rates:
            mean_r = sum(rates) / len(rates)
            var_r = sum((r - mean_r) ** 2 for r in rates) / len(rates)
            sd_r = var_r ** 0.5
            min_r = min(rates)
            max_r = max(rates)
            print(f"{cond_label:<14}{mean_r:>8.3f}{sd_r:>8.3f}{min_r:>8.3f}{max_r:>8.3f}")

    # Table 3 — Exact-match rate by condition
    print(f"\n{'=' * 70}")
    print("EXACT-MATCH RATE BY CONDITION  (higher = better)")
    print(f"{'=' * 70}")
    print(f"{'condition':<14}{'exact':>8}{'total':>8}{'rate':>9}")
    print("-" * 42)
    for cond_label in conditions:
        exacts = per_cond_exact[cond_label]
        if exacts:
            n_exact = sum(exacts)
            n_total = len(exacts)
            rate = n_exact / n_total if n_total else 0
            print(f"{cond_label:<14}{n_exact:>8}{n_total:>8}{rate:>8.0%}")

    # Table 4 — Per-prompt exact-match comparison
    print(f"\n{'=' * 70}")
    print(f"PER-PROMPT EXACT-MATCH COMPARISON  (out of {args.runs} runs each)")
    print(f"{'=' * 70}")
    print(f"{'prompt':<28}{'before':>10}{'after':>10}{'delta':>10}{'category':<18}")
    print("-" * 78)
    for prompt in prompts:
        pid = prompt["id"]
        b_exact = per_cond_prompt_exact["before"].get(pid, 0)
        a_exact = per_cond_prompt_exact["after"].get(pid, 0)
        b_total = per_cond_prompt_total["before"].get(pid, 1)
        a_total = per_cond_prompt_total["after"].get(pid, 1)
        b_rate = b_exact / b_total if b_total else 0
        a_rate = a_exact / a_total if a_total else 0
        delta = a_rate - b_rate
        delta_str = f"{delta:+.0%}"
        print(f"{pid:<28}{b_rate:>9.0%}{a_rate:>9.0%}{delta_str:>10}{prompt['category']:<18}")

    # Table 5 — Token usage comparison
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
    print("DELTA: before → after")
    print(f"{'=' * 70}")

    for metric, data, better in [
        ("Should-call rate", per_cond_should_call, "higher"),
        ("Wrong-tool rate", per_cond_wrong_tool, "lower"),
        ("Exact-match rate", per_cond_exact, "higher"),
    ]:
        b_vals = data["before"]
        a_vals = data["after"]
        if b_vals and a_vals:
            if metric == "Exact-match rate":
                b_mean = sum(b_vals) / len(b_vals)
                a_mean = sum(a_vals) / len(a_vals)
            else:
                b_mean = sum(b_vals) / len(b_vals)
                a_mean = sum(a_vals) / len(a_vals)
            delta = a_mean - b_mean
            direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            improved = (
                (better == "higher" and delta > 0)
                or (better == "lower" and delta < 0)
            )
            verdict = "✅ improved" if improved else ("⚠️  worsened" if delta != 0 else "→ unchanged")
            print(f"  {metric:<22} {b_mean:.3f} → {a_mean:.3f}  ({delta:+.3f} {direction})  {verdict}")

    # ---- Most common wrong-tool calls ----------------------------------------
    print(f"\n{'=' * 70}")
    print("MOST COMMON WRONG-TOOL CALLS  (extra tools that were not expected)")
    print(f"{'=' * 70}")
    for cond_label in conditions:
        wrong_counter: Counter = Counter()
        for r in all_results:
            if r.get("condition") == cond_label and "api_error" not in r:
                for extra_tool in r.get("extra", []):
                    wrong_counter[extra_tool] += 1
        if wrong_counter:
            print(f"\n  {cond_label}:")
            for tool, count in wrong_counter.most_common():
                print(f"    {tool:<28} {count:>3} times called unnecessarily")
        else:
            print(f"\n  {cond_label}: (none — no wrong-tool calls)")

    # ---- Most common missed tools --------------------------------------------
    print(f"\n{'=' * 70}")
    print("MOST COMMON MISSED TOOLS  (expected tools that were NOT called)")
    print(f"{'=' * 70}")
    for cond_label in conditions:
        missed_counter: Counter = Counter()
        for r in all_results:
            if r.get("condition") == cond_label and "api_error" not in r:
                for missed_tool in r.get("missed", []):
                    missed_counter[missed_tool] += 1
        if missed_counter:
            print(f"\n  {cond_label}:")
            for tool, count in missed_counter.most_common():
                print(f"    {tool:<28} {count:>3} times missed")
        else:
            print(f"\n  {cond_label}: (none — all expected tools were called)")

    # ---- Save results --------------------------------------------------------
    out_dir = _THIS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"day1_results_{timestamp}.json"

    # Build clean summary
    summary = {
        "experiment": "week05_day1_tool_descriptions",
        "model": MODEL,
        "runs_per_cell": args.runs,
        "conditions": conditions,
        "prompts": [p["id"] for p in prompts],
        "should_call_rate": {
            c: {
                "mean": sum(v) / len(v) if v else 0,
                "n": len(v),
            }
            for c, v in per_cond_should_call.items()
        },
        "wrong_tool_rate": {
            c: {
                "mean": sum(v) / len(v) if v else 0,
                "n": len(v),
            }
            for c, v in per_cond_wrong_tool.items()
        },
        "exact_match_rate": {
            c: {
                "rate": sum(v) / len(v) if v else 0,
                "n": len(v),
            }
            for c, v in per_cond_exact.items()
        },
        "per_prompt_exact": {
            pid: {
                "before": per_cond_prompt_exact["before"].get(pid, 0),
                "after": per_cond_prompt_exact["after"].get(pid, 0),
                "total": args.runs,
            }
            for pid in [p["id"] for p in prompts]
        },
        "token_usage": {
            c: {
                "mean": sum(v) / len(v) if v else 0,
                "total": sum(v),
                "n": len(v),
            }
            for c, v in per_cond_tokens.items()
        },
        "delta": {
            "should_call_rate": (
                cond_summary["after"]["mean"] - cond_summary["before"]["mean"]
                if "before" in cond_summary and "after" in cond_summary
                else None
            ),
            "exact_match_rate": (
                (sum(per_cond_exact["after"]) / len(per_cond_exact["after"]))
                - (sum(per_cond_exact["before"]) / len(per_cond_exact["before"]))
                if per_cond_exact["before"] and per_cond_exact["after"]
                else None
            ),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "prompt_details": [
                    {"id": p["id"], "prompt": p["prompt"], "expected_tools": p["expected_tools"],
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
