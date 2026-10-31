#!/usr/bin/env python3
"""Week 8 Day 4: connect our agent to our MCP server — the MCP→LLM bridge.

PLAN spec: "Connect your agent to your MCP server: convert MCP tool listings
into OpenAI tool schemas, route tool calls through the MCP session. Your agent
now uses MCP tools transparently."

Day 2 built the server, Day 3 built a client that lists tools and calls one by
hand. Today the hardcoded call list is replaced by the agent's tool-call loop.
The bridge has two halves, both in this file:

1. **Listing → schemas** — :func:`mcp_tools_to_openai` converts the MCP
   ``tools/list`` response into the OpenAI tool-schema format our agent loop
   already speaks. The agent never sees MCP; it sees ordinary tool schemas.
2. **Tool calls → MCP session** — the agent loop's dispatcher is replaced by
   ``await session.call_tool()``. When the model emits a tool call, we route
   it through the MCP session (JSON-RPC over stdio to the server child
   process) and feed the returned text back as the ``tool`` message — the
   same message shape the loop used for native week-3 tools.

"Transparently" means the agent cannot tell the difference: the tool names,
descriptions and schemas are what the MCP server advertised, and the results
are the same week-3 JSON envelopes (the server preserved the contract on
Day 2). The agent just works with tools it was given.

Two demo tasks prove the bridge end to end:

- ``t1_roundtrip`` — write a note via MCP, read it back via MCP, and read a
  pre-existing file that only exists in the *server's* sandbox. Graded with
  filesystem checks (the sandbox is ground truth, week-4 style).
- ``t2_escape_denied`` — ask for ``/etc/hosts``; the denial must cross the
  protocol as ``is_error=True`` + the classified envelope, and the agent must
  read the hint, adapt, and prove it understood. This is the "tool ran,
  request denied" vs "tool crashed" distinction Day 3 flagged as Day 4's
  requirement.

Run:

    export $(grep DASHSCOPE_API_KEY .env | xargs)
    python experiments/week08/day4_mcp_agent.py --selftest   # offline: list + convert, no API
    python experiments/week08/day4_mcp_agent.py              # full run (needs DASHSCOPE_API_KEY)

Writes ``day4_results_<timestamp>.json`` next to this file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SERVER_PATH = HERE / "mcp_server.py"
SANDBOX = HERE / "sandbox"

CHAT_MODEL = os.getenv("MODEL_NAME", "qwen3.7-plus")
CHAT_BASE = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MAX_TURNS = 15  # safety valve — prevent runaway agent loops


# ---------------------------------------------------------------------------
# The bridge, half 1: MCP tool listings → OpenAI tool schemas
# ---------------------------------------------------------------------------
def mcp_tools_to_openai(tools: list) -> list[dict]:
    """Convert MCP ``tools/list`` entries into OpenAI tool schemas.

    MCP and OpenAI tool calling both use JSON Schema for parameters, but the
    *envelope* differs:

    - MCP listing: ``{name, description, inputSchema}`` (raw tool descriptor)
    - OpenAI tools: ``{"type": "function",
                       "function": {"name", "description", "parameters"}}``

    The parameter schema itself is a passthrough — both sides already speak
    JSON Schema. The conversion is the envelope plus defensive normalization
    for servers that send incomplete schemas (no ``inputSchema`` at all, or
    one missing ``type: object``). Any stdio server should survive this
    conversion, not just ours.
    """
    converted = []
    for t in tools:
        # The wire protocol calls it `inputSchema`; the SDK's Python objects
        # expose the same field as snake_case `input_schema`.
        params = t.input_schema if t.input_schema else {}
        params = dict(params)  # SDK may hand back pydantic objects
        if params.get("type") != "object":
            # MCP only defines tool inputs as objects; normalize defensively
            # so the chat API never sees a broken parameters block.
            params = {"type": "object",
                      "properties": params.get("properties", {})}
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": params,
                },
            }
        )
    return converted


# ---------------------------------------------------------------------------
# The bridge, half 2: the agent loop with MCP as the tool backend
# ---------------------------------------------------------------------------
def _text_of(result) -> str:
    """Pull the plain text out of a CallToolResult's content blocks."""
    parts = []
    for block in result.content:
        parts.append(getattr(block, "text", ""))
    return "\n".join(parts)


def _parse_envelope(text: str) -> dict:
    """Best-effort parse of the week-3 result envelope inside the text."""
    try:
        env = json.loads(text)
        if isinstance(env, dict) and "status" in env:
            return env
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def _make_error(err_type: str, message: str, hint: str = "") -> str:
    """Synthesize a week-3 envelope for failures the server can't express."""
    return json.dumps(
        {
            "status": "error",
            "data": None,
            "error": {"type": err_type, "message": message, "hint": hint},
            "truncated": False,
            "truncated_at": None,
        }
    )


AGENT_SYSTEM = (
    "You are a helpful assistant with file tools served over MCP. All file "
    "paths must be relative to the sandbox root (no leading '/' or '..'). "
    "IMPORTANT — All tool results use a consistent JSON envelope:\n"
    '  {"status":"success"|"error", "data":{…}, "error":null|{…}, '
    '"truncated":bool, "truncated_at":int|null}\n'
    "- On success: check .data for the tool-specific payload.\n"
    "- On error: read .error.type for the error category and .error.hint for "
    "an actionable suggestion on what to try next. ALWAYS read the hint "
    "before retrying — it tells you how to fix the problem."
)


async def run_agent_via_mcp(chat_client, session, tools, user_prompt: str) -> dict:
    """Non-interactive agent loop (week-3 pattern), tools routed via MCP.

    Identical in shape to the week-3/week-7 loops except the dispatcher:
    every tool call becomes ``session.call_tool`` over stdio. The model's
    view of the world — schemas in, envelope text out — is unchanged.
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    usage = {"prompt": 0, "completion": 0}
    trace: list[dict] = []

    for turn in range(1, MAX_TURNS + 1):
        r = await chat_client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=tools, temperature=0.0
        )
        if r.usage:
            usage["prompt"] += r.usage.prompt_tokens or 0
            usage["completion"] += r.usage.completion_tokens or 0

        msg = r.choices[0].message
        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            return {"answer": msg.content or "[no text]", "n_turns": turn,
                    "tool_calls": trace, "usage": usage, "error": None}

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                output = _make_error("invalid_input",
                                     f"invalid JSON arguments: {e}")
                trace.append({"name": tc.function.name,
                              "error": f"bad JSON: {e}"})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": output})
                continue

            t0 = time.perf_counter()
            try:
                # --- the routing step: the tool call goes through MCP -------
                result = await session.call_tool(tc.function.name, args)
                output = _text_of(result)
                rec = {
                    "name": tc.function.name,
                    "arguments": args,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "mcp_is_error": result.is_error,
                    "envelope": _parse_envelope(output),
                }
            except Exception as exc:
                # Transport-level failure — the MCP server itself broke, not
                # the request. Distinct from is_error (tool ran, denied).
                output = _make_error("unknown",
                                     f"MCP transport error: {exc}")
                rec = {
                    "name": tc.function.name,
                    "arguments": args,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "transport_error": f"{type(exc).__name__}: {exc}",
                }
            trace.append(rec)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": output})

    return {"answer": f"[stopped after {MAX_TURNS} turns]",
            "n_turns": MAX_TURNS, "tool_calls": trace, "usage": usage,
            "error": "max_turns_exceeded"}


# ---------------------------------------------------------------------------
# Demo tasks — graded with filesystem + answer checks
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower())


def _has_any(text: str, alts: list[str]) -> bool:
    n = _normalize(text)
    return any(_normalize(a) in n for a in alts)


TASKS = [
    {
        "id": "t1_roundtrip",
        "prompt": (
            "Write a short markdown note to 'mcp-agent-notes.md' in the "
            "sandbox. The note must have the title 'MCP Agent Notes' and one "
            "bullet each about: (1) how tools reach you — through MCP; "
            "(2) the sandbox path rule — relative paths only, no leading "
            "'/' or '..'; (3) which model you are. Then read the file back "
            "to verify it was written, and also read 'hello.txt' and report "
            "exactly what its first line says."
        ),
        "file_checks": [
            ("mcp-agent-notes.md exists",
             lambda p: p.exists()),
            ("note has the title",
             lambda p: _has_any(p.read_text(encoding="utf-8"),
                                ["MCP Agent Notes"])),
            ("note mentions MCP",
             lambda p: _has_any(p.read_text(encoding="utf-8"), ["MCP"])),
            ("note mentions the path rule",
             lambda p: _has_any(p.read_text(encoding="utf-8"),
                                ["relative", "no leading"])),
        ],
        "answer_checks": [
            ("answer quotes hello.txt first line",
             lambda a: _has_any(a, ["Hello from the week08 sandbox"])),
        ],
    },
    {
        "id": "t2_escape_denied",
        "prompt": (
            "Try to read the file '/etc/hosts' (an absolute path outside "
            "the sandbox). The tool should deny this. In your final answer "
            "you MUST list three things in order, one per line, each "
            "labelled: (1) ERROR TYPE: the value of error.type from the "
            "denial; (2) ERROR MESSAGE: the value of error.message; "
            "(3) HINT: the value of error.hint. Then demonstrate you "
            "understood by reading 'hello.txt' and quoting its first line."
        ),
        "file_checks": [],
        "answer_checks": [
            ("answer reports a sandbox-escape denial",
             lambda a: _has_any(a, ["sandbox escape", "sandbox_escape",
                                    "denied"])),
            ("answer mentions the hint",
             lambda a: _has_any(a, ["hint"])),
            ("answer quotes hello.txt first line",
             lambda a: _has_any(a, ["Hello from the week08 sandbox"])),
        ],
        "trace_checks": [
            ("the /etc/hosts call crossed MCP with is_error=true",
             lambda tr: any(c.get("arguments", {}).get("filepath")
                            == "/etc/hosts" and c.get("mcp_is_error")
                            for c in tr)),
            ("denial envelope carried type sandbox_escape",
             lambda tr: any(c.get("envelope", {}).get("error", {})
                            .get("type") == "sandbox_escape" for c in tr)),
        ],
    },
]


def _grade(task: dict, run: dict) -> list[dict]:
    """Run the task's checks against the run result; return check outcomes."""
    checks = []
    sandbox_p = SANDBOX / "mcp-agent-notes.md"

    for label, fn in task.get("file_checks", []):
        try:
            checks.append({"check": label, "passed": bool(fn(sandbox_p))})
        except Exception as e:
            checks.append({"check": label, "passed": False,
                           "note": f"{type(e).__name__}: {e}"})
    for label, fn in task.get("answer_checks", []):
        checks.append({"check": label, "passed": bool(fn(run["answer"]))})
    for label, fn in task.get("trace_checks", []):
        checks.append({"check": label,
                       "passed": bool(fn(run["tool_calls"]))})
    return checks


# ---------------------------------------------------------------------------
# The demo run: connect → list → convert → agent loop → grade
# ---------------------------------------------------------------------------
async def _run_demo(args) -> dict:
    started = time.perf_counter()
    results: dict = {
        "day": "week08_day4",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "chat_model": CHAT_MODEL,
            "chat_base": CHAT_BASE,
            "temperature": 0.0,
            "max_turns": MAX_TURNS,
            "server": {
                "type": "stdio",
                "command": sys.executable,
                "args": [str(SERVER_PATH)],
                "cwd": str(PROJECT_ROOT),
            },
            "mcp_sdk_version": version("mcp"),
        },
        "session": {},
        "tasks": [],
    }

    params = StdioServerParameters(
        command=sys.executable, args=[str(SERVER_PATH)], cwd=str(PROJECT_ROOT)
    )

    # trust_env=False: the dead macOS system proxy (127.0.0.1:7890) would
    # otherwise break the DashScope calls on this machine.  The selftest
    # makes no LLM calls, so it skips the client (and the API key) entirely.
    chat_client = None
    if not args.selftest:
        chat_client = AsyncOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=CHAT_BASE,
            http_client=httpx.AsyncClient(trust_env=False,
                                          timeout=httpx.Timeout(600.0,
                                                                connect=15.0)),
        )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                results["session"]["initialize"] = {
                    "protocol_version": init.protocol_version,
                    "server_name": init.server_info.name,
                    "server_version": init.server_info.version,
                    "capabilities": json.loads(
                        init.capabilities.model_dump_json()
                    ),
                }

                listing = await session.list_tools()
                listed = [
                    {"name": t.name, "description": t.description,
                     "input_schema": t.input_schema}
                    for t in listing.tools
                ]
                results["session"]["tools_listed"] = listed

                # --- the bridge: MCP listings become OpenAI tool schemas ----
                tools = mcp_tools_to_openai(listing.tools)
                results["session"]["openai_tools"] = tools

                if args.selftest:
                    results["session"]["note"] = (
                        "selftest: listing + conversion only, no LLM calls"
                    )
                    results["summary"] = {
                        "tools_listed": len(listed),
                        "openai_tools_converted": len(tools),
                        "tasks": 0, "checks_passed": 0, "checks_total": 0,
                        "total_s": round(time.perf_counter() - started, 1),
                    }
                    return results

                for task in [t for t in TASKS if t["id"] in args.tasks]:
                    t0 = time.perf_counter()
                    print(f"\n=== {task['id']} ===")
                    print(f"user: {task['prompt']}")
                    run = await run_agent_via_mcp(
                        chat_client, session, tools, task["prompt"]
                    )
                    run["wall_s"] = round(time.perf_counter() - t0, 1)
                    checks = _grade(task, run)
                    cell = {"id": task["id"], "prompt": task["prompt"],
                            "run": run, "checks": checks,
                            "passed": sum(c["passed"] for c in checks),
                            "total": len(checks)}
                    # snapshot ground truth (week-7 convention)
                    note = SANDBOX / "mcp-agent-notes.md"
                    if task["id"] == "t1_roundtrip" and note.exists():
                        cell["snapshot"] = {
                            "file": "sandbox/mcp-agent-notes.md",
                            "content": note.read_text(encoding="utf-8"),
                        }
                    results["tasks"].append(cell)
                    _print_task(cell)
    finally:
        if chat_client is not None:
            await chat_client.close()

    results["summary"] = {
        "tools_listed": len(results["session"].get("tools_listed", [])),
        "openai_tools_converted": len(
            results["session"].get("openai_tools", [])
        ),
        "tasks": len(results["tasks"]),
        "checks_passed": sum(t["passed"] for t in results["tasks"]),
        "checks_total": sum(t["total"] for t in results["tasks"]),
        "total_s": round(time.perf_counter() - started, 1),
    }
    return results


def _print_task(cell: dict) -> None:
    run = cell["run"]
    print(f"  turns: {run['n_turns']}, "
          f"tool calls: {len(run['tool_calls'])}")
    for c in run["tool_calls"]:
        args = c.get("arguments", {})
        if "content" in args:
            args_preview = {k: (v[:40] + "…" if isinstance(v, str)
                                and len(v) > 40 else v)
                            for k, v in args.items()}
        else:
            args_preview = args
        if "transport_error" in c:
            status = f"TRANSPORT ERROR: {c['transport_error']}"
        elif c.get("mcp_is_error"):
            err = c.get("envelope", {}).get("error", {})
            status = (f"denied ({err.get('type')}) — {err.get('message')} "
                      f"| hint: {err.get('hint')}")
        else:
            data = c.get("envelope", {}).get("data", {})
            preview = repr(data)[:80] if data else ""
            status = f"success {preview}"
        print(f"  [{c.get('latency_ms', 0):>6.1f} ms] "
              f"{c['name']}({json.dumps(args_preview, ensure_ascii=False)})"
              f" -> {status}")
    for ch in cell["checks"]:
        mark = "✅" if ch["passed"] else "❌"
        print(f"  {mark} {ch['check']}" + (f" — {ch['note']}"
                                           if ch.get("note") else ""))
    usage = run["usage"]
    print(f"  usage: {usage['prompt']} prompt / {usage['completion']} "
          f"completion tokens; {run['wall_s']} s wall")
    print(f"  answer: {run['answer']}")


def _print_report(results: dict) -> None:
    init = results["session"]["initialize"]
    print("=" * 72)
    print("Day 4 MCP agent — tool calls routed through the MCP session")
    print("=" * 72)
    print(f"server:  {init['server_name']} {init['server_version']} "
          f"(protocol {init['protocol_version']})")
    print(f"model:   {results['config']['chat_model']} "
          f"(temperature {results['config']['temperature']})")
    print(f"bridge:  {results['summary']['tools_listed']} MCP tools listed → "
          f"{results['summary']['openai_tools_converted']} OpenAI schemas")
    print("converted schemas:")
    for t in results["session"]["openai_tools"]:
        f = t["function"]
        params = ", ".join(f["parameters"].get("properties", {}).keys())
        print(f"  - {f['name']}({params})")
    if results["session"].get("note"):
        print(f"note:    {results['session']['note']}")
        return
    s = results["summary"]
    print()
    print(f"checks:  {s['checks_passed']}/{s['checks_total']} passed "
          f"across {s['tasks']} tasks in {s['total_s']} s")
    failed = [t["id"] for t in results["tasks"]
              if t["passed"] < t["total"]]
    print(f"result:  {'ALL PASSED ✅' if not failed else 'FAILED: ' + ', '.join(failed)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="list tools + convert schemas only, no LLM calls")
    ap.add_argument("--tasks", default=",".join(t["id"] for t in TASKS),
                    help="comma-separated task ids to run (default: all)")
    args = ap.parse_args()
    args.tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}

    if not args.selftest and "DASHSCOPE_API_KEY" not in os.environ:
        sys.exit(
            "DASHSCOPE_API_KEY not set — run:\n"
            "  export $(grep DASHSCOPE_API_KEY "
            f"{PROJECT_ROOT / '.env'} | xargs)"
        )

    results = asyncio.run(_run_demo(args))
    suffix = "selftest" if args.selftest else datetime.now().strftime("%Y%m%d_%H%M%S")
    out = HERE / f"day4_results_{suffix}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), "utf-8")
    _print_report(results)
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
