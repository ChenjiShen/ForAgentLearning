#!/usr/bin/env python3
"""Week 8 Day 3: an MCP client for our Day-2 server — no LLM, just the protocol.

PLAN spec: "Write an MCP client that connects to your server, lists tools,
and calls one — no LLM yet, just the protocol."

This is the client half of the MCP relationship. Day 2 built the server
(``mcp_server.py``); this script is what a host would do *before* an LLM ever
gets involved:

1. **Spawn** the server as a child process (stdio transport — the client owns
   the server's lifecycle; no port, no auth, no HTTP).
2. **Initialize** the session (``initialize`` → server info, protocol version,
   capabilities) and **list tools** (``tools/list`` → the 2 JSON Schemas).
3. **Call tools** (``tools/call``): a write→read round trip through MCP to
   prove state flows, plus the two denial cases so we can observe how errors
   cross the protocol (``is_error`` + the week-3 classified envelope).

Design points:

- Built on the official SDK's client side (``mcp.client.stdio.stdio_client`` +
  ``ClientSession``), the mirror image of Day 2's server side. Both sides
  speak JSON-RPC over stdin/stdout; the SDK handles framing, request/response
  matching, and notifications for us.
- The client is **generic**: it discovers tools via ``list_tools`` and never
  hardcodes the tool set — the call list below is just data. Swap in any
  other stdio server and the discovery part still works.
- No LLM anywhere: no API key, no model, no prompt. Tool selection is a
  hardcoded list. (Day 4 replaces that list with the agent's tool-call loop.)
- Every call is timed (``perf_counter``) — cheap now, and the numbers seed
  Day 6's native-vs-MCP latency experiment.
- Unlike Day 1/2, nothing here touches HTTP, so the dead macOS system proxy
  (127.0.0.1:7890) cannot break it — stdio needs no proxy fix.

Run it:

    python experiments/week08/mcp_client.py

Writes ``day3_results_<timestamp>.json`` next to this file.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SERVER_PATH = HERE / "mcp_server.py"

# The protocol pokes: (label, tool, arguments, expected outcome).
# "denied" means the call should come back is_error=True with a classified
# week-3 envelope — a *successful protocol round trip* whose payload says no.
CALLS = [
    (
        "write mcp-via-client.md (round trip: write)",
        "write_file",
        {
            "filepath": "mcp-via-client.md",
            "content": (
                "# Written by the Day-3 MCP client\n\n"
                f"Generated at {datetime.now().isoformat(timespec='seconds')} "
                "by experiments/week08/mcp_client.py — no LLM involved, "
                "just the protocol.\n"
            ),
        },
        "success",
    ),
    (
        "read mcp-via-client.md back (round trip: read)",
        "read_file",
        {"filepath": "mcp-via-client.md"},
        "success",
    ),
    (
        "overwrite mcp-via-client.md (state changes stick)",
        "write_file",
        {"filepath": "mcp-via-client.md", "content": "overwritten via MCP\n"},
        "success",
    ),
    (
        "read pre-existing sandbox file hello.txt",
        "read_file",
        {"filepath": "hello.txt"},
        "success",
    ),
    (
        "read /etc/hosts (sandbox escape attempt)",
        "read_file",
        {"filepath": "/etc/hosts"},
        "denied",
    ),
    (
        "write binary.bin (bad extension attempt)",
        "write_file",
        {"filepath": "binary.bin", "content": "should be denied"},
        "denied",
    ),
]


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


async def _run_demo() -> dict:
    """Full MCP client session: connect, initialize, list, call, report."""
    started = time.perf_counter()
    results: dict = {
        "day": "week08_day3",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "client": {
            "type": "mcp python sdk stdio client",
            "sdk_version": version("mcp"),
            "llm": None,  # day 3: protocol only, no model
        },
        "server_config": {
            "type": "stdio",
            "command": sys.executable,
            "args": [str(SERVER_PATH)],
            "cwd": str(PROJECT_ROOT),
        },
        "session": {},
        "calls": [],
    }

    params = StdioServerParameters(
        command=sys.executable, args=[str(SERVER_PATH)], cwd=str(PROJECT_ROOT)
    )
    # stdio_client spawns the server as a child process; the context manager
    # gives us the read/write streams, ClientSession turns them into a
    # JSON-RPC session (handles initialize handshake, request ids, etc.).
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            results["session"]["initialize"] = {
                "protocol_version": init.protocol_version,
                "server_name": init.server_info.name,
                "server_version": init.server_info.version,
                "capabilities": json.loads(init.capabilities.model_dump_json()),
                "instructions": init.instructions,
            }

            listing = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in listing.tools
            ]
            results["session"]["tools"] = tools

            for label, tool, args, expected in CALLS:
                t0 = time.perf_counter()
                try:
                    result = await session.call_tool(tool, args)
                    text = _text_of(result)
                    call = {
                        "label": label,
                        "tool": tool,
                        "arguments": args,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "is_error": result.is_error,
                        "envelope": _parse_envelope(text),
                    }
                except Exception as exc:  # protocol/transport failure, not denial
                    call = {
                        "label": label,
                        "tool": tool,
                        "arguments": args,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "transport_error": f"{type(exc).__name__}: {exc}",
                    }
                results["calls"].append(call)

    n_ok = sum(
        1
        for c in results["calls"]
        if c.get("envelope", {}).get("status") == "success"
    )
    n_denied = sum(
        1
        for c in results["calls"]
        if c.get("is_error") and c.get("envelope", {}).get("status") == "error"
    )
    results["summary"] = {
        "tools_listed": len(results["session"].get("tools", [])),
        "calls_made": len(results["calls"]),
        "succeeded": n_ok,
        "denied_as_expected": n_denied,
        "total_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    return results


def _print_report(results: dict) -> None:
    """Human-readable summary on stdout."""
    init = results["session"]["initialize"]
    tools = results["session"]["tools"]
    print("=" * 72)
    print("Day 3 MCP client — direct SDK client, no LLM")
    print("=" * 72)
    print(f"server:  {init['server_name']} {init['server_version']} "
          f"(protocol {init['protocol_version']})")
    print(f"tools listed: {results['summary']['tools_listed']}")
    for t in tools:
        params = ", ".join(t["input_schema"].get("properties", {}).keys())
        desc = (t["description"] or "").split("\n")[0]
        print(f"  - {t['name']}({params}): {desc}")
    print()
    for c in results["calls"]:
        env = c.get("envelope", {})
        if "transport_error" in c:
            status = f"TRANSPORT ERROR: {c['transport_error']}"
        elif c["is_error"]:
            err = env.get("error", {})
            status = (f"denied ({err.get('type')}) — {err.get('message')}"
                      f" | hint: {err.get('hint')}")
        else:
            data = env.get("data", {})
            preview = ""
            if isinstance(data, dict) and "content" in data:
                preview = repr(data["content"])[:60]
            status = f"success {preview}"
        print(f"[{c['latency_ms']:>7.1f} ms] {c['tool']} — {c['label']}")
        print(f"            -> {status}")
    s = results["summary"]
    print()
    print(f"summary: {s['tools_listed']} tools listed, {s['calls_made']} calls "
          f"({s['succeeded']} success, {s['denied_as_expected']} denied as "
          f"expected) in {s['total_ms']} ms")


def main() -> None:
    results = asyncio.run(_run_demo())
    out = HERE / f"day3_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), "utf-8")
    _print_report(results)
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
