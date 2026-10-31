#!/usr/bin/env python3
"""Week 8 Day 2: our own MCP server (stdio transport) — 2 week-3 tools.

PLAN spec: "Build your own MCP server exposing 2 of your week-3 tools
(stdio transport). Test with the inspector."

Which two tools: **read_file** and **write_file** — the week-3 Day-1 file
pair. The round trip (write via MCP, read it back via MCP) proves state
actually flows through the protocol, and both tools exercise the week-3
sandbox path-safety boundary.

How it works:

- Built on the official MCP Python SDK (``mcp`` 2.x — FastMCP was renamed
  ``MCPServer`` in 2.0).  ``mcp.run()`` with the default ``stdio`` transport
  means the server speaks JSON-RPC over stdin/stdout; the Inspector (or any
  MCP host) spawns this file as a child process.
- The week-3 implementations are *imported, not copied*:
  ``experiments/week03/agent.py`` provides ``read_file`` / ``write_file``
  plus ``resolve_sandbox_path`` (realpath + commonpath containment check).
  Its module-level ``SANDBOX_ROOT`` is pointed at ``week08/sandbox`` (the
  directory already used on Day 1), so the MCP server is confined there.
- The week-3 result contract is preserved: tools return the same JSON
  envelope (``status`` / ``data`` / ``error`` / ``truncated``) as plain
  text, and error envelopes are flagged ``is_error=True`` so MCP clients
  (and, on Day 4, our agent) can tell success from denial apart.
- Week-3 tools never raise — every failure (escape attempt, bad extension,
  missing file) comes back as a classified error envelope with an
  actionable hint.  That is exactly the error model MCP wants: the tool
  *ran*, the request was denied.

Run it directly (the Inspector launches it the same way):

    python experiments/week08/mcp_server.py

Smoke-test the stdio handshake without the Inspector:

    printf '%s\n' \
      '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
      '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
      '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
    | python experiments/week08/mcp_server.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import mcp_types
from mcp.server.mcpserver import MCPServer
from pydantic import Field

HERE = Path(__file__).resolve().parent

# --- reuse the real week-3 tool implementations ---------------------------
# agent.py lives in experiments/week03; add it to sys.path explicitly so the
# server works no matter which cwd the Inspector spawns it from.
sys.path.insert(0, str(HERE.parent / "week03"))
import agent as w3  # noqa: E402

# Confine the week-3 sandbox to this week's sandbox directory.  The week-3
# tools read this module-level global at call time, so pointing it at
# week08/sandbox moves the whole security boundary here.
w3.SANDBOX_ROOT = HERE / "sandbox"

# --- the MCP server --------------------------------------------------------
mcp = MCPServer(
    name="week08-sandbox-files",
    version="0.1.0",
    description=(
        "Two week-3 sandbox file tools (read_file, write_file) exposed over "
        "MCP stdio. All paths are relative to the sandbox root; realpath + "
        "commonpath containment keeps every access inside it."
    ),
    instructions=(
        "Paths must be relative (no leading '/' or '..'). Allowed extensions: "
        ".txt, .md, .py, .json, .csv. Escapes and bad extensions return a "
        "classified error envelope with a hint — read it and adjust."
    ),
)


def _result(envelope: str) -> mcp_types.CallToolResult:
    """Map a week-3 result envelope to an MCP tool result.

    The week-3 tools never raise; they return a JSON envelope like::

        {"status":"success","data":{…},"error":null,"truncated":false,…}
        {"status":"error","data":null,"error":{"type":…,"message":…,"hint":…},…}

    We keep that exact contract as the text block (the model learned it in
    week 3) and flag error envelopes with ``is_error`` so MCP clients render
    and react to denials as errors.
    """
    try:
        is_error = json.loads(envelope).get("status") == "error"
    except (json.JSONDecodeError, AttributeError):
        is_error = False
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=envelope)],
        is_error=is_error,
    )


@mcp.tool(
    description=(
        "Read a text file from the sandbox workspace. Call when the user "
        "asks to see or summarize a file. Path must be relative to the "
        "sandbox root (no leading '/' or '..'); content is capped at 8000 "
        "chars with a truncation marker."
    )
)
def read_file(
    filepath: Annotated[str, Field(description="Relative path, e.g. 'hello.txt' or 'notes/mcp-notes.md'.")],
) -> mcp_types.CallToolResult:
    """Week-3 read_file over MCP."""
    return _result(w3.read_file(filepath))


@mcp.tool(
    description=(
        "Write or overwrite a text file in the sandbox workspace. Call when "
        "the user asks to save, create, or update a file. Path must stay "
        "inside the sandbox; only .txt/.md/.py/.json/.csv extensions are "
        "allowed."
    )
)
def write_file(
    filepath: Annotated[str, Field(description="Relative path inside the sandbox, e.g. 'summary.md'.")],
    content: Annotated[str, Field(description="Full file contents to write.")],
) -> mcp_types.CallToolResult:
    """Week-3 write_file over MCP."""
    return _result(w3.write_file(filepath, content))


if __name__ == "__main__":
    mcp.run(transport="stdio")
