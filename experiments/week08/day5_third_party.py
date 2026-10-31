#!/usr/bin/env python3
"""Week 8 Day 5: connect our agent to a *third-party* MCP server we didn't write.

PLAN spec: "Connect your agent to a third-party MCP server you didn't write.
Note every friction point."

Day 4 connected the agent to OUR server (``mcp_server.py``) — same author,
same result contract (week-3 JSON envelopes), same arg names. Today the same
agent connects to ``@modelcontextprotocol/server-filesystem``, the official
reference server already installed on Day 1. We did not write it; we only
know what ``tools/list`` tells us.

The bridge is unchanged (``mcp_tools_to_openai`` + the MCP-routed agent loop,
copied from ``day4_mcp_agent.py``). What changes is everything the bridge
does *not* cover — and that gap is exactly what this experiment measures:

- the system prompt (Day 4's taught the agent our JSON envelope; this server
  returns plain text)
- arg names (same tool names, different schemas)
- error shape (``isError`` + bare text, no ``error.type``/``error.hint``)
- startup cost (npx + node vs python)
- tool volume (14 tools vs 2 — a larger schema block in every request)

Every friction point is captured in the results JSON under ``friction_points``,
each with severity + evidence + workaround.

Three demo tasks, graded week-4 style (the sandbox filesystem is ground
truth):

- ``t1_discover_and_read`` — list the sandbox, read ``hello.txt``, quote it.
- ``t2_write_roundtrip`` — write ``thirdparty-agent-notes.md`` through the
  third-party server, read it back, quote the title.
- ``t3_escape_denied`` — read ``/etc/hosts``; the denial arrives as ``isError``
  + plain text (no classified envelope), and the agent must report it and
  adapt.

Run:

    export $(grep DASHSCOPE_API_KEY .env | xargs)
    python experiments/week08/day5_third_party.py --selftest   # offline: list + convert + friction evidence
    python experiments/week08/day5_third_party.py              # full run (needs DASHSCOPE_API_KEY)

Writes ``day5_results_<timestamp>.json`` next to this file.
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
OUR_SERVER_PATH = HERE / "mcp_server.py"
SANDBOX = HERE / "sandbox"

# The third-party server (user-approved): the official reference filesystem
# server, installed on Day 1.  We never wrote it and only know it through
# the protocol.
THIRD_PARTY_COMMAND = "npx"
THIRD_PARTY_ARGS = ["-y", "@modelcontextprotocol/server-filesystem", "sandbox"]

CHAT_MODEL = os.getenv("MODEL_NAME", "qwen3.7-plus")
CHAT_BASE = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MAX_TURNS = 15  # safety valve — prevent runaway agent loops


# ---------------------------------------------------------------------------
# The bridge, half 1: MCP tool listings → OpenAI tool schemas
# (unchanged from day4_mcp_agent.py — it was written server-agnostic)
# ---------------------------------------------------------------------------
def mcp_tools_to_openai(tools: list) -> list[dict]:
    """Convert MCP ``tools/list`` entries into OpenAI tool schemas.

    The envelope conversion is the same for any stdio server; the parameter
    schema is a passthrough (both sides speak JSON Schema).
    """
    converted = []
    for t in tools:
        # Accept SDK Tool objects or plain dicts (e.g. from our session
        # facts snapshot) — the schema content is the same either way.
        if isinstance(t, dict):
            name, desc = t.get("name"), t.get("description")
            params = t.get("input_schema") or {}
        else:
            name, desc = t.name, t.description
            params = t.input_schema if t.input_schema else {}
        params = dict(params)  # SDK may hand back pydantic objects
        if params.get("type") != "object":
            params = {"type": "object",
                      "properties": params.get("properties", {})}
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc or "",
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


def _make_error(err_type: str, message: str, hint: str = "") -> str:
    """Synthesize a week-3 envelope for failures the server can't express
    (kept for transport-level failures only — the agent's prompt no longer
    teaches the envelope)."""
    return json.dumps(
        {
            "status": "error",
            "data": None,
            "error": {"type": err_type, "message": message, "hint": hint},
            "truncated": False,
            "truncated_at": None,
        }
    )


# NOTE (friction #1): this prompt is NOT Day 4's.  Day 4's prompt taught the
# agent our week-3 JSON envelope (status/data/error/truncated, read
# error.hint).  The third-party server returns plain text and bare denial
# messages, so the prompt had to be rewritten for the new server's contract.
AGENT_SYSTEM = (
    "You are a helpful assistant with file tools served over MCP by a "
    "third-party filesystem server. You have not been given its source "
    "code; rely only on the tool schemas you were given.\n"
    "Rules:\n"
    "- All file paths must be inside the sandbox the server allows. Do not "
    "use absolute paths like '/etc/hosts' and do not use '..'.\n"
    "- Tool results are PLAIN TEXT. Success returns file contents or a "
    "listing; denial returns a short error message.\n"
    "- If a tool reports a denial, read the message, do not repeat the "
    "same call, and adapt using a path inside the sandbox.\n"
    "- Parameter names differ per tool — read each tool's schema before "
    "calling it."
)


async def run_agent_via_mcp(chat_client, session, tools, user_prompt: str) -> dict:
    """Non-interactive agent loop (week-3 pattern), tools routed via MCP.

    Identical in shape to the week-3/week-7 loops except the dispatcher:
    every tool call becomes ``session.call_tool`` over stdio. The model's
    view of the world — schemas in, result text out — is unchanged.
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
                              "error": f"bad JSON: {e}",
                              "output_preview": output[:200]})
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
                    "output_preview": output[:200],
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
# Friction capture
# ---------------------------------------------------------------------------
class _ErrCapture:
    """Capture the server child's stderr through a real pipe fd.

    stdio gives the server exactly one protocol channel (stdout); its logs
    go to stderr and would otherwise leak into our terminal.  The SDK
    passes ``errlog.fileno()`` straight to the subprocess as the child's
    stderr, so this object only needs to BE a pipe: the SDK hands the write
    end to the child, and we read the read end non-blockingly on demand.

    Deliberately NO blocking wait for EOF: npx's child tree can survive the
    SDK's session close (observed), which would keep the write end open and
    hang any EOF-waiting drain forever.
    """

    def __init__(self):
        self._r, self._w = os.pipe()
        os.set_blocking(self._r, False)
        self.lines: list[str] = []

    def fileno(self) -> int:
        return self._w

    def collect(self) -> None:
        """Non-blocking read of whatever the child has written so far."""
        while True:
            try:
                chunk = os.read(self._r, 65536)
            except BlockingIOError:
                return
            if not chunk:
                return
            self.lines.append(chunk.decode("utf-8", errors="replace"))

    def close_write_end(self) -> None:
        """Close the parent's copy of the write end (safe after session)."""
        try:
            os.close(self._w)
        except OSError:
            pass

    def text(self) -> str:
        self.collect()
        return "".join(self.lines).strip()


async def _session_facts(session, errlog, t0: float) -> dict:
    """Initialize + list tools + probe on an open session. Returns facts."""
    init = await session.initialize()
    spawn_ms = round((time.perf_counter() - t0) * 1000, 0)
    listing = await session.list_tools()
    t1 = time.perf_counter()
    probe = await session.call_tool("list_allowed_directories", {})
    probe_ms = round((time.perf_counter() - t1) * 1000, 1)
    return {
        "server_name": init.server_info.name,
        "server_version": init.server_info.version,
        "protocol_version": init.protocol_version,
        "capabilities": json.loads(
            init.capabilities.model_dump_json()
        ),
        "tools_listed": [
            {"name": t.name, "description": t.description,
             "input_schema": t.input_schema}
            for t in listing.tools
        ],
        "spawn_init_ms": spawn_ms,
        "probe_call_ms": probe_ms,
        "probe_result": _text_of(probe).strip(),
        "probe_is_error": probe.is_error,
        "stderr": errlog.text(),
    }


async def _connect(params, errlog) -> dict:
    """Spawn a server and return its session facts (for one-shot probes)."""
    t0 = time.perf_counter()
    async with stdio_client(params, errlog=errlog) as (read, write):
        async with ClientSession(read, write) as session:
            facts = await _session_facts(session, errlog, t0)
    errlog.close_write_end()
    return facts


def _schema_diff(their_listing: list[dict]) -> list[dict]:
    """Diff shared tool names between our Day-2 server and the third-party
    server — same tool names, different parameter names (friction #2)."""
    day4_selftest = HERE / "day4_results_selftest.json"
    ours = {}
    if day4_selftest.exists():
        data = json.loads(day4_selftest.read_text(encoding="utf-8"))
        ours = {t["name"]: t.get("input_schema") or {}
                for t in data["session"]["tools_listed"]}

    diffs = []
    for t in their_listing:
        if t["name"] not in ours:
            continue
        their_props = set((t["input_schema"] or {}).get("properties", {}))
        our_props = set(ours[t["name"]].get("properties", {}))
        diffs.append({
            "tool": t["name"],
            "our_params": sorted(our_props),
            "their_params": sorted(their_props),
            "only_ours": sorted(our_props - their_props),
            "only_theirs": sorted(their_props - our_props),
        })
    return diffs


# ---------------------------------------------------------------------------
# Demo tasks — graded with filesystem + answer checks (week-4 style)
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower())


def _has_any(text: str, alts: list[str]) -> bool:
    n = _normalize(text)
    return any(_normalize(a) in n for a in alts)


def _count_check(text: str) -> bool:
    """Answer reports a directory-listing count >= 3 (loose pattern)."""
    pat = re.compile(r"\b(\d+)(?:\s+\w+){0,4}\s*(?:item|items|file|files|entr\w+)",
                     re.I)
    return any(int(m.group(1)) >= 3 for m in pat.finditer(text))


TASKS = [
    {
        "id": "t1_discover_and_read",
        "prompt": (
            "Explore the sandbox this server gives you access to. First "
            "list the contents of the sandbox root directory, then read "
            "the file 'hello.txt' and report exactly what its first line "
            "says. In your final answer, also say how many items the "
            "directory listing showed."
        ),
        "file_checks": [],
        "answer_checks": [
            ("answer quotes hello.txt first line",
             lambda a: _has_any(a, ["Hello from the week08 sandbox"])),
            ("answer reports a listing count of 3 or more",
             lambda a: _count_check(a)),
        ],
        "trace_checks": [
            ("some directory-listing tool was called",
             lambda tr: any(c["name"] in {"list_directory", "directory_tree",
                                          "list_directory_with_sizes"}
                            for c in tr)),
            ("hello.txt was read through the third-party server",
             lambda tr: any(c["name"] in {"read_file", "read_text_file"}
                            and str(c.get("arguments", {}).get("path", ""))
                            .endswith("hello.txt") for c in tr)),
        ],
    },
    {
        "id": "t2_write_roundtrip",
        "prompt": (
            "Write a short markdown note to 'thirdparty-agent-notes.md' in "
            "the sandbox. The note must have the title 'Third-Party Server "
            "Notes' and one bullet naming the server that served you: the "
            "third-party filesystem server. Then read the file back and "
            "quote its title in your final answer."
        ),
        "file_checks": [
            ("thirdparty-agent-notes.md exists",
             lambda p: p.exists()),
            ("note has the title",
             lambda p: _has_any(p.read_text(encoding="utf-8"),
                                ["Third-Party Server Notes"])),
        ],
        "answer_checks": [
            ("answer quotes the title",
             lambda a: _has_any(a, ["Third-Party Server Notes"])),
        ],
        "trace_checks": [
            ("the write call was not denied",
             lambda tr: any(c["name"] == "write_file"
                            and str(c.get("arguments", {}).get("path", ""))
                            .endswith("thirdparty-agent-notes.md")
                            and not c.get("mcp_is_error") for c in tr)),
        ],
    },
    {
        "id": "t3_escape_denied",
        "prompt": (
            "Try to read the file '/etc/hosts'. The server should deny "
            "this because the path is outside the sandbox. In your final "
            "answer report three things, one per line, labelled: "
            "(1) DENIAL MESSAGE: the exact error text the tool returned; "
            "(2) LESSON: one sentence on what that teaches about paths; "
            "(3) then prove you adapted by reading 'hello.txt' and "
            "quoting its first line."
        ),
        "file_checks": [],
        "answer_checks": [
            ("answer reports the access denial",
             lambda a: _has_any(a, ["access denied", "outside allowed",
                                    "denied"])),
            ("answer quotes hello.txt first line",
             lambda a: _has_any(a, ["Hello from the week08 sandbox"])),
        ],
        "trace_checks": [
            ("the /etc/hosts call crossed MCP with is_error=true",
             lambda tr: any(c.get("arguments", {}).get("path") == "/etc/hosts"
                            and c.get("mcp_is_error") for c in tr)),
        ],
    },
]


def _grade(task: dict, run: dict, note_path: Path) -> list[dict]:
    """Run the task's checks against the run result; return check outcomes."""
    checks = []
    for label, fn in task.get("file_checks", []):
        try:
            checks.append({"check": label, "passed": bool(fn(note_path))})
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
# Friction points — assembled from live evidence
# ---------------------------------------------------------------------------
def _friction_points(their: dict, our: dict, diffs: list, cells: list) -> list:
    """The deliverable of the day: every friction point, with evidence."""
    # cells are task records {"id", "run": {...}, ...} — unwrap to runs
    runs = [c.get("run", {}) for c in cells]
    friction = []

    friction.append({
        "id": "prompt_contract_coupling",
        "severity": "blocker",
        "what": ("Day 4's system prompt taught the agent OUR server's result "
                 "contract (the week-3 JSON envelope: status/data/error/"
                 "truncated, 'read error.hint'). The third-party server "
                 "returns plain text with bare denial strings. A prompt "
                 "tuned to one server silently misinforms the agent about "
                 "another — the agent would go looking for error.hint that "
                 "does not exist."),
        "evidence": ("AGENT_SYSTEM had to be rewritten for the new server "
                     "(see the note above its definition in this file); "
                     "the third-party server's denial output contains no "
                     "JSON envelope"),
        "workaround": ("Treat the system prompt as server-specific config: "
                       "rewrite it whenever the server changes, or keep it "
                       "generic ('read each tool's result carefully')."),
    })

    friction.append({
        "id": "same_names_different_schemas",
        "severity": "warning",
        "what": ("Shared tool names hide different parameter names: both "
                 "our read_file and write_file take 'filepath' for the "
                 "file location, theirs take 'path'. Hardcoded args or "
                 "wrapper code written for one server break against the "
                 "other, and the names collide so the failure is "
                 "confusing."),
        "evidence": json.dumps(diffs),
        "workaround": ("Never hardcode tool args; let the LLM fill "
                       "parameters from the advertised schemas (the agent "
                       "loop already does this — no code change was "
                       "needed, which is the MCP promise working)."),
    })

    friction.append({
        "id": "error_contract_differs",
        "severity": "warning",
        "what": ("Denials arrive as isError=true plus a bare string "
                 "('Access denied - path outside allowed directories'), "
                 "with no classified error type or hint. Any client code "
                 "that parses the week-3 envelope gets nothing from this "
                 "server."),
        "evidence": _denial_evidence(runs),
        "workaround": ("Handle isError as the only cross-server error "
                       "signal; treat everything else about error shape "
                       "as server-specific."),
    })

    friction.append({
        "id": "spawn_and_call_latency",
        "severity": "annoyance",
        "what": ("npx spawns node, resolves the package, then starts the "
                 "server: spawn+initialize took ~1.4 s vs ~0.5 s for our "
                 "python server (measured same-day, same client); per-call "
                 "latency is also higher (~20-50 ms vs ~2-5 ms)."),
        "evidence": {
            "their_spawn_init_ms": their["spawn_init_ms"],
            "our_spawn_init_ms": our["spawn_init_ms"],
            "their_probe_call_ms": their["probe_call_ms"],
        },
        "workaround": ("Day 6 formalizes this comparison; for real apps, "
                       "keep long-lived sessions (pay the spawn once) or "
                       "prefer a local binary over npx indirection."),
    })

    friction.append({
        "id": "tool_volume_schema_bloat",
        "severity": "warning",
        "what": ("The third-party server advertises 14 tools vs our 2. "
                 "Every LLM request carries the full schema block, and a "
                 "wider surface means more chances for the model to pick "
                 "a near-miss tool (read_file vs read_text_file, "
                 "list_directory vs directory_tree)."),
        "evidence": {
            "their_tool_count": len(their["tools_listed"]),
            "our_tool_count": len(our["tools_listed"]),
            "their_openai_schema_bytes": their.get("openai_schema_bytes"),
        },
        "workaround": ("Filter the listing before conversion if the agent "
                       "doesn't need everything (the bridge makes this a "
                       "one-liner)."),
    })

    friction.append({
        "id": "absolute_path_convention",
        "severity": "warning",
        "what": ("This server's error messages, its "
                 "list_allowed_directories tool and its success replies "
                 "all use ABSOLUTE paths, and it accepts them in every "
                 "tool — unlike our server, which is relative-only. The "
                 "agent therefore stopped using relative paths even "
                 "though the prompt told it to: environment feedback "
                 "overrode the instruction. Worse, long absolute paths "
                 "invite transcription errors."),
        "evidence": _abs_path_evidence(runs),
        "workaround": ("Either teach the agent to prefer relative paths "
                       "explicitly, or normalize/alias paths in the client "
                       "bridge. Note the agent self-healed: after two "
                       "denials it re-read the allowed directory and "
                       "corrected the path."),
    })

    friction.append({
        "id": "roots_capability_not_sent",
        "severity": "annoyance",
        "what": ("The Python SDK client does not declare the MCP 'roots' "
                 "capability, so the server cannot ask the host which "
                 "directories to allow and falls back to CLI args — "
                 "printing a note to stderr. A full host (Claude Desktop) "
                 "would negotiate roots dynamically."),
        "evidence": ("server stderr: " + (their["stderr"] or "(empty)")),
        "workaround": ("Pass allowed directories as server CLI args (what "
                       "we did), or extend the client to send roots."),
    })

    friction.append({
        "id": "stderr_log_leakage",
        "severity": "annoyance",
        "what": ("stdio has exactly one protocol channel (stdout); server "
                 "logs go to stderr and leak into the host process's "
                 "output unless the client captures them (we did, with a "
                 "custom errlog)."),
        "evidence": ("captured stderr lines: " +
                     repr(their["stderr"].splitlines()[:3])),
        "workaround": ("Pass errlog to stdio_client and keep server logs "
                       "out of the agent transcript."),
    })

    friction.append({
        "id": "orphan_child_tree_after_session_close",
        "severity": "warning",
        "what": ("Spawned through npx, the server is npm -> node grandchild "
                 "chain. The SDK's session close kills the direct child "
                 "but the grandchild can survive as an orphan, keeping "
                 "its inherited stderr pipe open. Our first implementation "
                 "drained stderr with a blocking read-until-EOF, which "
                 "hung forever against those surviving children. npx "
                 "adds a process layer that makes lifecycle a real "
                 "problem; spawning the node binary directly would avoid "
                 "it."),
        "evidence": ("ps after the first selftest run showed orphaned "
                     "'npm exec' + node mcp-server-filesystem processes "
                     "still holding the sandbox cwd; the run hung in the "
                     "stderr drain until killed"),
        "workaround": ("Never block on child-stderr EOF when npx is in "
                       "the chain; read stderr non-blockingly on demand "
                       "(this script does exactly that)."),
    })

    friction.append({
        "id": "protocol_version_drift",
        "severity": "note",
        "what": ("Day 1's Inspector session negotiated protocol "
                 "2025-06-18; today's SDK client negotiated 2025-11-25; "
                 "the server accepted both. Version agreement is a "
                 "per-connection property — two clients of the same "
                 "server can be on different protocol versions."),
        "evidence": {
            "negotiated_now": their["protocol_version"],
            "negotiated_day1": "2025-06-18",
            "server": their["server_name"] + " " + their["server_version"],
        },
        "workaround": ("None needed — negotiation worked; just don't "
                       "assume a fixed protocol version in logs or "
                       "migrations."),
    })

    return friction


def _abs_path_evidence(runs: list) -> dict:
    """From the traces: the absolute-path convention in action (t1's
    transcription error + self-heal; t2's absolute-path write)."""
    if not runs:
        return {"note": "tasks not run in this mode"}
    ev = {}
    for run in runs:
        calls = run.get("tool_calls", [])
        denials = [c for c in calls if c.get("mcp_is_error")]
        if denials:
            ev.setdefault("denials", []).extend([
                {"tool": c["name"],
                 "path": c.get("arguments", {}).get("path", "")[:100],
                 "message": c.get("output_preview", "")[:100]}
                for c in denials
            ])
        abs_paths = [c.get("arguments", {}).get("path", "")
                     for c in calls
                     if str(c.get("arguments", {}).get("path", ""))
                     .startswith("/")]
        if abs_paths:
            ev.setdefault("absolute_paths_used", []).extend(
                p[:100] for p in abs_paths
            )
    return ev


def _denial_evidence(runs: list) -> dict:
    """What the /etc/hosts denial actually looked like on the wire."""
    for run in runs:
        for c in run.get("tool_calls", []):
            if c.get("arguments", {}).get("path") == "/etc/hosts":
                return {
                    "is_error": c.get("mcp_is_error"),
                    "output_preview": c.get("output_preview", ""),
                    "had_json_envelope": '"status"' in c.get(
                        "output_preview", ""),
                }
    return {"is_error": None, "note": "no /etc/hosts call found (t3 not run "
            "or not attempted)"}


# ---------------------------------------------------------------------------
# The demo run: connect → list → convert → agent loop → grade
# ---------------------------------------------------------------------------
async def _run_demo(args) -> dict:
    started = time.perf_counter()
    results: dict = {
        "day": "week08_day5",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "chat_model": CHAT_MODEL,
            "chat_base": CHAT_BASE,
            "temperature": 0.0,
            "max_turns": MAX_TURNS,
            "third_party_server": {
                "type": "stdio",
                "command": THIRD_PARTY_COMMAND,
                "args": THIRD_PARTY_ARGS,
                "cwd": str(HERE),
            },
            "mcp_sdk_version": version("mcp"),
        },
        "session": {},
        "tasks": [],
    }

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

    # --- third-party server: one spawn, used for facts AND the task loop -----
    their_err = _ErrCapture()
    their_params = StdioServerParameters(
        command=THIRD_PARTY_COMMAND, args=THIRD_PARTY_ARGS, cwd=str(HERE)
    )
    t0 = time.perf_counter()
    async with stdio_client(their_params, errlog=their_err) as (read, write):
        async with ClientSession(read, write) as session:
            their = await _session_facts(session, their_err, t0)
            their_tools = mcp_tools_to_openai(their["tools_listed"])
            their["openai_schema_bytes"] = len(json.dumps(their_tools))
            results["session"]["third_party"] = their
            results["session"]["openai_tools"] = their_tools

            # --- our Day-2 server, same-day timing + schema comparison -------
            our = await _connect(
                StdioServerParameters(command=sys.executable,
                                      args=[str(OUR_SERVER_PATH)],
                                      cwd=str(PROJECT_ROOT)),
                _ErrCapture(),
            )
            results["session"]["our_server_for_comparison"] = our

            diffs = _schema_diff(their["tools_listed"])

            if not args.selftest:
                for task in [t for t in TASKS if t["id"] in args.tasks]:
                    t_task = time.perf_counter()
                    print(f"\n=== {task['id']} ===")
                    print(f"user: {task['prompt']}")
                    run = await run_agent_via_mcp(
                        chat_client, session, their_tools, task["prompt"]
                    )
                    run["wall_s"] = round(time.perf_counter() - t_task, 1)
                    note_path = SANDBOX / "thirdparty-agent-notes.md"
                    checks = _grade(task, run, note_path)
                    cell = {"id": task["id"], "prompt": task["prompt"],
                            "run": run, "checks": checks,
                            "passed": sum(c["passed"] for c in checks),
                            "total": len(checks)}
                    if (task["id"] == "t2_write_roundtrip"
                            and note_path.exists()):
                        cell["snapshot"] = {
                            "file": "sandbox/thirdparty-agent-notes.md",
                            "content": note_path.read_text(encoding="utf-8"),
                        }
                    results["tasks"].append(cell)
                    _print_task(cell)

    their_err.close_write_end()
    their_err.collect()  # final flush of any stderr written late

    if chat_client is not None:
        await chat_client.close()

    results["friction_points"] = _friction_points(
        their, our, diffs, results["tasks"]
    )
    results["summary"] = {
        "third_party_tools_listed": len(their["tools_listed"]),
        "openai_tools_converted": len(their_tools),
        "schema_diff_tools": [d["tool"] for d in diffs],
        "friction_points": len(results["friction_points"]),
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
        args_preview = {k: (v[:40] + "…" if isinstance(v, str)
                            and len(v) > 40 else v)
                        for k, v in args.items()}
        if "transport_error" in c:
            status = f"TRANSPORT ERROR: {c['transport_error']}"
        elif c.get("mcp_is_error"):
            status = f"denied — {c.get('output_preview', '')[:80]}"
        else:
            status = f"success: {c.get('output_preview', '')[:60]!r}"
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
    their = results["session"]["third_party"]
    our = results["session"]["our_server_for_comparison"]
    print("=" * 72)
    print("Day 5 — our agent, a third-party MCP server we didn't write")
    print("=" * 72)
    print(f"third-party server: {their['server_name']} "
          f"{their['server_version']} (protocol {their['protocol_version']})")
    print(f"spawn+init: theirs {their['spawn_init_ms']:.0f} ms vs "
          f"ours {our['spawn_init_ms']:.0f} ms; probe call: "
          f"{their['probe_call_ms']} ms")
    print(f"tools:  {len(their['tools_listed'])} third-party → "
          f"{len(results['session']['openai_tools'])} OpenAI schemas "
          f"({their['openai_schema_bytes']} bytes in every request)")
    print("\nfriction points:")
    for f in results["friction_points"]:
        print(f"  [{f['severity']}] {f['id']}")
        print(f"    what: {f['what']}")
    if not results["tasks"]:
        print("\n(selftest — no LLM tasks run)")
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
                    help="list tools + convert schemas + friction evidence, "
                         "no LLM calls")
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
    out = HERE / f"day5_results_{suffix}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), "utf-8")
    _print_report(results)
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
