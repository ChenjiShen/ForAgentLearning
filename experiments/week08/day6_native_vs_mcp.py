#!/usr/bin/env python3
"""Week 8 Day 6 EXPERIMENT: same tools called natively vs. via MCP.

PLAN spec: "same tools called natively vs. via MCP — latency and reliability
overhead."

The comparison is apples-to-apples by construction: both conditions run the
SAME week-3 tool implementations (``experiments/week03/agent.py``) against
the SAME sandbox.  The only difference is the access path:

- **native** — the agent/harness calls ``w3.read_file`` / ``w3.write_file``
  directly, in-process.  No serialization, no child process, no protocol.
- **via MCP** — the same two functions served by our Day-2 server
  (``mcp_server.py``) over stdio; each call is a JSON-RPC round trip through
  a child process.

The third-party filesystem server is deliberately NOT used here: it runs on
node, has 14 tools and a different contract, so it would confound transport
overhead with implementation differences (see Day 5's friction log).

Three measurement layers:

1. **Micro-benchmark** (``--rounds``, default 25): interleaved rounds of the
   same 8 operations in both conditions — read a small file, write a
   ~0.5 KB note, cross-read it from the other path (proves both paths share
   state), and one sandbox-escape denial.  Per-op latency, plus integrity
   checks (read-back must contain the round's unique marker) and envelope
   fidelity (native vs MCP denial envelopes must be identical JSON).
2. **Setup cost**: importing the week-3 module vs spawning the MCP server
   and initializing the session — the fixed cost each path pays once.
3. **Agent-level comparison** (``--agent-runs``, default 5 per condition,
   interleaved): the same graded task (write note, read back, quote
   hello.txt) through the same agent loop with the same schemas, the only
   difference being the tool dispatcher.  Wall time, tool-call latency,
   tokens and pass/fail per run.

Run:

    export $(grep DASHSCOPE_API_KEY .env | xargs)
    python experiments/week08/day6_native_vs_mcp.py --agent-runs 0   # micro-benchmark only, no API
    python experiments/week08/day6_native_vs_mcp.py                  # everything (needs DASHSCOPE_API_KEY)

Writes ``day6_results_<timestamp>.json`` next to this file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
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
WARMUP_ROUNDS = 3  # excluded from stats (interpreter + session warmup)

BENCH_FILE = "bench_native_vs_mcp.md"  # micro-benchmark scratch file


# ---------------------------------------------------------------------------
# The two backends: same functions, same sandbox, different access path
# ---------------------------------------------------------------------------
def _load_native():
    """Import the week-3 tool module exactly as mcp_server.py does, and
    confine it to week08/sandbox — the native condition's backend."""
    sys.path.insert(0, str(HERE.parent / "week03"))
    import agent as w3
    w3.SANDBOX_ROOT = SANDBOX
    return w3


def _envelope(text: str) -> dict:
    try:
        env = json.loads(text)
        if isinstance(env, dict) and "status" in env:
            return env
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def _envelope_success(text: str) -> bool:
    return _envelope(text).get("status") == "success"


class _ErrCapture:
    """Capture the server child's stderr through a real pipe fd
    (copied from day5_third_party.py).  Non-blocking on purpose: never
    wait for EOF — the child tree can outlive the session."""

    def __init__(self):
        self._r, self._w = os.pipe()
        os.set_blocking(self._r, False)
        self.lines: list[str] = []

    def fileno(self) -> int:
        return self._w

    def collect(self) -> None:
        while True:
            try:
                chunk = os.read(self._r, 65536)
            except BlockingIOError:
                return
            if not chunk:
                return
            self.lines.append(chunk.decode("utf-8", errors="replace"))

    def close_write_end(self) -> None:
        try:
            os.close(self._w)
        except OSError:
            pass

    def text(self) -> str:
        self.collect()
        return "".join(self.lines).strip()


# ---------------------------------------------------------------------------
# Micro-benchmark: interleaved rounds of identical ops in both conditions
# ---------------------------------------------------------------------------
def _native_call(w3, op: str, args: dict) -> tuple[str, bool]:
    """Call the week-3 function directly; return (envelope text, is_error)."""
    fn = {"read_file": w3.read_file, "write_file": w3.write_file}[op]
    text = fn(**args)
    return text, _envelope(text).get("status") == "error"


async def _native_coro(w3, op: str, args: dict) -> tuple[str, bool]:
    """Coroutine form of _native_call so both backends fit one measure()."""
    return _native_call(w3, op, args)


async def _mcp_call(session, op: str, args: dict) -> tuple[str, bool]:
    """Call the same function through the MCP session; return (text, is_error)."""
    result = await session.call_tool(op, args)
    parts = [getattr(b, "text", "") for b in result.content]
    return "\n".join(parts), result.is_error


async def _run_microbench(w3, session, rounds: int) -> dict:
    """Interleaved measurements: each round runs the same 8 operations in
    both conditions, alternating so time-of-day drift hits both equally."""
    ops = ["read_hello", "write_bench", "cross_read", "deny_escape"]
    payload = "x" * 512  # fixed-size write content (plus the round marker)

    records: list[dict] = []
    integrity = {"checks": 0, "failures": []}

    async def measure(cond: str, op: str, coro) -> dict:
        t0 = time.perf_counter()
        try:
            text, is_error = await coro
            env = _envelope(text)
            rec = {
                "condition": cond, "op": op,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                "is_error": is_error,
                "status": env.get("status"),
                "transport_error": None,
            }
            if env.get("error"):
                rec["error_type"] = env["error"].get("type")
            # note: error envelopes carry "data": null, so .get() with a
            # default does not help — the key exists with a None value
            if (env.get("data") or {}).get("content") is not None:
                # read ops: keep the content for the integrity checks
                rec["_content"] = env["data"]["content"]
            return rec
        except Exception as exc:  # transport-level failure — count it
            return {
                "condition": cond, "op": op,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                "is_error": None,
                "status": None,
                "transport_error": f"{type(exc).__name__}: {exc}",
            }

    for i in range(WARMUP_ROUNDS + rounds):
        measured = i >= WARMUP_ROUNDS
        native_marker = f"round-{i}-native"
        mcp_marker = f"round-{i}-mcp"

        # 1+2 — read the small existing file, both paths
        r = await measure("native", "read_hello",
                          _native_coro(w3, "read_file",
                                       {"filepath": "hello.txt"}))
        if measured:
            records.append(r)
        r = await measure("mcp", "read_hello",
                          _mcp_call(session, "read_file",
                                    {"filepath": "hello.txt"}))
        if measured:
            records.append(r)

        # 3+4 — native write, then MCP reads the same state (cross-path)
        r = await measure("native", "write_bench",
                          _native_coro(w3, "write_file",
                                       {"filepath": BENCH_FILE,
                                        "content": payload + native_marker}))
        if measured:
            records.append(r)
        r = await measure("mcp", "cross_read",
                          _mcp_call(session, "read_file",
                                    {"filepath": BENCH_FILE}))
        if measured:
            records.append(r)
            integrity["checks"] += 1
            if native_marker not in (r.get("_content") or ""):
                integrity["failures"].append(
                    f"round {i}: mcp read did not see native write")

        # 5+6 — MCP write, then native reads the same state (cross-path)
        r = await measure("mcp", "write_bench",
                          _mcp_call(session, "write_file",
                                    {"filepath": BENCH_FILE,
                                     "content": payload + mcp_marker}))
        if measured:
            records.append(r)
        r = await measure("native", "cross_read",
                          _native_coro(w3, "read_file",
                                       {"filepath": BENCH_FILE}))
        if measured:
            records.append(r)
            integrity["checks"] += 1
            if mcp_marker not in (r.get("_content") or ""):
                integrity["failures"].append(
                    f"round {i}: native read did not see mcp write")

        # 7+8 — the same denial, both paths
        r = await measure("native", "deny_escape",
                          _native_coro(w3, "read_file",
                                       {"filepath": "/etc/hosts"}))
        if measured:
            records.append(r)
        r = await measure("mcp", "deny_escape",
                          _mcp_call(session, "read_file",
                                    {"filepath": "/etc/hosts"}))
        if measured:
            records.append(r)

    return {"rounds_measured": rounds, "records": records,
            "integrity": integrity, "summary": _summarize(records)}


def _summarize(records: list[dict]) -> dict:
    """Per-(condition, op) latency stats + overhead; reliability counts."""
    stats, reliability = {}, {}
    for cond in ("native", "mcp"):
        for op in ("read_hello", "write_bench", "cross_read", "deny_escape"):
            lats = [r["latency_ms"] for r in records
                    if r["condition"] == cond and r["op"] == op
                    and r["transport_error"] is None]
            if not lats:
                continue
            stats[f"{cond}:{op}"] = {
                "n": len(lats),
                "mean_ms": round(statistics.mean(lats), 3),
                "median_ms": round(statistics.median(lats), 3),
                "p90_ms": round(_percentile(lats, 90), 3),
                "min_ms": round(min(lats), 3),
                "max_ms": round(max(lats), 3),
            }
        for fail_kind in ("transport", "denied", "envelope_mismatch"):
            reliability[f"{cond}:{fail_kind}"] = _count_kind(records, cond,
                                                             fail_kind)

    overhead = {}
    for op in ("read_hello", "write_bench", "cross_read", "deny_escape"):
        n = stats.get(f"native:{op}", {}).get("mean_ms")
        m = stats.get(f"mcp:{op}", {}).get("mean_ms")
        if n is not None and m is not None:
            overhead[op] = {
                "mcp_mean_ms": m, "native_mean_ms": n,
                "overhead_ms": round(m - n, 3),
                "overhead_x": round(m / n, 1) if n > 0 else None,
            }
    return {"stats": stats, "overhead": overhead,
            "reliability": reliability}


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _count_kind(records: list[dict], cond: str, kind: str) -> int:
    if kind == "transport":
        return sum(1 for r in records
                   if r["condition"] == cond and r["transport_error"])
    if kind == "denied":
        return sum(1 for r in records
                   if r["condition"] == cond and r["op"] == "deny_escape"
                   and r["is_error"] is True
                   and r.get("error_type") == "sandbox_escape")
    return sum(1 for r in records if r["condition"] == cond
               and r["status"] != "success" and r["status"] is not None
               and not (r["op"] == "deny_escape"))


# ---------------------------------------------------------------------------
# Agent-level comparison: same loop, same schemas, different dispatcher
# ---------------------------------------------------------------------------
def mcp_tools_to_openai(tools: list) -> list[dict]:
    """MCP tool listings → OpenAI tool schemas (from day4_mcp_agent.py)."""
    converted = []
    for t in tools:
        if isinstance(t, dict):
            name, desc = t.get("name"), t.get("description")
            params = t.get("input_schema") or {}
        else:
            name, desc = t.name, t.description
            params = t.input_schema if t.input_schema else {}
        params = dict(params)
        if params.get("type") != "object":
            params = {"type": "object",
                      "properties": params.get("properties", {})}
        converted.append(
            {"type": "function",
             "function": {"name": name, "description": desc or "",
                          "parameters": params}}
        )
    return converted


AGENT_SYSTEM = (
    "You are a helpful assistant with file tools. All file paths must be "
    "relative to the sandbox root (no leading '/' or '..'). "
    "IMPORTANT — All tool results use a consistent JSON envelope:\n"
    '  {"status":"success"|"error", "data":{…}, "error":null|{…}, '
    '"truncated":bool, "truncated_at":int|null}\n'
    "- On success: check .data for the tool-specific payload.\n"
    "- On error: read .error.type for the error category and .error.hint for "
    "an actionable suggestion on what to try next. ALWAYS read the hint "
    "before retrying — it tells you how to fix the problem."
)


async def run_agent(chat_client, tools, dispatch, user_prompt: str) -> dict:
    """Week-3 agent loop with a pluggable dispatcher: the ONLY difference
    between conditions.  dispatch(name, args) -> (text, is_error)."""
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
                trace.append({"name": tc.function.name,
                              "error": f"bad JSON: {e}"})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(
                                     {"status": "error", "data": None,
                                      "error": {"type": "invalid_input",
                                                "message": f"bad JSON: {e}",
                                                "hint": ""},
                                      "truncated": False,
                                      "truncated_at": None})})
                continue

            t0 = time.perf_counter()
            try:
                text, is_error = await dispatch(tc.function.name, args)
                rec = {"name": tc.function.name, "arguments": args,
                       "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                       "is_error": is_error}
            except Exception as exc:
                text = json.dumps(
                    {"status": "error", "data": None,
                     "error": {"type": "unknown",
                               "message": f"dispatcher error: {exc}",
                               "hint": ""},
                     "truncated": False, "truncated_at": None})
                rec = {"name": tc.function.name, "arguments": args,
                       "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                       "transport_error": f"{type(exc).__name__}: {exc}"}
            trace.append(rec)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": text})

    return {"answer": f"[stopped after {MAX_TURNS} turns]",
            "n_turns": MAX_TURNS, "tool_calls": trace, "usage": usage,
            "error": "max_turns_exceeded"}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower())


def _has_any(text: str, alts: list[str]) -> bool:
    n = _normalize(text)
    return any(_normalize(a) in n for a in alts)


def _grade_task(fname: str, run: dict) -> list[dict]:
    path = SANDBOX / fname
    checks = []
    checks.append({"check": f"{fname} exists", "passed": path.exists()})
    if path.exists():
        checks.append({"check": "note has the title",
                       "passed": _has_any(path.read_text(encoding="utf-8"),
                                          ["Bench Notes"])})
    else:
        checks.append({"check": "note has the title", "passed": False})
    checks.append({"check": "answer quotes hello.txt first line",
                   "passed": _has_any(run["answer"],
                                      ["Hello from the week08 sandbox"])})
    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def _run_experiment(args) -> dict:
    started = time.perf_counter()

    # --- setup cost measurement: native import vs MCP spawn+init ------------
    t0 = time.perf_counter()
    w3 = _load_native()
    native_import_ms = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    errlog = _ErrCapture()
    params = StdioServerParameters(command=sys.executable,
                                   args=[str(SERVER_PATH)],
                                   cwd=str(PROJECT_ROOT))
    async with stdio_client(params, errlog=errlog) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            mcp_spawn_init_ms = round((time.perf_counter() - t0) * 1000, 1)
            listing = await session.list_tools()

            # --- micro-benchmark --------------------------------------------
            bench = await _run_microbench(w3, session, args.rounds)

            # --- envelope fidelity: native vs MCP denial envelope -----------
            native_deny, _ = _native_call(w3, "read_file",
                                          {"filepath": "/etc/hosts"})
            mcp_deny, _ = await _mcp_call(session, "read_file",
                                          {"filepath": "/etc/hosts"})
            env_fidelity = {
                "native_envelope": _envelope(native_deny),
                "mcp_envelope": _envelope(mcp_deny),
                "identical": _envelope(native_deny) == _envelope(mcp_deny),
            }

            # --- agent-level comparison -------------------------------------
            tools = mcp_tools_to_openai(listing.tools)

            async def native_dispatch(name, args):
                return _native_call(w3, name, args)

            async def mcp_dispatch(name, args):
                return await _mcp_call(session, name, args)

            agent_runs = []
            if args.agent_runs > 0:
                chat_client = AsyncOpenAI(
                    api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url=CHAT_BASE,
                    http_client=httpx.AsyncClient(
                        trust_env=False,
                        timeout=httpx.Timeout(600.0, connect=15.0)),
                )
                try:
                    for i in range(args.agent_runs):
                        # interleave conditions so API latency drift hits both
                        for cond, dispatch in (("native", native_dispatch),
                                               ("mcp", mcp_dispatch)):
                            fname = f"bench-agent-{cond}-{i}.md"
                            prompt = (
                                f"Write a short markdown note to '{fname}' "
                                "in the sandbox. The note must have the "
                                "title 'Bench Notes' and one bullet about "
                                "which tool backend served you: write "
                                "'unknown backend'. Then read the file "
                                "back to verify it was written, and also "
                                "read 'hello.txt' and report exactly what "
                                "its first line says."
                            )
                            t_task = time.perf_counter()
                            run = await run_agent(chat_client, tools,
                                                  dispatch, prompt)
                            run["wall_s"] = round(time.perf_counter() - t_task, 1)
                            checks = _grade_task(fname, run)
                            agent_runs.append({
                                "condition": cond, "run_index": i,
                                "fname": fname, "run": run, "checks": checks,
                                "passed": sum(c["passed"] for c in checks),
                                "total": len(checks),
                            })
                            print(f"[agent {cond} #{i}] "
                                  f"{sum(c['passed'] for c in checks)}/"
                                  f"{len(checks)} checks, "
                                  f"{len(run['tool_calls'])} calls, "
                                  f"{run['wall_s']} s wall")
                finally:
                    await chat_client.close()

    errlog.close_write_end()

    results = {
        "day": "week08_day6",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "chat_model": CHAT_MODEL,
            "rounds": args.rounds,
            "warmup_rounds": WARMUP_ROUNDS,
            "agent_runs_per_condition": args.agent_runs,
            "server": {"name": init.server_info.name,
                       "version": init.server_info.version,
                       "protocol": init.protocol_version},
            "mcp_sdk_version": version("mcp"),
            "sandbox": str(SANDBOX),
        },
        "setup_cost": {
            "native_import_ms": native_import_ms,
            "mcp_spawn_init_ms": mcp_spawn_init_ms,
            "mcp_over_native_ms": round(mcp_spawn_init_ms - native_import_ms, 1),
        },
        "microbench": bench,
        "envelope_fidelity": env_fidelity,
        "agent": _summarize_agent(agent_runs),
        "total_s": round(time.perf_counter() - started, 1),
    }
    return results


def _summarize_agent(runs: list) -> dict:
    if not runs:
        return {"runs": [], "note": "agent part skipped (--agent-runs 0)"}
    summary = {}
    for cond in ("native", "mcp"):
        cells = [r for r in runs if r["condition"] == cond]
        lats = [c["latency_ms"] for r in cells
                for c in r["run"]["tool_calls"]
                if "transport_error" not in c]
        walls = [r["run"]["wall_s"] for r in cells]
        summary[cond] = {
            "runs": len(cells),
            "checks_passed": sum(r["passed"] for r in cells),
            "checks_total": sum(r["total"] for r in cells),
            "wall_mean_s": round(statistics.mean(walls), 2),
            "wall_min_s": round(min(walls), 2),
            "wall_max_s": round(max(walls), 2),
            "tool_calls_mean": round(statistics.mean(
                [len(r["run"]["tool_calls"]) for r in cells]), 1),
            "tool_call_latency_mean_ms": (
                round(statistics.mean(lats), 3) if lats else None),
            "tool_call_latency_max_ms": (
                round(max(lats), 3) if lats else None),
            "tokens_mean": round(statistics.mean(
                [r["run"]["usage"]["prompt"] + r["run"]["usage"]["completion"]
                 for r in cells])),
            "transport_errors": sum(
                1 for r in cells
                for c in r["run"]["tool_calls"] if "transport_error" in c),
        }
    return {"runs": runs, "summary": summary}


def _print_report(results: dict) -> None:
    print("=" * 72)
    print("Day 6 — native vs MCP: latency and reliability overhead")
    print("=" * 72)
    cfg = results["config"]
    print(f"server:  {cfg['server']['name']} {cfg['server']['version']} "
          f"(protocol {cfg['server']['protocol']})")
    print(f"setup:   native import {results['setup_cost']['native_import_ms']} ms "
          f"vs MCP spawn+init {results['setup_cost']['mcp_spawn_init_ms']} ms "
          f"(+{results['setup_cost']['mcp_over_native_ms']} ms)")
    print(f"rounds:  {results['microbench']['rounds_measured']} measured "
          f"(+{cfg['warmup_rounds']} warmup)")
    print("\nmicro-benchmark (mean latency):")
    stats = results["microbench"]["summary"]["stats"]
    overhead = results["microbench"]["summary"]["overhead"]
    for op in ("read_hello", "write_bench", "cross_read", "deny_escape"):
        n = stats[f"native:{op}"]["mean_ms"]
        m = stats[f"mcp:{op}"]["mean_ms"]
        o = overhead[op]
        print(f"  {op:<12} native {n:>7.3f} ms | mcp {m:>7.3f} ms | "
              f"overhead {o['overhead_ms']:>+7.3f} ms ({o['overhead_x']}x)")
    rel = results["microbench"]["summary"]["reliability"]
    print(f"reliability: {json.dumps(rel)}")
    integ = results["microbench"]["integrity"]
    print(f"integrity: {integ['checks'] - len(integ['failures'])}/"
          f"{integ['checks']} cross-path reads saw the other path's write")
    for f in integ["failures"]:
        print(f"  ❌ {f}")
    ef = results["envelope_fidelity"]
    print(f"denial envelope fidelity: native == MCP ? "
          f"{'YES ✅' if ef['identical'] else 'NO ❌'}")
    if results["agent"]["runs"]:
        s = results["agent"]["summary"]
        print("\nagent-level (5 runs per condition, interleaved):")
        for cond in ("native", "mcp"):
            c = s[cond]
            print(f"  {cond:<7} {c['checks_passed']}/{c['checks_total']} checks, "
                  f"wall {c['wall_mean_s']} s mean, "
                  f"tool-call latency {c['tool_call_latency_mean_ms']} ms mean "
                  f"(max {c['tool_call_latency_max_ms']} ms), "
                  f"{c['transport_errors']} transport errors")
    print(f"\ntotal:  {results['total_s']} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=25,
                    help="measured micro-benchmark rounds per op per "
                         "condition (default 25)")
    ap.add_argument("--agent-runs", type=int, default=5,
                    help="agent task runs per condition (default 5; 0 skips "
                         "the LLM part entirely — no API key needed)")
    args = ap.parse_args()

    if args.agent_runs > 0 and "DASHSCOPE_API_KEY" not in os.environ:
        sys.exit(
            "DASHSCOPE_API_KEY not set — run:\n"
            "  export $(grep DASHSCOPE_API_KEY "
            f"{PROJECT_ROOT / '.env'} | xargs)"
        )

    # clean stale bench files from previous runs (only our own bench files)
    for stale in SANDBOX.glob("bench-*.md"):
        stale.unlink()

    results = asyncio.run(_run_experiment(args))
    out = HERE / f"day6_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), "utf-8")
    _print_report(results)
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
