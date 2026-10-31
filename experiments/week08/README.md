# Week 8 — MCP (Model Context Protocol)

## Day 1 — Install an existing MCP server and poke at it with the Inspector

**PLAN spec:** Install an existing MCP server (e.g. the filesystem server) and
poke at it with the MCP inspector tool.

### What was installed / run

- **MCP server:** `@modelcontextprotocol/server-filesystem` (latest, via `npx`).
  Launched by the Inspector, it identifies itself as
  `secure-filesystem-server 0.2.0`, speaking protocol `2025-06-18` over **stdio**.
- **MCP Inspector:** `@modelcontextprotocol/inspector` v2.4.0, web UI at
  http://127.0.0.1:6274, launched with:

  ```bash
  npx -y @modelcontextprotocol/inspector \
       npx -y @modelcontextprotocol/server-filesystem ./sandbox
  ```

  Only `./sandbox` is an allowed directory for the server (path-safety test below).

### How to poke at it yourself (interactive UI)

The Inspector is still running in the background. Open the URL it printed
(contains the API token) in a browser, click **Connect**, and use the
**Tools / Resources / Prompts** tabs:

```
http://127.0.0.1:6274?MCP_INSPECTOR_API_TOKEN=<token from logs/inspector.log>
```

Stop it when done: `pkill -f mcp-inspector`.

### Headless poking — `poke_inspector.py`

Because the UI is interactive, the experiment was also driven headlessly
through the Inspector's own REST proxy API (the same API its web UI uses):

| Endpoint | Purpose |
|---|---|
| `GET  /api/config` | preconfigured stdio server (command, args, cwd, env) |
| `POST /api/mcp/connect` | create a session — spawns the MCP server process |
| `GET  /api/mcp/events?sessionId=` | SSE stream of JSON-RPC messages |
| `POST /api/mcp/send` | send a JSON-RPC message on the session |

Auth: `x-mcp-remote-auth: Bearer <token>` header on every call.

`python poke_inspector.py` performs the full MCP handshake (`initialize` →
`notifications/initialized` → `tools/list` → `tools/call` × 6 → `resources/*` →
`prompts/list`) and writes a timestamped `day1_results_*.json`.

### Findings

1. **The server exposes 14 tools** (write-capable ones included!):
   `read_file, read_text_file, read_media_file, read_multiple_files,
   write_file, edit_file, create_directory, list_directory,
   list_directory_with_sizes, directory_tree, move_file, search_files,
   get_file_info, list_allowed_directories`.

2. **Tool calls work over stdio** — `list_directory` and `read_file` returned
   the sandbox contents correctly; `get_file_info` returned size + timestamps.

3. **Path safety is enforced server-side** (this is the point of the allowed-dir
   config). Both attempts to escape the sandbox were denied with `isError:
   true` and text `Access denied - path outside allowed directories`:
   - absolute path outside sandbox: `/etc/hosts`
   - relative traversal: `../PLAN.md`

4. **MCP capabilities are optional per server.** `resources/list`,
   `resources/templates/list`, `resources/read` and `prompts/list` all return
   `-32601 Method not found` — this filesystem server implements **tools only**.
   (Older versions exposed file:// resources; the current one dropped them.)

5. **Transport & protocol details observed:**
   - stdio transport: the Inspector spawns the server as a child process and
     proxies JSON-RPC to the UI/API over HTTP + SSE.
   - Protocol version negotiation is explicit: we sent `2025-06-18` and the
     server agreed.
   - Tool-call results are `content` blocks (text), not plain return values.

6. **Local gotcha:** Python's `urllib` on macOS honors the *system* proxy
   config (127.0.0.1:7890, which is dead on this machine) even when no proxy
   env vars are set, so requests to the local Inspector fail with 502.
   Fix: `urllib.request.build_opener(urllib.request.ProxyHandler({}))`.

### Files

| File | What it is |
|---|---|
| `poke_inspector.py` | headless poke script driving the Inspector REST API |
| `day1_results_*.json` | full results: initialize response, tool schemas, every poke + outcome, SSE transcript summary |
| `sandbox/` | the server's allowed directory (sample files) |
| `logs/inspector.log` | Inspector startup log (contains the API token) |

---

## Day 2 — Build our own MCP server (stdio) and test it with the Inspector

**PLAN spec:** Build your own MCP server exposing 2 of your week-3 tools
(stdio transport). Test with the inspector.

### What was built

`mcp_server.py` — a stdio MCP server built on the official MCP Python SDK
(`mcp` 2.1.1; in 2.x FastMCP was renamed `MCPServer`). It exposes **two
week-3 tools**: `read_file` and `write_file` (the week-3 Day-1 file pair —
picked so a write→read round trip proves state actually flows through MCP).

Key design points:

1. **The week-3 implementations are imported, not copied.**
   `experiments/week03/agent.py` supplies the tool functions and the sandbox
   resolver; the server just points the week-3 module's `SANDBOX_ROOT` at
   `week08/sandbox` and wraps the two functions as MCP tools. One security
   boundary, shared.
2. **The week-3 result contract survives intact.** Tools still return the
   same JSON envelope (`status` / `data` / `error` / `truncated`), and error
   envelopes are flagged `is_error: true` so MCP clients can tell success
   from denial apart — exactly the error model MCP wants (the tool *ran*,
   the request was denied).
3. **stdio transport** — `mcp.run(transport="stdio")` speaks JSON-RPC over
   stdin/stdout; the Inspector (or any host) spawns the file as a child
   process. No port, no auth, the host owns the process lifecycle.

Run it directly (the Inspector launches it the same way):

```bash
python experiments/week08/mcp_server.py
```

### Testing with the Inspector

```bash
npx -y @modelcontextprotocol/inspector python experiments/week08/mcp_server.py
```

The Inspector web UI is at http://127.0.0.1:6274?MCP_INSPECTOR_API_TOKEN=<token
from `logs/inspector_day2.log`>. Like Day 1, the test was also run headlessly
through the Inspector's REST proxy API: `python poke_day2_server.py`
(reuses Day 1's `InspectorClient`), writing `day2_results_*.json`.

### Findings

1. **Handshake + listing work.** `initialize` negotiated protocol
   `2025-06-18`, server identifies as `week08-sandbox-files 0.1.0`, and
   `tools/list` returns exactly the 2 tools with full JSON Schemas
   (descriptions + parameter descriptions, from the docstrings/Field args).

2. **Round trip through MCP.** `write_file mcp-via-inspector.md` → 
   `read_file` returned the exact content written; a second `write_file`
   (overwrite) → `read_file` showed the new content. State flows through
   the protocol, not just the reply.

3. **Sandbox safety is enforced through MCP.** All four denial attempts
   came back `isError: true` with the week-3 classified envelope + hint:
   - `/etc/hosts` → `sandbox_escape` (absolute path)
   - `../PLAN.md` → `sandbox_escape` (traversal)
   - `../evil.txt` (write) → `sandbox_escape`
   - `binary.bin` → `extension_not_allowed`

   The same protections tested in Day 1 against the reference filesystem
   server now hold for our own server.

4. **Capabilities differ from Day 1's server.** Our SDK server advertises
   `prompts` and `resources` capabilities (both return empty lists —
   `prompts/list` and `resources/list` answer instead of `-32601`), whereas
   the reference filesystem server is tools-only.

5. **SDK gotcha:** mcp 2.x renamed FastMCP to `MCPServer`
   (`from mcp.server.mcpserver import MCPServer`); tool functions can return
   `mcp_types.CallToolResult` directly to control `is_error` per call.

### Files

| File | What it is |
|---|---|
| `mcp_server.py` | our MCP server (read_file + write_file over stdio) |
| `poke_day2_server.py` | headless Inspector test for our server |
| `day2_results_*.json` | initialize response, tool schemas, all 13 pokes, SSE transcript |
| `sandbox/mcp-via-inspector.md` | file written through MCP during the test (round-trip artifact) |
| `logs/inspector_day2.log` | Inspector startup log for Day 2 (contains the API token) |

---

## Day 3 — MCP client: session lifecycle, list_tools, call_tool (no LLM)

**PLAN spec:** Write an MCP client that connects to your server, lists tools,
and calls one — no LLM yet, just the protocol.

### What was built

`mcp_client.py` — the client half of the MCP relationship, built on the
official SDK's client side (`mcp.client.stdio.stdio_client` + `ClientSession`),
the mirror image of Day 2's server side. It does exactly what a host does
*before* an LLM ever gets involved:

1. **Spawn** the Day-2 server as a child process (stdio — the client owns the
   server's lifecycle; no port, no auth, no HTTP).
2. **Initialize** the session (`initialize`) and **list tools**
   (`tools/list` → the 2 JSON Schemas).
3. **Call tools** (`tools/call`) × 6: a write→read→overwrite round trip, a
   read of a pre-existing file, and the two denial cases.

Run it:

```bash
python experiments/week08/mcp_client.py
```

Writes `day3_results_<timestamp>.json`.

### Findings

1. **The client is generic — discovery works.** It learns the tool set from
   `list_tools` (never hardcoded); the call list is just data. Swap in any
   other stdio server and the discovery part still works. Tool selection is a
   hardcoded list *on purpose* — Day 4 replaces it with the agent's tool-call
   loop.
2. **Round trip through MCP with the SDK client.** `write_file` →
   `read_file` returned the exact content; a second `write_file` (overwrite)
   → `read_file` showed the new content. State flows through the protocol,
   not just the reply — same proof as Day 2, now from a first-party client.
3. **Errors cross the protocol as data.** Both denial attempts came back
   `is_error: true` with the week-3 classified envelope + hint
   (`/etc/hosts` → `sandbox_escape`, `binary.bin` → `extension_not_allowed`).
   The client sees a *successful* protocol round trip whose payload says no —
   the exact error model Day 4's agent needs to tell "tool ran, request
   denied" from "tool crashed".
4. **Protocol version is negotiated per client.** This client negotiated
   `2025-11-25` (the SDK's current default) while Day 2's Inspector session
   negotiated `2025-06-18`; the server accepted both. Version agreement is a
   property of each client–server pair, not of the server.
5. **stdio avoids the Day-1 proxy gotcha entirely.** Unlike the Inspector
   REST API, a direct stdio client makes no HTTP calls, so the dead macOS
   system proxy (127.0.0.1:7890) cannot break it. No ProxyHandler fix needed.
6. **Latency numbers (seed for Day 6):** per-call latency 0.7–4.3 ms; the
   736 ms end-to-end figure is dominated by spawning the server process +
   the client interpreter. The per-call numbers are what Day 6 compares
   against native (in-process) tool calls.

### Files

| File | What it is |
|---|---|
| `mcp_client.py` | direct MCP SDK client: connect, initialize, list_tools, call_tool × 6 |
| `day3_results_*.json` | initialize response, tool schemas, all 6 calls + latency + envelopes, summary |
| `sandbox/mcp-via-client.md` | file written through MCP by the Day-3 client (round-trip artifact) |

---

## Day 4 — Connect our agent to our MCP server: the MCP→LLM bridge

**PLAN spec:** Connect your agent to your MCP server: convert MCP tool listings
into OpenAI tool schemas, route tool calls through the MCP session. Your agent
now uses MCP tools transparently.

### What was built

`day4_mcp_agent.py` — Day 3's hardcoded call list replaced by the agent's
tool-call loop. The bridge has two halves:

1. **Listing → schemas** (`mcp_tools_to_openai`): the `tools/list` response
   becomes the OpenAI tool-schema list the agent loop already speaks. The
   parameter schema is a passthrough — MCP and OpenAI tool calling both use
   JSON Schema — so the conversion is the envelope
   (`{name, description, inputSchema}` → `{"type":"function",
   "function":{…}}`) plus defensive normalization (missing schemas, missing
   `type: object`) so any stdio server survives it.
2. **Tool calls → MCP session**: the week-3/week-7 loop is reused unchanged
   except the dispatcher — every model tool call becomes
   `await session.call_tool(name, args)` over stdio, and the returned text is
   fed back as the `tool` message, the same shape the loop used for native
   tools. The agent never imports week-3 code; it only sees the schemas the
   server advertised.

Two graded demo tasks prove it (filesystem = ground truth, week-4 style):

- `t1_roundtrip` — write a note via MCP, read it back via MCP, quote
  `hello.txt` (a file that only exists in the *server's* sandbox).
- `t2_escape_denied` — read `/etc/hosts`; the denial must cross the protocol
  as `is_error=True` + the classified envelope, and the agent must report the
  error type/message/hint and then demonstrate adaptation.

Run it (selftest needs no API key):

```bash
python experiments/week08/day4_mcp_agent.py --selftest   # offline listing + conversion
python experiments/week08/day4_mcp_agent.py               # full run (needs DASHSCOPE_API_KEY)
python experiments/week08/day4_mcp_agent.py --tasks t2_escape_denied
```

### Findings

1. **Transparency is behavioral, not cosmetic.** The agent received the same
   tool names, descriptions and result envelopes it learned in week 3, served
   from a child process it never saw. `t1` needed zero new instructions: the
   agent wrote `mcp-agent-notes.md`, verified it by reading it back, and
   quoted `hello.txt` — 5/5 checks, three tool calls, all through MCP.
2. **The MCP error model works for the agent — Day 3's requirement met.**
   `t2`'s `/etc/hosts` call returned `is_error=True` with the classified
   envelope; the agent treated it as "tool ran, request denied" (not a
   crash), and its follow-up `read_file('hello.txt')` showed it applied the
   hint. The trace check confirms the envelope crossed the wire intact.
3. **Finding: adaptation ≠ reporting.** First `t2` run: the mechanism checks
   passed but the final answer glossed over the denial ("The file was read
   successfully…") instead of reporting type/message/hint as instructed —
   3/5. Tightening the prompt to demand three labelled lines produced a
   verbatim report — 5/5. The bridge was never the problem; the prompt was.
   (Both runs are kept in the results files — the audit trail.)
4. **SDK naming gotcha:** the wire protocol field is `inputSchema` but the
   SDK's Python objects expose `input_schema` — worth remembering for any
   code that walks `tools/list` payloads directly.
5. **Latency seeds for Day 6:** MCP `call_tool` latencies inside the agent
   loop were 1.4–6.3 ms per call (Day 3 measured 0.7–4.3 ms from a bare
   client). Task wall time (8–12 s) is dominated by LLM turns, not MCP —
   the exact comparison Day 6 formalizes against native calls.

### Files

| File | What it is |
|---|---|
| `day4_mcp_agent.py` | the bridge: MCP listings → OpenAI schemas, MCP-routed agent loop, 2 graded tasks |
| `day4_results_selftest.json` | offline selftest: listing + converted schemas, no API |
| `day4_results_20260908_214939.json` | first full run: t1 5/5, t2 3/5 (under-reporting finding) |
| `day4_results_20260908_215055.json` | t2 re-run with tightened prompt: 5/5 |
| `day4_run.log` | stdout log of both runs |
| `sandbox/mcp-agent-notes.md` | note written by the agent through MCP (round-trip artifact) |

---

## Day 5 — Connect our agent to a third-party MCP server we didn't write

**PLAN spec:** Connect your agent to a *third-party* MCP server you didn't
write. Note every friction point.

### What was built

`day5_third_party.py` — the Day-4 agent, same bridge
(`mcp_tools_to_openai` + the MCP-routed agent loop), pointed at
**`@modelcontextprotocol/server-filesystem`** (the official reference server
installed on Day 1, spawned via `npx`). We did not write it; the client only
knows what `tools/list` tells it. Three graded tasks:

- `t1_discover_and_read` — list the sandbox, read `hello.txt`, quote it.
- `t2_write_roundtrip` — write `thirdparty-agent-notes.md` through the
  third-party server, read it back, quote the title.
- `t3_escape_denied` — read `/etc/hosts`; the denial arrives as `isError`
  + plain text (no classified envelope), and the agent must report it and
  adapt.

The deliverable is not the tasks — it's the **friction log**: the results
JSON carries a `friction_points` list where every point has severity +
live evidence + workaround.

Run it:

```bash
python experiments/week08/day5_third_party.py --selftest   # offline: listing + friction evidence
python experiments/week08/day5_third_party.py               # full run (needs DASHSCOPE_API_KEY)
```

### Findings — the friction points (severity-ranked)

1. **[blocker] `prompt_contract_coupling`.** Day 4's system prompt taught
   the agent OUR result contract (the week-3 JSON envelope, "read
   error.hint"). The third-party server returns plain text with bare denial
   strings. A prompt tuned to one server silently misinforms the agent
   about another. The prompt had to be rewritten for the new server; treat
   the system prompt as server-specific config (or keep it contract-agnostic).
2. **[warning] `same_names_different_schemas`.** Same tool names, different
   parameter names: our `read_file`/`write_file` take `filepath`, theirs
   take `path`. Hardcoded args or wrapper code written for one server break
   against the other. (The agent loop itself was fine — it fills parameters
   from the advertised schemas, which is exactly the MCP promise working.)
3. **[warning] `error_contract_differs`.** Denials arrive as `isError=true`
   + a bare string (`Access denied - path outside allowed directories: …`),
   with no error type or hint. Evidence recorded on the wire:
   `had_json_envelope: false`. Any client that parses the week-3 envelope
   gets nothing. `isError` is the only cross-server error signal.
4. **[warning] `tool_volume_schema_bloat`.** 14 tools vs our 2 — 8,747 bytes
   of tool schemas ride in *every* LLM request, and the wider surface means
   near-miss choices (`read_file` vs `read_text_file`). Filter the listing
   before conversion if the agent doesn't need everything.
5. **[warning] `absolute_path_convention`.** This server's tool output,
   error messages and `list_allowed_directories` all use absolute paths,
   and it accepts them — unlike our relative-only server. The agent
   abandoned relative paths even though the prompt told it to use them
   (environment feedback overrode the instruction), then *hallucinated* the
   long path (dropped `MyPycharm`), got 2 denials, re-read the allowed
   directory and self-corrected. Self-healing worked; the round trip cost
   two extra tool calls.
6. **[warning] `orphan_child_tree_after_session_close`.** Via npx the
   server is an npm → node grandchild chain; the SDK kills the direct
   child but the grandchild can survive as an orphan holding its stderr
   pipe. Our first stderr capture blocked on read-until-EOF and hung
   forever. Fix: read stderr non-blockingly on demand (the script does).
7. **[annoyance] `spawn_and_call_latency`.** npx spawn+initialize: 1.4–2.0 s
   vs ~0.5 s for our python server (same-day, same client); probe call
   ~21 ms vs ~2–5 ms. Day 6's native-vs-MCP experiment formalizes this.
8. **[annoyance] `roots_capability_not_sent`.** The Python SDK client
   doesn't declare the MCP `roots` capability, so the server falls back to
   CLI args for allowed directories and says so on stderr. A full host
   (Claude Desktop) negotiates roots dynamically; a minimal client must
   pass them as CLI args.
9. **[annoyance] `stderr_log_leakage`.** stdio has one protocol channel
   (stdout); server logs go to stderr and leak into the host's terminal
   unless the client captures them (we did, via a custom `errlog` pipe).
10. **[note] `protocol_version_drift`.** Day 1's Inspector negotiated
    `2025-06-18`; today's SDK client negotiated `2025-11-25`; the server
    accepted both. Version agreement is per connection, not per server.

### Outcome

**11/11 checks passed** across the 3 tasks (4 turns, 3–5 tool calls and
~8 s wall per task — LLM turns dominate, as on Day 4). The agent used MCP
tools it had never seen from a server we didn't write, recovered from its
own path hallucination, and reported the third-party error contract
correctly.

### Files

| File | What it is |
|---|---|
| `day5_third_party.py` | agent + bridge against the reference filesystem server; builds the friction log from live evidence |
| `day5_results_selftest.json` | offline: listing, schema conversion, stderr capture, both servers' timing |
| `day5_results_20260909_210741.json` | first full run — 9/11; the 2 failures were overly strict path checks (agent used absolute allowed paths), corrected afterwards |
| `day5_results_20260909_211122.json` | final full run — 11/11, complete friction log with evidence |
| `sandbox/thirdparty-agent-notes.md` | note written by the agent through the third-party server (round-trip artifact) |

---

## Day 6 — EXPERIMENT: same tools called natively vs. via MCP

**PLAN spec:** same tools called natively vs. via MCP — latency and
reliability overhead.

### What was built

`day6_native_vs_mcp.py` — an apples-to-apples comparison: both conditions
run the SAME week-3 tool functions against the SAME sandbox. The only
difference is the access path:

- **native** — `w3.read_file` / `w3.write_file` called directly, in-process.
- **via MCP** — the same two functions served by our Day-2 server over
  stdio (JSON-RPC round trip through a child process).

The third-party filesystem server is deliberately NOT used: node, 14 tools
and a different contract would confound transport overhead with
implementation differences (Day 5's log).

Three measurement layers:

1. **Micro-benchmark** — 25 interleaved rounds of 8 operations each (read
   small file, write ~0.5 KB note, cross-read the other path's write,
   sandbox-escape denial), alternating native/MCP so drift hits both
   equally. 3 warmup rounds excluded.
2. **Setup cost** — importing the week-3 module vs spawning the MCP server
   + initializing the session.
3. **Agent-level** — 5 runs per condition, interleaved, of the same graded
   task (write note → read back → quote hello.txt) through the same agent
   loop with the same schemas; only the tool dispatcher differs.

Run it:

```bash
python experiments/week08/day6_native_vs_mcp.py --agent-runs 0   # micro-benchmark only, no API
python experiments/week08/day6_native_vs_mcp.py                  # everything (needs DASHSCOPE_API_KEY)
```

### Findings

1. **The overhead is a constant ~0.53 ms per call, independent of the
   operation** (read 7.6x, write 5.5x, cross-read 7.2x). It is the
   JSON-RPC encode/decode + stdio round trip, not the file work — the
   escape denial costs the same ~0.54 ms even though it does no I/O at
   all. As a ratio the denial is 79.7x because the native denial is
   nearly free (it fails at path resolution, before any file access).
2. **Setup cost is the larger fixed price**: native import ~60 ms vs MCP
   spawn+initialize ~465 ms (+~400 ms) — but it is paid once per process,
   and Day 3 already showed the whole spawn is ~0.7 s end-to-end. For a
   long-lived agent session it amortizes to nothing; for one-shot CLI
   invocations it dominates every per-call saving.
3. **Reliability is identical: zero transport errors in both conditions**
   (25/25 reads, writes, cross-reads and denials each); the denial
   envelopes are byte-identical JSON between conditions; all 50 cross-path
   reads saw the other path's write (same sandbox state, two access
   paths).
4. **Under agent load, both paths slow down ~6-8x** (event-loop contention
   with the live LLM session): native per-call latency rose from ~0.08 ms
   to ~0.68 ms mean, MCP from ~0.62 ms to ~4.15 ms mean. The MCP *delta*
   stays ~3.5 ms per call.
5. **At task level, the overhead disappears into LLM noise.** Wall time
   native 6.94 s vs MCP 6.16 s mean (ranges 5.8–8.3 s vs 5.1–7.4 s) —
   statistically indistinguishable; the ~10 ms of MCP overhead per task
   (3–5 calls × ~3.5 ms) is 0.2% of task time. Both conditions scored
   15/15 checks; token usage is near-identical.
6. **Verdict (the Day-5 question, answered):** MCP's indirection costs
   ~0.5 ms/call standalone, ~3.5 ms/call under agent load, and ~0.5 s to
   set up — trivial for anything that shares tools across agents/apps,
   and pure overhead for a single process calling its own functions.
   Pay it when tools must cross process boundaries or be shared; skip it
   in hot loops inside one process.

### Files

| File | What it is |
|---|---|
| `day6_native_vs_mcp.py` | the comparison harness: interleaved micro-benchmark, setup cost, envelope fidelity, agent-level runs |
| `day6_results_20260909_213613.json` | first run (audit trail) — a `data: null` parsing bug in the measure loop corrupted the deny records; fixed afterwards |
| `day6_results_20260909_213703.json` | micro-benchmark only (no LLM), corrected |
| `day6_results_20260909_213820.json` | final full run: micro + agent-level, all numbers above |
| `sandbox/bench-agent-{native,mcp}-{0..4}.md` | the 10 notes written by the agent during the comparison |


