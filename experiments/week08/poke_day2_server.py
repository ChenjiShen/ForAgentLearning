#!/usr/bin/env python3
"""Week08 Day2: poke OUR MCP server through the MCP Inspector.

Same technique as Day 1 (drive the Inspector's REST/SSE proxy API), but the
stdio server behind the Inspector is now our own `mcp_server.py` exposing
two week-3 tools: read_file and write_file.

    python poke_day2_server.py

Pokes performed (all through the Inspector, i.e. real stdio round trips):

1. initialize — handshake; record serverInfo + protocol version
2. tools/list — must be exactly [read_file, write_file] with schemas
3. read_file hello.txt                 -> ok
4. read_file notes/mcp-notes.md        -> ok (nested path)
5. write_file mcp-via-inspector.md     -> ok (new file)
6. read_file mcp-via-inspector.md      -> ok (round trip proves state flowed
                                            through MCP, not just the reply)
7. write_file mcp-via-inspector.md     -> ok (overwrite), then re-read to
                                            verify content changed
8. read_file /etc/hosts                -> isError (absolute path escape)
9. read_file ../PLAN.md                -> isError (traversal escape)
10. write_file ../evil.txt             -> isError (write escape)
11. write_file binary.bin              -> isError (extension not allowed)
12. prompts/list / resources/list      -> whatever the server advertises

Writes day2_results_<timestamp>.json next to this script.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from poke_inspector import (
    InspectorClient,
    InspectorError,
    NO_PROXY_OPENER,
    PROTOCOL_VERSION,
    call_tool,
    get_config,
)

HERE = Path(__file__).parent
URL_RE = re.compile(
    r"(http://127\.0\.0\.1:\d+)\?MCP_INSPECTOR_API_TOKEN=([a-f0-9]{16,})"
)


def read_token_and_port(log_path: Path) -> tuple[str | None, str | None]:
    """Parse the Inspector startup log for the API token and base URL."""
    text = log_path.read_text(errors="replace")
    m = URL_RE.search(text)
    if m:
        return m.group(1), m.group(2)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=None,
                    help="Inspector base URL (default: parsed from the log)")
    ap.add_argument("--token-file", default=str(HERE / "logs" / "inspector_day2.log"))
    args = ap.parse_args()

    log_path = Path(args.token_file)
    if not log_path.exists():
        print(f"error: {log_path} not found", file=sys.stderr)
        sys.exit(1)
    log_base, log_token = read_token_and_port(log_path)
    base_url = args.base_url or log_base
    token = log_token
    if not base_url:
        print("error: could not parse base URL from inspector log", file=sys.stderr)
        sys.exit(1)
    if not token:
        print("error: no MCP_INSPECTOR_API_TOKEN in log", file=sys.stderr)
        sys.exit(1)

    inspector = InspectorClient(base_url, token)

    print("== fetch /api/config ==")
    cfg = get_config(inspector)
    print(f"inspector stdio server: {cfg['command']} {' '.join(cfg['args'])}")

    print("== connect (spawns our mcp_server.py) ==")
    session_id = inspector.connect(cfg)
    print(f"sessionId: {session_id}")

    print("== initialize ==")
    init = inspector.send("initialize", {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "week08-day2-poke-cli", "version": "0.0.1"},
    })
    server_info = init["result"]["serverInfo"]
    print(f"server: {server_info['name']} {server_info['version']} "
          f"(protocol {init['result'].get('protocolVersion')})")
    inspector.notify("notifications/initialized")

    report = {
        "day": "week08_day2",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inspector_base_url": base_url,
        "server_config": cfg,
        "session_id": session_id,
        "initialize": init,
        "pokes": [],
    }

    def poke(label, fn):
        try:
            report["pokes"].append({"label": label, **fn()})
        except InspectorError as e:
            report["pokes"].append({"label": label, "status": "error",
                                    "error": str(e)})

    print("== tools/list ==")
    tools = inspector.send("tools/list")["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    print(f"{len(tools)} tools: {', '.join(tool_names)}")
    report["pokes"].append({
        "label": "tools/list",
        "status": "ok",
        "count": len(tools),
        "tools": [{"name": t["name"], "description": t.get("description", ""),
                   "inputSchema": t.get("inputSchema", {})}
                  for t in tools],
    })

    print("== tools/call pokes ==")
    mcp_file = "mcp-via-inspector.md"
    calls = [
        # happy path — read the Day-1 sandbox files
        ("read_file", {"filepath": "hello.txt"}, "read_file hello.txt"),
        ("read_file", {"filepath": "notes/mcp-notes.md"},
         "read_file notes/mcp-notes.md (nested path)"),
        # write → read round trip: state must flow through MCP
        ("write_file", {"filepath": mcp_file,
                        "content": "# Round-trip test\n\nWritten through the MCP Inspector.\n"},
         "write_file mcp-via-inspector.md (create)"),
        ("read_file", {"filepath": mcp_file}, "read_file back (round trip, v1)"),
        ("write_file", {"filepath": mcp_file,
                        "content": "# Round-trip test\n\nOverwritten through the MCP Inspector.\n"},
         "write_file mcp-via-inspector.md (overwrite)"),
        ("read_file", {"filepath": mcp_file}, "read_file back (round trip, v2)"),
        # path-safety denials — must come back isError with a hint
        ("read_file", {"filepath": "/etc/hosts"},
         "read_file /etc/hosts (absolute path, expect block)"),
        ("read_file", {"filepath": "../PLAN.md"},
         "read_file ../PLAN.md (traversal, expect block)"),
        ("write_file", {"filepath": "../evil.txt", "content": "x"},
         "write_file ../evil.txt (write escape, expect block)"),
        ("write_file", {"filepath": "binary.bin", "content": "x"},
         "write_file binary.bin (extension not allowed, expect block)"),
    ]
    for tool_name, tool_args, label in calls:
        result = call_tool(inspector, tool_name, tool_args, label)
        report["pokes"].append(result)
        status = result.get("status", "?")
        snippet = (result.get("text") or result.get("error") or "")[:120].replace("\n", " | ")
        print(f"  [{status}] {label}: {snippet}")

    def poke_method(method, params=None, result_keys=()):
        """Send a plain method; tolerate 'Method not found' responses."""
        try:
            resp = inspector.send(method, params)
        except InspectorError as e:
            print(f"  transport error: {e}")
            return {"label": method, "status": "error", "error": str(e)}
        if "error" in resp:
            print(f"  {resp['error'].get('code')} {resp['error'].get('message')}")
            return {"label": method, "status": "not_supported",
                    "error": resp["error"]}
        result = resp.get("result", {})
        out = {"label": method, "status": "ok"}
        for key in result_keys:
            out[key] = result.get(key, [])
        return out

    print("== prompts/list ==")
    pr_entry = poke_method("prompts/list", result_keys=["prompts"])
    if pr_entry.get("prompts") is not None:
        pr_entry["count"] = len(pr_entry["prompts"])
    report["pokes"].append(pr_entry)

    print("== resources/list ==")
    res_entry = poke_method("resources/list", result_keys=["resources"])
    if res_entry.get("resources") is not None:
        res_entry["count"] = len(res_entry["resources"])
    report["pokes"].append(res_entry)

    # SSE transcript: only MCP messages (id-bearing), to keep the file small
    report["sse_event_count"] = len(inspector._all_events)
    report["sse_events"] = [
        {"event_type": e.get("type"), "method": e.get("data", {}).get("method"),
         "id": e.get("data", {}).get("id")}
        for e in inspector._all_events
    ]

    out = HERE / f"day2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nresults -> {out.name}")

    pokes = report["pokes"]
    n_ok = sum(1 for p in pokes if p.get("status") == "ok")
    n_blocked = sum(1 for p in pokes if p.get("status") == "blocked")
    n_unsupported = sum(1 for p in pokes if p.get("status") == "not_supported")
    print(f"\nSUMMARY: {n_ok} ok, {n_blocked} blocked-as-designed, "
          f"{n_unsupported} not-supported ({len(pokes)} pokes total), "
          f"{len(tools)} tools listed")


if __name__ == "__main__":
    main()
