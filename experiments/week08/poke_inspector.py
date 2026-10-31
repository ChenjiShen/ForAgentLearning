#!/usr/bin/env python3
"""Week08 Day1: poke the filesystem MCP server through the MCP Inspector.

Drives the Inspector's own REST API (the same one its web UI uses) at
http://127.0.0.1:6274:

    POST /api/mcp/connect   create a session (spawns the stdio MCP server)
    GET  /api/mcp/events    SSE stream of JSON-RPC messages for the session
    POST /api/mcp/send      send a JSON-RPC message over the session

Usage:
    python poke_inspector.py [--base-url http://127.0.0.1:6274]
                             [--token-file logs/inspector.log]

Writes day1_results_<timestamp>.json next to this script.
"""

import argparse
import json
import queue
import re
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
PROTOCOL_VERSION = "2025-06-18"
TOKEN_RE = re.compile(r"MCP_INSPECTOR_API_TOKEN=([a-f0-9]{16,})")

# urllib on macOS honors the system proxy config (127.0.0.1:7890, which is
# dead on this machine). The Inspector is on localhost, so disable proxies.
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class InspectorError(RuntimeError):
    pass


class InspectorClient:
    """Minimal client for the MCP Inspector's REST/SSE proxy API."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.session_id = None
        self._events = queue.Queue()
        self._sse_thread = None
        self._sse_stop = threading.Event()
        self._all_events = []  # full transcript of every SSE event
        self._next_id = 1

    # ---- HTTP helpers ----------------------------------------------------
    def _request(self, method: str, path: str, body=None):
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "x-mcp-remote-auth": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with NO_PROXY_OPENER.open(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise InspectorError(f"HTTP {e.code} on {method} {path}: {detail}") from e

    # ---- session lifecycle ------------------------------------------------
    def connect(self, config: dict) -> str:
        """Create a session; returns sessionId. Spawns the stdio server."""
        result = self._request("POST", "/api/mcp/connect", {"config": config})
        sid = result.get("sessionId")
        if not sid:
            raise InspectorError(f"connect failed: {result}")
        self.session_id = sid
        self._start_sse()
        return sid

    def _start_sse(self):
        """Open the SSE event stream in a background thread."""
        url = f"{self.base_url}/api/mcp/events?sessionId={self.session_id}"
        req = urllib.request.Request(
            url,
            headers={
                "x-mcp-remote-auth": f"Bearer {self.token}",
                "Accept": "text/event-stream",
            },
        )

        def reader():
            try:
                with NO_PROXY_OPENER.open(req, timeout=None) as resp:
                    event_type, data_lines = None, []
                    for raw in resp:
                        if self._sse_stop.is_set():
                            break
                        line = raw.decode(errors="replace").rstrip("\n")
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].strip())
                        elif line == "" and data_lines:
                            payload = json.loads("\n".join(data_lines))
                            self._all_events.append(payload)
                            self._events.put(payload)
                            event_type, data_lines = None, []
            except Exception as e:  # stream closed when session ends
                if not self._sse_stop.is_set():
                    self._events.put({"type": "sse-error", "data": str(e)})

        self._sse_thread = threading.Thread(target=reader, daemon=True)
        self._sse_thread.start()

    # ---- MCP JSON-RPC -----------------------------------------------------
    def send(self, method: str, params: dict | None = None, msg_id=None) -> dict:
        """Send a JSON-RPC message; wait for and return the matching response."""
        msg_id = msg_id if msg_id is not None else self._next_id
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            message["params"] = params
        self._request(
            "POST",
            "/api/mcp/send",
            {"sessionId": self.session_id, "message": message,
             "protocolVersion": PROTOCOL_VERSION},
        )
        # The response arrives over the SSE stream; match it by id.
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                event = self._events.get(timeout=deadline - time.time())
            except queue.Empty:
                break
            if event.get("type") == "sse-error":
                raise InspectorError(f"SSE stream error: {event.get('data')}")
            msg = event.get("data")
            if isinstance(msg, dict) and msg.get("id") == msg_id:
                return msg
        raise InspectorError(f"no response for {method} (id={msg_id})")

    def notify(self, method: str, params: dict | None = None):
        """Send a notification (no id, no response expected)."""
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._request(
            "POST",
            "/api/mcp/send",
            {"sessionId": self.session_id, "message": message,
             "protocolVersion": PROTOCOL_VERSION},
        )


def get_config(inspector: InspectorClient) -> dict:
    """Fetch the preconfigured stdio server from the Inspector and adapt it
    to the connect-endpoint config shape."""
    cfg = inspector._request("GET", "/api/config")
    return {
        "type": "stdio",
        "command": cfg["defaultCommand"],
        "args": cfg["defaultArgs"],
        "cwd": cfg["defaultCwd"],
        "env": cfg.get("defaultEnvironment"),
    }


def call_tool(inspector, name, arguments, label):
    """Call one tool and shape the result for the report."""
    try:
        resp = inspector.send("tools/call", {"name": name, "arguments": arguments})
        if "error" in resp:
            return {"label": label, "tool": name, "args": arguments,
                    "status": "error", "error": resp["error"]}
        result = resp.get("result", {})
        content = result.get("content", [])
        is_error = result.get("isError", False) or any(
            c.get("type") == "text" and c.get("text", "").startswith("Error")
            for c in content)
        text = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
        # isError means the tool ran but the server denied the request
        # (e.g. path outside allowed directories) — the poke itself worked.
        return {"label": label, "tool": name, "args": arguments,
                "status": "blocked" if is_error else "ok",
                "isError": bool(is_error),
                "text": text}
    except InspectorError as e:
        return {"label": label, "tool": name, "args": arguments,
                "status": "error", "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:6274")
    ap.add_argument("--token-file", default=str(HERE / "logs" / "inspector.log"))
    args = ap.parse_args()

    token = None
    log_path = Path(args.token_file)
    if log_path.exists():
        m = TOKEN_RE.search(log_path.read_text(errors="replace"))
        if m:
            token = m.group(1)
    if not token:
        print(f"error: no MCP_INSPECTOR_API_TOKEN found in {log_path}", file=sys.stderr)
        sys.exit(1)

    inspector = InspectorClient(args.base_url, token)

    print("== fetch /api/config ==")
    cfg = get_config(inspector)
    print(f"inspector, stdio server: "
          f"{cfg['command']} {' '.join(cfg['args'])}")

    print("== connect (spawns filesystem server) ==")
    session_id = inspector.connect(cfg)
    print(f"sessionId: {session_id}")

    print("== initialize ==")
    init = inspector.send("initialize", {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "week08-poke-cli", "version": "0.0.1"},
    })
    server_info = init["result"]["serverInfo"]
    print(f"server: {server_info['name']} {server_info['version']} "
          f"(protocol {init['result'].get('protocolVersion')})")
    inspector.notify("notifications/initialized")

    report = {
        "day": "week08_day1",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inspector_base_url": args.base_url,
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
        "tools": [{"name": t["name"], "description": t.get("description", "")}
                  for t in tools],
    })

    sandbox = cfg["cwd"] + "/sandbox"

    print("== tools/call pokes ==")
    calls = [
        ("list_allowed_directories", {}, "tools/call list_allowed_directories"),
        ("list_directory", {"path": sandbox}, "tools/call list_directory (sandbox root)"),
        ("read_file", {"path": sandbox + "/hello.txt"}, "tools/call read_file (hello.txt)"),
        ("get_file_info", {"path": sandbox + "/notes/mcp-notes.md"},
         "tools/call get_file_info (notes/mcp-notes.md)"),
        ("read_file", {"path": "/etc/hosts"},
         "tools/call read_file (/etc/hosts — OUTSIDE sandbox, expect block)"),
        ("read_file", {"path": "../PLAN.md"},
         "tools/call read_file (../PLAN.md — traversal, expect block)"),
    ]
    for tool_name, tool_args, label in calls:
        result = call_tool(inspector, tool_name, tool_args, label)
        report["pokes"].append(result)
        status = result.get("status", "?")
        snippet = (result.get("text") or result.get("error") or "")[:100].replace("\n", " | ")
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

    print("== resources/list ==")
    res_entry = poke_method("resources/list", result_keys=["resources"])
    if res_entry.get("resources") is not None:
        resources = res_entry["resources"]
        res_entry["count"] = len(resources)
        res_entry["resources"] = [{"uri": r["uri"], "name": r.get("name")}
                                  for r in resources]
    report["pokes"].append(res_entry)

    print("== resources/templates/list ==")
    tpl_entry = poke_method("resources/templates/list",
                            result_keys=["resourceTemplates"])
    if tpl_entry.get("resourceTemplates") is not None:
        templates = tpl_entry["resourceTemplates"]
        tpl_entry["count"] = len(templates)
        tpl_entry["uriTemplates"] = [t["uriTemplate"] for t in templates]
    report["pokes"].append(tpl_entry)

    print("== resources/read (file:// hello.txt) ==")
    read_entry = poke_method("resources/read",
                             {"uri": f"file://{sandbox}/hello.txt"},
                             result_keys=["contents"])
    if read_entry.get("contents") is not None:
        read_entry["text"] = "\n".join(c.get("text", "")
                                       for c in read_entry["contents"])
        print(f"  content: {read_entry['text']!r}")
    report["pokes"].append(read_entry)

    print("== prompts/list ==")
    pr_entry = poke_method("prompts/list", result_keys=["prompts"])
    if pr_entry.get("prompts") is not None:
        pr_entry["count"] = len(pr_entry["prompts"])
    report["pokes"].append(pr_entry)

    # SSE transcript: only MCP messages (id-bearing), to keep the file small
    report["sse_event_count"] = len(inspector._all_events)
    report["sse_events"] = [
        {"event_type": e.get("type"), "method": e.get("data", {}).get("method"),
         "id": e.get("data", {}).get("id")}
        for e in inspector._all_events
    ]

    out = HERE / f"day1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nresults -> {out.name}")

    # summary
    pokes = report["pokes"]
    n_ok = sum(1 for p in pokes if p.get("status") == "ok")
    n_blocked = sum(1 for p in pokes if p.get("status") == "blocked")
    n_unsupported = sum(1 for p in pokes if p.get("status") == "not_supported")
    print(f"\nSUMMARY: {n_ok} ok, {n_blocked} blocked-as-designed, "
          f"{n_unsupported} not-supported ({len(pokes)} pokes total), "
          f"{len(tools)} tools listed")


if __name__ == "__main__":
    main()
