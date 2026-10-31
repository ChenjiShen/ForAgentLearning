"""Week 7 Day 4: file-based memory — read_memory/write_memory tools.

Experiment question (PLAN.md week 7 day 4):
    File-based memory: scratchpads, notes-to-self; why frontier agents write
    markdown files.
    Build: give your agent read_memory/write_memory tools over a memory/ dir
    + a system-prompt note about them. Run tasks that benefit from
    remembering.

Design:

    TOOLS — day 3's agent (search_knowledge) grows two memory tools:
      read_memory(filepath)            read a file from memory/, capped at
                                       8,000 chars with the week-3 truncation
                                       marker
      write_memory(filepath, content, mode="overwrite"|"append")
                                       write a markdown file in memory/;
                                       append mode turns the file into a real
                                       scratchpad (frontier agents append
                                       notes-to-self rather than rewriting)
    Both use the week-3 envelope (status/data/error/truncated) and the week-3
    sandbox discipline: every path is realpath-resolved and verified to stay
    inside memory/ — model-supplied paths are untrusted input (week 3 lesson).

    SYSTEM PROMPT NOTE — the plan asks for one. It tells the agent WHEN to
    use memory (multi-step tasks, intermediate state, facts to carry across
    steps), and the honest reason: files persist, context is re-sent every
    turn and its contents are what the model can afford to keep.

    TASKS — three demos where remembering pays, graded with filesystem
    checks (week 4: the filesystem is ground truth that text judging is
    blind to) plus final-answer checks:
      t1_write_read_prefs   write 3 user preferences to user_prefs.md, read
                            them back, confirm — the write/read round-trip,
                            and the seed for day 5's cross-session test
      t2_scratchpad_notes   3 notebook lookups (search_knowledge), one note
                            appended to scratchpad.md after each, then read
                            the scratchpad and synthesize — notes-to-self
                            mid-task
      t3_arithmetic         4 independent computations, each intermediate
                            appended to calc.md, then the sum — the pure
                            scratchpad use, no retrieval involved
    Memory files are snapshotted into the results JSON and the directory is
    reset before each run (fresh-session semantics; day 5 is where the
    cross-session part is tested).

Run:
    export $(grep DASHSCOPE_API_KEY .env | xargs)
    python experiments/week07/day4_memory_tools.py --selftest   # offline checks, no API
    python experiments/week07/day4_memory_tools.py --smoke      # truncated docs, sanity pass
    python experiments/week07/day4_memory_tools.py              # full run

Outputs (in experiments/week07/):
    day4_results_{ts}.json — config, per-task trace + usage + checks +
                             memory-file snapshots
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
import tempfile
from pathlib import Path

import httpx
import numpy as np
from openai import OpenAI

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Day 1/2 handoff: embedding client + normalize (grader robustness).
from day1_embeddings import get_client  # noqa: E402
from day2_mini_rag import normalize  # noqa: E402
# Day 3 handoff: the search tool, index builder, docs, chat config.
from day3_agent_retrieval import (  # noqa: E402
    CHAT_BASE,
    CHAT_MODEL,
    DOCS,
    DOCS_DIR,
    SEARCH_TOOL,
    build_index,
    make_search,
)

MEMORY_ROOT = _THIS_DIR / "memory"
ALLOWED_EXTENSIONS = {".md", ".txt", ".json"}
READ_CAP = 8_000  # read_memory output cap (chars)

MAX_TURNS = 12  # t2 needs ~9 turns; headroom for reformulated searches

# ---------------------------------------------------------------------------
# Memory tools — week-3 sandbox discipline applied to a memory/ directory.
# ---------------------------------------------------------------------------

def _make_success(data: dict, *, truncated: bool = False,
                  truncated_at: int | None = None) -> str:
    return json.dumps({"status": "success", "data": data, "error": None,
                       "truncated": truncated, "truncated_at": truncated_at},
                      ensure_ascii=False)


def _make_error(error_type: str, message: str) -> str:
    return json.dumps({"status": "error", "data": None,
                       "error": {"type": error_type, "message": message},
                       "truncated": False, "truncated_at": None},
                      ensure_ascii=False)


def resolve_memory_path(user_path: str, root: Path) -> Path:
    """Resolve a model-supplied path inside *root*, or raise ValueError.

    Week-3 lesson: a model-supplied path is untrusted input — join, realpath,
    commonpath against the real memory root.  Also rejects absolute paths
    (week-3 convention: relative paths only).
    """
    if not user_path or not user_path.strip():
        raise ValueError("path must be a non-empty string")
    user_path = user_path.strip()
    # Canonicalize model-supplied paths: the model frequently writes
    # "memory/x.md" or "./x.md" — both mean the same file inside the root.
    # Stripping is normalization of untrusted input, not a permission grant:
    # the realpath/commonpath check below still enforces the boundary.
    for prefix in ("./", "memory/"):
        if user_path.startswith(prefix):
            user_path = user_path[len(prefix):]
    if os.path.isabs(user_path):
        raise ValueError(f"absolute paths not allowed: {user_path!r}")
    root = root.resolve()
    candidate = (root / user_path).resolve()
    try:
        common = os.path.commonpath([str(root), str(candidate)])
    except ValueError:
        raise ValueError(f"path escapes memory dir: {user_path!r}") from None
    if common != str(root):
        raise ValueError(f"path escapes memory dir: {user_path!r}")
    return candidate


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """Week-3 truncation marker: '… [truncated at N chars]'."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n… [truncated at {max_chars} chars]", True


def read_memory(filepath: str, root: Path = MEMORY_ROOT) -> str:
    """Read a file from the memory directory.  Envelope with capped content."""
    try:
        path = resolve_memory_path(filepath, root)
    except ValueError as e:
        return _make_error("sandbox_escape" if "escapes" in str(e) or "absolute" in str(e)
                           else "invalid_input", str(e))
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return _make_error("extension_not_allowed",
                           f"extension {path.suffix!r} not allowed; allowed: "
                           f"{sorted(ALLOWED_EXTENSIONS)}")
    if not path.exists():
        return _make_error("not_found", f"no such memory file: {filepath!r}")
    if not path.is_file():
        return _make_error("not_a_file", f"not a file: {filepath!r}")

    content = path.read_text(encoding="utf-8")
    capped, was_truncated = _truncate(content, READ_CAP)
    return _make_success({"filepath": filepath, "content": capped,
                          "chars": len(content)},
                         truncated=was_truncated,
                         truncated_at=READ_CAP if was_truncated else None)


def write_memory(filepath: str, content: str, mode: str = "overwrite",
                 root: Path = MEMORY_ROOT) -> str:
    """Write (overwrite or append) a file in the memory directory.

    append is what turns a file into a real scratchpad: the agent adds
    notes-to-self incrementally instead of re-reading and rewriting the
    whole file every step.
    """
    try:
        path = resolve_memory_path(filepath, root)
    except ValueError as e:
        return _make_error("sandbox_escape" if "escapes" in str(e) or "absolute" in str(e)
                           else "invalid_input", str(e))
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return _make_error("extension_not_allowed",
                           f"extension {path.suffix!r} not allowed; allowed: "
                           f"{sorted(ALLOWED_EXTENSIONS)}")
    if mode not in ("overwrite", "append"):
        return _make_error("invalid_input",
                           f"mode must be 'overwrite' or 'append', got {mode!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append" and path.exists():
        content = path.read_text(encoding="utf-8") + content
    path.write_text(content, encoding="utf-8")
    return _make_success({"filepath": filepath, "bytes": len(content.encode("utf-8")),
                          "mode": mode})


# ---------------------------------------------------------------------------
# Tool schemas + dispatch.
# ---------------------------------------------------------------------------

READ_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "read_memory",
        "description": (
            "Read a file from your memory directory. Call this to recall "
            "notes-to-self, scratchpad contents, or facts you stored earlier."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string",
                             "description": ("Path relative to the memory "
                                             "directory, e.g. 'scratchpad.md' "
                                             "(no 'memory/' prefix).")},
            },
            "required": ["filepath"],
        },
    },
}

WRITE_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "write_memory",
        "description": (
            "Write a file in your memory directory. Call this to store "
            "intermediate results, decisions, or facts you will need later — "
            "files persist, so write them down instead of only remembering. "
            "Use mode='append' to add to an existing file (scratchpad), "
            "mode='overwrite' to replace it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string",
                             "description": ("Path relative to the memory "
                                             "directory, e.g. 'scratchpad.md' "
                                             "(no 'memory/' prefix).")},
                "content": {"type": "string",
                            "description": "Text to write (markdown preferred)."},
                "mode": {"type": "string", "enum": ["overwrite", "append"],
                         "description": "overwrite replaces the file, append adds to it (default overwrite)."},
            },
            "required": ["filepath", "content"],
        },
    },
}

TOOLS = [SEARCH_TOOL, READ_MEMORY_TOOL, WRITE_MEMORY_TOOL]


# ---------------------------------------------------------------------------
# System prompt — the plan's "system-prompt note" about memory.
# ---------------------------------------------------------------------------

AGENT_SYSTEM = (
    "You are a research assistant with a personal notebook of AI agent "
    "engineering experiment notes (weekly reports week1-week6) and a "
    "persistent memory directory. The notebook is NOT in your context — "
    "search_knowledge is the only way to read it. Make AT MOST 6 "
    "search_knowledge calls per task; after that, answer from what you "
    "have or say: I don't know.\n\n"
    "MEMORY — you have two memory tools over the memory directory. "
    "File paths are relative to the memory directory itself, e.g. "
    "'scratchpad.md' (do not prefix them with 'memory/').\n"
    "- write_memory(filepath, content, mode): store a note in a markdown "
    "file. mode='append' adds to an existing file, 'overwrite' replaces it.\n"
    "- read_memory(filepath): read a memory file back.\n"
    "Use memory as a scratchpad: when a task has several steps, write down "
    "intermediate results, numbers, and decisions as you go, then read the "
    "notes back before giving your final answer. Files persist — write, "
    "don't just remember. Prefer short, structured markdown.\n\n"
    "Answer ONLY from what your tools return. If the tools do not contain "
    "an answer, say exactly: I don't know."
)


def build_tool_functions(search_fn):
    """Dispatch table closed over the search index + the memory root."""
    return {
        "search_knowledge": lambda args: search_fn(args.get("query", ""),
                                                   args.get("k")),
        "read_memory": lambda args: read_memory(args.get("filepath", "")),
        "write_memory": lambda args: write_memory(args.get("filepath", ""),
                                                  args.get("content", ""),
                                                  args.get("mode", "overwrite")),
    }


def run_agent(chat_client, user_prompt: str, tool_fns: dict,
              system: str = AGENT_SYSTEM, tools: list = TOOLS) -> dict:
    """Non-interactive agent loop (week-3 day7 pattern) with 3 tools.

    *system* defaults to day 4's prompt; day 5 passes a cross-session
    variant — the system prompt IS the memory-consultation design, so it
    must be parameterizable.  *tools* likewise: day 5 adds list_memory
    (consultation requires discoverability).
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_prompt}]
    usage = {"prompt": 0, "completion": 0}
    trace: list[dict] = []

    for turn in range(1, MAX_TURNS + 1):
        r = chat_client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=tools, temperature=0.0)
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
                for tc in msg.tool_calls]
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
                                 "content": _make_error("invalid_input",
                                                        f"invalid JSON arguments: {e}")})
                continue
            try:
                output = tool_fns[tc.function.name](args)
            except Exception as e:
                output = _make_error("unknown", str(e))
                trace.append({"name": tc.function.name,
                              "error": f"{type(e).__name__}: {str(e)[:80]}"})
            else:
                rec = {"name": tc.function.name}
                if tc.function.name == "search_knowledge":
                    rec["query"] = args.get("query", "")
                    rec["k"] = args.get("k")
                else:
                    rec["filepath"] = args.get("filepath", "")
                    if tc.function.name == "write_memory":
                        rec["mode"] = args.get("mode", "overwrite")
                        rec["content_chars"] = len(args.get("content", ""))
                trace.append(rec)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": output})

    return {"answer": f"[stopped after {MAX_TURNS} turns]",
            "n_turns": MAX_TURNS, "tool_calls": trace, "usage": usage,
            "error": "max_turns_exceeded"}


# ---------------------------------------------------------------------------
# Tasks — graded with filesystem checks + final-answer checks.
# ---------------------------------------------------------------------------

def _has_any(text: str, alts: list[str]) -> bool:
    n = normalize(text)
    return any(normalize(a) in n for a in alts)


TASKS = [
    {
        "id": "t1_write_read_prefs",
        "memory_file": "user_prefs.md",
        "prompt": (
            "Remember three preferences about the user and save them to "
            "memory/user_prefs.md (markdown): (1) preferred answer language: "
            "English; (2) preferred output format: markdown tables; "
            "(3) favorite model: qwen3.7-plus. Then read the file back and "
            "confirm exactly what you stored."
        ),
        "file_checks": [
            ("user_prefs.md exists",
             lambda p: p.exists()),
            ("prefs mention English",
             lambda p: _has_any(p.read_text(encoding="utf-8"), ["English"])),
            ("prefs mention markdown tables",
             lambda p: _has_any(p.read_text(encoding="utf-8"),
                                ["markdown table", "tables", "table"])),
            ("prefs mention qwen3.7-plus",
             lambda p: _has_any(p.read_text(encoding="utf-8"),
                                ["qwen3.7-plus", "qwen 3.7 plus"])),
        ],
        "answer_checks": [
            ("answer confirms English",
             lambda a: _has_any(a, ["English"])),
            ("answer confirms qwen3.7-plus",
             lambda a: _has_any(a, ["qwen3.7-plus", "qwen 3.7 plus"])),
        ],
    },
    {
        "id": "t2_scratchpad_notes",
        "memory_file": "scratchpad.md",
        "prompt": (
            "Answer three questions about the notebook, one at a time, and "
            "after EACH answer append it as a numbered line to "
            "memory/scratchpad.md (mode='append'):\n"
            "1. In the week 1 day 6 trap_binary_file experiment, what were "
            "the wrong-tool rates of terse vs detailed tool descriptions?\n"
            "2. Roughly how many tokens does a raw HTML page cost, and what "
            "fraction is useful content?\n"
            "3. In the week 4 substring-vs-LLM-judge experiment, what "
            "fraction of the disagreements were substring false negatives?\n"
            "Then read the scratchpad back and give a final summary of all "
            "three findings."
        ),
        "file_checks": [
            ("scratchpad.md exists",
             lambda p: p.exists()),
            ("scratchpad has wrong-tool rates (4/5 or 80%)",
             lambda p: _has_any(p.read_text(encoding="utf-8"),
                                ["4/5", "4 of 5", "80%"])),
            ("scratchpad has 30,000 tokens",
             lambda p: _has_any(p.read_text(encoding="utf-8"),
                                ["30,000", "30000"])),
            ("scratchpad has 91%",
             lambda p: _has_any(p.read_text(encoding="utf-8"), ["91%"])),
        ],
        "answer_checks": [
            ("answer has wrong-tool rates",
             lambda a: _has_any(a, ["4/5", "4 of 5", "80%"])),
            ("answer has 30,000",
             lambda a: _has_any(a, ["30,000", "30000"])),
            ("answer has 91%",
             lambda a: _has_any(a, ["91%"])),
        ],
    },
    {
        "id": "t3_arithmetic_scratchpad",
        "memory_file": "calc.md",
        "prompt": (
            "Compute these four expressions one at a time. After EACH one, "
            "append the expression and its result as a numbered line to "
            "memory/calc.md (mode='append'):\n"
            "1. 17 * 23\n2. 456 * 7\n3. 89 * 64\n4. 1234 + 5678\n"
            "Then read calc.md back and report the sum of the four results."
        ),
        "file_checks": [
            ("calc.md exists",
             lambda p: p.exists()),
            ("calc.md has 391",
             lambda p: _has_any(p.read_text(encoding="utf-8"), ["391"])),
            ("calc.md has 3192",
             lambda p: _has_any(p.read_text(encoding="utf-8"), ["3192"])),
            ("calc.md has 5696",
             lambda p: _has_any(p.read_text(encoding="utf-8"), ["5696"])),
            ("calc.md has 6912",
             lambda p: _has_any(p.read_text(encoding="utf-8"), ["6912"])),
        ],
        "answer_checks": [
            ("answer gives sum 16191",
             lambda a: _has_any(a, ["16191", "16,191"])),
        ],
    },
]


# ---------------------------------------------------------------------------
# Offline selftest — memory tools only, no API.  Attack the path sandbox.
# ---------------------------------------------------------------------------

def selftest() -> None:
    checks: list[bool] = []

    def check(name: str, cond: bool) -> None:
        checks.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # tmp memory root so the real memory/ dir stays clean
    tmp = Path(tempfile.mkdtemp(prefix="day4_selftest_"))
    root = tmp / "mem"
    root.mkdir()

    def env(s: str) -> dict:
        return json.loads(s)

    # 1. path escapes: parent traversal, absolute, hidden traversal
    for label, fp in [("parent traversal", "../x.md"),
                      ("double-dot hidden by normpath", "a/../../x.md"),
                      ("absolute path", "/etc/passwd")]:
        e = env(read_memory(fp, root))
        check(f"read_memory blocks {label}",
              e["status"] == "error"
              and e["error"]["type"] in ("sandbox_escape", "invalid_input"))

    # 2. symlink inside memory dir pointing outside
    outside = tmp / "secret.md"
    outside.write_text("TOP SECRET", encoding="utf-8")
    link = root / "escape.md"
    link.symlink_to(outside)
    e = env(read_memory("escape.md", root))
    check("read_memory blocks symlink escape",
          e["status"] == "error" and e["error"]["type"] == "sandbox_escape")
    link.unlink()

    # 3. write -> read round-trip (overwrite)
    w = env(write_memory("notes.md", "alpha beta", root=root))
    check("write_memory succeeds with envelope",
          w["status"] == "success" and w["data"]["bytes"] == 10)
    r = env(read_memory("notes.md", root))
    check("read_memory returns the written content",
          r["status"] == "success" and r["data"]["content"] == "alpha beta")

    # 4. append mode concatenates (scratchpad semantics)
    write_memory("notes.md", "\ngamma", mode="append", root=root)
    r = env(read_memory("notes.md", root))
    check("append adds to the file",
          r["data"]["content"] == "alpha beta\ngamma")

    # 5. overwrite replaces
    write_memory("notes.md", "only this", root=root)
    r = env(read_memory("notes.md", root))
    check("overwrite replaces the file", r["data"]["content"] == "only this")

    # 6. missing file, bad extension, bad mode
    check("read_memory on missing file -> not_found",
          env(read_memory("nope.md", root))["error"]["type"] == "not_found")
    check("write_memory rejects .py extension",
          env(write_memory("x.py", "print(1)", root=root))["error"]["type"]
          == "extension_not_allowed")
    check("write_memory rejects bad mode",
          env(write_memory("a.md", "x", mode="merge", root=root))["error"]["type"]
          == "invalid_input")

    # 7. read cap + truncation marker
    big = "x" * (READ_CAP + 100)
    write_memory("big.md", big, root=root)
    r = env(read_memory("big.md", root))
    check("read_memory truncates at the cap with the week-3 marker",
          r["truncated"] is True and r["truncated_at"] == READ_CAP
          and f"[truncated at {READ_CAP} chars]" in r["data"]["content"]
          and len(r["data"]["content"]) < len(big))

    # 8. tool schemas: three tools, memory params required
    names = [t["function"]["name"] for t in TOOLS]
    check("three tools: search_knowledge, read_memory, write_memory",
          names == ["search_knowledge", "read_memory", "write_memory"])
    check("read_memory requires filepath",
          "filepath" in READ_MEMORY_TOOL["function"]["parameters"]["required"])
    check("write_memory requires filepath+content",
          set(WRITE_MEMORY_TOOL["function"]["parameters"]["required"])
          == {"filepath", "content"})
    check("write_memory mode enum: overwrite/append",
          WRITE_MEMORY_TOOL["function"]["parameters"]["properties"]["mode"]["enum"]
          == ["overwrite", "append"])

    # 9. system prompt carries the memory note
    check("system prompt tells the agent when to use memory",
          "scratchpad" in AGENT_SYSTEM and "persistent memory" in AGENT_SYSTEM)

    # 10. task ids + their memory files (guards typos in the verify wiring)
    check("three tasks defined",
          [t["id"] for t in TASKS] == ["t1_write_read_prefs",
                                       "t2_scratchpad_notes",
                                       "t3_arithmetic_scratchpad"])
    check("every task names its memory file",
          all("memory_file" in t for t in TASKS))

    # 11. model path habits: "memory/x.md" and "./x.md" canonicalize into the
    # root (the smoke run showed the model writes these forms — the resolver
    # must accept them, not create a nested memory/memory/ dir)
    write_memory("memory/nested.md", "nested write", root=root)
    check("'memory/nested.md' lands at root/nested.md, not memory/memory/",
          (root / "nested.md").exists() and not (root / "memory").exists())
    check("'./nested.md' resolves to the same file",
          env(read_memory("./nested.md", root))["data"]["content"] == "nested write")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nselftest: {sum(checks)}/{len(checks)} checks passed")
    if not all(checks):
        sys.exit(1)


# ---------------------------------------------------------------------------

def snapshot_memory(root: Path) -> dict[str, str]:
    """Capture every memory file's final content for the results JSON."""
    if not root.exists():
        return {}
    return {p.name: p.read_text(encoding="utf-8")
            for p in sorted(root.iterdir()) if p.is_file()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Week 7 Day 4: file-based memory tools + tasks that benefit from remembering")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline memory-tool checks only (no API calls)")
    ap.add_argument("--smoke", action="store_true",
                    help="truncate each doc to 10k chars (quick sanity pass)")
    ap.add_argument("--tasks", default="",
                    help="comma-separated task ids to run (default: all)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    # fresh-session semantics: reset the memory dir before the run
    if MEMORY_ROOT.exists():
        shutil.rmtree(MEMORY_ROOT)
    MEMORY_ROOT.mkdir(parents=True)

    docs = {}
    for name in DOCS:
        path = DOCS_DIR / name
        if not path.exists():
            sys.exit(f"missing document: {path}")
        docs[name] = path.read_text()[:10_000 if args.smoke else None]
    total_chars = sum(len(t) for t in docs.values())
    print(f"docs: {len(docs)} files, {total_chars:,} chars (smoke={args.smoke}); "
          f"tasks: {len(TASKS)}; chat={CHAT_MODEL}; memory root: {MEMORY_ROOT}")

    embed_client = get_client()
    chat_client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=CHAT_BASE,
        http_client=httpx.Client(trust_env=False,
                                 timeout=httpx.Timeout(600.0, connect=15.0)),
    )
    t0 = time.time()

    print("building index (fixed_400) ...")
    flat, E_hat, index_stats = build_index(docs, embed_client)
    print(f"  {index_stats['n_chunks']} chunks, {index_stats['embed_tokens']} embed tokens")

    search_fn = make_search(flat, E_hat, embed_client)
    tool_fns = build_tool_functions(search_fn)

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tasks = [t for t in TASKS if not task_ids or t["id"] in task_ids]

    runs = []
    for task in tasks:
        cell_t0 = time.time()
        print(f"\n=== {task['id']} ===")
        res = run_agent(chat_client, task["prompt"], tool_fns)
        elapsed = time.time() - cell_t0

        file_checks, answer_checks = [], []
        # locate the task's memory file by basename anywhere under the
        # memory root — grading cares that the agent persisted the file,
        # not which exact path it chose
        matches = list(MEMORY_ROOT.rglob(task["memory_file"]))
        mfile = matches[0] if matches else MEMORY_ROOT / task["memory_file"]
        for name, fn in task["file_checks"]:
            try:
                ok = fn(mfile)
            except Exception as e:
                ok, name = False, f"{name} (error: {e})"
            file_checks.append({"check": name, "pass": ok})
            print(f"  [{'PASS' if ok else 'FAIL'}] file: {name}")
        for name, fn in task["answer_checks"]:
            ok = fn(res["answer"])
            answer_checks.append({"check": name, "pass": ok})
            print(f"  [{'PASS' if ok else 'FAIL'}] answer: {name}")

        n_calls = {name: 0 for name in ("search_knowledge", "read_memory",
                                        "write_memory")}
        for c in res["tool_calls"]:
            n_calls[c["name"]] = n_calls.get(c["name"], 0) + 1
        print(f"  turns={res['n_turns']} calls={n_calls} "
              f"prompt={res['usage']['prompt']:,} "
              f"completion={res['usage']['completion']:,} {elapsed:.0f}s")

        runs.append({
            "task_id": task["id"], "prompt": task["prompt"],
            "answer": res["answer"],
            "n_turns": res["n_turns"],
            "tool_calls": res["tool_calls"],
            "calls_by_tool": n_calls,
            "usage": res["usage"],
            "elapsed_s": round(elapsed, 1),
            "file_checks": file_checks,
            "answer_checks": answer_checks,
            "error": res.get("error"),
        })

    print(f"\nelapsed {time.time() - t0:.0f}s")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment": "week07_day4_memory_tools",
        "timestamp": ts,
        "config": {
            "docs": DOCS, "doc_chars": {d: len(t) for d, t in docs.items()},
            "smoke": args.smoke, "max_turns": MAX_TURNS,
            "read_cap": READ_CAP, "memory_root": str(MEMORY_ROOT),
            "chat": {"model": CHAT_MODEL, "base_url": CHAT_BASE,
                     "temperature": 0.0},
        },
        "index": index_stats,
        "memory_files": snapshot_memory(MEMORY_ROOT),
        "runs": runs,
        "summary": {
            "tasks": len(runs),
            "file_checks_pass": sum(c["pass"] for r in runs for c in r["file_checks"]),
            "file_checks_total": sum(len(r["file_checks"]) for r in runs),
            "answer_checks_pass": sum(c["pass"] for r in runs for c in r["answer_checks"]),
            "answer_checks_total": sum(len(r["answer_checks"]) for r in runs),
            "total_tokens_gen": sum(r["usage"]["prompt"] + r["usage"]["completion"]
                                    for r in runs),
        },
    }
    out_path = _THIS_DIR / f"day4_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"results -> {out_path}")

    s = results["summary"]
    print(f"\nsummary: {s['file_checks_pass']}/{s['file_checks_total']} file checks, "
          f"{s['answer_checks_pass']}/{s['answer_checks_total']} answer checks, "
          f"{s['total_tokens_gen']:,} gen tokens across {s['tasks']} tasks")


if __name__ == "__main__":
    main()
