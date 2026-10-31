"""Week 3: sandboxed file tools + allowlisted shell commands + web tools.

Day 1 — File tools (read_file, write_file, list_dir) are confined to a sandbox
root. Every user-supplied path is resolved with os.path.realpath() and verified
to stay inside the sandbox via os.path.commonpath().

Day 2 — run_command executes only allowlisted executables (no shell), with a
timeout and output cap. Blocklists lose to injection; an allowlist does not.

Day 3 — web_fetch fetches a URL via requests, extracts plain text from HTML
with BeautifulSoup, and caps the output length. web_search queries DuckDuckGo
and returns a list of results. Both validate inputs and handle errors safely.

Day 4 — Combine all tools into one agent. Give it a real multi-step task:
"find X online, save a summary to notes.md". Watch the full trace.

Run tests (no API key needed):
    python experiments/week03/agent.py --test-sandbox
    python experiments/week03/agent.py --test-commands
    python experiments/week03/agent.py --test-web

Run the agent (needs DASHSCOPE_API_KEY):
    python experiments/week03/agent.py
    python experiments/week03/agent.py --day2
    python experiments/week03/agent.py --day3
    python experiments/week03/agent.py --day4
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import urllib.parse
from collections.abc import Callable
from pathlib import Path

import requests
from bs4 import BeautifulSoup
# from duckduckgo_search import DDGS
from ddgs import DDGS
from openai import OpenAI

MODEL = os.getenv("MODEL_NAME", "qwen3.7-plus")

# All file tools operate only inside this directory.
SANDBOX_ROOT = Path(__file__).resolve().parent / "sandbox"
ALLOWED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv"}

# Day 2: command tool limits — allowlist beats blocklist for injection defense.
ALLOWED_EXECUTABLES = frozenset(
    {"ls", "cat", "echo", "grep", "wc", "head", "tail", "python3", "python"}
)
COMMAND_TIMEOUT_SECONDS = 5
COMMAND_OUTPUT_CAP = 4_000
# Reject shell syntax we do not implement — not a executable blocklist, but a
# clear error when the model/user expects pipes or command chaining.
UNSUPPORTED_SHELL_CHARS = frozenset("|;&$`><")

# ---------------------------------------------------------------------------
# Day 3: Web tool constants
# ---------------------------------------------------------------------------

# web_fetch limits
WEB_FETCH_TIMEOUT = 10  # seconds
WEB_FETCH_MAX_RESPONSE_BYTES = 5_000_000  # 5 MB — cap raw response body size
WEB_FETCH_DEFAULT_MAX_CHARS = 5_000  # default cap on extracted text

# web_search limits
WEB_SEARCH_DEFAULT_MAX_RESULTS = 5
WEB_SEARCH_HARD_MAX_RESULTS = 10

# ---------------------------------------------------------------------------
# Day 5: Unified tool-result format — consistent envelope, truncation, errors
# ---------------------------------------------------------------------------

# Standardised error types the model can pattern-match and act on.
# Each type has a corresponding hint in _ERROR_HINTS that tells the model
# what to try next — this is what "actionable errors" means.


class ErrorType:
    SANDBOX_ESCAPE = "sandbox_escape"
    NOT_FOUND = "not_found"
    NOT_A_FILE = "not_a_file"
    NOT_A_DIRECTORY = "not_a_directory"
    EXTENSION_NOT_ALLOWED = "extension_not_allowed"
    EXECUTABLE_NOT_ALLOWED = "executable_not_allowed"
    DANGEROUS_ARGS = "dangerous_args"
    DANGEROUS_COMMAND = "dangerous_command"
    SHELL_METACHARACTER = "shell_metacharacter"
    COMMAND_TIMEOUT = "command_timeout"
    COMMAND_ERROR = "command_error"
    INVALID_URL_SCHEME = "invalid_url_scheme"
    PRIVATE_ADDRESS = "private_address"
    REQUEST_FAILED = "request_failed"
    SEARCH_FAILED = "search_failed"
    INVALID_INPUT = "invalid_input"
    PERMISSION_DENIED = "permission_denied"
    UNKNOWN = "unknown"


# Each error type maps to a hint that tells the model what to try next.
# Key design principle: every error must be *actionable* — the model should
# be able to read the hint and adjust its behaviour without human help.
_ERROR_HINTS: dict[str, str] = {
    ErrorType.SANDBOX_ESCAPE: (
        "Use list_dir to browse available files. All paths must be "
        "relative (no leading '/' or '..') and stay inside the sandbox."
    ),
    ErrorType.NOT_FOUND: (
        "Use list_dir to see what files exist, then try again with a valid path."
    ),
    ErrorType.NOT_A_FILE: (
        "The path points to a directory, not a file. Use list_dir to "
        "browse its contents, or specify a file path instead."
    ),
    ErrorType.NOT_A_DIRECTORY: (
        "The path points to a file, not a directory. Use read_file to "
        "read it, or specify a directory path instead."
    ),
    ErrorType.EXTENSION_NOT_ALLOWED: (
        "Use a file with one of the allowed extensions: "
        + ", ".join(sorted(ALLOWED_EXTENSIONS))
        + "."
    ),
    ErrorType.EXECUTABLE_NOT_ALLOWED: (
        "Allowed executables: "
        + ", ".join(sorted(ALLOWED_EXECUTABLES))
        + ". Use one of these instead. For HTTP requests, use web_fetch "
        "or web_search (curl and wget are blocked)."
    ),
    ErrorType.DANGEROUS_ARGS: (
        "The -c, -m, and -i flags are blocked for Python because they "
        "enable arbitrary code execution. Write your code to a .py file "
        "with write_file, then run it with 'python filename.py'."
    ),
    ErrorType.DANGEROUS_COMMAND: (
        "This command is blocked by the defense-in-depth layer. Use an "
        "allowlisted alternative: "
        + ", ".join(sorted(ALLOWED_EXECUTABLES))
        + "."
    ),
    ErrorType.SHELL_METACHARACTER: (
        "Shell metacharacters (| ; & $ ` > <) are not supported — "
        "run_command does not invoke a shell. Pass one executable and "
        "its arguments only. Use write_file + run_command for multi-step "
        "workflows."
    ),
    ErrorType.COMMAND_TIMEOUT: (
        "The command took too long. Try a more efficient approach: use "
        "narrower grep patterns, smaller file scopes, or split the work "
        "into smaller steps."
    ),
    ErrorType.COMMAND_ERROR: (
        "The command exited with a non-zero status. Check the stderr "
        "field for details and adjust the command arguments accordingly."
    ),
    ErrorType.INVALID_URL_SCHEME: (
        "Only http:// and https:// URLs are supported. file://, ftp://, "
        "and other schemes are blocked for security."
    ),
    ErrorType.PRIVATE_ADDRESS: (
        "The URL resolves to a private or internal IP address (blocked). "
        "Only public internet URLs can be fetched. Try searching for a "
        "public equivalent with web_search."
    ),
    ErrorType.REQUEST_FAILED: (
        "The web request failed. Verify the URL is correct (try "
        "web_search to find it), or retry later — the server may be "
        "temporarily unavailable."
    ),
    ErrorType.SEARCH_FAILED: (
        "The search query failed. Try a simpler, shorter query with "
        "fewer special characters, or try a different search engine."
    ),
    ErrorType.INVALID_INPUT: (
        "The input was empty or invalid. Provide a non-empty, valid value."
    ),
    ErrorType.PERMISSION_DENIED: (
        "This command requires human approval and was denied. Try an "
        "alternative approach using safe read-only commands (ls, cat, "
        "grep, wc, head, tail) instead."
    ),
}

# Per-tool output size caps (chars).  Shared across the codebase so the
# model sees a consistent truncation story no matter which tool it calls.
TOOL_OUTPUT_CAPS: dict[str, int] = {
    "read_file_content": 8_000,
    "run_command_stdout": COMMAND_OUTPUT_CAP,
    "run_command_stderr": COMMAND_OUTPUT_CAP,
    "web_fetch_text": WEB_FETCH_DEFAULT_MAX_CHARS,
}


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate *text* to *max_chars* with a consistent marker.

    Every tool uses this single function so the truncation marker is
    always the same: ``… [truncated at N chars]``.  The model learns this
    pattern once and recognises it everywhere.
    """
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n… [truncated at {max_chars} chars]", True


def _make_success(
    data: dict | None = None,
    *,
    truncated: bool = False,
    truncated_at: int | None = None,
) -> str:
    """Build a consistent success-result envelope (JSON string).

    Every tool that succeeds returns the same outer shape::

        {"status":"success", "data":{…}, "error":null,
         "truncated":false, "truncated_at":null}

    The model can always count on ``.status``, ``.data``, and
    ``.truncated`` being present.
    """
    return json.dumps(
        {
            "status": "success",
            "data": data or {},
            "error": None,
            "truncated": truncated,
            "truncated_at": truncated_at,
        }
    )


def _make_error(
    error_type: str,
    message: str,
    *,
    hint: str | None = None,
) -> str:
    """Build a consistent error-result envelope with an *actionable* hint.

    Every tool that fails returns the same outer shape::

        {"status":"error", "data":null,
         "error":{"type":"…","message":"…","hint":"…"},
         "truncated":false, "truncated_at":null}

    The ``hint`` field is the key innovation — it tells the model *what to
    try next* instead of just reporting what went wrong.
    """
    return json.dumps(
        {
            "status": "error",
            "data": None,
            "error": {
                "type": error_type,
                "message": message,
                "hint": hint
                or _ERROR_HINTS.get(
                    error_type, "Check the tool documentation for valid inputs."
                ),
            },
            "truncated": False,
            "truncated_at": None,
        }
    )


# Content types we are willing to process as HTML
_HTML_CONTENT_TYPES = frozenset(
    {"text/html", "application/xhtml+xml", "text/xml", "application/xml"}
)

# ---------------------------------------------------------------------------
# Day 2+: Argument validation & permission confirmation constants
# ---------------------------------------------------------------------------

# Per-executable argument flags that enable arbitrary code execution or
# interactive access. The allowlist only checks argv[0] — these flags turn
# an otherwise-safe executable into a code-execution backdoor.
DANGEROUS_ARG_BLOCKLIST: dict[str, frozenset[str]] = {
    "python": frozenset({"-c", "-m", "-i"}),
    "python3": frozenset({"-c", "-m", "-i"}),
}

# Defense-in-depth: command categories that are explicitly blocked even
# though they are already rejected by the executable allowlist. These
# provide clear error messages, an audit trail, and protection against
# accidental allowlist additions.
DESTRUCTIVE_COMMANDS: frozenset[str] = frozenset({
    "rm", "rmdir", "dd", "mkfs", "fdisk", "mv", "cp",
    "chmod", "chown", "shred", "unlink",
    "shutdown", "reboot", "halt", "poweroff",
})
NETWORK_COMMANDS: frozenset[str] = frozenset({
    "curl", "wget", "nc", "ncat", "netcat",
    "ssh", "scp", "sftp", "ftp", "telnet",
    "nmap", "ping", "traceroute",
})
PRIVILEGE_COMMANDS: frozenset[str] = frozenset({
    "sudo", "su", "chroot", "pkexec", "doas",
})
SHELL_INTERPRETERS: frozenset[str] = frozenset({
    "sh", "bash", "zsh", "dash", "ksh", "fish", "perl", "ruby", "lua",
})
DANGEROUS_COMMANDS: frozenset[str] = (
    DESTRUCTIVE_COMMANDS | NETWORK_COMMANDS | PRIVILEGE_COMMANDS | SHELL_INTERPRETERS
)

# Risk classification for permission confirmation prompts.
# - safe: read-only commands that cannot modify state or execute code
# - sensitive: commands that can execute code, write files, or access the network
SAFE_EXECUTABLES: frozenset[str] = frozenset({"ls", "cat", "grep", "wc", "head", "tail"})
SENSITIVE_EXECUTABLES: frozenset[str] = frozenset({"echo", "python", "python3"})


# ---------------------------------------------------------------------------
# Sandbox path resolution — the security boundary
# ---------------------------------------------------------------------------
def resolve_sandbox_path(user_path: str, *, must_exist: bool = False) -> Path:
    """Resolve a user path inside SANDBOX_ROOT, or raise ValueError.

    Why not just ban '..' and check startswith(workspace)?
      - normpath collapses '../' so a naive '..' check misses some forms
      - startswith('/tmp/sandbox') wrongly allows '/tmp/sandbox-evil/...'
      - symlinks can point outside the sandbox unless you realpath()

    The fix: join → realpath → commonpath with the real sandbox root.
    """
    if not user_path or not user_path.strip():
        raise ValueError("Path must be a non-empty string")

    user_path = user_path.strip()
    if os.path.isabs(user_path):
        raise ValueError(f"Absolute paths not allowed: {user_path!r}")

    sandbox_root = SANDBOX_ROOT.resolve()
    candidate = (sandbox_root / user_path).resolve()

    try:
        common = os.path.commonpath([str(sandbox_root), str(candidate)])
    except ValueError:
        # Different drives on Windows — treat as escape.
        raise ValueError(f"Path escapes sandbox: {user_path!r}") from None

    if common != str(sandbox_root):
        raise ValueError(f"Path escapes sandbox: {user_path!r}")

    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"Not found in sandbox: {user_path!r}")

    return candidate


def _check_extension(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Extension {path.suffix!r} not allowed. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )


# ---------------------------------------------------------------------------
# File tool implementations
# ---------------------------------------------------------------------------
def read_file(filepath: str) -> str:
    """Read a text file from the sandbox.

    Returns a unified ToolResult envelope (JSON string).  The file content
    is truncated at TOOL_OUTPUT_CAPS['read_file_content'] chars with a
    consistent ``… [truncated at N chars]`` marker.
    """
    # --- input validation + path resolution (with classified errors) -------
    try:
        path = resolve_sandbox_path(filepath, must_exist=True)
    except FileNotFoundError as e:
        return _make_error(ErrorType.NOT_FOUND, str(e))
    except ValueError as e:
        msg = str(e)
        if "escapes sandbox" in msg or "Absolute paths" in msg:
            return _make_error(ErrorType.SANDBOX_ESCAPE, msg)
        if "non-empty" in msg:
            return _make_error(ErrorType.INVALID_INPUT, msg)
        return _make_error(ErrorType.UNKNOWN, msg)

    try:
        _check_extension(path)
    except ValueError as e:
        return _make_error(ErrorType.EXTENSION_NOT_ALLOWED, str(e))

    if not path.is_file():
        return _make_error(ErrorType.NOT_A_FILE, f"Not a file: {filepath!r}")

    # --- read + truncate ---------------------------------------------------
    content = path.read_text(encoding="utf-8")
    capped_content, was_truncated = _truncate(
        content, TOOL_OUTPUT_CAPS["read_file_content"]
    )
    return _make_success(
        {
            "filepath": filepath,
            "content": capped_content,
            "bytes": len(content.encode("utf-8")),
            "extension": path.suffix.lower(),
        },
        truncated=was_truncated,
        truncated_at=TOOL_OUTPUT_CAPS["read_file_content"] if was_truncated else None,
    )


def write_file(filepath: str, content: str) -> str:
    """Write (overwrite) a text file inside the sandbox.

    Returns a unified ToolResult envelope (JSON string).
    """
    # --- input validation + path resolution ---------------------------------
    try:
        path = resolve_sandbox_path(filepath)
    except ValueError as e:
        msg = str(e)
        if "escapes sandbox" in msg or "Absolute paths" in msg:
            return _make_error(ErrorType.SANDBOX_ESCAPE, msg)
        if "non-empty" in msg:
            return _make_error(ErrorType.INVALID_INPUT, msg)
        return _make_error(ErrorType.UNKNOWN, msg)

    try:
        _check_extension(path)
    except ValueError as e:
        return _make_error(ErrorType.EXTENSION_NOT_ALLOWED, str(e))

    if path.exists() and not path.is_file():
        return _make_error(ErrorType.NOT_A_FILE, f"Not a file: {filepath!r}")

    # --- write -------------------------------------------------------------
    path.parent.mkdir(parents=True, exist_ok=True)
    byte_count = len(content.encode("utf-8"))
    path.write_text(content, encoding="utf-8")
    return _make_success({"filepath": filepath, "bytes": byte_count})


def list_dir(dirpath: str = ".") -> str:
    """List files and directories inside the sandbox.

    Returns a unified ToolResult envelope (JSON string).  Symlinks
    pointing outside the sandbox are listed but marked inaccessible.
    """
    # --- input validation + path resolution ---------------------------------
    try:
        path = resolve_sandbox_path(dirpath, must_exist=True)
    except FileNotFoundError as e:
        return _make_error(ErrorType.NOT_FOUND, str(e))
    except ValueError as e:
        msg = str(e)
        if "escapes sandbox" in msg or "Absolute paths" in msg:
            return _make_error(ErrorType.SANDBOX_ESCAPE, msg)
        if "non-empty" in msg:
            return _make_error(ErrorType.INVALID_INPUT, msg)
        return _make_error(ErrorType.UNKNOWN, msg)

    if not path.is_dir():
        return _make_error(ErrorType.NOT_A_DIRECTORY, f"Not a directory: {dirpath!r}")

    # --- enumerate ----------------------------------------------------------
    entries = []
    for name in sorted(path.iterdir()):
        entry_path = name.resolve()
        try:
            in_sandbox = os.path.commonpath(
                [str(SANDBOX_ROOT.resolve()), str(entry_path)]
            ) == str(SANDBOX_ROOT.resolve())
        except ValueError:
            in_sandbox = False

        kind = "dir" if name.is_dir() else "file"
        if in_sandbox:
            rel = str(entry_path.relative_to(SANDBOX_ROOT.resolve()))
        else:
            rel = name.name  # symlink escape — show name only, block via resolver
        entries.append(
            {
                "name": name.name,
                "path": rel,
                "type": kind,
                "accessible": in_sandbox,
            }
        )

    return _make_success({"directory": dirpath, "entries": entries})


# ---------------------------------------------------------------------------
# Command tool — allowlist + no shell + timeout + output cap
# ---------------------------------------------------------------------------
def _check_dangerous_args(argv: list[str]) -> None:
    """Reject dangerous argument combinations on allowlisted executables.

    Why per-executable checks?
      The allowlist only validates argv[0].  ``python -c 'import os'``
      passes the allowlist check but grants arbitrary code execution.
      We must inspect remaining arguments for known-dangerous flags.

    Why not a general argument blocklist?
      Blocklists on arbitrary strings are fragile — attackers use encoding
      tricks and alternative flag forms.  Per-executable patterns are
      narrow, well-defined, and auditable.
    """
    executable = Path(argv[0]).name
    blocked = DANGEROUS_ARG_BLOCKLIST.get(executable, frozenset())
    if not blocked:
        return

    for arg in argv[1:]:
        if arg in blocked:
            raise ValueError(
                f"Argument {arg!r} is blocked for executable {executable!r}. "
                f"This flag enables arbitrary code execution or interactive "
                f"access and is not permitted in the sandbox."
            )


def _check_defense_in_depth(argv: list[str]) -> None:
    """Explicitly block dangerous commands as defense-in-depth.

    The allowlist check in run_command() already rejects non-allowlisted
    executables.  This function provides:
    1. Clear, category-specific error messages when dangerous commands are attempted
    2. An audit trail for security monitoring
    3. Protection against accidental allowlist changes — if someone adds
       'rm' to ALLOWED_EXECUTABLES, this check still prevents destruction.
    """
    executable = Path(argv[0]).name
    if executable not in DANGEROUS_COMMANDS:
        return

    categories: list[str] = []
    if executable in DESTRUCTIVE_COMMANDS:
        categories.append("destructive (modifies system state)")
    if executable in NETWORK_COMMANDS:
        categories.append("network access")
    if executable in PRIVILEGE_COMMANDS:
        categories.append("privilege escalation")
    if executable in SHELL_INTERPRETERS:
        categories.append("shell interpreter")

    raise ValueError(
        f"Executable {executable!r} is blocked as a "
        f"{', '.join(categories)} command. "
        f"This is a defense-in-depth restriction that supplements the "
        f"executable allowlist."
    )


def _classify_executable(executable: str) -> str:
    """Classify an executable by risk level for confirmation prompts.

    Returns:
        'safe' — read-only commands that cannot modify state or execute code.
        'sensitive' — commands that can execute code, write files, or access
            the network.  These require human confirmation.
    """
    if executable in SAFE_EXECUTABLES:
        return "safe"
    return "sensitive"


def run_command(
    command: str,
    *,
    confirm_callback: Callable[[str, str], bool] | None = None,
) -> str:
    """Run a shell command inside the sandbox using an executable allowlist.

    Security layers (in order of evaluation):
      Layer 1 — shell=False: no ``;``, ``|``, ``$()`` interpretation
      Layer 2 — shell metacharacter rejection: explicit error on ``|;&$`><``
      Layer 3 — executable allowlist: only known-safe basenames
      Layer 4 — dangerous argument check: block ``python -c/-m/-i``
      Layer 5 — defense-in-depth: block destructive/network/privilege/shell cmds
      Layer 6 — permission confirmation: human approval for sensitive executables
      Layer 7 — cwd locked to SANDBOX_ROOT + timeout + output cap

    Returns a unified ToolResult envelope (JSON string).  Errors are
    *actionable* — the ``error.hint`` field tells the model what to try next
    instead of just reporting what went wrong.
    """
    # --- Layer 1-2: input validation ---------------------------------------
    if not command or not command.strip():
        return _make_error(ErrorType.INVALID_INPUT, "Command must be a non-empty string")

    command = command.strip()
    if any(ch in command for ch in UNSUPPORTED_SHELL_CHARS):
        return _make_error(
            ErrorType.SHELL_METACHARACTER,
            "Shell metacharacters not supported — run_command does not invoke "
            "a shell. Use one allowlisted executable and its arguments only.",
        )

    try:
        argv = shlex.split(command)
    except ValueError as e:
        return _make_error(
            ErrorType.INVALID_INPUT, f"Invalid command syntax: {e}"
        )

    if not argv:
        return _make_error(ErrorType.INVALID_INPUT, "Command must be a non-empty string")

    executable = Path(argv[0]).name

    # --- Layer 3: executable allowlist --------------------------------------
    if executable not in ALLOWED_EXECUTABLES:
        return _make_error(
            ErrorType.EXECUTABLE_NOT_ALLOWED,
            f"Executable {executable!r} not in allowlist. "
            f"Allowed: {sorted(ALLOWED_EXECUTABLES)}",
        )

    # --- Layer 4: dangerous argument check ----------------------------------
    try:
        _check_dangerous_args(argv)
    except ValueError as e:
        return _make_error(ErrorType.DANGEROUS_ARGS, str(e))

    # --- Layer 5: defense-in-depth ------------------------------------------
    try:
        _check_defense_in_depth(argv)
    except ValueError as e:
        return _make_error(ErrorType.DANGEROUS_COMMAND, str(e))

    # --- Layer 6: permission confirmation -----------------------------------
    if confirm_callback is not None:
        risk = _classify_executable(executable)
        if risk == "sensitive":
            allowed = confirm_callback(executable, command)
            if not allowed:
                return _make_error(
                    ErrorType.PERMISSION_DENIED,
                    f"Execution of {executable!r} denied by user. "
                    f"Command: {command!r}",
                )

    # --- Layer 7: execute (cwd locked, no shell, timeout, output cap) --------
    sandbox = SANDBOX_ROOT.resolve()
    sandbox.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            argv,
            cwd=sandbox,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            shell=False,
        )
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or "") + (e.stderr or "")
        stdout, was_truncated = _truncate(partial, TOOL_OUTPUT_CAPS["run_command_stdout"])
        return _make_success(
            {
                "command": command,
                "exit_code": None,
                "timed_out": True,
                "stdout": stdout,
                "stderr": "",
            },
            truncated=was_truncated,
            truncated_at=TOOL_OUTPUT_CAPS["run_command_stdout"] if was_truncated else None,
        )
    except FileNotFoundError:
        return _make_error(
            ErrorType.COMMAND_ERROR,
            f"Executable not found on PATH: {executable!r}",
            hint=f"Allowed executables: {', '.join(sorted(ALLOWED_EXECUTABLES))}. "
            f"Use 'which {executable}' or try a different executable.",
        )

    stdout, out_trunc = _truncate(
        result.stdout or "", TOOL_OUTPUT_CAPS["run_command_stdout"]
    )
    stderr, err_trunc = _truncate(
        result.stderr or "", TOOL_OUTPUT_CAPS["run_command_stderr"]
    )

    return _make_success(
        {
            "command": command,
            "exit_code": result.returncode,
            "timed_out": False,
            "stdout": stdout,
            "stderr": stderr,
        },
        truncated=out_trunc or err_trunc,
        truncated_at=TOOL_OUTPUT_CAPS["run_command_stdout"] if (out_trunc or err_trunc) else None,
    )


# ---------------------------------------------------------------------------
# Day 3: Web tools — web_fetch + web_search
# ---------------------------------------------------------------------------

# --- Private-IP detection (defense-in-depth for web_fetch) ---------------

# Networks that must never be fetched, even if DNS resolves to them.
# Covers IPv4 private ranges + loopback + link-local.
_PRIVATE_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    # IPv4
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("0.0.0.0/8"),
    ipaddress.IPv4Network("224.0.0.0/4"),  # multicast
    ipaddress.IPv4Network("240.0.0.0/4"),  # reserved
    # IPv6
    ipaddress.IPv6Network("::1/128"),       # loopback
    ipaddress.IPv6Network("fe80::/10"),     # link-local
    ipaddress.IPv6Network("fc00::/7"),      # unique local
)


def _is_private_host(hostname: str) -> bool:
    """Return True if *hostname* resolves to a private / internal IP address.

    Design note — why resolve here instead of inspecting the raw IP in the URL?
      The URL may contain a hostname, not an IP.  The attack vector is a
      hostname whose DNS record points at a private address (e.g. an internal
      service).  Resolving is the only way to catch this case.
    """
    try:
        for info in socket.getaddrinfo(hostname, None):
            try:
                ip = ipaddress.ip_address(info[4][0])
            except ValueError:
                continue  # non-IP address family — skip
            for net in _PRIVATE_NETWORKS:
                if ip in net:
                    return True
    except socket.gaierror:
        # Can't resolve — err on the side of safety and block.
        return True
    return False


# --- web_fetch ------------------------------------------------------------

def _html_to_text(html_bytes: bytes) -> str:
    """Extract readable plain text from HTML, stripping scripts/styles."""
    soup = BeautifulSoup(html_bytes, "html.parser")

    # Remove non-content elements before text extraction.
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Collapse runs of 3+ blank lines into 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def web_fetch(url: str, max_chars: int = WEB_FETCH_DEFAULT_MAX_CHARS) -> str:
    """Fetch a URL and return extracted plain text.

    Security layers:
      1. Scheme must be http or https — no file://, ftp://, etc.
      2. Hostname must not resolve to a private/internal IP (defense-in-depth).
      3. Response body is capped at ~5 MB before processing.
      4. Non-HTML content-types are returned as-is (plain text).
      5. Extracted text is truncated to *max_chars* with a consistent marker.

    Returns a unified ToolResult envelope (JSON string).
    """
    # --- input validation ---------------------------------------------------
    if not url or not url.strip():
        return _make_error(ErrorType.INVALID_INPUT, "URL must be a non-empty string")
    url = url.strip()

    # Layer 1 — scheme check.
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return _make_error(
            ErrorType.INVALID_URL_SCHEME,
            f"Only http/https URLs are allowed, got scheme {parsed.scheme!r}. "
            f"URL: {url!r}",
        )
    if not parsed.hostname:
        return _make_error(
            ErrorType.INVALID_INPUT, f"URL has no hostname: {url!r}"
        )

    # Layer 2 — block private/internal addresses.
    if _is_private_host(parsed.hostname):
        return _make_error(
            ErrorType.PRIVATE_ADDRESS,
            f"URL resolves to a private/internal address (blocked): {url!r}",
        )

    # Layer 3 — fetch with timeout and body-size cap.
    try:
        resp = requests.get(
            url,
            timeout=WEB_FETCH_TIMEOUT,
            stream=True,
            headers={"User-Agent": "MyAgent/1.0 (learning project)"},
        )
        resp.raise_for_status()
    except requests.Timeout:
        return _make_error(
            ErrorType.COMMAND_TIMEOUT,
            f"Request timed out after {WEB_FETCH_TIMEOUT}s: {url!r}",
        )
    except requests.RequestException as e:
        return _make_error(
            ErrorType.REQUEST_FAILED,
            f"Request failed: {e}",
        )

    # Read up to the hard byte limit (streaming — doesn't buffer the whole body).
    raw_body = resp.raw.read(WEB_FETCH_MAX_RESPONSE_BYTES + 1, decode_content=True)
    body_truncated = len(raw_body) > WEB_FETCH_MAX_RESPONSE_BYTES
    if body_truncated:
        raw_body = raw_body[:WEB_FETCH_MAX_RESPONSE_BYTES]

    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()

    # Layer 4 — text extraction (HTML) or passthrough (plain text).
    if content_type in _HTML_CONTENT_TYPES or content_type == "":
        text = _html_to_text(raw_body)
        soup_title = BeautifulSoup(raw_body, "html.parser").title
        title = soup_title.string.strip() if soup_title and soup_title.string else "(no title)"
    else:
        # Non-HTML: return raw text as-is.
        text = raw_body.decode("utf-8", errors="replace")
        title = "(non-HTML content)"

    # Layer 5 — length cap (uses the shared _truncate for consistent marker).
    capped_text, was_truncated = _truncate(text, max_chars)

    note = ""
    if body_truncated:
        note = (
            f"Response body was truncated at {WEB_FETCH_MAX_RESPONSE_BYTES} "
            f"bytes before text extraction."
        )

    return _make_success(
        {
            "url": resp.url,  # final URL after redirects
            "title": title,
            "text": capped_text,
            "chars": len(capped_text),
            "content_type": content_type,
            "status_code": resp.status_code,
            "note": note or None,
        },
        truncated=was_truncated,
        truncated_at=max_chars if was_truncated else None,
    )


# --- web_search -----------------------------------------------------------

def web_search(query: str, max_results: int = WEB_SEARCH_DEFAULT_MAX_RESULTS) -> str:
    """Search the web via DuckDuckGo and return results.

    Returns a unified ToolResult envelope (JSON string).
    """
    if not query or not query.strip():
        return _make_error(
            ErrorType.INVALID_INPUT, "Search query must be a non-empty string"
        )
    query = query.strip()

    max_results = max(1, min(max_results, WEB_SEARCH_HARD_MAX_RESULTS))

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return _make_error(
            ErrorType.SEARCH_FAILED,
            f"Search failed: {e}",
        )

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in raw
    ]

    return _make_success(
        {
            "query": query,
            "results": results,
            "count": len(results),
        }
    )


# ---------------------------------------------------------------------------
# Tool schemas for the LLM
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file from the sandbox workspace. Call when the user "
                "asks to see or summarize a file. Path must be relative to the "
                "sandbox root (no leading '/' or '..')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Relative path, e.g. 'notes.txt' or 'subdir/data.json'.",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write or overwrite a text file in the sandbox workspace. Call when "
                "the user asks to save, create, or update a file. Path must stay "
                "inside the sandbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Relative path inside sandbox."},
                    "content": {"type": "string", "description": "Full file contents to write."},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List files and subdirectories in the sandbox. Call when the user "
                "wants to browse or discover what files exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dirpath": {
                        "type": "string",
                        "description": "Relative directory path; default '.' for sandbox root.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a read-only shell command inside the sandbox workspace. "
                "Only allowlisted executables are permitted (ls, cat, echo, grep, "
                "wc, head, tail, python/python3). No shell — pipes and semicolons "
                "are not supported. Use for listing files, grepping content, or "
                "running short python one-liners."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command argv string, e.g. 'ls -la' or 'grep sandbox notes.txt'.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a web page by URL and return the page title plus extracted "
                "plain text. Only http/https URLs are allowed — internal/private "
                "addresses are blocked. The response is capped at 5000 chars by "
                "default. Call this when the user asks to read, fetch, visit, or "
                "summarize a URL. Also call this instead of curl or wget (which "
                "are blocked by the command allowlist)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL starting with http:// or https://.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max chars of extracted text (default 5000). Use a smaller value for quick summaries and a larger one for detailed reading.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web via DuckDuckGo and return a list of results with "
                "titles, URLs, and short text snippets. Call this when the user "
                "asks to search for something online, find information, look up "
                "facts, or research current events. Combine with web_fetch to "
                "open the most promising result for more detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query — keep it concise and keyword-focused.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1–10, default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "read_file": lambda inp: read_file(inp["filepath"]),
    "write_file": lambda inp: write_file(inp["filepath"], inp["content"]),
    "list_dir": lambda inp: list_dir(inp.get("dirpath", ".")),
    "run_command": lambda inp: run_command(inp["command"]),
    "web_fetch": lambda inp: web_fetch(inp["url"], inp.get("max_chars", WEB_FETCH_DEFAULT_MAX_CHARS)),
    "web_search": lambda inp: web_search(inp["query"], inp.get("max_results", WEB_SEARCH_DEFAULT_MAX_RESULTS)),
}


# ---------------------------------------------------------------------------
# Escape attempts — try to break your own sandbox
# ---------------------------------------------------------------------------
def run_escape_tests() -> None:
    """Run path-traversal and symlink attacks against the sandbox resolver."""
    sandbox = SANDBOX_ROOT.resolve()
    sandbox.mkdir(parents=True, exist_ok=True)

    # Symlink inside sandbox pointing outside — classic bypass if you skip realpath.
    outside_target = (SANDBOX_ROOT.parent / "outside_secret.txt").resolve()
    outside_target.write_text("TOP SECRET — outside the sandbox", encoding="utf-8")
    symlink_path = sandbox / "escape_link.txt"
    if symlink_path.is_symlink() or symlink_path.exists():
        symlink_path.unlink()
    symlink_path.symlink_to(outside_target)

    attacks: list[tuple[str, str, str]] = [
        # (label, tool, path_or_args)
        ("parent traversal", "read", "../agent.py"),
        ("double-dot hidden by normpath", "read", "subdir/../../agent.py"),
        ("absolute path", "read", "/etc/passwd"),
        ("absolute under sandbox prefix", "read", str(sandbox / "notes.txt")),
        ("null byte (legacy)", "read", "notes.txt\x00/../../etc/passwd"),
        ("symlink to outside file", "read", "escape_link.txt"),
        ("write outside via traversal", "write", "../../pwned.txt"),
        ("list parent directory", "list", ".."),
    ]

    print(f"Sandbox root: {sandbox}\n")
    print(f"{'Attack':<32} {'Expected':<10} {'Result'}")
    print("-" * 72)

    blocked = 0
    for label, tool, arg in attacks:
        try:
            if tool == "read":
                raw = read_file(arg)
            elif tool == "write":
                raw = write_file(arg, "pwned")
            else:
                raw = list_dir(arg)
            envelope = json.loads(raw)
            if envelope.get("status") == "error":
                err = envelope.get("error", {})
                outcome = f"BLOCKED ({err.get('type', 'unknown')})"
                blocked += 1
            else:
                outcome = "ALLOWED"
        except Exception as e:
            outcome = f"BLOCKED ({type(e).__name__})"
            blocked += 1

        status = "OK" if outcome.startswith("BLOCKED") else "FAIL"
        print(f"{label:<32} {status:<10} {outcome}")

    # Legitimate access should work.
    print("\nLegitimate access:")
    for label, fn in [
        ("read notes.txt", lambda: read_file("notes.txt")),
        ("list .", lambda: list_dir(".")),
        ("write sandbox/tmp.txt", lambda: write_file("tmp.txt", "hello")),
    ]:
        try:
            raw = fn()
            envelope = json.loads(raw)
            if envelope.get("status") == "success":
                data = envelope.get("data", {})
                # Show a short preview of the data
                preview = json.dumps(data)
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                print(f"  {label}: OK — {preview}")
            else:
                err = envelope.get("error", {})
                print(f"  {label}: FAIL — error: {err.get('type')}: {err.get('message')}")
        except Exception as e:
            print(f"  {label}: FAIL — {e}")

    print(f"\n{blocked}/{len(attacks)} escape attempts blocked.")

    # Cleanup
    symlink_path.unlink(missing_ok=True)
    outside_target.unlink(missing_ok=True)
    (sandbox / "tmp.txt").unlink(missing_ok=True)


def run_command_tests() -> None:
    """Run injection and policy attacks against run_command."""
    sandbox = SANDBOX_ROOT.resolve()
    sandbox.mkdir(parents=True, exist_ok=True)

    attacks: list[tuple[str, str, str]] = [
        ("rm not allowlisted", "BLOCKED", "rm notes.txt"),
        ("curl not allowlisted", "BLOCKED", "curl https://example.com"),
        ("env not allowlisted", "BLOCKED", "env"),
        ("shell via sh -c", "BLOCKED", 'sh -c "cat notes.txt"'),
        ("absolute path to forbidden binary", "BLOCKED", "/bin/rm notes.txt"),
        ("semicolon injection (no shell)", "BLOCKED", "ls; rm notes.txt"),
        ("pipe injection (no shell)", "BLOCKED", "cat notes.txt | wc -l"),
        # Dangerous argument patterns (Layer 4)
        ("python -c (inline code)", "BLOCKED", 'python -c "print(1)"'),
        ("python3 -c (inline code)", "BLOCKED", 'python3 -c "import os"'),
        ("python -m (module exec)", "BLOCKED", "python -m http.server"),
        ("python3 -m (module exec)", "BLOCKED", "python3 -m venv .env"),
        ("python -i (interactive)", "BLOCKED", "python -i"),
        # Redirection metacharacters (Layer 2 — defense-in-depth)
        ("stdout redirect >", "BLOCKED", "echo hello > out.txt"),
        ("stdin redirect <", "BLOCKED", "cat < notes.txt"),
        # Unicode homoglyph (not in allowlist basename)
        ("unicode homoglyph ls", "BLOCKED", "ｌｓ -la"),
        # Defense-in-depth (Layer 5 — redundant with allowlist)
        ("defense-in-depth: rm", "BLOCKED", "rm notes.txt"),
        ("defense-in-depth: sudo", "BLOCKED", "sudo ls"),
        ("defense-in-depth: bash", "BLOCKED", "bash -c 'ls'"),
        ("defense-in-depth: curl", "BLOCKED", "curl https://example.com"),
    ]

    print(f"Sandbox cwd: {sandbox}")
    print(f"Allowlist: {sorted(ALLOWED_EXECUTABLES)}")
    print(f"Timeout: {COMMAND_TIMEOUT_SECONDS}s | Output cap: {COMMAND_OUTPUT_CAP} chars\n")
    print(f"{'Attack':<36} {'Expected':<10} {'Result'}")
    print("-" * 80)

    blocked = 0
    for label, expected, cmd in attacks:
        raw = run_command(cmd)
        envelope = json.loads(raw)
        if envelope.get("status") == "error":
            err = envelope.get("error", {})
            outcome = f"BLOCKED ({err.get('type', 'unknown')}: {err.get('message', '')})"
            blocked += 1
        else:
            outcome = "ALLOWED"

        status = "OK" if outcome.startswith(expected) else "FAIL"
        print(f"{label:<36} {status:<10} {outcome[:120]}")

    print("\nPolicy limits:")
    # Use sandbox scripts instead of python -c (which is blocked by Layer 4).
    _sleep_script = sandbox / "_test_sleep.py"
    _sleep_script.write_text(
        f"import time\ntime.sleep({COMMAND_TIMEOUT_SECONDS + 3})\nprint('done')"
    )
    _output_script = sandbox / "_test_big_output.py"
    _output_script.write_text(
        f"print('x' * {COMMAND_OUTPUT_CAP + 500})"
    )
    limits: list[tuple[str, str, str]] = [
        (
            "timeout",
            "TIMEOUT",
            f"{_python_bin()} _test_sleep.py",
        ),
        (
            "output cap",
            "TRUNCATED",
            f"{_python_bin()} _test_big_output.py",
        ),
    ]
    for label, expected, cmd in limits:
        raw = run_command(cmd)
        envelope = json.loads(raw)
        data = envelope.get("data", {})
        if expected == "TIMEOUT":
            ok = data.get("timed_out") is True
            detail = "timed_out=True" if ok else f"unexpected: {raw[:80]}"
        else:
            ok = envelope.get("truncated") is True
            detail = (
                f"truncated=True, truncated_at={envelope.get('truncated_at')}"
                if ok
                else raw[:80]
            )
        print(f"  {label}: {'OK' if ok else 'FAIL'} — {detail}")

    print("\nLegitimate commands:")
    for label, cmd in [
        ("ls", "ls"),
        ("cat notes.txt", "cat notes.txt"),
        ("grep", "grep sandbox notes.txt"),
        ("echo", "echo hello"),
    ]:
        raw = run_command(cmd)
        envelope = json.loads(raw)
        data = envelope.get("data", {})
        preview = (data.get("stdout") or data.get("stderr") or "")[:60]
        print(f"  {label}: OK (exit={data.get('exit_code', '?')}) — {preview!r}")

    print(f"\n{blocked}/{len(attacks)} forbidden commands blocked.")

    # ------------------------------------------------------------------
    # Injection resilience — these should run safely, not raise errors
    # ------------------------------------------------------------------
    print("\nInjection resilience (shlex handles these correctly):")
    injection_tests: list[tuple[str, str]] = [
        (
            "newline as whitespace — stays in argv",
            "ls\ncat notes.txt",
        ),
        (
            "tab as separator — stays in argv",
            "ls\t-la",
        ),
        (
            "single quotes consumed by shlex",
            "grep -E 'sandbox' notes.txt",
        ),
        (
            "backslash escape consumed by shlex",
            "l\\s",
        ),
    ]
    for label, cmd in injection_tests:
        raw = run_command(cmd)
        envelope = json.loads(raw)
        if envelope.get("status") == "success":
            data = envelope.get("data", {})
            detail = f"exit={data.get('exit_code', '?')} (safe — inject chars consumed by shlex)"
        else:
            err = envelope.get("error", {})
            detail = f"BLOCKED (unexpected): [{err.get('type')}] {err.get('message')}"
        print(f"  {label}: {detail}")

    # ------------------------------------------------------------------
    # Permission confirmation callback
    # ------------------------------------------------------------------
    print("\nPermission confirmation callback:")
    _callback_record: list[tuple[str, str]] = []

    def _test_cb(exe: str, cmd: str) -> bool:
        _callback_record.append((exe, cmd))
        return False  # deny all for testing

    for label, expected_calls, cmd in [
        ("python → callback invoked (sensitive)", 1, "python --version"),
        ("echo → callback invoked (sensitive)", 1, "echo hello"),
        ("ls → no callback (safe)", 0, "ls"),
        ("cat → no callback (safe)", 0, "cat notes.txt"),
    ]:
        _callback_record.clear()
        raw = run_command(cmd, confirm_callback=_test_cb)
        envelope = json.loads(raw)
        # When callback denies, status is error with PERMISSION_DENIED type
        if envelope.get("status") == "error":
            err = envelope.get("error", {})
            if err.get("type") != ErrorType.PERMISSION_DENIED:
                pass  # some other error — still count callback calls
        ok = len(_callback_record) == expected_calls
        extra = f"(denied by callback)" if len(_callback_record) > 0 and not ok else ""
        detail = f"callback called {len(_callback_record)} time(s) {extra}"
        print(f"  {label}: {'OK' if ok else 'FAIL'} — {detail}")

    # Cleanup test scripts
    _sleep_script.unlink(missing_ok=True)
    _output_script.unlink(missing_ok=True)


def run_web_tests() -> None:
    """Run security and correctness tests for web_fetch and web_search."""
    print("=" * 72)
    print("  web_fetch tests")
    print("=" * 72)

    # --- Valid fetch ----------------------------------------------------
    print("\n-- Valid HTTP fetch --")
    try:
        raw = web_fetch("https://example.com", max_chars=600)
        envelope = json.loads(raw)
        assert envelope["status"] == "success", f"expected success: {envelope}"
        data = envelope["data"]
        assert data["url"] == "https://example.com/", f"unexpected final url: {data['url']}"
        assert data["title"] == "Example Domain", f"unexpected title: {data['title']!r}"
        assert "Example Domain" in data["text"], "body text missing"
        assert not envelope["truncated"], "should not be truncated at 600 chars"
        assert data["status_code"] == 200
        print(f"  OK — title={data['title']!r}, chars={data['chars']}, truncated={envelope['truncated']}")
    except Exception as e:
        print(f"  FAIL — {e}")

    # --- Length cap -----------------------------------------------------
    print("\n-- Length cap --")
    try:
        raw = web_fetch("https://example.com", max_chars=50)
        envelope = json.loads(raw)
        assert envelope["status"] == "success", f"expected success: {envelope}"
        assert envelope["truncated"], "should be truncated at 50 chars"
        data = envelope["data"]
        assert data["chars"] <= 120, f"chars should be near 50: got {data['chars']}"
        print(f"  OK — truncated={envelope['truncated']}, truncated_at={envelope['truncated_at']}, chars={data['chars']}")
    except Exception as e:
        print(f"  FAIL — {e}")

    # --- Invalid schemes ------------------------------------------------
    print("\n-- Scheme checks --")
    for label, url in [
        ("file scheme", "file:///etc/passwd"),
        ("ftp scheme", "ftp://example.com/file"),
        ("no scheme", "example.com"),
    ]:
        raw = web_fetch(url)
        envelope = json.loads(raw)
        if envelope.get("status") == "error":
            err = envelope.get("error", {})
            print(f"  OK — {label}: blocked ([{err.get('type')}] {err.get('message')})")
        else:
            print(f"  FAIL — {label}: {url!r} should have been blocked")

    # --- Private / internal IPs -----------------------------------------
    print("\n-- Private IP blocking --")
    for label, url in [
        ("localhost", "http://localhost:8080/admin"),
        ("127.0.0.1", "http://127.0.0.1/secret"),
        ("192.168.x.x", "http://192.168.1.1/"),
        ("10.x.x.x", "http://10.0.0.1/api"),
        ("[::1]", "http://[::1]:3000/"),
    ]:
        raw = web_fetch(url)
        envelope = json.loads(raw)
        if envelope.get("status") == "error":
            err = envelope.get("error", {})
            print(f"  OK — {label}: blocked ([{err.get('type')}] {err.get('message')})")
        else:
            print(f"  FAIL — {label}: {url!r} should have been blocked")

    # --- Timeout --------------------------------------------------------
    print("\n-- Timeout (fast) --")
    raw = web_fetch("http://192.0.2.1/test", max_chars=100)
    envelope = json.loads(raw)
    if envelope.get("status") == "error":
        err = envelope.get("error", {})
        err_type = err.get("type", "")
        if err_type in (ErrorType.COMMAND_TIMEOUT, ErrorType.REQUEST_FAILED):
            print(f"  OK — caught [{err_type}]: {err.get('message')}")
        else:
            print(f"  WARN — unexpected error type: {err_type}")
    else:
        print("  WARN — request completed (unexpected but network-dependent)")

    # --- 404 error ------------------------------------------------------
    print("\n-- HTTP error (non-200) --")
    raw = web_fetch("https://httpbin.org/status/404")
    envelope = json.loads(raw)
    if envelope.get("status") == "error":
        err = envelope.get("error", {})
        if "404" in err.get("message", ""):
            print(f"  OK — caught 404: [{err.get('type')}] {err.get('message')}")
        else:
            print(f"  WARN — {err.get('message')}")
    else:
        print("  FAIL — should have returned an error for 404")

    # --- Empty URL ------------------------------------------------------
    print("\n-- Empty URL --")
    for label, url in [("empty string", ""), ("whitespace only", "   ")]:
        raw = web_fetch(url)
        envelope = json.loads(raw)
        if envelope.get("status") == "error":
            err = envelope.get("error", {})
            print(f"  OK — {label}: [{err.get('type')}] {err.get('message')}")
        else:
            print(f"  FAIL — {label} should be rejected")

    # =========================================================================
    print("\n" + "=" * 72)
    print("  web_search tests")
    print("=" * 72)

    # --- Valid search ---------------------------------------------------
    print("\n-- Valid search --")
    try:
        raw = web_search("Python programming language", max_results=3)
        envelope = json.loads(raw)
        assert envelope["status"] == "success", f"expected success: {envelope}"
        data = envelope["data"]
        assert data["query"] == "Python programming language"
        assert 1 <= data["count"] <= 3, f"expected 1–3 results: got {data['count']}"
        for r in data["results"]:
            assert r["title"], f"result missing title: {r}"
            assert r["url"], f"result missing url: {r}"
            assert r["snippet"], f"result missing snippet: {r}"
        print(f"  OK — {data['count']} results")
        for i, r in enumerate(data["results"], 1):
            print(f"    {i}. {r['title'][:80]}")
            print(f"       {r['url'][:80]}")
    except Exception as e:
        print(f"  FAIL — {e}")

    # --- Max results clamped --------------------------------------------
    print("\n-- Max results clamped --")
    try:
        raw = web_search("test", max_results=999)
        envelope = json.loads(raw)
        assert envelope["status"] == "success", f"expected success: {envelope}"
        data = envelope["data"]
        assert data["count"] <= WEB_SEARCH_HARD_MAX_RESULTS, (
            f"should be capped at {WEB_SEARCH_HARD_MAX_RESULTS}: got {data['count']}"
        )
        print(f"  OK — capped at {data['count']} results")
    except Exception as e:
        print(f"  FAIL — {e}")

    # --- Empty query ----------------------------------------------------
    print("\n-- Empty query --")
    for label, q in [("empty string", ""), ("whitespace only", "   ")]:
        raw = web_search(q)
        envelope = json.loads(raw)
        if envelope.get("status") == "error":
            err = envelope.get("error", {})
            print(f"  OK — {label}: [{err.get('type')}] {err.get('message')}")
        else:
            print(f"  FAIL — {label} should be rejected")


def _python_bin() -> str:
    """Pick an allowlisted python executable available on this machine."""
    for name in ("python3", "python"):
        if name in ALLOWED_EXECUTABLES:
            return name
    return "python3"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def _confirm_command(executable: str, command: str) -> bool:
    """Prompt the user to approve or deny a sensitive command at runtime."""
    print()
    print("=" * 60)
    print("  ⚠️  Permission Required — Sensitive Command")
    print("=" * 60)
    print(f"  Executable: {executable!r}")
    print(f"  Full command: {command}")
    print(f"  Sandbox root: {SANDBOX_ROOT.resolve()}")
    answer: str = input("  Allow execution? (y/N): ").strip().lower()
    print("=" * 60)
    return answer == "y"


def _trace_preview(tool_name: str, data: dict) -> str:
    """Build a short human-readable preview of a tool result for trace output."""
    if tool_name == "read_file":
        content = data.get("content", "")
        preview = content[:100].replace("\n", " ")
        return f"read {data.get('filepath', '?')} ({data.get('bytes', 0)} bytes): {preview}…"
    elif tool_name == "write_file":
        return f"wrote {data.get('filepath', '?')} ({data.get('bytes', 0)} bytes)"
    elif tool_name == "list_dir":
        entries = data.get("entries", [])
        names = [e["name"] for e in entries[:8]]
        suffix = f"… (+{len(entries) - 8} more)" if len(entries) > 8 else ""
        return f"{data.get('directory', '?')}: {len(entries)} entries — {', '.join(names)}{suffix}"
    elif tool_name == "run_command":
        exit_code = data.get("exit_code", "?")
        timed_out = data.get("timed_out", False)
        if timed_out:
            return f"TIMED OUT — {data.get('command', '?')}"
        stdout = (data.get("stdout") or "").replace("\n", " ")[:100]
        return f"exit={exit_code} — {data.get('command', '?')} → {stdout}"
    elif tool_name == "web_fetch":
        truncated = data.get("chars", 0)
        return (
            f"fetched {data.get('url', '?')} — "
            f"title={data.get('title', '?')!r}, "
            f"{truncated} chars, status={data.get('status_code', '?')}"
        )
    elif tool_name == "web_search":
        results = data.get("results", [])
        titles = [r.get("title", "?")[:60] for r in results[:3]]
        return f"{data.get('count', 0)} results for '{data.get('query', '?')}': {titles}"
    return json.dumps(data)[:200]


def run_agent(user_input: str) -> str:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with sandboxed file, command, and web tools. "
                f"All file paths and shell commands run inside {SANDBOX_ROOT}. "
                "File tools cannot access paths outside the sandbox. "
                f"run_command only permits: {', '.join(sorted(ALLOWED_EXECUTABLES))} "
                "(no shell — pipes/semicolons unsupported). "
                "web_fetch can retrieve any public http/https URL and return "
                "extracted plain text (HTML is stripped automatically). "
                "web_search queries DuckDuckGo and returns titles + URLs + snippets. "
                "Combine web_search + web_fetch: search first, then open the most "
                "promising result for more detail.\n\n"
                "IMPORTANT — All tool results use a consistent JSON envelope:\n"
                '  {"status":"success"|"error", "data":{…}, "error":null|{…}, '
                '"truncated":bool, "truncated_at":int|null}\n'
                "- On success: check .data for the tool-specific payload.\n"
                "- On error: read .error.type for the error category and "
                ".error.hint for an actionable suggestion on what to try next. "
                "ALWAYS read the hint before retrying — it tells you how to fix "
                "the problem.\n"
                "- If .truncated is true, the output was capped at .truncated_at "
                "chars. The truncation marker is always: "
                '"… [truncated at N chars]". Consider narrowing your request '
                "or reading a smaller portion."
            ),
        },
        {"role": "user", "content": user_input},
    ]

    step = 0
    turn = 0
    MAX_TURNS = 15  # safety valve — prevent runaway agent loops
    while True:
        turn += 1
        if turn > MAX_TURNS:
            return f"[Agent stopped after {MAX_TURNS} LLM turns — possible loop.]"

        response = client.chat.completions.create(model=MODEL, tools=TOOLS, messages=messages)
        choice = response.choices[0]
        msg = choice.message
        messages.append(msg)

        if choice.finish_reason == "length":
            return "[Response was truncated — raise max_tokens.]"

        if not msg.tool_calls:
            return msg.content or "[no text in response]"

        print(f"  ── LLM turn {turn} ({len(msg.tool_calls)} tool call(s)) ──")
        for call in msg.tool_calls:
            step += 1
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError as e:
                output = _make_error(
                    ErrorType.INVALID_INPUT,
                    f"Invalid JSON arguments: {e}",
                )
                print(f"  [{step}] ❌ {call.function.name} — bad JSON: {e}")
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output}
                )
                continue

            # Summarise arguments for display
            arg_summary = call.function.arguments
            if call.function.name == "web_fetch" and len(arg_summary) > 120:
                arg_summary = arg_summary[:117] + "..."
            if call.function.name == "write_file" and len(arg_summary) > 150:
                arg_summary = arg_summary[:147] + "..."

            print(f"  [{step}] 🔧 {call.function.name}({arg_summary})")
            try:
                if call.function.name == "run_command":
                    output = run_command(
                        args["command"],
                        confirm_callback=_confirm_command,
                    )
                else:
                    output = TOOL_FUNCTIONS[call.function.name](args)

                # --- Parse the unified envelope for trace display --------------
                try:
                    envelope = json.loads(output)
                    status = envelope.get("status", "unknown")
                    truncated = envelope.get("truncated", False)
                    trunc_at = envelope.get("truncated_at")

                    if status == "success":
                        data = envelope.get("data", {})
                        # Pick a meaningful preview from the data payload
                        preview = _trace_preview(call.function.name, data)
                        icon = "✅"
                        if truncated:
                            icon = "✅📎"  # paperclip = truncated
                        print(f"  [{step}] {icon} {preview}")
                        if truncated and trunc_at:
                            print(f"         ⚠️  truncated at {trunc_at} chars")
                    else:
                        err = envelope.get("error", {})
                        err_type = err.get("type", "unknown")
                        err_msg = err.get("message", str(err))
                        err_hint = err.get("hint", "")
                        print(f"  [{step}] ❌ [{err_type}] {err_msg}")
                        if err_hint:
                            print(f"         💡 hint: {err_hint}")
                except (json.JSONDecodeError, TypeError):
                    # Fallback: raw output preview
                    preview = output[:250]
                    if len(output) > 250:
                        preview += f"… [{len(output)} total chars]"
                    print(f"  [{step}] ✅ result: {preview}")
            except Exception as e:
                output = _make_error(ErrorType.UNKNOWN, str(e))
                print(f"  [{step}] ❌ unhandled exception: {e}")

            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": output}
            )


if __name__ == "__main__":
    if "--test-sandbox" in sys.argv:
        run_escape_tests()
        sys.exit(0)

    if "--test-commands" in sys.argv:
        run_command_tests()
        sys.exit(0)

    if "--test-web" in sys.argv:
        run_web_tests()
        sys.exit(0)

    # Day 1 demo — path traversal:
    day1_question = "Try to read ../PLAN.md using read_file. If blocked, explain why."

    # Day 2 demo — forbidden command; watch how the agent copes:
    day2_question = (
        "Use run_command to print all environment variables (run `env`). "
        "If that fails, try curl to fetch https://example.com instead. "
        "Report what worked and what the tool errors said."
    )

    # Day 3 demo — web tools: search for something and fetch a page.
    day3_question = (
        "Use web_search to find the official Python website. Then use web_fetch "
        "to open the top result and tell me: what is the page title, and what "
        "is the first sentence or tagline you see on the page? Save a summary "
        "of what you found to 'python_website_notes.md' in the sandbox."
    )

    # Day 4 demo — Combine ALL tools. Give the agent a real multi-step task
    # that requires chaining web_search → web_fetch → write_file → read_file
    # → run_command.  The prompt must be sufficiently constrained to force
    # multi-step behaviour, but open enough that the agent must decide the
    # exact sequence.  Key idea from "Building Effective Agents":
    #   "Agents succeed when given clear, bounded tasks with the tools to
    #    accomplish them — not when asked to be omniscient."
    day4_question = (
        "You have access to web_search, web_fetch, write_file, read_file, "
        "list_dir, and run_command tools. Complete this multi-step task:\n\n"
        "1. Use web_search to look up \"Anthropic Building Effective Agents\" "
        "and find the official blog post or documentation.\n"
        "2. Use web_fetch to open the most relevant result and extract the key "
        "points — focus on the 5 core principles or patterns they recommend.\n"
        "3. Use write_file to save a well-formatted summary (with a title, the "
        "5 principles, and a short takeaway) to 'notes.md' in the sandbox.\n"
        "4. Use read_file to verify 'notes.md' was written correctly.\n"
        "5. Use list_dir to confirm the file appears in the sandbox.\n"
        "6. Use run_command with 'cat notes.md' to double-check the content.\n\n"
        "Important: Do EVERY step in order. Do not skip any step. After each "
        "tool call, report what you learned before proceeding to the next step."
    )

    if "--day4" in sys.argv:
        question = day4_question
    elif "--day3" in sys.argv:
        question = day3_question
    elif "--day2" in sys.argv:
        question = day2_question
    elif "--day1" in sys.argv:
        question = day1_question
    else:
        question = day4_question

    # Day 4 trace header — make the full trace visible
    if "--day4" in sys.argv or ("--day1" not in sys.argv and "--day2" not in sys.argv and "--day3" not in sys.argv):
        print("=" * 72)
        print("  WEEK 3 DAY 4 — All Tools Combined: Multi-Step Agent Trace")
        print("=" * 72)
        print(f"  Sandbox: {SANDBOX_ROOT.resolve()}")
        print(f"  Model:   {MODEL}")
        print(f"  Tools:   read_file, write_file, list_dir, run_command,")
        print(f"           web_fetch, web_search")
        print("=" * 72)
        print()

    print(f"User: {question}\n")
    print("-" * 72)
    print("  AGENT TRACE — tool calls and results")
    print("-" * 72)
    answer = run_agent(question)
    print("-" * 72)
    print(f"\nAgent: {answer}")
