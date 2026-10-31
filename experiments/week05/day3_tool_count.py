"""Week 5 Day 3: Tool count effects — schema token cost + wrong-tool rate.

Experiment question:
    How many tokens do tool schemas consume at different tool counts, and
    does adding irrelevant tools increase the wrong-tool-call rate?

Method — Part 1 (Schema token measurement):
    1. Build tool schema sets at exactly 5, 15, and 30 tools.
    2. Send one API call per count with a fixed neutral prompt + the tools.
    3. Also send a baseline call with the same prompt but NO tools.
    4. Schema token cost = prompt_tokens(with tools) - prompt_tokens(no tools).
    5. Repeat 5× per count to verify consistency.

Method — Part 2 (Wrong-tool rate):
    1. At each tool count (5 / 15 / 30), the agent gets 5 core real tools
       (read_file, write_file, list_dir, run_command, web_search) plus
       N dummy tools that look realistic but should NOT be called.
    2. Run on 8 tool-selection prompts (from the day1 experiment, excluding
       the 2 that require web_fetch which is absent at 5-tool count).
    3. Measure wrong-tool rate: fraction of calls to dummy tools.

    Default: 3 conditions × 5 runs × 8 prompts = 120 agent runs
           + 3 conditions × 5 runs = 15 schema-measurement calls

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week05/day3_tool_count.py
    python experiments/week05/day3_tool_count.py --runs 3   # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Import tool implementations from the week03 agent.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_WEEK03_DIR = _THIS_DIR.parent / "week03"
sys.path.insert(0, str(_WEEK03_DIR))

from agent import (  # noqa: E402
    MODEL,
    SANDBOX_ROOT,
    read_file,
    write_file,
    list_dir,
    run_command,
    web_search,
    _make_success,
    _make_error,
    ErrorType,
)

from openai import OpenAI  # noqa: E402

# ============================================================================
# Pricing (USD per 1M tokens)
# ============================================================================
_PRICING_PER_1M = {
    "qwen3.7-plus": (0.40, 1.20),
    "qwen-plus": (0.40, 1.20),
    "qwen-max": (1.60, 6.40),
    "qwen-turbo": (0.05, 0.20),
    "deepseek-chat": (0.27, 1.10),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def _model_prices() -> tuple[float, float]:
    table_in, table_out = _PRICING_PER_1M.get(MODEL, (0.0, 0.0))
    price_in = float(os.getenv("INPUT_PRICE_PER_1M", table_in))
    price_out = float(os.getenv("OUTPUT_PRICE_PER_1M", table_out))
    return price_in, price_out


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = _model_prices()
    return prompt_tokens / 1_000_000 * price_in + completion_tokens / 1_000_000 * price_out


# ============================================================================
# Core tool schemas (5 real tools from week03)
# ============================================================================
# These are the only tools that should be called by the eval prompts.
# web_fetch is intentionally excluded so the 5-tool condition can cover all
# 8 eval prompts without missing a needed tool.

CORE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file from the sandbox workspace. Call when the user "
                "asks to see, view, inspect, or read the contents of a specific "
                "named file. Path must be relative to the sandbox root (no "
                "leading '/' or '..')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Relative path to the file inside the sandbox, "
                                       "e.g. 'notes.txt' or 'subdir/data.json'.",
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
                "Write or overwrite a text file in the sandbox workspace. Call "
                "when the user asks to save, create, write, or store content to "
                "a file. Path must stay inside the sandbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Relative path inside the sandbox, e.g. 'output.txt'.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write to the file.",
                    },
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
                "List files and subdirectories in the sandbox workspace. Call "
                "when the user wants to browse or discover what files exist, or "
                "needs to find a file but doesn't know its exact path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dirpath": {
                        "type": "string",
                        "description": "Relative directory path; defaults to '.' for sandbox root.",
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
                "Run a shell command inside the sandbox workspace using "
                "allowlisted executables only: ls, cat, echo, grep, wc, head, "
                "tail, python, python3. No shell interpretation — pipes | and "
                "semicolons ; are NOT supported. Timeout: 5 seconds. Output "
                "capped at 4000 chars. Call when you need to search across "
                "multiple files, count lines/words, or run Python scripts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command argv string with an allowlisted executable "
                                       "and its arguments, e.g. 'grep pattern file.txt'.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web via DuckDuckGo and return a list of results with "
                "titles, URLs, and text snippets. Call when the user asks to "
                "search, find, look up, or research something online without "
                "providing a specific URL. Combine with web_fetch (if available) "
                "to open the most promising result for more detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — keep it concise and keyword-focused.",
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

CORE_TOOL_FUNCTIONS = {
    "read_file": lambda inp: read_file(inp["filepath"]),
    "write_file": lambda inp: write_file(inp["filepath"], inp["content"]),
    "list_dir": lambda inp: list_dir(inp.get("dirpath", ".")),
    "run_command": lambda inp: run_command(inp["command"]),
    "web_search": lambda inp: web_search(inp["query"], inp.get("max_results", 5)),
}

CORE_TOOL_NAMES = frozenset(CORE_TOOL_FUNCTIONS)

# ============================================================================
# Dummy tool schemas — 25 realistic-looking tools that should NOT be called
# ============================================================================
# Design principles for dummy tools:
#   1. Each has a plausible name, description, and parameters.
#   2. They cover diverse domains (productivity, data processing, utilities,
#      communication) so they read as a real agent toolkit.
#   3. None of them help with sandbox file operations or web search — the
#      tasks our eval prompts actually require.
#   4. Mix of parameter types: string, integer, boolean, enum, number.
#   5. Some are "near-miss" tools whose descriptions overlap slightly with
#      real tool capabilities (e.g. "search_documents" vs web_search) —
#      these are the most interesting for measuring confusion.

DUMMY_TOOL_SCHEMAS = [
    # ---- Productivity / Communication ----
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Send an email to one or more recipients. Call when the user "
                "asks to email, notify, or send a message to someone. Supports "
                "plain text and HTML bodies. Attachments not supported."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Comma-separated recipient email addresses."},
                    "subject": {"type": "string", "description": "Email subject line."},
                    "body": {"type": "string", "description": "Email body text (plain text)."},
                    "cc": {"type": "string", "description": "Comma-separated CC addresses (optional)."},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": (
                "Create a calendar event or meeting. Call when the user asks to "
                "schedule, book, or add an event to their calendar. Returns a "
                "confirmation with the event ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title."},
                    "start_time": {"type": "string", "description": "Start time in ISO 8601 format."},
                    "end_time": {"type": "string", "description": "End time in ISO 8601 format."},
                    "attendees": {"type": "string", "description": "Comma-separated email addresses (optional)."},
                    "location": {"type": "string", "description": "Physical or virtual location (optional)."},
                },
                "required": ["title", "start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": (
                "Translate text from one language to another. Call when the user "
                "asks to translate, convert language, or understand foreign text. "
                "Supports 50+ languages. Auto-detects source language if not specified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to translate."},
                    "target_language": {"type": "string", "description": "Target language, e.g. 'Chinese', 'French', 'Japanese'."},
                    "source_language": {"type": "string", "description": "Source language (optional; auto-detect if omitted)."},
                },
                "required": ["text", "target_language"],
            },
        },
    },
    # ---- Data / Database ----
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Execute a read-only SQL query against the application database. "
                "Call when the user asks to look up, find, or retrieve structured "
                "data stored in the database. Only SELECT queries are allowed — "
                "INSERT, UPDATE, DELETE are blocked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A read-only SQL SELECT statement."},
                    "max_rows": {"type": "integer", "description": "Max rows to return (default 100)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search through the company's internal document library for "
                "documents matching a query. Call when the user asks to find "
                "internal docs, policies, reports, or reference material. "
                "Returns document titles, excerpts, and links. This is for "
                "internal documents ONLY — not web pages or sandbox files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms for document retrieval."},
                    "doc_type": {
                        "type": "string",
                        "enum": ["all", "policy", "report", "guide", "spec"],
                        "description": "Filter by document type (default 'all').",
                    },
                    "max_results": {"type": "integer", "description": "Max results (default 10)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": (
                "Get the current or historical stock price for a given ticker "
                "symbol. Call when the user asks about stock prices, market data, "
                "or company valuations. Returns open, high, low, close, and volume."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL', 'GOOGL'."},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format (default: latest)."},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather conditions and forecast for a location. "
                "Call when the user asks about weather, temperature, rain, or "
                "climate. Returns temperature, humidity, wind speed, and "
                "conditions description."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name or geographic coordinates."},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units (default 'celsius').",
                    },
                    "days": {"type": "integer", "description": "Forecast days (1–7, default 1 for current)."},
                },
                "required": ["location"],
            },
        },
    },
    # ---- Text / Content utilities ----
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": (
                "Generate a concise summary of a long text passage. Call when "
                "the user asks to summarize, condense, or get the gist of a "
                "block of text they provide. NOT for summarizing files already "
                "in the sandbox — use read_file first, then pass the content here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to summarize."},
                    "max_length": {"type": "integer", "description": "Target summary length in words (default 100)."},
                    "style": {
                        "type": "string",
                        "enum": ["concise", "bullet_points", "paragraph"],
                        "description": "Summary style (default 'concise').",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sentiment_analysis",
            "description": (
                "Analyze the sentiment of a given text. Returns sentiment "
                "label (positive/negative/neutral) and confidence score. "
                "Call when the user asks about tone, sentiment, or emotional "
                "content of a piece of text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to analyze for sentiment."},
                    "language": {"type": "string", "description": "Language code (default 'en')."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spell_check",
            "description": (
                "Check text for spelling and grammar errors. Returns a list of "
                "errors with suggestions. Call when the user asks to proofread, "
                "check spelling, or correct grammar in a text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to spell-check."},
                    "language": {"type": "string", "description": "Language code (default 'en')."},
                    "auto_correct": {"type": "boolean", "description": "If true, return auto-corrected text (default false)."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": (
                "Extract named entities (people, organizations, locations, dates) "
                "from text. Call when the user asks to identify names, places, or "
                "organizations mentioned in a document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to extract entities from."},
                    "entity_types": {
                        "type": "string",
                        "description": "Comma-separated entity types to extract: "
                                       "person, org, location, date, money (default: all).",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "markdown_to_html",
            "description": (
                "Convert Markdown text to HTML. Call when the user asks to "
                "convert, render, or transform markdown into HTML format. "
                "Supports GitHub-Flavored Markdown extensions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "markdown": {"type": "string", "description": "Raw markdown content."},
                    "include_css": {"type": "boolean", "description": "Include minimal default CSS (default false)."},
                },
                "required": ["markdown"],
            },
        },
    },
    # ---- Generators / Utilities ----
    {
        "type": "function",
        "function": {
            "name": "generate_qr_code",
            "description": (
                "Generate a QR code image from text or a URL. Call when the user "
                "asks to create a QR code, barcode, or scannable link. Returns "
                "the QR code as a base64-encoded PNG image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "The text or URL to encode in the QR code."},
                    "size": {"type": "integer", "description": "Image size in pixels (default 256)."},
                    "error_correction": {
                        "type": "string",
                        "enum": ["L", "M", "Q", "H"],
                        "description": "Error correction level: L (~7%), M (~15%), Q (~25%), H (~30%). Default 'M'.",
                    },
                },
                "required": ["data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_password",
            "description": (
                "Generate a secure random password. Call when the user asks to "
                "create a password, generate credentials, or make a secure token. "
                "Configurable length and character sets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "length": {"type": "integer", "description": "Password length in characters (default 16, min 8)."},
                    "include_symbols": {"type": "boolean", "description": "Include special characters (default true)."},
                    "include_numbers": {"type": "boolean", "description": "Include digits (default true)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_uuid",
            "description": (
                "Generate one or more UUID v4 identifiers. Call when the user "
                "needs unique IDs, request IDs, or random identifiers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of UUIDs to generate (default 1, max 100)."},
                    "format": {
                        "type": "string",
                        "enum": ["standard", "hex", "urn"],
                        "description": "Output format: 'standard' (8-4-4-4-12), 'hex' (no dashes), 'urn' (URN prefix).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hash_text",
            "description": (
                "Compute a cryptographic hash (MD5, SHA-1, SHA-256, SHA-512) "
                "of a text string. Call when the user asks to hash, checksum, or "
                "fingerprint data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to hash."},
                    "algorithm": {
                        "type": "string",
                        "enum": ["md5", "sha1", "sha256", "sha512"],
                        "description": "Hash algorithm (default 'sha256').",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "base64_encode_decode",
            "description": (
                "Encode text to Base64 or decode Base64 back to text. Call when "
                "the user asks to encode, decode, or convert data to/from Base64."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to encode (or Base64 string to decode)."},
                    "operation": {
                        "type": "string",
                        "enum": ["encode", "decode"],
                        "description": "Whether to encode or decode (default 'encode').",
                    },
                },
                "required": ["text", "operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "json_validator",
            "description": (
                "Validate a JSON string and return any syntax errors with line "
                "and column positions. Call when the user wants to check if JSON "
                "is valid or debug malformed JSON."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "json_string": {"type": "string", "description": "The JSON string to validate."},
                    "strict": {"type": "boolean", "description": "If true, reject trailing commas and comments (default true)."},
                },
                "required": ["json_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "timezone_converter",
            "description": (
                "Convert a datetime from one timezone to another. Call when the "
                "user asks to convert time zones, schedule across regions, or "
                "figure out what time it is somewhere else."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "datetime_str": {"type": "string", "description": "Datetime in ISO 8601 format, e.g. '2026-08-12T14:30:00'."},
                    "from_tz": {"type": "string", "description": "Source timezone, e.g. 'America/New_York', 'Asia/Shanghai'."},
                    "to_tz": {"type": "string", "description": "Target timezone."},
                },
                "required": ["datetime_str", "from_tz", "to_tz"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": (
                "Convert a value between different units of measurement. Call "
                "when the user asks to convert lengths, weights, temperatures, "
                "volumes, or other units. Supports metric and imperial systems."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Numeric value to convert."},
                    "from_unit": {"type": "string", "description": "Source unit, e.g. 'km', 'lb', 'celsius', 'gallon'."},
                    "to_unit": {"type": "string", "description": "Target unit."},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_distance",
            "description": (
                "Calculate the great-circle distance between two geographic "
                "coordinates. Call when the user asks about distances between "
                "cities, travel distances, or geographic proximity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat1": {"type": "number", "description": "Latitude of point 1 (-90 to 90)."},
                    "lon1": {"type": "number", "description": "Longitude of point 1 (-180 to 180)."},
                    "lat2": {"type": "number", "description": "Latitude of point 2 (-90 to 90)."},
                    "lon2": {"type": "number", "description": "Longitude of point 2 (-180 to 180)."},
                    "unit": {
                        "type": "string",
                        "enum": ["km", "miles", "nautical"],
                        "description": "Distance unit (default 'km').",
                    },
                },
                "required": ["lat1", "lon1", "lat2", "lon2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "color_converter",
            "description": (
                "Convert a color between different representations: hex, RGB, "
                "HSL, HSV, CMYK. Call when the user asks to convert color codes "
                "or work with color values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {"type": "string", "description": "Color value, e.g. '#ff0000', 'rgb(255,0,0)', 'hsl(0,100,50)'."},
                    "target_format": {
                        "type": "string",
                        "enum": ["hex", "rgb", "hsl", "hsv", "cmyk"],
                        "description": "Desired output format.",
                    },
                },
                "required": ["color", "target_format"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ip_geolocation",
            "description": (
                "Look up the geographic location of an IP address. Returns "
                "country, region, city, ISP, and approximate coordinates. "
                "Call when the user asks where an IP address is from."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ip": {"type": "string", "description": "IPv4 or IPv6 address to look up."},
                },
                "required": ["ip"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compress_text",
            "description": (
                "Compress or decompress text using gzip/zlib. Call when the "
                "user asks to compress, decompress, or reduce the size of "
                "text data. Returns base64-encoded compressed data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to compress or base64 of compressed data to decompress."},
                    "operation": {
                        "type": "string",
                        "enum": ["compress", "decompress"],
                        "description": "Operation to perform.",
                    },
                },
                "required": ["text", "operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_ocr",
            "description": (
                "Extract text from an image using optical character recognition "
                "(OCR). Call when the user wants to read text from a screenshot, "
                "photo, or scanned document. Supports PNG, JPG, and PDF inputs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the image or PDF file."},
                    "language": {"type": "string", "description": "Language code for OCR (default 'eng')."},
                },
                "required": ["image_path"],
            },
        },
    },
]


def build_tools_for_count(n_real: int, n_dummy: int) -> list[dict]:
    """Build a tool-schema list with n_real core tools + n_dummy dummy tools."""
    assert 1 <= n_real <= len(CORE_TOOL_SCHEMAS)
    assert n_dummy <= len(DUMMY_TOOL_SCHEMAS)
    return list(CORE_TOOL_SCHEMAS[:n_real]) + list(DUMMY_TOOL_SCHEMAS[:n_dummy])


def build_tool_functions_for_real(n_real: int) -> dict:
    """Build the tool-function registry with the first n_real core tools."""
    core_names = [t["function"]["name"] for t in CORE_TOOL_SCHEMAS[:n_real]]
    return {name: CORE_TOOL_FUNCTIONS[name] for name in core_names}


# ============================================================================
# Part 1 — Schema token measurement
# ============================================================================

# A fixed neutral prompt that does NOT require any tool calls — this ensures
# the model won't emit tool_calls, keeping the prompt_tokens measurement clean.
_SCHEMA_MEASUREMENT_PROMPT = (
    "What is 2 + 2? Answer in one word."
)


def measure_schema_tokens(
    client: OpenAI,
    tools: list[dict],
    *,
    n_runs: int = 5,
) -> dict:
    """Measure how many prompt tokens the tool schemas contribute.

    Sends the same prompt twice per run:
      1. Without tools (baseline prompt_tokens).
      2. With tools.

    The difference isolates the per-request token cost of the tool schemas.
    The model's response is always brief (one word), so completion_tokens
    are negligible.
    """
    prompt = _SCHEMA_MEASUREMENT_PROMPT

    baseline_tokens: list[int] = []
    with_tools_tokens: list[int] = []
    schema_costs: list[int] = []
    schema_chars: list[int] = []

    tools_json = json.dumps(tools, ensure_ascii=False)
    tools_char_count = len(tools_json)

    for run_idx in range(1, n_runs + 1):
        # Basline — no tools
        resp_baseline = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        b_pt = resp_baseline.usage.prompt_tokens if resp_baseline.usage else 0
        baseline_tokens.append(b_pt)

        # With tools
        resp_with = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
        )
        w_pt = resp_with.usage.prompt_tokens if resp_with.usage else 0
        with_tools_tokens.append(w_pt)

        schema_cost = w_pt - b_pt
        schema_costs.append(schema_cost)

    mean_baseline = sum(baseline_tokens) / len(baseline_tokens) if baseline_tokens else 0
    mean_with = sum(with_tools_tokens) / len(with_tools_tokens) if with_tools_tokens else 0
    mean_cost = sum(schema_costs) / len(schema_costs) if schema_costs else 0

    # Token-per-tool efficiency ratio
    n_tools = len(tools)
    tokens_per_tool = mean_cost / n_tools if n_tools > 0 else 0
    chars_per_token = tools_char_count / mean_cost if mean_cost > 0 else 0

    return {
        "n_tools": n_tools,
        "tools_json_chars": tools_char_count,
        "tools_json_est_tokens": tools_char_count // 4,
        "baseline_prompt_tokens": baseline_tokens,
        "baseline_mean": mean_baseline,
        "with_tools_prompt_tokens": with_tools_tokens,
        "with_tools_mean": mean_with,
        "schema_token_cost": schema_costs,
        "schema_token_mean": mean_cost,
        "tokens_per_tool": tokens_per_tool,
        "chars_per_token_equivalent": chars_per_token,
    }


# ============================================================================
# Part 2 — Agent loop for wrong-tool measurement
# ============================================================================


def run_experiment_agent(
    user_input: str,
    tools: list[dict],
    tool_functions: dict,
    *,
    max_turns: int = 12,
) -> dict:
    """Run the agent with a given tool set and return structured results."""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    system_prompt = (
        "You are a helpful assistant with access to various tools. "
        f"All sandbox file paths are relative to {SANDBOX_ROOT}. "
        "Complete the user's task using the available tools.\n\n"
        "IMPORTANT:\n"
        "- Read each tool's description carefully before calling it.\n"
        "- Only call tools that are actually needed for the task.\n"
        "- If a task can be done without any tool, answer directly.\n"
        "- When you have the answer, state it clearly and stop calling tools."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    tool_calls_trace: list[dict] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for turn in range(1, max_turns + 1):
        response = client.chat.completions.create(
            model=MODEL,
            tools=tools,
            messages=messages,
        )

        if response.usage:
            usage_totals["prompt_tokens"] += response.usage.prompt_tokens or 0
            usage_totals["completion_tokens"] += response.usage.completion_tokens or 0
            usage_totals["total_tokens"] += response.usage.total_tokens or 0

        choice = response.choices[0]
        msg = choice.message

        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            return {
                "answer": msg.content or "[no text]",
                "tool_calls": tool_calls_trace,
                "n_turns": turn,
                "n_tool_calls": len(tool_calls_trace),
                "usage": dict(usage_totals),
                "estimated_cost": _estimate_cost(
                    usage_totals["prompt_tokens"], usage_totals["completion_tokens"]
                ),
                "error": None,
            }

        if choice.finish_reason == "length":
            return {
                "answer": "[truncated — max_tokens limit]",
                "tool_calls": tool_calls_trace,
                "n_turns": turn,
                "n_tool_calls": len(tool_calls_trace),
                "usage": dict(usage_totals),
                "estimated_cost": _estimate_cost(
                    usage_totals["prompt_tokens"], usage_totals["completion_tokens"]
                ),
                "error": "finish_reason=length",
            }

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                output = _make_error(
                    ErrorType.INVALID_INPUT,
                    f"Invalid JSON arguments: {e}",
                )
                tool_calls_trace.append({
                    "name": fn_name,
                    "arguments": tc.function.arguments,
                    "error": f"JSON parse: {e}",
                })
            else:
                fn = tool_functions.get(fn_name)
                if fn is None:
                    # Dummy tool — return a plausible success response
                    output = json.dumps({
                        "status": "success",
                        "data": {"message": f"Executed {fn_name} successfully.", "args": args},
                        "error": None,
                        "truncated": False,
                        "truncated_at": None,
                    })
                else:
                    try:
                        output = fn(args)
                    except Exception as e:
                        output = _make_error(ErrorType.UNKNOWN, str(e))

                tool_calls_trace.append({
                    "name": fn_name,
                    "arguments": args,
                    "error": None,
                    "is_dummy": fn is None,
                })

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

    return {
        "answer": f"[Agent stopped after {max_turns} turns — possible loop.]",
        "tool_calls": tool_calls_trace,
        "n_turns": max_turns,
        "n_tool_calls": len(tool_calls_trace),
        "usage": dict(usage_totals),
        "estimated_cost": _estimate_cost(
            usage_totals["prompt_tokens"], usage_totals["completion_tokens"]
        ),
        "error": "max_turns_exceeded",
    }


# ============================================================================
# Part 2 — Eval prompts (8 prompts, all satisfiable by 5 core tools)
# ============================================================================
# Same prompts as day1 except P5 (fetch-url, needs web_fetch) and P7
# (search-then-fetch, also needs web_fetch).  All 8 remaining prompts only
# require tools from the 5-tool core set.

EVAL_PROMPTS = [
    {
        "id": "P1-read-known-file",
        "prompt": (
            "Read the file 'subdir/data.json' from the sandbox. "
            "What fields does it contain? List all the field names."
        ),
        "expected_tools": ["read_file"],
        "category": "single-tool",
    },
    {
        "id": "P2-browse-sandbox",
        "prompt": (
            "I just started working in this sandbox. What files and directories "
            "are available here? Show me everything."
        ),
        "expected_tools": ["list_dir"],
        "category": "single-tool",
    },
    {
        "id": "P3-grep-across-files",
        "prompt": (
            "Search all text files (.txt, .md, .py) in the sandbox for any "
            "mention of the word 'sandbox'. For each file that contains it, "
            "tell me which file it is and show the matching line."
        ),
        "expected_tools": ["run_command"],
        "category": "grep-vs-read",
    },
    {
        "id": "P4-save-content",
        "prompt": (
            "Create a new file called 'week5_summary.md' in the sandbox with "
            "this exact content:\n\n"
            "# Week 5 Summary\n\n"
            "## Day 1\n"
            "- Experiment: tool description rewrite\n"
            "- Topic: when-to-call conditions\n"
            "- Status: running\n\n"
            "Save it now."
        ),
        "expected_tools": ["write_file"],
        "category": "single-tool",
    },
    {
        "id": "P6-search-online",
        "prompt": (
            "Search online: what are the most popular open-source AI agent "
            "frameworks in 2026? Give me the top 3 with a one-line description "
            "of each."
        ),
        "expected_tools": ["web_search"],
        "category": "single-tool",
    },
    {
        "id": "P8-count-lines-efficient",
        "prompt": (
            "How many lines are in the sandbox file 'notes.md'? Use the most "
            "efficient method — do NOT read the entire file just to count lines."
        ),
        "expected_tools": ["run_command"],
        "category": "grep-vs-read",
    },
    {
        "id": "P9-no-tool-trivia",
        "prompt": (
            "What year did World War II end? Just give me the year."
        ),
        "expected_tools": [],
        "category": "no-tool",
    },
    {
        "id": "P10-trap-web-search-vs-local",
        "prompt": (
            "I need the latest information about 'sandbox escape prevention "
            "techniques in AI agent systems'. Search for this topic and tell "
            "me what the top recommendations are."
        ),
        "expected_tools": ["web_search"],
        "category": "web-vs-local-trap",
    },
]


# ============================================================================
# Scoring
# ============================================================================


def score_run(trace: dict, expected_tools: list[str], core_names: frozenset[str]) -> dict:
    """Compute tool-selection metrics.

    Key metric for this experiment: **wrong-tool rate** — what fraction of
    calls went to dummy tools that should never have been called?

    Also tracks:
      - should-call rate: expected tools called / total expected
      - exact-match: called set == expected set
      - dummy-call rate: dummy calls / total calls (the main measure)

    core_names is the set of tool names that are "real" and should be used.
    Anything not in core_names is a dummy tool.
    """
    called_names = [tc["name"] for tc in trace.get("tool_calls", [])]
    called_set = set(called_names)
    expected = set(expected_tools)

    # Should-call rate
    if len(expected) == 0:
        should_call_rate = 1.0 if len(called_set) == 0 else 0.0
    else:
        should_call_rate = len(called_set & expected) / len(expected)

    # Dummy-tool rate (the main metric for this experiment)
    total_calls = len(called_names)
    if total_calls == 0:
        dummy_call_rate = 0.0
        dummy_count = 0
    else:
        dummy_count = sum(1 for n in called_names if n not in core_names)
        dummy_call_rate = dummy_count / total_calls

    # Wrong-tool rate (broader: any call not in expected)
    if total_calls == 0:
        wrong_tool_rate = 0.0
    else:
        wrong_count = len(called_set - expected)
        wrong_tool_rate = wrong_count / total_calls

    # Exact match
    exact_match = called_set == expected

    # Missed and extra
    missed = sorted(expected - called_set)
    extra = sorted(called_set - expected)
    dummy_calls_list = [n for n in called_names if n not in core_names]

    return {
        "called": called_names,
        "expected": sorted(expected),
        "should_call_rate": should_call_rate,
        "dummy_call_rate": dummy_call_rate,
        "dummy_count": dummy_count,
        "dummy_calls": dummy_calls_list,
        "wrong_tool_rate": wrong_tool_rate,
        "exact_match": exact_match,
        "missed": missed,
        "extra": extra,
        "n_calls": total_calls,
        "n_turns": trace.get("n_turns", 0),
    }


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Runs per (prompt, condition) pair (default 5).",
    )
    parser.add_argument(
        "--prompts", type=str, nargs="*",
        help="Prompt IDs to run (default: all 8).",
    )
    parser.add_argument(
        "--skip-part1", action="store_true",
        help="Skip the schema-token measurement (Part 1).",
    )
    parser.add_argument(
        "--skip-part2", action="store_true",
        help="Skip the wrong-tool-rate experiment (Part 2).",
    )
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Filter prompts if requested
    prompts = EVAL_PROMPTS
    if args.prompts:
        prompts = [p for p in EVAL_PROMPTS if p["id"] in args.prompts]
        if not prompts:
            raise SystemExit(f"No prompts matched: {args.prompts}")

    # ------------------------------------------------------------------
    # Configuration — tool counts and their compositions
    # ------------------------------------------------------------------
    # Each condition: (label, n_real_tools, n_dummy_tools)
    #   - 5 tools:  5 real (all core tools)
    #   - 15 tools: 5 real + 10 dummy
    #   - 30 tools: 5 real + 25 dummy
    # All conditions share the same 5 core tools → same expected_tools for
    # every eval prompt → clean comparison across counts.

    TOOL_COUNT_CONDITIONS = [
        ("5_tools", 5, 0),
        ("15_tools", 5, 10),
        ("30_tools", 5, 25),
    ]

    all_results: list[dict] = []
    schema_results: dict[str, dict] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ==================================================================
    # PART 1 — Schema token measurement
    # ==================================================================
    schema_baseline_runs: list[int] = []

    if not args.skip_part1:
        print("=" * 72)
        print("PART 1 — SCHEMA TOKEN MEASUREMENT")
        print("=" * 72)
        print(f"  Prompt: \"{_SCHEMA_MEASUREMENT_PROMPT}\"")
        print(f"  Runs per count: {args.runs}")
        print()

        for label, n_real, n_dummy in TOOL_COUNT_CONDITIONS:
            tools = build_tools_for_count(n_real, n_dummy)
            print(f"  [{label}] {len(tools)} total tools ({n_real} real + {n_dummy} dummy)")
            result = measure_schema_tokens(client, tools, n_runs=args.runs)
            schema_results[label] = result

            # Use first run's baseline as shared baseline (all runs use the same prompt)
            if not schema_baseline_runs:
                schema_baseline_runs = result["baseline_prompt_tokens"]

            print(f"    Baseline (no tools):     {result['baseline_mean']:>6.0f} prompt tokens")
            print(f"    With {len(tools):>2} tools:        {result['with_tools_mean']:>6.0f} prompt tokens")
            print(f"    Schema cost:             {result['schema_token_mean']:>6.0f} tokens "
                  f"({result['schema_token_mean'] / len(tools):.1f} per tool)")
            print(f"    Tools JSON size:         {result['tools_json_chars']:>6} chars "
                  f"(~{result['tools_json_est_tokens']} char/4 tokens)")
            print(f"    Chars/token equivalent:  {result['chars_per_token_equivalent']:.1f}")
            print()

        # Comparison table
        print("-" * 72)
        print(f"{'count':<12}{'tools':>6}{'schema chars':>14}{'schema tokens':>15}{'per-tool':>10}{'char/tok':>10}")
        print("-" * 72)
        for label in [c[0] for c in TOOL_COUNT_CONDITIONS]:
            r = schema_results[label]
            print(
                f"{label:<12}{r['n_tools']:>6}{r['tools_json_chars']:>14,}"
                f"{r['schema_token_mean']:>15.0f}{r['tokens_per_tool']:>10.1f}"
                f"{r['chars_per_token_equivalent']:>10.1f}"
            )

        # Scaling analysis
        print()
        print("SCALING ANALYSIS:")
        costs = [(label, schema_results[label]["schema_token_mean"], schema_results[label]["n_tools"])
                 for label in [c[0] for c in TOOL_COUNT_CONDITIONS]]
        for i in range(1, len(costs)):
            prev_label, prev_cost, prev_n = costs[i - 1]
            curr_label, curr_cost, curr_n = costs[i]
            tool_mult = curr_n / prev_n
            cost_mult = curr_cost / prev_cost
            print(f"  {prev_label} → {curr_label}: "
                  f"{tool_mult:.1f}× tools → {cost_mult:.1f}× schema tokens "
                  f"({'sub-linear' if cost_mult < tool_mult else 'super-linear' if cost_mult > tool_mult else 'linear'})")
        print()

    # ==================================================================
    # PART 2 — Wrong-tool rate at each tool count
    # ==================================================================
    if not args.skip_part2:
        print("=" * 72)
        print("PART 2 — WRONG-TOOL RATE BY TOOL COUNT")
        print("=" * 72)
        print(f"  Model: {MODEL}")
        print(f"  Prompts: {len(prompts)}  |  Runs per cell: {args.runs}")
        print(f"  Conditions: {[c[0] for c in TOOL_COUNT_CONDITIONS]}")
        print(f"  Total runs: {len(prompts) * len(TOOL_COUNT_CONDITIONS) * args.runs}")
        print()

        # Accumulators per condition
        per_cond: dict[str, dict] = {}
        for label, _, _ in TOOL_COUNT_CONDITIONS:
            per_cond[label] = {
                "dummy_call_rate": [],
                "wrong_tool_rate": [],
                "should_call_rate": [],
                "exact_match": [],
                "tokens": [],
                "n_calls": [],
                "per_prompt_exact": {},
                "per_prompt_total": {},
            }

        for cond_idx, (label, n_real, n_dummy) in enumerate(TOOL_COUNT_CONDITIONS):
            tools = build_tools_for_count(n_real, n_dummy)
            tool_functions = build_tool_functions_for_real(n_real)

            # Core names for this condition (all real tools)
            core_names = frozenset(
                t["function"]["name"] for t in CORE_TOOL_SCHEMAS[:n_real]
            )
            print(f"{'─' * 72}")
            print(f"CONDITION: {label}  ({len(tools)} tools: {n_real} real + {n_dummy} dummy)")
            print(f"  Real tools: {sorted(core_names)}")
            print(f"  Schema size: {len(json.dumps(tools, ensure_ascii=False)):,} chars")
            print()

            for prompt in prompts:
                prompt_id = prompt["id"]
                expected = prompt["expected_tools"]
                per_cond[label]["per_prompt_exact"][prompt_id] = 0
                per_cond[label]["per_prompt_total"][prompt_id] = 0

                for run_idx in range(1, args.runs + 1):
                    try:
                        trace = run_experiment_agent(
                            prompt["prompt"], tools, tool_functions,
                        )
                    except Exception as e:
                        print(f"  [{label}] {prompt_id} run {run_idx}: API ERROR — {e}")
                        all_results.append({
                            "condition": label,
                            "prompt_id": prompt_id,
                            "run": run_idx,
                            "api_error": str(e),
                        })
                        continue

                    score = score_run(trace, expected, core_names)
                    all_results.append({
                        "condition": label,
                        "prompt_id": prompt_id,
                        "run": run_idx,
                        "expected_tools": expected,
                        "core_tools_available": sorted(core_names),
                        **score,
                        "answer": trace["answer"][:300],
                        "tool_call_details": trace["tool_calls"],
                        "usage": trace["usage"],
                        "estimated_cost": trace["estimated_cost"],
                        "error": trace.get("error"),
                    })

                    per_cond[label]["dummy_call_rate"].append(score["dummy_call_rate"])
                    per_cond[label]["wrong_tool_rate"].append(score["wrong_tool_rate"])
                    per_cond[label]["should_call_rate"].append(score["should_call_rate"])
                    per_cond[label]["exact_match"].append(1 if score["exact_match"] else 0)
                    per_cond[label]["tokens"].append(trace["usage"]["total_tokens"])
                    per_cond[label]["n_calls"].append(score["n_calls"])
                    per_cond[label]["per_prompt_exact"][prompt_id] += 1 if score["exact_match"] else 0
                    per_cond[label]["per_prompt_total"][prompt_id] += 1

                    # Per-run detail
                    icon = "✅" if score["exact_match"] else "❌"
                    call_str = ", ".join(score["called"]) if score["called"] else "(none)"
                    exp_str = ", ".join(score["expected"]) if score["expected"] else "(none)"
                    extra_info = ""
                    if score["dummy_calls"]:
                        extra_info += f"  DUMMY: {score['dummy_calls']}"
                    if score["missed"]:
                        extra_info += f"  missed: {score['missed']}"
                    print(
                        f"  [{icon}] {prompt_id:<32} run {run_idx}/{args.runs}"
                        f"  called=[{call_str}]  expected=[{exp_str}]{extra_info}"
                    )

                # Per-prompt summary
                n_exact = per_cond[label]["per_prompt_exact"][prompt_id]
                n_total = per_cond[label]["per_prompt_total"][prompt_id]
                rate = n_exact / n_total if n_total else 0
                mark = "✅" if rate == 1.0 else ("❌" if rate == 0.0 else "⚠️")
                print(f"  {mark} {prompt_id}: {n_exact}/{n_total} exact-match ({rate:.0%})")
            print()

        # =================================================================
        # Summary tables
        # =================================================================

        # Table 1 — Dummy-tool call rate (THE key metric)
        print("=" * 72)
        print("DUMMY-TOOL CALL RATE BY TOOL COUNT  (lower = better)")
        print("  Fraction of all tool calls that went to irrelevant dummy tools.")
        print("=" * 72)
        header = f"{'condition':<14}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}{'n_runs':>8}"
        print(header)
        print("-" * len(header))
        for label, _, _ in TOOL_COUNT_CONDITIONS:
            rates = per_cond[label]["dummy_call_rate"]
            if rates:
                mean_r = sum(rates) / len(rates)
                var_r = sum((r - mean_r) ** 2 for r in rates) / len(rates)
                sd_r = var_r ** 0.5
                print(f"{label:<14}{mean_r:>8.3f}{sd_r:>8.3f}{min(rates):>8.3f}{max(rates):>8.3f}{len(rates):>8}")

        # Table 2 — Wrong-tool rate (broader)
        print(f"\n{'=' * 72}")
        print("WRONG-TOOL RATE BY TOOL COUNT  (lower = better)")
        print("  Fraction of all tool calls that were to unexpected tools.")
        print(f"{'=' * 72}")
        print(header.replace("dummy", "wrong "))
        print("-" * len(header))
        for label, _, _ in TOOL_COUNT_CONDITIONS:
            rates = per_cond[label]["wrong_tool_rate"]
            if rates:
                mean_r = sum(rates) / len(rates)
                var_r = sum((r - mean_r) ** 2 for r in rates) / len(rates)
                sd_r = var_r ** 0.5
                print(f"{label:<14}{mean_r:>8.3f}{sd_r:>8.3f}{min(rates):>8.3f}{max(rates):>8.3f}{len(rates):>8}")

        # Table 3 — Exact-match rate
        print(f"\n{'=' * 72}")
        print("EXACT-MATCH RATE BY TOOL COUNT  (higher = better)")
        print(f"{'=' * 72}")
        print(f"{'condition':<14}{'exact':>8}{'total':>8}{'rate':>9}")
        print("-" * 42)
        for label, _, _ in TOOL_COUNT_CONDITIONS:
            exacts = per_cond[label]["exact_match"]
            if exacts:
                n_exact = sum(exacts)
                n_total = len(exacts)
                rate = n_exact / n_total if n_total else 0
                print(f"{label:<14}{n_exact:>8}{n_total:>8}{rate:>8.0%}")

        # Table 4 — Should-call rate
        print(f"\n{'=' * 72}")
        print("SHOULD-CALL RATE BY TOOL COUNT  (higher = better)")
        print(f"{'=' * 72}")
        print(header.replace("dummy", "should"))
        print("-" * len(header))
        for label, _, _ in TOOL_COUNT_CONDITIONS:
            rates = per_cond[label]["should_call_rate"]
            if rates:
                mean_r = sum(rates) / len(rates)
                var_r = sum((r - mean_r) ** 2 for r in rates) / len(rates)
                sd_r = var_r ** 0.5
                print(f"{label:<14}{mean_r:>8.3f}{sd_r:>8.3f}{min(rates):>8.3f}{max(rates):>8.3f}{len(rates):>8}")

        # Table 5 — Token usage
        print(f"\n{'=' * 72}")
        print("TOKEN USAGE BY TOOL COUNT  (mean ± sd)")
        print(f"{'=' * 72}")
        for label, _, _ in TOOL_COUNT_CONDITIONS:
            tokens = per_cond[label]["tokens"]
            if tokens:
                mean_t = sum(tokens) / len(tokens)
                var_t = sum((t - mean_t) ** 2 for t in tokens) / len(tokens)
                sd_t = var_t ** 0.5
                print(f"  {label:<14} {mean_t:>8,.0f} ± {sd_t:>8,.0f} tokens/run")

        # Per-prompt comparison
        print(f"\n{'=' * 72}")
        print(f"PER-PROMPT EXACT-MATCH COMPARISON  (out of {args.runs} runs each)")
        print(f"{'=' * 72}")
        cond_labels = [c[0] for c in TOOL_COUNT_CONDITIONS]
        print(f"{'prompt':<34}" + "".join(f"{c:>10}" for c in cond_labels) + f"{'category':<22}")
        print("-" * (34 + 10 * len(cond_labels) + 22))
        for prompt in prompts:
            pid = prompt["id"]
            row = f"{pid:<34}"
            for label in cond_labels:
                n_ok = per_cond[label]["per_prompt_exact"].get(pid, 0)
                n_total = per_cond[label]["per_prompt_total"].get(pid, 1)
                rate = n_ok / n_total if n_total else 0
                row += f"{rate:>9.0%}"
            row += f"{prompt['category']:<22}"
            print(row)

        # Most commonly confused dummy tools
        print(f"\n{'=' * 72}")
        print("DUMMY TOOLS MOST FREQUENTLY CALLED BY MISTAKE")
        print(f"{'=' * 72}")
        for label in cond_labels:
            dummy_counter: Counter = Counter()
            for r in all_results:
                if r.get("condition") == label and "api_error" not in r:
                    for dc in r.get("dummy_calls", []):
                        dummy_counter[dc] += 1
            if dummy_counter:
                print(f"\n  {label}:")
                for tool, count in dummy_counter.most_common(10):
                    print(f"    {tool:<34} {count:>3} times")
            else:
                print(f"\n  {label}: (none — no dummy tools called)")

    # =====================================================================
    # Save results
    # =====================================================================
    out_dir = _THIS_DIR

    # Build summary
    summary: dict = {
        "experiment": "week05_day3_tool_count",
        "model": MODEL,
        "timestamp": timestamp,
        "runs_per_cell": args.runs,
        "tool_counts": [c[0] for c in TOOL_COUNT_CONDITIONS],
    }

    if not args.skip_part1:
        summary["part1_schema_measurement"] = {
            "prompt": _SCHEMA_MEASUREMENT_PROMPT,
            "results": schema_results,
        }

    if not args.skip_part2:
        summary["part2_wrong_tool_rate"] = {
            "dummy_call_rate": {
                label: {
                    "mean": sum(v) / len(v) if v else 0,
                    "std": (sum((x - (sum(v) / len(v))) ** 2 for x in v) / len(v)) ** 0.5 if v else 0,
                    "n": len(v),
                }
                for label, d in per_cond.items()
                for v in [d["dummy_call_rate"]]
            },
            "wrong_tool_rate": {
                label: {
                    "mean": sum(v) / len(v) if v else 0,
                    "n": len(v),
                }
                for label, d in per_cond.items()
                for v in [d["wrong_tool_rate"]]
            },
            "exact_match_rate": {
                label: {
                    "rate": sum(v) / len(v) if v else 0,
                    "n": len(v),
                }
                for label, d in per_cond.items()
                for v in [d["exact_match"]]
            },
            "token_usage": {
                label: {
                    "mean": sum(v) / len(v) if v else 0,
                    "total": sum(v),
                    "n": len(v),
                }
                for label, d in per_cond.items()
                for v in [d["tokens"]]
            },
            "per_prompt_exact": {
                pid: {
                    label: per_cond[label]["per_prompt_exact"].get(pid, 0)
                    for label in [c[0] for c in TOOL_COUNT_CONDITIONS]
                }
                for pid in [p["id"] for p in prompts]
            },
        }

    out_path = out_dir / f"day3_results_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "prompt_details": [
                    {"id": p["id"], "prompt": p["prompt"], "expected_tools": p["expected_tools"],
                     "category": p["category"]}
                    for p in prompts
                ],
                "dummy_tool_names": [t["function"]["name"] for t in DUMMY_TOOL_SCHEMAS],
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
