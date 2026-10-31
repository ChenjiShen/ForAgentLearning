"""Week 5 Day 5: Forced tool_choice router vs. one big agent.

Experiment question:
    Does a router architecture — force the FIRST call to be a
    ``classify_request`` tool, then dispatch to a fresh conversation with a
    specialist system prompt and only that specialty's tools — beat one big
    agent holding all 6 tools, on the same 10 tool-selection prompts?

Why this tests the day's topic (forced tool use / when to constrain):
    The router call is *forced* via ``tool_choice={"type": "function",
    "function": {"name": "classify_request"}}`` — the model cannot answer,
    dodge, or pick another tool; its first output MUST be the classification.
    Each specialist then sees a smaller tool surface (1-3 tools instead of
    6) and a prompt that only concerns its domain. The monolith pays for 6
    tool schemas on every call but can never be misrouted.

Method:
    1. Conditions:
       monolith — one agent, all 6 tools, general system prompt. Uses the
                  day1 "after" (current best) descriptions. Identical to
                  the best configuration measured so far.
       router   — stage 1: single call with classify_request as the ONLY
                  tool and tool_choice forced to that function. Stage 2:
                  dispatch to a specialist, each a FRESH conversation with
                  a specialist system prompt and only its own tools:
                      file    → read_file, write_file, list_dir
                      command → run_command
                      web     → web_search, web_fetch
                      none    → no tools, answer directly
    2. Same 10 prompts and same scoring as days 1-4 (tool-selection exact
       match, should-call, wrong-tool). classify_request is router
       infrastructure, not a task tool — it is excluded from tool-selection
       scoring and reported separately as routing accuracy.
    3. Compare: exact-match / should-call / wrong-tool rates, tokens, cost.
       Router adds: routing accuracy (gold category per prompt), confusion
       matrix, misroute cascade (do misrouted tasks degrade?), and the token
       overhead of the routing call itself.
    4. Routing errors (unparseable classification, missing tool call) count
       as FAILED runs — they are real router behavior, not noise.

    Default: 2 conditions × 5 runs × 10 prompts = 100 agent runs
             (router runs = 2 API conversations each)

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week05/day5_router.py
    python experiments/week05/day5_router.py --conditions router      # re-run router only
    python experiments/week05/day5_router.py --runs 3 --prompts P1 P5 P9  # quick test

Platform note: DashScope qwen3.x rejects tool_choice while thinking is
enabled. The router call falls back: forced tool_choice with
enable_thinking=False → forced tool_choice with defaults → prompt-only
forcing (classify_request is still the only tool; refusal = routing
failure). The mechanism used is recorded per run in the results JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_WEEK03_DIR = _THIS_DIR.parent / "week03"
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_WEEK03_DIR))

# Reuse the day1 harness: tool implementations, scoring, prompts, and the
# "after" (current best) description set for the monolith condition.
import day1_tool_descriptions as d1  # noqa: E402

from agent import MODEL, SANDBOX_ROOT, ALLOWED_EXECUTABLES  # noqa: E402
from openai import OpenAI  # noqa: E402

_API_BASE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ============================================================================
# Pricing (USD per 1M tokens) — same table as day1
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


def _make_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=_API_BASE_URL)


# ============================================================================
# Specialists — system prompt + the tool subset each one gets
# ============================================================================
# Each specialist is a FRESH conversation: no history of the routing call,
# only the original user request, a specialist system prompt, and a smaller
# tool surface. The smaller surface is half the point of routing: fewer
# schemas per call and fewer wrong choices available.

SPECIALIST_TOOLS: dict[str, list[str]] = {
    "file": ["read_file", "write_file", "list_dir"],
    "command": ["run_command"],
    "web": ["web_search", "web_fetch"],
    "none": [],
}

SPECIALIST_SYSTEMS = {
    "file": (
        "You are the FILE specialist of a multi-agent system. A router "
        "classified this request as a local-file task. Your ONLY tools are "
        f"read_file, write_file and list_dir, and all paths stay inside "
        f"{SANDBOX_ROOT}. You have NO shell access and NO web access.\n\n"
        "Tool selection rules:\n"
        "- The user names a specific file and wants its contents → read_file "
        "directly. Never list_dir first when the path is known.\n"
        "- The user wants to browse/discover what exists → list_dir.\n"
        "- The user asks to save/create/overwrite content → write_file.\n"
        "- When you have the answer, state it clearly and stop calling tools."
    ),
    "command": (
        "You are the COMMAND specialist of a multi-agent system. A router "
        "classified this request as a shell-command task. Your ONLY tool is "
        "run_command, restricted to allowlisted executables: "
        + ", ".join(sorted(ALLOWED_EXECUTABLES))
        + f". Pipes and semicolons are not supported. Commands run inside "
        f"{SANDBOX_ROOT}.\n\n"
        "Tool selection rules:\n"
        "- Searching text across one or many files → grep.\n"
        "- Counting lines/words/chars → wc.\n"
        "- Listing files with filters → ls.\n"
        "- Computation → python -c one-liners.\n"
        "- When you have the answer, state it clearly and stop calling tools."
    ),
    "web": (
        "You are the WEB specialist of a multi-agent system. A router "
        "classified this request as a web task. Your ONLY tools are "
        "web_search and web_fetch. You have NO local file access.\n\n"
        "Tool selection rules:\n"
        "- The user gives a specific URL → web_fetch directly; do not search "
        "first.\n"
        "- No URL given → web_search first, then answer from the snippets. "
        "Fetch at most ONE result page, and only if the snippets miss a "
        "specific detail the user asked for.\n"
        "- When you have the answer, state it clearly and stop calling tools."
    ),
    "none": (
        "You are the GENERAL knowledge specialist of a multi-agent system. "
        "A router classified this request as needing no tools. You have no "
        "tools at all. Answer directly from your knowledge, briefly and "
        "precisely."
    ),
}

# ============================================================================
# Router — forced first call to classify_request
# ============================================================================

ROUTER_SYSTEM = (
    "You are the REQUEST ROUTER of a multi-agent system. You never answer "
    "the user's question yourself. Your only job is to call classify_request "
    "exactly once, choosing the specialist that should handle the request.\n\n"
    "Available specialists:\n"
    f'- "file": local sandbox file operations (read_file, write_file, '
    f"list_dir) inside {SANDBOX_ROOT}. Choose when the request is about "
    "local files, their paths, or their contents.\n"
    '- "command": shell commands (grep, wc, ls, python one-liners). Choose '
    "when the task searches across many files, counts, or does batch/compute "
    "work on local files.\n"
    '- "web": web_search + web_fetch. Choose when the request asks for '
    "online information, a URL, current events, or research.\n"
    '- "none": no tools at all — pure knowledge questions. Choose only when '
    "the request clearly needs no tool.\n\n"
    "Routing rules:\n"
    "- The word 'sandbox' can mean local files OR a web-search topic — read "
    "the whole request before choosing.\n"
    "- A named local file → file. Content search across files → command. "
    "URL or online research → web. Trivia → none."
)

ROUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_request",
            "description": (
                "Classify the user's request into the single specialist "
                "category that should handle it. Call this exactly once with "
                "your classification. Do NOT answer the user's question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["file", "command", "web", "none"],
                        "description": (
                            "The specialist that should handle the request: "
                            "file, command, web, or none."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "How confident you are in this classification.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One short sentence: why this specialist fits the request.",
                    },
                },
                "required": ["category", "confidence", "reason"],
            },
        },
    }
]

# Gold routing per eval prompt — the category the router SHOULD pick.
# Derived from each prompt's expected_tools and what the specialists own.
GOLD_ROUTES = {
    "P1-read-known-file": "file",
    "P2-browse-sandbox": "file",
    "P3-grep-across-files": "command",
    "P4-save-content": "file",
    "P5-fetch-url": "web",
    "P6-search-online": "web",
    "P7-search-then-fetch": "web",
    "P8-count-lines-efficient": "command",
    "P9-no-tool-trivia": "none",
    "P10-trap-web-search-vs-local": "web",
}

# ============================================================================
# Tool-subset schema builder (same schemas/descriptions as day1 "after" —
# the monolith and each specialist share identical tool definitions, so the
# only variable is the architecture around them)
# ============================================================================

_PARAM_TYPES: dict[str, dict[str, str]] = {
    "web_fetch": {"max_chars": "integer"},
    "web_search": {"max_results": "integer"},
}


def build_subset_tools(names: list[str]) -> list[dict]:
    """Build schemas for a subset of the 6 tools using day1's "after"
    descriptions (CONDITION_B)."""
    tools = []
    for name in names:
        props = {}
        type_overrides = _PARAM_TYPES.get(name, {})
        for pname, pdesc in d1._PARAM_DESCRIPTIONS.get(name, {}).items():
            ptype = type_overrides.get(pname, "string")
            props[pname] = {"type": ptype, "description": pdesc}

        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": d1.CONDITION_B_DESCRIPTIONS[name],
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": d1._PARAM_REQUIRED.get(name, []),
                },
            },
        })
    return tools


# ============================================================================
# Agent loop — parameterized version of day1's run_experiment_agent
# (custom system prompt, optional tool subset, may have zero tools)
# ============================================================================


def _run_agent(
    user_input: str,
    tools: list[dict],
    system_prompt: str,
    *,
    max_turns: int = 12,
) -> dict:
    """Run one conversation: system prompt + user input + given tools.

    Returns a dict with: answer, tool_calls, n_turns, n_tool_calls, usage,
    estimated_cost, error — the same shape as d1.run_experiment_agent.
    """
    client = _make_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    tool_calls_trace: list[dict] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for turn in range(1, max_turns + 1):
        kwargs: dict = {"model": MODEL, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = client.chat.completions.create(**kwargs)

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
                output = d1._make_error(
                    d1.ErrorType.INVALID_INPUT,
                    f"Invalid JSON arguments: {e}",
                )
                tool_calls_trace.append({
                    "name": fn_name,
                    "arguments": tc.function.arguments,
                    "error": f"JSON parse: {e}",
                })
            else:
                try:
                    fn = d1.TOOL_FUNCTIONS[fn_name]
                    output = fn(args)
                except Exception as e:
                    output = d1._make_error(d1.ErrorType.UNKNOWN, str(e))

                tool_calls_trace.append({
                    "name": fn_name,
                    "arguments": args,
                    "error": None,
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


def _usage_sum(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in a}


def _routing_failure(mechanism: str, message: str, usage: dict) -> dict:
    """Trace for a run where the router itself failed to classify."""
    return {
        "answer": "",
        "tool_calls": [],
        "n_turns": 0,
        "n_tool_calls": 0,
        "usage": dict(usage),
        "usage_router": dict(usage),
        "usage_specialist": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "estimated_cost": _estimate_cost(
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        ),
        "router": {"error": message, "mechanism": mechanism},
        "error": "routing_error",
    }


def run_router(user_input: str) -> dict:
    """Router architecture: forced classify_request call → specialist.

    Stage 1 — routing call. tool_choice forces classify_request, which is
    the ONLY tool in the router's schema, so the model's first output MUST
    be that call. Some OpenAI-compatible gateways reject the single-function
    form of tool_choice; fall back to tool_choice="required" (still forced,
    still the only tool available) and record which mechanism was used.

    Stage 2 — dispatch. A fresh conversation per specialist: specialist
    system prompt, only that specialty's tools, same user request.
    """
    client = _make_client()

    # ---- Stage 1: forced classify_request ---------------------------------
    # Forcing a single-function tool_choice is exactly the day's subject —
    # but it only works where the platform allows it. DashScope qwen3.x
    # rejects tool_choice entirely while thinking is enabled (HTTP 400:
    # "The tool_choice parameter does not support being set to required or
    # object in thinking mode"). Attempt, in order:
    #   1. forced single-function tool_choice with thinking disabled
    #      (enable_thinking=False via extra_body — documented fix)
    #   2. forced single-function tool_choice with default settings
    #      (for gateways that never enable thinking by default)
    #   3. prompt-only forcing: no tool_choice at all. classify_request is
    #      still the ONLY tool, so a refusal to call it is a routing
    #      failure — real router behavior, scored as such.
    # The mechanism actually used is recorded in the trace.
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": user_input},
    ]
    attempts = [
        ("forced_function_no_thinking", {
            "tool_choice": {"type": "function", "function": {"name": "classify_request"}},
            "extra_body": {"enable_thinking": False},
        }),
        ("forced_function", {
            "tool_choice": {"type": "function", "function": {"name": "classify_request"}},
        }),
        ("prompt_only", {}),
    ]
    first_error = None
    response = None
    for mechanism, extra in attempts:
        try:
            response = client.chat.completions.create(
                model=MODEL, tools=ROUTER_TOOLS, messages=messages, **extra
            )
            break
        except Exception as e:
            if first_error is None:
                first_error = str(e)
            continue
    if response is None:
        raise RuntimeError(f"all router-call attempts failed: {first_error}")

    usage_router = {
        "prompt_tokens": response.usage.prompt_tokens or 0 if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens or 0 if response.usage else 0,
        "total_tokens": response.usage.total_tokens or 0 if response.usage else 0,
    }

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        return _routing_failure(mechanism, "no tool call returned despite tool_choice", usage_router)

    tc = tool_calls[0]
    if tc.function.name != "classify_request":
        return _routing_failure(
            mechanism, f"unexpected first call: {tc.function.name}", usage_router
        )

    try:
        args = json.loads(tc.function.arguments)
        category = args["category"]
        confidence = args.get("confidence", "?")
        reason = args.get("reason", "")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return _routing_failure(mechanism, f"bad classify_request args: {e}", usage_router)

    if category not in SPECIALIST_TOOLS:
        return _routing_failure(mechanism, f"unknown category: {category!r}", usage_router)

    # ---- Stage 2: dispatch to the specialist -------------------------------
    spec_tools = build_subset_tools(SPECIALIST_TOOLS[category])
    spec_trace = _run_agent(user_input, spec_tools, SPECIALIST_SYSTEMS[category])

    return {
        "answer": spec_trace["answer"],
        "tool_calls": spec_trace["tool_calls"],
        "n_turns": spec_trace["n_turns"],
        "n_tool_calls": spec_trace["n_tool_calls"],
        "usage": _usage_sum(usage_router, spec_trace["usage"]),
        "usage_router": dict(usage_router),
        "usage_specialist": dict(spec_trace["usage"]),
        "estimated_cost": _estimate_cost(
            usage_router["prompt_tokens"] + spec_trace["usage"]["prompt_tokens"],
            usage_router["completion_tokens"] + spec_trace["usage"]["completion_tokens"],
        ),
        "router": {
            "category": category,
            "confidence": confidence,
            "reason": reason[:200],
            "mechanism": mechanism,
            "first_error": first_error[:200] if first_error else None,
        },
        "error": spec_trace["error"],
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
        help="Prompt IDs to run — full IDs or prefixes, e.g. P1 or P1-read-known-file (default: all 10).",
    )
    parser.add_argument(
        "--conditions", type=str, nargs="*", default=["monolith", "router"],
        help="Conditions to run: monolith and/or router (default: both).",
    )
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    prompts = d1.EVAL_PROMPTS
    if args.prompts:
        prompts = [
            p for p in d1.EVAL_PROMPTS
            if any(
                # bare segment token ("P1") matches the ID's first segment
                # exactly; a token with "-" matches as a full-ID prefix
                p["id"] == tok
                or (p["id"].startswith(tok) if "-" in tok else p["id"].split("-")[0] == tok)
                for tok in args.prompts
            )
        ]
        if not prompts:
            raise SystemExit(f"No prompts matched: {args.prompts}")

    conditions = args.conditions
    for c in conditions:
        if c not in ("monolith", "router"):
            raise SystemExit(f"Unknown condition: {c} (choose from monolith, router)")
    monolith_tools = d1.build_tools("after")  # day1 condition B = current best

    print(f"Model: {MODEL}")
    print(f"Prompts: {len(prompts)}  |  Runs per cell: {args.runs}")
    print(f"Conditions: {conditions}")
    print(f"Total runs: {len(prompts) * len(conditions) * args.runs}"
          f"  (router runs = 2 API conversations each)")
    print(f"Sandbox:  {SANDBOX_ROOT.resolve()}")
    print()
    print("Specialists:")
    for cat, tools in SPECIALIST_TOOLS.items():
        print(f"  {cat:<8} → {', '.join(tools) if tools else '(no tools)'}")
    print()

    # ---- Run all cells ------------------------------------------------------
    all_results: list[dict] = []

    per_cond_should_call: dict[str, list[float]] = {c: [] for c in conditions}
    per_cond_wrong_tool: dict[str, list[float]] = {c: [] for c in conditions}
    per_cond_exact: dict[str, list[int]] = {c: [] for c in conditions}
    per_cond_tokens: dict[str, list[int]] = {c: [] for c in conditions}
    per_cond_costs: dict[str, list[float]] = {c: [] for c in conditions}
    per_cond_prompt_exact: dict[str, dict[str, int]] = {c: {} for c in conditions}
    per_cond_prompt_total: dict[str, dict[str, int]] = {c: {} for c in conditions}

    # Router-only accumulators
    route_matrix: Counter = Counter()          # (gold, predicted)
    route_correct_total = 0
    route_runs = 0
    route_errors = 0
    route_mechanisms: Counter = Counter()
    per_prompt_route: dict[str, dict[str, int]] = {p["id"]: {"correct": 0, "total": 0} for p in prompts}
    conf_buckets: dict[str, dict[str, int]] = {c: {"correct": 0, "total": 0} for c in ["high", "medium", "low", "?"]}
    cascade: dict[str, list[int]] = {"correct_route": [], "misroute": []}  # exact-match per bucket
    router_token_split: dict[str, list[int]] = {"router": [], "specialist": []}

    for cond_label in conditions:
        print(f"{'=' * 70}")
        print(f"CONDITION: {cond_label.upper()}")
        print(f"{'=' * 70}")
        print()

        for prompt in prompts:
            prompt_id = prompt["id"]
            expected = prompt["expected_tools"]
            gold = GOLD_ROUTES[prompt_id]
            per_cond_prompt_exact[cond_label][prompt_id] = 0
            per_cond_prompt_total[cond_label][prompt_id] = 0

            for run_idx in range(1, args.runs + 1):
                try:
                    if cond_label == "monolith":
                        trace = d1.run_experiment_agent(prompt["prompt"], monolith_tools)
                    else:
                        trace = run_router(prompt["prompt"])
                except Exception as e:
                    print(f"  [{cond_label}] {prompt_id} run {run_idx}: API ERROR — {e}")
                    all_results.append({
                        "condition": cond_label,
                        "prompt_id": prompt_id,
                        "run": run_idx,
                        "api_error": str(e),
                    })
                    continue

                # Routing failures are real router behavior: count as failed.
                if trace.get("error") == "routing_error":
                    score = {
                        "called": [], "expected": sorted(expected),
                        "should_call_rate": 0.0, "wrong_tool_rate": 0.0,
                        "exact_match": False,
                        "missed": sorted(expected), "extra": [],
                        "n_calls": 0, "n_turns": 0,
                    }
                else:
                    score = d1.score_run(trace, expected)

                all_results.append({
                    "condition": cond_label,
                    "prompt_id": prompt_id,
                    "run": run_idx,
                    "expected_tools": expected,
                    **score,
                    "answer": trace["answer"][:300],
                    "tool_call_details": trace["tool_calls"],
                    "usage": trace["usage"],
                    "estimated_cost": trace["estimated_cost"],
                    "router": trace.get("router"),
                    "usage_router": trace.get("usage_router"),
                    "usage_specialist": trace.get("usage_specialist"),
                    "error": trace.get("error"),
                })

                per_cond_should_call[cond_label].append(score["should_call_rate"])
                per_cond_wrong_tool[cond_label].append(score["wrong_tool_rate"])
                per_cond_exact[cond_label].append(1 if score["exact_match"] else 0)
                per_cond_tokens[cond_label].append(trace["usage"]["total_tokens"])
                per_cond_costs[cond_label].append(trace["estimated_cost"])
                per_cond_prompt_exact[cond_label][prompt_id] += 1 if score["exact_match"] else 0
                per_cond_prompt_total[cond_label][prompt_id] += 1

                # Router-only bookkeeping
                route_suffix = ""
                if cond_label == "router":
                    r = trace.get("router") or {}
                    route_runs += 1
                    if trace.get("error") == "routing_error":
                        route_errors += 1
                        route_suffix = f"  ROUTE=ERROR({r.get('error', '')[:40]})"
                    else:
                        pred = r.get("category", "?")
                        route_matrix[(gold, pred)] += 1
                        per_prompt_route[prompt_id]["total"] += 1
                        correct = pred == gold
                        if correct:
                            route_correct_total += 1
                            per_prompt_route[prompt_id]["correct"] += 1
                            cascade["correct_route"].append(1 if score["exact_match"] else 0)
                        else:
                            cascade["misroute"].append(1 if score["exact_match"] else 0)
                        conf_buckets[r.get("confidence", "?")]["total"] += 1
                        if correct:
                            conf_buckets[r.get("confidence", "?")]["correct"] += 1
                        mark = "✓" if correct else "✗"
                        route_suffix = (
                            f"  route={pred}(gold {gold}) {mark}"
                            f"  conf={r.get('confidence', '?')}"
                        )
                    if trace.get("usage_router"):
                        router_token_split["router"].append(trace["usage_router"]["total_tokens"])
                    if trace.get("usage_specialist"):
                        router_token_split["specialist"].append(trace["usage_specialist"]["total_tokens"])
                    if trace.get("router", {}).get("mechanism"):
                        route_mechanisms[trace["router"]["mechanism"]] += 1

                icon = "✅" if score["exact_match"] else "❌"
                call_str = ", ".join(score["called"]) if score["called"] else "(none)"
                exp_str = ", ".join(score["expected"]) if score["expected"] else "(none)"
                extra_info = ""
                if score["missed"]:
                    extra_info += f"  missed: {score['missed']}"
                if score["extra"]:
                    extra_info += f"  extra: {score['extra']}"
                print(
                    f"  [{icon}] {prompt_id:<28} run {run_idx}/{args.runs}"
                    f"  called=[{call_str}]  expected=[{exp_str}]{extra_info}{route_suffix}"
                )

            n_exact = per_cond_prompt_exact[cond_label][prompt_id]
            n_total = per_cond_prompt_total[cond_label][prompt_id]
            rate = n_exact / n_total if n_total else 0
            mark = "✅" if rate == 1.0 else ("❌" if rate == 0.0 else "⚠️")
            print(f"  {mark} {prompt_id}: {n_exact}/{n_total} exact-match ({rate:.0%})")
        print()

    # =========================================================================
    # Summary tables
    # =========================================================================
    def _mean_sd(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        return mean, sd

    print(f"{'=' * 70}")
    print("TOOL-SELECTION BY CONDITION  (monolith = one agent, all 6 tools)")
    print(f"{'=' * 70}")
    header = f"{'metric':<22}" + "".join(f"{c:>14}" for c in conditions)
    print(header)
    print("-" * len(header))
    for metric, data, better in [
        ("Should-call rate", per_cond_should_call, "higher"),
        ("Wrong-tool rate", per_cond_wrong_tool, "lower"),
        ("Exact-match rate", per_cond_exact, "higher"),
    ]:
        row = f"{metric:<22}"
        for c in conditions:
            mean, sd = _mean_sd(data[c])
            row += f"{mean:>13.3f}±{sd:<6.3f}"
        print(f"{row}  ({better})")
    n_scored = len(per_cond_exact[conditions[0]])
    wins_row = f"{'exact-match wins':<22}"
    for c in conditions:
        wins_row += f"{sum(per_cond_exact[c]):>14}"
    print(f"{wins_row}  (of {n_scored} runs each)")

    # ---- Routing accuracy ----------------------------------------------------
    print(f"\n{'=' * 70}")
    print("ROUTING (router condition only)")
    print(f"{'=' * 70}")
    if route_runs:
        acc = route_correct_total / route_runs
        print(f"  routing accuracy: {route_correct_total}/{route_runs} = {acc:.0%}"
              f"  |  routing errors: {route_errors}")
        print(f"  tool_choice mechanism: {dict(route_mechanisms)}")

        # Confusion matrix (gold rows × predicted cols)
        cats = ["file", "command", "web", "none"]
        print(f"\n  confusion matrix (gold \\ predicted):")
        print(f"  {'gold':<8}" + "".join(f"{c:>10}" for c in cats) + f"{'err':>6}")
        for g in cats:
            row = f"  {g:<8}"
            for p in cats:
                row += f"{route_matrix.get((g, p), 0):>10}"
            row += f"{'—':>6}"
            print(row)
        print(f"  {'(err)':<8}{route_errors:>10}")

        # Per-prompt routing
        print("\n  per-prompt routing accuracy:")
        for prompt in prompts:
            pid = prompt["id"]
            n_c = per_prompt_route[pid]["correct"]
            n_t = per_prompt_route[pid]["total"]
            rate = n_c / n_t if n_t else 0.0
            mark = "✅" if rate == 1.0 else ("❌" if rate == 0.0 else "⚠️")
            print(f"    {mark} {pid:<28} {n_c}/{n_t} → gold {GOLD_ROUTES[pid]}")

        # Confidence vs. accuracy
        print("\n  routing accuracy by confidence:")
        for conf in ["high", "medium", "low", "?"]:
            b = conf_buckets[conf]
            if b["total"]:
                print(f"    {conf:<7} {b['correct']}/{b['total']} = {b['correct'] / b['total']:.0%}")

        # Misroute cascade
        c_vals, m_vals = cascade["correct_route"], cascade["misroute"]
        if c_vals and m_vals:
            c_rate = sum(c_vals) / len(c_vals)
            m_rate = sum(m_vals) / len(m_vals)
            print(f"\n  misroute cascade — task exact-match rate when:")
            print(f"    routed correctly: {sum(c_vals)}/{len(c_vals)} = {c_rate:.0%}")
            print(f"    misrouted:        {sum(m_vals)}/{len(m_vals)} = {m_rate:.0%}")
        elif m_vals:
            print(f"\n  misroute cascade — misrouted runs only: "
                  f"{sum(m_vals)}/{len(m_vals)} = {sum(m_vals) / len(m_vals):.0%} exact")
    else:
        print("  (no router runs)")

    # ---- Per-prompt exact-match comparison ----------------------------------
    print(f"\n{'=' * 70}")
    print(f"PER-PROMPT EXACT-MATCH COMPARISON  (out of {args.runs} runs each)")
    print(f"{'=' * 70}")
    header = f"{'prompt':<28}" + "".join(f"{c:>10}" for c in conditions)
    if len(conditions) == 2:
        header += f"{'delta':>10}"
    header += f"{'gold route':<12}{'category':<18}"
    print(header)
    print("-" * len(header))
    for prompt in prompts:
        pid = prompt["id"]
        rates = []
        for c in conditions:
            n = per_cond_prompt_exact[c].get(pid, 0)
            total = per_cond_prompt_total[c].get(pid, 1)
            rates.append(n / total if total else 0)
        row = f"{pid:<28}" + "".join(f"{rate:>9.0%}" for rate in rates)
        if len(conditions) == 2:
            row += f"{rates[1] - rates[0]:>+10.0%}"
        row += f"{GOLD_ROUTES[pid]:<12}{prompt['category']:<18}"
        print(row)

    # ---- Tokens and cost ------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("TOKENS AND COST BY CONDITION  (mean per run)")
    print(f"{'=' * 70}")
    for cond_label in conditions:
        tokens = per_cond_tokens[cond_label]
        costs = per_cond_costs[cond_label]
        if tokens:
            t_mean, t_sd = _mean_sd(tokens)
            c_mean = sum(costs) / len(costs) if costs else 0.0
            print(f"  {cond_label:<10} {t_mean:>8,.0f} ± {t_sd:>7,.0f} tokens/run"
                  f"  |  ${c_mean:.5f}/run  (n={len(tokens)})")
    if router_token_split["router"] and router_token_split["specialist"]:
        r_mean = sum(router_token_split["router"]) / len(router_token_split["router"])
        s_mean = sum(router_token_split["specialist"]) / len(router_token_split["specialist"])
        print(f"\n  router-run token split: routing call {r_mean:,.0f} + "
              f"specialist {s_mean:,.0f} = {r_mean + s_mean:,.0f} "
              f"({r_mean / (r_mean + s_mean):.0%} routing overhead)")

    # ---- Most common wrong-tool calls ----------------------------------------
    print(f"\n{'=' * 70}")
    print("MOST COMMON WRONG-TOOL CALLS  (extra tools that were not expected)")
    print(f"{'=' * 70}")
    for cond_label in conditions:
        wrong_counter: Counter = Counter()
        for r in all_results:
            if r.get("condition") == cond_label and "api_error" not in r:
                for extra_tool in r.get("extra", []):
                    wrong_counter[extra_tool] += 1
        if wrong_counter:
            print(f"\n  {cond_label}:")
            for tool, count in wrong_counter.most_common():
                print(f"    {tool:<28} {count:>3} times called unnecessarily")
        else:
            print(f"\n  {cond_label}: (none — no wrong-tool calls)")

    print(f"\n{'=' * 70}")
    print("MOST COMMON MISSED TOOLS  (expected tools that were NOT called)")
    print(f"{'=' * 70}")
    for cond_label in conditions:
        missed_counter: Counter = Counter()
        for r in all_results:
            if r.get("condition") == cond_label and "api_error" not in r:
                for missed_tool in r.get("missed", []):
                    missed_counter[missed_tool] += 1
        if missed_counter:
            print(f"\n  {cond_label}:")
            for tool, count in missed_counter.most_common():
                print(f"    {tool:<28} {count:>3} times missed")
        else:
            print(f"\n  {cond_label}: (none — all expected tools were called)")

    # ---- Delta summary (only when both conditions ran) ------------------------
    if len(conditions) == 2:
        m_label, r_label = conditions
        print(f"\n{'=' * 70}")
        print(f"DELTA: {m_label} → {r_label}")
        print(f"{'=' * 70}")
        for metric, data, better in [
            ("Should-call rate", per_cond_should_call, "higher"),
            ("Wrong-tool rate", per_cond_wrong_tool, "lower"),
            ("Exact-match rate", per_cond_exact, "higher"),
        ]:
            m_vals = data[m_label]
            r_vals = data[r_label]
            if m_vals and r_vals:
                m_mean = sum(m_vals) / len(m_vals)
                r_mean = sum(r_vals) / len(r_vals)
                delta = r_mean - m_mean
                direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                improved = (
                    (better == "higher" and delta > 0)
                    or (better == "lower" and delta < 0)
                )
                verdict = "✅ improved" if improved else ("⚠️  worsened" if delta != 0 else "→ unchanged")
                print(f"  {metric:<22} {m_mean:.3f} → {r_mean:.3f}  ({delta:+.3f} {direction})  {verdict}")
        if per_cond_tokens[m_label] and per_cond_tokens[r_label]:
            m_t = sum(per_cond_tokens[m_label]) / len(per_cond_tokens[m_label])
            r_t = sum(per_cond_tokens[r_label]) / len(per_cond_tokens[r_label])
            print(f"  {'tokens/run':<22} {m_t:,.0f} → {r_t:,.0f}  ({r_t - m_t:+,.0f})")
        if per_cond_costs[m_label] and per_cond_costs[r_label]:
            m_c = sum(per_cond_costs[m_label]) / len(per_cond_costs[m_label])
            r_c = sum(per_cond_costs[r_label]) / len(per_cond_costs[r_label])
            print(f"  {'$/run':<22} ${m_c:.5f} → ${r_c:.5f}  (${r_c - m_c:+.5f})")

    # ---- Save results --------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _THIS_DIR / f"day5_results_{timestamp}.json"

    summary = {
        "experiment": "week05_day5_router_vs_monolith",
        "model": MODEL,
        "runs_per_cell": args.runs,
        "conditions": conditions,
        "prompts": [p["id"] for p in prompts],
        "specialists": SPECIALIST_TOOLS,
        "gold_routes": {pid: GOLD_ROUTES[pid] for pid in [p["id"] for p in prompts]},
        "exact_match_rate": {
            c: {"rate": sum(v) / len(v) if v else 0, "n": len(v)}
            for c, v in per_cond_exact.items()
        },
        "should_call_rate": {
            c: {"mean": sum(v) / len(v) if v else 0, "n": len(v)}
            for c, v in per_cond_should_call.items()
        },
        "wrong_tool_rate": {
            c: {"mean": sum(v) / len(v) if v else 0, "n": len(v)}
            for c, v in per_cond_wrong_tool.items()
        },
        "per_prompt_exact": {
            pid: {
                **{c: per_cond_prompt_exact[c].get(pid, 0) for c in conditions},
                "total": args.runs,
            }
            for pid in [p["id"] for p in prompts]
        },
        "routing": {
            "accuracy": route_correct_total / route_runs if route_runs else None,
            "runs": route_runs,
            "errors": route_errors,
            "mechanisms": dict(route_mechanisms),
            "matrix": {f"{g}->{p}": n for (g, p), n in route_matrix.items()},
            "per_prompt": {
                pid: {
                    "correct": per_prompt_route[pid]["correct"],
                    "total": per_prompt_route[pid]["total"],
                    "gold": GOLD_ROUTES[pid],
                }
                for pid in [p["id"] for p in prompts]
            },
            "confidence_buckets": conf_buckets,
            "misroute_cascade": {
                "correct_route_exact": sum(cascade["correct_route"]) / len(cascade["correct_route"])
                if cascade["correct_route"] else None,
                "misroute_exact": sum(cascade["misroute"]) / len(cascade["misroute"])
                if cascade["misroute"] else None,
                "n_correct_route": len(cascade["correct_route"]),
                "n_misroute": len(cascade["misroute"]),
            },
        },
        "token_usage": {
            c: {
                "mean": sum(v) / len(v) if v else 0,
                "total": sum(v),
                "n": len(v),
            }
            for c, v in per_cond_tokens.items()
        },
        "router_token_split": {
            "router_mean": sum(router_token_split["router"]) / len(router_token_split["router"])
            if router_token_split["router"] else 0,
            "specialist_mean": sum(router_token_split["specialist"]) / len(router_token_split["specialist"])
            if router_token_split["specialist"] else 0,
        },
        "cost_per_run": {
            c: {
                "mean": sum(v) / len(v) if v else 0,
                "n": len(v),
            }
            for c, v in per_cond_costs.items()
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "prompt_details": [
                    {"id": p["id"], "prompt": p["prompt"], "expected_tools": p["expected_tools"],
                     "category": p["category"], "gold_route": GOLD_ROUTES[p["id"]],
                     "note": p["note"]}
                    for p in prompts
                ],
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
