"""Week 5 Day 4: Worked examples in tool descriptions — re-measure.

Experiment question:
    Does adding concrete worked examples (few-shot input→action pairs,
    including "do NOT do this" counter-examples) to the descriptions of the
    two most-misused tools reduce their wrong-call rate?

Which two tools, and why (from the day1 + day3 result data):
    Wrong-call counts across both experiments:

        list_dir     22   (7 in day1, 15 in day3) — called as unnecessary
                          pre-exploration on P3 "search all text files for
                          the word X", where a single grep would do.
        web_fetch    20   (all day1; still 10 under the day1 "after"
                          descriptions) — after a successful web_search the
                          model fetches 2-5 result URLs even though the
                          search snippets already answer the question
                          (P6, P10).
        write_file   19   (same P3 over-engineering pattern — close third,
                          not treated here to keep the experiment variable
                          to exactly two tools).

    Note: day3 couldn't exhibit the web_fetch misuse because web_fetch was
    not in that experiment's tool sets.

Method:
    1. Baseline condition = the day1 "after" descriptions (explicit
       when-to-call / do-NOT-call clauses) — the current state of the tools.
    2. Examples condition = baseline, plus a "**Worked examples:**" section
       appended to list_dir and web_fetch ONLY. Everything else is identical.
    3. Re-run the same 10 tool-selection prompts, 5 runs per cell.
    4. Compare wrong-tool rate, exact-match rate, and per-target-tool
       wrong-call counts. Also verify the examples do not suppress
       LEGITIMATE use of the two tools (P2 → list_dir, P5/P7 → web_fetch).

    Default: 2 conditions × 5 runs × 10 prompts = 100 agent runs

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week05/day4_worked_examples.py
    python experiments/week05/day4_worked_examples.py --runs 3 --prompts P3 P6  # quick test
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

# Reuse the day1 harness: agent loop, scoring, prompts, and the "after"
# (current) description set that becomes this experiment's baseline.
import day1_tool_descriptions as d1  # noqa: E402

from agent import MODEL, SANDBOX_ROOT  # noqa: E402
from openai import OpenAI  # noqa: E402

# ============================================================================
# Descriptions — baseline (day1 "after") vs. baseline + worked examples
# ============================================================================

BASELINE_DESCRIPTIONS = d1.CONDITION_B_DESCRIPTIONS


def _with_examples(base: str, examples: str) -> str:
    return base.rstrip() + "\n\n" + examples


# ---------------------------------------------------------------------------
# Worked examples for list_dir.
# Targeted failure mode (day1 P3 ×7, day3 P3 ×15): the model calls list_dir
# as "exploration" before a task that a single direct tool call solves
# (content search → grep; known path → read_file).
# ---------------------------------------------------------------------------
_LIST_DIR_EXAMPLES = (
    "**Worked examples:**\n"
    "- User: \"What files and directories are in the sandbox?\" → "
    "list_dir(\".\") ✅\n"
    "- User: \"Read 'subdir/data.json'.\" → read_file(\"subdir/data.json\") "
    "directly — the path is known. Do NOT list_dir first. ❌ "
    "list_dir → read_file\n"
    "- User: \"Search all .txt/.md/.py files for the word 'sandbox'.\" → "
    "run_command with grep — it searches file CONTENTS across many files in "
    "one step. Do NOT call list_dir first; browsing adds a wasted turn "
    "before the real work. ❌ list_dir → grep\n"
    "- User: \"Do I have any .md files?\" → list_dir(\".\") and filter the "
    "names ✅"
)

# ---------------------------------------------------------------------------
# Worked examples for web_fetch.
# Targeted failure mode (day1 P6 ×10, P10 ×10): after web_search returns
# snippets that already answer the question, the model fetches 2-5 result
# URLs "to verify". The examples teach: fetch only when the user asked for
# page-level detail, and then fetch ONE page, not five.
# ---------------------------------------------------------------------------
_WEB_FETCH_EXAMPLES = (
    "**Worked examples:**\n"
    "- User: \"Open https://example.com and summarize it.\" → "
    "web_fetch(\"https://example.com\") ✅\n"
    "- User: \"Search for 'Anthropic building effective agents', then visit "
    "the most relevant result and give me a detailed summary.\" → web_search "
    "first, then web_fetch on the SINGLE most relevant result ✅\n"
    "- User: \"Search online: what are the top 3 open-source AI agent "
    "frameworks in 2026? Give me a one-line description of each.\" → "
    "web_search only, then answer from the snippets. The snippets already "
    "name the frameworks and what they do — fetching the pages adds turns "
    "but no information. ❌ web_search → web_fetch × 5\n"
    "- User: \"Search for 'sandbox escape prevention' and tell me the top "
    "recommendations.\" → web_search only, then summarize the snippets. "
    "Fetch a page ONLY if the snippets miss a specific detail the user "
    "asked for, and fetch at most one."
)

EXAMPLES_DESCRIPTIONS = {
    **BASELINE_DESCRIPTIONS,
    "list_dir": _with_examples(BASELINE_DESCRIPTIONS["list_dir"], _LIST_DIR_EXAMPLES),
    "web_fetch": _with_examples(BASELINE_DESCRIPTIONS["web_fetch"], _WEB_FETCH_EXAMPLES),
}

CONDITIONS = {
    "baseline": BASELINE_DESCRIPTIONS,
    "examples": EXAMPLES_DESCRIPTIONS,
}

# The two tools the experiment modifies — used for targeted analysis.
TARGET_TOOLS = ["list_dir", "web_fetch"]

# ============================================================================
# Schema builder (same assembly as day1.build_tools, but takes a dict)
# ============================================================================

_TOOL_ORDER = ["read_file", "write_file", "list_dir", "run_command", "web_fetch", "web_search"]

_PARAM_TYPES: dict[str, dict[str, str]] = {
    "web_fetch": {"max_chars": "integer"},
    "web_search": {"max_results": "integer"},
}


def build_tools(descriptions: dict[str, str]) -> list[dict]:
    """Build the OpenAI tool-schema list for a descriptions dict.

    Tool implementations and parameter schemas are identical across
    conditions — only ``function.description`` differs.
    """
    tools = []
    for name in _TOOL_ORDER:
        props = {}
        for pname, pdesc in d1._PARAM_DESCRIPTIONS.get(name, {}).items():
            ptype = _PARAM_TYPES.get(name, {}).get(pname, "string")
            props[pname] = {"type": ptype, "description": pdesc}

        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": descriptions[name],
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": d1._PARAM_REQUIRED.get(name, []),
                },
            },
        })
    return tools


# ============================================================================
# Schema token cost (same method as day3 Part 1)
# ============================================================================

def measure_schema_cost(
    tools: list[dict],
    neutral_prompt: str = "What is 2 + 2? Answer in one word.",
) -> dict:
    """One API call with the tools and one without; the prompt-token
    difference is what the schemas alone cost in context, every call."""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    base = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": neutral_prompt}],
    )
    with_tools = client.chat.completions.create(
        model=MODEL,
        tools=tools,
        messages=[{"role": "user", "content": neutral_prompt}],
    )
    base_tokens = base.usage.prompt_tokens or 0
    wt_tokens = with_tools.usage.prompt_tokens or 0
    return {
        "baseline_prompt_tokens": base_tokens,
        "with_tools_prompt_tokens": wt_tokens,
        "schema_token_cost": wt_tokens - base_tokens,
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
        help="Prompt IDs to run (default: all 10).",
    )
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    prompts = d1.EVAL_PROMPTS
    if args.prompts:
        prompts = [p for p in d1.EVAL_PROMPTS if p["id"] in args.prompts]
        if not prompts:
            raise SystemExit(f"No prompts matched: {args.prompts}")

    cond_labels = list(CONDITIONS)

    print(f"Model: {MODEL}")
    print(f"Prompts: {len(prompts)}  |  Runs per cell: {args.runs}")
    print(f"Conditions: {cond_labels} (identical except {TARGET_TOOLS} descriptions)")
    print(f"Total runs: {len(prompts) * len(cond_labels) * args.runs}")
    print(f"Sandbox:  {SANDBOX_ROOT.resolve()}")
    print()

    # Description-size delta (the token cost of the examples)
    for tool in TARGET_TOOLS:
        b_len = len(BASELINE_DESCRIPTIONS[tool])
        e_len = len(EXAMPLES_DESCRIPTIONS[tool])
        print(f"  {tool}: description {b_len} → {e_len} chars (+{e_len - b_len})")
    print()

    # ---- Schema token cost per condition (paid on every API call) ----------
    schema_cost: dict[str, dict] = {}
    for cond_label in cond_labels:
        tools = build_tools(CONDITIONS[cond_label])
        try:
            schema_cost[cond_label] = measure_schema_cost(tools)
            print(
                f"  schema token cost [{cond_label}]: "
                f"{schema_cost[cond_label]['schema_token_cost']} tokens per call"
            )
        except Exception as e:
            print(f"  schema token cost [{cond_label}]: API ERROR — {e}")
            schema_cost[cond_label] = {"error": str(e)}
    print()

    # ---- Run all cells ------------------------------------------------------
    all_results: list[dict] = []

    per_cond_should_call: dict[str, list[float]] = {c: [] for c in cond_labels}
    per_cond_wrong_tool: dict[str, list[float]] = {c: [] for c in cond_labels}
    per_cond_exact: dict[str, list[int]] = {c: [] for c in cond_labels}
    per_cond_tokens: dict[str, list[int]] = {c: [] for c in cond_labels}
    per_cond_prompt_tokens: dict[str, list[int]] = {c: [] for c in cond_labels}
    per_cond_prompt_exact: dict[str, dict[str, int]] = {c: {} for c in cond_labels}
    per_cond_prompt_total: dict[str, dict[str, int]] = {c: {} for c in cond_labels}
    # Targeted analysis: {condition: {tool: {"wrong": Counter, "expected_hits": n}}}
    target_stats: dict[str, dict[str, dict]] = {
        c: {t: {"wrong": Counter(), "expected_runs": 0, "expected_called": 0} for t in TARGET_TOOLS}
        for c in cond_labels
    }

    for cond_label in cond_labels:
        tools = build_tools(CONDITIONS[cond_label])

        print(f"{'=' * 70}")
        print(f"CONDITION: {cond_label.upper()}")
        print(f"{'=' * 70}")
        print()

        for prompt in prompts:
            prompt_id = prompt["id"]
            expected = prompt["expected_tools"]
            per_cond_prompt_exact[cond_label][prompt_id] = 0
            per_cond_prompt_total[cond_label][prompt_id] = 0

            for run_idx in range(1, args.runs + 1):
                try:
                    trace = d1.run_experiment_agent(prompt["prompt"], tools)
                except Exception as e:
                    print(f"  [{cond_label}] {prompt_id} run {run_idx}: API ERROR — {e}")
                    all_results.append({
                        "condition": cond_label,
                        "prompt_id": prompt_id,
                        "run": run_idx,
                        "api_error": str(e),
                    })
                    continue

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
                    "error": trace.get("error"),
                })

                per_cond_should_call[cond_label].append(score["should_call_rate"])
                per_cond_wrong_tool[cond_label].append(score["wrong_tool_rate"])
                per_cond_exact[cond_label].append(1 if score["exact_match"] else 0)
                per_cond_tokens[cond_label].append(trace["usage"]["total_tokens"])
                per_cond_prompt_tokens[cond_label].append(trace["usage"]["prompt_tokens"])
                per_cond_prompt_exact[cond_label][prompt_id] += 1 if score["exact_match"] else 0
                per_cond_prompt_total[cond_label][prompt_id] += 1

                # Targeted analysis for the two modified tools
                for tool in TARGET_TOOLS:
                    st = target_stats[cond_label][tool]
                    if tool in score["extra"]:
                        st["wrong"][prompt_id] += 1
                    if tool in expected:
                        st["expected_runs"] += 1
                        if tool in score["called"]:
                            st["expected_called"] += 1

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
                    f"  called=[{call_str}]  expected=[{exp_str}]{extra_info}"
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
    b_label, e_label = cond_labels

    print(f"{'=' * 70}")
    print("SHOULD-CALL RATE BY CONDITION  (higher = better)")
    print(f"{'=' * 70}")
    header = f"{'condition':<14}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}"
    print(header)
    print("-" * len(header))
    cond_summary = {}
    for cond_label in cond_labels:
        rates = per_cond_should_call[cond_label]
        if rates:
            mean_r = sum(rates) / len(rates)
            sd_r = (sum((r - mean_r) ** 2 for r in rates) / len(rates)) ** 0.5
            cond_summary[cond_label] = {"mean": mean_r, "sd": sd_r}
            print(f"{cond_label:<14}{mean_r:>8.3f}{sd_r:>8.3f}{min(rates):>8.3f}{max(rates):>8.3f}")

    print(f"\n{'=' * 70}")
    print("WRONG-TOOL RATE BY CONDITION  (lower = better)")
    print(f"{'=' * 70}")
    print(header)
    print("-" * len(header))
    for cond_label in cond_labels:
        rates = per_cond_wrong_tool[cond_label]
        if rates:
            mean_r = sum(rates) / len(rates)
            sd_r = (sum((r - mean_r) ** 2 for r in rates) / len(rates)) ** 0.5
            print(f"{cond_label:<14}{mean_r:>8.3f}{sd_r:>8.3f}{min(rates):>8.3f}{max(rates):>8.3f}")

    print(f"\n{'=' * 70}")
    print("EXACT-MATCH RATE BY CONDITION  (higher = better)")
    print(f"{'=' * 70}")
    print(f"{'condition':<14}{'exact':>8}{'total':>8}{'rate':>9}")
    print("-" * 42)
    for cond_label in cond_labels:
        exacts = per_cond_exact[cond_label]
        if exacts:
            n_exact = sum(exacts)
            print(f"{cond_label:<14}{n_exact:>8}{len(exacts):>8}{n_exact / len(exacts):>8.0%}")

    # ---- Targeted analysis: the two modified tools --------------------------
    print(f"\n{'=' * 70}")
    print(f"TARGET-TOOL ANALYSIS  ({', '.join(TARGET_TOOLS)})")
    print(f"{'=' * 70}")
    for tool in TARGET_TOOLS:
        print(f"\n  {tool}:")
        for cond_label in cond_labels:
            st = target_stats[cond_label][tool]
            wrong_total = sum(st["wrong"].values())
            wrong_detail = ", ".join(f"{pid}×{n}" for pid, n in st["wrong"].most_common()) or "none"
            ret = (
                f"{st['expected_called']}/{st['expected_runs']}"
                if st["expected_runs"] else "n/a (not expected by any prompt)"
            )
            print(
                f"    {cond_label:<10} wrong calls: {wrong_total:>3} ({wrong_detail})  |  "
                f"called-when-expected: {ret}"
            )

    # ---- Per-prompt exact-match comparison ----------------------------------
    print(f"\n{'=' * 70}")
    print(f"PER-PROMPT EXACT-MATCH COMPARISON  (out of {args.runs} runs each)")
    print(f"{'=' * 70}")
    print(f"{'prompt':<28}{b_label:>10}{e_label:>10}{'delta':>10}{'category':<18}")
    print("-" * 78)
    for prompt in prompts:
        pid = prompt["id"]
        b_n = per_cond_prompt_exact[b_label].get(pid, 0)
        e_n = per_cond_prompt_exact[e_label].get(pid, 0)
        b_total = per_cond_prompt_total[b_label].get(pid, 1)
        e_total = per_cond_prompt_total[e_label].get(pid, 1)
        b_rate = b_n / b_total if b_total else 0
        e_rate = e_n / e_total if e_total else 0
        delta = e_rate - b_rate
        print(f"{pid:<28}{b_rate:>9.0%}{e_rate:>9.0%}{delta:>+10.0%}{prompt['category']:<18}")

    # ---- Token usage ---------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("TOKEN USAGE BY CONDITION  (mean ± sd)")
    print(f"{'=' * 70}")
    for cond_label in cond_labels:
        tokens = per_cond_tokens[cond_label]
        ptokens = per_cond_prompt_tokens[cond_label]
        if tokens and ptokens:
            mean_t = sum(tokens) / len(tokens)
            sd_t = (sum((t - mean_t) ** 2 for t in tokens) / len(tokens)) ** 0.5
            mean_p = sum(ptokens) / len(ptokens)
            print(f"  {cond_label:<14} total {mean_t:>8,.0f} ± {sd_t:>7,.0f}  |  prompt {mean_p:>8,.0f} tokens/run")

    # ---- Delta summary --------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"DELTA: {b_label} → {e_label}")
    print(f"{'=' * 70}")

    for metric, data, better in [
        ("Should-call rate", per_cond_should_call, "higher"),
        ("Wrong-tool rate", per_cond_wrong_tool, "lower"),
        ("Exact-match rate", per_cond_exact, "higher"),
    ]:
        b_vals = data[b_label]
        e_vals = data[e_label]
        if b_vals and e_vals:
            b_mean = sum(b_vals) / len(b_vals)
            e_mean = sum(e_vals) / len(e_vals)
            delta = e_mean - b_mean
            direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            improved = (
                (better == "higher" and delta > 0)
                or (better == "lower" and delta < 0)
            )
            verdict = "✅ improved" if improved else ("⚠️  worsened" if delta != 0 else "→ unchanged")
            print(f"  {metric:<22} {b_mean:.3f} → {e_mean:.3f}  ({delta:+.3f} {direction})  {verdict}")

    # ---- Most common wrong-tool calls ----------------------------------------
    print(f"\n{'=' * 70}")
    print("MOST COMMON WRONG-TOOL CALLS  (extra tools that were not expected)")
    print(f"{'=' * 70}")
    for cond_label in cond_labels:
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

    # ---- Save results --------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _THIS_DIR / f"day4_results_{timestamp}.json"

    summary = {
        "experiment": "week05_day4_worked_examples",
        "model": MODEL,
        "runs_per_cell": args.runs,
        "conditions": cond_labels,
        "target_tools": TARGET_TOOLS,
        "description_chars": {
            "baseline": {t: len(BASELINE_DESCRIPTIONS[t]) for t in TARGET_TOOLS},
            "examples": {t: len(EXAMPLES_DESCRIPTIONS[t]) for t in TARGET_TOOLS},
        },
        "schema_token_cost": schema_cost,
        "prompts": [p["id"] for p in prompts],
        "should_call_rate": {
            c: {"mean": sum(v) / len(v) if v else 0, "n": len(v)}
            for c, v in per_cond_should_call.items()
        },
        "wrong_tool_rate": {
            c: {"mean": sum(v) / len(v) if v else 0, "n": len(v)}
            for c, v in per_cond_wrong_tool.items()
        },
        "exact_match_rate": {
            c: {"rate": sum(v) / len(v) if v else 0, "n": len(v)}
            for c, v in per_cond_exact.items()
        },
        "per_prompt_exact": {
            pid: {
                b_label: per_cond_prompt_exact[b_label].get(pid, 0),
                e_label: per_cond_prompt_exact[e_label].get(pid, 0),
                "total": args.runs,
            }
            for pid in [p["id"] for p in prompts]
        },
        "target_tool_stats": {
            c: {
                t: {
                    "wrong_calls": dict(st["wrong"]),
                    "wrong_total": sum(st["wrong"].values()),
                    "expected_runs": st["expected_runs"],
                    "expected_called": st["expected_called"],
                }
                for t, st in target_stats[c].items()
            }
            for c in cond_labels
        },
        "token_usage": {
            c: {
                "mean": sum(v) / len(v) if v else 0,
                "total": sum(v),
                "n": len(v),
            }
            for c, v in per_cond_tokens.items()
        },
        "prompt_tokens_mean": {
            c: sum(v) / len(v) if v else 0
            for c, v in per_cond_prompt_tokens.items()
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "prompt_details": [
                    {"id": p["id"], "prompt": p["prompt"], "expected_tools": p["expected_tools"],
                     "category": p["category"], "note": p["note"]}
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
