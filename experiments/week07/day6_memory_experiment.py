"""Week 7 Day 6: does the memory scratchpad improve multi-step task success,
or just add tokens?

Experiment question (PLAN.md week 7 day 6):
    **Experiment:** does the memory scratchpad improve multi-step task
    success, or just add tokens?

Design:

    Two agent variants, ONE variable — the memory tools:
      with_memory   day-5 stack: search_knowledge + read/write/list memory,
                    system prompt describes the scratchpad pattern
      no_memory     search_knowledge only; the system prompt has NO memory
                    section and the tools list has NO memory tools
    Everything else is identical: same user prompts, same model, temp 0,
    same MAX_TURNS, same search budget.

    TASKS — three genuinely multi-step tasks where intermediate state must
    survive to the end:
      t1_three_facts_summary   3 notebook retrievals + a final summary that
                               must include every number
      t2_arithmetic_chain      5-step chained arithmetic (each step depends
                               on the previous one) — the classic scratchpad
                               shape
      t3_verify_peak_temp      retrieve a fact, RE-verify it with a second
                               search, then answer — a mini verify-and-answer
                               loop
    All three are answerable WITHOUT memory (the no_memory condition must
    have a fair chance — otherwise the experiment measures nothing).

    GRADING — normalized evidence presence over gold groups (any-of per
    group, all groups required), no LLM judge: the answers are numbers and
    facts, and day 2's lesson was that evidence graders have false
    negatives only when the answer rephrases facts — numbers don't rephrase.

    METRICS — per condition and task: success rate, mean turns, mean tool
    calls, mean memory calls (from the TRACE — day 5's lesson: answer style
    lies, traces don't), mean prompt/completion tokens, latency.

Run:
    export $(grep DASHSCOPE_API_KEY .env | xargs)
    python experiments/week07/day6_memory_experiment.py --selftest   # offline, no API
    python experiments/week07/day6_memory_experiment.py --smoke      # truncated docs
    python experiments/week07/day6_memory_experiment.py              # full run, 5 runs/cell

Outputs (in experiments/week07/):
    day6_results_{ts}.json — config, every cell (trace + usage + grade),
                             per-condition metrics
    day6_plot_{ts}.png     — success rate + tokens per task by condition
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from day1_embeddings import get_client  # noqa: E402
from day3_agent_retrieval import (  # noqa: E402
    CHAT_BASE,
    CHAT_MODEL,
    DOCS,
    DOCS_DIR,
    SEARCH_TOOL,
    build_index,
    make_search,
)
from day4_memory_tools import (  # noqa: E402
    MEMORY_ROOT,
    _has_any,
    run_agent,
)
from day5_cross_session import TOOLS5, build_tool_functions5  # noqa: E402

# ---------------------------------------------------------------------------
# Conditions — the experiment variable is the memory tools (+ their prompt).
# ---------------------------------------------------------------------------

WITH_MEMORY_SYSTEM = (
    "You are a research assistant with a personal notebook of AI agent "
    "engineering experiment notes (weekly reports week1-week6) and a "
    "persistent memory directory. The notebook is NOT in your context — "
    "search_knowledge is the only way to read it. Make AT MOST 6 "
    "search_knowledge calls per task.\n\n"
    "MEMORY — you have memory tools over the memory directory:\n"
    "- list_memory(): list what memory files exist.\n"
    "- write_memory(filepath, content, mode): store a note ('append' adds "
    "to an existing file, 'overwrite' replaces it). File paths are relative "
    "to the memory directory itself, e.g. 'scratchpad.md' (no 'memory/' "
    "prefix).\n"
    "- read_memory(filepath): read a memory file back.\n"
    "Use memory as a scratchpad: when a task has several steps, write down "
    "intermediate results, numbers, and decisions as you go, then read the "
    "notes back before giving your final answer. Files persist — write, "
    "don't just remember. Prefer short, structured markdown.\n\n"
    "Answer ONLY from what your tools return. If the tools do not contain "
    "an answer, say exactly: I don't know."
)

NO_MEMORY_SYSTEM = (
    "You are a research assistant with a personal notebook of AI agent "
    "engineering experiment notes (weekly reports week1-week6). The "
    "notebook is NOT in your context — search_knowledge is the only way "
    "to read it. Make AT MOST 6 search_knowledge calls per task.\n\n"
    "Answer ONLY from what your tools return. If the tools do not contain "
    "an answer, say exactly: I don't know."
)

CONDITIONS = [
    {"id": "with_memory", "tools": TOOLS5,
     "system": WITH_MEMORY_SYSTEM, "uses_memory": True},
    {"id": "no_memory", "tools": [SEARCH_TOOL],
     "system": NO_MEMORY_SYSTEM, "uses_memory": False},
]

MEMORY_TOOL_NAMES = {"read_memory", "write_memory", "list_memory"}

# ---------------------------------------------------------------------------
# Tasks — multi-step, answerable by both conditions.  Gold groups: any-of
# per group (normalized), ALL groups required.
# ---------------------------------------------------------------------------

TASKS = [
    {
        "id": "t1_three_facts_summary",
        "prompt": (
            "Using the notebook, find these three facts one at a time and "
            "then give a final summary with all three:\n"
            "1. In the week 1 day 6 trap_binary_file experiment, what were "
            "the wrong-tool rates of terse vs detailed tool descriptions?\n"
            "2. In the week 2 temperature sweep, what success rates did "
            "temperatures 0.0, 0.7, and 1.2 achieve?\n"
            "3. In week 5 day 1, how did rewriting the tool descriptions "
            "change token consumption?\n"
            "Your final summary must include every number."
        ),
        "gold_groups": [
            ["4/5", "4of5", "80"],       # terse wrong-tool
            ["0/5", "0of5"],             # detailed wrong-tool
            ["97.5"],                    # temp 0.0 (and 1.2)
            ["100"],                     # temp 0.7
            ["33.6"],                    # week-5 token change
        ],
    },
    {
        "id": "t2_arithmetic_chain",
        "prompt": (
            "Compute this chain step by step (each step builds on the "
            "previous one):\n"
            "a = 17 * 23\n"
            "b = a + 456 * 7\n"
            "c = b * 2\n"
            "d = c - 89 * 64\n"
            "e = d + 1234 + 5678\n"
            "Report the value of e."
        ),
        "gold_groups": [["8382"]],
    },
    {
        "id": "t3_verify_peak_temp",
        "prompt": (
            "Using the notebook: what success rate did temperature 0.7 "
            "achieve in the week 2 temperature sweep? To be safe, verify "
            "your finding with a SECOND search before answering. State "
            "which temperature had the highest success rate and its value."
        ),
        "gold_groups": [["0.7"], ["100"]],
    },
]


def grade(answer: str, gold_groups: list[list[str]]) -> dict:
    """Normalized evidence presence: every group needs at least one hit."""
    matched = [bool(_has_any(answer, g)) for g in gold_groups]
    return {"pass": all(matched), "matched": sum(matched),
            "n_groups": len(gold_groups)}


# ---------------------------------------------------------------------------
# Offline selftest — no API.  Proves the chain values, the grader, and that
# the two conditions differ ONLY in the memory tools + their prompt.
# ---------------------------------------------------------------------------

def selftest() -> None:
    checks: list[bool] = []

    def check(name: str, cond: bool) -> None:
        checks.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1. chain values (the gold for t2 must itself be correct)
    a, b = 17 * 23, 17 * 23 + 456 * 7
    c, d = b * 2, b * 2 - 89 * 64
    e = d + 1234 + 5678
    check("chain: a..e = 391, 3583, 7166, 1470, 8382",
          (a, b, c, d, e) == (391, 3583, 7166, 1470, 8382))

    # 2. grader: all-groups-required, any-of per group
    ans = "terse 4/5, detailed 0/5; temps 97.5, 100, 97.5; +33.6%"
    g = grade(ans, TASKS[0]["gold_groups"])
    check("grader: full answer passes all 5 groups",
          g["pass"] and g["matched"] == 5)
    ans_missing = "terse 4/5, detailed 0/5; +33.6%"
    check("grader: missing temp facts fails",
          grade(ans_missing, TASKS[0]["gold_groups"])["pass"] is False)
    check("grader: any-of works (4 of 5 matches the 4/5 group)",
          grade("the rate was 4 of 5", [["4/5", "4of5"]])["pass"] is True)
    check("grader: 0% must NOT match via the 0 substring",
          grade("100% success", [["0/5", "0of5"]])["pass"] is False)

    # 3. the conditions differ ONLY in memory (tools + prompt mention)
    check("no_memory tools = search_knowledge only",
          [t["function"]["name"] for t in CONDITIONS[1]["tools"]]
          == ["search_knowledge"])
    check("with_memory tools = the day-5 stack",
          [t["function"]["name"] for t in CONDITIONS[0]["tools"]]
          == ["search_knowledge", "read_memory", "write_memory",
              "list_memory"])
    check("no_memory system prompt has no memory section",
          "memory" not in NO_MEMORY_SYSTEM.lower())
    check("with_memory system prompt describes the scratchpad",
          "scratchpad" in WITH_MEMORY_SYSTEM.lower())
    check("both conditions share the same search budget line",
          "AT MOST 6" in WITH_MEMORY_SYSTEM and "AT MOST 6" in NO_MEMORY_SYSTEM)

    # 4. task ids + prompts reference distinct tasks
    check("three tasks in order",
          [t["id"] for t in TASKS] == ["t1_three_facts_summary",
                                       "t2_arithmetic_chain",
                                       "t3_verify_peak_temp"])

    print(f"\nselftest: {sum(checks)}/{len(checks)} checks passed")
    if not all(checks):
        sys.exit(1)


# ---------------------------------------------------------------------------
# Plot: success rate + tokens per task by condition.
# ---------------------------------------------------------------------------

# two categorical hues: the memory condition is the entity under test; the
# no-memory baseline is muted gray (identity: "the absence", as day 2 used
# for no_retrieval)
COND_COLORS = {"with_memory": "#2a78d6", "no_memory": "#9CA3AF"}


def plot_results(results: dict, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    m = results["metrics"]
    tasks = [t["id"].replace("t1_three_facts_summary", "t1\n3 facts")
                        .replace("t2_arithmetic_chain", "t2\nchain")
                        .replace("t3_verify_peak_temp", "t3\nverify")
             for t in TASKS]
    conds = [c["id"] for c in CONDITIONS]
    colors = [COND_COLORS[c] for c in conds]
    x = np.arange(len(tasks))
    width = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ink, grid = "#0b0b0b", "#d8d6d0"

    for i, cond in enumerate(conds):
        accs = [m[cond]["per_task"][t["id"]]["success_rate"] for t in TASKS]
        toks = [m[cond]["per_task"][t["id"]]["mean_total_tokens"] for t in TASKS]
        off = (i - 0.5) * width
        ax1.bar(x + off, accs, width, color=colors[i], label=cond)
        for xi, v in zip(x + off, accs):
            ax1.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=ink)
        ax2.bar(x + off, toks, width, color=colors[i], label=cond)
        for xi, v in zip(x + off, toks):
            ax2.text(xi, v + 20, f"{v / 1000:.1f}K", ha="center", fontsize=8, color=ink)

    for ax, title, ylab in [
        (ax1, "Success rate by task and condition", "success rate"),
        (ax2, "Mean tokens per cell by task and condition", "total tokens (gen)"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=grid, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\nplot -> {out_path}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Week 7 Day 6: memory scratchpad vs no memory on multi-step tasks")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline checks only (no API calls)")
    ap.add_argument("--smoke", action="store_true",
                    help="truncate each doc to 10k chars (quick sanity pass)")
    ap.add_argument("--runs", type=int, default=5,
                    help="runs per (condition, task) cell (default 5)")
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    docs = {}
    for name in DOCS:
        path = DOCS_DIR / name
        if not path.exists():
            sys.exit(f"missing document: {path}")
        docs[name] = path.read_text()[:10_000 if args.smoke else None]
    total_chars = sum(len(t) for t in docs.values())
    n_cells = len(TASKS) * len(CONDITIONS) * args.runs
    print(f"docs: {len(docs)} files, {total_chars:,} chars (smoke={args.smoke}); "
          f"tasks: {len(TASKS)}; conditions: {[c['id'] for c in CONDITIONS]}; "
          f"runs={args.runs} -> {n_cells} cells; chat={CHAT_MODEL}")

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
    tool_fns = build_tool_functions5(search_fn)

    cells = []
    for task in TASKS:
        for run_i in range(args.runs):
            for cond in CONDITIONS:
                if MEMORY_ROOT.exists():
                    shutil.rmtree(MEMORY_ROOT)
                MEMORY_ROOT.mkdir(parents=True)

                cell_t0 = time.time()
                res = run_agent(chat_client, task["prompt"], tool_fns,
                                system=cond["system"], tools=cond["tools"])
                elapsed = time.time() - cell_t0

                g = grade(res["answer"], task["gold_groups"])
                mem_calls = sum(1 for c in res["tool_calls"]
                                if c["name"] in MEMORY_TOOL_NAMES)
                n_tools = len(res["tool_calls"])
                mark = "PASS" if g["pass"] else "fail"
                print(f"  [{run_i}] {cond['id']:<12} {task['id']:<24} {mark}  "
                      f"{res['n_turns']}t/{n_tools}c/{mem_calls}mem  "
                      f"({res['usage']['prompt'] + res['usage']['completion']:,} tok, "
                      f"{elapsed:.0f}s)")

                cells.append({
                    "condition": cond["id"], "task_id": task["id"],
                    "run": run_i,
                    "prompt": task["prompt"],
                    "answer": res["answer"],
                    "grade": g,
                    "n_turns": res["n_turns"],
                    "tool_calls": res["tool_calls"],
                    "n_tool_calls": n_tools,
                    "n_memory_calls": mem_calls,
                    "usage": res["usage"],
                    "elapsed_s": round(elapsed, 1),
                    "error": res.get("error"),
                })

    print(f"\nelapsed {time.time() - t0:.0f}s")

    # per-condition metrics
    metrics = {}
    for cond in CONDITIONS:
        cid = cond["id"]
        rs = [c for c in cells if c["condition"] == cid]
        metrics[cid] = {"n_cells": len(rs),
                        "cells_with_memory_use": sum(1 for c in rs if c["n_memory_calls"] > 0),
                        "mean_memory_calls": round(sum(c["n_memory_calls"] for c in rs) / len(rs), 2),
                        "per_task": {}}
        for task in TASKS:
            tr = [c for c in rs if c["task_id"] == task["id"]]
            metrics[cid]["per_task"][task["id"]] = {
                "success_rate": round(sum(c["grade"]["pass"] for c in tr) / len(tr), 3),
                "mean_turns": round(sum(c["n_turns"] for c in tr) / len(tr), 2),
                "mean_tool_calls": round(sum(c["n_tool_calls"] for c in tr) / len(tr), 2),
                "mean_memory_calls": round(sum(c["n_memory_calls"] for c in tr) / len(tr), 2),
                "mean_prompt_tokens": round(sum(c["usage"]["prompt"] for c in tr) / len(tr), 1),
                "mean_completion_tokens": round(sum(c["usage"]["completion"] for c in tr) / len(tr), 1),
                "mean_total_tokens": round(sum(c["usage"]["prompt"] + c["usage"]["completion"] for c in tr) / len(tr), 1),
                "mean_elapsed_s": round(sum(c["elapsed_s"] for c in tr) / len(tr), 1),
            }

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment": "week07_day6_memory_experiment",
        "timestamp": ts,
        "config": {
            "docs": DOCS, "doc_chars": {d: len(t) for d, t in docs.items()},
            "runs": args.runs, "smoke": args.smoke,
            "conditions": {c["id"]: {"tools": [t["function"]["name"]
                                              for t in c["tools"]],
                                     "uses_memory": c["uses_memory"]}
                           for c in CONDITIONS},
            "chat": {"model": CHAT_MODEL, "base_url": CHAT_BASE,
                     "temperature": 0.0},
        },
        "index": index_stats,
        "cells": cells,
        "metrics": metrics,
    }
    out_path = _THIS_DIR / f"day6_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"results -> {out_path}")

    # report
    print(f"\n{'task':<26} {'condition':<12} {'acc':>6} {'turns':>6} "
          f"{'calls':>6} {'mem':>5} {'tok/cell':>10} {'sec':>6}")
    for task in TASKS:
        for cond in CONDITIONS:
            p = metrics[cond["id"]]["per_task"][task["id"]]
            print(f"{task['id']:<26} {cond['id']:<12} {p['success_rate']:>6.3f} "
                  f"{p['mean_turns']:>6.2f} {p['mean_tool_calls']:>6.2f} "
                  f"{p['mean_memory_calls']:>5.2f} {p['mean_total_tokens']:>10.1f} "
                  f"{p['mean_elapsed_s']:>6.1f}")
    for cond in CONDITIONS:
        print(f"{cond['id']}: memory used in "
              f"{metrics[cond['id']]['cells_with_memory_use']}/"
              f"{metrics[cond['id']]['n_cells']} cells "
              f"(mean {metrics[cond['id']]['mean_memory_calls']} calls/cell)")

    if not args.no_plot:
        plot_path = _THIS_DIR / f"day6_plot_{ts}.png"
        plot_results(results, str(plot_path))


if __name__ == "__main__":
    main()
