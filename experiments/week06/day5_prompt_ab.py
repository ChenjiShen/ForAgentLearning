"""Week 6 Day 5: System prompt design — restructure the agent's system prompt
(role, tool guidance, constraints, output format) and A/B the old vs. the new
on the eval suites.

Experiment question:
    Does a restructured system prompt — explicit sections for role, tool
    guidance, constraints and output format — beat the current flat prompt
    on the compounding eval suite (evals/tasks.json) and the week-5
    tool-selection eval set?

The old prompt (day1_context_growth.SYSTEM_PROMPT, still used by the week-6
agent) is a flat "helpful assistant + tool rules" block that has accreted
session-specific lines ("This is a long working session…", "When the user
says 'no tools needed'…") — those belong in USER messages of the turns that
need them, not in the stable system layer. It has no role statement beyond
"helpful assistant", no explicit constraints, and no output-format contract.

The new prompt restructures the SAME facts into four sections and moves the
session-specific lines out. Two deliberate content additions, both disclosed:
  - a direct-tool mapping in Tool guidance (named file -> read_file, etc.),
  - "report only what the tools actually returned" in Constraints — a
    response to day 4's finding that the agent's false-success echo was the
    real information hazard (tool rejection + "Done." reply).

Design:
    TWO conditions, ONE thing varied (the system prompt):
      old — day1's SYSTEM_PROMPT (current).
      new — NEW_PROMPT below (sectioned).
    Same agent loop, same tools (week-5 "after" descriptions, 6 tools), same
    sandbox (week03/sandbox), same tasks, same checkers.
    Cell order (task x condition x run) shuffled with a fixed seed so
    time-of-day effects can't confound the comparison (week-5 day-6 lesson).

    Suite A — evals/tasks.json (14 tasks, substring checkers via the root
      harness's check_task). Some tasks' ground truth has drifted since
      week 3 (files other experiments rewrote/added) — both conditions see
      the identical environment, so the comparison stays fair; drifted tasks
      are flagged and a valid-subset analysis is reported separately.
    Suite B — week-5 EVAL_PROMPTS (10 prompts) scored on tool selection
      (should-call rate, wrong-tool rate, exact match). Ground truth is the
      expected tool set, which does not drift with sandbox contents.

    Secondary metrics (signal beyond pass/fail): tool-call counts, answer
    length (output-format compliance), tokens and cost, system-prompt token
    delta.

    Default: 2 suites x (14 + 10) tasks x 2 conditions x 5 runs = 240 runs.

Run:
    export $(grep DASHSCOPE .env | head -1 | xargs)
    python experiments/week06/day5_prompt_ab.py --selftest
    python experiments/week06/day5_prompt_ab.py --smoke
    python experiments/week06/day5_prompt_ab.py --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent
_WEEK05_DIR = _THIS_DIR.parent / "week05"
_EVALS_DIR = _ROOT / "evals"
for _d in (_THIS_DIR, _WEEK05_DIR, _EVALS_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from day1_context_growth import (  # noqa: E402
    SYSTEM_PROMPT,
    est_tokens,
)
from day1_tool_descriptions import (  # noqa: E402
    EVAL_PROMPTS,
    TOOL_FUNCTIONS,
    _estimate_cost,
    build_tools,
    score_run,
)
from day6_tool_distraction import (  # noqa: E402
    fisher_exact_two_sided,
    wilson_ci,
)
from run_evals import check_task  # noqa: E402  (root eval harness)
from openai import OpenAI  # noqa: E402
import httpx  # noqa: E402

MODEL = os.getenv("MODEL_NAME", "qwen3.7-plus")
API_BASE = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

OLD_PROMPT = SYSTEM_PROMPT

# ============================================================================
# The deliverable — the restructured system prompt.
# Same core facts as OLD_PROMPT, organized into: role / tool guidance /
# constraints / output format. Session-specific lines moved out (they belong
# in user messages). New facts: the direct-tool mapping and the
# report-only-what-tools-returned rule (see module docstring).
# ============================================================================

NEW_PROMPT = (
    "# Role\n"
    "You are a sandboxed task assistant: you complete user requests using "
    "the tools available to you, and report results plainly.\n\n"
    "# Tool guidance\n"
    "- Read a tool's description before calling it, and match the tool to "
    "the task.\n"
    "- Prefer the most direct tool: a named file -> read_file; browsing "
    "the workspace -> list_dir; multi-file text search -> run_command; a "
    "URL -> web_fetch; online research -> web_search.\n"
    "- Call only the tools the task actually needs. If the task can be "
    "answered without any tool, answer directly and call none.\n"
    "- On a tool error, read the error and retry once with corrected "
    "arguments before giving up.\n\n"
    "# Constraints\n"
    "- All file paths and shell commands are confined to the sandbox "
    "workspace.\n"
    "- Stop calling tools as soon as the task is complete.\n"
    "- Report only what the tools actually returned. Never invent file "
    "contents or tool results.\n\n"
    "# Output format\n"
    "- End your reply with the direct answer to the user's question.\n"
    "- Keep replies brief: the answer plus the minimum supporting detail.\n"
    "- If the task cannot be completed, say so plainly instead of guessing."
)

CONDITIONS = ["old", "new"]
PROMPT_BY_CONDITION = {"old": OLD_PROMPT, "new": NEW_PROMPT}

TOOLS = build_tools("after")  # 6 tools, week-5 "after" descriptions
assert len(TOOLS) == 6, f"expected 6 tools, got {len(TOOLS)}"


def load_suite_a() -> list[dict]:
    """The compounding eval suite (root evals/tasks.json)."""
    tasks = json.loads((_EVALS_DIR / "tasks.json").read_text())
    for t in tasks:
        assert t.get("id") and t.get("prompt") and t.get("expected_substring"), t
    return tasks


def load_suite_b() -> list[dict]:
    """The week-5 tool-selection eval set (expected tool per prompt)."""
    for p in EVAL_PROMPTS:
        assert p.get("id") and "expected_tools" in p, p
    return list(EVAL_PROMPTS)


# Suite-A tasks whose ground truth has drifted since week 3 — the reason is
# re-verified at run time and recorded in the results meta (see
# environment_drift()). Both conditions see the same environment, so these
# stay in the suite; they are flagged and excluded from the valid-subset
# analysis.
DRIFTED_TASKS: dict[str, str] = {
    "read-file-basic": "requirements.txt does not exist in the week03 sandbox",
    "search-then-locate": "no file in the sandbox defines MOCK_ORDERS",
    "orders-revenue-sequential": "MOCK_ORDERS not present in the sandbox",
    "orders-filter-sort": "MOCK_ORDERS not present in the sandbox",
    "orders-compare-multitool": "MOCK_ORDERS not present in the sandbox",
    "sandbox-list-py": "sandbox root .py count changed (task expects 3)",
    "sandbox-count-lines": "notes.txt was rewritten by later experiments "
                           "(task expects 4 lines)",
    "sandbox-multi-step": "notes.txt first line changed from "
                          "'Week 3 sandbox notes.'",
}


def environment_drift() -> dict:
    """Re-check the drifted-task claims against the current sandbox."""
    import sys as _sys
    _W3 = _THIS_DIR.parent / "week03"
    if str(_W3) not in _sys.path:
        _sys.path.insert(0, str(_W3))
    from agent import SANDBOX_ROOT as SB

    py_count = len(list(SB.glob("*.py")))
    notes = SB / "notes.txt"
    notes_lines = notes.read_text().splitlines() if notes.is_file() else []
    mock_orders_hits = [
        str(p.relative_to(SB))
        for p in SB.rglob("*")
        if p.is_file() and "MOCK_ORDERS" in p.read_text(errors="ignore")
    ]
    return {
        "sandbox_root": str(SB),
        "requirements_txt_exists": (SB / "requirements.txt").is_file(),
        "mock_orders_files_in_sandbox": mock_orders_hits,
        "sandbox_root_py_count": py_count,
        "notes_txt_lines": len(notes_lines),
        "notes_txt_first_line": notes_lines[0] if notes_lines else None,
    }


# ============================================================================
# The agent loop — week-5's runner with a parameterized system prompt and the
# day-4 client hardening (trust_env=False skips the macOS system proxy,
# 300s/15s timeout so a stalled DashScope request fails instead of hanging
# the whole batch).
# ============================================================================


def run_agent(user_input: str, system_prompt: str,
              max_turns: int = 12) -> dict:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=API_BASE,
        http_client=httpx.Client(trust_env=False,
                                 timeout=httpx.Timeout(300.0, connect=15.0)),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    tool_calls: list[dict] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.monotonic()

    for turn in range(1, max_turns + 1):
        response = client.chat.completions.create(
            model=MODEL, tools=TOOLS, messages=messages)
        if response.usage:
            usage["prompt_tokens"] += response.usage.prompt_tokens or 0
            usage["completion_tokens"] += response.usage.completion_tokens or 0
        choice = response.choices[0]
        msg = choice.message

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
            answer = msg.content or "[no text]"
            if choice.finish_reason == "length":
                answer = "[truncated — max_tokens limit]"
            return {
                "answer": answer,
                "tool_calls": tool_calls,
                "n_turns": turn,
                "n_tool_calls": len(tool_calls),
                "usage": usage,
                "estimated_cost": _estimate_cost(usage["prompt_tokens"],
                                                 usage["completion_tokens"]),
                "elapsed_s": round(time.monotonic() - t0, 2),
                "error": None,
            }

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                out = f"[tool error] invalid JSON arguments: {e}"
                tool_calls.append({"name": name,
                                   "arguments": tc.function.arguments,
                                   "error": f"JSON parse: {e}"})
            else:
                try:
                    out = str(TOOL_FUNCTIONS[name](args))
                except Exception as e:
                    out = f"[tool error] {name}: {e}"
                tool_calls.append({"name": name, "arguments": args,
                                   "error": None})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": out})

    return {
        "answer": f"[Agent stopped after {max_turns} turns — possible loop.]",
        "tool_calls": tool_calls,
        "n_turns": max_turns,
        "n_tool_calls": len(tool_calls),
        "usage": usage,
        "estimated_cost": _estimate_cost(usage["prompt_tokens"],
                                         usage["completion_tokens"]),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "error": "max_turns_exceeded",
    }


def run_cell(suite: str, task: dict, condition: str, run_index: int) -> dict:
    """One (task, condition, run) cell: run the agent, score, record."""
    trace = run_agent(task["prompt"], PROMPT_BY_CONDITION[condition])
    rec = {
        "suite": suite,
        "task": task["id"],
        "condition": condition,
        "run": run_index,
        "answer": trace["answer"],
        "answer_len": len(trace["answer"]),
        "n_turns": trace["n_turns"],
        "n_tool_calls": trace["n_tool_calls"],
        "tools_called": sorted({tc["name"] for tc in trace["tool_calls"]}),
        "prompt_tokens": trace["usage"]["prompt_tokens"],
        "completion_tokens": trace["usage"]["completion_tokens"],
        "estimated_cost": trace["estimated_cost"],
        "elapsed_s": trace["elapsed_s"],
        "error": trace["error"],
    }
    if suite == "A":
        try:
            rec["pass"], rec["checker"] = check_task(task, trace["answer"])
        except Exception as e:
            rec["pass"], rec["checker"] = False, f"checker exception: {e}"
    else:  # suite B — tool-selection scoring (week-5 score_run)
        s = score_run(trace, task["expected_tools"])
        rec.update({
            "pass": bool(s["exact_match"]),
            "should_call_rate": s["should_call_rate"],
            "wrong_tool_rate": s["wrong_tool_rate"],
            "exact_match": s["exact_match"],
            "missed": s["missed"],
            "extra": s["extra"],
        })
    return rec


# ============================================================================
# Statistics / reporting
# ============================================================================


def pooled_rates(recs: list[dict]) -> dict:
    ok = [r for r in recs if not r.get("error")]
    n_pass = sum(r["pass"] for r in ok)
    n = len(ok)
    return {"pass": n_pass, "n": n, "rate": n_pass / n if n else 0.0,
            "n_errors": len(recs) - n}


def fmt_ci(k: int, n: int) -> str:
    lo, hi = wilson_ci(k, n)
    return f"{k/n:.1%} [{lo:.1%}, {hi:.1%}] (n={n})" if n else "n/a"


def print_report(results: dict, save_path: Path) -> None:
    print("\n" + "=" * 78)
    print("WEEK 6 DAY 5 — system prompt restructure: old vs. new")
    print("=" * 78)
    print(f"model {results['model']}  runs/cell {results['runs_per_cell']}  "
          f"seed {results['seed']}  save {save_path.name}")

    env = results["environment"]
    print(f"\nsandbox: {env['sandbox_root']}")
    print(f"  .py files at root: {env['sandbox_root_py_count']} | "
          f"requirements.txt: {env['requirements_txt_exists']} | "
          f"notes.txt lines: {env['notes_txt_lines']} | "
          f"MOCK_ORDERS files: {env['mock_orders_files_in_sandbox'] or 'none'}")

    sp_old, sp_new = results["system_prompt_est_tokens"]
    print(f"\nsystem prompt (est. tokens): old {sp_old} -> new {sp_new} "
          f"(Δ {sp_new - sp_old:+d})")

    for suite in ("A", "B"):
        print(f"\n--- Suite {suite} ---")
        rows = results["raw"]
        for task_id in results["task_order"][suite]:
            sub = [r for r in rows if r["task"] == task_id]
            old = pooled_rates([r for r in sub if r["condition"] == "old"])
            new = pooled_rates([r for r in sub if r["condition"] == "new"])
            flag = ""
            if suite == "A" and task_id in results["drifted_tasks"]:
                flag = "  * drifted"
            print(f"  {task_id:<30} old {old['pass']}/{old['n']:<2}  "
                  f"new {new['pass']}/{new['n']:<2}{flag}")
        if suite == "A" and results["drifted_tasks"]:
            print("  (* drifted: ground truth changed since week 3 — "
                  "excluded from the valid-subset analysis below)")
            print("  drifted reasons:")
            for tid, why in results["drifted_tasks"].items():
                print(f"    - {tid}: {why}")

    print("\n--- Pooled comparison ---")
    for suite, valid_only in (("A", False), ("A-valid", True), ("B", False)):
        rows = results["raw"]
        recs = [r for r in rows if r["suite"] == "A"
                if (not valid_only) or r["task"] not in results["drifted_tasks"]]
        if suite == "B":
            recs = [r for r in rows if r["suite"] == "B"]
        old = pooled_rates([r for r in recs if r["condition"] == "old"])
        new = pooled_rates([r for r in recs if r["condition"] == "new"])
        # Fisher exact on pass/fail counts (errors excluded)
        a, b = old["pass"], old["n"] - old["pass"]
        c, d = new["pass"], new["n"] - new["pass"]
        p = fisher_exact_two_sided(a, b, c, d) if old["n"] and new["n"] else 1.0
        err_note = (f"  [errors old {old['n_errors']} / new {new['n_errors']}]"
                    if old["n_errors"] or new["n_errors"] else "")
        print(f"  {suite:<9} old {fmt_ci(old['pass'], old['n']):<28} "
              f"new {fmt_ci(new['pass'], new['n']):<28} "
              f"Fisher p={p:.4f}{err_note}")

    if any(r["suite"] == "B" for r in results["raw"]):
        print("\n--- Suite B: tool-selection detail (pooled over runs) ---")
        for cond in CONDITIONS:
            recs = [r for r in results["raw"]
                    if r["suite"] == "B" and r["condition"] == cond
                    and not r.get("error")]
            n = len(recs)
            sc = sum(r["should_call_rate"] for r in recs) / n if n else 0.0
            wt = sum(r["wrong_tool_rate"] for r in recs) / n if n else 0.0
            ex = sum(r["exact_match"] for r in recs) / n if n else 0.0
            print(f"  {cond:<4} should-call {sc:.1%}  wrong-tool {wt:.1%}  "
                  f"exact-match {ex:.1%}  (n={n})")

    print("\n--- Secondary metrics (pooled over all runs) ---")
    for cond in CONDITIONS:
        recs = [r for r in results["raw"]
                if r["condition"] == cond and not r.get("error")]
        n = len(recs)
        if not n:
            continue
        ans_lens = sorted(r["answer_len"] for r in recs)
        calls = sorted(r["n_tool_calls"] for r in recs)
        tin = sum(r["prompt_tokens"] for r in recs)
        tout = sum(r["completion_tokens"] for r in recs)
        cost = sum(r["estimated_cost"] for r in recs)
        med = lambda xs: xs[len(xs) // 2]
        print(f"  {cond:<4} n={n:<3} median answer {med(ans_lens)} chars | "
              f"median tool calls {med(calls)} | tokens in {tin:,} out {tout:,} "
              f"| est. cost ${cost:.3f}")

    print(f"\nFull results -> {save_path}")


# ============================================================================
# Selftest — deterministic checks, no API calls
# ============================================================================


def selftest() -> None:
    checks = 0

    # NEW_PROMPT has the four required sections, and differs from OLD_PROMPT
    for section in ("# Role", "# Tool guidance", "# Constraints",
                    "# Output format"):
        assert section in NEW_PROMPT, f"missing section {section}"
    checks += 1
    assert NEW_PROMPT != OLD_PROMPT
    checks += 1

    # suites load
    a = load_suite_a()
    b = load_suite_b()
    assert len(a) == 14, len(a)
    assert len(b) == 10, len(b)
    checks += 1

    # every drifted task id actually exists in suite A
    assert all(t in {x["id"] for x in a} for t in DRIFTED_TASKS)
    checks += 1

    # substring checker (root harness) behaves
    ok, _ = check_task({"id": "t", "prompt": "p",
                        "expected_substring": "395"}, "The answer is 395.")
    assert ok
    ok2, _ = check_task({"id": "t", "prompt": "p",
                         "expected_substring": "395"}, "The answer is 394.")
    assert not ok2
    checks += 1

    # score_run: exact match / wrong tool / no-tool cases
    s = score_run({"tool_calls": [{"name": "read_file", "arguments": {}}],
                   "n_turns": 2}, ["read_file"])
    assert s["exact_match"] and s["should_call_rate"] == 1.0 \
        and s["wrong_tool_rate"] == 0.0
    s = score_run({"tool_calls": [{"name": "web_fetch", "arguments": {}},
                                  {"name": "read_file", "arguments": {}}],
                   "n_turns": 3}, ["web_search"])
    assert not s["exact_match"] and s["wrong_tool_rate"] == 1.0 \
        and s["should_call_rate"] == 0.0
    s = score_run({"tool_calls": [], "n_turns": 1}, [])
    assert s["exact_match"] and s["should_call_rate"] == 1.0
    checks += 1

    # stats sanity
    lo, hi = wilson_ci(5, 10)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert fisher_exact_two_sided(5, 0, 0, 5) < 0.01
    assert fisher_exact_two_sided(5, 5, 5, 5) == 1.0
    checks += 1

    # cell-order shuffle: deterministic per seed, covers both conditions
    cells = [(s, t, c, r)
             for s in ("A", "B")
             for t in ("t1", "t2")
             for c in CONDITIONS
             for r in range(5)]
    def shuffle(seed):
        rnd = random.Random(seed)
        cs = list(cells)
        rnd.shuffle(cs)
        return cs
    assert shuffle(7) == shuffle(7)
    assert shuffle(1) != shuffle(2)
    assert all(c in shuffle(3) for c in cells)
    checks += 1

    # environment drift claims re-verified against the live sandbox
    env = environment_drift()
    assert not env["requirements_txt_exists"], env
    assert env["mock_orders_files_in_sandbox"] == [], env
    assert env["sandbox_root_py_count"] != 3, env
    assert env["notes_txt_lines"] != 4 or (
        env["notes_txt_first_line"] != "Week 3 sandbox notes."), env
    checks += 1

    print(f"SELFTEST PASSED ({checks} checks)")
    print("drift check (current sandbox values):")
    for k, v in env.items():
        print(f"  {k}: {v}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 6 Day 5: system prompt restructure A/B")
    parser.add_argument("--runs", type=int, default=5,
                        help="runs per (task, condition) cell")
    parser.add_argument("--seed", type=int, default=42,
                        help="shuffle seed for cell order")
    parser.add_argument("--suites", choices=["a", "b", "ab"], default="ab",
                        help="which eval suites to run")
    parser.add_argument("--tasks", type=str, nargs="*",
                        help="task ids to run (both suites)")
    parser.add_argument("--smoke", action="store_true",
                        help="1 run per cell on a tiny subset, end-to-end")
    parser.add_argument("--selftest", action="store_true",
                        help="deterministic checks only, no API")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY is not set — export it first "
                 "(e.g. `export $(grep DASHSCOPE "
                 "../../.env | head -1 | xargs)`).")

    tasks: list[tuple[str, dict]] = []
    if args.suites in ("a", "ab"):
        tasks += [("A", t) for t in load_suite_a()]
    if args.suites in ("b", "ab"):
        tasks += [("B", t) for t in load_suite_b()]
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [(s, t) for s, t in tasks if t["id"] in wanted]
        if not tasks:
            sys.exit(f"--tasks {args.tasks} matched nothing")

    if args.smoke:
        tasks = [(s, t) for s, t in tasks if t["id"] in
                 ("math-basic", "P9-no-tool-trivia")]

    runs_per_cell = 1 if args.smoke else args.runs
    cells = [(suite, t["id"], cond, r)
             for suite, t in tasks
             for cond in CONDITIONS
             for r in range(runs_per_cell)]
    rnd = random.Random(args.seed)
    rnd.shuffle(cells)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = _THIS_DIR / f"day5_results_{timestamp}.json"
    results = {
        "experiment": "week6_day5_system_prompt_ab",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "runs_per_cell": runs_per_cell,
        "seed": args.seed,
        "smoke": args.smoke,
        "suites": args.suites,
        "old_prompt": OLD_PROMPT,
        "new_prompt": NEW_PROMPT,
        "system_prompt_est_tokens": (est_tokens(OLD_PROMPT),
                                     est_tokens(NEW_PROMPT)),
        "environment": environment_drift(),
        "drifted_tasks": DRIFTED_TASKS,
        "task_order": {"A": [t["id"] for s, t in tasks if s == "A"],
                       "B": [t["id"] for s, t in tasks if s == "B"]},
        "cell_order": cells,
        "raw": [],
    }

    t_start = time.monotonic()
    n_done = 0
    for suite, task_id, cond, run_i in cells:
        task = next(t for s, t in tasks if t["id"] == task_id)
        try:
            rec = run_cell(suite, task, cond, run_i)
        except Exception as e:
            rec = {"suite": suite, "task": task_id, "condition": cond,
                   "run": run_i, "error": f"{type(e).__name__}: {e}"}
        results["raw"].append(rec)
        n_done += 1
        mark = "?"
        if not rec.get("error"):
            mark = ("✓" if rec["pass"]
                    else ("∅" if suite == "B"
                          and rec.get("should_call_rate") is not None
                          else "✗"))
        err_txt = ("  ERROR " + str(rec["error"])[:60]) if rec.get("error") else ""
        print(f"[{n_done:>3}/{len(cells)}] {suite} {task_id:<28} {cond} "
              f"r{run_i} {mark}  "
              f"{rec.get('n_tool_calls', 0)} calls  "
              f"{rec.get('answer_len', 0)} chars  "
              f"{rec.get('elapsed_s', 0):.1f}s{err_txt}",
              flush=True)
        # incremental save — a crash mid-batch loses nothing
        results["elapsed_min"] = round((time.monotonic() - t_start) / 60, 1)
        save_path.write_text(json.dumps(results, indent=2,
                                        ensure_ascii=False))

    print_report(results, save_path)


if __name__ == "__main__":
    main()
