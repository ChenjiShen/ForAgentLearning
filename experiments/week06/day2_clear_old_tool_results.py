"""Week 6 Day 2: clear-old-tool-results — prune old tool results, verify the
agent still functions.

Experiment question:
    Does replacing tool results older than N turns with "[cleared]" keep a
    long-session agent functional, and how much context does it save?

Method:
    1. `clear_old_tool_results(messages, msg_turns, current_turn, n)` — a pure
       function: mutates role="tool" messages whose turn t satisfies
       `current_turn - t >= n`, replacing content with CLEARED = "[cleared]".
       Assistant/user messages (the *decisions*) are never touched, and the
       cleared messages keep their structure (role, tool_call_id) so the
       request stays API-valid.
    2. `--selftest` — deterministic assertions on the pruning function, no API.
    3. Live run: the SAME 20-turn session as day 1 (mini research project with
       3 recall turns — T08/T14/T18 — that ask about facts established in
       earlier turns), pruned at every turn boundary with the given `--n`.
       Instrumentation mirrors day 1 (API ground-truth tokens + component
       decomposition), plus per-turn cleared counts and final-answer capture.
    4. Verification (the day's task): the agent must still
       - complete all 20 turns (no exceptions, every tool turn uses tools),
       - answer the recall turns correctly (their facts live in assistant
         messages, which pruning keeps),
       - and we measure input-token savings vs the day-1 unpruned baseline.

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week06/day2_clear_old_tool_results.py --selftest
    python experiments/week06/day2_clear_old_tool_results.py --n 3
    python experiments/week06/day2_clear_old_tool_results.py --n 6

Reuses the day-1 session (SESSION_PROMPTS, SYSTEM_PROMPT, instrumentation
helpers) and the week05/week03 tool stack, so the pruned run is directly
comparable to the day-1 unpruned baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_WEEK05_DIR = _THIS_DIR.parent / "week05"
if str(_WEEK05_DIR) not in sys.path:
    sys.path.insert(0, str(_WEEK05_DIR))

from day1_context_growth import (  # noqa: E402
    SESSION_PROMPTS,
    SYSTEM_PROMPT,
    execute_tool,
    measure_components,
)
from day1_tool_descriptions import _estimate_cost, build_tools  # noqa: E402
from openai import OpenAI  # noqa: E402
import httpx  # noqa: E402

MODEL = os.getenv("MODEL_NAME", "qwen3.7-plus")
API_BASE = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

CLEARED = "[cleared]"


# ============================================================================
# The pruning function
# ============================================================================


def clear_old_tool_results(messages: list[dict], msg_turns: list[int],
                           current_turn: int, n: int) -> list[dict]:
    """Replace tool results older than n turns with CLEARED (in place).

    Semantics: at the start of user turn `current_turn`, a tool result from
    turn t is pruned when `current_turn - t >= n` — i.e. results from the
    last n-1 turns (plus any produced during the current turn) are kept.
    Only role="tool" content is touched; role, tool_call_id, and all
    assistant/user/system messages survive untouched.

    Returns one record per message actually cleared:
        {index, turn, orig_chars}
    Calling again with the same arguments returns [] (idempotent).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    cleared: list[dict] = []
    for i, m in enumerate(messages):
        if m["role"] == "tool" and current_turn - msg_turns[i] >= n:
            if m["content"] != CLEARED:
                cleared.append({"index": i, "turn": msg_turns[i],
                                "orig_chars": len(m["content"])})
                m["content"] = CLEARED
    return cleared


# ============================================================================
# Selftest — deterministic checks of the pruning function (no API)
# ============================================================================


def _make_synthetic(turns: int = 5) -> tuple[list[dict], list[int]]:
    """A synthetic conversation: each turn = user + assistant(tool_call) + tool result."""
    msgs, ts = [], []
    for t in range(1, turns + 1):
        msgs.append({"role": "user", "content": f"question {t}"})
        ts.append(t)
        msgs.append({"role": "assistant", "content": f"answer {t}",
                     "tool_calls": [{"id": f"c{t}", "type": "function",
                                     "function": {"name": "t", "arguments": "{}"}}]})
        ts.append(t)
        msgs.append({"role": "tool", "tool_call_id": f"c{t}",
                     "content": f"RESULT-{t} " * 40})
        ts.append(t)
    return msgs, ts


def selftest() -> None:
    checks = 0

    # At turn 4 with n=2: turns 1-2 cleared, turns 3-4 kept, decisions intact.
    msgs, ts = _make_synthetic(5)
    rec = clear_old_tool_results(msgs, ts, current_turn=4, n=2)
    assert [r["turn"] for r in rec] == [1, 2], rec
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            if ts[i] <= 2:
                assert m["content"] == CLEARED
                assert m["tool_call_id"] == f"c{ts[i]}"  # structure kept
            else:
                assert m["content"].startswith("RESULT-")  # recent kept
        else:
            # user/assistant content untouched
            assert m["content"] in (f"question {ts[i]}", f"answer {ts[i]}")
    checks += 1

    # Assistant messages keep their tool_call payload (decisions survive).
    a2 = [m for m, t in zip(msgs, ts)
          if m["role"] == "assistant" and t == 2][0]
    assert "tool_calls" in a2 and a2["tool_calls"][0]["id"] == "c2"
    checks += 1

    # Idempotent: second pass records nothing new.
    assert clear_old_tool_results(msgs, ts, current_turn=4, n=2) == []
    checks += 1

    # n=1 at turn 3: every previous turn's result is already old.
    msgs, ts = _make_synthetic(5)
    assert [r["turn"] for r in clear_old_tool_results(msgs, ts, 3, 1)] == [1, 2]
    checks += 1

    # n larger than the session: nothing is ever old enough.
    msgs, ts = _make_synthetic(5)
    assert clear_old_tool_results(msgs, ts, 5, 100) == []
    checks += 1

    # Early boundaries: nothing is n turns old yet.
    msgs, ts = _make_synthetic(5)
    assert clear_old_tool_results(msgs, ts, 1, 3) == []
    assert clear_old_tool_results(msgs, ts, 3, 3) == []
    checks += 1

    # n < 1 is rejected.
    try:
        clear_old_tool_results(_make_synthetic(2)[0], _make_synthetic(2)[1], 2, 0)
        raise AssertionError("n=0 should raise")
    except ValueError:
        checks += 1

    print(f"SELFTEST PASSED ({checks} checks)")


# ============================================================================
# Recall verification — facts that live in assistant messages (kept by pruning)
# ============================================================================

RECALL_FACTS = {
    # T08 asks: mean of data1 (357/8) and the first research topic.
    "T08": [
        ("mean of data1 = 44.625", ("44.6",)),
        ("first topic", ("context window", "token limit")),
    ],
    # T14 asks: the two topics researched and the combined mean (1872/13).
    "T14": [
        ("topic 1", ("context window", "token limit")),
        ("topic 2", ("prompt caching",)),
        ("combined mean = 144.0", ("144",)),
    ],
}


def check_recall(answer: str, prompt_id: str) -> dict:
    """Check a recall turn's final answer against facts established earlier."""
    a = answer.lower()
    if prompt_id == "T18":  # "summarize everything we have done"
        return {"summary covers session": len(answer.strip()) >= 120
                and any(k in a for k in ("notes", "research", "data"))}
    return {label: any(needle in a for needle in needles)
            for label, needles in RECALL_FACTS[prompt_id]}


# ============================================================================
# The pruned session — day-1 loop + turn tracking + pruning at turn boundaries
# ============================================================================


def run_pruned_session(prompts: list[dict], tools: list[dict],
                       n: int, max_iter_per_turn: int = 10) -> dict:
    # trust_env=False: ignore the macOS system proxy (127.0.0.1:7890 — a local
    # proxy app that may not be running). The Beijing DashScope endpoint is
    # reachable directly; going through a dead proxy breaks every API call.
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=API_BASE,
        http_client=httpx.Client(trust_env=False),
    )
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    msg_turns: list[int] = [0]  # parallel array: which user turn each msg belongs to

    calib_factor: float | None = None
    factors_seen: list[float] = []
    calls: list[dict] = []
    turns: list[dict] = []
    clear_events: list[dict] = []  # every cleared tool result, across the session

    for pi, p in enumerate(prompts, start=1):
        messages.append({"role": "user", "content": p["prompt"]})
        msg_turns.append(pi)

        # ---- prune at the turn boundary, BEFORE any API call -------------
        cleared_now = clear_old_tool_results(messages, msg_turns, pi, n)
        clear_events.extend(cleared_now)

        turn_calls, tool_names, t0_turn = [], [], time.monotonic()
        final_answer = ""

        for _ in range(max_iter_per_turn):
            t0 = time.monotonic()
            comp = measure_components(messages, tools)
            response = client.chat.completions.create(
                model=MODEL, tools=tools, messages=messages)
            elapsed = time.monotonic() - t0

            usage = response.usage
            api_in = usage.prompt_tokens if usage else None
            api_out = usage.completion_tokens if usage else None

            est_total = sum(comp.values())
            if calib_factor is None and api_in:
                calib_factor = api_in / est_total
            if api_in and est_total:
                factors_seen.append(api_in / est_total)
            f = calib_factor if calib_factor else 1.0
            calib = {k: round(v * f) for k, v in comp.items()}

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
            msg_turns.append(pi)

            call = {
                "call_index": len(calls) + 1,
                "turn": pi,
                "api_prompt_tokens": api_in,
                "api_completion_tokens": api_out,
                "n_tool_calls": len(msg.tool_calls or []),
                "finish_reason": choice.finish_reason,
                "components_est": calib,
                "n_cleared_in_prompt": sum(
                    1 for m2, t2 in zip(messages, msg_turns)
                    if m2["role"] == "tool" and m2["content"] == CLEARED),
                "elapsed_s": round(elapsed, 2),
            }
            calls.append(call)
            turn_calls.append(call)

            if not msg.tool_calls:
                final_answer = msg.content or ""
                break  # final answer — turn over

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    out = f"[tool error] invalid JSON arguments: {e}"
                else:
                    out = execute_tool(name, args)
                tool_names.append(name)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": out})
                msg_turns.append(pi)

        # Sanity: pruning happens only at turn boundaries, so this turn's own
        # fresh tool results must all still be real content.
        fresh_intact = all(
            m2["content"] != CLEARED
            for m2, t2 in zip(messages, msg_turns)
            if m2["role"] == "tool" and t2 == pi)

        # ---- roll up the turn ------------------------------------------
        in_tok = sum(c["api_prompt_tokens"] or 0 for c in turn_calls)
        out_tok = sum(c["api_completion_tokens"] or 0 for c in turn_calls)
        turn = {
            "turn": pi,
            "prompt_id": p["id"],
            "kind": p["kind"],
            "prompt": p["prompt"],
            "n_api_calls": len(turn_calls),
            "n_tool_calls": len(tool_names),
            "tools_called": tool_names,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cumulative_input": in_tok + (turns[-1]["cumulative_input"] if turns else 0),
            "n_cleared_this_turn": len(cleared_now),
            "cleared_chars_this_turn": sum(r["orig_chars"] for r in cleared_now),
            "fresh_results_intact": fresh_intact,
            "final_answer": final_answer,
            "elapsed_s": round(time.monotonic() - t0_turn, 2),
        }
        if p["kind"] == "recall":
            turn["recall_checks"] = check_recall(final_answer, p["id"])
        turns.append(turn)

    # ---- session rollup ------------------------------------------------
    total_in = sum(t["input_tokens"] for t in turns)
    total_out = sum(t["output_tokens"] for t in turns)
    factors = [x for x in factors_seen if x]
    factor_mean = sum(factors) / len(factors) if factors else None

    function_checks = {
        "all_turns_completed": len(turns) == len(prompts),
        "every_tool_turn_used_tools": all(
            t["n_tool_calls"] > 0 for t in turns if t["kind"] == "tool"),
        "every_fresh_result_intact": all(t["fresh_results_intact"] for t in turns),
        "recall_turns_all_facts_ok": all(
            all(v for v in t["recall_checks"].values())
            for t in turns if t["kind"] == "recall"),
    }

    return {
        "prune_n": n,
        "model": MODEL,
        "n_turns": len(turns),
        "n_api_calls": len(calls),
        "n_tool_calls": sum(t["n_tool_calls"] for t in turns),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "estimated_cost_usd": round(_estimate_cost(total_in, total_out), 4),
        "n_cleared_total": len(clear_events),
        "cleared_chars_total": sum(e["orig_chars"] for e in clear_events),
        "n_cleared_in_final_prompt": sum(
            1 for m, t in zip(messages, msg_turns)
            if m["role"] == "tool" and m["content"] == CLEARED),
        "calibration_factor": round(calib_factor, 4) if calib_factor else None,
        "calibration_factor_mean": round(factor_mean, 4) if factor_mean else None,
        "function_checks": function_checks,
        "turns": turns,
        "calls": calls,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================================
# Reporting — pruned vs baseline comparison table
# ============================================================================


def print_table(s: dict) -> None:
    base = s["baseline"]
    sv = s["savings"]
    print(f"\n=== 20-turn session, clear-old-tool-results n={s['prune_n']} ===")
    print(f"model={s['model']}  api_calls={s['n_api_calls']}  "
          f"tool_calls={s['n_tool_calls']}  "
          f"cleared events={s['n_cleared_total']} "
          f"({s['cleared_chars_total']:,} chars dropped)")
    print(f"input tokens: {s['total_input_tokens']:,} "
          f"(baseline {base['total_input_tokens']:,})  →  "
          f"saved {sv['input_tokens']:,} ({sv['input_pct']:.1f}%)")
    print(f"output tokens: {s['total_output_tokens']:,} "
          f"(baseline {base['total_output_tokens']:,})")
    print(f"est. cost: ${s['estimated_cost_usd']:.4f} "
          f"(baseline ${base['estimated_cost_usd']:.4f})")
    fc = s["function_checks"]
    print("\nfunction checks:")
    for k, v in fc.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    for t in s["turns"]:
        if t["kind"] == "recall":
            for label, ok in t["recall_checks"].items():
                print(f"  [{'PASS' if ok else 'FAIL'}] {t['prompt_id']}: {label}")

    print("\n turn  calls tools  in_tok   cum_in   cum_base   saved_cum  "
          "cleared  | recall")
    for t, bt in zip(s["turns"], base["turns"]):
        cum_saved = bt["cumulative_input"] - t["cumulative_input"]
        recall = ""
        if t["kind"] == "recall":
            recall = " ".join(
                ("✓" if ok else "✗") for ok in t["recall_checks"].values())
        print(f" {t['turn']:>4} {t['n_api_calls']:>5} {t['n_tool_calls']:>5} "
              f"{t['input_tokens']:>7,} {t['cumulative_input']:>8,} "
              f"{bt['cumulative_input']:>9,} {cum_saved:>11,} "
              f"{t['n_cleared_this_turn']:>7}  | {recall}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 6 Day 2: clear-old-tool-results pruning")
    parser.add_argument("--n", type=int, default=3,
                        help="prune tool results older than N turns "
                             "(default: 3)")
    parser.add_argument("--selftest", action="store_true",
                        help="run deterministic pruning checks only, no API")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="max API iterations per user turn")
    parser.add_argument("--baseline", metavar="RESULTS_JSON", default=None,
                        help="day-1 unpruned results JSON (default: newest "
                             "day1_results_*.json in this directory)")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY is not set — export it first "
                 "(e.g. `set -a; source ../../.env; set +a`).")

    if args.baseline:
        baseline_path = Path(args.baseline)
    else:
        baseline_path = sorted(_THIS_DIR.glob("day1_results_*.json"))[-1]
    baseline = json.loads(baseline_path.read_text())

    print(f"Starting 20-turn session with pruning n={args.n} "
          f"(model={MODEL}) — baseline: {baseline_path.name} ...")
    t_start = time.monotonic()
    summary = run_pruned_session(SESSION_PROMPTS, build_tools("after"),
                                 n=args.n, max_iter_per_turn=args.max_iter)
    wall = time.monotonic() - t_start

    b_in = baseline["total_input_tokens"]
    summary["baseline"] = {
        "file": baseline_path.name,
        "total_input_tokens": b_in,
        "total_output_tokens": baseline["total_output_tokens"],
        "estimated_cost_usd": baseline["estimated_cost_usd"],
        "n_api_calls": baseline["n_api_calls"],
        "n_tool_calls": baseline["n_tool_calls"],
        "turns": baseline["turns"],
    }
    summary["savings"] = {
        "input_tokens": b_in - summary["total_input_tokens"],
        "input_pct": round((b_in - summary["total_input_tokens"]) / b_in * 100, 2),
    }
    summary["wall_seconds"] = round(wall, 1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = _THIS_DIR / f"day2_results_n{args.n}_{ts}.json"
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Raw trace saved to {results_path}")

    print_table(summary)
    print(f"\nWall time: {wall:.0f}s")


if __name__ == "__main__":
    main()
