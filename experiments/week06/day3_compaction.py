"""Week 6 Day 3: Compaction — when history > N tokens, summarize the older
half with an LLM call and splice it in.

Experiment question:
    Does replacing the older half of a long session's history with an
    LLM-written summary keep the agent functional, how much context does it
    save (net of the summarizer's own cost), and what survives the summary?

Method:
    1. `compact_history(messages, msg_turns, summarizer, max_tokens)` — a pure
       orchestration function (the summarizer is injected, so it is fully
       testable without an API):
       - trigger: calibrated estimate of system+messages > max_tokens;
       - `find_split`: the older half = messages[1:split] (by message count),
         with the split never landing between an assistant tool_call and its
         tool results (orphan tool results are API-invalid, so the split walks
         forward past any role="tool" messages);
       - guards: need ≥4 messages in the older half, and compaction is skipped
         when the kept half alone already exceeds max_tokens (one summary
         cannot fix that — it would thrash every turn);
       - the summarizer receives the serialized older half (roles + content +
         tool-call arguments flattened to text — avoids any API role-structure
         quirks) and is instructed to preserve numbers, filenames, facts and
         decisions; its output is spliced in as a single role="user" message
         with a prefix that marks it as a summary. The summary replaces the
         older half; the system prompt and the kept half survive untouched.
    2. `--selftest` — deterministic assertions on split/splice/trigger logic
       with a fake summarizer (no API).
    3. Live run: the SAME 20-turn session as days 1-2, compacted at every turn
       boundary with the given `--max-tokens`. Instrumentation mirrors day 1
       (API ground-truth tokens + component decomposition), plus:
       - per-boundary history estimate (pre/post compaction) — the sawtooth;
       - one event per compaction: what was summarized (turns, messages,
         chars), the summary text itself, and the summarizer's token usage
         (accounted in the session totals and cost).
    4. Verification (the day's task):
       - all 20 turns complete, every tool turn uses tools,
       - the recall turns (T08/T14/T18) still answer correctly — their facts
         may now live in a summary instead of verbatim history,
       - input-token savings vs the day-1 unpruned baseline, NET of the
         summarizer calls.

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week06/day3_compaction.py --selftest
    python experiments/week06/day3_compaction.py --max-tokens 12000
    python experiments/week06/day3_compaction.py --plot-only <results.json>

Reuses the day-1 session (SESSION_PROMPTS, SYSTEM_PROMPT, instrumentation
helpers) and the week05/week03 tool stack, so the compacted run is directly
comparable to the day-1 baseline and the day-2 pruned runs.
"""

from __future__ import annotations

import argparse
import json
import math
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
    est_tokens,
    execute_tool,
    measure_components,
)
from day1_tool_descriptions import _estimate_cost, build_tools  # noqa: E402
from day2_clear_old_tool_results import check_recall  # noqa: E402
from openai import OpenAI  # noqa: E402
import httpx  # noqa: E402

MODEL = os.getenv("MODEL_NAME", "qwen3.7-plus")
API_BASE = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# Marks a spliced-in summary; also what we count to report "summaries in prompt".
SUMMARY_PREFIX = "[Earlier conversation summary — replaces the full history up to this point]\n"

# Compaction needs at least this many messages in the older half to be worth
# an LLM call (system excluded): roughly one full user→tool→assistant turn.
MIN_OLD_MSGS = 4

SUMMARIZER_SYSTEM = (
    "You are compressing an agent's conversation history. Write ONE compact "
    "summary that preserves every detail the agent might need later: "
    "specific numbers and computed results, filenames and file contents, "
    "web-research topics and findings, facts learned from tool results, "
    "decisions made, and any constraints or instructions from the user. "
    "Use terse bullet form. This summary REPLACES the conversation in the "
    "agent's memory — nothing else is kept, so any detail you drop is lost "
    "forever. Output only the summary text."
)


# ============================================================================
# The compaction function
# ============================================================================


def find_split(messages: list[dict]) -> int | None:
    """Index of the first message of the KEPT (newer) half, or None.

    The older half is messages[1:split] (message-count based; messages[0] is
    the system prompt, which is never summarized). The split then walks
    forward past any role="tool" messages so it never cuts between an
    assistant tool_call and its results — orphan tool results in the kept
    half would make the next API request invalid.

    Returns None when the conversation is too short to split at all.
    """
    n = len(messages)
    if n <= 3:  # system + fewer than 2 messages — nothing sensible to split
        return None
    split = 1 + math.ceil((n - 1) / 2)
    while split < n and messages[split]["role"] == "tool":
        split += 1
    return split if split < n else None


def history_estimate(messages: list[dict]) -> int:
    """Est tokens of system+messages (schemas excluded) — the trigger measure."""
    return sum(est_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)


def _serialize_for_summarizer(messages: list[dict]) -> str:
    """Flatten messages to readable text for the summarizer LLM call.

    Assistant tool_calls are rendered as 'name(args)' so the summary knows
    what was *done*; plain text roles render as 'role: content'.
    """
    lines = []
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc["function"]
                lines.append(f"assistant: called {fn['name']}({fn['arguments']})")
            if m.get("content"):
                lines.append(f"assistant: {m['content']}")
        elif role == "tool":
            lines.append(f"tool result (for {m.get('tool_call_id', '?')}): "
                         f"{m['content']}")
        else:
            lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def compact_history(messages: list[dict], msg_turns: list[int],
                    summarizer, max_tokens: int, calib: float = 1.0) -> dict | None:
    """Compact the older half of `messages` into one summary message (in place).

    `summarizer` is a callable(serialized_older_half: str) -> dict with keys
    `text` (str) and optionally `prompt_tokens` / `completion_tokens` — the
    fake summarizer in the selftest supplies None, the live one fills in the
    LLM usage. `msg_turns` is the parallel turn array; the summary message
    inherits the newest turn it absorbed. `calib` scales the raw chars/4
    token estimates (the API-calibrated factor from the session).

    Returns an event record, or None when nothing was compacted:
        {status, turn, trigger_est_tokens, split_index, n_messages_summarized,
         turns_absorbed: [min, max], chars_before, chars_after, summary_text,
         summarizer_prompt_tokens, summarizer_completion_tokens}
    (status="error" records summarizer failures; messages are left unchanged.)
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    n = len(messages)
    if n - 1 < MIN_OLD_MSGS:
        return None  # nothing to summarize yet

    if history_estimate(messages) * calib <= max_tokens:
        return None  # under the threshold

    split = find_split(messages)
    if split is None or split - 1 < MIN_OLD_MSGS:
        return None  # older half too small to be worth an LLM call

    # Thrash guard: if the kept half alone already exceeds the threshold, one
    # summary cannot bring us under it — we would re-trigger every turn.
    if history_estimate(messages[split:]) * calib > max_tokens:
        return None

    older, kept = messages[1:split], messages[split:]
    turns_absorbed = [min(msg_turns[1:split]), max(msg_turns[1:split])]
    chars_before = sum(len(json.dumps(m, ensure_ascii=False)) for m in older)

    event = {
        "status": "ok",
        "trigger_est_tokens": round(history_estimate(messages) * calib),
        "split_index": split,
        "n_messages_summarized": len(older),
        "turns_absorbed": turns_absorbed,
        "chars_before": chars_before,
        "chars_after": None,
        "summary_text": None,
        "summarizer_prompt_tokens": None,
        "summarizer_completion_tokens": None,
    }
    try:
        result = summarizer(_serialize_for_summarizer(older))
    except Exception as e:  # summarizer failed — keep history, log the failure
        event["status"] = "error"
        event["error"] = str(e)
        return event

    summary_msg = {"role": "user",
                   "content": SUMMARY_PREFIX + result["text"].strip()}
    event["summary_text"] = result["text"].strip()
    event["chars_after"] = len(summary_msg["content"])
    event["summarizer_prompt_tokens"] = result.get("prompt_tokens")
    event["summarizer_completion_tokens"] = result.get("completion_tokens")

    # Splice: system stays, summary takes the older half's slot, kept half
    # follows untouched.
    messages[1:split] = [summary_msg]
    msg_turns[1:split] = [turns_absorbed[1]]
    return event


# ============================================================================
# Selftest — deterministic checks of the compaction logic (no API)
# ============================================================================


def _fake_summarizer(serialized: str) -> dict:
    return {"text": f"SUMMARY({len(serialized.splitlines())} msgs)",
            "prompt_tokens": 0, "completion_tokens": 0}


def _synthetic(n_blocks: int) -> tuple[list[dict], list[int]]:
    """n_blocks of: user, assistant(tool_call), tool result, assistant(final)."""
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    ts: list[int] = [0]
    for t in range(1, n_blocks + 1):
        msgs.append({"role": "user", "content": f"question {t}" * 20})
        ts.append(t)
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{t}", "type": "function",
                                     "function": {"name": "read_file",
                                                  "arguments": '{"filepath": "x"}'}}]})
        ts.append(t)
        msgs.append({"role": "tool", "tool_call_id": f"c{t}",
                     "content": f"RESULT-{t} " * 200})
        ts.append(t)
        msgs.append({"role": "assistant", "content": f"answer {t} " * 10})
        ts.append(t)
    return msgs, ts


def _assert_no_orphans(msgs: list[dict]) -> None:
    """Every kept tool result has its assistant tool_call in the same half,
    and every kept assistant tool_call has all its results after the split."""
    seen_calls: set[str] = set()
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            seen_calls |= {tc["id"] for tc in m["tool_calls"]}
        elif m["role"] == "tool":
            assert m["tool_call_id"] in seen_calls, \
                f"orphan tool result {m['tool_call_id']}"


def selftest() -> None:
    checks = 0

    # 1. find_split: system + user + assistant(tc) + tool + tool + user + assistant
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c1", "content": "r1b"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    # mid = 1 + ceil(6/2) = 4 → lands on the second tool result → walk to 5
    assert find_split(msgs) == 5, find_split(msgs)
    checks += 1

    # 2. find_split: too short
    assert find_split([{"role": "system", "content": "S"},
                       {"role": "user", "content": "u"}]) is None
    checks += 1

    # 3. Under threshold → no compaction, messages untouched.
    msgs, ts = _synthetic(3)
    before = [json.dumps(m) for m in msgs]
    assert compact_history(msgs, ts, _fake_summarizer, max_tokens=10_000_000) is None
    assert [json.dumps(m) for m in msgs] == before
    checks += 1

    # 4. Over threshold → compacts exactly the older half; system + kept half
    #    survive (same content, same order); summary spliced at index 1.
    #    (N=1500: above the kept half's ~1100 est tokens, below the full
    #    history's ~2450 — so exactly one compaction fires, no guard trips.)
    msgs, ts = _synthetic(4)  # 17 msgs incl system: mid = 1+ceil(16/2) = 9
    kept_before = [json.dumps(m) for m in msgs[9:]]
    kept_turns = ts[9:]
    event = compact_history(msgs, ts, _fake_summarizer, max_tokens=1_500)
    assert event and event["status"] == "ok", event
    assert event["n_messages_summarized"] == 8, event
    assert event["turns_absorbed"] == [1, 2], event
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["role"] == "user" and msgs[1]["content"].startswith(SUMMARY_PREFIX)
    assert "SUMMARY(" in msgs[1]["content"]  # fake summarizer text spliced verbatim
    assert [json.dumps(m) for m in msgs[2:]] == kept_before  # kept half untouched
    assert ts[2:] == kept_turns and ts[1] == 2  # summary inherits newest turn
    _assert_no_orphans(msgs)
    checks += 1

    # 5. A tool-pair straddling the midpoint: the walk-forward must absorb the
    #    whole assistant(tc)+results block into the older half — no orphans.
    #    (N=500: history ~870, kept half ~340 — compacts, no guard trips.)
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u1 " * 200},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1 " * 200},
        {"role": "tool", "tool_call_id": "c1", "content": "r1b " * 200},
        {"role": "user", "content": "u2 " * 200},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c2", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "r2 " * 200},
        {"role": "assistant", "content": "a2"},
    ]
    ts = [0, 1, 1, 1, 1, 2, 2, 2, 2]
    event = compact_history(msgs, ts, _fake_summarizer, max_tokens=500)
    assert event and event["n_messages_summarized"] == 4, event  # u1, asst(c1), r1, r1b
    _assert_no_orphans(msgs)  # kept half: u2, asst(c2), r2, a2 — complete pair
    checks += 1

    # 6. Kept-half-alone-over-N guard: one summary cannot fix it → no-op.
    # (History > 100, and the kept half alone — 2 synthetic blocks of ~4k
    # chars each — also > 100, so compacting the older half would never bring
    # us under N; it would just thrash every turn.)
    msgs, ts = _synthetic(4)
    before = [json.dumps(m) for m in msgs]
    event = compact_history(msgs, ts, _fake_summarizer, max_tokens=100)
    assert event is None, "thrash guard should skip when kept half alone > N"
    assert [json.dumps(m) for m in msgs] == before
    checks += 1

    # 7. MIN_OLD_MSGS guard: history over N but older half has <4 messages.
    msgs, ts = _synthetic(1)  # 5 msgs: mid = 1+ceil(4/2) = 3 → older half = 2 msgs
    before = [json.dumps(m) for m in msgs]
    assert compact_history(msgs, ts, _fake_summarizer, max_tokens=1) is None
    assert [json.dumps(m) for m in msgs] == before
    checks += 1

    # 8. max_tokens < 1 rejected.
    msgs, ts = _synthetic(2)
    try:
        compact_history(msgs, ts, _fake_summarizer, max_tokens=0)
        raise AssertionError("max_tokens=0 should raise")
    except ValueError:
        checks += 1

    # 9. Summary-of-summary: grow history again past N after a compaction —
    #    the previous summary message itself gets absorbed into the next one.
    msgs, ts = _synthetic(4)
    event1 = compact_history(msgs, ts, _fake_summarizer, max_tokens=1_500)
    assert event1 and event1["status"] == "ok"
    first_summary = msgs[1]["content"]
    # pad the kept half until we're over the threshold again
    while history_estimate(msgs) <= 1_500:
        t = max(ts) + 1
        msgs.append({"role": "user", "content": "grow " * 100})
        ts.append(t)
        msgs.append({"role": "assistant", "content": "grow back " * 100})
        ts.append(t)
    event2 = compact_history(msgs, ts, _fake_summarizer, max_tokens=1_500)
    assert event2 and event2["status"] == "ok"
    assert msgs[1]["role"] == "user" and msgs[1]["content"].startswith(SUMMARY_PREFIX)
    assert first_summary not in [m.get("content") for m in msgs], \
        "first summary should have been absorbed into the second"
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == SYSTEM_PROMPT
    _assert_no_orphans(msgs)
    checks += 1

    print(f"SELFTEST PASSED ({checks} checks)")


# ============================================================================
# The compacted session — day-1/2 loop + compaction at turn boundaries
# ============================================================================


def run_compacted_session(prompts: list[dict], tools: list[dict],
                          max_tokens: int,
                          max_iter_per_turn: int = 10) -> dict:
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
    compaction_events: list[dict] = []
    summarizer_calls: list[dict] = []
    boundary_history: list[dict] = []  # per-turn pre/post compaction estimates

    def summarizer(serialized: str) -> dict:
        """LLM summarizer: one plain chat completion over the serialized older half."""
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SUMMARIZER_SYSTEM},
                      {"role": "user",
                       "content": ("<conversation>\n" + serialized +
                                   "\n</conversation>\nSummarize the conversation "
                                   "above. Output only the summary.")}])
        usage = resp.usage
        rec = {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "elapsed_s": round(time.monotonic() - t0, 2),
        }
        summarizer_calls.append(rec)
        return {"text": resp.choices[0].message.content or "",
                "prompt_tokens": rec["prompt_tokens"],
                "completion_tokens": rec["completion_tokens"]}

    for pi, p in enumerate(prompts, start=1):
        messages.append({"role": "user", "content": p["prompt"]})
        msg_turns.append(pi)

        # ---- compact at the turn boundary, BEFORE any API call -------------
        f = calib_factor if calib_factor else 1.0
        est_pre = round(history_estimate(messages) * f)
        event = compact_history(messages, msg_turns, summarizer, max_tokens,
                                calib=f)
        est_post = round(history_estimate(messages) * f) if event else None
        if event:
            event.update({"turn": pi, "est_pre": est_pre, "est_post": est_post})
            compaction_events.append(event)
        boundary_history.append({"turn": pi, "est_pre": est_pre,
                                 "est_post": est_post,
                                 "compacted": bool(event
                                                    and event["status"] == "ok")})

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
                # Calibrate the estimator on the very first API call.
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
                "n_summaries_in_prompt": sum(
                    1 for m2 in messages
                    if m2["role"] == "user"
                    and m2["content"].startswith(SUMMARY_PREFIX)),
                "summary_tokens_in_prompt": sum(
                    est_tokens(m2["content"]) * f
                    for m2 in messages
                    if m2["role"] == "user"
                    and m2["content"].startswith(SUMMARY_PREFIX)),
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
            "n_compactions_before_turn": len(compaction_events),
            "elapsed_s": round(time.monotonic() - t0_turn, 2),
            "final_answer": final_answer,
        }
        if p["kind"] == "recall":
            turn["recall_checks"] = check_recall(final_answer, p["id"])
        turns.append(turn)

    # ---- session rollup ------------------------------------------------
    total_in = sum(t["input_tokens"] for t in turns)
    total_out = sum(t["output_tokens"] for t in turns)
    sum_in = sum(c["prompt_tokens"] or 0 for c in summarizer_calls)
    sum_out = sum(c["completion_tokens"] or 0 for c in summarizer_calls)
    factors = [x for x in factors_seen if x]
    factor_mean = sum(factors) / len(factors) if factors else None

    function_checks = {
        "all_turns_completed": len(turns) == len(prompts),
        "every_tool_turn_used_tools": all(
            t["n_tool_calls"] > 0 for t in turns if t["kind"] == "tool"),
        "recall_turns_all_facts_ok": all(
            all(v for v in t["recall_checks"].values())
            for t in turns if t["kind"] == "recall"),
        "compaction_triggered": len(compaction_events) > 0,
        "every_compaction_ok": all(
            e["status"] == "ok" for e in compaction_events),
    }

    return {
        "max_tokens": max_tokens,
        "model": MODEL,
        "n_turns": len(turns),
        "n_api_calls": len(calls),
        "n_tool_calls": sum(t["n_tool_calls"] for t in turns),
        "agent_input_tokens": total_in,
        "agent_output_tokens": total_out,
        "summarizer_input_tokens": sum_in,
        "summarizer_output_tokens": sum_out,
        "total_input_tokens": total_in + sum_in,
        "total_output_tokens": total_out + sum_out,
        "total_tokens": total_in + total_out + sum_in + sum_out,
        "estimated_cost_usd": round(_estimate_cost(total_in + sum_in,
                                                   total_out + sum_out), 4),
        "n_compactions": len(compaction_events),
        "compaction_events": compaction_events,
        "summarizer_calls": summarizer_calls,
        "boundary_history": boundary_history,
        "n_summaries_in_final_prompt": sum(
            1 for m in messages
            if m["role"] == "user" and m["content"].startswith(SUMMARY_PREFIX)),
        "calibration_factor": round(calib_factor, 4) if calib_factor else None,
        "calibration_factor_mean": round(factor_mean, 4) if factor_mean else None,
        "function_checks": function_checks,
        "turns": turns,
        "calls": calls,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================================
# Reporting — compacted vs baseline comparison table
# ============================================================================


def print_table(s: dict) -> None:
    base = s["baseline"]
    sv = s["savings"]
    print(f"\n=== 20-turn session, compaction max_tokens={s['max_tokens']:,} ===")
    print(f"model={s['model']}  api_calls={s['n_api_calls']}  "
          f"tool_calls={s['n_tool_calls']}  "
          f"compactions={s['n_compactions']}")
    print(f"agent input tokens: {s['agent_input_tokens']:,} "
          f"(baseline {base['total_input_tokens']:,})")
    print(f"summarizer cost: {s['summarizer_input_tokens']:,} in + "
          f"{s['summarizer_output_tokens']:,} out = "
          f"{s['summarizer_input_tokens'] + s['summarizer_output_tokens']:,} tokens")
    print(f"NET input tokens: {s['total_input_tokens']:,}  →  "
          f"saved {sv['net_input_tokens']:,} ({sv['net_input_pct']:.1f}%) vs baseline")
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

    print("\ncompaction events:")
    for e in s["compaction_events"]:
        ok = e["status"] == "ok"
        print(f"  turn {e['turn']:>2}: est {e['est_pre']:>6,} → {e['est_post']:>6,} "
              f"({e['n_messages_summarized']} msgs, turns "
              f"{e['turns_absorbed'][0]}-{e['turns_absorbed'][1]}, "
              f"{e['chars_before']:,} → {e['chars_after']:,} chars, "
              f"summarizer {e['summarizer_prompt_tokens'] or 0}+"
              f"{e['summarizer_completion_tokens'] or 0} tok)"
              + (f"  ERROR: {e['error'][:60]}" if not ok else ""))
    if s["compaction_events"]:
        ok_events = [e for e in s["compaction_events"] if e["status"] == "ok"]
        if ok_events:
            print("\nsummary texts (what survived):")
            for i, e in enumerate(ok_events, 1):
                print(f"  --- compaction #{i} (turn {e['turn']}, absorbed turns "
                      f"{e['turns_absorbed'][0]}-{e['turns_absorbed'][1]}) ---")
                for line in e["summary_text"].splitlines()[:6]:
                    print(f"    {line[:110]}")
                if len(e["summary_text"].splitlines()) > 6:
                    print("    ...")

    print("\n turn  calls tools  in_tok   cum_in   cum_base   saved_cum  "
          "compacts  | recall")
    for t, bt in zip(s["turns"], base["turns"]):
        cum_saved = bt["cumulative_input"] - t["cumulative_input"]
        recall = ""
        if t["kind"] == "recall":
            recall = " ".join(
                ("✓" if ok else "✗") for ok in t["recall_checks"].values())
        print(f" {t['turn']:>4} {t['n_api_calls']:>5} {t['n_tool_calls']:>5} "
              f"{t['input_tokens']:>7,} {t['cumulative_input']:>8,} "
              f"{bt['cumulative_input']:>9,} {cum_saved:>11,} "
              f"{t['n_compactions_before_turn']:>8}  | {recall}")


# ============================================================================
# Plot — cumulative input vs baseline + the history sawtooth at turn boundaries
# ============================================================================


def make_plot(s: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # -- dataviz palette (light mode), validated slots — same family as day 1
    SURFACE = "#fcfcfb"
    INK = "#0b0b0b"
    SECONDARY = "#52514e"
    MUTED = "#898781"
    GRID = "#e1e0d9"
    C_COMPACT = "#2a78d6"    # slot 1 blue   — compacted session (the subject)
    C_BASE = "#898781"       # muted gray    — day-1 baseline (the reference)
    C_EVENT = "#eb6834"      # slot 2 orange — compaction events
    C_THRESH = "#52514e"     # secondary ink — the N threshold line

    turns = s["turns"]
    base_turns = s["baseline"]["turns"]
    x = [t["turn"] for t in turns]
    y_base = [t["cumulative_input"] for t in base_turns]
    y_comp = [t["cumulative_input"] for t in turns]
    events = [e for e in s["compaction_events"] if e["status"] == "ok"]
    ev_x = [e["turn"] for e in events]
    bh = s["boundary_history"]
    x_pre = [b["turn"] for b in bh]
    y_pre = [b["est_pre"] for b in bh]
    n_thresh = s["max_tokens"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig = plt.figure(figsize=(12, 8.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0],
                          hspace=0.38, left=0.08, right=0.97,
                          top=0.95, bottom=0.07)

    # ---- Panel A: cumulative input tokens — baseline vs compacted --------
    ax_a = fig.add_subplot(gs[0])
    ax_a.plot(x, y_base, color=C_BASE, linewidth=2, linestyle="--",
              marker="o", markersize=4, markerfacecolor=SURFACE,
              markeredgecolor=C_BASE, markeredgewidth=1.2,
              label="day-1 baseline (no compaction)", zorder=2)
    ax_a.plot(x, y_comp, color=C_COMPACT, linewidth=2.2,
              marker="o", markersize=5, markerfacecolor=C_COMPACT,
              markeredgecolor=SURFACE, markeredgewidth=1.2,
              label="day-3 compaction", zorder=3)
    # compaction events: vertical ticks at the turn where they fired
    for ex in ev_x:
        ax_a.axvline(ex, color=C_EVENT, linewidth=1.4, linestyle=":",
                     alpha=0.85, zorder=1)
    if ev_x:
        ax_a.scatter(ev_x, [y_base[e - 1] for e in ev_x], color=C_EVENT,
                     marker="v", s=70, zorder=4,
                     edgecolor=SURFACE, linewidth=1.2)
        ax_a.annotate("compaction", xy=(ev_x[0], y_base[ev_x[0] - 1]),
                      xytext=(ev_x[0] + 0.3, y_base[ev_x[0] - 1] + 6000),
                      fontsize=10, color=C_EVENT)
    ax_a.set_ylabel("cumulative input tokens", fontsize=11)
    ax_a.set_title("Cumulative input tokens over 20 turns — compacting the "
                   "older half whenever history > "
                   f"{n_thresh:,} tokens", fontsize=13, loc="left", pad=10)
    ax_a.set_xticks(range(1, 21))
    ax_a.set_xlim(0.6, 21.6)
    ax_a.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
    ax_a.grid(axis="y", color=GRID, linewidth=1)
    ax_a.set_axisbelow(True)
    ax_a.spines[["top", "right"]].set_visible(False)
    for s_ in ax_a.spines.values():
        s_.set_color("#c3c2b7")
    ax_a.legend(loc="upper left", frameon=False, fontsize=10.5,
                labelcolor=SECONDARY)

    # ---- Panel B: history size at each turn boundary — the sawtooth ------
    ax_b = fig.add_subplot(gs[1])
    ax_b.plot(x_pre, y_pre, color=C_COMPACT, linewidth=2,
              marker="o", markersize=4.5, markerfacecolor=C_COMPACT,
              markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    # drop lines: pre → post when a compaction fired
    for b in bh:
        if b["compacted"]:
            ax_b.plot([b["turn"], b["turn"]],
                      [b["est_pre"], b["est_post"]],
                      color=C_EVENT, linewidth=1.6, linestyle=":", zorder=2)
            ax_b.scatter([b["turn"]], [b["est_post"]], color=C_EVENT,
                         marker="v", s=60, zorder=4,
                         edgecolor=SURFACE, linewidth=1.2)
    ax_b.axhline(n_thresh, color=C_THRESH, linewidth=1.4, linestyle="--",
                 zorder=1)
    ax_b.annotate(f"threshold N = {n_thresh:,} tokens", xy=(20, n_thresh),
                  xytext=(14.3, n_thresh * 1.28), fontsize=10, color=SECONDARY)
    ax_b.set_ylabel("history at turn boundary (est tokens)", fontsize=11)
    ax_b.set_xlabel("user turn", fontsize=11)
    ax_b.set_title("History size at each turn boundary — compactions pull "
                   "it back under N (sawtooth)", fontsize=13, loc="left", pad=10)
    ax_b.set_xticks(range(1, 21))
    ax_b.set_xlim(0.6, 21.6)
    ax_b.grid(axis="y", color=GRID, linewidth=1)
    ax_b.set_axisbelow(True)
    ax_b.spines[["top", "right"]].set_visible(False)
    for s_ in ax_b.spines.values():
        s_.set_color("#c3c2b7")

    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nPlot saved to {out_png}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 6 Day 3: LLM-summary compaction")
    parser.add_argument("--max-tokens", type=int, default=12_000,
                        help="compaction threshold N: summarize the older "
                             "half when history exceeds N estimated tokens "
                             "(default: 12000)")
    parser.add_argument("--selftest", action="store_true",
                        help="run deterministic compaction checks only, no API")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="max API iterations per user turn")
    parser.add_argument("--baseline", metavar="RESULTS_JSON", default=None,
                        help="day-1 uncompacted results JSON (default: newest "
                             "day1_results_*.json in this directory)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the matplotlib chart (table only)")
    parser.add_argument("--plot-only", metavar="RESULTS_JSON",
                        help="regenerate the plot from a saved results JSON "
                             "(no API calls); overwrites the matching PNG name")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    if args.plot_only:
        rp = Path(args.plot_only)
        s = json.loads(rp.read_text())
        out = rp.with_name(rp.stem.replace("results", "compaction") + ".png")
        make_plot(s, out)
        print_table(s)
        return

    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY is not set — export it first "
                 "(e.g. `set -a; source ../../.env; set +a`).")

    if args.baseline:
        baseline_path = Path(args.baseline)
    else:
        baseline_path = sorted(_THIS_DIR.glob("day1_results_*.json"))[-1]
    baseline = json.loads(baseline_path.read_text())

    print(f"Starting 20-turn session with compaction max_tokens="
          f"{args.max_tokens:,} (model={MODEL}) — "
          f"baseline: {baseline_path.name} ...")
    t_start = time.monotonic()
    summary = run_compacted_session(SESSION_PROMPTS, build_tools("after"),
                                    max_tokens=args.max_tokens,
                                    max_iter_per_turn=args.max_iter)
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
        "net_input_tokens": b_in - summary["total_input_tokens"],
        "net_input_pct": round((b_in - summary["total_input_tokens"]) / b_in * 100, 2),
    }
    summary["wall_seconds"] = round(wall, 1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = _THIS_DIR / f"day3_results_n{args.max_tokens}_{ts}.json"
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Raw trace saved to {results_path}")

    print_table(summary)
    if not args.no_plot:
        png_path = _THIS_DIR / f"day3_compaction_{ts}.png"
        make_plot(summary, png_path)
    print(f"\nWall time: {wall:.0f}s")


if __name__ == "__main__":
    main()
