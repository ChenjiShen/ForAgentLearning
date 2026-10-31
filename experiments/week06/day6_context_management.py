"""Week 6 Day 6: Context-management experiment — pruning vs. compaction vs.
nothing on long-session task success + cost.

Experiment question (PLAN.md week 6 day 6):
    Which context strategy wins on a long session — pruning (day 2),
    compaction (day 3), or nothing? Measured on task success AND cost.

Design — day 4's probe battery as the success measure, three conditions
(plus one optional extension):

    The same 12-turn session as day 4: plants (T1-3: a random 5-digit
    number, a model-invented filename, a naming constraint), filler (T4-9:
    bulk tool results that push history past the threshold), probes
    (T10-12: recall the number, recall the filename, follow the
    constraint). Ground truth comes from the SANDBOX FILESYSTEM, never from
    planted prompt text. A probe is *void* if the agent called tools on a
    no-tools turn.

    Conditions — the ONE thing varied is the context strategy applied at
    every turn boundary, before the turn's API calls:
      none          control: history untouched.
      prune         day 2's clear_old_tool_results, n=3: tool results older
                    than 3 turns become "[cleared]" stubs; user/assistant
                    messages (the decisions) survive verbatim.
      compact       day 3's compact_history, N=8,000 tokens: when history
                    crosses N, the older half is LLM-summarized with day 3's
                    preserve-everything summarizer prompt and spliced back.
      compact_lazy  (--with-lazy) same mechanism, but a deliberately sloppy
                    summarizer prompt ("2-3 sentences, gist only, details
                    unimportant") — day 4's lesson: the preserve-everything
                    summarizer killed nothing, so a failure gradient needs a
                    weaker summarizer to materialize.

    System prompt = day 5's NEW_PROMPT for every condition. Day 5's open
    question was whether to keep the new prompt only if it helps the
    long-session failure modes — this run exercises exactly those, so the
    results also close day 5.

    Run order (condition x run) is shuffled with a fixed seed so
    time-of-day drift cannot confound the comparison (week-5 day-6 lesson).

    Genuineness: a probe in a strategy condition is *genuine* only if the
    strategy actually touched the probe's planted material before the probe
    fired — compact: a compaction absorbed turns >= 3 before the probe
    (day 4's definition); prune: a pruning cleared a tool result from the
    probe's plant turn or earlier before the probe fired.

Run:
    export $(grep DASHSCOPE .env | head -1 | xargs)
    python experiments/week06/day6_context_management.py --selftest
    python experiments/week06/day6_context_management.py --smoke
    python experiments/week06/day6_context_management.py --runs 5 [--with-lazy]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_WEEK05_DIR = _THIS_DIR.parent / "week05"
_WEEK03_DIR = _THIS_DIR.parent / "week03"
for _d in (_THIS_DIR, _WEEK05_DIR, _WEEK03_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from day1_context_growth import (  # noqa: E402
    execute_tool,
    measure_components,
)
from day1_tool_descriptions import _estimate_cost, build_tools  # noqa: E402
from day2_clear_old_tool_results import (  # noqa: E402
    clear_old_tool_results,
)
from day3_compaction import (  # noqa: E402
    SUMMARIZER_SYSTEM,
    compact_history,
    history_estimate,
)
from day4_compaction_failure_modes import (  # noqa: E402
    DAY4_PROMPTS,
    PROBE_PLANT_TURN,
    check,
    check_constraint,
    check_filename,
    check_number,
    new_files,
    plants_absorbed_before,
    read_number_truth,
    reset_sandbox,
    snapshot_sandbox,
)
from day5_prompt_ab import NEW_PROMPT  # noqa: E402
from day6_tool_distraction import fisher_exact_two_sided  # noqa: E402 (week05)
from openai import OpenAI  # noqa: E402
import httpx  # noqa: E402

MODEL = os.getenv("MODEL_NAME", "qwen3.7-plus")
API_BASE = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

CORE_CONDITIONS = ["none", "prune", "compact"]
LAZY_CONDITION = "compact_lazy"

# The deliberately sloppy summarizer (day 4 lesson 1: the preserve-everything
# summarizer kills nothing, so the failure mode needs a weaker one to show).
SUMMARIZER_SYSTEM_LAZY = (
    "You are a lazy note-taker. Summarize the conversation in 2-3 short "
    "sentences capturing only the general gist. Do not list numbers, "
    "filenames, or specific details — they are not important."
)

# The sandbox write_file whitelist rejects extensionless names (day 4), so
# the agent now creates the log via a python script instead — extensionless
# files DO exist on disk. Day 4's truth glob ("d4_log_*.txt") misses those
# and auto-fails the probe; day 6 accepts any d4_log_* file.
SANDBOX_ROOT = Path(_WEEK03_DIR / "sandbox").resolve()


def read_filename_truth() -> str | None:
    """Ground truth for the filename probe: the d4_log_* file created at T03
    (captured right after T03 in the run loop; any extension)."""
    research = SANDBOX_ROOT / "research"
    if not research.is_dir():
        return None
    logs = sorted(p.name for p in research.glob("d4_log_*") if p.is_file())
    return logs[0] if logs else None


# ============================================================================
# Watchdog — one stalled DashScope request must fail its run, not the batch
# ============================================================================
# Day 4's notes: the endpoint can stall a request indefinitely (~1h
# observed); the httpx timeout bounds ordinary stalls, but a trickle-style
# stall can outlive it. SIGALRM gives each run a hard wall-clock budget:
# the alarm interrupts the hung call, the run is recorded as an error, and
# the batch moves to the next run.


class RunBudgetExceeded(Exception):
    pass


def _watchdog_handler(signum, frame):  # noqa: ARG001
    raise RunBudgetExceeded("run exceeded its wall-clock budget")


class _Deadline:
    """Per-run wall-clock budget enforcing a per-call cap (SIGALRM; handler
    installed in main). One instance per run.

    Observed failure mode that motivates this: the DashScope endpoint can
    stall an SSL read past httpx's timeout (stack: _ssl__SSLSocket_read ->
    poll, no data ever) and day 3's compact_history swallows Exception
    around the summarizer. So the alarm must be armed around EVERY API call
    with a fresh budget — a swallowed alarm then costs at most one call —
    and re-armed between calls so a hung tool call also dies.
    """

    def __init__(self, run_budget_seconds: int, call_budget_seconds: int):
        self._deadline = time.monotonic() + run_budget_seconds
        self._call_budget = call_budget_seconds

    def _remain(self) -> int:
        return self._deadline - time.monotonic()

    def _arm(self, seconds: int) -> None:
        signal.alarm(max(1, seconds))

    def call(self, fn):
        """Run fn; a stall longer than the call budget (or the remaining
        run budget) raises RunBudgetExceeded."""
        remain = self._remain()
        if remain <= 0:
            raise RunBudgetExceeded("run exceeded its wall-clock budget")
        self._arm(min(int(remain) + 1, self._call_budget))
        ok = False
        try:
            result = fn()
            ok = True
            return result
        finally:
            if ok:
                # call finished inside budget: keep the run-level budget
                # armed until the next call or tool action (next call()
                # raises if the run budget is already spent)
                remain = self._remain()
                if remain > 0:
                    self._arm(int(remain) + 1)

# ============================================================================
# Prune genuineness — the prune counterpart of day 4's plants_absorbed_before
# ============================================================================


def plants_pruned_before(events: list[dict], probe_turn: int,
                         plant_turn: int) -> bool:
    """Did pruning clear a tool result from the probe's plant turn before
    the probe fired?

    `events` are prune records enriched with `boundary` (the turn at whose
    boundary the clearing happened). Pruning clears by age, so a plant
    turn's results all clear at the same boundary — one cleared result from
    that turn means the probe's raw tool material is gone. Strict `<` on
    boundary matches day 4's convention for compaction events.
    """
    return any(e["boundary"] < probe_turn and e["turn"] == plant_turn
               for e in events)


# ============================================================================
# Selftest — deterministic checks (no API)
# ============================================================================


def selftest() -> None:
    checks = 0

    # plants_pruned_before: clearing AT the probe boundary doesn't count
    # (strict <, same convention as day 4); earlier boundaries do.
    ev = [{"boundary": 4, "turn": 1}, {"boundary": 5, "turn": 2}]
    assert plants_pruned_before(ev, 10, 2)          # turn 2 cleared at T5 < T10
    assert plants_pruned_before(ev, 10, 1)
    assert not plants_pruned_before([{"boundary": 10, "turn": 2}], 10, 2)
    assert not plants_pruned_before([{"boundary": 5, "turn": 2}], 5, 2)
    assert not plants_pruned_before(ev, 10, 3)      # plant 3 never cleared
    assert not plants_pruned_before([], 10, 2)
    checks += 1

    # prune events: a turn is pruned exactly when current - turn >= n, and
    # the record keeps the originating turn (day 2 contract, re-verified).
    from day2_clear_old_tool_results import _make_synthetic
    msgs, ts = _make_synthetic(5)
    rec = clear_old_tool_results(msgs, ts, current_turn=4, n=2)
    assert [r["turn"] for r in rec] == [1, 2]
    checks += 1

    # day 4's checker battery, re-asserted so day 6 owns its scoring
    assert check_number("The last line was 48,371.", "48371")
    assert not check_number("The last line was 48372.", "48371")
    assert check_filename("It was d4_log_2026_08_20_09_15_42.txt",
                          "d4_log_2026_08_20_09_15_42.txt")
    assert not check_filename("a log file", "d4_log_2026_08_20_09_15_42.txt")
    assert check_constraint(["research/d4_final_notes.md"])
    assert not check_constraint(["research/notes.md"])
    assert not check_constraint([])
    assert new_files({"research/a.md": 1.0}, {"research/a.md": 9.0,
                                              "research/c.md": 9.0}) \
        == ["research/c.md"]
    assert plants_absorbed_before([{"status": "ok", "turn": 8,
                                    "turns_absorbed": [1, 4]}], 10)
    assert not plants_absorbed_before([], 10)
    checks += 1

    print(f"SELFTEST PASSED ({checks} checks)")


# ============================================================================
# The session — day 4's loop, plus the prune hook and lazy summarizer option
# ============================================================================


def run_session(condition: str, run_index: int, tools: list[dict],
                max_tokens: int, prune_n: int, deadline: _Deadline,
                max_iter_per_turn: int = 10) -> dict:
    # trust_env=False: ignore the macOS system proxy (127.0.0.1:7890 — a
    # local proxy app that may not be running). timeout: a stalled server
    # must fail the call (SDK retries) instead of blocking the run forever.
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=API_BASE,
        # max_retries=1: halve the stall tail (2 attempts x 300s read timeout
        # worst case per call); the per-run watchdog below is the hard bound.
        max_retries=1,
        http_client=httpx.Client(trust_env=False,
                                 timeout=httpx.Timeout(300.0, connect=15.0)),
    )
    messages: list[dict] = [{"role": "system", "content": NEW_PROMPT}]
    msg_turns: list[int] = [0]

    # compact_lazy uses the sloppy summarizer; every other condition uses
    # day 3's preserve-everything one when it summarizes at all.
    summarizer_system = (SUMMARIZER_SYSTEM_LAZY
                         if condition == LAZY_CONDITION else SUMMARIZER_SYSTEM)

    calib_factor: float | None = None
    factors_seen: list[float] = []
    calls: list[dict] = []
    turns: list[dict] = []
    compaction_events: list[dict] = []
    prune_events: list[dict] = []
    summarizer_calls: list[dict] = []
    boundary_history: list[dict] = []

    def summarizer(serialized: str) -> dict:
        t0 = time.monotonic()
        resp = deadline.call(
            lambda: client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": summarizer_system},
                          {"role": "user",
                           "content": ("<conversation>\n" + serialized +
                                       "\n</conversation>\nSummarize the "
                                       "conversation above. Output only the "
                                       "summary.")}]))
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

    number_truth = read_number_truth()   # None until T01/T02 create the file
    filename_truth: str | None = None    # captured right after T03
    snap_pre_t12: dict[str, float] | None = None

    for pi, p in enumerate(DAY4_PROMPTS, start=1):
        messages.append({"role": "user", "content": p["prompt"]})
        msg_turns.append(pi)

        # ---- apply the strategy at the turn boundary ----------------------
        f = calib_factor if calib_factor else 1.0
        est_pre = round(history_estimate(messages) * f)
        event = None
        if condition.startswith("compact"):
            event = compact_history(messages, msg_turns, summarizer,
                                    max_tokens, calib=f)
            if event:
                event.update({"turn": pi, "est_pre": est_pre,
                              "est_post": round(history_estimate(messages) * f)})
                compaction_events.append(event)
        elif condition == "prune":
            recs = clear_old_tool_results(messages, msg_turns, pi, prune_n)
            for rec in recs:
                rec["boundary"] = pi
            prune_events.extend(recs)
            if recs:
                event = {"status": "ok", "turn": pi,
                         "n_cleared": len(recs),
                         "chars_cleared": sum(r["orig_chars"] for r in recs),
                         "est_pre": est_pre,
                         "est_post": round(history_estimate(messages) * f)}
        boundary_history.append({
            "turn": pi, "est_pre": est_pre,
            "compacted": bool(event and event["status"] == "ok"
                              and condition.startswith("compact")),
            "pruned": bool(condition == "prune" and prune_events
                           and prune_events[-1]["boundary"] == pi),
        })

        if p["id"] == "T12":
            snap_pre_t12 = snapshot_sandbox()

        turn_calls, tool_names, t0_turn = [], [], time.monotonic()
        final_answer = ""

        for _ in range(max_iter_per_turn):
            t0 = time.monotonic()
            comp = measure_components(messages, tools)
            response = deadline.call(
                lambda: client.chat.completions.create(
                    model=MODEL, tools=tools, messages=messages))
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
                "elapsed_s": round(elapsed, 2),
            }
            calls.append(call)
            turn_calls.append(call)

            if not msg.tool_calls:
                final_answer = msg.content or ""
                break

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

        # ---- ground truth capture at the plant turns ----------------------
        if p["id"] == "T02":
            number_truth = read_number_truth()
        elif p["id"] == "T03":
            filename_truth = read_filename_truth()

        # ---- probe scoring ------------------------------------------------
        turn = {
            "turn": pi,
            "prompt_id": p["id"],
            "kind": p["kind"],
            "n_api_calls": len(turn_calls),
            "n_tool_calls": len(tool_names),
            "tools_called": tool_names,
            "input_tokens": sum(c["api_prompt_tokens"] or 0 for c in turn_calls),
            "output_tokens": sum(c["api_completion_tokens"] or 0
                                 for c in turn_calls),
            "final_answer": final_answer,
            "elapsed_s": round(time.monotonic() - t0_turn, 2),
        }
        if p["kind"] == "probe":
            probe = p["probe"]
            if probe == "constraint":
                snap_post = snapshot_sandbox()
                created = new_files(snap_pre_t12 or {}, snap_post)
                turn["probe"] = {
                    "probe": probe,
                    "created_files": created,
                    "expected": "new file(s) in research/ with d4_ prefix",
                    "pass": check_constraint(created),
                    "void": False,
                }
            else:
                truth = number_truth if probe == "number" else filename_truth
                used_tools = len(tool_names) > 0
                turn["probe"] = {
                    "probe": probe,
                    "answer": final_answer,
                    "expected": truth,
                    # truth is None when the plant step failed to create the
                    # file — scored as a probe failure, not a void
                    "pass": (truth is not None) and not used_tools
                            and check(final_answer, truth, probe),
                    "void": used_tools,  # re-derived instead of recalling
                }
        turns.append(turn)

    # ---- run rollup -------------------------------------------------------
    total_in = sum(t["input_tokens"] for t in turns)
    total_out = sum(t["output_tokens"] for t in turns)
    sum_in = sum(c["prompt_tokens"] or 0 for c in summarizer_calls)
    sum_out = sum(c["completion_tokens"] or 0 for c in summarizer_calls)
    factors = [x for x in factors_seen if x]
    factor_mean = sum(factors) / len(factors) if factors else None

    probes = {}
    for t in turns:
        if t["kind"] == "probe":
            pr = dict(t["probe"])
            pr["turn"] = t["turn"]
            pr["plants_absorbed"] = (
                plants_absorbed_before(compaction_events, t["turn"])
                if condition.startswith("compact") else False)
            pr["plants_pruned"] = (
                plants_pruned_before(prune_events, t["turn"],
                                     PROBE_PLANT_TURN[pr["probe"]])
                if condition == "prune" else False)
            probes[t["prompt_id"]] = pr

    return {
        "condition": condition,
        "run_index": run_index,
        "max_tokens": max_tokens if condition.startswith("compact") else None,
        "prune_n": prune_n if condition == "prune" else None,
        "model": MODEL,
        "system_prompt": NEW_PROMPT,
        "n_turns": len(turns),
        "n_api_calls": len(calls),
        "n_tool_calls": sum(t["n_tool_calls"] for t in turns),
        "agent_input_tokens": total_in,
        "agent_output_tokens": total_out,
        "summarizer_input_tokens": sum_in,
        "summarizer_output_tokens": sum_out,
        "total_tokens": total_in + total_out + sum_in + sum_out,
        "estimated_cost_usd": round(_estimate_cost(total_in + sum_in,
                                                   total_out + sum_out), 4),
        "n_compactions": len(compaction_events),
        "compaction_events": compaction_events,
        "n_pruned": len(prune_events),
        "chars_pruned": sum(e["orig_chars"] for e in prune_events),
        "prune_events": prune_events,
        "boundary_history": boundary_history,
        "number_truth": number_truth,
        "filename_truth": filename_truth,
        "probes": probes,
        "calibration_factor": round(calib_factor, 4) if calib_factor else None,
        "calibration_factor_mean": round(factor_mean, 4) if factor_mean else None,
        "turns": turns,
        "calls": calls,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================================
# Reporting
# ============================================================================


def aggregate(runs: list[dict], condition: str) -> dict:
    """Per-probe pass rates over one condition's runs (day 4's aggregate,
    extended with the prune genuineness bucket)."""
    by_probe: dict[str, dict] = {}
    for r in runs:
        if "error" in r:
            continue
        for pid, pr in r["probes"].items():
            name = pr["probe"]
            agg = by_probe.setdefault(
                name, {"n_runs": 0, "n_valid": 0, "n_pass": 0,
                       "n_genuine": 0, "n_genuine_pass": 0, "answers": [],
                       "fail_answers": []})
            agg["n_runs"] += 1
            if pr["void"]:
                continue
            agg["n_valid"] += 1
            if pr["pass"]:
                agg["n_pass"] += 1
            genuine = pr["plants_pruned"] if condition == "prune" \
                else pr["plants_absorbed"]
            if condition in ("prune", "compact", LAZY_CONDITION):
                if not genuine:
                    continue
                agg["n_genuine"] += 1
                if pr["pass"]:
                    agg["n_genuine_pass"] += 1
                if not pr["pass"] and "answer" in pr:
                    agg["fail_answers"].append((r["run_index"], pr["answer"],
                                                pr["expected"]))
    return by_probe


def _pass_counts(runs: list[dict], probe: str) -> tuple[int, int]:
    """(passes, valid) pooled over a condition's runs for one probe."""
    k = n = 0
    for r in runs:
        if "error" in r:
            continue
        for pid, pr in r["probes"].items():
            if pr["probe"] != probe:
                continue
            if pr["void"]:
                continue
            n += 1
            k += 1 if pr["pass"] else 0
    return k, n


def print_report(results: dict, conditions: list[str]) -> None:
    print("\n" + "=" * 78)
    print("WEEK 6 DAY 6 — pruning vs. compaction vs. nothing: success + cost")
    print("=" * 78)
    for condition in conditions:
        runs = results[condition]
        ok_runs = [r for r in runs if "error" not in r]
        print(f"\n--- condition: {condition} ({len(ok_runs)} ok runs, "
              f"{len(runs) - len(ok_runs)} errored) ---")
        tot_in = sum(r["agent_input_tokens"] for r in ok_runs)
        tot_all = sum(r["total_tokens"] for r in ok_runs)
        cost = sum(r["estimated_cost_usd"] for r in ok_runs)
        comps = sum(r["n_compactions"] for r in ok_runs)
        prunes = sum(r["n_pruned"] for r in ok_runs)
        mean_in = tot_in / len(ok_runs) if ok_runs else 0
        print(f"    agent input {tot_in:,} (mean {mean_in:,.0f}/run)  "
              f"total tokens {tot_all:,}  est. cost ${cost:.3f}  "
              f"compactions {comps}  pruned {prunes}")
        agg = aggregate(runs, condition)
        for name, a in agg.items():
            base = a["n_valid"] or 1
            line = (f"    probe {name:<11} pass {a['n_pass']}/{a['n_valid']} "
                    f"({a['n_pass'] / base * 100:.0f}%)  "
                    f"[void {a['n_runs'] - a['n_valid']}]")
            if a["n_genuine"]:
                line += (f"  genuine {a['n_genuine_pass']}/{a['n_genuine']}"
                         f" ({a['n_genuine_pass'] / a['n_genuine'] * 100:.0f}%)")
            print(line)
            for ri, ans, exp in a["fail_answers"][:3]:
                print(f"      x run {ri}: answered {ans!r}  expected {exp!r}")
        for r in runs:
            if "error" in r:
                print(f"      run {r['run_index']}: ERROR {r['error'][:100]}")
                continue
            marks = " ".join(
                ("✓" if r["probes"][pid]["pass"]
                 else ("∅" if r["probes"][pid]["void"] else "✗"))
                for pid in ("T10", "T11", "T12"))
            extra = (f"{r['n_compactions']}c" if r["n_compactions"] else
                     f"{r['n_pruned']}p" if r["n_pruned"] else "-")
            print(f"      run {r['run_index']}: {marks}  {extra}  "
                  f"{r['agent_input_tokens']:,} in  ${r['estimated_cost_usd']:.3f}"
                  f"  {r.get('wall_seconds', '?')}s")

    # ---- head-to-head: Fisher exact on per-probe pass rates ---------------
    print("\n--- Fisher exact (2x2, valid probes): pass rate between conditions")
    probes = ["number", "filename", "constraint"]
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            a, b = conditions[i], conditions[j]
            for probe in probes:
                ka, na = _pass_counts(results[a], probe)
                kb, nb = _pass_counts(results[b], probe)
                if na == 0 or nb == 0:
                    continue
                p = fisher_exact_two_sided(ka, na - ka, kb, nb - kb)
                marker = "  <-- significant" if p < 0.05 else ""
                print(f"    {a:<13} vs {b:<13} {probe:<11} "
                      f"{ka}/{na} vs {kb}/{nb}  p={p:.3f}{marker}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 6 Day 6: pruning vs. compaction vs. nothing")
    parser.add_argument("--condition",
                        choices=["all", "none", "prune", "compact",
                                 LAZY_CONDITION],
                        default="all")
    parser.add_argument("--runs", type=int, default=5,
                        help="runs per condition")
    parser.add_argument("--max-tokens", type=int, default=8_000,
                        help="compaction threshold N (compact conditions)")
    parser.add_argument("--prune-n", type=int, default=3,
                        help="pruning window n (prune condition)")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="max API iterations per user turn")
    parser.add_argument("--with-lazy", action="store_true",
                        help="add the compact_lazy condition (day 4 lesson 1)")
    parser.add_argument("--smoke", action="store_true",
                        help="one run per condition, in order (validation)")
    parser.add_argument("--run-budget", type=int, default=1200,
                        help="per-run wall-clock watchdog in seconds "
                             "(a stalled API request fails the run, not "
                             "the batch)")
    parser.add_argument("--call-budget", type=int, default=300,
                        help="per-API-call wall-clock budget in seconds "
                             "(the endpoint's stall variant outlives "
                             "httpx's timeout)")
    parser.add_argument("--selftest", action="store_true",
                        help="run deterministic checker tests only, no API")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY is not set — export it first "
                 "(e.g. `export $(grep DASHSCOPE ../../.env | "
                 "head -1 | xargs)`).")

    if args.condition == "all":
        conditions = list(CORE_CONDITIONS)
        if args.with_lazy:
            conditions.append(LAZY_CONDITION)
    else:
        conditions = [args.condition]

    signal.signal(signal.SIGALRM, _watchdog_handler)

    tools = build_tools("after")
    results: dict[str, list[dict]] = {c: [] for c in conditions}
    t_start = time.monotonic()

    # Interleave (condition x run) pairs with a fixed seed so time-of-day
    # drift cannot confound the comparison (week-5 day-6 lesson). Smoke runs
    # are one run per condition, in condition order for easy reading.
    runs_per_condition = 1 if args.smoke else args.runs
    pairs = [(c, ri) for c in conditions
             for ri in range(1, runs_per_condition + 1)]
    if not args.smoke:
        random.Random(20260823).shuffle(pairs)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _THIS_DIR / f"day6_results_{ts}.json"
    summary = {
        "experiment": "week06 day6 pruning vs compaction vs nothing",
        "model": MODEL,
        "max_tokens": args.max_tokens,
        "prune_n": args.prune_n,
        "conditions": results,
        "wall_seconds_total": 0.0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f">>> {len(pairs)} runs over {conditions} "
          f"(model={MODEL}, max_tokens={args.max_tokens:,}, "
          f"prune_n={args.prune_n}, run_budget={args.run_budget}s, "
          f"call_budget={args.call_budget}s) ...")
    for c, ri in pairs:
        reset_sandbox()
        deadline = _Deadline(args.run_budget, args.call_budget)
        t0 = time.monotonic()
        try:
            r = run_session(c, ri, tools, args.max_tokens, args.prune_n,
                            deadline, max_iter_per_turn=args.max_iter)
        except Exception as e:  # one bad run must not kill the batch
            r = {"condition": c, "run_index": ri, "error": str(e)}
            print(f"  [{c}] run {ri}: EXCEPTION {type(e).__name__}: "
                  f"{str(e)[:120]}")
        signal.alarm(0)  # the run consumed or left a stale alarm — disarm
        r["wall_seconds"] = round(time.monotonic() - t0, 1)
        results[c].append(r)
        summary["wall_seconds_total"] = round(time.monotonic() - t_start, 1)
        # incremental save: a killed batch never loses completed runs
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        if "error" in r:
            continue
        marks = " ".join(
            ("✓" if r["probes"][pid]["pass"]
             else ("∅" if r["probes"][pid]["void"] else "✗"))
            for pid in ("T10", "T11", "T12"))
        extra = (f"compactions={r['n_compactions']}" if r["n_compactions"]
                 else f"pruned={r['n_pruned']}" if r["n_pruned"] else "-")
        print(f"  [{c}] run {ri}: {marks}  {extra}  "
              f"in={r['agent_input_tokens']:,}  ${r['estimated_cost_usd']:.3f}  "
              f"{r['wall_seconds']}s", flush=True)

    print(f"\nRaw traces saved to {out}")

    print_report(summary["conditions"], conditions)
    print(f"\nTotal wall time: {summary['wall_seconds_total']:.0f}s")


if __name__ == "__main__":
    main()
