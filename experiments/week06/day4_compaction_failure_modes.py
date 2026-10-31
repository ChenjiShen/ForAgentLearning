"""Week 6 Day 4: Compaction failure modes — 3 tasks that REQUIRE details from
early context (a number, a filename, a constraint), with vs without
compaction.

Experiment question:
    When history crosses N and the older half is LLM-summarized, which early
    context details die — and does the same session survive without
    compaction?

Design — one 12-turn session per run, three probes at the end:

    Plant (turns 1-3):
      T01  CONSTRAINT: "every new file you create must go in research/ and
           start with the prefix d4_" — plus generate 30 numbers via a python
           script: 29 formula numbers (1000+i*37) + ONE final line with a
           RANDOM 5-digit number (random → the value exists ONLY in tool
           results / files, never in the user prompt).
      T02  read research/d4_numbers.md, sort ascending, confirm the full list
           — this puts all 30 numbers into history (read result + echo).
      T03  create a log file whose name embeds a timestamp the MODEL invents
           (YYYY_MM_DD_HH_MM_SS) — the exact filename exists only in tool-call
           args/results, not in the user prompt.
    Filler (turns 4-9): file listing, web search + fetch (bulk tool results
      that push history past N), a compute script, grep, a session-notes file.
    Probes (turns 10-12), "No tools needed" for 10-11 so the answer must come
      from history/summary, not re-derivation:
      T10  NUMBER:  "what was the value of the FINAL line of
           research/d4_numbers.md?"  — ground truth read from the sandbox file.
      T11  FILENAME: "what was the EXACT file name, including the full
           timestamp, of the log file we created early on?" — ground truth =
           the actual file created at T03.
      T12  CONSTRAINT: "Create a NEW file in research/ with your final notes"
           (no prefix hint) — ground truth = filesystem diff before/after the
           turn: new file(s) in research/ must all start with d4_.

Conditions (the ONE thing varied):
    compact — day-3's compaction with max_tokens = N (default 8,000).
    none    — identical session, compaction disabled (control).

Instrumentation mirrors day 3 (API ground-truth tokens + calibrated
estimates, per-boundary history size, compaction events with absorbed turn
ranges). A probe is *genuine* (in the compact condition) only if a
compaction absorbed turns ≥3 before the probe fired — otherwise the run
didn't actually test compaction for that probe. A probe is *void* if the
agent called tools on a no-tools turn (re-derived instead of recalling).

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week06/day4_compaction_failure_modes.py --selftest
    python experiments/week06/day4_compaction_failure_modes.py --runs 5
    python experiments/week06/day4_compaction_failure_modes.py --condition compact --runs 3

The sandbox is reset between runs (research/d4_* removed — files this
experiment owns).
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
_WEEK03_DIR = _THIS_DIR.parent / "week03"
for _d in (_WEEK05_DIR, _WEEK03_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from day1_context_growth import (  # noqa: E402
    SYSTEM_PROMPT,
    execute_tool,
    measure_components,
)
from day1_tool_descriptions import _estimate_cost, build_tools  # noqa: E402
from day3_compaction import (  # noqa: E402
    SUMMARIZER_SYSTEM,
    SUMMARY_PREFIX,
    compact_history,
    history_estimate,
)
from openai import OpenAI  # noqa: E402
import httpx  # noqa: E402

MODEL = os.getenv("MODEL_NAME", "qwen3.7-plus")
API_BASE = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# day4's sandbox lives in week03 (same tool stack as days 1-3)
SANDBOX_ROOT = Path(_WEEK03_DIR / "sandbox").resolve()

DAY4_PROMPTS: list[dict] = [
    {"id": "T01", "kind": "plant", "probe": "constraint",
     "prompt": (
         "Setting up. Two things. (1) A naming rule that applies to this "
         "WHOLE session: every new file you create must go in research/ and "
         "its name must start with the prefix d4_ — this matters later. "
         "(2) Write a python script research/d4_gen.py that generates 29 "
         "numbers using the formula 1000 + i*37 for i in range(29), one per "
         "line, then appends ONE final line with a random 5-digit number, "
         "writes all 30 lines to research/d4_numbers.md, and prints the last "
         "line. Then run the script.")},
    {"id": "T02", "kind": "plant", "probe": "number",
     "prompt": (
         "Read research/d4_numbers.md, sort the numbers in ascending order, "
         "and confirm the full sorted list in your reply.")},
    {"id": "T03", "kind": "plant", "probe": "filename",
     "prompt": (
         "Create a log file in research/ named d4_log_ followed by the "
         "current date and time in YYYY_MM_DD_HH_MM_SS format, containing "
         "the single line 'session start'.")},
    {"id": "T04", "kind": "filler",
     "prompt": "List the files in the research directory with their sizes."},
    {"id": "T05", "kind": "filler",
     "prompt": (
         "Search the web for 'context window token limits in LLM agents' "
         "and save the top results to research/d4_search_notes.md.")},
    {"id": "T06", "kind": "filler",
     "prompt": (
         "Fetch https://en.wikipedia.org/wiki/Prompt_engineering and add "
         "two bullet points to research/d4_search_notes.md.")},
    {"id": "T07", "kind": "filler",
     "prompt": (
         "Write a python script research/d4_sum.py that reads "
         "research/d4_numbers.md, computes the count and sum of the numbers, "
         "and prints them. Run it and report the result.")},
    {"id": "T08", "kind": "filler",
     "prompt": (
         "Use grep (case-insensitive) to find lines mentioning 'cache' in "
         "the research directory.")},
    {"id": "T09", "kind": "filler",
     "prompt": (
         "Create research/d4_notes.md with a short summary of everything we "
         "have done so far in this session.")},
    {"id": "T10", "kind": "probe", "probe": "number",
     "prompt": (
         "No tools needed — answer from memory. Back at the very start of "
         "this session we generated a file with 30 numbers. What was the "
         "value of the FINAL line of research/d4_numbers.md?")},
    {"id": "T11", "kind": "probe", "probe": "filename",
     "prompt": (
         "No tools needed — answer from memory. Early in this session we "
         "created a log file. What was the EXACT file name, including the "
         "full timestamp?")},
    {"id": "T12", "kind": "probe", "probe": "constraint",
     "prompt": (
         "Create a NEW file in research/ with your final notes for this "
         "session (a couple of sentences on what we did).")},
]

# plant turn each probe's detail lives in (T10's number is seen at T02 via
# read_file, T11's filename at T03, T12's constraint stated at T01).
PROBE_PLANT_TURN = {"number": 2, "filename": 3, "constraint": 1}


# ============================================================================
# Sandbox ground truth — the probes are scored against the FILESYSTEM, not
# against text that was planted in the prompts.
# ============================================================================


def snapshot_sandbox() -> dict[str, float]:
    """{relative path: mtime} for every file under the sandbox."""
    out: dict[str, float] = {}
    for p in SANDBOX_ROOT.rglob("*"):
        if p.is_file():
            out[str(p.relative_to(SANDBOX_ROOT))] = p.stat().st_mtime
    return out


def new_files(pre: dict[str, float], post: dict[str, float]) -> list[str]:
    """Files created between the two snapshots (new paths, sorted)."""
    return sorted(set(post) - set(pre))


def reset_sandbox() -> None:
    """Remove files this experiment owns (d4_* under the sandbox root and
    research/)."""
    for base in (SANDBOX_ROOT, SANDBOX_ROOT / "research"):
        if base.is_dir():
            for p in base.glob("d4_*"):
                if p.is_file():
                    p.unlink()


def read_number_truth() -> str | None:
    """Ground truth for the number probe: the last non-empty line of
    research/d4_numbers.md."""
    f = SANDBOX_ROOT / "research" / "d4_numbers.md"
    if not f.is_file():
        return None
    lines = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
    return lines[-1] if lines else None


def read_filename_truth() -> str | None:
    """Ground truth for the filename probe: the d4_log_*.txt file created at
    T03 (existence checked right after T03 in the run loop)."""
    research = SANDBOX_ROOT / "research"
    if not research.is_dir():
        return None
    logs = sorted(p.name for p in research.glob("d4_log_*.txt"))
    return logs[0] if logs else None


# ============================================================================
# Scoring — pure checkers (selftest-able)
# ============================================================================


def _normalize(s: str) -> str:
    return s.replace(",", "").replace(" ", "")


def check_number(answer: str, truth: str) -> bool:
    """Answer names the number (comma/space-insensitive)."""
    return _normalize(truth) in _normalize(answer)


def check_filename(answer: str, truth: str) -> bool:
    """Answer names the full log filename or its embedded timestamp."""
    stem = truth.replace("d4_log_", "").replace(".txt", "")
    return truth in answer or stem in answer


def check_constraint(new: list[str]) -> bool:
    """Every new file created during T12 lives in research/ and starts with
    d4_ — and at least one new file was actually created."""
    in_research = [r for r in new if r.startswith("research/")]
    if not in_research:
        return False  # nothing created in research/ (or nothing at all)
    return all(Path(r).name.startswith("d4_") for r in in_research)


def selftest() -> None:
    checks = 0

    # number checker: comma/space tolerance, absent value fails
    assert check_number("The last line was 48,371.", "48371")
    assert check_number("48371", "48371")
    assert not check_number("The last line was 48372.", "48371")
    assert not check_number("I don't remember", "48371")
    checks += 1

    # filename checker: full name or bare timestamp
    t = "d4_log_2026_08_20_09_15_42.txt"
    assert check_filename("It was d4_log_2026_08_20_09_15_42.txt", t)
    assert check_filename("timestamp 2026_08_20_09_15_42", t)
    assert not check_filename("d4_log_2026_08_20_09_15_43.txt", t)
    assert not check_filename("a log file", t)
    checks += 1

    # constraint checker: new-file sets
    assert check_constraint(["research/d4_final_notes.md"])
    assert check_constraint(["research/d4_a.md", "research/d4_b.md"])
    assert not check_constraint(["research/notes.md"])
    assert not check_constraint(["research/d4_a.md", "research/bad.md"])
    assert not check_constraint([])                      # created nothing
    assert not check_constraint(["final_notes.md"])      # wrong location
    checks += 1

    # new_files diff: only new paths appear, overwrites don't
    pre = {"research/a.md": 1.0, "research/b.md": 2.0}
    post = {"research/a.md": 1.0, "research/b.md": 9.0, "research/c.md": 9.0}
    assert new_files(pre, post) == ["research/c.md"]
    assert new_files(post, pre) == []
    checks += 1

    # plants_absorbed: needs a compaction with max absorbed turn >= 3 BEFORE
    # the probe turn
    ev = [{"status": "ok", "turn": 8, "turns_absorbed": [1, 4]}]
    assert plants_absorbed_before(ev, 10)
    assert not plants_absorbed_before(ev, 8)             # must be before probe
    assert not plants_absorbed_before(
        [{"status": "ok", "turn": 7, "turns_absorbed": [1, 2]}], 10)
    assert not plants_absorbed_before([], 10)            # control runs: none
    checks += 1

    print(f"SELFTEST PASSED ({checks} checks)")


def plants_absorbed_before(events: list[dict], probe_turn: int) -> bool:
    """Did any compaction absorb turns >=3 before this probe fired?"""
    return any(e["status"] == "ok" and e["turn"] < probe_turn
               and e["turns_absorbed"][1] >= 3
               for e in events)


# ============================================================================
# The session — day-3 loop, compaction optional, probe scoring
# ============================================================================


def run_session(condition: str, run_index: int, tools: list[dict],
                max_tokens: int, max_iter_per_turn: int = 10) -> dict:
    # trust_env=False: ignore the macOS system proxy (127.0.0.1:7890 — a
    # local proxy app that may not be running). timeout: a stalled server
    # must fail the call (SDK retries) instead of blocking the run forever.
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=API_BASE,
        http_client=httpx.Client(trust_env=False,
                                 timeout=httpx.Timeout(300.0, connect=15.0)),
    )
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    msg_turns: list[int] = [0]

    calib_factor: float | None = None
    factors_seen: list[float] = []
    calls: list[dict] = []
    turns: list[dict] = []
    compaction_events: list[dict] = []
    summarizer_calls: list[dict] = []
    boundary_history: list[dict] = []

    def summarizer(serialized: str) -> dict:
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SUMMARIZER_SYSTEM},
                      {"role": "user",
                       "content": ("<conversation>\n" + serialized +
                                   "\n</conversation>\nSummarize the "
                                   "conversation above. Output only the "
                                   "summary.")}])
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

        # ---- compact at the turn boundary (compact condition only) --------
        f = calib_factor if calib_factor else 1.0
        est_pre = round(history_estimate(messages) * f)
        event = None
        if condition == "compact":
            event = compact_history(messages, msg_turns, summarizer,
                                    max_tokens, calib=f)
            if event:
                event.update({"turn": pi, "est_pre": est_pre,
                              "est_post": round(history_estimate(messages) * f)})
                compaction_events.append(event)
        boundary_history.append({"turn": pi, "est_pre": est_pre,
                                 "compacted": bool(event and event["status"] == "ok")})

        if p["id"] == "T12":
            snap_pre_t12 = snapshot_sandbox()

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
            pr = t["probe"]
            pr = dict(pr)
            pr["turn"] = t["turn"]
            pr["plants_absorbed"] = (
                plants_absorbed_before(compaction_events, t["turn"])
                if condition == "compact" else False)
            probes[t["prompt_id"]] = pr

    return {
        "condition": condition,
        "run_index": run_index,
        "max_tokens": max_tokens if condition == "compact" else None,
        "model": MODEL,
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


def check(answer: str, truth: str, probe: str) -> bool:
    """Dispatch a (number|filename) probe to its checker."""
    return (check_filename if probe == "filename" else check_number)(answer, truth)


# ============================================================================
# Reporting
# ============================================================================


def aggregate(runs: list[dict], condition: str) -> dict:
    """Per-probe pass rates over one condition's runs."""
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
            if condition == "compact":
                if not pr["plants_absorbed"]:
                    continue
                agg["n_genuine"] += 1
                if pr["pass"]:
                    agg["n_genuine_pass"] += 1
                if not pr["pass"] and "answer" in pr:
                    agg["fail_answers"].append((r["run_index"], pr["answer"],
                                                pr["expected"]))
    return by_probe


def print_report(results: dict) -> None:
    print("\n" + "=" * 78)
    print("WEEK 6 DAY 4 — compaction failure modes: number / filename / constraint")
    print("=" * 78)
    for condition, runs in results["conditions"].items():
        ok_runs = [r for r in runs if "error" not in r]
        print(f"\n--- condition: {condition} ({len(ok_runs)} ok runs, "
              f"{len(runs)-len(ok_runs)} errored) ---")
        tot_in = sum(r["agent_input_tokens"] for r in ok_runs)
        cost = sum(r["estimated_cost_usd"] for r in ok_runs)
        comps = sum(r["n_compactions"] for r in ok_runs)
        print(f"    total agent input: {tot_in:,}  est. cost: ${cost:.3f}  "
              f"compactions: {comps}")
        agg = aggregate(runs, condition)
        for name, a in agg.items():
            base = a["n_valid"] or 1
            line = (f"    probe {name:<11} pass {a['n_pass']}/{a['n_valid']} "
                    f"({a['n_pass']/base*100:.0f}%)  "
                    f"[void {a['n_runs']-a['n_valid']}]")
            if condition == "compact" and a["n_genuine"]:
                line += (f"  genuine-only {a['n_genuine_pass']}/{a['n_genuine']}"
                         f" ({a['n_genuine_pass']/a['n_genuine']*100:.0f}%)")
            print(line)
            for ri, ans, exp in a["fail_answers"][:3]:
                print(f"      ✗ run {ri}: answered {ans!r}  expected {exp!r}")
        # per-run one-liner
        for r in runs:
            if "error" in r:
                print(f"      run {r['run_index']}: ERROR {r['error'][:100]}")
                continue
            marks = " ".join(
                ("✓" if r["probes"][pid]["pass"]
                 else ("∅" if r["probes"][pid]["void"] else "✗"))
                for pid in ("T10", "T11", "T12"))
            ab = (" absorbed" if r["probes"]["T10"]["plants_absorbed"] else "")
            print(f"      run {r['run_index']}: {marks}  "
                  f"{r['n_compactions']} compactions{ab}  "
                  f"{r['agent_input_tokens']:,} in  ${r['estimated_cost_usd']:.3f}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 6 Day 4: compaction failure modes")
    parser.add_argument("--condition", choices=["both", "compact", "none"],
                        default="both")
    parser.add_argument("--runs", type=int, default=5,
                        help="runs per condition")
    parser.add_argument("--max-tokens", type=int, default=8_000,
                        help="compaction threshold N (compact condition)")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="max API iterations per user turn")
    parser.add_argument("--selftest", action="store_true",
                        help="run deterministic checker tests only, no API")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY is not set — export it first "
                 "(e.g. `set -a; source ../../.env; set +a`).")

    tools = build_tools("after")
    conditions = {"compact", "none"} if args.condition == "both" \
        else {args.condition}
    results: dict[str, list[dict]] = {}
    t_start = time.monotonic()

    for condition in sorted(conditions):
        print(f"\n>>> condition={condition}  {args.runs} runs, "
              f"max_tokens={args.max_tokens:,} (model={MODEL}) ...")
        runs: list[dict] = []
        for ri in range(1, args.runs + 1):
            reset_sandbox()
            t0 = time.monotonic()
            try:
                r = run_session(condition, ri, tools, args.max_tokens,
                                max_iter_per_turn=args.max_iter)
            except Exception as e:  # one bad run must not kill the batch
                r = {"condition": condition, "run_index": ri, "error": str(e)}
                print(f"  run {ri}: EXCEPTION {type(e).__name__}: {str(e)[:120]}")
            r["wall_seconds"] = round(time.monotonic() - t0, 1)
            runs.append(r)
            if "error" in r:
                continue
            marks = " ".join(
                ("✓" if r["probes"][pid]["pass"]
                 else ("∅" if r["probes"][pid]["void"] else "✗"))
                for pid in ("T10", "T11", "T12"))
            print(f"  run {ri}: {marks}  compactions={r['n_compactions']}  "
                  f"in={r['agent_input_tokens']:,}  "
                  f"${r['estimated_cost_usd']:.3f}  {r['wall_seconds']}s",
                  flush=True)
        results[condition] = runs

    summary = {
        "experiment": "week06 day4 compaction failure modes",
        "model": MODEL,
        "max_tokens": args.max_tokens,
        "conditions": results,
        "wall_seconds_total": round(time.monotonic() - t_start, 1),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _THIS_DIR / f"day4_results_{ts}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nRaw traces saved to {out}")

    print_report(summary)
    print(f"\nTotal wall time: {summary['wall_seconds_total']:.0f}s")


if __name__ == "__main__":
    main()
